"""Measured rerank-prefix K, retrieval cost, and real online query cost."""
from __future__ import annotations
import time
import torch
from .assets import load_bundle,checkpoint
from .runtime import atomic_json,read_json,fingerprint,sha,Blocked
from .engine import rank_of,write_jsonl

def synchronize(run):
 if run.device.type=='cuda':torch.cuda.synchronize(run.device)
def experiments(cfg,rr,cat,run,ks):
 b=load_bundle(rr,cat,'val',run);Q=b['query']['query_features'];I=b['image']['features'];top=b['topk'];L=len(top['sorted_ids'][0]);ks=sorted(set(ks))
 if not ks or min(ks)<1 or max(ks)>L:raise ValueError(f'requested rerank K exceeds actual cached candidates {L}; regenerate assets with candidate_k >= requested K')
 path=run.path/'checkpoints'/cat/'best.pt'
 if not path.exists():raise Blocked('new true-similarity RR checkpoint required')
 tick=time.perf_counter();model,_,_=checkpoint(rr,cat,run,path);model.to(run.device).eval();synchronize(run);load_seconds=time.perf_counter()-tick
 bs=cfg['export_batch_size'];N=len(Q);folder=run.path/'predictions'/f'K-experiments-{cat}';folder.mkdir(parents=True,exist_ok=True);allrows=[];profiles=[];token=fingerprint({'model':sha(path),'inputs':{p.name:sha(p) for p in b['files']},'ks':ks,'batch':bs,'code':cfg['_code_fingerprint']});started=time.monotonic()
 for start in range(0,N,bs):
  p=folder/f'{start:08d}.json';chunk=read_json(p)
  if chunk:
   if chunk['fingerprint']!=token or fingerprint(chunk['rows'])!=chunk['rows_hash']:raise ValueError('experiment chunk corrupt/incompatible')
  else:
   run.check_stop();end=min(start+bs,N);rows=[];timings=[];query=Q[start:end].to(run.device).float()
   synchronize(run);tick=time.perf_counter()
   with torch.no_grad():
    sims=query@I.to(run.device).float().T
    for offset,source in enumerate(b['query']['source_ids'][start:end]):sims[offset,b['index'][source]]=-torch.inf
    order=torch.argsort(sims,dim=-1,descending=True)
   synchronize(run);retrieval_seconds=time.perf_counter()-tick # raw operation timing; separate from protocol-controlled prefix candidates
   for K in ks:
    ids=[row[:K] for row in top['sorted_ids'][start:end]];indices=torch.tensor([[b['index'][x] for x in row] for row in ids]);candidates=I[indices].to(run.device).float();base=torch.tensor([row[:K] for row in top['sorted_scores'][start:end]],device=run.device)
    synchronize(run);tick=time.perf_counter()
    with torch.no_grad():scores=base+rr['categories'][cat]['blend']*(model(query,candidates,base)-base);perms=torch.argsort(scores,dim=-1,descending=True).cpu().tolist()
    synchronize(run);elapsed=time.perf_counter()-tick;timings.append({'start':start,'batch_size':end-start,'rerank_K':K,'seconds':elapsed,'warmup':start==0,'device':str(run.device),'precision':'float32','cuda_synchronized':run.device.type=='cuda'})
    for offset,perm in enumerate(perms):
     j=start+offset;ranked=[ids[offset][x] for x in perm]+list(top['sorted_ids'][j][K:]);target=b['query']['target_ids'][j];rank=rank_of(ranked,target)
     rows.append({'query_id':b['query_ids'][j],'reference_image_id':b['query']['source_ids'][j],'target_image_id':target,'rerank_K':K,'candidate_L':L,'retrieved_image_ids':ranked,'target_rank':rank,'rank_status':'exact within cached candidates' if rank else f'outside Top-{L}','rank_lower_bound':None if rank else L+1,'prefix_protocol':'reorder first K, leave tail ordering unchanged; never insert target'})
   chunk={'fingerprint':token,'rows':rows,'rows_hash':fingerprint(rows),'timings':timings,'retrieval':{'start':start,'batch_size':end-start,'seconds':retrieval_seconds,'scope':'cached feature matmul+sort including gallery transfer; not encoder cost','warmup':start==0}};atomic_json(p,chunk)
  allrows.extend(chunk['rows']);profiles.extend(chunk['timings']+[chunk['retrieval']]);run.progress(min(N,start+bs),N,started,category=cat,phase='K-ablation')
 metrics={}
 for K in ks:
  rows=[r for r in allrows if r['rerank_K']==K];metrics[str(K)]={'count':len(rows),'hits':{str(k):sum(r['target_rank'] is not None and r['target_rank']<=k for r in rows) for k in cfg['cutoffs'] if k<=L},'target_coverage_at_K':sum(r['target_rank'] is not None and r['target_rank']<=K for r in rows)/len(rows)*100,'candidate_L':L}
  metrics[str(K)]['recall']={k:v*100/len(rows) for k,v in metrics[str(K)]['hits'].items()}
 pred=run.path/'predictions'/f'K-experiments-{cat}.jsonl';write_jsonl(pred,allrows);out=run.path/'metrics'/f'K-experiments-{cat}.json';atomic_json(out,metrics)
 perf=run.path/'profiling'/f'cached-retrieval-RR-{cat}.json';atomic_json(perf,{'model_load_seconds':load_seconds,'raw_batches':profiles,'hardware':str(run.device),'precision':'float32','first_batch_marked_warmup':True,'scope':'cached retrieval includes gallery transfer+source exclusion+sort; RR timing is forward+blend+sort/host result transfer, excludes candidate gather/transfer; not end-to-end'})
 return [pred,out,perf]

def online_cost(cfg,cat,run):
 from .upstream import model_for,ckpt_path,sync
 from .raw_data import FashionData
 data=FashionData(cfg,cat,'val');path=ckpt_path(run,'cn',cat)
 if not path.exists():raise Blocked('trained new CN required for online encoding cost')
 model=model_for(cfg,'cn',run,path).eval();rr_cfg=__import__('cir_full.full_cli',fromlist=['rr_config']).rr_config(cfg,run);b=load_bundle(rr_cfg,cat,'val',run);reranker,_,_=checkpoint(rr_cfg,cat,run,run.path/'checkpoints'/cat/'best.pt');reranker.to(run.device).eval();gallery=b['image']['features'].to(run.device);rows,x,_=data.batch(list(range(min(len(data.rows),cfg['export_batch_size']))),model,False,include_targets=False);x=x.to(run.device);measure=[]
 if run.device.type=='cuda':torch.cuda.reset_peak_memory_stats(run.device)
 for j in range(cfg['benchmark_warmup']+cfg['benchmark_steps']):
  run.check_stop();sync(run);tick=time.perf_counter()
  with torch.no_grad():
   q=model.extract_query([r['text'] for r in rows],x);sims=q@gallery.T
   for offset,row in enumerate(rows):sims[offset,data.gallery_index[row['source_id']]]=-torch.inf
   K=min(cfg['candidate_k'],len(gallery)-1);values,indices=sims.topk(K,dim=-1);scores=reranker(q,gallery[indices],values);final=values+rr_cfg['categories'][cat]['blend']*(scores-values);torch.argsort(final,dim=-1,descending=True)
  sync(run);measure.append({'iteration':j,'warmup':j<cfg['benchmark_warmup'],'seconds':time.perf_counter()-tick,'batch_size':len(rows)})
  if j%cfg['log_every']==0:run.say('online-benchmark',category=cat,**measure[-1])
 p=run.path/'profiling'/f'online-cost-{cat}.json';atomic_json(p,{'raw':measure,'device':str(run.device),'gpu':torch.cuda.get_device_name(run.device) if run.device.type=='cuda' else None,'peak_allocated_bytes':torch.cuda.max_memory_allocated(run.device) if run.device.type=='cuda' else None,'precision':'float32','cuda_synchronized':run.device.type=='cuda','scope':'preprocessed reference+text encoding + retrieval + RR, gallery cached offline, excludes disk image read/CPU preprocessing and gallery encoding; offline export timings recorded separately'});return [p]
