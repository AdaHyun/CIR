from __future__ import annotations
import copy, hashlib, json, math, os, random, shutil, time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from .assets import checkpoint, load_bundle, load_pt, paths
from .model import FeatureReranker, training_item
from .runtime import ROOT, Blocked, Paused, atomic_json, atomic_torch, fingerprint, read_json, relative, sha
CUTS=(5,10,20,30,50,100)
def write_jsonl(p,rows):
 p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_name(p.name+'.partial')
 with tmp.open('w',encoding='utf8') as f:
  for r in rows:f.write(json.dumps(r,ensure_ascii=False,default=str)+'\n')
  f.flush();os.fsync(f.fileno())
 os.replace(tmp,p)
def read_jsonl(p):
 with p.open(encoding='utf8') as f:return [json.loads(x) for x in f if x.strip()]
def identity_files(files):return {relative(p):{'sha256':sha(p),'bytes':p.stat().st_size} for p in files if p.is_file()}
def rank_of(ids,target):return ids.index(target)+1 if target in ids else None

def evaluate(cfg,cat,run,kind='old',model=None,bundle=None,tag=None,blend=None,persist=True):
 b=bundle or load_bundle(cfg,cat,'val',run);q,im,t=b['query'],b['image'],b['topk'];n=len(b['query_ids']);bs=cfg['eval_batch_size'];started=time.monotonic()
 ckpath=None
 if model is None and kind!='cn':
  ckpath=(run.path/'checkpoints'/cat/'best.pt') if kind=='new' else None
  if kind=='new' and not ckpath.exists():raise Blocked(f'new checkpoint unavailable for {cat}; training blocked or incomplete')
  model,ck,ckpath=checkpoint(cfg,cat,run,ckpath)
 if model is not None:model.to(run.device).eval()
 if blend is None:blend=cfg['categories'][cat]['blend'] if kind!='cn' else 0.
 label=tag or f'{kind}-{cat}';chunkdir=run.path/'predictions'/label/'chunks';chunkdir.mkdir(parents=True,exist_ok=True)
 inp=identity_files(b['files']+([ckpath] if ckpath else []))
 # In-memory models are validation-only and never reuse predictions from a previous epoch.
 token=fingerprint({'inputs':inp,'kind':kind,'blend':blend,'batch':bs,'config':cfg['categories'][cat],'schema':2})
 rows=[];raw_profile=[]
 for start in range(0,n,bs):
  stop=min(n,start+bs);block=chunkdir/f'{start:08d}.json';prior=read_json(block)
  if persist and prior:
   if prior['fingerprint']!=token:raise ValueError('evaluation chunk fingerprint changed')
   if fingerprint(prior['rows'])!=prior['rows_hash']:raise ValueError('corrupt committed evaluation chunk')
   rows.extend(prior['rows']);raw_profile.append(prior['profiling'])
   if (start//bs)%cfg['log_every']==0 or stop==n:run.say('block-skipped',category=cat,block=start,method=kind)
   continue
  run.check_stop();tick=time.perf_counter();cand_ids=t['sorted_ids'][start:stop]
  scores=torch.tensor(t['sorted_scores'][start:stop],dtype=torch.float32,device=run.device)
  if run.device.type=='cuda':torch.cuda.synchronize(run.device)
  inference_start=time.perf_counter()
  with torch.no_grad():
   if kind=='cn':final=scores
   else:
    c=im['features'][torch.tensor([[b['index'][x] for x in row] for row in cand_ids])].to(run.device).float()
    query=q['query_features'][start:stop].to(run.device).float()
    learned=model(query,c,scores);final=scores+blend*(learned-scores)
   order=torch.argsort(final,dim=-1,descending=True).cpu().tolist()
  if run.device.type=='cuda':torch.cuda.synchronize(run.device)
  inference_seconds=time.perf_counter()-inference_start
  vals=final.cpu().tolist();records=[]
  for offset,perm in enumerate(order):
   j=start+offset;ranked=[cand_ids[offset][k] for k in perm];r=rank_of(ranked,q['target_ids'][j]);base_rank=rank_of(cand_ids[offset],q['target_ids'][j])
   records.append({'query_id':b['query_ids'][j],'query_index':j,'category':cat,'split':'val','reference_image_id':q['source_ids'][j],'target_image_id':q['target_ids'][j],'modification_text':q.get('modification_texts',[None]*n)[j],'text_status':'from verified query asset metadata' if 'modification_texts' in q else 'missing original annotations','method':kind,'retrieved_image_ids':ranked,'scores':[vals[offset][k] for k in perm],'target_rank':r,'rank_status':'exact within Top-K' if r else f'outside Top-{len(ranked)}','rank_lower_bound':None if r else len(ranked)+1,'cn_topk_rank':base_rank,'gallery_protocol':'query-union / VAL-Split; reference excluded' if b['gallery_is_query_union'] else 'explicit gallery mapping; reference excluded','input_identity':inp,'blend':blend,'code_fingerprint':cfg['_code_fingerprint']})
  profile={'block':start,'samples':stop-start,'inference_seconds':inference_seconds,'block_seconds':time.perf_counter()-tick,'device':str(run.device),'kind':kind,'scope':'frozen-feature RR only; includes transfer/gather/sort, excludes backbone','warmup':start==0,**run.memory()}
  if persist:atomic_json(block,{'fingerprint':token,'rows_hash':fingerprint(records),'rows':records,'profiling':profile})
  rows.extend(records);raw_profile.append(profile)
  if (start//bs)%cfg['log_every']==0 or stop==n:run.progress(stop,n,started,category=cat,method=kind,batch=start//bs+1,total_batches=math.ceil(n/bs))
 count=len(rows);hits={str(k):sum(r['target_rank'] is not None and r['target_rank']<=k for r in rows) for k in CUTS};metrics={'count':count,'hits':hits,'recall':{str(k):hits[str(k)]*100/count for k in CUTS},'category':cat,'method':kind,'blend':blend}
 if not persist:return metrics,rows
 p=run.path/'metrics'/f'{label}.json';atomic_json(p,metrics);pred=run.path/'predictions'/f'{label}.jsonl';write_jsonl(pred,rows)
 prof=run.path/'profiling'/f'{label}.json';atomic_json(prof,{'raw':raw_profile,'total_inference_seconds':sum(x['inference_seconds'] for x in raw_profile),'scope':'frozen-feature RR only; not end-to-end cost'})
 run.say('evaluation-completed',**metrics);return [p,pred,prof]

def multik(cfg,cat,run):
 # CN uses the same argsort path as zero-blend legacy evaluation, preserving tie behavior.
 outputs=evaluate(cfg,cat,run,kind='cn');old=run.path/'metrics'/f'old-{cat}.json'
 if old.exists():
  a=read_json(outputs[0]);b=read_json(old);p=run.path/'metrics'/f'multik-{cat}.json';atomic_json(p,{'CN':a,'old_RR':b,'same_top100_invariance':a['hits']['100']==b['hits']['100'],'note':'CN zero blend argsort; archived dress coverage may differ by one R@10 hit due to tie/order path; never adjusted'});outputs.append(p)
 return outputs

def rng_state():return {'python':random.getstate(),'numpy':np.random.get_state(),'torch_cpu':torch.get_rng_state(),'torch_cuda':torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}
def restore_rng(x):
 random.setstate(x['python']);np.random.set_state(x['numpy']);torch.set_rng_state(x['torch_cpu'])
 if x['torch_cuda'] is not None:
  if not torch.cuda.is_available():raise ValueError('CUDA checkpoint requires matching CUDA runtime for training resume')
  torch.cuda.set_rng_state_all(x['torch_cuda'])
def train(cfg,cat,run,stop_after=None):
 c={**cfg['training'],**cfg['categories'][cat]}
 if c['train_k']<2 or c['train_pool']<c['train_k'] or c['save_every_steps']<1 or c['save_every_seconds']<=0:raise ValueError('invalid candidate/save configuration')
 b=load_bundle(cfg,cat,'train',run);val=load_bundle(cfg,cat,'val',run)
 if b['query']['query_features'].shape[1]!=val['query']['query_features'].shape[1]:raise ValueError('train/val dimensions differ')
 provenance=Path(cfg['asset_dir'])/cat/'provenance.json'
 if not provenance.exists():raise Blocked('train/val feature-space evidence required: assets/<category>/provenance.json; see MISSING_ASSETS.md')
 prov=read_json(provenance)
 for split,bun in [('train',b),('val',val)]:
  producer=prov.get(split,{}).get('producer_checkpoint_sha256','')
  if not cfg.get('fixture',False) and (len(producer)!=64 or any(x not in '0123456789abcdefABCDEF' for x in producer)):raise Blocked('producer identity must be a real 64-character SHA-256; template placeholders are not evidence')
  if not prov.get(split,{}).get('producer_checkpoint_sha256'):raise Blocked('missing producer checkpoint identity')
  for p in bun['files']:
   if prov[split].get('files',{}).get(p.name)!=sha(p):raise ValueError('provenance file hash mismatch: '+p.name)
 if prov['train']['producer_checkpoint_sha256']!=prov['val']['producer_checkpoint_sha256']:raise ValueError('train/val feature producers differ')
 if prov['train'].get('feature_definition')!=prov['val'].get('feature_definition') or not prov['train'].get('feature_definition'):raise ValueError('train/val feature definition differs/missing')
 random.seed(c['seed']);np.random.seed(c['seed']);torch.manual_seed(c['seed'])
 if torch.cuda.is_available():torch.cuda.manual_seed_all(c['seed'])
 torch.use_deterministic_algorithms(c.get('deterministic',True))
 model=FeatureReranker(b['query']['query_features'].shape[1],c['hidden_dim'],c['dropout'],c['base_score_scale']).to(run.device)
 optimizer=torch.optim.AdamW(model.parameters(),lr=c['lr'],weight_decay=c['weight_decay'])
 files=b['files']+val['files']+[provenance];ident={'inputs':identity_files(files),'config':c,'code':cfg['_code_fingerprint'],'device_type':run.device.type,'torch':torch.__version__}
 token=fingerprint(ident);d=run.path/'checkpoints'/cat;d.mkdir(parents=True,exist_ok=True);last=d/'last.pt';best_path=d/'best.pt'
 epoch=cursor=step=0;best=-math.inf;epoch_loss=0.;epoch_items=0;step_rows=[];epoch_rows=[];best_blob=None;last_save=time.monotonic();total_steps=math.ceil(len(b['query_ids'])/c['batch_size'])*c['epochs'];started=time.monotonic()
 if last.exists():
  state=load_pt(last,run)
  if state['fingerprint']!=token:raise ValueError('training checkpoint config/assets/model/environment mismatch')
  model.load_state_dict(state['model_state_dict'],strict=True);optimizer.load_state_dict(state['optimizer']);epoch=state['epoch'];cursor=state['cursor'];step=state['global_step'];best=state['best_score'];epoch_loss=state['epoch_loss'];epoch_items=state['epoch_items'];step_rows=state['step_records'];epoch_rows=state['epoch_records'];best_blob=state['best_checkpoint'];restore_rng(state['rng']);run.say('training-restored',category=cat,epoch=epoch,global_step=step,cursor=cursor)
 initial_step=step
 def materialize():
  write_jsonl(run.path/'metrics'/f'train-steps-{cat}.jsonl',step_rows);write_jsonl(run.path/'metrics'/f'train-epochs-{cat}.jsonl',epoch_rows)
  if best_blob is not None:atomic_torch(best_path,best_blob)
 def save():
  nonlocal last_save
  state={'model_state_dict':model.state_dict(),'optimizer':optimizer.state_dict(),'scheduler':None,'amp_grad_scaler':None,'epoch':epoch,'cursor':cursor,'global_step':step,'sampler':{'seed':c['seed'],'epoch':epoch,'cursor':cursor,'strategy':'epoch randperm; per-query SHA256 candidate seed; num_workers=0; no prefetch'},'rng':rng_state(),'best_score':best,'selection_rule':'max unblended validation R@10 + R@50; strict improvement','epoch_loss':epoch_loss,'epoch_items':epoch_items,'args':c,'effective_config':cfg,'fingerprint':token,'input_identity':ident,'step_records':step_rows,'epoch_records':epoch_rows,'best_checkpoint':best_blob}
  atomic_torch(last,state);materialize();last_save=time.monotonic();run.say('checkpoint-saved',path=relative(last),category=cat,epoch=epoch,global_step=step,cursor=cursor)
 materialize()
 while epoch<c['epochs']:
  order=torch.randperm(len(b['query_ids']),generator=torch.Generator().manual_seed(c['seed']+epoch)).tolist();model.train()
  while cursor<len(order):
   if run.stop:save();raise Paused('training interrupted at completed optimizer step')
   indices=order[cursor:cursor+c['batch_size']];items=[training_item(b,i,epoch,c) for i in indices]
   query=torch.stack([x[0] for x in items]).to(run.device);candidates=torch.stack([x[1] for x in items]).to(run.device);scores=torch.stack([x[2] for x in items]).to(run.device);labels=torch.tensor([x[3] for x in items],device=run.device)
   tick=time.perf_counter();optimizer.zero_grad(set_to_none=True);loss=F.cross_entropy(model(query,candidates,scores),labels);loss.backward();optimizer.step()
   if run.device.type=='cuda':torch.cuda.synchronize(run.device)
   value=float(loss.detach());cursor+=len(indices);step+=1;epoch_loss+=value*len(indices);epoch_items+=len(indices)
   step_rows.append({'global_step':step,'epoch':epoch+1,'cursor':cursor,'query_ids':[b['query_ids'][i] for i in indices],'candidate_ids':[x[4] for x in items],'labels':[x[3] for x in items],'loss':value,'average_loss':epoch_loss/epoch_items,'lr':optimizer.param_groups[0]['lr'],'step_seconds':time.perf_counter()-tick})
   if step%cfg['log_every']==0 or cursor==len(order):run.progress(step,total_steps,started,completed_before=initial_step,category=cat,epoch=epoch+1,epochs=c['epochs'],epoch_processed=cursor,epoch_samples=len(order),loss=value,average_loss=epoch_loss/epoch_items,lr=optimizer.param_groups[0]['lr'])
   if step%c['save_every_steps']==0 or time.monotonic()-last_save>=c['save_every_seconds']:save()
   if stop_after is not None and step>=stop_after:save();raise Paused('explicit short-test interruption')
  save() # completed optimizer epoch survives interruption during validation
  metrics,_=evaluate(cfg,cat,run,kind='new',model=model,bundle=val,tag=f'validation-{cat}',blend=1.,persist=False)
  score=metrics['recall']['10']+metrics['recall']['50'];record={'epoch':epoch+1,'global_step':step,'loss':epoch_loss/epoch_items,'selection_score':score,'validation':metrics};epoch_rows.append(record)
  if score>best:
   best=score
   best_blob={'model_state_dict':copy.deepcopy(model.state_dict()),'args':c,'optimizer':copy.deepcopy(optimizer.state_dict()),'scheduler':None,'amp_grad_scaler':None,'epoch':epoch+1,'cursor':0,'global_step':step,'rng':rng_state(),'best_score':best,'selection_rule':'max unblended validation R@10 + R@50','effective_config':cfg,'input_identity':ident,'fingerprint':token,'best_metrics':metrics,'sampler':{'seed':c['seed'],'epoch':epoch+1,'cursor':0},'purpose':'best model selection; resume training from last.pt'}
   run.say('best-updated',category=cat,epoch=epoch+1,score=best)
  run.say('epoch-validation',category=cat,**record);epoch+=1;cursor=0;epoch_loss=0.;epoch_items=0;save()
 return [last,best_path,run.path/'metrics'/f'train-steps-{cat}.jsonl',run.path/'metrics'/f'train-epochs-{cat}.jsonl']

def build_topk(cfg,cat,run,split):
 b=load_bundle(cfg,cat,split,run,require_topk=False);q,im=b['query'],b['image'];Q=q['query_features'].float();I=im['features'].float();n=len(Q);size=cfg['topk_block_size'];k=min(cfg['topk'],len(I)-1);directory=run.path/'predictions'/f'topk-{cat}-{split}';directory.mkdir(parents=True,exist_ok=True)
 token=fingerprint({'inputs':identity_files(b['files']),'k':k,'block_size':size,'reference_exclusion':True});rows=[];started=time.monotonic()
 for start in range(0,n,size):
  p=directory/f'{start:08d}.json';chunk=read_json(p)
  if chunk:
   if chunk['fingerprint']!=token or fingerprint(chunk['rows'])!=chunk['rows_hash']:raise ValueError('topk block identity mismatch')
   rows.extend(chunk['rows']);continue
  run.check_stop();sims=Q[start:start+size]@I.T
  block=[]
  for j in range(len(sims)):
   ix=start+j;sims[j,b['index'][q['source_ids'][ix]]]=-torch.inf
   values,indices=torch.topk(sims[j],k,sorted=True);ids=[im['image_ids'][i] for i in indices.tolist()]
   block.append({'sorted_ids':ids,'sorted_scores':values.tolist(),'source_id':q['source_ids'][ix],'target_id':q['target_ids'][ix]})
  atomic_json(p,{'fingerprint':token,'rows_hash':fingerprint(block),'rows':block});rows.extend(block);run.progress(min(n,start+size),n,started,category=cat,phase='topk',split=split)
 payload={'sorted_ids':[r['sorted_ids'] for r in rows],'sorted_scores':[r['sorted_scores'] for r in rows],'source_ids':q['source_ids'],'target_ids':q['target_ids'],'target_in_topk':[r['target_id'] in r['sorted_ids'] for r in rows]}
 out=run.path/'reports'/f'{cat}_{split}_topk.pt';atomic_torch(out,payload);return [out]

def diagnostics(cfg,cat,run):
 cn=run.path/'predictions'/f'cn-{cat}.jsonl';old=run.path/'predictions'/f'old-{cat}.jsonl'
 if not cn.exists() or not old.exists():raise Blocked('requires completed CN and old RR predictions')
 baseline=read_jsonl(cn);outputs=[];report={}
 for method in ['old','new']:
  p=run.path/'predictions'/f'{method}-{cat}.jsonl'
  if not p.exists():report[method]={'status':'blocked','reason':'new checkpoint/evaluation unavailable'};continue
  records=read_jsonl(p);assert len(records)==len(baseline)
  changes={str(k):{'corrected':0,'harmed':0} for k in CUTS};fails=[]
  for a,b in zip(baseline,records):
   assert a['query_id']==b['query_id'] and a['gallery_protocol']==b['gallery_protocol']
   for k in CUTS:
    before=a['target_rank'] is not None and a['target_rank']<=k;after=b['target_rank'] is not None and b['target_rank']<=k
    changes[str(k)]['corrected']+=int(after and not before);changes[str(k)]['harmed']+=int(before and not after)
   if b['target_rank'] is None or b['target_rank']>10:fails.append(b)
  report[method]={'changes':changes,'count':len(records),'failed_at_10':len(fails)};f=run.path/'reports'/f'failure-queries-{method}-{cat}.jsonl';write_jsonl(f,fails);outputs.append(f)
 out=run.path/'reports'/f'diagnostics-{cat}.json';atomic_json(out,report);return outputs+[out]

def visualization(cfg,cat,run):
 # Enrich complete per-query predictions only via exact reference/target identity matches.
 annotation=Path(cfg['data_dir'])/'captions'/f'cap.{cat}.val.json';image_root=Path(cfg['data_dir'])/'resized_image'/cat
 if not annotation.exists():raise Blocked('original modification texts/annotations absent; full CN/old RR ID+score predictions retained, no fabricated text/images; DQU predictions absent')
 ann=read_json(annotation);lookup={}
 for x in ann:lookup.setdefault((x['candidate'],x['target']),[]).append(x)
 outputs=[]
 for method in ['dqu','cn','old','new']:
  p=run.path/'predictions'/f'{method}-{cat}.jsonl'
  if not p.exists():continue
  rows=read_jsonl(p)
  for r in rows:
   matches=lookup.get((r['reference_image_id'],r['target_image_id']),[])
   if len(matches)!=1:raise Blocked('ambiguous or missing annotation identity; supply explicit query mapping instead of guessing')
   r['modification_text']=matches[0]['captions'];r['text_status']='verified reference+target annotation match'
  out=run.path/'predictions'/f'visualization-{method}-{cat}.jsonl';write_jsonl(out,rows);outputs.append(out)
 if not image_root.is_dir():raise Blocked('full text-enriched records exported; real FashionIQ images absent; image visualization blocked')
 # Save reproducible HTML with actual local images copied into the run, no invented retrievals.
 cases=run.path/'reports'/f'visualization-{cat}';cases.mkdir(parents=True,exist_ok=True)
 import html
 body=['<meta charset="utf-8"><h1>Available verified methods (missing methods omitted explicitly in status report)</h1>']
 available=[(method,read_jsonl(run.path/'predictions'/f'visualization-{method}-{cat}.jsonl')) for method in ['dqu','cn','old','new'] if (run.path/'predictions'/f'visualization-{method}-{cat}.jsonl').exists()]
 selected=[j for j,r in enumerate(available[0][1]) if r['target_rank'] is None or r['target_rank']>10][:cfg.get('visualization_cases',12)]
 for j in selected:
  body.append('<h2>'+html.escape(available[0][1][j]['query_id'])+'</h2>')
  for method,rows in available:
   r=rows[j];body.append('<p>'+html.escape(method+' '+str(r['modification_text']))+'</p>')
   for imageid in [r['reference_image_id'],r['target_image_id']]+r['retrieved_image_ids'][:5]:
    src=image_root/(imageid+'.jpg')
    if not src.exists():raise Blocked(f'missing real image: {imageid}')
    dest=cases/src.name;shutil.copy2(src,dest);body.append(f'<figure style="display:inline-block"><img width="120" src="{html.escape(src.name)}"><figcaption>{html.escape(imageid)}</figcaption></figure>');outputs.append(dest)
 out=cases/'index.html';out.write_text('\n'.join(body),encoding='utf8');outputs.append(out);return outputs

def cost(cfg,cat,run):
 p=run.path/'profiling'/f'old-{cat}.json'
 if not p.exists():raise Blocked('requires old RR evaluation profiling')
 x=read_json(p);raw=x['raw'];timed=[r for r in raw if not r['warmup']];sec=sum(r['inference_seconds'] for r in timed);count=sum(r['samples'] for r in timed)
 result={'scope':'RR with frozen features only, includes transfer/gather/sort; not DQU/CN or whole system','device':str(run.device),'warmup_blocks_excluded':1,'samples':count,'seconds':sec,'samples_per_second':count/sec if sec else None,'training_cost_status':'blocked without true training assets; fixtures are not experimental timings','raw_source':relative(p)}
 out=run.path/'profiling'/f'cost-{cat}.json';atomic_json(out,result);return [out]

def summary(cfg,categories,run):
 states=read_json(run.path/'reports'/'stage_state.json',{});result={'stage_states':{k:v for k,v in states.items() if k!='summary'},'macro':{},'limitations':['No A100 validation','Frozen features only; training assets absent unless explicitly supplied','DQU per-query predictions/weights missing','Raw images/annotations missing in current material']}
 for method in ['cn','old','new']:
  files=[run.path/'metrics'/f'{method}-{c}.json' for c in categories]
  if not all(p.exists() for p in files):result['macro'][method]={'status':'blocked','reason':'not all selected categories evaluated'};continue
  vals=[read_json(p) for p in files];r={str(k):sum(x['recall'][str(k)] for x in vals)/len(vals) for k in CUTS}
  result['macro'][method]={'categories':categories,'equal_weight':True,'recall':r,'mR':(r['10']+r['50'])/2,'source_files':[relative(p) for p in files]}
 p=run.path/'reports'/'summary.json';atomic_json(p,result);run.say('summary',macro=result['macro']);return [p]


def dqu_export(cfg,cat,run):
 from .runtime import resolve
 path=resolve(cfg.get('dqu_predictions',{}).get(cat,f'predictions/dqu/{cat}.jsonl'))
 if not path.exists():raise Blocked('DQU complete query-level predictions, DQU frozen features and DQU checkpoint all absent. Aggregate metrics and case PNG files cannot reconstruct them.')
 b=load_bundle(cfg,cat,'val',run);rows=read_jsonl(path);query=b['query'];lookup={}
 for row in rows:
  key=(row['reference_image_id'],row['target_image_id'])
  if key in lookup:raise ValueError('ambiguous DQU reference/target mapping; provide explicit query IDs')
  ids=row['retrieved_image_ids'];scores=row['scores']
  assert len(ids)==len(scores)>=100 and len(set(ids))==len(ids)
  assert row['reference_image_id'] not in ids
  assert set(ids)<=set(b['index']), 'DQU gallery differs'
  assert all(math.isfinite(float(x)) for x in scores)
  assert row['gallery_protocol']=='query-union / VAL-Split; reference excluded'
  assert all(float(scores[j])>=float(scores[j+1]) for j in range(len(scores)-1)), 'DQU scores not sorted'
  assert row.get('checkpoint_sha256') or row.get('feature_sha256'), 'DQU provenance missing'
  lookup[key]=row
 if len(rows)!=len(b['query_ids']):raise ValueError('DQU export incomplete; all queries must remain in denominator')
 out=[]
 for j,qid in enumerate(b['query_ids']):
  key=(query['source_ids'][j],query['target_ids'][j])
  if key not in lookup:raise ValueError('DQU query identity does not match CN')
  row=dict(lookup[key]);row.update(query_id=qid,query_index=j,method='dqu',category=cat,code_fingerprint=cfg['_code_fingerprint'],input_identity=identity_files([path]))
  rank=rank_of(row['retrieved_image_ids'],row['target_image_id']);row['target_rank']=rank;row['rank_status']='exact within Top-K' if rank else 'outside Top-K';row['rank_lower_bound']=None if rank else len(row['retrieved_image_ids'])+1
  out.append(row)
 p=run.path/'predictions'/f'dqu-{cat}.jsonl';write_jsonl(p,out);return [p]
