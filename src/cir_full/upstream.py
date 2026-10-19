"""Raw-image DQU -> reference-conditioned model -> StageP/CN -> RR.
Historical formulas are retained; changed engineering choices are recorded in config.
"""
from __future__ import annotations
import copy,json,math,random,time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from .runtime import ROOT,Blocked,Paused,atomic_json,atomic_torch,fingerprint,read_json,relative,resolve,sha
from .assets import load_pt
from .engine import rng_state,restore_rng,write_jsonl,read_jsonl,identity_files
from .raw_data import FashionData
from .upstream_model import DQU_CIR,DQU_PFP_REFCOND_CIR,DQU_PFP_CIR

def sync(run):
 if run.device.type=='cuda':torch.cuda.synchronize(run.device)
def model_for(cfg,method,run,checkpoint=None):
 cls=DQU_CIR if method=='dqu' else {'dqu':DQU_CIR,'dqu_pfp':DQU_PFP_CIR,'dqu_pfp_refcond':DQU_PFP_REFCOND_CIR}[cfg.get('rc_variant','dqu_pfp_refcond')]
 start=time.perf_counter();model=cls(hidden_dim=cfg['dimension'],dropout=cfg['dropout'],backbone=cfg['backbone'],pretrained=cfg['pretrained']).to(run.device)
 if checkpoint:
  c=load_pt(checkpoint,run);model.load_state_dict(c['model_state_dict'],strict=True)
 sync(run);duration=time.perf_counter()-start;record={'method':method,'stage':run.stage,'seconds':duration,'backbone':cfg['backbone'],'device':str(run.device),'precision':'float32','cuda_synchronized':run.device.type=='cuda','time':time.time()}
 prof=run.path/'profiling/model-loads.jsonl';prior=read_jsonl(prof) if prof.exists() else [];write_jsonl(prof,prior+[record]);run.say('model-loaded',**record,**run.memory());return model

def committed_chunk(path,token):
 commit=read_json(path.with_suffix('.commit.json'))
 if commit:
  if commit['fingerprint']!=token or not path.is_file() or sha(path)!=commit['sha256']:raise ValueError('committed chunk changed: '+relative(path))
  return load_pt(path)
 return None
def save_chunk(path,value,token):
 atomic_torch(path,value);atomic_json(path.with_suffix('.commit.json'),{'fingerprint':token,'sha256':sha(path)})
def ckpt_path(run,method,cat):return run.path/'checkpoints'/f'{method}-{cat}'/'best.pt'

def encode_split(cfg,cat,split,model,checkpoint,method,run):
 producer_sha=sha(checkpoint)
 data=FashionData(cfg,cat,split);base=run.path/'assets'/method/cat;base.mkdir(parents=True,exist_ok=True)
 token=fingerprint({'checkpoint':producer_sha,'data':data.identity,'export':'deterministic validation preprocessing both splits','config':cfg['query'],'method':method,'code':cfg['_code_fingerprint']})
 bs=cfg['export_batch_size'];started=time.monotonic();image_features=[];qfeatures=[];textfeatures=[];model.eval()
 for kind,items in [('image',data.gallery_ids),('query',data.rows)]:
  for start in range(0,len(items),bs):
   run.check_stop();p=base/'blocks'/f'{split}-{kind}-{start:08d}.pt';p.parent.mkdir(parents=True,exist_ok=True);chunk=committed_chunk(p,token)
   if chunk is None:
    sync(run);compute_start=time.perf_counter()
    with torch.no_grad():
     if kind=='image':
      ids=items[start:start+bs];x=torch.stack([data.image(x,model.preprocess_val) for x in ids]).to(run.device);chunk={'ids':ids,'features':model.extract_target(x).float().cpu()}
     else:
      rows,x,_=data.batch(list(range(start,min(start+bs,len(items)))),model,train=False,include_targets=False);texts=[x['text'] for x in rows];chunk={'rows':rows,'features':model.extract_query(texts,x.to(run.device)).float().cpu(),'text_features':model.encode_textual_query(texts).float().cpu()}
    sync(run);chunk['compute_seconds']=time.perf_counter()-compute_start
    assert torch.isfinite(chunk['features']).all();save_chunk(p,chunk,token)
   if kind=='image':image_features.append(chunk['features'])
   else:qfeatures.append(chunk['features']);textfeatures.append(chunk['text_features'])
   if (start//bs)%cfg['log_every']==0 or start+bs>=len(items):run.progress(min(start+bs,len(items)),len(items),started,category=cat,method=method,split=split,encoding=kind)
 Q=torch.cat(qfeatures);I=torch.cat(image_features);query={'query_features':Q,'text_features':torch.cat(textfeatures),'query_ids':[r['query_id'] for r in data.rows],'source_ids':[r['source_id'] for r in data.rows],'target_ids':[r['target_id'] for r in data.rows],'modification_texts':[r['modification_text'] for r in data.rows],'metadata':{'producer_sha256':producer_sha,'method':method,'data_fingerprint':data.fingerprint,'split':split,'query_config':cfg['query'],'backbone':cfg['backbone'],'normalization':'L2','preprocessing':'deterministic validation transform','dimension':Q.shape[1]}}
 gallery={'features':I,'image_ids':data.gallery_ids,'metadata':query['metadata']};files=[]
 for kind,x in [('query_features',query),('image_features',gallery)]:
  p=base/f'{cat}_{split}_{kind}.pt';atomic_torch(p,x);files.append(p)
 # Exact full-gallery ranking in query blocks; do not allocate all N x M similarities.
 K=min(cfg['candidate_k'],len(I)-1);records=[]
 for start in range(0,len(Q),bs):
  run.check_stop();p=base/'blocks'/f'{split}-topk-{start:08d}.pt';chunk=committed_chunk(p,token+f':K={K}')
  if chunk is None:
   sims=Q[start:start+bs]@I.T;chunk=[]
   for j,row in enumerate(sims):
    r=data.rows[start+j];row[data.gallery_index[r['source_id']]]=-torch.inf
    order=torch.argsort(row,descending=True,stable=True);indices=order[:K].tolist();target=data.gallery_index[r['target_id']];rank=int((order==target).nonzero()[0])+1
    chunk.append({'ids':[data.gallery_ids[x] for x in indices],'scores':row[indices].tolist(),'rank':rank})
   save_chunk(p,chunk,token+f':K={K}')
  records.extend(chunk)
  if (start//bs)%cfg['log_every']==0:run.progress(min(start+bs,len(Q)),len(Q),started,category=cat,phase='full-gallery-topk',K=K)
 top={'sorted_ids':[r['ids'] for r in records],'sorted_scores':[r['scores'] for r in records],'source_ids':query['source_ids'],'target_ids':query['target_ids'],'target_ranks':[r['rank'] for r in records],'target_in_topk':[r['rank']<=K for r in records],'metadata':query['metadata']}
 p=base/f'{cat}_{split}_topk.pt';atomic_torch(p,top);files.append(p)
 provenance=read_json(base/'provenance.json',{});provenance[split]={'producer_checkpoint_sha256':producer_sha,'feature_definition':{'model':method,'backbone':cfg['backbone'],'query':cfg['query'],'preprocess':'deterministic validation transform','normalization':'L2','dimension':Q.shape[1]},'data_fingerprint':data.fingerprint,'gallery_protocol':'query-union; reference excluded','files':{p.name:sha(p) for p in files}};atomic_json(base/'provenance.json',provenance)
 if split=='val':
  rows=[]
  for j,r in enumerate(data.rows):
   record=records[j];rank=record['rank'] if record['rank']<=K else None
   rows.append({'query_id':r['query_id'],'query_index':j,'category':cat,'reference_image_id':r['source_id'],'target_image_id':r['target_id'],'modification_text':r['modification_text'],'retrieved_image_ids':record['ids'],'scores':record['scores'],'target_rank':rank,'rank_status':'exact within Top-K' if rank else f'outside Top-{K}','rank_lower_bound':None if rank else K+1,'gallery_protocol':'query-union / VAL-Split; reference excluded','checkpoint_sha256':producer_sha,'feature_sha256':sha(files[0]),'code_fingerprint':cfg['_code_fingerprint'],'method':method})
  p=run.path/'predictions'/f'{method}-{cat}.jsonl';write_jsonl(p,rows);files.append(p)
  hits={str(k):sum(x['target_rank'] is not None and x['target_rank']<=k for x in rows) for k in cfg['cutoffs'] if k<=K};out=run.path/'metrics'/f'{method}-{cat}.json';atomic_json(out,{'count':len(rows),'hits':hits,'recall':{k:v*100/len(rows) for k,v in hits.items()},'K':K});files.append(out)
 files.append(base/'provenance.json');return files

def export(cfg,cat,method,run):
 path=ckpt_path(run,method,cat)
 if not path.exists():raise Blocked('prerequisite trained checkpoint missing '+relative(path))
 model=model_for(cfg,method,run,path);outputs=[]
 for split in ['train','val']:outputs+=encode_split(cfg,cat,split,model,path,method,run)
 profile=run.path/'profiling'/f'offline-export-{method}-{cat}.json'
 blocks=[]
 for p in (run.path/'assets'/method/cat/'blocks').glob('*.pt'):
  if '-topk-' not in p.name:
   x=load_pt(p);blocks.append({'block':p.name,'compute_seconds':x.get('compute_seconds'),'rows':len(x['features'])})
 atomic_json(profile,{'raw_blocks':blocks,'scope':'offline gallery/query/text blocks: includes image read/preprocessing+encoding+transfer; excludes model construction/input hashing/topK/export IO','precision':'float32','device':str(run.device)})
 return list(dict.fromkeys(outputs))+[profile]

def bank(cfg,cat,run):
 bundles=[]
 for method in ['rc','dqu']:
  p=run.path/'assets'/method/cat
  files=[p/f'{cat}_train_{kind}.pt' for kind in ['query_features','image_features']]
  if not all(p.exists() for p in files):raise Blocked('teacher training features missing: '+method)
  provenance=read_json(p/'provenance.json',{}).get('train',{}).get('files',{})
  for x in files:
   if provenance.get(x.name)!=sha(x):raise ValueError('teacher feature integrity/provenance mismatch: '+x.name)
  bundles.append([load_pt(x,run) for x in files])
 (qa,ia),(qb,ib)=bundles
 assert qa['query_ids']==qb['query_ids'] and ia['image_ids']==ib['image_ids'],'teachers must share exact IDs, not shapes'
 ids=ia['image_ids'];lookup={v:i for i,v in enumerate(ids)};N=len(qa['query_ids']);bs=cfg['export_batch_size'];records=[];token=fingerprint({'a':qa['metadata'],'b':qb['metadata'],'config':cfg['bank']})
 started=time.monotonic();directory=run.path/'assets/banks'/cat;directory.mkdir(parents=True,exist_ok=True)
 for start in range(0,N,bs):
  p=directory/f'{start:08d}.pt';chunk=committed_chunk(p,token)
  if chunk is None:
   run.check_stop();chunk=[]
   sources=qa['source_ids'];targets=qa['target_ids'];scores=[qa['query_features'][start:start+bs]@ia['features'].T,qb['query_features'][start:start+bs]@ib['features'].T,qa['text_features'][start:start+bs]@ia['features'].T]
   for offset in range(len(scores[0])):
    j=start+offset;fused=torch.zeros(len(ids))
    for sims in scores:
     row=sims[offset].clone();row[lookup[sources[j]]]=-torch.inf;order=torch.argsort(row,descending=True,stable=True);ranks=torch.empty_like(order);ranks[order]=torch.arange(1,len(ids)+1);fused+=1/(60+ranks.float())
    fused[lookup[sources[j]]]=-torch.inf;order=torch.argsort(fused,descending=True,stable=True).tolist();target=targets[j];rank=order.index(lookup[target])+1
    negatives=[ids[x] for x in order if ids[x] not in [target,sources[j]]];soft=[target]+negatives[:cfg['bank']['size']-1]
    if len(soft)!=cfg['bank']['size']:raise ValueError('gallery too small for configured bank')
    z=torch.tensor([float(fused[lookup[x]]) for x in soft[1:]]);weights=[.55]+(.45*torch.softmax((z-z.max())*50,0)).tolist()
    chunk.append({'query_id':qa['query_ids'][j],'source_id':sources[j],'target_id':target,'use_bank':rank<=cfg['bank']['rank_gate'],'target_rank':rank,'soft_ids':soft,'soft_weights':weights,'conflict_id':negatives[0]})
   save_chunk(p,chunk,token)
  records.extend(chunk)
  run.progress(min(start+bs,N),N,started,category=cat,phase='teacher-bank')
 out=run.path/'assets/banks'/f'{cat}.jsonl';write_jsonl(out,records);return [out]

def validate_model(cfg,data,model,run):
 model.eval();bs=cfg['export_batch_size'];started=time.monotonic();images=[];queries=[]
 with torch.no_grad():
  for start in range(0,len(data.gallery_ids),bs):
   run.check_stop();x=torch.stack([data.image(i,model.preprocess_val) for i in data.gallery_ids[start:start+bs]]).to(run.device);images.append(model.extract_target(x).float().cpu())
   if (start//bs)%cfg['log_every']==0:run.progress(min(start+bs,len(data.gallery_ids)),len(data.gallery_ids),started,phase='validation-gallery')
  for start in range(0,len(data.rows),bs):
   run.check_stop();rows,x,_=data.batch(list(range(start,min(start+bs,len(data.rows)))),model,False,include_targets=False);queries.append(model.extract_query([r['text'] for r in rows],x.to(run.device)).float().cpu())
   if (start//bs)%cfg['log_every']==0:run.progress(min(start+bs,len(data.rows)),len(data.rows),started,phase='validation-query')
 I=torch.cat(images);Q=torch.cat(queries);hits={str(k):0 for k in cfg['cutoffs']}
 for start in range(0,len(Q),bs):
  sims=Q[start:start+bs]@I.T
  for offset,row in enumerate(sims):
   q=data.rows[start+offset];row[data.gallery_index[q['source_id']]]=-torch.inf;order=torch.argsort(row,descending=True,stable=True);rank=int((order==data.gallery_index[q['target_id']]).nonzero()[0])+1
   for k in cfg['cutoffs']:hits[str(k)]+=int(rank<=k)
 sync(run);return {'count':len(Q),'hits':hits,'recall':{k:v*100/len(Q) for k,v in hits.items()},'validation_seconds':time.monotonic()-started}

def train(cfg,cat,method,run,pause_after=None,benchmark=False):
 train_data=FashionData(cfg,cat,'train');val_data=FashionData(cfg,cat,'val');c=cfg['stages'][method];batch_size=c.get('batch_size',cfg['batch_size']);decay=c.get('lr_decay',cfg['lr_decay']);max_decay=c.get('max_decay_epoch',cfg['max_decay_epoch']);patience=c.get('patience',cfg['patience'])
 random.seed(cfg['seed']);np.random.seed(cfg['seed']);torch.manual_seed(cfg['seed']);torch.use_deterministic_algorithms(cfg.get('deterministic',True))
 initial=ckpt_path(run,'rc',cat) if method=='cn' else None
 if initial and not initial.exists():raise Blocked('CN initializes from newly trained RC, missing '+relative(initial))
 if benchmark and run.device.type=='cuda':torch.cuda.reset_peak_memory_stats(run.device)
 model=model_for(cfg,method,run,initial);model.train()
 groups=[{'params':[p for n,p in model.named_parameters() if n.startswith('clip.')],'lr':c['clip_lr']},{'params':[p for n,p in model.named_parameters() if not n.startswith('clip.')],'lr':c['lr']}]
 optimizer=torch.optim.AdamW(groups,weight_decay=cfg['weight_decay'],eps=1e-8)
 # Same historical decay condition; the schedule state is explicit and resumable.
 scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,lambda e:cfg['lr_div']**sum(1 for j in range(1,e+1) if (j+1)%decay==0 and j<max_decay))
 amp=bool(cfg['amp'] and run.device.type=='cuda');scaler=torch.cuda.amp.GradScaler(enabled=amp)
 bank_rows=None;bank_path=run.path/'assets/banks'/f'{cat}.jsonl'
 if method=='cn':
  if not bank_path.exists():raise Blocked('teacher bank missing')
  bank_rows=read_jsonl(bank_path)
  if [r['query_id'] for r in bank_rows]!=[r['query_id'] for r in train_data.rows]:raise ValueError('bank/train query identity mismatch')
 ident={'train':train_data.fingerprint,'val':val_data.fingerprint,'model':method,'config':cfg,'initial':sha(initial) if initial else None,'bank':sha(bank_path) if bank_rows is not None else None,'torch':torch.__version__,'device_type':run.device.type};token=fingerprint(ident)
 directory=run.path/'checkpoints'/f'{method}-{cat}';directory.mkdir(parents=True,exist_ok=True);last=directory/'latest.pt';best=directory/'best.pt'
 epoch=cursor=step=bad_epochs=0;best_score=-math.inf;history=[];steps=[];best_state=None;loss_total=0.;items_total=0;last_saved=time.monotonic();started=time.monotonic()
 if last.exists():
  x=load_pt(last,run)
  if x['fingerprint']!=token:raise ValueError('upstream resume configuration/data/code/device incompatible')
  model.load_state_dict(x['model_state_dict'],strict=True);optimizer.load_state_dict(x['optimizer']);scheduler.load_state_dict(x['scheduler']);scaler.load_state_dict(x['amp']);epoch=x['epoch'];cursor=x['cursor'];step=x['global_step'];bad_epochs=x['bad_epochs'];best_score=x['best_score'];history=x['history'];steps=x['steps'];best_state=None;loss_total=x['loss_total'];items_total=x['items_total'];restore_rng(x['rng']);run.say('upstream-restored',method=method,category=cat,epoch=epoch,global_step=step,cursor=cursor)
 if last.exists() and x.get('pending_best'):
  recovered={k:v for k,v in x.items() if k not in ['steps','history','pending_best']};recovered.update(method=method,config=cfg,selection_score=best_score);atomic_torch(best,recovered);run.say('best-recovered-from-latest',path=relative(best))
 initial_step=step
 def save():
  nonlocal last_saved,best_state
  state={'model_state_dict':model.state_dict(),'optimizer':optimizer.state_dict(),'scheduler':scheduler.state_dict(),'amp':scaler.state_dict(),'rng':rng_state(),'epoch':epoch,'cursor':cursor,'global_step':step,'bad_epochs':bad_epochs,'best_score':best_score,'selection_rule':'unrounded val R@10+R@50 strict improvement','history':history,'steps':steps,'best_state':None,'pending_best':best_state is not None,'loss_total':loss_total,'items_total':items_total,'config':cfg,'fingerprint':token,'input_identity':ident,'sampler':{'seed':cfg['seed'],'epoch':epoch,'cursor':cursor,'workers':0}}
  atomic_torch(last,state)
  if best_state:
   atomic_torch(best,best_state);best_state=None
  write_jsonl(run.path/'metrics'/f'{method}-{cat}-steps.jsonl',steps);write_jsonl(run.path/'metrics'/f'{method}-{cat}-epochs.jsonl',history);last_saved=time.monotonic();run.say('upstream-checkpoint-saved',path=relative(last),method=method,category=cat,epoch=epoch,global_step=step,cursor=cursor)
 total=math.ceil(len(train_data.rows)/batch_size)*c['epochs']
 while epoch<c['epochs'] and bad_epochs<patience:
  order=torch.randperm(len(train_data.rows),generator=torch.Generator().manual_seed(cfg['seed']+epoch)).tolist();model.train()
  # Historical source sets BatchNorm modules to eval during fine-tuning.
  for m in model.modules():
   if isinstance(m,torch.nn.modules.batchnorm._BatchNorm):m.eval()
  while cursor<len(order):
   if run.stop:save();raise Paused('upstream safely stopped after committed optimizer step')
   indices=order[cursor:cursor+batch_size]
   if len(indices)<2:
    # A singleton contrastive batch has no negative. Explicit skip is logged, not trained on validation.
    cursor+=len(indices);run.say('singleton-training-batch-skipped',indices=indices);continue
   rows,visual,target=train_data.batch(indices,model,True);visual=visual.to(run.device);target=target.to(run.device);texts=[r['text'] for r in rows]
   optimizer.zero_grad(set_to_none=True);sync(run);tick=time.perf_counter()
   with torch.autocast(device_type=run.device.type,dtype=torch.float16,enabled=amp):
    loss_dict=model.compute_loss(texts,visual,target)
    if bank_rows is not None:
     br=[bank_rows[j] for j in indices];mask=torch.tensor([r['use_bank'] for r in br],device=run.device).float();denom=mask.sum().clamp_min(1)
     query=model.extract_query(texts,visual);positive=model.extract_target(target)
     with torch.no_grad():
      conflict=model.extract_target(torch.stack([train_data.image(r['conflict_id'],model.preprocess_train) for r in br]).to(run.device))
      soft_images=torch.stack([torch.stack([train_data.image(i,model.preprocess_train) for i in r['soft_ids'][:cfg['bank']['train_size']]]) for r in br]).to(run.device);B,K=soft_images.shape[:2];soft=model.extract_target(soft_images.flatten(0,1)).reshape(B,K,-1)
     loss_dict['conflict']=cfg['bank']['conflict_weight']*(F.relu(cfg['bank']['margin']-(query*positive).sum(-1)+(query*conflict).sum(-1))*mask).sum()/denom
     weights=torch.tensor([r['soft_weights'][:K] for r in br],device=run.device);weights=weights/weights.sum(-1,keepdim=True)
     logits=torch.einsum('bd,bkd->bk',query,soft)/cfg['bank']['temperature'];kl=F.kl_div(F.log_softmax(logits,-1),weights,reduction='none').sum(-1);loss_dict['soft']=cfg['bank']['soft_weight']*(kl*mask).sum()/denom
    loss=sum(loss_dict.values())
   if not torch.isfinite(loss):raise FloatingPointError('nonfinite upstream loss')
   scaler.scale(loss).backward();scaler.step(optimizer);scaler.update();sync(run);elapsed=time.perf_counter()-tick
   cursor+=len(indices);step+=1;value=float(loss.detach());loss_total+=value*len(indices);items_total+=len(indices)
   rec={'global_step':step,'epoch':epoch+1,'cursor':cursor,'query_ids':[r['query_id'] for r in rows],'loss':value,'loss_components':{k:float(v.detach()) for k,v in loss_dict.items()},'average_loss':loss_total/items_total,'lr':[g['lr'] for g in optimizer.param_groups],'seconds':elapsed,'batch_size':len(indices),**run.memory()};steps.append(rec)
   if step%cfg['log_every']==0:run.progress(step,total,started,completed_before=initial_step,method=method,category=cat,epoch=epoch+1,loss=value,average_loss=loss_total/items_total,lr=rec['lr'],samples=cursor)
   if step%cfg['save_every_steps']==0 or time.monotonic()-last_saved>cfg['save_every_seconds']:save()
   if pause_after and step>=pause_after:save();raise Paused('explicit upstream short-step pause')
   if benchmark and step-initial_step>=cfg['benchmark_steps']:break
  save();metrics=validate_model(cfg,val_data,model,run);score=metrics['recall']['10']+metrics['recall']['50'];history.append({'epoch':epoch+1,'global_step':step,'train_loss':loss_total/max(items_total,1),'validation':metrics,'score':score})
  if benchmark:
   measured=steps[initial_step+cfg['benchmark_warmup']:];durations=[x['seconds'] for x in measured];mean=sum(durations)/max(len(durations),1);out=run.path/'profiling'/f'benchmark-{cat}-{method}.json';atomic_json(out,{'real_hardware':str(run.device),'gpu':torch.cuda.get_device_name(run.device) if run.device.type=='cuda' else None,'precision':'float16 AMP' if amp else 'float32','batch_size':batch_size,'raw_steps':steps[initial_step:],'warmup_steps':cfg['benchmark_warmup'],'measured_steps':len(durations),'mean_train_step_seconds':mean,'peak_allocated_bytes':torch.cuda.max_memory_allocated(run.device) if run.device.type=='cuda' else None,'full_validation_seconds':metrics['validation_seconds'],'estimated_epochs':c['epochs'],'estimated_total_seconds':mean*total+metrics['validation_seconds']*c['epochs'],'scope':'train step forward+backward+optimizer excludes CPU loading; validation includes encoding; use as measured estimate, not performance result',**run.memory()});return [out,last]
  bad_epochs+=1
  if score>best_score:
   best_score=score;bad_epochs=0;best_state={'model_state_dict':model.state_dict(),'optimizer':optimizer.state_dict(),'scheduler':scheduler.state_dict(),'amp':copy.deepcopy(scaler.state_dict()),'rng':rng_state(),'epoch':epoch+1,'cursor':0,'global_step':step,'selection_score':score,'metrics':metrics,'config':cfg,'method':method,'input_identity':ident,'fingerprint':token};run.say('upstream-best-updated',method=method,category=cat,epoch=epoch+1,score=score)
  run.say('upstream-validation',method=method,category=cat,**metrics);epoch+=1;cursor=0;loss_total=0.;items_total=0;scheduler.step();save()
 return [last,best,run.path/'metrics'/f'{method}-{cat}-steps.jsonl',run.path/'metrics'/f'{method}-{cat}-epochs.jsonl']
