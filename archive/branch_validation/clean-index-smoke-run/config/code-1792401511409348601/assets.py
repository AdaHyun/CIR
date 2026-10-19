from __future__ import annotations
import csv, json, pickle, types
from pathlib import Path, PurePosixPath, PureWindowsPath
import torch
from .runtime import ROOT, Blocked, atomic_json, relative, sha
from .model import FeatureReranker
class PortableUnpickler(pickle.Unpickler):
 def find_class(self,module,name):
  if module=='pathlib' and name=='PosixPath':return PurePosixPath
  if module=='pathlib' and name=='WindowsPath':return PureWindowsPath
  return super().find_class(module,name)
PORTABLE=types.ModuleType('cir_portable_pickle');PORTABLE.Unpickler=PortableUnpickler;PORTABLE.load=pickle.load

def load_pt(p,run=None):
 if not p.is_file():raise Blocked(f'missing asset: {relative(p)}')
 h=sha(p)
 # Published archive hashes are authoritative for copied old artifacts.
 manifest=ROOT/'archive'/'CIR-minimal-reproduction-20260919'/'manifests'/'SHA256SUMS.txt'
 if manifest.exists() and (p.is_relative_to(ROOT/'assets') or p.is_relative_to(ROOT/'checkpoints') or p.is_relative_to(ROOT/'archive'/'CIR-minimal-reproduction-20260919')):
  suffix='/'.join(p.parts[-3:])
  for line in manifest.read_text().splitlines():
   expected,name=line.split(maxsplit=1)
   if p.as_posix().endswith('/'+name.lstrip('./')) and p.suffix=='.pt' and h!=expected:raise ValueError(f'published hash mismatch: {p}')
 if run:run.say('load',path=relative(p),bytes=p.stat().st_size,sha256=h)
 # Trusted, locally supplied artifacts only; pickle files may execute code.
 return torch.load(p,map_location='cpu',weights_only=False,pickle_module=PORTABLE)

def paths(cfg,cat,split):
 base=Path(cfg['asset_dir'])/cat
 return [Path(cfg.get('asset_files',{}).get(cat,{}).get(split,{}).get(x,base/f'{cat}_{split}_{x}.pt')) for x in ['query_features','image_features','topk']]
def describe(x):
 if isinstance(x,torch.Tensor):
  y=x.float();return {'type':'Tensor','shape':list(x.shape),'dtype':str(x.dtype),'finite':bool(torch.isfinite(y).all()),'norm_min':float(y.norm(dim=-1).min()) if y.ndim==2 else None,'norm_max':float(y.norm(dim=-1).max()) if y.ndim==2 else None}
 if isinstance(x,dict):return {str(k):describe(v) for k,v in x.items()}
 if isinstance(x,(list,tuple)):return {'type':type(x).__name__,'length':len(x),'first':str(x[:1])[:250]}
 return x if isinstance(x,(str,int,float,bool,type(None))) else str(x)

def load_bundle(cfg,cat,split,run=None,require_topk=True):
 files=paths(cfg,cat,split)
 if not cfg.get('fixture',False) and any(p.resolve().is_relative_to((ROOT/'tests').resolve()) for p in files):raise ValueError('fixture assets cannot enter real experiment results')
 provenance_path=Path(cfg['asset_dir'])/cat/'provenance.json'
 if provenance_path.exists():
  declared=__import__('json').loads(provenance_path.read_text()).get(split,{}).get('files',{})
  for p in files if require_topk else files[:2]:
   if declared.get(p.name)!=sha(p):raise ValueError('asset provenance SHA mismatch: '+p.name)
 q=load_pt(files[0],run);im=load_pt(files[1],run)
 top=load_pt(files[2],run) if require_topk else None
 assert isinstance(q,dict) and isinstance(im,dict)
 Q=q['query_features'];I=im['features'];ids=im['image_ids'];n=len(q['source_ids'])
 assert Q.ndim==I.ndim==2 and Q.shape[1]==I.shape[1] and Q.shape[0]==n>0 and len(q['target_ids'])==n
 assert len(ids)==I.shape[0] and len(set(ids))==len(ids),'duplicate/missing gallery IDs'
 assert all(isinstance(x,str) for x in ids),'gallery IDs must be strings'
 lookup={v:i for i,v in enumerate(ids)}
 for arr in [Q,I]:
  assert arr.dtype.is_floating_point and torch.isfinite(arr).all(),'nonfinite/nonfloat features'
  assert torch.allclose(arr.float().norm(dim=1),torch.ones(arr.shape[0]),atol=2e-5,rtol=2e-5),'features must be unit normalized; no silent normalization'
 assert all(x in lookup for x in q['source_ids']+q['target_ids']),'IDs absent from gallery'
 assert all(s!=t for s,t in zip(q['source_ids'],q['target_ids'])),'reference equals target'
 query_ids=q.get('query_ids',[f'{cat}:{split}:{j}:{s}:{t}' for j,(s,t) in enumerate(zip(q['source_ids'],q['target_ids']))])
 assert len(query_ids)==n and len(set(query_ids))==n,'invalid query IDs'
 maxerr=0.;rank_disagreements=[]
 if top is not None:
  assert top['source_ids']==q['source_ids'] and top['target_ids']==q['target_ids'],'query identity/order mismatch'
  assert len(top['sorted_ids'])==len(top['sorted_scores'])==n
  k=len(top['sorted_ids'][0]);assert k>0
  if split=='val' and not cfg.get('fixture',False):assert k==cfg.get('expected_val_k',100),'candidate K does not match configured evaluation protocol'
  started=__import__('time').monotonic()
  for j in range(n):
   row=top['sorted_ids'][j];scores=torch.tensor(top['sorted_scores'][j]);assert len(row)==len(scores)==k and len(set(row))==k
   assert q['source_ids'][j] not in row,'reference image retained'
   assert all(x in lookup for x in row) and torch.isfinite(scores).all()
   actual=I[[lookup[x] for x in row]].float()@Q[j].float();err=float((actual-scores).abs().max());maxerr=max(maxerr,err)
   assert err<2e-5,f'score/ID mapping mismatch query {j}: {err}'
   assert bool((scores[:-1]>=scores[1:]-1e-7).all()),'TopK scores unsorted'
   if 'target_ranks' in top:
    rank=row.index(q['target_ids'][j])+1 if q['target_ids'][j] in row else None
    archived=int(top['target_ranks'][j])
    if (rank is not None and archived!=rank) or (rank is None and archived<=k):rank_disagreements.append({'query_index':j,'archived_rank':archived,'topk_rank':rank})
   if 'target_in_topk' in top:assert bool(top['target_in_topk'][j])==(q['target_ids'][j] in row)
   if run and ((j+1)%max(100,cfg.get('log_every',20)*10)==0 or j+1==n):run.progress(j+1,n,started,phase='validate-mapping',category=cat,split=split)
 if run:run.say('asset-validated',category=cat,split=split,queries=n,gallery=len(ids),dimension=Q.shape[1],dtype=str(Q.dtype),max_score_error=maxerr,rank_disagreements=len(rank_disagreements))
 return {'query':q,'image':im,'topk':top,'index':lookup,'query_ids':query_ids,'files':files if require_topk else files[:2],'score_max_abs_error':maxerr,'rank_disagreements':rank_disagreements,'gallery_is_query_union':set(ids)==set(q['source_ids']+q['target_ids'])}

def checkpoint(cfg,cat,run=None,path=None):
 p=path or Path(cfg['checkpoint_dir'])/cat/f'{cat}_best.pt';c=load_pt(p,run)
 a=c['args'];state=c['model_state_dict'];dim=state['query_proj.weight'].shape[1]
 model=FeatureReranker(dim,int(a['hidden_dim']),float(a['dropout']),float(a['base_score_scale']));model.load_state_dict(state,strict=True)
 return model,c,p

def audit(cfg,cat,run):
 report={'category':cat,'sources_separate':True,'splits':{}}
 for split in ['val','train']:
  missing=[relative(p) for p in paths(cfg,cat,split) if not p.exists()]
  if missing:
   report['splits'][split]={'status':'缺失','missing':missing,'topk_rebuildable':all(p.exists() for p in paths(cfg,cat,split)[:2])};continue
  b=load_bundle(cfg,cat,split,run)
  report['splits'][split]={'status':'已验证可用','schema':{relative(p):describe(load_pt(p)) for p in b['files']},'query_count':len(b['query_ids']),'gallery_count':len(b['index']),'gallery_is_query_union':b['gallery_is_query_union'],'score_max_abs_error':b['score_max_abs_error'],'rank_disagreements':b['rank_disagreements'],'feature_space_provenance':'unconfirmed producer checkpoint hash; dimensional compatibility alone is insufficient'}
 m,c,p=checkpoint(cfg,cat,run);report['checkpoint']={'status':'已验证可用','path':relative(p),'schema':describe(c),'strict_state_dict':True,'train_resume_capable':all(k in c for k in ['optimizer','rng','cursor'])}
 report['cross_package_compatibility']='model class matches strictly; assets of phase_packages absent, so feature-space identity unverified'
 out=run.path/'reports'/f'audit-{cat}.json';atomic_json(out,report);return [out]
