from __future__ import annotations
import argparse,json,os,sys,time,traceback,shutil
from pathlib import Path
import torch
from .runtime import ROOT,Run,Blocked,Paused,resolve,relative,read_json,atomic_json,fingerprint,sha
from .raw_data import FashionData
from . import upstream
from .cli import merge
from . import engine
ORDER=['doctor','dqu','rc','export-dqu','export-rc','bank','cn','export-cn','rr-train','rr-eval','analysis','report']

def rr_config(cfg,run):
 c=read_json(ROOT/'configs/default.json');merge(c,cfg.get('rr_overrides',{}));c.update(asset_dir=str(run.path/'assets/cn'),data_dir=str(resolve(cfg['data_dir'])),checkpoint_dir=str(run.path/'checkpoints/new-only'),output_dir=str(run.path),fixture=cfg.get('fixture',False),_code_fingerprint=cfg['_code_fingerprint'],expected_val_k=cfg['candidate_k'])
 c['training']['val_k']=cfg['candidate_k']
 if c['training'].get('score_rule','true-similarity')!='true-similarity':raise ValueError('main Proposed must use true-similarity; legacy score is only allowed via the isolated legacy-control command')
 return c

def doctor(cfg,cat,run,persist=True):
 for name in ['PIL','cv2','open_clip'] if not cfg.get('fixture') else []:
  __import__(name)
 if not cfg.get('fixture'):
  p=resolve(cfg['pretrained'])
  if not p.is_file():raise Blocked('public pretrained weights missing; scripts/prepare.py model --kind clip')
  side=read_json(p.with_suffix(p.suffix+'.integrity.json'))
  if not side or side.get('sha256')!=sha(p):raise Blocked('pretrained SHA verification record missing/mismatch; import using prepare.py import --sha256 ...')
 train=FashionData(cfg,cat,'train');val=FashionData(cfg,cat,'val')
 if set(train.gallery_ids)&set(val.gallery_ids):raise ValueError('official train/val image partitions overlap; verify supplied split files')
 nbytes=shutil.disk_usage(run.path).free;estimate=0 if cfg.get('fixture') else 80*1024**3
 report={'category':cat,'train_queries':len(train.rows),'val_queries':len(val.rows),'train_gallery':len(train.gallery_ids),'val_gallery':len(val.gallery_ids),'train_val_image_overlap':0,'train_identity':train.identity,'val_identity':val.identity,'free_bytes':nbytes,'estimated_per_category_checkpoint_budget_bytes':estimate,'query_config':cfg['query'],'scope':'new raw-image workflow; no old CN or RR weights are prerequisites','correction_cache':'present' if train.correction_path.exists() else 'identity correction; documented departure from historical cache','keywords_mode':cfg['query']['keywords']}
 out=run.path/'reports'/(f'doctor-{cat}.json' if persist else f'preflight-{cat}-{time.time_ns()}.json');atomic_json(out,report);run.say('preflight-complete',**{k:v for k,v in report.items() if not k.endswith('identity')})
 if estimate>nbytes:raise Blocked('insufficient free disk for retained full optimizer latest/best checkpoints; budget ~80 GiB per category plus data/features; choose a larger output volume')
 return [out]

def analysis(cfg,cat,run):
 methods={};records={}
 for method,file in [('dqu',f'dqu-{cat}.jsonl'),('rc',f'rc-{cat}.jsonl'),('cn',f'cn-{cat}.jsonl'),('proposed',f'new-{cat}.jsonl')]:
  p=run.path/'predictions'/file
  if not p.exists():raise Blocked('predictions missing: '+relative(p))
  rows=engine.read_jsonl(p);records[method]=rows;N=len(rows);hits={str(k):sum(r['target_rank'] is not None and r['target_rank']<=k for r in rows) for k in cfg['cutoffs'] if k<=cfg['candidate_k']};methods[method]={'count':N,'hits':hits,'recall':{k:v*100/N for k,v in hits.items()}}
 ids=[r['query_id'] for r in records['cn']];assert all([r['query_id'] for r in rows]==ids for rows in records.values())
 changes={};cases=[]
 for method in ['dqu','rc','proposed']:
  changes[method]={}
  for k in cfg['cutoffs']:
   if k>cfg['candidate_k']:continue
   before=[r['target_rank'] is not None and r['target_rank']<=k for r in records['cn']];after=[r['target_rank'] is not None and r['target_rank']<=k for r in records[method]]
   changes[method][str(k)]={'corrected':sum(b and not a for a,b in zip(before,after)),'harmed':sum(a and not b for a,b in zip(before,after))}
 for j,qid in enumerate(ids):cases.append({'query_id':qid,'methods':{k:rows[j] for k,rows in records.items()}})
 path=run.path/'reports'/f'aligned-cases-{cat}.jsonl';engine.write_jsonl(path,cases);out=run.path/'metrics'/f'analysis-{cat}.json';atomic_json(out,{'methods':methods,'vs_cn':changes,'all_query_count':len(ids),'protocol':'same new training run, same val annotation identity and query-union gallery'})
 # Generate contact sheets/HTML only from real image IDs in complete exported records.
 import html
 directory=run.path/'reports'/f'cases-{cat}';directory.mkdir(parents=True,exist_ok=True);body=['<meta charset="utf-8"><h1>New DQU → CN → Proposed</h1>'];data=FashionData(cfg,cat,'val')
 for c in [x for x in cases if x['methods']['cn']['target_rank']!=x['methods']['proposed']['target_rank']][:12]:
  body.append('<h2>'+html.escape(c['query_id'])+'</h2>')
  for method in ['dqu','cn','proposed']:
   row=c['methods'][method];body.append('<p>'+html.escape(method+' '+str(row.get('modification_text')))+' rank '+html.escape(str(row['target_rank']) if row['target_rank'] else row['rank_status'])+'</p>')
   for imageid in [row['reference_image_id'],row['target_image_id']]+row['retrieved_image_ids'][:5]:
    src=data.image_path(imageid)
    if cfg.get('fixture'):continue
    dest=directory/src.name;shutil.copy2(src,dest);body.append('<figure style="display:inline-block"><img width="120" src="'+html.escape(src.name)+'"><figcaption>'+html.escape(imageid)+'</figcaption></figure>')
 out_html=directory/'index.html';out_html.write_text('\n'.join(body),encoding='utf8');return [path,out,out_html]+[p for p in directory.glob('*.jpg')]

def report(cfg,categories,run):
 outputs={};macro={};rows=[]
 for c in categories:
  p=run.path/'metrics'/f'analysis-{c}.json'
  if p.exists():outputs[c]=read_json(p)
 if len(outputs)==len(categories):
  for method in ['dqu','rc','cn','proposed']:
   keys=outputs[categories[0]]['methods'][method]['recall'];macro[method]={k:sum(x['methods'][method]['recall'][k] for x in outputs.values())/len(outputs) for k in keys}
   for k,v in macro[method].items():rows.append({'method':method,'cutoff':k,'macro_recall':v})
 out=run.path/'reports/final.json';atomic_json(out,{'categories':outputs,'macro_equal_weight':macro,'states':read_json(run.path/'reports/stage_state.json',{}),'fixture_only':cfg.get('fixture',False),'actual_gpu':torch.cuda.get_device_name(run.device) if run.device.type=='cuda' else None,'cuda_execution_verified':run.device.type=='cuda','limitations':'hardware validation only refers to actual run device; no historical number target asserted'})
 import csv
 table=run.path/'reports/metrics.csv'
 with table.open('w',newline='',encoding='utf8') as f:
  w=csv.DictWriter(f,fieldnames=['method','cutoff','macro_recall']);w.writeheader();w.writerows(rows)
 # Self-contained SVG from actual machine-readable metrics, no plotting package requirement.
 svg=run.path/'reports/metrics.svg';parts=['<svg xmlns="http://www.w3.org/2000/svg" width="800" height="260">']
 for j,(method,values) in enumerate(macro.items()):
  value=values.get('10',values.get('5',0));parts.append(f'<text x="10" y="{30+j*50}">{method} R@10/available: {value:.4f}</text><rect x="300" y="{15+j*50}" width="{value*4}" height="25" fill="#346ca8"/>')
 parts.append('</svg>');svg.write_text(''.join(parts));return [out,table,svg]

def main(argv=None):
 p=argparse.ArgumentParser(description='New raw-image CIR pipeline. No old upstream checkpoints required.')
 p.add_argument('stage',choices=ORDER+['pipeline','benchmark','experiments','online-cost','legacy-control']);p.add_argument('--config',default='configs/full.json');p.add_argument('--category',choices=['dress','shirt','toptee','all'],default='dress');p.add_argument('--run-id');p.add_argument('--run-dir');p.add_argument('--resume',action='store_true');p.add_argument('--device',default='cpu');p.add_argument('--stop-after-steps',type=int);p.add_argument('--benchmark-method',choices=['dqu','rc','cn'],default='dqu');p.add_argument('--data-dir');p.add_argument('--output-dir');p.add_argument('--pretrained');p.add_argument('--log-every',type=int);p.add_argument('--rerank-ks',default='5,10,20,30,50,100');a=p.parse_args(argv)
 cfg=read_json(ROOT/'configs/full.json');merge(cfg,read_json(resolve(a.config)))
 for key in ['data_dir','output_dir','pretrained','log_every']:
  if getattr(a,key,None) is not None:cfg[key]=getattr(a,key)
 for field in ['data_dir','pretrained','output_dir']:
  value=Path(cfg[field]);cfg[field]=relative(resolve(value)) if resolve(value).is_relative_to(ROOT) else str(value)
 if min(cfg['batch_size'],cfg['export_batch_size'],cfg['candidate_k'],cfg['save_every_steps'],cfg['log_every'])<1:p.error('batch/candidate/save/log counts must be positive')
 if cfg['backbone']=='fixture-tiny' and not cfg.get('fixture'):p.error('tiny model is exclusively a marked test fixture')
 if not cfg.get('fixture') and cfg['candidate_k']<max(cfg['cutoffs']):p.error('candidate_k must cover all metric cutoffs; rerank-prefix experiments use --rerank-ks')
 if not {10,50}.issubset(set(cfg['cutoffs'])):p.error('checkpoint selection requires cutoffs 10 and 50')
 if cfg['benchmark_steps']<=cfg['benchmark_warmup']:p.error('benchmark must include measured steps after warmup')
 if cfg['bank']['train_size']>cfg['bank']['size']:p.error('train bank size exceeds generated teacher bank')
 cfg['_code_fingerprint']=fingerprint({x.name:sha(x) for x in sorted((ROOT/'src/cir_full').glob('*.py'))})
 os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8');torch.set_num_threads(cfg['threads']);device=torch.device(a.device)
 if device.type=='cuda' and not torch.cuda.is_available():p.error('CUDA explicitly requested but unavailable')
 if device.type=='cuda' and torch.cuda.device_count()!=1:p.error('Set CUDA_VISIBLE_DEVICES to exactly your allocated physical GPU, then use --device cuda:0; avoid touching other visible GPUs when saving RNG states')
 if a.resume and not (a.run_id or a.run_dir):p.error('resume requires explicit run ID/directory')
 name=a.run_id or time.strftime('%Y%m%dT%H%M%S')+'-'+str(os.getpid())
 if Path(name).name!=name:p.error('run ID must be one directory component')
 path=resolve(a.run_dir) if a.run_dir else resolve(cfg['output_dir'])/name
 if a.resume and not (path/'config/effective.json').exists():p.error('resume run missing')
 cats=['dress','shirt','toptee'] if a.category=='all' else [a.category];run=None
 try:
  previous=read_json(path/'reports/stage_state.json',{})
  if a.stage=='benchmark' and any(k.split(':')[0] in ['dqu','rc','cn'] for k in previous):raise ValueError('benchmark must use a separate run, never a formal training run')
  if a.stage!='benchmark' and any(k.startswith('benchmark:') for k in previous):raise ValueError('benchmark checkpoints are isolated; start a new formal run')
  run=Run(path,cfg,vars(a),device,a.resume);stages=ORDER if a.stage=='pipeline' else [a.stage]
  if a.stage=='pipeline':
   # Fail before any expensive training when required raw materials are absent.
   preflight_inputs=[resolve(cfg['data_dir'])/'annotations.manifest.json',resolve(cfg['pretrained'])]+[resolve(cfg['data_dir'])/'resized_image'/c/'import.manifest.json' for c in cats]
   run.execute('preflight',{'config':cfg,'known_input_manifests':engine.identity_files(preflight_inputs)},lambda:[p for c in cats for p in doctor(cfg,c,run,persist=False)])
   preflight_state=read_json(path/'reports/stage_state.json',{})['preflight']
   if preflight_state['status']=='blocked':raise Blocked(preflight_state['reason'])
   if not cfg.get('fixture') and shutil.disk_usage(path).free<len(cats)*80*1024**3:raise Blocked('insufficient retained checkpoint disk budget for selected categories')
  for stage in stages:
   if stage=='report':
    run.execute('report',{'code':cfg['_code_fingerprint'],'config':cfg,'metrics':engine.identity_files(list((path/'metrics').glob('analysis-*.json')))},lambda:report(cfg,cats,run));continue
   for cat in cats:
    run.check_stop();run.stage=stage+':'+cat
    data=FashionData(cfg,cat,'train');val=FashionData(cfg,cat,'val');inputs={'train':data.identity,'val':val.identity}
    prereqs=[]
    if stage=='cn':prereqs=[upstream.ckpt_path(run,'rc',cat),path/'assets/banks'/f'{cat}.jsonl']
    if stage.startswith('export-'):prereqs=[upstream.ckpt_path(run,stage.split('-')[1],cat)]
    if stage=='bank':prereqs=[path/'assets'/m/cat/f'{cat}_train_{k}.pt' for m in ['dqu','rc'] for k in ['query_features','image_features']]+[path/'assets'/m/cat/'provenance.json' for m in ['dqu','rc']]
    if stage=='rr-train':prereqs=list((path/'assets/cn'/cat).glob('*.pt'))+[path/'assets/cn'/cat/'provenance.json']
    if stage in ['rr-eval','analysis','experiments','online-cost','legacy-control']:prereqs=[path/'checkpoints'/cat/'best.pt']+list((path/'assets/cn'/cat).glob('*.pt'))
    if stage=='analysis':prereqs+=[path/'predictions'/f'{m}-{cat}.jsonl' for m in ['dqu','rc','cn','new']]
    if stage=='online-cost':prereqs+=[upstream.ckpt_path(run,'cn',cat)]
    weight=resolve(cfg['pretrained'])
    if weight.is_file():inputs['pretrained']={relative(weight):sha(weight)}
    ident={'config':cfg,'data':inputs,'prerequisites':engine.identity_files(prereqs),'missing':[relative(x) for x in prereqs if not x.exists()]}
    rr=rr_config(cfg,run)
    from .experiments import experiments,online_cost
    fn={'doctor':lambda:doctor(cfg,cat,run),'dqu':lambda:upstream.train(cfg,cat,'dqu',run,a.stop_after_steps),'rc':lambda:upstream.train(cfg,cat,'rc',run,a.stop_after_steps),'cn':lambda:upstream.train(cfg,cat,'cn',run,a.stop_after_steps),'export-dqu':lambda:upstream.export(cfg,cat,'dqu',run),'export-rc':lambda:upstream.export(cfg,cat,'rc',run),'export-cn':lambda:upstream.export(cfg,cat,'cn',run),'bank':lambda:upstream.bank(cfg,cat,run),'rr-train':lambda:engine.train(rr,cat,run,a.stop_after_steps),'rr-eval':lambda:engine.evaluate(rr,cat,run,'new'),'analysis':lambda:analysis(cfg,cat,run),'benchmark':lambda:upstream.train(cfg,cat,a.benchmark_method,run,benchmark=True),'experiments':lambda:experiments(cfg,rr,cat,run,[int(k) for k in a.rerank_ks.split(',')]),'online-cost':lambda:online_cost(cfg,cat,run),'legacy-control':lambda:legacy_control(cfg,rr,cat,run)}[stage]
    if stage=='experiments':ident['rerank_ks']=a.rerank_ks
    if stage=='benchmark':ident['benchmark_method']=a.benchmark_method
    run.execute(stage+':'+cat,ident,fn)
    state=read_json(path/'reports/stage_state.json',{})[stage+':'+cat]
    if state['status']=='blocked' and a.stage=='pipeline':raise Blocked(state['reason'])
  state=read_json(path/'reports/stage_state.json',{});run.say('new-workflow-completed',fixture=cfg.get('fixture',False),states={k:v['status'] for k,v in state.items()});return 2 if any(v['status']=='blocked' for v in state.values()) else 0
 except Blocked as e:
  if run:
   run.say('blocked',reason=str(e));atomic_json(run.path/'reports'/f'blocked-{time.time_ns()}.json',{'stage':run.stage,'reason':str(e),'time':time.time()})
  else:print(str(e),file=sys.stderr)
  return 2
 except Paused:traceback.print_exc();return 130
 except Exception:traceback.print_exc();return 1
 finally:
  if run:run.close()


def legacy_control(cfg,rr,cat,run):
 # Explicit controlled experiment on NEW assets; never a default input to Proposed.
 import copy
 c=copy.deepcopy(rr);c['training']['score_rule']='legacy-positive-constant-ablation'
 d=run.path/'controls/legacy-positive-constant';exists=(d/'config/effective.json').exists();child=Run(d,c,{'control':'legacy target score=1; new CN assets'},run.device,exists)
 try:
  result=engine.train(c,cat,child)
  result+=engine.evaluate(c,cat,child,'new')
  corrected=read_json(run.path/'metrics'/f'new-{cat}.json')
  if corrected is None:raise Blocked('complete corrected RR evaluation first for the paired comparison')
  baseline=read_json(child.path/'metrics'/f'new-{cat}.json');assert baseline['count']==corrected['count']
  comparison=run.path/'reports'/f'legacy-vs-corrected-{cat}.json';atomic_json(comparison,{'corrected_true_similarity':corrected,'legacy_constant_control_on_same_new_CN':baseline,'note':'both use new CN assets, same train parameters and deterministic sample strategy; not historical old RR vs a new feature space'})
  return result+[comparison]
 finally:child.close()
