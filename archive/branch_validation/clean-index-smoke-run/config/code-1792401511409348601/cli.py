from __future__ import annotations
import argparse, copy, json, os, sys, time, traceback
from pathlib import Path
import torch
from .assets import audit, paths
from .runtime import ROOT, Run, Paused, fingerprint, read_json, relative, resolve
from .engine import build_topk, cost, diagnostics, dqu_export, evaluate, identity_files, multik, summary, train, visualization
STAGES=['audit','old-eval','multik','train','new-eval','diagnostics','dqu-export','visualization','cost','summary']
def merge(base,other):
 for k,v in other.items():
  if isinstance(v,dict) and isinstance(base.get(k),dict):merge(base[k],v)
  else:base[k]=v
 return base

def main(argv=None):
 parser=argparse.ArgumentParser(description='Portable CIR frozen-feature experiments; no remote access or downloads')
 parser.add_argument('stage',choices=STAGES+['pipeline','build-topk'])
 parser.add_argument('--config',type=str);parser.add_argument('--category',choices=['all','dress','shirt','toptee'],default='all')
 parser.add_argument('--run-id');parser.add_argument('--run-dir');parser.add_argument('--resume',action='store_true');parser.add_argument('--device',default='cpu',help='cpu, cuda:0, cuda:1; respect CUDA_VISIBLE_DEVICES')
 for field in ['asset-dir','checkpoint-dir','data-dir','output-dir']:parser.add_argument('--'+field)
 parser.add_argument('--log-every',type=int);parser.add_argument('--heartbeat-seconds',type=float);parser.add_argument('--eval-batch-size',type=int)
 parser.add_argument('--split',choices=['train','val'],default='train');parser.add_argument('--stop-after-steps',type=int,help='explicit short-test pause, not a training success')
 args=parser.parse_args(argv);cfg=read_json(ROOT/'configs'/'default.json')
 if args.config:merge(cfg,read_json(resolve(args.config)))
 for k in ['asset_dir','checkpoint_dir','data_dir','output_dir','log_every','heartbeat_seconds','eval_batch_size']:
  value=getattr(args,k,None)
  if value is not None:cfg[k]=value
 if cfg['log_every']<1 or cfg['heartbeat_seconds']<=0 or cfg['eval_batch_size']<1:parser.error('progress intervals/batch size must be positive')
 for k in ['asset_dir','checkpoint_dir','data_dir','output_dir']:cfg[k]=str(resolve(cfg[k]))
 for cat,splits in cfg.get('asset_files',{}).items():
  for split,items in splits.items():
   for kind,p in items.items():items[kind]=str(resolve(p))
 if cfg['training'].get('distill_weight',0)!=0:parser.error('this experiment has no distillation loss; distill_weight must be zero')
 from .runtime import sha
 cfg['_code_fingerprint']=fingerprint({p.name:sha(p) for p in sorted((ROOT/'src'/'cir_full').glob('*.py'))})
 os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
 device=torch.device(args.device)
 if device.type=='cuda' and not torch.cuda.is_available():parser.error('CUDA requested but unavailable; no automatic CPU fallback')
 if device.type=='cuda' and torch.cuda.device_count()!=1:parser.error('Mask exactly your allocated GPU with CUDA_VISIBLE_DEVICES, then use --device cuda:0')
 torch.set_num_threads(cfg.get('threads',4))
 categories=['dress','shirt','toptee'] if args.category=='all' else [args.category]
 if args.resume and not (args.run_id or args.run_dir):parser.error('--resume requires explicit --run-id or --run-dir')
 name=args.run_id or time.strftime('%Y%m%dT%H%M%S')+'-'+str(os.getpid())
 if Path(name).name!=name:parser.error('run ID must be a single directory name')
 path=resolve(args.run_dir) if args.run_dir else Path(cfg['output_dir'])/name
 if args.resume and not (path/'config'/'effective.json').exists():parser.error('resume run does not exist')
 run=None
 try:
  run=Run(path,cfg,vars(args),device,args.resume)
  inputfiles=[]
  for c in categories:
   inputfiles+=paths(cfg,c,'val')+paths(cfg,c,'train')+[Path(cfg['checkpoint_dir'])/c/f'{c}_best.pt',Path(cfg['asset_dir'])/c/'provenance.json']
  from .runtime import atomic_json
  attempt=str(time.time_ns());atomic_json(path/'manifests'/f'inputs-{attempt}.json',{'files':identity_files(inputfiles),'missing':[relative(p) for p in inputfiles if not p.exists()]})
  stages=STAGES if args.stage=='pipeline' else [args.stage]
  canonical={k:relative(Path(v)) if k in ['asset_dir','checkpoint_dir','data_dir','output_dir'] else v for k,v in cfg.items()}
  def portable(x):
   if isinstance(x,dict):return {k:portable(v) for k,v in x.items()}
   if isinstance(x,list):return [portable(v) for v in x]
   if isinstance(x,str) and x.startswith(str(ROOT)+os.sep):return '@PROJECT_ROOT/'+Path(x).relative_to(ROOT).as_posix()
   return x
  canonical=portable(canonical)
  for stage in stages:
   if stage=='summary':
    state=read_json(path/'reports'/'stage_state.json',{})
    ident={'config':canonical,'stages':{k:{'status':v['status'],'fingerprint':v['fingerprint'],'outputs':v.get('outputs',{})} for k,v in state.items() if k!='summary'}}
    # Summaries are derived views; retain previous versions when stages advance.
    if 'summary' in state and state['summary']['fingerprint']!=fingerprint(ident):
     old=path/'reports'/'summary.json'
     if old.exists():old.rename(old.with_name(f'summary-history-{time.time_ns()}.json'))
     state.pop('summary');atomic_json(path/'reports'/'stage_state.json',state)
    run.execute('summary',ident,lambda:summary(cfg,categories,run));continue
   for cat in categories:
    run.check_stop();key=f'{stage}:{cat}';files=paths(cfg,cat,'val')
    if stage=='train':files+=paths(cfg,cat,'train')+[Path(cfg['asset_dir'])/cat/'provenance.json']
    if stage=='audit':files+=paths(cfg,cat,'train')
    if stage=='old-eval' or stage=='audit':files+=[Path(cfg['checkpoint_dir'])/cat/f'{cat}_best.pt']
    if stage=='new-eval':files+=[path/'checkpoints'/cat/'best.pt']
    if stage=='multik':files+=[path/'metrics'/f'old-{cat}.json']
    if stage in ['diagnostics','visualization']:
     files+=[path/'predictions'/f'{method}-{cat}.jsonl' for method in ['cn','old','new']]
    if stage=='cost':files+=[path/'profiling'/f'old-{cat}.json']
    if stage=='visualization':files+=[Path(cfg['data_dir'])/'captions'/f'cap.{cat}.val.json']
    if stage=='dqu-export':files+=[resolve(cfg.get('dqu_predictions',{}).get(cat,f'predictions/dqu/{cat}.jsonl'))]
    if stage=='build-topk':files=paths(cfg,cat,args.split)[:2];key+=f':{args.split}'
    ident={'config':canonical,'inputs':identity_files(files),'missing':[relative(p) for p in files if not p.exists()],'category':cat,'stage':stage}
    fn={'audit':lambda:audit(cfg,cat,run),'old-eval':lambda:evaluate(cfg,cat,run,'old'),'multik':lambda:multik(cfg,cat,run),'train':lambda:train(cfg,cat,run,args.stop_after_steps),'new-eval':lambda:evaluate(cfg,cat,run,'new'),'diagnostics':lambda:diagnostics(cfg,cat,run),'dqu-export':lambda:dqu_export(cfg,cat,run),'visualization':lambda:visualization(cfg,cat,run),'cost':lambda:cost(cfg,cat,run),'build-topk':lambda:build_topk(cfg,cat,run,args.split)}[stage]
    try:run.execute(key,ident,fn)
    except Paused:raise
    except Exception:
     if args.stage!='pipeline':raise
     run.say('continuing-independent-stages',failed_stage=key)
  states=read_json(path/'reports'/'stage_state.json',{});blocked=[k for k,v in states.items() if v['status']=='blocked'];run.say('run-finished',blocked_stages=blocked)
  return 1 if any(v['status']=='failed' for v in states.values()) else 2 if blocked else 0
 except Paused:
  if run:traceback.print_exc()
  return 130
 except BaseException:
  traceback.print_exc();return 1
 finally:
  if run:run.close()
