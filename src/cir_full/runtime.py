from __future__ import annotations
import contextlib, hashlib, json, os, platform, signal, subprocess, sys, threading, time, traceback
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[2]
class Blocked(RuntimeError): pass
class Paused(RuntimeError): pass
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def fingerprint(x):return hashlib.sha256(json.dumps(x,sort_keys=True,default=str).encode()).hexdigest()
def atomic_json(path,x):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_name(path.name+'.partial')
 with tmp.open('w',encoding='utf8') as f:json.dump(x,f,indent=2,ensure_ascii=False,default=str);f.write('\n');f.flush();os.fsync(f.fileno())
 os.replace(tmp,path)
def atomic_torch(path,x):
 path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_name(path.name+'.partial')
 with tmp.open('wb') as f:torch.save(x,f);f.flush();os.fsync(f.fileno())
 os.replace(tmp,path)
def read_json(p,default=None):return json.loads(Path(p).read_text(encoding='utf8')) if Path(p).exists() else default
def stamp():return time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())
def relative(p):
 try:return Path(p).resolve().relative_to(ROOT).as_posix()
 except ValueError:return str(Path(p).resolve())
def resolve(p):
 p=Path(p);return p.resolve() if p.is_absolute() else (ROOT/p).resolve()
class Tee:
 def __init__(self,stream,log):self.stream,self.log=stream,log
 def write(self,s):self.stream.write(s);self.stream.flush();self.log.write(s);self.log.flush()
 def flush(self):self.stream.flush();self.log.flush()
class Run:
 def __init__(self,path,cfg,args,device,resume=False):
  if cfg.get('fixture',False) and not path.resolve().is_relative_to((ROOT/'tests'/'artifacts').resolve()):raise ValueError('test fixtures must write only under tests/artifacts; never real runs/')
  self.path=path;self.cfg=cfg;self.device=device;self.stop=False;self.stage='initialization';self.started=time.monotonic();self.resume=resume
  if (path/'config'/'effective.json').exists() and not resume:raise ValueError('run exists: use --resume or a new run ID')
  for name in ['config','manifests','logs','checkpoints','metrics','predictions','profiling','reports']:(path/name).mkdir(parents=True,exist_ok=True)
  self.log=(path/'logs'/'terminal.log').open('a',encoding='utf8');self.origout,self.origerr=sys.stdout,sys.stderr
  sys.stdout=Tee(sys.stdout,self.log);sys.stderr=Tee(sys.stderr,self.log)
  self.halt=threading.Event();self.thread=threading.Thread(target=self.heartbeat,daemon=True);self.thread.start()
  self.handlers={}
  for s in (signal.SIGINT,signal.SIGTERM):
   self.handlers[s]=signal.getsignal(s);signal.signal(s,self.signal)
  self.say('restart' if resume else 'start',run_id=path.name,output=relative(path),device=str(device))
  attempt=str(time.time_ns());atomic_json(path/'config'/f'invocation-{attempt}.json',args)
  atomic_json(path/'config'/'effective.json',cfg)
  atomic_json(path/'config'/f'environment-{attempt}.json',{'python':sys.version,'platform':platform.platform(),'torch':torch.__version__,'cuda_runtime':torch.version.cuda,'cuda_available':torch.cuda.is_available(),'device':str(device),'gpu':torch.cuda.get_device_name(device) if device.type=='cuda' else None})
  snap=path/'config'/f'code-{attempt}';snap.mkdir()
  import shutil
  for p in (ROOT/'src'/'cir_full').glob('*.py'):shutil.copy2(p,snap/p.name)
  for p in [ROOT/'scripts'/'cir.py',ROOT/'configs'/'default.json']:
   if p.exists():shutil.copy2(p,snap/p.name)
  try:
   for cmd,name in [(['git','rev-parse','HEAD'],'git-head.txt'),(['git','diff','--binary','--','.'],'git-diff.patch'),(['git','status','--short'],'git-status.txt')]:
    result=subprocess.run(cmd,cwd=ROOT,capture_output=True,text=True);(snap/name).write_text(result.stdout+result.stderr,encoding='utf8')
  except OSError as e:self.say('git-metadata-unavailable',reason=str(e))
 def say(self,event,**kw):print(json.dumps({'time':stamp(),'stage':self.stage,'event':event,**kw},ensure_ascii=False,default=str),flush=True)
 def signal(self,s,*_):self.stop=True;self.say('interrupt-requested',signal=s)
 def check_stop(self):
  if self.stop:raise Paused('interrupted; resume with the same run directory and --resume')
 def heartbeat(self):
  while not self.halt.wait(self.cfg.get('heartbeat_seconds',15)):
   self.say('heartbeat',elapsed_seconds=time.monotonic()-self.started,**self.memory())
 def memory(self):
  return {'gpu_allocated_bytes':torch.cuda.memory_allocated(self.device),'gpu_reserved_bytes':torch.cuda.memory_reserved(self.device)} if self.device.type=='cuda' else {'gpu':None}
 def progress(self,done,total,started,completed_before=0,**kw):
  elapsed=time.monotonic()-started;speed=(done-completed_before)/max(elapsed,1e-9)
  self.say('progress',processed=done,processed_this_attempt=done-completed_before,total=total,elapsed_seconds=elapsed,items_per_second=speed,eta_seconds=(total-done)/max(speed,1e-9),**kw,**self.memory())
 def close(self):
  self.halt.set();self.thread.join();sys.stdout,self.origout=self.origout,None;sys.stderr=self.origerr
  for s,h in self.handlers.items():signal.signal(s,h)
  self.log.close()
 def execute(self,key,identity,fn):
  self.stage=key;p=self.path/'reports'/'stage_state.json';states=read_json(p,{})
  previous=states.get(key);fp=fingerprint(identity)
  if previous and previous['fingerprint']!=fp and previous['status']=='completed':raise ValueError(f'{key}: inputs/config/code changed: choose a new run directory')
  if previous and previous['status']=='completed':
   outputs=previous.get('outputs',{})
   if not all((self.path/k).is_file() and sha(self.path/k)==v for k,v in outputs.items()):raise ValueError(f'{key}: completed outputs changed or missing; new run required')
   self.say('skipped',reason='completed; input and output hashes verified');return
  history=previous.get('attempts',[]) if previous else []
  entry={'start':stamp(),'status':'running','fingerprint':fp};history.append(entry)
  state={'fingerprint':fp,'identity':identity,'status':'running','attempts':history};states[key]=state;atomic_json(p,states)
  self.say('running',input_fingerprint=fp)
  try:
   outputs=fn() or [];self.check_stop();state['status']='completed';state['outputs']={x.relative_to(self.path).as_posix():sha(x) for x in outputs}
  except Blocked as e:state.update(status='blocked',reason=str(e));self.say('blocked',reason=str(e))
  except Paused as e:state.update(status='pending',reason=str(e));raise
  except BaseException as e:state.update(status='failed',reason=repr(e));traceback.print_exc();raise
  finally:
   entry.update(end=stamp(),status=state['status'],reason=state.get('reason'));states[key]=state;atomic_json(p,states);self.say(state['status'],reason=state.get('reason'))
