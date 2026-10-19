"""Explicit synthetic images/tiny encoder; never experimental evidence."""
import copy,json,os,shutil,subprocess,sys,time,unittest,signal,tempfile
from pathlib import Path
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from cir_full.runtime import ROOT,atomic_json,read_json,sha
from cir_full.assets import load_pt

def fixture(base):
 torch.manual_seed(123);root=base/'data';cat='dress';(root/'captions').mkdir(parents=True);(root/'image_splits').mkdir();(root/'resized_image'/cat).mkdir(parents=True)
 for split,N in [('train',8),('val',6)]:
  ids=[f'EXPLICIT-TEST-{split}-{i}' for i in range(N)];rows=[]
  for j,imageid in enumerate(ids):
   torch.save(torch.rand(3,8,8),root/'resized_image'/cat/(imageid+'.pt'));rows.append({'candidate':imageid,'target':ids[(j+1)%N],'captions':['test synthetic brighter','test synthetic stripe']})
  atomic_json(root/'captions'/f'cap.{cat}.{split}.json',rows);atomic_json(root/'image_splits'/f'split.{cat}.{split}.json',ids);atomic_json(root/'captions'/f'image_captions_{cat}_{split}.json',{x:'explicit test source garment' for x in ids})
 atomic_json(root/'captions/correction_dict_dress.json',{})
 cfg=read_json(ROOT/'configs/full.json');cfg.update(fixture=True,backbone='fixture-tiny',dimension=8,data_dir=str(root.relative_to(ROOT)),pretrained='FIXTURE-NO-WEIGHTS',output_dir=str((base/'outputs').relative_to(ROOT)),batch_size=4,export_batch_size=3,amp=False,threads=2,log_every=1,save_every_steps=1,benchmark_steps=3,benchmark_warmup=1,candidate_k=5,patience=5)
 for x in cfg['stages'].values():x.update(epochs=2,batch_size=4)
 cfg['rr_overrides']={'training':{'hidden_dim':6,'batch_size':3,'save_every_steps':1},'categories':{c:{'epochs':2,'train_k':3,'train_pool':4} for c in ['dress','shirt','toptee']}}
 atomic_json(base/'config.json',cfg);return cfg
class FullPipelineTests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  cls.base=ROOT/'tests/artifacts'/('full-fixture-'+str(time.time_ns()));cls.base.mkdir(parents=True);cls.cfg=fixture(cls.base)
 @classmethod
 def call(cls,stage,run='complete',extra=(),config=None):
  cmd=[sys.executable,'-u',str(ROOT/'scripts/full.py'),stage,'--config',str(config or cls.base/'config.json'),'--run-id',run,'--category','dress']+list(extra)
  p=subprocess.run(cmd,cwd=tempfile.gettempdir(),text=True,capture_output=True,timeout=90);(cls.base/f'{stage}-{run}-{time.time_ns()}.log').write_text(p.stdout+p.stderr,encoding='utf8');return p
 def test_01_complete_pipeline(self):
  p=self.call('pipeline');self.assertEqual(p.returncode,0,p.stdout+p.stderr)
  run=self.base/'outputs/complete';state=read_json(run/'reports/stage_state.json');self.assertTrue(all(x['status']=='completed' for x in state.values()))
  for method in ['dqu','rc','cn']:
   self.assertTrue((run/f'checkpoints/{method}-dress/best.pt').is_file())
   for split in ['train','val']:
    q=load_pt(run/f'assets/{method}/dress/dress_{split}_query_features.pt');self.assertEqual(q['metadata']['split'],split);self.assertTrue(torch.isfinite(q['query_features']).all())
  self.assertTrue(read_json(run/'reports/final.json')['fixture_only'])
  prov=read_json(run/'assets/cn/dress/provenance.json');self.assertEqual(prov['train']['producer_checkpoint_sha256'],prov['val']['producer_checkpoint_sha256'])
  model=load_pt(run/'checkpoints/dress/best.pt');self.assertNotIn('legacy',model['args'].get('score_rule','true-similarity'))
  p=self.call('pipeline',extra=['--resume']);self.assertEqual(p.returncode,0,p.stdout+p.stderr);self.assertIn('"event": "skipped"',p.stdout)
 def test_02_upstream_resume(self):
  p=self.call('dqu',run='interrupted',extra=['--stop-after-steps','1']);self.assertEqual(p.returncode,130,p.stdout+p.stderr)
  p=self.call('dqu',run='interrupted',extra=['--resume']);self.assertEqual(p.returncode,0,p.stdout+p.stderr)
  p=self.call('dqu',run='continuous');self.assertEqual(p.returncode,0,p.stdout+p.stderr)
  a=load_pt(self.base/'outputs/interrupted/checkpoints/dqu-dress/latest.pt');b=load_pt(self.base/'outputs/continuous/checkpoints/dqu-dress/latest.pt')
  for k in a['model_state_dict']:self.assertTrue(torch.equal(a['model_state_dict'][k],b['model_state_dict'][k]),k)
  self.assertEqual(a['scheduler'],b['scheduler']);self.assertEqual(a['global_step'],4)
  self.assertEqual([s['query_ids'] for s in a['steps']],[s['query_ids'] for s in b['steps']]);self.assertEqual([s['loss'] for s in a['steps']],[s['loss'] for s in b['steps']])
  cmd=[sys.executable,'-u',str(ROOT/'scripts/full.py'),'dqu','--config',str(self.base/'config.json'),'--run-id','actual-signal']
  proc=subprocess.Popen(cmd,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True);lines=[];sent=False
  for line in proc.stdout:
   lines.append(line)
   if 'upstream-checkpoint-saved' in line and not sent:proc.send_signal(signal.SIGTERM);sent=True
  code=proc.wait(timeout=30);(self.base/'actual-upstream-signal.log').write_text(''.join(lines),encoding='utf8');self.assertTrue(sent);self.assertEqual(code,130)
  p=self.call('dqu',run='actual-signal',extra=['--resume']);self.assertEqual(p.returncode,0,p.stdout+p.stderr)
  signal_ckpt=load_pt(self.base/'outputs/actual-signal/checkpoints/dqu-dress/latest.pt')
  for k in b['model_state_dict']:self.assertTrue(torch.equal(signal_ckpt['model_state_dict'][k],b['model_state_dict'][k]),k)
  changed=copy.deepcopy(self.cfg);changed['stages']['dqu']['lr']*=2;atomic_json(self.base/'changed.json',changed)
  p=self.call('dqu',run='interrupted',extra=['--resume'],config=self.base/'changed.json');self.assertNotEqual(p.returncode,0)
 def test_03_experiments_and_boundaries(self):
  p=self.call('experiments',extra=['--resume','--rerank-ks','1,3,5']);self.assertEqual(p.returncode,0,p.stdout+p.stderr)
  p=self.call('experiments',extra=['--resume','--rerank-ks','100']);self.assertNotEqual(p.returncode,0)
  p=self.call('online-cost',extra=['--resume']);self.assertEqual(p.returncode,0,p.stdout+p.stderr)
  p=self.call('legacy-control',extra=['--resume']);self.assertEqual(p.returncode,0,p.stdout+p.stderr)
  p=self.call('benchmark',run='short-benchmark');self.assertEqual(p.returncode,0,p.stdout+p.stderr)
  self.assertTrue((self.base/'outputs/short-benchmark/profiling/benchmark-dress-dqu.json').exists())
 def test_04_corruption_and_relocation(self):
  project=self.base/'moved project';
  for folder in ['src','scripts','configs']:shutil.copytree(ROOT/folder,project/folder,ignore=shutil.ignore_patterns('__pycache__'))
  shutil.copytree(self.base/'data',project/'data/fixture');cfg=copy.deepcopy(self.cfg);cfg['data_dir']='data/fixture';cfg['output_dir']='tests/artifacts/moved-output';atomic_json(project/'configs/fixture.json',cfg)
  p=subprocess.run([sys.executable,str(project/'scripts/full.py'),'doctor','--config','configs/fixture.json','--run-id','relocated'],cwd=tempfile.gettempdir(),capture_output=True,text=True,timeout=30);self.assertEqual(p.returncode,0,p.stdout+p.stderr)
  run=self.base/'outputs/complete';p=run/'assets/cn/dress/blocks/val-image-00000000.pt';p.write_bytes(p.read_bytes()+b'corrupt')
  from cir_full.upstream import committed_chunk
  with self.assertRaises(ValueError):committed_chunk(p,'different-token')
if __name__=='__main__':unittest.main(verbosity=2)
