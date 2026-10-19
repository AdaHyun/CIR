"""Explicit synthetic fixtures. Never write to assets/ or runs/."""
import copy, json, os, shutil, subprocess, sys, tempfile, unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import torch
from cir_full.runtime import ROOT, Run, Paused, atomic_json, sha, fingerprint
from cir_full.assets import load_bundle, load_pt
from cir_full.model import FeatureReranker, training_item
from cir_full.engine import train, evaluate, build_topk, read_jsonl

def fixture(base):
 torch.manual_seed(5);cfg=json.loads((ROOT/'configs/default.json').read_text());cfg.update(asset_dir=str(base/'assets'),checkpoint_dir=str(base/'checkpoints'),data_dir=str(base/'data'),output_dir=str(base/'outputs'),fixture=True,eval_batch_size=3,log_every=2,heartbeat_seconds=30,_code_fingerprint='explicit-test-fixture')
 cfg['training'].update(hidden_dim=6,batch_size=3,save_every_steps=1,save_every_seconds=60)
 cfg['categories']['dress'].update(epochs=2,train_k=4,train_pool=8,dropout=.2)
 p=base/'assets'/'dress';p.mkdir(parents=True)
 I=torch.nn.functional.normalize(torch.randn(12,8),dim=1);ids=[f'FIXTURE-{j}' for j in range(12)];provenance={}
 for split in ['train','val']:
  Q=torch.nn.functional.normalize(torch.randn(9,8),dim=1);src=ids[:9];target=ids[1:10];sims=Q@I.T
  for j in range(9):sims[j,j]=-torch.inf
  vals,indices=sims.topk(10,dim=1);top={'sorted_ids':[[ids[x] for x in row] for row in indices.tolist()],'sorted_scores':vals.tolist(),'source_ids':src,'target_ids':target}
  for kind,data in [('query_features',{'query_features':Q,'source_ids':src,'target_ids':target}),('image_features',{'features':I,'image_ids':ids}),('topk',top)]:torch.save(data,p/f'dress_{split}_{kind}.pt')
  provenance[split]={'producer_checkpoint_sha256':'FIXTURE-ONLY-NOT-REAL','feature_definition':{'mode':'explicit synthetic unit vectors'},'files':{f.name:sha(f) for f in p.glob(f'dress_{split}_*.pt')}}
 atomic_json(p/'provenance.json',provenance);return cfg

def run_train(path,cfg,pause=None,resume=False):
 r=Run(path,cfg,{'fixture':True},torch.device('cpu'),resume)
 try:return train(cfg,'dress',r,pause)
 finally:r.close()

class WorkflowTests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  torch.set_num_threads(2);cls.base=ROOT/'tests'/'artifacts'/('fixture-'+str(os.getpid()));cls.base.mkdir(parents=True,exist_ok=True);cls.cfg=fixture(cls.base)
 def test_01_true_similarity_and_resume(self):
  b=load_bundle(self.cfg,'dress','train');c={**self.cfg['training'],**self.cfg['categories']['dress']};x=training_item(b,0,0,c)
  self.assertTrue(torch.equal(x[2],x[1]@x[0]));self.assertNotEqual(float(x[2][x[3]]),1.)
  self.assertEqual(x[4][x[3]],b['query']['target_ids'][0]);self.assertNotIn(b['query']['source_ids'][0],x[4]);self.assertEqual(len(set(x[4])),len(x[4]))
  a=self.base/'continuous';z=self.base/'resumed';run_train(a,self.cfg)
  with self.assertRaises(Paused):run_train(z,self.cfg,2)
  paused=load_pt(z/'checkpoints/dress/last.pt');self.assertEqual(paused['cursor'],6);self.assertEqual(paused['global_step'],2)
  run_train(z,self.cfg,resume=True)
  one=load_pt(a/'checkpoints/dress/last.pt');two=load_pt(z/'checkpoints/dress/last.pt')
  self.assertEqual(one['global_step'],6);self.assertEqual(two['global_step'],6)
  def equal(x,y):
   if isinstance(x,torch.Tensor):self.assertTrue(torch.equal(x,y))
   elif isinstance(x,dict):
    self.assertEqual(x.keys(),y.keys())
    for k in x:equal(x[k],y[k])
   elif isinstance(x,(list,tuple)):
    for i,j in zip(x,y):equal(i,j)
   else:self.assertEqual(x,y)
  equal(one['model_state_dict'],two['model_state_dict']);equal(one['optimizer'],two['optimizer']);equal(one['rng']['torch_cpu'],two['rng']['torch_cpu'])
  rows1=read_jsonl(a/'metrics/train-steps-dress.jsonl');rows2=read_jsonl(z/'metrics/train-steps-dress.jsonl')
  self.assertEqual(len(rows2),len({r['global_step'] for r in rows2}))
  for x,y in zip(rows1,rows2):
   for k in ['global_step','epoch','cursor','query_ids','candidate_ids','labels','loss','average_loss','lr']:self.assertEqual(x[k],y[k])
  self.assertEqual(read_jsonl(a/'metrics/train-epochs-dress.jsonl'),read_jsonl(z/'metrics/train-epochs-dress.jsonl'))
  changed=copy.deepcopy(self.cfg);changed['categories']['dress']['lr']*=2
  with self.assertRaisesRegex(ValueError,'mismatch'):run_train(z,changed,resume=True)
 def test_02_eval_chunk_reuse_and_topk(self):
  p=self.base/'eval';r=Run(p,self.cfg,{'fixture':True},torch.device('cpu'))
  try:
   out=evaluate(self.cfg,'dress',r,kind='cn');h=sha(out[1]);mtime=(p/'predictions/cn-dress/chunks/00000000.json').stat().st_mtime_ns
   evaluate(self.cfg,'dress',r,kind='cn');self.assertEqual(h,sha(out[1]));self.assertEqual(mtime,(p/'predictions/cn-dress/chunks/00000000.json').stat().st_mtime_ns)
   out2=build_topk(self.cfg,'dress',r,'train');built=load_pt(out2[0]);b=load_bundle(self.cfg,'dress','train')
   for j,row in enumerate(built['sorted_ids']):self.assertNotIn(b['query']['source_ids'][j],row)
   # Corrupt committed chunk must never be accepted.
   chunk=p/'predictions/cn-dress/chunks/00000000.json';x=json.loads(chunk.read_text());x['rows'][0]['target_rank']=999;atomic_json(chunk,x)
   with self.assertRaisesRegex(ValueError,'corrupt'):evaluate(self.cfg,'dress',r,kind='cn')
  finally:r.close()
 def test_03_mapping_rejected(self):
  b=load_bundle(self.cfg,'dress','train');bad=copy.deepcopy(b['topk']);bad['source_ids']=list(reversed(bad['source_ids']));dest=self.base/'invalid'/'dress';dest.mkdir(parents=True)
  for p in (self.base/'assets/dress').glob('dress_train_*.pt'):shutil.copy2(p,dest/p.name)
  torch.save(bad,dest/'dress_train_topk.pt');cfg={**self.cfg,'asset_dir':str(dest.parent)}
  with self.assertRaisesRegex(AssertionError,'identity'):load_bundle(cfg,'dress','train')
 def test_04_relocated_entrypoint(self):
  # Copy only runtime files; no runtime can reach back to its original checkout.
  relocated=self.base/'moved project'
  for d in ['src','scripts','configs']:shutil.copytree(ROOT/d,relocated/d,ignore=shutil.ignore_patterns('__pycache__'),dirs_exist_ok=True)
  result=subprocess.run([sys.executable,str(relocated/'scripts/cir.py'),'--help'],cwd=self.base,text=True,capture_output=True)
  self.assertEqual(result.returncode,0,result.stderr);self.assertIn('pipeline',result.stdout)
  shutil.copytree(self.base/'assets',relocated/'assets')
  config=copy.deepcopy(self.cfg)
  for k,v in [('asset_dir','assets'),('checkpoint_dir','checkpoints'),('data_dir','data'),('output_dir','tests/artifacts/test_outputs')]:config[k]=v
  atomic_json(relocated/'configs'/'fixture.json',config)
  result=subprocess.run([sys.executable,str(relocated/'scripts/cir.py'),'multik','--config','configs/fixture.json','--category','dress','--run-id','relocated-fixture'],cwd=self.base,text=True,capture_output=True)
  (self.base/'relocation.log').write_text(result.stdout+result.stderr,encoding='utf8')
  self.assertEqual(result.returncode,0,result.stdout+result.stderr)
  self.assertEqual(json.loads((relocated/'tests/artifacts/test_outputs/relocated-fixture/metrics/cn-dress.json').read_text())['count'],9)
if __name__=='__main__':unittest.main(verbosity=2)
