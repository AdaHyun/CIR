"""Real SIGTERM and CLI pipeline state checks, synthetic fixtures only."""
import copy,json,os,signal,subprocess,sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from test_workflow import fixture
from cir_full.runtime import ROOT, atomic_json
from cir_full.assets import load_pt
from cir_full.engine import read_jsonl
import torch
class SignalResumeTest(unittest.TestCase):
 def test_signal_and_resume(self):
  base=ROOT/'tests'/'artifacts'/('signal-fixture-'+str(os.getpid()));base.mkdir(parents=True)
  cfg=fixture(base);cfg['categories']['dress']['epochs']=4;cfg['log_every']=1
  config=base/'config.json';atomic_json(config,cfg)
  def cmd(name,resume=False):return [sys.executable,'-u',str(ROOT/'scripts/cir.py'),'train','--config',str(config),'--category','dress','--run-dir',str(base/name)]+(['--resume'] if resume else [])
  proc=subprocess.Popen(cmd('interrupted'),stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
  lines=[];sent=False
  for line in proc.stdout:
   lines.append(line)
   if 'checkpoint-saved' in line and not sent:proc.send_signal(signal.SIGTERM);sent=True
  code=proc.wait(timeout=30);(base/'signal-terminal.log').write_text(''.join(lines),encoding='utf8')
  self.assertTrue(sent);self.assertEqual(code,130)
  before=load_pt(base/'interrupted/checkpoints/dress/last.pt');self.assertLess(before['global_step'],12)
  result=subprocess.run(cmd('interrupted',True),text=True,capture_output=True,timeout=45);(base/'resume-terminal.log').write_text(result.stdout+result.stderr,encoding='utf8');self.assertEqual(result.returncode,0,result.stdout+result.stderr)
  result=subprocess.run(cmd('continuous'),text=True,capture_output=True,timeout=45);(base/'continuous-terminal.log').write_text(result.stdout+result.stderr,encoding='utf8');self.assertEqual(result.returncode,0,result.stdout+result.stderr)
  a=load_pt(base/'interrupted/checkpoints/dress/last.pt');b=load_pt(base/'continuous/checkpoints/dress/last.pt')
  self.assertEqual(a['global_step'],12)
  for k in a['model_state_dict']:self.assertTrue(torch.equal(a['model_state_dict'][k],b['model_state_dict'][k]),k)
  ar=read_jsonl(base/'interrupted/metrics/train-steps-dress.jsonl');br=read_jsonl(base/'continuous/metrics/train-steps-dress.jsonl')
  self.assertEqual(len(ar),12);self.assertEqual([r['global_step'] for r in ar],list(range(1,13)))
  for a,b in zip(ar,br):
   for key in ['query_ids','candidate_ids','labels','loss']:self.assertEqual(a[key],b[key])
  result=subprocess.run(cmd('interrupted',True),text=True,capture_output=True,timeout=30);(base/'completed-restart.log').write_text(result.stdout+result.stderr,encoding='utf8');self.assertEqual(result.returncode,0,result.stdout+result.stderr);self.assertIn('"event": "skipped"',result.stdout)
if __name__=='__main__':unittest.main(verbosity=2)
