"""Real package interface, random lightweight model and synthetic image ONLY."""
from pathlib import Path
import importlib.util,sys,unittest,os
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import torch
@unittest.skipUnless(importlib.util.find_spec('open_clip') and importlib.util.find_spec('torchvision'),'optional CPU OpenCLIP stack is not installed')
class RealLibraryInterfaceTest(unittest.TestCase):
 def test_openclip_dqu_rc_forward(self):
  os.environ.setdefault('HF_HOME',str(Path(__file__).resolve().parents[1]/'.cache/huggingface'))
  import open_clip
  from PIL import Image
  from cir_full.upstream_model import DQU_CIR,DQU_PFP_REFCOND_CIR
  torch.set_num_threads(2);torch.manual_seed(19)
  encoder,train,val=open_clip.create_model_and_transforms('ViT-B-32',pretrained=None)
  tokenizer=open_clip.get_tokenizer('ViT-B-32');image=val(Image.new('RGB',(96,96),'red'));images=torch.stack([image,image.flip(-1)]);texts=['explicit synthetic garment, but test brighter','explicit synthetic garment, but test striped']
  # No pretrained weights or public performance claims. Exercise the actual OpenCLIP package API.
  with patch('cir_full.upstream_model.create_backbone',return_value=(encoder,train,val,tokenizer)):
   for cls in [DQU_CIR,DQU_PFP_REFCOND_CIR]:
    model=cls(hidden_dim=512,dropout=.1,backbone='test-random-real-package',pretrained='NONE').eval()
    with torch.no_grad():q=model.extract_query(texts,images);i=model.extract_target(images);loss=model.compute_loss(texts,images,images)['ranking']
    self.assertEqual(tuple(q.shape),(2,512));self.assertTrue(torch.isfinite(q).all());self.assertTrue(torch.allclose(q.norm(dim=1),torch.ones(2),atol=2e-5));self.assertTrue(torch.isfinite(i).all());self.assertGreater(float(loss),0)
if __name__=='__main__':unittest.main(verbosity=2)
