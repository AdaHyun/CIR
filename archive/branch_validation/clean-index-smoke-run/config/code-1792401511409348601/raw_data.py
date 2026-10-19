"""FashionIQ records: explicit identities and modification-only visual writing."""
from __future__ import annotations
import json,random,string,os
from pathlib import Path
import numpy as np
import torch
from torch import nn
from .runtime import ROOT,Blocked,resolve,sha,fingerprint,relative,atomic_json
os.environ.setdefault('HF_HOME',str(ROOT/'.cache/huggingface'))
class TinyEncoder(nn.Module):
 """Synthetic-test backend, never an experimental model."""
 def __init__(self):
  super().__init__();self.image=nn.Linear(3*8*8,8);self.text=nn.Embedding(257,8)
 def encode_image(self,x):return self.image(x.flatten(1))
 def encode_text(self,x):return self.text(x).mean(1)
def tiny_tokenizer(texts):return torch.tensor([[ord(c)%256+1 for c in t[:24].ljust(24)] for t in texts])
def tiny_transform(x):return torch.as_tensor(x).float()
def create_backbone(name,pretrained):
 if name=='fixture-tiny':return TinyEncoder(),tiny_transform,tiny_transform,tiny_tokenizer
 import open_clip
 p=resolve(pretrained)
 if not p.is_file():raise Blocked('public CLIP weights missing: run scripts/prepare.py model --kind clip')
 record=__import__('json').loads(p.with_suffix(p.suffix+'.integrity.json').read_text()) if p.with_suffix(p.suffix+'.integrity.json').exists() else None
 if not record or record.get('sha256')!=sha(p):raise Blocked('public CLIP integrity record missing or SHA mismatch; prepare/import verified weights first')
 model,train,val=open_clip.create_model_and_transforms(name,pretrained=str(p))
 return model,train,val,open_clip.get_tokenizer(name)
def correct(text,corrections):
 tokens=str(text).lower().translate(str.maketrans({p:' ' for p in string.punctuation})).split()
 return ' '.join(corrections.get(t,t) for t in tokens)
class FashionData:
 def __init__(self,cfg,category,split):
  self.cfg,self.category,self.split=cfg,category,split;self.root=resolve(cfg['data_dir'])
  path=self.root/'captions'/f'cap.{category}.{split}.json'
  if not path.is_file():raise Blocked(f'missing official annotations: {relative(path)}')
  print(json.dumps({'phase':'load-annotations','category':category,'split':split,'path':relative(path),'bytes':path.stat().st_size}),flush=True)
  raw=json.loads(path.read_text(encoding='utf8'));self.correction_path=self.root/'captions'/f'correction_dict_{category}.json'
  self.corrections=json.loads(self.correction_path.read_text()) if self.correction_path.exists() else {}
  self.caption_path=self.root/'captions'/f'image_captions_{category}_{split}.json'
  self.keyword_path=self.root/'captions'/f'keywords_in_mods_{category}.json'
  if not self.caption_path.is_file():raise Blocked(f'missing source-image captions {relative(self.caption_path)}; import existing or run prepare.py captions')
  captions=json.loads(self.caption_path.read_text());keywords=json.loads(self.keyword_path.read_text()) if self.keyword_path.exists() else {}
  self.rows=[]
  split_path=self.root/'image_splits'/f'split.{category}.{split}.json'
  if not split_path.exists():raise Blocked(f'official split missing {relative(split_path)}')
  split_ids=set(json.loads(split_path.read_text()));used=set()
  for idx,x in enumerate(raw):
   source,target=x['candidate'],x['target'];assert source!=target and source in split_ids and target in split_ids
   if source not in captions:raise Blocked(f'missing caption for source image {source}')
   if not isinstance(captions[source],str) or not captions[source].strip():raise ValueError('source-image caption must be a nonempty string with explicit ID mapping')
   mod=' and '.join(correct(t,self.corrections) for t in x['captions'])
   key=source+'_'+target
   if cfg['query']['keywords']=='historical' and key not in keywords:raise Blocked('historical keyword entry missing: '+key)
   if cfg['query']['keywords'] in ['historical','auto'] and key in keywords:writing=keywords[key][-1]
   else:writing=mod # explicit new-run alternative, no hidden API/cache or target identity as feature
   if not isinstance(writing,str):raise ValueError('keyword cache must provide a real string, not a guessed ID/vector mapping')
   qid=f'{category}:{split}:{idx}:{source}:{target}'
   self.rows.append({'query_id':qid,'source_id':source,'target_id':target,'modification_text':x['captions'],'writing':writing,'text':str(captions[source])+', but '+mod});used.update([source,target])
  self.gallery_ids=sorted(used);self.gallery_index={x:i for i,x in enumerate(self.gallery_ids)}
  files=[path,split_path,self.caption_path]+([self.correction_path] if self.correction_path.exists() else [])+([self.keyword_path] if cfg['query']['keywords'] in ['historical','auto'] and self.keyword_path.exists() else [])
  self.files=files;self.identity={relative(p):sha(p) for p in files}
  # Verify actual image content identity once per stage/run, not just filenames.
  for count,imageid in enumerate(self.gallery_ids,1):
   p=self.image_path(imageid)
   if not p.is_file():raise Blocked('missing actual FashionIQ image '+relative(p))
   self.identity[relative(p)]=sha(p)
   if count%500==0 or count==len(self.gallery_ids):print(json.dumps({'phase':'verify-images','category':category,'split':split,'count':count,'total':len(self.gallery_ids),'current':relative(p)}),flush=True)
  self.fingerprint=fingerprint(self.identity)
 def image_path(self,imageid):return self.root/'resized_image'/self.category/(imageid+('.pt' if self.cfg.get('fixture') else '.jpg'))
 def image(self,imageid,transform,writing=None):
  p=self.image_path(imageid)
  if self.cfg.get('fixture'):
   return transform(torch.load(p,weights_only=True))
  from PIL import Image
  with Image.open(p) as image:img=image.convert('RGB').copy()
  if writing and self.cfg['query']['visual_writing']:
   import cv2
   # Exact historical layout: 512px, comma-delimited lines, OpenCV white box/black text.
   arr=cv2.resize(np.asarray(img)[:,:,::-1].copy(),(512,512));font=cv2.FONT_HERSHEY_SIMPLEX;y=15
   for text in writing.split(', '):
    if not text:continue
    (w,h),baseline=cv2.getTextSize(text,font,.7,5);origin=(15,y+h)
    cv2.rectangle(arr,(14,origin[1]-2-baseline),(15+w,origin[1]+h),(255,255,255),-1)
    cv2.putText(arr,text,origin,font,.5,(0,0,0),2,8);y+=h+2+baseline
   img=Image.fromarray(cv2.cvtColor(arr,cv2.COLOR_BGR2RGB))
  return transform(img)
 def batch(self,indices,model,train=True,include_targets=True):
  transform=model.preprocess_train if train else model.preprocess_val;rows=[self.rows[i] for i in indices]
  refs=torch.stack([self.image(x['source_id'],transform,x['writing']) for x in rows]);targets=torch.stack([self.image(x['target_id'],transform) for x in rows]) if include_targets else None
  return rows,refs,targets
