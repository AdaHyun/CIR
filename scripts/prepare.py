#!/usr/bin/env python3
"""Explicit local preparation. Never called implicitly by training; no paid API."""
from pathlib import Path
import argparse,contextlib,hashlib,json,os,shutil,sys,threading,time,traceback,urllib.request
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from cir_full.runtime import ROOT,resolve,relative,atomic_json,read_json,sha,Tee
os.environ.setdefault('HF_HOME',str(ROOT/'.cache/huggingface'))
FASHION_SHA='f71b16c3b09ef03b02c0fed160fa072fb3b8aa16'
MODELS={'clip':('laion/CLIP-ViT-H-14-laion2B-s32B-b79K','weights/clip'),'captioner':('Salesforce/blip-image-captioning-base','weights/captioner')}
def verified_copy(source,dest,expected=None):
 source=Path(source);h=sha(source)
 if expected and h.lower()!=expected.lower():raise ValueError('source SHA256 mismatch')
 dest.parent.mkdir(parents=True,exist_ok=True)
 if dest.exists():
  if sha(dest)!=h:raise FileExistsError('different existing material retained: '+str(dest))
  print('verified existing',relative(dest),flush=True);return h
 tmp=dest.with_name(dest.name+'.partial');shutil.copy2(source,tmp)
 if sha(tmp)!=h:raise ValueError('copied bytes mismatch')
 os.replace(tmp,dest);return h
def annotations(a):
 out=resolve(a.data_dir);records=[];trusted=read_json(ROOT/'configs/fashioniq_annotations.json');expected_by_name={x['target_relative_path']:x for x in trusted}
 for cat in ['dress','shirt','toptee']:
  for split in ['train','val']:
   for folder,name in [('captions',f'cap.{cat}.{split}.json'),('image_splits',f'split.{cat}.{split}.json')]:
    p=out/folder/name;expected=expected_by_name[folder+'/'+name]['sha256'];url=f'https://raw.githubusercontent.com/XiaoxiaoGuo/fashion-iq/{FASHION_SHA}/{folder}/{name}'
    if not p.exists():
     p.parent.mkdir(parents=True,exist_ok=True)
     # Small immutable annotation files are restarted on interruption; no partial JSON is accepted.
     with urllib.request.urlopen(url,timeout=60) as f:data=f.read()
     parsed=json.loads(data)
     if hashlib.sha256(data).hexdigest()!=expected:raise ValueError('pinned official annotation SHA mismatch')
     if not isinstance(parsed,list) or not parsed:raise ValueError('invalid official annotation/split')
     tmp=p.with_name(p.name+'.partial');tmp.write_bytes(data);os.replace(tmp,p)
    else:
     if sha(p)!=expected:raise ValueError('existing annotation differs from verified pinned official content; preserve it, do not silently relabel it')
     parsed=json.loads(p.read_text())
     if not isinstance(parsed,list) or not parsed:raise ValueError('invalid existing annotation/split')
    records.append({'path':relative(p),'bytes':p.stat().st_size,'sha256':sha(p),'url':url,'source_revision':FASHION_SHA});print('annotation verified',records[-1],flush=True)
 atomic_json(out/'annotations.manifest.json',records)
def model(a):
 from huggingface_hub import HfApi,hf_hub_download
 repo,default=MODELS[a.kind];dest=resolve(a.dest or default);os.environ.setdefault('HF_HOME',str(ROOT/'.cache/huggingface'));api=HfApi(token=False);info=api.model_info(repo,revision=a.revision,files_metadata=True);revision=info.sha
 existing=read_json(dest/'download.manifest.json',[])
 if existing and any(x['revision']!=revision for x in existing):raise ValueError('another pinned model revision already exists; choose a different --dest, do not overwrite it')
 wanted=['open_clip_pytorch_model.bin'] if a.kind=='clip' else [x.rfilename for x in info.siblings if x.rfilename in ['pytorch_model.bin','model.safetensors','config.json','preprocessor_config.json','tokenizer_config.json','special_tokens_map.json','vocab.txt','tokenizer.json']]
 if a.kind=='captioner' and 'model.safetensors' in wanted:wanted=[x for x in wanted if x!='pytorch_model.bin']
 records=[]
 for name in wanted:
  item=next(x for x in info.siblings if x.rfilename==name);lfs=getattr(item,'lfs',None);expected=getattr(lfs,'sha256',None) if lfs else None
  if isinstance(lfs,dict):expected=lfs.get('sha256')
  print('downloading/verifying',repo,revision,name,'published SHA256',expected,flush=True)
  path=Path(hf_hub_download(repo,name,revision=revision,local_dir=str(dest),token=False));actual=sha(path)
  if expected and actual!=expected:raise ValueError('authoritative LFS SHA256 mismatch; artifact not usable')
  if name.endswith(('.bin','.safetensors')) and not expected:raise ValueError('model has no authoritative LFS SHA256; import an independently verified file explicitly')
  entry={'repository':repo,'revision':revision,'filename':name,'sha256':actual,'bytes':path.stat().st_size,'published_lfs_sha256':expected};records.append(entry);atomic_json(path.with_suffix(path.suffix+'.integrity.json'),entry)
 atomic_json(dest/'download.manifest.json',records)
def images(a):
 src=Path(a.source).resolve();dest=resolve(a.data_dir)/'resized_image'/a.category
 if not src.is_dir():raise FileNotFoundError(src)
 mapping=[]
 for n,p in enumerate(sorted(src.rglob('*.jpg')),1):
  imageid=p.stem
  if any(x in imageid for x in ['/', '\\']):raise ValueError('invalid image ID')
  target=dest/(imageid+'.jpg');h=verified_copy(p,target);mapping.append({'source':str(p),'path':relative(target),'sha256':h,'bytes':target.stat().st_size})
  if n%100==0:print('images copied/verified',n,'latest',imageid,flush=True)
 if not mapping:raise ValueError('no .jpg files found; do not treat an empty import as success')
 atomic_json(dest/'import.manifest.json',mapping)
def caches(a):
 src=Path(a.source).resolve();dest=resolve(a.data_dir)/'captions';records=[]
 for p in sorted(src.rglob('*.json')):
  if not (p.name.startswith(('image_captions_','keywords_in_mods_','correction_dict_'))):continue
  if not isinstance(json.loads(p.read_text(encoding='utf8')),dict):raise ValueError('cache must be dict: '+str(p))
  h=verified_copy(p,dest/p.name);records.append({'source':str(p),'path':relative(dest/p.name),'sha256':h});print('cache imported',p.name,flush=True)
 if not records:raise ValueError('no recognized real cache files')
 atomic_json(dest/'cache-import.manifest.json',records)
def captions(a):
 import torch
 from PIL import Image
 from transformers import BlipProcessor,BlipForConditionalGeneration
 root=resolve(a.data_dir);directory=resolve(a.model_dir);manifest=read_json(directory/'download.manifest.json')
 if not manifest:raise ValueError('caption model manifest missing; use prepare.py model --kind captioner')
 for r in manifest:
  if sha(directory/r['filename'])!=r['sha256']:raise ValueError('caption model changed')
 device=torch.device(a.device)
 if device.type=='cuda' and not torch.cuda.is_available():raise RuntimeError('CUDA requested but unavailable')
 processor=BlipProcessor.from_pretrained(str(directory),local_files_only=True);net=BlipForConditionalGeneration.from_pretrained(str(directory),local_files_only=True).to(device).eval()
 for split in ['train','val']:
  annotation=root/'captions'/f'cap.{a.category}.{split}.json';rows=json.loads(annotation.read_text());ids=sorted(set(x['candidate'] for x in rows));out=root/'captions'/f'image_captions_{a.category}_{split}.json';metadata=out.with_suffix('.generation.json');identity={'model_files':manifest,'generation':{'num_beams':3,'do_sample':False,'max_new_tokens':32},'annotation_sha256':sha(annotation),'input_image_sha256':{i:sha(root/'resized_image'/a.category/(i+'.jpg')) for i in ids},'precision':'float32','source':'free local public BLIP; not historical authors cache'}
  existing=read_json(out,{})
  if existing and read_json(metadata)!=identity:raise ValueError('existing captions have another/unknown source; preserve them and choose import, do not overwrite')
  atomic_json(metadata,identity)
  for start in range(0,len(ids),a.batch_size):
   batch=[x for x in ids[start:start+a.batch_size] if x not in existing]
   if not batch:continue
   imgs=[]
   for imageid in batch:
    with Image.open(root/'resized_image'/a.category/(imageid+'.jpg')) as image:imgs.append(image.convert('RGB').copy())
   inputs=processor(images=imgs,return_tensors='pt').to(device)
   with torch.no_grad():tokens=net.generate(**inputs,num_beams=3,do_sample=False,max_new_tokens=32)
   text=processor.batch_decode(tokens,skip_special_tokens=True);existing.update(zip(batch,text));atomic_json(out,existing);print('captions committed',a.category,split,min(start+a.batch_size,len(ids)),'/',len(ids),flush=True)
def main():
 p=argparse.ArgumentParser();sub=p.add_subparsers(dest='command',required=True)
 sub.add_parser('annotations')
 x=sub.add_parser('model');x.add_argument('--kind',choices=MODELS,required=True);x.add_argument('--revision',default='main');x.add_argument('--dest')
 x=sub.add_parser('images');x.add_argument('--source',required=True);x.add_argument('--category',choices=['dress','shirt','toptee'],required=True)
 x=sub.add_parser('caches');x.add_argument('--source',required=True)
 x=sub.add_parser('import');x.add_argument('--source',required=True);x.add_argument('--dest',required=True);x.add_argument('--sha256',required=True)
 x=sub.add_parser('captions');x.add_argument('--category',choices=['dress','shirt','toptee'],required=True);x.add_argument('--model-dir',default='weights/captioner');x.add_argument('--device',default='cpu');x.add_argument('--batch-size',type=int,default=8)
 for x in sub.choices.values():x.add_argument('--data-dir',default='data/fashioniq')
 a=p.parse_args();logdir=ROOT/'reports/preparation';logdir.mkdir(parents=True,exist_ok=True);log=(logdir/(time.strftime('%Y%m%dT%H%M%S')+'-'+str(time.time_ns())+'.log')).open('a',encoding='utf8');oldout,olderr=sys.stdout,sys.stderr;sys.stdout=Tee(sys.stdout,log);sys.stderr=Tee(sys.stderr,log);stop=threading.Event()
 def heartbeat():
  while not stop.wait(15):print('preparation heartbeat',a.command,time.strftime('%H:%M:%S'),flush=True)
 thread=threading.Thread(target=heartbeat,daemon=True);thread.start()
 try:
  print('preparation start',vars(a),flush=True)
  if a.command=='import':
   dest=resolve(a.dest);h=verified_copy(Path(a.source),dest,a.sha256);atomic_json(dest.with_suffix(dest.suffix+'.integrity.json'),{'source':str(Path(a.source).resolve()),'sha256':h,'bytes':dest.stat().st_size,'verification':'explicit user-supplied independent SHA256'})
  else:globals()[a.command](a)
  print('preparation completed',a.command,flush=True)
 except BaseException:traceback.print_exc();return 1
 finally:stop.set();thread.join();sys.stdout,sys.stderr=oldout,olderr;log.close()
 return 0
if __name__=='__main__':raise SystemExit(main())
