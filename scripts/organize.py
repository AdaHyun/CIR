"""Audit, then verified incremental copy. Sources are never modified."""
from pathlib import Path
import argparse, csv, hashlib, json, os, shutil, time, zipfile, tarfile
ROOT=Path(__file__).resolve().parents[1]
def digest(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--source-root',type=Path,required=True);a=ap.parse_args()
 stamp=time.strftime('%Y%m%dT%H%M%S'); logs=ROOT/'reports'/'organization';logs.mkdir(parents=True,exist_ok=True)
 log=(logs/f'{stamp}-{time.time_ns()}.jsonl').open('a',encoding='utf8')
 def say(**kw):
  rec={'time':time.time(),**kw};print(json.dumps(rec,ensure_ascii=False),flush=True);log.write(json.dumps(rec,ensure_ascii=False)+'\n');log.flush()
 plans=[];exclusions=[]
 for name in ['CIR-minimal-reproduction-20260919','phase_packages']:
  src=a.source_root/name
  if not src.is_dir():raise FileNotFoundError(src)
  say(stage='scan',directory=str(src))
  for base,dirs,files in os.walk(src,followlinks=False):
   for d in list(dirs):
    p=Path(base)/d
    if p.is_symlink():raise RuntimeError(f'symlink must be resolved explicitly: {p}')
    if d in {'.git','__pycache__','.venv','venv','node_modules','.cache'}:
     exclusions.append({'path':str(p),'reason':'nonportable environment / regenerable cache / Git metadata'});dirs.remove(d)
   for f in sorted(files):
    p=Path(base)/f
    if p.is_symlink():raise RuntimeError(f'symlink: {p}')
    rel=p.relative_to(src);s=digest(p)
    dest=ROOT/'archive'/name/rel
    plans.append((p,dest,name,s))
    if p.stat().st_size>1024*1024:say(stage='scan-file',path=str(p),bytes=p.stat().st_size,sha256=s)
    # Preserve full package plus explicit active asset copies, without mixing sources.
    if name.startswith('CIR-minimal') and rel.parts[0] in {'assets','checkpoints'}:
     plans.append((p,ROOT/rel,name,s))
    if zipfile.is_zipfile(p):
     with zipfile.ZipFile(p) as z:
      say(stage='archive-inspection',path=str(p),members=len(z.infolist()),names=[x.filename for x in z.infolist()])
    elif tarfile.is_tarfile(p):
     with tarfile.open(p) as z:say(stage='archive-inspection',path=str(p),names=z.getnames())
 needed=sum(p.stat().st_size for p,d,_,s in plans if not d.exists() or digest(d)!=s)
 free=shutil.disk_usage(ROOT).free;say(stage='preflight',files=len(plans),needed_bytes=needed,free_bytes=free)
 if needed+128*1024*1024>free:raise RuntimeError('insufficient disk space')
 rows=[];copied=skipped=0
 for n,(p,d,name,s) in enumerate(plans,1):
  try:
   if d.exists() and digest(d)!=s:
    d=ROOT/'archive'/'conflicts'/name/p.relative_to(a.source_root/name)
    d=d.with_name(d.name+'.'+s)
   d.parent.mkdir(parents=True,exist_ok=True)
   if d.exists() and digest(d)==s:skipped+=1
   else:
    tmp=d.with_name(d.name+'.partial')
    with p.open('rb') as src,tmp.open('wb') as out:
     done=0;t=time.monotonic()
     for b in iter(lambda:src.read(8*1024*1024),b''):
      out.write(b);done+=len(b)
      if time.monotonic()-t>5:say(stage='copy-progress',path=str(p),bytes=done,total=p.stat().st_size);t=time.monotonic()
     out.flush();os.fsync(out.fileno())
    if digest(tmp)!=s:raise RuntimeError(f'copy hash mismatch: {p}')
    os.replace(tmp,d);copied+=1
   rel=str(p.relative_to(a.source_root/name));cat=next((c for c in ['dress','shirt','toptee'] if c in rel),'shared')
   rows.append({'source_package':name,'original_path':str(p),'original_relative_path':rel,'new_path':d.relative_to(ROOT).as_posix(),'size_bytes':p.stat().st_size,'sha256':s,'category':cat,'stage':next((x for x in p.parts if x[:2].isdigit()),'minimal'),'split':'train' if 'train' in p.name else 'val' if 'val' in p.name else 'unknown','purpose':p.suffix,'compatibility':'byte-identical copy; semantic audit separate','issues':'see reports/LOCAL_ASSET_AUDIT.md'})
   if n%50==0 or p.stat().st_size>1024*1024:say(stage='copy',current=str(p),processed=n,total=len(plans),copied=copied,skipped=skipped,failed=0)
  except BaseException as e:say(stage='failed',path=str(p),reason=repr(e));raise
 existing=ROOT/'manifests'/'source_mapping.csv'
 if existing.exists():
  with existing.open(encoding='utf8',newline='') as f:prior=list(csv.DictReader(f))
  combined={(x['original_path'],x['new_path'],x['sha256']):x for x in prior}
  combined.update({(x['original_path'],x['new_path'],x['sha256']):x for x in rows});rows=list(combined.values())
 for name in ['source_mapping.csv']:
  out=ROOT/'manifests'/name;tmp=out.with_suffix('.partial')
  with tmp.open('w',newline='',encoding='utf8') as f:
   w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
  os.replace(tmp,out)
 (ROOT/'manifests'/'exclusions.json').write_text(json.dumps(exclusions,indent=2),encoding='utf8')
 say(stage='completed',copied=copied,skipped=skipped,total=len(plans),failed=0)
if __name__=='__main__':main()
