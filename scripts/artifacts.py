"""Inventory/export existing non-Git experiment artifacts, never silently omit them."""
from pathlib import Path
import argparse,csv,hashlib,json,os,subprocess,tarfile,time
ROOT=Path(__file__).resolve().parents[1]
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def inventory():
 git=set(subprocess.check_output(['git','ls-files','-z'],cwd=ROOT).decode().split('\0'));rows=[]
 for p in ROOT.rglob('*'):
  if not p.is_file() or any(x in p.parts for x in ['.cache','__pycache__','.git','.venv','.tools']) or p.suffix=='.partial':continue
  rel=p.relative_to(ROOT).as_posix()
  if rel in git or rel=='manifests/EXTERNAL_ARTIFACTS.csv':continue
  h=sha(p);equivalent=None
  if rel.startswith(('assets/','checkpoints/')):
   old=ROOT/'CIR-minimal-reproduction-20260919'/rel
   if old.is_file() and sha(old)==h:equivalent=old.relative_to(ROOT).as_posix()
  elif rel.startswith('archive/CIR-minimal-reproduction-20260919/'):
   old=ROOT/rel.removeprefix('archive/')
   if old.is_file() and sha(old)==h:equivalent=old.relative_to(ROOT).as_posix()
  rows.append({'path':rel,'size_bytes':p.stat().st_size,'sha256':h,'source':'see source_mapping.csv and run/config+manifests','in_git':False,'git_equivalent':equivalent or '', 'recovery':'scripts/materialize_old.py / original Git source package' if equivalent else 'new outputs regenerable from raw data/config; existing historical run evidence requires this export','purpose':'explicit fixture evidence' if rel.startswith('tests/artifacts/') else 'actual historical or experiment assets'})
 out=ROOT/'manifests/EXTERNAL_ARTIFACTS.csv';out.parent.mkdir(parents=True,exist_ok=True)
 with out.open('w',newline='',encoding='utf8') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]) if rows else ['path','size_bytes','sha256','source','in_git','git_equivalent','recovery','purpose']);w.writeheader();w.writerows(rows)
 print('Artifact inventory',len(rows),'files',sum(r['size_bytes'] for r in rows),'bytes',flush=True);return rows
def main():
 p=argparse.ArgumentParser();p.add_argument('command',choices=['inventory','export']);p.add_argument('--output',type=Path);a=p.parse_args();rows=inventory()
 if a.command=='export':
  if not a.output:raise ValueError('--output required')
  if not rows:raise ValueError('no non-Git artifacts to export')
  out=a.output.resolve()
  if out.is_relative_to(ROOT):raise ValueError('archive must be outside the project')
  if out.exists():raise FileExistsError(out)
  out.parent.mkdir(parents=True,exist_ok=True);tmp=out.with_name(out.name+'.partial')
  with tarfile.open(tmp,'w:gz') as tar:
   tar.add(ROOT/'manifests/EXTERNAL_ARTIFACTS.csv',arcname='manifests/EXTERNAL_ARTIFACTS.csv')
   for j,r in enumerate(rows,1):
    p=ROOT/r['path']
    if sha(p)!=r['sha256']:raise ValueError('asset changed during export')
    if p.is_symlink():raise ValueError('symlink not accepted')
    tar.add(p,arcname=r['path'],recursive=False)
    if j%100==0:print('artifact export',j,'/',len(rows),r['path'],flush=True)
  os.replace(tmp,out);h=sha(out);out.with_name(out.name+'.sha256').write_text(h+'  '+out.name+'\n');print('export complete',str(out),out.stat().st_size,h,flush=True)
if __name__=='__main__':main()
