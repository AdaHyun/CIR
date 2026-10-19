"""Explicit local packaging only. No upload, no deletion, no network access."""
from pathlib import Path
import argparse, hashlib, tarfile, time
ROOT=Path(__file__).resolve().parents[1]
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def main():
 a=argparse.ArgumentParser();a.add_argument('--output',type=Path,required=True);x=a.parse_args();out=x.output.resolve()
 if out.is_relative_to(ROOT):raise ValueError('archive must be outside CIR-FULL to prevent recursive inclusion')
 if out.exists():raise FileExistsError(out)
 sums=ROOT/'manifests/PACKAGE_SHA256SUMS.txt';print('Verifying package inputs before packaging',flush=True)
 for n,line in enumerate(sums.read_text(encoding='utf8').splitlines(),1):
  expected,rel=line.split('  ',1);p=ROOT/rel
  if not p.is_file() or sha(p)!=expected:raise ValueError('inventory changed; rerun scripts/inventory.py: '+rel)
  if n%200==0:print(f'validated {n} inputs',flush=True)
 files=[p for p in sorted(ROOT.rglob('*')) if p.is_file() and not any(x in p.parts for x in ['__pycache__','.git','.venv','.cache','.tools']) and p.suffix!='.partial']
 total=sum(p.stat().st_size for p in files);print(f'Packing {len(files)} files, {total} input bytes to {out}',flush=True)
 tmp=out.with_name(out.name+'.partial');out.parent.mkdir(parents=True,exist_ok=True)
 with tarfile.open(tmp,'w:gz') as tar:
  for n,p in enumerate(files,1):
   if p.is_symlink():raise ValueError('symlink: '+str(p))
   tar.add(p,arcname=str(Path('CIR-FULL')/p.relative_to(ROOT)),recursive=False)
   if n%100==0 or p.stat().st_size>8*1024*1024:print(f'packed {n}/{len(files)} {p.relative_to(ROOT)} ({p.stat().st_size} bytes)',flush=True)
 tmp.rename(out);print(f'Completed {out}, bytes={out.stat().st_size}, sha256={sha(out)}',flush=True)
if __name__=='__main__':main()
