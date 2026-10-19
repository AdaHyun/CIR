"""Copy only authenticated historical artifacts shipped in the original Git source package."""
from pathlib import Path
import sys,json,shutil,hashlib
ROOT=Path(__file__).resolve().parents[1]
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 src=ROOT/'CIR-minimal-reproduction-20260919';manifest=src/'manifests/SHA256SUMS.txt';rows=[]
 for line in manifest.read_text().splitlines():
  expected,name=line.split(maxsplit=1);rel=Path(name)
  if rel.parts[0] not in ['assets','checkpoints'] or rel.suffix!='.pt':continue
  p=src/rel;dest=ROOT/rel
  if digest(p)!=expected:raise ValueError('historical model/feature SHA mismatch: '+str(p))
  dest.parent.mkdir(parents=True,exist_ok=True)
  if dest.exists() and digest(dest)!=expected:raise FileExistsError('different existing asset preserved: '+str(dest))
  if not dest.exists():
   temp=dest.with_name(dest.name+'.partial');shutil.copy2(p,temp)
   if digest(temp)!=expected:raise ValueError('copy hash failed')
   temp.replace(dest)
  print('old asset verified',dest.relative_to(ROOT),p.stat().st_size,expected,flush=True);rows.append(str(dest.relative_to(ROOT)))
 if len(rows)!=12:raise RuntimeError('expected 12 historical files; do not call incomplete materialization ready')
if __name__=='__main__':main()
