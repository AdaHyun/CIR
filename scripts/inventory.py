"""Refresh a portable file inventory; excludes only self-reference and transient bytecode."""
from pathlib import Path
import csv,hashlib,json,os,time
ROOT=Path(__file__).resolve().parents[1]
def digest(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def main():
 mapping={}
 with (ROOT/'manifests/source_mapping.csv').open(encoding='utf8') as f:
  for row in csv.DictReader(f):mapping.setdefault(row['new_path'],[]).append(row)
 rows=[];started=time.monotonic()
 for n,p in enumerate(sorted(ROOT.rglob('*'))):
  if p.is_symlink():raise RuntimeError('symlink not portable: '+str(p))
  if not p.is_file():continue
  rel=p.relative_to(ROOT).as_posix()
  if rel in {'manifests/file_manifest.csv','manifests/PACKAGE_SHA256SUMS.txt'} or p.suffix=='.partial' or any(x in p.parts for x in ['__pycache__','.git','.venv','.cache','.tools']):continue
  sources=mapping.get(rel,[]);cat=next((c for c in ['dress','shirt','toptee'] if c in rel),'shared')
  rows.append({'new_path':rel,'size_bytes':p.stat().st_size,'sha256':digest(p),'source_package':';'.join(dict.fromkeys(x['source_package'] for x in sources)) or ('test_fixture' if rel.startswith('tests/artifacts/') else 'CIR-FULL generated'),'original_path':';'.join(x['original_path'] for x in sources),'original_relative_path':';'.join(x['original_relative_path'] for x in sources),'category':cat,'stage':'fixture' if rel.startswith('tests/') else 'archive' if rel.startswith('archive/') else 'active','split':'train' if 'train' in p.name else 'val' if 'val' in p.name else 'unknown','purpose':'real material' if sources else 'generated code/report/run evidence','compatibility':'byte-identical verified copy' if sources else 'see local validation reports','issues':'archived absolute paths are provenance only' if rel.startswith('archive/') else ''})
  if len(rows)%200==0:print(json.dumps({'stage':'inventory','files':len(rows),'current':rel,'elapsed':time.monotonic()-started}),flush=True)
 out=ROOT/'manifests/file_manifest.csv';temp=out.with_suffix('.partial')
 with temp.open('w',newline='',encoding='utf8') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 os.replace(temp,out)
 checksum=ROOT/'manifests/PACKAGE_SHA256SUMS.txt'
 checksum.write_text(''.join(r['sha256']+'  '+r['new_path']+'\n' for r in rows)+digest(out)+'  manifests/file_manifest.csv\n',encoding='utf8')
 print(json.dumps({'stage':'inventory-completed','files':len(rows),'bytes':sum(r['size_bytes'] for r in rows),'elapsed':time.monotonic()-started}),flush=True)
if __name__=='__main__':main()
