"""Preserve prior local logs/validation, explicitly supplied roots, without moving sources."""
from pathlib import Path
import argparse,csv,hashlib,os,shutil,time
ROOT=Path(__file__).resolve().parents[1]
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--record-root',type=Path,required=True);ap.add_argument('--validation-copy',type=Path);a=ap.parse_args()
 m=ROOT/'manifests/source_mapping.csv';rows=list(csv.DictReader(m.open(encoding='utf8')));fields=list(rows[0]);keys={(x['original_path'],x['new_path'],x['sha256']) for x in rows}
 plans=[(p,ROOT/'archive/onboarding/logs'/p.name,'onboarding execution log') for p in sorted(a.record_root.glob('cir-*.log')) if p.is_file()]
 if a.validation_copy:
  plans.extend((p,ROOT/'archive/onboarding/validation'/p.relative_to(a.validation_copy),'onboarding validation') for p in sorted(a.validation_copy.rglob('*')) if p.is_file())
 needed=sum(p.stat().st_size for p,d,_ in plans if not d.exists());free=shutil.disk_usage(ROOT).free
 print(f'Preflight {len(plans)} files; {needed} bytes plus conflict copies may be needed; free={free}',flush=True)
 if free<sum(p.stat().st_size for p,_,_ in plans)+128*1024*1024:raise RuntimeError('insufficient free space for worst-case conflicts')
 for n,(p,d,source) in enumerate(plans,1):
  h=sha(p);d.parent.mkdir(parents=True,exist_ok=True)
  if d.exists() and sha(d)!=h:d=d.with_name(d.name+'.'+h)
  if not d.exists():
   tmp=d.with_name(d.name+'.partial');shutil.copy2(p,tmp)
   if sha(tmp)!=h:raise ValueError('copy hash mismatch: '+str(p))
   os.replace(tmp,d)
  if sha(d)!=h:raise ValueError('destination hash mismatch: '+str(d))
  key=(str(p),d.relative_to(ROOT).as_posix(),h)
  if key not in keys:
   x=dict.fromkeys(fields,'');x.update(original_path=str(p),original_relative_path=p.name,new_path=key[1],size_bytes=p.stat().st_size,sha256=h,source_package=source,category='shared',stage='onboarding',split='unknown',purpose='preserved actual execution records',compatibility='byte-identical copy',issues='historical records, not new experimental results');rows.append(x);keys.add(key)
  if n%25==0:print(f'verified {n}/{len(plans)} {p}',flush=True)
 tmp=m.with_suffix('.partial')
 with tmp.open('w',newline='',encoding='utf8') as f:
  w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows);f.flush();os.fsync(f.fileno())
 os.replace(tmp,m);print(f'Preservation complete: {len(plans)} files verified, {len(rows)} source mappings',flush=True)
if __name__=='__main__':main()
