"""Relabel known existing FVM results without rerunning simulations.

Only title rectangles and table text are changed. Frame count, duration, all
pixels outside those rectangles, and numerical notebook results are verified.
This is a label-only refresh, not a new computation or a new validation run.
"""
from pathlib import Path
import io,base64,hashlib,json,argparse,shutil
from PIL import Image,ImageDraw,ImageFont,ImageChops,ImageSequence
from matplotlib import font_manager
ROOT=Path(__file__).resolve().parents[1]
FONT=font_manager.findfont('DejaVu Sans')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def label(im,rect,text,xy,size):
 bg=im.getpixel((rect[0],rect[1])); d=ImageDraw.Draw(im);d.rectangle(rect,fill=bg)
 font=font_manager.findfont(font_manager.FontProperties(family='DejaVu Sans',weight='light' if size>=24 else 'normal'))
 d.text(xy,text,font=ImageFont.truetype(font,size),fill='black',anchor='lt')
def outside_unchanged(before,after,rects):
 delta=ImageChops.difference(before.convert('RGB'),after.convert('RGB'))
 d=ImageDraw.Draw(delta)
 for r in rects:d.rectangle(r,fill=0)
 assert delta.getbbox() is None,'label-only edit changed plot pixels'
def gif(path,expected,edits,backup):
 if sha(path)!=expected:raise ValueError(f'{path.name}: not the known original; refusing a second or unverified edit')
 shutil.copy2(path,backup/path.name)
 src=Image.open(path);frames=[];dur=[];original=[]
 for fr in ImageSequence.Iterator(src):
  old=fr.convert('RGB');im=old.copy();original.append(old);dur.append(fr.info.get('duration',250))
  # Antialias text using only the original frame's palette so no plot pixels
  # are altered by GIF quantization and no white boxes cover the canvas tint.
  palette=Image.new('P',(1,1)); colors=[rgb for count,rgb in old.getcolors(256)]
  palette.putpalette([v for rgb in colors for v in rgb]+[0]*(768-3*len(colors)))
  for rect,text,xy,size in edits:
   label(im,rect,text,xy,size)
   crop=im.crop((rect[0],rect[1],rect[2]+1,rect[3]+1))
   im.paste(crop.quantize(palette=palette,dither=Image.Dither.NONE).convert('RGB'),(rect[0],rect[1]))
  outside_unchanged(old,im,[e[0] for e in edits])
  indexed=im.quantize(colors=256,dither=Image.Dither.NONE)
  assert ImageChops.difference(im,indexed.convert('RGB')).getbbox() is None,'GIF quantization was not lossless'
  frames.append(indexed)
 frames[0].save(path,save_all=True,append_images=frames[1:],duration=dur,loop=src.info.get('loop',0),disposal=2,optimize=False)
 check=Image.open(path);assert check.n_frames==len(original)
 for i,fr in enumerate(ImageSequence.Iterator(check)):
  assert fr.info.get('duration',250)==dur[i]
  outside_unchanged(original[i],fr,[e[0] for e in edits])
 return {'path':str(path.relative_to(ROOT)),'sha256':sha(path),'frames':len(frames),'treatment':'label-only refresh from original frames; no simulation rerun'}
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--solver-framing',action='store_true',help='Clarify the already-relabelled Kansas GIFs as two FVM solvers');ap.add_argument('--backup-dir',type=Path,required=True);a=ap.parse_args();a.backup_dir.mkdir(parents=True,exist_ok=True)
 report=[]
 if a.solver_framing:
  report.append(gif(ROOT/'image/methods_compare_kansas.gif','acccd64ce1fb987ea017cbcf7051f9d966e7c50b455ffa1ce6f236d2e49b62d6',[
   ((35,35,450,70),'USGS MODFLOW 6: pressure',(40,38),26),
   ((493,35,932,70),'USGS MODFLOW 6: temperature',(495,38),26),
   ((8,8,830,30),'Mississippian interval, Kansas: FVM vs USGS MODFLOW 6, 0.05 m³/s',(11,11),16),
   ((95,602,420,642),'FVM: pressure',(137,608),28),
   ((550,602,905,642),'FVM: temperature',(581,608),28),
  ],a.backup_dir))
  report.append(gif(ROOT/'image/fluids_compare_kansas.gif','0e8774209551bd16e2825076ad82a4d53fe9367ba8f01bb8da31836ca5f131d0',[
   ((8,8,1000,30),'Mississippian interval, Kansas: fluid injection with FVM over 100 years',(11,11),16),
  ],a.backup_dir))
  (a.backup_dir/'solver-framing-report.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2));return
 report.append(gif(ROOT/'image/methods_compare_kansas.gif','8523c36869fb1c5c191af89b72f5b282bc4dba92c5e9d64eb0d884094286b44e',[
 ((8,8,745,30),'Mississippian interval, Kansas: cold-water doublet, MODFLOW 6 vs FVM, 0.05 m³/s',(11,11),16),
 ((95,602,402,642),'FVM: pressure',(137,608),28),
 ((550,602,889,642),'FVM: temperature',(581,608),28),
 ],a.backup_dir))
 report.append(gif(ROOT/'image/fluids_compare_kansas.gif','bce0ffd5847b956a98936bcb967409287d39c39daea1e29f80b07f995cbf4a95',[
 ((8,8,680,30),'Mississippian interval, Kansas: fluid injection with FVM over 100 years',(11,11),16),
 ],a.backup_dir))
 for name in ['13_cornerpoint_and_co2_ve.ipynb','14_flow_diagnostics.ipynb']:
  p=ROOT/'examples'/name;shutil.copy2(p,a.backup_dir/name);nb=json.loads(p.read_text())
  for c in nb['cells']:
   for o in c.get('outputs',[]):
    for k in ['text/plain','text/markdown']:
     if k in o.get('data',{}):
      v=o['data'][k];s=''.join(v) if isinstance(v,list) else v;s=s.replace('FIM-OBL','FVM').replace('FIM water–oil','FVM water–oil').replace('FIM run:','FVM run:');o['data'][k]=s.splitlines(keepends=True) if isinstance(v,list) else s
  if name.startswith('13_'):
   data=nb['cells'][12]['outputs'][1]['data'];old=Image.open(io.BytesIO(base64.b64decode(''.join(data['image/png'])))).convert('RGB');new=old.copy()
   label(new,(371,24,431,42),'FVM:',(373,26),13);outside_unchanged(old,new,[(371,24,431,42)])
   out=io.BytesIO();new.save(out,format='PNG');data['image/png']=base64.b64encode(out.getvalue()).decode()
  for c in nb['cells']:
   s=''.join(c.get('source',[]))
   s=s.replace('Saved outputs below predate this wording update and may still say FIM-OBL or FIM; their labels will be refreshed in a later rendering pass.','Saved outputs were relabelled from the original results; no simulation was rerun.')
   s=s.replace('Saved outputs below predate this wording update and may still say FIM; their labels will be refreshed in a later rendering pass.','Saved outputs were relabelled from the original results; no simulation was rerun.')
   c['source']=s.splitlines(keepends=True)
  p.write_text(json.dumps(nb,ensure_ascii=False,indent=1)+'\n');report.append({'path':str(p.relative_to(ROOT)),'sha256':sha(p),'treatment':'saved table/title relabel only; original numerical results and execution history preserved'})
 for rel in ['examples/_build_nb13.py','examples/_build_nb14.py']:
  p=ROOT/rel;s=p.read_text();s=s.replace('Saved outputs below predate this wording update and may still say FIM-OBL or FIM; their labels will be refreshed in a later rendering pass.','Saved outputs were relabelled from the original results; no simulation was rerun.').replace('Saved outputs below predate this wording update and may still say FIM; their labels will be refreshed in a later rendering pass.','Saved outputs were relabelled from the original results; no simulation was rerun.');p.write_text(s)
 p=ROOT/'README.md';s=p.read_text();s=s.replace('*The saved animation still shows the previous FIM-OBL labels; these mean FVM and will be refreshed in a later rendering pass.*','*Animation labels were refreshed to FVM from the original frames; no simulation was rerun.*').replace('*The saved animation still shows the previous FIM-OBL label; this means FVM and will be refreshed in a later rendering pass.*','*Animation labels were refreshed to FVM from the original frames; no simulation was rerun.*');p.write_text(s)
 (a.backup_dir/'label-refresh-report.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
if __name__=='__main__':main()
