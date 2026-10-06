from pathlib import Path
import subprocess,io
from PIL import Image,ImageDraw
root=Path('validation_artifacts/tw154_flat_walk');out=Path('validation_artifacts/tw154_review')
for cid in ['000832','001191','005309','009516','000016','000124','000346','000211']:
 canvas=Image.new('RGB',(1200,660),'white');draw=ImageDraw.Draw(canvas)
 for row,v in enumerate(['source','path','rate']):
  p=root/'rate'/'humanml_source'/f'{cid}_source.mp4' if v=='source' else root/v/'videos'/f'{cid}.mp4'
  duration=float(subprocess.check_output(['ffprobe','-v','error','-show_entries','format=duration','-of','csv=p=0',str(p)]))
  for col,frac in enumerate([.15,.3,.45,.6,.75,.9]):
   data=subprocess.check_output(['ffmpeg','-nostdin','-v','error','-ss',str(duration*frac),'-i',str(p),'-frames:v','1','-vf','scale=200:190:force_original_aspect_ratio=decrease','-f','image2pipe','-vcodec','mjpeg','-threads','1','pipe:1'])
   im=Image.open(io.BytesIO(data));canvas.paste(im,(col*200,row*220+25))
   draw.text((col*200+3,row*220+3),f'{cid} {v} {duration*frac:.2f}s',fill='black')
 canvas.save(out/f'{cid}.jpg')
print('sheets done')
