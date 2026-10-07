from pathlib import Path
import subprocess
from PIL import Image,ImageDraw
r=Path('validation_artifacts/tw215_flat_walk');o=Path('validation_artifacts/tw215_review')
for clip in ['000832','000016','000211']:
 canvas=Image.new('RGB',(1600,500),'white');draw=ImageDraw.Draw(canvas)
 for i,v in enumerate(['baseline','neck_only']):
  p=o/(clip+'_'+v+'.jpg')
  subprocess.run(['ffmpeg','-v','error','-y','-i',str(r/v/'videos'/(clip+'.mp4')),'-vf','fps=1/2,scale=400:225,tile=4x1','-frames:v','1',str(p)],check=True)
  canvas.paste(Image.open(p),(0,i*250+25));draw.text((10,i*250+5),v+' / '+clip+' / about 1,3,5,7 seconds; recordings may reset after failure',fill='black')
 canvas.save(o/(clip+'_sheet.jpg'))
print('sheets complete')
