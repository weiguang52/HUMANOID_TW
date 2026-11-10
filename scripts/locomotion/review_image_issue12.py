"""Stream a small review image in ASCII chunks; never copy remote data to local disk."""
import argparse,base64,io,json,math
from pathlib import Path
from PIL import Image
p=argparse.ArgumentParser()
p.add_argument("path")
p.add_argument("--chunk",type=int,default=0)
p.add_argument("--chunk_chars",type=int,default=10000)
p.add_argument("--max_width",type=int,default=1200)
a=p.parse_args()
path=Path(a.path).resolve()
assert path.is_relative_to(Path("/root/gpufree-data/tw_stage1"))
assert path.suffix.lower() in [".png",".jpg",".jpeg"]
im=Image.open(path).convert("RGB")
if im.width>a.max_width:im=im.resize((a.max_width,round(im.height*a.max_width/im.width)))
buffer=io.BytesIO();im.save(buffer,format="JPEG",quality=88)
data=base64.b64encode(buffer.getvalue()).decode()
chunks=math.ceil(len(data)/a.chunk_chars);assert 0<=a.chunk<chunks
print("REVIEW_META "+json.dumps({"chunks":chunks,"chunk":a.chunk,"width":im.width,"height":im.height,"mime":"image/jpeg"}))
print("REVIEW_IMAGE_BEGIN")
part=data[a.chunk*a.chunk_chars:(a.chunk+1)*a.chunk_chars]
for i in range(0,len(part),64):print(part[i:i+64])
print("REVIEW_IMAGE_END",flush=True)
