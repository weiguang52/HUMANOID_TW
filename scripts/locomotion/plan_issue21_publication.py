"""Prepare exact new Git objects after validated remote-Linux commit creation."""
import subprocess,json
from issue21_common import ROOT,save
BASE='b8ddc926a017d479869dfcc9183d15c653a5a4d7'
head=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip();assert head!=BASE and subprocess.check_output(['git','rev-parse','HEAD^'],text=True).strip()==BASE;assert subprocess.check_output(['git','branch','--show-current'],text=True).strip()=='supermimic/tw-locomotion-20261108'
objects=subprocess.check_output(['git','rev-list','--objects',head,'^'+BASE],text=True).splitlines();oids=[x.split()[0] for x in objects];checked=subprocess.check_output(['git','cat-file','--batch-check=%(objectname) %(objecttype) %(objectsize)'],input=('\n'.join(oids)+'\n').encode()).decode().splitlines();blobs=[]
for row in checked:
 oid,kind,size=row.split()
 if kind=='blob':blobs.append({'oid':oid,'bytes':int(size)})
assert all(x['bytes']<=90*1024*1024 for x in blobs)
save('publication/new_git_objects.json',{'objects':objects});save('publication/api_blob_upload_plan.json',{'base':BASE,'final':head,'tree':subprocess.check_output(['git','rev-parse','HEAD^{tree}'],text=True).strip(),'blobs':blobs});print('EXACT_GIT_PUBLICATION_PLANNED',head,len(blobs),sum(x['bytes'] for x in blobs))
