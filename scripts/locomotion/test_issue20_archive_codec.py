"""Independent byte restoration checks for scientific binary/text archives."""
import unittest,tempfile,random
from pathlib import Path
from issue20_archive_codec import encode,restore,digest,contained
class CodecTests(unittest.TestCase):
 def test_raw_and_gzip_chunks_restore_exactly(self):
  with tempfile.TemporaryDirectory(prefix='issue20-codec-test-',dir='/tmp') as d:
   root=Path(d);src=root/'source';dst=root/'archive';out=root/'restore';src.mkdir();dst.mkdir();samples={'binary.npz':random.Random(20).randbytes(100001),'geometry.usd':b'#usda 1.0\n'+b'0.1 0.2 -0.3\n'*20000}
   for name,data in samples.items():
    p=src/name;p.write_bytes(data);row=encode(p,dst,name,root,chunk_bytes=4096);self.assertIsNotNone(row);self.assertTrue(all(x['bytes']<=4096 for x in row['parts']));restore(dst,row,out);self.assertEqual((out/name).read_bytes(),data)
 def test_corruption_and_parent_traversal_rejected(self):
  with tempfile.TemporaryDirectory(prefix='issue20-codec-test-',dir='/tmp') as d:
   root=Path(d);p=root/'input.npz';p.write_bytes(b'abcde'*1000);dst=root/'archive';dst.mkdir();row=encode(p,dst,'nested/raw.npz',root,chunk_bytes=1024);part=dst/row['parts'][0]['path'];part.write_bytes(b'corrupted')
   with self.assertRaises(AssertionError):restore(dst,row,root/'out')
   with self.assertRaises(AssertionError):contained(root,'../outside')
 def test_shareable_video_keeps_original_path(self):
  with tempfile.TemporaryDirectory(prefix='issue20-codec-test-',dir='/tmp') as d:
   root=Path(d);p=root/'clip.mp4';p.write_bytes(b'video-payload'*1000);dst=root/'archive';dst.mkdir();self.assertIsNone(encode(p,dst,'videos/clip.mp4',root,chunk_bytes=1024));self.assertEqual(digest(p),digest(dst/'videos/clip.mp4'))
 def test_small_file_exact_copy(self):
  with tempfile.TemporaryDirectory(prefix='issue20-codec-test-',dir='/tmp') as d:
   root=Path(d);p=root/'input';p.write_bytes(b'raw\r\nterminal\x00bytes');dst=root/'archive';dst.mkdir();self.assertIsNone(encode(p,dst,'small.log',root));self.assertEqual(digest(p),digest(dst/'small.log'))
if __name__=='__main__':unittest.main()
