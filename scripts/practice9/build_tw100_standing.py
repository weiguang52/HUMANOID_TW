"""Select the entire standing class; copy contact slices byte-for-byte."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def build(source, output):
    d = json.loads(source.read_text())
    if output.exists():
        raise FileExistsError(output)
    spec = d['contact_reference']
    contact_path = Path(spec['file'])
    if not contact_path.is_absolute():
        contact_path = source.parent / contact_path
    assert hashlib.sha256(contact_path.read_bytes()).hexdigest() == spec['sha256']
    with np.load(contact_path, allow_pickle=False) as src:
        assert src['clip_ids'].tolist() == [r['id'] for r in d['motions']]
        lengths = src['lengths']
        assert lengths.tolist() == [r['frames'] for r in d['motions']]
        starts = np.r_[0, np.cumsum(lengths)]
        chosen = [i for i, r in enumerate(d['motions']) if r['category'] == 'standing_upper']
        assert chosen
        indices = np.concatenate([np.arange(starts[i], starts[i+1]) for i in chosen])
        bundle = {k: src[k].copy() for k in src.files}
        for k in ('contact', 'known', 'source_contact4'):
            bundle[k] = src[k][indices].copy()
        for k in ('clip_ids', 'lengths'):
            bundle[k] = src[k][chosen].copy()
    rows = [dict(d['motions'][i], weight=1.0 / len(chosen) / (int(d['motions'][i]['frames']) - 1)) for i in chosen]
    output.parent.mkdir(parents=True, exist_ok=True)
    sidecar = output.with_suffix('.contacts.npz')
    np.savez_compressed(sidecar, **bundle)
    d['motions'] = rows
    d['contact_reference'] = dict(spec, file=str(sidecar.resolve()),
        sha256=hashlib.sha256(sidecar.read_bytes()).hexdigest())
    d['tw100_selection'] = dict(source=str(source), category='standing_upper',
        note='All class clips; contact/known/source_contact4 selected unchanged, no retiming.')
    output.write_text(json.dumps(d, indent=2))
    return len(rows)

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    print('clips', build(args.source, args.output))
