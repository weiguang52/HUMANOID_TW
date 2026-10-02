import json
import sys
from pathlib import Path
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts/practice9'))
from run_native_motion_pipeline import require_manifest
from retarget_humanml3d import CUSTOM_JOINT_NAMES
from joint_coordinates import CONTRACT_VERSION


def test_derived_curriculum_requires_explicit_backend_and_all_gates(tmp_path):
    clip = tmp_path / 'clip.npz'; clip.touch()
    data = dict(backend={'backend':'s2_curriculum_v1'}, joint_coordinate_contract=CONTRACT_VERSION,
                joint_names=CUSTOM_JOINT_NAMES, target_fps=50,
                motions=[dict(file=str(clip),quality_pass=True,fk_quality_pass=True)])
    path=tmp_path/'manifest.json'
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError,match='native'): require_manifest(path,training=True)
    require_manifest(path,training=True,expected_backend='s2_curriculum_v1')
    for field in ('quality_pass','fk_quality_pass'):
        data['motions'][0][field]=False;path.write_text(json.dumps(data))
        with pytest.raises(ValueError):
            require_manifest(path,training=True,expected_backend='s2_curriculum_v1')
        data['motions'][0][field]=True
    data['joint_coordinate_contract']='obsolete';path.write_text(json.dumps(data))
    with pytest.raises(ValueError,match='Stale'):
        require_manifest(path,training=True,expected_backend='s2_curriculum_v1')
