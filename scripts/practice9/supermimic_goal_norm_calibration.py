"""Explicit, hash-bound loading of audited statistics; never load this file as a policy."""
import hashlib,json
from pathlib import Path
import torch


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def calibrated_normalizers(source, path, parent_sha256, manifest_sha256):
    source={k:v.detach().cpu() for k,v in source.items() if k.startswith(('obs_norm.','goal_norm.','critic_norm.'))}
    path=Path(path)
    data=torch.load(path,map_location='cpu',weights_only=False)
    assert data.get('kind')=='offline_training_goal_normalizer_calibration_not_policy_checkpoint'
    assert data['parent_sha256']==parent_sha256
    assert data['manifest_sha256']==manifest_sha256
    assert data['contract']=='all_training_frames_reference_root_goal_nonorientation_v1'
    audit_path=path.parent/'independent_named_audit.json'
    audit=json.loads(audit_path.read_text())
    assert audit['passed'] and audit['calibration_sha256']==digest(path)
    assert audit['frames']==data['fit_frames'] and audit['dimensions_checked']==690
    assert audit['absolute_moment_tolerance']<=1e-6
    assert all(0<=v<=1e-6 for v in audit['max_errors'].values())
    assert data['epsilon']==1e-5 and data['orientation_statistics_preserved']
    mask=data['replacement_mask']
    expected=torch.tensor([i%75 not in range(63,69) for i in range(750)])
    assert mask.dtype==torch.bool and torch.equal(mask,expected)
    assert torch.equal(data['frozen_mask'],source['goal_norm.frozen_mask'])
    assert not bool(source['goal_norm.frozen_mask'][mask].any())
    assert source['critic_norm.mean'].shape==(857,)
    assert not bool(source['critic_norm.frozen_mask'][73:823][mask].any())
    result={k:v.clone() for k,v in source.items() if k.startswith(('obs_norm.','goal_norm.','critic_norm.'))}
    for suffix in ['mean','var']:
        value=data['goal_'+suffix]
        old=source['goal_norm.'+suffix]
        assert value.shape==old.shape==(750,) and value.dtype==old.dtype
        assert torch.isfinite(value).all()
        if suffix=='var':assert bool((value>=0).all())
        assert torch.equal(value[~mask],old[~mask])
        result['goal_norm.'+suffix][mask]=value[mask]
        result['critic_norm.'+suffix][73:823][mask]=value[mask]
    metadata={'file':str(path.resolve()),'sha256':digest(path),'independent_audit_sha256':digest(audit_path),'contract':data['contract'],'replaced_goal_dimensions':690,'critic_goal_slice':[73,823],'fit_frames':data['fit_frames']}
    return result,metadata
