"""Independent all-case control/PD audit using native GPU recordings."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--collection',type=Path,required=True);a=p.parse_args();root=a.collection;assert not (root/'audit.json').exists()
r=json.loads((root/'results.json').read_text());meta=json.loads((root/'control_metadata.json').read_text());assert len(r['rows'])==6
expected={(w,amp) for w in [0,10] for amp in [.02,.05,8.]};assert {(x['warmup'],x['amplitude']) for x in r['rows']}==expected
ai=np.array(meta['action_index_for_actuator']);si=np.array([meta['state_joint_names'].index(n) for n in meta['actuator_joint_names']]);di=np.array(meta['dof_for_actuator']);assert sorted(ai.tolist())==list(range(30)) and len(set(di.tolist()))==30
scale=np.array(meta['scale'],np.float32);offset=np.array(meta['offset'],np.float32);gain=np.array(meta['gainprm'],np.float32)[:,0];bias=np.array(meta['biasprm'],np.float32)[:,:3];limits=np.array(meta['forcerange'],np.float32)
checks=[];sources=[]
for row in r['rows']:
 path=(root/row['file']).resolve();assert path.is_relative_to(root.resolve())
 with np.load(path,allow_pickle=False) as z:d={k:z[k].copy() for k in z.files}
 assert all(np.isfinite(v).all() for v in d.values());assert d['state_before'].shape==d['next_state'].shape==(61,73)
 for k in ['action','expected_target','ctrl','cached_length','cached_velocity','native_force','analytic_force','unclamped_force','saturated']:assert d[k].shape==(61,30)
 assert d['generalized_force'].shape==(61,36)
 intended=np.zeros((61,30),np.float32);intended[np.arange(1,61),np.repeat(np.arange(30),2)]=np.tile([-1,1],30)*row['amplitude'];np.testing.assert_array_equal(d['action'],intended)
 target=offset+scale*d['action'];np.testing.assert_allclose(target,d['expected_target'],atol=1e-7,rtol=0);np.testing.assert_array_equal(d['ctrl'],d['expected_target'][:,ai])
 raw=d['ctrl']*gain+bias[:,0]+bias[:,1]*d['cached_length']+bias[:,2]*d['cached_velocity'];force=np.maximum(np.minimum(raw,limits[:,1]),limits[:,0]);force_err=float(np.max(np.abs(force-d['native_force'])));assert force_err<=1e-5
 np.testing.assert_array_equal(d['analytic_force'],force)
 projected=np.zeros((61,36),np.float32);projected[:,di]=d['native_force'];np.testing.assert_array_equal(projected,d['generalized_force'])
 length_err=float(np.max(np.abs(d['cached_length']-d['next_state'][:,13+si])));velocity_err=float(np.max(np.abs(d['cached_velocity']-d['next_state'][:,43+si])))
 assert length_err<=5e-7 and velocity_err<=5e-7
 from_state=gain*(d['ctrl']-d['next_state'][:,13+si])-(-bias[:,2])*d['next_state'][:,43+si];from_state=np.maximum(np.minimum(from_state,limits[:,1]),limits[:,0]);state_torque_error=float(np.max(np.abs(from_state-d['native_force'])));assert state_torque_error<=1e-5
 saturated=(raw<limits[:,0])|(raw>limits[:,1]);np.testing.assert_array_equal(saturated,d['saturated'])
 if row['amplitude']==8:
  active=np.array([np.flatnonzero(ai==j)[0] for j in np.repeat(np.arange(30),2)]);assert saturated[np.arange(1,61),active].all()
 else:assert not saturated.any()
 assert int(d['terminated'].sum())==row['terminated_count'] and int(d['truncated'].sum())==row['truncated_count']
 checks.append(dict(warmup=row['warmup'],amplitude=row['amplitude'],native_pd_abs_error=force_err,state_pd_abs_error=state_torque_error,length_vs_post_state_abs_error=length_err,velocity_vs_post_state_abs_error=velocity_err,saturated_components=int(saturated.sum()),terminated_count=row['terminated_count'],truncated_count=row['truncated_count']))
 sources.append(dict(file=row['file'],sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
report=dict(passed=True,scope='366one-step probe worlds,30actuators each. Exact compiled native PD/ctrl mapping and stress saturation algebra only, not hardware validation or control skill. No physics/weights changes.',cache_timing_correction='Original metadata last-substep wording was inaccurate. env.step performs sim.forward after integration; recorded actuator length/velocity equal recorded post-step joint state, verified here. Original metadata preserved.',checks=checks,source_sha256=sources,promoted=False)
(root/'audit.json').write_text(json.dumps(report,indent=2,allow_nan=False));(root/'status').write_text('audited_pd_chain_complete\n');print(json.dumps(dict(passed=True,checks=checks)))
