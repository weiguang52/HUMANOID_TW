exec(open('/root/gpufree-data/tw_stage1/issue17_20261113/check_reward_phase.py').read().split('results={}')[0])
phase=((steps*.02)%.4/.4);shift=(((steps+10)*.02)%.4/.4)
pa=(phase[:,None]+torch.tensor([0.,.5]))%1.;pb=(shift[:,None]+torch.tensor([0.,.5]))%1.
stance=pa<.55;shifted=(pb<.55).flip(1);bad=(stance!=shifted).any(1)
distance=torch.stack([abs(pa-.55),torch.minimum(pa,1-pa)],dim=-1).amin((1,2))
print('STATE_MASK_MISMATCH',int(bad.sum()),'ALL_NEAR_BOUNDARY',bool((distance[bad]<1e-4).all()),'MAX_BOUNDARY_DISTANCE',float(distance[bad].max()))
print('FIRST_CASES',[(int(steps[i]),pa[i].tolist(),pb[i].tolist()) for i in torch.where(bad)[0][:12]])
# Exact integer clock oracle:20steps/cycle and threshold11/20,offset10/20.
integer_phase=(steps[:,None]+torch.tensor([0,10]))%20
integer_shift=((steps+10)[:,None]+torch.tensor([0,10]))%20
assert torch.equal(integer_phase<11,(integer_shift<11).flip(1))
print('INTEGER_ORACLE_HALF_CYCLE_EXACT',True)
away=distance>1e-4
for name,fn,kw in [('feet_gait',gait,{'sensor_cfg':cfg}),('feet_swing_height',swing,{'asset_cfg':cfg,'target_height':.03,'rest_height':.011})]:
 x=fn(a,period=.4,offset=[0.,.5],threshold=.55,command_name='base_velocity',**kw);y=fn(b,period=.4,offset=[0.,.5],threshold=.55,command_name='base_velocity',**kw)
 print(name,'AWAY_FROM_BOUNDARIES_EXACT',bool(torch.equal(x[away],y[away])),'TOTAL',int(away.sum()))
