from pathlib import Path
import hashlib,json,shutil,time
R=Path('/root/gpufree-data/tw_stage1/issue16_20261111')
for pid in [88400,89049]:
 p=Path('/proc')/str(pid);assert not p.exists() or (p/'stat').read_text().split()[2]=='Z'
source=R/'smoke';dest=R/'preflight_phase_definition_superseded'
assert source.resolve().is_relative_to(R.resolve()) and dest.resolve().is_relative_to(R.resolve()) and not dest.exists()
source.rename(dest)
shutil.copy2(R/'smoke_driver.log',R/'smoke_driver_initial_phase_preflight.log')
shutil.copy2(R/'mirror_unit_tests.log',R/'mirror_unit_tests_initial_phase_preflight.log')
shutil.copy2(R/'protocol_plan.md',R/'protocol_plan_initial_preflight.md')
p=Path('scripts/locomotion/issue16_mirror.py');s=p.read_text().replace('y[...,99:101]=-x[...,99:101]','y[...,99:101]=x[...,99:101]').replace("'(sin,cos)->(-sin,-cos), half-cycle left/right role exchange'","'Global scalar clock (sin,cos) unchanged by pure spatial reflection; no time/phase intervention'");p.write_text(s)
p=Path('tests/locomotion/test_issue16_mirror.py');s=p.read_text();a=s.index('    def test_gait_half_cycle_and_involution');b=s.index('    def test_joint_blocks',a)
replacement="""    def test_global_gait_clock_invariant(self):
        phases=torch.arange(19,dtype=torch.float32)/19*2*math.pi
        x=torch.zeros(19,101);x[:,99]=phases.sin();x[:,100]=phases.cos()
        self.assertTrue(torch.equal(self.map.observation(x)[:,99:101],x[:,99:101]))
    def test_clock_driven_policy_true_spatial_conjugacy(self):
        class ClockPolicy:
            def __call__(self,obs):
                return obs['policy'][:,9:39]+obs['policy'][:,99:100]*torch.arange(1,31)
        policy=ClockPolicy();wrapped=ReflectedPolicy(policy,self.map,'mirror')
        x=torch.arange(202,dtype=torch.float32).reshape(2,101)/101
        physical_mirror=self.map.observation(x)
        physical_mirror[:,99:101]=x[:,99:101]  # Time does not reflect with spatial coordinates.
        actual=wrapped({'policy':physical_mirror})
        expected=self.map.action(policy({'policy':x}))
        self.assertTrue(torch.equal(actual,expected))
"""
s=s[:a]+replacement+s[b:];compile(s,str(p),'exec');p.write_text(s)
p=Path('docs/stage1_issue16_20261111.md');s=p.read_text().replace('The global gait sin/cos pair is negated (half-cycle phase shift) so left/right gait roles swap.','The global gait sin/cos pair is unchanged: a pure spatial reflection does not change elapsed time. The evaluation conjugates the original policy at the SAME clock, rather than combining reflection with a temporal phase intervention.').replace('gait-phase half-cycle','global-clock invariance and clock-driven spatial policy conjugacy')
s+='\n## Preflight operator-definition correction (before formal runs)\n\nThe initial draft negated both global gait-phase features. Independent coordinate reasoning showed that this would combine spatial reflection with a half-cycle temporal intervention. The physical mirrored trajectory must retain the same exogenous clock, so the final operator leaves sin/cos unchanged. The initial64-env runtime preflight was stopped, preserved separately and excluded from scientific analysis. No heading result or acceptance threshold was used to choose this correction; no formal job had started. A clock-driven conjugacy counterexample test now detects the invalid phase shift. The30degree criterion,33 mirrored jobs plus one no-policy job, checkpoints and all other budgets are unchanged.\n'
p.write_text(s);(R/'protocol_plan.md').write_text(s)
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
(R/'operator_definition_amendment.json').write_text(json.dumps(dict(reason='Pure spatial reflection preserves the global clock. Corrected before formal evaluation; half-cycle temporal intervention removed.',formal_jobs_started=0,heading_outcomes_used=False,initial_plan_sha256=sha(R/'protocol_plan_initial_preflight.md'),final_plan_sha256=sha(R/'protocol_plan.md'),preflight_preserved_directory=str(dest),stopped_child_pid=89049,stopped_driver_pid=88400,threshold_and_formal_budget_unchanged=True,updated_unix=time.time()),indent=2)+'\n')
(R/'stop').unlink()
print('PURE_SPATIAL_CLOCK_INVARIANCE_CORRECTED')
