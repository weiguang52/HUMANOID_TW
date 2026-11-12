from pathlib import Path
p=Path('scripts/locomotion/analyze_issue16.py');s=p.read_text();needle='   if finite:\n    delta='
replacement="""   if cmd not in MAIN:
    record['operational_mirror_pattern']='not_applicable_commanded_turn_control'
    record['direction_reversal_inference_applicable']=False
    record['control_scope']='Raw same-command turning performance only. A pure reflection maps positive to negative yaw commands; same-command turn direction is not a zero-yaw bias reversal test. Original reset ordinals are preserved, not silently swapped.'
    comparisons.append(record)
    continue
   record['direction_reversal_inference_applicable']=True
   if finite:
    delta="""
assert needle in s;s=s.replace(needle,replacement);compile(s,str(p),'exec');p.write_text(s)
r=Path('/root/gpufree-data/tw_stage1/issue16_20261111')
(r/'analysis_scope_notes.md').write_text('''# Issue16 analysis scope clarifications

The frozen primary direction/magnitude interpretation applies only to stand and forward0.3/0.5/0.8 with zero yaw target. The two commanded turning conditions are retained in their original reset ordinals as supplementary raw tracking controls. Their same-command heading direction is not used as evidence of non-reversal: spatial reflection also reverses a nonzero yaw command. No supplementary turn is treated as a30degree heading-hold gate or used for the primary causal classification. The formal budget and all frozen primary criteria are unchanged.

The phase pair is a limb-independent scalar clock computed from each environment's episode_length_buf. Calling it global gait phase does not mean a wall clock shared across resets. Pure spatial reflection leaves its current value unchanged. Divergent failures may reset the two trajectories at different times; paired START conditions do not imply identical later reset/RNG histories. Full20s endpoint comparisons preserve missingness rather than dropping a censored seed.
''')
print('ZERO_YAW_INFERENCE_SCOPE_EXPLICIT')
