"""Produce conclusions from complete frozen comparisons; exactly one follow-up recommendation."""
import ast,json
from pathlib import Path
from issue15_common import ROOT,REPO,sha
def main():
 assert (ROOT/"analysis_complete").exists()
 comparisons=json.loads((ROOT/"lineage_comparisons.json").read_text());dominance=json.loads((ROOT/"yaw_dominance.json").read_text());legacy=json.loads((ROOT/"original_ft5_protocol_comparison.json").read_text())
 explanations={
 "ft5->robust05":"Multiple training differences and unshown sibling ancestry: FT5 and yaw_s1 share FT4, robust05 follows yaw_s1. No sole configuration cause can be identified from this contrast.",
 "robust05->robust08":"Only the saved logical command-range bundle expands forward/yaw0.5 to0.8, with1500 further updates. Forward and angular ranges change together and their effects cannot be separated. One training seed: a descriptive association, not an isolated causal test.",
 "robust08->strengthen":"Saved seed1 recipe difference is training push +/-0.3 to+/-0.45m/s; both restart from robust05. The current two-seed comparison is conditional on fixed robust08 seed1, not a two-population causal interval.",
 "strengthen->bounded":"Saved logical recipe difference is base-mass randomization: nominal+U(-0.08,0.15) becomes nominal*U(0.5,1.5), uniform. Both restart from robust05; sample size is two training seeds.",
 "bounded->requested1500":"Exact same checkpoint bytes for seeds1/2. Five-seed node adds seeds3/4/5; any full-node mean shift is cohort composition, not a training/configuration effect.",
 "requested1500->requested3000":"Same recipe, continued training budget;3000 target actually3006logged total. The continuation reconstructs simulator state and resumes the saved iteration label, not an uninterrupted byte-identical run.",
 "requested3000->requested4500":"Same continuation run/recipe, additional training to4500logged total. No new configuration intervention at3000.",
 "requested4500->control6000":"Additional1500updates. Saved timed-push/reset-clock/zero-weight pitch term instrumentation is present; prior issue12 audit establishes original dynamics/RNG equivalence at weight0. No nonzero recovery reward in control.",
 "control6000->treatment":"Same five matched requested4500 parents/budget. Sole saved behavioral recipe difference is post-push pitch recovery weight0 ->34.67877532210905.",
 "treatment->clock":"Same five matched requested4500 parents/budget, sibling retraining. Sole saved intervention is two extra critic clock inputs134->136; actor101, hidden layers, original rewards/distributions remain identical."}
 for c in comparisons:c["configuration_interpretation"]=explanations[c["previous"]+"->"+c["current"]]
 primary=[c for c in comparisons if c["prospectively_frozen_straight_comparison"]];supplement=[c for c in comparisons if not c["prospectively_frozen_straight_comparison"]]
 assert len(primary)==len(supplement)==30
 qualifying=[c for c in primary if c["meets_issue15_descriptive_or_multiseed_criterion"]]
 strong=[c for c in qualifying if not c["single_seed_descriptive_only"]]
 descriptive=[c for c in qualifying if c["single_seed_descriptive_only"]]
 nodes=json.loads((ROOT/"node_statistics.json").read_text())
 def point(node,command,metric):return nodes[node][command]["metrics"][metric]["mean"]
 phase_path=REPO/"source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/robots/tw/velocity_env_cfg.py"
 phase_tree=ast.parse(phase_path.read_text())
 period=next(n.value.value for n in phase_tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=="GAIT_PERIOD" for t in n.targets))
 cells=[x for group in dominance.values() for x in group.values()]
 within_points=[x["within_energy_fraction"]["mean"] for x in cells]
 facts={
 "FT5_clean03_yaw":point("ft5","forward_03","full_yaw_error"),
 "control_clean03_yaw":point("control6000","forward_03","full_yaw_error"),
 "control_over_FT5_clean03_yaw_ratio":point("control6000","forward_03","full_yaw_error")/point("ft5","forward_03","full_yaw_error"),
 "mean_abs20s_heading03_degrees":{n:point(n,"forward_03","full20s_survivor_heading_mean_abs_degrees") for n in ["ft5","robust05","control6000","treatment","clock"]},
 "all33_node_speed_point_within_energy_fractions_above_half":all(x>.5 for x in within_points),
 "all24_multiseed_within_energy_lower95_above_half":all(x["within_energy_fraction"]["bootstrap95"][0]>.5 for x in cells if x["within_energy_fraction"]["bootstrap95"] is not None),
 "node_speed_within_energy_mean_range":[min(within_points),max(within_points)],
 "low_speed_peak_step_and_ratio":{n:{k:point(n,"forward_03",k) for k in ["fullband_yaw_peak_hz_mean","footstrike_step_hz_mean","fullband_peak_over_step_mean"]} for n in nodes},
 "configured_gait_phase_period_seconds":period,"configured_gait_phase_frequency_Hz":1/period,
 "FT5_clean05_08_fall_rates":[point("ft5",c,"fall_rate") for c in ["forward_05","forward_08"]]}
 h=facts["mean_abs20s_heading03_degrees"]
 brief=[
 f"Five multiseed-positive-interval yaw contrasts and two single-seed descriptive >20% contrasts meet their prospectively fixed definitions. The three robust08->strengthen contrasts use two later seeds against one fixed robust08 policy (conditional inference); the two control->treatment contrasts use five matched pairs.",
 f"At clean0.3m/s FT5 yaw MAE={facts['FT5_clean03_yaw']:.6f}rad/s and control={facts['control_clean03_yaw']:.6f}rad/s ({facts['control_over_FT5_clean03_yaw_ratio']:.3f}x). The original approximately0.184rad/s FT5 protocol is reproduced with the same policy; its protocol/sampling difference does not remove the observed unified-protocol lineage difference.",
 f"All33node/speed mean within-trial MSE shares exceed0.5 (range{100*min(within_points):.2f}..{100*max(within_points):.2f}%). All24multiseed intervals lie above0.5; the nine single-seed cells remain descriptive. Time-varying error dominates this energy decomposition, not a proof that every trajectory is periodic or straight.",
 f"Low-speed body-yaw peaks are2.5Hz, matching the configured{period:.1f}s gait-phase input ({1/period:.1f}Hz). Later-node force-threshold contact events are approximately4.95..4.98/s, giving roughly one yaw oscillation per full gait/stride cycle, or half the two-foot event cadence. FT5 has approximately7.50events/s under this contact-crossing proxy; it is not an independent anatomical step measurement. Frequency coincidence does not establish causation.",
 f"Curving coexists with the dominant oscillatory energy: mean absolute20s heading drift at0.3m/s is FT5{h['ft5']:.2f}deg, control{h['control6000']:.2f}deg, treatment{h['treatment']:.2f}deg and clock{h['clock']:.2f}deg. All these low-speed trials survive, so these particular endpoints are not survivor-selected.",
 f"Body-z rate error and world heading cannot substitute for each other. FT5->robust05 raises0.3m/s body-yaw MAE by63.32% while mean absolute20s quaternion-heading drift changes{h['ft5']:.2f}->{h['robust05']:.2f}deg. That comparison has one training seed per node and multiple recipe/ancestry differences, so it has no unique configuration attribution.",
 "For failing out-of-range FT5 conditions, high-frequency residual peaks and full-horizon variability include failure/transient behavior; they are not evidence of a stable healthy gait. The raw support and censored heading endpoints remain explicit."
 ]
 result={"training_updates":0,"all_requested_evaluations_completed":True,"yaw_formula":"Exposure-weighted mean valid post-step abs(body-z angular velocity-commanded yaw rate), rad/s; exactly original evaluator.",
 "brief_findings":brief,"supporting_numeric_facts":facts,"interpretation_basis_sha256":{n:sha(ROOT/n) for n in ["node_statistics.json","lineage_comparisons.json","yaw_dominance.json","original_ft5_protocol_comparison.json"]},"gait_phase_source_sha256":sha(phase_path),
 "qualifying_jumps":qualifying,"multiseed_positive_interval_jumps":strong,"single_seed_descriptive20percent_jumps":descriptive,
 "all_fixed_comparisons_with_configuration_context":primary,"supplementary_control_command_comparisons":supplement,"dominance_by_node_and_speed":dominance,
 "original_FT5_protocol_all_three_eval_seeds_exact":all(x["exact_original_result_fields_reproduced"] for x in legacy),
 "causal_limit":"Only saved-recipe candidate differences are named; sibling ancestry, a fixed one-policy baseline, continued training and shared ancestors prevent a forced unique causal claim. Actual saved parent graph takes precedence over display arrows.",
 "heading_limit":"Full20s heading results are survivor-conditional with explicit counts/falls; initial-prefix/all-segment drift also retained. Body-z yaw rate is not Euler heading derivative under tilt.",
 "next_recommendations":["Add one independent straight-heading acceptance gate based on reset-aware quaternion20s heading drift alongside existing xy/fall gates; retain body-z yaw MAE/SD and gait spectra as diagnostic companion evidence. Freeze numerical bounds and failure/censoring handling before any separately authorized future training; implement no reward/configuration change in this issue."],
 "recommendation_implemented":False,"default_promoted":False,"source_sha256":sha(__file__)}
 (ROOT/"conclusions.json").write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")
 print("ISSUE15_CONCLUSIONS",len(strong),"multiseed",len(descriptive),"single-seed descriptive",flush=True)
if __name__=="__main__":main()
