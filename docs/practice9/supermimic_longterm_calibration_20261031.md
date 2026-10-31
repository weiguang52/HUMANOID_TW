# TW SuperMimic long-term calibration: ongoing

The new goal is stable control, not just a runnable method transfer. Keep the
original seven test motions and termination conditions, require their full
requested durations and at least three independent training seeds, compare
controlled baselines, and deliver all videos. Do not promote automatically.

All artifacts below are on the data disk under
`/root/gpufree-data/datasets/practice9/supermimic_tw_20261031/longterm_diagnostics`.

## Reproducible development sequence

1. `prepare_supermimic_calibration.py`: frozen source-family split of the332
   training clips/331 families;66 families for validation. No overlap with the
   existing final holdout. Both partitions were seen by the old policy; this is
   a development split for new calibration, not fresh generalization evidence.
   Nominal inverse PD offsets across153960frames show shoulder-roll absolute
   p95 around3.69/3.60. Values outside[-1,1] are not automatically errors;
   clipping there would remove valid reference targets.
2. `collect_supermimic_calibration.py`: first four manifest entries per partition,
   selected before simulation, three controllers(nominal, hold-initial,
   reference PD targets), original terminations, no reset after failure.
   `physics_collection_v1` has24 episodes/1232 transitions; all finite and
   adjacent state/next-state continuity checked. Average durations are
   .9925/.8675/1.22s, all0/8 completed. These are diagnostic controls, not skills.
   `physics_collection_mapping_v2` additionally verifies every processed PD
   target and records action/state joint order, scale, offset and clip.
   Orders match the original checkpoint. This does not prove balance or full
   dynamic equivalence.
3. `audit_supermimic_calibration_model.py`: frozen-model H1/H8/H32 plus constant
   root-velocity baseline on each real episode, without padding/reset crossing.
   Original checkpoint baseline is `calibration_model_baseline.json`.
4. `calibrate_supermimic_world_model.py`: `wm_calibration_v1`, fixed seed42,
   1000 updates, batch128, window8, fresh Adam lr1e-4 on dynamics net only.
   Only fit trajectories are used. Normalizers and termination head are checked
   bitwise unchanged; no actor or critic is trained. Final artifact is explicitly
   offline dynamics calibration, not a resumable policy checkpoint. Only final
   update1000 is considered; no checkpoint selection.
5. `compare_supermimic_calibration.py`: predefined validation episode-median
   root-position/orientation p95 must improve>=20% at H1/H8/H32, with other
   reported RMSE medians worsening no more than5%. This is a development
   diagnostic condition, not a replacement for stable-control acceptance.

## Actual result and reporting correction

Calibration exited0 and all18 development checks passed. H32 validation median
of episode root-position p95:.29215→.16518m; orientation:2.15416→.71257rad.
Root linear/angular velocity RMSE and joint position/velocity RMSE also improve.
Absolute root error remains substantial; short and narrowly sampled episodes
cannot establish robust learned dynamics or reliable control.

`wm_calibration_v1/comparison.json` is an invalid initial comparator result:
GPU float mean of an all-true mask sometimes equals.9999999404, and comparison
with exactly1 falsely flagged nonfinite rollouts. Raw reports had no failure
and included all accuracy metrics. The comparator now recovers integer counts;
new evaluations directly record finite_count and integer-count/window fraction.
`comparison_countfix.json` is the valid result. No actual nonfinite state is
accepted and no performance gate was relaxed. Preserve both reports.

## Next work

Collect sustained nominal-pose and controlled small-perturbation trajectories
with separate seeds, plus wider motion/control coverage. Verify model quality
on new physical sampling before resuming actor optimization. A static nominal
PD calibration is not learned standing success. Existing final holdout has
already been observed; retain its reuse limitation and add a genuinely untouched
source-isolated confirmation set for final claims. Main long-term goal remains
active; existing production policies and original Isaac runtime are unchanged.


## 2026-10-31 08:02 CST: static coverage and combined calibration

Static perturbation collection produced 16 episodes / 7775 transitions. Six reached 20 seconds: four identical zero-input nominal-PD cases and two amplitude-0.02 cases. These are not learned-policy successes. All higher-amplitude cases terminated under unchanged gates.

The first motion-only calibration failed the static H8 root-position improvement gate (3.7% improvement versus the required 20%). Combined v2 therefore adds static fit trajectories, with exactly 64 motion and 64 static windows per batch. It starts from the same original model and retains 1000 updates, seed42, H8, batch128, Adam1e-4, and frozen normalizers/termination head. Validation and final holdout are not fitting data.

Combined v2 passes both 18-condition comparisons against the original model and all 18 static nonregression conditions against v1. It fails five motion nonregression conditions against v1: H1 orientation p95 +18.65%, angular velocity RMSE +5.09%, joint-position RMSE +5.25%; H8 angular velocity RMSE +5.16%, joint-position RMSE +10.09%. The prespecified 5% cap remains unchanged. Overall result: rejected for promotion.

H32 median episode root-position p95 improves from v1 0.16518 to 0.12775 m on motion validation and 0.12775 to 0.07765 m on static validation. Absolute errors remain substantial. These are offline prediction results, not closed-loop policy success. Evidence is under longterm_diagnostics/wm_combined_v2, including four *_vs_*.json files and the evaluation_protocol.json saved before evaluation.

Next: diagnose short-horizon versus rollout-loss tradeoffs and sampling coverage before another fixed-protocol experiment. Stable learned standing, stepping, all seven fixed motion durations, and at least three independent training seeds remain outstanding. Long-term goal remains active.


### Fixed teacher-forcing experiment v3

The combined collection has 522 motion and 3249 static fit windows, sampled exactly 64/64 per batch. V2 optimizes an H8 free rollout, which does not directly constrain one-step predictions from the real states at steps 2–8. This motivates a hypothesis test, not a proven explanation of its regressions.

V3 changes only the loss to 0.5 times the original H8 rollout loss plus 0.5 times the mean real-state one-step loss over the same eight transitions. All original initialization, seed, update count, optimizer, frozen parameters and group sampling remain fixed. Contiguity of every trace is asserted before using previous next_state as the teacher input. The runner records its protocol before fitting and evaluates only the final 1000-update artifact. It requires the original baseline gates and <=5% nonregression versus both v1 and v2 separately on motion and static validation. Run: scripts/practice9/run_supermimic_teacher_calibration.sh. Output: longterm_diagnostics/wm_combined_tf_v3. This produces an offline dynamics artifact, never a deployable policy checkpoint.


### 2026-10-31 08:08 CST: v3 rejected; real-physics standing baseline

V3 completed training and six comparisons with process exit0, but failed acceptance. Both original-model comparisons pass, static versus v1 passes, and the other three comparisons fail. Motion H32 position episode-p95 median is 0.17265m (+35.14% versus v2); static H32 position is +53.08% versus v2. Thus the equal-weight one-step loss does not resolve the short/long horizon tradeoff. All artifacts remain preserved; no policy loads this rejected calibration.

Started `tw_standing_v1` with `run_tw_supermimic_standing_v1.sh`: fresh seed42,256 environments,1000 iterations, unchanged online SuperMimic/physics/randomization/reward parameters. The single changed input is the synthetic1001-frame nominal-standing motion instead of332 motion clips. This tests whether the learner can acquire the simplest real-physics control task. It does not use any prior actor or offline calibration artifact. The runner automatically evaluates final checkpoint for20s against zero-offset nominal PD, recording front/side video and original termination reasons. Actual finite states and speed caps must be checked, plus drift/error reported. Success here is a synthetic training-task diagnostic only; all seven original independent motions and three independent training seeds remain required for the long-term goal. State/protocol/logs live in `supermimic_tw_20261031/standing_v1_seed42` on the data disk.

## Standing and contact follow-up, 2026-10-31

Standing v1 completed1000 iterations. Final learned policy terminates at16 steps/0.32s (ee_body_pos), nominal zero-offset PD completes1000 steps/20s. All four H264/50fps videos have exact step counts and pass full ffmpeg decode. Actual joint speeds obey caps in both trajectories; imagined H1/H8 speeds violate caps about28% of entries. Policy max absolute action2.731; root XY displacement46.7mm at0.32s versus nominalPD2.06mm at20s (different durations, not comparable20s drift). See standing_v1_seed42/evaluation/results.json, video_integrity.json and action_speed_audit.json.

Contact probe v2 records actual Warp solver contact data through get_data_into/mj_contactForce. V1 omitted robot/ prefixes and its empty contact arrays are invalid, retained for provenance. V2 has16 fixed perturbation episodes,780 samples, finite/contiguous traces,6 full20s episodes. Four zero-input duplicates are not independent learned successes. Nominal seed42 samples2-19.8s: four loaded points per foot,49x22.85mm footprint, total normal force8.248N matching robot weight. Perturbations lose support area while soles tilt25-35degrees. This does not establish a collision-model defect. URDF box extents match original STL bounds. Probe code is independent of policy training.

Source comparison must use corresponding HumanML3D ankles7/8, not toes10/11 against robot foot origins. Across332 clips, normalized-duration height correlations have medians0.987/0.992 before contact correction and0.935/0.933 afterward, none below0.5. This supports retained lift timing but is not clearance/dynamic-feasibility acceptance. Reports under longterm_diagnostics include all_source_ankle_stage_screen.json and near_ground_sole_screen.json; no training/test data were modified.

Optional imagined_velocity_penalty defaults0 and subtracts mean squared relative joint-speed excess from imagined reward only. It leaves predictions and real physics untouched. Math/gradient tests pass;8env/4iteration smoke completes with coefficient1 and191 finite checkpoint tensors. Fresh standing_velocity_v2_seed42 uses256env,1000iterations,seed42 and all v1 settings, with this coefficient as the sole experimental change. Final1000 only, automatic same20s trained/zero videos, unchanged termination. Run script and data-disk protocol are preserved. No claim of improved control until actual evaluation. Original fixed7 motions and3 independent training seeds remain outstanding.

## Initial-pose and mass audit follow-up

No hardware mass/CAD reference is available per user. Total model mass0.8407428kg, head/neck16.2%, near-symmetric left/right groups. All principal inertias positive and satisfy triangle inequalities; all33 checked COMs lie inside visual AABBs, and necessary AABB covariance bounds pass. Vertex merging closes7/33 STL meshes;26 remain unclosed, so uniform-solid mesh volume/inertia mismatch is not reliable evidence of wrong URDF inertia. Keep original mass parameters. Data-disk reports: mass_distribution_audit.json, mass_geometry_screen.json, processed_mesh_closure.json, inertia_aabb_bound_check.json.

Nominal shoulder roll +/-90deg is a T-pose; roll0 yields arms down but raw collision geometry overlaps wrist/thigh. +/-10deg clears that overlap and lowers COM about12mm. Fixed16 open-loop perturbation tests: original6 full20s, arm10 straight7, arm10+2mm lower hip5. Lower hip alone reduces steady full-sole geometric support margin8.88mm to6.71mm. These are diagnostics, not learned-policy or independent training-seed successes.

Bounded IK candidate keeps feet fixed, arms10deg, rootz0.262m, rootx0.0139258m, targetCOMx0.010m, knee flexion13-14deg and leg yaw<4deg. All16 fixed tests complete;10 full20s. Four strongest perturbations still fail1.92-2.48s; moderate seed42/44 fail4.40/5.96s. Each common observed window versus each of the three earlier poses shows lower maximum root tilt and XY displacement16/16, no extrapolation of failures to20s. Evidence pose_forward2_v1/summary.json and common_window_comparisons.json. Not automatically adopted. Actual target mapping assertion passes in forward2_mapping_check. Full front/side recorded-state replays are running; nominal seed42 final20s frames were inspected and show upright small-arm-abduction stance. Other frames/contact and independent forcing remain unverified.

Pose-reference generator now accepts arm angle and validated IK JSON, preserving defaults. Pose collector converts actual initial targets using action ids/offset/scale, records manifest hash and offers mapping-only check. Renderer replays every recorded next_state at50fps, including failures, with no extension and explicit open-loop-PD label. Changes do not alter the ongoing standing_velocity_v2_seed42 experiment.


## 2026-10-31: independent pose confirmation and standing velocity result

The frozen forward2 pose improved full 20s open-loop PD forcing episodes from 7/16 to 9/16 on new forcing seeds 46--49, but seed47 amplitude0.10 regressed: lifetime 2.26s to 1.60s, common-window max tilt 6.44 to 54.34 degrees and XY excursion 24.59 to 217.78mm. It is not adopted. These are forcing seeds, not independent learned-policy seeds. All 32 development replay videos decoded and matched trace frame counts; terminal frames were reviewed. Full temporal visual review and loaded-contact verification of this candidate remain incomplete.

The fresh standing_velocity_v2_seed42 completed 1000 iterations with imagined relative joint-speed violation penalty coefficient1; the final checkpoint has 191 finite tensors. Learned control failed at 0.40s with original ee_body_pos termination versus standing_v1 0.32s. Nominal zero-offset PD completed 20s in both. On the common first0.32s, action absolute p95 fell 1.82052 to0.89523 and root maximum tilt fell53.74 to33.08 degrees, but XY maximum grew46.69 to48.92mm. This is not a successful stability fix. Full-episode world-model statistics use different windows and cannot establish paired model improvement. All four H264/50fps videos match20/1000 frames and fully decode. First/last front/side frames confirm learned tilt and nominal upright stance; no ground-impact claim or temporal visual completeness claim.

Evidence: data state standing_velocity_v2_seed42/evaluation/{results.json,paired_standing_audit.json,integrity.json,visual_review.json}. Independent forcing evidence: longterm_diagnostics/pose_confirmation_comparison.json. Neither policy nor pose is promoted. Next: diagnose reward/model gradients at nominal standing against physically recorded transitions before another policy-training change; retain all checkpoints and original gates. Final7 complete-duration and three-training-seed acceptance remain unmet.


## 2026-10-31: local action-response evidence

The final standing_velocity_v2 world model was evaluated at the identical nominal reset state with independent per-joint plus/minus normalized residuals0.02 and0.05, one real step each (120 responses per diagnostic run). Both runs retained all finite responses, identical initial states and no terminations. Joint-velocity derivative cosines were0.9925/0.9876 and joint-position0.9880/0.9879, but original tracking-reward derivative cosines were only0.1857/0.2130. Model reward autograd versus model finite differences agreed0.9993/0.9983, so this does not support a broken autograd implementation.

A second diagnostic measured the actual zero-action response and added its detached prediction residual to imagined next states only for reward-gradient analysis. Reward derivative cosines then became0.9131/0.9240. This uses an oracle real response at the same state and is NOT a deployable correction or model calibration result. It implicates local prediction bias, but does not establish long-horizon or policy-training causality. Original tracking reward was analyzed here; velocity-feasibility penalty, termination survival and critic-bootstrap gradients remain separate diagnostic work. No policy/model weights were changed. Evidence: longterm_diagnostics/action_response_v1 and action_response_v2/{plan.json,paired_responses.npz,results.json,integrity.json}. Next: test this mechanism at separately fixed physically recorded standing states, with fit/validation separation, before choosing a model-calibration change.

## 2026-10-31: fixed later-standing response protocol

Fixed warmup lengths10/50/100 zero-action controls (0.2/1/2s) before each central-difference branch, replayed from reset rather than restoring only qpos/qvel. Each state has all120 plus/minus responses; this is one synthetic trajectory, not independent training seeds. The first10-step attempt stopped on its strict1e-7 replay-state check at a1.0547e-7 joint-speed difference. The failed folder/log remain. Revised diagnostics record five repeat replays and every branch's maximum state discrepancy, with a fixed1e-5 absolute numerical replay bound. This only concerns diagnostic replay reproducibility, not any stability or model acceptance gate. Raw measured discrepancies must be reviewed relative to action-response magnitudes before interpreting gradients. Revised outputs use separate action_response_replay_v2_warmup{10,50,100} folders. No model or policy update is made.

## 2026-10-31: standing-specific dynamics calibration v1

Fixed final standing_velocity_v2 checkpoint, static_perturb_v1 fit seeds42/44 only,1000 updates seed42 Adam1e-4 batch128 H8 pure rollout loss. All18 existing validation diagnostic gates passed without threshold changes. H32 root-position p95 episode median fell0.332124 to0.070262m; root-orientation ratio0.11171. All non-dynamics model tensors remained bitwise equal and all model tensors finite. Eight fit trace filenames are disjoint from validation filenames. Artifact SHA2565883d551ec0647401b0f864b40fbb51b9a291cd2bcfd43c966cbabca972a9757, kind offline_dynamics_calibration_not_policy_checkpoint. No actor/critic or policy promotion.

Important limitation: zero-amplitude synthetic trajectories repeat across forcing seeds, so filename separation is not source-disjoint motion or independent training-seed validation. The collection has already been inspected. This passes a development model diagnostic only, not stable standing or final skill acceptance. The audit utility's legacy scope text referring to an old motion policy is generic historical text; current checkpoint and actual supplied split are recorded in plans and hashes, and do not establish fresh generalization. Next evaluate local reward gradients with this final frozen calibrated model and test against actual recorded trajectories before deciding on a policy update. Evidence directory longterm_diagnostics/standing_wm_calibration_v1, including before/after.json,gate.json,fit andintegrity.json.

## 2026-10-31: calibrated reward response check (partial,2s pending)

Reused every recorded physical branch at0/0.2/1s, reconstructed deterministic context and checked initial state tolerance1e-5 plus all120 real rewards against stored values at1e-7. Original trace did not store context: this limitation is explicit. No oracle residual correction. Final calibrated reward-gradient cosines at eps0.02/0.05: initial0.8412/0.8645 versus0.1857/0.2130;0.2s0.6627/0.4895 versus-0.1617/-0.1640;1s0.7066/-0.6275 versus-0.1143/-0.0119. Gradient RMSE decreased at all six conditions, but1s eps0.05 direction remains a counterexample. No claim of uniformly trustworthy optimization gradients, no policy update. Two-second original diagnostic remains live and must be included when ready. Evaluate actual finite-difference sensitivity across amplitudes rather than ignoring small-gradient or contact-transition effects. Evidence: calibrated_response_warmup{0,10,50} and standing_wm_calibration_v1/reward_response_comparison_partial.json.

## 2026-10-31: prespecified smaller-scale response diagnostic

Actual reward central differences across0.02/0.05 had cosine0.99588 at reset,0.93033 at0.2s, but-0.57802 at1s. Fixed follow-up at1s uses all30 action dimensions plus/minus0.002/0.005/0.01 (180 responses) and five repeated zero-action rewards to quantify numerical variation, keeping every old result. The new output is action_response_smallscale_warmup50. This is an investigation of finite-scale nonlinearity versus noise, not a relaxed acceptance gate. The original2s task is preserved and still live. Scorer accepts every prescribed amplitude rather than selecting a passing one. Pending results must not be interpreted as successful control.

## 2026-10-31: response provenance and action-support limitations

Future response collection now stores actual context34 and normalized action30 for every physical branch, and preserves partial responses on interruption. Scoring prefers these recorded arrays, checks context against deterministic replay and exact action encoding, and retains an explicit reconstruction limitation for legacy traces. A new independent60-response smoke collection and calibrated scoring completed: all arrays finite, context/action shapes60x34/60x30, actual reward reconstruction within1e-7, recorded context/action paths exercised. Partial/interruption preservation is implemented but not exercised by this successful smoke; no healthy job was interrupted. Evidence: action_response_provenance_smoke_v1 and calibrated_response_provenance_smoke_v1.

The fixed control1100 policy still terminated at0.40s; zero PD completed20s. Four H26450fps videos fully decoded with20/1000 frames; front/side start/end frames reviewed (not full temporal visual QA). Shared0.4s versus its1000 parent had XY max77.60 to38.79mm and tilt53.32 to51.84deg, without longer survival. Calibrated continuation remains pending; no policy promotion.

Fit support was3305 transitions with action absolute maximum0.1, but control1100 reached2.610 and81.5% of action components exceeded their empirical fit-dimension range; every one of20 steps exceeded at least one dimension. This is empirical range coverage, not a causal proof. On that identical failed20-step actual trajectory, original1000 versus offline-calibrated dynamics root-position p95: H1 15.98 to43.31mm (regression), H8 66.50 to56.50mm (improvement); no H32 windows. Static-only calibration cannot claim full-policy coverage. Evidence in standing_calibration_pair_v1/control: action_support_audit.json, model_support_audit.json, continuation_audit.json and video/visual integrity files. These counterexamples must remain in subsequent model validation.

## 2026-10-31: fixed paired continuation final result (failed standing gate)

Both branches completed exactly100 updates1000 to1100, retained actor/critic optimizers and RNG from the identical source, reset WM optimizer state in both, and used the original final learning rate6e-6. All191 checkpoint tensors were finite in each. Control learned20 steps0.40s; calibrated learned41 steps0.82s; both original ee_body_pos termination. Zero-offset PD in both completed1000 steps20s. Calibrated is not promoted. This is one seed continuation, not three independent training seeds or final7-motion acceptance.

On the common0.40s, calibrated versus control root XY max31.88 versus38.79mm and max tilt15.08 versus51.84deg. The longer0.82s endpoint is not a20s drift measure. All eight H26450fps front/side videos fully decoded and frame counts match20/41/1000 actual steps. Start/end images confirm strong learned tilt and upright nominal PD; no ground-impact or full temporal visual-review claim. Combined index: supermimic_tw_20261031/longterm_diagnostics/standing_calibration_pair_v1/index.html on the18768 forwarding service, HEAD200 verified. Synthetic standing, no original human motion source clip.

Calibrated policy actions still reached2.984;82.85% of components exceeded static fit empirical per-dimension ranges (fit abs max0.1). Same16 static development trajectories were audited with final models; all18 validation metric summaries matched trajectories and had finite windows. H32 root-position p95 episode medians: source1000 0.332124m, offline calibrated0.070262m, control1100 0.670827m, calibrated1100 0.270575m. Thus online continuation lost much of the static prediction improvement; this does not identify whether weights, normalizers, data support or their interaction caused it. Full18-metric report is model_retention_audit.json. Preserve both this regression and the failed-policy H1 regression in future validation.

Next planned single-variable test: retain fixed good-standing fit windows during online WM fitting, with half the same total dynamics batch from the existing fit-only static collection and half online, while leaving termination sampling, normalizer behavior, actor/critic settings, physics and the fixed100-update/final-only protocol unchanged. Validate loading, no reset-crossing and source split before pilot; do not use validation records in updates. No blanket model/control success claim; compare against the already completed calibrated continuation and retain all counterexamples. Implementation and pilot remain pending.

## 2026-10-31: fit-window retention implementation and verification

New optional wm_anchor_collection is disabled by default. When enabled at fraction0.5, dynamics batch512 comprises256 online and256 fixed fit windows; termination-head sampling remains512 and normalizer/policy/critic mechanisms stay unchanged. Fit loader verifies joint/action order, broadcast offset/scale equality, finite dimensions, exact next-state continuity, constant context, within-collection paths and disjoint split filenames. Uses k+1 stored live states, no padding/terminal-next-state insertion. Actual static fit input is8 files/3241 windows. Fit file hashes, mapping hash and independent sampler state are saved; resume checks source hashes and restores sampler. Inference disables training-only anchor loading.

Five data-integrity tests passed, including reset crossing/split overlap/order/nonfinite rejection and exact sampler replay. First8-env4-update smoke stopped at a strict offset shape check (stored1x30 versus replicated8x30), before fitting; original failed log/run retained. Corrected verification broadcasts the stored per-joint mapping across envs without loosening value tolerance. Second fresh smoke completed4 updates,192 finite checkpoint tensors; actual CUDA trainer reload checked16fit+16online,8files/3241windows, source hashes and sampler-state restoration/replay. SHA25619bc765a276f482537f43d287a25763819f7ca8b4e5a1d63bfdeb70a4526c6b4. Smoke is not a control result and is never used as formal initialization. Evidence: anchor_retention_smoke_check.json and anchor_retention_smoke_v2.log; all temporary/test data remained on the data disk.

Formal retained pilot is fixed100 updates from the same1000 source plus the same offline-calibrated dynamics as the previous calibrated arm. Only dynamics-window source mixture changes. Final1100 checkpoint only, same256env seed42,6e-6 terminal schedule, physics/randomization/reward/velocity penalty and WM optimizer reset. Run standing_retention_v1. Report all original18 validation metrics against prior calibrated1100 and initial offline calibration; existing comparison convention20% p95 improvement/5% other-RMSE limit, and separate retention5% nonregression report. Neither diagnostic report replaces actual20s standing or final7-motion/3-independent-seed requirements. No automatic promotion.

## 2026-10-31: retained-window pilot and head coverage diagnostic

Retained pilot finished fixed1100 with192 finite checkpoint tensors and lr6e-6. Model-improvement18-metric gate versus prior calibrated1100 passed;5% retention gate versus initial offline calibration failed9 metrics. H32 root-position p95 median0.083655m versus prior0.270575 and initial0.070262. Preliminary learned actual evaluation37 steps0.74s terminated ee_body_pos, below prior0.82s; no policy improvement or promotion claim. Full zero/video review remained pending at this entry.

Reusable audit_supermimic_action_support.py now records fit/actual trace hashes, policy/model ordering and physics contracts, matmul precision, empirical action-range coverage and all availableH1/H8/H32 actual-trajectory errors. It reproduced prior control0.815 outside-range fraction and root p95H1/H8 to1e-7; runtime highest/TF32false recorded, matching prior frozen-model diagnostics. Short episodes have no inventedH32 windows.

Head coverage evidence:14 directly tracked bodies omit head/neck, while3 head joints are actuated. Static target is0 but previous calibrated-policy actual angles reached neck-0.121rad,roll0.162rad,pitch-0.235rad. They retain indirect root/body/action-rate/joint-limit constraints; this is not evidence of no constraints or sole failure causality. Existing Isaac NeckOnly configuration had direct neck position/velocity costs, and HeadLocked inherited it with physical locked-head asset; current SuperMimic reward lacks that direct group. Across all332 existing full-motion references153960frames, head targets are largely nonzero: yaw[-0.798,0.730],roll+/-0.52,pitch[-0.790,-0.158]rad. Distribution alone does not prove retargeting error, and permanent full-motion head zeroing is not a valid full-scope fix. Evidence: head_tracking_gap_audit.json and head_reference_coverage_audit.json.

Prespecified diagnostic only: freeze the latest retained1100 policy, repeat trained baseline versus trained_head_zero on the same synthetic static reference, only setting the3 normalized head position commands to0 through actual action-name mapping. Not a weld or physical joint lock. Runtime guards require all reference joint velocities zero, head reference positions zero and actual PD offsets zero. Save actual masked actions in histories/transitions, retain original physics/20s/first failure and front/side videos. Default evaluation behavior stays unchanged; no policy training or promotion. Evaluate all results, including a null/negative outcome.

## 2026-10-31: retained pilot and nominal-head diagnostic fully evaluated

Retained learned37 steps0.74s (original ee_body_pos), zero PD1000 steps20s. Model improvement gate18/18 passed versus prior online calibrated endpoint; retention gate failed9/18 versus initial offline model. Retained actual action abs max2.8277 and82.8829% of components outside fit empirical ranges. All actual episodes were audited with checkpoint/trace hashes and matching policy/model coordinate/action contracts. No policy promotion or control improvement claim versus prior0.82s.

Frozen latest retained1100 diagnostic repeated original37 steps0.74s exactly; nominal head commands produced40 steps0.80s, original ee_body_pos termination. Actual3 masked head commands were verified exactly0. Common0.74s root XY max60.04 to116.80mm, max tilt52.50 to26.54degrees. This is a tradeoff/negative repair result, not evidence that head zeroing fixes balance or should be adopted. Full-motion head targets remain nonzero and unchanged. Eight new front/side H26450fps videos fully decoded with actual37/40/1000 frames; start/end visual frames reviewed, not full temporal QA or proof of ground impact. Combined index: supermimic_tw_20261031/longterm_diagnostics/standing_retention_v1/index.html. Stage statuses explicitly failed standing gate; long-term goal remains incomplete.

Next planned hypothesis: current failed actor starts with outputs far beyond the calibrated neighborhood, whereas nominal PD already completes20s. Test zero initialization of only the final actor decoder readout (weight/bias plus only their Adam state reset), preserving encoder/earlier decoder/critic, same source1000+offline dynamics, retained fit mixture, physics/reward and fixed100-update protocol. All30 action channels stay controllable and no inference clipping/head mask is added. First verify exact nominal output and unchanged other weights, then a separate short smoke before formal pilot. This is not permission to call zero-action PD learned feedback control: final acceptance still requires actual learned behavior, disturbances, full7-motion durations and3 independent training seeds. Implementation/pilot pending at this entry.

## 2026-10-31: isolated actor-readout initialization verified

Opt-in zero_readout initialization zeros only final decoder linear weight/bias and removes only their two Adam state entries. All other actor/critic tensors are checked bitwise unchanged before training, and real-checkpoint zero/one-input probes produce exact nominal normalized commands. No inference mask, clipping, frozen head or loss change; all30 action channels retain gradients. Two targeted tests verified unchanged earlier weights/moments, zero output and nonzero readout gradient, plus rejection of non-linear output modules.

Separate8-env4-update smoke continued the same1000 source with identical offline-calibrated dynamics and fit mixture, ending1004. All192 checkpoint tensors finite; readout parameter norm became0.00173409 from0, confirming actual optimization. Initial checks recorded decoder.8 prefix, two removed optimizer entries and unchanged other weights. SHA256b59d180c511a42318b287b8b79e499815cdcb35ee66c14bb5e2df860136e51ce. Smoke is never used as formal initialization. Evidence: zero_readout_smoke_v1/initialization_check.json,zero_readout_smoke_check.json andzero_readout_smoke.log.

Formal pilot standing_retention_zero_readout_v1 uses256env, fixed100 updates1000 to1100, same source1000/offline-calibrated WM/retained batch512=256+256, WM optimizer reset and original6e-6 terminal schedule as retained control. Only readout initialization and its own optimizer state differ. Model diagnostic nonregression reports against prior retained endpoint and initial offline model are separate from original20s/first-termination real control acceptance. Final1100 only, trained/zero front/side videos, no automatic promotion, no final7-motion or3-training-seed claim.
