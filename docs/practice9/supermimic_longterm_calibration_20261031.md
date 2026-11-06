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

## 2026-10-31: fixed PD reset/exploration factorial diagnostic

Predetermined32-env2x2 experiment, seed42, same static frame0 and same independent Gaussian sequence seed10042. Action noise std0.1 read from actual training config. All four groups disable COM/friction/encoder events to isolate reset-state perturbations and action noise; this is not the full training return-time/event distribution and not a learned-policy or independent-training-seed test.

Results (completed20s/32,median actual seconds): nominal_zero32/32,20.0; nominal_noise0/32,0.90 (0.68--1.42); perturbed_zero0/32,0.49 (0.28--1.40); perturbed_noise0/32,0.46 (0.26--0.88). Inherited reset ranges remain unchanged in training. This establishes strong effects on this fixed PD baseline, not impossibility of recovery by a learned policy or proof of the sole training-failure cause.

First collector v1 completed nominal_zero then stopped at backend's mandatory reset guard after a noise-case termination; partial data/log retained. Corrected v2 resets and nominally parks only already stopped worlds to satisfy the batched API; never records another episode. Live physical state/context and command/episode clocks are checked unchanged; largest observed live-state cache discrepancy5.96e-8. All128 first episodes have correct lengths, finite arrays, exact serialized continuity, matching initial-state pairs and actual noise equal to the fixed sequence. Evidence reset_noise_probe_v2/{plan.json,results.json,integrity.json,initial arrays,128traces}; v1 preserved separately.

Zero-readout formal evaluation fully completed: learned84steps1.68s original ee_body_pos; zero1000steps20s. Four H26450fps videos fully decoded, exact84/1000 frames. Six evenly spaced trained front/side video-time samples plus zero start/end reviewed: early near-upright stance, pronounced late lean; no full frame-by-frame or ground-impact claim. Common0.74s versus retained control: root XY max60.04 to14.40mm, tilt52.51 to2.22deg; still no20s skill pass. Actual action abs max0.15889 and28.5317% of components outside empirical fit ranges (prior2.8277/82.8829%). Both18-metric model nonregression reports passed, but absolute H32 validation episode root p95 ranges9.88--123.59mm; relative gate success does not prove universal predictive accuracy. Combined index standing_retention_zero_readout_v1/index.html. No policy promotion; final7-motion and3-independent-training-seed acceptance remain unmet.

Source inspection also confirms fit anchors currently feed only dynamics updates; imagination starts use only online replay. Next investigate healthy-standing coverage of policy-start sampling and a prespecified data-mixture test before lowering reset/noise ranges. No new curriculum or policy-start mixture implemented in this entry, and no validation threshold changes.

---
## 2026-11-05: current continuation state (TW-RL / Notion004)

The long-term goal is active and incomplete. No learned policy is accepted or
promoted. Earlier runnable PPO/standing pilots are failed evidence, not a usable
walking baseline. Do not resume PPO with a rejected world model. Final acceptance
still requires all fixed seven independent motions for their original durations,
without the original termination trigger, at least three independent training
seeds, matched baselines, front/side videos and original motion provenance.
Keep the original physics, samples, horizons and numerical gates.

All paths below are relative to the remote data directory
`/root/gpufree-data/datasets/practice9/supermimic_tw_20261031/longterm_diagnostics`.
All work and caches stay on the data disk; never copy mounted data/media locally.
The repository is `/root/gpufree-data/projects/HUMANOID_TW`, SSH port 30416,
branch `supermimic/tw-adaptation-20261104`. Credentials do not belong in this file.
Notion004: https://app.notion.com/p/403325d8231d8172995bf19510b8d760

### Latest rejected full experiment

`two_batch_full18_pair_v1`: both fixed 1000-update arms completed; matched control
weights/optimizer, frozen fields, paired RNG and finite same-batch constraints
passed integrity checks. Independent evaluation rejected the candidate: 6/18
static, 18/18 dynamic, 8/18 original-reference static and 11/18 original-reference
dynamic checks failed; held branch gates also failed. Coordinate contracts passed.
The expanded fit diagnostic failed 18/18 comparisons, but is not a generalization
gate. See `fit_integrity.json`, `paired_metric_review.json`,
`root_state_response/audit.json`, `root_pose_response/audit.json`. Retain both
final checkpoints and all failed experiments; do not choose an intermediate.

The frozen 16-batch comparison is saved in
`two_batch_full18_initial_gradient_diagnostic_v3/initial_final_comparison.json`.
Full-objective means: constructed initial model 2.29016814, final control
0.07282845, final candidate 0.27429338. Static means: 0.000301907,
0.000432276, 0.000206100 respectively. The candidate improved over initialization
but learned dynamics substantially less well than the control. Do not claim it
is globally worse than initialization. All three audits verified unchanged source
weights and zero source optimizer steps. Same trainer/runner hashes and fixed
batch sequence were used; historical reports do not include first-batch RNG
hashes. Failed initial-load attempts v1/v2 are preserved. The actual initial model
must be constructed by the original runner (153 inputs), not loaded directly
from its unadapted 137-input parent.

### Current experiment, not yet accepted

`moment_reset_static_guard_pair_v1`: both arms retain the original complete
objective and original static finite-step guard. The sole candidate variable is
clearing Adam `exp_avg` immediately before each real step; second moments, step
counts, learning rate, clipping and samplers remain unchanged. This follows the
frozen final-control diagnostic: saved-momentum full steps increased static loss
16/16 times, while zero-first-moment steps offered simultaneous finite static/full
descent on 16/16 batches. This is a hypothesis, not physical validation.

The separate 4-update smoke passed `smoke4/fit_integrity.json`: control weights
and optimizer are bit-exact to the original guarded trainer, 22 frozen tensors
match, eight Adam step counts are four, native/static indices match and the first
step guard is identical. Candidate resets are zero on step one, eight thereafter.
`reviewer_compatibility.json` confirms the extended reviewer reproduces every
historical full18 decision (excluding timestamp and reviewer hash).

Live formal tmux session: `moment_reset_pair`, fixed 1000 updates per arm, followed
by integrity and unchanged static/dynamic/expanded H1/H8/H32, coordinate, branch
and original-reference checks. Inspect the actual process plus `control_fit.log`,
`candidate_fit.log` and `status`; do not infer failure from a quiet log or stale
status. `moment_reset_supplementary` waits for that session and passed fit integrity
before root velocity/pose response audits of parent, original reference and both
final models. Never restart a normal run. Primary entry point:
`scripts/practice9/run_supermimic_moment_reset_pair.sh`.

Next: verify final integrity and all independent reports; inspect fit/validation
errors and response behavior before selecting another single-variable change.
PPO remains gated on physical prediction consistency. Code is committed remotely;
new-branch push is still unverified because HTTPS authentication failed. A local
commit does not establish a successful push. Do not record tokens in this file.


---
### 2026-11-06 02:18 Asia/Shanghai — dynamic action response is improved, full calibration still rejected

Branch: supermimic/tw-adaptation-20261104. `motion_action_response_pair_v1` completed both fixed 1000 updates. Integrity passed: original control weights/Adam bitexact, 22 frozen tensors, eight Adam states, paired indices, sole coefficient 0/1 difference, 2128 endpoints/update. Raw candidate SHA256 f418e0616182a05cc248c1577eb99418f3f8da9fdd713a548274716708d92737; declared candidate 314b5e00278e706b3d58157e4a3d2a4d888ab7c3e06c849d870755932e6ba28f.

Full review rejected: static 4/18 and dynamic 6/18 regressions relative to paired control; expanded fit diagnostic 9/18; original-reference static 6/18 and dynamic 1/18. Coordinates pass, held response branches fail. Dynamic H32 root-position p95 trajectory median 129.501 -> 147.478 mm. Static H32 candidate 29.498 mm alone does not establish acceptance. No PPO or promotion.

Dynamic response validation normalized RMSE (control -> candidate): root linear .239084 -> .212535; root angular .692180 -> .475110; joint velocity .617161 -> .167450. Frozen offset decomposition at the same archived perturbed endpoints finds joint-velocity centered error .0215444 -> .00574442 but symmetric-mean error .126453 -> .158635. Mean per-condition offset MSE fraction .915258 -> .996883. Each condition has an exact MSE decomposition assertion; symmetric means approximate local action centers and are not exact zero-perturbation evaluations. These descriptive results support investigating absolute next-state offset alongside action derivatives, not claiming a proven causal explanation of all rollout errors. Fit-only decisions and unchanged validation gates remain required.

Evidence under `/root/gpufree-data/datasets/practice9/supermimic_tw_20261031/longterm_diagnostics/motion_action_response_pair_v1`: fit_integrity.json, paired_metric_review.json, response_summary.json, dynamic_action_offset/audit.json. New frozen diagnostic script `scripts/practice9/audit_supermimic_motion_action_offset.py` audits all 192 conditions per model and asserts source model tensors unchanged. All failed finals retained. Next: verify absolute predictions at archived unperturbed dynamic action centers, then design a single-variable fit-only correction if justified. Long-term fixed7/full-duration/three-training-seed requirements are still unmet.

Push remains blocked by HTTPS authentication; no claim that these latest commits reached origin.

---
### 2026-11-06 03:46 Asia/Shanghai: center supervision rejected
motion_action_center_pair_v1: both fixed1000 arms completed. Integrity passed: control weights/Adam bitexact prior candidate,22 frozen tensors,8 Adam states,2144 endpoints/update,center coefficient0/1 only.
Review failures: static9/18,dynamic7/18,expanded-fit7/18,original-static12/18,original-dynamic1/18. Coordinates pass; both branch comparisons fail. No PPO or promotion.
Dynamic H32 root147.478->121.436mm improves,static29.498->36.140mm and dynamic orientation.554866->.727941rad worsen.
Exact center validation normalized RMSE control->candidate: linear.144597->.139797,angular.243330->.256189,jointvelocity.158642->.145998. Fit angular.122794->.099141,jointvelocity.085229->.066307 improve.
Action derivative validation jointvelocity.167450->.166837,angular.475110->.474648 approximately maintained. All frozen audit model tensors unchanged.
Evidence: motion_action_center_pair_v1/{fit_integrity.json,paired_metric_review.json,dynamic_action_center/audit.json,dynamic_action_response/audit.json,dynamic_action_offset/audit.json}.
Next inspect fit coverage and joint loss effects before another full run. Fixed7/full durations/3training-seed acceptance remains unmet. Latest branch commits still not pushed: unresolved HTTPS authentication.

---
### 2026-11-06 05:18 Asia/Shanghai: dense fit coverage still rejected
dense_motion_center_pair_v1 completed1000+1000; integrity passed with2144 endpoints/update,22frozen,8Adam,original paired indices and center replay verified. Only sparse48 versus dense895 fit sampling differs.
Raw candidate f2882fc642011d6b179e2c1a1fe78a39749fa1b664cb5ce7e6f8593b94aeb1d5; declared candidate4df80e96fdd89b2f69c4ca46d37a5eeedd2804ccd9945a4d0b3736b54982a872.
Review rejected: paired static0/18 failures,dynamic6/18,expanded-fit1/18; original static7/18,dynamic1/18; coordinate checks pass,both held branch checks fail. Dynamic H32 root121.436->158.425mm. No PPO or promotion.
Full1740 frozen steps: other847 fit rootangular normalized error.112362->.083699,jointvelocity.070450->.050628 improve; rootlinear.050666->.055671 worsens. Other797 validation angular.175637->.171424,jointvelocity.110199->.104547 improve; rootlinear.081479->.095755 worsens.
Exact48 validation centers all three worsen: linear.139797->.150405,angular.256189->.277202,jointvelocity.145998->.151270. Action derivative validation jointvelocity.166837->.162985 slightly improves. Frozen weights unchanged.
Evidence paired_metric_review.json,dynamic_action_response/audit.json,dynamic_action_center/audit.json,dynamic_action_offset/audit.json,dense_center_coverage/audit.json. Center supervised legacy flag denotes sampled centers only,not validation fitting.
Next inspect omitted root-linear component of added center loss against fit-only evidence. Do not claim coverage expansion alone solves calibration. Full fixed7/duration/3seed requirements remain unmet; branch still unpushed due HTTPS authentication.

---
2026-11-06 05:25 Asia/Shanghai — root-linear dense-center paired experiment
Goal: test the omitted root-linear velocity target without changing dense fit coverage, original objective, sample budget, or acceptance gates. Both arms use 895 fit states and 2144 prediction points/update; only the added normalized root-linear center MSE coefficient switches 0/1. No validation labels enter fitting.
The fixed 4-update smoke passed: control weights and Adam state bitexact against the previous dense candidate, 22 frozen tensors and 8 Adam states checked, all paired indices identical and center seed replay verified. This establishes reproducibility, not policy quality.
Next: fixed 1000 updates/arm, unchanged H1/H8/H32, coordinate and held-branch audits; retain rejected finals. Fixed7 full-duration and >=3 independent policy seeds remain unfulfilled. No policy promotion.
State: longterm_diagnostics/linear_dense_center_pair_v1. Push remains unverified because remote HTTPS credentials are unavailable; local remote-server commits retained.

---
2026-11-06 06:44 Asia/Shanghai — root-linear dense-center pair rejected
Both fixed1000 arms completed with paired integrity passed (22 frozen tensors, 8 Adam states, 2144 points/update, sampled indices replayed). Control weights+Adam are bitexact against prior dense candidate. Raw candidate SHA256 527ae017a9dfa05e19621365b1535cc903e79a1f5e725f2b8b20584b4bd0d7d0.
Unchanged reviewer rejects: paired static8/18, dynamic6/18, expanded fit diagnostic5/18 failures; original static12/18 and dynamic1/18 failures. Both coordinate audits pass; held branches fail. Dynamic H32 root position improves 158.425 ->117.985mm, but static35.578 ->38.599mm regresses. Dynamic H1/H8 angular errors also regress. No policy promotion or PPO restart.
Supplementary center/offset/all1740 coverage audits continue in their existing queue. Next decision must use these results to distinguish center bias, derivative response and fit/validation behavior; do not launch blind coefficient tuning or change gates. Fixed7 full-duration and >=3 policy seeds remain unfulfilled.

---
2026-11-06 06:50 Asia/Shanghai — complete center errors and frozen gradient scope
All1740 coverage is complete, weights unchanged. Exact mean normalized squared errors over895 fit states (control ->candidate): linear .00444209 ->.00495532; angular .00943586 ->.01027087; joint velocity .00292474 ->.00286758. Validation845 linear .01699953 ->.01775244; angular .05703460 ->.05762708; joint velocity .01769220 ->.01696667. Thus increased linear error is not merely a mean-RMSE aggregation artifact.
Frozen dense-center gradient diagnostic v2 completed on fixed16 fit-only batches32 for original/control/candidate; zero optimizer updates and weights verified unchanged. Control linear-vs-angular/joint gradient dot products are nonnegative16/16; candidate angular conflicts2/16 and joint0/16, median cosines .03581/.05009. This does NOT support a blanket claim that the three center velocity targets conflict; remaining complete-objective terms and optimizer trajectory require evidence before another trial.
Initial gradient diagnostic failed because evaluation loader freezes params; retained failure.json. Corrected script enables gradients only for actual prepared model.net parameters. No policy accepted, no PPO launched.

---
2026-11-06 06:57 Asia/Shanghai — full objective frozen replay completed
linear_full_objectives_v1 both16 batches complete; original sampler state hashes, dynamic indices and center indices equal. All source weights unchanged, source optimizer steps0.
Candidate center-linear gradient conflicts with root-state15/16 and root-pose16/16 (median cosine -.06305/-.09300), but with original full0/16 and new complete full0/16 (median cosine .77581/.74225). Control full conflicts1/16 each. Thus component conflict exists but total Euclidean gradient usually aligns; this does not establish the cause of failed training or support blindly dropping response terms. Next inspect actual saved Adam update direction against these fixed objectives, preserve existing weights and optimizer, no policy promotion.

---
2026-11-06 07:04 Asia/Shanghai — actual Adam direction probe complete
linear_adam_direction_v1 fixed16 paired fit batches complete, source weights/checkpoint unchanged and source optimizer steps0. Candidate saved-Adam and zero-first-moment each0/16 center-linear slope ascent,0/16 finite center-linear increases,0/16 actual-full slope ascent; median finite center-linear changes -.00132514/-.00136150. Control saved moments6/16 slope ascent and8/16 finite increases, zero-first0/16 each.
These are frozen final-weight local probes, not the historical optimization trajectory. They do not support blaming candidate failure on saved first moments or repeating the already rejected global moment-reset training. Next check the finite step against all895 fit states, not only the sampled16 states, to test batch-vs-population discrepancy before another training trial. Acceptance gates and final checkpoints remain unchanged.

---
2026-11-06 07:10 Asia/Shanghai — population Adam probe complete
linear_adam_population_v1 both16 paired batches complete; all895 fit states evaluated per step, restoration reproduces population errors exactly, source weights/checkpoint unchanged, source optimizer steps0. Control saved-moment population linear/angular/joint error increases10/15/11 of16; zero-first all0/16. Candidate saved-moment linear increases0/16 (median delta-.001224854), angular1/16 (-.000845688), joint14/16 (+.000064346); zero-first all0/16 (linear-.001314878/angular-.002171407/joint-.000310651).
Candidate batch-local linear improvement also holds over full fit population at final weights; it does not explain the historical final-vs-control linear regression. Saved-Adam steps instead expose a joint-velocity tradeoff. These are fit-only final-weight counterfactuals, not held-out/policy proof. Next localize joint-velocity harm to shared hidden vs output-head saved first moments before selecting a training intervention; prior global moment-reset failure remains relevant. No policy promotion.

---
2026-11-06 07:17 Asia/Shanghai — layer scope completed; bounded training hypothesis
linear_adam_layer_scope_v1 both16 paired batches complete, source weights unchanged, source optimizer steps0. Candidate saved/output-head-only/joint-head-only probes each14/16 population joint-error increases and11/16 static increases. Shared-hidden-only first-moment reset yields0/16 population linear/angular/joint and0/16 static increases; all-first-moment reset has3/16 static increases. This supports testing a narrower shared-hidden intervention rather than repeating global reset blindly.
Limits: shared-hidden probe root-state4/16 and root-pose14/16 ascent slopes remain (saved3/16 and13/16); no response or policy acceptance inferred. Proposed next experiment: same rejected linear candidate final checkpoint+Adam1000 as common starting state; identical complete objective/samples/2144-point budget, fixed continuation1000 each; sole variable reset shared-hidden first moment before original Adam step, retaining all head/second moments and step/lr. Require reproducible4-update smoke before launch; use all unchanged gates and retain failures. No source checkpoint overwritten or promoted.

---
2026-11-06 07:19 Asia/Shanghai — shared hidden moment continuation smoke passed
shared_hidden_moment_pair_v1 smoke4 passed: repeated disabled control weights+Adam bitexact; both restore exact rejected linear-candidate checkpoint SHA527ae017... and Adam1000,22 frozen tensors,8 Adam states at1004,paired samplers,2144 points/update. Per-update proof verifies only first6 net tensor exp_avg reset in candidate, all head/second moments/steps preserved. Smoke raw control85441bfec3647fe73a5ea3d0de012e676c48a6611cbcac1fb2339c32c3a27bef candidate7ffc37e37c1bfa458c90722cea07ae506b621b1ae10f90a65fccd18c9c64e7db.
Formal plan: fixed continuation1000 each from same model+Adam1000; no loss or data changes. Exact repeat proof covers smoke4 only, not an independent historical formal1000 control. Final audit will check resume hash, paired samples/budgets/reset history and Adam2000 explicitly. Reviewer --hidden-moment historical linear report identical except time/script hash; original gates unchanged. All failure artifacts retained; no PPO/policy promotion.

---
2026-11-06 08:40 Asia/Shanghai — shared hidden moment continuation rejected
Both fixed continuation1000 arms completed, Adam2000, paired integrity passed. Raw control09a7410bdce35423b598f93a90d9fee8a24f2edeba62793ed06e177b6a270176 candidate0c3206c97ecfea1a214cddd98ed42058b0107deb9bd2d969230f436593397e95.
Unchanged reviewer rejects: paired static7/18,dynamic8/18,expanded fit4/18 failures; original static11/18,original dynamic0/18. Coordinates both pass; held branches fail. Static H32 root29.995 ->56.598mm; dynamic136.599 ->163.943mm. Short frozen one-step fit/static gains did not predict full training validation gains. Reject this candidate, retain finals, no PPO/policy promotion.
Existing supplementary dynamic action/center/offset/all1740 audits continue. Next analysis must distinguish improved fit from rollout/source/response degradation and avoid another small optimizer-only tweak solely on frozen probes. Fixed7 full-duration and >=3 policy seeds remain unmet; push still unverified.

---
2026-11-06 08:43 Asia/Shanghai — hidden moment all-state audits complete
All1740 frozen audits completed, weights unchanged. Exact mean normalized MSE control->candidate over895 fit: linear .00219534->.00167138, angular .01121175->.00480069, joint .00283489->.00128925; over845 validation: linear .01650025->.01314224, angular .05873424->.05221448, joint .01930970->.01597099. All three improve at native one-step states, despite worse static/dynamic autoregressive H32 trajectories.
This contradicts a simple all-native one-step generalization explanation; it suggests examining rollout exposure and error accumulation, without asserting causality yet. Next compare matched teacher-forced native-state velocity errors versus autoregressive errors over fixed H1/H8/H32 windows, and retain native integration defect separately. Do not launch further optimizer-only tweaks or promote this rejected candidate. All supplementary outputs summarized in frozen_response_summary.json.

---
2026-11-06 08:56 Asia/Shanghai — frozen rollout exposure audit, not policy acceptance

The dynamic and static v3 audits reproduce all 1,692 endpoint values from the paired original H1/H8/H32 audits exactly (maximum absolute difference zero). All model weights remain unchanged. Preserved early outputs include a static missing-trace-hash failure and endpoint quaternion metric precision mismatch. Static inputs now have an explicitly current hash snapshot, not a claim of pretraining provenance; endpoint statistics use the original API dtype, without changing gates.

Validation trajectory-median H32 root-position P95 (mm), autoregressive / real-input predicted-velocity integration: dynamic control 136.60 / 13.53, candidate 163.94 / 12.26; static control 29.99 / 7.75, candidate 56.60 / 6.06. Native endpoint velocity integration defects are identical across all three models (dynamic 2.44 mm, static 0.41 mm). Teacher-state diagnostics are not deployable results. Improved one-step predictions coexist with worse autoregressive validation, motivating isolation of state-feedback error rather than another optimizer-only trial.

Evidence: longterm_diagnostics/hidden_moment_rollout_exposure_v1/{dynamic_v3,static_v3}/audit.json and consistency_v3.json. Next frozen diagnostic hidden_moment_feedback_groups_v1 replaces only model-input state groups with recorded truth while integrating velocities on each branch's own state. The unchanged-input branch must match the actual model bitwise at every step; initial groups, hashes and weights are checked. All 32 dynamic traces and H32 windows are retained, no fitting or policy promotion. Goal remains active and unaccepted.

---
2026-11-06 09:02 Asia/Shanghai — recurrent input-group diagnostic completed

The three frozen models passed baseline equality at every recurrent step and all 522 H32 endpoint values; initial groups and immutable inputs/weights also passed. Validation trajectory-median root P95 (mm), unchanged input / true root-velocity input / all true inputs: reference 158.60 / 27.99 / 19.34, control 136.60 / 26.89 / 18.26, candidate 163.94 / 22.64 / 16.68. Candidate root-pose-only is 126.76, joint-position-only 173.59, joint-velocity-only 169.36. Evidence: hidden_moment_feedback_groups_v1/summary.json. These oracle mixed-state sensitivity tests do not establish physically causal changes or deployable stability.

New script audit_supermimic_recurrent_feedback_groups.py preserves the earlier fit-only single-step audit_supermimic_feedback_groups.py. The active command used the new code under the older name at 3f88aef; source_path_note.txt records the rename, and 72dc042 restores the original diagnostic. No running evaluation was restarted.

Next frozen native-root response audit reuses grounded_motion_response_alignment_v1/serial_full: 5,760 endpoints, 480 batches, all archive/write/paired-repeat/native-replay/termination checks passed. Both source-isolated partitions, all 48 anchors per partition, two magnitudes and all six axes are retained. It compares full-precision finite response, normalized error and spectral/singular gains; finite responses near contact are not global Jacobians or stability proofs. No optimizer, new acquisition or policy promotion. Latest commits remain unpushed because of the previously verified GitHub authentication failure.

---
2026-11-06 09:04 Asia/Shanghai — native root response audited; exact drift ledger started

All frozen native root-response conditions completed with inputs and weights verified. Control/candidate normalized derivative MSE: fit .08752/.11991, validation .18670/.18700. Validation absolute root-linear RMSE .05376/.04883 m/s; root-angular .81872/.85250 rad/s. Validation local six-axis response median spectral radius control/candidate/native .93346/.94604/1.00428; normalized maximum singular median .97502/.99036/1.05633. This does not support a simplistic claim of excessive median model velocity gain; finite contact responses are not the full coupled-state Jacobian or stability proofs. Evidence hidden_moment_native_root_feedback_v1/audit/audit.json.

Next: hidden_moment_root_drift_ledger_v1, tmux hidden_root_drift_ledger, all three models and all32 dynamic traces/H32 windows. Decompose actual root-position drift into native endpoint-velocity integration defect, real-state one-step bias, predicted-input feedback, previous-velocity blending feedback and arithmetic rounding. Require per-window reconstruction residual below 1e-12 m, hashes and frozen weights unchanged; this only validates algebra and does not replace original acceptance gates. No new PPO or policy promotion; long-term goal remains Active and incomplete.

---
2026-11-06 09:12 Asia/Shanghai — fit-only exposure range evidence and native label expansion

Completed root-drift ledger reproduces original endpoints and reconstructs every root-position vector within 7e-16 m. Candidate validation trajectory-median component P95: native integration 2.439 mm, one-step bias 10.396 mm, predicted-input feedback 103.059 mm, old-velocity blending feedback 48.752 mm, rounding .000085 mm. Component quantiles are separate vector summaries and must not be added. Exact exposure-coverage rerun preserves all original ledger/metric values.

Existing velocity branches reach only +/- .01 m/s linear and .04 rad/s angular. Across the three models, the maximum FIT trajectory-median per-axis P95 autoregressive input discrepancy is .21392 m/s and 1.25988 rad/s. Validation statistics are evaluation only and were not used to select acquisition levels. Retain base [.005,.005,.005,.02,.02,.02] and levels1,2; add8,32,64, reaching .32 m/s and1.28 rad/s. Overlapping H32 inputs are not independent physical samples. Evidence hidden_moment_root_velocity_exposure_v1/dynamic/audit.json.

wide_native_root_velocity_v1 smoke:132 endpoints,11 batches, zero write/repeat/replay/termination failures; all5 original-level batches regress to the verified old archive. smoke_integrity_v2.json also verifies that the final proof file is written only after both original checks and overlap regression pass. The base-only report is separately named and cannot stand in for the wide final proof.

Full acquisition started09:12 in tmux wide_root_velocity_full: all32 traces, first/middle/final anchors,11 batches per anchor, expected1,056 batches/12,672 endpoints. Only root-velocity amplitudes change; original PD actions, native state/warmstart, diagnostic dense Jacobian physics, and source isolation remain. Original termination outputs retained. Runner transitions collecting -> auditing -> audited_no_training only after full integrity and all480 original-level regression checks pass; errors set failed. No fitting or PPO is launched automatically. Do not restart a live collector. Long-term full7/3-seed acceptance remains unfulfilled.

---
2026-11-06 09:15 Asia/Shanghai — wide frozen evaluation prepared and queued

Previous goal turn was progress (coverage evidence, verified smoke, full native acquisition started). The collector remains live as PID47834 in wide_root_velocity_full;132/1,056 batches from four source episodes are logged. It was not restarted.

Prepared supermimic_wide_root_response.py: loading requires the complete12,672-endpoint proof, zero failures,480 original-level regressions, correct physics/asset/mapping, frozen source-isolated partitions and all input hashes. Default split is fit; validation is explicitly evaluation only. It retains terminal endpoints rather than filtering them.

Prepared audit_supermimic_wide_root_models.py to compare the same three immutable models at each of levels1,2,8,32,64. It reports root/joint finite-response MSE, absolute velocity RMSE and all terminal fractions. Original levels1/2 must reproduce the prior frozen audit within1e-10 (numerical comparison only; original acceptance gates unchanged). All1,440 four-endpoint conditions per split and all48 anchors are retained; models and input files must remain unchanged. Syntax parsed; full end-to-end evaluation remains UNVERIFIED until acquisition finishes.

Queue tmux wide_root_models, live bash PID47937, waits for the actual acquisition tmux to end and requires audited_no_training before loading the complete proof; state wide_root_models_v1/status and queue.log. Failure stops the queue; no fitting, checkpoint selection, PPO or promotion occurs. Goal remains Active, unaccepted; necessary commits are remote, push remains blocked by previously verified GitHub authentication.

---
2026-11-06 09:36 Asia/Shanghai — wide acquisition verified; controlled continuation smoke passed

All1,056 native batches/12,672 endpoints completed with zero archive/write/repeat/native-replay/termination failures; all480 original-level batches regress to the old archive. Frozen three-model evaluation at all levels completed. Initial metric check failed at4.14e-9 because new denominator inference usedfloat64 versus the legacy explicitfloat32 then promotion. Preserved failure and restored exact legacy arithmetic; original endpoint/response comparisons then agree within7e-17 without changing1e-10 threshold. Wide labels alone do not prove that coverage causes long-horizon drift, and improved finite responses need not improve rollout stability.

The proposed single variable is dynamic root-response amplitude sampling, not a new loss, optimizer or physics. Both arms resume raw shared_hidden_moment_pair_v1/control SHA09a7410bdce35423b598f93a90d9fee8a24f2edeba62793ed06e177b6a270176 and exactAdam2000; retain complete objective, center linear1, original Adam/lr/clipping,2144 prediction endpoints/update. Original sampler52 selects the SAME source anchor and axis in both arms. Independent sampler62 draws a level in both arms; control retains original1/2 sample while candidate uses1/2/8/32/64. New loader verifies original data/denominator overlap bitexact, onlyfit arrays used.

wide_root_response_pair_v1/smoke4/fit_integrity.json passed4 updates/arm plus repeated control: control raw3e437aa5ffde1d749168d8552caeb1c896a31c944c5d06ee4963ac25176e8366, candidateba48a42c2f88bbdd2a059ca5c06476541c4b48e208597dd23e1ff492611e8ddb. Repeated control weights AND Adam are bitexact;22 frozen tensors, all8 Adam steps2004, all paired samplers and point budgets passed. This repetition evidence applies to4 steps only, not the unrun1000-step formal result.

Reviewer adds an explicit wide-root plan flag and unchanged original trajectory/branch/coordinate thresholds; replay of the complete historical H review is identical except timestamp and reviewer hash. Formal runner is gated on smoke, fixed1000 each arm and no checkpoint selection, then exports declared precision, audits original static/dynamic/expanded/coordinates/branches and original-reference comparisons, finally reviews without promoting any policy. Error trap records failed. Long-term fixed7/full durations/3seeds/videos remains incomplete.

---
2026-11-06 10:55 Asia/Shanghai — fixed1000 pair completed; outcome evaluation still pending

wide_root_response_pair_v1/fit_integrity.json passed: both1000 updates, Adam3000,22 frozen tensors,2144 endpoints/update, original and added amplitude/anchor/axis sampler replay. Raw hashes: control8574e57a49f3e88c25da24c124025add618922ebf44e9fe998212a37cd88466b; candidate3094670a3f47dcc30769da56cc8d0116f73045b09742c6db7325252cd21af6b6. Formal repeated-control bitexactness remains null; only the4-step smoke was repeated. This is training-integrity evidence, not evidence of control improvement or policy acceptance.

Primary original H1/H8/H32/coordinate/branch/reference evaluation remains live (control expanded PID49225 at this observation). Supplementary native-root then all-wide-level evaluation is queued as tmux wide_root_supplementary, bash49280, waiting for the actual primary tmux to finish and a reviewed_no_policy_promotion status. Wide-model audit accepts an explicit original-level audit with its hash, requiring those same final models' old-level metrics to reproduce within unchanged1e-10 arithmetic check. No fitting, terminal filtering, checkpoint selection or policy promotion. New optional argument syntax parsed; queued end-to-end invocation remains unverified until it runs. Long-term fixed7/full-duration/3-seed/video acceptance remains incomplete.

---
2026-11-06 10:58 Asia/Shanghai — wide root-response continuation REJECTED by primary gates

Both fixed1000 finals passed training integrity but failed the unchanged development review: static10/18, dynamic1/18, expanded fit diagnostic1/18, original-reference static12/18, original-reference dynamic0/18. Both coordinate contracts pass; paired and original-reference held-branch requirements do not pass. No promotion.

Validation trajectory-median H32 root-position P95: control192.701 -> candidate155.765 mm dynamic improvement, but46.836 ->56.235 mm static regression. Dynamic failure is H1 root-orientation P95 .0253833 -> .0270562 rad, beyond original1.05 upper limit .0266524. All final artifacts and failures retained. Partial dynamic improvement is not accepted stability.

Evidence wide_root_response_pair_v1/paired_metric_review.json. Supplementary frozen native-root response PID49812 is live, followed by all wide amplitudes and original-level arithmetic consistency. No further fitting is queued; finish supplementary evidence before choosing the next single-variable intervention. Long-term fixed7/full durations/3seeds/videos remains unverified and goal Active.

---
2026-11-06 11:00 Asia/Shanghai — wide response benefit verified, rejected outcome unchanged

Final supplementary native and wide response audits completed for all three models, both partitions and all1,440 four-endpoint conditions per split. Final pair old-level metrics reproduce independent audits exactly, weights remain frozen, all terminated endpoints (~27.01%) retained. Validation all-level normalized root-response MSE .1015073 -> .0885739; absolute root-angular RMSE .853499 -> .787886 rad/s. Validation root-response MSE improves at every individual acquired level. This does not override static/orientation/branch gate failures.

FIT original levels1/2 root-response MSE regresses .082509/.036268 -> .126089/.043642. Replacement of the sampling pool may dilute original local constraints, but this observation does not prove the cause of static drift. Do not accept only favorable amplitudes or dynamic metrics. Evidence wide_root_response_pair_v1/wide_root_response/audit.json and native_root_response/audit.json.

Detailed frozen audit tmux wide_root_detailed_audits started11:00 (root-state response PID50097); subsequent static root-pose, dynamic action response, exact center/offset and all1,740 native one-step state comparisons reuse unchanged source-isolated collections. Hashes/weights and all groups retained, no fitting or terminal filtering. Finish these checks before deciding whether a preserving-original-sampling intervention is justified; no additional training is currently queued. Long-term goal remains Active and incomplete.

---
2026-11-06 11:04 Asia/Shanghai — detailed frozen audit complete, static regression still unresolved

Evidence wide_root_response_pair_v1/detailed_summary.json uses arithmetic mean normalized MSE, not mean RMSE. All1,740 native states retained (895fit/845validation). Validation linear/angular/joint velocity MSE control .014980/.054075/.019615 -> candidate .013672/.046141/.017865; all three fit metrics also improve. Both partitions'48 exact centers improve all three velocity metrics, as do the96 validation dynamic action-response groups. Static root-pose angular-response MSE improves1.164e-5 ->4.694e-6, while static root-velocity linear-response MSE regresses2.158e-10 ->3.286e-10. These statistics do not establish static autoregressive stability; primary rejection stands.

Static exposure diagnostic started11:04 in tmux wide_static_exposure, same three final models, all16 static traces and all fixedH1/H8/H32 windows. Reuses current input hash snapshot; weights frozen and original autoregressive endpoint metrics must reproduce exactly. Separates real-state-input velocity integration and native integration defect from free rollout. Oracle-state inputs cannot be deployed. No new fitting/threshold changes/promotion; goal Active and unaccepted.

---
2026-11-06 11:08 Asia/Shanghai — static exposure reproduced, finer feedback groups started

Static H1/H8/H32 endpoint metrics reproduce the primary audits exactly with frozen weights. H32 fit root-position P95 control23.240 -> candidate39.374 mm, despite real-input independent velocity integration4.186 ->3.216 mm; validation46.836 ->56.235 mm versus real-input6.704 ->5.696 mm. Native integration defect is identical across models (~.413 mm validation). Feedback regression also occurs on fit, so a purely held-out-source generalization explanation is insufficient. Evidence static_exposure_summary.json; oracle-input results are not deployable performance.

Root-pose response code uses acquisition base_delta(.0005 m/.001 rad) as its input response scale, with logged loss ~1e-6. This is a current objective scaling choice, not yet evidence of a bug or justification to increase weights blindly.

At11:07:57 started tmux wide_static_feedback/PID50998 on all16 static trajectories/H32 windows and the same three models. Retains existing groups and adds separate root linear/angular velocity and root position/orientation truth substitutions in model INPUT only; integration stays on branch state. Unchanged-input branch must remain bitexact at every step, weights/hashes frozen. Mixed-state sensitivity does not establish physical causality or policy acceptance. No additional fitting, gate changes or promotion; long-term goal Active and incomplete.

---
2026-11-06 11:18 Asia/Shanghai — static group diagnosis completed; additive-wide smoke passed

All three models/all16 static traces/H32 groups completed. All288 unchanged-input endpoint metrics reproduce the prior audit and every baseline recurrent state is bitexact. Candidate validation root-position P95 (mm): unchanged56.235, true root velocity4.399, onlylinear5.530, onlyangular14.440, onlyorientation29.125; onlyposition worsens to116.416. These mixed-input oracle diagnostics identify sensitivity, not a deployable correction or physical causality. Evidence static_feedback_summary.json.

Next controlled intervention additive_wide_root_pair_v1 retains EVERY original1/2 root response sample and complete objective, adds32 paired anchor/axis endpoints at8/32/64 in BOTH arms, and changes only coefficient0/1 on the extra half-weighted root response MSE (the original dynamic coefficient.5). Original root sampler52, independent new sampler63, center/dynamic action/native samplers, model/Adam2000 resume, normalization, lr and clipping remain fixed. No validation arrays loaded for fitting. Both2176 endpoints/update.

Four-update smoke for both arms plus repeated control passed: repeated weights AND optimizer bitexact,22 frozen tensors,8 Adam steps2004, paired original and additional amplitude/index replay,2176 endpoints/update. Raw control7ce05852aeb0fb76d8935298ef1e36689b747377a78cdaa3cc90b6a6178693a0; candidate7e2cb1573d37ec50744b51cf186d9db2ce11a2eda21a01785204644a84b00633. Evidence additive_wide_root_pair_v1/smoke4/fit_integrity.json. Formal1000-per-arm remains unverified. Reviewer adds an explicit additive-wide plan flag; complete historical W review remains identical apart from timestamp/script hash, thresholds unchanged. Formal runner remains gated by smoke and does not promote policies. Goal Active and incomplete.

---
2026-11-06 11:58 Asia/Shanghai — additive-wide control1000 completed and historical control reproduced

Raw control SHA06b201bd1e283685e1fd47bced7976d513f53d57a6a59c7d0f14316cc8122324; Adam3000,2,176,000 actual prediction points. CPU comparison verifies ALL model weights and optimizer state bitexact against previous wide_root_response_pair_v1/control1000 SHA8574e57a49f3e88c25da24c124025add618922ebf44e9fe998212a37cd88466b. Extra coefficient0 forward therefore preserves the original full training result; this is measured1000-step control evidence, not an extrapolation from smoke. Evidence additive_wide_root_pair_v1/control_reference_integrity.json.

Candidate PID51593 started sequentially in the same tmux at fixed1000, coefficient1; do not restart a live trainer. Pair outcome and full original gate review remain pending, no promotion. Long-term goal remains Active and incomplete.

---
2026-11-06 12:38:50 Asia/Shanghai | Additive wide pair1000 finished; offline evaluation running

Both formal arms completed the fixed1000 updates. fit_integrity.json passed:22 frozen tensors,8 Adam states,2176 points/update, original paired indices and center replay consistent, sole additional-wide coefficient variable verified. Candidate raw SHA256 d66bad324b0374088d0204a9f5850592b4717e6c94f24b15efd6f582c8c7f3ea; control raw06b201bd1e283685e1fd47bced7976d513f53d57a6a59c7d0f14316cc8122324. Formal auditor repeated-control scope remains4 smoke updates; separate control_reference_integrity.json already proves control1000 model and Adam bitexact to historical Wcontrol1000. These scopes must not be conflated.

Primary offline evaluation is active under tmux additive_wide_root_pair. A supplemental frozen queue (tmux additive_wide_supplementary) waits for primary session completion AND reviewed_no_policy_promotion, then checks narrow/wide native root responses and detailed static root/dynamic action/all native state coverage. Queue scripts stay in the experiment directory, bash -n passed; no training edits or output overwrites. All artifacts remain on the data disk. No PPO/promotion or policy acceptance; fixed7 full durations,3 seeds and videos remain unfulfilled.

---
2026-11-06 12:42:15 Asia/Shanghai | Additive wide pair REJECTED; frozen supplemental audits active

paired_metric_review.json rejects the unchanged development gate: failed static6/18,dynamic4/18,expanded-fit2/18,original-static11/18,original-dynamic1/18; both coordinate checks pass, held branches fail. DynamicH32 root position P95 improves192.701 to132.905mm versus paired control, and158.598 to132.905mm versus original. StaticH32 root position46.836 to47.158mm passes5% paired tolerance but remains worse than original29.838mm; staticH32 angular velocity RMSE0.248770 to0.302732rad/s and joint velocity0.0138742 to0.0184902rad/s regress. DynamicH1 orientation0.0253833 to0.0307869rad and H8/H32 joint position also regress. This is not accepted policy performance or a full-duration test.

Primary ended reviewed_no_policy_promotion; supplementary native root feedback audit PID52875 observed live after queue51838 released. Continue fixed narrow/wide and detailed native/static/dynamic diagnostics before selecting a single next hypothesis. No new optimizer/PPO trial or promotion launched. Preserve final checkpoints, all original thresholds and fixed7/3seed/video requirements.

---
2026-11-06 12:52:47 Asia/Shanghai | Additive diagnostics complete; static velocity coverage audit launched

All native and detailed audits completed without model updates. wide_response_summary.json reproduces old-level reference exactly; all five validation root-response MSEs improve, total0.1015073 to0.0883444, but fit level1 worsens0.0825091 to0.1155045 despite preserved original samples. detailed_summary.json covers all895fit/845val native states: mean normalized MSE fit(linear,angular,joint velocity)0.00160769/0.0119365/0.00304187 to0.00139486/0.00876288/0.00256309; validation0.0149796/0.0540751/0.0196149 to0.0153293/0.0466758/0.0170366. Validation linear bias worsens, not all improvements. Sparse48 validation center linear MSE0.0299852 to0.0356903;96 action-response rows improve all3 groups slightly.

Static exposure and recurrent group audits finished on all16 traces and3 models; all weights unchanged, group baseline every step bitexact. H32 validation median per-trace P95 rootpos: control autoregressive46.836mm/teacher-integrated6.704mm; candidate47.158/6.239mm; native integration defect0.413mm. Candidate oracle-input rootvelocity group4.618mm, linear5.178mm, angular13.097mm, orientation27.329mm, rootposition93.889mm, jointposition34.838mm,jointvelocity36.249mm. This sensitivity implicates root velocity feedback for further investigation, not physical causal proof; oracle input replacement is never deployable control. Summaries preserve audit hashes and all groups.

Next read-only measurement launched under tmux additive_static_velocity (pane54147): reuse audit_supermimic_root_velocity_exposure_coverage.py with original static collection, fixed static manifest and same3 models. Quantify actual fit-side rollout input velocity error range versus existing native branch amplitudes before proposing static wider collection; no amplitude chosen from validation, no training started or threshold changed. Runner syntax passed; output static_velocity_coverage. Policy remains rejected; full fixed7/3seed/video acceptance unfulfilled.

---
2026-11-06 12:54:17 Asia/Shanghai | Static velocity coverage metadata fix; v2 measurement running

Initial static_velocity_coverage audit stopped with KeyError sha256 before producing a final audit: static results store hashes in the independently fixed trace manifest, unlike dynamic inline row hashes. Preserve failed output/log. Updated coverage auditor to use the same manifest fallback as the original rollout-exposure auditor; inline hashes retain original priority and exact equality. Added exact consecutive state/next-state continuity check. Results manifest hash and all16 static file hashes verified before relaunch; AST parse and runner bash syntax passed. No metric/threshold/data/model/optimizer changes. Relaunched separate output static_velocity_coverage_v2 under tmux additive_static_velocity_v2 (pane54246); initial failure is not treated as a model result.

---
2026-11-06 12:57:15 Asia/Shanghai | Static velocity coverage verified; wider native collection proposed

static_velocity_coverage_v2 completed3models x16traces, all weights unchanged, state continuity checked; exact ledger residual <2.9e-16m. Fit-only max across models of per-trace P95 medians on six velocity axes:0.06633864,0.01939638,0.01031283m/s;0.05803167,0.30401587,0.31820194rad/s. Existing native branch max0.01m/s and0.04rad/s undercovers these fit exposure statistics; overlapping H32 inputs are not independent native samples and do not prove causality.

Saved wide_static_root_velocity_v1/proposal.json with evidence hash and no validation-based amplitude selection. Preserve old1/2 levels; add8/16, where16 is next power-of-two covering max fit/base ratio15.9101. Maxdelta0.08m/s and0.32rad/s. Preserve static fit warmups5/25/75/150 and validation15/35/125/175, asset/PD/mapping/physics. This is only planned, not collected or trained; require native smoke integrity, all original-level overlap regression, retained failures and full audit before any training. Static temporal split does not replace source-isolated fixed7 acceptance.

---
2026-11-06 13:02:04 Asia/Shanghai | Wide static native smoke passed; full collection launched

Added collect_supermimic_wide_static_velocity.py: verified fit-only frozen proposal/evidence hash; AST changes exactly original plan contract and levels1/2/8/16. Original reset/PD/physics/mapping remain executed unchanged. Added auditor reusing all original branch archive checks, then complete retained-level regression before final integrity.json; base_checks.json alone is insufficient. Initial wrapper smoke failed before environment creation because a results dictionary had dynamic contract expression; fixed constant-node matching, verified exactly one plan match, preserved smoke.log.

smoke_v2 passed9files108endpoints, zero terminations, requested-write error5.96e-8, replay error1.66e-7; retained5 original files maxdifference1.51247e-6 at original1e-5 tolerance. Verified all40 original archived hashes against original integrity before full launch. Full expected72files864endpoints, including40 retained files; tmux wide_static_full (pane54556), output wide_static_root_velocity_v1/serial_full, full.log. All files remain data disk; available22GiB. No training/PPO/promotion; full integrity must pass first.

---
2026-11-06 13:06:32 Asia/Shanghai | Wide static full integrity and frozen responses complete

wide_static_root_velocity_v1/serial_full integrity passed72files864endpoints,0terminations; requested state write error8.94e-8, replay3.88977e-6; complete40 original files max difference4.51133e-6 at unchanged1e-5. No rejected branch omitted. Frozen3model x32condition response audit completed, all weights/norms unchanged. New wrapper audit_supermimic_wide_static_models.py changes only accepted collection contract in original auditor, requires complete72file/40retained-file proof, records base/wrapper/integrity hashes; preserves finite-response formula and all outcomes.

frozen_models_summary.json contains all per-level fit/validation mean squared response RMSEs with physical output units (base input scale normalization, not parent output-standard normalization). Additive dynamic-wide candidate is worse than paired control on every static level and each reported root linear/angular/position response group. Example validation16x linear2.73375e-9 to3.38493e-9, angular7.39178e-8 to1.10957e-7. These are finite interval responses, not global infinitesimal Jacobians or closed-loop stability proof. Together with fit velocity undercoverage and static recurrent sensitivity, this supports investigating an additional static-wide response objective while preserving narrow/static/dynamic samples; inspect exact loss scales and paired contract before any training. No new training launched; final fixed7/3seed/video requirements remain unmet.

---
2026-11-06 13:11:06 Asia/Shanghai | Static-wide fit loader/formula verified; paired experiment frozen

Added AdditionalStaticWideRootLoss loading only16 verified fit branch files at8/16 amplitudes,48 four-endpoint conditions; no validation arrays. Original static derivative and frozen root input/output std normalization, half weight matching original static mix,8conditions32points, independent seed64. Actual float32 yaw-canonical training response model probe:4 matched calls bitexact, all weights unchanged,0 optimizer updates; half-loss values1.38668e-5,5.42360e-5,2.36942e-5,4.16784e-5. Separate original RootStateResponseLoss method on identical arrays/sampler produces bitexact four losses (loss_formula_regression.json). Initial probe used declared offline predictor without state_norm and failed before fitting; failure preserved, corrected probe to actual training response API, not loss formula.

static_wide_response_pair_v1/proposal.json freezes sole extra-static-wide coefficient0/1, common Hcontrol09a741 model+Adam2000,1000updates each, all previous objectives retained (dynamic-wide addition absent),2176points both,seed64. Next implement paired runner/auditor, require repeated-control4 smoke and frozen/Adam/all samplers/point accounting before formal run; compare control1000 with historical Wcontrol1000. No training started, thresholds/fixed7/3seed/video scope unchanged.

---
2026-11-06 13:16:27 Asia/Shanghai | Static wide paired4 smoke passed; formal launch ready

Implemented calibrate/audit/run_supermimic_static_wide_pair and review_supermimic_static_wide_metrics. Paired control/candidate4 updates and repeatedcontrol4 pass;22frozen tensors,8Adam2004,2176points/update, original dynamic/center/native/observer sample histories paired, extra static48condition seed64 histories independently replayed, fit-only16files. Repeated control model+Adam bitexact; additionally historical Wcontrol4 model+Adam bitexact (raw3e437aa5ffde1d749168d8552caeb1c896a31c944c5d06ee4963ac25176e8366). Smoke raw control30a88efe7c6207b186c2dd7636516574ee671cd05155e0bd0d49df2d8cf2d0fd,candidate2575e718ed10674a96bab9ff7a2201be4c01dfc8f01d4fae54e5f857f2b24389. Preserved initial smoke proof before adding historical comparison. Formal auditor now requires model+Adam exact historical Wcontrol1000, not extrapolating4-update proof.

Reviewer source equivalence verified: inverse mapping only explicit trial flag and sole coefficient field reproduces original entire reviewer text exactly; all gates unchanged (reviewer_contract_equivalence.json). Both arms retain original complete objective; only extra static-wide half-response coefficient0/1,32 extra points each, common Hcontrol09a741+Adam2000,original lr/clipping,no reset. Formal1000perarm runner ready with all primary offline audits and original-reference checks, no overwrite, no automatic policy promotion. Full fixed7/3seed/video acceptance unfulfilled.

---
2026-11-06 13:56:25 Asia/Shanghai | Static-wide control1000 exact historical match; candidate running

Control finished fixed1000, Adam3000. Independent CPU comparison of every model tensor and complete Adam state is bitexact to historical Wcontrol1000 (control_reference_integrity.json). Current raw6c89e408fd9105c956a330bb9955922ad6388bebbd17b3369b2fac6ffacb2d01; historical8574e57a49f3e88c25da24c124025add618922ebf44e9fe998212a37cd88466b (metadata hashes differ). This is actual1000 proof, not4-step extrapolation or candidate acceptance. Candidate started automatically, PID55694 confirmed live; no restart. Primary runner and gated supplemental queue remain active.

---
2026-11-06 14:35:10 Asia/Shanghai | Static-wide formal1000 pair integrity passed; offline audits active

Both arms finished fixed1000. fit_integrity.json verifies sole static-wide coefficient variable, paired original and extra seed64 indices,2176points/update,22frozen tensors,8Adam3000 states, exact historical control1000 model+Adam. Raw candidate d34266c3c43a1626a5e8a343ef4b1f230e3178e3167604a31cf6e38a79739483; control6c89e408fd9105c956a330bb9955922ad6388bebbd17b3369b2fac6ffacb2d01. Declared control exported with unchanged stored weights; primary offline evaluation now running. This is experiment integrity only, not development-gate or policy acceptance. Supplemental queue remains gated on primary completion. No PPO or promotion; original fixed7/3seed/full-duration/video requirements remain.

---
2026-11-06 14:39:09 Asia/Shanghai | Static-wide candidate REJECTED on unchanged complete gates

paired_metric_review.json rejects candidate: failed paired static2/18,dynamic4/18,expanded fit1/18,original static8/18,original dynamic0/18. Both coordinate checks pass; held response branches fail. DynamicH32 rootposP95 improves192.701 to143.267mm (original158.598), static46.836 to48.062mm remains within paired5% but worse than original29.838mm. StaticH32 angular velocity RMSE0.248770 to0.277229rad/s and jointvelocity0.0138742 to0.0163349rad/s regress. DynamicH1 orientation,jointposition,jointvelocity and H32jointposition exceed paired thresholds. No thresholds changed, no checkpoint selected or policy promoted. Declared candidate13797590a8cfd525ddc486139d7e725278163fdfc35f3980f377785407c7f592. Primary finished reviewed_no_policy_promotion; existing supplemental queue continues frozen native narrow/wide/static-wide response and static recurrence diagnostics before another hypothesis. Fixed7 full-duration/3seed/videos remain unfulfilled.

---
2026-11-06 14:50:26 Asia/Shanghai | Static wide supplemental audits complete; endpoint-bias diagnostic planned

All supplemental audits completed, original weights/norms unchanged. All16 static feedback traces and3models verify baseline every step bitexact. Candidate validation H32rootpos48.062mm; oracle rootvelocity4.6177,linear4.1215,angular12.7905,orientation26.4812,jointpos32.7118,jointvel40.2167mm; oracle input replacement is sensitivity only. Teacher-integrated validation rootpos6.704 to5.915mm improves while autoregressive46.836 to48.062mm worsens.

New static wide8/16 response improves all reported root linear/angular/position groups in both partitions; validation16x linear MSE2.73375e-9 to1.05459e-9,angular7.39178e-8 to5.64279e-8. Narrow1/2 linear response worsens (validation mean2.15848e-10 to2.57998e-10),angular improves4.48043e-8 to1.99897e-8. Wide dynamic validation root response0.101507 to0.102387 slightly worse. All1740 native states mean normalized MSE fit all3 improve; validation linear0.0149796 to0.0146577,jointvel0.0196149 to0.0188406 improve,angular0.0540751 to0.0557700 worsens. Full summaries retain source hashes and all split/level/group results.

Next proposal static_wide_endpoint_bias_v1: measure absolute root/joint velocity error at all four native endpoints; separate common/even bias from signed odd response and repeat residual, verify exact squared-error decomposition and unchanged model state. Derivative supervision cancels common mode, so wider response gains alone cannot prove endpoint accuracy. This is a hypothesis for frozen measurement, not a proven cause or new training choice. No weight changes or PPO; preserve all full-goal gates.

---
2026-11-06 14:54:33 Asia/Shanghai | Endpoint decomposition verified; root common-bias-only explanation unsupported

New frozen static endpoint audit complete3models x768perturbed endpoints,all4levels and bothpartitions. Original finite-response metrics reproduce prior audit bitexact; weights/norms unchanged,0optimizer updates. Orthogonal common/signed-odd/repeat-residual squared-error identity passes1e-12 for everycondition/group. At validation16x, normalized rootlinear endpoint MSE5.01681e-5 to2.34120e-5,rootangular5.30521e-5 to2.23337e-5; corresponding common biases4.46573e-5 to2.13303e-5 and5.03528e-5 to2.00132e-5 improve. Thus do not assume missing root common-bias correction explains rollout failure.

Jointvelocity16x endpoint MSE fit2.01563e-4 to2.06307e-4,val2.04904e-4 to2.09700e-4 worsens. Candidate signed-odd joint contribution fit1.67659e-4 (~81.3% total),val1.69730e-4; repeat residual~1e-13 is negligible. Existing root derivative objective supervises only7:13, not43:73. Evidence motivates checking omitted root-input/joint-output response, not a proven stability cause. Saved next_hypothesis.json: inspect original slices, probe fit-only normalized joint response and finite gradients while verifying retained rootloss/sampler unchanged before any new paired training. Full-goal acceptance and no-policy-promotion remain unchanged.

---
2026-11-06 14:57:56 Asia/Shanghai | Root-to-joint static response probe passed; paired proposal frozen

Added StaticWideCoupledResponseLoss reusing same8 fit conditions and32forward endpoints; retains original half root6 loss exactly and optionally adds half jointvelocity30 response with original frozen input rootstd/output jointstd normalization. Source audit confirms existing root-only derivative slices7:13 omit43:73. Actual training float32 yaw-canonical model probe4calls passes pairedindices and rootloss bitexact with original for both disabled/enabled cases; unchanged weights,0optimizer updates; all8net gradient tensors finite/nonzero. Joint half losses0.00197976/0.00142229/0.00185632/0.00183830; gradient norms0.0115002/0.0106324/0.0109606/0.0120804. This proves executable loss and gradient, not improved training or stability.

Frozen static_coupled_response_pair_v1/proposal.json: both retain static-wide root coefficient1 plus complete old objective, sole jointresponse coefficient0/1; same common Hcontrol09a741+Adam2000,1000updates,2176points(no new forwards),fit-only48conditions seed64. Control should match previous Scandidate at same4/1000 updates; require actual model+Adam comparison and repeatedcontrol smoke before formal launch. No new paired training started; no threshold/sample changes or promotion.

---
2026-11-06 15:02:37 Asia/Shanghai | Static coupled4 smoke passed; formal launch ready

Implemented static_coupled calibrator,auditor,runner,reviewer. Both arms retain static-wide root1 and full original objective; sole additional joint-response coefficient0/1. Same8 fit conditions/32forward endpoints;2176total/update, common Hcontrol09a741+Adam2000,original lr/clipping,no reset. Four updates each and repeatedcontrol pass:22frozen tensors,8Adam2004,paired original/extra seed64 histories,fit-only16files,point tally. Currentcontrol model+Adam bitexact repeatedcontrol and prior static-wide candidate4 (historical raw2575e718ed10674a96bab9ff7a2201be4c01dfc8f01d4fae54e5f857f2b24389). New rawcontrol3b8928852a6dad0c637499e64021199bdda9da8c144ea3cf703bc1be6d133ff5,candidate9da32ecb89b173f1d10a96bda2f46736d81c37b573f6665e8ba2aa6cd77e283f. Formal auditor requires exact historical static-wide candidate1000 model+Adam too.

Reviewer inverse trial-name/field mapping plus removal of added retained-root guard reproduces entire previous reviewer source exactly; all metric gates unchanged. Runner formal1000each ready, followed by complete primary offline/coordinate/branch/original reference audits, no promotion. This experimental baseline is unaccepted, not an active robot policy. Full fixed7/3seed/full-duration/video acceptance remains unfulfilled.

---
2026-11-06 15:04:42 Asia/Shanghai | Static coupled supplemental queue prepared

Formal control58851 confirmed live;22GiB available. tmux static_coupled_supplementary (pane58911) waits primary session completion and reviewed_no_policy_promotion before all native narrow/wide responses,detailed action/center/all-state coverage,static exposure/feedback,static-wide native responses,and exact endpoint bias decomposition. Only after all commands succeed writes all_supplementary_status. All scripts syntax checked. Endpoint auditor gained optional explicit response-reference path, removed before original argument parser; inverse source mapping proves all original metric/identity/weight checks unchanged, recorded endpoint_auditor_compatibility.json. No training edits, thresholds or data changes.

### 2026-11-06 15:44 Asia/Shanghai — static coupled control1000 integrity

The control completed 1000 updates; candidate PID59361 started automatically.
CPU recursive comparisons confirmed every world-model tensor and complete Adam
state bitexact against static_wide_response_pair_v1/candidate at 1000 updates.
All eight Adam states reached step3000. Evidence:
`static_coupled_response_pair_v1/control_reference_integrity.json` under the
remote longterm_diagnostics root. Current serialized SHA256:
`5db4de92934324384086264a5efd1de05bc4a6a14906b9498b9b22e29e80dd4c`;
historical reference SHA256:
`d34266c3c43a1626a5e8a343ef4b1f230e3178e3167604a31cf6e38a79739483`.
Different serialization hashes do not imply different tensor/optimizer contents.
Candidate1000 and the complete paired and supplementary diagnostics remain
pending. No policy acceptance or PPO launch; original final gates unchanged.

### 2026-11-06 16:27 Asia/Shanghai — static coupled candidate rejected

Both fixed1000 arms completed; paired integrity passed. Candidate raw SHA256:
`f3d6e74d1061af8731996c22490730bd8f37bc727110dc3d8367978ae9136387`.
Declared candidate: `add8f62743295bfd16174e832d9824ccbd08e2e4cc25843416a695c01ce3f579`.
`static_coupled_response_pair_v1/paired_metric_review.json` rejects the candidate:
paired static12/18 and dynamic7/18 fail; expanded fit-only4/18 fail;
original-reference static13/18 fail and dynamic0/18 fail. Both coordinate
checks pass; held branches fail. Gates and final acceptance remain unchanged.
Validation trajectory median H32 root-position P95 improves static48.062 to
44.066mm, dynamic143.267 to139.019mm. However static H1 linear-velocity RMSE
increases .006492 to.007291m/s and joint velocity .027347 to.029703rad/s;
dynamic H32 angular velocity increases1.468126 to1.641061rad/s and joint
velocity .073920 to.087957rad/s. Original static root-position29.838mm
remains better. Local position improvement is not stable-control acceptance.
The supplementary queue started native response diagnostics (PID60804 at16:27).
Complete coverage, exposure, feedback and endpoint decomposition before
choosing another single-variable hypothesis. No PPO or policy promotion.

### 2026-11-06 16:41 Asia/Shanghai — isolated coupled gradient diagnosis

All supplementary J audits completed. At16x validation static endpoints, joint
normalized MSE2.09700e-4 to2.44318e-5 and odd component1.69730e-4 to1.22795e-6
improve, but root linear common MSE2.13303e-5 to5.98521e-5 and angular common
2.00132e-5 to6.45821e-5 worsen. Fit shows the same tradeoff. See J endpoint
_decomposition_summary.json and original static_endpoint_bias/audit.json.
Frozen fit-only gradient audit `static_coupled_gradient_v1/results/audit.json`
completed16 batches on parent/control/candidate with8 finite network gradients,
source weights restored bitexact every batch, unchanged checkpoint hashes and
zero source optimizer updates. Joint-response vs root-common gradient conflicts
(linear/angular): parent16/1, control0/0, candidate12/12 of16. Isolated saved-Adam
root-only/joint-only/combined-response probes all increase linear/angular common
bias on16/16 batches for all three models. These probes omit the full objective
and retain its historical moments; they do not establish historical causality or
justify resetting Adam. Next inspect actual complete-objective directions before
choosing a new training intervention. No acceptance or policy promotion.

### 2026-11-06 16:43 Asia/Shanghai — full-objective frozen direction audit

Started `static_coupled_full_direction_v1` in tmux static_coupled_full. The
wrapper executes the unchanged actual coupled trainer AST through all objectives
and point-count checks, stopping before backward/real Adam/export. For each of
parent/control/candidate,16 fixed new fit batches compare full_with_joint and
full_without_joint using shadow copies of the saved Adam state and original
learning rate/clipping. Every batch verifies source model and full optimizer
unchanged; endpoint common errors are fit probes outside training point counts.
This is a frozen local diagnostic, not historical1000-update reconstruction or
policy acceptance. Source and checkpoint hashes are included in final reports.

### 2026-11-06 16:47 Asia/Shanghai — complete directions and common-bias probe

Full-objective audits completed16 batches each, source weights and complete Adam
unchanged. With-joint vs without-joint root common-bias increases (linear/angular):
parent16/0, control2/0, candidate12/11. Candidate full-with-joint step increases
linear common bias2/16 and angular15/16 versus its own pre-step values.
`static_coupled_full_direction_v1/summary.json` and per-model audit.json retain
all outcomes; neither uniform gradient conflict nor uniform Adam failure follows.
Started `static_coupled_common_direction_v1` (tmux static_coupled_common): same
three models and16 fit batches, plus one predefined shadow direction adding
0.5MSE of four-endpoint mean normalized root-linear error and0.5MSE root-angular
error, coefficient1. Response terms retained. This directly tests common-bias
supervision on the same endpoints; no new training, coefficient search, validation
selection, optimizer reset or acceptance. Optional probe adds auditor SHA256 and
fit-input provenance; baseline directions remain unchanged. Final gates unchanged.
