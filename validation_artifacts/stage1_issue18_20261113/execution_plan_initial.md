# Issue18 prospective scope and execution plan

Base commit:8d5d18ff089e6583f5c9641c82e6e0447f894a30. Issue18 is latest by creation; no comments at initial read. This is evaluation-only: zero training updates, no reward/config/USD edits, no default promotion.

Implement deterministic strict policy projection p_sym(s)=0.5*(p(s)+Ma*p(Ts)) with issue17 TrainingMirror actor101 (including negated gait sin/cos) and action30 map. Verify involutions, all15models on the exact issue17 fixed102400states at tolerance1e-5, identity bitwise equality, immutable model/optimizer, RNG-neutral inference and actual applied last_action feedback.

Use existing15finalmodels:oldcontrol6000_s1..5,control_int_s1..5,sym_int_s1..5. Reuse ALL original-policy data without rerunning original rollouts. Preserve each cohort original clock (oldcontrol legacy floating, newgroups integer); projection is the sole intervention. Newformal jobs15models*3eval seeds=45,1024env*6commands*20s deterministic clean, same command order and reset ordinals. Formal denominator270commandrecords/276480initialtrials. Validation smoke tests are separate and never enter formal counts.

Freeze classification before first formal rollout. Training seeds are five paired statistical units, three eval seeds averaged within each. Retain reset-aware signed/absolute20s heading, failures/prefix exposure and all initial trials, xy/yaw errors, forward speed and turning commands. For forward0.5 derive integer episode phase at action time and decompose yaw increments into idx0..9 and10..19; reconstruct original cohorts from saved reset masks and reset-to-zero episode counter, with a new smoke recording validating that reconstruction. Never relabel absent full20s endpoints or drop missing seed endpoints.

Generate new sym_int projected-policy median forward0.5 full initial-cohort representative colored front/overhead saved-state videos with worldXY trail. Preserve exact reconstruction, source/weights/USD hashes, automatic and actual visual review.

Prepare complete paired tables and group/overall A/B/C interpretations under issue18. Document numerical conventions explicitly before formal work and preserve uncertainty/mixed outcomes. Recommend ONE next action only, and do not execute that future experiment. Original15model data are read-only. Do not reuse issue17 adapter directly because it writes its own provenance into the old evidence directory; add isolated issue18 adapter and outputs instead.

Storage is currently about294MB free. Before GPU work, audit genuinely unused historical intermediate weights under the user's existing explicit cleanup authorization. Preserve every archived/tracked/referenced model and all15current finals/parents. Delete only proven unused individual weight files with pre/post SHA receipts; do not copy mounted files to the local computer. All outputs remain on the remote data disk.
