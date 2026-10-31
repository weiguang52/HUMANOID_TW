# Fixed policy-start pilot and native foot contacts

The fixed100-update seed42 policy-start pilot did not improve standing: trained68steps/1.36s original ee_body_pos termination, previous zero-readout control84steps/1.68s. Both nominal PD controls complete1000steps/20s. No policy promotion.

Same original1000 parent, same offline-calibrated dynamics, zero readout initialization, half-fit WM batch512, original reset/exploration/sensor noise, velocity projection, reward and6e-6 schedule. Only policy-start batch512 changed to256online+256fit. Fit8files/3217past-only12-frame windows, no validation reads. Final1100 SHA256 c37b96f66cbf72c3bb4dd5ed61f6e63946481d942817dde37ec0899ee01a5880,193finite tensors, real CUDA trainer restore/source/RNG replay passed.

Against previous control at common actual1.36s: maximum XY45.87→196.39mm, root tilt10.08→47.55deg. Actual action max0.10563 and9.56%components outside fit ranges (previous full episode0.15889/28.53%); smaller commands did not imply better balance. Initial offline18model nonregression passes; previous control4/18fail. H32root-position p95 median55.01mm vs52.55mm. Do not interpret model gates as control success.

All4videos H264/50fps,68or1000actual frames, full decode verified. Front/side trained sampled0/.4/.8/1.34s show late marked lean; zero endpoints upright. Not all-frame visual review or ground-impact evidence.

State root: /root/gpufree-data/datasets/practice9/supermimic_tw_20261031/longterm_diagnostics
Video index: standing_policy_anchor_v1/evaluation/verified_index.html

## Native contacts

Frozen previous/current final policies and PD replayed with the same original evaluation conditions. Each control step copies actual GPU MuJoCo data and reads mj_contactForce without CPU forward/step. Extracted state before/after each read exactly unchanged. Replays preserve all original durations/termination reasons, but are not bitwise identical: maximum whole-state difference up to2.35e-4 (mixed position/velocity units), action difference up to1.85e-6; contexts identical. These numerical differences are reported, not hidden or called exact replay.

Raw diagnostic: learned_policy_native_contacts_v1. The initial ground classifier incorrectly assumed geom_bodyid0; actual plane is named terrain. Raw per-contact positions/forces were retained. Original derived ground flags/counts are invalid. The corrected helper asserts terrain is a plane in the real runtime model; a one-step CUDA smoke passes. contact_audit.json and all four full contact_series.json recompute aggregates from actual terrain contact records and preserve original raw files.

At0.02s both learned cases have4loaded points per foot spanning49x22.85mm. At0.2s previous control left2/right1points with root tilt0.95deg; current left2/right4 with root tilt0.77deg. At0.4s left foot1point in both. At termination both feet1point and sole tilts about47–49deg. PD at20s retains4points each, full sole span, sole tilt about0.42deg. This does not support an inherently point-only sole at nominal pose; contact degrades early under policy commands. Timing is descriptive, not proof of unique causality or hardware correctness.

Next: quantify early sole orientation/command changes and model sensitivity to support-preserving actions before selecting another single-variable intervention. No collision/mass changes or new policy training based on these findings alone. Full7motion original-duration acceptance and3independent training seeds remain unmet.
