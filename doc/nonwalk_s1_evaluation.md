# Non-walking diagnostic replay with the S1 controller

210 HumanML3D clips selected by captions across wave, punch, squat, bend,
dance, kick and jump categories. None passed the existing strict retarget gate.
Selected low-limit-occupancy examples were regenerated with quality_pass=false
in an isolated diagnostic directory. All four passed simulator FK readback checks.
They were not added to training; training admission rules remain unchanged.

Same final S1 model_9999.pt, physical plane/contact/PD control, fixed clip start,
no retraining. Diagnostic replay explicitly allows rejected references via a
process-scoped P9_CUSTOM_ALLOW_UNSAFE_MOTIONS=1. Frame-zero resets follow errors.

| Type | Clip | Steps | Anchor height term | End-effector term | Motion end |
| --- | --- | ---: | ---: | ---: | ---: |
| Right-hand wave | 000113 | 1000 | 8 | 12 | 0 |
| Slight squat, hands on thighs | 000672 | 1832 | 35 | 49 | 0 |
| Single kick | 000433 | 1000 | 0 | 4 | 0 |
| Salsa steps | 000653 | 2155 | 1 | 24 | 0 |

Termination term counts may overlap and are not unique episode counts.
None completed an entire reference clip. Errors triggered automatic resets;
this does not mean every reset was a visible fall. Reference limitations confound
policy generalization: wave/squat have neck-limit dwell; dance/kick also have
ankle/knee limit issues. These trials must not be labeled successful validation.

Artifacts stay remote at
`/root/gpufree-data/datasets/practice9/nonwalk_s1_v2/diagnostic/`:
4 videos (20-43 seconds), per-clip evaluation.json, evaluation_summary.json,
source captions/retarget quality, manifests with quality_pass=false and FK logs.
Original 4-clip walking training manifest and policy weights are unchanged.
