# Project instructions

## Validation reference videos

For every motion validation delivery, include the original HumanML3D NPY skeleton video alongside the robot replay. Generate all evaluated clips with scripts/practice9/render_validation_references.py and deliver its index. Keep original source fps, fixed cameras, and clearly distinguish source speed from retargeted robot reference timing. Do not present source and robot videos as synchronized unless explicitly phase-aligned. Generate artifacts remotely; do not copy mounted files to the local computer.

For project work, record tasks, progress, decisions, and results in Notion and update the corresponding Notion page status upon completion. Do not create or synchronize Linear issues; this requirement was superseded by the user on 2026-11-08. Keep GitHub issue replies aligned with the Notion record. The existing TW-RL record is https://www.notion.so/403325d8231d8172995bf19510b8d760.

## Robot evaluation video presentation (user requirement, 2026-10-24)

Every future robot evaluation video delivery must use a colored robot model and include a front-view video for every evaluated motion, in addition to the existing side view. Supply a joint-to-color legend and clear front/side labels and links in the report. An uncolored or side-only delivery is incomplete.

Load colors from a separately saved, precolored USD. Do not dynamically bind materials after simulation initialization. Derive the colored asset from the exact robot asset used by the experiment, preserving mass, inertia, collisions, joint limits and any locked joints. Do not substitute an unlocked colored asset into a head-locked experiment.

For the front view, start in front of the robot and follow its translation. State whether the camera follows heading; do not describe a fixed world-facing camera as always frontal during turns. Verify colors, full-body framing, attachment of small parts during arm movement and turning, video encoding/frame counts, and consistency with the experiment's replay configuration before delivery. Preserve the original HumanML3D reference requirements above.
