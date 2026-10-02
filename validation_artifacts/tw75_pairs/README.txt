HumanML3D source NPY (left) vs corrected URDF kinematic reference replay (right).
Five full clips, matched phase, retarget time scaling shown on screen. Fixed cameras; separately fitted human/robot scale.
No policy, no gravity/contact dynamics, no foot snapping or pose correction. Ground is visual only. This compares retarget pose, not controller stability.
Source and reference paths and quality flags are in manifest.json; all reference files from pilot_training_geometry_aligned. No new retarget tuning for these videos.
Original URDF unchanged; mesh copies converted to OBJ on server for renderer. MuJoCo vs Pinocchio paired body position check over 5 seed42 random poses gave max 8.90e-8 m discrepancy (MJCF serialization precision).
All videos H264 yuv420p 1280x736 25fps; frame counts/hashes verified.
