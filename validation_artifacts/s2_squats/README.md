# Standing-to-squat diagnostic test

Latest long_velocity/model_15596.pt; physical contact + PD, seed42, observation noise disabled. Source HumanML3D000890 (squat then stand) and001240 (squat then stand). Stable side camera with fixed height and smoothed XY.

Both references preserve tw44_table_v2 and pass FK conversion, but fail retarget quality due to prolonged joint-limit proximity. Both knees reach -90 degrees. Explicit unsafe-motion flag is confined to diagnostic replay; no training manifest or joint limit is changed. These results test the current policy plus imperfect reference, not policy capability in isolation.

|Clip|First abnormal termination (s)|Termination counts|
|---|---:|---|
|000890|4.26|{'time_out': 0, 'motion_end': 0, 'anchor_pos': 0, 'anchor_ori': 0, 'ee_body_pos': 6}|
|001240|0.78|{'time_out': 0, 'motion_end': 0, 'anchor_pos': 10, 'anchor_ori': 0, 'ee_body_pos': 24}|

Neither motion completes successfully. Videos include automatic episode resets; discontinuities at reset must not be interpreted as recovery or camera jitter. first_attempt.png stops before the first termination. No sitting-on-floor capability is claimed. The model was trained on walking, stand, weight-shift and wave, not these squat motions.

Next step: build a feasible shallow-squat reference within joint limits and verified foot support, train a stand-to-shallow-squat curriculum before increasing depth. Ground sitting needs its own contact model/reward/termination audit and cannot be inferred from this test.
