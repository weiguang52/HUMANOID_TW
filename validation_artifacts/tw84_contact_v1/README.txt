TW84 contact ablation completed, improvement NOT accepted.
Both variants: 4000 extra PPO iterations from same full-corpus seed123 checkpoint.
Contact phase cost: control 0 vs contact -0.5; other rewards/controller unchanged.
All 36 evaluation reports and 12 videos generated. Six walking videos archived here,
including failed 000139. Standing videos and raw telemetry remain on server.
Clean replay completion: original baseline15/18, control13/18, contact15/18.
Contact vs control: walking slip -9%, raw force-threshold switching +20%; standing
switching -26%. These are validation results, not independent test acceptance.
Raw switching is not equivalent to full foot flight. Band RMS includes intentional
motion; it is not a universal jitter score. Baseline was not retrained in this round.
Original HumanML contact bits retained. Confidence masking yields no trusted stance
for treadmill000039; some robot FK reference stance feet remain 1-2cm above ground.
Reference labels were not changed to improve these validation scores.
Next experiment TW91 varies EXISTING feet_slide weight -0.2 vs -1.0, not a duplicate.
Video frame sampling of000139 shows low foot clearance/forward lean; isolated frames
cannot establish high-frequency motion quality. Full perceptual review remains limited.
Camera follows horizontal translation only, tau0.5, fixed direction/height; reset cuts.
Keep final weights alongside params/control_runtime.json to preserve runtime behavior.
