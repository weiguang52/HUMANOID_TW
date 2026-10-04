# TW91 final paired-seed slip ablation

Existing feet_slide weight: control -0.2, slip -1.0; contact phase -0.5 in both.
Each variant: training seeds 42/123, 4000 additional PPO iterations from the same TW84 checkpoint.
Seed123 resumed after shutdown; final model_43593.pt retained with runtime params.
72 evaluations (6 validation clips x 3 evaluation seeds x 4 policies), 24 videos verified.
12 walking videos included, including failed clips. Original NPY videos: ../tw98_reference_videos.
Robot videos: 50fps, 999 frames. Original NPY: 20fps; NOT phase-synchronized.

Standing: aggregate slip about -33%, body error -25%, 2-8Hz actual-joint RMS -43%.
Walking: slip about -3%, force-threshold switching -22%, joint RMS about +8%.
Clean walking replays: control 8/18, slip 7/18. Standing both 18/18.
000039 clean: control 3/6, slip 1/6; 000139 both 0/6; 000211 5/6 vs 6/6.
Conclusion: retain standing candidate; walking improvement NOT accepted. Baseline preserved.
Band RMS includes intentional movement; force-threshold switches do not prove foot flight.
These clips are validation, not independent test. Body-average error is not hand-expression fidelity.
Some recovery videos used performance rendering after GPU rendering failure; physics unchanged.
