# TW stage1 GitHub issues #1–4 results

SuperMimic remains stopped: historical control pilots passed 0/7 fixed motions. These stage1 velocity-control experiments do not validate SuperMimic.

## FT5 independent baseline

128 environments per command per seed, deterministic inference, 20 s. Observation noise and startup mass/friction randomization remain active; external pushes disabled. Fall rate counts trials with any non-timeout termination, not resets per step.

| Seed | Command | Failures / 128 | XY error m/s | Yaw error rad/s | Foot contact slide m/s | Joint acceleration RMS rad/s² |
|---|---|---:|---:|---:|---:|---:|
| 1 | stand | 0 | 0.02033 | 0.05010 | 0.00724 | 6.447 |
| 1 | forward_01 | 0 | 0.02695 | 0.06811 | 0.01777 | 12.831 |
| 1 | forward_02 | 0 | 0.03191 | 0.11461 | 0.03034 | 18.516 |
| 1 | forward_03 | 1 | 0.05457 | 0.18381 | 0.03706 | 20.779 |
| 1 | yaw_positive | 0 | 0.02196 | 0.27988 | 0.00939 | 8.130 |
| 1 | yaw_negative | 0 | 0.02442 | 0.29583 | 0.00662 | 5.870 |
| 2 | stand | 0 | 0.02035 | 0.05015 | 0.00717 | 6.427 |
| 2 | forward_01 | 0 | 0.02709 | 0.06821 | 0.01767 | 12.810 |
| 2 | forward_02 | 0 | 0.03155 | 0.11491 | 0.03025 | 18.502 |
| 2 | forward_03 | 0 | 0.05423 | 0.18413 | 0.03709 | 20.763 |
| 2 | yaw_positive | 0 | 0.02197 | 0.27986 | 0.00937 | 8.117 |
| 2 | yaw_negative | 0 | 0.02448 | 0.29607 | 0.00661 | 5.827 |
| 3 | stand | 0 | 0.02040 | 0.05017 | 0.00728 | 6.450 |
| 3 | forward_01 | 0 | 0.02695 | 0.06865 | 0.01798 | 12.954 |
| 3 | forward_02 | 0 | 0.03184 | 0.11624 | 0.03076 | 18.511 |
| 3 | forward_03 | 0 | 0.05442 | 0.18514 | 0.03770 | 20.884 |
| 3 | yaw_positive | 0 | 0.02191 | 0.27968 | 0.00948 | 8.148 |
| 3 | yaw_negative | 0 | 0.02447 | 0.29618 | 0.00671 | 5.871 |

Issue #1 empirical fall-rate gate (<1% for every command and seed): PASS.

Training logs previously reported roughly 0.067 XY and 0.21 yaw error. Training episode statistics and independent fixed-command trial statistics have different distributions and denominators; training termination proportions are not independent trial fall rates.

## Matched final-checkpoint comparisons

| Model | Protocol | Worst command/seed fall rate | Mean +/-0.3 yaw error | Mean old-range XY error |
|---|---|---:|---:|---:|
| ft5 | nominal | 100.00% | 0.28792 | 0.02999 |
| ft5 | challenge | 100.00% | 0.30538 | 0.05968 |
| control_s1 | nominal | 100.00% | 0.26703 | 0.04430 |
| control_s2 | nominal | 100.00% | 0.27258 | 0.04822 |
| entropy_s1 | nominal | 100.00% | 0.22039 | 0.03046 |
| entropy_s2 | nominal | 100.00% | 0.20282 | 0.02879 |
| yaw_s1 | nominal | 100.00% | 0.15677 | 0.03166 |
| yaw_s2 | nominal | 100.00% | 0.18602 | 0.02984 |
| range05 | nominal | 87.50% | 0.10545 | 0.03223 |
| range05 | challenge | 98.44% | 0.18538 | 0.06958 |
| robust05 | nominal | 91.41% | 0.15563 | 0.03618 |
| robust05 | challenge | 99.22% | 0.18357 | 0.04983 |
| range08 | nominal | 1.56% | 0.08374 | 0.04645 |
| range08 | challenge | 89.84% | 0.18319 | 0.07951 |
| robust08 | nominal | 0.78% | 0.12865 | 0.03361 |
| robust08 | challenge | 21.09% | 0.15134 | 0.04328 |

The complete per-command/per-seed records are in the JSON files; pooled means do not replace individual acceptance checks. Range/robustness comparisons use a combined randomization package, so individual randomizer effects are not identified. No wider-range configuration is automatically promoted to production defaults.

## Videos

Camera follows translation and heading. Front is along the robot heading; side is perpendicular. Both use the separately precolored exact experiment USD. Joint-to-color legend: [joint_colors.json](joint_colors.json). Stage1 has no HumanML3D source motion; skeleton-reference comparison is not applicable.

- [ft5_video_front_forward_01_front](videos/ft5_video_front_forward_01_front.mp4)
- [ft5_video_front_forward_02_front](videos/ft5_video_front_forward_02_front.mp4)
- [ft5_video_front_forward_03_front](videos/ft5_video_front_forward_03_front.mp4)
- [ft5_video_front_stand_front](videos/ft5_video_front_stand_front.mp4)
- [ft5_video_front_yaw_negative_front](videos/ft5_video_front_yaw_negative_front.mp4)
- [ft5_video_front_yaw_positive_front](videos/ft5_video_front_yaw_positive_front.mp4)
- [ft5_video_side_forward_01_side](videos/ft5_video_side_forward_01_side.mp4)
- [ft5_video_side_forward_02_side](videos/ft5_video_side_forward_02_side.mp4)
- [ft5_video_side_forward_03_side](videos/ft5_video_side_forward_03_side.mp4)
- [ft5_video_side_stand_side](videos/ft5_video_side_stand_side.mp4)
- [ft5_video_side_yaw_negative_side](videos/ft5_video_side_yaw_negative_side.mp4)
- [ft5_video_side_yaw_positive_side](videos/ft5_video_side_yaw_positive_side.mp4)
- [robust08_video_front_forward_01_front](videos/robust08_video_front_forward_01_front.mp4)
- [robust08_video_front_forward_02_front](videos/robust08_video_front_forward_02_front.mp4)
- [robust08_video_front_forward_03_front](videos/robust08_video_front_forward_03_front.mp4)
- [robust08_video_front_forward_05_front](videos/robust08_video_front_forward_05_front.mp4)
- [robust08_video_front_forward_08_front](videos/robust08_video_front_forward_08_front.mp4)
- [robust08_video_front_stand_front](videos/robust08_video_front_stand_front.mp4)
- [robust08_video_front_yaw_minus08_front](videos/robust08_video_front_yaw_minus08_front.mp4)
- [robust08_video_front_yaw_negative_front](videos/robust08_video_front_yaw_negative_front.mp4)
- [robust08_video_front_yaw_plus08_front](videos/robust08_video_front_yaw_plus08_front.mp4)
- [robust08_video_front_yaw_positive_front](videos/robust08_video_front_yaw_positive_front.mp4)
- [robust08_video_side_forward_01_side](videos/robust08_video_side_forward_01_side.mp4)
- [robust08_video_side_forward_02_side](videos/robust08_video_side_forward_02_side.mp4)
- [robust08_video_side_forward_03_side](videos/robust08_video_side_forward_03_side.mp4)
- [robust08_video_side_forward_05_side](videos/robust08_video_side_forward_05_side.mp4)
- [robust08_video_side_forward_08_side](videos/robust08_video_side_forward_08_side.mp4)
- [robust08_video_side_stand_side](videos/robust08_video_side_stand_side.mp4)
- [robust08_video_side_yaw_minus08_side](videos/robust08_video_side_yaw_minus08_side.mp4)
- [robust08_video_side_yaw_negative_side](videos/robust08_video_side_yaw_negative_side.mp4)
- [robust08_video_side_yaw_plus08_side](videos/robust08_video_side_yaw_plus08_side.mp4)
- [robust08_video_side_yaw_positive_side](videos/robust08_video_side_yaw_positive_side.mp4)

## Execution evidence

Original console logs and intermediate checkpoints remain on the remote data disk. Diagnostic mixed-environment preliminary results were isolated and excluded. Formal runs use the original Isaac Lab environment; the launcher clears inherited PYTHONPATH. Per the user decision on 2026-11-08, Notion replaces Linear synchronization. The canonical project record is https://www.notion.so/403325d8231d8172995bf19510b8d760; AGENTS.md and GitHub issue #4 record this decision.


## Interpretation and acceptance

Exploration bonus 0.001 raises the final-100-iteration policy mean std from about 0.023–0.024 (control) to 0.059 for both training seeds. This is evidence against continued std collapse in those arms; it does not by itself prove robust control. Increasing yaw reward further lowers independent turn error. Training command distributions and action noise differ from deterministic fixed-command evaluation, so their error values are not interchangeable.

| Training seed | Entropy yaw-error reduction vs control | Yaw-reward reduction vs control |
|---|---:|---:|
| 1 | 17.47% | 41.29% |
| 2 | 25.59% | 31.76% |

All entropy/yaw arms still fail the strict per-command/per-seed <1% fall gate. Historical FT5 re-inflated std to 0.3 on resume; the matched new arms deliberately preserve FT4 std and optimizer. Historical FT5 comparisons therefore include exploration-initialization differences; causal reward comparisons use the matched new arms.

## Range and robustness outcomes

| Model | Protocol | 0.3 m/s falls (seeds 1/2/3) | 0.5 m/s falls | 0.8 m/s falls |
|---|---|---|---|---|
| ft5 | nominal | 1/0/0 | 84/100/94 | 128/128/128 |
| ft5 | challenge | 116/110/114 | 128/125/125 | 128/128/128 |
| range05 | nominal | 0/0/0 | 1/0/1 | 105/112/109 |
| range05 | challenge | 110/111/109 | 118/121/116 | 126/126/125 |
| robust05 | nominal | 0/0/0 | 0/0/0 | 117/114/114 |
| robust05 | challenge | 62/59/53 | 81/85/89 | 127/123/123 |
| range08 | nominal | 0/0/0 | 0/0/0 | 0/1/2 |
| range08 | challenge | 94/90/95 | 90/89/91 | 106/104/110 |
| robust08 | nominal | 0/0/0 | 0/0/0 | 0/0/1 |
| robust08 | challenge | 1/3/1 | 11/15/7 | 27/24/27 |

Each denominator is 128 trials. Wider-range training substantially improves deterministic nominal tracking, and the combined robustness package improves the matched challenge outcomes. The final robust08 policy still has material challenge failures; wider command coverage is not a claim of perturbation acceptance. Defaults remain unchanged. The challenge protocol combines startup friction/mass changes, 0/20 ms action delay, observation noise and repeated +/-0.3 m/s velocity pushes. This experiment does not isolate which individual randomizer or mechanical factor causes a failure.

## Left/right gait diagnostics

| Model | Command | Mean left contact fraction | Mean right contact fraction | Left acceleration RMS | Right acceleration RMS |
|---|---|---:|---:|---:|---:|
| ft5 | forward_02 | 0.49135 | 0.54839 | 19.35767 | 27.52016 |
| ft5 | forward_03 | 0.49162 | 0.52296 | 21.24655 | 31.02321 |
| control_s1 | forward_02 | 0.42675 | 0.61760 | 18.26473 | 18.98427 |
| control_s1 | forward_03 | 0.42187 | 0.61032 | 20.09615 | 20.80045 |
| entropy_s1 | forward_02 | 0.46080 | 0.56406 | 21.81325 | 22.25232 |
| entropy_s1 | forward_03 | 0.48492 | 0.55638 | 24.83717 | 23.44171 |
| yaw_s1 | forward_02 | 0.47828 | 0.54281 | 21.00753 | 21.18288 |
| yaw_s1 | forward_03 | 0.49668 | 0.53572 | 24.17294 | 22.28979 |
| robust08 | forward_02 | 0.53115 | 0.49070 | 23.91978 | 26.19949 |
| robust08 | forward_03 | 0.52182 | 0.49247 | 25.38208 | 26.92245 |

Left/right contact duty and acceleration differences identify observable gait asymmetry; they do not establish a hardware or reward-causality diagnosis. Turn commands naturally require asymmetric motion. Full per-foot height, root-height, contact-slide and joint-acceleration metrics remain in the JSON records.

## Visual validation

32 videos passed H.264, 960x720, 50fps and 1000-frame checks. Five timestamps per video (160 sampled frames) passed nonblack and colored-body framing checks. A 3x3 opening removes isolated render noise from the colored-pixel mask; framing checks do not replace visual inspection. Contact sheets of all baseline/final commands were inspected for colors, full body and attached visible parts. Camera follows translation and heading. No runtime material binding is used. Initial black-frame videos remain excluded in the remote preliminary directory.
