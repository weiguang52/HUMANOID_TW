import json
from pathlib import Path
p=Path('validation_artifacts/tw154_review');m=json.loads((p/'metrics.json').read_text())
html='''<!doctype html><meta charset="utf-8"><title>TW169 review</title><style>body{font:18px/1.6 sans-serif;max-width:1150px;margin:30px auto}img{width:100%}pre{white-space:pre-wrap}</style><h1>TW169 / TW154 quality review</h1>
<p>Decision: retain rate as the next candidate; neither policy passes human-like gait quality acceptance.</p>
<h2>Findings</h2><p>Rate flat walking: 24/24 clean ends, but knee amplitude only 30.6% of reference and foot-link Z amplitude 33.8%. Known swing frames still have contact 42.9% of the time. Path: 23/24, 25.3%, 35.9%, 49.3% respectively.</p>
<p>Rate improves horizontal path RMSE (5.76 vs 7.30 cm) and wrist error (0.89 vs 1.64 cm). Standing wrist error improves (0.75 vs 2.34 cm), but standing path RMSE regresses (5.73 vs 3.90 cm) and net displacement error increases (3.20 vs 1.71 cm).</p>
<p>Compared with old TW151 residual on the same 8 flat clips: net displacement error improves from 27.97 to 9.74 cm, foot Z amplitude from 26.9% to 33.8%, but knee amplitude regresses from 49.3% to 30.6%. Data and training duration changed too; not a single-factor causal comparison.</p>
<p>Rate hip/knee pitch reference error is 13.92 deg RMS; target-to-actual error is only 1.73 deg. This supports a policy/target-generation and objective tradeoff issue, not simply PD failing to reach commanded angles. It does not rule out actuator limits or reference feasibility.</p>
<p>Rate left/right knee spans average 18.6/10.4 deg. Highest flat-walk actual 2-8Hz RMS: right knee 1.44 deg, right ankle yaw 1.43, right ankle pitch 1.32. This band includes intentional motion. Rapid target alternation in standing sections requires an additional 8-25Hz audit; it is not captured by this band alone.</p>
'''
html+='''<h2>Visual review and limitations</h2><p>Inspected 8 temporal contact sheets: 4 flat walks, 3 standing upper-body clips and 1 turn, 6 samples per source/path/rate. Low knee flexion and limited foot lift agree with metrics. This is sampled visual review, not exhaustive video playback. Source 20fps and robot 50fps have different durations and are NOT phase aligned.</p><p>All 84 numeric replays reviewed. Continuous camera vertical movement is zero. Horizontal motion is smoothed; resets excluded. 254 files verified against SHA manifests; 56 H264 videos passed recorded frame checks. All 8 flat clips are KIT, not cross-dataset acceptance. 5s metrics are not HumanScore. Foot-link Z is not sole clearance, contact speed includes rolling, and known-contact masks are incomplete.</p>
<h2>Next experiments, not launched in this review</h2><ol><li>Audit per-term reward scales and saturation against knee/hip, swing-height, contact errors and smoothness/acceleration costs before changing weights.</li><li>Test one non-saturating reference error term for swing-leg knee/foot tracking while preserving original contact timing and known masks. Do not impose fixed clearance or artificial symmetry.</li><li>Audit raw action to rate-limited target alternation and 8-25Hz spectra. Do not simultaneously change PD, friction, reward and limiter.</li><li>Keep mixed standing/walking training. Retain rate and old baselines; use short controlled runs before long training, then independent training seeds. Gate on amplitude, contact, path and standing drift, not survival alone.</li></ol>
<h2>Detailed first-episode metrics</h2>'''
html+='<pre>'+json.dumps(m,indent=2)+'</pre>'
for n in ['rate_000832.png','path_000832.png']:
 html+=f'<h2>{n}</h2><img src="{n}">'
for n in ['000832','001191','005309','009516','000016','000124','000346','000211']:
 html+=f'<h2>{n}</h2><img src="{n}.jpg">'
html+='<p><a href="../tw154_flat_walk/rate/index.html">Rate full video report</a> | <a href="../tw154_flat_walk/path/index.html">Path full video report</a></p>'
(p/'index.html').write_text(html)
