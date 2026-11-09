# Secondary concentration sensitivity, frozen before stratified outcome analysis

The primary STRICT/ENVELOPE definitions and thresholds remain unchanged.
To avoid equating a narrow primary mask failing with proof that every possible
measured rare subset fails, add a secondary five-fold leave-one-training-seed-out
concentration diagnostic using the SAME records, no new evaluation or training.
This addition is made after partial record acquisition but before any stratified
outcome analysis; it is secondary, not retroactively claimed as the initial plan.

For4500/0.3 only: (A) initial staticQ,dynamicQ,massQ,delay joint cells with20s
trial outcomes; (B) eligible push staticQ,massQ,delay,amplitudeQ,direction,support
where support merges left/right single, keeps double/flight, using1s outcomes.
For each held-out training seed fit cell risks on the OTHER FOUR seeds only.
Eligible cells have>=20training observations and at least1event. Sort descending
risk, then descending n, then lexicographic cell. Greedily include a cell only if
the UNION of exposed initial training trial IDs stays<=20% of all training trials.
For push cells budget is unique initial trials, not push count. Never rank on
held-out outcomes. Evaluate selected cells on the held-out seed; report actual
held-out trial coverage and fraction of ALL low-speed first failures captured
(push capture uses1s events, so unrelated/late failures stay in the denominator).
Aggregate all five held-out folds. An actual held-out coverage<=20% AND capture
>=80% flags secondary concentration in a measured subset. Otherwise it does not.
No p-value or independent confirmation claim: folds share training seeds and all
models share evaluation physical draws. Selected cells need independent physical
stream validation. All joint tables and primary corner results remain visible.
If primary or secondary finds concentration, the one next proposal is sampling
reweighting for that measured subset. Otherwise propose only a low-speed post-push
pitch/COM recovery reward study. Neither proposal is implemented.
