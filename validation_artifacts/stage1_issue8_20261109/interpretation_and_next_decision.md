# Interpretation and next decision

Both frozen acceptance conditions fail in both training seeds. Stop this replacement-only variant as a demonstrated robustness fix; do not extend its training budget or promote production defaults.

The effects are heterogeneous, not uniformly negative: seed1 is worse than the matching issue6 control at all three speeds (66/269/602 vs41/121/473 falls per3072). Seed2 is worse at0.3m/s but improves at0.5/0.8m/s (43/64/398 vs25/230/650). Its two low/mid-speed upper95% bounds are still1.880%/2.651%; seed1 upper bounds are2.724%/9.808%, all above1%.

The absolute20ms-vs0ms gap widens in both seeds:6.097->13.190percentage points and3.157->6.288percentage points. This does not mean both delay strata always worsen. Seed2 improves from8.256%to2.365%at0ms and11.413%to8.653%at20ms; the larger0ms gain widens the gap. Seed1 improves0ms slightly (3.870%to3.634%) but worsens20ms (9.967%to16.824%). Do not pool training seeds into a single claimed benefit.

A structural hypothesis for a future study, not a proven failure mechanism: with one-step delay, after requesta_t the applied action and new observation area_(t-1), while the queued requesta_t will act next. The original observation exposeda_t; replacement drops that exact pending-command channel. The tested sequence requests0.1,0.3,-0.2and observes applied0,0.1,0.3. Existing tests prove timing/buffer correctness, not that discarding pending information is beneficial. Actor feature-meaning migration from a requested-action parent is another unresolved factor.

If a future issue continues this research, test retaining both requested/pending and applied-target histories together instead of replacing one with the other, with a separately specified architecture/checkpoint migration and controlled baseline. No such change is implemented, trained or evaluated here. The present results do not prove that applied-target feedback is intrinsically useless, only that this single-parameter replacement under this fixed shared-parent budget is not a reliable fix.

Both training seeds share one robust05 seed1 parent; all confidence intervals are conditional fixed-policy evaluation intervals, not full-pipeline replication guarantees.
