# Frozen-policy action anchoring

TW-RL / Notion004. User authorized continued optimization.
Contact ablation rejected: turning1/3; static177.92mm; upper terminates9.52s.
Unmodified matched continuation also loses static stability(5.16s termination).
Hypothesis: constrain departure from original27995 behavior while allowing PPO updates.

Candidate retains original HeadLocked rewards (no failed contact-loss addition),
adds coefficient1.0 times mean squared deterministic action-mean difference to a
frozen original27995 actor on current PPO minibatch observations. This is mean-action
anchoring, not full distribution KL or a claim of reproducing PPD.
Teacher normalization/weights are frozen, excluded from optimizer; no teacher action
sampling/RNG draws. Explicit teacher checkpoint allows student recovery without
silently replacing the teacher. Audit stores teacher SHA256 and coefficient.
PPO.update derivative preserves BSD-3-Clause copyright and full license; runtime
source hash guard rejects unreviewed upstream changes. Unsupported recurrent/RND/
symmetry/distributed modes fail closed. Default coefficient0 uses untouched PPO.

CPU checks: exact actor/critic parameter parity for copied update with zero anchor;
anchor-only real optimizer reduces mean-action error; teacher parameters/gradients
unchanged; deterministic teacher consumes no RNG; invalid coefficients rejected.
Explicit original teacher state is retained even if student parameters differ.
Actual simulator smoke required before formal execution.

Predeclared protocol: original27995,seed42,1024env,500iterations, matched baseline
and anchor1.0. Both restore optimizer scheduler lr. Existing sampler/RNG resume
limitation remains. Do not select checkpoints/weights by fixed test performance.
All prior numerical acceptance gates remain, including all-group nonregression,
static<109.15mm and baseline, upper20s and <=35.06mm and baseline, standing6/6,
walking>=23/24,turning3/3. Inspect full colored views and tracking error; independent
validation needed before broad improvement claims. No automatic promotion.

Primary inspiration: https://arxiv.org/abs/2407.15134 (PPO plus policy distillation),
https://github.com/NVlabs/ProtoMotions/tree/main/protomotions/agents/fine_tuning
(frozen pretrained module infrastructure). Neither establishes efficacy here.
State: /root/gpufree-data/datasets/practice9/anchor_ablation_20261025.
Current contact experiment video workers retain priority; do not interrupt them.
