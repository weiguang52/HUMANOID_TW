Third-party implementation, not author official code.
Upstream: https://github.com/YHFone/supermimic-mjlab
Pinned commit: 9aab0a588c6941bf9862a2d444b711fd73e2b832
Original files verified against Git blob hashes. Local changes are tracked by this repository. Preserve Apache-2.0 LICENSE.

Physics-v3 includes an Apache-licensed copy of the installed MuJoCo Warp
`_src/forward.py::_advance` integration routine, with its Newton Developers
copyright/license header retained in `config/tw/velocity_limits.py` and its
original source SHA256 checked at runtime. Our inserted mass-metric speed
projection is in `velocity_projection.py`; it is a TW-specific operator-split
constraint implementation, not an author-provided SuperMimic component or a
claim of identical PhysX dynamics. CPU/GPU inertia, momentum, energy, static
and forced-impulse diagnostics are documented in the server30416 report.
