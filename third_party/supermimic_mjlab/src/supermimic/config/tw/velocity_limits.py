"""TW joint-speed projection before semi-implicit position integration.

MuJoCo has no PhysX ``velocity_limit_sim`` equivalent. This explicit projection
preserves the TW cap but is not a claim of cross-engine contact equivalence.
Only registered TW simulations are affected; root velocities remain unconstrained.
"""

from __future__ import annotations

import functools
import hashlib
import inspect
import mujoco
import numpy as np
import warp as wp
from mujoco_warp._src import forward
from .contract import (
    LEG_MAJOR_JOINTS,
    FOOT_JOINTS,
    WAIST_JOINTS,
    ARM_JOINTS,
    NECK_JOINTS,
)

LIMITS = {
    name: cap
    for names, cap in (
        (LEG_MAJOR_JOINTS, 3.0),
        (FOOT_JOINTS, 3.0),
        (WAIST_JOINTS, 2.0),
        (ARM_JOINTS, 2.0),
        (NECK_JOINTS, 2.0),
    )
    for name in names
}
_REGISTERED = {}
_ORIGINAL = forward._advance
# Fail closed if the integrator's ordering changes in another runtime version.
_ADVANCE_HASH = "8771fda254aff2c717fcd432e43faeb8a7fdfe027772b57eeb851f72d0fca90e"


@wp.kernel
def _bounded_acceleration(
    dt: wp.array(dtype=float),
    velocity: wp.array2d(dtype=float),
    acceleration: wp.array2d(dtype=float),
    limits: wp.array(dtype=float),
    result: wp.array2d(dtype=float),
):
    world, dof = wp.tid()
    a = acceleration[world, dof]
    cap = limits[dof]
    if cap > 0.0:
        h = dt[world % dt.shape[0]]
        v = velocity[world, dof]
        a = (wp.clamp(v + h * a, -cap, cap) - v) / h
    result[world, dof] = a


@functools.wraps(_ORIGINAL)
def _advance_with_limits(model, data, acceleration, qvel=None):
    record = _REGISTERED.get(id(data))
    if record is None:
        return _ORIGINAL(model, data, acceleration, qvel)
    if qvel is not None:
        raise RuntimeError(
            "TW speed projection supports semi-implicit integration only"
        )
    _, limits, scratch = record
    wp.launch(
        _bounded_acceleration,
        dim=(data.nworld, model.nv),
        inputs=[model.opt.timestep, data.qvel, acceleration, limits],
        outputs=[scratch],
    )
    return _ORIGINAL(model, data, scratch)


def enable_tw_velocity_limits(env):
    """Validate names/DOFs, register per-sim arrays, and recapture CUDA graphs."""
    original_hash = hashlib.sha256(
        inspect.getsource(_ORIGINAL).strip().encode()
    ).hexdigest()
    if original_hash != _ADVANCE_HASH:
        raise RuntimeError(
            f"Unsupported MuJoCo Warp integrator source: {original_hash}"
        )
    sim = env.sim
    model = sim.mj_model
    if model.opt.integrator != mujoco.mjtIntegrator.mjINT_IMPLICITFAST:
        raise ValueError("TW speed limits require implicitfast")
    caps = np.zeros(model.nv, dtype=np.float32)
    found = set()
    for jid in range(model.njnt):
        if model.jnt_type[jid] == mujoco.mjtJoint.mjJNT_FREE:
            continue
        name = model.joint(jid).name.split("/")[-1]
        if name not in LIMITS or name in found:
            raise ValueError(f"Unexpected TW joint: {name}")
        if model.jnt_type[jid] != mujoco.mjtJoint.mjJNT_HINGE:
            raise ValueError(f"Expected hinge joint: {name}")
        caps[model.jnt_dofadr[jid]] = LIMITS[name]
        found.add(name)
    if found != set(LIMITS) or model.nv != 36:
        raise ValueError("Incomplete TW 30-DOF speed-limit contract")
    with wp.ScopedDevice(sim.wp_device):
        limits = wp.array(caps, dtype=float)
        scratch = wp.empty_like(sim.wp_data.qvel)
        _REGISTERED[id(sim.wp_data)] = (sim.wp_data, limits, scratch)
        forward._advance = _advance_with_limits
        sim.create_graph()
    return caps


def disable_tw_velocity_limits(env):
    """Release retained simulation arrays after closing a TW environment."""
    _REGISTERED.pop(id(env.sim.wp_data), None)
