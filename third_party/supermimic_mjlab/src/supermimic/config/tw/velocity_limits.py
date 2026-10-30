# Copyright 2025 The Newton Developers
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""TW speed caps through native velocity/position integration order.

The vendored _advance_tw is MuJoCo Warp's pinned _advance with one velocity-cap
launch inserted. Preserve native acceleration buffers and apply momentum-consistent root reactions.
This is a physical-constraint adaptation, not cross-engine contact equivalence.
"""

from __future__ import annotations
import functools
import hashlib
import inspect
from typing import Optional
import mujoco
import numpy as np
import warp as wp
import torch
from .velocity_projection import project_velocity_mass_metric
from mujoco_warp._src import forward
from .contract import (
    LEG_MAJOR_JOINTS,
    FOOT_JOINTS,
    WAIST_JOINTS,
    ARM_JOINTS,
    NECK_JOINTS,
)

Model, Data = forward.Model, forward.Data
_next_activation = forward._next_activation
_next_velocity = forward._next_velocity
_next_position = forward._next_position
_next_time_builder = forward._next_time_builder
history, sleep = forward.history, forward.sleep
EnableBit, DisableBit = forward.EnableBit, forward.DisableBit
fwd_velocity = forward.fwd_velocity
PHYSICS_CONTRACT = "explicit_default_pd_mass_metric_velocity_fp64_v3"

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
_ADVANCE_HASH = "8771fda254aff2c717fcd432e43faeb8a7fdfe027772b57eeb851f72d0fca90e"


def _project_registered_velocity(model, data):
    record = _REGISTERED[id(data)]
    # Borrow the actual Warp stream; native physics kernels and Torch projection
    # must execute in the same stream. CUDA graph mode is disabled explicitly.
    with torch.cuda.stream(wp.stream_to_torch(wp.get_stream())):
        velocity = wp.to_torch(data.qvel)
        caps = record["caps"]
        if not bool((velocity[:, 6:].abs() > caps[6:]).any()):
            record["steps"] += 1
            return
        dense = record["mass"]
        values = wp.to_torch(data.M)
        dense.zero_()
        dense[:, record["rows"], record["columns"]] = values
        dense[:, record["columns"], record["rows"]] = values
        projected, stats = project_velocity_mass_metric(velocity, dense, caps)
        velocity.copy_(projected)
        record["steps"] += 1
        record["active_steps"] += 1
        for field in ("residual", "momentum_error", "energy_increase"):
            record[field] = max(record[field], stats[field])


@functools.wraps(_ORIGINAL)
def _advance_with_limits(model, data, acceleration, qvel=None):
    if id(data) not in _REGISTERED:
        return _ORIGINAL(model, data, acceleration, qvel)
    if qvel is not None:
        raise RuntimeError(
            "TW speed projection requires native semi-implicit position update"
        )
    return _advance_tw(model, data, acceleration)


def _advance_tw(m: Model, d: Data, qacc: wp.array, qvel: Optional[wp.array] = None):
    """Advance state and time given activation derivatives and acceleration."""
    # TODO(team): can we assume static timesteps?

    # advance activations
    wp.launch(
        _next_activation,
        dim=(d.nworld, m.nu),
        inputs=[
            m.opt.timestep,
            m.actuator_dyntype,
            m.actuator_actadr,
            m.actuator_actnum,
            m.actuator_actlimited,
            m.actuator_dynprm,
            m.actuator_gainprm,
            m.actuator_biasprm,
            m.actuator_actrange,
            d.act,
            d.act_dot,
            d.actuator_velocity,
            1.0,
            True,
        ],
        outputs=[d.act],
    )

    wp.launch(
        _next_velocity,
        dim=(d.nworld, m.nv),
        inputs=[m.opt.timestep, d.qvel, qacc, 1.0],
        outputs=[d.qvel],
    )

    # Apply internal velocity-limit impulses in the full physical inertia metric.
    _project_registered_velocity(m, d)

    # advance positions with qvel if given, d.qvel otherwise (semi-implicit)
    qvel_in = qvel or d.qvel

    wp.launch(
        _next_position,
        dim=(d.nworld, m.njnt),
        inputs=[
            m.opt.timestep,
            m.jnt_type,
            m.jnt_qposadr,
            m.jnt_dofadr,
            d.qpos,
            qvel_in,
            1.0,
        ],
        outputs=[d.qpos],
    )

    # advance history buffers before time advance
    history.insert_ctrl_history(m, d)

    wp.launch(
        _next_time_builder(bool(m.opt.warn_overflow)),
        dim=d.nworld,
        inputs=[
            m.opt.timestep,
            m.is_sparse,
            d.nefc,
            d.time,
            d.efc.J_rownnz,
            d.efc.J_rowadr,
            d.nworld,
            d.naconmax,
            d.njmax,
            d.njmax_nnz,
            d.nacon,
            d.ncollision,
        ],
        outputs=[d.time, d.overflow],
    )

    wp.copy(d.qacc_warmstart, d.qacc)

    sleep_enabled = bool(m.opt.enableflags & EnableBit.SLEEP) and not bool(
        m.opt.disableflags & DisableBit.ISLAND
    )
    if sleep_enabled:
        sleep.sleep(m, d)
        fwd_velocity(m, d)
        sleep.update_sleep(m, d)


def enable_tw_velocity_limits(env):
    """Register mass-metric constraints with eager Warp/Torch stream ordering."""
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
    free = [
        j for j in range(model.njnt) if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE
    ]
    if (
        len(free) != 1
        or model.jnt_dofadr[free[0]] != 0
        or model.jnt_qposadr[free[0]] != 0
    ):
        raise ValueError(
            "TW projection requires the floating root in the first six DOFs"
        )
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
        _REGISTERED[id(sim.wp_data)] = {
            "data": sim.wp_data,
            "caps": torch.tensor(caps, device=env.device),
            "rows": torch.tensor(
                sim.wp_model.M_hinit_i.numpy(), device=env.device, dtype=torch.long
            ),
            "columns": torch.tensor(
                sim.wp_model.M_colind.numpy(), device=env.device, dtype=torch.long
            ),
            "mass": torch.empty((sim.wp_data.nworld, 36, 36), device=env.device),
            "steps": 0,
            "active_steps": 0,
            "residual": 0.0,
            "momentum_error": 0.0,
            "energy_increase": 0.0,
        }
        forward._advance = _advance_with_limits
        sim.use_cuda_graph = False
    return caps


def disable_tw_velocity_limits(env):
    """Release retained simulation arrays after closing a TW environment."""
    _REGISTERED.pop(id(env.sim.wp_data), None)


def velocity_projection_stats(env):
    record = _REGISTERED[id(env.sim.wp_data)]
    return {
        key: record[key]
        for key in (
            "steps",
            "active_steps",
            "residual",
            "momentum_error",
            "energy_increase",
        )
    }
