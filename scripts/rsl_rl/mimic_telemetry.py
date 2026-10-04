"""Opt-in, single-environment mimic diagnostics at control and physics rates."""
import json
import os
from pathlib import Path
import numpy as np


def high_frequency_fraction(values, sample_rate, valid, cutoff=8.0):
    """Welch-style power fraction; never join samples across reset boundaries."""
    window = min(256, len(values))
    if window < 32:
        return None
    spectra = []
    taper = np.hanning(window)[:, None]
    for start in range(0, len(values) - window + 1, window // 2):
        if not valid[start:start + window].all():
            continue
        part = values[start:start + window]
        part = part - part.mean(axis=0, keepdims=True)
        spectra.append(np.abs(np.fft.rfft(part * taper, axis=0)) ** 2)
    if not spectra:
        return None
    power = np.mean(spectra, axis=0).sum(axis=1)
    freq = np.fft.rfftfreq(window, 1 / sample_rate)
    total = power[freq >= .5].sum()
    return float(power[freq >= cutoff].sum() / total) if total > 1e-15 else 0.0


class MimicTelemetry:
    def __init__(self, env, path):
        if env.num_envs != 1:
            raise ValueError('Telemetry currently requires exactly one environment')
        self.env, self.path = env, Path(path)
        self.robot = env.scene['robot']
        self.command = env.command_manager.get_term('motion')
        self.index = self.command.joint_indexes
        self.feet, _ = self.robot.find_bodies(['left_foot', 'right_foot'], preserve_order=True)
        self.sensor = env.scene['contact_forces']
        self.sensor_feet = [self.sensor.body_names.index(n) for n in ['left_foot', 'right_foot']]
        self.physics, self.control, self.terminals = [], [], []
        self.expression_diagnostics = os.environ.get('P9_EXPRESSION_TELEMETRY') == '1'
        if self.expression_diagnostics:
            from unitree_rl_lab.tasks.mimic.mdp.contact_rewards import load_reference
            self.contact_labels, self.contact_known = load_reference(self.command)
            self.upper_names = ['left_upper_arm', 'left_force_arm', 'left_wrist',
                                'right_upper_arm', 'right_force_arm', 'right_wrist']
            self.upper_indexes = [self.command.cfg.body_names.index(n) for n in self.upper_names]
        self.active = False
        self.original_update = env.scene.update
        def update(dt):
            self.original_update(dt)
            if self.active and dt > 0:
                self.sample()
        env.scene.update = update

    @staticmethod
    def array(value):
        return value.detach().cpu().numpy().copy()

    def begin(self, actions):
        self.active = True
        self.control.append(dict(action=self.array(actions[0]),
            reference_frame=int(self.command.frame_indices[0].item()),
            reference_q=self.array(self.command.joint_pos[0]),
            reference_qd=self.array(self.command.joint_vel[0])))

        if self.expression_diagnostics:
            from isaaclab.utils.math import quat_apply_inverse, yaw_quat
            c = self.command
            ix = self.upper_indexes
            ref = quat_apply_inverse(yaw_quat(c.anchor_quat_w[0]).expand(len(ix), -1),
                c.body_pos_w[0, ix] - c.anchor_pos_w[0])
            actual = quat_apply_inverse(yaw_quat(c.robot_anchor_quat_w[0]).expand(len(ix), -1),
                c.robot_body_pos_w[0, ix] - c.robot_anchor_pos_w[0])
            frame = c.frame_indices[0]
            self.control[-1].update(
                upper_reference=self.array(ref), upper_actual=self.array(actual),
                reference_contact=self.array(self.contact_labels[frame]),
                reference_known=self.array(self.contact_known[frame]),
                anchor_reference=self.array(c.anchor_pos_w[0]),
                anchor_actual=self.array(c.robot_anchor_pos_w[0]))

    def sample(self):
        d = self.robot.data
        self.physics.append(dict(control_step=len(self.control)-1,
            q=self.array(d.joint_pos[0, self.index]),
            qd=self.array(d.joint_vel[0, self.index]),
            target_q=self.array(d.joint_pos_target[0, self.index]),
            torque=self.array(d.applied_torque[0, self.index]),
            force=self.array(self.sensor.data.net_forces_w[0, self.sensor_feet]),
            foot_velocity=self.array(d.body_lin_vel_w[0, self.feet]),
            foot_position=self.array(d.body_pos_w[0, self.feet]),
            foot_quaternion=self.array(d.body_quat_w[0, self.feet]),
            foot_angular_velocity=self.array(d.body_ang_vel_w[0, self.feet]),
            root_height=float(d.root_pos_w[0, 2].item())))

    def end(self, dones):
        self.active = False
        self.terminals.append(bool(dones[0].item()))

    def save(self):
        self.env.scene.update = self.original_update
        if not self.physics:
            raise ValueError('No physics telemetry samples recorded')
        arrays = {k: np.asarray([x[k] for x in self.physics]) for k in self.physics[0]}
        arrays.update({k: np.asarray([x[k] for x in self.control]) for k in self.control[0]})
        if self.expression_diagnostics:
            arrays['upper_body_names'] = np.asarray(self.upper_names)
        arrays['terminal'] = np.asarray(self.terminals)
        arrays['joint_names'] = np.asarray([self.robot.joint_names[i] for i in self.index])
        arrays['effort_limits'] = self.array(self.robot.data.joint_effort_limits[0, self.index])
        arrays['physics_dt'] = np.asarray(self.env.cfg.sim.dt)
        arrays['control_dt'] = np.asarray(self.env.step_dt)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(self.path, **arrays)
        # Exclude the terminating control step and its neighbours from derivatives.
        valid = ~arrays['terminal']
        for shift in (-1, 1):
            valid &= ~np.roll(arrays['terminal'], shift)
        valid[0] = False
        step = arrays['control_step']
        physics_valid = valid[step]
        end_indexes = np.r_[np.flatnonzero(np.diff(step)), len(step)-1]
        target = arrays['target_q'][end_indexes]
        rate = 1 / self.env.step_dt
        physics_rate = 1 / self.env.cfg.sim.dt
        pair = physics_valid[1:] & physics_valid[:-1]
        acc = np.diff(arrays['qd'], axis=0) * physics_rate
        contact = np.linalg.norm(arrays['force'], axis=-1) > 1.0
        horizontal = np.linalg.norm(arrays['foot_velocity'][..., :2], axis=-1)
        eligible = contact & physics_valid[:, None]
        report = dict(control_steps=len(self.control), physics_samples=len(self.physics),
            control_hz=rate, physics_hz=physics_rate, resets=int(arrays['terminal'].sum()),
            reference_q_high_frequency_fraction=high_frequency_fraction(arrays['reference_q'],rate,valid),
            target_q_high_frequency_fraction=high_frequency_fraction(target,rate,valid),
            actual_qd_high_frequency_fraction=high_frequency_fraction(arrays['qd'],physics_rate,physics_valid),
            acceleration_rms_by_joint=np.sqrt(np.mean(acc[pair]**2,axis=0)).tolist(),
            torque_saturation_fraction_by_joint=np.mean(np.abs(arrays['torque'][physics_valid]) >= .99*arrays['effort_limits'],axis=0).tolist(),
            foot_force_peak_n=np.linalg.norm(arrays['force'][physics_valid],axis=-1).max(axis=0).tolist(),
            contact_horizontal_speed_mean_m_s=float(horizontal[eligible].mean()) if eligible.any() else None,
            joint_names=arrays['joint_names'].tolist(),
            note='Applied torque is the actuator-reported value; not hardware torque. Reset boundaries excluded from smoothness metrics.')
        self.path.with_suffix('.summary.json').write_text(json.dumps(report,indent=2)+'\n')
