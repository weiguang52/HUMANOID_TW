# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# PPO.update copied from the installed rsl_rl implementation; only anchor loss added.
import copy
import hashlib
import inspect
import math
import torch
import torch.nn as nn
from rsl_rl.algorithms import PPO

UPSTREAM_UPDATE_SHA256 = '507b4a88853dc6eda17c7c2e79bd081430c97ec8ca2e2efc24572729c94907b0'


HIP_ANCHOR_NAMES = ('left_hip_pitch_joint', 'left_hip_roll_joint',
                    'right_hip_pitch_joint', 'right_hip_roll_joint')

def hip_anchor_weights(names, multiplier):
    if not math.isfinite(multiplier) or multiplier < 1:
        raise ValueError('Hip multiplier must be finite and at least one')
    if len(set(names)) != len(names) or any(names.count(n) != 1 for n in HIP_ANCHOR_NAMES):
        raise ValueError('Expected unique action names including all four hips')
    return [float(multiplier) if n in HIP_ANCHOR_NAMES else 1.0 for n in names]

ARM_ANCHOR_NAMES = tuple(side + '_' + joint for side in ('left', 'right')
    for joint in ('shoulder_pitch_joint', 'shoulder_roll_joint',
                  'shoulder_yaw_joint', 'elbow_pitch_joint', 'wrist_pitch'))

def arm_anchor_weights(names, multiplier, existing=None):
    if not math.isfinite(multiplier) or multiplier < 1:
        raise ValueError('Arm multiplier must be finite and at least one')
    if len(names) != 30 or len(set(names)) != len(names) or any(names.count(n) != 1 for n in ARM_ANCHOR_NAMES):
        raise ValueError('Expected unique30 action names including all10 arm joints')
    weights = [1.0] * len(names) if existing is None else list(existing)
    if len(weights) != len(names) or any(not math.isfinite(w) or w <= 0 for w in weights):
        raise ValueError('Existing anchor weights must be finite positive and match actions')
    return [w * float(multiplier) if n in ARM_ANCHOR_NAMES else w for n, w in zip(names, weights)]

def validate_arm_resume(saved, multiplier):
    if not math.isfinite(multiplier) or multiplier < 1:
        raise ValueError('Arm multiplier must be finite and at least one')
    previous = saved.get('arm_multiplier', 1.0)
    if previous != 1.0 and previous != multiplier:
        raise ValueError('Resume must explicitly preserve saved arm anchor multiplier')

def candidate_anchor_weights(names, multiplier, existing=None):
    from candidate_action_scope import make_scope_contract
    if isinstance(multiplier, bool) or not isinstance(multiplier, (int, float)) or not math.isfinite(multiplier) or multiplier < 1:
        raise ValueError('Candidate multiplier must be finite numeric and at least one')
    scope = make_scope_contract(names, 'lower_body_waist')
    weights = [1.0] * 30 if existing is None else [float(v) for v in existing]
    if len(weights) != 30 or any(not math.isfinite(w) or w <= 0 for w in weights):
        raise ValueError('Existing candidate anchor weights must be finite positive30')
    selected = set(scope['candidate_joint_names'])
    return [w * float(multiplier) if n in selected else w for n, w in zip(scope['action_joint_names'], weights)]


def validate_candidate_resume(saved, multiplier, allow_frozen_initialization=False):
    if not isinstance(saved, dict):
        raise ValueError('Expected saved anchor metadata')
    if isinstance(multiplier, bool) or not isinstance(multiplier, (int, float)) or not math.isfinite(multiplier) or multiplier < 1:
        raise ValueError('Candidate multiplier must be finite numeric and at least one')
    previous = saved.get('candidate_multiplier', 1.0)
    if isinstance(previous, bool) or not isinstance(previous, (int, float)) or not math.isfinite(previous) or previous < 1:
        raise ValueError('Invalid saved candidate multiplier')
    if previous != 1:
        if saved.get('candidate_scope') != 'lower_body_waist':
            raise ValueError('Saved candidate anchor requires exact action partition')
        names = saved.get('action_joint_names')
        if saved.get('action_weights') != candidate_anchor_weights(names, previous):
            raise ValueError('Saved candidate anchor weights/order inconsistent')
    if previous != multiplier and not allow_frozen_initialization:
        raise ValueError('Resume must preserve candidate anchor multiplier; only exact frozen teacher may initialize a new value')


def anchor_mse(student, teacher, weights=None):
    if student.shape != teacher.shape:
        raise ValueError('Student/teacher action dimensions differ')
    error = (student-teacher).square()
    if weights is not None:
        if weights.ndim != 1 or weights.numel() != student.shape[-1]:
            raise ValueError('Action weights do not match policy action dimensions')
        error = error * weights
    return error.mean()

class AnchoredPPO(PPO):
    def update(self) -> dict[str, float]:
        """Run optimization epochs over stored batches and return mean losses."""
        mean_anchor_loss = 0.0
        mean_temporal_loss = 0.0
        mean_value_loss = 0
        mean_surrogate_loss = 0
        mean_entropy = 0
        # RND loss
        mean_rnd_loss = 0 if self.rnd else None
        # Symmetry loss
        mean_symmetry_loss = 0 if self.symmetry else None

        # Get mini batch generator
        if self.actor.is_recurrent or self.critic.is_recurrent:
            generator = self.storage.recurrent_mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        else:
            generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)

        # Iterate over batches
        for batch in generator:
            original_batch_size = batch.observations.batch_size[0]

            # Check if we should normalize advantages per mini batch
            if self.normalize_advantage_per_mini_batch:
                with torch.no_grad():
                    batch.advantages = (batch.advantages - batch.advantages.mean()) / (batch.advantages.std() + 1e-8)  # type: ignore

            # Perform symmetric augmentation
            if self.symmetry and self.symmetry["use_data_augmentation"]:
                # Augmentation using symmetry
                data_augmentation_func = self.symmetry["data_augmentation_func"]
                # Returned shape: [batch_size * num_aug, ...]
                batch.observations, batch.actions = data_augmentation_func(
                    env=self.symmetry["_env"],
                    obs=batch.observations,
                    actions=batch.actions,
                )
                # Compute number of augmentations per sample
                num_aug = int(batch.observations.batch_size[0] / original_batch_size)
                # Repeat the rest of the batch
                batch.old_actions_log_prob = batch.old_actions_log_prob.repeat(num_aug, 1)
                batch.values = batch.values.repeat(num_aug, 1)
                batch.advantages = batch.advantages.repeat(num_aug, 1)
                batch.returns = batch.returns.repeat(num_aug, 1)

            # Recompute actions log prob and entropy for current batch of transitions
            # Note: We need to do this because we updated the policy with the new parameters
            self.actor(
                batch.observations,
                masks=batch.masks,
                hidden_state=batch.hidden_states[0],
                stochastic_output=True,
            )
            actions_log_prob = self.actor.get_output_log_prob(batch.actions)  # type: ignore
            values = self.critic(batch.observations, masks=batch.masks, hidden_state=batch.hidden_states[1])
            # Note: We only keep the distribution parameters and entropy of the first augmentation (the original one)
            distribution_params = tuple(p[:original_batch_size] for p in self.actor.output_distribution_params)
            entropy = self.actor.output_entropy[:original_batch_size]

            # Compute KL divergence and adapt the learning rate
            if self.desired_kl is not None and self.schedule == "adaptive":
                with torch.inference_mode():
                    kl = self.actor.get_kl_divergence(batch.old_distribution_params, distribution_params)  # type: ignore
                    kl_mean = torch.mean(kl)

                    # Reduce the KL divergence across all GPUs
                    if self.is_multi_gpu:
                        torch.distributed.all_reduce(kl_mean, op=torch.distributed.ReduceOp.SUM)
                        kl_mean /= self.gpu_world_size

                    # Update the learning rate only on the main process
                    if self.gpu_global_rank == 0:
                        if kl_mean > self.desired_kl * 2.0:
                            self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                        elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                            self.learning_rate = min(1e-2, self.learning_rate * 1.5)

                    # Update the learning rate for all GPUs
                    if self.is_multi_gpu:
                        lr_tensor = torch.tensor(self.learning_rate, device=self.device)
                        torch.distributed.broadcast(lr_tensor, src=0)
                        self.learning_rate = lr_tensor.item()

                    # Update the learning rate for all parameter groups
                    for param_group in self.optimizer.param_groups:
                        param_group["lr"] = self.learning_rate

            # Surrogate loss
            ratio = torch.exp(actions_log_prob - torch.squeeze(batch.old_actions_log_prob))  # type: ignore
            surrogate = -torch.squeeze(batch.advantages) * ratio  # type: ignore
            surrogate_clipped = -torch.squeeze(batch.advantages) * torch.clamp(  # type: ignore
                ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
            )
            surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

            # Value function loss
            if self.use_clipped_value_loss:
                value_clipped = batch.values + (values - batch.values).clamp(-self.clip_param, self.clip_param)
                value_losses = (values - batch.returns).pow(2)
                value_losses_clipped = (value_clipped - batch.returns).pow(2)
                value_loss = torch.max(value_losses, value_losses_clipped).mean()
            else:
                value_loss = (batch.returns - values).pow(2).mean()

            loss = surrogate_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy.mean()
            with torch.no_grad():
                teacher_mean = self.anchor_teacher(batch.observations)
            anchor_loss = anchor_mse(self.actor.output_mean, teacher_mean, self.anchor_action_weights)
            loss = loss + self.anchor_coefficient*anchor_loss
            mean_anchor_loss += anchor_loss.detach().item()
            if self.anchor_temporal_coefficient:
                if self.anchor_temporal_order == 1:
                    from temporal_anchor import PREVIOUS_POLICY, PREVIOUS_VALID, temporal_anchor_mse
                    previous_obs = batch.observations.clone(False)
                    previous_obs['policy'] = batch.observations[PREVIOUS_POLICY]
                    with torch.no_grad():
                        teacher_previous = self.anchor_teacher(previous_obs)
                    student_current = self.actor.output_mean
                    student_previous = self.actor(previous_obs)
                    temporal_loss = temporal_anchor_mse(student_current, teacher_mean,
                        student_previous, teacher_previous, batch.observations[PREVIOUS_VALID])
                else:
                    from second_order_anchor import PREVIOUS, OLDER, VALID, second_order_anchor_mse
                    previous_obs = batch.observations.clone(False)
                    previous_obs['policy'] = batch.observations[PREVIOUS]
                    older_obs = batch.observations.clone(False)
                    older_obs['policy'] = batch.observations[OLDER]
                    if self.anchor_temporal_target=='teacher_relative':
                        with torch.no_grad():
                            teacher_previous = self.anchor_teacher(previous_obs)
                            teacher_older = self.anchor_teacher(older_obs)
                    student_current = self.actor.output_mean
                    if getattr(self, 'anchor_temporal_observation_source', 'policy') == 'clean_critic':
                        from clean_temporal_policy import CLEAN
                        current_obs = batch.observations.clone(False)
                        current_obs['policy'] = batch.observations[CLEAN]
                        student_current = self.actor(current_obs)
                    student_previous = self.actor(previous_obs)
                    student_older = self.actor(older_obs)
                    if self.anchor_temporal_target=='teacher_relative':
                        temporal_loss = second_order_anchor_mse(student_current, teacher_mean,
                            student_previous, teacher_previous, student_older, teacher_older,
                            batch.observations[VALID])
                    else:
                        from second_order_anchor import student_mean_curvature_mse
                        temporal_loss = student_mean_curvature_mse(student_current,
                            student_previous,student_older,batch.observations[VALID])
                loss = loss + self.anchor_temporal_coefficient*temporal_loss
                mean_temporal_loss += temporal_loss.detach().item()


            # Symmetry loss
            if self.symmetry:
                # Obtain the symmetric actions
                # Note: If we did augmentation before then we don't need to augment again
                if not self.symmetry["use_data_augmentation"]:
                    data_augmentation_func = self.symmetry["data_augmentation_func"]
                    batch.observations, _ = data_augmentation_func(
                        obs=batch.observations, actions=None, env=self.symmetry["_env"]
                    )

                # Actions predicted by the actor for symmetrically-augmented observations
                mean_actions = self.actor(batch.observations.detach().clone())

                # Compute the symmetrically augmented actions
                # Note: We are assuming the first augmentation is the original one. We do not use the batch.actions from
                # earlier since that action was sampled from the distribution. However, the symmetry loss is computed
                # using the mean of the distribution.
                action_mean_orig = mean_actions[:original_batch_size]
                _, actions_mean_symm = data_augmentation_func(
                    obs=None, actions=action_mean_orig, env=self.symmetry["_env"]
                )

                # Compute the loss
                mse_loss = torch.nn.MSELoss()
                symmetry_loss = mse_loss(
                    mean_actions[original_batch_size:], actions_mean_symm.detach()[original_batch_size:]
                )
                # Add the loss to the total loss
                if self.symmetry["use_mirror_loss"]:
                    loss += self.symmetry["mirror_loss_coeff"] * symmetry_loss
                else:
                    symmetry_loss = symmetry_loss.detach()

            # RND loss
            if self.rnd:
                # Extract the rnd_state
                with torch.no_grad():
                    rnd_state = self.rnd.get_rnd_state(batch.observations[:original_batch_size])  # type: ignore
                    rnd_state = self.rnd.state_normalizer(rnd_state)
                # Predict the embedding and the target
                predicted_embedding = self.rnd.predictor(rnd_state)
                target_embedding = self.rnd.target(rnd_state).detach()
                # Compute the loss as the mean squared error
                mseloss = torch.nn.MSELoss()
                rnd_loss = mseloss(predicted_embedding, target_embedding)

            # Compute the gradients for PPO
            self.optimizer.zero_grad()
            loss.backward()
            # Compute the gradients for RND
            if self.rnd:
                self.rnd_optimizer.zero_grad()
                rnd_loss.backward()

            # Collect gradients from all GPUs
            if self.is_multi_gpu:
                self.reduce_parameters()

            # Apply the gradients for PPO
            nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
            nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
            self.optimizer.step()
            # Apply the gradients for RND
            if self.rnd_optimizer:
                self.rnd_optimizer.step()

            # Store the losses
            mean_value_loss += value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()
            mean_entropy += entropy.mean().item()
            # RND loss
            if mean_rnd_loss is not None:
                mean_rnd_loss += rnd_loss.item()
            # Symmetry loss
            if mean_symmetry_loss is not None:
                mean_symmetry_loss += symmetry_loss.item()

        # Divide the losses by the number of updates
        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        mean_entropy /= num_updates
        if mean_rnd_loss is not None:
            mean_rnd_loss /= num_updates
        if mean_symmetry_loss is not None:
            mean_symmetry_loss /= num_updates

        # Clear the storage
        self.storage.clear()

        # Construct the loss dictionary
        loss_dict = {
            "value": mean_value_loss,
            "surrogate": mean_surrogate_loss,
            "entropy": mean_entropy,
        }
        if self.rnd:
            loss_dict["rnd"] = mean_rnd_loss
        if self.symmetry:
            loss_dict["symmetry"] = mean_symmetry_loss

        loss_dict["anchor_mse"] = mean_anchor_loss/num_updates
        if self.anchor_temporal_coefficient:
            loss_dict["anchor_temporal_mse"] = mean_temporal_loss/num_updates
        return loss_dict


def install_anchor(alg, coefficient, teacher_state=None, action_weights=None, temporal_coefficient=0.0, temporal_order=1, temporal_target="teacher_relative"):
    if not math.isfinite(coefficient) or coefficient < 0:
        raise ValueError('Anchor coefficient must be finite and nonnegative')
    if not math.isfinite(temporal_coefficient) or temporal_coefficient < 0:
        raise ValueError('Temporal coefficient must be finite and nonnegative')
    if temporal_order not in (1,2) or (temporal_order==2 and temporal_coefficient==0):
        raise ValueError('Temporal order2 requires a positive temporal coefficient')
    from temporal_target_contract import validate_target
    validate_target(temporal_coefficient,temporal_order,temporal_target)
    if coefficient == 0 and temporal_coefficient == 0:
        return {'enabled':False,'coefficient':0.0}
    if type(alg) is not PPO or alg.actor.is_recurrent or alg.critic.is_recurrent:
        raise ValueError('Audited only for feed-forward PPO')
    if alg.symmetry or alg.rnd or alg.is_multi_gpu:
        raise ValueError('Anchor extension not audited for symmetry, RND or distributed PPO')
    source=inspect.getsource(PPO.update).rstrip()+'\n'
    if hashlib.sha256(source.encode()).hexdigest()!=UPSTREAM_UPDATE_SHA256:
        raise RuntimeError('PPO.update changed; re-audit anchored update before using')
    from rsl_rl.modules import GaussianDistribution
    if type(alg.actor.distribution) is not GaussianDistribution:
        raise ValueError('Audited only for Gaussian MLP policies')
    # Cached Normal contains transient non-leaf tensor views, not model state.
    cached=alg.actor.distribution._distribution
    teacher=copy.deepcopy(alg.actor,{id(cached):None} if cached is not None else {}).eval()
    if teacher_state is not None:
        teacher.load_state_dict(teacher_state,strict=True)
    teacher.requires_grad_(False)
    teacher_ids={id(p) for p in teacher.parameters()}
    assert not any(id(p) in teacher_ids for g in alg.optimizer.param_groups for p in g['params'])
    alg.anchor_action_weights = None
    if action_weights is not None:
        weights = torch.as_tensor(action_weights, device=alg.device, dtype=torch.float32)
        if weights.ndim != 1 or not weights.numel() or not torch.isfinite(weights).all() or not (weights > 0).all():
            raise ValueError('Action weights must be a finite positive vector')
        alg.anchor_action_weights = weights.detach().clone()
    alg.anchor_temporal_coefficient=float(temporal_coefficient)
    alg.anchor_temporal_order=int(temporal_order)
    alg.anchor_temporal_target=temporal_target
    alg.anchor_teacher=teacher
    alg.anchor_coefficient=float(coefficient)
    alg.__class__=AnchoredPPO
    audit = {'enabled':True,'coefficient':coefficient,'kind':'deterministic_action_mean_mse',
             'teacher_frozen':True,'upstream_update_sha256':UPSTREAM_UPDATE_SHA256}
    if alg.anchor_action_weights is not None:
        audit['action_weights'] = alg.anchor_action_weights.cpu().tolist()
        audit['reduction'] = 'mean over batch and all action dimensions; no weight-sum normalization'
    if temporal_coefficient:
        audit['temporal_coefficient'] = float(temporal_coefficient)
        audit['temporal_kind'] = 'valid consecutive teacher-relative mean-action increments; training-only auxiliary observations'
        if temporal_order==2:
            audit['temporal_order']=2
            audit['temporal_kind']='valid three-observation teacher-relative mean-action second differences; no dt scaling'
    if temporal_target=='student_mean':
        audit['temporal_target']='student_mean'
        audit['temporal_kind']='valid three-observation student mean second differences; no teacher subtraction or dt scaling'
    return audit
