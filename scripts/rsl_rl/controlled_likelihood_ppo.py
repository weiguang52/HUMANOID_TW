# AnchoredPPO.update copied with only the two audited score/entropy replacements.
# Original attribution is preserved in anchored_ppo.py (BSD-3-Clause).
import hashlib
import inspect
import torch
import torch.nn as nn
from anchored_ppo import AnchoredPPO, anchor_mse
from rsl_rl.algorithms import PPO
from rsl_rl.modules import GaussianDistribution
from controlled_action_likelihood import ACTIVE, projected_statistics, selected_indices
ANCHOR_UPDATE_SHA256 = '77132b8ab510075f14aa9b1c684abd266f3414b551c190fd2e19ef82d6ff2242'
PPO_ACT_SHA256 = '930a814dd21dd2003e6694fd8faeb1d91aecab7ae468741276b44860b6de5d47'

class ControlledLikelihoodPPO(AnchoredPPO):
    def controlled_statistics(self, obs, actions):
        params = self.actor.output_distribution_params
        if len(params) != 2:
            raise ValueError('Expected full Gaussian mean/std')
        return projected_statistics(params[0], params[1], actions,
                                    obs[ACTIVE], self.controlled_action_indices)

    def act(self, obs):
        actions = super().act(obs)
        log_prob, _ = self.controlled_statistics(obs, actions)
        self.transition.actions_log_prob = log_prob.detach()
        return actions

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
            actions_log_prob, controlled_entropy = self.controlled_statistics(batch.observations, batch.actions)
            values = self.critic(batch.observations, masks=batch.masks, hidden_state=batch.hidden_states[1])
            # Note: We only keep the distribution parameters and entropy of the first augmentation (the original one)
            distribution_params = tuple(p[:original_batch_size] for p in self.actor.output_distribution_params)
            entropy = controlled_entropy[:original_batch_size]

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


def install_controlled_likelihood(alg, contract):
    if type(alg) is not AnchoredPPO or alg.actor.is_recurrent or alg.critic.is_recurrent:
        raise ValueError('Requires already installed feed-forward AnchoredPPO')
    if alg.symmetry or alg.rnd or alg.is_multi_gpu:
        raise ValueError('Not audited for symmetry, RND or distributed PPO')
    if type(alg.actor.distribution) is not GaussianDistribution or alg.actor.distribution.output_dim != 30:
        raise ValueError('Expected raw30 Gaussian policy')
    if hashlib.sha256(inspect.getsource(AnchoredPPO.update).encode()).hexdigest() != ANCHOR_UPDATE_SHA256:
        raise RuntimeError('Anchored update changed; re-audit controlled extension')
    if hashlib.sha256(inspect.getsource(PPO.act).encode()).hexdigest() != PPO_ACT_SHA256:
        raise RuntimeError('Upstream act changed; re-audit raw sampling parity')
    indices = selected_indices(contract)
    stored = alg.storage.observations
    if ACTIVE not in stored or stored[ACTIVE].dtype != torch.float32 or stored[ACTIVE].shape[-1] != 1:
        raise ValueError('Activity must be in rollout storage before runner construction')
    if ACTIVE in alg.actor.obs_groups or ACTIVE in alg.critic.obs_groups:
        raise ValueError('Activity is training-only and cannot be an NN input')
    alg.controlled_action_indices = indices
    alg.__class__ = ControlledLikelihoodPPO
    return {'enabled': True, 'contract': contract,
            'anchor_update_sha256': ANCHOR_UPDATE_SHA256,
            'ppo_act_sha256': PPO_ACT_SHA256,
            'changes': ['stored/current score projected to active17', 'entropy projected to active17'],
            'full30_KL_unchanged': True, 'raw30_sampling_unchanged': True}
