"""Bounded contact-loss cost only for stationary, known double-support references."""
import torch


def standing_contact_cost(force_z, reference_xy, reference_yaw_rate, standing, contact, known):
    speed = torch.linalg.vector_norm(reference_xy, dim=-1)
    speed_gate = ((.025-speed)/.020).clamp(0.,1.)
    yaw_gate = ((.3-reference_yaw_rate.abs())/.2).clamp(0.,1.)
    eligible = standing.bool() & contact.bool().all(-1) & known.bool().all(-1)
    # Evaluate every stored physics sample; a brief contact cannot mask dropout.
    loss = (1.-force_z.clamp(0.,1.)).square().mean(dim=(1,2))
    return loss*speed_gate*yaw_gate*eligible


def stationary_contact_loss(env, command_name, sensor_cfg):
    from .expression import reference_contacts
    from .stability_tracking import standing_mask
    command=env.command_manager.get_term(command_name)
    contact,known=reference_contacts(command)
    index=command.cfg.body_names.index('base_link')
    force=env.scene.sensors[sensor_cfg.name].data.net_forces_w_history[:,:,sensor_cfg.body_ids,2]
    return standing_contact_cost(force,command.body_lin_vel_w[:,index,:2],
        command.body_ang_vel_w[:,index,2],standing_mask(command),contact,known)
