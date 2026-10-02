"""Joint-order-aligned velocity tracking cost (rad/s squared)."""
def motion_joint_velocity_error_l2(env, command_name: str):
    command = env.command_manager.get_term(command_name)
    # MotionCommand exposes both tensors in the same reference joint order.
    # Penalize excess motion relative to reference, not intentional movement.
    return (command.robot_joint_vel - command.joint_vel).square().mean(dim=-1)
