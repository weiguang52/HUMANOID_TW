from pathlib import Path
p=Path('scripts/locomotion/issue16_runtime_readback.py')
s=p.read_text()
old="raw_PhysX_link_names=link_names,raw_PhysX_DOF_names=dof_names,robot_body_names=robot.body_names,robot_joint_names=robot.joint_names,"
new=old+"action_joint_names=list(term._joint_names),"
assert old in s;s=s.replace(old,new)
p.write_text(s)
print('ACTION_ORDER_READBACK_ADDED')
