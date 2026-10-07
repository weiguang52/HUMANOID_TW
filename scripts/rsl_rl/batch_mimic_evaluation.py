"""Single-seed, multi-clip headless replay with one simulator startup."""
import json
from pathlib import Path
import torch
from mimic_telemetry import MimicTelemetry
from control_runtime import snapshot

@torch.inference_mode()
def run(env,policy,policy_nn,args,checkpoint,cfg):
    if env.unwrapped.num_envs!=1:raise ValueError('Batch telemetry requires one environment')
    jobs=json.loads(Path(args.evaluation_batch).read_text())
    if not isinstance(jobs,list) or not jobs:raise ValueError('Empty batch')
    if any(job.get('controller','policy') not in ('policy','reference_pd') for job in jobs):raise ValueError('Unknown controller')
    if any(job.get('controller','policy')!='policy' for job in jobs) and not args.standing_diagnostic:
        raise ValueError('Non-policy controllers require explicit diagnostic mode')
    command=env.unwrapped.command_manager.get_term('motion')
    for job in jobs:
        motion_id=int(job['motion_id']);steps=int(job['steps'])
        if steps<=0 or motion_id<0 or motion_id>=command.motion.num_motions:
            raise ValueError('Invalid batch motion/steps')
    resetter=policy if hasattr(policy,'reset') else policy_nn
    for job in jobs:
        output=Path(job['output']);output.parent.mkdir(parents=True,exist_ok=True)
        if output.exists():raise FileExistsError(output)
        command.set_evaluation_motion_ids([int(job['motion_id'])])
        env.seed(args.seed)
        obs,_=env.reset()
        if resetter is not None:resetter.reset(torch.ones(1,device=env.unwrapped.device,dtype=torch.bool))
        report={'steps':0,'termination_counts':{},'motion_metrics':{}}
        telemetry=MimicTelemetry(env.unwrapped,output.with_suffix('.npz'))
        writer=None
        if args.standing_diagnostic and args.enable_cameras:
            import imageio.v2 as imageio
            import numpy as np
            origin=env.unwrapped.scene['robot'].data.root_pos_w[0].cpu().numpy()
            env.unwrapped.sim.set_camera_view(eye=origin+np.array([0,-1.2,.15]),target=origin+np.array([0,0,-.03]))
            writer=imageio.get_writer(str(output.with_suffix('.mp4')),fps=25,codec='libx264',macro_block_size=None)
        with torch.inference_mode():
            for step in range(int(job['steps'])):
                if job.get("controller","policy")=="reference_pd":
                    from standing_diagnostic import reference_actions
                    actions=reference_actions(env.unwrapped,command)
                else:actions=policy(obs)
                telemetry.begin(actions)
                obs,_,dones,_=env.step(actions);telemetry.end(dones)
                report['steps']+=1
                if writer is not None and step%2==0:writer.append_data(env.unwrapped.render())
                manager=env.unwrapped.termination_manager
                for name in manager.active_terms:
                    count=int(manager.get_term(name).sum().item())
                    report['termination_counts'][name]=report['termination_counts'].get(name,0)+count
                for name,value in command.metrics.items():
                    if name.startswith('error_'):
                        report['motion_metrics'][name]=report['motion_metrics'].get(name,0.)+float(value.mean().item())
                if resetter is not None:resetter.reset(dones)
                if args.standing_diagnostic and dones.any():break
        if writer is not None:writer.close()
        telemetry.save()
        report['motion_metrics']={k:v/report['steps'] for k,v in report['motion_metrics'].items()}
        report.update(diagnostic=bool(args.standing_diagnostic),controller=job.get('controller','policy'),physics_dt=env.unwrapped.physics_dt,checkpoint=checkpoint,motion_id=job['motion_id'],seed=args.seed,
            seconds=report['steps']*env.unwrapped.step_dt,num_envs=1,
            control_runtime=snapshot(cfg),batch_protocol='single_startup_per_seed_reset_per_clip_v1',
            upper_body_termination=bool(args.evaluation_upper_body_termination),
            note='Startup physical randomization shared across clips within seed. Compare only matched batch protocol.')
        output.write_text(json.dumps(report,indent=2)+'\n')
        print('BATCH_COMPLETED',output,flush=True)

    Path(str(args.evaluation_batch)+'.done').write_text('completed\n')
