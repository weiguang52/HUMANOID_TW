"""Single-seed, multi-clip headless replay with one simulator startup."""
import json
from pathlib import Path
import torch
from mimic_telemetry import MimicTelemetry
from control_runtime import snapshot

@torch.inference_mode()
def run(env,policy,policy_nn,args,checkpoint,cfg,perturbation=None):
    if env.unwrapped.num_envs!=1:raise ValueError('Batch telemetry requires one environment')
    jobs=json.loads(Path(args.evaluation_batch).read_text())
    if not isinstance(jobs,list) or not jobs:raise ValueError('Empty batch')
    from training_speed_blend import validate_checkpoint_jobs as validate_training_speed_jobs
    trained_speed_contract=validate_training_speed_jobs(checkpoint,jobs)
    from filter_evaluation_contract import validate_checkpoint_jobs
    trained_filter_contract=validate_checkpoint_jobs(checkpoint,jobs)
    if any(job.get('controller','policy') not in ('policy','reference_pd') for job in jobs):raise ValueError('Unknown controller')
    if any(job.get('controller','policy')!='policy' for job in jobs) and not args.standing_diagnostic:
        raise ValueError('Non-policy controllers require explicit diagnostic mode')
    if any('blend_release_tau' in j and 'reference_speed_teacher_export' not in j for j in jobs):
        raise ValueError('Release gate requires explicit reference speed blend')
    if perturbation is not None:perturbation.validate(jobs,args)
    command=env.unwrapped.command_manager.get_term('motion')
    import os
    damping_schedule=None
    if os.environ.get('P9_LOCOMOTION_DAMPING')=='1':
        from locomotion_damping import LocomotionDamping
        damping_schedule=LocomotionDamping(env.unwrapped)
    hip_gains=[job.get('hip_pitch_damping_gain') for job in jobs]
    if any(v is not None for v in hip_gains):
        if damping_schedule is not None:raise ValueError('Cannot combine ankle and hip damping schedules')
        if any(v is None for v in hip_gains) or len(set(hip_gains))!=1:
            raise ValueError('Hip damping gain must match across all batch jobs')
        from locomotion_damping import LocomotionDamping
        damping_schedule=LocomotionDamping(env.unwrapped,
            names=['left_hip_pitch_joint','right_hip_pitch_joint'],maximum_gain=float(hip_gains[0]))
    for job in jobs:
        motion_id=int(job['motion_id']);steps=int(job['steps'])
        if steps<=0 or motion_id<0 or motion_id>=command.motion.num_motions:
            raise ValueError('Invalid batch motion/steps')
    resetter=policy if hasattr(policy,'reset') else policy_nn
    for job in jobs:
        output=Path(job['output']);output.parent.mkdir(parents=True,exist_ok=True)
        if output.exists():raise FileExistsError(output)
        command.set_evaluation_motion_ids([int(job['motion_id'])])
        if damping_schedule is not None:damping_schedule.reset()
        env.seed(args.seed)
        obs,_=env.reset()
        if resetter is not None:resetter.reset(torch.ones(1,device=env.unwrapped.device,dtype=torch.bool))
        report={'steps':0,'termination_counts':{},'motion_metrics':{}}
        if trained_filter_contract is not None:report['training_action_filter_contract']=trained_filter_contract
        if trained_speed_contract is not None:report['training_speed_blend_contract']=trained_speed_contract
        perturbation_records=[]
        telemetry=MimicTelemetry(env.unwrapped,output.with_suffix('.npz'))
        blend=None
        if "blend_teacher_export" in job:
            from policy_action_blend import PolicyActionBlend
            scope=job.get("blend_scope","all")
            action_names=None
            if scope!="all":
                manager=env.unwrapped.action_manager
                if len(manager.active_terms)!=1:raise ValueError("Scoped blending requires one joint action term")
                term=manager.get_term(manager.active_terms[0])
                action_names=list(term._joint_names)
                if len(action_names)!=term.action_dim:raise ValueError("Action joint order unavailable")
            blend=PolicyActionBlend(job["blend_teacher_export"],job["blend_teacher_weight"],env.unwrapped.device,action_names,scope)
            report["policy_blend"]=blend.audit
        if 'reference_speed_teacher_export' in job:
            if blend is not None:raise ValueError('Cannot combine fixed and speed blend')
            from reference_speed_blend import ReferenceSpeedBlend
            action_names=None
            if 'candidate_hip_filter_alpha' in job or 'candidate_leg_filter_alpha' in job or job.get('candidate_action_scope','all')!='all':
                manager=env.unwrapped.action_manager
                if len(manager.active_terms)!=1:raise ValueError('Hip filter requires one action term')
                term=manager.get_term(manager.active_terms[0]);action_names=list(term._joint_names)
                if len(action_names)!=term.action_dim:raise ValueError('Action order unavailable')
            if job.get('candidate_action_scope','all')!='all':
                if trained_speed_contract is None:raise ValueError('Action partition requires a matching trained contract')
                from candidate_scope_contract import require_scope_action_names
                require_scope_action_names(trained_speed_contract['contract'],action_names)
            blend=ReferenceSpeedBlend(job['reference_speed_teacher_export'],env.unwrapped.device,command,
                job.get('maximum_candidate_weight',.5),job.get('blend_speed_low',.05),job.get('blend_speed_high',.1),
                hip_alpha=job.get('candidate_hip_filter_alpha'),action_names=action_names,
                hip_mode=job.get('candidate_hip_filter_mode','absolute'),leg_alpha=job.get('candidate_leg_filter_alpha'),leg_mode=job.get('candidate_leg_filter_mode','absolute'),candidate_scope=job.get('candidate_action_scope','all'))
            if 'blend_release_tau' in job:
                from release_speed_blend import ReleaseSpeedBlend
                blend=ReleaseSpeedBlend(blend,float(job['blend_release_tau']),env.unwrapped.step_dt)
            report['policy_blend']=blend.audit
        action_filter=None
        if "leg_filter_alpha" in job:
            from policy_action_filter import LegActionFilter
            manager=env.unwrapped.action_manager
            if len(manager.active_terms)!=1:raise ValueError("Filter requires one action term")
            term=manager.get_term(manager.active_terms[0])
            action_filter=LegActionFilter(list(term._joint_names),env.unwrapped.device,job["leg_filter_alpha"],job.get("leg_filter_scope","legs"))
            mode=job.get("leg_filter_mode","absolute")
            if mode not in ("absolute","reference_residual","two_tap"):raise ValueError("Invalid filter mode")
            if mode=="two_tap":
                from policy_action_filter import TwoTapAnkleFilter
                action_filter=TwoTapAnkleFilter(list(term._joint_names),env.unwrapped.device,job["leg_filter_alpha"],job.get("leg_filter_scope","legs"))
            action_filter.audit["mode"]=mode
            report["action_filter"]=action_filter.audit
        contact_trace=[] if job.get('record_contact_feedback',False) else None
        trace=None
        if job.get("record_policy_observations",False):
            from policy_observation_trace import PolicyObservationTrace
            trace=PolicyObservationTrace(env.unwrapped)
        writer=None
        batch_video=None
        batch_video_reset=False
        if job.get('record_video',False):
            if not args.enable_cameras:raise ValueError('Batch video requires enable_cameras')
            from batch_video import BatchVideo
            batch_video=BatchVideo(env,output,job.get('camera_view','side'),warmup_frames=job.get('video_warmup_frames',0))
            report['video_camera']=batch_video.audit
        if batch_video is None and args.standing_diagnostic and args.enable_cameras:
            import imageio.v2 as imageio
            import numpy as np
            origin=env.unwrapped.scene['robot'].data.root_pos_w[0].cpu().numpy()
            env.unwrapped.sim.set_camera_view(eye=origin+np.array(args.viewer_eye if args.viewer_eye is not None else [0,-1.2,.15]),target=origin+np.array(args.viewer_lookat if args.viewer_lookat is not None else [0,0,-.03]))
            writer=imageio.get_writer(str(output.with_suffix('.mp4')),fps=25,codec='libx264',macro_block_size=None)
        with torch.inference_mode():
            for step in range(int(job['steps'])):
                if contact_trace is not None:
                    from contact_feedback import ground_contact_features
                    contact_trace.append(ground_contact_features(env.unwrapped,sensor_prefix="diagnostic_").cpu().numpy().copy())
                if job.get("controller","policy")=="reference_pd":
                    from standing_diagnostic import reference_actions
                    actions=reference_actions(env.unwrapped,command)
                else:actions=policy(obs)
                if blend is not None:actions=blend(obs,actions)
                if action_filter is not None:
                    reference=None
                    if job.get("leg_filter_mode","absolute")=="reference_residual":
                        from standing_diagnostic import reference_actions
                        reference=reference_actions(env.unwrapped,command)
                    actions=action_filter(actions,reference)
                if trace is not None:trace.append(obs,actions)
                kick=perturbation.apply(env.unwrapped,job,step,args.seed) if perturbation is not None else None
                if damping_schedule is not None:damping_schedule.step()
                telemetry.begin(actions)
                obs,_,dones,_=env.step(actions);telemetry.end(dones)
                if kick is not None:
                    perturbation.after_step(env.unwrapped,kick);perturbation_records.append(kick)
                report['steps']+=1
                if batch_video is not None:
                    batch_video_reset=batch_video_reset or bool(dones.any())
                    if step%2==0:
                        batch_video.append(batch_video_reset);batch_video_reset=False
                if writer is not None and step%2==0:
                    if perturbation is not None and args.viewer_eye is not None:
                        origin=env.unwrapped.scene['robot'].data.root_pos_w[0].cpu().numpy()
                        env.unwrapped.sim.set_camera_view(eye=origin+np.array(args.viewer_eye),target=origin+np.array(args.viewer_lookat))
                    writer.append_data(env.unwrapped.render())
                manager=env.unwrapped.termination_manager
                for name in manager.active_terms:
                    count=int(manager.get_term(name).sum().item())
                    report['termination_counts'][name]=report['termination_counts'].get(name,0)+count
                for name,value in command.metrics.items():
                    if name.startswith('error_'):
                        report['motion_metrics'][name]=report['motion_metrics'].get(name,0.)+float(value.mean().item())
                if blend is not None and hasattr(blend,'reset'):blend.reset(dones)
                if action_filter is not None:action_filter.reset(dones)
                if resetter is not None:resetter.reset(dones)
                if args.standing_diagnostic and dones.any():break
        if writer is not None:writer.close()
        if batch_video is not None:batch_video.close()
        telemetry.save()
        if damping_schedule is not None:
            schedule_path=output.with_suffix(".damping.npz")
            damping_schedule.save(schedule_path)
            report["locomotion_damping_trace"]=str(schedule_path)
            report["locomotion_damping_contract"]={"joints":damping_schedule.names,"maximum_gain":damping_schedule.maximum_gain,"speed_low":.05,"speed_high":.1,"schedule":"smoothstep","base":1.0,"reset":"before each clip"}
        if trace is not None:
            trace_path=output.with_suffix(".policy_observations.npz")
            trace.save(trace_path)
            report["policy_observations"]=str(trace_path)
        if contact_trace is not None:
            import numpy as np
            contact_path=output.with_suffix('.contact_feedback.npz')
            columns=[side+'_'+term for side in ['left','right'] for term in ['load_100N','vx_heading_0.1m_s','vy_heading_0.1m_s','xy_rms_0.1m_s']]
            np.savez_compressed(contact_path,features=np.concatenate(contact_trace,axis=0),columns=np.array(columns))
            report['contact_feedback']={'trace':str(contact_path),'policy_input_changed':False,'timing':'before policy inference; reset samples masked to zero'}
        report['motion_metrics']={k:v/report['steps'] for k,v in report['motion_metrics'].items()}
        report.update(diagnostic=bool(args.standing_diagnostic),controller=job.get('controller','policy'),physics_dt=env.unwrapped.physics_dt,checkpoint=checkpoint,motion_id=job['motion_id'],seed=args.seed,
            seconds=report['steps']*env.unwrapped.step_dt,num_envs=1,
            control_runtime=snapshot(cfg),batch_protocol='single_startup_per_seed_reset_per_clip_v1',
            upper_body_termination=bool(args.evaluation_upper_body_termination),
            note='Startup physical randomization shared across clips within seed. Compare only matched batch protocol.')
        if perturbation is not None:
            report['perturbations']=perturbation_records
            report['requested_push_speed']=job['push_speed']
        if blend is not None and hasattr(blend,'save_trace'):
            blend.save_trace(output.with_suffix('.blend.npz'))
        output.write_text(json.dumps(report,indent=2)+'\n')
        print('BATCH_COMPLETED',output,flush=True)

    Path(str(args.evaluation_batch)+'.done').write_text('completed\n')
