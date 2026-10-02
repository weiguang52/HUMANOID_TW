"""Compare physical oscillation and completion at the same replay settings."""
import json
from pathlib import Path
import numpy as np
from analyze_s2_joint_commands import read
ROOT=Path('/root/gpufree-data/datasets/practice9/s2_control_fix')
OLD=Path('/root/gpufree-data/datasets/practice9/s2_trials/smooth')
IDS=['000801','006680','010407','010782','s2_stand','s2_weight_shift','s2_wave_000113']

def summarize(path,ids):
    rows=[]
    for clip in ids:
        file=path/(clip+'.evaluation.json')
        if not file.exists():continue
        data=read(path/(clip+'.telemetry.npz'));e=json.load(open(file))
        a=np.sqrt(np.nanmean(data['actual_band']**2,axis=0))
        c=np.sqrt(np.nanmean(data['band']**2,axis=0))
        rows.append(dict(clip=clip,termination=e['termination_counts'],body_error_m=e['motion_metrics']['error_body_pos'],
            actual_2_8_rms_deg=float(np.sqrt(np.nanmean(data['actual_band']**2))),
            command_2_8_rms_deg=float(np.sqrt(np.nanmean(data['band']**2))),
            joint_actual_2_8_rms_deg=dict(zip(data['names'],a.tolist())),
            joint_command_2_8_rms_deg=dict(zip(data['names'],c.tolist())),control=e.get('control_runtime')))
    return rows

if __name__=='__main__':
    report={'reference_smooth':summarize(OLD,IDS)}
    for directory in sorted(ROOT.iterdir()):
        if directory.is_dir():
            rows=summarize(directory,IDS)
            if rows:report[directory.name]=rows
    (ROOT/'comparison.json').write_text(json.dumps(report,indent=2)+'\n')
    for variant,rows in report.items():
        print(variant)
        for r in rows: print(r['clip'],round(r['actual_2_8_rms_deg'],3),r['termination'])
