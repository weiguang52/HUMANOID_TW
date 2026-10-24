"""Reuse the verified colored evaluation protocol in isolated output directories."""
from pathlib import Path
import hashlib
import json
import subprocess

REPO = Path('/root/gpufree-data/projects/HUMANOID_TW')
OLD = Path('/root/gpufree-data/datasets/practice9/head_locked_20261024')
ROOT = Path('/root/gpufree-data/datasets/practice9/support_ablation_20261024')
source = OLD / 'evaluation_colored'
provenance = {}
for variant, task in [('baseline', 'HeadLocked'), ('support', 'SupportVelocity')]:
    state = ROOT / variant
    target = state / 'evaluation_colored'
    if target.exists():
        raise FileExistsError(target)
    for folder in ['validation/neck_only', 'eval/neck_only', 'diagnostic/neck_only',
                   'diagnostic_front/neck_only', 'front/neck_only', 'neck_only']:
        (target / folder).mkdir(parents=True, exist_ok=True)
    # References and physics are shared read-only; outputs are always new paths.
    (state / 'evaluation').symlink_to(OLD / 'evaluation', target_is_directory=True)
    for name in ['robot_colored.usd', 'robot_colored.audit.json', 'robot_colored.colors.json']:
        (state / name).symlink_to(OLD / name)
    paths = [source / 'validation/neck_only/manifest.json',
             source / 'diagnostic/neck_only/jobs.json',
             source / 'diagnostic_front/neck_only/jobs.json']
    paths += list((source / 'eval/neck_only').glob('*.json'))
    paths += [source / f'validation/neck_only/jobs{s}.json' for s in [42, 123, 2026]]
    for path in paths:
        raw = path.read_text()
        # Only relocate job OUTPUT paths; manifest source paths remain immutable.
        if path.name.startswith('jobs'):
            raw = raw.replace(str(source), str(target))
        (target / path.relative_to(source)).write_text(raw)
    (target / 'neck_only/data_audit.json').write_text(json.dumps({
        'training_manifest': '/root/gpufree-data/datasets/practice9/cpp_refresh_20261024/train.json',
        'evaluation_manifest_sha256': hashlib.sha256((source/'validation/neck_only/manifest.json').read_bytes()).hexdigest(),
        'variant': variant, 'training_seed': 42, 'additional_iterations': 2000,
    }, indent=2))
    for name in ['run.sh', 'front.sh', 'diagnostic_front.sh', 'report.py']:
        path = source / name
        raw = path.read_text()
        provenance[name] = hashlib.sha256(path.read_bytes()).hexdigest()
        raw = raw.replace(str(source), str(target))
        raw = raw.replace('neck_only) KIND=HeadLocked', f'neck_only) KIND={task}')
        if name in ['front.sh', 'diagnostic_front.sh']:
            raw = '\n'.join('CHECKPOINT=$(cat "$STATE/neck_only/final_checkpoint")'
                            if line.startswith('CHECKPOINT=') else line for line in raw.splitlines())+'\n'
        raw = raw.replace('validation_artifacts/head_locked_20261024/evaluation_colored',
                          f'validation_artifacts/support_ablation_20261024/{variant}/evaluation_colored')
        (target / name).write_text(raw)
        if name.endswith('.sh'):
            subprocess.run(['bash', '-n', str(target / name)], check=True)
    raw = (REPO/'scripts/practice9/finish_head_locked_color.py').read_text()
    provenance['finish_head_locked_color.py'] = hashlib.sha256(raw.encode()).hexdigest()
    raw = raw.replace("S=Path('/root/gpufree-data/datasets/practice9/head_locked_20261024')",f'S=Path({str(state)!r})')
    raw = raw.replace('validation_artifacts/head_locked_20261024/evaluation_colored',
                      f'validation_artifacts/support_ablation_20261024/{variant}/evaluation_colored')
    raw = raw.replace('model_27995.pt', 'see model_final.pt and checkpoint params')
    raw = raw.replace('Colored rerun', variant+' continuation')
    raw = raw.replace('colored rerun', variant+' continuation')
    raw = raw.replace('metric_reproducibility.json', 'previous_checkpoint_differences.json')
    (target/'finish.py').write_text(raw)
    # Metric differences are recorded, never asserted to be zero after training.
    compile(raw, str(target/'finish.py'), 'exec')
(ROOT/'evaluation_template_sha256.json').write_text(json.dumps(provenance,indent=2))
print('Prepared isolated evaluation jobs for both variants')
