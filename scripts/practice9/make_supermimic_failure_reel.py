"""All fixed-test failures, front/side at explicitly labelled quarter-speed."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

import mediapy
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--evaluation', type=Path, required=True)
    a = p.parse_args()
    results_path = a.evaluation / 'results.json'
    report = json.loads(results_path.read_text())
    audit = json.loads((a.evaluation / 'video_audit.json').read_text())
    assert audit['passed'] and all(v['full_decode_passed'] for v in audit['videos'])
    rows = [r for r in report['rows'] if r['agent'] == 'trained']
    assert len(rows) == 7 and len({r['id'] for r in rows}) == 7
    assert all(r['terminated'] and not r['completed_requested_window'] for r in rows)
    output = a.evaluation / 'trained_failure_reel_025x.mp4'
    assert not output.exists(), 'Refusing existing output overwrite'
    font = ImageFont.load_default(size=27)
    frame_count = 0
    with mediapy.VideoWriter(str(output), shape=(688, 1920), fps=50) as writer:
        for row in rows:
            front = mediapy.read_video(str(a.evaluation / row['front_video']))
            side = mediapy.read_video(str(a.evaluation / row['side_video']))
            assert len(front) == len(side) == row['steps']
            assert front.shape[1:] == side.shape[1:] == (640, 960, 3)
            for f, s in zip(front, side, strict=True):
                image = Image.new('RGB', (1920, 688), (15, 15, 15))
                image.paste(Image.fromarray(f), (0, 0))
                image.paste(Image.fromarray(s), (960, 0))
                ImageDraw.Draw(image).text(
                    (12, 647),
                    f"0.25x SLOW PLAYBACK | {row['id']} FAILED at {row['duration_s']:.2f}s | Original episode time above; no restart or frame extension",
                    fill='white', font=font,
                )
                frame = np.asarray(image)
                for _ in range(4):
                    writer.add_image(frame)
                    frame_count += 1
    stream = json.loads(subprocess.check_output([
        'ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_entries',
        'stream=codec_name,nb_frames,r_frame_rate,width,height', '-of', 'json', str(output),
    ], text=True))['streams'][0]
    assert stream['codec_name'] == 'h264' and stream['r_frame_rate'] == '50/1'
    assert (stream['width'], stream['height']) == (1920, 688)
    assert int(stream['nb_frames']) == frame_count == sum(r['steps'] for r in rows) * 4
    subprocess.run(['ffmpeg', '-v', 'error', '-xerror', '-i', str(output), '-f', 'null', '-'], check=True)
    (output.with_suffix('.json')).write_text(json.dumps(dict(
        kind='all_seven_trained_failures_quarter_speed', playback_speed=.25,
        source_results_sha256=hashlib.sha256(results_path.read_bytes()).hexdigest(),
        checkpoint_sha256=report['checkpoint_sha256'],
        ids=[r['id'] for r in rows], actual_episode_duration_s=[r['duration_s'] for r in rows],
        playback_duration_s=frame_count/50, frame_count=frame_count,
        sha256=hashlib.sha256(output.read_bytes()).hexdigest(), full_decode_passed=True,
        note='Every recorded source frame repeated exactly four times. No held terminal frame or fabricated continuation. Original real-time source videos remain separate.',
        stream=stream,
    ), indent=2)+'\n')
    index = a.evaluation / 'index.html'
    summary = '<section><h2>Result: 0/7 completed; all trials terminated early</h2><p>Quarter-speed failure reel below includes every trained test clip in fixed order. Actual episode times remain in the original labels. Slow playback is not longer survival.</p><video controls preload="metadata" style="width:100%;max-width:1400px" src="trained_failure_reel_025x.mp4"></video><p><a href="trained_failure_reel_025x.json">Reel audit</a> | <a href="../../kinematic_reference/index.html">Kinematic references: NOT POLICY</a></p><p>Zero means zero normalized PD action offset, which drives the original task nominal pose; it is not zero torque or reference feedforward.</p></section>'
    index.write_text(index.read_text().replace('</h1>', '</h1>'+summary, 1))
    print(output, frame_count, 'frames; full decode passed')


if __name__ == '__main__':
    main()
