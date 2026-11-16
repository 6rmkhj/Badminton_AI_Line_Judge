"""Re-encode selected rally pairs into other qualities to test robustness.

Variants simulate other cameras/phones: lower resolution, lower frame rate and
heavy compression. Output: runs/variants/<variant>/<same file names>.
"""
import subprocess, sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PAIRS = ['8k 30 4k 24 {v}시점 1', '8k 30 4k 24 {v}시점 4', '8k 30 4k 24 {v}시점 7', '{v}시점 1', '{v}시점 5',
         '{v}시점 12', '{v}시점 14', '{v}시점 17', '{v}시점 19', '{v}-4시점(uhd30 4k30)']
VARIANTS = {
    'hd720': ['-vf', 'scale=1280:-2', '-c:v', 'h264_nvenc', '-cq', '23'],
    'sd480_15fps': ['-vf', 'scale=854:-2,fps=15', '-c:v', 'h264_nvenc', '-cq', '28'],
    'fhd_2mbps': ['-vf', "scale='min(1920,iw)':-2", '-c:v', 'h264_nvenc', '-b:v', '2M', '-maxrate', '2M',
                  '-bufsize', '2M'],
}


def encode(job):
    source, target, args = job
    if target.exists():
        return target.name, 'exists'
    target.parent.mkdir(parents=True, exist_ok=True)
    command = ['ffmpeg', '-hide_banner', '-loglevel', 'error', '-y', '-i', str(source), '-an', *args,
               '-pix_fmt', 'yuv420p', str(target)]
    done = subprocess.run(command, capture_output=True, text=True, encoding='utf-8', errors='replace')
    return target.name, 'ok' if done.returncode == 0 else done.stderr[-300:]


jobs = []
for pattern in PAIRS:
    for view in ('a', 'b'):
        name = pattern.format(v=view) + '.mp4'
        source = ROOT / 'video' / name
        if not source.exists():
            sys.exit('missing ' + name)
        for variant, args in VARIANTS.items():
            jobs.append((source, ROOT / 'runs/variants' / variant / name, args))
with ThreadPoolExecutor(4) as pool:
    for name, status in pool.map(encode, jobs):
        if status != 'ok':
            print(name, status)
print('encoded', len(jobs))
