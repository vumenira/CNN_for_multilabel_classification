import argparse
import json
from pathlib import Path

TEMPLATE = '''import glob, os, subprocess, sys, time
JOBS = {jobs}
BRANCH = "{branch}"
subprocess.run("pip install -q -U timm", shell=True)
subprocess.run(f"git clone -q -b {{BRANCH}} https://github.com/vumenira/CNN_for_multilabel_classification.git /tmp/repo", shell=True, check=True)
data = os.path.dirname(sorted(glob.glob("/kaggle/input/**/test.csv", recursive=True))[0])
print("data", data, os.listdir(data), flush=True)
env = dict(os.environ, DATA_DIR=data, CACHE_DIR="/tmp/cache", OUT_DIR="/kaggle/working/out", PYTHONPATH="/tmp/repo")
subprocess.run([sys.executable, "-c", "from scoremax.common import build_cache; [print(build_cache(s), flush=True) for s in ('train', 'test')]"], cwd="/tmp/repo", env=env, check=True)
subprocess.run("nvidia-smi --query-gpu=name,memory.total --format=csv,noheader; nproc; free -g | sed -n 2p; df -h /tmp /kaggle/working | tail -2", shell=True)
procs = []
for gpu, commands in enumerate(JOBS):
    log = open(f"/kaggle/working/gpu{{gpu}}.log", "w")
    procs.append(subprocess.Popen(" ; ".join(commands), shell=True, cwd="/tmp/repo", env=dict(env, CUDA_VISIBLE_DEVICES=str(gpu)), stdout=log, stderr=subprocess.STDOUT))
seen = [0] * len(procs)
while True:
    alive = any(p.poll() is None for p in procs)
    for gpu in range(len(procs)):
        lines = [l for l in open(f"/kaggle/working/gpu{{gpu}}.log").read().splitlines() if "Warning" not in l and l.strip()]
        for line in lines[seen[gpu]:]:
            print(f"[gpu{{gpu}}] {{line}}", flush=True)
        seen[gpu] = len(lines)
    if not alive:
        break
    time.sleep(60)
print("KERNEL DONE", flush=True)
'''

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--slug", required=True)
    ap.add_argument("--dir", required=True)
    ap.add_argument("--branch", default="scoremax")
    ap.add_argument("--gpu0", nargs="+", required=True)
    ap.add_argument("--gpu1", nargs="*", default=[])
    a = ap.parse_args()
    folder = Path(a.dir)
    folder.mkdir(parents=True, exist_ok=True)
    jobs = [a.gpu0] + ([a.gpu1] if a.gpu1 else [])
    (folder / "job.py").write_text(TEMPLATE.format(jobs=json.dumps(jobs), branch=a.branch))
    meta = {
        "id": f"vumenira/{a.slug}",
        "title": a.slug,
        "code_file": "job.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": True,
        "enable_internet": True,
        "competition_sources": ["cats-and-dogs"],
        "dataset_sources": [],
        "kernel_sources": [],
    }
    (folder / "kernel-metadata.json").write_text(json.dumps(meta, indent=1))
    print(folder)
