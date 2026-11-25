import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from scoremax.vm import sh


def kg(*args):
    try:
        return subprocess.run([str(HERE / "kg.sh"), *args], capture_output=True, text=True, timeout=90).stdout
    except Exception as error:
        return f"ERR {error}"


def snapshot(kernels, vms):
    state = {}
    for kernel in kernels:
        out = kg("kernels", "status", f"vumenira/{kernel}").strip().splitlines()
        state[f"kernel {kernel}"] = out[-1].split("status")[-1].strip() if out else "?"
    for account, session, command in vms:
        state[f"vm {account}/{session}"] = " | ".join(sh(account, session, command).strip().splitlines()[-6:])
    subs = [l for l in kg("competitions", "submissions", "-c", "cats-and-dogs").splitlines()[2:5]]
    state["subs"] = " || ".join(" ".join(l.split()[1:2] + l.split()[-2:]) for l in subs)
    return state


if __name__ == "__main__":
    kernels = [a for a in sys.argv[1:] if ":" not in a]
    vms = [tuple(a.split(":", 2)) for a in sys.argv[1:] if ":" in a]
    last = {}
    while True:
        now = snapshot(kernels, vms)
        for key, value in now.items():
            if last.get(key) != value:
                print(f"{time.strftime('%H:%M')} {key}: {value}", flush=True)
        last = now
        time.sleep(120)
