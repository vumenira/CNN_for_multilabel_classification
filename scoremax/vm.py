import base64
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SH = '''import base64, subprocess
r = subprocess.run(base64.b64decode("{cmd}").decode(), shell=True, capture_output=True, text=True, errors="replace", cwd="/content")
print(r.stdout[-6000:], r.stderr[-2000:], sep="")
'''
BG = '''import base64, os, subprocess
os.makedirs("/content/logs", exist_ok=True)
log = open("/content/logs/{log}.log", "a")
p = subprocess.Popen(base64.b64decode("{cmd}").decode(), shell=True, stdout=log, stderr=subprocess.STDOUT, cwd="/content", start_new_session=True)
print("started pid", p.pid)
'''


def call(account, session, code, timeout=120, tries=3):
    args = [str(HERE / "colabx.sh"), account, "exec", "-s", session]
    for _ in range(tries):
        try:
            r = subprocess.run(args, input=code, text=True, capture_output=True, timeout=timeout)
        except subprocess.TimeoutExpired:
            print("[vm] timeout, retrying", file=sys.stderr)
            continue
        if "Connection was lost" in r.stdout + r.stderr:
            print("[vm] connection lost, retrying", file=sys.stderr)
            continue
        return r.stdout + r.stderr[-1500:]
    return "[vm] FAILED after retries"


def sh(account, session, command):
    return call(account, session, SH.format(cmd=base64.b64encode(command.encode()).decode()))


def bg(account, session, log, command):
    return call(account, session, BG.format(log=log, cmd=base64.b64encode(command.encode()).decode()), tries=1)


if __name__ == "__main__":
    account, session, mode = sys.argv[1:4]
    print(sh(account, session, sys.argv[4]) if mode == "sh" else bg(account, session, sys.argv[4], sys.argv[5]))
