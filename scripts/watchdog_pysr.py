#!/usr/bin/env python3
"""
RAM watchdog for PySR runs.
Launches a PySR script as subprocess, monitors the entire process tree
every POLL_SECONDS, kills it if RSS exceeds KILL_GB, warns at WARN_GB.
"""
import os
import sys
import time
import signal
import subprocess
from pathlib import Path

import psutil

POLL_SECONDS = 20
WARN_GB = 6.0
KILL_GB = 10.0
LOG = Path("data/results/watchdog_pysr.log")
LOG.parent.mkdir(parents=True, exist_ok=True)


def tree_rss(proc: psutil.Process) -> float:
    """Total RSS of a process and all its children, in GB."""
    rss = 0
    try:
        rss += proc.memory_info().rss
        for child in proc.children(recursive=True):
            try:
                rss += child.memory_info().rss
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        pass
    return rss / 1e9


def main():
    if len(sys.argv) < 2:
        print("Usage: watchdog_pysr.py <script.py> [args...]")
        sys.exit(2)

    cmd = [sys.executable] + sys.argv[1:]
    print(f"[watchdog] Launching: {' '.join(cmd)}")
    print(f"[watchdog] Thresholds: warn={WARN_GB} GB, kill={KILL_GB} GB, poll={POLL_SECONDS}s")
    print(f"[watchdog] Log: {LOG}")

    logf = open(LOG, "w", buffering=1)
    start = time.time()
    proc = subprocess.Popen(cmd, stdout=logf, stderr=subprocess.STDOUT)
    try:
        parent = psutil.Process(proc.pid)
    except psutil.NoSuchProcess:
        print("[watchdog] Subprocess died immediately")
        sys.exit(proc.returncode if proc.returncode is not None else 1)

    peak = 0.0
    warned = False
    try:
        while proc.poll() is None:
            time.sleep(POLL_SECONDS)
            try:
                rss = tree_rss(parent)
            except psutil.NoSuchProcess:
                break
            peak = max(peak, rss)
            elapsed = time.time() - start
            status = f"[watchdog t={elapsed:7.0f}s] RSS = {rss:5.2f} GB  peak = {peak:5.2f} GB"
            print(status)

            if rss > KILL_GB:
                print(f"[watchdog] *** RSS exceeded kill threshold ({KILL_GB} GB); terminating ***")
                # SIGTERM tree
                for child in parent.children(recursive=True):
                    try: child.send_signal(signal.SIGTERM)
                    except psutil.NoSuchProcess: pass
                parent.send_signal(signal.SIGTERM)
                time.sleep(3)
                # SIGKILL if still alive
                for child in parent.children(recursive=True):
                    try: child.kill()
                    except psutil.NoSuchProcess: pass
                try: parent.kill()
                except psutil.NoSuchProcess: pass
                print(f"[watchdog] killed after reaching {rss:.2f} GB")
                sys.exit(137)

            if rss > WARN_GB and not warned:
                print(f"[watchdog] *** WARNING: RSS exceeded {WARN_GB} GB ***")
                warned = True
    except KeyboardInterrupt:
        print("[watchdog] SIGINT received; forwarding to subprocess")
        proc.send_signal(signal.SIGINT)
        proc.wait()

    rc = proc.wait()
    elapsed = time.time() - start
    print(f"\n[watchdog] DONE in {elapsed:.0f}s, peak RSS = {peak:.2f} GB, exit code = {rc}")
    logf.close()
    sys.exit(rc)


if __name__ == "__main__":
    main()
