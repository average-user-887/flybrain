#!/usr/bin/env python3
"""Real-browser check that the dashboard never shows a frozen simulation as LIVE (audit F).

Starts its OWN daemon (through tests/fixtures/silent_freeze/faultd.py, which only adds
fault switches) and its own static web server, then drives real Firefox (selenium +
geckodriver) through:

  a. healthy run: the pill says LIVE and the step age keeps resetting;
  b. disk full (every checkpoint write raises ENOSPC): the amber "NOT SAVING" banner
     appears while the step counter keeps advancing and the pill stays LIVE;
  c. disk freed: the banner clears on the next retry and nothing was lost;
  d. a step that hangs past --step-hard-limit: STEP RUNNING, then the red
     "SIMULATION NOT ADVANCING" pill; it recovers by itself when the step returns;
  e. simulation thread death: the red "SIMULATION NOT ADVANCING" pill within the
     stall threshold (never LIVE), then re-selecting the assay in the page restarts the
     thread and the pill returns to LIVE with an advancing step.

    python scripts/silent_freeze_browser_check.py --daemon-port 8861 --web-port 8860 \
        --scratch /path/to/scratch --out docs/receipts/<dir>

Ports 8769/8780/8781 (a running observatory) are refused.  Nothing fills a disk.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.firefox.options import Options
from selenium.webdriver.firefox.service import Service

ROOT = Path(__file__).resolve().parents[1]
FAULTD = ROOT / "tests" / "fixtures" / "silent_freeze" / "faultd.py"

UI_JS = """
const t = id => document.getElementById(id)?.textContent ?? null;
const d = window.neuroflyDiagnostics, b = d?.bridge;
const banner = document.getElementById('persistenceBanner');
return {pill: t('clusterStatusPill'), pillTitle: document.getElementById('clusterStatusPill')?.title ?? null,
        pillColor: document.getElementById('clusterStatusPill')?.style.color ?? null,
        stepAge: t('statStepAge'), dataAge: t('statDataAge'), step: t('statStep'),
        lastStep: d ? d.lastStep() : null, stepSeen: b?.lastStepSeen ?? null,
        stepAgeS: b ? b.stepAgeSeconds() : null, liveness: b?.daemonLiveness?.state ?? null,
        banner: banner && banner.style.display !== 'none' ? banner.textContent : '',
        badge: t('navbarParadigmBadge'),
        errors: (window.neuroflyErrors?.log || []).map(e => ({phase: e.phase, message: e.message})),
        consoleErrors: (window.__fcConsole || []).slice()};
"""
CONSOLE_HOOK = """
if (!window.__fcConsole) { window.__fcConsole = [];
  const orig = console.error.bind(console);
  console.error = (...a) => { try { window.__fcConsole.push(a.map(String).join(' ').slice(0, 300)); } catch (e) {} orig(...a); };
  window.addEventListener('error', e => window.__fcConsole.push(String(e.message).slice(0, 300))); }
return true;
"""


def now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--daemon-port", type=int, default=8861)
    ap.add_argument("--web-port", type=int, default=8860)
    ap.add_argument("--scratch", required=True, help="run directory (output dir, fault switches, logs)")
    ap.add_argument("--out", required=True, help="receipt directory (JSON + screenshots)")
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--speed", default="3")
    ap.add_argument("--geckodriver", default=None)
    args = ap.parse_args()
    for port in (args.daemon_port, args.web_port):
        if port in (8769, 8780, 8781):
            raise SystemExit(f"port {port} belongs to the observatory; refusing")
    scratch = Path(args.scratch).resolve()
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    faults = scratch / "faults"
    if scratch.exists():
        shutil.rmtree(scratch / "faults", ignore_errors=True)
    faults.mkdir(parents=True, exist_ok=True)
    daemon_url = f"http://127.0.0.1:{args.daemon_port}"
    env = dict(os.environ, FAULT_DIR=str(faults), PYTHONPATH=str(ROOT), PYTHONUNBUFFERED="1")
    dlog = open(scratch / "daemon.log", "w")
    daemon = subprocess.Popen(
        [args.python, "-u", str(FAULTD), "--host", "127.0.0.1", "--port", str(args.daemon_port),
         "--backend", "connectome-fixed", "--paradigm", "t-maze", "--speed", args.speed, "--continuous",
         "--trial-seconds", "120", "--checkpoint-interval", "5", "--step-hard-limit", "12",
         "--output-dir", str(scratch / "out"), "--data-dir", str(scratch / "out" / "learning"),
         "--pid-file", str(scratch / "daemon.pid")], cwd=ROOT, env=env, stdout=dlog, stderr=subprocess.STDOUT)
    web = subprocess.Popen([args.python, "-m", "http.server", str(args.web_port), "--bind", "127.0.0.1",
                            "--directory", str(ROOT / "web")], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    R = {"kind": "silent-freeze-browser-check", "started_at": now(), "daemon_port": args.daemon_port,
         "web_port": args.web_port, "checks": {}, "samples": {}, "screenshots": []}
    driver = None

    def status():
        with urllib.request.urlopen(daemon_url + "/api/status", timeout=5) as r:
            return json.loads(r.read())

    def arm(name, spec):
        (faults / name).write_text(spec)

    def disarm(name):
        (faults / name).unlink(missing_ok=True)

    try:
        end = time.time() + 300
        while time.time() < end:
            try:
                if status()["total_steps"] > 20:
                    break
            except Exception:
                pass
            time.sleep(1)
        gecko = args.geckodriver or shutil.which("geckodriver")
        if not gecko:
            raise SystemExit("geckodriver not found on PATH")
        if Path(gecko).resolve().as_posix().startswith("/snap/") or "/snap/" in gecko:
            profile_root = Path.home() / "snap" / "firefox" / "common" / "neurofly-freeze-check"
        else:
            profile_root = Path(tempfile.mkdtemp(prefix="neurofly-freeze-"))
        profile_root.mkdir(parents=True, exist_ok=True)
        opts = Options()
        opts.add_argument("-headless")
        opts.add_argument("--width=1600")
        opts.add_argument("--height=1000")
        driver = webdriver.Firefox(options=opts, service=Service(
            executable_path=gecko, service_args=["--profile-root", str(profile_root)],
            log_output=str(scratch / "geckodriver.log")))
        caps = driver.capabilities
        R["browser"] = {"name": caps.get("browserName"), "version": caps.get("browserVersion"),
                        "headless": caps.get("moz:headless")}

        def ui():
            return driver.execute_script(UI_JS)

        def wait_for(pred, timeout, poll=0.25):
            t0, last = time.time(), None
            while time.time() - t0 < timeout:
                last = ui()
                try:
                    if pred(last):
                        return last, True, round(time.time() - t0, 2)
                except Exception:
                    pass
                time.sleep(poll)
            return last, False, round(time.time() - t0, 2)

        def sample(seconds, every=1.0):
            rows = []
            for _ in range(int(seconds / every)):
                u = ui()
                rows.append({k: u[k] for k in ("pill", "stepAge", "lastStep", "liveness", "banner")})
                time.sleep(every)
            return rows

        def shot(name):
            p = out / name
            driver.save_screenshot(str(p))
            R["screenshots"].append(p.name)

        driver.get(f"http://127.0.0.1:{args.web_port}/index.html?daemon=http%3A%2F%2F127.0.0.1%3A{args.daemon_port}")
        driver.execute_script(CONSOLE_HOOK)

        # a. healthy ---------------------------------------------------------------
        # The step-age readout is painted by the page's 4 Hz freshness tick, which can
        # follow the first LIVE paint by up to 250 ms: wait for both.
        u, ok, _ = wait_for(lambda u: "LIVE" in (u["pill"] or "") and u["lastStep"]
                            and u["stepAge"] not in (None, "--"), 60)
        rows = sample(5)
        R["samples"]["a_healthy"] = rows
        R["checks"]["a_healthy_live"] = ok and all("LIVE" in (r["pill"] or "") for r in rows)
        R["checks"]["a_step_advancing"] = rows[-1]["lastStep"] > rows[0]["lastStep"]
        R["checks"]["a_step_age_resets"] = all(r["stepAge"] not in (None, "--") and float(r["stepAge"].rstrip("s")) < 3
                                               for r in rows)
        R["checks"]["a_no_banner"] = all(not r["banner"] for r in rows)
        shot("a-healthy-live.png")

        # b. disk full: degraded but stepping ---------------------------------------
        arm("ckpt_write", "ENOSPC")
        t_arm = time.time()
        u, ok, took = wait_for(lambda u: "NOT SAVING" in (u["banner"] or ""), 30)
        rows = sample(6)
        R["samples"]["b_disk_full"] = rows
        R["checks"]["b_banner_shown"] = ok and all("NOT SAVING" in r["banner"] for r in rows)
        R["checks"]["b_banner_says_disk_full"] = all("disk full" in r["banner"] for r in rows)
        R["checks"]["b_steps_still_advance"] = rows[-1]["lastStep"] > rows[0]["lastStep"]
        R["checks"]["b_pill_still_live"] = all("LIVE" in (r["pill"] or "") for r in rows)
        st = status()
        R["checks"]["b_status_degraded"] = st["status"] == "degraded"
        R["b_banner_after_s"] = took
        R["b_status_persistence"] = {k: st["persistence"].get(k) for k in ("state", "reason")}
        shot("b-disk-full-not-saving.png")

        # c. disk freed: retry clears the banner (back-off 30 s) --------------------
        disarm("ckpt_write")
        u, ok, took = wait_for(lambda u: not u["banner"] and "LIVE" in (u["pill"] or ""), 75)
        R["checks"]["c_banner_clears_after_retry"] = ok
        R["c_cleared_after_s"] = took
        R["checks"]["c_status_online"] = status()["status"] == "online"

        # d. a hanging step: slow, then NOT ADVANCING after the hard limit -----------
        arm("arena_step", "once:HANG25")
        u, ok_slow, _ = wait_for(lambda u: "STEP RUNNING" in (u["pill"] or ""), 15)
        u, ok_red, took = wait_for(lambda u: "NOT ADVANCING" in (u["pill"] or ""), 30)
        R["checks"]["d_slow_then_not_advancing"] = ok_slow and ok_red
        R["d_red_after_s"] = took
        R["d_pill"] = u["pill"]
        shot("d-hung-step-not-advancing.png")
        u, ok, took = wait_for(lambda u: "LIVE" in (u["pill"] or "") and u["liveness"] == "advancing", 40)
        R["checks"]["d_recovers_when_step_returns"] = ok

        # e. thread death: red within the threshold, re-select assay restarts ---------
        before = ui()["lastStep"]
        arm("arena_step", "once:THREADEXIT")
        t_kill = time.time()
        u, ok, took = wait_for(lambda u: "NOT ADVANCING" in (u["pill"] or ""), 15)
        rows = sample(5)
        R["samples"]["e_dead"] = rows
        st = status()
        R["checks"]["e_red_pill_within_threshold"] = ok and took <= 10.0
        R["checks"]["e_never_live_while_dead"] = all("LIVE" not in (r["pill"] or "") for r in rows)
        R["checks"]["e_step_frozen"] = len({r["lastStep"] for r in rows}) == 1
        R["checks"]["e_status_dead"] = st["liveness"]["state"] == "dead" and st["status"] == "error"
        R["e_red_after_s"] = took
        R["e_pill"] = u["pill"]
        R["e_status_error"] = st["error"]
        shot("e-thread-dead-not-advancing.png")
        el = driver.find_element(By.CSS_SELECTOR, '.experiment-card[data-paradigm="t-maze"]')
        driver.execute_script("arguments[0].scrollIntoView({block:'center'});", el)
        time.sleep(0.2)
        el.click()
        u, ok, took = wait_for(lambda u: "LIVE" in (u["pill"] or "") and u["lastStep"] and u["lastStep"] > before + 10, 60)
        rows = sample(4)
        R["checks"]["e_reselect_restarts_and_live"] = ok and rows[-1]["lastStep"] > rows[0]["lastStep"]
        R["checks"]["e_status_online_after"] = status()["status"] == "online"
        shot("e-recovered-live.png")
        final = ui()
        R["checks"]["no_page_errors"] = not final["errors"] and not final["consoleErrors"]
        R["final_errors"] = final["errors"] + final["consoleErrors"]
    finally:
        if driver is not None:
            try:
                driver.quit()
            except Exception:
                pass
        for f in faults.iterdir():
            if f.name not in ("fired.log", "threads.json"):
                f.unlink(missing_ok=True)
        if daemon.poll() is None:
            daemon.send_signal(signal.SIGTERM)
            try:
                R["daemon_exit_code"] = daemon.wait(timeout=60)
            except subprocess.TimeoutExpired:
                daemon.kill()
                R["daemon_exit_code"] = "killed"
        web.terminate()
        dlog.close()
    R["finished_at"] = now()
    R["pass"] = bool(R["checks"]) and all(R["checks"].values())
    (out / "silent_freeze_browser_check.json").write_text(json.dumps(R, indent=1) + "\n")
    print(json.dumps({"pass": R["pass"], "checks": R["checks"]}, indent=1))
    return 0 if R["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
