#!/usr/bin/env python3
"""Real-browser check of the no-silent-freeze work and the save policy (audit F; Codex
direction 5 Oct 2026, decision 3).

Starts its OWN daemons (through tests/fixtures/silent_freeze/faultd.py, which only adds
fault switches) and its own static web server, then drives real Firefox (selenium +
geckodriver).  Daemon 1, default scientific mode:

  a. healthy run: pill LIVE, step age keeps resetting, no banner;
  b. required save fails (every checkpoint write raises ENOSPC): red
     "SIMULATION STOPPED · NOT SAVED" pill, "STOPPED · RESULT INCOMPLETE" banner, step frozen;
  c. disk freed, the assay re-selected in the page: LIVE again and advancing, while a
     "RESULT INCOMPLETE" banner keeps the failure on the record;
  d. genuine CuPy CUDARuntimeError from the connectome step: "GPU/COMPUTE ERROR" halt;
     re-select recovers;
  e. slow computer (every step sleeps 3 s): never "NOT ADVANCING";
  f. a step that hangs past --step-hard-limit: STEP RUNNING, then red NOT ADVANCING,
     recovers when the step returns;
  g. simulation thread death: red NOT ADVANCING within the threshold, never LIVE;
     re-selecting the assay restarts the thread.

Daemon 2, --exploratory:
  h. pill "LIVE DAEMON · EXPLORATORY"; the same disk-full fault gives an amber
     "EXPLORATORY · NOT SAVING" banner while the step keeps advancing.

    python scripts/silent_freeze_browser_check.py --daemon-port 8861 --web-port 8860 \
        --scratch /path/to/scratch --out /path/to/receipt

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
        stepAge: t('statStepAge'), dataAge: t('statDataAge'), step: t('statStep'),
        lastStep: d ? d.lastStep() : null, stepSeen: b?.lastStepSeen ?? null,
        stepAgeS: b ? b.stepAgeSeconds() : null, liveness: b?.daemonLiveness?.state ?? null,
        validity: b?.daemonValidity?.state ?? null,
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


class Daemon:
    def __init__(self, args, scratch: Path, name: str, extra: list):
        self.dir = scratch / name
        shutil.rmtree(self.dir / "faults", ignore_errors=True)
        self.faults = self.dir / "faults"
        self.faults.mkdir(parents=True, exist_ok=True)
        self.url = f"http://127.0.0.1:{args.daemon_port}"
        env = dict(os.environ, FAULT_DIR=str(self.faults), PYTHONPATH=str(ROOT), PYTHONUNBUFFERED="1")
        self.log = open(self.dir / "daemon.log", "w")
        self.proc = subprocess.Popen(
            [args.python, "-u", str(FAULTD), "--host", "127.0.0.1", "--port", str(args.daemon_port),
             "--backend", "connectome-fixed", "--paradigm", "t-maze", "--speed", args.speed, "--continuous",
             "--trial-seconds", "120", "--checkpoint-interval", "5", "--step-hard-limit", "12",
             "--output-dir", str(self.dir / "out"), "--data-dir", str(self.dir / "out" / "learning"),
             "--pid-file", str(self.dir / "daemon.pid")] + extra,
            cwd=ROOT, env=env, stdout=self.log, stderr=subprocess.STDOUT)
        end = time.time() + 300
        while time.time() < end:
            try:
                if self.status()["total_steps"] > 20:
                    return
            except Exception:
                pass
            time.sleep(1)
        raise SystemExit(f"daemon {name} did not start stepping")

    def status(self):
        with urllib.request.urlopen(self.url + "/api/status", timeout=5) as r:
            return json.loads(r.read())

    def arm(self, name, spec):
        (self.faults / name).write_text(spec)

    def disarm(self, name):
        (self.faults / name).unlink(missing_ok=True)

    def stop(self):
        for f in self.faults.iterdir():
            if f.name not in ("fired.log", "threads.json"):
                f.unlink(missing_ok=True)
        code = None
        if self.proc.poll() is None:
            self.proc.send_signal(signal.SIGTERM)
            try:
                code = self.proc.wait(timeout=60)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                code = "killed"
        self.log.close()
        return code


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--daemon-port", type=int, default=8861)
    ap.add_argument("--web-port", type=int, default=8860)
    ap.add_argument("--scratch", required=True, help="run directory (output dirs, fault switches, logs)")
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
    scratch.mkdir(parents=True, exist_ok=True)
    web = subprocess.Popen([args.python, "-m", "http.server", str(args.web_port), "--bind", "127.0.0.1",
                            "--directory", str(ROOT / "web")], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    R = {"kind": "silent-freeze-browser-check", "started_at": now(), "daemon_port": args.daemon_port,
         "web_port": args.web_port, "checks": {}, "timings_s": {}, "samples": {}, "screenshots": [],
         "pills": {}}
    driver, daemon = None, None
    page = f"http://127.0.0.1:{args.web_port}/index.html?daemon=http%3A%2F%2F127.0.0.1%3A{args.daemon_port}"
    try:
        daemon = Daemon(args, scratch, "scientific", [])
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
                rows.append({k: u[k] for k in ("pill", "stepAge", "lastStep", "liveness", "validity", "banner")})
                time.sleep(every)
            return rows

        def shot(name):
            p = out / name
            driver.save_screenshot(str(p))
            R["screenshots"].append(p.name)

        def reselect(assay="t-maze"):
            el = driver.find_element(By.CSS_SELECTOR, f'.experiment-card[data-paradigm="{assay}"]')
            driver.execute_script("arguments[0].scrollIntoView({block:'center'});", el)
            time.sleep(0.2)
            el.click()

        def live(u):
            return "LIVE" in (u["pill"] or "")

        def load():
            driver.get(page)
            driver.execute_script(CONSOLE_HOOK)
            return wait_for(lambda u: live(u) and u["lastStep"] and u["stepAge"] not in (None, "--"), 60)

        def fresh_tab(pred, timeout, shot_name):
            """Open the page in a NEW tab while the fault is active: the probe must connect
            and show the fault, never DISCONNECTED or the local preview."""
            main = driver.current_window_handle
            driver.switch_to.new_window("tab")
            driver.get(page)
            driver.execute_script(CONSOLE_HOOK)
            u, ok, took = wait_for(pred, timeout)
            shot(shot_name)
            driver.close()
            driver.switch_to.window(main)
            return u, ok

        C = R["checks"]
        # a. healthy ---------------------------------------------------------------
        _, ok, _ = load()
        rows = sample(5)
        R["samples"]["a_healthy"] = rows
        C["a_healthy_live"] = ok and all(live(r) for r in rows)
        C["a_step_advancing"] = rows[-1]["lastStep"] > rows[0]["lastStep"]
        C["a_step_age_resets"] = all(r["stepAge"] not in (None, "--") and float(r["stepAge"].rstrip("s")) < 3
                                     for r in rows)
        C["a_no_banner"] = all(not r["banner"] for r in rows)
        C["a_result_valid"] = rows[-1]["validity"] == "valid_so_far"
        shot("a-healthy-live.png")

        # b. scientific mode: a required save fails -> stopped, result incomplete -----
        daemon.arm("ckpt_write", "ENOSPC")
        u, ok, took = wait_for(lambda u: "NOT SAVED" in (u["pill"] or ""), 30)
        rows = sample(4)
        R["samples"]["b_required_save_failed"] = rows
        st = daemon.status()
        C["b_red_stopped_pill"] = ok and all("SIMULATION STOPPED" in (r["pill"] or "") for r in rows)
        C["b_banner_stopped_incomplete"] = all("STOPPED · RESULT INCOMPLETE" in r["banner"] for r in rows)
        C["b_never_live"] = all(not live(r) for r in rows)
        C["b_step_frozen"] = len({r["lastStep"] for r in rows}) == 1
        C["b_status"] = (st["status"] == "error" and st["halted"] is True
                         and st["error_detail"]["failure_class"] == "persistence"
                         and st["result_validity"]["state"] == "incomplete")
        R["timings_s"]["b_stopped_after"] = took
        R["pills"]["b"] = rows[-1]["pill"]
        R["b_banner"] = rows[-1]["banner"]
        shot("b-required-save-failed-stopped.png")
        u, ok = fresh_tab(lambda u: "NOT SAVED" in (u["pill"] or "") and "STOPPED" in (u["banner"] or ""), 40,
                          "b-new-tab-opened-during-fault.png")
        C["b_new_tab_during_fault_shows_stopped"] = ok and "DISCONNECTED" not in (u["pill"] or "") \
            and "LOCAL ENGINE" not in (u["pill"] or "")
        R["pills"]["b_new_tab"] = u["pill"]
        # Re-select while the disk is still full: refused, still stopped.
        reselect()
        time.sleep(3)
        u = ui()
        C["b_reselect_while_full_stays_stopped"] = "NOT SAVED" in (u["pill"] or "") and daemon.status()["halted"]

        # c. disk freed, re-select: LIVE again; the failure stays on the record -------
        daemon.disarm("ckpt_write")
        reselect()
        u, ok, took = wait_for(lambda u: live(u) and u["liveness"] == "advancing", 40)
        rows = sample(4)
        R["samples"]["c_recovered"] = rows
        C["c_live_and_advancing"] = ok and rows[-1]["lastStep"] > rows[0]["lastStep"]
        C["c_incomplete_banner_retained"] = all("RESULT INCOMPLETE" in r["banner"] for r in rows)
        st = daemon.status()
        C["c_status_records_recovery"] = (st["status"] == "online" and st["result_validity"]["state"] == "incomplete"
                                          and st["result_validity"]["incidents"][0]["recovered_at"] is not None)
        R["c_banner"] = rows[-1]["banner"]
        shot("c-recovered-result-incomplete.png")

        # d. genuine CuPy CUDARuntimeError from the connectome step -------------------
        daemon.arm("brain_step", "once:CUPY")
        u, ok, took = wait_for(lambda u: "GPU/COMPUTE ERROR" in (u["pill"] or ""), 20)
        st = daemon.status()
        C["d_compute_halt_pill"] = ok
        C["d_status_compute"] = (st["halted"] is True and st["error_detail"]["failure_class"] == "compute"
                                 and st["error_detail"]["type"] == "CUDARuntimeError")
        R["pills"]["d"] = u["pill"]
        R["d_banner"] = u["banner"]
        shot("d-cuda-error-compute-halt.png")
        reselect()
        u, ok, _ = wait_for(lambda u: live(u) and u["liveness"] == "advancing", 40)
        C["d_recovers_after_reselect"] = ok

        # e. slow computer: every step takes ~3 s -------------------------------------
        daemon.arm("arena_step", "HANG3")
        rows = sample(16)
        daemon.disarm("arena_step")
        R["samples"]["e_slow_cpu"] = rows
        C["e_slow_cpu_never_not_advancing"] = all("NOT ADVANCING" not in (r["pill"] or "") for r in rows)
        C["e_slow_cpu_seen_slow"] = any(r["liveness"] == "slow" or "STEP RUNNING" in (r["pill"] or "") for r in rows)
        C["e_slow_cpu_steps_advance"] = rows[-1]["lastStep"] > rows[0]["lastStep"]
        u, ok, _ = wait_for(lambda u: live(u) and u["liveness"] == "advancing", 30)

        # f. a hanging step: slow, then NOT ADVANCING after the hard limit -----------
        daemon.arm("arena_step", "once:HANG25")
        u, ok_slow, _ = wait_for(lambda u: "STEP RUNNING" in (u["pill"] or ""), 15)
        u, ok_red, took = wait_for(lambda u: "NOT ADVANCING" in (u["pill"] or ""), 30)
        C["f_slow_then_not_advancing"] = ok_slow and ok_red
        R["timings_s"]["f_red_after"] = took
        R["pills"]["f"] = u["pill"]
        shot("f-hung-step-not-advancing.png")
        u, ok, _ = wait_for(lambda u: live(u) and u["liveness"] == "advancing", 40)
        C["f_recovers_when_step_returns"] = ok

        # g. thread death -----------------------------------------------------------
        before = ui()["lastStep"]
        daemon.arm("arena_step", "once:THREADEXIT")
        u, ok, took = wait_for(lambda u: "NOT ADVANCING" in (u["pill"] or ""), 15)
        rows = sample(5)
        st = daemon.status()
        C["g_red_pill_within_threshold"] = ok and took <= 10.0
        C["g_never_live_while_dead"] = all(not live(r) for r in rows)
        C["g_step_frozen"] = len({r["lastStep"] for r in rows}) == 1
        C["g_status_dead"] = st["liveness"]["state"] == "dead" and st["status"] == "error"
        R["timings_s"]["g_red_after"] = took
        R["pills"]["g"] = u["pill"]
        shot("g-thread-dead-not-advancing.png")
        u, ok = fresh_tab(lambda u: "NOT ADVANCING" in (u["pill"] or ""), 40, "g-new-tab-opened-while-dead.png")
        C["g_new_tab_while_dead_shows_not_advancing"] = ok and "LIVE" not in (u["pill"] or "") \
            and "DISCONNECTED" not in (u["pill"] or "")
        R["pills"]["g_new_tab"] = u["pill"]
        reselect()
        u, ok, _ = wait_for(lambda u: live(u) and u["lastStep"] and u["lastStep"] > before + 10, 60)
        C["g_reselect_restarts_and_live"] = ok
        # i. pause, transport loss, reconnect into a fault, recovery ------------------
        u, _, _ = wait_for(lambda u: live(u) and u["liveness"] == "advancing", 30)
        driver.find_element(By.ID, "btnPauseToggle").click()
        u, ok, _ = wait_for(lambda u: "PAUSED" in (u["pill"] or ""), 10)
        rows = sample(14)                                     # longer than the 10 s stall threshold
        C["i_pause_shows_paused_never_not_advancing"] = ok and all(
            "PAUSED" in (r["pill"] or "") and "NOT ADVANCING" not in (r["pill"] or "") for r in rows)
        R["pills"]["i_paused"] = rows[-1]["pill"]
        driver.find_element(By.ID, "btnPauseToggle").click()
        u, ok, _ = wait_for(lambda u: live(u) and u["liveness"] == "advancing", 30)
        C["i_resume_live"] = ok
        daemon.arm("arena_step", "once:RUNTIME")              # an honest halt (software)
        u, ok, _ = wait_for(lambda u: "HALTED" in (u["pill"] or ""), 20)
        C["i_halt_shown"] = ok
        os.kill(daemon.proc.pid, signal.SIGSTOP)              # transport loss: the daemon stops answering
        try:
            u, ok, took = wait_for(lambda u: "DISCONNECTED" in (u["pill"] or ""), 45)
            C["i_transport_loss_shows_disconnected"] = ok and "LIVE" not in (u["pill"] or "")
            R["pills"]["i_transport_loss"] = u["pill"]
            R["timings_s"]["i_disconnected_after"] = took
            shot("i-transport-loss-disconnected.png")
        finally:
            os.kill(daemon.proc.pid, signal.SIGCONT)
        u, ok, took = wait_for(lambda u: "HALTED" in (u["pill"] or ""), 60)   # reconnect INTO the fault
        C["i_reconnect_into_fault_shows_halt"] = ok and "LIVE" not in (u["pill"] or "") \
            and "LOCAL ENGINE" not in (u["pill"] or "")
        R["pills"]["i_reconnected"] = u["pill"]
        R["timings_s"]["i_reconnected_after"] = took
        shot("i-reconnected-into-halt.png")
        reselect()
        u, ok, _ = wait_for(lambda u: live(u) and u["liveness"] == "advancing", 40)
        C["i_recovery_live"] = ok

        final = ui()
        C["scientific_no_page_errors"] = not final["errors"] and not final["consoleErrors"]
        R["scientific_final_errors"] = final["errors"] + final["consoleErrors"]
        R["daemon_exit_codes"] = {"scientific": daemon.stop()}
        daemon = None

        # h. exploratory daemon: degraded, NOT SAVING, still stepping ------------------
        daemon = Daemon(args, scratch, "exploratory", ["--exploratory"])
        _, ok, _ = load()
        u = ui()
        C["h_exploratory_shown"] = ok and "EXPLORATORY" in (u["pill"] or "")
        daemon.arm("ckpt_write", "ENOSPC")
        u, ok, took = wait_for(lambda u: "EXPLORATORY · NOT SAVING" in (u["banner"] or ""), 30)
        rows = sample(6)
        R["samples"]["h_exploratory_disk_full"] = rows
        st = daemon.status()
        C["h_banner_not_saving"] = ok and all("EXPLORATORY · NOT SAVING" in r["banner"] for r in rows)
        C["h_steps_still_advance"] = rows[-1]["lastStep"] > rows[0]["lastStep"]
        C["h_pill_live_exploratory"] = all(live(r) and "EXPLORATORY" in r["pill"] for r in rows)
        C["h_status_degraded_incomplete"] = (st["status"] == "degraded" and st["mode"] == "exploratory"
                                             and st["result_validity"]["state"] == "incomplete")
        R["h_banner"] = rows[-1]["banner"]
        shot("h-exploratory-not-saving.png")
        u, ok = fresh_tab(lambda u: "EXPLORATORY · NOT SAVING" in (u["banner"] or "") and live(u), 40,
                          "h-new-tab-opened-while-degraded.png")
        C["h_new_tab_while_degraded_shows_it"] = ok
        daemon.disarm("ckpt_write")
        final = ui()
        C["exploratory_no_page_errors"] = not final["errors"] and not final["consoleErrors"]
        R["daemon_exit_codes"]["exploratory"] = daemon.stop()
        daemon = None
    finally:
        if driver is not None:
            try:
                driver.quit()
            except Exception:
                pass
        if daemon is not None:
            daemon.stop()
        web.terminate()
    R["finished_at"] = now()
    R["pass"] = bool(R["checks"]) and all(R["checks"].values())
    (out / "silent_freeze_browser_check.json").write_text(json.dumps(R, indent=1) + "\n")
    print(json.dumps({"pass": R["pass"], "checks": R["checks"]}, indent=1))
    return 0 if R["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
