#!/usr/bin/env python3
"""Headless real-Firefox check of the dashboard against an ISOLATED daemon.

Run with a Python that has ``selenium`` installed (kept out of the project venv):

    <bvenv>/bin/python scripts/browser_firefox_check.py --work-dir /tmp/job \
        --receipt outputs/rethink-audit/firefox_headless_receipt.json

It starts its own daemon (``--output-dir``/``--data-dir``/``--pid-file`` under
``--work-dir``, port >= 19000) and its own static web server, drives Firefox through
geckodriver, and records what it observed. Scenarios: live 1x load, 100x via the
real speed selector for ``--seconds`` wall seconds, pause/resume, the three
test-build fault injections (``?inject=malformed|apply|render``) and a daemon
disconnect/restart.

A headless check is NOT the final live-UI sign-off required by AGENTS.md; it
supplements a person clicking the real controls in the running browser.
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.firefox.options import Options
from selenium.webdriver.firefox.service import Service
from selenium.webdriver.support.ui import Select

ROOT = Path(__file__).resolve().parents[1]
PROJECT_PY = ROOT / ".venv" / "bin" / "python"


class _SpeedOnlyDone(Exception):
    pass


def free_port(start):
    for port in range(start, start + 500):
        with socket.socket() as s:
            try:
                s.bind(("127.0.0.1", port))
                return port
            except OSError:
                continue
    raise RuntimeError("no free port")


def start_daemon(work: Path, port: int, tag: str, script: Path = ROOT / "neurofly_daemon.py"):
    out = work / f"daemon-{tag}"
    out.mkdir(parents=True, exist_ok=True)
    proc = subprocess.Popen(
        [str(PROJECT_PY), str(script), "--host", "127.0.0.1", "--port", str(port),
         "--paradigm", "wind-tunnel", "--speed", "1", "--checkpoint-interval", "3600",
         "--output-dir", str(out / "outputs"), "--data-dir", str(out / "data"), "--pid-file", str(out / "daemon.pid")],
        cwd=str(work), stdout=open(out / "daemon.log", "w"), stderr=subprocess.STDOUT,
        env=dict(os.environ, PYTHONPATH=str(ROOT)))
    deadline = time.time() + 30
    while time.time() < deadline:
        try:
            status(port)
            return proc
        except Exception:
            time.sleep(0.2)
    raise RuntimeError(f"daemon did not start; see {out / 'daemon.log'}")


def status(port, timeout=3):
    with urllib.request.urlopen(f"http://127.0.0.1:{port}/api/status", timeout=timeout) as r:
        return json.loads(r.read())


def stop(proc):
    if proc and proc.poll() is None:
        proc.send_signal(signal.SIGTERM)
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()


def ui(driver):
    return driver.execute_script("""
        const t = id => document.getElementById(id)?.textContent ?? null;
        const d = window.neuroflyDiagnostics;
        return {pill: t('clusterStatusPill'), step: t('statStep'), simTime: t('statSimTime'),
                speed: t('statSpeed'), achieved: t('statAchieved'), dataAge: t('statDataAge'),
                runState: t('arenaRunState'), banner: document.getElementById('errorBanner')?.style.display === 'none' ? '' : t('errorBannerText'),
                resumeVisible: !document.getElementById('btnErrorResume')?.hidden,
                dataAgeMs: d ? d.dataAgeMs() : null, lastStep: d ? d.lastStep() : null,
                errors: (window.neuroflyErrors?.log || []).map(e => ({phase: e.phase, message: e.message, assay: e.assay, run_id: e.run_id, step: e.step, count: e.count})),
                suspended: window.neuroflyErrors ? [...window.neuroflyErrors.suspended] : null,
                remoteDriven: !!window.arena?.remoteDriven, trail: window.arena?.fly?.trail?.length ?? null};
    """)


def wait_for(driver, pred, timeout=20.0):
    end = time.time() + timeout
    last = None
    while time.time() < end:
        last = ui(driver)
        if pred(last):
            return last
        time.sleep(0.25)
    return last


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work-dir", required=True)
    ap.add_argument("--receipt", required=True)
    ap.add_argument("--seconds", type=float, default=60.0)
    ap.add_argument("--geckodriver", default="/snap/bin/geckodriver")
    ap.add_argument("--web-root", default=str(ROOT / "web"))
    ap.add_argument("--daemon-script", default=str(ROOT / "neurofly_daemon.py"))
    ap.add_argument("--speed-only", action="store_true",
                    help="only the 1x load and the 100x observation (e.g. to reproduce the baseline freeze)")
    args = ap.parse_args()
    work = Path(args.work_dir).resolve()
    work.mkdir(parents=True, exist_ok=True)
    dport, wport = free_port(19100), free_port(19300)
    receipt = {"scenarios": {}, "note": "Headless real Firefox via geckodriver against an isolated daemon; "
               "not the final live-UI sign-off required by AGENTS.md."}
    daemon = start_daemon(work, dport, "a", Path(args.daemon_script))
    receipt["daemon_script"], receipt["web_root"] = args.daemon_script, args.web_root
    web = subprocess.Popen([sys.executable, "-m", "http.server", str(wport), "--bind", "127.0.0.1",
                            "--directory", args.web_root], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    # Snap Firefox cannot read hidden directories under $HOME or /tmp; use its own area.
    profile_root = Path.home() / "snap" / "firefox" / "common" / "neurofly-headless"
    profile_root.mkdir(parents=True, exist_ok=True)
    opts = Options()
    opts.add_argument("-headless")
    opts.add_argument("--width=1440")
    opts.add_argument("--height=900")
    service = Service(executable_path=args.geckodriver, service_args=["--profile-root", str(profile_root)],
                      log_output=str(work / "geckodriver.log"))
    driver = webdriver.Firefox(options=opts, service=service)
    base = f"http://127.0.0.1:{wport}/index.html?daemon=http%3A%2F%2F127.0.0.1%3A{dport}"
    try:
        caps = driver.capabilities
        receipt["browser"] = {"name": caps.get("browserName"), "version": caps.get("browserVersion"),
                              "platform": caps.get("platformName"), "headless": caps.get("moz:headless"),
                              "user_agent": driver.execute_script("return navigator.userAgent"),
                              "viewport": driver.execute_script("return [innerWidth, innerHeight, devicePixelRatio]"),
                              "geckodriver": args.geckodriver}
        receipt["daemon_port"], receipt["web_port"] = dport, wport

        # 1. Live load at 1x
        driver.get(base)
        s = wait_for(driver, lambda u: u["pill"] and "LIVE" in u["pill"] and u["step"] not in (None, "0"))
        receipt["scenarios"]["load_1x"] = {"ui": s, "pass": bool(s["pill"] and "LIVE" in s["pill"] and not s["errors"])}

        # 2. 100x through the real speed selector, observed for --seconds
        Select(driver.find_element(By.ID, "selectSpeed")).select_by_value("100")
        samples, t_end = [], time.time() + args.seconds
        while time.time() < t_end:
            u = ui(driver)
            t0 = time.perf_counter()
            try:
                st = status(dport, timeout=10)
            except Exception as exc:
                st = {"error": f"{type(exc).__name__}: {exc}"}
            ui_step = u["lastStep"] if u["lastStep"] is not None else (int(u["step"]) if (u["step"] or "").isdigit() else None)
            samples.append({"t": round(time.time(), 2), "pill": u["pill"], "ui_step": ui_step,
                            "server_step": st.get("total_steps"), "status_ms": round((time.perf_counter() - t0) * 1e3, 1),
                            "data_age_ms": u["dataAgeMs"], "achieved": u["achieved"],
                            "requested": st.get("sim_speed"), "server_achieved": (st.get("timing") or {}).get("achieved_speed")})
            time.sleep(1.0)
        ages = [x["data_age_ms"] for x in samples if x["data_age_ms"] is not None]
        lag = [x["server_step"] - x["ui_step"] for x in samples if x["ui_step"] is not None and x["server_step"] is not None]
        disconnected = [x for x in samples if "DISCONNECTED" in (x["pill"] or "")]
        stale = [x for x in samples if "STALE" in (x["pill"] or "")]
        receipt["scenarios"]["speed_100x"] = {
            "seconds": args.seconds, "samples": samples,
            "max_data_age_ms": round(max(ages), 1) if ages else None,
            "max_step_lag_ui_vs_server": max(lag) if lag else None,
            "disconnected_samples": len(disconnected), "stale_samples": len(stale),
            "ui_steps_advanced": samples[-1]["ui_step"] - samples[0]["ui_step"] if samples else 0,
            "server_timing": status(dport, timeout=10).get("timing"),
            "errors": ui(driver)["errors"],
            "pass": bool(samples) and bool(ages) and not disconnected and not stale and max(ages) < 3000,
        }
        if args.speed_only:
            raise _SpeedOnlyDone()

        # 3. Pause / resume through the real button
        driver.find_element(By.ID, "btnPauseToggle").click()
        p1 = wait_for(driver, lambda u: "PAUSED" in (u["runState"] or ""), 10)
        time.sleep(2.0)
        p2 = ui(driver)
        paused_status = status(dport)
        driver.find_element(By.ID, "btnPauseToggle").click()
        r1 = wait_for(driver, lambda u: u["lastStep"] and p2["lastStep"] and u["lastStep"] > p2["lastStep"], 10)
        receipt["scenarios"]["pause_resume"] = {
            "paused_ui": p1, "after_2s": p2, "server_paused": paused_status["paused"],
            "server_step_vs_ui": [paused_status["total_steps"], p2["lastStep"]], "resumed_ui": r1,
            "pass": bool("PAUSED" in (p1["runState"] or "") and p1["lastStep"] == p2["lastStep"]
                         and "DISCONNECTED" not in (p2["pill"] or "") and r1["lastStep"] > p2["lastStep"])}
        Select(driver.find_element(By.ID, "selectSpeed")).select_by_value("1")

        # 4. Fault injections (one-shot, test build)
        for mode, phases in (("malformed", {"packet-parse", "packet-validate"}), ("apply", {"packet-apply"}),
                             ("render", {"render"})):
            driver.get(base + f"&inject={mode}")
            seen = wait_for(driver, lambda u: phases <= {e["phase"] for e in u["errors"]}, 20)
            got = {e["phase"] for e in seen["errors"]}
            entry = {"errors": seen["errors"], "banner": seen["banner"], "suspended": seen["suspended"],
                     "resume_button_visible": seen["resumeVisible"]}
            step_at_error = seen["lastStep"]
            if mode == "render":
                driver.find_element(By.ID, "btnErrorResume").click()
            time.sleep(3.0)
            after = ui(driver)
            entry["after_recovery"] = {k: after[k] for k in ("pill", "lastStep", "dataAgeMs", "banner", "suspended")}
            entry["errors_after_recovery"] = len(after["errors"])
            well_formed = all(e["assay"] and e["run_id"] and e["step"] is not None for e in seen["errors"])
            entry["pass"] = bool(phases <= got and well_formed and seen["banner"]
                                 and after["lastStep"] and step_at_error is not None and after["lastStep"] > step_at_error
                                 and not after["suspended"] and after["dataAgeMs"] is not None and after["dataAgeMs"] < 3000
                                 and len(after["errors"]) == len(seen["errors"]))
            receipt["scenarios"][f"inject_{mode}"] = entry

        # 5. Disconnect (stop my own isolated daemon) and restart
        driver.get(base)
        live = wait_for(driver, lambda u: u["pill"] and "LIVE" in u["pill"] and u["lastStep"], 20)
        time.sleep(1.5)
        before = ui(driver)
        stop(daemon)
        d1 = wait_for(driver, lambda u: "DISCONNECTED" in (u["pill"] or ""), 30)
        time.sleep(3.0)
        d2 = ui(driver)
        daemon = start_daemon(work, dport, "b", Path(args.daemon_script))
        rc = wait_for(driver, lambda u: "LIVE" in (u["pill"] or "") and u["dataAgeMs"] is not None and u["dataAgeMs"] < 2000, 60)
        receipt["scenarios"]["disconnect_reconnect"] = {
            "before": before, "disconnected": d1, "disconnected_after_3s": d2, "reconnected": rc,
            "pass": bool("DISCONNECTED" in (d1["pill"] or "")
                         and d2["lastStep"] == d1["lastStep"] and d2["step"] == d1["step"]
                         and d2["remoteDriven"] and "(frozen)" in (d2["dataAge"] or "")
                         and d2["runState"] == d1["runState"] and "LIVE" in (rc["pill"] or ""))}
    except _SpeedOnlyDone:
        pass
    finally:
        try:
            driver.quit()
        finally:
            stop(daemon)
            web.terminate()
            web.wait(timeout=10)
    receipt["pass"] = all(v.get("pass") for v in receipt["scenarios"].values())
    Path(args.receipt).parent.mkdir(parents=True, exist_ok=True)
    Path(args.receipt).write_text(json.dumps(receipt, indent=2) + "\n")
    summary = {k: v.get("pass") for k, v in receipt["scenarios"].items()}
    print(json.dumps({"browser": receipt["browser"], "scenarios": summary, "pass": receipt["pass"]}, indent=2))


if __name__ == "__main__":
    main()
