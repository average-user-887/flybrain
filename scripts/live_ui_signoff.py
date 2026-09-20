#!/usr/bin/env python3
"""AGENTS.md live-UI sign-off against the OWNER'S RUNNING observatory.

Unlike scripts/browser_firefox_check.py this starts no daemon and no web server:
it drives real Firefox (geckodriver + selenium) against the already-running
services and only ever uses the real controls in the page. It never stops or
restarts the daemon and never writes under outputs/.

    <venv>/bin/python scripts/live_ui_signoff.py \
        --web http://127.0.0.1:8780/ --daemon http://127.0.0.1:8781 \
        --out docs/receipts/live-signoff
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.firefox.options import Options
from selenium.webdriver.firefox.service import Service
from selenium.webdriver.support.ui import Select

ASSAYS = ["open-arena", "t-maze", "y-maze", "heat-maze", "buridan", "visual-operant",
          "wind-tunnel", "looming-escape", "optomotor", "gap-crossing", "circadian-dam",
          "courtship", "labyrinth", "multisensory-sandbox"]

UI_JS = """
const t = id => document.getElementById(id)?.textContent ?? null;
const d = window.neuroflyDiagnostics;
const a = window.arena;
return {
  pill: t('clusterStatusPill'), badge: t('navbarParadigmBadge'),
  step: t('statStep'), simTime: t('statSimTime'), speed: t('statSpeed'),
  achieved: t('statAchieved'), dataAge: t('statDataAge'), fps: t('statFps'),
  runState: t('arenaRunState'),
  pauseLabel: document.getElementById('btnPauseToggle')?.textContent ?? null,
  banner: document.getElementById('errorBanner')?.style.display === 'none' ? '' : t('errorBannerText'),
  dataAgeMs: d ? d.dataAgeMs() : null, lastStep: d ? d.lastStep() : null,
  errors: (window.neuroflyErrors?.log || []).map(e => ({phase: e.phase, message: e.message, assay: e.assay, step: e.step, count: e.count})),
  consoleErrors: (window.__signoffConsole || []).slice(),
  toolsPanel: {present: !!document.querySelector('.arena-toolbar, #toolSelect'),
               buttons: [...document.querySelectorAll('.tool-btn')].map(b => b.textContent.trim()),
               visible: !!document.getElementById('toolSelect')?.offsetParent},
  arena: {assay: a?.activeParadigmId ?? null, remoteDriven: !!a?.remoteDriven,
          bounds: a?.worldBounds ?? null, walls: a?.currentWalls?.length ?? null,
          fly: a?.fly ? {x: a.fly.x, y: a.fly.y} : null,
          packetAssay: a?.remotePacket?.paradigm ?? null,
          packetBounds: a?.remotePacket?.world_bounds ?? null,
          packetStep: a?.remotePacket?.step ?? null},
  ident: {backend: t('identBackend'), label: t('identLabel'), assists: t('identAssists'),
          motor: t('identMotor'), fault: t('identFault'), run: t('identRun'),
          assistance: t('identAssistance'), optoMap: t('identOptoMap'),
          banner: document.getElementById('identityBanner')?.style.display === 'none' ? '' : t('identityBanner'),
          packetRun: a?.remotePacket?.identity?.run_id ?? null,
          packetActivation: a?.remotePacket?.identity?.activation ?? null,
          packetAssay: a?.remotePacket?.identity?.assay ?? null,
          ackActivation: d?.bridge?.lastSwitchAck?.identity?.activation ?? null,
          ackAssay: d?.bridge?.lastSwitchAck?.identity?.assay ?? null,
          rejected: d?.bridge?.rejectedIdentityPackets ?? null}};
"""

CONSOLE_HOOK = """
if (!window.__signoffConsole) {
  window.__signoffConsole = [];
  const orig = console.error.bind(console);
  console.error = (...a) => { try { window.__signoffConsole.push(a.map(String).join(' ').slice(0, 400)); } catch (e) {} orig(...a); };
}
return true;
"""


def now():
    return datetime.now(timezone.utc).isoformat()


class Ctx:
    def __init__(self, driver, daemon):
        self.driver, self.daemon = driver, daemon

    def ui(self):
        return self.driver.execute_script(UI_JS)

    def status(self, timeout=8):
        with urllib.request.urlopen(f"{self.daemon}/api/status", timeout=timeout) as r:
            return json.loads(r.read())

    def wait_for(self, pred, timeout=25.0, poll=0.25):
        end, last = time.time() + timeout, None
        while time.time() < end:
            last = self.ui()
            try:
                if pred(last):
                    return last, True
            except Exception:
                pass
            time.sleep(poll)
        return last, False

    def canvas_hash(self):
        url = self.driver.execute_script(
            "const c=document.getElementById('arenaCanvas');return c?c.toDataURL():null;")
        if not url:
            return None, None
        return hashlib.sha256(url.encode()).hexdigest()[:16], len(url)

    def shot(self, out: Path, name):
        p = out / name
        self.driver.save_screenshot(str(p))
        return {"file": str(p), "bytes": p.stat().st_size}


def inside(bounds, fly, tol=3.0):
    if not bounds or not fly:
        return None
    return (bounds["minX"] - tol <= fly["x"] <= bounds["maxX"] + tol
            and bounds["minY"] - tol <= fly["y"] <= bounds["maxY"] + tol)


def click_card(driver, assay):
    el = driver.find_element(By.CSS_SELECTOR, f'.experiment-card[data-paradigm="{assay}"]')
    driver.execute_script("arguments[0].scrollIntoView({block:'center'});", el)
    time.sleep(0.15)
    el.click()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--web", default="http://127.0.0.1:8780/")
    ap.add_argument("--daemon", default="http://127.0.0.1:8781")
    ap.add_argument("--out", default="docs/receipts/live-signoff")
    ap.add_argument("--geckodriver", default="/snap/bin/geckodriver")
    ap.add_argument("--restore-assay", default="open-arena")
    ap.add_argument("--restore-speed", default="1")
    args = ap.parse_args()
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)

    R = {"kind": "live-ui-signoff", "started_at": now(), "web_url": args.web, "daemon_url": args.daemon,
         "automated_real_firefox": True, "scenarios": {}, "screenshots": []}

    profile_root = Path.home() / "snap" / "firefox" / "common" / "neurofly-signoff"
    profile_root.mkdir(parents=True, exist_ok=True)
    opts = Options()
    opts.add_argument("-headless")
    opts.add_argument("--width=1600")
    opts.add_argument("--height=1000")
    service = Service(executable_path=args.geckodriver,
                      service_args=["--profile-root", str(profile_root)],
                      log_output=str(out / "geckodriver.log"))
    driver = webdriver.Firefox(options=opts, service=service)
    c = Ctx(driver, args.daemon)
    try:
        caps = driver.capabilities
        R["browser"] = {"name": caps.get("browserName"), "version": caps.get("browserVersion"),
                        "platform": caps.get("platformName"), "headless": caps.get("moz:headless"),
                        "user_agent": driver.execute_script("return navigator.userAgent"),
                        "geckodriver": args.geckodriver,
                        "firefox_binary": caps.get("moz:geckodriverVersion")}
        R["status_at_start"] = c.status()

        # ---- 1. fresh load -------------------------------------------------
        t0 = now()
        driver.get(args.web)
        driver.execute_script(CONSOLE_HOOK)
        s1, ok = c.wait_for(lambda u: u["pill"] and "LIVE" in u["pill"] and u["lastStep"], 40)
        R["browser"]["viewport"] = driver.execute_script("return [innerWidth, innerHeight, devicePixelRatio]")
        R["browser"]["loaded_url"] = driver.current_url
        time.sleep(3.0)
        s2 = c.ui()
        st = c.status()
        ch, clen = c.canvas_hash()
        R["screenshots"].append(c.shot(out, "01-initial-load.png"))
        R["scenarios"]["initial_load"] = {
            "at": t0, "live_reached": ok, "first": s1, "after_3s": s2, "daemon_status": {
                k: st[k] for k in ("active_paradigm", "total_steps", "sim_speed", "paused")},
            "canvas_sha256_16": ch, "canvas_data_len": clen,
            "checks": {
                "pill_live": bool(s2["pill"] and "LIVE" in s2["pill"]),
                "assay_matches_daemon": s2["arena"]["packetAssay"] == st["active_paradigm"],
                "clock_advanced": s2["simTime"] != s1["simTime"],
                "steps_advanced": bool(s2["lastStep"] and s1["lastStep"] and s2["lastStep"] > s1["lastStep"]),
                "identity_backend_modular": s2["ident"]["backend"] == "modular",
                "identity_label_present": bool(s2["ident"]["label"]),
                "achieved_live": bool(s2["achieved"]) and s2["achieved"] != "--",
                "data_age_live": s2["dataAgeMs"] is not None and s2["dataAgeMs"] < 3000,
                "tools_panel": bool(s2["toolsPanel"]["present"] and s2["toolsPanel"]["buttons"]),
                "arena_rendered": bool(clen and clen > 5000),
                "no_errors": not s2["errors"] and not s2["consoleErrors"]}}
        R["scenarios"]["initial_load"]["pass"] = all(R["scenarios"]["initial_load"]["checks"].values())

        # ---- 2. all 14 assays ---------------------------------------------
        rows, prev_hash = [], ch
        for assay in ASSAYS:
            before = c.ui()
            before_steps = c.status()["total_steps"]
            click_card(driver, assay)
            u, ok = c.wait_for(lambda x: x["arena"]["packetAssay"] == assay
                               and x["ident"]["ackAssay"] == assay
                               and x["ident"]["ackActivation"] == x["ident"]["packetActivation"], 30)
            time.sleep(2.0)
            after = c.ui()
            st = c.status()
            h, ln = c.canvas_hash()
            row = {"assay": assay, "switch_ok": ok, "at": now(),
                   "badge": after["badge"], "arena_assay": after["arena"]["assay"],
                   "packet_assay": after["arena"]["packetAssay"],
                   "daemon_assay": st["active_paradigm"],
                   "bounds": after["arena"]["bounds"], "packet_bounds": after["arena"]["packetBounds"],
                   "walls": after["arena"]["walls"], "fly": after["arena"]["fly"],
                   "ui_step_before": before["lastStep"], "ui_step_after": after["lastStep"],
                   "daemon_steps_before": before_steps, "daemon_steps_after": st["total_steps"],
                   "identity": {k: after["ident"][k] for k in
                                ("backend", "label", "assists", "motor", "fault", "assistance",
                                 "packetActivation", "ackActivation", "packetAssay", "rejected", "banner")},
                   "canvas_sha256_16": h, "canvas_changed_from_prev": h != prev_hash,
                   "data_age_ms": after["dataAgeMs"], "pill": after["pill"],
                   "errors": after["errors"], "console_errors": after["consoleErrors"],
                   "fly_inside_bounds": inside(after["arena"]["bounds"], after["arena"]["fly"])}
            row["checks"] = {
                "acked": ok,
                "ui_matches_daemon": row["packet_assay"] == row["daemon_assay"] == assay,
                "identity_assay_matches": after["ident"]["packetAssay"] == assay,
                "activation_current": after["ident"]["ackActivation"] == after["ident"]["packetActivation"],
                "steps_advancing": bool(after["lastStep"] and before["lastStep"] and after["lastStep"] > before["lastStep"]),
                "daemon_steps_monotonic": st["total_steps"] > before_steps,
                "fly_inside": row["fly_inside_bounds"] is True,
                "live": bool(after["pill"] and "LIVE" in after["pill"]),
                "no_errors": not after["errors"] and not after["consoleErrors"]}
            row["pass"] = all(row["checks"].values())
            prev_hash = h
            if assay in ("labyrinth", "optomotor"):
                R["screenshots"].append(c.shot(out, f"02-assay-{assay}.png"))
            rows.append(row)
        R["scenarios"]["all_assays"] = {"rows": rows, "pass": all(r["pass"] for r in rows),
                                        "distinct_canvases": len({r["canvas_sha256_16"] for r in rows}),
                                        "distinct_bounds": len({json.dumps(r["bounds"]) for r in rows})}
        R["screenshots"].append(c.shot(out, "03-identity-bar.png"))

        # ---- 3. rapid consecutive selections -------------------------------
        seq = ["t-maze", "wind-tunnel", "looming-escape", "heat-maze", "y-maze"]
        steps_before = c.status()["total_steps"]
        rej_before = c.ui()["ident"]["rejected"]
        t_rapid = now()
        for a in seq:
            click_card(driver, a)
            time.sleep(0.35)
        settle, ok = c.wait_for(lambda x: x["arena"]["packetAssay"] == seq[-1]
                                and x["ident"]["ackAssay"] == seq[-1]
                                and x["ident"]["ackActivation"] == x["ident"]["packetActivation"], 30)
        obs = []
        for _ in range(12):
            u = c.ui()
            obs.append({"packetAssay": u["arena"]["packetAssay"], "arenaAssay": u["arena"]["assay"],
                        "badge": u["badge"], "step": u["lastStep"],
                        "act": u["ident"]["packetActivation"], "ack": u["ident"]["ackActivation"]})
            time.sleep(0.4)
        st = c.status()
        R["scenarios"]["rapid_selection"] = {
            "at": t_rapid, "sequence": seq, "settled": ok, "settle_ui": settle,
            "post_settle_samples": obs, "daemon_assay": st["active_paradigm"],
            "daemon_steps_before": steps_before, "daemon_steps_after": st["total_steps"],
            "rejected_before": rej_before, "rejected_after": settle["ident"]["rejected"],
            "checks": {"settled_on_last": ok and settle["arena"]["packetAssay"] == seq[-1],
                       "daemon_agrees": st["active_paradigm"] == seq[-1],
                       "no_stale_after_settle": all(o["packetAssay"] == seq[-1] for o in obs),
                       "activation_current": all(o["act"] == o["ack"] for o in obs),
                       "no_hidden_reset": st["total_steps"] > steps_before,
                       "steps_monotonic_in_ui": all(obs[i]["step"] <= obs[i + 1]["step"] for i in range(len(obs) - 1)),
                       "no_errors": not settle["errors"] and not settle["consoleErrors"]}}
        R["scenarios"]["rapid_selection"]["pass"] = all(R["scenarios"]["rapid_selection"]["checks"].values())
        R["screenshots"].append(c.shot(out, "04-after-rapid.png"))

        # ---- 4. two-tab synchronisation ------------------------------------
        tab_a = driver.current_window_handle
        driver.switch_to.new_window("tab")
        tab_b = driver.current_window_handle
        driver.get(args.web)
        driver.execute_script(CONSOLE_HOOK)
        b_live, okb = c.wait_for(lambda u: u["pill"] and "LIVE" in u["pill"] and u["lastStep"], 40)
        click_card(driver, "buridan")
        b_sw, okbs = c.wait_for(lambda u: u["arena"]["packetAssay"] == "buridan"
                                and u["ident"]["ackAssay"] == "buridan", 30)
        time.sleep(2.0)
        b_after = c.ui()
        driver.switch_to.window(tab_a)
        a_after, oka = c.wait_for(lambda u: u["arena"]["packetAssay"] == "buridan", 30)
        time.sleep(1.5)
        a_final = c.ui()
        R["screenshots"].append(c.shot(out, "05-tabA-follows-tabB.png"))
        R["scenarios"]["two_tab_sync"] = {
            "tab_b_live": okb, "tab_b_switch_ok": okbs, "tab_b": b_after,
            "tab_a_followed": oka, "tab_a": a_final, "daemon": c.status()["active_paradigm"],
            "checks": {"tab_b_switched": b_after["arena"]["packetAssay"] == "buridan",
                       "tab_a_followed": a_final["arena"]["packetAssay"] == "buridan",
                       "tab_a_badge": "BURIDAN" in (a_final["badge"] or "").upper(),
                       "tab_a_still_live": bool(a_final["pill"] and "LIVE" in a_final["pill"]),
                       "tab_a_steps_advancing": bool(a_final["lastStep"] and a_after["lastStep"]
                                                     and a_final["lastStep"] >= a_after["lastStep"]),
                       "tab_a_no_errors": not a_final["errors"] and not a_final["consoleErrors"]}}
        R["scenarios"]["two_tab_sync"]["pass"] = all(R["scenarios"]["two_tab_sync"]["checks"].values())
        driver.switch_to.window(tab_b)
        driver.close()
        driver.switch_to.window(tab_a)

        # ---- 5. speed control ----------------------------------------------
        speeds = []
        for value in ("1", "20", "100"):
            Select(driver.find_element(By.ID, "selectSpeed")).select_by_value(value)
            time.sleep(6.0)
            samples = []
            for _ in range(5):
                u = c.ui()
                st = c.status()
                samples.append({"pill": u["pill"], "ui_speed": u["speed"], "ui_achieved": u["achieved"],
                                "ui_data_age_ms": u["dataAgeMs"], "ui_step": u["lastStep"],
                                "server_requested": st["timing"]["requested_speed"],
                                "server_achieved": st["timing"]["achieved_speed"],
                                "server_overloaded": st["timing"]["overloaded"],
                                "server_step": st["total_steps"]})
                time.sleep(1.0)
            entry = {"requested": float(value), "samples": samples,
                     "checks": {
                         "server_requested_matches": all(s["server_requested"] == float(value) for s in samples),
                         "ui_shows_requested": all((s["ui_speed"] or "").startswith(value) for s in samples),
                         "achieved_reported": all(s["ui_achieved"] and s["ui_achieved"] != "--" for s in samples),
                         "never_disconnected": not any("DISCONNECTED" in (s["pill"] or "") for s in samples),
                         "view_not_frozen": samples[-1]["ui_step"] > samples[0]["ui_step"],
                         "data_fresh": all(s["ui_data_age_ms"] is not None and s["ui_data_age_ms"] < 3000 for s in samples)}}
            entry["achieved_vs_requested"] = [s["server_achieved"] for s in samples]
            entry["pass"] = all(entry["checks"].values())
            speeds.append(entry)
            if value == "100":
                R["screenshots"].append(c.shot(out, "06-speed-100x.png"))
        R["scenarios"]["speed_control"] = {"steps": speeds, "pass": all(s["pass"] for s in speeds),
                                           "errors": c.ui()["errors"]}

        # ---- 6. pause / resume ----------------------------------------------
        Select(driver.find_element(By.ID, "selectSpeed")).select_by_value("1")
        time.sleep(2.0)
        pre = c.ui()
        driver.find_element(By.ID, "btnPauseToggle").click()
        p1, okp = c.wait_for(lambda u: "PAUSED" in (u["runState"] or "").upper()
                             or "RESUME" in (u["pauseLabel"] or "").upper(), 15)
        time.sleep(4.0)
        p2 = c.ui()
        paused_status = c.status()
        R["screenshots"].append(c.shot(out, "07-paused.png"))
        driver.find_element(By.ID, "btnPauseToggle").click()
        r1, okr = c.wait_for(lambda u: u["lastStep"] and p2["lastStep"] and u["lastStep"] > p2["lastStep"], 20)
        resumed_status = c.status()
        R["scenarios"]["pause_resume"] = {
            "before": pre, "paused_first": p1, "paused_after_4s": p2, "resumed": r1,
            "server_paused": paused_status["paused"], "server_resumed_paused": resumed_status["paused"],
            "server_steps_paused": paused_status["total_steps"],
            "checks": {"ui_shows_paused": okp,
                       "server_paused": paused_status["paused"] is True,
                       "steps_frozen": p1["lastStep"] == p2["lastStep"],
                       "clock_frozen": p1["simTime"] == p2["simTime"],
                       "metrics_retained": bool(p2["ident"]["backend"] == "modular" and p2["step"] not in (None, "")),
                       "not_disconnected_flag": "DISCONNECTED" not in (p2["pill"] or ""),
                       "resumed": okr, "server_running_again": resumed_status["paused"] is False,
                       "no_errors": not r1["errors"] and not r1["consoleErrors"]}}
        R["scenarios"]["pause_resume"]["pass"] = all(R["scenarios"]["pause_resume"]["checks"].values())

        # ---- 7. restore owner state -----------------------------------------
        click_card(driver, args.restore_assay)
        rest, okrest = c.wait_for(lambda u: u["arena"]["packetAssay"] == args.restore_assay
                                  and u["ident"]["ackAssay"] == args.restore_assay, 30)
        Select(driver.find_element(By.ID, "selectSpeed")).select_by_value(args.restore_speed)
        time.sleep(4.0)
        final_ui = c.ui()
        final_status = c.status()
        R["screenshots"].append(c.shot(out, "08-restored-open-arena-1x.png"))
        R["scenarios"]["restore"] = {
            "target": {"assay": args.restore_assay, "sim_speed": float(args.restore_speed), "paused": False},
            "ui": final_ui,
            "daemon_status": {k: final_status[k] for k in ("active_paradigm", "sim_speed", "paused", "total_steps")},
            "checks": {"assay_restored": final_status["active_paradigm"] == args.restore_assay,
                       "speed_restored": final_status["sim_speed"] == float(args.restore_speed),
                       "running": final_status["paused"] is False,
                       "ui_agrees": final_ui["arena"]["packetAssay"] == args.restore_assay,
                       "ui_live": bool(final_ui["pill"] and "LIVE" in final_ui["pill"]),
                       "acked": okrest}}
        R["scenarios"]["restore"]["pass"] = all(R["scenarios"]["restore"]["checks"].values())
        R["final_console_errors"] = final_ui["consoleErrors"]
        R["final_app_errors"] = final_ui["errors"]
    finally:
        try:
            driver.quit()
        except Exception:
            pass
    R["finished_at"] = now()
    R["pass"] = all(v.get("pass") for v in R["scenarios"].values())
    (out / "live_ui_signoff.json").write_text(json.dumps(R, indent=2) + "\n")
    print(json.dumps({"pass": R["pass"],
                      "scenarios": {k: v.get("pass") for k, v in R["scenarios"].items()},
                      "assay_failures": [r["assay"] for r in R["scenarios"].get("all_assays", {}).get("rows", [])
                                         if not r["pass"]],
                      "screenshots": [s["file"] for s in R["screenshots"]]}, indent=2))


if __name__ == "__main__":
    main()
