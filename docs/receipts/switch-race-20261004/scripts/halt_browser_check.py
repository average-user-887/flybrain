"""Real-Firefox check: fault -> pill HALTED + status error; switch -> pill LIVE, steps advance."""
import json, pathlib, time, urllib.request
from selenium import webdriver
from selenium.webdriver.firefox.options import Options
from selenium.webdriver.firefox.service import Service
from selenium.webdriver.common.by import By

D = "http://127.0.0.1:8791"
FLAG = pathlib.Path("<scratch>/race/inject.flag")
OUT = pathlib.Path("<scratch>/race/browser/halt-check")
OUT.mkdir(parents=True, exist_ok=True)


def status():
    return json.loads(urllib.request.urlopen(D + "/api/status", timeout=10).read())


opts = Options(); opts.add_argument("-headless"); opts.add_argument("--width=1600"); opts.add_argument("--height=1000")
prof = pathlib.Path.home() / "snap" / "firefox" / "common" / "neurofly-signoff"
drv = webdriver.Firefox(options=opts, service=Service("/snap/bin/geckodriver", service_args=["--profile-root", str(prof)],
                                                      log_output=str(OUT / "geckodriver.log")))
R = {}
try:
    drv.get("http://127.0.0.1:8790/?daemon=" + D)
    pill = lambda: drv.find_element(By.ID, "clusterStatusPill").text if drv.find_elements(By.ID, "clusterStatusPill") else None
    run_state = lambda: drv.find_element(By.ID, "arenaRunState").text
    for _ in range(80):
        if pill() and "LIVE" in pill(): break
        time.sleep(0.25)
    time.sleep(2)
    R["before"] = {"pill": pill(), "status": status()["status"], "steps": status()["total_steps"]}
    FLAG.touch()
    time.sleep(3)
    s1 = status(); time.sleep(3); s2 = status()
    R["halted"] = {"pill": pill(), "pill_title": drv.find_element(By.ID, "clusterStatusPill").get_attribute("title"),
                   "run_state": run_state(), "status": s2["status"], "halted": s2.get("halted"),
                   "paused": s2["paused"], "error": s2["error"], "steps_3s_apart": [s1["total_steps"], s2["total_steps"]]}
    drv.save_screenshot(str(OUT / "halted.png"))
    # Recover through the real dashboard control: select another assay card.
    card = drv.find_element(By.CSS_SELECTOR, '[data-paradigm="y-maze"]')
    drv.execute_script("arguments[0].scrollIntoView({block:'center'})", card); card.click()
    for _ in range(80):
        if pill() and "LIVE" in pill() and status()["active_paradigm"] == "y-maze": break
        time.sleep(0.25)
    a = status(); time.sleep(3); b = status()
    R["recovered"] = {"pill": pill(), "run_state": run_state(), "status": b["status"], "halted": b.get("halted"),
                      "assay": b["active_paradigm"], "steps_3s_apart": [a["total_steps"], b["total_steps"]],
                      "cleared_errors": b.get("cleared_errors")}
    drv.save_screenshot(str(OUT / "recovered.png"))
    R["console_errors"] = drv.execute_script("return (window.neuroflyErrors && window.neuroflyErrors.list && window.neuroflyErrors.list()) || null")
finally:
    drv.quit()
h, r = R["halted"], R["recovered"]
R["checks"] = {
    "pill_live_before": "LIVE" in (R["before"]["pill"] or ""),
    "pill_halted": "HALTED" in (h["pill"] or "") and "LIVE" not in (h["pill"] or ""),
    "status_error": h["status"] == "error" and h["halted"] is True and h["paused"] is False,
    "frozen_while_halted": h["steps_3s_apart"][0] == h["steps_3s_apart"][1],
    "pill_live_after_switch": "LIVE" in (r["pill"] or "") and "HALTED" not in (r["pill"] or ""),
    "status_online_after": r["status"] == "online" and r["halted"] is False,
    "advancing_after": r["steps_3s_apart"][1] > r["steps_3s_apart"][0],
    "cleared_recorded": bool(r["cleared_errors"]) and r["cleared_errors"][-1]["cleared_by"] == "switch_paradigm"}
R["pass"] = all(R["checks"].values())
(OUT / "halt_check.json").write_text(json.dumps(R, indent=1, default=str))
print(json.dumps({"checks": R["checks"], "halted_pill": h["pill"], "run_state": h["run_state"],
                  "recovered_pill": r["pill"], "pass": R["pass"]}, indent=1))
