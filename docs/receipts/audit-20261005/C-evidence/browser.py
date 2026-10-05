#!/usr/bin/env python3
"""Real Firefox (headless) walk over the 14 assays on the audit daemon (8831/8830)."""
import json, os, sys, time
from pathlib import Path
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.firefox.options import Options
from selenium.webdriver.firefox.service import Service

OUT = Path("<SCRATCH>/browser")
OUT.mkdir(parents=True, exist_ok=True)
URL = "http://127.0.0.1:8830/index.html?daemon=http://127.0.0.1:8831"
ASSAYS = os.environ.get("ASSAYS", "open-arena,t-maze,y-maze,heat-maze,buridan,visual-operant,wind-tunnel,"
                        "looming-escape,optomotor,gap-crossing,circadian-dam,courtship,labyrinth,"
                        "multisensory-sandbox").split(",")
DWELL = float(os.environ.get("DWELL", "30"))

JS = r"""
const t = id => { const e = document.getElementById(id); return e ? e.textContent.trim() : null; };
const vis = id => { const e = document.getElementById(id); return !!(e && e.offsetParent && getComputedStyle(e).display !== 'none'); };
const a = window.arena, p = a && a.remotePacket;
const card = document.querySelector(`.experiment-card[data-paradigm="${arguments[0]}"]`);
return {
  wall: Date.now(), runState: t('arenaRunState'), pill: t('clusterStatusPill'),
  identBackend: t('identBackend'), identMotor: t('identMotor'), identAssists: t('identAssists'),
  identAssistance: t('identAssistance'), identFault: t('identFault'),
  identityBanner: vis('identityBanner') ? t('identityBanner') : '',
  errorBanner: vis('errorBanner') ? t('errorBannerText') : '',
  badge: t('controllerBadge'), achieved: t('statAchieved'), step: t('statStep'),
  fly: a && a.fly ? [Math.round(a.fly.x*100)/100, Math.round(a.fly.y*100)/100, Math.round((a.fly.heading||0)*100)/100] : null,
  pktParadigm: p ? p.paradigm : null, pktStep: p ? p.step : null, pktState: p && p.fly ? p.fly.state : null,
  pktSpeed: p && p.fly ? p.fly.speed : null,
  spikes: p && p.connectome ? p.connectome.total_spikes : null,
  dn: { l: t('hudDna02L'), r: t('hudDna02R'), dnp09: t('hudDnp09'), mdn: t('hudMdn'), gf: t('hudGf') },
  activityGrouping: t('activityGrouping'), activity: (t('activityRegions') || '').slice(0, 600),
  kcActive: t('valKcActive'), netValence: t('valNetValence'), compassSource: t('valCompassSource'),
  epgBump: t('valEpgBump'), dopamine: t('valDopamineState'), cpg: t('valCpgFreq'),
  card: card ? card.textContent.replace(/\s+/g, ' ').trim() : null,
  assayTools: vis('assayToolsPanel') ? (t('assayToolsPanel') || '').replace(/\s+/g, ' ').slice(0, 800) : '',
  errors: (window.neuroflyErrors?.log || window.neuroflyDiagnostics?.errors || []).map ? (window.neuroflyErrors?.log || []).map(e => e.phase + ': ' + e.message + ' x' + e.count) : [],
};
"""


def make_driver(w=1600, h=1000):
    root = Path.home() / "snap" / "firefox" / "common" / "neurofly-auditC"
    root.mkdir(parents=True, exist_ok=True)
    o = Options(); o.add_argument("-headless")
    o.add_argument(f"--width={w}"); o.add_argument(f"--height={h}")
    s = Service(executable_path="geckodriver", service_args=["--profile-root", str(root)],
                log_output=str(OUT / "geckodriver.log"))
    d = webdriver.Firefox(options=o, service=s)
    d.set_window_size(w, h)
    return d


def main():
    d = make_driver()
    log = []
    try:
        d.get(URL)
        for _ in range(60):
            s = d.execute_script(JS, "t-maze")
            if s.get("pktStep"):
                break
            time.sleep(1)
        log.append({"assay": "_initial", "sample": s})
        d.save_screenshot(str(OUT / "00-initial.png"))
        for n, a in enumerate(ASSAYS, 1):
            card = d.find_element(By.CSS_SELECTOR, f'.experiment-card[data-paradigm="{a}"]')
            d.execute_script("arguments[0].scrollIntoView({block:'center'});", card)
            card.click()
            samples = []
            t0 = time.time()
            for at in (2, DWELL / 3, 2 * DWELL / 3, DWELL):
                while time.time() - t0 < at:
                    time.sleep(0.2)
                samples.append(d.execute_script(JS, a))
            d.save_screenshot(str(OUT / f"{n:02d}-{a}.png"))
            log.append({"assay": a, "samples": samples})
            last = samples[-1]
            print(a, json.dumps({k: last[k] for k in ("pktParadigm", "pktStep", "fly", "spikes", "dn", "identMotor",
                                                       "identityBanner", "errorBanner", "pktState", "epgBump",
                                                       "kcActive")}), flush=True)
        # restore the starting assay
        d.find_element(By.CSS_SELECTOR, '.experiment-card[data-paradigm="t-maze"]').click()
        time.sleep(3)
        logs = []
        try:
            logs = d.get_log("browser")
        except Exception as e:
            logs = [f"get_log unsupported: {e}"]
        log.append({"browser_console": logs})
    finally:
        (OUT / "walk.json").write_text(json.dumps(log, indent=1, default=str))
        d.quit()


if __name__ == "__main__":
    main()
