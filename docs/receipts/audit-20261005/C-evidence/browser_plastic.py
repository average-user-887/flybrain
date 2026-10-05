#!/usr/bin/env python3
"""Switch the live dashboard to connectome-plastic, visit three assays, then restore fixed + t-maze."""
import json, time, sys
sys.path.insert(0, "<SCRATCH>")
from browser import make_driver, JS, URL, OUT
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import Select

d = make_driver()
log = []
try:
    d.get(URL)
    time.sleep(8)
    Select(d.find_element(By.ID, "selectBackend")).select_by_value("connectome-plastic")
    time.sleep(15)
    for a in ("t-maze", "buridan", "optomotor"):
        d.find_element(By.CSS_SELECTOR, f'.experiment-card[data-paradigm="{a}"]').click()
        s = []
        for _ in range(3):
            time.sleep(10)
            s.append(d.execute_script(JS, a))
        d.save_screenshot(str(OUT / f"plastic-{a}.png"))
        hud = d.execute_script("const t=id=>{const e=document.getElementById(id);return e?e.textContent.trim():null};"
                               "return {mean:t('hudWp6MeanDelta'),max:t('hudWp6MaxDelta'),pct:t('hudWp6Pct')}")
        log.append({"assay": a, "samples": s, "wp6": hud})
        l = s[-1]
        print(a, json.dumps({k: l[k] for k in ("identBackend", "badge", "pktStep", "fly", "spikes", "dn", "epgBump")}), hud, flush=True)
    Select(d.find_element(By.ID, "selectBackend")).select_by_value("connectome-fixed")
    time.sleep(15)
    d.find_element(By.CSS_SELECTOR, '.experiment-card[data-paradigm="t-maze"]').click()
    time.sleep(5)
    print("restored", d.execute_script(JS, "t-maze")["identBackend"], flush=True)
finally:
    (OUT / "walk-plastic.json").write_text(json.dumps(log, indent=1, default=str))
    d.quit()
