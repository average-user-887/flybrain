"""Audit D helpers: real Firefox (headless) via selenium against the audit daemon 8841 / web 8840."""
from __future__ import annotations

import json
import os
import time
import urllib.request
from pathlib import Path

from selenium import webdriver
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.common.by import By
from selenium.webdriver.firefox.options import Options
from selenium.webdriver.firefox.service import Service
from selenium.webdriver.support.ui import Select

SCRATCH = Path("<scratch>")
DAEMON = "http://127.0.0.1:8841"
WEB = "http://127.0.0.1:8840"
DL = SCRATCH / "downloads"
SHOTS = SCRATCH / "shots"
DL.mkdir(parents=True, exist_ok=True)
SHOTS.mkdir(parents=True, exist_ok=True)

HOOK = """
window.__errs = window.__errs || [];
window.__alerts = window.__alerts || [];
if (!window.__hooked) {
  window.__hooked = true;
  window.addEventListener('error', e => window.__errs.push('error: ' + e.message));
  window.addEventListener('unhandledrejection', e => window.__errs.push('rejection: ' + (e.reason && e.reason.message || e.reason)));
  const ce = console.error.bind(console);
  console.error = (...a) => { window.__errs.push('console.error: ' + a.map(String).join(' ')); ce(...a); };
  const cw = console.warn.bind(console);
  console.warn = (...a) => { window.__errs.push('console.warn: ' + a.map(String).join(' ')); cw(...a); };
  window.alert = (m) => { window.__alerts.push(String(m)); };
  // record every /api/command POST the page makes, with its reply
  const of = window.fetch.bind(window);
  window.__cmds = [];
  window.fetch = async (url, opts) => {
    const r = await of(url, opts);
    try {
      if (String(url).includes('/api/command') || String(url).includes('/api/controller')) {
        const c = r.clone(); const body = await c.text();
        window.__cmds.push({url: String(url), req: opts && opts.body, status: r.status, reply: body.slice(0, 400)});
      }
    } catch (e) {}
    return r;
  };
}
"""


class Log:
    def __init__(self, name):
        self.path = SCRATCH / f"{name}.jsonl"
        self.f = self.path.open("a", encoding="utf-8")

    def __call__(self, control, **kw):
        kw = {"t": time.strftime("%H:%M:%S"), "control": control, **kw}
        line = json.dumps(kw, default=str)
        print(line[:600], flush=True)
        self.f.write(line + "\n")
        self.f.flush()


def driver():
    opts = Options()
    if os.environ.get("HEADED") != "1":
        opts.add_argument("-headless")
    opts.add_argument("--width=1700")
    opts.add_argument("--height=1150")
    opts.set_preference("browser.download.folderList", 2)
    opts.set_preference("browser.download.dir", str(DL))
    opts.set_preference("browser.download.useDownloadDir", True)
    opts.set_preference("browser.helperApps.neverAsk.saveToDisk",
                        "application/zip,application/json,text/csv,application/octet-stream,text/plain")
    opts.set_preference("pdfjs.disabled", True)
    d = webdriver.Firefox(service=Service("/snap/bin/geckodriver"), options=opts)
    d.set_window_size(1700, 1150)
    return d


def hook(d):
    d.execute_script(HOOK)


def js(d, s, *a):
    return d.execute_script(s, *a)


def errs(d, clear=True):
    e = js(d, "const e = (window.__errs||[]).slice(); if (arguments[0]) window.__errs = []; return e;", clear)
    return e


def alerts(d):
    return js(d, "const a=(window.__alerts||[]).slice(); window.__alerts=[]; return a;")


def cmds(d):
    return js(d, "const a=(window.__cmds||[]).slice(); window.__cmds=[]; return a;")


def wait(fn, timeout=30.0, step=0.25):
    end = time.time() + timeout
    last = None
    while time.time() < end:
        try:
            last = fn()
            if last:
                return last
        except Exception as e:  # noqa: BLE001
            last = e
        time.sleep(step)
    return None


def get(path, base=DAEMON, timeout=6):
    with urllib.request.urlopen(base + path, timeout=timeout) as r:
        return json.loads(r.read().decode())


def post(action, base=DAEMON, **params):
    body = json.dumps({"action": action, **params}).encode()
    req = urllib.request.Request(base + "/api/command", data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.loads(r.read().decode())


def status():
    return get("/api/status")


def tel():
    return get("/api/telemetry")


def click(d, el_id):
    el = d.find_element(By.ID, el_id)
    d.execute_script("arguments[0].scrollIntoView({block:'center'})", el)
    try:
        el.click()
    except Exception:  # noqa: BLE001 - covered element: fall back to a DOM click
        d.execute_script("arguments[0].click()", el)
    return el


def text(d, el_id):
    try:
        return d.find_element(By.ID, el_id).text
    except Exception:  # noqa: BLE001
        return None


def shot(d, name):
    p = SHOTS / f"{name}.png"
    d.save_screenshot(str(p))
    return str(p)


def set_range(d, el, value, event="input"):
    """Set a range input and dispatch input + change like a user drag would."""
    d.execute_script(
        "const e=arguments[0]; e.value=arguments[1]; e.dispatchEvent(new Event('input',{bubbles:true}));"
        "e.dispatchEvent(new Event('change',{bubbles:true}));", el, str(value))


def connected(d):
    return js(d, "const b=window.app&&window.app.hud&&window.app.hud.daemonBridge; return !!(b&&b.connected&&window.app.arena.remoteDriven);")


def open_dash(d, daemon=DAEMON, extra=""):
    d.get(f"{WEB}/index.html?daemon={daemon}{extra}")
    hook(d)
    return wait(lambda: connected(d), 60)


__all__ = [n for n in dir() if not n.startswith("_")] + ["By", "ActionChains", "Select"]
