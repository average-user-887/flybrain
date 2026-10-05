#!/usr/bin/env python3
"""PR #30 review: drive the experiment studio's own features in real (headless) Firefox.

Studio: python -m neurofly_studio serve on 127.0.0.1:8796 from the review worktree,
real MaleCNS graph, CPU brain backend, real FlyGym body (no stand-in runner).

Phases (run separately, because real runs take minutes):
  build   - Gallery/Build/Queue: catalog, builder, silencing, controls, queue, cancel
  results - after the runs finished (and one pair was curated): gallery cards, Watch,
            Download bundle, Compare with pair, synced play/seek/pause, metrics table
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.firefox.options import Options
from selenium.webdriver.firefox.service import Service
from selenium.webdriver.support.ui import Select

import os
BASE = "http://127.0.0.1:8796/"
SKIP_FIRST = os.environ.get("SKIP_FIRST") == "1"
HOOK = """
window.__errs = window.__errs || [];
if (!window.__hooked) {
  window.__hooked = true;
  window.addEventListener('error', e => window.__errs.push('error: ' + e.message));
  window.addEventListener('unhandledrejection', e => window.__errs.push('rejection: ' + (e.reason && e.reason.message || e.reason)));
  const ce = console.error.bind(console);
  console.error = (...a) => { window.__errs.push('console.error: ' + a.map(String).join(' ')); ce(...a); };
}
"""


def driver():
    opts = Options()
    opts.add_argument("-headless")
    opts.add_argument("--width=1600")
    opts.add_argument("--height=1100")
    opts.set_preference("browser.download.folderList", 2)
    opts.set_preference("browser.download.dir", "<scratch>/pr30-studio/downloads")
    opts.set_preference("browser.helperApps.neverAsk.saveToDisk", "application/zip")
    return webdriver.Firefox(service=Service("/snap/bin/geckodriver"), options=opts)


def wait(fn, timeout=30.0, step=0.25, what=""):
    end = time.time() + timeout
    last = None
    while time.time() < end:
        try:
            last = fn()
            if last:
                return last
        except Exception as error:  # noqa: BLE001 - polled condition
            last = error
        time.sleep(step)
    raise TimeoutError(f"timed out: {what} (last={last!r})")


def load(d):
    d.get(BASE)
    d.execute_script(HOOK)
    wait(lambda: "queue worker" in d.find_element(By.ID, "worker").text, what="worker status")


def tab(d, name):
    d.find_element(By.CSS_SELECTOR, f"nav [data-tab='{name}']").click()
    wait(lambda: d.find_element(By.ID, f"tab-{name}").is_displayed(), what=f"tab {name}")


def errs(d):
    return d.execute_script("return (window.__errs||[]).slice()") or []


def build_phase(d, out: Path, report: dict):
    load(d)
    checks = report.setdefault("checks", {})
    d.save_screenshot(str(out / "01-gallery-empty.png"))
    checks["gallery_lead_text"] = d.find_element(By.CSS_SELECTOR, "#tab-gallery .lead").text
    tab(d, "build")
    cards = d.find_elements(By.CSS_SELECTOR, "#paradigms .card")
    badges = [c.find_element(By.CSS_SELECTOR, ".badge").text for c in cards]
    checks["paradigm_cards"] = len(cards)
    checks["badges"] = sorted(set(badges))
    checks["buildable"] = [b.get_attribute("data-build") for b in d.find_elements(By.CSS_SELECTOR, "[data-build]")]
    d.find_element(By.CSS_SELECTOR, "[data-build='optomotor']").click()
    wait(lambda: d.find_element(By.ID, "builder").is_displayed(), what="builder")
    checks["builder_explanation"] = d.find_element(By.ID, "b-explain").text
    checks["silence_groups"] = [s.get_attribute("data-silence") for s in d.find_elements(By.CSS_SELECTOR, "[data-silence]")]
    checks["prereg_text"] = d.find_element(By.ID, "b-prereg").text
    radios = lambda: {r.get_attribute("value"): (r.is_selected(), r.is_enabled())  # noqa: E731
                      for r in d.find_elements(By.CSS_SELECTOR, "input[name=b-control]")}
    checks["controls_default_unsilenced"] = radios()

    def set_param(name, value):
        box = d.find_element(By.ID, "p-" + name)
        box.clear()
        box.send_keys(str(value))

    # Experiment 1: DNa02 silenced on both sides, paired by default with the same fly un-silenced.
    name = d.find_element(By.ID, "b-name"); name.clear(); name.send_keys("Review DNa02 silenced")
    set_param("duration_s", 0.5); set_param("seed", 11)
    Select(d.find_element(By.ID, "s-dna02")).select_by_value("both")
    checks["controls_after_silencing"] = radios()
    d.save_screenshot(str(out / "02-builder-silenced.png"))
    if SKIP_FIRST:
        checks["submit1"] = "skipped (queued by the first build pass)"
    else:
        d.find_element(By.ID, "b-submit").click()
        wait(lambda: d.find_element(By.ID, "b-result").is_displayed(), what="result 1")
        checks["submit1"] = d.find_element(By.ID, "b-result").text

    # Experiment 2: nothing silenced -> default control is the output-disconnected brain.
    Select(d.find_element(By.ID, "s-dna02")).select_by_value("")
    checks["controls_after_unsilencing"] = radios()
    name.clear(); name.send_keys("Review intact vs disconnected")
    set_param("seed", 12)
    if SKIP_FIRST:
        checks["submit2"] = "skipped (queued by the first build pass)"
    else:
        d.find_element(By.ID, "b-submit").click()
        time.sleep(1.5)
        checks["submit2"] = d.find_element(By.ID, "b-result").text

    # Experiment 3: two repeats, one-sided T4/T5 silencing; queued only to be cancelled.
    name.clear(); name.send_keys("Review cancel me")
    set_param("seed", 20)
    Select(d.find_element(By.ID, "b-repeats")).select_by_visible_text("2")
    Select(d.find_element(By.ID, "s-t4t5")).select_by_value("L")
    d.find_element(By.ID, "b-submit").click()
    time.sleep(1.5)
    checks["submit3"] = d.find_element(By.ID, "b-result").text

    # A bad value is refused by the server and shown, not queued.
    Select(d.find_element(By.ID, "s-t4t5")).select_by_value("")
    Select(d.find_element(By.ID, "b-repeats")).select_by_visible_text("1")
    set_param("contrast", 3)
    d.execute_script("document.getElementById('p-contrast').removeAttribute('max')")
    d.execute_script("document.getElementById('builder').requestSubmit()")
    time.sleep(1.5)
    checks["submit_bad_contrast"] = d.find_element(By.ID, "b-result").text
    set_param("contrast", 1)

    tab(d, "queue")
    time.sleep(1.0)
    rows = lambda: [r.text for r in d.find_elements(By.CSS_SELECTOR, "#queue-rows tr")]  # noqa: E731
    checks["queue_rows_before_cancel"] = rows()
    d.save_screenshot(str(out / "03-queue.png"))
    cancel = d.find_elements(By.CSS_SELECTOR, "[data-cancel*='review-cancel-me']")
    checks["cancel_buttons_for_cancel_me"] = len(cancel)
    for _ in range(len(cancel)):
        button = d.find_elements(By.CSS_SELECTOR, "[data-cancel*='review-cancel-me']")[0]
        button.click()
        time.sleep(1.5)
    checks["queue_rows_after_cancel"] = rows()
    checks["worker_text"] = d.find_element(By.ID, "worker").text
    d.save_screenshot(str(out / "04-queue-after-cancel.png"))
    checks["errors"] = errs(d)


def results_phase(d, out: Path, report: dict):
    load(d)
    checks = report.setdefault("checks", {})
    time.sleep(1.0)
    checks["curated_cards"] = [c.text for c in d.find_elements(By.CSS_SELECTOR, "#curated .card")]
    checks["finished_cards"] = [c.text for c in d.find_elements(By.CSS_SELECTOR, "#finished .card")]
    d.save_screenshot(str(out / "05-gallery-with-runs.png"))

    # Watch: open the first curated card's replay in a new window and check it loaded.
    watch = d.find_elements(By.CSS_SELECTOR, "#curated .card a.act[href*='embodied_replay']")
    checks["curated_watch_links"] = len(watch)
    main = d.current_window_handle
    if watch:
        watch[0].click()
        wait(lambda: len(d.window_handles) > 1, what="replay window")
        d.switch_to.window([h for h in d.window_handles if h != main][0])
        info = wait(lambda: d.execute_script("return window.embodiedReplay && {frames: window.embodiedReplay.frames, verified: window.embodiedReplay.verified, duration: window.embodiedReplay.duration}"),
                    timeout=60, what="replay loaded")
        checks["watch_replay"] = info
        d.save_screenshot(str(out / "06-watch-curated.png"))
        d.close()
        d.switch_to.window(main)

    # Download: the card's link, fetched by the page itself (same request the click makes).
    link = d.find_element(By.CSS_SELECTOR, "#finished .card a[download]").get_attribute("href")
    checks["download"] = d.execute_async_script("""
      const url = arguments[0], done = arguments[arguments.length - 1];
      fetch(url).then(async r => { const b = new Uint8Array(await r.arrayBuffer());
        done({url, status: r.status, type: r.headers.get('Content-Type'),
              disposition: r.headers.get('Content-Disposition'), bytes: b.length,
              zip_magic: b[0] === 0x50 && b[1] === 0x4b}); }).catch(e => done({error: String(e)}));""", link)
    # Click it too: Firefox should save the zip (download dir set in the profile).
    d.find_element(By.CSS_SELECTOR, "#finished .card a[download]").click()
    time.sleep(3.0)
    checks["downloaded_files"] = sorted(p.name for p in Path("<scratch>/pr30-studio/downloads").glob("*"))

    # Compare with pair: from the silenced experiment's card.
    cards = d.find_elements(By.CSS_SELECTOR, "#finished .card")
    target = None
    for card in cards:
        if "Silenced: DNa02" in card.text:
            target = card
            break
    checks["silenced_card_found"] = target is not None
    button = (target or cards[0]).find_element(By.CSS_SELECTOR, "[data-compare]")
    checks["compare_button_text"] = button.text
    button.click()
    wait(lambda: d.find_element(By.ID, "tab-compare").is_displayed(), what="compare tab")
    wait(lambda: not d.find_element(By.ID, "sync").get_attribute("hidden"), timeout=90, what="sync bar")
    checks["compare_left"] = Select(d.find_element(By.ID, "cmp-a")).first_selected_option.text
    checks["compare_right"] = Select(d.find_element(By.ID, "cmp-b")).first_selected_option.text
    wait(lambda: len(d.find_elements(By.CSS_SELECTOR, "#cmp-table tbody tr")) >= 6, timeout=60, what="metrics table")
    checks["metrics_table"] = [r.text for r in d.find_elements(By.CSS_SELECTOR, "#cmp-table tr")]
    frames = """return ['cmp-frame-a','cmp-frame-b'].map(id => { const r = document.getElementById(id).contentWindow.embodiedReplay;
                 return r ? {frame: r.frame, frames: r.frames, duration: r.duration} : null; })"""
    checks["frames_loaded"] = d.execute_script(frames)
    d.save_screenshot(str(out / "07-compare-loaded.png"))
    d.find_element(By.ID, "sync-play").click()
    time.sleep(0.2)
    checks["play_label"] = d.find_element(By.ID, "sync-play").text
    time.sleep(0.3)
    first = d.execute_script(frames)
    time.sleep(0.3)
    second = d.execute_script(frames)
    checks["while_playing"] = [first, second, d.find_element(By.ID, "sync-clock").text]
    d.find_element(By.ID, "sync-play").click()
    checks["pause_label"] = d.find_element(By.ID, "sync-play").text
    paused_a = d.execute_script(frames); time.sleep(0.5); paused_b = d.execute_script(frames)
    checks["paused_stays"] = paused_a == paused_b
    # Seek to the middle: both replays land on the same frame.
    d.execute_script("const s = document.getElementById('sync-seek'); s.value = '500'; s.dispatchEvent(new Event('input'));")
    time.sleep(0.5)
    checks["after_seek_500"] = [d.execute_script(frames), d.find_element(By.ID, "sync-clock").text]
    d.save_screenshot(str(out / "08-compare-seek-middle.png"))
    # Speed 4x and play to the end: both stop on their last frame.
    Select(d.find_element(By.ID, "sync-speed")).select_by_value("4")
    d.find_element(By.ID, "sync-play").click()
    wait(lambda: d.find_element(By.ID, "sync-play").text == "Play both", timeout=30, what="played to end")
    checks["after_play_to_end"] = [d.execute_script(frames), d.find_element(By.ID, "sync-clock").text]
    # Curated pair compare, if installed.
    tab(d, "gallery")
    curated_compare = d.find_elements(By.CSS_SELECTOR, "#curated [data-compare][data-with]")
    checks["curated_compare_with_pair"] = len(curated_compare)
    if curated_compare:
        curated_compare[0].click()
        wait(lambda: not d.find_element(By.ID, "sync").get_attribute("hidden"), timeout=90, what="curated sync bar")
        wait(lambda: len(d.find_elements(By.CSS_SELECTOR, "#cmp-table tbody tr")) >= 5, timeout=60, what="curated metrics")
        checks["curated_compare_pair"] = [Select(d.find_element(By.ID, "cmp-a")).first_selected_option.text,
                                          Select(d.find_element(By.ID, "cmp-b")).first_selected_option.text]
        checks["curated_metrics_table"] = [r.text for r in d.find_elements(By.CSS_SELECTOR, "#cmp-table tr")]
        d.save_screenshot(str(out / "09-compare-curated.png"))
    # Phone width: page still renders without horizontal overflow.
    d.set_window_size(390, 844)
    tab(d, "gallery")
    checks["phone_overflow_px"] = d.execute_script("return document.documentElement.scrollWidth - document.documentElement.clientWidth")
    d.save_screenshot(str(out / "10-phone-gallery.png"))
    checks["global_error_visible"] = d.find_element(By.ID, "global-error").is_displayed()
    checks["errors"] = errs(d)


def sync_phase(d, out: Path, report: dict):
    """Synced play/pause at 0.25x on the silenced pair (0.5 s runs play for 2 s of wall time)."""
    load(d)
    checks = report.setdefault("checks", {})
    time.sleep(1.0)
    card = [c for c in d.find_elements(By.CSS_SELECTOR, "#finished .card") if "Silenced: DNa02" in c.text][0]
    card.find_element(By.CSS_SELECTOR, "[data-compare]").click()
    wait(lambda: not d.find_element(By.ID, "sync").get_attribute("hidden"), timeout=90, what="sync bar")
    frames = """return ['cmp-frame-a','cmp-frame-b'].map(id => document.getElementById(id).contentWindow.embodiedReplay.frame)"""
    Select(d.find_element(By.ID, "sync-speed")).select_by_value("0.25")
    d.find_element(By.ID, "sync-play").click()
    samples = []
    for _ in range(4):
        time.sleep(0.25)
        samples.append([d.execute_script(frames), d.find_element(By.ID, "sync-clock").text])
    checks["playing_samples"] = samples
    checks["advanced_while_playing"] = samples[-1][0][0] > samples[0][0][0]
    checks["both_on_same_frame"] = all(a == b for (a, b), _ in samples)
    d.find_element(By.ID, "sync-play").click()
    checks["label_after_pause"] = d.find_element(By.ID, "sync-play").text
    first = d.execute_script(frames); time.sleep(1.0); second = d.execute_script(frames)
    checks["paused_frames"] = [first, second]
    checks["paused_stays"] = first == second
    d.save_screenshot(str(out / "11-compare-paused-mid.png"))
    checks["errors"] = errs(d)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=("build", "results", "sync"))
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    Path("<scratch>/pr30-studio/downloads").mkdir(parents=True, exist_ok=True)
    report = {"phase": args.phase, "base": BASE, "started": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
    d = driver()
    try:
        report["browser"] = d.capabilities.get("browserName") + " " + d.capabilities.get("browserVersion", "")
        {"build": build_phase, "results": results_phase, "sync": sync_phase}[args.phase](d, args.out, report)
        report["ok"] = True
    except Exception as error:  # noqa: BLE001 - record and keep the screenshot
        report["ok"] = False
        report["exception"] = f"{type(error).__name__}: {error}"
        try:
            d.save_screenshot(str(args.out / "zz-failure.png"))
            report.setdefault("checks", {})["errors"] = errs(d)
        except Exception:  # noqa: BLE001
            pass
    finally:
        d.quit()
    report["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    (args.out / f"studio_{args.phase}.json").write_text(json.dumps(report, indent=1, default=str) + "\n")
    print(json.dumps(report, indent=1, default=str))


if __name__ == "__main__":
    main()
