#!/usr/bin/env python3
"""Headless Chromium walk-through of the experiment studio (web/studio.html).

    python scripts/studio_headless_check.py --out /tmp/studio-shots

By default it starts an in-process studio on a temporary queue whose runner is
tests/studio_fakes.py (modular baseline + stick-fly body, NOT a simulation), so
it needs neither the graph nor FlyGym.  It clicks through Build, Queue, Gallery
and Compare, waits for the runs to finish, opens both replays side by side and
prints a JSON report (console errors, checks, screenshots).  This supplements,
and does not replace, the live browser check required by AGENTS.md.
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
import threading
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from playwright.sync_api import sync_playwright  # noqa: E402

from neurofly_studio.server import Studio, make_server  # noqa: E402
from tests.studio_fakes import fake_runner  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    tmp = Path(tempfile.mkdtemp(prefix="studio-check-"))
    studio = Studio(tmp / "queue", tmp / "curated", runner=fake_runner)
    studio.start_worker(poll_s=0.1, log=lambda m: print(m, file=sys.stderr))
    server = make_server(studio, "127.0.0.1", 0)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{server.server_address[1]}"

    report: dict = {"base": base, "checks": {}, "console_errors": [], "page_errors": [], "screenshots": []}
    checks = report["checks"]

    def shot(page, name):
        path = args.out / f"{name}.png"
        page.screenshot(path=str(path), full_page=True)
        report["screenshots"].append(str(path))

    with sync_playwright() as pw:
        browser = pw.chromium.launch()
        for width, label in ((1280, "desktop"), (390, "phone")):
            context = browser.new_context(viewport={"width": width, "height": 900})
            page = context.new_page()
            page.on("console", lambda m: m.type == "error" and report["console_errors"].append(m.text))
            page.on("pageerror", lambda e: report["page_errors"].append(str(e)))
            page.goto(base + "/")
            page.wait_for_selector("#curated .empty")
            checks[f"{label}_no_horizontal_scroll"] = page.evaluate(
                "document.documentElement.scrollWidth <= document.documentElement.clientWidth")
            if label == "phone":
                shot(page, "phone-gallery")
                context.close()
                continue
            shot(page, "1-gallery-empty")

            page.click("nav [data-tab=build]")
            page.wait_for_selector("[data-build=optomotor]")
            badges = page.eval_on_selector_all("#paradigms .badge", "els => els.map(e => e.textContent)")
            checks["fourteen_paradigms_with_badges"] = len(badges) == 14
            checks["no_validated_badge_while_matrix_has_none"] = "Validated" not in badges
            checks["only_optomotor_buildable"] = page.eval_on_selector_all(
                "[data-build]", "els => els.map(e => e.dataset.build)") == ["optomotor"]
            page.click("[data-build=optomotor]")
            page.fill("#p-world_angular_velocity_rad_s", "3")
            page.fill("#p-duration_s", "1")
            page.fill("#b-name", "Headless check")
            checks["prereg_listed_read_only"] = "optomotor-yaw-v3-2" in page.inner_text("#b-prereg")
            shot(page, "2-builder")
            page.click("#b-submit")
            page.wait_for_selector("#b-result.ok")
            checks["queued_message"] = page.inner_text("#b-result")

            # A bad value is refused by the server and explained.
            page.fill("#p-contrast", "2")
            page.eval_on_selector("#p-contrast", "e => e.removeAttribute('max')")
            errors_before = len(report["console_errors"])
            page.click("#b-submit")
            page.wait_for_selector("#b-result.err")
            # The browser logs the deliberate 400 as a failed resource; that one is expected.
            expected = [e for e in report["console_errors"][errors_before:] if "status of 400" in e]
            if len(expected) == 1:
                report["console_errors"].remove(expected[0])
                checks["expected_400_logged_once"] = True
            checks["out_of_range_refused"] = page.inner_text("#b-result")

            page.click("nav [data-tab=queue]")
            page.wait_for_selector("#queue-rows tr")
            shot(page, "3-queue")
            page.wait_for_function(
                "() => [...document.querySelectorAll('#queue-rows .badge')].length === 2 && "
                "[...document.querySelectorAll('#queue-rows .badge')].every(b => b.textContent === 'done')",
                timeout=120_000)
            shot(page, "4-queue-done")

            page.click("nav [data-tab=gallery]")
            page.wait_for_selector("#finished .card")
            checks["gallery_cards"] = page.locator("#finished .card").count()
            shot(page, "5-gallery")
            page.click("#finished [data-with]")
            page.wait_for_selector("#cmp-table td")
            page.wait_for_function("() => !document.querySelector('#cmp-table').textContent.includes('undefined')")
            rows = page.eval_on_selector_all("#cmp-table tbody tr", "rs => rs.map(r => [...r.children].map(c => c.textContent))")
            checks["compare_rows"] = rows
            for frame_id in ("cmp-frame-a", "cmp-frame-b"):
                frame = page.frame_locator("#" + frame_id)
                frame.locator("#status").wait_for()
                page.wait_for_timeout(1500)
                checks[f"{frame_id}_status"] = frame.locator("#status").inner_text()
            shot(page, "6-compare")

            # Silencing (only when this install's runner has --silence).
            page.click("nav [data-tab=build]")
            page.click("[data-build=optomotor]")
            if page.locator("#b-silence").is_hidden():
                checks["silence_ui"] = "hidden: runner has no --silence"
            else:
                checks["intact_control_disabled_until_silenced"] = page.is_disabled(
                    "input[name=b-control][value=intact]")
                page.select_option("#s-dna02", "both")
                checks["intact_control_default_when_silenced"] = page.is_checked(
                    "input[name=b-control][value=intact]")
                page.fill("#p-duration_s", "1")
                page.fill("#b-name", "Silence DNa02")
                shot(page, "7-builder-silence")
                page.click("#b-submit")
                page.wait_for_selector("#b-result.ok")
                page.click("nav [data-tab=queue]")
                page.wait_for_function(
                    "() => [...document.querySelectorAll('#queue-rows .badge')].length === 4 && "
                    "[...document.querySelectorAll('#queue-rows .badge')].every(b => b.textContent === 'done')",
                    timeout=120_000)
                page.click("nav [data-tab=gallery]")
                page.locator("#finished .card", has_text="Silence DNa02").first.locator("[data-with]").click()
                page.wait_for_function("() => document.querySelector('#cmp-table').textContent.includes('Silenced cell types')")
                page.wait_for_timeout(500)
                rows = page.eval_on_selector_all("#cmp-table tbody tr", "rs => rs.map(r => [...r.children].map(c => c.textContent))")
                table = {r[0]: r[1:] for r in rows}
                checks["silence_compare_rows"] = rows
                checks["silence_pair_intact_left"] = table["Silenced cell types"] == ["–", "DNa02"]
                checks["silence_clamp_reported"] = table["Silencing held (no silenced neuron spiked)"][1] == "yes"
                shot(page, "8-compare-silenced")
            context.close()
        browser.close()
    server.shutdown()

    turning = {r[0]: r[1:] for r in checks.get("compare_rows", [])}
    gain = turning.get("Turning gain (fly ÷ world)", ["", ""])
    report["pass"] = (not report["console_errors"] and not report["page_errors"]
                      and all(v for k, v in checks.items() if isinstance(v, bool))
                      and "Queued 2 runs" in checks.get("queued_message", "")
                      and "contrast" in checks.get("out_of_range_refused", "")
                      and gain[1] in ("0.000", "–") and gain[0] not in ("0.000", "–"))
    print(json.dumps(report, indent=2))
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
