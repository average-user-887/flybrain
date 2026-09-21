#!/usr/bin/env python3
"""Headless browser exercise of the NeuroFly dashboard (web/index.html + web/app.js).

Usage:
    python3 scripts/dashboard_headless_check.py --mode offline   --out /tmp/shots
    python3 scripts/dashboard_headless_check.py --mode connected --out /tmp/shots

Requires Playwright with Firefox in the interpreter that runs it.

offline mode   : every request to the daemon (port 8769) is aborted in the browser, so
                 the page must fall back to its in-browser engine; every paradigm card,
                 tab, slider, action button, lesion button and control-bar control is
                 exercised and screenshotted.
connected mode : GET requests to the daemon are allowed; POST /api/command is aborted
                 inside the browser so the live learning run is never touched. Only
                 read-only UI (tabs) is exercised.

Prints a JSON report with console errors, page errors, NaN sightings, layout overflow
and screenshot paths.
"""
import argparse
import json
import os
import sys
import time

from playwright.sync_api import sync_playwright

PARADIGMS = [
    'open-arena', 't-maze', 'y-maze', 'heat-maze', 'buridan', 'visual-operant',
    'wind-tunnel', 'looming-escape', 'optomotor', 'gap-crossing', 'circadian-dam',
    'courtship', 'labyrinth', 'multisensory-sandbox',
]

OVERFLOW_JS = """
() => {
  const out = { pageScrollX: document.documentElement.scrollWidth > document.documentElement.clientWidth,
                pageScrollY: document.documentElement.scrollHeight > document.documentElement.clientHeight,
                overflowing: [] };
  const vw = document.documentElement.clientWidth, vh = document.documentElement.clientHeight;
  for (const el of document.querySelectorAll('.panel, .hud-card, canvas, .top-navbar, .bottom-stage-panel, .catalog-scroll, .assay-tools-panel, .limb-deck-panel')) {
    const r = el.getBoundingClientRect();
    const cs = getComputedStyle(el);
    if (r.width === 0 || cs.display === 'none') continue;
    const hidden = cs.overflow === 'hidden' || cs.overflowX === 'hidden' || cs.overflow === 'auto' || cs.overflowY === 'auto' || cs.overflowX === 'auto' || cs.overflowY === 'scroll';
    if (r.right > vw + 1 || r.bottom > vh + 1 || (!hidden && (el.scrollWidth > el.clientWidth + 2)))
      out.overflowing.push({ tag: el.tagName, id: el.id, cls: el.className.toString().slice(0, 40), right: Math.round(r.right), bottom: Math.round(r.bottom), sw: el.scrollWidth, cw: el.clientWidth, vw, vh });
  }
  return out;
}
"""

NAN_JS = """
() => {
  const hits = [];
  const walker = document.createTreeWalker(document.body, NodeFilter.SHOW_TEXT);
  let n;
  while ((n = walker.nextNode())) {
    const t = n.nodeValue;
    if (/\\bNaN\\b|undefined|Infinity/.test(t)) {
      const p = n.parentElement;
      hits.push({ id: p && p.id, cls: p && p.className.toString().slice(0, 30), text: t.trim().slice(0, 60) });
    }
  }
  return hits;
}
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', choices=['offline', 'connected'], default='offline')
    ap.add_argument('--url', default='http://127.0.0.1:8770/')
    ap.add_argument('--out', default='/tmp/neurofly_shots')
    ap.add_argument('--width', type=int, default=1600)
    ap.add_argument('--height', type=int, default=900)
    ap.add_argument('--settle', type=float, default=2.0)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    report = {'mode': args.mode, 'url': args.url, 'console': [], 'pageerrors': [], 'requests': [],
              'blocked_posts': 0, 'nan': {}, 'overflow': {}, 'shots': [], 'notes': []}

    def shot(page, name):
        path = os.path.join(args.out, f'{args.mode}_{name}.png')
        page.screenshot(path=path, full_page=False)
        report['shots'].append(path)
        return path

    def audit(page, name):
        report['nan'][name] = page.evaluate(NAN_JS)
        report['overflow'][name] = page.evaluate(OVERFLOW_JS)

    with sync_playwright() as p:
        browser = p.firefox.launch(headless=True)
        ctx = browser.new_context(viewport={'width': args.width, 'height': args.height}, accept_downloads=True)
        page = ctx.new_page()

        page.on('console', lambda m: report['console'].append({'type': m.type, 'text': m.text[:300]}) if m.type in ('error', 'warning') else None)
        page.on('pageerror', lambda e: report['pageerrors'].append(str(e)[:400]))
        page.on('dialog', lambda d: d.accept())

        def route_handler(route, request):
            url = request.url
            is_daemon = ':8769' in url or '/api/' in url
            if not is_daemon:
                return route.continue_()
            if request.method == 'POST' or '/api/command' in url:
                report['blocked_posts'] += 1
                report['requests'].append(f'BLOCKED {request.method} {url}')
                return route.abort()
            if args.mode == 'offline':
                report['requests'].append(f'ABORTED(offline) {request.method} {url}')
                return route.abort()
            report['requests'].append(f'{request.method} {url}')
            return route.continue_()

        page.route('**/*', route_handler)

        page.goto(args.url, wait_until='load')
        time.sleep(args.settle + 1.5)
        shot(page, '00_initial')
        audit(page, 'initial')
        report['status_pill'] = page.evaluate("() => (document.getElementById('clusterStatusPill')||{}).textContent")

        if args.mode == 'offline':
            for pid in PARADIGMS:
                card = page.locator(f'.experiment-card[data-paradigm="{pid}"]')
                card.scroll_into_view_if_needed()
                card.click()
                time.sleep(args.settle)
                shot(page, f'10_{pid}_guide')
                audit(page, f'{pid}_guide')
                # Assay tools tab: sliders + buttons
                page.click('#tabAssayTools')
                time.sleep(0.4)
                sliders = page.locator('#assayToolsPanel input[type=range]')
                for i in range(sliders.count()):
                    s = sliders.nth(i)
                    page.evaluate("(el) => { el.value = el.max; el.dispatchEvent(new Event('input', {bubbles:true})); el.value = el.min; el.dispatchEvent(new Event('input', {bubbles:true})); }", s.element_handle())
                btns = page.locator('#assayToolsPanel button')
                for i in range(btns.count()):
                    btns.nth(i).click()
                    time.sleep(0.15)
                time.sleep(0.8)
                shot(page, f'11_{pid}_assay')
                audit(page, f'{pid}_assay')
                # Guide-tab dynamic sliders
                page.click('#tabGuide')
                time.sleep(0.2)
                gs = page.locator('#dynamicSlidersContainer input[type=range]')
                for i in range(gs.count()):
                    page.evaluate("(el) => { el.value = el.max; el.dispatchEvent(new Event('input', {bubbles:true})); }", gs.nth(i).element_handle())
                time.sleep(0.3)

            # Limb deck & neuro-stim
            page.click('.experiment-card[data-paradigm="multisensory-sandbox"]')
            time.sleep(0.5)
            page.click('#tabLimbDeck')
            page.click('#btnToggleManualControl')
            for sid in ['sliderStimDna02', 'sliderStimThrust', 'sliderStimMdn', 'sliderStimCpg', 'sliderStimWing']:
                page.evaluate(f"() => {{ const el = document.getElementById('{sid}'); el.value = el.max; el.dispatchEvent(new Event('input', {{bubbles:true}})); }}")
            for bid in ['btnFlareGf', 'btnFlareHeat', 'btnFlareOdor', 'btnFlareWind']:
                page.click(f'#{bid}')
            time.sleep(2.0)
            shot(page, '20_limb_deck_manual')
            audit(page, 'limb_deck')
            page.click('#btnToggleManualControl')

            # Lesions, science guide modal
            for bid in ['btnLesionMB', 'btnLesionCX', 'btnLesionGF', 'btnLesionJO', 'btnLesionOFF', 'btnLesionWT']:
                page.click(f'#{bid}')
                time.sleep(0.1)
            page.click('#btnOpenScienceGuide')
            time.sleep(0.4)
            shot(page, '30_science_guide_modal')
            page.keyboard.press('Escape')
            time.sleep(0.2)

            # Control bar: speed select, toggle, pause/resume, reset, tools, downloads, clear
            page.select_option('#selectSpeed', '100')
            time.sleep(1.5)
            page.click('#btnSpeedToggle')
            page.click('#btnPauseToggle')
            time.sleep(0.3)
            page.click('#btnPauseToggle')
            page.click('#btnResetTrial')
            for tid in ['toolFood', 'toolAlarm', 'toolPredator', 'toolWind', 'toolSelect']:
                page.click(f'#{tid}')
                page.click('#arenaCanvas', position={'x': 300, 'y': 200})
            page.select_option('#selectSpeed', '1')
            page.click('#btnDownloadCsv')
            page.click('#btnDownloadJson')
            page.click('#btnClearTelemetry')
            time.sleep(1.0)
            shot(page, '40_controls_open_arena')
            audit(page, 'controls')

            # Long soak at 100x on labyrinth to catch runtime errors / leaks
            page.click('.experiment-card[data-paradigm="labyrinth"]')
            page.select_option('#selectSpeed', '100')
            t0 = time.time()
            heap0 = page.evaluate("() => ({ trail: window.arena.fly.trail.length, telem: window.arena.telemetryBuffer.length, trials: (window.arena.trialHistory||[]).length, scope: window.hud.scopeHistory.length, learn: window.hud.learningTrials.length, steps: window.arena.stepCount })")
            time.sleep(12)
            heap1 = page.evaluate("() => ({ trail: window.arena.fly.trail.length, telem: window.arena.telemetryBuffer.length, trials: (window.arena.trialHistory||[]).length, scope: window.hud.scopeHistory.length, learn: window.hud.learningTrials.length, steps: window.arena.stepCount, x: window.arena.fly.x, y: window.arena.fly.y, fps: document.getElementById('statFps').textContent })")
            report['soak'] = {'before': heap0, 'after': heap1, 'wall_seconds': round(time.time() - t0, 1)}
            shot(page, '50_soak_labyrinth_100x')
            audit(page, 'soak')

        else:  # connected: read-only
            time.sleep(4)
            shot(page, '01_connected_after_6s')
            audit(page, 'connected_6s')
            report['bridge'] = page.evaluate("""() => { const b = window.hud && window.hud.daemonBridge; return b ? { connected: b.connected, url: b.activeUrl, lastPacketAgeMs: b.lastPacketTime ? Math.round(performance.now() - b.lastPacketTime) : null, es: b.eventSource ? b.eventSource.readyState : null, paradigm: window.arena.activeParadigmId, fly: { x: window.arena.fly.x, y: window.arena.fly.y } } : null; }""")
            page.click('#tabLimbDeck')
            time.sleep(2.5)
            shot(page, '02_connected_limb_deck')
            audit(page, 'connected_limb')
            page.click('#tabAssayTools')
            time.sleep(1.5)
            shot(page, '03_connected_assay_tools')
            audit(page, 'connected_assay')
            page.click('#tabGuide')

        # Small viewport layout check
        page.set_viewport_size({'width': 1280, 'height': 720})
        time.sleep(1.0)
        shot(page, '90_viewport_1280x720')
        audit(page, 'viewport_1280')
        page.set_viewport_size({'width': 1024, 'height': 700})
        time.sleep(1.0)
        shot(page, '91_viewport_1024x700')
        audit(page, 'viewport_1024')

        browser.close()

    # Compact NaN/overflow: drop empty entries
    report['nan'] = {k: v for k, v in report['nan'].items() if v}
    report['overflow'] = {k: v for k, v in report['overflow'].items() if v.get('overflowing') or v.get('pageScrollX')}
    with open(os.path.join(args.out, f'report_{args.mode}.json'), 'w') as f:
        json.dump(report, f, indent=1)
    print(json.dumps({k: v for k, v in report.items() if k not in ('requests',)}, indent=1))
    print('requests:', len(report['requests']), 'blocked_posts:', report['blocked_posts'])
    print('sample requests:', report['requests'][:8])
    return 0


if __name__ == '__main__':
    sys.exit(main())
