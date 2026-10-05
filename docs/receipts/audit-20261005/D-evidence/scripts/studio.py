"""Experiment studio controls in Firefox. usage: studio.py build|results"""
import sys, time, glob, os, json, zipfile
sys.path.insert(0, "<scratch>/scripts")
from lib import *  # noqa

BASE = "http://127.0.0.1:8842/"
import getpass
# Strings that must not appear in a shared bundle (the run used the literal values).
LOCAL_PATH_NEEDLES = (os.path.expanduser("~") + "/", str(SCRATCH.parents[2]) + "/", getpass.getuser())
PHASE = sys.argv[1]
log = Log(f"studio-{PHASE}")
d = driver()


def tab(name):
    d.find_element(By.CSS_SELECTOR, f"nav [data-tab='{name}']").click(); time.sleep(1)
    return js(d, "return [...document.querySelectorAll('main > section')].filter(s=>!s.hidden).map(s=>s.id)")


def sapi(path):
    return get(path, base="http://127.0.0.1:8842")


try:
    d.get(BASE); hook(d); time.sleep(3)
    log("studio-load", worker=text(d, "worker"), error=text(d, "global-error"), curated=text(d, "curated")[:200],
        finished=text(d, "finished")[:200], errs=errs(d))
    shot(d, f"studio-{PHASE}-gallery")
    for t in ("build", "queue", "compare", "gallery"):
        log(f"tab:{t}", visible=tab(t), selected=js(d, "return [...document.querySelectorAll('nav [data-tab]')].map(b=>b.dataset.tab+':'+b.getAttribute('aria-selected'))"))

    if PHASE == "build":
        tab("build")
        cards = d.find_elements(By.CSS_SELECTOR, "#paradigms .card")
        builds = d.find_elements(By.CSS_SELECTOR, "[data-build]")
        log("build-catalog", cards=len(cards), buildable=[b.get_attribute("data-build") for b in builds],
            badges=sorted(set(e.text for e in d.find_elements(By.CSS_SELECTOR, "#paradigms .badge"))))
        builds[0].click(); time.sleep(1.5)
        log("Build(optomotor)", builder_hidden=js(d, "return document.getElementById('builder').hidden"), title=text(d, "b-title"),
            params=[e.get_attribute("data-param") for e in d.find_elements(By.CSS_SELECTOR, "[data-param]")],
            silence=[e.get_attribute("data-silence") for e in d.find_elements(By.CSS_SELECTOR, "[data-silence]")],
            controls=js(d, "return [...document.querySelectorAll('input[name=b-control]')].map(r=>r.value+':'+(r.checked?'checked':'')+(r.disabled?'disabled':''))"),
            prereg=text(d, "b-prereg")[:300])
        shot(d, "studio-builder")
        # range <-> number sync
        rng = d.find_element(By.CSS_SELECTOR, "[data-range='world_angular_velocity_rad_s']")
        set_range(d, rng, "-6")
        n1 = d.find_element(By.ID, "p-world_angular_velocity_rad_s").get_attribute("value")
        num = d.find_element(By.ID, "p-duration_s"); num.clear(); num.send_keys("0.5")
        r2 = d.find_element(By.CSS_SELECTOR, "[data-range='duration_s']").get_attribute("value")
        log("param range<->number", range_to_number=n1, number_to_range=r2,
            seed_range_hidden=js(d, "return document.querySelector(\"[data-range='seed']\").hidden"))
        # silencing changes the default control
        Select(d.find_element(By.ID, "s-dna02")).select_by_value("both"); time.sleep(0.3)
        c1 = js(d, "return [...document.querySelectorAll('input[name=b-control]')].map(r=>r.value+':'+(r.checked?'checked':'')+(r.disabled?'disabled':''))")
        Select(d.find_element(By.ID, "s-dna02")).select_by_value(""); time.sleep(0.3)
        c2 = js(d, "return [...document.querySelectorAll('input[name=b-control]')].map(r=>r.value+':'+(r.checked?'checked':'')+(r.disabled?'disabled':''))")
        log("silence-select->controls", silenced=c1, unsilenced=c2)
        # invalid submit (contrast out of range, bypassing the browser's own check)
        js(d, "document.getElementById('builder').noValidate=true")
        c = d.find_element(By.ID, "p-contrast"); c.clear(); c.send_keys("3")
        if os.environ.get("SKIP") != "1": d.find_element(By.ID, "b-submit").click(); time.sleep(2)
        log("submit(invalid contrast=3)", result=text(d, "b-result"), cls=d.find_element(By.ID, "b-result").get_attribute("class"))
        c.clear(); c.send_keys("1")
        # valid: silence DNa02, intact control (2 runs)
        SKIP = os.environ.get("SKIP") == "1"
        d.find_element(By.ID, "b-name").clear(); d.find_element(By.ID, "b-name").send_keys("auditD dna02 silenced")
        Select(d.find_element(By.ID, "s-dna02")).select_by_value("both"); time.sleep(0.3)
        seed = d.find_element(By.ID, "p-seed"); seed.clear(); seed.send_keys("21")
        if not SKIP: d.find_element(By.ID, "b-submit").click(); time.sleep(3)
        log("submit(dna02 silenced + intact)", result=text(d, "b-result"), queue=[(j["name"], j["state"], j.get("role")) for j in sapi("/api/studio/runs")["queue"]])
        # valid: nothing silenced + output-disconnected control, 1 repeat
        Select(d.find_element(By.ID, "s-dna02")).select_by_value(""); time.sleep(0.3)
        d.find_element(By.ID, "b-name").clear(); d.find_element(By.ID, "b-name").send_keys("auditD intact")
        seed.clear(); seed.send_keys("22")
        if not SKIP: d.find_element(By.ID, "b-submit").click(); time.sleep(3)
        log("submit(intact + disconnected)", result=text(d, "b-result"))
        # a set to cancel: 2 repeats, left side T4/T5
        Select(d.find_element(By.ID, "s-t4t5")).select_by_value("L"); time.sleep(0.3)
        Select(d.find_element(By.ID, "b-repeats")).select_by_visible_text("2")
        d.find_element(By.ID, "b-name").clear(); d.find_element(By.ID, "b-name").send_keys("auditD cancel me")
        seed.clear(); seed.send_keys("30")
        d.find_element(By.ID, "b-submit").click(); time.sleep(3)
        log("submit(repeats=2, T4/T5 left)", result=text(d, "b-result"))
        tab("queue"); time.sleep(5)
        rows = d.find_elements(By.CSS_SELECTOR, "#queue-rows tr")
        log("queue-rows", rows=[r.text.replace("\n", " | ")[:160] for r in rows], worker=text(d, "worker"))
        shot(d, "studio-queue")
        cancels = d.find_elements(By.CSS_SELECTOR, "[data-cancel]")
        names = [c.get_attribute("data-cancel") for c in cancels if "cancel" in c.get_attribute("data-cancel")]
        for n in names:
            b = d.find_element(By.CSS_SELECTOR, f"[data-cancel='{n}']"); b.click(); time.sleep(2)
        time.sleep(3)
        q = sapi("/api/studio/runs")["queue"]
        log("Cancel", cancelled=names, remaining=[(j["name"], j["state"]) for j in q], global_error=text(d, "global-error"))
        # try cancelling a running job via API (no UI button is offered)
        running = [j["name"] for j in q if j["state"] == "running"]
        log("queue-running-has-cancel-button", running=running,
            buttons=[c.get_attribute("data-cancel") for c in d.find_elements(By.CSS_SELECTOR, "[data-cancel]")])

    if PHASE == "results":
        tab("gallery")
        cards = d.find_elements(By.CSS_SELECTOR, "#finished .card, #curated .card")
        log("gallery-cards", n=len(cards), text=[c.text.replace("\n", " | ")[:220] for c in cards])
        shot(d, "studio-results-gallery")
        # Watch
        watch = d.find_elements(By.CSS_SELECTOR, "#finished a.act[target=_blank], #curated a.act[target=_blank]")
        if watch and os.environ.get("SKIPWATCH") != "1":
            href = watch[0].get_attribute("href")
            main = d.current_window_handle
            watch[0].click(); time.sleep(4)
            handles = [h for h in d.window_handles if h != main]
            if handles:
                d.switch_to.window(handles[-1]); hook(d); time.sleep(4)
                info = js(d, "const r=window.embodiedReplay; return r?{frames:r.frames&&r.frames.length,duration:r.duration,frame:r.frame}:null")
                log("Watch(embodied_replay)", url=href, replay=info, play_disabled=not d.find_element(By.ID, "play").is_enabled(),
                    errs=errs(d), body=d.find_element(By.TAG_NAME, "body").text[:300].replace("\n", " | "))
                shot(d, "studio-watch")
                # embodied replay controls
                d.find_element(By.ID, "play").click(); time.sleep(2)
                f1 = js(d, "return document.getElementById('seek').value")
                d.find_element(By.ID, "play").click(); time.sleep(0.5)
                f2 = js(d, "return document.getElementById('seek').value"); time.sleep(1)
                f3 = js(d, "return document.getElementById('seek').value")
                set_range(d, d.find_element(By.ID, "seek"), "0"); time.sleep(0.3)
                f4 = js(d, "return document.getElementById('seek').value")
                Select(d.find_element(By.ID, "speed")).select_by_index(0)
                fol = d.find_element(By.ID, "follow"); fol.click()
                log("embodied_replay controls", play_advanced=f1, paused_holds=(f2, f3), seek0=f4,
                    speed_options=[o.text for o in Select(d.find_element(By.ID, "speed")).options], follow=fol.is_selected(), errs=errs(d))
                d.close(); d.switch_to.window(main)
        # Download bundle
        dl = d.find_elements(By.CSS_SELECTOR, "#finished a[download], #curated a[download]")
        before = set(glob.glob(str(DL / "*.zip")))
        if dl:
            dl_href = dl[0].get_attribute("href")
            js(d, "document.querySelector('#curated a[download], #finished a[download]').click()"); time.sleep(5)
            new = sorted(set(glob.glob(str(DL / "*.zip"))) - before)
            info = {}
            for f in new:
                with zipfile.ZipFile(f) as z:
                    info[os.path.basename(f)] = {"files": z.namelist()[:20], "bytes": os.path.getsize(f)}
                    leak = [n for n in z.namelist() if n.endswith((".json", ".jsonl"))]
                    hits = []
                    for n in leak:
                        txt = z.read(n).decode("utf-8", "replace")
                        for needle in LOCAL_PATH_NEEDLES:  # home dir prefix, media mount prefix, account name (redacted)
                            if needle in txt:
                                hits.append((n, needle))
                    info[os.path.basename(f)]["path_leaks"] = hits
            log("Download(bundle)", href=dl_href, saved=info)
        # Compare with pair
        pair = d.find_elements(By.CSS_SELECTOR, "[data-compare][data-with]")
        if pair:
            js(d, "document.querySelector('[data-compare][data-with]').click()"); time.sleep(6)
            log("Compare with pair", tab=js(d, "return [...document.querySelectorAll('main > section')].filter(s=>!s.hidden).map(s=>s.id)"),
                a=d.find_element(By.ID, "cmp-a").get_attribute("value"), b=d.find_element(By.ID, "cmp-b").get_attribute("value"),
                table=text(d, "cmp-table")[:600], sync_hidden=js(d, "return document.getElementById('sync').hidden"), errs=errs(d))
            shot(d, "studio-compare")
            fr = lambda: js(d, "return ['cmp-frame-a','cmp-frame-b'].map(id=>{try{const r=document.getElementById(id).contentWindow.embodiedReplay;return r?r.frame:null}catch(e){return 'x'}})")
            if not js(d, "return document.getElementById('sync').hidden"):
                Select(d.find_element(By.ID, "sync-speed")).select_by_value("0.25")
                d.find_element(By.ID, "sync-play").click(); time.sleep(0.8); a1 = fr(); c1 = text(d, "sync-clock"); time.sleep(0.6); a2 = fr(); c2 = text(d, "sync-clock")
                d.find_element(By.ID, "sync-play").click(); time.sleep(0.2); p1 = (fr(), text(d, "sync-clock"), text(d, "sync-play")); time.sleep(1.2); p2 = (fr(), text(d, "sync-clock"))
                set_range(d, d.find_element(By.ID, "sync-seek"), "500"); time.sleep(0.5); s1 = fr(); sc = text(d, "sync-clock")
                Select(d.find_element(By.ID, "sync-speed")).select_by_value("4")
                d.find_element(By.ID, "sync-play").click(); time.sleep(3); e1 = fr()
                log("synced compare", play=(a1, c1, a2, c2), pause=(p1, p2), seek50=(s1, sc), speed4_end=(e1, text(d, "sync-clock"), text(d, "sync-play")), errs=errs(d))
        # Compare… on a run without pair
        lone = d.find_elements(By.CSS_SELECTOR, "[data-compare]:not([data-with])")
        tab("gallery")
        lone = d.find_elements(By.CSS_SELECTOR, "[data-compare]:not([data-with])")
        if lone:
            js(d, "document.querySelector('[data-compare]:not([data-with])').click()"); time.sleep(3)
            log("Compare…(no pair)", a=d.find_element(By.ID, "cmp-a").get_attribute("value"), b=d.find_element(By.ID, "cmp-b").get_attribute("value"),
                table=text(d, "cmp-table")[:300])
        # manual selects
        opts = [o.get_attribute("value") for o in Select(d.find_element(By.ID, "cmp-b")).options if o.get_attribute("value")]
        if len(opts) >= 2:
            js(d, "for (const [id,v] of [['cmp-a',arguments[0]],['cmp-b',arguments[1]]]) {const s=document.getElementById(id); s.value=v; s.dispatchEvent(new Event('change',{bubbles:true}));}", opts[0], opts[-1]); time.sleep(4)
            log("cmp-a/cmp-b selects", a=opts[0], b=opts[-1], table=text(d, "cmp-table")[:400], sync_hidden=js(d, "return document.getElementById('sync').hidden"))
        # phone width
        d.set_window_size(390, 900); time.sleep(1)
        log("phone-width", overflow=js(d, "return document.documentElement.scrollWidth > document.documentElement.clientWidth"))
        d.set_window_size(1700, 1150)
    log("end", errs=errs(d))
finally:
    d.quit()
