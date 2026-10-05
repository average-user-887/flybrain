"""Standalone preview (no daemon reachable): which controls act on the in-browser engine."""
import sys, time
sys.path.insert(0, "<scratch>/scripts")
from lib import *  # noqa

log = Log("preview")
d = driver()
A = "window.app.arena"
try:
    # explicit dead daemon: the page must not try any other address
    d.get(f"{WEB}/index.html?daemon=http://127.0.0.1:8849")
    hook(d)
    time.sleep(8)
    log("page-load(no daemon)", pill=text(d, "clusterStatusPill"), addr=text(d, "daemonAddress"), run_state=text(d, "arenaRunState"),
        remote=js(d, f"return {A}.remoteDriven"), awaiting=js(d, f"return {A}.awaitingDaemon"),
        step=text(d, "statStep"), errs=errs(d))
    s1 = text(d, "statStep"); time.sleep(2); s2 = text(d, "statStep")
    log("local-engine-running", steps=(s1, s2), fly=js(d, f"return [{A}.fly.x,{A}.fly.y]"))
    shot(d, "preview-load")

    # pause
    click(d, "btnPauseToggle"); time.sleep(0.3)
    a = js(d, f"return [{A}.fly.x,{A}.fly.y,{A}.simTime]"); time.sleep(1.5); b = js(d, f"return [{A}.fly.x,{A}.fly.y,{A}.simTime]")
    log("btnPauseToggle(preview)", label=text(d, "btnPauseToggle"), before=a, after=b)
    click(d, "btnPauseToggle")
    click(d, "btnSpeedToggle"); time.sleep(0.3)
    log("btnSpeedToggle(preview)", label=text(d, "btnSpeedToggle"), simSpeed=js(d, "return window.app.hud.simSpeed"))
    Select(d.find_element(By.ID, "selectSpeed")).select_by_value("10"); time.sleep(0.3)
    log("selectSpeed(preview)", simSpeed=js(d, "return window.app.hud.simSpeed"))
    Select(d.find_element(By.ID, "selectSpeed")).select_by_value("1")
    t0 = js(d, f"return {A}.currentTrial")
    click(d, "btnResetTrial"); time.sleep(0.5)
    log("btnResetTrial(preview)", trial=(t0, js(d, f"return {A}.currentTrial")))

    # backend select with no daemon
    Select(d.find_element(By.ID, "selectBackend")).select_by_value("modular"); time.sleep(1.5)
    log("selectBackend(preview)", value=d.find_element(By.ID, "selectBackend").get_attribute("value"), cmds=cmds(d), errs=errs(d),
        options=js(d, "return [...document.getElementById('selectBackend').options].map(o=>o.value)"))

    # tools on the local arena
    canvas = d.find_element(By.ID, "arenaCanvas")
    for tool, off, expr in (("toolFood", (-60, -40), f"{A}.foodItems.length"), ("toolAlarm", (60, 40), f"{A}.alarms.length"),
                            ("toolPredator", (-80, 60), f"{A}.predators.length"), ("toolWind", (90, -70), f"JSON.stringify({A}.windVector)")):
        click(d, tool)
        before = js(d, "return " + expr)
        ActionChains(d).move_to_element_with_offset(canvas, off[0], off[1]).click().perform(); time.sleep(0.5)
        log(f"{tool}(preview)", disabled=not d.find_element(By.ID, tool).is_enabled(), before=before, after=js(d, "return " + expr), cmds=cmds(d))
    shot(d, "preview-tools")
    click(d, "toolSelect")

    # lesions
    for b in ("btnLesionMB", "btnLesionCX", "btnLesionGF", "btnLesionJO", "btnLesionOFF", "btnLesionWT"):
        click(d, b); time.sleep(0.3)
        log(f"{b}(preview)", enabled=d.find_element(By.ID, b).is_enabled(), label=text(d, "activeLesionLabel"), lesion=js(d, f"return {A}.lesion"),
            flags=js(d, f"return [{A}.cx.isLesioned,{A}.mb.plasticityEnabled,{A}.gfLesioned,{A}.joLesioned]"),
            off_used=js(d, f"return /DELTA_OFF/.test(String(Object.getPrototypeOf({A}).constructor))"))

    # guide sliders
    for s in d.find_elements(By.CSS_SELECTOR, "#dynamicSlidersContainer input[type=range]"):
        sid = s.get_attribute("id")
        set_range(d, s, s.get_attribute("max")); time.sleep(0.3)
        log(f"guide:{sid}(preview)", enabled=s.is_enabled(), wind=js(d, f"return JSON.stringify({A}.windVector)"))

    # limb deck
    click(d, "tabLimbDeck")
    click(d, "btnToggleManualControl"); time.sleep(0.3)
    log("btnToggleManualControl(preview)", label=text(d, "btnToggleManualControl"), mode=text(d, "labelControlMode"),
        manual=js(d, f"return !!({A}.paradigmState||{{}}).manualActive"), paradigm=js(d, f"return {A}.activeParadigmId"))
    h0 = js(d, f"return {A}.fly.heading")
    for sid, v in (("sliderStimDna02", "1"), ("sliderStimThrust", "100"), ("sliderStimMdn", "100"), ("sliderStimCpg", "14"), ("sliderStimWing", "90")):
        el = d.find_element(By.ID, sid)
        set_range(d, el, v); time.sleep(0.6)
        log(f"{sid}(preview)", enabled=el.is_enabled(), state=js(d, f"const p={A}.paradigmState||{{}}; return [p.overrideDna02,p.overrideThrust,p.overrideMdn,p.overrideWings,{A}.cpg.baseFreq]"))
    time.sleep(1.5)
    log("manual-stim-effect(open-arena)", heading=(h0, js(d, f"return {A}.fly.heading")), speed=js(d, f"return {A}.fly.speed"))
    # does any paradigm read the overrides?
    for bid, expr in (("btnFlareGf", f"[{A}.dn.dnp01Gf,{A}.dn.escapeActive]"), ("btnFlareHeat", f"({A}.paradigmState||{{}}).temp"),
                      ("btnFlareOdor", f"{A}.dn.dnp09"), ("btnFlareWind", f"JSON.stringify({A}.windVector)")):
        before = js(d, "return " + expr)
        click(d, bid); time.sleep(0.1)
        log(f"{bid}(preview)", before=before, after=js(d, "return " + expr))
    click(d, "btnToggleManualControl")

    # legacy assay tools panel in preview, per assay
    click(d, "tabAssayTools")
    cards = js(d, "return [...document.querySelectorAll('.experiment-card')].map(c=>c.dataset.paradigm)")
    for pid in cards:
        js(d, "document.querySelector(\".experiment-card[data-paradigm='\"+arguments[0]+\"']\").click()", pid)
        time.sleep(0.8)
        sliders = d.find_elements(By.CSS_SELECTOR, "#assaySlidersContainer input[type=range]")
        btns = d.find_elements(By.CSS_SELECTOR, "#assayActionsContainer button")
        res = []
        for s in sliders:
            set_range(d, s, s.get_attribute("max")); time.sleep(0.1)
            res.append((s.get_attribute("id"), "ok"))
        bres = []
        for b in btns:
            lab = b.text
            snap = js(d, f"return JSON.stringify([{A}.fly.x,{A}.fly.y,{A}.fly.heading,{A}.paradigmState,{A}.foodItems.length,{A}.predators.length,{A}.mb.weights?{A}.mb.weights.length:null]).length")
            js(d, "arguments[0].click()", b); time.sleep(0.2)
            snap2 = js(d, f"return JSON.stringify([{A}.fly.x,{A}.fly.y,{A}.fly.heading,{A}.paradigmState,{A}.foodItems.length,{A}.predators.length]).length")
            bres.append(lab)
        log(f"preview-assay:{pid}", active=js(d, f"return {A}.activeParadigmId"), badge=text(d, "navbarParadigmBadge"),
            title=(text(d, "assayToolsPanel") or "")[:60].replace("\n", " | "), sliders=[r[0] for r in res], buttons=bres,
            cmds=cmds(d), errs=errs(d), alerts=alerts(d))
    shot(d, "preview-assaytools")

    # training tab without a daemon
    click(d, "tabTraining"); time.sleep(2)
    log("training(preview)", disabled=js(d, "return ['trainingTeach','trainingReverse','trainingProbe','trainingFreeze','trainingSave','trainingExport'].map(i=>i+':'+document.getElementById(i).disabled)"),
        conn=text(d, "trainingConnection"), msg=text(d, "trainingMessage"))
    # 3D in preview
    click(d, "btnToggle3D"); time.sleep(2)
    log("btnToggle3D(preview)", banner=text(d, "errorBannerText"), errs=errs(d))
    shot(d, "preview-3d")
    click(d, "btnToggle3D")
    # status pill with an unreachable daemon
    click(d, "clusterStatusPill"); time.sleep(5)
    log("clusterStatusPill(preview)", pill=text(d, "clusterStatusPill"), run_state=text(d, "arenaRunState"))
    # replay chooser without daemon
    click(d, "btnReplay"); time.sleep(1)
    log("btnReplay(preview)", list=text(d, "replayList"))
    log("end", errs=errs(d), alerts=alerts(d))
finally:
    d.quit()
