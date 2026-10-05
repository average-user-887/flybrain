"""Audit D live-dashboard pass. usage: live.py <tag> <section,section,...|all>"""
import sys, time, json, glob, os
sys.path.insert(0, "<scratch>/scripts")
from lib import *  # noqa

TAG = sys.argv[1]
SECTIONS = sys.argv[2].split(",")
log = Log(f"live-{TAG}")
d = driver()


def want(s):
    return "all" in SECTIONS or s in SECTIONS


def deck(tab):
    click(d, tab)
    time.sleep(0.4)


def sc(name):
    return shot(d, f"{TAG}-{name}")


try:
    ok = open_dash(d)
    log("page-load", connected=bool(ok), pill=text(d, "clusterStatusPill"), errs=errs(d))
    time.sleep(2)

    # ---------------------------------------------------------------- navbar
    if want("nav"):
        s0 = status()
        cmds(d)
        click(d, "btnPauseToggle")
        p1 = wait(lambda: status()["paused"] is True, 20)
        time.sleep(1)
        log("btnPauseToggle(pause)", daemon_paused=status()["paused"], btn=text(d, "btnPauseToggle"),
            run_state=text(d, "arenaRunState"), cmds=cmds(d), ok=bool(p1))
        st1 = status()["total_steps"]; time.sleep(2); st2 = status()["total_steps"]
        log("pause-holds", steps_during_pause=st2 - st1)
        sc("paused")
        click(d, "btnPauseToggle")
        p2 = wait(lambda: status()["paused"] is False, 20)
        time.sleep(1)
        log("btnPauseToggle(resume)", daemon_paused=status()["paused"], btn=text(d, "btnPauseToggle"), cmds=cmds(d), ok=bool(p2))

        click(d, "btnSpeedToggle")
        time.sleep(1.5)
        log("btnSpeedToggle", daemon_speed=status()["sim_speed"], btn=text(d, "btnSpeedToggle"),
            stat=text(d, "statSpeed"), select=d.find_element(By.ID, "selectSpeed").get_attribute("value"), cmds=cmds(d))
        Select(d.find_element(By.ID, "selectSpeed")).select_by_value("5")
        time.sleep(1.5)
        log("selectSpeed=5", daemon_speed=status()["sim_speed"], btn=text(d, "btnSpeedToggle"), cmds=cmds(d))
        time.sleep(3)
        log("achieved@5x", achieved=text(d, "statAchieved"))
        Select(d.find_element(By.ID, "selectSpeed")).select_by_value("1")
        time.sleep(1.5)
        log("selectSpeed=1", daemon_speed=status()["sim_speed"], cmds=cmds(d))

        s0 = status(); t0 = tel()
        click(d, "btnResetTrial")
        time.sleep(2)
        s1 = status(); t1 = tel()
        log("btnResetTrial", trial_before=s0["current_trial"], trial_after=s1["current_trial"],
            seg_changed=t0["segment_id"] != t1["segment_id"], transition=t1.get("transition"),
            run_state=text(d, "arenaRunState"), cmds=cmds(d))

        click(d, "clusterStatusPill")
        time.sleep(4)
        log("clusterStatusPill(click)", connected=connected(d), pill=text(d, "clusterStatusPill"), errs=errs(d))

        click(d, "researchLink")
        time.sleep(1)
        log("researchLink", training_active="active" in d.find_element(By.ID, "tabTraining").get_attribute("class"),
            panel=js(d, "return document.getElementById('trainingPanel').style.display"))
        deck("tabGuide")

        # deck tabs
        for tab, panel in (("tabGuide", "guideTabContent"), ("tabAssayTools", "assayToolsPanel"),
                           ("tabLimbDeck", "limbDeckPanel"), ("tabTraining", "trainingPanel")):
            deck(tab)
            log(tab, shown=js(d, f"return getComputedStyle(document.getElementById('{panel}')).display"),
                others=js(d, "return ['guideTabContent','assayToolsPanel','limbDeckPanel','trainingPanel'].map(i=>i+':'+getComputedStyle(document.getElementById(i)).display)"))
            sc(f"deck-{tab}")
        deck("tabGuide")

    # ---------------------------------------------------------------- arena tools
    if want("tools"):
        if status()["active_paradigm"] != "open-arena":
            post("switch_paradigm", paradigm="open-arena"); time.sleep(3)
        canvas = d.find_element(By.ID, "arenaCanvas")
        w, h = canvas.size["width"], canvas.size["height"]
        for tool, off in (("toolSelect", (40, 40)), ("toolFood", (-60, -40)), ("toolAlarm", (60, 50)),
                          ("toolPredator", (-80, 60)), ("toolWind", (90, -70))):
            click(d, tool)
            mode = js(d, "return window.app.arena.toolMode")
            sc0 = tel()["scene"]; w0 = tel()["stimuli"]["wind"]
            cmds(d); alerts(d)
            ActionChains(d).move_to_element_with_offset(canvas, off[0], off[1]).click().perform()
            time.sleep(2)
            sc1 = tel()["scene"]; w1 = tel()["stimuli"]["wind"]
            log(tool, toolMode=mode, active=("active" in d.find_element(By.ID, tool).get_attribute("class")),
                food=(len(sc0["food"]), len(sc1["food"])), hazards=(len(sc0["hazards"]), len(sc1["hazards"])),
                predators=(len(sc0["predators"]), len(sc1["predators"])), wind=(w0, w1),
                cmds=cmds(d), alerts=alerts(d), errs=errs(d))
            sc(f"tool-{tool}")
        # tools in another assay (not open arena)
        post("switch_paradigm", paradigm="t-maze"); time.sleep(4)
        click(d, "toolFood"); cmds(d)
        ActionChains(d).move_to_element_with_offset(canvas, 10, 10).click().perform(); time.sleep(1.5)
        log("toolFood@t-maze", cmds=cmds(d), alerts=alerts(d), run_state=text(d, "arenaRunState"),
            tools_visible=js(d, "return getComputedStyle(document.querySelector('.arena-overlay-tools')).display"))
        post("switch_paradigm", paradigm="open-arena"); time.sleep(4)
        click(d, "toolSelect")

        # 3D toggle and camera
        click(d, "btnToggle3D"); time.sleep(2)
        log("btnToggle3D(on)", label=text(d, "btnToggle3D"), v3d=js(d, "return getComputedStyle(document.getElementById('viewport3DContainer')).display"),
            arena2d=js(d, "return getComputedStyle(document.getElementById('arenaCanvas')).display"),
            cam_btn=js(d, "return getComputedStyle(document.getElementById('btnCameraMode')).display"),
            init=js(d, "return !!(window.viewport3D&&window.viewport3D.initialized)"), errs=errs(d))
        sc("3d-orbit")
        cam0 = js(d, "const c=window.viewport3D.camera.position; return [c.x,c.y,c.z]")
        click(d, "btnCameraMode"); time.sleep(2)
        cam1 = js(d, "const c=window.viewport3D.camera.position; return [c.x,c.y,c.z]")
        log("btnCameraMode(chase)", label=text(d, "btnCameraMode"), mode=js(d, "return window.viewport3D.cameraMode"),
            camera_moved=cam0 != cam1, cam0=cam0, cam1=cam1)
        sc("3d-chase")
        click(d, "btnCameraMode"); time.sleep(1)
        log("btnCameraMode(orbit)", label=text(d, "btnCameraMode"), mode=js(d, "return window.viewport3D.cameraMode"))
        # tool click while in 3D
        click(d, "toolFood"); cmds(d)
        v3 = d.find_element(By.ID, "viewport3DCanvas")
        ActionChains(d).move_to_element_with_offset(v3, 20, 20).click().perform(); time.sleep(1)
        log("toolFood-in-3D", cmds=cmds(d), note="2D canvas hidden in 3D view")
        click(d, "toolSelect")
        click(d, "btnToggle3D"); time.sleep(1)
        log("btnToggle3D(off)", label=text(d, "btnToggle3D"),
            arena2d=js(d, "return getComputedStyle(document.getElementById('arenaCanvas')).display"),
            cam_btn=js(d, "return getComputedStyle(document.getElementById('btnCameraMode')).display"))

    # ---------------------------------------------------------------- lesions, guide modal
    if want("lesion"):
        for b in ("btnLesionMB", "btnLesionCX", "btnLesionGF", "btnLesionJO", "btnLesionOFF", "btnLesionWT"):
            t0 = tel(); cmds(d)
            click(d, b); time.sleep(1.5)
            t1 = tel()
            log(b, label=text(d, "activeLesionLabel"), card=text(d, "lesionCardName"),
                local_lesion=js(d, "return window.app.arena.lesion"), cmds=cmds(d),
                daemon_identity_same=t0["identity"] == t1["identity"],
                daemon_has_lesion_field=any("lesion" in k for k in t1.keys()))
        sc("lesion")
        click(d, "btnOpenScienceGuide"); time.sleep(0.5)
        o = js(d, "return document.getElementById('scienceGuideModal').style.display")
        click(d, "btnCloseScienceGuide"); time.sleep(0.3)
        c = js(d, "return document.getElementById('scienceGuideModal').style.display")
        log("btnOpenScienceGuide/btnCloseScienceGuide", opened=o, closed=c)
        click(d, "btnOpenScienceGuide"); time.sleep(0.3)
        sc("science-modal")
        d.find_element(By.TAG_NAME, "body").send_keys("")
        time.sleep(0.3)
        log("scienceGuide-Escape", display=js(d, "return document.getElementById('scienceGuideModal').style.display"))
        click(d, "btnOpenScienceGuide"); time.sleep(0.3)
        js(d, "document.getElementById('scienceGuideModal').dispatchEvent(new MouseEvent('click',{bubbles:true}))")
        time.sleep(0.3)
        log("scienceGuide-backdrop", display=js(d, "return document.getElementById('scienceGuideModal').style.display"))

    # ---------------------------------------------------------------- guide sliders (standalone preview)
    if want("guide"):
        deck("tabGuide")
        sliders = d.find_elements(By.CSS_SELECTOR, "#dynamicSlidersContainer input[type=range]")
        for s in sliders:
            sid = s.get_attribute("id"); t0 = tel(); cmds(d)
            set_range(d, s, s.get_attribute("max")); time.sleep(1.5)
            t1 = tel()
            log(f"guide:{sid}", paradigm=t1["paradigm"], cmds=cmds(d),
                daemon_stimuli=(t0["stimuli"], t1["stimuli"]),
                shown=text(d, "val_" + sid.replace("slider_", "")))

    # ---------------------------------------------------------------- limb & neuro-stim deck
    if want("limb"):
        deck("tabLimbDeck")
        t0 = tel(); cmds(d)
        click(d, "btnToggleManualControl"); time.sleep(1)
        log("btnToggleManualControl", label=text(d, "btnToggleManualControl"), mode=text(d, "labelControlMode"),
            cmds=cmds(d), local_manual=js(d, "return !!(window.app.arena.paradigmState||{}).manualActive"))
        sc("limb-manual")
        for sid, v in (("sliderStimDna02", "1"), ("sliderStimThrust", "100"), ("sliderStimMdn", "100"),
                       ("sliderStimCpg", "14"), ("sliderStimWing", "90")):
            ta = tel()
            set_range(d, d.find_element(By.ID, sid), v); time.sleep(2)
            tb = tel()
            log(sid, cmds=cmds(d), shown=text(d, sid.replace("slider", "val")),
                daemon_dna02=(ta["descending"].get("dna02_yaw"), tb["descending"].get("dna02_yaw")),
                daemon_speed=(ta["fly"]["speed"], tb["fly"]["speed"]))
        for bid in ("btnFlareGf", "btnFlareHeat", "btnFlareOdor", "btnFlareWind"):
            ta = tel()
            click(d, bid); time.sleep(1.5)
            tb = tel()
            log(bid, cmds=cmds(d), errs=errs(d), daemon_state=(ta["fly"].get("state"), tb["fly"].get("state")),
                daemon_wind=(ta["stimuli"].get("wind"), tb["stimuli"].get("wind")))
        click(d, "btnToggleManualControl"); time.sleep(0.5)
        log("btnToggleManualControl(off)", label=text(d, "btnToggleManualControl"), mode=text(d, "labelControlMode"))

    # ---------------------------------------------------------------- telemetry export panel
    if want("telemetry"):
        before = set(glob.glob(str(DL / "*")))
        click(d, "btnDownloadCsv"); time.sleep(2)
        click(d, "btnDownloadJson"); time.sleep(2)
        new = sorted(set(glob.glob(str(DL / "*"))) - before)
        info = {}
        for f in new:
            info[os.path.basename(f)] = os.path.getsize(f)
        log("btnDownloadCsv+btnDownloadJson", files=info, steps_label=text(d, "telemetryStepCount"))
        for f in new:
            if f.endswith(".csv"):
                with open(f) as fh:
                    lines = fh.readlines()
                log("csv-content", rows=len(lines), header=lines[0][:300] if lines else None, row1=lines[1][:300] if len(lines) > 1 else None)
            if f.endswith(".json"):
                try:
                    j = json.load(open(f))
                    log("json-content", keys=list(j.keys())[:30], identity=j.get("identity") or j.get("run", {}))
                except Exception as e:  # noqa: BLE001
                    log("json-content", error=str(e))
        click(d, "btnClearTelemetry"); time.sleep(0.2)
        a = text(d, "telemetryStepCount"); time.sleep(3)
        log("btnClearTelemetry", immediately=a, after3s=text(d, "telemetryStepCount"),
            buf=js(d, "return window.app.arena.telemetryBuffer.length"))
        # research summary links in the Training tab
        deck("tabTraining")
        hrefs = js(d, "return [...document.querySelectorAll('#trainingPanel a[download]')].map(a=>a.href)")
        res = {}
        import urllib.request, urllib.error
        for hrf in hrefs:
            try:
                with urllib.request.urlopen(hrf, timeout=5) as r:
                    res[hrf] = r.status
            except urllib.error.HTTPError as e:
                res[hrf] = e.code
        log("research-download-links", links=res)
        deck("tabGuide")

    # ---------------------------------------------------------------- preview wall assist
    if want("assist"):
        t0 = tel(); cmds(d)
        click(d, "previewWallAssist"); time.sleep(1.5)
        t1 = tel()
        log("previewWallAssist(uncheck)", checked=d.find_element(By.ID, "previewWallAssist").is_selected(),
            local=js(d, "return window.app.arena.previewWallAssist"), cmds=cmds(d),
            daemon_assists=(t0["motor"]["motor_assists"], t1["motor"]["motor_assists"]), ident_assists=text(d, "identAssists"))
        click(d, "previewWallAssist"); time.sleep(0.5)

    # ---------------------------------------------------------------- training & data
    if want("training"):
        deck("tabTraining"); time.sleep(2)
        dis = js(d, "return ['trainingTeach','trainingReverse','trainingProbe','trainingFreeze','trainingSave','trainingExport'].map(i=>i+':'+document.getElementById(i).disabled)")
        log("training-initial", disabled=dis, conn=text(d, "trainingConnection"), msg=text(d, "trainingMessage"),
            title=text(d, "trainingTitle"), brain=text(d, "trainingBrainId"))
        sc("training")
        obs0 = get("/api/observatory")
        b0 = obs0["brain"]
        for bid, check in (("trainingProbe", None), ("trainingTeach", "teach"), ("trainingReverse", "teach"),
                           ("trainingFreeze", None), ("trainingFreeze", None), ("trainingSave", None)):
            wait(lambda: not d.find_element(By.ID, bid).get_attribute("disabled"), 120)
            ob = get("/api/observatory")["brain"]
            hist0 = len(ob["history"])
            click(d, bid); time.sleep(2.5)
            ob2 = get("/api/observatory")["brain"]
            log(bid, label=text(d, bid), msg=text(d, "trainingMessage"), phase=text(d, "trainingPhase"),
                teaching=ob2.get("teaching"), learning=(ob["learning_enabled"], ob2["learning_enabled"]),
                history_kinds=[e["kind"] for e in ob2["history"][-3:]], hist_len=(hist0, len(ob2["history"])),
                weight_l2=(ob["weight_change_l2"], ob2["weight_change_l2"]), disc=(ob["probe"].get("discrimination"), ob2["probe"].get("discrimination")),
                errs=errs(d))
            if check == "teach":
                done = wait(lambda: not get("/api/observatory")["brain"].get("teaching"), 240, 2)
                ob3 = get("/api/observatory")["brain"]
                log(bid + "(finished)", finished=bool(done), weight_l2=ob3["weight_change_l2"],
                    disc=ob3["probe"].get("discrimination"), A=ob3["probe"].get("A"), B=ob3["probe"].get("B"))
        ck = sorted(glob.glob(str(SCRATCH / f"daemon-*/out/**/*instrument*"), recursive=True))
        log("trainingSave-files", files=ck[-3:])
        before = set(glob.glob(str(DL / "*")))
        click(d, "trainingExport"); time.sleep(2)
        new = sorted(set(glob.glob(str(DL / "*"))) - before)
        log("trainingExport", files={os.path.basename(f): os.path.getsize(f) for f in new})
        sc("training-after")
        deck("tabGuide")

    # ---------------------------------------------------------------- all 14 assays + live controls
    if want("assays"):
        cards = js(d, "return [...document.querySelectorAll('.experiment-card')].map(c=>c.dataset.paradigm)")
        log("experiment-cards", cards=cards)
        deck("tabAssayTools")
        for pid in cards:
            card = d.find_element(By.CSS_SELECTOR, f".experiment-card[data-paradigm='{pid}']")
            d.execute_script("arguments[0].scrollIntoView({block:'center'})", card)
            t_click = time.time()
            card.click()
            okp = wait(lambda: status()["active_paradigm"] == pid and js(d, "return window.app.arena.remotePacket&&window.app.arena.remotePacket.paradigm") == pid, 120, 0.5)
            dt = round(time.time() - t_click, 1)
            time.sleep(2)
            badge = text(d, "navbarParadigmBadge")
            active = js(d, "return [...document.querySelectorAll('.experiment-card.active')].map(c=>c.dataset.paradigm)")
            tl = tel()
            la = tl.get("live_assay") or {}
            entry = dict(switched=bool(okp), seconds=dt, badge=badge, active_card=active, step=tl["step"],
                         panel_title=(d.find_element(By.ID, "assayToolsPanel").text or "")[:90].replace("\n", " | "),
                         params=[p["name"] for p in la.get("parameters", [])], actions=[a["name"] for a in la.get("actions", [])],
                         cmds=cmds(d), errs=errs(d))
            log(f"card:{pid}", **entry)
            sc(f"assay-{pid}")
            # each connected parameter
            for p in la.get("parameters", []):
                el = d.find_element(By.ID, "live_param_" + p["name"])
                nv = p["max"] if p["value"] != p["max"] else p["min"]
                set_range(d, el, nv); time.sleep(2.5)
                la2 = tel().get("live_assay") or {}
                after = {q["name"]: q["value"] for q in la2.get("parameters", [])}.get(p["name"])
                log(f"live_param:{pid}:{p['name']}", sent=nv, before=p["value"], after=after,
                    status_line=text(d, "liveCommandStatus"), cmds=cmds(d))
                if p["name"] == "gapWidth":
                    # also test the guarded case
                    pass
            for a in la.get("actions", []):
                s0 = tel()
                click(d, "live_action_" + a["name"]); time.sleep(2.5)
                s1 = tel()
                log(f"live_action:{pid}:{a['name']}", status_line=text(d, "liveCommandStatus"), cmds=cmds(d),
                    scene_changed=s0.get("scene") != s1.get("scene"), assay_state_changed=s0.get("assay_state") != s1.get("assay_state"),
                    scene_after=str(s1.get("scene"))[:300], assay_state_after=str(s1.get("assay_state"))[:300])
        # rapid consecutive selections
        seq = ["y-maze", "buridan", "optomotor", "courtship"]
        for pid in seq:
            js(d, "document.querySelector(\".experiment-card[data-paradigm='\"+arguments[0]+\"']\").click()", pid)
            time.sleep(0.15)
        okr = wait(lambda: status()["active_paradigm"] == seq[-1] and js(d, "return window.app.arena.remotePacket.paradigm") == seq[-1], 180, 0.5)
        time.sleep(2)
        log("rapid-card-switch", target=seq[-1], landed=status()["active_paradigm"], page=js(d, "return window.app.arena.remotePacket.paradigm"),
            badge=text(d, "navbarParadigmBadge"), active=js(d, "return [...document.querySelectorAll('.experiment-card.active')].map(c=>c.dataset.paradigm)"),
            ok=bool(okr), cmds=[c["req"] for c in cmds(d)], errs=errs(d))
        deck("tabGuide")

    log("end-errors", errs=errs(d), alerts=alerts(d))
finally:
    d.quit()
