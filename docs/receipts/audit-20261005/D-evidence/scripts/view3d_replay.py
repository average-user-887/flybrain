"""3D view + error banner + replay controls. usage: view3d_replay.py <tag> <3d|replay|inject>"""
import sys, time, glob, os, hashlib
sys.path.insert(0, "<scratch>/scripts")
from lib import *  # noqa

TAG = sys.argv[1]
WHAT = sys.argv[2].split(",")
log = Log(f"v3r-{TAG}")
d = driver()


def canvas_hash(cid="arenaCanvas"):
    data = js(d, f"return document.getElementById('{cid}').toDataURL()")
    return hashlib.sha1(data.encode()).hexdigest()[:10]


def banner():
    return dict(shown=js(d, "return getComputedStyle(document.getElementById('errorBanner')).display"),
                text=text(d, "errorBannerText"),
                resume_hidden=js(d, "return document.getElementById('btnErrorResume').hidden"))


try:
    if "3d" in WHAT:
        open_dash(d); time.sleep(3)
        h1 = canvas_hash(); time.sleep(1.5); h2 = canvas_hash()
        log("2d-before", canvas_updating=h1 != h2, banner=banner(), errs=errs(d))
        click(d, "btnToggle3D"); time.sleep(3)
        v1 = canvas_hash("viewport3DCanvas"); time.sleep(1.5); v2 = canvas_hash("viewport3DCanvas")
        fx = js(d, "const g=window.viewport3D.flyGroup; return g?[g.position.x,g.position.y,g.position.z]:null")
        time.sleep(1.5)
        fx2 = js(d, "const g=window.viewport3D.flyGroup; return g?[g.position.x,g.position.y,g.position.z]:null")
        log("3d-on", banner=banner(), errs=errs(d), webgl_canvas_changes=v1 != v2, fly3d_pos=(fx, fx2),
            arena_fly_keys=js(d, "return Object.keys(window.app.arena.fly).slice(0,25)"))
        shot(d, f"{TAG}-3d-on")
        click(d, "btnToggle3D"); time.sleep(2)
        h1 = canvas_hash(); time.sleep(1.5); h2 = canvas_hash()
        st1 = text(d, "statSimTime")
        log("3d-off", canvas_updating=h1 != h2, banner=banner(), simtime=st1)
        shot(d, f"{TAG}-3d-off-suspended")
        if not js(d, "return document.getElementById('btnErrorResume').hidden"):
            click(d, "btnErrorResume"); time.sleep(2)
            h1 = canvas_hash(); time.sleep(1.5); h2 = canvas_hash()
            log("btnErrorResume", canvas_updating=h1 != h2, banner=banner(), errs=errs(d))
        click(d, "btnErrorDismiss"); time.sleep(0.5)
        log("btnErrorDismiss", banner=banner())

    if "inject" in WHAT:
        open_dash(d, extra="&inject=render"); time.sleep(4)
        b = banner(); h1 = canvas_hash(); time.sleep(1.5); h2 = canvas_hash()
        log("inject=render", banner=b, canvas_updating=h1 != h2, errs=errs(d))
        shot(d, f"{TAG}-inject")
        click(d, "btnErrorResume"); time.sleep(2)
        h1 = canvas_hash(); time.sleep(1.5); h2 = canvas_hash()
        log("btnErrorResume(inject)", banner=banner(), canvas_updating=h1 != h2)
        click(d, "btnErrorDismiss"); time.sleep(0.5)
        log("btnErrorDismiss(inject)", banner=banner())

    if "replay" in WHAT:
        open_dash(d); time.sleep(3)
        click(d, "btnReplay"); time.sleep(2)
        items = d.find_elements(By.CSS_SELECTOR, "#replayList button")
        log("btnReplay(open chooser)", chooser_hidden=js(d, "return document.getElementById('replayChooser').hidden"),
            list=text(d, "replayList")[:400], n=len(items))
        shot(d, f"{TAG}-replay-chooser")
        click(d, "btnReplayChooserClose"); time.sleep(0.3)
        log("btnReplayChooserClose", hidden=js(d, "return document.getElementById('replayChooser').hidden"))
        click(d, "btnReplay"); time.sleep(2)
        items = d.find_elements(By.CSS_SELECTOR, "#replayList button")
        if items:
            items[0].click(); time.sleep(4)
            log("replay-list-item", bar_hidden=js(d, "return document.getElementById('replayBar').hidden"),
                info=text(d, "replayInfo"), time=text(d, "replayTime"), run_state=text(d, "arenaRunState"),
                replayMode=js(d, "return window.app.hud.daemonBridge.replayMode"), errs=errs(d))
            shot(d, f"{TAG}-replay-loaded")
            t1 = text(d, "replayTime"); time.sleep(2); t2 = text(d, "replayTime")
            log("replay-playing", t1=t1, t2=t2)
            click(d, "btnReplayPlay"); time.sleep(0.5)
            t1 = text(d, "replayTime"); time.sleep(2); t2 = text(d, "replayTime")
            log("btnReplayPlay(pause)", label=text(d, "btnReplayPlay"), t1=t1, t2=t2)
            set_range(d, d.find_element(By.ID, "replaySeek"), 800); time.sleep(1)
            log("replaySeek=0.8", time=text(d, "replayTime"), simtime=text(d, "statSimTime"))
            Select(d.find_element(By.ID, "replaySpeed")).select_by_value("5")
            click(d, "btnReplayPlay"); time.sleep(0.3)
            t1 = text(d, "replayTime"); time.sleep(1); t2 = text(d, "replayTime")
            log("replaySpeed=5 + play", t1=t1, t2=t2, label=text(d, "btnReplayPlay"))
            Select(d.find_element(By.ID, "replaySpeed")).select_by_value("1")
            click(d, "btnPauseToggle"); time.sleep(0.5)
            t1 = text(d, "replayTime"); time.sleep(1.5); t2 = text(d, "replayTime")
            log("btnPauseToggle-in-replay", t1=t1, t2=t2, replay_btn=text(d, "btnReplayPlay"), daemon_paused=status()["paused"])
            click(d, "btnPauseToggle"); time.sleep(0.3)
            Select(d.find_element(By.ID, "selectSpeed")).select_by_value("2"); time.sleep(0.5)
            log("selectSpeed-in-replay", replaySpeed=d.find_element(By.ID, "replaySpeed").get_attribute("value"), daemon_speed=status()["sim_speed"])
            card = d.find_element(By.CSS_SELECTOR, ".experiment-card[data-paradigm='y-maze']")
            js(d, "arguments[0].click()", card); time.sleep(1)
            log("card-click-in-replay", run_state=text(d, "arenaRunState"), daemon_paradigm=status()["active_paradigm"])
            s0 = status()["current_trial"]
            click(d, "btnResetTrial"); time.sleep(1)
            log("btnResetTrial-in-replay", trial=(s0, status()["current_trial"]), cmds=cmds(d))
            click(d, "btnReplayExit"); time.sleep(5)
            log("btnReplayExit", bar_hidden=js(d, "return document.getElementById('replayBar').hidden"),
                replayMode=js(d, "return window.app.hud.daemonBridge.replayMode"), connected=connected(d), pill=text(d, "clusterStatusPill"), errs=errs(d))
            Select(d.find_element(By.ID, "selectSpeed")).select_by_value("1")
        # file input path
        recs = sorted(glob.glob(str(SCRATCH / "daemon-*/out/recordings/*.nfrec")))
        if recs:
            click(d, "btnReplay"); time.sleep(1)
            d.find_element(By.ID, "replayFileInput").send_keys(recs[0]); time.sleep(4)
            log("replayFileInput", file=os.path.basename(recs[0]), bar_hidden=js(d, "return document.getElementById('replayBar').hidden"),
                info=text(d, "replayInfo"), time=text(d, "replayTime"), errs=errs(d))
            shot(d, f"{TAG}-replay-file")
            click(d, "btnReplayExit"); time.sleep(3)
    log("end", errs=errs(d), alerts=alerts(d))
finally:
    d.quit()
