import sys, time
sys.path.insert(0, "<scratch>/scripts")
from lib import *  # noqa

log = Log("raster")
d = driver()
try:
    open_dash(d); time.sleep(3)
    log("activity-live(modular)", grouping=text(d, "activityGrouping"), raster_hidden=js(d, "return document.getElementById('rasterCanvas').hidden"),
        note=text(d, "rasterNote"))
    click(d, "btnReplay"); time.sleep(2)
    items = d.find_elements(By.CSS_SELECTOR, "#replayList button")
    graph = [i for i in items if "auditd-graph" in i.text]
    log("replay-list", items=[i.text for i in items])
    graph[0].click(); time.sleep(5)
    log("replay(graph recording)", info=text(d, "replayInfo"), grouping=text(d, "activityGrouping"),
        raster_hidden=js(d, "return document.getElementById('rasterCanvas').hidden"), note=text(d, "rasterNote"),
        regions=(text(d, "activityRegions") or "")[:200].replace("\n", " | "), ident=text(d, "identBackend"), errs=errs(d))
    shot(d, "raster-replay")
    click(d, "btnReplayExit"); time.sleep(4)
    log("after-exit", raster_hidden=js(d, "return document.getElementById('rasterCanvas').hidden"), grouping=text(d, "activityGrouping"))

    # backend switch from the dropdown while a recording runs (daemon refuses)
    r = post("record_start", name="auditd-guard", record_every=5, raster="none")
    time.sleep(1)
    Select(d.find_element(By.ID, "selectBackend")).select_by_value("connectome-fixed")
    time.sleep(3)
    v1 = d.find_element(By.ID, "selectBackend").get_attribute("value")
    js(d, "document.getElementById('selectBackend').blur()"); time.sleep(3)
    log("selectBackend-while-recording", record_start=r.get("status"), daemon=status()["identity"]["backend"],
        select_right_after=v1, select_after_blur=d.find_element(By.ID, "selectBackend").get_attribute("value"),
        run_state=text(d, "arenaRunState"), cmds=cmds(d), alerts=alerts(d), errs=errs(d))
    shot(d, "backend-while-recording")
    post("record_stop")
finally:
    d.quit()
