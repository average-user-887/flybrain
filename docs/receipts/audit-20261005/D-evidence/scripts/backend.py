"""selectBackend live: switch the running daemon's controller from the dashboard. usage: backend.py <target>"""
import sys, time
sys.path.insert(0, "<scratch>/scripts")
from lib import *  # noqa

target = sys.argv[1]
log = Log("backend")
d = driver()
try:
    open_dash(d); time.sleep(2)
    before = status()["identity"]["backend"]
    log("selectBackend-options", options=js(d, "return [...document.getElementById('selectBackend').options].map(o=>o.value+'='+o.text)"),
        shown=d.find_element(By.ID, "selectBackend").get_attribute("value"), daemon=before)
    t0 = time.time()
    Select(d.find_element(By.ID, "selectBackend")).select_by_value(target)
    js(d, "document.getElementById('selectBackend').blur()")
    samples = []
    ok = None
    for i in range(400):
        time.sleep(1.5)
        try:
            s = status()
            b = s["identity"]["backend"]
        except Exception as e:  # noqa: BLE001 - daemon busy while it loads the graph
            b = f"unreachable: {type(e).__name__}"
        if i % 4 == 0:
            samples.append((round(time.time() - t0), b, text(d, "arenaRunState"), text(d, "clusterStatusPill"), text(d, "identBackend")))
        if b == target and js(d, "return (window.app.arena.remotePacket||{}).identity&&window.app.arena.remotePacket.identity.backend") == target:
            ok = round(time.time() - t0, 1)
            break
    time.sleep(5)
    s = status()
    log("selectBackend->" + target, ok_after_s=ok, samples=samples, daemon_backend=s["identity"]["backend"],
        brain=s.get("brain_backend") or s.get("compute"), ident=text(d, "identityBar"), badge=text(d, "controllerBadge"),
        selected=d.find_element(By.ID, "selectBackend").get_attribute("value"), cmds=cmds(d), errs=errs(d),
        achieved=text(d, "statAchieved"), pill=text(d, "clusterStatusPill"))
    shot(d, f"backend-{target}")
finally:
    d.quit()
