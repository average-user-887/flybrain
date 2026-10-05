import sys, time
sys.path.insert(0, "<scratch>/scripts")
from lib import *  # noqa

log = Log("backend")
r = post("switch_backend", backend="connectome-with-trained-readout")
log("api:switch_backend(trained-readout)", reply=str(r)[:300])
wait(lambda: status()["identity"]["backend"] == "connectome-with-trained-readout", 120, 1)
d = driver()
try:
    open_dash(d); time.sleep(4)
    sel = d.find_element(By.ID, "selectBackend")
    log("selectBackend-shows(trained-readout)", daemon=status()["identity"]["backend"], value=sel.get_attribute("value"),
        selectedIndex=js(d, "return document.getElementById('selectBackend').selectedIndex"),
        visible_text=js(d, "const s=document.getElementById('selectBackend'); return s.selectedIndex>=0?s.options[s.selectedIndex].text:'(blank)'"),
        ident=text(d, "identBackend"))
    shot(d, "backend-trained-readout")
    # record_start is refused for a backend switch while recording? (no UI) -> API check below
    Select(sel).select_by_value("modular"); js(d, "document.getElementById('selectBackend').blur()")
    ok = wait(lambda: status()["identity"]["backend"] == "modular", 120, 1)
    time.sleep(3)
    log("selectBackend->modular", ok=bool(ok), ident=text(d, "identBackend"), value=sel.get_attribute("value"), errs=errs(d))
finally:
    d.quit()
