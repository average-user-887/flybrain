import sys, time
sys.path.insert(0, "<scratch>/scripts")
from lib import *  # noqa

log = Log("studio-cmp")
d = driver()
try:
    d.get("http://127.0.0.1:8842/"); hook(d); time.sleep(3)
    d.find_element(By.CSS_SELECTOR, "nav [data-tab='compare']").click(); time.sleep(1)
    opts = js(d, "return [...document.getElementById('cmp-a').options].map(o=>o.value).filter(Boolean)")
    sil = [o for o in opts if "dna02-silenced-s21" in o and not o.endswith("control")][0]
    ctl = [o for o in opts if "dna02-silenced-s21-control" in o][0]
    js(d, "for (const [id,v] of [['cmp-a',arguments[0]],['cmp-b',arguments[1]]]) {const s=document.getElementById(id); s.value=v; s.dispatchEvent(new Event('change',{bubbles:true}));}", sil, ctl)
    time.sleep(5)
    log("cmp-a/cmp-b(manual: dna02 silenced vs intact)", options=len(opts), table=text(d, "cmp-table"),
        sync_hidden=js(d, "return document.getElementById('sync').hidden"), errs=errs(d))
    shot(d, "studio-compare-silenced")
    # one side only
    js(d, "const s=document.getElementById('cmp-b'); s.value=''; s.dispatchEvent(new Event('change',{bubbles:true}));"); time.sleep(3)
    log("cmp-b cleared", table=text(d, "cmp-table")[:300], sync_hidden=js(d, "return document.getElementById('sync').hidden"))
    # values survive the 5 s auto-refresh?
    js(d, "const s=document.getElementById('cmp-b'); s.value=arguments[0]; s.dispatchEvent(new Event('change',{bubbles:true}));", ctl); time.sleep(11)
    log("cmp selection survives auto-refresh", a=d.find_element(By.ID, "cmp-a").get_attribute("value"), b=d.find_element(By.ID, "cmp-b").get_attribute("value"))
finally:
    d.quit()
