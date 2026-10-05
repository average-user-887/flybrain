import sys, time, hashlib
sys.path.insert(0, "<scratch>/scripts")
from lib import *  # noqa

log = Log("embodied")
d = driver()
NF = "<scratch>/studio/curated/auditd-intro-control/body.nfbody"
try:
    d.get("http://127.0.0.1:8842/embodied_replay.html"); hook(d); time.sleep(2)
    log("embodied_replay(no src)", play_disabled=not d.find_element(By.ID, "play").is_enabled(), body=d.find_element(By.TAG_NAME, "body").text[:200].replace("\n", " | "))
    d.find_element(By.ID, "file").send_keys(NF); time.sleep(3)
    log("file input .nfbody", play_disabled=not d.find_element(By.ID, "play").is_enabled(),
        seek_max=d.find_element(By.ID, "seek").get_attribute("max"), status=d.find_element(By.TAG_NAME, "body").text[:260].replace("\n", " | "), errs=errs(d))
    canvas = d.find_element(By.TAG_NAME, "canvas")
    h0 = hashlib.sha1(js(d, "return document.querySelector('canvas').toDataURL()").encode()).hexdigest()[:8]
    d.find_element(By.TAG_NAME, "body").send_keys(" "); time.sleep(1.5)
    f1 = d.find_element(By.ID, "seek").get_attribute("value")
    h1 = hashlib.sha1(js(d, "return document.querySelector('canvas').toDataURL()").encode()).hexdigest()[:8]
    log("Space key toggles play", seek_after=f1, play_label=d.find_element(By.ID, "play").text, canvas_changed=h0 != h1)
    # follow checkbox: camera follows?
    f = d.find_element(By.ID, "follow"); f.click()
    log("follow checkbox", checked=f.is_selected(), errs=errs(d))
    shot(d, "embodied-file")
finally:
    d.quit()
