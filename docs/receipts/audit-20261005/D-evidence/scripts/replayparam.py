import sys, time
sys.path.insert(0, "<scratch>/scripts")
from lib import *  # noqa
log = Log("replayparam")
d = driver()
try:
    d.get(f"{WEB}/index.html?daemon={DAEMON}&replay={DAEMON}/api/recordings/auditd-modular.nfrec"); hook(d); time.sleep(8)
    log("?replay=URL", bar_hidden=js(d, "return document.getElementById('replayBar').hidden"), info=text(d, "replayInfo"),
        time=text(d, "replayTime"), errs=errs(d))
finally:
    d.quit()
