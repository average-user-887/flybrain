import sys
sys.path.insert(0, "<scratch>/scripts")
from lib import *  # noqa

d = driver()
try:
    ok = open_dash(d)
    print("connected", ok)
    import time; time.sleep(3)
    print("pill", text(d, "clusterStatusPill"), "|addr", text(d, "daemonAddress"), "|run", text(d, "arenaRunState"))
    print("ident", text(d, "identityBar"))
    print("errs", errs(d))
    print(shot(d, "smoke"))
finally:
    d.quit()
