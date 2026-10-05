import json, sys
d = json.load(open(sys.argv[1] + "/live_ui_signoff.json"))
sc = d["scenarios"]
passed = [k for k, v in sc.items() if v.get("pass")]
print(sys.argv[1].split("/")[-1], "overall pass:", d.get("pass"), f"{len(passed)}/{len(sc)}",
      "failed:", [k for k, v in sc.items() if not v.get("pass")], "assay_failures:", d.get("assay_failures"))
