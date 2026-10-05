"""Leak and checksum check of the bundle Firefox downloaded from the studio."""
import getpass, gzip, hashlib, json, re, socket, sys, zipfile
z = zipfile.ZipFile(sys.argv[1])
pat = re.compile(r'(?:/run/media|/home|/Users|/media|/mnt|/tmp)/|' + re.escape(getpass.getuser()) + '|' + re.escape(socket.gethostname()))
bundle = json.loads(z.read([n for n in z.namelist() if n.endswith('bundle.json')][0]))
out = {"members": len(z.namelist()), "leaks": {}, "checksums_ok": True}
for n in z.namelist():
    data = z.read(n)
    if n.endswith('.nfbody'):
        data = gzip.decompress(data)
    hits = pat.findall(data.decode('utf-8', 'replace')) if not n.endswith('.parquet') else []
    if n.endswith('.parquet'):
        hits = pat.findall(data.decode('latin-1'))
    if hits:
        out["leaks"][n] = hits[:3]
    short = n.split('/', 1)[1]
    if short in bundle["files"] and hashlib.sha256(z.read(n)).hexdigest() != bundle["files"][short]["sha256"]:
        out["checksums_ok"] = False
out["redaction"] = bundle.get("redaction")
print(json.dumps(out, indent=1))
