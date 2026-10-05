import hashlib
import pathlib

W = pathlib.Path('<redacted-path>/Documents/ChatGPT/flybrain/<redacted-path>/.wt-graded')
locked = W / 'docs/receipts/graded_transmission_v4_declaration.locked.md'
text = locked.read_text()
digest = hashlib.sha256(locked.read_bytes()).hexdigest()
assert digest == '2634c824476c9b799bd2c0c656040255360cd10f8837e2d874a97af5f939872e', digest

header = f"""

---

## 7. v4 — hybrid graded/spiking transmission (new)

**Written before the v4 engine was run on anything larger than a two-neuron
test graph, before any graded class was resolved against the real graph, and
before any v4 network measurement.** The graded membrane equation, the release
function and its derivation, the declared graded cell classes, the predicted
consequences, the falsification criteria and the stop rule before the
expensive protocol were all fixed and hashed first. The locked declaration is

    sha256 {digest}

written 2026-09-27T11:15:16+02:00, and is reproduced verbatim in
[`receipts/graded_transmission_v4_declaration.locked.md`](receipts/graded_transmission_v4_declaration.locked.md).
If this section and that file ever differ, **that file is the declaration.**

v1, v2 and v3 are **not** deleted, superseded or reinterpreted. All three stay
selectable, keep their pins, stay bit-reproducible, and every number published
under them stands as a result of that controller version. The default stays
v3.

"""
body = text.split('---\n', 2)[2] if text.count('---\n') >= 2 else text
# drop the locked file's own preamble: keep from the first "## 7.1"
i = text.index('## 7.1 Why a graded mode')
body = text[i:]
body = body.replace('--- END OF LOCKED DECLARATION ---\n', '')
spec = W / 'docs/LIF_DYNAMICS_SPEC.md'
s = spec.read_text()
# insert before the final "## 7. References" section, renaming it to 8
assert '\n## 7. References\n' in s, 'references heading not found'
s = s.replace('\n## 7. References\n', header + body + '\n---\n\n## 8. References\n', 1)
# update the title line and the pointer in the header block
s = s.replace('# Declared LIF dynamics for `brainlab` — v1 current-based, v2 conductance-based, v3 recalibrated',
              '# Declared LIF dynamics for `brainlab` — v1 current-based, v2 conductance-based, v3 recalibrated, v4 hybrid graded/spiking')
s = s.replace('**§6 adds v3** and was written under the same rule:',
              '**§7 adds v4**, the hybrid graded/spiking mode, under the same rule; see §7 and its\nlocked declaration.\n\n**§6 adds v3** and was written under the same rule:')
spec.write_text(s)
print('spec now', len(s.splitlines()), 'lines')
