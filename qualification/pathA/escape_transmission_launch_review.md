# GF downstream diagnostic: separate root launch decision

1. **Decision requested:** independently review the clean candidate and issue GO or reject for exactly one 75-row CPU launch. This package authorizes no execution; A2 FAIL and the accepted original evidence remain unchanged.
2. **External pins:** derived contract `4a7f354c6d28291e2fb7f3458ff80decd799c3ba06fb8e316f2e701f925474ef`; prereg `0aa3b98144416fb26e67391e273e5628a053c7780ae473635f7f954dc607b9f2`; analyzer `90affa4083b060440ddf1752d82e0d97efd6d732be88bbf0223fb420d7409fe0`. Use the exact clean review-commit SHA as the separately supplied source pin.
3. **Read the prereg:** it pins four data files, original evidence, full reset/weight identities, every source dependency, Python 3.12.3 / NumPy 2.5.3 / Numba 0.67.0 and all 75 input/RNG records. No-input uses `[seed,0,0]`; irrelevant shuffled/timing fields are removed and the budget is one hour/3500 s.
4. **Verification:** 44 focused static tests cover synthetic positive evidence and negative source/document, integer-seed, row/order, retained-input, reset/weight, count/control, exit/argv/PID/log/time gates. Independently reconstructed default-v3 weights match the accepted digest. No model dynamics were run; these checks verify the adapter/audit, not circuit responses.
5. **Preflight:** root checks CPU/RAM and owner-service availability; use one process/thread with CUDA disabled. Put evidence outside the checkout in a new directory. Set the five `ESCAPE_*` variables below from externally reviewed pins and use an absolute Python executable reporting the frozen versions.
6. **After explicit root GO**, invoke the exclusive launcher; its stdout is the actual startup receipt. Record the observed wrapper and timeout-child PIDs independently. The timeout stays inside its child; logs/evidence are never resumed or overwritten.

```sh
CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMBA_NUM_THREADS=1 \
"$ESCAPE_PYTHON" -B scripts/pathA_escape_launch.py \
 --contract qualification/pathA/escape_transmission_contract.json --contract-sha256 "$ESCAPE_CONTRACT_SHA" \
 --prereg qualification/pathA/escape_transmission_prereg.json --prereg-sha256 "$ESCAPE_PREREG_SHA" \
 --expected-code-sha "$ESCAPE_SOURCE_SHA" --out "$ESCAPE_EVIDENCE_DIR"
```

7. **Audit after exit:** supply the independently observed PIDs, rather than learning them from the evidence being checked. Only integer exit0 and all 75 valid rows yield a descriptive report; any failed gate yields INVALID and no interpretation.

```sh
"$ESCAPE_PYTHON" -B scripts/pathA_escape_analyse.py \
 --contract qualification/pathA/escape_transmission_contract.json --contract-sha256 "$ESCAPE_CONTRACT_SHA" \
 --prereg qualification/pathA/escape_transmission_prereg.json --prereg-sha256 "$ESCAPE_PREREG_SHA" --expected-code-sha "$ESCAPE_SOURCE_SHA" \
 --expected-python "$ESCAPE_PYTHON" --observed-child-pid "$ESCAPE_CHILD_PID" --observed-wrapper-pid "$ESCAPE_WRAPPER_PID" \
 --runs "$ESCAPE_EVIDENCE_DIR" --graph-dir "$NEUROFLY_GRAPH_DIR" --connectome-dir "$NEUROFLY_CONNECTOME_DIR" --out "$ESCAPE_EVIDENCE_DIR/analysis.json"
```

8. **Evidence limit:** shared full reset/weight artifacts and recorded per-row canonical digests bind this run to the frozen state/weights; they are not saved full per-row state snapshots. Captured startup/exit receipts require root's actual observations. No biological relay validation, route fraction, necessity, mediation, motion inference or A2 regrade is permitted.
