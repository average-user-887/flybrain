// Headless containment harness for the browser fly physics in web/app.js.
//
//   gjs scripts/containment_harness.js [path/to/app.js] [seeds] [steps] [--stress]
//
// Loads app.js under gjs (SpiderMonkey) with a minimal DOM stub, then steps every
// paradigm for many seeds and checks, every step, that the fly:
//   1. stays inside the paradigm enclosure (in bounds),
//   2. never penetrates a wall segment by more than a tolerance,
//   3. never teleports (per-step displacement bounded by max speed * dt, except at
//      trial resets, which relocate the fly on purpose),
//   4. never stays frozen against a wall (must keep moving while in contact).
// Exit code is 1 when any check fails.

const GLib = imports.gi.GLib;

const appPath = ARGV[0] || 'web/app.js';
const nSeeds = parseInt(ARGV[1] || '12', 10);
const nSteps = parseInt(ARGV[2] || '3000', 10);
const stress = ARGV.includes('--stress');
const DT = 0.02;

// ---------------------------------------------------------------- DOM stub
function makeCtx() {
    return new Proxy({}, {
        get: (t, k) => {
            if (k === 'canvas') return null;
            if (k === 'measureText') return () => ({ width: 10 });
            if (k === 'createRadialGradient' || k === 'createLinearGradient') return () => ({ addColorStop: () => {} });
            if (k === 'getImageData') return () => ({ data: new Uint8ClampedArray(4) });
            return () => {};
        },
        set: () => true
    });
}
function makeEl(id) {
    return {
        id, style: {}, dataset: {}, textContent: '', innerHTML: '', value: '0', width: 300, height: 100,
        clientWidth: 300, clientHeight: 100,
        classList: { add() {}, remove() {}, toggle() {}, contains() { return false; } },
        getContext: () => makeCtx(),
        getBoundingClientRect: () => ({ left: 0, top: 0, width: 800, height: 600 }),
        addEventListener() {}, removeEventListener() {}, appendChild() {}, removeChild() {}, click() {},
        querySelector: () => null, querySelectorAll: () => []
    };
}
const elCache = {};
globalThis.document = {
    getElementById: (id) => (elCache[id] = elCache[id] || makeEl(id)),
    querySelector: () => null, querySelectorAll: () => [],
    createElement: (tag) => makeEl(tag), body: makeEl('body'),
    addEventListener() {}
};
// gjs aliases `window` to the global object, so the stubs go on globalThis itself.
const windowStub = {
    addEventListener() {}, removeEventListener() {}, devicePixelRatio: 1,
    location: { hostname: 'localhost', origin: 'http://localhost:8770', protocol: 'http:', search: '', href: 'http://localhost:8770/' },
    requestAnimationFrame() {}
};
if (typeof globalThis.window === 'undefined') {
    globalThis.window = windowStub;
} else {
    for (const [k, v] of Object.entries(windowStub)) {
        if (!(k in globalThis.window) || typeof globalThis.window[k] !== 'function') globalThis.window[k] = v;
    }
}
globalThis.performance = globalThis.performance || { now: () => Number(GLib.get_monotonic_time()) / 1000 };
globalThis.requestAnimationFrame = () => {};
globalThis.alert = () => {};
globalThis.fetch = () => Promise.reject(new Error('offline'));
globalThis.EventSource = class { close() {} };
globalThis.AbortSignal = { timeout: () => null };
globalThis.URLSearchParams = globalThis.URLSearchParams || class { get() { return null; } };

// ---------------------------------------------------------------- load app.js
const src = new TextDecoder().decode(GLib.file_get_contents(appPath)[1]);
new Function(src + '\nglobalThis.__NF = { ScientificBioArena, WallSegment, MushroomBodyCircuit, daemonFrameOffset };')();
const { ScientificBioArena, MushroomBodyCircuit, daemonFrameOffset } = globalThis.__NF;
if (String(daemonFrameOffset({paradigm:'multisensory-sandbox',world_bounds:[-75,-75,75,75]})) !== '0,0') throw new Error('Modern multisensory frame shifted outside arena');
if (String(daemonFrameOffset({paradigm:'open-arena',world_bounds:[0,0,100,100]})) !== '-50,-50') throw new Error('Foraging frame not centered');
// Reset must clear both long-term efficacy pathways, while keepMemory retains them.
const resetCircuit = new MushroomBodyCircuit();
resetCircuit.u[0] = [0.3, -0.4];
resetCircuit.w[0] = [0.2, -0.1];
resetCircuit.reset(true);
if (resetCircuit.u[0][1] !== -0.4) throw new Error('keepMemory discarded efficacy');
resetCircuit.reset(false);
if (resetCircuit.u[0].some(v => v !== 0) || resetCircuit.w[0].some(v => v !== 0)) {
    throw new Error('memory reset left a plasticity pathway uncleared');
}

// Regression: live telemetry has exclusive ownership of pose, clock and recording.
// Every local assay previously kept resetting the remote fly behind the stream.
for (const pid of ['open-arena','t-maze','y-maze','heat-maze','buridan','visual-operant',
    'wind-tunnel','looming-escape','optomotor','gap-crossing','circadian-dam','courtship','labyrinth','multisensory-sandbox']) {
    const live = new ScientificBioArena('arenaCanvas');
    live.initParadigm(pid);
    live.remoteDriven = true;
    const snapshot = () => JSON.stringify([live.fly.x,live.fly.y,live.fly.heading,live.currentTrial,
        live.stepCount,live.simTime,live.paradigmElapsedSec,live.telemetryBuffer]);
    const before = snapshot();
    for (let i=0;i<5000;i++) live.step(.02);
    if (snapshot() !== before) throw new Error(`${pid}: local simulation changed a live recording`);
    live.remoteDriven = false;
    live.awaitingDaemon = true;
    live.step(.02);
    if (snapshot() !== before) throw new Error(`${pid}: outage silently resumed synthetic data`);
}
print('PASS live ownership: all 14 assays preserve daemon pose, clock and samples');

// ---------------------------------------------------------------- seeded RNG
function seededRandom(seed) {
    let s = (seed >>> 0) || 1;
    return () => {
        s ^= s << 13; s >>>= 0; s ^= s >>> 17; s ^= s << 5; s >>>= 0;
        return (s >>> 0) / 4294967296;
    };
}

// ---------------------------------------------------------------- enclosures
// inside(x, y, r): true when a fly of radius r centred at (x, y) is inside the arena.
const TOL = 0.25; // mm of allowed geometric slack (numerical resolution offset)
const circ = (cx, cy, R) => (x, y, r) => Math.hypot(x - cx, y - cy) <= R - r + TOL;
const rect = (x0, y0, x1, y1) => (x, y, r) => x >= x0 + r - TOL && x <= x1 - r + TOL && y >= y0 + r - TOL && y <= y1 - r + TOL;
const ENCLOSURE = {
    'open-arena': rect(-140, -100, 140, 100),
    't-maze': (x, y, r) => rect(63, 10, 77, 57)(x, y, r) || rect(10, 43, 130, 57)(x, y, r)
        || (x >= 63 - TOL && x <= 77 + TOL && y >= 10 + r - TOL && y <= 57 - r + TOL),
    'y-maze': circ(60, 60, 41.5),
    'heat-maze': circ(60, 60, 55),
    'buridan': circ(60, 60, 47),
    'visual-operant': rect(0, 0, 80, 80),
    'wind-tunnel': rect(0, 0, 200, 60),
    'looming-escape': rect(0, 0, 80, 80),
    'optomotor': rect(0, 0, 90, 90),
    'gap-crossing': rect(5, 7, 95, 13),
    'circadian-dam': rect(5, 1, 60, 9),
    'courtship': circ(10, 10, 8.5),
    'labyrinth': rect(0, 0, 140, 100),
    'multisensory-sandbox': (x, y, r) => circ(0, 0, 75)(x, y, r)
        && [[35, 35], [-35, 35], [-35, -35], [35, -35]].every(pc => Math.hypot(x - pc[0], y - pc[1]) >= 6 + r - TOL - 0.5)
};
const TETHERED = new Set(['visual-operant', 'looming-escape', 'optomotor']);
const PARADIGMS = Object.keys(ENCLOSURE);

const MAX_SPEED = 42.0;            // escape speed in app.js
const TELEPORT_MM = MAX_SPEED * DT * 1.5 + 0.6;
const FROZEN_WINDOW = Math.round(2.0 / DT); // 2 s
const FROZEN_MM = 1.0;

function runOne(pid, seed) {
    Math.random = seededRandom(seed * 7919 + 17);
    const arena = new ScientificBioArena('arenaCanvas');
    arena.initParadigm(pid);
    const inside = ENCLOSURE[pid];
    const r = arena.fly.radius;
    const res = { pid, seed, oob: 0, penetr: 0, teleport: 0, frozen: 0, maxPen: 0, maxJump: 0, contactSteps: 0, maxFrozenSec: 0, first: null };
    let px = arena.fly.x, py = arena.fly.y;
    let lastTrial = arena.currentTrial;
    const hist = [];
    let frozenRun = 0;
    for (let i = 0; i < nSteps; i++) {
        if (stress && i % 250 === 100 && !TETHERED.has(pid)) {
            arena.fly.heading = Math.random() * 2 * Math.PI - Math.PI;
            arena.fly.speed = 32.0;
        }
        arena.step(DT);
        const { x, y } = arena.fly;
        if (!Number.isFinite(x) || !Number.isFinite(y)) {
            res.oob++; res.first = res.first || `step ${i}: non-finite position`; break;
        }
        // 1. enclosure
        if (!inside(x, y, r)) {
            res.oob++; res.first = res.first || `step ${i}: out of bounds at (${x.toFixed(2)}, ${y.toFixed(2)})`;
        }
        // 2. wall penetration
        let minD = Infinity;
        for (const w of arena.currentWalls || []) {
            const d = w.distanceToPoint(x, y);
            if (d < minD) minD = d;
        }
        if (minD < r + 0.6) res.contactSteps++;
        if (minD < r - TOL) {
            res.penetr++; res.maxPen = Math.max(res.maxPen, r - minD);
            res.first = res.first || `step ${i}: wall penetration ${(r - minD).toFixed(2)} mm at (${x.toFixed(2)}, ${y.toFixed(2)})`;
        }
        // 3. teleport (skip on trial reset)
        const reset = arena.currentTrial !== lastTrial || arena.paradigmElapsedSec < DT * 1.5;
        lastTrial = arena.currentTrial;
        const jump = Math.hypot(x - px, y - py);
        if (!reset) {
            res.maxJump = Math.max(res.maxJump, jump);
            if (jump > TELEPORT_MM) {
                res.teleport++;
                res.first = res.first || `step ${i}: teleport ${jump.toFixed(2)} mm from (${px.toFixed(2)}, ${py.toFixed(2)}) to (${x.toFixed(2)}, ${y.toFixed(2)})`;
            }
        }
        px = x; py = y;
        // 4. frozen against a wall
        if (!TETHERED.has(pid)) {
            hist.push({ x, y, contact: minD < r + 0.6, reset });
            if (hist.length > FROZEN_WINDOW) hist.shift();
            if (hist.length === FROZEN_WINDOW && !hist.some(h => h.reset)) {
                let maxDisp = 0;
                for (const h of hist) maxDisp = Math.max(maxDisp, Math.hypot(h.x - hist[0].x, h.y - hist[0].y));
                const allContact = hist.filter(h => h.contact).length > FROZEN_WINDOW * 0.8;
                if (maxDisp < FROZEN_MM && allContact) {
                    frozenRun++;
                    if (frozenRun === 1) {
                        res.frozen++;
                        res.first = res.first || `step ${i}: frozen against wall for ${(FROZEN_WINDOW * DT).toFixed(1)} s at (${x.toFixed(2)}, ${y.toFixed(2)})`;
                    }
                    res.maxFrozenSec = Math.max(res.maxFrozenSec, (FROZEN_WINDOW + frozenRun) * DT);
                } else {
                    frozenRun = 0;
                }
            }
        }
    }
    return res;
}

// Optional trace mode: --trace <pid> <seed> <fromStep> <toStep> prints the fly state per step.
const traceIdx = ARGV.indexOf('--trace');
if (traceIdx >= 0) {
    const [tp, ts, tf, tt] = ARGV.slice(traceIdx + 1, traceIdx + 5);
    Math.random = seededRandom(parseInt(ts, 10) * 7919 + 17);
    const arena = new ScientificBioArena('arenaCanvas');
    arena.initParadigm(tp);
    for (let i = 0; i <= parseInt(tt, 10); i++) {
        if (stress && i % 250 === 100 && !TETHERED.has(tp)) {
            arena.fly.heading = Math.random() * 2 * Math.PI - Math.PI;
            arena.fly.speed = 32.0;
        }
        arena.step(DT);
        if (i >= parseInt(tf, 10)) {
            const f = arena.fly;
            print(`${i} x=${f.x.toFixed(2)} y=${f.y.toFixed(2)} h=${f.heading.toFixed(2)} v=${f.speed.toFixed(2)} yaw=${f.yawRate.toFixed(2)} cmd=${(f.yawCommand||0).toFixed(2)} turnDir=${f.wallTurnDir} dna02=${arena.dn.dna02Diff.toFixed(1)} mdn=${arena.dn.mdn.toFixed(1)} contactT=${(arena.contactTime || 0).toFixed(2)} stuckT=${(arena.stuckEscapeTimer || 0).toFixed(2)} avoid=${(arena.lastWallAvoid || 0).toFixed(2)} status=${arena.paradigmStatus}`);
        }
    }
    imports.system.exit(0);
}

let anyFail = false;
const summary = [];
for (const pid of PARADIGMS) {
    const agg = { pid, runs: 0, oob: 0, penetr: 0, teleport: 0, frozen: 0, maxPen: 0, maxJump: 0, contactPct: 0, maxFrozenSec: 0, firsts: [] };
    for (let seed = 1; seed <= nSeeds; seed++) {
        let r;
        try {
            r = runOne(pid, seed);
        } catch (e) {
            r = { oob: 1, penetr: 0, teleport: 0, frozen: 0, maxPen: 0, maxJump: 0, contactSteps: 0, maxFrozenSec: 0, first: `exception: ${e.message}` };
        }
        agg.runs++;
        agg.oob += r.oob; agg.penetr += r.penetr; agg.teleport += r.teleport; agg.frozen += r.frozen;
        agg.maxPen = Math.max(agg.maxPen, r.maxPen); agg.maxJump = Math.max(agg.maxJump, r.maxJump);
        agg.contactPct += r.contactSteps / nSteps; agg.maxFrozenSec = Math.max(agg.maxFrozenSec, r.maxFrozenSec);
        if (r.first && agg.firsts.length < 2) agg.firsts.push(`seed ${seed} ${r.first}`);
    }
    agg.contactPct = (100 * agg.contactPct / agg.runs);
    const fail = agg.oob + agg.penetr + agg.teleport + agg.frozen > 0;
    anyFail = anyFail || fail;
    summary.push(agg);
    print(`${fail ? 'FAIL' : 'ok  '} ${pid.padEnd(22)} oob=${String(agg.oob).padStart(6)} penetr=${String(agg.penetr).padStart(6)} teleport=${String(agg.teleport).padStart(5)} frozen=${String(agg.frozen).padStart(3)}  maxPen=${agg.maxPen.toFixed(2)}mm maxJump=${agg.maxJump.toFixed(2)}mm contact=${agg.contactPct.toFixed(1)}% maxFrozen=${agg.maxFrozenSec.toFixed(1)}s`);
    for (const f of agg.firsts) print(`       ${f}`);
}
print(`\n${anyFail ? 'RESULT: FAIL' : 'RESULT: PASS'}  (${PARADIGMS.length} paradigms x ${nSeeds} seeds x ${nSteps} steps, dt=${DT}${stress ? ', stress impulses' : ''})`);
imports.system.exit(anyFail ? 1 : 0);
