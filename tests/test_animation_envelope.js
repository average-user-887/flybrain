'use strict';
// Independent check of the exported W3 animation GLBs against the W2 rig v2 joint
// contract (display-safe envelope), sampled over each clip's full duration and across
// the loop wrap.  Reads only the staged GLBs and W2's joints.json, never W3's manifest.
// Skips when the presentation assets are not staged (they are git-ignored).
const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');

const WEB = path.resolve(__dirname, '../web');
const ANIM = path.join(WEB, 'assets/hq/anim');
const read = (rel) => fs.readFileSync(path.join(WEB, rel), 'utf8');
const G = require(path.join(WEB, 'asset_gallery.js'));

const ctx = vm.createContext({TextDecoder, console, URL});
vm.runInContext('var self = globalThis;', ctx);
vm.runInContext(read('vendor/three.min.js'), ctx);
vm.runInContext(read('vendor/GLTFLoader.js'), ctx);
const THREE = ctx.THREE;
const parse = (file) => {
    const buf = fs.readFileSync(file);
    return new Promise((resolve, reject) => new THREE.GLTFLoader().parse(buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.length), '', resolve, reject));
};

const CLIPS = ['wing_open_fold', 'wingbeat_loop', 'antenna_sweep', 'groom_head_forelegs', 'groom_abdomen_wings_hindlegs', 'groom_full_cycle',
    'walk_tripod_loop', 'takeoff_prep', 'idle_antenna_twitch'];
// The two clips W3 designs to repeat (stated in the animation brief), named here so the
// check does not depend on W3's manifest flags.
const LOOP_CLIPS = ['wingbeat_loop', 'antenna_sweep', 'walk_tripod_loop', 'idle_antenna_twitch'];
const isTarsus = (m) => { for (let o = m; o; o = o.parent) if (/^(lf|lm|lh|rf|rm|rh)_tarsus$/.test(o.name || '')) return true; return false; };
const OFF_AXIS_TOL = 0.005;      // rad-equivalent quaternion component off the documented axis
const WRAP_TOL = 0.02;           // rad jump allowed between the end and the start of a looping clip
const POP_TOL = 1.0;             // rad between consecutive 30 fps keys: larger is a one-frame pop, not a fast stroke
const REST_TOL = 0.05;           // rad: a non-looping clip starts (and, unless it ends in a declared held pose, ends) at rest
// Clips that end in a held, non-rest pose by design; their return to rest is reported, not failed.
const HELD_END_CLIPS = ['takeoff_prep'];
const FLOOR_TOL = 1e-4;          // viewport-mm: a tarsus vertex may not sit below the floor by more
function angleBetween(a, b) { return 2 * Math.acos(Math.min(1, Math.abs(a.dot(b)))); }

// v3 = the repaired final clips; v2 = the failed export, kept with its label as evidence.
const SETS = {
    v3: {prefix: 'w3_v3_', label: 'W3 v3 (repaired, final)'},
    v2: {prefix: 'w3_v2_', label: 'W3 v2 (FAILED: feet through the floor, loop pops; kept as evidence)'}
};
function variants(set) {
    const out = [];
    // NF_ANIM_ONLY=female|male restricts an early check to the variants already exported.
    const sexes = process.env.NF_ANIM_ONLY ? [process.env.NF_ANIM_ONLY] : ['female', 'male'];
    for (const sex of sexes) for (const lod of [0, 1]) out.push({set, sex, lod,
        clips: path.join(ANIM, `${SETS[set].prefix}fly_${sex}_v2_lod${lod}_clips.glb`),
        rig: path.join(ANIM, `fly_${sex}_v2_lod${lod}.glb`),
        joints: path.join(ANIM, `fly_${sex}_v2_lod${lod}_joints.json`)});
    return out;
}
const stagedSet = (set) => variants(set).every((v) => fs.existsSync(v.clips) && fs.existsSync(v.joints) && fs.existsSync(v.rig));

// Measure every clip of one variant. The floor is taken from W2's own rig GLB, not from
// the clip file: the lowest tarsus vertex of the rest pose, which the gallery stands on
// the floor.  Every tarsus vertex is then scanned at 240 Hz plus every key time.
async function measureVariant(v) {
    const rig = await parse(v.rig);
    rig.scene.updateMatrixWorld(true);
    const floorY = G.vertexBounds(THREE, rig.scene, isTarsus).min.y;
    const gltf = await parse(v.clips);
    const contract = JSON.parse(fs.readFileSync(v.joints, 'utf8'));
    const axes = {}, ranges = {};
    for (const j of contract.joints) if (Array.isArray(j.axis)) { axes[j.node] = j.axis; ranges[j.node] = j.range_rad; }
    const root = gltf.scene;
    root.updateMatrixWorld(true);
    const restFoot = G.vertexBounds(THREE, root, isTarsus).min.y;
    const out = {names: gltf.animations.map((a) => a.name), floorY, restFoot, clips: {}};
    const restQ = (name) => { const o = rig.scene.getObjectByName(name); return o ? o.quaternion : null; };
    for (const clip of gltf.animations) {
        const mixer = new THREE.AnimationMixer(root);
        const action = mixer.clipAction(clip);
        action.play();
        const nodes = G.trackNodes(clip).map((n) => root.getObjectByName(n));
        const r = out.clips[clip.name] = {missingTargets: nodes.filter((o) => !o).length, undocumented: [], violations: [], offAxis: 0,
            lowestFoot: null, lowestAt: null, wrap: 0, pops: [], samples: 0};
        const live = nodes.filter(Boolean);
        const dofs = live.filter((o) => axes[o.name]);
        const legs = live.filter((o) => !axes[o.name]);
        r.undocumented = legs.map((o) => o.name).filter((n) => !/^(lf|lm|lh|rf|rm|rh)_(coxa|femur|tibia)_joint$/.test(n));
        const n = Math.max(240, Math.ceil(clip.duration * 240));
        const times = new Set();
        for (let k = 0; k <= n; k += 1) times.add(clip.duration * k / n);
        for (const tr of clip.tracks) for (const t of tr.times) times.add(t);
        const pose = (t) => { action.time = t; mixer.update(0); root.updateMatrixWorld(true); };
        let lowest = Infinity, lowestAt = null;
        for (const t of [...times].sort((a, b) => a - b)) {
            pose(t);
            r.samples += 1;
            const ang = Object.fromEntries(dofs.map((o) => [o.name, G.dofAngle(o.quaternion, axes[o.name])]));
            for (const x of G.envelopeViolations(ang, ranges)) r.violations.push(`t=${t.toFixed(3)} ${x}`);
            for (const o of dofs) {
                const a = axes[o.name], len = Math.hypot(...a);
                const along = (o.quaternion.x * a[0] + o.quaternion.y * a[1] + o.quaternion.z * a[2]) / len;
                r.offAxis = Math.max(r.offAxis, Math.hypot(o.quaternion.x - along * a[0] / len, o.quaternion.y - along * a[1] / len, o.quaternion.z - along * a[2] / len));
            }
            const y = G.vertexBounds(THREE, root, isTarsus).min.y;      // every tarsus vertex, claws included
            if (y < lowest) { lowest = y; lowestAt = t; }
        }
        r.lowestFoot = lowest; r.lowestAt = lowestAt;
        for (const tr of clip.tracks) {
            if (!/\.quaternion$/.test(tr.name)) continue;
            const v4 = tr.values;
            for (let i = 1; i < tr.times.length; i += 1) {
                const a = new THREE.Quaternion(v4[4 * i - 4], v4[4 * i - 3], v4[4 * i - 2], v4[4 * i - 1]);
                const b = new THREE.Quaternion(v4[4 * i], v4[4 * i + 1], v4[4 * i + 2], v4[4 * i + 3]);
                const jump = angleBetween(a, b);
                if (jump > POP_TOL) r.pops.push(`${tr.name.split('.')[0]} key ${i - 1}->${i} (t=${tr.times[i - 1].toFixed(3)} s) jumps ${jump.toFixed(3)} rad`);
            }
        }
        // Loop wrap, per rotation track: the jump from the last key back to the first.  A
        // seamless loop omits the duplicate end frame, so the wrap may be as large as an
        // ordinary step next to it; anything beyond 1.5x the neighbouring steps (and
        // WRAP_TOL) is a discontinuity.
        // Transitions to and from rest: first and last key against W2's rest pose.
        r.fromRest = 0; r.toRest = 0; r.fromRestNode = null; r.toRestNode = null;
        for (const tr of clip.tracks) {
            if (!/\.quaternion$/.test(tr.name)) continue;
            const node = tr.name.split('.')[0], rq = restQ(node);
            if (!rq) continue;
            const k = tr.times.length;
            const a0 = angleBetween(new THREE.Quaternion().fromArray(tr.values, 0), rq);
            const a1 = angleBetween(new THREE.Quaternion().fromArray(tr.values, 4 * (k - 1)), rq);
            if (a0 > r.fromRest) { r.fromRest = a0; r.fromRestNode = node; }
            if (a1 > r.toRest) { r.toRest = a1; r.toRestNode = node; }
        }
        r.wrap = 0; r.wrapExcess = [];
        for (const tr of clip.tracks) {
            if (!/\.quaternion$/.test(tr.name) || tr.times.length < 3) continue;
            const q = (i) => new THREE.Quaternion(tr.values[4 * i], tr.values[4 * i + 1], tr.values[4 * i + 2], tr.values[4 * i + 3]);
            const k = tr.times.length;
            const jump = angleBetween(q(k - 1), q(0));
            const allowed = Math.max(WRAP_TOL, 1.5 * Math.max(angleBetween(q(0), q(1)), angleBetween(q(k - 2), q(k - 1))));
            r.wrap = Math.max(r.wrap, jump);
            if (jump > allowed) r.wrapExcess.push(`${tr.name.split('.')[0]} wrap ${jump.toFixed(3)} rad > allowed ${allowed.toFixed(3)}`);
        }
        mixer.stopAllAction(); mixer.uncacheRoot(root);
    }
    return out;
}
const cache = {};
const measured = (v) => (cache[v.clips] = cache[v.clips] || measureVariant(v));
function problemsOf(m) {
    const p = [];
    for (const [name, r] of Object.entries(m.clips)) {
        if (r.lowestFoot < m.floorY - FLOOR_TOL) p.push(`${name}: a tarsus vertex goes ${(m.floorY - r.lowestFoot).toFixed(4)} below the floor at t=${r.lowestAt.toFixed(3)} s`);
        p.push(...r.pops.map((x) => name + ': ' + x));
        if (LOOP_CLIPS.includes(name)) p.push(...r.wrapExcess.map((x) => `${name}: loop ${x}`));
    }
    return p;
}
function report(v, m) {
    if (!process.env.NF_ANIM_REPORT) return;
    for (const [name, r] of Object.entries(m.clips)) console.log(`${v.set} ${v.sex} LOD${v.lod} ${name}: ${r.samples} samples, envelope violations ${r.violations.length}, off-axis ${r.offAxis.toFixed(5)}, wrap ${r.wrap.toFixed(3)}, pops ${r.pops.length}, lowest tarsus vertex ${r.lowestFoot.toFixed(4)} at t=${r.lowestAt.toFixed(3)} (floor ${m.floorY.toFixed(4)}), from rest ${r.fromRest.toFixed(3)}, to rest ${r.toRest.toFixed(3)}`);
}

// Repaired final clips: every check is a hard assertion.
for (const v of variants('v3')) {
    test(`W3 v3 clips on ${v.sex} LOD${v.lod}: nine clips; W2 envelope; on axis; every tarsus vertex on or above the floor; no pops; loops close; start/end at rest`,
        {skip: !stagedSet('v3') && 'v3 animation GLBs not staged under web/assets/hq/anim'}, async () => {
            const m = await measured(v);
            report(v, m);
            for (const c of (v.set === 'v2' ? CLIPS.slice(0, 6) : CLIPS)) assert.ok(m.names.includes(c), `${c} missing (have ${m.names})`);
            assert.ok(Math.abs(m.restFoot - m.floorY) < FLOOR_TOL, `clip file rest pose differs from the W2 rig floor (${m.restFoot} vs ${m.floorY})`);
            for (const [name, r] of Object.entries(m.clips)) {
                assert.equal(r.missingTargets, 0, `${name}: a track targets a node absent in this variant`);
                assert.deepEqual(r.undocumented, [], `${name}: undocumented joints`);
                assert.deepEqual(r.violations.slice(0, 5), [], `${name} leaves the W2 display-safe envelope`);
                assert.ok(r.offAxis < OFF_AXIS_TOL, `${name}: rotation leaves the documented DOF axis (${r.offAxis})`);
                assert.ok(r.lowestFoot >= m.floorY - FLOOR_TOL, `${name}: a tarsus vertex goes ${(m.floorY - r.lowestFoot).toFixed(4)} below the floor at t=${r.lowestAt.toFixed(3)} s`);
            }
            for (const [name, r] of Object.entries(m.clips)) {
                if (LOOP_CLIPS.includes(name)) continue;    // a loop need not contain the rest pose; its entry/exit is reported
                assert.ok(r.fromRest < REST_TOL, `${name}: starts ${r.fromRest.toFixed(3)} rad from rest (${r.fromRestNode})`);
                if (!HELD_END_CLIPS.includes(name)) assert.ok(r.toRest < REST_TOL, `${name}: ends ${r.toRest.toFixed(3)} rad from rest (${r.toRestNode})`);
            }
            assert.deepEqual(problemsOf(m), []);
        });
}

// The failed v2 export stays as evidence; the same check must still catch its defects.
for (const v of variants('v2')) {
    test(`W3 v2 (failed, kept) on ${v.sex} LOD${v.lod}: the check still catches its floor and pop defects`,
        {skip: !stagedSet('v2') && 'v2 animation GLBs not staged'}, async () => {
            const m = await measured(v);
            report(v, m);
            const p = problemsOf(m).join('\n');
            assert.match(p, /groom_head_forelegs: a tarsus vertex goes 0\.1[0-9]+ below the floor/);
            assert.match(p, /wingbeat_loop: .*(jumps|wrap) 2\.7/);
        });
}

test('a deliberately bad clip is caught (the check is not vacuous)', () => {
    const ranges = {l_wing_sweep: [0, 3], l_wing_elevate: [-1.7, 0.8], l_wing_pitch: [-0.27, 3.92]};
    // Stroke-reversal pitch on an open wing, and pitch on a folded wing.
    assert.ok(G.envelopeViolations({l_wing_sweep: 1.5, l_wing_elevate: 0.3, l_wing_pitch: 1.2}, ranges).some((x) => /open-wing safe pitch/.test(x)));
    assert.ok(G.envelopeViolations({l_wing_sweep: 0.3, l_wing_elevate: 0, l_wing_pitch: 0.06}, ranges).some((x) => /while folded/.test(x)));
});
