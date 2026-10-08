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

const CLIPS = ['wing_open_fold', 'wingbeat_loop', 'antenna_sweep', 'groom_head_forelegs', 'groom_abdomen_wings_hindlegs', 'groom_full_cycle'];
const VARIANTS = [];
for (const sex of ['female', 'male']) for (const lod of [0, 1]) VARIANTS.push({sex, lod,
    clips: path.join(ANIM, `w3_v2_fly_${sex}_v2_lod${lod}_clips.glb`), joints: path.join(ANIM, `fly_${sex}_v2_lod${lod}_joints.json`)});
const staged = VARIANTS.every((v) => fs.existsSync(v.clips) && fs.existsSync(v.joints));
const isTarsus = (m) => { for (let o = m; o; o = o.parent) if (/^(lf|lm|lh|rf|rm|rh)_tarsus$/.test(o.name || '')) return true; return false; };
const OFF_AXIS_TOL = 0.005;      // rad-equivalent quaternion component off the documented axis
const WRAP_TOL = 0.02;           // rad jump allowed between the end and the start of a looping clip

function angleBetween(a, b) { return 2 * Math.acos(Math.min(1, Math.abs(a.dot(b)))); }

// The two clips W3 designs to repeat (stated in the animation brief), named here so the
// check does not depend on W3's manifest flags.
const LOOP_CLIPS = ['wingbeat_loop', 'antenna_sweep'];
const POP_TOL = 1.0;             // rad between consecutive 30 fps keys: larger is a one-frame pop, not a fast stroke

// Measure every clip of one variant; returns per-clip findings.
async function measureVariant(v) {
    const gltf = await parse(v.clips);
    const contract = JSON.parse(fs.readFileSync(v.joints, 'utf8'));
    const axes = {}, ranges = {};
    for (const j of contract.joints) if (Array.isArray(j.axis)) { axes[j.node] = j.axis; ranges[j.node] = j.range_rad; }
    const root = gltf.scene;
    root.updateMatrixWorld(true);
    const bindFoot = G.vertexBounds(THREE, root, isTarsus).min.y;
    const out = {names: gltf.animations.map((a) => a.name), bindFoot, clips: {}};
    for (const clip of gltf.animations) {
        const mixer = new THREE.AnimationMixer(root);
        const action = mixer.clipAction(clip);
        action.play();
        const nodes = G.trackNodes(clip).map((n) => root.getObjectByName(n));
        const r = out.clips[clip.name] = {missingTargets: nodes.filter((o) => !o).length, undocumented: [], violations: [], offAxis: 0,
            lowestFoot: null, wrap: 0, pops: []};
        const live = nodes.filter(Boolean);
        const dofs = live.filter((o) => axes[o.name]);
        const legs = live.filter((o) => !axes[o.name]);
        r.undocumented = legs.map((o) => o.name).filter((n) => !/^(lf|lm|lh|rf|rm|rh)_(coxa|femur|tibia)_joint$/.test(n));
        const n = Math.max(240, Math.ceil(clip.duration * 240));
        const pose = (t) => { action.time = t; mixer.update(0); root.updateMatrixWorld(true); };
        let lowest = Infinity;
        for (let k = 0; k <= n; k += 1) {
            const t = clip.duration * k / n;
            pose(t);
            const ang = Object.fromEntries(dofs.map((o) => [o.name, G.dofAngle(o.quaternion, axes[o.name])]));
            for (const x of G.envelopeViolations(ang, ranges)) r.violations.push(`t=${t.toFixed(3)} ${x}`);
            for (const o of dofs) {
                const a = axes[o.name], len = Math.hypot(...a);
                const along = (o.quaternion.x * a[0] + o.quaternion.y * a[1] + o.quaternion.z * a[2]) / len;
                r.offAxis = Math.max(r.offAxis, Math.hypot(o.quaternion.x - along * a[0] / len, o.quaternion.y - along * a[1] / len, o.quaternion.z - along * a[2] / len));
            }
            if (legs.length) lowest = Math.min(lowest, G.vertexBounds(THREE, root, isTarsus).min.y);
        }
        r.lowestFoot = legs.length ? lowest : null;
        // Key-to-key continuity straight from the exported tracks (one-frame pops).
        for (const tr of clip.tracks) {
            if (!/\.quaternion$/.test(tr.name)) continue;
            const v4 = tr.values;
            for (let i = 1; i < tr.times.length; i += 1) {
                const a = new THREE.Quaternion(v4[4 * i - 4], v4[4 * i - 3], v4[4 * i - 2], v4[4 * i - 1]);
                const b = new THREE.Quaternion(v4[4 * i], v4[4 * i + 1], v4[4 * i + 2], v4[4 * i + 3]);
                const jump = angleBetween(a, b);
                if (jump > POP_TOL) r.pops.push(`${tr.name.split('.')[0]} key ${i - 1}->${i} (t=${tr.times[i - 1].toFixed(3)}->${tr.times[i].toFixed(3)} s) jumps ${jump.toFixed(3)} rad`);
            }
        }
        // Loop wrap: the end pose against the start pose.
        pose(0); const first = live.map((o) => o.quaternion.clone());
        pose(clip.duration); const last = live.map((o) => o.quaternion.clone());
        r.wrap = Math.max(0, ...live.map((o, i) => angleBetween(first[i], last[i])));
        mixer.stopAllAction(); mixer.uncacheRoot(root);
    }
    return out;
}

const results = {};
const measured = (v) => (results[v.clips] = results[v.clips] || measureVariant(v));

for (const v of VARIANTS) {
    const skip = !staged && 'animation GLBs not staged under web/assets/hq/anim';
    test(`W3 clips on ${v.sex} LOD${v.lod}: six clips, inside the W2 display-safe envelope, every rotation on its documented axis`, {skip}, async () => {
        const m = await measured(v);
        for (const c of CLIPS) assert.ok(m.names.includes(c), `${c} missing (have ${m.names})`);
        for (const [name, r] of Object.entries(m.clips)) {
            assert.equal(r.missingTargets, 0, `${name}: a track targets a node absent in this variant`);
            assert.deepEqual(r.undocumented, [], `${name}: undocumented joints`);
            assert.deepEqual(r.violations.slice(0, 5), [], `${name} leaves the W2 display-safe envelope`);
            assert.ok(r.offAxis < OFF_AXIS_TOL, `${name}: rotation leaves the documented DOF axis (${r.offAxis})`);
        }
        if (process.env.NF_ANIM_REPORT) for (const [name, r] of Object.entries(m.clips)) console.log(`${v.sex} LOD${v.lod} ${name}: violations ${r.violations.length}, off-axis ${r.offAxis.toFixed(5)}, wrap ${r.wrap.toFixed(3)}, pops ${r.pops.length}${r.lowestFoot !== null ? ', lowest foot ' + r.lowestFoot.toFixed(4) + ' (rest ' + m.bindFoot.toFixed(4) + ')' : ''}${r.pops.length ? '\n    ' + r.pops.slice(0, 4).join('\n    ') : ''}`);
    });
}

// Feet and continuity are checked separately and recorded as known defects while they
// fail (found by this independent check on 2026-10-08, reported to the clip author;
// the gallery shows them and never patches the keys):
//  * groom_head_forelegs / groom_full_cycle: at frame ~203 (t = 6.767 s) both fore
//    tarsi sit 0.18 viewport-mm below the rest foot plane, i.e. through the floor;
//  * the first key of antenna_sweep, groom_head_forelegs and wingbeat_loop holds
//    another clip's pose (one-frame pop), so the looping clips do not close.
const KNOWN_CONTINUITY_DEFECT = 'W3 v2 export: fore tarsi below the foot plane at t=6.767 s in the grooming clips; first key of several clips holds another clip\'s pose (reported 2026-10-08, fix pending)';
for (const v of VARIANTS) {
    test(`W3 clips on ${v.sex} LOD${v.lod}: feet stay on or above the rest foot plane, no one-frame pops, looping clips close at the wrap`,
        {skip: !staged && 'animation GLBs not staged', todo: staged ? KNOWN_CONTINUITY_DEFECT : false}, async () => {
            const m = await measured(v);
            const problems = [];
            for (const [name, r] of Object.entries(m.clips)) {
                if (r.lowestFoot !== null && r.lowestFoot < m.bindFoot - 1e-3) problems.push(`${name}: a foot goes ${(m.bindFoot - r.lowestFoot).toFixed(4)} below the rest foot plane`);
                problems.push(...r.pops.map((p) => name + ': ' + p));
                if (LOOP_CLIPS.includes(name) && r.wrap > WRAP_TOL) problems.push(`${name}: loop wrap jumps ${r.wrap.toFixed(3)} rad`);
            }
            assert.deepEqual(problems, []);
        });
}

test('a deliberately bad clip is caught (the check is not vacuous)', () => {
    const ranges = {l_wing_sweep: [0, 3], l_wing_elevate: [-1.7, 0.8], l_wing_pitch: [-0.27, 3.92]};
    // Stroke-reversal pitch on an open wing, and pitch on a folded wing.
    assert.ok(G.envelopeViolations({l_wing_sweep: 1.5, l_wing_elevate: 0.3, l_wing_pitch: 1.2}, ranges).some((x) => /open-wing safe pitch/.test(x)));
    assert.ok(G.envelopeViolations({l_wing_sweep: 0.3, l_wing_elevate: 0, l_wing_pitch: 0.06}, ranges).some((x) => /while folded/.test(x)));
});
