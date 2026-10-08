'use strict';
// Independent check of the W3 environment-animal clips (jumping spider, mantis nymph),
// the wind-vane mapping and the composed scenes, read from the EXPORTED GLBs plus each
// rig's joints JSON.  W3's own verify output and clip metadata are compared against,
// never trusted.  Physical units: viewport-mm are converted to mm with the scale MEASURED
// here (body length in viewport-mm over the stated body length in mm).
// Skips when the presentation assets are not staged (web/assets/hq is git-ignored).
const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');

const WEB = path.resolve(__dirname, '../web');
const DIR = path.join(WEB, 'assets/hq/envanim');
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
const SET = process.env.NF_ENVANIM_SET || 'v3';
const ANIMALS = [
    {key: 'spider', file: 'env_jumping_spider', p: 'sp', bodyMeshes: /^spider_(body|eyes|chelicera_[LR])$/},
    {key: 'mantis', file: 'env_mantis_nymph', p: 'mn', bodyMeshes: /^mantis_(body|head)$/}
];
const files = (a) => ({
    clips: path.join(DIR, `${SET}_${a.file}_rig_clips.glb`), rig: path.join(DIR, `${SET}_${a.file}_rig.glb`),
    joints: path.join(DIR, `${SET}_${a.file}_rig_joints.json`), meta: path.join(DIR, `${SET}_${a.file}_rig_clips.json`)});
const staged = ANIMALS.every((a) => Object.values(files(a)).every((f) => fs.existsSync(f)));
if (process.env.NF_ENV_REQUIRE_ASSETS === '1') assert.ok(staged, `${SET}: required exported assets are missing`);
const SKIP = !staged && `env-animation ${SET} assets not staged under web/assets/hq/envanim`;

const DOF_TOL = 0.01;            // rad beyond a documented display range
const OFF_AXIS_TOL = 0.005;      // quaternion component off the documented axis
const FLOOR_TOL = 1e-3;          // viewport-mm a tarsus vertex may sit below the rest foot plane
const STANCE_BAND = 0.06;        // viewport-mm above the foot plane that still counts as stance
const DRIFT_BL = 0.02;           // max cumulative stance-foot drift per stance phase, as a fraction of body length
const WRAP_FACTOR = 1.5;         // a loop seam may be up to 1.5x the neighbouring key step
const SPEED_TOL = 0.10;          // relative agreement between in-place stance-foot speed and root-motion speed

const angleBetween = (a, b) => 2 * Math.acos(Math.min(1, Math.abs(a.dot(b))));
const isTarsus = (p) => (m) => { for (let o = m; o; o = o.parent) if (new RegExp(`^${p}_[LR]\\d_tarsus$`).test(o.name || '')) return true; return false; };

async function measureAnimal(a) {
    const f = files(a);
    const rig = await parse(f.rig);
    const gltf = await parse(f.clips);
    const joints = JSON.parse(fs.readFileSync(f.joints, 'utf8'));
    const meta = JSON.parse(fs.readFileSync(f.meta, 'utf8'));
    const axes = {}, ranges = {}, shiftLimits = {};
    for (const j of joints.joints) {
        if (Array.isArray(j.axis)) { axes[j.node] = j.axis; ranges[j.node] = j.range_rad; }
        if (j.kind === 'translation' && j.translation_axes) shiftLimits[j.node] = j.translation_axes;
    }
    const legs = Object.keys(joints.legs || {});
    // Floor and scale from the RIG GLB at rest (bind pose = every DOF 0).
    rig.scene.updateMatrixWorld(true);
    const floorY = G.vertexBounds(THREE, rig.scene, isTarsus(a.p)).min.y;
    const bodyBox = G.vertexBounds(THREE, rig.scene, (m) => { for (let o = m; o; o = o.parent) if (a.bodyMeshes.test(o.name || '')) return true; return false; });
    const bodyLenVp = bodyBox.max.z - bodyBox.min.z;
    const bodyLenMm = meta.parameters && meta.parameters.body_length_mm;
    const scale = bodyLenVp / bodyLenMm;      // viewport-mm per mm, measured
    const root = gltf.scene;
    root.updateMatrixWorld(true);
    const motion = root.getObjectByName(`${a.p}_motion`);
    // Foot tip per leg: the tarsus vertex lowest at rest (fixed vertex index).
    const tips = {};
    for (const leg of legs) {
        const mesh = root.getObjectByName(`${a.p}_${leg}_tarsus`);
        if (!mesh) continue;
        let best = null;
        mesh.traverse((m) => {
            const pos = m.isMesh && m.geometry && m.geometry.attributes.position;
            if (!pos) return;
            const v = new THREE.Vector3();
            for (let k = 0; k < pos.count; k += 1) {
                v.fromBufferAttribute(pos, k).applyMatrix4(m.matrixWorld);
                if (!best || v.y < best.y) best = {mesh: m, k, y: v.y};
            }
        });
        tips[leg] = best;
    }
    const tipWorld = (t) => new THREE.Vector3().fromBufferAttribute(t.mesh.geometry.attributes.position, t.k).applyMatrix4(t.mesh.matrixWorld);
    // Which side the L legs are on, measured from the rest tarsi (the agreed contract:
    // anatomical sides as authored, no loader reflection for the environment animals).
    const lx = legs.filter((l) => l.startsWith('L') && tips[l]).map((l) => tipWorld(tips[l]).x);
    const rx = legs.filter((l) => l.startsWith('R') && tips[l]).map((l) => tipWorld(tips[l]).x);
    const lSide = Math.sign(lx.reduce((x, y) => x + y, 0) / lx.length);
    const rSide = Math.sign(rx.reduce((x, y) => x + y, 0) / rx.length);
    const out = {lSide, rSide, floorY, bodyLenVp, bodyLenMm, scale, claimedScale: meta.viewport_mm_per_mm, meta, clips: {}, names: gltf.animations.map((c) => c.name)};
    for (const clip of gltf.animations) {
        const mixer = new THREE.AnimationMixer(root);
        const action = mixer.clipAction(clip);
        action.play();
        const nodes = G.trackNodes(clip).map((n) => root.getObjectByName(n));
        const r = out.clips[clip.name] = {missing: nodes.filter((o) => !o).length, dof: [], offAxis: 0, shift: [], lowest: Infinity,
            lowestAt: 0, stance: {}, maxDriftVp: 0, maxDriftLeg: null, wrapExcess: [], motionStill: true, samples: 0};
        r.invalidChannels = clip.tracks.filter((tr) => {
            const [node, property] = tr.name.split('.');
            if (property === 'quaternion') return !axes[node] && node !== `${a.p}_motion`;
            if (property === 'position') return !shiftLimits[node] && node !== `${a.p}_motion`;
            return true;
        }).map((tr) => tr.name);
        r.motionFloorError = 0;
        r.motionOffAxis = 0;
        const live = nodes.filter(Boolean);
        const times = new Set();
        const n = Math.max(240, Math.ceil(clip.duration * 240));
        for (let k = 0; k <= n; k += 1) times.add(clip.duration * k / n);
        for (const tr of clip.tracks) for (const t of tr.times) times.add(t);
        const sorted = [...times].sort((x, y) => x - y);
        const pose = (t) => { action.time = t; mixer.update(0); root.updateMatrixWorld(true); };
        const legState = Object.fromEntries(legs.filter((l) => tips[l]).map((l) => [l, {inStance: false, start: null, maxDrift: 0, phases: 0, speeds: []}]));
        const track = [];   // [t, motion position, motion yaw]
        let prevTips = null, prevT = null;
        for (const t of sorted) {
            pose(t);
            r.samples += 1;
            for (const o of live) {
                const ax = axes[o.name];
                if (ax) {
                    const th = G.dofAngle(o.quaternion, ax);
                    const rg = ranges[o.name];
                    if (rg && (th < rg[0] - DOF_TOL || th > rg[1] + DOF_TOL)) r.dof.push(`t=${t.toFixed(3)} ${o.name} ${th.toFixed(3)} outside [${rg}]`);
                    const len = Math.hypot(...ax), along = (o.quaternion.x * ax[0] + o.quaternion.y * ax[1] + o.quaternion.z * ax[2]) / len;
                    r.offAxis = Math.max(r.offAxis, Math.hypot(o.quaternion.x - along * ax[0] / len, o.quaternion.y - along * ax[1] / len, o.quaternion.z - along * ax[2] / len));
                }
                const lim = shiftLimits[o.name];
                if (lim) {
                    for (const [axis, [lo, hi]] of Object.entries(lim)) {
                        const v = o.position[axis];
                        if (v < lo - 1e-6 || v > hi + 1e-6) r.shift.push(`t=${t.toFixed(3)} ${o.name}.${axis} ${v.toFixed(3)} outside [${lo}, ${hi}]`);
                    }
                }
            }
            const low = G.vertexBounds(THREE, root, isTarsus(a.p)).min.y;
            if (low < r.lowest) { r.lowest = low; r.lowestAt = t; }
            const tipsNow = {};
            for (const [leg, st] of Object.entries(legState)) {
                const w = tipWorld(tips[leg]);
                tipsNow[leg] = w;
                const stance = w.y <= floorY + STANCE_BAND;
                if (stance && !st.inStance) { st.inStance = true; st.start = w.clone(); st.phases += 1; }
                if (stance && st.inStance) {
                    const d = Math.hypot(w.x - st.start.x, w.z - st.start.z);
                    st.maxDrift = Math.max(st.maxDrift, d);
                    if (prevTips && prevT !== null && t > prevT) st.speeds.push(Math.hypot(w.x - prevTips[leg].x, w.z - prevTips[leg].z) / (t - prevT));
                }
                if (!stance) st.inStance = false;
            }
            prevTips = tipsNow; prevT = t;
            if (motion) {
                r.motionFloorError = Math.max(r.motionFloorError, Math.abs(motion.position.y));
                r.motionOffAxis = Math.max(r.motionOffAxis, Math.hypot(motion.quaternion.x, motion.quaternion.z));
                const yaw = new THREE.Euler().setFromQuaternion(motion.quaternion, 'YXZ').y;
                track.push([t, motion.position.clone(), yaw]);
                if (motion.position.lengthSq() > 1e-12 || Math.abs(yaw) > 1e-9) r.motionStill = false;
            }
        }
        for (const [leg, st] of Object.entries(legState)) {
            r.stance[leg] = {phases: st.phases, maxDriftVp: st.maxDrift, medianSpeedVp: st.speeds.length ? st.speeds.sort((x, y) => x - y)[Math.floor(st.speeds.length / 2)] : 0};
            if (st.maxDrift > r.maxDriftVp) { r.maxDriftVp = st.maxDrift; r.maxDriftLeg = leg; }
        }
        // Root motion: speed, heading, turn rate (unwrapped yaw).
        if (track.length > 1) {
            let yawTotal = 0;
            for (let i = 1; i < track.length; i += 1) {
                let d = track[i][2] - track[i - 1][2];
                while (d > Math.PI) d -= 2 * Math.PI;
                while (d < -Math.PI) d += 2 * Math.PI;
                yawTotal += d;
            }
            const first = track[0], last = track[track.length - 1];
            const disp = last[1].clone().sub(first[1]);
            r.distanceVp = disp.length();
            r.speedVp = r.distanceVp / clip.duration;
            r.yawTotal = yawTotal;
            r.turnRateDeg = yawTotal / clip.duration * 180 / Math.PI;
            // Facing at the middle of the clip vs the direction actually travelled (+Z forward).
            const mid = track[Math.floor(track.length / 2)];
            const facing = new THREE.Vector3(Math.sin(mid[2]), 0, Math.cos(mid[2]));
            r.headingCos = r.distanceVp > 1e-6 ? facing.dot(disp.clone().setY(0).normalize()) : null;
            // + yaw about +Y turns +Z toward +X: toward the L legs when they sit at +X.
            r.turnsTowardL = Math.abs(yawTotal) < 1e-6 ? null : Math.sign(yawTotal) === lSide;
        }
        // Loop seam per rotation track (and per translation track except the root-motion carrier).
        if (clip.name && (r.meta = (meta.clips || {})[clip.name]) && r.meta.loop) {
            for (const tr of clip.tracks) {
                const node = tr.name.split('.')[0];
                if (node === `${a.p}_motion`) continue;           // accumulates by design; velocity seam below
                const k = tr.times.length;
                if (k < 3) continue;
                const isQ = /\.quaternion$/.test(tr.name);
                const dist = (i, j) => isQ
                    ? angleBetween(new THREE.Quaternion().fromArray(tr.values, 4 * i), new THREE.Quaternion().fromArray(tr.values, 4 * j))
                    : Math.hypot(tr.values[3 * i] - tr.values[3 * j], tr.values[3 * i + 1] - tr.values[3 * j + 1], tr.values[3 * i + 2] - tr.values[3 * j + 2]);
                const seam = dist(k - 1, 0), allowed = Math.max(isQ ? 0.02 : 0.02, WRAP_FACTOR * Math.max(dist(0, 1), dist(k - 2, k - 1)));
                if (seam > allowed) r.wrapExcess.push(`${tr.name} seam ${seam.toFixed(3)} > ${allowed.toFixed(3)}`);
            }
        }
        mixer.stopAllAction(); mixer.uncacheRoot(root);
        pose(0);
    }
    return out;
}

const cache = {};
const measured = (a) => (cache[a.key] = cache[a.key] || measureAnimal(a));
const mm = (vp, m) => vp / m.scale;

function report(a, m) {
    if (!process.env.NF_ENV_REPORT) return;
    console.log(`${SET} ${a.key}: L legs at ${m.lSide > 0 ? '+X' : '-X'}, R legs at ${m.rSide > 0 ? '+X' : '-X'}; body ${m.bodyLenVp.toFixed(2)} vp-mm / ${m.bodyLenMm} mm -> scale ${m.scale.toFixed(3)} vp-mm/mm (W3 claims ${m.claimedScale}); floor ${m.floorY.toFixed(4)}`);
    for (const [name, r] of Object.entries(m.clips)) {
        console.log(`  ${name}: ${r.samples} samples; DOF ${r.dof.length}, off-axis ${r.offAxis.toFixed(5)}, shift ${r.shift.length}; lowest tarsus ${(r.lowest - m.floorY).toFixed(4)} vs floor`
            + `; max stance drift ${mm(r.maxDriftVp, m).toFixed(4)} mm (${(100 * r.maxDriftVp / m.bodyLenVp).toFixed(2)}% BL, ${r.maxDriftLeg})`
            + (r.speedVp !== undefined ? `; root ${mm(r.speedVp, m).toFixed(2)} mm/s = ${(r.speedVp / m.bodyLenVp).toFixed(2)} BL/s, turn ${r.turnRateDeg.toFixed(0)} deg/s, heading cos ${r.headingCos === null ? '-' : r.headingCos.toFixed(3)}${r.turnsTowardL === null || r.turnsTowardL === undefined ? '' : (r.turnsTowardL ? ', turns toward the L legs' : ', turns toward the R legs')}` : '')
            + (r.wrapExcess.length ? `; SEAM ${r.wrapExcess.slice(0, 2).join(' | ')}` : ''));
    }
}

for (const a of ANIMALS) {
    test(`${SET} ${a.key}: measured body scale matches metadata; authored L=+X and R=-X`, {skip: SKIP}, async () => {
        const m = await measured(a);
        assert.ok(Number.isFinite(m.scale) && m.scale > 0 && Number.isFinite(m.claimedScale) && m.claimedScale > 0);
        assert.ok(Math.abs(m.scale - m.claimedScale) / m.scale <= 0.02,
            `measured scale ${m.scale.toFixed(3)} vs metadata ${m.claimedScale}`);
        assert.equal(m.lSide, 1, 'environment L legs must be at +X without a reflection');
        assert.equal(m.rSide, -1, 'environment R legs must be at -X without a reflection');
    });
    test(`${SET} ${a.key}: every clip inside the documented DOF and body-shift limits, on axis, no tarsus vertex below the foot plane`, {skip: SKIP}, async () => {
        const m = await measured(a);
        report(a, m);
        for (const [name, r] of Object.entries(m.clips)) {
            assert.equal(r.missing, 0, `${name}: a track targets a missing node`);
            assert.deepEqual(r.invalidChannels, [], `${name}: animated root, pivot, mesh or unsupported property`);
            assert.ok(r.motionFloorError < 1e-6, `${name}: locomotion carrier leaves the floor plane`);
            assert.ok(r.motionOffAxis < OFF_AXIS_TOL, `${name}: locomotion carrier rotates away from +Y`);
            assert.deepEqual(r.dof.slice(0, 3), [], `${name}: DOF outside its display limits`);
            assert.ok(r.offAxis < OFF_AXIS_TOL, `${name}: rotation off its documented axis (${r.offAxis})`);
            assert.deepEqual(r.shift.slice(0, 3), [], `${name}: body shift outside its limits`);
            assert.ok(r.lowest >= m.floorY - FLOOR_TOL, `${name}: a tarsus vertex goes ${(m.floorY - r.lowest).toFixed(4)} below the foot plane at t=${r.lowestAt.toFixed(3)}`);
        }
    });

    test(`${SET} ${a.key}: cumulative stance-foot drift per stance phase in mm, in root-motion clips`, {skip: SKIP}, async () => {
        const m = await measured(a);
        for (const [name, r] of Object.entries(m.clips)) {
            if (!/rootmotion/.test(name)) continue;
            const limitVp = DRIFT_BL * m.bodyLenVp;
            assert.ok(r.maxDriftVp <= limitVp, `${name}: ${r.maxDriftLeg} drifts ${mm(r.maxDriftVp, m).toFixed(3)} mm in one stance phase `
                + `(limit ${mm(limitVp, m).toFixed(3)} mm = ${100 * DRIFT_BL}% of ${m.bodyLenMm} mm body length)`);
            assert.ok(Object.values(r.stance).some((s) => s.phases > 0), `${name}: no stance phase detected`);
        }
    });

    test(`${SET} ${a.key}: in-place clips never move the root; root-motion speed, heading and turn rate agree with the gait`, {skip: SKIP}, async () => {
        const m = await measured(a);
        for (const [name, r] of Object.entries(m.clips)) if (/inplace/.test(name)) assert.ok(r.motionStill, `${name}: in-place clip moves ${a.p}_motion`);
        const walk = m.clips.walk_forward_rootmotion, walkIn = m.clips.walk_forward_inplace;
        assert.ok(walk && walkIn, 'walk_forward clips present');
        // Heading: the walk goes where the animal faces.
        assert.ok(walk.headingCos > 0.99, `walk_forward_rootmotion travels off its facing (cos ${walk.headingCos})`);
        // Visible speed vs body scale: the measured root speed matches W3's stated physical speed.
        const stated = m.meta.estimated_physical_speed_mm_s;
        const measuredMmS = mm(walk.speedVp, m);
        assert.ok(Math.abs(measuredMmS - stated) / stated < 0.15, `walk speed ${measuredMmS.toFixed(2)} mm/s vs stated ${stated} mm/s`);
        // Treadmill consistency: in-place stance feet move backward at the root-motion speed.
        const inSpeeds = Object.values(walkIn.stance).map((s) => s.medianSpeedVp).filter((v) => v > 0).sort((x, y) => x - y);
        const inMedian = inSpeeds[Math.floor(inSpeeds.length / 2)];
        assert.ok(Math.abs(inMedian - walk.speedVp) / walk.speedVp < SPEED_TOL, `in-place stance feet move ${mm(inMedian, m).toFixed(2)} mm/s, root motion ${measuredMmS.toFixed(2)} mm/s`);
        // Turns: opposite signs, rate matches W3's stated yaw rate.
        const tl = m.clips.turn_left_rootmotion, tr = m.clips.turn_right_rootmotion;
        assert.ok(tl.yawTotal * tr.yawTotal < 0, 'turn_left and turn_right turn the same way');
        // L/R contract: L legs and R legs on opposite sides; turn_left turns toward the L legs.
        assert.equal(m.lSide * m.rSide, -1, 'L and R legs on the same side');
        assert.equal(tl.turnsTowardL, true, 'turn_left turns toward the R-leg side');
        assert.equal(tr.turnsTowardL, false, 'turn_right turns toward the L-leg side');
        const statedRate = m.meta.turn && m.meta.turn.yaw_rate_rad_s;
        if (statedRate) assert.ok(Math.abs(Math.abs(tl.yawTotal / tl.meta.seconds) - statedRate) / statedRate < 0.15, `turn rate ${tl.turnRateDeg.toFixed(0)} deg/s vs stated ${(statedRate * 180 / Math.PI).toFixed(0)} deg/s`);
    });

    test(`${SET} ${a.key}: looping clips close at the seam (root-motion carrier excluded)`, {skip: SKIP}, async () => {
        const m = await measured(a);
        for (const [name, r] of Object.entries(m.clips)) assert.deepEqual(r.wrapExcess, [], `${name}: loop seam`);
    });
}

// Wind vane: the telemetry mapping from W3 (rotor yaw = atan2(wx, -wy)), with zero wind
// (field present, calm) kept distinct from missing wind (no field / not simulated).
test('wind vane mapping: direction from telemetry; calm and missing are different states', () => {
    const W = G.windVaneState;
    assert.equal(W(0, 1).state, 'directed');
    assert.ok(Math.abs(W(0, 1).yaw - Math.PI) < 1e-12, '(0, +1) -> pi (arrow to arena +y)');
    assert.ok(Math.abs(W(1, 0).yaw - Math.PI / 2) < 1e-12, '(+1, 0) -> pi/2');
    assert.equal(W(0, 0).state, 'calm');
    assert.equal(W(0, 0).yaw, 0);
    assert.match(W(0, 0).label, /0\.00 mm\/s/);
    for (const miss of [[null, null], [undefined, 1], [NaN, 0], [Infinity, 0]]) assert.equal(W(...miss).state, 'missing', String(miss));
    assert.match(W(null, null).label, /NOT SIMULATED|no wind field/i);
    assert.notEqual(W(0, 0).label, W(null, null).label);
});

test('scenes: within the 40k triangle budget, no textures, predator scene labelled ILLUSTRATIVE', {skip: !fs.existsSync(path.join(DIR, 'scene_predator_encounter_illustrative.glb')) && 'scenes not staged'}, async () => {
    for (const s of ['scene_fermenting_fruit_patch', 'scene_sugar_water_feeder', 'scene_predator_encounter_illustrative']) {
        const g = await parse(path.join(DIR, s + '.glb'));
        assert.ok(G.triangles(g.scene) <= 40000, `${s}: ${G.triangles(g.scene)} triangles`);
        let textures = 0;
        g.scene.traverse((o) => { for (const mat of [].concat(o.material || [])) if (mat && mat.map) textures += 1; });
        assert.equal(textures, 0, `${s}: textures`);
        assert.equal(g.animations.length, 0, `${s}: scenes are static`);
    }
    const meta = JSON.parse(fs.readFileSync(path.join(DIR, 'scene_predator_encounter_illustrative.json'), 'utf8'));
    assert.match(meta.status, /ILLUSTRATIVE/);
    assert.match(meta.status, /must not change looming or retinal input/);
});
