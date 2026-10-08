'use strict';
// Dashboard 3D viewport presentation layer: geometric floor placement (never a pose
// clamp) and the frame-time quality fallback.
const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');

const WEB = path.resolve(__dirname, '../web');
const read = (rel) => fs.readFileSync(path.join(WEB, rel), 'utf8');
const app = read('app.js');

function declaration(prefix) {
    const start = app.indexOf(prefix);
    assert.notEqual(start, -1, `${prefix} missing`);
    const body = app.indexOf('{', start);
    let depth = 0;
    for (let i = body; i < app.length; i += 1) {
        if (app[i] === '{') depth += 1;
        if (app[i] === '}' && --depth === 0) return app.slice(start, i + 1);
    }
    throw new Error(`${prefix} is incomplete`);
}

const statusNode = () => ({style: {}, textContent: ''});
const ctx = vm.createContext({
    TextDecoder, console, URL, performance,
    window: {location: {search: ''}},
    document: {getElementById: () => null, createElement: () => ({style: {}, getContext: () => null})}
});
vm.runInContext('var self = globalThis;', ctx);
vm.runInContext(read('vendor/three.min.js'), ctx);
vm.runInContext([
    declaration('function orbitFrameValues('),
    declaration('const DAEMON_FRAME_OFFSET ='),
    declaration('function daemonFrameOffset('),
    declaration('function arenaPointFor3D('),
    declaration('function assayGeometry3DDescriptor('),
    declaration('function disposeThreeTree('),
    declaration('class ArticulatedFly3DViewport '),
    'this.Viewport = ArticulatedFly3DViewport;'
].join('\n'), ctx);
const {THREE, Viewport} = ctx;
const FLOOR = Viewport.floorTop();

function viewport(packet) {
    const vp = Object.create(Viewport.prototype);
    vp.scene = new THREE.Scene();
    vp.buildFlyMesh();
    vp.standOffset = Viewport.standOffsetFor(Viewport.proceduralLowestLegY());
    vp.initialized = true;
    vp.cameraMode = 'orbit';
    vp.poseStatus = statusNode();
    vp.arena = {
        activeParadigmId: 'open-arena', awaitingDaemon: false, width: 100, height: 100,
        worldBounds: {minX: 0, maxX: 100, minY: 0, maxY: 100},
        fly: {x: 40, y: 60, heading: 0.7},
        remotePacket: Object.assign({type: 'telemetry', paradigm: 'open-arena', world_bounds: [0, 0, 100, 100],
            fly: {heading: 0.7}, identity: {run_id: 'r', instance_id: 'i', activation: 1}, segment_id: 's',
            scene: {geometry: {schema: 'neurofly.assay-geometry.v1', coordinate_frame: 'arena-mm',
                source: 'python-arena', bounds: [0, 0, 100, 100],
                containment: {kind: 'rectangle', bounds: [0, 0, 100, 100]}, walls: [],
                surfaces: [{shape: 'rectangle', bounds: [0, 0, 100, 100]}]}}}, packet)
    };
    return vp;
}

// Lowest point (exact vertex scan) of all visible leg geometry, in world y.
function lowestLegWorldY(vp) {
    vp.scene.updateMatrixWorld(true);
    let lowest = Infinity;
    const v = new THREE.Vector3();
    for (const leg of vp.legs) {
        leg.coxa.traverse((o) => {
            const a = o.isMesh && o.geometry.attributes.position;
            if (a) for (let k = 0; k < a.count; k += 1) lowest = Math.min(lowest, v.fromBufferAttribute(a, k).applyMatrix4(o.matrixWorld).y);
        });
    }
    return lowest;
}

function lcg(seed) {
    let s = seed >>> 0;
    return () => ((s = (Math.imul(s, 1664525) + 1013904223) >>> 0) / 4294967296);
}

test('the stand offset comes from the rig geometry: the exact scan matches the analytic rest pose', () => {
    const vp = viewport({});
    const scanned = Viewport.lowestLegPointY(THREE, vp.flyGroup, vp.legs);
    assert.ok(Math.abs(scanned - Viewport.proceduralLowestLegY()) < 1e-6, `scan ${scanned}`);
    assert.ok(Math.abs(Viewport.proceduralLowestLegY() + 3.85) < 1e-12);
    assert.ok(Math.abs(Viewport.standOffsetFor(scanned) - 3.33) < 1e-6);
    // The scan restores every transform it touched.
    vp.flyGroup.position.set(5, 6, 7);
    vp.flyGroup.rotation.set(0, 1.1, 0);
    vp.legs[2].femur.rotation.z = 0.4;
    const before = [vp.flyGroup.position.toArray(), vp.flyGroup.quaternion.toArray(), vp.legs[2].femur.rotation.z];
    Viewport.lowestLegPointY(THREE, vp.flyGroup, vp.legs);
    assert.deepEqual([vp.flyGroup.position.toArray(), vp.flyGroup.quaternion.toArray(), vp.legs[2].femur.rotation.z], before);
});

test('at the illustrative lift no joint-angle stream can draw a foot below the floor top', () => {
    // Before the fix the root sat at z + 1.2 and the rest-pose feet at y = -2.15.
    const vp = viewport({});
    vp.updatePose();
    assert.ok(Math.abs(lowestLegWorldY(vp) - FLOOR) < 1e-6, `rest pose touches the floor: ${lowestLegWorldY(vp)}`);
    const rnd = lcg(7);
    for (let trial = 0; trial < 400; trial += 1) {
        const angles = Array.from({length: 18}, () => (rnd() * 2 - 1) * Math.PI);
        const contacts = Array.from({length: 6}, () => rnd() > 0.5);
        const p = viewport({joint_angles_rad: angles, leg_contacts: contacts, body_position_mm: [40, 60, 0.5]});
        p.updatePose();
        assert.ok(lowestLegWorldY(p) >= FLOOR - 1e-6, `trial ${trial}: ${lowestLegWorldY(p)}`);
    }
});

test('streamed body height and joint angles are drawn unchanged: no clamp, no per-frame lift', () => {
    // femur -0.7 and tibia 0.4 map to straight legs (femur 0.35 - 0.35 = 0, tibia
    // 0.59 + 0.5 * -0.7 - 0.6 * 0.4 = 0), the lowest pose the mapping can draw.
    const angles = Array.from({length: 18}, (_, i) => [0.14, -0.7, 0.4][i % 3]);
    for (const z of [1.11, 0.5, 0.2]) {
        const vp = viewport({joint_angles_rad: angles, leg_contacts: [true, false, true, false, true, false],
            body_position_mm: [40, 60, z]});
        vp.updatePose();
        assert.ok(Math.abs(vp.flyGroup.position.y - (z + 3.33)) < 1e-9, `root follows the stream at z=${z}`);
        assert.ok(Math.abs(vp.legs[0].femur.rotation.z) < 1e-12);
        assert.ok(Math.abs(lowestLegWorldY(vp) - (FLOOR + z - 0.5)) < 1e-6, `feet at ${lowestLegWorldY(vp)}`);
        // A recorded body height below the reference lift is shown as recorded (feet below).
        if (z < 0.5) assert.ok(lowestLegWorldY(vp) < FLOOR);
    }
});

test('the stand is re-derived from attached geometry and display scale (male 0.8696)', () => {
    const vp = viewport({});
    // A longer visible foot mesh (as an HQ tarsus with claws would be) lowers the scan.
    const claw = new THREE.Mesh(new THREE.BoxGeometry(0.1, 0.4, 0.1), new THREE.MeshBasicMaterial());
    claw.position.set(0, -2.6, 0);
    vp.legs[0].tibia.add(claw);
    vp.refreshStand();
    assert.ok(Math.abs(vp.standLowestY + 4.0) < 1e-6, `${vp.standLowestY}`);
    vp.flyGroup.scale.setScalar(0.8696);
    vp.refreshStand();
    assert.ok(Math.abs(vp.standLowestY + 4.0 * 0.8696) < 1e-6);
    vp.updatePose();
    assert.ok(Math.abs(lowestLegWorldY(vp) - FLOOR) < 1e-6);
    // Hidden geometry (procedural materials hidden by the HQ loader) does not count.
    claw.material.visible = false;
    vp.flyGroup.scale.setScalar(1);
    vp.refreshStand();
    assert.ok(Math.abs(vp.standLowestY + 3.85) < 1e-6);
});

// ------------------------------------------------------------------ quality fallback
function run(governor, frames, interval, renderMs = 3) {
    const changes = [];
    for (let i = 0; i < frames; i += 1) {
        const c = governor.sample(typeof interval === 'function' ? interval(i) : interval, renderMs);
        if (c) changes.push(c);
    }
    return changes;
}

test('a frame time within budget keeps the starting quality', () => {
    const g = Viewport.createQualityGovernor({startLevel: 0});
    assert.deepEqual(run(g, 3000, 16.7), []);
    assert.equal(g.level, 0);
    assert.ok(Math.abs(g.frameMs - 16.7) < 1e-6);
    assert.equal(g.budgetMs, 25);
});

test('sustained over-budget frames step the quality down one level at a time, then stop at Low', () => {
    const g = Viewport.createQualityGovernor({startLevel: 0});
    const changes = run(g, 2000, 45);
    assert.deepEqual(changes.map((c) => c.level), [1, 2]);
    assert.equal(g.level, 2);
    assert.equal(JSON.stringify(g.failedLevels.sort()), '[0,1]');
    // A failed level is never re-entered automatically: no oscillation.
    assert.deepEqual(run(g, 5000, 5), []);
    assert.equal(g.level, 2);
});

test('the fallback waits for a sustained trend: short spikes and stalls do not drop quality', () => {
    const g = Viewport.createQualityGovernor({startLevel: 0});
    // A 10-frame spike every 200 frames.
    assert.deepEqual(run(g, 4000, (i) => (i % 200 < 10 ? 60 : 16.7)), []);
    // Hidden tab / paused debugger: intervals over 250 ms are not samples.
    assert.deepEqual(run(g, 500, 4000), []);
    assert.equal(g.level, 0);
});

test('the budget is 40 frames per second; a display capped at 30 Hz is treated as over budget', () => {
    const g = Viewport.createQualityGovernor({startLevel: 0});
    assert.equal(g.budgetMs, 25);
    assert.deepEqual(run(g, 3000, 24), []);
    assert.equal(run(g, 3000, 33.3).length >= 1, true);
});

test('recovery is allowed only into a level that has not failed', () => {
    const g = Viewport.createQualityGovernor({startLevel: 1, recoverFrames: 300});
    const up = run(g, 2000, 5);
    assert.deepEqual(up.map((c) => c.level), [0]);
    assert.equal(g.level, 0);
});

// ------------------------------------------------------------------ honesty of colours and labels
test('measured and cue colours bypass tone mapping; labels and status text are untouched', () => {
    const cues = declaration('    updateAssayCues() {');
    assert.match(cues, /this\.cueTexture\.encoding = THREE\.sRGBEncoding;/);
    assert.match(cues, /material\.toneMapped = false;/);
    const vp = viewport({});
    for (const leg of vp.legs) assert.equal(leg.contactMat.toneMapped, false);
    vp.setContactColor(vp.legs[0], 0x22c55e, 0.9);
    const shown = vp.legs[0].contactMat.color.clone().convertLinearToSRGB().getHex();
    assert.equal(shown, 0x22c55e);
    assert.match(app, /`Illustrative 3D view · \$\{poseLabel\} · \$\{limbLabel\} · \$\{contactLabel\} · Drag to Orbit`/);
    // Geometry: floors stay at -0.02, walls 2.0 high centred at 1.0.
    const geometry = declaration('    updateAssayGeometry() {');
    assert.match(geometry, /floor\.position\.set\(surfaceCenter\.x, -0\.02, -surfaceCenter\.y\);/);
    assert.match(geometry, /new THREE\.BoxGeometry\(length, 2\.0, 0\.45\)/);
    assert.match(geometry, /\(wall\.p1\.x \+ wall\.p2\.x\) \/ 2, 1\.0,/);
});

test('illustrative animation clips are never played in the dashboard viewport', () => {
    assert.ok(!/AnimationMixer|animations\b|clip/i.test(declaration('class ArticulatedFly3DViewport ')));
});
