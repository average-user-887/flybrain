'use strict';
// Left/right and heading of the dashboard 3D viewport, checked on the ACTUAL viewport
// class (updatePose, setDisplayScale, the Top camera preset and three.js projection)
// against recorded FlyGym body frames and the 2D arena's heading convention.
// Background: docs/RENDERER_HANDEDNESS_20261008.md.
const test = require('node:test');
const assert = require('node:assert/strict');
const path = require('node:path');
const fs = require('node:fs');
const vm = require('node:vm');
const check = require('../scripts/check_viewport_handedness.js');

const WEB = path.resolve(__dirname, '../web');
const read = (rel) => fs.readFileSync(path.join(WEB, rel), 'utf8');
const app = read('app.js');
const rec = JSON.parse(fs.readFileSync(path.join(__dirname, 'fixtures/viewport_handedness_frames.json'), 'utf8'));

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

const ctx = vm.createContext({
    TextDecoder, console, URL, performance,
    window: {location: {search: ''}},
    document: {getElementById: () => null, createElement: () => ({style: {}, getContext: () => null})}
});
vm.runInContext('var self = globalThis;', ctx);
vm.runInContext(read('vendor/three.min.js'), ctx);
vm.runInContext(read('vendor/GLTFLoader.js'), ctx);
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
const HQ = require(path.join(WEB, 'hq_assets.js'));

const GEOMETRY = {schema: 'neurofly.assay-geometry.v1', coordinate_frame: 'arena-mm', source: 'python-arena',
    bounds: [0, 0, 100, 100], containment: {kind: 'rectangle', bounds: [0, 0, 100, 100]}, walls: [],
    surfaces: [{shape: 'rectangle', bounds: [0, 0, 100, 100]}]};

function viewport(heading, x = 0, y = 0) {   // near the Top camera axis (little parallax)
    const vp = Object.create(Viewport.prototype);
    vp.scene = new THREE.Scene();
    vp.buildFlyMesh();
    vp.standOffset = Viewport.standOffsetFor(Viewport.proceduralLowestLegY());
    vp.initialized = true;
    vp.cameraMode = 'orbit';
    vp.poseStatus = {style: {}};
    vp.camera = new THREE.PerspectiveCamera(45, 1, 0.1, 1000);
    vp.controls = {target: new THREE.Vector3(), enabled: true, update() { vp.camera.lookAt(this.target); }};
    vp.arena = {activeParadigmId: 'open-arena', awaitingDaemon: false, width: 100, height: 100,
        worldBounds: {minX: 0, maxX: 100, minY: 0, maxY: 100}, fly: {x, y, heading},
        remotePacket: {type: 'telemetry', paradigm: 'open-arena', world_bounds: [0, 0, 100, 100], fly: {heading},
            identity: {run_id: 'r', instance_id: 'i', activation: 1}, segment_id: 's', scene: {geometry: GEOMETRY},
            joint_angles_rad: Array(18).fill(0.2), leg_contacts: [true, false, true, false, true, false],
            body_position_mm: [x, y, 0.5]}};
    vp.updatePose();
    return vp;
}

// Screen position (pixels, y up) through the real Top preset camera.
function topScreen(vp) {
    vp.setCameraMode('top');
    vp.camera.updateMatrixWorld(true);
    vp.scene.updateMatrixWorld(true);
    return (obj) => {
        const v = obj.getWorldPosition(new THREE.Vector3()).project(vp.camera);
        return [v.x * 500, v.y * 500];
    };
}

const deg = (d) => d * Math.PI / 180;
const sub = (a, b) => [a[0] - b[0], a[1] - b[1]];
const dot = (a, b) => a[0] * b[0] + a[1] * b[1];
const cross = (a, b) => a[0] * b[1] - a[1] * b[0];   // > 0: b is to the left of a (y up)
const unit = (a) => { const n = Math.hypot(a[0], a[1]); return [a[0] / n, a[1] / n]; };

test('Top view: the drawn head points where the 2D arena points, at cardinal and oblique headings', () => {
    // The 2D arena draws the heading as (cos h, -sin h) in canvas pixels, i.e.
    // (cos h, sin h) with screen y up.
    for (const d of [0, 45, 90, 135, 180, 225, 270, 315]) {
        const vp = viewport(deg(d));
        const screen = topScreen(vp);
        const head = vp.thoraxMesh.children[0];                       // head mesh at +Z
        const fwd = unit(sub(screen(head), screen(vp.thoraxMesh)));
        assert.ok(dot(fwd, [Math.cos(deg(d)), Math.sin(deg(d))]) > 0.999, `heading ${d}: ${fwd}`);
    }
});

test('recorded left legs land on the drawn fly\'s anatomical left (procedural and display scales)', () => {
    const recorded = check.analyse(THREE, rec).recorded;
    assert.equal(recorded.lfLeft, recorded.frames);                    // the data itself is consistent
    for (const frame of rec.frames) {
        for (const extra of [0, 60, 90, 180, 250]) {
            const f = check.rotateFrame(frame, deg(extra));
            const r = check.recordedSides(rec.segments, f);
            for (const scale of [1, 0.8696]) {
                const vp = viewport(f.yaw);
                vp.setDisplayScale(scale);
                vp.updatePose();
                const screen = topScreen(vp);
                const fwd = sub(screen(vp.thoraxMesh.children[0]), screen(vp.thoraxMesh));
                // The recorded heading agrees with the drawn head ...
                assert.ok(dot(unit(fwd), r.forward) > 0.99, `yaw ${f.yaw}`);
                for (const leg of vp.legs) {
                    const side = cross(fwd, sub(screen(leg.coxa), screen(vp.thoraxMesh)));
                    // ... and every L coxa is on its left and every R coxa on its right,
                    // exactly as the recorded lf/lm/lh and rf/rm/rh coxae are.
                    const recordedSide = r.side[check.VIEWPORT[leg.name]];
                    assert.ok(side !== 0 && Math.sign(side) === Math.sign(recordedSide), `${leg.name} yaw ${f.yaw} scale ${scale}`);
                }
            }
        }
    }
});

test('the reflection keeps the floor offset, the contact colours and the Chase camera behind the head', () => {
    for (const scale of [1, 0.8696]) {
        const vp = viewport(deg(30));
        vp.setDisplayScale(scale);
        assert.ok(vp.flyGroup.scale.x < 0 && Math.abs(vp.flyGroup.scale.x) === scale);
        vp.refreshStand();
        assert.ok(Math.abs(vp.standLowestY + 3.85 * scale) < 1e-6, `${vp.standLowestY}`);
        vp.updatePose();
        vp.scene.updateMatrixWorld(true);
        let low = Infinity;
        const v = new THREE.Vector3();
        for (const leg of vp.legs) leg.coxa.traverse((o) => {
            const a = o.isMesh && o.geometry.attributes.position;
            if (a) for (let k = 0; k < a.count; k += 1) low = Math.min(low, v.fromBufferAttribute(a, k).applyMatrix4(o.matrixWorld).y);
        });
        assert.ok(low >= Viewport.floorTop() - 1e-6);
        // Contact colours follow the leg index, not the position: L1 stance green, R1 swing blue.
        const shown = (leg) => leg.contactMat.color.clone().convertLinearToSRGB().getHex();
        assert.equal(shown(vp.legs[0]), 0x22c55e);
        assert.equal(shown(vp.legs[3]), 0x38bdf8);
        // Chase sits behind the drawn head.
        vp.cameraMode = 'chase';
        vp.controls.enabled = false;
        for (let i = 0; i < 200; i += 1) vp.updatePose();
        const head = vp.thoraxMesh.children[0].getWorldPosition(new THREE.Vector3()).sub(vp.thoraxMesh.getWorldPosition(new THREE.Vector3()));
        const cam = vp.camera.position.clone().sub(vp.flyGroup.position);
        assert.ok(head.x * cam.x + head.z * cam.z < 0, 'camera behind the head');
    }
});

test('contact-shadow sizing uses the magnitude of the signed root scale', () => {
    const vp = viewport(deg(10));
    vp.setDisplayScale(0.8696);
    vp.contactShadow = {position: new THREE.Vector3(), rotation: {y: 0}, scale: new THREE.Vector3(), material: {}, visible: false};
    vp.updatePresentation(1 / 60);
    assert.ok(vp.contactShadow.scale.x > 0 && Math.abs(vp.contactShadow.scale.x - 5.2 * 0.8696) < 1e-9);
});

// Staged rig v2 GLBs (git-ignored, optional): the real HQ parts on the anatomical left.
const STAGED = ['female', 'male'].map((sex) => [sex, path.join(WEB, `assets/hq/fly_${sex}_v2_lod0.glb`)])
    .filter(([, f]) => fs.existsSync(f));
test('HQ rig v2 female/male: l_wing and lf_coxa meshes draw on the anatomical left', {skip: STAGED.length === 0}, async () => {
    for (const [sex, file] of STAGED) {
        const bytes = fs.readFileSync(file);
        const gltf = await new Promise((resolve, reject) => new THREE.GLTFLoader()
            .parse(bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.length), '', resolve, reject));
        const scale = gltf.scene.getObjectByName('neurofly_fly').userData.nf_display_scale;
        const vp = viewport(deg(120));
        const doc = {createElement: () => ({style: {}}), head: {appendChild() {}}};
        const hq = HQ.create({location: {search: '?assets=hq'}, THREE, document: doc, loadGlb: () => Promise.resolve(gltf)});
        assert.equal(await hq.attachViewport(vp), 'hq', sex);
        vp.setDisplayScale(scale);
        vp.updatePose();
        const screen = topScreen(vp);
        const fwd = sub(screen(vp.thoraxMesh.children[0]), screen(vp.thoraxMesh));
        const centre = (name) => {
            const box = new THREE.Box3().setFromObject(vp.flyGroup.getObjectByName(name));
            const c = box.getCenter(new THREE.Vector3()).project(vp.camera);
            return [c.x * 500, c.y * 500];
        };
        for (const [name, left] of [['l_wing', true], ['r_wing', false], ['lf_coxa', true], ['rh_coxa', false], ['l_haltere', true], ['r_pedicel', false]]) {
            const side = cross(fwd, sub(centre(name), screen(vp.thoraxMesh)));
            assert.equal(side > 0, left, `${sex} ${name}`);
        }
    }
});
