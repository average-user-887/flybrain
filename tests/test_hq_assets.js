'use strict';
// Opt-in HQ presentation assets (web/hq_assets.js): off by default, clean fallback,
// and with ?assets=hq the GLB meshes hang on the viewport's own pose groups.
const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');

const WEB = path.resolve(__dirname, '../web');
const read = (rel) => fs.readFileSync(path.join(WEB, rel), 'utf8');
const HQ = require(path.join(WEB, 'hq_assets.js'));
const app = read('app.js');

// Real three.js r128 + the vendored r128 GLTFLoader in an isolated context.
const ctx = vm.createContext({TextDecoder, console, URL});
vm.runInContext('var self = globalThis;', ctx);
vm.runInContext(read('vendor/three.min.js'), ctx);
vm.runInContext(read('vendor/GLTFLoader.js'), ctx);
const THREE = ctx.THREE;

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
vm.runInContext(declaration('class ArticulatedFly3DViewport ') + '\nthis.Viewport = ArticulatedFly3DViewport;', ctx);

function proceduralViewport() {
    const vp = Object.create(ctx.Viewport.prototype);
    vp.scene = new THREE.Scene();
    vp.buildFlyMesh();             // the production procedural fly, unchanged
    return vp;
}

// ------------------------------------------------------------------ synthetic GLB
function glb(names, size = 2) {
    const pos = new Float32Array([0, 0, 0, size / 4, -size, 0, -size / 4, -size, 0.1]);
    const json = {
        asset: {version: '2.0'}, scene: 0, scenes: [{nodes: names.map((_, i) => i)}],
        nodes: names.map((name) => ({name, mesh: 0})),
        meshes: [{primitives: [{attributes: {POSITION: 0}}]}],
        accessors: [{bufferView: 0, componentType: 5126, count: 3, type: 'VEC3',
            min: [-size / 4, -size, 0], max: [size / 4, 0, 0.1]}],
        bufferViews: [{buffer: 0, byteOffset: 0, byteLength: pos.byteLength}],
        buffers: [{byteLength: pos.byteLength}]
    };
    let text = Buffer.from(JSON.stringify(json));
    text = Buffer.concat([text, Buffer.alloc((4 - text.length % 4) % 4, 0x20)]);
    const bin = Buffer.from(pos.buffer);
    const out = Buffer.alloc(12 + 8 + text.length + 8 + bin.length);
    out.writeUInt32LE(0x46546C67, 0); out.writeUInt32LE(2, 4); out.writeUInt32LE(out.length, 8);
    out.writeUInt32LE(text.length, 12); out.writeUInt32LE(0x4E4F534A, 16); text.copy(out, 20);
    const b = 20 + text.length;
    out.writeUInt32LE(bin.length, b); out.writeUInt32LE(0x004E4942, b + 4); bin.copy(out, b + 8);
    return out.buffer.slice(out.byteOffset, out.byteOffset + out.length);
}
const parse = (buffer) => new Promise((resolve, reject) => new THREE.GLTFLoader().parse(buffer, '', resolve, reject));
const FULL = HQ.requiredFlyNodes().concat(['c_scutum', 'l_haltere', 'r_haltere']);

function fakeDoc() {
    const created = [];
    return {created, createElement(tag) {
        const el = {tag, style: {cssText: ''}, textContent: '', children: [], appendChild(c) { this.children.push(c); }};
        created.push(el);
        return el;
    }, body: {appendChild() {}}, head: {appendChild() {}}};
}
function materialStates(vp) {
    const out = [];
    vp.flyGroup.traverse((o) => { if (o.isMesh) out.push(o.material.visible); });
    return out;
}

// ------------------------------------------------------------------ flag
test('the flag is exactly ?assets=hq and is off by default', () => {
    for (const search of ['', '?assets=', '?assets=HQ', '?assets=hq2', '?asset=hq', '?inject=hq'])
        assert.equal(HQ.flagEnabled({search}), false, search);
    for (const search of ['?assets=hq', '?daemon=x&assets=hq'])
        assert.equal(HQ.flagEnabled({search}), true, search);
    assert.equal(HQ.flagEnabled(null), false);
});

test('without the flag nothing is fetched, injected or touched', async () => {
    const doc = fakeDoc();
    let fetched = 0;
    const hq = HQ.create({location: {search: ''}, THREE, document: doc, loadGlb: () => { fetched += 1; return Promise.reject(new Error('x')); }});
    const vp = proceduralViewport();
    const before = JSON.stringify(vp.flyGroup.toJSON());
    const update = vp.updateAssayGeometry;
    assert.equal(hq.attachViewport(vp), null);
    assert.equal(hq.attachReplay({THREE, scene: new THREE.Scene(), state: {}}), null);
    await new Promise((r) => setImmediate(r));
    assert.equal(fetched, 0);
    assert.equal(doc.created.length, 0, 'no label, no script tag');
    assert.equal(JSON.stringify(vp.flyGroup.toJSON()), before, 'procedural fly byte-identical');
    assert.equal(vp.updateAssayGeometry, update);
    assert.equal(hq.status, 'off');
});

test('dashboard and replay hooks are single guarded calls; GLTFLoader is never loaded statically', () => {
    // One guarded hook in init(); the only other references are the viewport's opt-in
    // variant selection (configureHQVariant) and its status read (syncHQPresentation),
    // both of which return at once without the flag.
    const methods = ['static configureHQVariant(', 'syncHQPresentation() {'].map((m) => declaration(m.startsWith('static') ? m : '    ' + m));
    const rest = methods.reduce((text, body) => text.replace(body, ''), app);
    assert.equal(rest.split('NeuroflyHQAssets').length - 1, 2);
    assert.match(methods[0], /if \(!lib \|\| !lib\.flagEnabled \|\| !lib\.flagEnabled\(window\.location\)\) return null;/);
    assert.match(app, /this\.buildFlyMesh\(\);\n.*\n\s+if \(window\.NeuroflyHQAssets\) window\.NeuroflyHQAssets\.attachViewport\(this\);/);
    const replay = read('embodied_replay.js');
    assert.match(replay, /if \(window\.NeuroflyHQAssets\) window\.NeuroflyHQAssets\.attachReplay\(\{ THREE, scene, state \}\);/);
    for (const page of ['index.html', 'embodied_replay.html']) {
        const html = read(page);
        assert.ok(html.includes('<script src="hq_assets.js"></script>'), page);
        assert.ok(!html.includes('GLTFLoader'), page);
    }
});

// ------------------------------------------------------------------ fallback
test('a missing GLB keeps the procedural fly and says so', async () => {
    const doc = fakeDoc();
    const hq = HQ.create({location: {search: '?assets=hq'}, THREE, document: doc,
        loadGlb: () => Promise.reject(new Error('HTTP 404'))});
    const vp = proceduralViewport();
    vp.statusPanel = {appendChild() {}};
    const before = JSON.stringify(vp.flyGroup.toJSON());
    assert.equal(await hq.attachViewport(vp), null);
    assert.equal(hq.status, 'fallback');
    assert.equal(JSON.stringify(vp.flyGroup.toJSON()), before);
    assert.ok(materialStates(vp).every(Boolean));
    assert.match(doc.created[0].textContent, /HQ assets unavailable \(HTTP 404\) · procedural fly shown/);
});

test('an invalid or incomplete GLB is rejected by the real r128 loader path', async () => {
    for (const [buffer, reason] of [[glb(['c_thorax', 'c_head']), /missing nodes: l_eye/],
        [glb(FULL, 40), /thorax size/], [new ArrayBuffer(16), /./]]) {
        const hq = HQ.create({location: {search: '?assets=hq'}, THREE, document: fakeDoc(), loadGlb: () => parse(buffer)});
        const vp = proceduralViewport();
        const before = JSON.stringify(vp.flyGroup.toJSON());
        assert.equal(await hq.attachViewport(vp), null);
        assert.equal(hq.status, 'fallback');
        assert.match(hq.reason, reason);
        assert.equal(JSON.stringify(vp.flyGroup.toJSON()), before);
    }
});

// ------------------------------------------------------------------ success
test('with ?assets=hq the GLB meshes hang on the existing pose groups; contacts stay visible', async () => {
    const hq = HQ.create({location: {search: '?assets=hq'}, THREE, document: fakeDoc(),
        loadGlb: (url) => parse(url.includes('arena') ? glb(['shell_rect', 'shell_round'], 1) : glb(FULL))});
    const vp = proceduralViewport();
    vp.assayGeometryDescriptor = {available: true, width: 100, depth: 60, walls: [{}], bounds: {minX: 0, maxX: 100, minY: 0, maxY: 60}};
    const groupsBefore = vp.legs.map((l) => [l.coxa.position.toArray(), l.coxa.rotation.toArray(), l.femur.position.toArray(), l.tibia.position.toArray()]);
    assert.equal(await hq.attachViewport(vp), 'hq');
    assert.equal(hq.status, 'hq');
    const byName = {};
    vp.flyGroup.traverse((o) => { if (o.name) byName[o.name] = o; });
    assert.equal(byName.c_thorax.parent, vp.thoraxMesh);
    assert.equal(byName.l_wing.parent, vp.thoraxMesh);
    for (const leg of vp.legs) {
        const p = HQ.VIEWPORT_LEGS[leg.name];
        assert.equal(byName[p + '_coxa'].parent, leg.coxa, leg.name);
        assert.equal(byName[p + '_trochanterfemur'].parent, leg.femur, leg.name);
        assert.equal(byName[p + '_tibia'].parent, leg.tibia, leg.name);
        assert.equal(byName[p + '_tarsus'].parent, leg.tibia, leg.name);
        assert.equal(leg.contact.material.visible, true, 'contact sphere carries measured stance');
    }
    assert.deepEqual(vp.legs.map((l) => [l.coxa.position.toArray(), l.coxa.rotation.toArray(), l.femur.position.toArray(), l.tibia.position.toArray()]), groupsBefore);
    assert.equal(vp.thoraxMesh.material.visible, false);
    assert.equal(vp.legs[0].coxa.children[0].material.visible, false);
    // Arena shell: scaled to the simulation's own bounds, top below the floor.
    const shell = vp.hqArenaShell;
    assert.ok(shell);
    assert.equal(shell.position.y, -0.06);
    assert.ok(shell.scale.x > 100 && shell.scale.z > 60 && shell.scale.x - 100 === shell.scale.z - 60);
    // Restoring returns the procedural fly exactly.
    hq.restoreViewport();
    assert.ok(materialStates(vp).every(Boolean));
    let leftovers = 0;
    vp.flyGroup.traverse((o) => { if (FULL.includes(o.name)) leftovers += 1; });
    assert.equal(leftovers, 0);
});

test('replay rig stretches each leg mesh exactly between recorded joint positions', async () => {
    const segments = ['c_thorax', 'c_head', 'lf_coxa', 'lf_trochanterfemur', 'lf_tibia', 'lf_tarsus1', 'lf_tarsus5'];
    const pos = [0, 0, 1, 0.6, 0, 1.1, 0.1, 0.2, 0.9, 0.3, 0.5, 0.6, 0.6, 0.9, 0.3, 0.8, 1.2, 0.05, 1.0, 1.4, 0.0];
    const state = {rec: {header: {skeleton: {segments, parents: [-1, 0, 0, 2, 3, 4, 5]}}, frames: [{pos, yaw: 0}]}, frame: 0};
    const scene = new THREE.Scene();
    const hq = HQ.create({location: {search: '?assets=hq'}, THREE, document: fakeDoc(), requestAnimationFrame: () => {},
        loadGlb: () => parse(glb(FULL))});
    assert.equal(await hq.attachReplay({THREE, scene, state}), 'hq');
    const rig = scene.getObjectByName('hq-replay-rig');
    assert.ok(rig);
    rig.updateMatrixWorld(true);
    const {edges} = hq._replay.replayEdges(segments);
    assert.deepEqual(edges.map((e) => e.mesh), ['lf_coxa', 'lf_trochanterfemur', 'lf_tibia', 'lf_tarsus']);
    rig.children.slice(0, edges.length).forEach((wrapper, i) => {
        const e = edges[i];
        const p0 = new THREE.Vector3().fromArray(pos, 3 * e.from), p1 = new THREE.Vector3().fromArray(pos, 3 * e.to);
        const dir = p1.clone().sub(p0);
        const length = dir.length();
        dir.normalize();
        let lo = Infinity, hi = -Infinity;
        wrapper.traverse((o) => {
            if (!o.isMesh) return;
            const a = o.geometry.attributes.position;
            for (let k = 0; k < a.count; k += 1) {
                const t = new THREE.Vector3().fromBufferAttribute(a, k).applyMatrix4(o.matrixWorld).sub(p0).dot(dir);
                lo = Math.min(lo, t); hi = Math.max(hi, t);
            }
        });
        assert.ok(Math.abs(lo) < 1e-5 && Math.abs(hi - length) < 1e-5, `${e.mesh}: ${lo}..${hi} vs ${length}`);
    });
    const body = rig.children[edges.length];
    const femur = Math.hypot(0.3, 0.4, 0.3);
    assert.ok(Math.abs(body.scale.x - femur / 2.2) < 1e-9, 'body size comes from the recorded femur');
    const head = new THREE.Vector3(0, 0.25, 1.75).applyMatrix4(body.matrixWorld);
    assert.ok(head.distanceTo(new THREE.Vector3(0.6, 0, 1.1)) < 1e-6, 'drawn head sits on the recorded head point');
    // FlyGym 2 recordings have no c_head body: the eye bodies stand in for it.
    const real = hq._replay.replayEdges(['c_thorax', 'l_eye', 'r_eye', 'rm_trochanterfemur', 'rm_tibia']);
    assert.equal(real.head, 1);
    assert.deepEqual(real.femurs, [[3, 4]]);
});

test('lowestFootY measures the tarsus vertices in the root frame and leaves the root where it was', async () => {
    const gltf = await parse(glb(FULL, 2));
    gltf.scene.position.set(5, 7, -3);
    assert.equal(HQ.lowestFootY(THREE, gltf.scene), -2);      // synthetic tarsus spans y 0..-2
    assert.equal(JSON.stringify(gltf.scene.position.toArray()), "[5,7,-3]");
    assert.equal(HQ.lowestFootY(THREE, (await parse(glb(['c_thorax'], 2))).scene), null);
});

test('replay success label appears only once the posed mesh is in the scene, and tracks every rendered frame', async () => {
    const segments = ['c_thorax', 'l_eye', 'lf_coxa', 'lf_trochanterfemur', 'lf_tibia', 'lf_tarsus1', 'lf_tarsus5'];
    const frame = (dx) => ({pos: [0, 0, 1, 0.3, 0, 1, 0.1, 0.2, 0.9, 0.3, 0.5, 0.6, 0.6, 0.9, 0.3, 0.8, 1.2, 0.05, 1.0, 1.4, 0]
        .map((v, i) => (i % 3 === 0 ? v + dx : v)), yaw: 0});
    const rec = {header: {skeleton: {segments, parents: [-1, 0, 0, 2, 3, 4, 5]}}, frames: [frame(0), frame(2)]};
    const state = {rec: null, frame: 0};
    const scene = new THREE.Scene();
    const doc = fakeDoc();
    const hq = HQ.create({location: {search: '?assets=hq'}, THREE, document: doc, loadGlb: () => parse(glb(FULL))});
    assert.equal(await hq.attachReplay({THREE, scene, state}), 'ready');
    assert.match(doc.created[0].textContent, /open a recording/);
    assert.equal(scene.getObjectByName('hq-replay-rig'), undefined);
    state.rec = rec;                         // the page loads a recording ...
    scene.onBeforeRender();                  // ... and renders: the rig is posed in the same call
    const rig = scene.getObjectByName('hq-replay-rig');
    assert.ok(rig && rig.parent === scene);
    assert.equal(hq.status, 'hq');
    assert.equal(doc.created[0].textContent, HQ.REPLAY_LABEL);
    const coxa = rig.children[0];
    assert.equal(JSON.stringify(coxa.position.toArray().map((v) => +v.toFixed(6))), "[0.1,0.2,0.9]");
    state.frame = 1;                         // Play advances the frame; the next render moves the mesh
    scene.onBeforeRender();
    assert.equal(hq.replayFrame, 1);
    assert.equal(JSON.stringify(coxa.position.toArray().map((v) => +v.toFixed(6))), "[2.1,0.2,0.9]");
    let culled = 0;
    rig.traverse((o) => { if (o.isMesh && o.frustumCulled) culled += 1; });
    assert.equal(culled, 0);
    // A skeleton without FlyGym names keeps the skeleton and says the surface is unavailable.
    state.rec = {header: {skeleton: {segments: ['thorax', 'head'], parents: [-1, 0]}}, frames: [{pos: [0, 0, 0, 1, 0, 0]}]};
    state.frame = 0;
    scene.onBeforeRender();
    assert.equal(hq.status, 'fallback');
    assert.match(doc.created[0].textContent, /skeleton shown/);
});

// The real Blender export, when it has been staged (tools/assets/README.md); skipped otherwise.
const BUILT = path.join(WEB, 'assets/hq/fly_hq_lod0.glb');
test('the staged Blender GLB (if present) passes the same validation', {skip: !fs.existsSync(BUILT)}, async () => {
    for (const lod of [0, 1]) {
        const file = path.join(WEB, `assets/hq/fly_hq_lod${lod}.glb`);
        if (!fs.existsSync(file)) continue;
        const bytes = fs.readFileSync(file);
        const gltf = await parse(bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.length));
        assert.equal(HQ.validateFly(THREE, HQ.nodeIndex(gltf.scene)), null, `lod${lod}`);
        // Static preview placement: the feet are the lowest geometry, and standing the
        // root on the floor top (-0.02) puts nothing below it.
        const foot = HQ.lowestFootY(THREE, gltf.scene);
        let lowest = Infinity;
        gltf.scene.updateMatrixWorld(true);
        gltf.scene.traverse((o) => {
            const a = o.isMesh && o.geometry.attributes.position;
            if (a) for (let k = 0; k < a.count; k += 1) lowest = Math.min(lowest, new THREE.Vector3().fromBufferAttribute(a, k).applyMatrix4(o.matrixWorld).y);
        });
        assert.ok(Math.abs(foot - lowest) < 1e-9, `lod${lod}: feet ${foot} vs mesh ${lowest}`);
        const vp = proceduralViewport();
        const hq = HQ.create({location: {search: '?assets=hq'}, THREE, document: fakeDoc(),
            loadGlb: () => Promise.resolve(gltf)});
        assert.equal(await hq.attachViewport(vp), 'hq');
    }
});

// Female/male appearance variants (tools/assets/SEX_VARIANTS.md), when staged.
const VARIANTS = ['female', 'male'].flatMap((sex) => [0, 1].map((lod) => [sex, lod]))
    .filter(([sex, lod]) => fs.existsSync(path.join(WEB, `assets/hq/fly_${sex}_lod${lod}.glb`)));
test('staged sex variants share the rig interface, carry scale metadata and stand on their feet',
    {skip: VARIANTS.length === 0}, async () => {
        for (const [sex, lod] of VARIANTS) {
            const bytes = fs.readFileSync(path.join(WEB, `assets/hq/fly_${sex}_lod${lod}.glb`));
            const gltf = await parse(bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.length));
            const tag = `${sex} lod${lod}`;
            assert.equal(HQ.validateFly(THREE, HQ.nodeIndex(gltf.scene)), null, tag);
            const root = gltf.scene.getObjectByName('neurofly_fly');
            assert.equal(root.userData.nf_rig, 'neurofly-viewport-fly-v1', tag);
            assert.equal(root.userData.nf_variant, sex, tag);
            assert.equal(root.userData.nf_display_scale, sex === 'male' ? 0.8696 : 1, tag);
            assert.match(root.userData.nf_appearance_only, /no brain dataset, physiology or behaviour/);
            // Same joint origins as the neutral rig: the coxa joints sit where the viewport puts them.
            for (const [leg, x, z] of [['lf', -1, 0.8], ['lm', -1, 0], ['lh', -1, -0.8], ['rf', 1, 0.8], ['rm', 1, 0], ['rh', 1, -0.8]]) {
                const p = gltf.scene.getObjectByName(leg + '_coxa_joint').position;
                assert.ok(Math.abs(p.x - x) < 1e-5 && Math.abs(p.y + 0.2) < 1e-5 && Math.abs(p.z - z) < 1e-5, `${tag} ${leg}`);
            }
            assert.ok(Math.abs(HQ.lowestFootY(THREE, gltf.scene) + 2.1315) < 1e-4, tag);
            const vp = proceduralViewport();
            const hq = HQ.create({location: {search: '?assets=hq'}, THREE, document: fakeDoc(), loadGlb: () => Promise.resolve(gltf)});
            assert.equal(await hq.attachViewport(vp), 'hq', tag);
        }
    });
