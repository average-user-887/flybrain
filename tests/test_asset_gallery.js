'use strict';
// Opt-in asset gallery (web/asset_gallery.html + asset_gallery.js): manifest-driven,
// measured from geometry, and never loaded by the dashboard or the replay page.
const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');

const WEB = path.resolve(__dirname, '../web');
const read = (rel) => fs.readFileSync(path.join(WEB, rel), 'utf8');
const G = require(path.join(WEB, 'asset_gallery.js'));
const HQ = require(path.join(WEB, 'hq_assets.js'));
const MANIFEST = JSON.parse(read('asset_manifest.json'));

const ctx = vm.createContext({TextDecoder, console, URL});
vm.runInContext('var self = globalThis;', ctx);
vm.runInContext(read('vendor/three.min.js'), ctx);
const THREE = ctx.THREE;

test('the dashboard, replay and studio pages never load the gallery', () => {
    for (const page of ['index.html', 'embodied_replay.html', 'studio.html', 'research.html']) {
        const html = read(page);
        assert.ok(!html.includes('asset_gallery'), page);
        assert.ok(!html.includes('asset_manifest'), page);
    }
    for (const js of ['app.js', 'hq_assets.js', 'embodied_replay.js', 'live_assays.js', 'assay_cues_3d.js'])
        assert.ok(!read(js).includes('asset_gallery') && !read(js).includes('NeuroflyAssetGallery'), js);
    const daemon = fs.readFileSync(path.resolve(__dirname, '../neurofly_daemon.py'), 'utf8');
    assert.ok(!daemon.includes('asset_gallery'), 'not part of the dashboard web-build identity');
});

test('the committed manifest validates and every LOD is a relative path under assets/', () => {
    const v = G.validateManifest(MANIFEST);
    assert.deepEqual(v.errors, []);
    assert.ok(v.assets.length >= 3);
    for (const a of v.assets) {
        for (const l of a.lods) assert.match(l.url, /^assets\/[\w./-]+\.glb$/, a.id);
        assert.ok(['illustrative', 'decorative', 'measured', 'reference'].includes(a.status), a.id);
        assert.ok(a.notes.length > 20, a.id + ' carries an honesty note');
    }
    const fly = v.assets.find((a) => a.id === 'fly-hq');
    assert.deepEqual(fly.lods.map((l) => l.level), [0, 1]);
    assert.equal(fly.stand, 'tarsus');
});

test('manifest validation rejects remote urls, duplicates, empty LODs and wrong schema', () => {
    const base = {schema: G.MANIFEST_SCHEMA, assets: []};
    assert.match(G.validateManifest(null).errors[0], /not a JSON object/);
    assert.match(G.validateManifest({assets: []}).errors.join(), /schema/);
    const v = G.validateManifest(Object.assign({}, base, {assets: [
        {id: 'a', lods: [{url: 'https://example.org/x.glb'}]},
        {id: 'b', lods: [{url: '/abs.glb'}]},
        {id: 'c', lods: []},
        {id: 'd', lods: [{url: 'assets/d.glb'}], layers: 'nope'},
        {id: 'd', lods: [{url: 'assets/d2.glb'}]},
        {lods: [{url: 'assets/e.glb'}]}]}));
    assert.deepEqual(v.assets.map((a) => a.id), ['d']);
    const all = v.errors.join('\n');
    for (const re of [/a: LOD urls must be relative/, /b: LOD urls must be relative/, /c: no LOD/, /unknown layer set nope/, /duplicate id d/, /missing id/])
        assert.match(all, re);
});

test('every node of the fly hook contract falls in a declared layer, none in "other"', () => {
    const layers = G.compileLayers(MANIFEST.layer_sets.fly);
    const names = HQ.requiredFlyNodes().concat(['c_scutum', 'c_thorax_bristles', 'c_head_bristles', 'c_proboscis',
        'l_pedicel', 'r_pedicel', 'l_funiculus', 'r_funiculus', 'l_arista', 'r_arista', 'c_abdomen3', 'c_abdomen6',
        'l_wing_veins', 'r_wing_veins', 'l_haltere', 'r_haltere']);
    for (const n of names) assert.notEqual(G.classifyName(n, layers), null, n);
    assert.equal(G.classifyName('l_wing', layers), 'wings');
    assert.equal(G.classifyName('l_wing_veins', layers), 'wings');
    assert.equal(G.classifyName('rh_tarsus', layers), 'legs');
    assert.equal(G.classifyName('c_head_bristles', layers), 'bristles');
    assert.equal(G.classifyName('lf_coxa_joint', layers), null, 'joint empties are not layers');
    // A primitive mesh inherits the layer of its nearest named ancestor.
    const parent = new THREE.Group(); parent.name = 'r_eye';
    const prim = new THREE.Mesh(new THREE.BufferGeometry()); prim.name = 'r_eye_1';
    parent.add(prim);
    assert.equal(G.layerOf(prim, layers, null), 'eyes');
    assert.equal(G.layerOf(new THREE.Mesh(), layers, null), 'other');
});

test('floor status, plinth status and stand lift are exact and signed', () => {
    assert.equal(G.floorStatus(-0.02, -0.02).state, 'contact');
    assert.equal(G.floorStatus(-0.0205, -0.02).state, 'contact');
    assert.equal(G.floorStatus(-0.05, -0.02).state, 'penetrates');
    assert.equal(G.floorStatus(0.5, -0.02).state, 'floating');
    assert.equal(G.floorStatus(NaN, -0.02).state, 'unknown');
    assert.equal(G.plinthStatus(-0.06, -0.02).state, 'below-floor');
    assert.equal(G.plinthStatus(0.1, -0.02).state, 'pokes-through');
    // INTERFACE.md: stand height on the dashboard floor is -0.02 - (-2.1315) = 2.1115.
    assert.ok(Math.abs(G.standLift('tarsus', -2.1315, -2.5, -0.02) - 2.1115) < 1e-12);
    assert.equal(G.standLift('bbox', null, -1, 0), 1);
    assert.equal(G.standLift('none', -2, -2, 0), 0);
    assert.equal(G.standLift('tarsus', NaN, -1, 0), 0, 'no feet: no invented lift');
});

test('vertexBounds is exact for rotated meshes and honours a filter', () => {
    const root = new THREE.Group();
    const m = new THREE.Mesh(new THREE.BoxGeometry(2, 2, 2));
    m.rotation.z = Math.PI / 4; m.position.y = 3; m.name = 'c_thorax';
    const leg = new THREE.Mesh(new THREE.BoxGeometry(0.2, 2, 0.2)); leg.name = 'lf_tarsus'; leg.position.y = -1;
    root.add(m, leg);
    const all = G.vertexBounds(THREE, root);
    assert.ok(Math.abs(all.min.y - -2) < 1e-9);
    const body = G.vertexBounds(THREE, root, (o) => !G.isLegMesh(o));
    assert.ok(Math.abs(body.min.y - (3 - Math.SQRT2)) < 1e-9, 'rotated cube corner, not the loose box');
    assert.equal(G.vertexBounds(THREE, new THREE.Group()), null);
    assert.equal(G.triangles(root), 24);
});

test('query parsing: defaults, LOD, hidden layers, overlays off', () => {
    const d = G.parseQuery('');
    assert.equal(d.manifest, 'asset_manifest.json');
    assert.equal(d.lod, null);
    assert.equal(d.hide.size, 0);
    const q = G.parseQuery('?asset=fly-hq&lod=1&hide=wings,bristles&off=grid&sheet=1&lights=neutral');
    assert.equal(q.asset, 'fly-hq'); assert.equal(q.lod, 1);
    assert.deepEqual([...q.hide], ['wings', 'bristles']);
    assert.deepEqual([...q.overlaysOff], ['grid']);
    assert.equal(q.sheet, true); assert.equal(q.lighting, 'neutral');
    assert.equal(G.parseQuery('?lod=x').lod, null);
});

test('grid step and camera presets', () => {
    assert.equal(G.gridStep(8), 1);
    assert.equal(G.gridStep(72), 10);
    assert.equal(G.gridStep(0), 1);
    assert.deepEqual(G.viewDirection('top').up, [0, 0, 1], 'top view: head (+Z) toward image top');
    assert.equal(G.viewDirection('side').ortho, true);
    assert.equal(G.viewDirection('anything').ortho, false);
});

test('the gallery page is self-contained and loads only vendored scripts', () => {
    const html = read('asset_gallery.html');
    const srcs = [...html.matchAll(/<script src="([^"]+)"/g)].map((m) => m[1]);
    assert.deepEqual(srcs, ['vendor/three.min.js', 'vendor/OrbitControls.js', 'vendor/GLTFLoader.js', 'hq_assets.js', 'asset_gallery.js']);
    assert.ok(!/https?:\/\//.test(html.replace(/<!--[\s\S]*?-->/g, '')), 'no remote resources');
    assert.match(html, /No simulation runs on this page/);
});

test('display scale: manifest number, GLB extras, or 1; a stand height carries it (W2 male 1.8335)', () => {
    assert.deepEqual(G.displayScaleOf(null, {nf_display_scale: 0.8}), {scale: 1, source: 'none'}, 'opt-in per manifest entry');
    assert.equal(G.displayScaleOf('extras', {nf_display_scale: 0.8696}).scale, 0.8696);
    assert.equal(G.displayScaleOf('extras', {}).scale, 1);
    assert.equal(G.displayScaleOf('extras', {nf_display_scale: 1e6}).scale, 1, 'absurd value ignored');
    assert.equal(G.displayScaleOf(2, {}).scale, 2);
    const lift = G.standLift('tarsus', -2.1315 * 0.8696, NaN, -0.02);
    assert.ok(Math.abs(lift - 1.8335) < 1e-4, String(lift));
    const root = new THREE.Group();
    const node = new THREE.Group(); node.userData = {nf_rig: 'neurofly-viewport-fly-v1', nf_variant: 'male', nf_display_scale: 0.8696};
    root.add(node);
    assert.equal(G.rootExtras(root).nf_variant, 'male');
    assert.deepEqual(G.rootExtras(new THREE.Group()), {});
});

test('sex variants in the manifest are appearance-only and read their scale from the GLB', () => {
    const v = G.validateManifest(MANIFEST);
    for (const id of ['fly-female', 'fly-male']) {
        const a = v.assets.find((x) => x.id === id);
        assert.ok(a, id);
        assert.equal(a.displayScale, 'extras');
        assert.equal(a.appearanceOnly, true);
        assert.match(a.notes, /never switches the brain dataset/);
    }
});

test('W3 props and predators: decorative / illustrative, standing on the floor, predators never claim a stimulus', () => {
    const v = G.validateManifest(MANIFEST);
    const env = v.assets.filter((a) => a.id.startsWith('env-'));
    assert.equal(env.length, 7);
    for (const a of env) {
        assert.equal(a.stand, 'bbox', a.id);
        assert.ok(a.layers.length >= 1, a.id);
        assert.match(a.notes, /DECORATIVE|ILLUSTRATIVE/, a.id);
    }
    for (const a of env.filter((x) => x.category === 'predator')) {
        assert.equal(a.status, 'illustrative');
        assert.match(a.notes, /never changes looming or retinal input/);
    }
    assert.match(env.find((a) => a.id === 'env-sugar-water').notes, /Sugar taste is NOT SIMULATED/);
    assert.ok(v.sheets.some((s) => /env_legend/.test(s.url)));
});

test('animation helpers: track targets, clip summary, joint readout from bind', () => {
    const clip = new THREE.AnimationClip('wing_open', 1.5, [
        new THREE.QuaternionKeyframeTrack('l_wing_root.quaternion', [0, 1.5], [0, 0, 0, 1, 0, 0, 0.3826834, 0.9238795]),
        new THREE.QuaternionKeyframeTrack('r_wing_root.quaternion', [0, 1.5], [0, 0, 0, 1, 0, 0, -0.3826834, 0.9238795]),
        new THREE.VectorKeyframeTrack('l_wing_root.position', [0, 1.5], [0, 0, 0, 0, 0, 0])]);
    assert.deepEqual(G.trackNodes(clip), ['l_wing_root', 'r_wing_root']);
    assert.deepEqual(G.clipSummary([clip]), [{name: 'wing_open', duration: 1.5, tracks: 3, nodes: ['l_wing_root', 'r_wing_root']}]);
    const root = new THREE.Group();
    const j = new THREE.Group(); j.name = 'l_wing_root'; root.add(j);
    const bind = j.quaternion.clone();
    const mixer = new THREE.AnimationMixer(root);
    mixer.clipAction(clip).play();
    mixer.update(1.5 - 1e-9);
    const r = G.jointReadout(THREE, j, bind);
    assert.ok(Math.abs(r.z - 45) < 0.01, String(r.z));
    assert.ok(Math.abs(r.delta - 45) < 0.01, String(r.delta));
    assert.equal(G.ANIMATION_LABEL, 'Illustrative animation, not simulated behaviour');
    assert.ok(G.SPEEDS.includes(1));
});

test('canned clips stay in the standalone gallery: no telemetry view plays an animation', () => {
    for (const js of ['app.js', 'hq_assets.js', 'embodied_replay.js', 'replay.js', 'env_inspector.js'])
        assert.ok(!/AnimationMixer|clipAction/.test(read(js)), js);
    const html = read('asset_gallery.html');
    assert.match(html, /Illustrative animation, not simulated behaviour/);
    assert.match(html, /Never applied to recorded or live telemetry/);
    const v = G.validateManifest({schema: G.MANIFEST_SCHEMA, assets: [{id: 'a', lods: [{url: 'assets/a.glb'}],
        clips: [{url: 'assets/c.glb', label: 'ok'}, {url: 'https://x/c.glb'}, {url: '/abs.glb'}]}]});
    assert.deepEqual(v.assets[0].clips.map((c) => c.url), ['assets/c.glb'], 'remote and absolute clip urls dropped');
});

test('rig v2 joint envelope: signed DOF angle and the W2 contract limits', () => {
    const q = new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(0, -1, 0), 0.5);
    assert.ok(Math.abs(G.dofAngle(q, [0, -1, 0]) - 0.5) < 1e-9);
    assert.ok(Math.abs(G.dofAngle(q, [0, 1, 0]) + 0.5) < 1e-9, 'sign follows the documented axis');
    const ranges = {l_wing_sweep: [0, 3], l_wing_elevate: [-1.7, 0.8], l_wing_pitch: [-0.27, 3.92], l_antenna_abduct: [-0.4, 0.8], l_antenna_extend: [-0.2, 0.5]};
    assert.deepEqual(G.envelopeViolations({l_wing_sweep: 1.4, l_wing_elevate: 0.09, l_wing_pitch: 0.1}, ranges), []);
    assert.match(G.envelopeViolations({l_wing_sweep: 3.2}, ranges).join(), /outside source range/);
    assert.match(G.envelopeViolations({l_wing_sweep: 2.0, l_wing_elevate: 0.0}, ranges).join(), /below display-safe minimum 0\.200/);
    assert.match(G.envelopeViolations({l_wing_sweep: 0.2, l_wing_elevate: 0, l_wing_pitch: 0.3}, ranges).join(), /while folded/);
    assert.match(G.envelopeViolations({l_antenna_abduct: 0.6, l_antenna_extend: 0.3}, ranges).join(), /above display-safe 0\.200/);
    assert.deepEqual(G.envelopeViolations({l_antenna_abduct: 0.6, l_antenna_extend: 0.2}, ranges), []);
});

test('layer wording is variant-aware: v1 static, v2 animated presentation joints', () => {
    const v = G.validateManifest(MANIFEST);
    const lab = (id, layer) => v.assets.find((a) => a.id === id).layers.find((l) => l.id === layer).label;
    for (const layer of ['wings', 'antennae', 'halteres']) {
        assert.match(lab('fly-hq', layer), /static/);
        assert.match(lab('fly-female-v2', layer), /animated, not simulated/);
        assert.doesNotMatch(lab('fly-female-v2', layer), /static/);
    }
    assert.match(read('asset_gallery.html'), /presentation joints \(animated, not simulated\)/);
});

test('rig v2 clips: repaired v3 first, failed v2 and early preview kept with their labels', () => {
    const v = G.validateManifest(MANIFEST);
    for (const id of ['fly-female-v2', 'fly-male-v2']) {
        const a = v.assets.find((x) => x.id === id);
        assert.match(a.clips[0].url, /w3_v3_.*\{lod\}_clips\.glb$/);
        assert.match(a.clips[0].label, /v3 \(final, repaired\)/);
        assert.ok(a.clips.some((c) => /w3_v2_/.test(c.url) && /FAILED/.test(c.label)));
        assert.ok(a.clips.some((c) => /PREVIEW_NOT_FINAL/.test(c.url) && /PREVIEW_NOT_FINAL/.test(c.label)));
        assert.match(a.notes, /thorax \(notum\) grooming is omitted/);
        assert.match(a.notes, /stroke reversal is NOT shown/);
        assert.match(a.notes, /no flight is shown/);
    }
});
