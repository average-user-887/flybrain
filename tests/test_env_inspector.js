'use strict';
// Opt-in environment inspector (web/env_inspector.html + env_inspector.js): shows only
// published telemetry, keeps concentration / delivered input / response apart, labels
// everything not simulated, and never sends a command.
const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');

const WEB = path.resolve(__dirname, '../web');
const read = (rel) => fs.readFileSync(path.join(WEB, rel), 'utf8');
const I = require(path.join(WEB, 'env_inspector.js'));
const FX = JSON.parse(fs.readFileSync(path.join(__dirname, 'fixtures/env_inspector_frames.json'), 'utf8')).frames;
const clone = (o) => JSON.parse(JSON.stringify(o));

test('nothing in the dashboard, replay, studio or daemon loads the inspector', () => {
    for (const page of ['index.html', 'embodied_replay.html', 'studio.html', 'research.html', 'asset_gallery.html'])
        assert.ok(!read(page).includes('env_inspector'), page);
    for (const js of ['app.js', 'hq_assets.js', 'embodied_replay.js', 'live_assays.js', 'asset_gallery.js'])
        assert.ok(!read(js).includes('env_inspector') && !read(js).includes('NeuroflyEnvInspector'), js);
    assert.ok(!fs.readFileSync(path.resolve(__dirname, '../neurofly_daemon.py'), 'utf8').includes('env_inspector'));
});

test('read-only: the inspector never posts, puts or sends a command', () => {
    const src = read('env_inspector.js');
    assert.ok(!/method\s*:/.test(src), 'no fetch method override (GET only)');
    assert.ok(!/\/api\/command|XMLHttpRequest|sendBeacon|WebSocket/.test(src));
    assert.match(src, /\/api\/stream/);
    const html = read('env_inspector.html');
    assert.deepEqual([...html.matchAll(/<script src="([^"]+)"/g)].map((m) => m[1]), ['env_inspector.js']);
});

test('only loopback daemons and relative replay files are accepted', () => {
    assert.equal(I.parseQuery('?daemon=http://127.0.0.1:19151').daemon, 'http://127.0.0.1:19151');
    assert.equal(I.parseQuery('?daemon=http://localhost:8769').daemon, 'http://localhost:8769');
    for (const bad of ['http://203.0.113.5:8769', 'https://evil.example', 'http://127.0.0.1:1/x', 'javascript:alert(1)']) {
        const q = I.parseQuery('?daemon=' + encodeURIComponent(bad));
        assert.equal(q.daemon, null, bad); assert.equal(q.daemonRejected, true, bad);
    }
    assert.equal(I.parseQuery('?src=assets/hq/replay.json').src, 'assets/hq/replay.json');
    assert.equal(I.parseQuery('?src=https://x/y.json').src, null);
    assert.equal(I.parseQuery('?src=//x/y.json').src, null);
});

test('freshness: no source, waiting, live, paused, halted, stale, disconnected, replay', () => {
    const pkt = {step: 5};
    assert.equal(I.freshness({haveSource: false}).state, 'no-source');
    assert.equal(I.freshness({haveSource: true, error: 'daemon not reachable'}).state, 'unavailable');
    assert.equal(I.freshness({haveSource: true, pkt: null}).state, 'connecting');
    assert.equal(I.freshness({haveSource: true, mode: 'live', pkt, ageS: 0.2}).state, 'live');
    assert.equal(I.freshness({haveSource: true, mode: 'live', pkt: {step: 5, paused: true}, ageS: 60}).state, 'paused', 'paused wins over age');
    assert.equal(I.freshness({haveSource: true, mode: 'live', pkt, ageS: 60, beatPaused: true}).state, 'paused');
    assert.equal(I.freshness({haveSource: true, mode: 'live', pkt: {step: 5, halted: true}, ageS: 0}).state, 'halted');
    assert.equal(I.freshness({haveSource: true, mode: 'live', pkt, ageS: I.STALE_AFTER_S + 0.1}).state, 'stale');
    assert.equal(I.freshness({haveSource: true, mode: 'live', pkt, ageS: 1, disconnected: true}).state, 'disconnected');
    assert.equal(I.freshness({haveSource: true, mode: 'replay', pkt}).state, 'replay');
    assert.match(I.freshness({haveSource: true, mode: 'replay', pkt, replayPaused: true}).text, /not live/);
});

test('real multisensory connectome frame: wind input shown exactly as reported, with the key-mismatch caveat', () => {
    const vm = I.viewModel(FX.multisensory_connectome);
    assert.equal(vm.graph, true);
    const jo = vm.input.rows.find((r) => r.name === 'jon_wind');
    const reported = FX.multisensory_connectome.connectome.input_stage.find((r) => r.name === 'jon_wind').current_injected;
    assert.equal(jo.current, reported, 'never recomputed from wind_magnitude');
    assert.ok(FX.multisensory_connectome.stimuli.wind_magnitude > 3 && !('wind_speed' in FX.multisensory_connectome.stimuli));
    assert.match(vm.caveats.join('\n'), /NOT DELIVERED in this assay.*wind_magnitude.*wind_speed/);
    assert.equal(vm.wind.published.wind_speed, null);
    assert.equal(vm.wind.published.wind_magnitude, FX.multisensory_connectome.stimuli.wind_magnitude);
    // Three separate things.
    assert.ok(vm.concentration.length >= 3);
    for (const r of vm.concentration) assert.equal(r.units, I.UNITS_ODOUR);
    assert.equal(vm.response.available, true);
    assert.deepEqual(vm.response.rates.map(([k]) => k).sort(), Object.keys(FX.multisensory_connectome.connectome.dn_rates).sort());
    assert.match(vm.lateral, /NOT PUBLISHED/);
    assert.deepEqual(vm.sources.map((s) => s.id), ['F1', 'R1', 'P1', 'H1', 'C1']);
    for (const s of vm.sources) assert.match(s.emission, /emission rate NOT SIMULATED/);
});

test('real wind-tunnel connectome frame: delivered and spiking, delivered but silent, and zero are distinct', () => {
    const vm = I.viewModel(FX.wind_tunnel_connectome);
    const by = Object.fromEntries(vm.input.rows.map((r) => [r.name, r]));
    assert.equal(by.orn_food.state, by.orn_food.spiking > 0 ? 'delivered-spiking' : 'delivered-silent');
    assert.equal(by.jon_wind.state, by.jon_wind.spiking > 0 ? 'delivered-spiking' : 'delivered-silent');
    if (by.jon_wind.state === 'delivered-silent') assert.match(by.jon_wind.text, /delivered but not detected/);
    assert.equal(by.orn_danger.state, 'zero');
    assert.ok(vm.plume, 'nozzle_pos and filament_sigma published');
    assert.match(vm.caveats.join('\n'), /only above 3 mm\/s/);
    assert.equal(vm.wind.published.wind_speed, FX.wind_tunnel_connectome.stimuli.wind_speed);
});

test('real modular frame: no connectome input or response is shown, and it says why', () => {
    const vm = I.viewModel(FX.multisensory_modular);
    assert.equal(vm.input.available, false);
    assert.match(vm.input.text, /UNAVAILABLE · modular controller/);
    assert.equal(vm.response.available, false);
    assert.ok(vm.concentration.length > 0, 'the field values still show');
});

test('null current is UNAVAILABLE, never zero; RPC bridge odour/cVA/wind/thermo read NOT DELIVERED', () => {
    const pkt = clone(FX.wind_tunnel_connectome);
    pkt.connectome.input_stage[0].current_injected = null;
    pkt.connectome.input_stage[0].unavailable = 'no neurons resolved for this probe on this graph';
    const r0 = I.viewModel(pkt).input.rows[0];
    assert.equal(r0.state, 'unavailable'); assert.equal(r0.current, null);
    assert.match(r0.text, /no neurons resolved/);
    const bridge = clone(FX.wind_tunnel_connectome);
    bridge.identity.backend = I.BRIDGE_RPC;
    const vm = I.viewModel(bridge);
    for (const r of vm.input.rows) {
        if (['orn_food', 'orn_danger', 'courtship_cva', 'jon_wind', 'thermo'].includes(r.name)) assert.equal(r.state, 'not-delivered', r.name);
        else assert.notEqual(r.state, 'not-delivered', r.name);
    }
    assert.match(vm.caveats.join('\n'), /RPC bridge mode/);
});

test('antennal points are derived (2 mm, ±45°) and labelled so', () => {
    const a = I.antennaPoints({x: 10, y: 20, heading: 0});
    assert.ok(Math.abs(a.left[0] - (10 + Math.SQRT2)) < 1e-12 && Math.abs(a.left[1] - (20 + Math.SQRT2)) < 1e-12);
    assert.ok(Math.abs(a.right[1] - (20 - Math.SQRT2)) < 1e-12);
    assert.match(a.note, /derived from model constants/);
    assert.equal(I.antennaPoints({x: 1, y: 2}), null, 'no heading, no points');
});

test('illustrative plume is zero upwind, peaks on the axis and decays downwind', () => {
    const n = [180, 30], wind = [-25, 0];
    assert.equal(I.illustrativePlume(n, 3.5, wind, [190, 30]), 0);
    assert.ok(Math.abs(I.illustrativePlume(n, 3.5, wind, [180, 30]) - 1) < 1e-12);
    assert.ok(Math.abs(I.illustrativePlume(n, 3.5, wind, [130, 30]) - Math.exp(-50 / 250)) < 1e-12);
    assert.ok(I.illustrativePlume(n, 3.5, wind, [130, 33.5]) < I.illustrativePlume(n, 3.5, wind, [130, 30]));
    assert.equal(I.illustrativePlume(n, 3.5, [0, 0], [130, 30]), null, 'calm: nothing drawn');
    assert.match(I.PLUME_ASSUMPTIONS, /Not telemetry/);
});

test('predators: none live is said plainly; looming comes only from published theta', () => {
    const vm = I.viewModel(FX.multisensory_connectome);
    assert.match(vm.predators, /none live/);
    assert.equal(vm.looming, null);
    const loom = clone(FX.multisensory_connectome);
    loom.stimuli.theta_deg = 12; loom.stimuli.theta_rad = 0.2094;
    assert.equal(I.viewModel(loom).looming.theta_deg, 12);
    assert.ok(I.NOT_SIMULATED_LIST.some((s) => /Predator bodies.*never changes looming/.test(s)));
});

test('labels never name a chemical the engine does not simulate', () => {
    for (const s of I.SOURCE_KEYS) assert.ok(!/vinegar|acetic|ester|geosmin\b(?! as)|CO2|yeast/i.test(s.label), s.label);
    const html = read('env_inspector.html');
    assert.match(html, /arbitrary normalised units \(0–1\), not ppm or molar/);
});

test('graph I/O v3+ mapping: multisensory wind DELIVERED as reported, vector-only stays NOT DELIVERED', () => {
    const release = I.viewModel(FX.multisensory_connectome);
    assert.equal(release.io.v3Mapping, false, 'v2 telemetry has no v3 mapping');
    const pkt = clone(FX.multisensory_connectome);
    pkt.identity.graph_io = {version: 'graph-arena-io-v3-unassisted'};
    const jo = pkt.connectome.input_stage.find((r) => r.name === 'jon_wind');
    Object.assign(jo, {current_injected: 2.25, n_spiking_this_step: 0, stimulus_key: 'wind_magnitude', delivery: 'delivered'});
    const vm = I.viewModel(pkt);
    assert.equal(vm.io.v3Mapping, true);
    const row = vm.input.rows.find((r) => r.name === 'jon_wind');
    assert.equal(row.current, 2.25, 'the reported current, not computed');
    assert.equal(row.state, 'delivered-silent');
    assert.match(row.text, /reported: delivered \(key wind_magnitude\)/);
    assert.doesNotMatch(vm.caveats.join('\n'), /key mismatch/);
    const vec = clone(pkt);
    delete vec.stimuli.wind_magnitude;
    Object.assign(vec.connectome.input_stage.find((r) => r.name === 'jon_wind'), {current_injected: 0, stimulus_key: null,
        delivery: 'NOT DELIVERED: the assay publishes a wind vector but no scalar wind_speed/wind_magnitude; no vector-to-probe mapping is defined'});
    const v2 = I.viewModel(vec);
    assert.equal(v2.input.rows.find((r) => r.name === 'jon_wind').state, 'not-delivered');
    assert.match(v2.caveats.join('\n'), /vector-only wind/);
    assert.match(release.caveats.join('\n'), /NOT DELIVERED in this assay/);
});

test('label regression: the v3+ mapping is a feature label; candidate/deployed status only from explicit build provenance', () => {
    const pkt = clone(FX.multisensory_connectome);
    pkt.identity.graph_io = {version: 'graph-arena-io-v4-unassisted'};   // a future version
    Object.assign(pkt.connectome.input_stage.find((r) => r.name === 'jon_wind'), {delivery: 'delivered', stimulus_key: 'wind_magnitude'});
    const cav = I.viewModel(pkt).caveats.join('\n');
    assert.match(cav, new RegExp(I.IO_V3_LABEL.replace(/[+]/g, '\\+')));
    assert.match(cav, /Delivery is not detection/);
    assert.doesNotMatch(cav, /candidate|deployed|CANDIDATE|not deployed/i, 'no status claimed without provenance');
    assert.doesNotMatch(read('env_inspector.js'), /CANDIDATE BUILD|not deployed\]/);
    pkt.build_provenance = {status: 'candidate (unreleased)'};
    assert.match(I.viewModel(pkt).caveats.join('\n'), /Build status \(from build provenance\): candidate \(unreleased\)/);
});
