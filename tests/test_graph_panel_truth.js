'use strict';

const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');

const root = path.resolve(__dirname, '..');
const app = fs.readFileSync(path.join(root, 'web/app.js'), 'utf8');
const training = fs.readFileSync(path.join(root, 'web/training.js'), 'utf8');
const html = fs.readFileSync(path.join(root, 'web/index.html'), 'utf8');

const start = app.indexOf("const GRAPH_BACKENDS =");
const end = app.indexOf('window.neuroflyGraphDnReadout = graphDnReadout;')
    + 'window.neuroflyGraphDnReadout = graphDnReadout;'.length;
assert.notEqual(start, -1);
assert.ok(end > start);
const context = {window:{}};
vm.createContext(context);
vm.runInContext(app.slice(start, end), context);
const capabilities = context.window.neuroflyGraphPanelCapabilities;
const graphDnReadout = context.window.neuroflyGraphDnReadout;

const trainingStart = training.indexOf('const trainingReplayActive =');
const trainingEnd = training.indexOf('window.neuroflyTrainingWriteAllowed = trainingWriteAllowed;')
    + 'window.neuroflyTrainingWriteAllowed = trainingWriteAllowed;'.length;
assert.notEqual(trainingStart, -1);
assert.ok(trainingEnd > trainingStart);
const trainingContext = {window:{}};
vm.createContext(trainingContext);
vm.runInContext(training.slice(trainingStart, trainingEnd), trainingContext);
const trainingWriteAllowed = trainingContext.window.neuroflyTrainingWriteAllowed;

test('fixed graph refuses modular memory actions while checkpoint remains valid', () => {
    const got = capabilities({identity:{backend:'connectome-fixed', label:'MaleCNS fixed'},
        neural:{kc_hz:Array(120).fill(99), net_valence:42}});
    assert.equal(got.graph, true);
    assert.equal(got.modularMemory, false);
    assert.equal(got.teach, false);
    assert.equal(got.reverse, false);
    assert.equal(got.probe, false);
    assert.equal(got.learningControl, false);
    assert.equal(got.saveCheckpoint, true);
    assert.match(got.modularReason, /not the controller/);
});

test('plastic backend name alone does not invent learning; measured zero remains available', () => {
    assert.equal(capabilities({identity:{backend:'connectome-plastic'}}).learningControl, false);
    const measured = capabilities({identity:{backend:'connectome-plastic'},
        plasticity:{wp6:{n_edges:3081, mean_delta:0, max_delta:0}}});
    assert.equal(measured.wp6Measured, true);
    assert.equal(measured.learningControl, true);
    for (const backend of ['connectome-fixed', 'connectome-with-trained-readout']) {
        const contradictory = capabilities({identity:{backend},
            plasticity:{wp6:{n_edges:1, mean_delta:0, max_delta:0}}});
        assert.equal(contradictory.wp6Measured, false);
        assert.equal(contradictory.learningControl, false);
    }
});

test('EPG availability requires complete finite streamed evidence and preserves zero', () => {
    const base = {identity:{backend:'connectome-fixed'}, connectome:{epg_bump_phase:0}};
    assert.equal(capabilities(base).epgMeasured, false);
    assert.equal(capabilities({...base, neural:{epg_wedges:Array(16).fill(0)}}).epgMeasured, false);
    const resolved = {...base, connectome:{...base.connectome, epg_available:true, epg_resolved_count:12},
        neural:{epg_wedges:Array(16).fill(0)}};
    assert.equal(capabilities(resolved).epgMeasured, true);
    assert.equal(capabilities({...resolved, neural:{epg_wedges:[...Array(15).fill(0), null]}}).epgMeasured, false);
    assert.equal(capabilities({...resolved, connectome:{...resolved.connectome, epg_available:false}}).epgMeasured, false);
});

test('graph DN readout never promotes top-level pose-derived values', () => {
    const top = {dna02_l:8, dna02_r:9, dnp09:25, mdn:0, gf:0};
    const missing = graphDnReadout({identity:{backend:'connectome-fixed'}, dn_rates:top});
    assert.deepEqual(Object.keys(missing.rates), []);
    assert.match(missing.missingReason, /graph-controller/);
    const nested = {dna02_l:0, dna02_r:0, dnp09:0, mdn:0, gf:0};
    const resolved = graphDnReadout({identity:{backend:'connectome-fixed'}, dn_rates:top,
        connectome:{dn_rates:nested, dn_unavailable:{}}});
    assert.equal(resolved.rates.dnp09, 0);
    assert.equal(resolved.missingReason, null);
});

test('replay is read only independently of daemon stream policy', () => {
    assert.equal(trainingWriteAllowed({read_only:false, commands_require_token:false}, false), true);
    assert.equal(trainingWriteAllowed({read_only:false, commands_require_token:false}, true), false);
    assert.match(training, /if\(trainingReplayActive\(\)\)\{[\s\S]*Replay is read only/);
});

test('modular controller retains its memory actions and panel semantics', () => {
    const got = capabilities({identity:{backend:'modular', label:'modular'}});
    assert.equal(got.modularMemory, true);
    assert.equal(got.teach && got.reverse && got.probe && got.learningControl, true);
});

test('training and markup expose persistent capability explanations', () => {
    assert.match(training, /supported=\{teach_brain:caps\.teach,probe_brain:caps\.probe,set_learning:caps\.learningControl/);
    assert.match(training, /Checkpoint save remains available/);
    for (const id of ['mushroomBodyPanelReason', 'mushroomBodyPanelTitle', 'gaitPanelTitle',
                      'trainingDeltaLabel', 'trainingDiscriminationLabel', 'trainingWeightsTitle']) {
        assert.match(html, new RegExp(`id="${id}"`));
    }
});

test('graph training copy uses packet evidence and clears modular cohort rows immediately', () => {
    const start = training.indexOf('function renderGraph(n,caps)');
    const end = training.indexOf('function renderResearch()', start);
    assert.ok(start >= 0 && end > start);
    const renderGraph = training.slice(start, end);
    assert.match(renderGraph, /only graph-controller measurements declared in the current telemetry packet/);
    assert.match(renderGraph, /observed arena outcome and graph packet channels as separate evidence/);
    assert.doesNotMatch(renderGraph, /n\[3\]|n\[4\]/);
    assert.match(renderGraph, /\$\('researchRows'\)\.innerHTML=''/);
});
