'use strict';
// CARD71: the Training & Data weight tile's caption and accessible name follow the data actually
// drawn. Modular wording ("daemon's saved weights", heatmap) never appears for a graph run.
const test = require('node:test'), assert = require('node:assert/strict'), fs = require('node:fs'), vm = require('node:vm'), path = require('node:path');
const source = fs.readFileSync(path.join(__dirname, '../web/training.js'), 'utf8');
const app = fs.readFileSync(path.join(__dirname, '../web/app.js'), 'utf8');
const html = fs.readFileSync(path.join(__dirname, '../web/index.html'), 'utf8');
const identityHelper = app.slice(app.indexOf('function identityRejection('), app.indexOf('window.neuroflyIdentityRejection = identityRejection;') + 'window.neuroflyIdentityRejection = identityRejection;'.length);
const capsHelper = app.slice(app.indexOf('const GRAPH_BACKENDS ='), app.indexOf('window.neuroflyGraphPanelCapabilities = graphPanelCapabilities;') + 'window.neuroflyGraphPanelCapabilities = graphPanelCapabilities;'.length);

function harness(backend, extraTelemetry = {}) {
    const elements = {};
    const identity = {daemon_run_id: 'daemon-a', activation: 1, run_id: 'run-a', instance_id: 'instance-a', assay: 'open-arena', backend,
        label: backend === 'modular' ? 'modular' : 'Full prepared MaleCNS graph, fixed weights, LIF proxy dynamics.'};
    const bridge = {connected: true, activeUrl: 'http://intended', replayMode: false, readOnly: false,
        lastOrderedPacket: {identity, brain_id: 'brain-a', run_id: 'daemon-a'}, sendCommand: async () => ({status: 'ok', ack: {applied: true}})};
    const hud = {daemonBridge: bridge, arena: {activeParadigmId: 'open-arena'}};
    function el(id) { return elements[id] ||= {style: {}, attrs: {}, textContent: '', innerHTML: '', firstChild: {textContent: ''}, classList: {toggle() {}},
        addEventListener() {}, setAttribute(k, v) { this.attrs[k] = v; }, removeAttribute(k) { delete this.attrs[k]; }}; }
    const weights = Array.from({length: 120}, () => [1, 1]);
    const data = () => ({brain: {brain_id: 'brain-a', paradigm: 'open-arena', seed: 1, steps: 4, trials: 0, learning_enabled: false, probe: {discrimination: 0},
        weight_change_l2: 0, history: [], weights}, telemetry: {identity: {...identity}, brain_id: 'brain-a', ...extraTelemetry}, brains: [], status: {stream: {}, sim_speed: 1}});
    const c = {window: {app: {hud}, addEventListener() {}}, document: {getElementById: el, createElement() { return {click() {}}; }}, AbortSignal: {timeout: () => ({})},
        fetch: async () => ({ok: true, json: async () => data()}), Blob, URL: {createObjectURL: () => '', revokeObjectURL() {}}, setTimeout() {}, setInterval() {}, Date, console};
    vm.createContext(c);
    vm.runInContext(identityHelper, c);
    vm.runInContext(capsHelper, c);
    vm.runInContext(source.replace(/\}\)\(\);\s*$/, 'window.testOwner={refresh,reconcileTrainingOwner};})();'), c);
    return {el, api: c.window.testOwner, bridge};
}
const MODULAR_WORDING = /daemon’s saved weights|daemon's saved weights|heatmap|Hover for exact weights/i;

test('static markup claims no weights before any owner data', () => {
    assert.match(html, /<svg id="trainingWeights"[^>]*aria-label="Weight readout not loaded"/);
    assert.match(html, /<p id="trainingWeightsCaption">No weight readout is loaded for the displayed owner yet\.<\/p>/);
    assert.doesNotMatch(html, /Saved mushroom body weight heatmap|These are the daemon’s saved weights/);
});

test('fixed graph without a measured subset: truthful empty caption and label, no modular leak', async () => {
    const h = harness('connectome-fixed');
    for (let frame = 0; frame < 3; frame++) {
        await h.api.refresh();
        assert.equal(h.el('trainingWeightsTitle').textContent, 'Graph plasticity readout unavailable');
        assert.equal(h.el('trainingWeights').attrs['aria-label'], 'No graph weight readout available');
        assert.equal(h.el('trainingWeightsCaption').textContent,
            'No saved weight readout is available: the current telemetry packet declares no measured plastic subset for Full prepared MaleCNS graph, fixed weights, LIF proxy dynamics. The 120-KC modular weights are not shown because they do not belong to this graph run.');
        assert.doesNotMatch(h.el('trainingWeights').attrs['aria-label'] + h.el('trainingWeightsCaption').textContent, MODULAR_WORDING);
        assert.doesNotMatch(h.el('trainingWeights').innerHTML, /<rect/);
    }
});

test('measured WP6 subset: summary only, no heatmap claimed', async () => {
    const h = harness('connectome-plastic', {plasticity: {wp6: {n_edges: 3081, mean_delta: 0, max_delta: 0}}});
    await h.api.refresh();
    assert.equal(h.el('trainingWeightsTitle').textContent, 'Measured WP6 ER→EPG subset');
    assert.equal(h.el('trainingWeights').attrs['aria-label'], 'Measured WP6 ER→EPG subset summary: edge count and mean weight change');
    assert.match(h.el('trainingWeightsCaption').textContent, /No per-synapse heatmap is drawn\.$/);
    assert.doesNotMatch(h.el('trainingWeightsCaption').textContent, /daemon’s saved weights|learning (is )?supported/i);
});

test('modular controller keeps its real heatmap caption, labelled as the 120-KC modular body', async () => {
    const h = harness('modular');
    await h.api.refresh();
    assert.match(h.el('trainingWeights').innerHTML, /<rect/);
    assert.equal(h.el('trainingWeights').attrs['aria-label'], 'Saved weight heatmap of the 120-KC modular mushroom body');
    assert.match(h.el('trainingWeightsCaption').textContent, /saved weights of the 120-KC modular mushroom body/);
});

test('switching modular → graph replaces the modular caption; losing the owner clears it', async () => {
    const h = harness('modular');
    await h.api.refresh();
    assert.match(h.el('trainingWeightsCaption').textContent, /120-KC modular/);
    h.bridge.replayMode = true;
    h.api.reconcileTrainingOwner();
    assert.equal(h.el('trainingWeights').attrs['aria-label'], 'Weight readout unavailable');
    assert.equal(h.el('trainingWeightsCaption').textContent, 'No weight readout is available for the displayed owner.');
    const g = harness('connectome-fixed');
    await g.api.refresh();
    assert.doesNotMatch(g.el('trainingWeightsCaption').textContent, MODULAR_WORDING);
});
