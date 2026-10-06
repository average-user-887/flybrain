'use strict';
const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const app = fs.readFileSync(path.resolve(__dirname, '../web/app.js'), 'utf8');
const renderer = require('../web/observation_renderer.js');
const context = {window:{NeuroFlyObservationRenderer:renderer, hud:{daemonBridge:{connected:true}}}};
vm.createContext(context);
vm.runInContext([
    app.slice(app.indexOf('const ARENA_TOOL_IDS ='), app.indexOf('function identityRejection(')),
    app.slice(app.indexOf('function sandboxScoreView('), app.indexOf('class ScientificBioArena')),
    app.slice(app.indexOf('class ScientificBioArena'), app.indexOf('const EXPERIMENT_GUIDES')),
    app.slice(app.indexOf('const ASSAY_CONFIGS ='), app.indexOf('const LESION_INFO =')),
    app.slice(app.indexOf('class ScientificHUD'), app.indexOf('// 9. APPLICATION INITIALIZATION')),
    'globalThis.hudUpdate = ScientificHUD.prototype.update;',
    'globalThis.cardMetricUpdate = ScientificHUD.prototype.updateCardMetric;',
    'globalThis.backendReconcile = ScientificHUD.prototype.reconcileBackendSelector;',
    'globalThis.toolReconcile = ScientificHUD.prototype.reconcileToolCapabilities;',
    'globalThis.canonical = ScientificBioArena.prototype.getCanonicalMetricInfo;',
    'globalThis.rawTrial = ScientificBioArena.prototype.getRawTrialMetric;',
    'globalThis.observation = ScientificBioArena.prototype.getObservationDisplay;',
    'globalThis.preview = ScientificBioArena.prototype.getPreviewMetricInfo;',
    "globalThis.guide = ASSAY_CONFIGS['multisensory-sandbox'];",
].join('\n'), context);
const local = {compositeScore:88.5, coordinationScore:.94, sensoryIntegrationScore:.88,
    efficiencyScore:.82, smoothnessScore:.91};
const measured = {composite_benchmark_score:73.1, locomotor_coordination_index:1,
    multisensory_integration_score:.924, biomechanical_efficiency:0, kinematic_smoothness:1};
function remote(metrics = measured) {
    return {activeParadigmId:'multisensory-sandbox', remoteDriven:true, awaitingDaemon:false,
        remotePacket:{paradigm:'multisensory-sandbox', run_id:'sandbox-run', identity:{run_id:'sandbox-run'}, metrics},
        paradigmState:{...local}, getObservationDisplay:context.observation, getPreviewMetricInfo:context.preview,
        getCanonicalMetricInfo:context.canonical};
}
function elements() {
    const ids = ['deckCompositeScore', 'deckCoordScore', 'deckSensoryScore', 'deckEfficacyScore',
        'deckSmoothScore', 'cardMetricMultisensory', 'deckScorecardSource'];
    const els = Object.fromEntries(ids.map(id => [id, {textContent:id==='cardMetricMultisensory'?'Not selected':'old', title:''}]));
    return {els, doc:{getElementById:id=>els[id] || null, querySelectorAll:()=>[els.cardMetricMultisensory]}};
}

test('sandbox → T-maze clears every deck/catalog scalar, retaining no old text', () => {
    const arena = remote();
    const {els, doc} = elements();
    context.renderSandboxScorecard(arena, doc);
    assert.equal(els.deckCompositeScore.textContent, 'Unavailable');
    assert.equal(els.deckCoordScore.textContent, 'Unavailable');
    assert.equal(els.deckSensoryScore.textContent, 'Unavailable');
    arena.activeParadigmId = 't-maze';
    // Even a retained sandbox packet/state cannot supply another assay's score.
    context.renderSandboxScorecard(arena, doc);
    for (const [id, el] of Object.entries(els)) {
        if (id === 'cardMetricMultisensory') assert.equal(el.textContent,'Not selected');
        else if (id !== 'deckScorecardSource') assert.equal(el.textContent, 'Unavailable', id);
    }
    assert.match(els.deckScorecardSource.textContent, /unavailable.*Waiting for selected owner/);
    assert.equal(context.guide.metrics[0].get(arena), 'Unavailable');
    assert.equal(context.guide.metrics[1].get(arena), 'Unavailable');
    assert.equal(context.guide.metrics[2].get(arena), 'Unavailable');
});

test('current packet owns the main metric, guide and scorecard despite stale mirrored state', () => {
    const arena = remote();
    const canonical = context.canonical.call(arena);
    assert.equal(canonical.rawValue, null);
    assert.equal(canonical.value, 'Unavailable');
    assert.match(canonical.label, /Scientific composite.*live provisional/);
    assert.equal(context.rawTrial.call(arena), null); // legacy proxy is never scientific evidence
    assert.equal(context.guide.metrics[0].get(arena), '73.1 / 100 · body proxy');
    assert.equal(context.guide.metrics[1].get(arena), '100.0%');
    assert.equal(context.guide.metrics[2].get(arena), '92.4%');
});

for (const invalid of [undefined, null, NaN, Infinity, '0', false, {}, []]) {
    test(`missing/malformed packet metric ${String(invalid)} never falls back to local values`, () => {
        const metrics = Object.fromEntries(Object.keys(measured).map(k=>[k,invalid]));
        const arena = remote(metrics);
        const score = context.sandboxScoreView(arena);
        for (const value of Object.values(score.text)) assert.equal(value, 'Unavailable');
        assert.equal(context.canonical.call(arena).rawValue, null);
        assert.equal(context.canonical.call(arena).value, 'Unavailable');
        assert.equal(context.rawTrial.call(arena), null);
        for (const scalar of context.guide.metrics) assert.equal(scalar.get(arena), 'Unavailable');
    });
}

test('later partial/absent packet clears previously available fields independently', () => {
    const arena = remote();
    const {els, doc} = elements();
    context.renderSandboxScorecard(arena, doc);
    arena.remotePacket = {...arena.remotePacket, metrics:{composite_benchmark_score:0}};
    context.renderSandboxScorecard(arena, doc);
    assert.equal(els.deckCompositeScore.textContent, 'Unavailable');
    for (const id of ['deckCoordScore','deckSensoryScore','deckEfficacyScore','deckSmoothScore']) {
        assert.equal(els[id].textContent, 'Unavailable');
    }
    delete arena.remotePacket.metrics;
    context.renderSandboxScorecard(arena, doc);
    assert.equal(els.deckCompositeScore.textContent, 'Unavailable');
    assert.equal(els.cardMetricMultisensory.textContent, 'Not selected');
});

test('genuine numeric zeros remain zero in every consumer', () => {
    const arena = remote(Object.fromEntries(Object.keys(measured).map(k=>[k,0])));
    const score = context.sandboxScoreView(arena);
    assert.equal(score.text.composite, '0.0 / 100');
    for (const key of ['coordination','sensory','efficiency','smoothness']) assert.equal(score.text[key], '0.0%');
    assert.equal(context.rawTrial.call(arena), null);
    assert.equal(context.canonical.call(arena).rawValue, null);
    assert.equal(context.guide.metrics[1].get(arena), '0.0%');
});

for (const state of ['waiting', 'mismatch', 'invalid-pose-mode', 'no-packet']) {
    test(`${state} cannot borrow a previous packet or local initialization`, () => {
        const arena = remote();
        if (state === 'waiting') arena.awaitingDaemon = true;
        if (state === 'mismatch') arena.remotePacket.paradigm = 't-maze';
        if (state === 'invalid-pose-mode') arena.remoteDriven = false;
        if (state === 'no-packet') arena.remotePacket = null;
        const score = context.sandboxScoreView(arena);
        for (const value of Object.values(score.text)) assert.equal(value, 'Unavailable');
    });
}

test('local preview is preserved with visible preview/proxy provenance and null semantics', () => {
    const arena = {activeParadigmId:'multisensory-sandbox', paradigmState:{...local}, getPreviewMetricInfo:context.preview};
    const {els, doc} = elements();
    context.renderSandboxScorecard(arena, doc);
    assert.equal(els.deckCompositeScore.textContent, '88.5 / 100');
    assert.equal(els.cardMetricMultisensory.textContent, 'Not selected');
    assert.match(els.deckScorecardSource.textContent, /Local preview.*heuristic body proxy/);
    assert.match(context.canonical.call(arena).label, /Local preview/);
    arena.paradigmState.compositeScore = null;
    arena.paradigmState.coordinationScore = 0;
    assert.equal(context.sandboxScoreView(arena).text.composite, 'Unavailable');
    assert.equal(context.guide.metrics[1].get(arena), '0.0%');
});

test('recordings carry recording provenance', () => {
    const arena = remote();
    arena.remotePacket.timing = {replay:true};
    assert.match(context.canonical.call(arena).label, /replay provisional/);
    assert.match(context.sandboxScoreView(arena).note, /Recording/);
});

test('guide radar uses current available values and draws no invented partial polygon', () => {
    const arena = remote();
    const points = [], labels = [];
    const ctx = {beginPath(){}, moveTo(...p){points.push(p);}, lineTo(...p){points.push(p);},
        stroke(){}, closePath(){}, fill(){}, fillText(t){labels.push(t);}};
    context.guide.drawChart(ctx, 200, 100, arena);
    assert.deepEqual(points.slice(-4), [[100,22],[133.264,58],[100,58],[64,58]]);
    points.length = 0;
    arena.remotePacket.metrics = {};
    context.guide.drawChart(ctx, 200, 100, arena);
    assert.equal(points.length, 0);
    assert.match(labels.at(-1), /Unavailable/);
});

test('actual HUD update clears the scorecard and catalog on the reproduced switch', () => {
    const arena = Object.assign(remote(), {simTime:1, paradigmElapsedSec:1, stepCount:50,
        currentTrial:1, mb:{kcFiring:[]}, windVector:[0,0], dn:{}, cpg:{legStates:{}}});
    arena.getCanonicalMetricInfo = function() {return context.canonical.call(this);};
    const {els, doc} = elements();
    context.document = doc;
    context.graphPanelCapabilities = () => ({graph:true, modularMemory:false});
    const hud = {arena, scopeHistory:[], updateCardMetric:context.cardMetricUpdate, reconcileBackendSelector:context.backendReconcile, reconcileToolCapabilities:context.toolReconcile};
    for (const method of ['renderKcMatrix','renderLearningCurve','renderCompass','renderOscilloscope',
                          'updateAssayTools','updatePremotorHUD']) hud[method] = () => {};
    context.hudUpdate.call(hud);
    assert.equal(els.deckCompositeScore.textContent, 'Unavailable');
    arena.activeParadigmId = 't-maze';
    arena.remotePacket = {paradigm:'t-maze', metrics:{performance_index:0}};
    context.hudUpdate.call(hud);
    for (const [id, el] of Object.entries(els)) {
        if (id === 'cardMetricMultisensory') assert.equal(el.textContent,'Not selected');
        else if (id !== 'deckScorecardSource') assert.equal(el.textContent, 'Unavailable', id);
    }
});
