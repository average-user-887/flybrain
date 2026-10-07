'use strict';
const test = require('node:test'), assert = require('node:assert/strict');
const fs = require('node:fs'), path = require('node:path'), vm = require('node:vm');
const renderer = require('../web/observation_renderer.js');
const fixtures = JSON.parse(fs.readFileSync(path.join(__dirname, 'fixtures/observation_display_all14.json'))).fixtures;
const app = fs.readFileSync(path.join(__dirname, '../web/app.js'), 'utf8');
const clone = value => JSON.parse(JSON.stringify(value));
function harness() {
    const x = clone(fixtures[0]), listeners = {}, elements = {};
    for (const id of ['identFault', 'statAchieved', 'arenaRunState', 'btnPauseToggle', 'statDataAge', 'statStepAge'])
        elements[id] = {textContent: '', style: {}};
    const packet = {identity: x.identity, brain_id: x.brainId, segment_id: x.observation.segment_id,
        observation: x.observation, observation_publication: x.observationPublication,
        fly: {x: 3, y: 4, heading: .5}, step: 10, sim_time_s: .2};
    const arena = {remoteDriven: true, activeParadigmId: x.identity.assay, remotePacket: packet};
    const bridge = {arena, connected: true, replayMode: false, switchPending: false,
        activeUrl: 'http://test-only', lastOrderedPacket: packet, observationValidityUpdate: null,
        noteStep() {}, updateFreshness() {this.refreshes = (this.refreshes || 0) + 1;}, receive() {}};
    const c = {window: {NeuroFlyObservationRenderer: renderer, hud: {daemonBridge: bridge}},
        document: {getElementById: id => elements[id] || null}, performance: {now: () => 100}, EventSource: class {addEventListener(name, fn) {listeners[name] = fn;}}};
    vm.createContext(c);
    vm.runInContext(app.slice(app.indexOf('function identityRejection('), app.indexOf('window.neuroflyIdentityRejection ='))
        + app.slice(app.indexOf('class DaemonBridgeClient'), app.indexOf('const ASSAY_CONFIGS ='))
        + app.slice(app.indexOf('class ScientificBioArena'), app.indexOf('const EXPERIMENT_GUIDES'))
        + '\nthis.B=DaemonBridgeClient.prototype;this.A=ScientificBioArena.prototype;', c);
    for (const name of ['acceptsHeartbeatOwner', 'applySimProcessLiveness', 'applyDaemonHealth', 'currentHaltForPacket', 'renderCurrentHalt', 'renderPlaybackState', 'applyObservationValidityUpdate', 'reconcileObservationValidityUpdate', 'startStreaming']) bridge[name] = c.B[name];
    arena.getObservationDisplay = c.A.getObservationDisplay;
    bridge.startStreaming();
    const notice = {schema: 'neurofly-observation-validity-update/1', identity: {...x.identity, brain_id: x.brainId},
        segment_id: packet.segment_id, validity: 'invalidated', reason: 'unexpected_loop_exit'};
    function beat(update = notice) {return {identity: clone(update.identity), brain_id: update.identity.brain_id,
        segment_id: update.segment_id, step: packet.step, liveness: {state: 'dead'}, paused: false,
        result_validity: {run_id: update.identity.run_id, state: 'incomplete'}, observation_validity_update: update};}
    return {arena, bridge, packet, notice, beat, elements, prototype: c.B, send(data) {listeners.heartbeat({data: JSON.stringify(data)});}};
}
test('matching heartbeat invalidates the real display cache without mutating pose/frame/records/terminal', () => {
    const h = harness(), before = JSON.stringify(h.packet), display = h.arena.getObservationDisplay();
    assert.equal(display.currentValidity, 'valid');
    h.send(h.beat());
    const fault = h.arena.getObservationDisplay();
    assert.notEqual(fault, display); assert.equal(fault.currentValidity, 'invalidated');
    assert.match(fault.transportLabel, /Last pre-fault frame/);
    assert.match(renderer.primary(fault).sub, /invalidated/); assert.equal(renderer.primary(fault).rawValue, null);
    assert.deepEqual(fault.live.records, display.live.records); assert.deepEqual(fault.terminal, display.terminal);
    assert.equal(JSON.stringify(h.packet), before);
    h.notice.identity.run_id = 'caller-corruption';
    assert.equal(h.arena.getObservationDisplay().currentValidity, 'invalidated');
});
test('exploratory validity stays degraded while the recorded values remain unchanged', () => {
    const h = harness(); h.notice.validity = 'exploratory_degraded';
    const records = h.arena.getObservationDisplay().live.records;
    h.send(h.beat()); const display = h.arena.getObservationDisplay();
    assert.equal(display.currentValidity, 'exploratory_degraded'); assert.deepEqual(display.live.records, records);
});
test('delayed heartbeat cannot alter any current health or metrics after owner or segment changes', () => {
    for (const field of ['daemon_run_id', 'run_id', 'instance_id', 'activation', 'assay', 'backend', 'controller_version', 'brain_id', 'synthetic', 'test_mode', 'segment_id']) {
        const h = harness(), stale = h.beat();
        if (field === 'segment_id') stale.segment_id = 'old-segment';
        else if (field === 'brain_id') stale.brain_id = 'old-brain';
        else stale.identity[field] = typeof stale.identity[field] === 'number' ? stale.identity[field] + 1 : 'old-owner';
        h.bridge.lastHeartbeat = {sentinel: 'current'}; h.bridge.daemonValidity = {sentinel: 'current'};
        h.send(stale);
        assert.equal(h.bridge.observationValidityUpdate, null, field);
        assert.equal(h.bridge.lastHeartbeat.sentinel, 'current', field);
        assert.equal(h.bridge.daemonValidity.sentinel, 'current', field);
        assert.equal(h.arena.getObservationDisplay().currentValidity, 'valid', field);
    }
});
test('ACK-before-frame, pending switch and replay fence matching old fault updates', () => {
    for (const field of ['lastSwitchAck', 'lastBackendAck', 'lastAck', 'switchPending', 'replayMode']) {
        const h = harness();
        h.bridge[field] = field.endsWith('Ack') ? {identity: {...h.packet.identity, activation: h.packet.identity.activation + 1}} : true;
        assert.equal(h.bridge.applyObservationValidityUpdate(h.notice), false, field);
        h.send(h.beat()); assert.equal(h.bridge.observationValidityUpdate, null, field);
    }
});
test('fresh recovered owner clears the overlay; a delayed previous-owner notice cannot reacquire it', () => {
    const h = harness(); h.send(h.beat());
    const recovered = clone(h.packet);
    recovered.identity.activation++; recovered.identity.run_id = 'recovered-run'; recovered.identity.instance_id = 'recovered-instance';
    recovered.observation.identity = {...recovered.identity, brain_id: recovered.brain_id};
    h.bridge.lastOrderedPacket = h.arena.remotePacket = recovered;
    h.bridge.reconcileObservationValidityUpdate(recovered);
    assert.equal(h.bridge.observationValidityUpdate, null);
    assert.equal(h.arena.getObservationDisplay().currentValidity, 'valid');
    h.send(h.beat()); assert.equal(h.arena.getObservationDisplay().currentValidity, 'valid');
});
test('fault updates may only downgrade a complete matching owner, never promote validity', () => {
    for (const edit of [x => x.validity = 'valid', x => x.reason = 'unqualified', x => delete x.identity.instance_id]) {
        const h = harness(), notice = clone(h.notice); edit(notice);
        assert.equal(h.bridge.applyObservationValidityUpdate(notice), false);
    }
    const h = harness(), conflicting = h.beat(); conflicting.identity.brain_id = 'conflicting-owner';
    h.send(conflicting); assert.equal(h.bridge.observationValidityUpdate, null);
});

test('captured publish halt overrides a stale paused frame without refreshing pose or data age', () => {
    const h = harness();
    const status = JSON.parse(fs.readFileSync(path.join(__dirname, 'fixtures/publish_halt_health.json')));
    h.packet.paused = true; h.packet.step = 656; h.packet.fly.state = 'still';
    h.bridge.lastValidDataTime = 1; h.bridge.lastStepSeen = 656; h.bridge.lastStepAdvanceTime = 1;
    h.bridge.staleAfterMs = 5; h.bridge.statusPill = {textContent: '', title: '', style: {}};
    h.bridge.updateFreshness = h.prototype.updateFreshness;
    for (const name of ['stepAgeSeconds', 'slowStepSeconds', 'notAdvancing', 'updatePersistenceBanner'])
        h.bridge[name] = h.prototype[name];
    const before = JSON.stringify(h.packet);
    h.send({...status, identity: {...h.packet.identity, brain_id: h.packet.brain_id},
        brain_id: h.packet.brain_id, segment_id: h.packet.segment_id,
        result_validity: {run_id: h.packet.identity.run_id, state: 'incomplete'}});
    assert.match(h.bridge.statusPill.textContent, /SIMULATION HALTED.*ERROR/);
    assert.match(h.elements.arenaRunState.textContent, /SIMULATION ERROR:.*publication failed/);
    assert.equal(h.elements.btnPauseToggle.textContent, 'Halted');
    assert.equal(h.elements.btnPauseToggle.disabled, true);
    assert.equal(h.elements.statAchieved.textContent, '0x · halted');
    assert.match(h.elements.identFault.textContent, /publication failed/);
    assert.equal(h.bridge.lastValidDataTime, 1);
    assert.equal(JSON.stringify(h.packet), before);
    assert.equal(h.bridge.daemonPaused, false);
    assert.ok(h.bridge.currentHaltForPacket(h.packet));
    assert.ok(h.bridge.currentHaltForPacket({...h.packet, step: status.step}));
    assert.equal(h.bridge.currentHaltForPacket({...h.packet, step: status.step + 1}), false);
});
test('halt health is fenced from replay and recovery clears the pause-control halt', () => {
    const h = harness(), b = h.beat();
    b.halted = true; b.error = 'publish failed'; b.liveness.state = 'halted';
    h.bridge.replayMode = true; h.send(b);
    assert.equal(h.bridge.daemonHalt, undefined);
    h.bridge.replayMode = false; h.send(b);
    assert.equal(h.elements.btnPauseToggle.disabled, true);
    b.halted = false; b.error = null; b.liveness.state = 'paused'; b.paused = true;
    h.send(b); assert.equal(h.bridge.daemonHalt, null);
    assert.equal(h.elements.btnPauseToggle.disabled, false);
    assert.equal(h.elements.btnPauseToggle.textContent, 'Resume');
});
