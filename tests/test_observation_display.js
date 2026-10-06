'use strict';
const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const {buildObservationDisplay: display, ASSAYS} = require('../web/observation_display.js');
const generated = JSON.parse(fs.readFileSync(process.env.NEUROFLY_DISPLAY_FIXTURES || path.join(__dirname, 'fixtures/observation_display_all14.json')));
const fixtures = generated.fixtures;
const clone = x => JSON.parse(JSON.stringify(x));
const base = () => clone(fixtures.find(x => x.identity.assay === 'gap-crossing'));
function rejectLive(change) { const x = base(); change(x); assert.equal(display(x).live.state, 'unavailable'); }
function rejectTerminal(change) { const x = base(); change(x); assert.equal(display(x).terminal.state, 'unavailable'); }

test('browser global API loads without DOM or external dependencies', () => {
    const context = vm.createContext({});
    for (const name of ['metric_records', 'observation_display']) vm.runInContext(fs.readFileSync(path.join(__dirname, `../web/${name}.js`), 'utf8'), context);
    assert.equal(typeof context.NeuroFlyObservationDisplay.buildObservationDisplay, 'function');
    vm.runInContext(`const input = JSON.parse(${JSON.stringify(JSON.stringify(fixtures[0]))});
        if (NeuroFlyObservationDisplay.buildObservationDisplay(input).live.state !== 'available') throw Error('browser rejection');`, context);
});
test('fixtures cover exactly all fourteen declared assay identities', () => {
    assert.deepEqual(fixtures.map(x => x.identity.assay).sort(), Object.keys(ASSAYS).sort());
});
for (const fixture of fixtures) test(`Python producer and actual ledger compatibility: ${fixture.identity.assay}`, () => {
    const before = JSON.stringify(fixture); const result = display(fixture);
    assert.equal(result.live.state, 'available', result.live.error);
    assert.equal(result.terminal.state, 'available', result.terminal.error);
    assert.equal(result.live.provisional, true); assert.equal(result.live.frozen, false);
    assert.equal(result.terminal.provisional, false); assert.equal(result.terminal.frozen, true);
    assert.equal(result.terminal.historical, false);
    assert.equal(result.terminal.receiptCheck, 'consistent_transport_only');
    assert.equal(result.terminal.completeness, 'incomplete');
    assert.equal(result.terminal.endReason, 'manual_reset');
    assert.equal(JSON.stringify(fixture), before);
    const name = Object.keys(result.live.records)[0];
    result.live.records[name].counts.changed = true;
    assert.equal(Object.values(fixture.observation.records).some(r => r.counts.changed), false);
});
test('real zero, false and unavailable records remain distinct across generated assays', () => {
    const records = fixtures.flatMap(x => Object.values(display(x).live.records));
    const deadline = display(generated.deadlineFixture).terminal;
    assert.equal(deadline.state, 'available', deadline.error);
    assert.equal(deadline.completeness, 'complete');
    records.push(...Object.values(deadline.records));
    assert.ok(records.some(r => r.available && r.rawValue === 0 && r.valueText === '0'));
    assert.ok(records.some(r => r.available && r.rawValue === false && r.text === 'false'));
    assert.ok(records.some(r => !r.available && r.rawValue === null && r.valueText === null));
});
test('same-owner older daemon/run/activation terminal is labelled historical without rewriting bytes', () => {
    const x = base(); const terminal = JSON.stringify(x.observationPublication.last_terminal);
    x.identity.run_id = 'new-run'; x.identity.daemon_run_id = 'new-daemon'; x.identity.activation++;
    x.observation.identity = clone(x.identity);
    const r = display(x); assert.equal(r.live.state, 'available'); assert.equal(r.terminal.historical, true);
    assert.equal(r.terminal.identity.run_id, 'run-gap_crossing');
    assert.equal(JSON.stringify(x.observationPublication.last_terminal), terminal);
});
test('same-run older segment remains historical', () => {
    const x = base(); x.observation.segment_id = 'new-segment'; x.observation.presentation_id = 'new-segment:0';
    assert.equal(display(x).terminal.historical, true);
});
test('current fault validity does not invalidate saved earlier terminal', () => {
    const x = base(); x.observation.validity = 'invalidated';
    const r = display(x); assert.equal(r.currentValidity, 'invalidated'); assert.equal(r.terminal.validity, 'valid');
});
test('degraded and incomplete saved observations retain independent distinctions', () => {
    const x = base(); x.observation.validity = 'exploratory_degraded';
    x.observationPublication.last_terminal.observation.validity = 'exploratory_degraded';
    const r = display(x); assert.equal(r.live.validity, 'exploratory_degraded');
    assert.equal(r.terminal.validity, 'exploratory_degraded'); assert.equal(r.terminal.completeness, 'incomplete');
});
for (const key of ['daemon_run_id', 'run_id', 'instance_id', 'activation', 'assay', 'backend', 'controller_version', 'brain_id']) {
    test(`rejects stale live identity field ${key}`, () => rejectLive(x => {
        x.observation.identity[key] = key === 'activation' ? 8 : `different-${key}`;
    }));
}
for (const key of ['assay', 'instance_id', 'backend', 'brain_id']) test(`different terminal owner ${key} unavailable`, () => {
    const x = base(); x.identity[key] = key === 'assay' ? 't-maze' : `different-${key}`;
    if (key === 'brain_id') x.brainId = x.identity.brain_id;
    assert.equal(display(x).terminal.state, 'unavailable');
});
test('explicit current brain ID mismatch rejects both sources', () => {
    const x = base(); x.brainId = 'other'; assert.equal(display(x).state, 'unavailable');
});
for (const field of ['daemon_run_id', 'run_id', 'instance_id', 'segment_id', 'presentation_id']) {
    test(`full publication key field ${field} required`, () => rejectTerminal(x => { delete x.observationPublication.last_terminal.observation_key[field]; }));
    test(`receipt key field ${field} agrees`, () => rejectTerminal(x => { x.observationPublication.last_terminal.receipt.observation_key[field] = 'wrong'; }));
}
for (const [label, change] of [
    ['wrapper identity', x => { x.identity.activation++; }],
    ['digest mismatch', x => { x.payload_sha256 = 'f'.repeat(64); }],
    ['missing digest', x => { delete x.payload_sha256; }],
    ['noncanonical digest', x => { x.payload_sha256 = 'bad'; }],
    ['missing receipt', x => { delete x.receipt; }],
    ['not durable', x => { x.receipt.durable = false; }],
    ['coerced durable', x => { x.receipt.durable = 1; }],
    ['receipt line', x => { x.receipt.line = false; }],
    ['receipt offset', x => { x.receipt.offset = -1; }],
    ['receipt idempotent', x => { x.receipt.idempotent = null; }],
    ['receipt directory', x => { x.receipt.file = '../trials.jsonl'; }],
]) test(`rejects inconsistent terminal ${label}`, () => rejectTerminal(x => change(x.observationPublication.last_terminal)));

test('pending queue candidates never substitute for a saved terminal', () => {
    const x = base(); const pending = x.observationPublication.last_terminal;
    x.observationPublication = {pending_count: 1, entries: [pending], last_terminal: null};
    const r = display(x); assert.equal(r.terminal.state, 'unavailable'); assert.equal(r.terminal.reason, 'no_durable_terminal');
});
for (const [label, change] of [
    ['schema', e => { e.schema = 'neurofly.metric/9'; }],
    ['contract', e => { e.contract = 'metric-contract/9'; }],
    ['spec', e => { e.spec_version = 'gap_crossing/9'; }],
    ['assay', e => { e.assay = 'unknown'; }],
    ['presentation ID', e => { e.presentation_id = 'wrong'; }],
    ['presentation index', e => { e.presentation_index = 0.5; }],
    ['negative elapsed', e => { e.presentation_elapsed_s = -1; }],
    ['boolean origin', e => { e.segment_start_sim_s = false; }],
    ['nonfinite origin', e => { e.segment_start_sim_s = Infinity; }],
    ['missing origin', e => { delete e.segment_start_sim_s; }],
    ['clock cutoff', e => { e.measurement_end_rel_s = 100; e.measurement_end_sim_s = 223.25; }],
    ['absolute cutoff', e => { e.measurement_end_sim_s = 3; }],
    ['relative interval', e => { Object.values(e.records)[0].interval_rel_s = [-1, 0]; }],
    ['absolute interval', e => { Object.values(e.records)[0].interval_sim_s = [0, 1]; }],
    ['missing stamped interval', e => { delete Object.values(e.records)[0].interval_sim_s; }],
    ['interval beyond prefix', e => { const r=Object.values(e.records)[0]; r.interval_rel_s=[0,2];r.interval_sim_s=[123.25,125.25]; }],
    ['unknown evidence ref', e => { Object.values(e.records)[0].evidence_ref = 'absent'; }],
    ['unsupported evidence type', e => { e.evidence.bad={type:'unknown',data:[]}; }],
    ['provenance', e => { e.provenance.motor_assists_enabled = 0; }],
    ['unknown validity', e => { e.validity = 'healthy'; }],
]) test(`rejects malformed live ${label}`, () => rejectLive(x => change(x.observation)));
for (const [label, change] of [
    ['final records', e => { Object.values(e.records)[0].final = true; }],
    ['completion', e => { e.completeness = 'complete'; }],
    ['terminal pose', e => { e.terminal_pose_post_step = {x_mm:0,y_mm:0,heading_rad:0}; }],
    ['receipt', e => { e.receipt = {}; }],
]) test(`live may not carry ${label}`, () => rejectLive(x => change(x.observation)));

test('prototype-shaped JSON is detached inert data without pollution', () => {
    const x = base(); const raw = JSON.parse('{"__proto__":{"polluted":true},"constructor":{"prototype":{"polluted":true}}}');
    x.observation.provenance.extension = raw;
    const r = display(x); assert.equal(r.live.state, 'available');
    assert.equal(r.live.provenance.extension.__proto__.polluted, true);
    assert.equal(Object.getPrototypeOf(r.live.provenance.extension), null);
    assert.equal({}.polluted, undefined);
});
test('non-JSON accessors are rejected without invoking them', () => {
    const x = base(); let invoked=false;
    Object.defineProperty(x.observation, 'untrusted', {enumerable:true,get(){invoked=true;throw Error('called');}});
    assert.equal(display(x).live.state, 'unavailable'); assert.equal(invoked,false);
});
test('null missing sources are explicit, never manufactured measurements', () => {
    const x = base(); delete x.observation; delete x.observationPublication;
    const r = display(x); assert.equal(r.live.reason,'no_live_observation'); assert.equal(r.terminal.reason,'no_publication');
    assert.equal(r.currentValidity,null);
});
for (const [label, change] of [
    ['missing pose', e => { e.terminal_pose_post_step = null; }],
    ['nonfinite pose', e => { e.terminal_pose_post_step.x_mm = NaN; }],
    ['provisional record', e => { Object.values(e.records)[0].final = false; }],
    ['unknown completeness', e => { e.completeness = 'pending'; }],
    ['interrupted labelled complete', e => { e.completeness = 'complete'; }],
    ['fault labelled valid', e => { e.end_reason = 'fault_halt'; }],
]) test(`rejects malformed frozen ${label}`, () => rejectTerminal(x => change(x.observationPublication.last_terminal.observation)));
test('completed-window terminal context is checked without calculating outcomes', () => {
    const x=clone(generated.deadlineFixture); x.observationPublication.last_terminal.observation.window_end_rel_s++;
    assert.equal(display(x).terminal.state,'unavailable');
});
test('segment unavailable does not assert a saved terminal is current', () => {
    const x=base(); x.observation=null; const r=display(x);
    assert.equal(r.terminal.state,'available'); assert.equal(r.terminal.historical,null);
    assert.match(r.terminal.contextLabel,/segment unavailable/);
});
test('malformed outer input and inherited identities cannot become current context', () => {
    for(const input of [null,undefined,[],Object.create(base())]) assert.equal(display(input).state,'unavailable');
    const x=base(); let invoked=false;
    Object.defineProperty(x,'identity',{get(){invoked=true;return {};}});
    assert.equal(display(x).state,'unavailable'); assert.equal(invoked,false);
});
test('rejecting either source does not hide or relabel the other source', () => {
    const badLive=base(); badLive.observation.identity.activation++;
    const a=display(badLive); assert.equal(a.live.state,'unavailable'); assert.equal(a.terminal.state,'available');
    const badTerminal=base(); badTerminal.observationPublication.last_terminal.receipt.durable=false;
    const b=display(badTerminal); assert.equal(b.live.state,'available'); assert.equal(b.terminal.state,'unavailable');
});
for (const item of [{type:'sequence',data:null}, {type:'pose',data:{}},
    {type:'event_sequence',data:[{event:'bad'}]}, {type:'bins',data:{counts:[]}}]) {
    test(`malformed typed evidence ${item.type} is unavailable`, () => rejectLive(x => { x.observation.evidence.bad=item; }));
}

for (const field of ['daemon_run_id', 'run_id', 'instance_id', 'brain_id', 'activation', 'backend', 'controller_version', 'synthetic', 'test_mode']) {
    test(`extra frozen identity shadow ${field} cannot redirect receipt ownership`, () => {
        const x = base(), w = x.observationPublication.last_terminal;
        w.observation[field] = w.observation.identity[field];
        assert.equal(display(x).terminal.state, 'unavailable');
        w.observation[field] = `different-${field}`;
        if (['daemon_run_id', 'run_id', 'instance_id'].includes(field)) {
            w.observation_key[field] = w.observation[field];
            w.receipt.observation_key[field] = w.observation[field];
        }
        assert.equal(display(x).terminal.state, 'unavailable');
    });
}
for (const source of ['live', 'terminal']) {
    const target = x => source === 'live' ? x.observation : x.observationPublication.last_terminal.observation;
    test(`${source} nested array getter rejected without invocation`, () => {
        const x = base(); let calls = 0;
        const a = target(x).provenance.extension = {nested: [{deeper: ['sound']}]};
        Object.defineProperty(a.nested[0].deeper, '0', {enumerable:true,get(){calls++;return 'hostile';}});
        assert.equal(display(x)[source].state, 'unavailable'); assert.equal(calls, 0);
    });
    test(`${source} own callable map cannot conceal nonfinite values`, () => {
        const x = base(); let calls = 0;
        const a = target(x).provenance.controller_states = [NaN];
        a.map = () => {calls++;return [];};
        assert.equal(display(x)[source].state, 'unavailable'); assert.equal(calls, 0);
    });
    test(`${source} inherited array accessors rejected without invocation`, () => {
        const x = base(); let calls = 0;
        const a = target(x).provenance.controller_states = new Array(1);
        const proto = Object.create(Array.prototype);
        Object.defineProperty(proto, '0', {get(){calls++;return 'hostile';}});
        Object.setPrototypeOf(a, proto);
        assert.equal(display(x)[source].state, 'unavailable'); assert.equal(calls, 0);
    });
    test(`${source} sparse arrays cannot manufacture a context`, () => {
        const x = base(); target(x).provenance.extension = {deep: [new Array(2)]};
        assert.equal(display(x)[source].state, 'unavailable');
    });
    test(`${source} own array iterator is never called`, () => {
        const x = base(); let calls = 0;
        const a = target(x).provenance.controller_states = [];
        a[Symbol.iterator] = () => {calls++;throw Error('called');};
        assert.equal(display(x)[source].state, 'unavailable'); assert.equal(calls, 0);
    });
    test(`${source} dense nested arrays remain detached inert JSON`, () => {
        const x = base(); const a = target(x).provenance.extension = {deep: [[0, false, null], []]};
        const r = display(x)[source]; assert.equal(r.state, 'available', r.error);
        assert.deepEqual(r.provenance.extension.deep[0], [0, false, null]);
        a.deep[0][0] = 99; assert.equal(r.provenance.extension.deep[0][0], 0);
    });
}

for (const flag of ['synthetic', 'test_mode']) {
    test(`live ${flag} mismatch cannot become current provenance`, () => {
        const x = base(); x.observation.identity[flag] = !x.identity[flag];
        const r = display(x);
        assert.equal(r.live.state, 'unavailable');
        assert.equal(r.currentValidity, null);
        assert.equal(r.terminal.state, 'available');
        assert.equal(r.terminal.identity[flag], x.identity[flag]);
    });
    test(`saved ${flag} mismatch remains historical with original frozen flags`, () => {
        const x = base(); const original = JSON.stringify(x.observationPublication.last_terminal);
        const frozenFlag = x.observationPublication.last_terminal.identity[flag];
        x.identity[flag] = !frozenFlag;
        x.observation.identity[flag] = !frozenFlag;
        const r = display(x);
        assert.equal(r.live.state, 'available');
        assert.equal(r.terminal.state, 'available');
        assert.equal(r.terminal.historical, true);
        assert.equal(r.terminal.identity[flag], frozenFlag);
        assert.equal(JSON.stringify(x.observationPublication.last_terminal), original);
    });
}
test('matching false flags and zero activation remain valid current context', () => {
    const x = base(), w = x.observationPublication.last_terminal;
    for (const identity of [x.identity, x.observation.identity, w.identity, w.observation.identity]) {
        identity.synthetic = false; identity.test_mode = false; identity.activation = 0;
    }
    const r = display(x);
    assert.equal(r.live.state, 'available'); assert.equal(r.terminal.state, 'available');
    assert.equal(r.terminal.historical, false);
    assert.equal(r.currentIdentity.activation, 0);
    assert.equal(r.live.identity.synthetic, false); assert.equal(r.live.identity.test_mode, false);
    assert.equal(r.terminal.identity.synthetic, false); assert.equal(r.terminal.identity.test_mode, false);
});
