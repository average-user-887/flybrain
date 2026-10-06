'use strict';

const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const {
    SUPPORTED_SCHEMA,
    MetricRecordError,
    validateMetricRecord,
    formatMetricRecord,
} = require('../web/metric_records.js');

test('exposes the same pure API to a browser-style global without a DOM', () => {
    const source = fs.readFileSync(path.join(__dirname, '..', 'web', 'metric_records.js'), 'utf8');
    const context = vm.createContext({});
    vm.runInContext(source, context, {filename: 'metric_records.js'});
    assert.equal(context.NeuroFlyMetricRecords.SUPPORTED_SCHEMA, SUPPORTED_SCHEMA);
    assert.equal(typeof context.NeuroFlyMetricRecords.validateMetricRecord, 'function');
    assert.equal(typeof context.NeuroFlyMetricRecords.formatMetricRecord, 'function');
});

function record(value, overrides = {}) {
    return {
        value,
        available: value !== null,
        reason: value === null ? 'not_observed' : null,
        note: null,
        kind: 'count',
        unit: 'count',
        counts: {},
        numerator: null,
        denominator: null,
        interval_rel_s: [0, 1],
        evidence_ref: null,
        capability: 'measured',
        scope: 'presentation',
        final: false,
        ...overrides,
    };
}

test('formats every supported kind from producer-shaped records', () => {
    const fixtures = [
        ['count', 'count', 3, '3 count'],
        ['duration', 's', 1.25, '1.25 s'],
        ['integral', 'degC*s', 0.22999999501443, '0.23 degC*s'],
        ['latency', 'ms', 18, '18 ms'],
        ['ratio', 'ratio', 0.5, '0.5 ratio'],
        ['index', 'index', -0.25, '-0.25 index'],
        ['fraction', 'fraction', 0.75, '0.75 fraction'],
        ['event', 'bool', true, 'true'],
        ['label', 'label', 'CROSS', 'CROSS'],
        ['kinematic', 'mm/s', -2.5, '-2.5 mm/s'],
        ['mean', 'rad/s', -0.125, '-0.125 rad/s'],
    ];
    for (const [kind, unit, value, text] of fixtures) {
        const display = formatMetricRecord(SUPPORTED_SCHEMA, record(value, {kind, unit}));
        assert.equal(display.text, text, kind);
        assert.equal(display.rawValue, value, kind);
        assert.equal(display.rawUnit, unit, kind);
        assert.equal(display.state, 'available', kind);
    }
});

test('preserves measured zero and false as available values', () => {
    const zero = formatMetricRecord(SUPPORTED_SCHEMA, record(0, {kind: 'mean', unit: 'rad/s'}));
    assert.equal(zero.text, '0 rad/s');
    assert.equal(zero.rawValue, 0);
    assert.equal(zero.reason, null);

    const noEvent = formatMetricRecord(SUPPORTED_SCHEMA, record(false, {kind: 'event', unit: 'bool'}));
    assert.equal(noEvent.text, 'false');
    assert.equal(noEvent.rawValue, false);
    assert.equal(noEvent.available, true);
    assert.equal(noEvent.reason, null);
});

test('keeps every unavailable reason distinct without synthesizing a value', () => {
    const expected = {
        pending: 'Pending',
        not_observed: 'Not observed',
        zero_denominator: 'Unavailable: zero denominator',
        insufficient_events: 'Insufficient events',
        censored: 'Censored',
        unsupported: 'Unsupported',
        invalidated: 'Invalidated',
    };
    for (const [reason, text] of Object.entries(expected)) {
        const capability = reason === 'unsupported' ? 'unsupported' : 'measured';
        const relationship = reason === 'zero_denominator'
            ? {kind: 'ratio', unit: 'ratio', numerator: 0, denominator: 0}
            : {};
        const display = formatMetricRecord(SUPPORTED_SCHEMA, record(null, {reason, capability, ...relationship}));
        assert.equal(display.text, text, reason);
        assert.equal(display.state, reason, reason);
        assert.equal(display.rawValue, null, reason);
        assert.equal(display.valueText, null, reason);
    }
});

test('integral exposure keeps its declared unit and raw value without a generic unit rule', () => {
    const raw = 287.345678901234;
    const display = formatMetricRecord(SUPPORTED_SCHEMA, record(raw, {
        kind: 'integral', unit: 'degC*s', scope: 'segment', final: true,
    }));
    assert.equal(display.text, '287.346 degC*s');
    assert.equal(display.unitText, 'degC*s');
    assert.equal(display.rawValue, raw);
    assert.equal(display.rawUnit, 'degC*s');

    const generic = formatMetricRecord(SUPPORTED_SCHEMA, record(2.5, {kind: 'integral', unit: 's'}));
    assert.equal(generic.text, '2.5 s');
    assert.equal(generic.rawValue, 2.5);
    assert.equal(generic.rawUnit, 's');
});

test('preserves counts, intervals, ratio components, and evidence references', () => {
    const source = record(-0.5, {
        kind: 'index', unit: 'index', capability: 'derived', scope: 'segment', final: true,
        counts: {left: 1, right: 3, nested: {finite: -2.5}},
        numerator: -2,
        denominator: 4,
        interval_rel_s: [6, 87.42],
        interval_sim_s: [1207.3, 1288.72],
        evidence_ref: 'entries',
        note: 'measured preference index',
    });
    const display = formatMetricRecord(SUPPORTED_SCHEMA, source);
    assert.deepEqual(display.counts, source.counts);
    assert.notEqual(display.counts, source.counts);
    assert.deepEqual(display.intervalRelS, [6, 87.42]);
    assert.deepEqual(display.intervalSimS, [1207.3, 1288.72]);
    assert.equal(display.numerator, -2);
    assert.equal(display.denominator, 4);
    assert.equal(display.evidenceRef, 'entries');
    assert.equal(display.note, 'measured preference index');
});

test('label values remain inert plain text', () => {
    const label = '<img src=x onerror="globalThis.compromised=true">';
    const display = formatMetricRecord(SUPPORTED_SCHEMA, record(label, {kind: 'label', unit: 'label'}));
    assert.equal(display.text, label);
    assert.equal(display.rawValue, label);
    assert.equal(globalThis.compromised, undefined);
    assert.equal(typeof display.text, 'string');
});

test('preserves real T-maze label counts in live and frozen records', () => {
    const live = record('cs_plus', {
        kind: 'label', unit: 'label', counts: {arm: 'arm_a'},
        interval_rel_s: [0, 1.02], evidence_ref: 'first_choice_pose', final: false,
    });
    const liveDisplay = formatMetricRecord(SUPPORTED_SCHEMA, live);
    assert.equal(liveDisplay.text, 'cs_plus');
    assert.deepEqual(liveDisplay.counts, {arm: 'arm_a'});
    assert.equal(liveDisplay.final, false);

    const frozenDisplay = formatMetricRecord(SUPPORTED_SCHEMA, {...live, final: true});
    assert.equal(frozenDisplay.text, 'cs_plus');
    assert.deepEqual(frozenDisplay.counts, {arm: 'arm_a'});
    assert.equal(frozenDisplay.final, true);
});

test('clones finite JSON counts with prototype-shaped keys as own detached data', () => {
    const counts = JSON.parse(
        '{"__proto__":{"n":2},"constructor":{"label":"kept"},' +
        '"prototype":{"series":[1,true,null,"arm_b"]},"arm":"arm_a"}',
    );
    const display = formatMetricRecord(SUPPORTED_SCHEMA, record('cs_plus', {
        kind: 'label', unit: 'label', counts,
    }));

    assert.deepEqual(display.counts, counts);
    assert.notEqual(display.counts, counts);
    assert.equal(Object.getPrototypeOf(display.counts), Object.prototype);
    for (const key of ['__proto__', 'constructor', 'prototype', 'arm']) {
        assert.equal(Object.hasOwn(display.counts, key), true, key);
    }
    assert.deepEqual(display.counts.__proto__, {n: 2});
    counts.__proto__.n = 99;
    counts.constructor.label = 'changed';
    counts.prototype.series[0] = 99;
    assert.deepEqual(display.counts.__proto__, {n: 2});
    assert.deepEqual(display.counts.constructor, {label: 'kept'});
    assert.deepEqual(display.counts.prototype, {series: [1, true, null, 'arm_b']});

    const nullPrototype = Object.create(null);
    Object.defineProperty(nullPrototype, '__proto__', {value: {finite: 3}, enumerable: true});
    const cloned = validateMetricRecord(SUPPORTED_SCHEMA, record(1, {counts: nullPrototype})).counts;
    assert.equal(Object.getPrototypeOf(cloned), null);
    assert.equal(Object.hasOwn(cloned, '__proto__'), true);
    assert.deepEqual(cloned.__proto__, {finite: 3});

    const cyclic = {};
    cyclic.self = cyclic;
    assert.throws(
        () => validateMetricRecord(SUPPORTED_SCHEMA, record(1, {counts: cyclic})),
        error => error instanceof MetricRecordError && error.path === 'record.counts.self',
    );
    assert.throws(
        () => validateMetricRecord(SUPPORTED_SCHEMA, record(1, {counts: {bad: undefined}})),
        error => error instanceof MetricRecordError && error.path === 'record.counts.bad',
    );
});

test('refuses unknown schemas instead of interpreting legacy records', () => {
    for (const schema of ['neurofly.metric/1.1', 'neurofly.metric/99', null, undefined]) {
        assert.throws(
            () => formatMetricRecord(schema, record(1)),
            error => error instanceof MetricRecordError && error.path === 'schema',
        );
    }
});

test('enforces closed vocabularies and boolean metadata', () => {
    const cases = [
        ['kind', 'bogus'], ['unit', 'furlong'], ['reason', 'maybe'],
        ['capability', 'guessed'], ['scope', 'run'],
    ];
    for (const [field, value] of cases) {
        const fixture = field === 'reason'
            ? record(null, {[field]: value})
            : record(1, {[field]: value});
        assert.throws(() => validateMetricRecord(SUPPORTED_SCHEMA, fixture), MetricRecordError, field);
    }
    assert.throws(() => validateMetricRecord(SUPPORTED_SCHEMA, record(1, {available: 1})), MetricRecordError);
    assert.throws(() => validateMetricRecord(SUPPORTED_SCHEMA, record(1, {final: 0})), MetricRecordError);
    assert.throws(
        () => validateMetricRecord(SUPPORTED_SCHEMA, record(null, {reason: 'pending', capability: 'unsupported'})),
        /unsupported capability requires reason unsupported/,
    );
});

test('rejects contradictory availability, value, and reason combinations', () => {
    const fixtures = [
        record(null, {available: true, reason: null}),
        record(0, {available: false, reason: 'not_observed'}),
        record(1, {available: true, reason: 'censored'}),
        record(null, {available: false, reason: null}),
    ];
    for (const fixture of fixtures) {
        assert.throws(() => validateMetricRecord(SUPPORTED_SCHEMA, fixture), MetricRecordError);
    }
});

test('enforces component, zero-denominator, and ordered-interval relationships', () => {
    for (const fixture of [
        record(1, {numerator: 1}),
        record(1, {denominator: 1}),
        record(null, {reason: 'zero_denominator', denominator: 0}),
        record(null, {
            kind: 'mean', unit: 'min', reason: 'zero_denominator', numerator: 0, denominator: 0,
        }),
        record(null, {kind: 'ratio', unit: 'ratio', reason: 'zero_denominator', denominator: 2}),
    ]) {
        assert.throws(() => validateMetricRecord(SUPPORTED_SCHEMA, fixture), MetricRecordError);
    }

    const emptyIndex = record(null, {
        kind: 'index', unit: 'index', reason: 'zero_denominator', numerator: 0, denominator: 0,
    });
    assert.equal(formatMetricRecord(SUPPORTED_SCHEMA, emptyIndex).state, 'zero_denominator');
    for (const reason of ['not_observed', 'unsupported']) {
        const capability = reason === 'unsupported' ? 'unsupported' : 'measured';
        const freshRatio = record(null, {kind: 'ratio', unit: 'ratio', reason, capability});
        assert.equal(formatMetricRecord(SUPPORTED_SCHEMA, freshRatio).state, reason);
    }

    for (const field of ['interval_rel_s', 'interval_sim_s']) {
        assert.throws(
            () => validateMetricRecord(SUPPORTED_SCHEMA, record(1, {[field]: [3, 1]})),
            error => error instanceof MetricRecordError && error.path === `record.${field}`,
        );
    }
});

test('preserves timed and fault-invalidated outcome semantics', () => {
    const completedFalse = formatMetricRecord(SUPPORTED_SCHEMA, record(false, {
        kind: 'event', unit: 'bool', final: true, interval_rel_s: [0, 2],
        counts: {samples: 100},
    }));
    assert.equal(completedFalse.text, 'false');
    assert.equal(completedFalse.rawValue, false);

    const faulted = formatMetricRecord(SUPPORTED_SCHEMA, record(null, {
        kind: 'event', unit: 'bool', reason: 'invalidated', final: true,
        interval_rel_s: [0, 1.25], counts: {samples: 62, faulted_at_rel_s: 1.25},
    }));
    assert.equal(faulted.text, 'Invalidated');
    assert.equal(faulted.rawValue, null);
    assert.deepEqual(faulted.counts, {samples: 62, faulted_at_rel_s: 1.25});
});

test('rejects malformed scalar values and structured record values', () => {
    const fixtures = [
        record(NaN), record(Infinity), record(-Infinity),
        record('3', {kind: 'count'}),
        record(1, {kind: 'event', unit: 'bool'}),
        record('true', {kind: 'event', unit: 'bool'}),
        record(false, {kind: 'label', unit: 'label'}),
        record([]), record({}),
    ];
    for (const fixture of fixtures) {
        assert.throws(() => validateMetricRecord(SUPPORTED_SCHEMA, fixture), MetricRecordError);
    }
});

test('recursively rejects nonfinite nested metadata and malformed intervals', () => {
    const badCounts = record(1, {counts: {outer: {bad: NaN}}});
    assert.throws(
        () => validateMetricRecord(SUPPORTED_SCHEMA, badCounts),
        error => error instanceof MetricRecordError && error.path === 'record.counts.outer.bad',
    );
    for (const interval of [[0, Infinity], [0], [0, 1, 2], '0,1']) {
        assert.throws(
            () => validateMetricRecord(SUPPORTED_SCHEMA, record(1, {interval_rel_s: interval})),
            MetricRecordError,
        );
    }
    assert.throws(
        () => validateMetricRecord(SUPPORTED_SCHEMA, record(1, {numerator: NaN})),
        MetricRecordError,
    );
});

test('requires the producer record shape and refuses undeclared fields', () => {
    const missing = record(1);
    delete missing.final;
    assert.throws(() => validateMetricRecord(SUPPORTED_SCHEMA, missing), /record.final: is required/);
    assert.throws(
        () => validateMetricRecord(SUPPORTED_SCHEMA, record(1, {run_validity: 'valid'})),
        /is not part of the schema 1.2 record shape/,
    );
});


test('numeric presentation has six significant digits without changing raw data or units', () => {
    const cases = [
        [26.417029773597136, '26.417 mm'], [1/3, '0.333333 mm'],
        [-0.987654321, '-0.987654 mm'], [1.23456789e-9, '1.23457e-9 mm'],
        [-1.23456789e-9, '-1.23457e-9 mm'], [0, '0 mm'], [-0, '0 mm'],
        [0.0001, '0.0001 mm'], [0.00009999999, '0.0001 mm'],
        [999999.9, '1e+6 mm'], [1.23456789e12, '1.23457e+12 mm'],
        [Number.MIN_VALUE, '5e-324 mm'], [Number.MAX_VALUE, '1.79769e+308 mm'],
    ];
    for (const [raw, text] of cases) {
        const source = record(raw, {kind:'kinematic', unit:'mm', evidence_ref:'original-proof'});
        const display = formatMetricRecord(SUPPORTED_SCHEMA, source);
        assert.equal(display.text, text);assert.ok(Object.is(display.rawValue,raw));
        assert.ok(Object.is(source.value,raw));assert.equal(display.evidenceRef,'original-proof');
    }
    const source=record(1/3,{kind:'ratio',unit:'ratio',numerator:1,denominator:3,counts:{n:1234567}});
    const ratio=formatMetricRecord(SUPPORTED_SCHEMA,source);
    assert.equal(ratio.text,'0.333333 ratio');assert.equal(ratio.numerator,1);assert.equal(ratio.denominator,3);
    assert.deepEqual(ratio.counts,{n:1234567});assert.equal(ratio.rawValue,1/3);
    const count=formatMetricRecord(SUPPORTED_SCHEMA,record(1234567));
    assert.equal(count.text,'1.23457e+6 count');assert.equal(count.rawValue,1234567);
});
test('nonfinite numbers remain refused before numeric presentation', () => {
    for(const raw of [NaN,Infinity,-Infinity])assert.throws(()=>formatMetricRecord(SUPPORTED_SCHEMA,record(raw)),/finite/);
});
