/*
 * Pure validation and display formatting for neurofly.metric/1.2 records.
 *
 * This module deliberately knows nothing about envelopes, assays, run identity,
 * validity, or the DOM. Record-specific label/unit declarations and envelope time
 * coordinates also belong to the parent consumer, which has the record spec and identity.
 */
(function exposeMetricRecords(root, factory) {
    const api = factory();
    if (typeof module === 'object' && module.exports) module.exports = api;
    if (root) root.NeuroFlyMetricRecords = api;
}(typeof globalThis === 'object' ? globalThis : this, function buildMetricRecords() {
    'use strict';

    const SUPPORTED_SCHEMA = 'neurofly.metric/1.2';
    const KINDS = new Set([
        'count', 'duration', 'integral', 'latency', 'ratio', 'index',
        'fraction', 'event', 'label', 'kinematic', 'mean',
    ]);
    const UNITS = new Set([
        's', 'ms', 'min', 'mm', 'mm/s', 'mm/s^3', 'deg', 'deg/s',
        'rad', 'rad/s', 'degC', 'degC*s', 'count', 'fraction', 'index',
        'ratio', 'gain', 'bool', 'label',
    ]);
    const REASONS = new Set([
        'pending', 'not_observed', 'zero_denominator', 'insufficient_events',
        'censored', 'unsupported', 'invalidated',
    ]);
    const CAPABILITIES = new Set(['measured', 'derived', 'unsupported']);
    const SCOPES = new Set(['segment', 'presentation']);
    const NUMERIC_KINDS = new Set([
        'count', 'duration', 'integral', 'latency', 'ratio', 'index',
        'fraction', 'kinematic', 'mean',
    ]);
    const COMPONENT_KINDS = new Set(['ratio', 'index', 'fraction']);
    const REQUIRED_FIELDS = [
        'value', 'available', 'reason', 'note', 'kind', 'unit', 'counts',
        'numerator', 'denominator', 'interval_rel_s', 'evidence_ref',
        'capability', 'scope', 'final',
    ];
    const OPTIONAL_FIELDS = new Set(['interval_sim_s']);
    const REASON_TEXT = Object.freeze({
        pending: 'Pending',
        not_observed: 'Not observed',
        zero_denominator: 'Unavailable: zero denominator',
        insufficient_events: 'Insufficient events',
        censored: 'Censored',
        unsupported: 'Unsupported',
        invalidated: 'Invalidated',
    });

    class MetricRecordError extends Error {
        constructor(path, message) {
            super(`${path}: ${message}`);
            this.name = 'MetricRecordError';
            this.path = path;
        }
    }

    function fail(path, message) {
        throw new MetricRecordError(path, message);
    }

    function isPlainObject(value) {
        if (value === null || typeof value !== 'object' || Array.isArray(value)) return false;
        const prototype = Object.getPrototypeOf(value);
        return prototype === Object.prototype || prototype === null;
    }

    function cloneFinite(value, path, seen) {
        if (value === null || typeof value === 'string' || typeof value === 'boolean') return value;
        if (typeof value === 'number') {
            if (!Number.isFinite(value)) fail(path, 'number must be finite');
            return value;
        }
        if (typeof value !== 'object') fail(path, `unsupported ${typeof value} value`);
        if (seen.has(value)) fail(path, 'cyclic data is not supported');
        seen.add(value);
        let copy;
        if (Array.isArray(value)) {
            copy = value.map((item, index) => cloneFinite(item, `${path}[${index}]`, seen));
        } else if (isPlainObject(value)) {
            copy = Object.getPrototypeOf(value) === null ? Object.create(null) : {};
            for (const [key, item] of Object.entries(value)) {
                Object.defineProperty(copy, key, {
                    value: cloneFinite(item, `${path}.${key}`, seen),
                    enumerable: true,
                    configurable: true,
                    writable: true,
                });
            }
        } else {
            fail(path, 'only plain objects and arrays are supported');
        }
        seen.delete(value);
        return copy;
    }

    function checkVocabulary(path, value, vocabulary) {
        if (typeof value !== 'string' || !vocabulary.has(value)) {
            fail(path, `unsupported value ${String(value)}`);
        }
    }

    function checkNullableText(path, value) {
        if (value !== null && typeof value !== 'string') fail(path, 'must be a string or null');
    }

    function checkNullableNumber(path, value) {
        if (value !== null && (typeof value !== 'number' || !Number.isFinite(value))) {
            fail(path, 'must be a finite number or null');
        }
    }

    function checkInterval(path, value) {
        if (value === null) return null;
        if (!Array.isArray(value) || value.length !== 2) fail(path, 'must be [start, end] or null');
        const interval = value.map((part, index) => {
            if (typeof part !== 'number' || !Number.isFinite(part)) {
                fail(`${path}[${index}]`, 'must be a finite number');
            }
            return part;
        });
        if (interval[1] < interval[0]) fail(path, 'end must not precede start');
        return interval;
    }

    function validateMetricRecord(schema, record) {
        if (schema !== SUPPORTED_SCHEMA) {
            fail('schema', `unsupported metric schema ${String(schema)}`);
        }
        if (!isPlainObject(record)) fail('record', 'must be a plain object');
        for (const field of REQUIRED_FIELDS) {
            if (!Object.prototype.hasOwnProperty.call(record, field)) fail(`record.${field}`, 'is required');
        }
        const allowed = new Set([...REQUIRED_FIELDS, ...OPTIONAL_FIELDS]);
        for (const field of Object.keys(record)) {
            if (!allowed.has(field)) fail(`record.${field}`, 'is not part of the schema 1.2 record shape');
        }

        checkVocabulary('record.kind', record.kind, KINDS);
        checkVocabulary('record.unit', record.unit, UNITS);
        checkVocabulary('record.capability', record.capability, CAPABILITIES);
        checkVocabulary('record.scope', record.scope, SCOPES);
        if (typeof record.available !== 'boolean') fail('record.available', 'must be boolean');
        if (typeof record.final !== 'boolean') fail('record.final', 'must be boolean');
        checkNullableText('record.note', record.note);
        checkNullableText('record.evidence_ref', record.evidence_ref);

        if (record.available) {
            if (record.reason !== null) fail('record.reason', 'must be null when available is true');
            if (record.value === null || record.value === undefined) {
                fail('record.value', 'must be a supported scalar when available is true');
            }
            if (NUMERIC_KINDS.has(record.kind)) {
                if (typeof record.value !== 'number' || !Number.isFinite(record.value)) {
                    fail('record.value', `must be a finite number for kind ${record.kind}`);
                }
            } else if (record.kind === 'event') {
                if (typeof record.value !== 'boolean') fail('record.value', 'must be boolean for kind event');
            } else if (record.kind === 'label') {
                if (typeof record.value !== 'string') fail('record.value', 'must be a string for kind label');
            }
        } else {
            if (record.value !== null) fail('record.value', 'must be null when available is false');
            checkVocabulary('record.reason', record.reason, REASONS);
        }
        if (record.capability === 'unsupported' && record.reason !== 'unsupported') {
            fail('record.capability', 'unsupported capability requires reason unsupported');
        }

        if (!isPlainObject(record.counts)) fail('record.counts', 'must be a plain object');
        const counts = cloneFinite(record.counts, 'record.counts', new Set());
        checkNullableNumber('record.numerator', record.numerator);
        checkNullableNumber('record.denominator', record.denominator);
        if (!COMPONENT_KINDS.has(record.kind)) {
            if (record.numerator !== null) fail('record.numerator', `must be null for kind ${record.kind}`);
            if (record.denominator !== null) fail('record.denominator', `must be null for kind ${record.kind}`);
        }
        if (record.reason === 'zero_denominator') {
            if (!COMPONENT_KINDS.has(record.kind)) {
                fail('record.reason', 'zero_denominator applies only to ratio, index, or fraction records');
            }
            if (record.denominator !== 0) {
                fail('record.denominator', 'must be zero when reason is zero_denominator');
            }
        }
        const intervalRelS = checkInterval('record.interval_rel_s', record.interval_rel_s);
        const intervalSimS = Object.prototype.hasOwnProperty.call(record, 'interval_sim_s')
            ? checkInterval('record.interval_sim_s', record.interval_sim_s)
            : undefined;

        const validated = {
            value: record.value,
            available: record.available,
            reason: record.reason,
            note: record.note,
            kind: record.kind,
            unit: record.unit,
            counts,
            numerator: record.numerator,
            denominator: record.denominator,
            interval_rel_s: intervalRelS,
            evidence_ref: record.evidence_ref,
            capability: record.capability,
            scope: record.scope,
            final: record.final,
        };
        if (intervalSimS !== undefined) validated.interval_sim_s = intervalSimS;
        return validated;
    }

    function availableValueText(record) {
        if (record.kind === 'label') return record.value;
        if (record.kind === 'event') return record.value ? 'true' : 'false';
        // Six significant digits are presentation only; rawValue remains exact.
        if (record.value === 0) return '0';
        const rounded = Number(record.value.toPrecision(6));
        const magnitude = Math.abs(rounded);
        return magnitude >= 1e6 || magnitude < 1e-4
            ? rounded.toExponential() : String(rounded);
    }

    function formatMetricRecord(schema, record) {
        const validated = validateMetricRecord(schema, record);
        const valueText = validated.available ? availableValueText(validated) : null;
        const unitText = validated.available && validated.kind !== 'event' && validated.kind !== 'label'
            ? validated.unit : null;
        const text = validated.available
            ? (unitText === null ? valueText : `${valueText} ${unitText}`)
            : REASON_TEXT[validated.reason];
        return {
            schema: SUPPORTED_SCHEMA,
            text,
            valueText,
            unitText,
            state: validated.available ? 'available' : validated.reason,
            available: validated.available,
            reason: validated.reason,
            kind: validated.kind,
            unit: validated.unit,
            capability: validated.capability,
            scope: validated.scope,
            final: validated.final,
            note: validated.note,
            rawValue: validated.value,
            rawUnit: validated.unit,
            counts: validated.counts,
            numerator: validated.numerator,
            denominator: validated.denominator,
            intervalRelS: validated.interval_rel_s,
            intervalSimS: validated.interval_sim_s,
            evidenceRef: validated.evidence_ref,
        };
    }

    return Object.freeze({
        SUPPORTED_SCHEMA,
        MetricRecordError,
        validateMetricRecord,
        formatMetricRecord,
    });
}));
