/* Pure transport/ownership adapter. It does not calculate scientific metrics,
 * recompute payload hashes, assert disk durability, or touch browser state.
 * Load metric_records.js first in a browser; Node resolves it locally.
 */
(function expose(root, factory) {
    const records = typeof module === 'object' && module.exports
        ? require('./metric_records.js') : root && root.NeuroFlyMetricRecords;
    const api = factory(records);
    if (typeof module === 'object' && module.exports) module.exports = api;
    if (root) root.NeuroFlyObservationDisplay = api;
}(typeof globalThis === 'object' ? globalThis : this, function build(records) {
    'use strict';
    const CONTRACT = 'metric-contract/1.2';
    const ASSAYS = Object.freeze({
        'open-arena': 'open_arena', 't-maze': 't_maze', 'y-maze': 'y_maze',
        'heat-maze': 'heat_maze', buridan: 'buridan', 'visual-operant': 'visual_operant',
        'wind-tunnel': 'wind_tunnel', 'looming-escape': 'looming_escape', optomotor: 'optomotor',
        'gap-crossing': 'gap_crossing', 'circadian-dam': 'circadian_dam', courtship: 'courtship',
        labyrinth: 'labyrinth', 'multisensory-sandbox': 'multisensory',
    });
    const LIVE_ID = ['daemon_run_id', 'run_id', 'instance_id', 'activation', 'assay', 'backend', 'controller_version', 'brain_id'];
    const CONTEXT_ID = LIVE_ID.concat(['synthetic', 'test_mode']);
    const OWNER = ['assay', 'backend', 'instance_id', 'brain_id'];
    const VALIDITIES = ['valid', 'invalidated', 'exploratory_degraded'];
    function fail(path, message) { throw new Error(`${path}: ${message}`); }
    function plain(value) {
        if (!value || typeof value !== 'object' || Array.isArray(value)) return false;
        const p = Object.getPrototypeOf(value);
        return p === Object.prototype || p === null;
    }
    function own(value, key) { return Object.prototype.hasOwnProperty.call(value, key); }
    function object(value, path) { if (!plain(value)) fail(path, 'requires plain object'); }
    function text(value, path) { if (typeof value !== 'string' || !value) fail(path, 'requires nonempty string'); }
    function number(value, path, nullable = false) {
        if (nullable && value === null) return;
        if (typeof value !== 'number' || !Number.isFinite(value)) fail(path, 'requires finite number');
    }
    function integer(value, path, minimum = 0) {
        number(value, path);
        if (!Number.isSafeInteger(value) || value < minimum) fail(path, 'requires safe integer in range');
    }
    function bool(value, path) { if (typeof value !== 'boolean') fail(path, 'requires boolean'); }
    function vocab(value, values, path) { if (!values.includes(value)) fail(path, 'unsupported value'); }
    // Define properties rather than assign prototype-shaped JSON keys.
    function detach(value, path = 'input', seen = new Set()) {
        if (value === null || typeof value === 'string' || typeof value === 'boolean') return value;
        if (typeof value === 'number') { number(value, path); return value; }
        if (typeof value !== 'object' || seen.has(value)) fail(path, 'requires acyclic finite JSON');
        seen.add(value);
        let copy;
        if (Array.isArray(value)) {
            if (Object.getPrototypeOf(value) !== Array.prototype) fail(path, 'requires plain JSON array');
            const descriptors = Object.getOwnPropertyDescriptors(value);
            const keys = Reflect.ownKeys(descriptors);
            const length = descriptors.length.value;
            // Inspect descriptors before reading any element. Dense JSON arrays
            // have only length and own indexed data; no methods or accessors.
            if (keys.length !== length + 1) fail(path, 'requires dense JSON array');
            for (const key of keys) {
                if (key === 'length') continue;
                const index = typeof key === 'string' ? Number(key) : NaN;
                if (!Number.isSafeInteger(index) || index < 0 || index >= length || String(index) !== key
                    || !own(descriptors[key], 'value')) fail(path, 'requires own indexed JSON data');
            }
            copy = [];
            for (let i = 0; i < length; i++) {
                Object.defineProperty(copy, i, {value: detach(descriptors[i].value, `${path}[${i}]`, seen), enumerable: true});
            }
        }
        else {
            object(value, path); copy = Object.create(null);
            for (const key of Object.keys(value)) {
                const d = Object.getOwnPropertyDescriptor(value, key);
                if (!d || !own(d, 'value')) fail(`${path}.${key}`, 'accessors are not JSON');
                Object.defineProperty(copy, key, {value: detach(d.value, `${path}.${key}`, seen), enumerable: true});
            }
        }
        seen.delete(value); return copy;
    }
    function equal(a, b) {
        if (a === b) return true;
        if (a === null || b === null || typeof a !== 'object' || typeof b !== 'object') return false;
        if (Array.isArray(a) !== Array.isArray(b)) return false;
        const keys = Object.keys(a);
        return keys.length === Object.keys(b).length && keys.every(k => own(b, k) && equal(a[k], b[k]));
    }
    function identity(value, path) {
        object(value, path);
        for (const key of LIVE_ID) if (key !== 'activation') text(value[key], `${path}.${key}`);
        integer(value.activation, `${path}.activation`);
        bool(value.synthetic, `${path}.synthetic`); bool(value.test_mode, `${path}.test_mode`);
        if (!own(ASSAYS, value.assay)) fail(`${path}.assay`, 'unsupported assay');
    }
    function interval(value, path) {
        if (value === null) return;
        if (!Array.isArray(value) || value.length !== 2) fail(path, 'requires interval or null');
        value.forEach((v, i) => number(v, `${path}[${i}]`));
        if (value[0] < 0 || value[1] < value[0]) fail(path, 'invalid interval');
    }
    function fields(value, keys, path) {
        object(value, path);
        if (!equal(Object.keys(value).sort(), keys.slice().sort())) fail(path, 'wrong typed evidence shape');
    }
    function evidence(item, path) {
        fields(item, ['type', 'data'], path);
        const d = item.data;
        switch (item.type) {
        case 'pose':
            fields(d, ['x_mm', 'y_mm', 'heading_rad', 't_rel_s', 'step', 'sample_point'], path);
            for (const k of ['x_mm', 'y_mm', 'heading_rad', 't_rel_s']) number(d[k], `${path}.${k}`);
            integer(d.step, `${path}.step`); vocab(d.sample_point, ['pre_motor', 'post_solver'], path);
            break;
        case 'sequence':
            fields(d, ['symbols', 'collapse'], path);
            if (!Array.isArray(d.symbols) || d.symbols.some(v => typeof v !== 'string')) fail(path, 'requires symbol labels');
            vocab(d.collapse, ['none', 'consecutive_repeats'], path); break;
        case 'bins':
            fields(d, ['bin_s', 'start_rel_s', 'quantity', 'counts'], path);
            number(d.bin_s, path); number(d.start_rel_s, path);
            if (typeof d.quantity !== 'string' || !Array.isArray(d.counts)) fail(path, 'invalid bins');
            d.counts.forEach(v => number(v, path)); break;
        case 'event_sequence':
        case 'phase_boundaries':
        case 'intervention_log':
            if (!Array.isArray(d)) fail(path, 'requires evidence array');
            for (const row of d) {
                if (item.type === 'event_sequence') {
                    fields(row, ['event', 't_rel_s', 'step', 'sample_point', 'attrs'], path);
                    if (typeof row.event !== 'string') fail(path, 'requires event label');
                    number(row.t_rel_s, path); integer(row.step, path);
                    vocab(row.sample_point, ['pre_motor', 'post_solver'], path); object(row.attrs, path);
                    for (const value of Object.values(row.attrs)) if (!['string', 'number'].includes(typeof value)) fail(path, 'invalid event attribute');
                } else if (item.type === 'phase_boundaries') {
                    fields(row, ['phase', 'start_rel_s', 'end_rel_s'], path);
                    if (typeof row.phase !== 'string') fail(path, 'requires phase label');
                    interval([row.start_rel_s, row.end_rel_s], path);
                } else {
                    fields(row, ['name', 'value', 't_rel_s', 'step'], path);
                    if (typeof row.name !== 'string' || (row.value !== null && !['string', 'number', 'boolean'].includes(typeof row.value))) fail(path, 'invalid intervention');
                    number(row.t_rel_s, path); integer(row.step, path);
                }
            }
            break;
        default: fail(path, 'unsupported evidence type');
        }
    }
    function envelope(value, frozen) {
        const e = detach(value, 'observation'); object(e, 'observation');
        if (!records) fail('metric_records', 'helper dependency unavailable');
        if (e.schema !== records.SUPPORTED_SCHEMA || e.contract !== CONTRACT) fail('schema', 'unsupported schema/contract');
        identity(e.identity, 'identity');
        for (const field of CONTEXT_ID) {
            if (field !== 'assay' && own(e, field)) fail(`observation.${field}`, 'identity belongs only in identity');
        }
        if (e.assay !== ASSAYS[e.identity.assay] || e.spec_version !== `${e.assay}/1.2`) fail('assay/spec', 'unsupported identity');
        vocab(e.mode, ['continuous', 'fixed', 'until_terminal', 'presentation'], 'mode');
        vocab(e.state, ['observing', 'measurement_ended', 'closed'], 'state');
        vocab(e.sample_point, ['pre_motor', 'post_solver'], 'sample_point');
        vocab(e.window_source, ['spec', 'override', 'continuous_flag'], 'window_source');
        vocab(e.validity, VALIDITIES, 'validity');
        text(e.config_id, 'config_id'); text(e.segment_id, 'segment_id');
        integer(e.presentation_index, 'presentation_index');
        if (e.presentation_id !== `${e.segment_id}:${e.presentation_index}`) fail('presentation_id', 'does not match segment/index');
        number(e.segment_start_sim_s, 'segment_start_sim_s');
        for (const field of ['presentation_start_rel_s', 'presentation_elapsed_s', 'hold_s']) {
            number(e[field], field); if (e[field] < 0) fail(field, 'must be nonnegative');
        }
        for (const field of ['effective_window_s', 'window_s', 'window_end_rel_s', 'dt_s', 'measurement_end_rel_s', 'measurement_end_sim_s']) number(e[field], field, true);
        bool(e.automatic_end, 'automatic_end'); bool(e.dt_fixed, 'dt_fixed');
        if (e.window_s !== e.effective_window_s || (e.effective_window_s !== null && e.effective_window_s <= 0)
            || (e.automatic_end && e.effective_window_s === null) || (e.dt_s !== null && e.dt_s <= 0)) fail('window/dt', 'invalid finite context');
        if (e.override !== null) {
            object(e.override, 'override');
            vocab(e.override.source, ['cli:--trial-seconds', 'cli:--continuous', 'api'], 'override.source');
            number(e.override.set_at_sim_s, 'override.set_at_sim_s'); number(e.override.value, 'override.value', true);
            if (typeof e.override.manifest_run_id !== 'string') fail('override.manifest_run_id', 'requires string');
        }
        if (!own(e, 'window_params') || !own(e, 'light_schedule')) fail('context', 'missing producer context');
        object(e.provenance, 'provenance');
        for (const field of ['gf_source', 'stimulus_entry_stage']) {
            if (e.provenance[field] !== null && typeof e.provenance[field] !== 'string') fail(`provenance.${field}`, 'requires string or null');
        }
        bool(e.provenance.motor_assists_enabled, 'provenance.motor_assists_enabled');
        if (!Array.isArray(e.provenance.controller_states)) fail('provenance.controller_states', 'requires array');
        e.provenance.controller_states.forEach(v => text(v, 'provenance.controller_states'));
        const cutoff = e.measurement_end_rel_s;
        const observedEnd = e.presentation_start_rel_s + e.presentation_elapsed_s;
        number(observedEnd, 'presentation clock end');
        if (cutoff !== null && (cutoff < e.presentation_start_rel_s || cutoff > observedEnd + 0.000001)) fail('measurement_end_rel_s', 'outside presentation clock');
        if (e.measurement_end_sim_s !== (cutoff === null ? null : e.segment_start_sim_s + cutoff)) fail('measurement_end_sim_s', 'incorrect translation');
        if (e.end_reason !== null && !['window_elapsed', 're_presentation_user', 'policy_change', 'manual_reset', 'experiment_selected', 'backend_switch', 'fault_halt', 'shutdown'].includes(e.end_reason)
            && !(typeof e.end_reason === 'string' && /^terminal_event:.+/.test(e.end_reason))) fail('end_reason', 'unsupported reason');
        if (e.terminal_event !== null && typeof e.terminal_event !== 'string') fail('terminal_event', 'requires string or null');
        if (typeof e.end_reason === 'string' && e.end_reason.startsWith('terminal_event:') && e.terminal_event !== e.end_reason.slice(15)) fail('terminal_event', 'does not match end reason');
        if (e.end_reason === 'fault_halt' && e.validity !== 'invalidated') fail('validity', 'fault requires invalidated');
        if (e.end_reason === 'window_elapsed') {
            if (!e.automatic_end || e.effective_window_s === null || e.state !== 'closed'
                || e.terminal_event !== null || cutoff !== e.window_end_rel_s
                || Math.abs(cutoff - e.presentation_start_rel_s - e.effective_window_s) > 0.000001
                || e.presentation_elapsed_s + 0.000001 < e.effective_window_s) fail('window_elapsed', 'inconsistent deadline context');
        }
        if (frozen) {
            vocab(e.completeness, ['complete', 'incomplete'], 'completeness');
            text(e.end_reason, 'end_reason'); if (cutoff === null) fail('cutoff', 'terminal requires cutoff');
            if ((e.end_reason === 'window_elapsed' || e.end_reason.startsWith('terminal_event:'))
                && e.completeness !== 'complete') fail('completeness', 'completed end requires complete');
            if (['manual_reset', 'experiment_selected', 'backend_switch', 'fault_halt', 'shutdown'].includes(e.end_reason)
                && e.completeness !== 'incomplete') fail('completeness', 'interrupted end requires incomplete');
            object(e.terminal_pose_post_step, 'terminal_pose_post_step');
            for (const field of ['x_mm', 'y_mm', 'heading_rad']) number(e.terminal_pose_post_step[field], `terminal_pose.${field}`);
        } else {
            if (e.completeness !== null || e.terminal_pose_post_step !== null || own(e, 'receipt')) fail('live', 'provisional has no terminal receipt or pose');
            if (e.state === 'observing' && (e.end_reason !== null || e.terminal_event !== null || cutoff !== null)) fail('live', 'observing has end receipt');
            if (e.state !== 'observing' && (e.end_reason === null || cutoff === null)) fail('live', 'ended provisional lacks cutoff/reason');
        }
        object(e.evidence, 'evidence');
        for (const [name, item] of Object.entries(e.evidence)) {
            text(name, 'evidence key'); evidence(item, `evidence.${name}`);
        }
        object(e.records, 'records'); if (!Object.keys(e.records).length) fail('records', 'requires records');
        const formatted = Object.create(null);
        for (const [name, record] of Object.entries(e.records)) {
            text(name, 'record key');
            const r = records.validateMetricRecord(e.schema, record);
            if (r.final !== frozen) fail(`records.${name}.final`, 'wrong source state');
            if (r.evidence_ref !== null && !own(e.evidence, r.evidence_ref)) fail('evidence_ref', 'unknown evidence');
            if (!own(record, 'interval_sim_s')) fail('interval_sim_s', 'required stamped interval');
            interval(r.interval_rel_s, 'interval_rel_s');
            const expected = r.interval_rel_s === null ? null : r.interval_rel_s.map(v => e.segment_start_sim_s + v);
            if (!equal(r.interval_sim_s, expected)) fail('interval_sim_s', 'incorrect translation');
            if (r.interval_rel_s !== null && r.interval_rel_s[1] > (cutoff === null ? observedEnd : cutoff) + 0.000001) fail('interval_rel_s', 'outside observed prefix');
            formatted[name] = records.formatMetricRecord(e.schema, record);
        }
        return {state: 'available', source: frozen ? 'durable_terminal' : 'live_provisional',
            provisional: !frozen, frozen, historical: false, identity: e.identity,
            segmentId: e.segment_id, presentationId: e.presentation_id, presentationIndex: e.presentation_index,
            segmentStartSimS: e.segment_start_sim_s, presentationStartRelS: e.presentation_start_rel_s,
            presentationElapsedS: e.presentation_elapsed_s, measurementEndRelS: cutoff,
            measurementEndSimS: e.measurement_end_sim_s, validity: e.validity,
            completeness: e.completeness, endReason: e.end_reason, terminalEvent: e.terminal_event,
            records: formatted, evidence: e.evidence, provenance: e.provenance, envelope: e};
    }
    function unavailable(reason, error = null) { return {state: 'unavailable', reason, error}; }
    function guarded(value, operation, absent) {
        if (value === null || value === undefined) return unavailable(absent);
        try { return operation(value); } catch (error) { return unavailable('rejected_transport', String(error.message)); }
    }
    function buildObservationDisplay(input) {
        const invalid = error => ({state: 'unavailable', reason: 'invalid_current_identity', error,
            currentValidity: null, live: unavailable('invalid_current_identity'), terminal: unavailable('invalid_current_identity')});
        if (!plain(input)) return invalid('input: requires plain context object');
        for (const key of ['identity', 'brainId', 'observation', 'observationPublication']) {
            const descriptor = Object.getOwnPropertyDescriptor(input, key);
            if (descriptor && !own(descriptor, 'value')) return invalid(`input.${key}: accessors are not JSON`);
        }
        const current = own(input, 'identity') ? input.identity : undefined;
        const brainId = own(input, 'brainId') ? input.brainId : undefined;
        const observation = own(input, 'observation') ? input.observation : null;
        const observationPublication = own(input, 'observationPublication') ? input.observationPublication : null;
        let currentIdentity;
        try {
            currentIdentity = detach(current, 'currentIdentity'); identity(currentIdentity, 'currentIdentity');
            text(brainId, 'brainId'); if (brainId !== currentIdentity.brain_id) fail('brainId', 'does not match current identity');
        } catch (error) {
            return invalid(String(error.message));
        }
        const live = guarded(observation, value => {
            const result = envelope(value, false);
            if (!CONTEXT_ID.every(k => result.identity[k] === currentIdentity[k])) fail('live.identity', 'does not match current packet');
            return result;
        }, 'no_live_observation');
        const terminal = guarded(observationPublication, publication => {
            const p = detach(publication, 'publication'); object(p, 'publication');
            if (p.last_terminal === null || p.last_terminal === undefined) return unavailable('no_durable_terminal');
            const wrapper = p.last_terminal; object(wrapper, 'last_terminal');
            const result = envelope(wrapper.observation, true);
            if (!OWNER.every(k => result.identity[k] === currentIdentity[k])) fail('terminal.owner', 'different current owner');
            if (!equal(wrapper.identity, result.identity)) fail('terminal.identity', 'wrapper mismatch');
            const expectedKey = {
                daemon_run_id: result.identity.daemon_run_id,
                run_id: result.identity.run_id,
                instance_id: result.identity.instance_id,
                segment_id: result.segmentId,
                presentation_id: result.presentationId,
            };
            if (!equal(wrapper.observation_key, expectedKey)) fail('observation_key', 'incomplete or mismatched key');
            if (typeof wrapper.payload_sha256 !== 'string' || !/^[a-f0-9]{64}$/.test(wrapper.payload_sha256)) fail('payload_sha256', 'requires canonical digest');
            const receipt = wrapper.receipt; object(receipt, 'receipt');
            if (receipt.durable !== true || !equal(receipt.observation_key, expectedKey) || receipt.payload_sha256 !== wrapper.payload_sha256) fail('receipt', 'not consistent durable acknowledgement');
            text(receipt.file, 'receipt.file');
            if (['.', '..'].includes(receipt.file) || /[/\\]/.test(receipt.file)) fail('receipt.file', 'requires ledger basename');
            integer(receipt.line, 'receipt.line', 1); integer(receipt.offset, 'receipt.offset'); bool(receipt.idempotent, 'receipt.idempotent');
            result.historical = !CONTEXT_ID.every(k => result.identity[k] === currentIdentity[k]) ? true
                : live.state === 'available' ? (result.segmentId !== live.segmentId || result.presentationId !== live.presentationId) : null;
            result.contextLabel = result.historical === true ? 'Historical saved terminal'
                : result.historical === false ? 'Current-context saved terminal' : 'Saved terminal; current segment unavailable';
            result.receipt = receipt; result.observationKey = wrapper.observation_key; result.payloadSha256 = wrapper.payload_sha256;
            result.receiptCheck = 'consistent_transport_only'; return result;
        }, 'no_publication');
        return {state: 'available', currentIdentity, currentValidity: live.state === 'available' ? live.validity : null, live, terminal};
    }
    return Object.freeze({CONTRACT, ASSAYS, buildObservationDisplay});
}));
