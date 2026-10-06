"""Pure assembly and validation for metric-contract/1.2 observation envelopes.

The producer owns measurement state and relative timing.  The daemon-side caller
supplies only packet/run context and remains responsible for locking and freezing.
"""

import copy
import json
import math

from online_metrics import (
    ASSAY_SPEC_VERSIONS,
    CAPABILITIES,
    CONTRACT_VERSION,
    END_REASONS,
    EVIDENCE_TYPES,
    KINDS,
    MODES,
    REASONS,
    SAMPLE_POINTS,
    SCHEMA,
    SCOPES,
    STATES,
    UNITS,
    WINDOW_SOURCES,
    completeness_for,
    s_to_us,
    us_to_s,
    validate_observation_config,
)


VALIDITIES = frozenset({'valid', 'invalidated', 'exploratory_degraded'})
COMPLETENESS = frozenset({'complete', 'incomplete'})

_PRODUCER_KEYS = frozenset({
    'schema', 'contract', 'assay', 'spec_version', 'mode', 'config_id',
    'effective_window_s', 'window_s', 'automatic_end', 'window_source',
    'override', 'hold_s', 'window_params', 'window_end_rel_s',
    'light_schedule', 'sample_point', 'dt_s', 'dt_fixed',
    'presentation_index', 'presentation_start_rel_s',
    'presentation_elapsed_s', 'state', 'completeness', 'end_reason',
    'terminal_event', 'measurement_end_rel_s', 'records', 'evidence',
})
_RECORD_KEYS = frozenset({
    'value', 'available', 'reason', 'note', 'kind', 'unit', 'counts',
    'numerator', 'denominator', 'interval_rel_s', 'evidence_ref',
    'capability', 'scope', 'final',
})
_NUMERIC_KINDS = frozenset({
    'count', 'duration', 'integral', 'latency', 'ratio', 'index', 'fraction',
    'kinematic', 'mean',
})
_CONTEXT_KEYS = frozenset({
    'identity', 'provenance', 'segment_id', 'presentation_id',
    'segment_start_sim_s', 'validity', 'measurement_end_sim_s',
    'terminal_pose_post_step',
})
_ASSAY_ID_TO_PRODUCER = {
    'open-arena': 'open_arena',
    't-maze': 't_maze',
    'y-maze': 'y_maze',
    'heat-maze': 'heat_maze',
    'buridan': 'buridan',
    'visual-operant': 'visual_operant',
    'wind-tunnel': 'wind_tunnel',
    'looming-escape': 'looming_escape',
    'optomotor': 'optomotor',
    'gap-crossing': 'gap_crossing',
    'circadian-dam': 'circadian_dam',
    'courtship': 'courtship',
    'labyrinth': 'labyrinth',
    'multisensory-sandbox': 'multisensory',
}


class ObservationEnvelopeError(ValueError):
    """A snapshot or daemon context violates metric-contract/1.2."""


def _fail(path, message):
    raise ObservationEnvelopeError(f'{path}: {message}')


def _json_value(value, path):
    """Validate a detached JSON-shaped value and all finite numeric leaves."""
    if value is None or isinstance(value, (bool, str)):
        return
    if isinstance(value, (int, float)):
        if not math.isfinite(value):
            _fail(path, 'number must be finite')
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _json_value(item, f'{path}[{index}]')
        return
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                _fail(path, 'object keys must be strings')
            _json_value(item, f'{path}.{key}')
        return
    _fail(path, f'unsupported JSON type {type(value).__name__}')


def _number(value, path, *, nullable=False, integer=False):
    if value is None and nullable:
        return
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        _fail(path, 'must be a number, not a bool' if isinstance(value, bool) else 'must be a number')
    if not math.isfinite(value):
        _fail(path, 'must be finite')
    if integer and not isinstance(value, int):
        _fail(path, 'must be an integer')


def _string(value, path, *, nullable=False, nonempty=False):
    if value is None and nullable:
        return
    if not isinstance(value, str):
        _fail(path, 'must be a string')
    if nonempty and not value:
        _fail(path, 'must not be empty')


def _interval(value, path):
    if value is None:
        return
    if not isinstance(value, list) or len(value) != 2:
        _fail(path, 'must be null or a two-number array')
    _number(value[0], f'{path}[0]')
    _number(value[1], f'{path}[1]')
    if value[1] < value[0]:
        _fail(path, 'end must be greater than or equal to start')


def _validate_identity(identity, producer_assay):
    if not isinstance(identity, dict):
        _fail('identity', 'must be an object')
    _json_value(identity, 'identity')
    for field in ('run_id', 'instance_id', 'assay', 'backend', 'controller_version',
                  'daemon_run_id', 'brain_id'):
        _string(identity.get(field), f'identity.{field}', nonempty=True)
    _number(identity.get('activation'), 'identity.activation', integer=True)
    for field in ('synthetic', 'test_mode'):
        if not isinstance(identity.get(field), bool):
            _fail(f'identity.{field}', 'must be a bool')
    declared = _ASSAY_ID_TO_PRODUCER.get(identity['assay'])
    if declared is None:
        _fail('identity.assay', f'unknown registry assay id {identity["assay"]!r}')
    if declared != producer_assay:
        _fail('identity.assay', f'declares {declared!r}, envelope is {producer_assay!r}')


def _validate_provenance(provenance):
    if not isinstance(provenance, dict):
        _fail('provenance', 'must be an object')
    _json_value(provenance, 'provenance')
    required = ('gf_source', 'stimulus_entry_stage', 'motor_assists_enabled', 'controller_states')
    missing = [key for key in required if key not in provenance]
    if missing:
        _fail('provenance', f'missing required fields {missing}')
    _string(provenance['gf_source'], 'provenance.gf_source', nullable=True)
    _string(provenance['stimulus_entry_stage'], 'provenance.stimulus_entry_stage', nullable=True)
    if not isinstance(provenance['motor_assists_enabled'], bool):
        _fail('provenance.motor_assists_enabled', 'must be a bool')
    states = provenance['controller_states']
    if not isinstance(states, list) or any(not isinstance(state, str) or not state for state in states):
        _fail('provenance.controller_states', 'must be an array of non-empty strings')


def _keys(value, required, path):
    if not isinstance(value, dict) or set(value) != set(required):
        _fail(path, f'must contain exactly {list(required)}')


def _label(value, path, *, nullable=False):
    _string(value, path, nullable=nullable)


def _step(value, path):
    _number(value, path, integer=True)


def _pose(data, path):
    fields = ('x_mm', 'y_mm', 'heading_rad', 't_rel_s', 'step', 'sample_point')
    _keys(data, fields, path)
    for field in fields[:4]:
        _number(data[field], f'{path}.{field}')
    _step(data['step'], f'{path}.step')
    if data['sample_point'] not in SAMPLE_POINTS:
        _fail(f'{path}.sample_point', 'is not in the closed vocabulary')


def _event_sequence(data, path):
    if not isinstance(data, list):
        _fail(path, 'must be an array')
    for index, event_item in enumerate(data):
        item_path = f'{path}[{index}]'
        _keys(event_item, ('event', 't_rel_s', 'step', 'sample_point', 'attrs'), item_path)
        _label(event_item['event'], f'{item_path}.event')
        _number(event_item['t_rel_s'], f'{item_path}.t_rel_s')
        _step(event_item['step'], f'{item_path}.step')
        if event_item['sample_point'] not in SAMPLE_POINTS:
            _fail(f'{item_path}.sample_point', 'is not in the closed vocabulary')
        if not isinstance(event_item['attrs'], dict):
            _fail(f'{item_path}.attrs', 'must be an object')
        for name, value in event_item['attrs'].items():
            _string(name, f'{item_path}.attrs key', nonempty=True)
            if isinstance(value, bool) or not isinstance(value, (int, float, str)):
                _fail(f'{item_path}.attrs.{name}', 'must be a finite number or label')
            if isinstance(value, (int, float)):
                _number(value, f'{item_path}.attrs.{name}')


def _sequence(data, path):
    _keys(data, ('symbols', 'collapse'), path)
    if not isinstance(data['symbols'], list):
        _fail(f'{path}.symbols', 'must be an array of labels')
    for index, symbol in enumerate(data['symbols']):
        _label(symbol, f'{path}.symbols[{index}]')
    if data['collapse'] not in ('none', 'consecutive_repeats'):
        _fail(f'{path}.collapse', 'must be none or consecutive_repeats')


def _bins(data, path):
    _keys(data, ('bin_s', 'start_rel_s', 'quantity', 'counts'), path)
    _number(data['bin_s'], f'{path}.bin_s')
    _number(data['start_rel_s'], f'{path}.start_rel_s')
    _label(data['quantity'], f'{path}.quantity')
    if not isinstance(data['counts'], list):
        _fail(f'{path}.counts', 'must be an array of finite numbers')
    for index, count in enumerate(data['counts']):
        _number(count, f'{path}.counts[{index}]')


def _phase_boundaries(data, path):
    if not isinstance(data, list):
        _fail(path, 'must be an array')
    prior_end = None
    for index, phase in enumerate(data):
        item_path = f'{path}[{index}]'
        _keys(phase, ('phase', 'start_rel_s', 'end_rel_s'), item_path)
        _label(phase['phase'], f'{item_path}.phase')
        interval = [phase['start_rel_s'], phase['end_rel_s']]
        _interval(interval, item_path)
        if prior_end is not None and phase['start_rel_s'] < prior_end:
            _fail(item_path, 'phase intervals must be ordered and non-overlapping')
        prior_end = phase['end_rel_s']


def _intervention_log(data, path):
    if not isinstance(data, list):
        _fail(path, 'must be an array')
    for index, intervention in enumerate(data):
        item_path = f'{path}[{index}]'
        _keys(intervention, ('name', 'value', 't_rel_s', 'step'), item_path)
        _label(intervention['name'], f'{item_path}.name')
        value = intervention['value']
        if value is not None and not isinstance(value, (bool, int, float, str)):
            _fail(f'{item_path}.value', 'must be a scalar or null')
        _json_value(value, f'{item_path}.value')
        _number(intervention['t_rel_s'], f'{item_path}.t_rel_s')
        _step(intervention['step'], f'{item_path}.step')


_EVIDENCE_VALIDATORS = {
    'pose': _pose,
    'event_sequence': _event_sequence,
    'sequence': _sequence,
    'bins': _bins,
    'phase_boundaries': _phase_boundaries,
    'intervention_log': _intervention_log,
}


def _validate_evidence(evidence_items):
    if not isinstance(evidence_items, dict):
        _fail('evidence', 'must be an object')
    for name, item in evidence_items.items():
        _string(name, 'evidence key', nonempty=True)
        if not isinstance(item, dict) or set(item) != {'type', 'data'}:
            _fail(f'evidence.{name}', 'must contain exactly type and data')
        if item['type'] not in EVIDENCE_TYPES:
            _fail(f'evidence.{name}.type', 'is not in the closed vocabulary')
        data_path = f'evidence.{name}.data'
        _json_value(item['data'], data_path)
        _EVIDENCE_VALIDATORS[item['type']](item['data'], data_path)


def _validate_record(name, rec, evidence_items):
    path = f'records.{name}'
    if not isinstance(rec, dict):
        _fail(path, 'must be an object')
    missing = sorted(_RECORD_KEYS - set(rec))
    if missing:
        _fail(path, f'missing fields {missing}')
    if rec['kind'] not in KINDS:
        _fail(f'{path}.kind', 'is not in the closed vocabulary')
    if rec['unit'] not in UNITS:
        _fail(f'{path}.unit', 'is not in the closed vocabulary')
    if rec['capability'] not in CAPABILITIES:
        _fail(f'{path}.capability', 'is not in the closed vocabulary')
    if rec['scope'] not in SCOPES:
        _fail(f'{path}.scope', 'is not in the closed vocabulary')
    if not isinstance(rec['available'], bool) or not isinstance(rec['final'], bool):
        _fail(path, 'available and final must be bools')
    if rec['available']:
        if rec['value'] is None or rec['reason'] is not None:
            _fail(path, 'available requires a non-null value and null reason')
    elif rec['value'] is not None or rec['reason'] not in REASONS:
        _fail(path, 'unavailable requires null value and a declared reason')
    if rec['capability'] == 'unsupported' and rec['reason'] != 'unsupported':
        _fail(path, 'unsupported capability requires unsupported reason')
    value = rec['value']
    if isinstance(value, (dict, list)):
        _fail(f'{path}.value', 'structured data belongs in evidence')
    if isinstance(value, str) and rec['kind'] != 'label':
        _fail(f'{path}.value', 'strings require kind label')
    if rec['kind'] == 'event' and value is not None and not isinstance(value, bool):
        _fail(f'{path}.value', 'available event values must be bools')
    if rec['kind'] == 'label' and value is not None and not isinstance(value, str):
        _fail(f'{path}.value', 'available label values must be strings')
    if isinstance(value, bool) and rec['kind'] in _NUMERIC_KINDS:
        _fail(f'{path}.value', f'bool is not a numeric {rec["kind"]} value')
    _json_value(value, f'{path}.value')
    _string(rec['note'], f'{path}.note', nullable=True)
    if not isinstance(rec['counts'], dict):
        _fail(f'{path}.counts', 'must be an object')
    _json_value(rec['counts'], f'{path}.counts')
    for field in ('numerator', 'denominator'):
        if rec[field] is not None:
            _number(rec[field], f'{path}.{field}')
    ratio_kinds = {'ratio', 'index', 'fraction'}
    if rec['kind'] not in ratio_kinds and (rec['numerator'] is not None or rec['denominator'] is not None):
        _fail(path, 'numerator and denominator belong only to ratio, index or fraction records')
    if rec['reason'] == 'zero_denominator':
        if rec['kind'] not in ratio_kinds or rec['denominator'] != 0 or rec['numerator'] is None:
            _fail(path, 'zero_denominator requires a ratio/index/fraction with numerator and denominator 0')
    _interval(rec['interval_rel_s'], f'{path}.interval_rel_s')
    _string(rec['evidence_ref'], f'{path}.evidence_ref', nullable=True)
    if rec['evidence_ref'] is not None and rec['evidence_ref'] not in evidence_items:
        _fail(f'{path}.evidence_ref', 'does not name published evidence')


def _validate_snapshot(snapshot):
    if not isinstance(snapshot, dict):
        _fail('snapshot', 'must be an object')
    missing = sorted(_PRODUCER_KEYS - set(snapshot))
    if missing:
        _fail('snapshot', f'missing producer fields {missing}')
    collisions = sorted(_CONTEXT_KEYS & set(snapshot))
    if collisions:
        _fail('snapshot', f'producer snapshot contains daemon-owned fields {collisions}')
    if snapshot['schema'] != SCHEMA or snapshot['contract'] != CONTRACT_VERSION:
        _fail('schema', f'unsupported schema/contract {snapshot["schema"]!r}, {snapshot["contract"]!r}')
    assay = snapshot['assay']
    if assay not in ASSAY_SPEC_VERSIONS:
        _fail('assay', f'unsupported assay {assay!r}')
    if snapshot['spec_version'] != ASSAY_SPEC_VERSIONS[assay]:
        _fail('spec_version', f'does not match {assay}')
    if snapshot['mode'] not in MODES or snapshot['state'] not in STATES:
        _fail('snapshot', 'mode or state is not in the closed vocabulary')
    if snapshot['sample_point'] not in SAMPLE_POINTS:
        _fail('sample_point', 'is not in the closed vocabulary')
    if snapshot['window_source'] not in WINDOW_SOURCES:
        _fail('window_source', 'is not in the closed vocabulary')
    if not isinstance(snapshot['automatic_end'], bool) or not isinstance(snapshot['dt_fixed'], bool):
        _fail('snapshot', 'automatic_end and dt_fixed must be bools')
    _string(snapshot['config_id'], 'config_id', nonempty=True)
    validate_observation_config({key: snapshot[key] for key in (
        'config_id', 'effective_window_s', 'automatic_end', 'window_source', 'override', 'hold_s')})
    if snapshot['override'] is not None:
        _number(snapshot['override']['set_at_sim_s'], 'override.set_at_sim_s')
        if snapshot['override']['value'] is not None:
            _number(snapshot['override']['value'], 'override.value')
    if snapshot['window_s'] != snapshot['effective_window_s']:
        _fail('window_s', 'must equal effective_window_s')
    for field in ('window_end_rel_s', 'dt_s', 'measurement_end_rel_s'):
        _number(snapshot[field], field, nullable=True)
    for field in ('presentation_start_rel_s', 'presentation_elapsed_s'):
        _number(snapshot[field], field)
    _number(snapshot['presentation_index'], 'presentation_index', integer=True)
    _json_value(snapshot['window_params'], 'window_params')
    _json_value(snapshot['light_schedule'], 'light_schedule')
    _string(snapshot['terminal_event'], 'terminal_event', nullable=True)
    if snapshot['end_reason'] is not None and not (
            snapshot['end_reason'] in END_REASONS
            or (isinstance(snapshot['end_reason'], str)
                and snapshot['end_reason'].startswith('terminal_event:')
                and len(snapshot['end_reason']) > len('terminal_event:'))):
        _fail('end_reason', 'is not in the closed vocabulary')
    _validate_evidence(snapshot['evidence'])
    if not isinstance(snapshot['records'], dict) or not snapshot['records']:
        _fail('records', 'must be a non-empty object')
    for name, rec in snapshot['records'].items():
        _string(name, 'record key', nonempty=True)
        if 'interval_sim_s' in rec:
            _fail(f'records.{name}.interval_sim_s', 'is daemon-owned and not valid in a producer snapshot')
        _validate_record(name, rec, snapshot['evidence'])
    finals = {rec['final'] for rec in snapshot['records'].values()}
    if len(finals) != 1:
        _fail('records', 'cannot mix provisional and final records')
    frozen = finals == {True}
    if frozen:
        if snapshot['completeness'] not in COMPLETENESS or snapshot['end_reason'] is None:
            _fail('snapshot', 'frozen records require completeness and end_reason')
        if snapshot['measurement_end_rel_s'] is None:
            _fail('measurement_end_rel_s', 'frozen records require a cutoff')
    elif snapshot['completeness'] is not None:
        _fail('completeness', 'live provisional records have no completion receipt')
    return frozen


def _validate_lifecycle(snapshot, frozen, validity):
    reason = snapshot['end_reason']
    if reason == 'fault_halt' and validity != 'invalidated':
        _fail('validity', 'fault_halt requires invalidated validity')
    _validate_chronology(snapshot)
    if not frozen:
        if snapshot['state'] == 'observing':
            if (snapshot['end_reason'] is not None or snapshot['terminal_event'] is not None
                    or snapshot['measurement_end_rel_s'] is not None):
                _fail('state', 'an observing live snapshot cannot carry an end receipt')
        elif snapshot['end_reason'] is None or snapshot['measurement_end_rel_s'] is None:
            _fail('state', 'an ended live snapshot requires end_reason and measurement_end_rel_s')
        if isinstance(snapshot['end_reason'], str) and snapshot['end_reason'].startswith('terminal_event:'):
            expected_event = snapshot['end_reason'].split(':', 1)[1]
            if snapshot['terminal_event'] != expected_event:
                _fail('terminal_event', 'must agree with terminal_event:<name> end_reason')
        return
    if isinstance(reason, str) and reason.startswith('terminal_event:'):
        expected_event = reason.split(':', 1)[1]
        if snapshot['terminal_event'] != expected_event:
            _fail('terminal_event', 'must agree with terminal_event:<name> end_reason')
    window_elapsed = (
        snapshot['automatic_end']
        and snapshot['effective_window_s'] is not None
        and s_to_us(snapshot['presentation_elapsed_s']) >= s_to_us(snapshot['effective_window_s'])
    )
    expected = completeness_for(reason, window_elapsed, snapshot['automatic_end'])
    if snapshot['completeness'] != expected:
        _fail('completeness', f'{reason!r} requires {expected!r}')


def _validate_chronology(snapshot):
    """Check producer-relative clocks using the producer's integer-microsecond convention."""
    start = snapshot['presentation_start_rel_s']
    elapsed = snapshot['presentation_elapsed_s']
    if start < 0 or elapsed < 0:
        _fail('presentation_elapsed_s', 'presentation clocks must be nonnegative')
    start_us, elapsed_us = s_to_us(start), s_to_us(elapsed)
    cutoff = snapshot['measurement_end_rel_s']
    if cutoff is not None:
        cutoff_us = s_to_us(cutoff)
        if cutoff_us < start_us or cutoff_us > start_us + elapsed_us:
            _fail('measurement_end_rel_s', 'must lie within the observed presentation clock range')

    if snapshot['end_reason'] != 'window_elapsed':
        return
    if not snapshot['automatic_end'] or snapshot['effective_window_s'] is None:
        _fail('end_reason', 'window_elapsed requires an automatic finite window')
    window_us = s_to_us(snapshot['effective_window_s'])
    deadline = us_to_s(start_us + window_us)
    if elapsed_us < window_us:
        _fail('presentation_elapsed_s', 'window_elapsed requires the effective deadline to be reached')
    if snapshot['window_end_rel_s'] != deadline:
        _fail('window_end_rel_s', 'does not match the producer-rounded effective deadline')
    if cutoff != deadline:
        _fail('measurement_end_rel_s', 'window_elapsed cutoff must equal the effective deadline')
    if snapshot['state'] != 'closed':
        _fail('state', 'window_elapsed requires the producer closed state')
    if snapshot['terminal_event'] is not None:
        _fail('terminal_event', 'window_elapsed cannot carry a terminal event')


def _validate_terminal_pose(pose, frozen):
    if not frozen:
        if pose is not None:
            _fail('terminal_pose_post_step', 'live provisional envelopes cannot carry a terminal pose')
        return
    if not isinstance(pose, dict):
        _fail('terminal_pose_post_step', 'frozen envelopes require a finite pose object')
    _json_value(pose, 'terminal_pose_post_step')
    for field in ('x_mm', 'y_mm', 'heading_rad'):
        if field not in pose:
            _fail('terminal_pose_post_step', f'is missing {field}')
        _number(pose[field], f'terminal_pose_post_step.{field}')


def build_observation_envelope(snapshot, *, identity, provenance, segment_id,
                               segment_start_sim_s, validity,
                               terminal_pose_post_step=None):
    """Return a detached, validated v1.2 envelope from producer and daemon facts."""
    producer = copy.deepcopy(snapshot)
    frozen = _validate_snapshot(producer)
    _validate_identity(identity, producer['assay'])
    _validate_provenance(provenance)
    _string(segment_id, 'segment_id', nonempty=True)
    _number(segment_start_sim_s, 'segment_start_sim_s')
    if validity not in VALIDITIES:
        _fail('validity', 'is not in the closed vocabulary')
    _validate_lifecycle(producer, frozen, validity)
    _validate_terminal_pose(terminal_pose_post_step, frozen)

    result = producer
    result.update({
        'identity': copy.deepcopy(identity),
        'provenance': copy.deepcopy(provenance),
        'segment_id': segment_id,
        'presentation_id': f'{segment_id}:{producer["presentation_index"]}',
        'segment_start_sim_s': segment_start_sim_s,
        'validity': validity,
        'measurement_end_sim_s': None,
        'terminal_pose_post_step': copy.deepcopy(terminal_pose_post_step),
    })
    end_rel = producer['measurement_end_rel_s']
    if end_rel is not None:
        end_sim = segment_start_sim_s + end_rel
        if not math.isfinite(end_sim):
            _fail('measurement_end_sim_s', 'origin plus relative time is non-finite')
        result['measurement_end_sim_s'] = end_sim
    for name, rec in result['records'].items():
        rel = rec['interval_rel_s']
        if rel is None:
            rec['interval_sim_s'] = None
            continue
        absolute = [segment_start_sim_s + rel[0], segment_start_sim_s + rel[1]]
        if not all(math.isfinite(value) for value in absolute):
            _fail(f'records.{name}.interval_sim_s', 'origin plus relative time is non-finite')
        rec['interval_sim_s'] = absolute
    _json_value(result, 'envelope')
    return result


def validate_observation_envelope(envelope):
    """Validate an already stamped envelope and return a detached copy."""
    if not isinstance(envelope, dict):
        _fail('envelope', 'must be an object')
    required = ('identity', 'provenance', 'segment_id', 'presentation_id',
                'segment_start_sim_s', 'validity', 'measurement_end_sim_s',
                'terminal_pose_post_step')
    missing = [key for key in required if key not in envelope]
    if missing:
        _fail('envelope', f'missing context fields {missing}')
    unstamped = copy.deepcopy(envelope)
    context = {key: unstamped.pop(key) for key in required}
    for rec in unstamped.get('records', {}).values():
        rec.pop('interval_sim_s', None)
    expected = build_observation_envelope(
        unstamped, identity=context['identity'], provenance=context['provenance'],
        segment_id=context['segment_id'], segment_start_sim_s=context['segment_start_sim_s'],
        validity=context['validity'], terminal_pose_post_step=context['terminal_pose_post_step'])
    if context['presentation_id'] != expected['presentation_id']:
        _fail('presentation_id', 'does not match segment_id and presentation_index')
    supplied_end = context['measurement_end_sim_s']
    expected_end = expected['measurement_end_sim_s']
    if (supplied_end is None) != (expected_end is None):
        _fail('measurement_end_sim_s', 'does not match measurement_end_rel_s')
    if supplied_end is not None:
        _number(supplied_end, 'measurement_end_sim_s')
        if supplied_end != expected_end:
            _fail('measurement_end_sim_s', 'does not match segment origin plus relative time')
    for name, supplied in envelope['records'].items():
        if 'interval_sim_s' not in supplied:
            _fail(f'records.{name}.interval_sim_s', 'is required in a stamped envelope')
        expected_interval = expected['records'][name]['interval_sim_s']
        supplied_interval = supplied['interval_sim_s']
        if (supplied_interval is None) != (expected_interval is None):
            _fail(f'records.{name}.interval_sim_s', 'does not match interval_rel_s')
        if supplied_interval is not None:
            _interval(supplied_interval, f'records.{name}.interval_sim_s')
            if supplied_interval != expected_interval:
                _fail(f'records.{name}.interval_sim_s', 'does not match segment origin plus relative time')
    return copy.deepcopy(envelope)


def dumps_observation_envelope(envelope, **kwargs):
    """Validate and serialize without non-finite coercion or a string fallback."""
    if 'default' in kwargs:
        _fail('serialization.default', 'fallback coercion is forbidden')
    if kwargs.get('allow_nan') is not None:
        _fail('serialization.allow_nan', 'is fixed false')
    validated = validate_observation_envelope(envelope)
    return json.dumps(validated, allow_nan=False, **kwargs)
