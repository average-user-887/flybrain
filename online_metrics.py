"""Bounded display histories with cumulative scientific statistics.

Continuous observation must not rescan an ever-growing trajectory every tick.
Evicting display samples does not discard the cumulative distance or mean.
"""
from collections import deque
import math


class ScalarHistory(deque):
    def __init__(self, maxlen=2048):
        super().__init__(maxlen=maxlen)
        self.count = 0
        self.total = 0.0

    def append(self, value):
        self.total += float(value)
        self.count += 1
        super().append(value)

    @property
    def mean(self):
        return self.total / self.count if self.count else 0.0

    def clear(self):
        super().clear()
        self.count = 0
        self.total = 0.0


class PathHistory(deque):
    def __init__(self, maxlen=2048):
        super().__init__(maxlen=maxlen)
        self.origin = None
        self.distance = 0.0
        self.count = 0

    def append(self, point):
        if self:
            self.distance += math.dist(self[-1], point)
        else:
            self.origin = point
        self.count += 1
        super().append(point)

    @property
    def tortuosity(self):
        net = math.dist(self.origin, self[-1]) if self else 0.0
        return self.distance / net if net > 1e-9 else None

    def clear(self):
        super().clear()
        self.origin = None
        self.distance = 0.0
        self.count = 0


# =============================================================================
# Metric contract v1.2 (metric-contract/1.2, schema neurofly.metric/1.2)
#
# Shared vocabularies, record and evidence constructors, recursive finite
# validation, the observation configuration (v1.1 A1), the freeze helper and the
# producer lifecycle used by every assay. docs/ASSAY_SEMANTICS.md defines the
# meaning of each record.
# =============================================================================
import copy

CONTRACT_VERSION = 'metric-contract/1.2'
SCHEMA = 'neurofly.metric/1.2'

# Closed producer identities. These are the assay names emitted by the 13
# registered paradigms plus OpenArenaObserver, not registry aliases or UI ids.
ASSAY_SPEC_VERSIONS = {
    assay: f'{assay}/1.2' for assay in (
        'buridan', 'circadian_dam', 'courtship', 'gap_crossing', 'heat_maze',
        'labyrinth', 'looming_escape', 'multisensory', 'open_arena', 'optomotor',
        't_maze', 'visual_operant', 'wind_tunnel', 'y_maze',
    )
}

REASONS = frozenset({'pending', 'not_observed', 'zero_denominator', 'insufficient_events',
                     'censored', 'unsupported', 'invalidated'})
UNITS = frozenset({'s', 'ms', 'min', 'mm', 'mm/s', 'mm/s^3', 'deg', 'deg/s', 'rad', 'rad/s',
                   'degC', 'degC*s', 'count', 'fraction', 'index', 'ratio', 'gain', 'bool', 'label'})
KINDS = frozenset({'count', 'duration', 'integral', 'latency', 'ratio', 'index', 'fraction',
                   'event', 'label', 'kinematic', 'mean'})
CAPABILITIES = frozenset({'measured', 'derived', 'unsupported'})
SCOPES = frozenset({'segment', 'presentation'})                      # v1.1 A4 (I-20)
EVIDENCE_TYPES = frozenset({'pose', 'event_sequence', 'sequence', 'bins', 'phase_boundaries',
                            'intervention_log'})
MODES = frozenset({'continuous', 'fixed', 'until_terminal', 'presentation'})
STATES = frozenset({'observing', 'measurement_ended', 'closed'})
SAMPLE_POINTS = frozenset({'pre_motor', 'post_solver'})
END_REASONS = frozenset({'window_elapsed', 're_presentation_user', 'policy_change', 'manual_reset',
                         'experiment_selected', 'backend_switch', 'fault_halt', 'shutdown'})
WINDOW_SOURCES = frozenset({'spec', 'override', 'continuous_flag'})
OVERRIDE_SOURCES = frozenset({'cli:--trial-seconds', 'cli:--continuous', 'api'})
# The outcome booleans that obey the deadline-outcome invariant (v1.1 A5).
OUTCOME_BOOLEANS = frozenset({'crossing_success', 'refuge_reached', 'source_reached', 'goal_reached',
                              'escape_initiated', 'escape_completed', 'turn_complete'})
EVIDENCE_EVENT_CAP = 1000  # event_sequence evidence keeps the first N events (counts stay exact)

_US = 1_000_000


class MetricFault(ValueError):
    """An unexpected non-finite number in a measurement, provenance or evidence item.

    ``path`` names the first offending field. The daemon applies the shared F policy.
    """

    def __init__(self, path, message=None):
        self.path = path
        super().__init__(message or f'non-finite number at {path}')


class ConfigError(ValueError):
    """An invalid ObservationConfig, or one delivered after observation started (v1.1 A1.4)."""


def validate_finite(obj, path=''):
    """Return the path of the first non-finite number inside ``obj`` (recursively), else None."""
    if isinstance(obj, bool) or obj is None or isinstance(obj, str):
        return None
    if isinstance(obj, (int, float)):
        return None if math.isfinite(obj) else (path or '<root>')
    if isinstance(obj, dict):
        for key, value in obj.items():
            found = validate_finite(value, f'{path}.{key}' if path else str(key))
            if found:
                return found
        return None
    if isinstance(obj, (list, tuple)):
        for i, value in enumerate(obj):
            found = validate_finite(value, f'{path}[{i}]')
            if found:
                return found
        return None
    try:  # numpy scalars and similar
        value = float(obj)
    except (TypeError, ValueError):
        raise TypeError(f'{path or "<root>"}: unsupported type {type(obj).__name__} in a metric') from None
    return None if math.isfinite(value) else (path or '<root>')


def _require_finite(obj, prefix=''):
    found = validate_finite(obj, prefix)
    if found:
        raise MetricFault(found)


def _check_vocab(name, value, vocab):
    if value not in vocab:
        raise ValueError(f'{name} {value!r} is not in the closed metric-contract vocabulary')


def us_to_s(us):
    return us / _US


def s_to_us(seconds):
    if not math.isfinite(seconds):
        raise MetricFault('dt')
    return int(round(seconds * _US))


def record(value=None, *, kind, unit, reason=None, note=None, counts=None, numerator=None,
           denominator=None, interval_rel_s=None, evidence_ref=None, capability='measured', scope=None):
    """Build one metric record (contract §2.1). ``reason`` given means unavailable.

    ``scope`` (v1.1 A4) is 'segment' or 'presentation'; when None the producer
    lifecycle stamps the assay's default scope before publishing.
    """
    _check_vocab('kind', kind, KINDS)
    _check_vocab('unit', unit, UNITS)
    _check_vocab('capability', capability, CAPABILITIES)
    if scope is not None:
        _check_vocab('scope', scope, SCOPES)
    if reason is not None:
        _check_vocab('reason', reason, REASONS)
        value = None
    elif value is None:
        raise ValueError('an available record needs a value; give a reason instead')
    if isinstance(value, (list, tuple, dict, set)):
        raise TypeError('structured data belongs in evidence, not in a record value')
    if isinstance(value, str) and kind != 'label':
        raise TypeError('string values are allowed only for kind "label"')
    if capability == 'unsupported' and reason != 'unsupported':
        raise ValueError('an unsupported capability must carry reason "unsupported"')
    rec = {
        'value': value,
        'available': reason is None,
        'reason': reason,
        'note': note,
        'kind': kind,
        'unit': unit,
        'counts': dict(counts or {}),
        'numerator': numerator,
        'denominator': denominator,
        'interval_rel_s': list(interval_rel_s) if interval_rel_s is not None else None,
        'evidence_ref': evidence_ref,
        'capability': capability,
        'scope': scope,
        'final': False,
    }
    _require_finite(rec)
    return rec


def ratio(numerator, denominator, *, kind='ratio', unit='ratio', counts=None, note=None,
          interval_rel_s=None, reason_if_zero='zero_denominator', capability='measured', scope=None):
    """A derived ratio, index or fraction. A zero denominator is unavailable, never 0 or 1000."""
    if denominator == 0:
        return record(kind=kind, unit=unit, reason=reason_if_zero, counts=counts, note=note,
                      numerator=numerator, denominator=denominator,
                      interval_rel_s=interval_rel_s, capability=capability, scope=scope)
    return record(numerator / denominator, kind=kind, unit=unit, counts=counts, note=note,
                  numerator=numerator, denominator=denominator,
                  interval_rel_s=interval_rel_s, capability=capability, scope=scope)


def unsupported(kind, unit, note, interval_rel_s=None, scope=None):
    return record(kind=kind, unit=unit, reason='unsupported', note=note,
                  interval_rel_s=interval_rel_s, capability='unsupported', scope=scope)


def evidence(type_, data):
    """One typed evidence item (contract §2.4)."""
    _check_vocab('evidence type', type_, EVIDENCE_TYPES)
    item = {'type': type_, 'data': copy.deepcopy(data)}
    _require_finite(item, 'evidence')
    return item


def pose_data(x, y, heading, t_rel_s, step, sample_point='pre_motor'):
    _check_vocab('sample_point', sample_point, SAMPLE_POINTS)
    return {'x_mm': float(x), 'y_mm': float(y), 'heading_rad': float(heading),
            't_rel_s': float(t_rel_s), 'step': int(step), 'sample_point': sample_point}


def event(name, t_rel_s, step, sample_point='pre_motor', **attrs):
    _check_vocab('sample_point', sample_point, SAMPLE_POINTS)
    return {'event': name, 't_rel_s': float(t_rel_s), 'step': int(step),
            'sample_point': sample_point, 'attrs': dict(attrs)}


def freeze_records(records):
    """The only place ``final`` becomes true: a deep, frozen copy of the records."""
    frozen = copy.deepcopy(records)
    for rec in frozen.values():
        rec['final'] = True
    return frozen


def completeness_for(end_reason, window_elapsed=False, automatic_end=True):
    """Contract §5.5 plus v1.1 A1.5/A4.3. The daemon owns the published value; this is the shared rule.

    A user re-presentation or a policy change completes the observation only when no
    planned window was cut short: the window had elapsed, or there was no automatic end.
    """
    if end_reason == 'window_elapsed' or str(end_reason).startswith('terminal_event:'):
        return 'complete'
    _check_vocab('end_reason', end_reason, END_REASONS)
    if end_reason == 're_presentation_user':
        return 'complete' if (window_elapsed or not automatic_end) else 'incomplete'
    if end_reason == 'policy_change':
        return 'complete' if window_elapsed else 'incomplete'
    return 'incomplete'


def check_spec(spec):
    """Validate a current static ``observation_spec`` at the producer boundary."""
    schema, contract = spec.get('schema'), spec.get('contract')
    if schema != SCHEMA or contract != CONTRACT_VERSION:
        raise ValueError(f'unsupported metric schema/contract: {schema!r}, {contract!r}')
    assay = spec.get('assay')
    if not isinstance(assay, str) or assay not in ASSAY_SPEC_VERSIONS:
        raise ValueError(f'unsupported assay identity {assay!r}')
    expected_spec = ASSAY_SPEC_VERSIONS[assay]
    if spec.get('spec_version') != expected_spec:
        raise ValueError(f'unsupported assay spec version {spec.get("spec_version")!r}; expected {expected_spec!r}')
    _check_vocab('mode', spec['mode'], MODES)
    if spec['learning_claim'] != 'none':
        raise ValueError('metric-contract/1.2 makes no learning claim for any assay')
    _require_finite(spec, 'spec')
    return spec


# ---- ObservationConfig (v1.1 A1, I-19) ------------------------------------------
_CONFIG_KEYS = ('config_id', 'effective_window_s', 'automatic_end', 'window_source', 'override', 'hold_s')
_OVERRIDE_KEYS = ('source', 'value', 'set_at_sim_s', 'manifest_run_id')


def validate_observation_config(config):
    """Return a normalised copy of an ObservationConfig, or raise ConfigError."""
    if not isinstance(config, dict):
        raise ConfigError('an ObservationConfig is a dict')
    missing = [k for k in _CONFIG_KEYS if k not in config]
    if missing:
        raise ConfigError(f'ObservationConfig is missing {missing}')
    cfg = {k: copy.deepcopy(config[k]) for k in _CONFIG_KEYS}
    if not isinstance(cfg['config_id'], str) or not cfg['config_id']:
        raise ConfigError('config_id must be a non-empty string')
    if not isinstance(cfg['automatic_end'], bool):
        raise ConfigError('automatic_end must be a bool')
    if cfg['window_source'] not in WINDOW_SOURCES:
        raise ConfigError(f'window_source {cfg["window_source"]!r} is not one of {sorted(WINDOW_SOURCES)}')
    window = cfg['effective_window_s']
    if window is not None:
        if isinstance(window, bool) or not isinstance(window, (int, float)) or not math.isfinite(window) \
                or window <= 0:
            raise ConfigError('effective_window_s must be a positive finite number or null')
        cfg['effective_window_s'] = float(window)
    elif cfg['automatic_end']:
        raise ConfigError('effective_window_s may be null only when automatic_end is false')
    hold = cfg['hold_s']
    if isinstance(hold, bool) or not isinstance(hold, (int, float)) or not math.isfinite(hold) or hold < 0:
        raise ConfigError('hold_s must be a finite number >= 0')
    cfg['hold_s'] = float(hold)
    override = cfg['override']
    if override is not None:
        if not isinstance(override, dict) or any(k not in override for k in _OVERRIDE_KEYS):
            raise ConfigError(f'override must be null or a dict with {list(_OVERRIDE_KEYS)}')
        if override['source'] not in OVERRIDE_SOURCES:
            raise ConfigError(f'override source {override["source"]!r} is not one of {sorted(OVERRIDE_SOURCES)}')
        if validate_finite(override):
            raise ConfigError('override numbers must be finite')
    return cfg


def build_observation_config(spec, *, config_id, override_window_s=None, continuous=False,
                             override_source=None, set_at_sim_s=0.0, manifest_run_id=''):
    """The daemon-side helper that turns a spec default plus the CLI/API policy into a config.

    - ``continuous`` (``--continuous``): no automatic end anywhere; window_source 'continuous_flag'.
    - a continuous-mode assay: never an automatic end; a window override is carried in the
      ``override`` block for provenance but not applied (window_source stays 'spec').
    - ``override_window_s`` (``--trial-seconds`` or the API): the effective window.
    """
    override = None
    if continuous:
        override = {'source': override_source or 'cli:--continuous', 'value': None,
                    'set_at_sim_s': float(set_at_sim_s), 'manifest_run_id': str(manifest_run_id)}
        cfg = {'effective_window_s': None, 'automatic_end': False, 'window_source': 'continuous_flag'}
    else:
        if override_window_s is not None:
            override = {'source': override_source or 'cli:--trial-seconds', 'value': float(override_window_s),
                        'set_at_sim_s': float(set_at_sim_s), 'manifest_run_id': str(manifest_run_id)}
        if spec['mode'] == 'continuous' or (spec['window_s'] is None and override_window_s is None):
            cfg = {'effective_window_s': None, 'automatic_end': False, 'window_source': 'spec'}
        elif override_window_s is not None:
            cfg = {'effective_window_s': float(override_window_s), 'automatic_end': True,
                   'window_source': 'override'}
        else:
            cfg = {'effective_window_s': float(spec['window_s']), 'automatic_end': True, 'window_source': 'spec'}
    cfg.update(config_id=str(config_id), override=override, hold_s=float(spec.get('hold_s') or 0.0))
    return validate_observation_config(cfg)


class Sample:
    """One pre-motor observation: segment time before this step's dt is added (contract §4, v1.1 A3.3)."""
    __slots__ = ('step', 't_us', 'dt_us', 'observation_dt_us', 'x', 'y', 'heading', 'speed',
                 'angular_velocity', 'fly')

    @property
    def t_rel_s(self):
        return us_to_s(self.t_us)

    @property
    def dt(self):
        return us_to_s(self.dt_us)

    @property
    def observation_dt(self):
        """Part of the raw controller interval that lies inside the observation window."""
        return us_to_s(self.observation_dt_us)


def _fly_field(fly, name, default=None):
    if isinstance(fly, dict):
        return fly.get(name, default)
    return getattr(fly, name, default)


class ObservationLifecycle:
    """Producer side of the v1.1 lifecycle (§5, A1-A5): clocks, configuration, presentations,
    measurement end and the frozen envelope.

    Subclasses define ``OBSERVATION_SPEC`` and the hooks ``_v1_clear()`` (presentation-scoped
    state), ``_v1_observe(sample)``, ``_v1_records(cut)`` and ``_v1_evidence()``. Optional hooks:
    ``_v1_clear_segment()`` (segment-scoped state), ``_v1_carry()``/``_v1_restore(carry)``
    (physical hysteresis state that survives an in-place presentation change),
    ``_v1_segment_carry()``/``_v1_segment_restore(carry)`` (physical hysteresis state
    that survives a measurement-only segment boundary), and
    ``_v1_on_new_presentation(reason)``. Producers never count steps for a duration and
    never read the global clock.
    """

    OBSERVATION_SPEC = None
    provenance = None

    # ---- state -----------------------------------------------------------------
    def _v1_init(self):
        self._v1_seg_us = 0
        self._v1_pres_index = 0
        self._v1_pres_start_us = 0
        self._v1_pres_samples = 0      # observed (pre-end) samples in this presentation
        self._v1_pres_ticks = 0        # every sample in this presentation, hold included
        self._v1_seg_samples = 0       # observed samples retained across presentations
        self._v1_pending_step = None
        self._v1_auto_step = 0
        self._v1_last_step = None
        self._v1_last_t_us = None
        self._v1_last_dt_us = None
        self._v1_dt_fixed = True
        self._v1_end = None            # {'event', 'end_us', 'reason', 'kind': 'deadline'|'terminal'}
        self._v1_freeze_cut = None     # (cut_us, 'interrupt'|'fault') while an interruption freezes
        self._v1_interventions = []
        self._v1_events_seen = []      # terminal events recorded without ending (automatic_end false)
        if not hasattr(self, '_v1_config'):
            self._v1_config = None     # an explicit ObservationConfig; None = the spec default
        self._v1_clear_segment()
        self._v1_clear()

    def _v1_ready(self):
        if not hasattr(self, '_v1_seg_us'):
            self._v1_init()

    def _v1_clear_segment(self):
        pass

    def _v1_carry(self):
        return None

    def _v1_restore(self, carry):
        pass

    def _v1_segment_carry(self):
        return None

    def _v1_segment_restore(self, carry):
        pass

    def _v1_on_new_observation_segment(self):
        pass

    # ---- declarations -----------------------------------------------------------
    def _v1_window_params(self):
        return {}

    def _v1_window_s(self):
        """The spec default window (a dynamic looming window is recomputed from its parameters)."""
        return self.OBSERVATION_SPEC['window_s']

    def _v1_default_scope(self):
        spec = self.OBSERVATION_SPEC
        return 'presentation' if (spec['mode'] == 'presentation' or spec['re_presentation_triggers']) else 'segment'

    def observation_spec(self):
        spec = copy.deepcopy(self.OBSERVATION_SPEC)
        spec['schema'] = SCHEMA
        spec['contract'] = CONTRACT_VERSION
        spec['window_s'] = self._v1_window_s()
        spec['window_params'] = self._v1_window_params()
        spec['default_scope'] = self._v1_default_scope()
        return check_spec(spec)

    # ---- configuration (v1.1 A1) -------------------------------------------------
    def default_observation_config(self):
        """The spec-default config, used when the daemon has delivered none."""
        spec = self.OBSERVATION_SPEC
        return build_observation_config({'mode': spec['mode'], 'window_s': self._v1_window_s(),
                                         'hold_s': spec.get('hold_s')},
                                        config_id=f'spec-default:{spec["assay"]}')

    def _v1_cfg(self):
        """The effective config (not a copy; the hot path)."""
        return self._v1_config if self._v1_config is not None else self.default_observation_config()

    def observation_config(self):
        self._v1_ready()
        return copy.deepcopy(self._v1_cfg())

    def configure_observation(self, config):
        """I-19: deliver the effective config before the first sample of a segment or presentation.

        After the first sample of the current presentation it raises ConfigError and changes
        nothing; a mid-presentation policy change goes through freeze ('policy_change') and
        ``begin_next_presentation(reason, config)`` instead.
        """
        self._v1_ready()
        cfg = validate_observation_config(config)
        if self._v1_pres_ticks > 0:
            raise ConfigError('configure_observation after the first sample of a presentation; '
                              'freeze with end reason "policy_change" and begin the next presentation')
        self._v1_config = cfg
        return copy.deepcopy(cfg)

    def _v1_apply_config(self, config):
        """Install the config of a new segment/presentation and log changed policy fields."""
        if config is None:
            return
        before = self.observation_config()
        cfg = validate_observation_config(config)
        self._v1_config = cfg
        for field in ('effective_window_s', 'automatic_end', 'window_source', 'hold_s', 'override'):
            if before[field] != cfg[field]:
                value = cfg[field]
                if field == 'override':
                    value = value['source'] if value else None
                self.log_intervention(f'policy:{field}', value)

    def _v1_eff_window_us(self):
        cfg = self._v1_cfg()
        if not cfg['automatic_end'] or cfg['effective_window_s'] is None:
            return None
        return s_to_us(cfg['effective_window_s'])

    # ---- timing -----------------------------------------------------------------
    def begin_sample(self, arena_step, dt):
        """I-8: called by the arena immediately before ``step`` with the arena's step id."""
        self._v1_ready()
        self._v1_pending_step = (int(arena_step), dt)

    def _v1_begin(self, fly, dt, pose):
        """Open the pre-motor sample for this step and return it."""
        self._v1_ready()
        if self._v1_pending_step is not None:
            step, _ = self._v1_pending_step
            self._v1_pending_step = None
        else:
            self._v1_auto_step += 1
            step = self._v1_auto_step
        dt_us = s_to_us(float(dt))
        if dt_us <= 0:
            raise MetricFault('dt', f'non-positive dt {dt!r}')
        _require_finite([float(v) for v in pose], 'pose')
        if self._v1_last_dt_us is not None and dt_us != self._v1_last_dt_us:
            self._v1_dt_fixed = False
        s = Sample()
        s.step, s.t_us, s.dt_us, s.fly = step, self._v1_seg_us, dt_us, fly
        window_us = self._v1_eff_window_us()
        if window_us is None:
            s.observation_dt_us = dt_us
        else:
            remaining_us = self._v1_pres_start_us + window_us - s.t_us
            s.observation_dt_us = max(0, min(dt_us, remaining_us))
        s.x, s.y, s.heading, s.speed, s.angular_velocity = (float(v) for v in pose)
        self._v1_last_step, self._v1_last_t_us, self._v1_last_dt_us = step, s.t_us, dt_us
        self._v1_pres_ticks += 1
        return s

    def _v1_after(self, sample):
        """Advance the clock past this sample and end the measurement at the effective window."""
        self._v1_seg_us += sample.dt_us
        window_us = self._v1_eff_window_us()
        if self._v1_end is None and window_us is not None:
            if self._v1_seg_us - self._v1_pres_start_us >= window_us:
                self._v1_end = {'event': None, 'end_us': self._v1_pres_start_us + window_us,
                                'reason': 'window_elapsed', 'kind': 'deadline'}

    def _v1_sample(self, fly, dt, pose):
        sample = self._v1_begin(fly, dt, pose)
        if self._v1_end is None:
            self._v1_pres_samples += 1
            self._v1_seg_samples += 1
            self._v1_observe(sample)
        self._v1_after(sample)

    def _v1_observed_samples(self):
        return self._v1_pres_samples

    def _v1_segment_samples(self):
        return self._v1_seg_samples

    def _v1_terminal(self, name, sample):
        """A terminal event at this pre-motor sample.

        With automatic_end the measurement ends at the event's time and True is returned
        (the event sample's forward interval is excluded from durations). Without it the
        event is only recorded and False is returned (v1.1 A1.3).
        """
        if self._v1_end is not None:
            return True
        if self._v1_cfg()['automatic_end']:
            self._v1_end = {'event': name, 'end_us': sample.t_us, 'reason': f'terminal_event:{name}',
                            'kind': 'terminal'}
            return True
        if name not in [e['event'] for e in self._v1_events_seen]:
            self._v1_events_seen.append(event(name, sample.t_rel_s, sample.step))
        return False

    def _v1_interval(self, end_us=None):
        if end_us is None:
            end_us = self._v1_cut()[0] if self._v1_cut() else self._v1_seg_us
        return [us_to_s(self._v1_pres_start_us), us_to_s(end_us)]

    def _v1_seg_interval(self):
        end_us = self._v1_cut()[0] if self._v1_cut() else self._v1_seg_us
        return [0.0, us_to_s(end_us)]

    def _v1_cut(self):
        """(cut_us, kind) once the observation is cut, else None.

        kind: 'deadline' (window fully observed), 'terminal' (ended at a declared terminal
        event), 'interrupt' (frozen early by an interruption) or 'fault' (fault_halt).
        """
        if self._v1_end is not None:
            return self._v1_end['end_us'], self._v1_end['kind']
        return self._v1_freeze_cut

    def _v1_cut_us(self):
        cut = self._v1_cut()
        return cut[0] if cut else None

    def _v1_censor_counts(self, cut_us):
        return {'censored_at_rel_s': us_to_s(cut_us), 'censored_after_s': us_to_s(cut_us - self._v1_pres_start_us)}

    def _v1_states(self):
        prov = self.provenance if isinstance(self.provenance, dict) else None
        states = prov.get('controller_states') if prov else None
        return set(states) if states else None

    def _v1_state_gate(self, required):
        """None when the active controller emits every required state, else the missing one."""
        states = self._v1_states()
        if states is None:
            return required[0]
        for name in required:
            if name not in states:
                return name
        return None

    def _v1_pending_value(self, kind, unit, note=None, counts=None):
        """A value that waits for an event: pending, censored at the cut, or invalidated on a fault."""
        cut = self._v1_cut()
        if cut is None:
            return record(kind=kind, unit=unit, reason='pending', note=note, counts=counts,
                          interval_rel_s=self._v1_interval())
        cut_us, how = cut
        merged = dict(counts or {})
        merged.update(self._v1_censor_counts(cut_us))
        return record(kind=kind, unit=unit, reason='invalidated' if how == 'fault' else 'censored', note=note,
                      counts=merged, interval_rel_s=self._v1_interval(cut_us))

    def _v1_segment_pending_value(self, kind, unit, note=None, counts=None):
        """Pending/censored value whose clock and interval begin at segment origin."""
        cut = self._v1_cut()
        if cut is None:
            return record(kind=kind, unit=unit, reason='pending', note=note, counts=counts,
                          interval_rel_s=self._v1_seg_interval())
        cut_us, how = cut
        merged = dict(counts or {})
        merged.update({'censored_at_rel_s': us_to_s(cut_us), 'censored_after_s': us_to_s(cut_us)})
        return record(kind=kind, unit=unit, reason='invalidated' if how == 'fault' else 'censored',
                      note=note, counts=merged, interval_rel_s=[0.0, us_to_s(cut_us)])

    def _v1_segment_latency(self, event_us, *, unit='s', kind='latency', note=None):
        cut = self._v1_cut()
        if cut is not None and cut[1] == 'fault':
            counts = {}
            if event_us is not None:
                counts = {'observed_at_rel_s': us_to_s(event_us),
                          'observed_after_s': us_to_s(event_us)}
            return self._v1_segment_pending_value(kind, unit, note, counts)
        if event_us is not None:
            return record(us_to_s(event_us), kind=kind, unit=unit, note=note,
                          interval_rel_s=self._v1_seg_interval())
        return self._v1_segment_pending_value(kind, unit, note)

    def _v1_segment_outcome(self, happened, *, note=None):
        cut = self._v1_cut()
        if cut is not None and cut[1] == 'fault':
            return self._v1_segment_pending_value('event', 'bool', note,
                                                  {'observed_events': 1 if happened else 0})
        if happened:
            return record(True, kind='event', unit='bool', note=note,
                          interval_rel_s=self._v1_seg_interval())
        if cut is not None and cut[1] in ('deadline', 'terminal'):
            return record(False, kind='event', unit='bool', note=note,
                          counts={'completed_at_rel_s': us_to_s(cut[0])},
                          interval_rel_s=[0.0, us_to_s(cut[0])])
        return self._v1_segment_pending_value('event', 'bool', note)

    def _v1_latency(self, event_us, *, unit='s', kind='latency', note=None):
        """An event latency, elapsed from the presentation start (v1.1 A3.2)."""
        cut = self._v1_cut()
        if cut is not None and cut[1] == 'fault':
            counts = {}
            if event_us is not None:
                counts = {'observed_at_rel_s': us_to_s(event_us),
                          'observed_after_s': us_to_s(event_us - self._v1_pres_start_us)}
            return self._v1_pending_value(kind, unit, note, counts)
        if event_us is not None:
            return record(us_to_s(event_us - self._v1_pres_start_us), kind=kind, unit=unit,
                          note=note, interval_rel_s=self._v1_interval())
        return self._v1_pending_value(kind, unit, note)

    def _v1_faulted_value(self, value, *, kind, unit, note=None, counts=None, evidence_ref=None):
        """Invalidate an observed companion value while retaining its diagnostic value."""
        cut = self._v1_cut()
        if cut is None or cut[1] != 'fault':
            return record(value, kind=kind, unit=unit, note=note, counts=counts,
                          interval_rel_s=self._v1_interval(), evidence_ref=evidence_ref)
        diagnostic = dict(counts or {})
        diagnostic['observed_value'] = value
        diagnostic.update(self._v1_censor_counts(cut[0]))
        return record(kind=kind, unit=unit, reason='invalidated', note=note, counts=diagnostic,
                      interval_rel_s=self._v1_interval(cut[0]), evidence_ref=evidence_ref)

    def _v1_outcome(self, happened, *, note=None):
        """An outcome boolean under the deadline-outcome invariant (v1.1 A5.3)."""
        cut = self._v1_cut()
        if cut is not None and cut[1] == 'fault':
            return self._v1_pending_value('event', 'bool', note,
                                          {'observed_events': 1 if happened else 0})
        if happened:
            return record(True, kind='event', unit='bool', note=note, interval_rel_s=self._v1_interval())
        if cut is not None and cut[1] in ('deadline', 'terminal'):
            return record(False, kind='event', unit='bool', note=note,
                          counts={'completed_at_rel_s': us_to_s(cut[0])}, interval_rel_s=self._v1_interval(cut[0]))
        return self._v1_pending_value('event', 'bool', note)

    # Backwards name used by the Deck WIP observers.
    _v1_event_flag = _v1_outcome

    # ---- published views --------------------------------------------------------
    def observation_status(self):
        """I-2, read once per daemon tick after ``arena.step``."""
        self._v1_ready()
        cfg = self._v1_cfg()
        hold_us = s_to_us(cfg['hold_s'])
        state, hold_remaining = 'observing', 0.0
        if self._v1_end is not None:
            elapsed_since = self._v1_seg_us - self._v1_end['end_us']
            if self._v1_end['kind'] == 'terminal' and elapsed_since < hold_us:
                state, hold_remaining = 'measurement_ended', us_to_s(hold_us - elapsed_since)
            else:
                state = 'closed'
        window_us = self._v1_eff_window_us()
        return {
            'state': state,
            'end_reason': self._v1_end['reason'] if self._v1_end else None,
            'terminal_event': self._v1_end['event'] if self._v1_end else None,
            'measurement_end_rel_s': us_to_s(self._v1_end['end_us']) if self._v1_end else None,
            'presentation_index': self._v1_pres_index,
            'presentation_start_rel_s': us_to_s(self._v1_pres_start_us),
            'presentation_elapsed_s': us_to_s(self._v1_seg_us - self._v1_pres_start_us),
            'segment_elapsed_s': us_to_s(self._v1_seg_us),
            'hold_remaining_s': hold_remaining,
            'config_id': cfg['config_id'],
            'effective_window_s': cfg['effective_window_s'] if cfg['automatic_end'] else None,
            'automatic_end': cfg['automatic_end'],
            'window_end_rel_s': us_to_s(self._v1_pres_start_us + window_us) if window_us is not None else None,
        }

    def get_metric_records(self):
        """I-3: live, provisional records (``final`` false), each with a ``scope``."""
        self._v1_ready()
        recs = self._v1_records(self._v1_cut_us())
        default_scope = self._v1_default_scope()
        for name, rec in recs.items():
            _check_vocab('record', type(rec), {dict})
            if (rec['value'] is None) == rec['available'] or (rec['reason'] is None) != rec['available']:
                raise AssertionError(f'{name}: available/value/reason invariant broken')
            if name in OUTCOME_BOOLEANS and rec['value'] is False and rec['reason'] is not None:
                raise AssertionError(f'{name}: false with a reason is forbidden (v1.1 A5)')
            if rec.get('scope') is None:
                rec['scope'] = default_scope
        return recs

    def log_intervention(self, name, value=None):
        """A logged intervention (food placement, policy change, ...); nothing is closed (§5.4)."""
        self._v1_ready()
        if not isinstance(value, (type(None), bool, int, float, str)):
            raise TypeError('an intervention value is a scalar or a label')
        entry = {'name': str(name), 'value': value, 't_rel_s': us_to_s(self._v1_seg_us),
                 'step': int(self._v1_last_step or 0)}
        _require_finite(entry, 'intervention')
        self._v1_interventions.append(entry)

    def get_evidence(self):
        self._v1_ready()
        ev = dict(self._v1_evidence())
        ev['interventions'] = evidence('intervention_log', self._v1_interventions)
        if self._v1_events_seen:
            ev['terminal_events_observed'] = evidence('event_sequence', self._v1_events_seen)
        _require_finite(ev, 'evidence')
        return ev

    def _v1_envelope_extra(self):
        return {}

    def _v1_producer_snapshot(self, *, records, evidence_items, completeness, end_reason,
                              measurement_end_rel_s):
        """Build the detached producer-owned part shared by live and frozen views."""
        status = self.observation_status()
        cfg = self.observation_config()
        spec = self.observation_spec()
        snapshot = {
            'schema': SCHEMA,
            'contract': CONTRACT_VERSION,
            'assay': spec['assay'],
            'spec_version': spec['spec_version'],
            'mode': spec['mode'],
            'config_id': cfg['config_id'],
            'effective_window_s': cfg['effective_window_s'] if cfg['automatic_end'] else None,
            'window_s': cfg['effective_window_s'] if cfg['automatic_end'] else None,  # alias through v0.4
            'automatic_end': cfg['automatic_end'],
            'window_source': cfg['window_source'],
            'override': copy.deepcopy(cfg['override']),
            'hold_s': cfg['hold_s'],
            'window_params': spec['window_params'],
            'window_end_rel_s': status['window_end_rel_s'],
            'light_schedule': spec.get('light_schedule'),
            'sample_point': 'pre_motor',
            'dt_s': us_to_s(self._v1_last_dt_us) if self._v1_last_dt_us else None,
            'dt_fixed': self._v1_dt_fixed,
            'presentation_index': status['presentation_index'],
            'presentation_start_rel_s': status['presentation_start_rel_s'],
            'presentation_elapsed_s': status['presentation_elapsed_s'],
            'state': status['state'],
            'completeness': completeness,
            'end_reason': end_reason,
            'terminal_event': status['terminal_event'],
            'measurement_end_rel_s': measurement_end_rel_s,
            'records': copy.deepcopy(records),
            'evidence': copy.deepcopy(evidence_items),
        }
        snapshot.update(copy.deepcopy(self._v1_envelope_extra()))
        _require_finite(snapshot, 'snapshot')
        return snapshot

    def snapshot_observation(self):
        """Return a detached live producer snapshot without advancing or freezing it."""
        self._v1_ready()
        status = self.observation_status()
        return self._v1_producer_snapshot(
            records=self.get_metric_records(), evidence_items=self.get_evidence(),
            completeness=None, end_reason=status['end_reason'],
            measurement_end_rel_s=status['measurement_end_rel_s'])

    def freeze_observation(self, end_reason=None):
        """The producer part of a frozen envelope (§1, §5.3 step 3), built before any mutation.

        ``end_reason`` defaults to the measurement end's own reason. An interruption
        censors pending values at the actual cutoff and keeps the measured prefix; a
        ``fault_halt`` marks them invalidated (v1.1 A5).
        """
        self._v1_ready()
        status = self.observation_status()
        if end_reason is None:
            end_reason = status['end_reason']
            if end_reason is None:
                raise ValueError('the measurement has not ended; give an interruption end_reason')
        elif not str(end_reason).startswith('terminal_event:'):
            _check_vocab('end_reason', end_reason, END_REASONS)
        if end_reason == 'policy_change' and self._v1_end is not None:
            # Policy cannot retroactively close an observation that already completed.
            # Rebuild its immutable envelope with the original completion reason.
            end_reason = self._v1_end['reason']
        cfg = self.observation_config()
        window_us = self._v1_eff_window_us()
        window_elapsed = window_us is not None and self._v1_seg_us - self._v1_pres_start_us >= window_us
        completeness = completeness_for(end_reason, window_elapsed, cfg['automatic_end'])
        if self._v1_end is None:
            self._v1_freeze_cut = (self._v1_seg_us, 'fault' if end_reason == 'fault_halt' else 'interrupt')
        elif end_reason == 'fault_halt':
            self._v1_freeze_cut = (self._v1_end['end_us'], 'fault')
        saved_end = self._v1_end
        if end_reason == 'fault_halt':
            self._v1_end = None  # a fault overrides a completed verdict: nothing frozen as valid
        try:
            records = freeze_records(self.get_metric_records())
            ev = self.get_evidence()
        finally:
            self._v1_freeze_cut = None
            self._v1_end = saved_end
        measurement_end_rel_s = (status['measurement_end_rel_s']
                                 if status['measurement_end_rel_s'] is not None
                                 else status['segment_elapsed_s'])
        return self._v1_producer_snapshot(
            records=records, evidence_items=ev, completeness=completeness,
            end_reason=end_reason, measurement_end_rel_s=measurement_end_rel_s)

    def begin_next_presentation(self, reason, config=None):
        """I-6: called after the freeze. Clears presentation-scoped state only; no pose change.

        ``config`` (v1.1 A1) is the ObservationConfig of the new presentation; changed
        policy fields are logged as ``policy:<field>`` interventions.
        """
        self._v1_ready()
        if not isinstance(reason, str) or not reason:
            raise ValueError('begin_next_presentation needs a reason')
        if config is not None:
            validate_observation_config(config)
        carry = self._v1_carry()
        self._v1_pres_index += 1
        self._v1_pres_start_us = self._v1_seg_us
        self._v1_pres_samples = 0
        self._v1_pres_ticks = 0
        self._v1_end = None
        self._v1_freeze_cut = None
        self._v1_events_seen = []
        self._v1_clear()
        self._v1_restore(carry)
        self._v1_apply_config(config)
        self._v1_on_new_presentation(reason)
        return self._v1_pres_index

    def begin_observation_segment(self, config):
        """Start a fresh measurement segment without resetting the retained arena.

        The caller freezes/persists the preceding observation before invoking this
        method under its arena lock.  This method validates the complete prospective
        config before mutation, clears measurement clocks and accumulators, and keeps
        only observer-specific physical hysteresis needed to avoid a fabricated entry
        or contact onset.  It does not inspect or mutate a fly, controller, or stimulus.
        """
        cfg = validate_observation_config(config)
        self._v1_ready()
        carry = self._v1_segment_carry()
        self._v1_init()
        self._v1_config = cfg
        self._v1_segment_restore(carry)
        self._v1_on_new_observation_segment()
        return copy.deepcopy(cfg)

    def _v1_on_new_presentation(self, reason):
        pass

    def _v1_reset_segment(self, config=None):
        """I-5: a new segment (called after the legacy ``reset_trial``)."""
        if config is not None:
            validate_observation_config(config)
        self._v1_init()
        self._v1_apply_config(config)

    def observe_contact(self, arena_step, dt, sample_point='post_solver', in_contact=False,
                        normals=None, source='solver'):
        """I-9: the arena's single post-solver contact result for ``arena_step``."""
        self._v1_ready()
        _check_vocab('sample_point', sample_point, SAMPLE_POINTS)
        if sample_point != 'post_solver':
            raise ValueError('contact handoffs are post_solver samples')
        if self._v1_last_step != int(arena_step) or self._v1_last_t_us is None:
            raise ValueError('observe_contact must follow the pre-motor sample of the same arena step')
        raw_dt_us = s_to_us(float(dt))
        t_us = self._v1_last_t_us + raw_dt_us  # post-solver: the end of this step
        end_us = self._v1_end['end_us'] if self._v1_end is not None else None
        if end_us is not None and self._v1_last_t_us >= end_us:
            return
        observed_dt_us = raw_dt_us if end_us is None else min(raw_dt_us, end_us - self._v1_last_t_us)
        if end_us is not None and t_us > end_us:
            self._v1_contact_prefix(observed_dt_us)
            return
        self._v1_contact(int(arena_step), t_us, observed_dt_us, bool(in_contact), source)

    def _v1_contact(self, step, t_us, dt_us, in_contact, source):
        pass  # assays without a contact metric ignore the handoff

    def _v1_contact_prefix(self, dt_us):
        pass  # assays with contacts may integrate their last in-window state to the deadline


class ContactCounter:
    """Contact onsets and contact time from the arena's post-solver handoff (§9.2)."""

    def __init__(self):
        self.handoffs = 0
        self.onsets = 0
        self.contact_us = 0
        self.prev = False
        self.events = []

    def observe(self, step, t_us, dt_us, in_contact, source):
        self.handoffs += 1
        if in_contact:
            self.contact_us += dt_us
            if not self.prev:
                self.onsets += 1
                if len(self.events) < EVIDENCE_EVENT_CAP:
                    self.events.append(event('contact_onset', us_to_s(t_us), step, 'post_solver',
                                             source=str(source)))
        self.prev = in_contact

    def integrate_prefix(self, dt_us):
        """Extend a known contact state without accepting an out-of-window handoff/event."""
        if self.prev:
            self.contact_us += dt_us

    def records(self, interval):
        counts = {'handoffs': self.handoffs}
        if not self.handoffs:
            return {
                'wall_contact_onsets': record(kind='count', unit='count', reason='not_observed', counts=counts,
                                              note='no solver contact feed received', interval_rel_s=interval),
                'wall_contact_s': record(kind='duration', unit='s', reason='not_observed', counts=counts,
                                         note='no solver contact feed received', interval_rel_s=interval),
            }
        return {
            'wall_contact_onsets': record(self.onsets, kind='count', unit='count', counts=counts,
                                          interval_rel_s=interval, evidence_ref='contact_events'),
            'wall_contact_s': record(us_to_s(self.contact_us), kind='duration', unit='s', counts=counts,
                                     interval_rel_s=interval),
        }


class OpenArenaObserver(ObservationLifecycle):
    """Open-arena foraging metrics (no paradigm object; I-13) with the full v1.1 lifecycle.

    Segment-scoped run measures survive a presentation change; presentation-scoped
    measures describe one airflow condition (v1.1 A4). The arena calls
    ``observe(fly, food_contact, dt, airflow_mm_s=...)`` once per step at the pre-motor sample.
    """

    OBSERVATION_SPEC = {
        'assay': 'open_arena', 'spec_version': 'open_arena/1.2', 'mode': 'continuous',
        'window_s': None, 'hold_s': 0.0, 'terminal_events': [],
        're_presentation_triggers': ['windStrength'],
        'logged_interventions': ['food'],
        'headline_metric': 'food_contacts',
        'companion_metrics': ['time_to_first_contact_s', 'observation_s', 'distance_mm',
                              'presentation_food_contacts', 'airflow_mm_s'],
        'curve_metrics': [], 'learning_claim': 'none', 'required_states': {},
    }
    SEGMENT_RECORDS = ('observation_s', 'food_contacts', 'time_to_first_contact_s', 'distance_mm')

    def __init__(self, airflow_mm_s=0.0):
        airflow = float(airflow_mm_s)
        _require_finite(airflow, 'open_arena.airflow_mm_s')
        self.airflow_mm_s = airflow
        self._v1_init()

    def _v1_window_params(self):
        declared = self._pres.get('airflow') if hasattr(self, '_pres') else None
        return {'airflow_mm_s': float(self.airflow_mm_s if declared is None else declared)}

    def _v1_clear_segment(self):
        self._seg = {'obs_us': 0, 'contacts': 0, 'first_us': None, 'distance': 0.0, 'n': 0}
        self._in_contact = False
        self._last_xy = None
        self._contact_events = []

    def _v1_clear(self):
        self._pres = {'obs_us': 0, 'contacts': 0, 'distance': 0.0, 'n': 0, 'airflow': None}

    def _v1_segment_carry(self):
        return self._in_contact

    def _v1_segment_restore(self, carry):
        self._in_contact = carry

    def validate_airflow_condition(self, airflow_mm_s):
        """Validate a sample against this presentation without changing observer state."""
        sampled_airflow = float(self.airflow_mm_s if airflow_mm_s is None else airflow_mm_s)
        _require_finite(sampled_airflow, 'open_arena.airflow_mm_s')
        declared_airflow = self._pres['airflow']
        if declared_airflow is not None and sampled_airflow != declared_airflow:
            raise MetricFault(
                'open_arena.airflow_mm_s',
                f'sampled airflow changed from declared {declared_airflow} to {sampled_airflow}; '
                'freeze and begin a new presentation before sampling the new condition')
        return sampled_airflow

    def observe(self, fly, food_contact, dt, airflow_mm_s=None, arena_step=None):
        """The arena's own food-contact test for this step, at the pre-motor sample."""
        sampled_airflow = self.validate_airflow_condition(airflow_mm_s)
        if arena_step is not None:
            # Condition validation precedes the pending-step mutation, while the actual
            # arena step/raw dt still reaches the common sampling path.
            self.begin_sample(arena_step, dt)
        self.airflow_mm_s = sampled_airflow
        pos = _fly_field(fly, 'pos')
        x = float(pos.x) if pos is not None else float(_fly_field(fly, 'x', 0.0))
        y = float(pos.y) if pos is not None else float(_fly_field(fly, 'y', 0.0))
        pose = (x, y, float(_fly_field(fly, 'heading', 0.0)), float(_fly_field(fly, 'speed', 0.0)),
                float(_fly_field(fly, 'angular_velocity', 0.0)))
        sample = self._v1_begin(fly, dt, pose)
        if self._v1_end is None:
            self._v1_pres_samples += 1
            self._observe_contact(sample, bool(food_contact))
        self._v1_after(sample)

    def _observe_contact(self, s, contact):
        seg, pres = self._seg, self._pres
        if pres['airflow'] is None:
            pres['airflow'] = float(self.airflow_mm_s)
        if self._last_xy is not None:
            step = math.dist(self._last_xy, (s.x, s.y))
            seg['distance'] += step
            if pres['n']:
                pres['distance'] += step
        self._last_xy = (s.x, s.y)
        if contact and not self._in_contact:
            seg['contacts'] += 1
            pres['contacts'] += 1
            if seg['first_us'] is None:
                seg['first_us'] = s.t_us
            if len(self._contact_events) < EVIDENCE_EVENT_CAP:
                self._contact_events.append(event('food_contact_onset', s.t_rel_s, s.step,
                                                  presentation_index=self._v1_pres_index))
        self._in_contact = contact
        seg['obs_us'] += s.observation_dt_us
        pres['obs_us'] += s.observation_dt_us
        seg['n'] += 1
        pres['n'] += 1

    def reset_trial(self, config=None):
        self._v1_reset_segment(config)

    def _v1_observe(self, sample):  # observe() drives the sample directly
        pass

    def _v1_records(self, cut):
        seg, pres = self._seg, self._pres
        siv, piv = self._v1_seg_interval(), self._v1_interval()

        def measured(value, n, kind, unit, iv, scope, **kw):
            if n == 0:
                return record(kind=kind, unit=unit, reason='not_observed', interval_rel_s=iv, scope=scope)
            return record(value, kind=kind, unit=unit, interval_rel_s=iv, scope=scope, **kw)

        if seg['first_us'] is not None:
            first = record(us_to_s(seg['first_us']), kind='latency', unit='s', interval_rel_s=siv, scope='segment',
                           note='elapsed from the segment start', evidence_ref='contact_events')
        else:
            pending = self._v1_pending_value('latency', 's', note='elapsed from the segment start')
            pending['interval_rel_s'] = [0.0, pending['interval_rel_s'][1]]
            if pending['reason'] == 'censored':
                pending['counts']['censored_after_s'] = pending['counts']['censored_at_rel_s']
            pending['scope'] = 'segment'
            first = pending
        airflow = (record(pres['airflow'], kind='kinematic', unit='mm/s', interval_rel_s=piv, scope='presentation',
                          note='airflow condition of this presentation (toward -x)')
                   if pres['airflow'] is not None else
                   record(kind='kinematic', unit='mm/s', reason='not_observed', interval_rel_s=piv,
                          scope='presentation'))
        return {
            'observation_s': measured(us_to_s(seg['obs_us']), seg['n'], 'duration', 's', siv, 'segment'),
            'food_contacts': measured(seg['contacts'], seg['n'], 'count', 'count', siv, 'segment',
                                      evidence_ref='contact_events'),
            'time_to_first_contact_s': first,
            'distance_mm': measured(seg['distance'], seg['n'], 'kinematic', 'mm', siv, 'segment'),
            'presentation_observation_s': measured(us_to_s(pres['obs_us']), pres['n'], 'duration', 's', piv,
                                                   'presentation'),
            'presentation_food_contacts': measured(pres['contacts'], pres['n'], 'count', 'count', piv,
                                                   'presentation', evidence_ref='contact_events'),
            'presentation_distance_mm': measured(pres['distance'], pres['n'], 'kinematic', 'mm', piv,
                                                 'presentation'),
            'airflow_mm_s': airflow,
        }

    def _v1_evidence(self):
        return {'contact_events': evidence('event_sequence', self._contact_events)}
