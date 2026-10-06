/* Dashboard presentation only; accepted transport helper owns validation. */
(function expose(root, factory) {
    const helper = typeof module === 'object' && module.exports
        ? require('./observation_display.js') : root.NeuroFlyObservationDisplay;
    const api = factory(helper);
    if (typeof module === 'object' && module.exports) module.exports = api;
    if (root) root.NeuroFlyObservationRenderer = api;
}(typeof globalThis === 'object' ? globalThis : this, function build(helper) {
    'use strict';
    const PRIMARY = Object.freeze({
        'open-arena': ['distance_mm', 'Distance'], 't-maze': ['arm_entry_preference_index', 'Arm entry preference'],
        'y-maze': ['spontaneous_alternation_rate', 'Spontaneous alternation'],
        'heat-maze': ['escape_latency_s', 'Escape latency'], buridan: ['centrophobism_index', 'Centrophobism'],
        'visual-operant': ['safe_occupancy_fraction', 'Safe-zone occupancy'],
        'wind-tunnel': ['upwind_displacement_mm', 'Upwind displacement'],
        'looming-escape': ['initiation_latency_s', 'Escape initiation latency'], optomotor: ['gain', 'Optomotor gain'],
        'gap-crossing': ['crossing_success', 'Crossing success'],
        'circadian-dam': ['bout_immobility_min', 'Bout immobility'], courtship: ['courtship_state_fraction', 'Courtship state occupancy'],
        labyrinth: ['tortuosity', 'Path tortuosity'], 'multisensory-sandbox': ['composite_benchmark_score', 'Scientific composite score'],
    });
    function matchesObservationOwner(packet, identity, segmentId) {
        const fields = ['daemon_run_id', 'run_id', 'instance_id', 'activation', 'assay',
            'backend', 'controller_version', 'brain_id', 'synthetic', 'test_mode'];
        const expected = {...packet?.identity, brain_id: packet?.brain_id};
        return !!(identity && (!Object.prototype.hasOwnProperty.call(packet?.identity || {}, 'brain_id')
                || packet.identity.brain_id === packet.brain_id)
            && fields.every(key => Object.prototype.hasOwnProperty.call(identity, key)
                && Object.is(identity[key], expected[key]))
            && typeof segmentId === 'string' && segmentId.length
            && segmentId === (packet?.segment_id || packet?.observation?.segment_id));
    }
    function matchesValidityUpdate(packet, update) {
        return !!(update?.schema === 'neurofly-observation-validity-update/1'
            && update.reason === 'unexpected_loop_exit'
            && ['invalidated', 'exploratory_degraded'].includes(update.validity)
            && matchesObservationOwner(packet, update.identity, update.segment_id));
    }
    function view(packet, {connected = false, pending = false, assay = null, replay = false, validityUpdate = null} = {}) {
        const selected = assay || packet?.identity?.assay;
        const ownerMatches = packet?.identity?.assay === selected;
        const ready = connected && !pending && ownerMatches;
        // The daemon's wire identity excludes brain_id; that owner is a sibling
        // field. Preserve any explicitly supplied nested value so disagreement
        // still fails validation instead of being silently overwritten.
        const wireIdentity = packet?.identity;
        const identity = wireIdentity && !Object.prototype.hasOwnProperty.call(wireIdentity, 'brain_id')
            ? {...wireIdentity, brain_id: packet?.brain_id} : wireIdentity;
        const faultUpdate = ready && !replay && matchesValidityUpdate(packet, validityUpdate);
        const observation = faultUpdate && packet?.observation
            ? {...packet.observation, validity: validityUpdate.validity} : packet?.observation;
        const result = helper.buildObservationDisplay({identity, brainId: packet?.brain_id,
            observation: ready ? observation : null,
            observationPublication: ownerMatches && !pending && !replay ? packet?.observation_publication : null});
        return {...result, transportLabel: replay ? 'Recording replay · not live durable data'
            : pending ? 'Switch pending' : !connected ? 'Disconnected · current observation unavailable'
                : !ownerMatches ? 'Waiting for selected owner' : faultUpdate
                    ? 'Last pre-fault frame · unexpected_loop_exit' : 'Daemon transport',
            assay: selected, replay};
    }
    function primary(display) {
        const [key, label] = PRIMARY[display.assay] || ['', 'Observation'];
        const live = display.live, record = live?.state === 'available' ? live.records[key] : null;
        const valid = live?.validity === 'valid';
        return {label: `${label} · ${display.replay ? 'replay provisional' : 'live provisional'}`,
            value: record ? `${record.text}${valid ? '' : ' · ' + live.validity}` : 'Unavailable',
            rawValue: valid && record?.available && typeof record.rawValue === 'number' ? record.rawValue : null,
            unit: record?.unitText || '',
            sub: record ? `${live.validity} · ${record.scope} · ${record.note || record.reason || display.transportLabel}` : display.transportLabel,
            contextKey: JSON.stringify([display.currentIdentity, live?.segmentId, live?.presentationId, key, record?.unit, live?.validity, display.transportLabel]),
        };
    }
    function node(document, tag, text, className) {
        const el = document.createElement(tag);
        if (text !== undefined) el.textContent = text;
        if (className) el.className = className;
        return el;
    }
    function section(document, title, observation, transportLabel) {
        const panel = node(document, 'section', undefined, 'observation-panel');
        panel.appendChild(node(document, 'h4', title));
        if (observation?.state !== 'available') {
            panel.appendChild(node(document, 'p', `${transportLabel} · Unavailable: ${observation?.reason || 'no observation'}`));
            if (observation?.error) panel.appendChild(node(document, 'p', observation.error));
            return panel;
        }
        const status = observation.provisional ? `Provisional · ${observation.validity} · ${observation.envelope.state}${observation.endReason ? ' · ' + observation.endReason : ''}`
            : `${observation.contextLabel} · ${observation.validity} · ${observation.completeness} · ${observation.endReason}`;
        panel.appendChild(node(document, 'p', status, 'observation-status'));
        panel.appendChild(node(document, 'p', `Experiment record brain ${observation.identity.brain_id} · Controller instance ${observation.identity.instance_id} · run ${observation.identity.run_id} · presentation ${observation.presentationId}`));
        const prov = observation.provenance;
        panel.appendChild(node(document, 'p', `Controller: ${observation.identity.backend} · ${observation.identity.controller_version} · synthetic: ${observation.identity.synthetic} · test mode: ${observation.identity.test_mode}. Motor assists: ${prov.motor_assists_enabled}. GF: ${prov.gf_source ?? 'not declared'}. Stimulus: ${prov.stimulus_entry_stage ?? 'not declared'}.`));
        for (const [name, record] of Object.entries(observation.records)) {
            const row = node(document, 'div', undefined, 'observation-record');
            row.appendChild(node(document, 'span', name.replace(/_/g, ' ')));
            const value = node(document, 'strong', record.text);
            if (record.available && typeof record.rawValue === 'number') {
                const raw = Object.is(record.rawValue, -0) ? '-0' : String(record.rawValue);
                value.title = `Raw value: ${raw} ${record.rawUnit}`;
                value.setAttribute('aria-label', `${record.text} · ${value.title}`);
            }
            row.appendChild(value);
            row.appendChild(node(document, 'small', `${record.scope} · ${record.capability} · unit: ${record.unit} · ${record.reason || 'available'}${record.note ? ' · ' + record.note : ''}`));
            panel.appendChild(row);
        }
        if (observation.receipt) panel.appendChild(node(document, 'p', `Saved receipt: ${observation.receipt.file}:${observation.receipt.line}. Receipt consistency checked in transport; browser does not verify disk durability.`));
        return panel;
    }
    function render(container, display) {
        if (!container) return;
        const doc = container.ownerDocument;
        container.replaceChildren(node(doc, 'p', `Current validity: ${display.currentValidity ?? 'unknown'} · ${display.transportLabel}`), section(doc, 'Live observation (provisional)', display.live, display.transportLabel),
            section(doc, 'Saved observation', display.terminal, display.replay ? display.transportLabel : 'No accepted saved observation'));
    }
    return Object.freeze({PRIMARY, matchesObservationOwner, matchesValidityUpdate, view, primary, render});
}));
