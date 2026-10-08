/*
 * Opt-in environmental inspector (web/env_inspector.html).  Read-only.
 *
 * Shows only telemetry the daemon actually publishes, following the W1 sensory
 * capability contract (docs/SENSORY_CAPABILITY_CONTRACT.md §4).  It never computes
 * a delivered input, never fills a missing value, and never sends a command: it
 * opens the daemon's read-only /api/stream (or replays a recorded packet file).
 *
 * Three separate things are kept apart on purpose:
 *   1. field concentration (what the assay field held at the fly),
 *   2. delivered receptor input (connectome.input_stage[].current_injected),
 *   3. neural response (n_spiking_this_step per probe, descending-neuron rates).
 * Delivering an input is not a response, and a decorative model is not detection.
 *
 * Nothing in the dashboard, replay or daemon loads this file; default UI is unchanged.
 */
(function (root, factory) {
    'use strict';
    const api = factory();
    if (typeof module === 'object' && module.exports) module.exports = api;
    if (root) root.NeuroflyEnvInspector = api;
}(typeof window !== 'undefined' ? window : null, function () {
    'use strict';

    const STALE_AFTER_S = 3;
    const UNITS_ODOUR = 'arbitrary normalised units (0–1), not ppm or molar';
    const NOT_SIMULATED = 'NOT SIMULATED';
    const NOT_PUBLISHED = 'NOT PUBLISHED';
    const NOT_DELIVERED = 'NOT DELIVERED';
    const UNAVAILABLE = 'UNAVAILABLE';
    // Model constants quoted by the W1 contract (arena.py:81-82).  Used only to DRAW the
    // antennal sample points, which the daemon does not publish; always labelled derived.
    const ANTENNA_LENGTH_MM = 2.0;
    const ANTENNA_ANGLE_RAD = Math.PI / 4;
    const GRAPH_BACKENDS = ['connectome-fixed', 'connectome-plastic', 'connectome-with-trained-readout'];
    const BRIDGE_RPC = 'hybrid-bridge-rpc-experimental';
    // Probes whose stimulus is suspected never to reach the graph in RPC bridge mode (W1 §0.5).
    const BRIDGE_MISSING = ['orn_food', 'orn_danger', 'courtship_cva', 'jon_wind', 'thermo'];

    // Source markers the daemon publishes in `scene`.  Shape + ID carry the meaning;
    // colour only supplements.  Labels follow W1 §4.2 (no chemical identity).
    const SOURCE_KEYS = [
        {key: 'food_pos', id: 'F', shape: 'square', colour: '#4ade80',
            label: 'generic attractive odour source → ORN_DM1 (Or42b) channel'},
        {key: 'repellent_pos', id: 'R', shape: 'triangle', colour: '#f87171',
            label: 'generic aversive odour source → ORN_DA2 (Or56a) channel'},
        {key: 'pheromone_pos', id: 'P', shape: 'diamond', colour: '#c084fc',
            label: 'cVA field source (model field, body-centre sampled) → ORN_DA1 channel'},
        {key: 'nozzle_pos', id: 'N', shape: 'circle', colour: '#38bdf8',
            label: 'wind-tunnel odour nozzle (static analytic plume shape, not transported)'},
        {key: 'goal_pos', id: 'G', shape: 'star', colour: '#facc15', label: 'goal odour source'},
        {key: 'hotspot_pos', id: 'H', shape: 'cross', colour: '#fb923c', label: 'heat spot (temperature field, not odour)'},
        {key: 'cool_pos', id: 'C', shape: 'cross', colour: '#93c5fd', label: 'cool spot (temperature field, not odour)'}
    ];

    const num = (v) => (typeof v === 'number' && Number.isFinite(v) ? v : null);
    const pt = (v) => (Array.isArray(v) && v.length >= 2 && num(v[0]) !== null && num(v[1]) !== null ? [v[0], v[1]] : null);

    // ------------------------------------------------------------------ freshness
    // How current the shown values are.  Paused and halted come from the packet; stale
    // and disconnected from wall time.  `mode` is 'live' or 'replay'.
    function freshness(o) {
        if (!o || !o.haveSource) return {state: 'no-source', text: 'NO SOURCE · nothing connected, nothing shown'};
        if (o.error) return {state: 'unavailable', text: UNAVAILABLE + ' · ' + o.error};
        if (!o.pkt) return {state: 'connecting', text: 'WAITING for the first frame'};
        const age = num(o.ageS);
        if (o.mode === 'replay') {
            return o.replayPaused ? {state: 'replay-paused', text: 'REPLAY PAUSED · recorded frame, not live'}
                : {state: 'replay', text: 'REPLAY · recorded frames, not live'};
        }
        if (o.disconnected) return {state: 'disconnected', text: 'DISCONNECTED · last known values' + (age !== null ? ', ' + age.toFixed(1) + ' s old' : '')};
        if (o.pkt.halted || o.pkt.error) return {state: 'halted', text: 'HALTED · the simulation is not advancing (' + (o.pkt.error || 'halted') + ')'};
        if (o.pkt.paused || o.beatPaused) return {state: 'paused', text: 'PAUSED · values frozen at step ' + o.pkt.step};
        if (age !== null && age > STALE_AFTER_S) return {state: 'stale', text: 'STALE · no new frame for ' + age.toFixed(1) + ' s'};
        return {state: 'live', text: 'LIVE · step ' + o.pkt.step};
    }

    // ------------------------------------------------------------------ view model
    function antennaPoints(fly) {
        if (!fly || num(fly.x) === null || num(fly.y) === null || num(fly.heading) === null) return null;
        // +heading CCW; left antenna at heading + 45°, right at heading − 45°.
        const at = (a) => [fly.x + ANTENNA_LENGTH_MM * Math.cos(fly.heading + a), fly.y + ANTENNA_LENGTH_MM * Math.sin(fly.heading + a)];
        return {left: at(ANTENNA_ANGLE_RAD), right: at(-ANTENNA_ANGLE_RAD),
            note: 'derived from model constants (2.0 mm at heading ±45°, arena.py:81-82); not published by the daemon'};
    }

    function sourcesOf(scene) {
        const out = [];
        if (!scene) return out;
        for (const s of SOURCE_KEYS) {
            const p = pt(scene[s.key]);
            if (p) out.push({id: s.id + '1', key: s.key, pos: p, shape: s.shape, colour: s.colour, label: s.label,
                emission: 'unit amplitude (model constant); emission rate ' + NOT_SIMULATED});
        }
        (Array.isArray(scene.food) ? scene.food : []).forEach((f, i) => {
            const p = pt(f);
            if (p) out.push({id: 'F' + (i + 2), key: 'food[' + i + ']', pos: p, shape: 'square', colour: '#4ade80',
                label: 'open-arena food (attractive odour field source)', emission: 'unit amplitude (model constant); emission rate ' + NOT_SIMULATED});
        });
        (Array.isArray(scene.hazards) ? scene.hazards : []).forEach((h, i) => {
            const p = pt(h);
            if (p) out.push({id: 'R' + (i + 2), key: 'hazards[' + i + ']', pos: p, shape: 'triangle', colour: '#f87171',
                label: 'open-arena hazard (aversive odour field source)', emission: 'unit amplitude (model constant); emission rate ' + NOT_SIMULATED});
        });
        if (pt(scene.female_pos)) out.push({id: 'Q1', key: 'female_pos', pos: pt(scene.female_pos), shape: 'diamond', colour: '#f472b6',
            label: 'female fly (' + (scene.female_type || 'type ?') + '); cVA/aphrodisiac exp(−d/3 mm) at the body centre',
            emission: 'unit amplitude (model constant); emission rate ' + NOT_SIMULATED});
        return out;
    }

    // Field concentration at the fly, exactly as published.  L/R never published.
    function concentrationRows(pkt) {
        const st = pkt.stimuli || {};
        const se = pkt.sensory || {};
        const rows = [];
        const add = (key, label, value, where) => rows.push({key, label, value: num(value), where, units: UNITS_ODOUR});
        if ('odor_a' in st || 'odor_a' in se) add('sensory.odor_a', 'generic attractive odour (food channel)', num(se.odor_a) !== null ? se.odor_a : st.odor_a, 'mean of the two antennal samples');
        if ('odor_b' in st || 'odor_b' in se) add('sensory.odor_b', 'generic aversive odour (danger channel)', num(se.odor_b) !== null ? se.odor_b : st.odor_b, 'mean of the two antennal samples');
        if ('odor_conc' in st) add('stimuli.odor_conc', 'assay odour field', st.odor_conc, 'body centre');
        if ('odor_cs_plus' in st) add('stimuli.odor_cs_plus', 'CS+ odour', st.odor_cs_plus, 'body centre');
        if ('odor_cs_minus' in st) add('stimuli.odor_cs_minus', 'CS− odour', st.odor_cs_minus, 'body centre');
        if ('cva_concentration' in st) add('stimuli.cva_concentration', 'cVA field', st.cva_concentration, 'body centre (not antennal)');
        if ('aphrodisiac_concentration' in st) add('stimuli.aphrodisiac_concentration', 'aphrodisiac field', st.aphrodisiac_concentration, 'body centre');
        return rows;
    }

    // Wind as published.  Direction is the direction the air flows TOWARD.
    function windOf(pkt) {
        const st = pkt.stimuli || {};
        const se = pkt.sensory || {};
        let v = pt(st.wind);
        let key = 'stimuli.wind';
        if (!v && num(se.wind_x) !== null && num(se.wind_y) !== null) { v = [se.wind_x, se.wind_y]; key = 'sensory.wind_x/y'; }
        if (!v) return {available: false, text: UNAVAILABLE + ' · no wind vector in this frame'};
        const speed = Math.hypot(v[0], v[1]);
        return {
            available: true, vx: v[0], vy: v[1], key, speed,
            towardDeg: speed > 0 ? (Math.atan2(v[1], v[0]) * 180 / Math.PI) : null,
            published: {
                wind_speed: num(st.wind_speed), wind_direction_rad: num(st.wind_direction_rad),
                wind_magnitude: num(st.wind_magnitude), egocentric_wind: num(st.egocentric_wind)
            },
            note: 'uniform vector for the whole assay (mm/s, arena frame): one arrow, not a flow field'
        };
    }

    // Feature detection, not release status: telemetry with the graph I/O v3+ mapping
    // publishes a per-frame `delivery` on the jon_wind row and declares
    // graph-arena-io-v3 or later.  Whether a build is a candidate or deployed is claimed
    // ONLY from explicit build provenance in the packet (`build_provenance.status` or
    // `identity.build_status`), never inferred from the I/O version.
    const IO_V3_LABEL = 'Graph I/O v3+ sensory-delivery mapping';
    function ioFeatures(pkt) {
        const id = (pkt && pkt.identity) || {};
        const declared = JSON.stringify(id.graph_io || (pkt && pkt.connectome && pkt.connectome.graph_io) || '');
        const m = declared.match(/graph-arena-io-v(\d+)/);
        const stages = pkt && pkt.connectome && Array.isArray(pkt.connectome.input_stage) ? pkt.connectome.input_stage : [];
        const rowDelivery = stages.some((r) => r && typeof r.delivery === 'string');
        const version = m ? Number(m[1]) : null;
        const prov = (pkt && pkt.build_provenance && typeof pkt.build_provenance.status === 'string') ? pkt.build_provenance.status
            : (typeof id.build_status === 'string' ? id.build_status : null);
        return {v3Mapping: rowDelivery || (version !== null && version >= 3), version, rowDelivery, buildStatus: prov};
    }

    // Delivered receptor input, row by row as the telemetry reports it.  Never computed here.
    function inputRows(pkt) {
        const id = pkt.identity || {};
        const backend = String(id.backend || '');
        const stages = pkt.connectome && Array.isArray(pkt.connectome.input_stage) ? pkt.connectome.input_stage : null;
        if (!stages) {
            const why = backend === 'modular' ? 'modular controller: no connectome receptor input exists in this mode'
                : backend === 'reference-flygym' ? 'reference fly: not the connectome'
                    : 'this frame carries no connectome.input_stage';
            return {available: false, text: UNAVAILABLE + ' · ' + why, rows: []};
        }
        const feat = ioFeatures(pkt);
        const rows = stages.map((s) => {
            const cur = s.current_injected;
            const spikes = s.n_spiking_this_step;
            let state, text;
            if (typeof s.delivery === 'string' && /^NOT DELIVERED/.test(s.delivery)) {
                state = 'not-delivered'; text = s.delivery + ' (as reported)';
            } else if (typeof s.delivery === 'string' && /^UNAVAILABLE/.test(s.delivery)) {
                state = 'unavailable'; text = s.delivery + ' (as reported)';
            } else if (backend === BRIDGE_RPC && !feat.v3Mapping && BRIDGE_MISSING.includes(s.name)) {
                state = 'not-delivered'; text = NOT_DELIVERED + ' (RPC bridge packet keys do not match; W1 §0.5, unverified by a run)';
            } else if (cur === null || cur === undefined) {
                state = 'unavailable'; text = UNAVAILABLE + (s.unavailable ? ' · ' + s.unavailable : (s.detail ? ' · ' + s.detail : ''));
            } else if (!(cur > 0)) {
                state = 'zero'; text = 'no input this step (0, as reported)';
            } else if (spikes === null || spikes === undefined) {
                state = 'delivered'; text = 'delivered; spike count not exposed';
            } else if (spikes > 0) {
                state = 'delivered-spiking'; text = 'delivered; ' + spikes + ' of ' + s.n_cells + ' cells spiked';
            } else {
                state = 'delivered-silent'; text = 'delivered but not detected: 0 of ' + s.n_cells + ' cells spiked';
            }
            if (typeof s.delivery === 'string' && state !== 'not-delivered' && state !== 'unavailable')
                text += ' · reported: ' + s.delivery + (s.stimulus_key ? ' (key ' + s.stimulus_key + ')' : '');
            return {name: String(s.name || '?'), cellTypes: s.cell_types, delivery: s.delivery || null, stimulusKey: s.stimulus_key || null, current: num(cur), nCells: num(s.n_cells),
                spiking: num(spikes), status: s.status || null, threshold: s.threshold, formula: s.formula || null, state, text};
        });
        return {available: true, rows, units: 'model current units (arbitrary)'};
    }

    function responseOf(pkt) {
        const rates = (pkt.connectome && pkt.connectome.dn_rates) || null;
        if (!rates || typeof rates !== 'object') return {available: false, text: UNAVAILABLE + ' · no connectome descending-neuron rates in this frame'};
        return {available: true, units: 'Hz', rates: Object.entries(rates).filter(([, v]) => num(v) !== null).map(([k, v]) => [k, v])};
    }

    // Contract notes that apply to the current assay/backend (quoted, not computed).
    function caveats(pkt) {
        const id = pkt.identity || {};
        const assay = String(id.assay || pkt.paradigm || '');
        const st = pkt.stimuli || {};
        const out = [];
        const feat = ioFeatures(pkt);
        if (feat.v3Mapping) {
            out.push(IO_V3_LABEL + (feat.version ? ' (declared graph-arena-io-v' + feat.version + ')' : '') + ': wind delivery to the graph is reported per frame in the jon_wind row. Delivery is not detection.'
                + (feat.buildStatus ? ' Build status (from build provenance): ' + feat.buildStatus + '.' : ''));
            if (pt(st.wind) && !('wind_speed' in st) && !('wind_magnitude' in st)) out.push('Wind → graph: ' + NOT_DELIVERED + ' (vector-only wind; no vector-to-probe mapping is defined).');
            if (/wind-tunnel/.test(assay)) out.push('Wind → graph (jon_wind) only above 3 mm/s, speed only; wind direction is not encoded in the graph.');
            if (String(id.backend) === BRIDGE_RPC) out.push('RPC bridge mode with the v3+ mapping: odour, cVA, wind and temperature values are shown as reported.');
            if ('food_contact' in st) out.push('Food contact: distance test (3.5–4 mm), not taste. Taste is ' + NOT_SIMULATED + '.');
            return out;
        }
        if (/multisensory/.test(assay)) out.push('Wind → graph: ' + NOT_DELIVERED + ' in this assay. The arena emits wind_magnitude, the graph probe reads wind_speed (known key mismatch, W1 §0.4; backend fix is a separate card). The jon_wind row below is shown exactly as the telemetry reports it.');
        else if (/t-maze|open-arena|labyrinth/.test(assay) || (pt(st.wind) && !('wind_speed' in st))) out.push('Wind → graph: ' + NOT_DELIVERED + ' (no wind_speed in this assay; W1 §0.4).');
        if (/wind-tunnel/.test(assay)) out.push('Wind → graph (jon_wind) only above 3 mm/s, speed only; wind direction is not encoded in the graph.');
        if (String(id.backend) === BRIDGE_RPC) out.push('RPC bridge mode: odour, cVA, wind and temperature are suspected NOT DELIVERED to the graph (packet key mismatch, W1 §0.5). Only looming (and optomotor) arrive.');
        if ('food_contact' in st) out.push('Food contact: distance test (3.5–4 mm), not taste. Taste is ' + NOT_SIMULATED + '.');
        return out;
    }

    const NOT_SIMULATED_LIST = [
        'Odour transport, advection by wind, turbulence, filaments, flow around obstacles',
        'Chemical identity (vinegar, acids, esters, CO2, geosmin as such, yeast volatiles)',
        'Emission rate, source depletion, odour lifetime',
        'Left/right odour at each antenna (not published) and any bilateral input to the connectome',
        'Wind direction as an input to the connectome',
        'Sugar, bitter and water taste; gustatory neuron input',
        'Predator bodies and predator movement (decoration never changes looming θ or retinal input)',
        'Sex-specific chemosensory processing beyond the fixed MaleCNS dataset'
    ];

    function viewModel(pkt) {
        if (!pkt || typeof pkt !== 'object') return null;
        const id = pkt.identity || {};
        const fly = pkt.fly && num(pkt.fly.x) !== null ? {x: pkt.fly.x, y: pkt.fly.y, heading: num(pkt.fly.heading)} : null;
        const scene = pkt.scene || {};
        const st = pkt.stimuli || {};
        const looming = 'theta_rad' in st || 'theta_deg' in st ? {theta_deg: num(st.theta_deg), theta_rad: num(st.theta_rad),
            expansion_rate_rad_s: num(st.expansion_rate_rad_s), time_to_collision_s: num(st.time_to_collision_s)} : null;
        const predators = Array.isArray(scene.predators) ? scene.predators.length : null;
        return {
            step: num(pkt.step), simTime: num(pkt.sim_time_s),
            assay: String(id.assay || pkt.paradigm || '?'), backend: String(id.backend || '?'),
            graph: GRAPH_BACKENDS.includes(String(id.backend)), bridgeRpc: String(id.backend) === BRIDGE_RPC,
            bounds: Array.isArray(pkt.world_bounds) && pkt.world_bounds.length === 4 && pkt.world_bounds.every((v) => num(v) !== null) ? pkt.world_bounds : null,
            fly, antennae: antennaPoints(fly), sources: sourcesOf(scene),
            plume: pt(scene.nozzle_pos) && num(scene.filament_sigma) !== null ? {nozzle: pt(scene.nozzle_pos), sigma: scene.filament_sigma} : null,
            concentration: concentrationRows(pkt), lateral: NOT_PUBLISHED + ' (left/right antennal values stay inside the simulation)',
            wind: windOf(pkt), input: inputRows(pkt), response: responseOf(pkt), looming,
            predators: predators === null ? UNAVAILABLE : (predators === 0 ? 'none live (the daemon builds arenas with no predators)' : predators + ' in scene'),
            caveats: caveats(pkt), notSimulated: NOT_SIMULATED_LIST.slice(), io: ioFeatures(pkt)
        };
    }

    // Illustrative only (W1 §1.2 wind tunnel): C = exp(−(lateral)²/(2σ²)) · exp(−downwind/250 mm),
    // downwind measured along the published wind vector from the nozzle; 0 upwind.
    function illustrativePlume(nozzle, sigma, wind, p) {
        const s = Math.hypot(wind[0], wind[1]);
        if (!(s > 0) || !(sigma > 0)) return null;
        const ux = wind[0] / s, uy = wind[1] / s;
        const dx = p[0] - nozzle[0], dy = p[1] - nozzle[1];
        const down = dx * ux + dy * uy;
        if (down < 0) return 0;
        const lat = -dx * uy + dy * ux;
        return Math.exp(-(lat * lat) / (2 * sigma * sigma)) * Math.exp(-down / 250);
    }
    const PLUME_EQUATION = 'C(p) = exp(−ℓ²/(2σ²)) · exp(−d/250 mm), d = downwind distance from the nozzle along the published wind vector, ℓ = lateral offset, σ = filament_sigma; C = 0 upwind';
    const PLUME_ASSUMPTIONS = 'Reconstructed in the browser from the W1 contract (maze.py:2644-2651) and published nozzle_pos, filament_sigma and wind. Static analytic shape: no transport, no turbulence, no obstacles. Not telemetry; compare with the measured odor_conc at the fly.';

    function parseQuery(search) {
        const q = new URLSearchParams(search || '');
        const daemon = q.get('daemon');
        const src = q.get('src');
        return {
            daemon: daemon && /^https?:\/\/(127\.0\.0\.1|localhost)(:\d+)?$/.test(daemon) ? daemon : null,
            daemonRejected: !!daemon && !/^https?:\/\/(127\.0\.0\.1|localhost)(:\d+)?$/.test(daemon),
            src: src && !/^[a-z]+:\/\//i.test(src) && !src.startsWith('//') ? src : null,
            hide: new Set((q.get('hide') || '').split(',').filter(Boolean))
        };
    }

    // ------------------------------------------------------------------ browser
    function start(env) {
        const doc = env.document, win = env.window;
        const $ = (id) => doc.getElementById(id);
        const query = parseQuery(env.location.search);
        const st = {pkt: null, lastAt: null, beatPaused: false, disconnected: false, error: null, mode: null,
            frames: [], frameIndex: 0, replayPaused: false, layers: new Set(['sources', 'ids', 'fly', 'antennae', 'wind', 'scale']),
            errors: []};
        for (const h of query.hide) st.layers.delete(h);
        win.__inspector = st;
        const report = (t) => { st.errors.push(t); const e = $('errors'); e.hidden = false; e.textContent = 'Problems:\n' + st.errors.join('\n'); };
        win.addEventListener('error', (e) => report('page error: ' + e.message));

        const now = () => (env.now ? env.now() : Date.now()) / 1000;
        function fresh() {
            return freshness({haveSource: !!st.mode, error: st.error, pkt: st.pkt, ageS: st.lastAt === null ? null : now() - st.lastAt,
                mode: st.mode, replayPaused: st.replayPaused, disconnected: st.disconnected, beatPaused: st.beatPaused});
        }

        let es = null;
        function connect(base) {
            if (es) es.close();
            st.mode = 'live'; st.error = null; st.disconnected = false; st.pkt = null;
            $('sourceInfo').textContent = 'Live, read-only: ' + base + '/api/stream (no command is ever sent)';
            try { es = new win.EventSource(base + '/api/stream'); } catch (e) { st.error = 'cannot open stream: ' + e.message; render(); return; }
            es.onmessage = (ev) => {
                let pkt;
                try { pkt = JSON.parse(ev.data); } catch (e) { report('unparseable frame skipped'); return; }
                if (!pkt || pkt.type !== 'telemetry') return;
                st.pkt = pkt; st.lastAt = now(); st.disconnected = false; st.beatPaused = false; render();
            };
            es.addEventListener('heartbeat', (ev) => {
                try { const b = JSON.parse(ev.data); st.beatPaused = !!(b.paused || (b.liveness && b.liveness.state === 'paused')); } catch (e) { /* ignore */ }
            });
            es.onerror = () => {
                if (!st.pkt) st.error = 'daemon not reachable at ' + base;
                else st.disconnected = true;
                render();
            };
        }
        function loadReplay(url) {
            st.mode = 'replay'; st.error = null;
            $('sourceInfo').textContent = 'Replay of recorded frames: ' + url + ' (not live)';
            fetch(url, {cache: 'no-store'}).then((r) => { if (!r.ok) throw new Error('HTTP ' + r.status); return r.text(); }).then((text) => {
                let frames;
                const t = text.trim();
                if (t.startsWith('[')) frames = JSON.parse(t);
                else if (t.startsWith('{') && !t.includes('\n{')) { const o = JSON.parse(t); frames = o.packets || [o]; }
                else frames = t.split('\n').map((l) => l.replace(/^data:\s*/, '')).filter((l) => l.startsWith('{')).map((l) => JSON.parse(l));
                st.frames = frames.filter((p) => p && p.type === 'telemetry');
                if (!st.frames.length) throw new Error('no telemetry frames in ' + url);
                st.frameIndex = 0; st.pkt = st.frames[0]; st.lastAt = now(); render();
            }).catch((e) => { st.error = 'replay unavailable: ' + e.message; render(); });
        }

        // ------------------------------------------------------------ drawing
        const canvas = $('map');
        function shapePath(g, shape, x, y, r) {
            g.beginPath();
            if (shape === 'square') g.rect(x - r, y - r, 2 * r, 2 * r);
            else if (shape === 'triangle') { g.moveTo(x, y - r * 1.2); g.lineTo(x + r * 1.1, y + r * 0.8); g.lineTo(x - r * 1.1, y + r * 0.8); g.closePath(); }
            else if (shape === 'diamond') { g.moveTo(x, y - r * 1.3); g.lineTo(x + r, y); g.lineTo(x, y + r * 1.3); g.lineTo(x - r, y); g.closePath(); }
            else if (shape === 'cross') { g.moveTo(x - r, y); g.lineTo(x + r, y); g.moveTo(x, y - r); g.lineTo(x, y + r); }
            else if (shape === 'star') { for (let k = 0; k < 10; k += 1) { const a = -Math.PI / 2 + k * Math.PI / 5, rr = k % 2 ? r * 0.45 : r * 1.2; g[k ? 'lineTo' : 'moveTo'](x + rr * Math.cos(a), y + rr * Math.sin(a)); } g.closePath(); }
            else g.arc(x, y, r, 0, 2 * Math.PI);
        }
        function arrow(g, x0, y0, x1, y1, colour, width) {
            const a = Math.atan2(y1 - y0, x1 - x0);
            g.strokeStyle = colour; g.fillStyle = colour; g.lineWidth = width;
            g.beginPath(); g.moveTo(x0, y0); g.lineTo(x1, y1); g.stroke();
            g.beginPath(); g.moveTo(x1, y1); g.lineTo(x1 - 10 * Math.cos(a - 0.4), y1 - 10 * Math.sin(a - 0.4));
            g.lineTo(x1 - 10 * Math.cos(a + 0.4), y1 - 10 * Math.sin(a + 0.4)); g.closePath(); g.fill();
        }
        function draw(vm, f) {
            const W = canvas.clientWidth || 600, H = canvas.clientHeight || 600;
            const dpr = Math.min(win.devicePixelRatio || 1, 2);
            canvas.width = W * dpr; canvas.height = H * dpr;
            const g = canvas.getContext('2d');
            g.setTransform(dpr, 0, 0, dpr, 0, 0);
            g.fillStyle = '#060913'; g.fillRect(0, 0, W, H);
            if (!vm || !vm.bounds) {
                g.fillStyle = '#fbbf24'; g.font = '14px monospace';
                g.fillText(vm ? UNAVAILABLE + ': no world_bounds in this frame' : f.text, 16, 30);
                return;
            }
            const [x0, y0, x1, y1] = vm.bounds;
            const pad = 34, s = Math.min((W - 2 * pad) / (x1 - x0), (H - 2 * pad) / (y1 - y0));
            const ox = (W - s * (x1 - x0)) / 2, oy = (H - s * (y1 - y0)) / 2;
            const X = (x) => ox + (x - x0) * s, Y = (y) => H - oy - (y - y0) * s;   // arena +y is up on screen
            g.strokeStyle = '#334155'; g.lineWidth = 1; g.strokeRect(X(x0), Y(y1), (x1 - x0) * s, (y1 - y0) * s);

            if (st.layers.has('plume') && vm.plume && vm.wind.available) {
                const step = 6;
                for (let py = Y(y1); py < Y(y0); py += step) for (let px = X(x0); px < X(x1); px += step) {
                    const c = illustrativePlume(vm.plume.nozzle, vm.plume.sigma, [vm.wind.vx, vm.wind.vy], [(px - ox) / s + x0, (H - oy - py) / s + y0]);
                    if (c > 0.02) { g.fillStyle = 'rgba(56,189,248,' + (0.55 * c).toFixed(3) + ')'; g.fillRect(px, py, step, step); }
                }
                g.fillStyle = '#fbbf24'; g.font = 'bold 11px monospace';
                g.fillText('ILLUSTRATIVE static shape (not telemetry, not transport)', X(x0) + 6, Y(y0) - 8);
            }
            if (st.layers.has('scale')) {
                const len = Math.pow(10, Math.floor(Math.log10((x1 - x0) / 4)));
                g.strokeStyle = '#e2e8f0'; g.lineWidth = 2;
                g.beginPath(); g.moveTo(X(x0) + 8, H - 12); g.lineTo(X(x0) + 8 + len * s, H - 12); g.stroke();
                g.fillStyle = '#e2e8f0'; g.font = '11px monospace'; g.fillText(len + ' mm', X(x0) + 8, H - 16);
                g.fillText('+x →  +y ↑ (arena frame, mm)', X(x1) - 200, H - 12);
            }
            const placed = [];
            if (st.layers.has('sources')) for (const src of vm.sources) {
                const px = X(src.pos[0]), py = Y(src.pos[1]);
                // Co-located sources (e.g. food and cool spot) keep separate, stacked IDs.
                const stack = placed.filter(([qx, qy]) => Math.hypot(qx - px, qy - py) < 6).length;
                placed.push([px, py]);
                shapePath(g, src.shape, px, py, 8);
                g.lineWidth = 2; g.strokeStyle = src.colour;
                if (src.shape === 'cross') g.stroke(); else { g.fillStyle = src.colour + '55'; g.fill(); g.stroke(); }
                if (st.layers.has('ids')) { g.fillStyle = '#e2e8f0'; g.font = 'bold 12px monospace'; g.fillText(src.id + (stack ? ' (same point)' : ''), px + 11, py - 9 + 14 * stack); }
            }
            if (vm.plume && st.layers.has('sources')) {
                g.setLineDash([4, 4]); g.strokeStyle = '#38bdf8'; g.beginPath();
                g.arc(X(vm.plume.nozzle[0]), Y(vm.plume.nozzle[1]), vm.plume.sigma * s, 0, 2 * Math.PI); g.stroke(); g.setLineDash([]);
            }
            if (vm.fly && st.layers.has('fly')) {
                const fx = X(vm.fly.x), fy = Y(vm.fly.y);
                g.fillStyle = '#e2e8f0'; g.beginPath(); g.arc(fx, fy, 5, 0, 2 * Math.PI); g.fill();
                if (vm.fly.heading !== null) arrow(g, fx, fy, fx + 18 * Math.cos(vm.fly.heading), fy - 18 * Math.sin(vm.fly.heading), '#e2e8f0', 2);
                if (st.layers.has('ids')) { g.font = 'bold 12px monospace'; g.fillText('fly', fx + 8, fy + 16); }
            }
            if (vm.antennae && st.layers.has('antennae')) {
                for (const [lab, p] of [['L*', vm.antennae.left], ['R*', vm.antennae.right]]) {
                    const ax = X(p[0]), ay = Y(p[1]);
                    g.strokeStyle = '#facc15'; g.lineWidth = 1.5; g.beginPath(); g.arc(ax, ay, 4, 0, 2 * Math.PI); g.stroke();
                    if (st.layers.has('ids')) { g.fillStyle = '#facc15'; g.font = '11px monospace'; g.fillText(lab, ax + 5, ay - 5); }
                }
            }
            if (vm.wind.available && st.layers.has('wind')) {
                const cx = W - 70, cy = 60;
                g.strokeStyle = '#334155'; g.lineWidth = 1; g.beginPath(); g.arc(cx, cy, 34, 0, 2 * Math.PI); g.stroke();
                if (vm.wind.speed > 0) arrow(g, cx - 26 * vm.wind.vx / vm.wind.speed, cy + 26 * vm.wind.vy / vm.wind.speed,
                    cx + 26 * vm.wind.vx / vm.wind.speed, cy - 26 * vm.wind.vy / vm.wind.speed, '#e2e8f0', 3);
                g.fillStyle = '#e2e8f0'; g.font = 'bold 11px monospace';
                g.fillText('wind ' + vm.wind.speed.toFixed(1) + ' mm/s', cx - 52, cy + 50);
                g.fillText(vm.wind.towardDeg === null ? 'calm' : 'toward ' + vm.wind.towardDeg.toFixed(0) + '°', cx - 40, cy + 64);
            }
            if (f.state !== 'live' && f.state !== 'replay') {
                g.fillStyle = 'rgba(6,9,19,0.45)'; g.fillRect(0, 0, W, H);
                g.fillStyle = f.state === 'paused' || f.state === 'replay-paused' ? '#94a3b8' : '#fbbf24';
                g.font = 'bold 16px monospace'; g.fillText(f.text, 16, 26);
            }
        }

        // ------------------------------------------------------------ panels
        const fmt = (v, d) => (v === null || v === undefined ? '–' : Number(v).toFixed(d === undefined ? 3 : d));
        function el(tag, attrs, text) {
            const e = doc.createElement(tag);
            for (const [k, v] of Object.entries(attrs || {})) e.setAttribute(k, v);
            if (text !== undefined) e.textContent = text;
            return e;
        }
        function table(id, head, rows) {
            const t = $(id); t.textContent = '';
            const tr = el('tr'); for (const h of head) tr.append(el('th', {}, h)); t.append(tr);
            for (const r of rows) {
                const row = el('tr', r.attrs || {});
                for (const c of r.cells) row.append(el('td', c.cls ? {class: c.cls} : {}, c.text));
                t.append(row);
            }
        }
        function render() {
            const f = fresh();
            const vm = viewModel(st.pkt);
            const banner = $('freshness');
            banner.textContent = f.text; banner.dataset.state = f.state;
            doc.body.dataset.freshness = f.state;
            const dim = f.state !== 'live' && f.state !== 'replay';
            for (const sec of doc.querySelectorAll('.data')) sec.classList.toggle('dim', dim);
            draw(vm, f);
            if (!vm) {
                for (const id of ['srcTable', 'concTable', 'inputTable', 'respTable']) $(id).textContent = '';
                $('wind').textContent = f.text; $('caveats').textContent = ''; $('ident').textContent = '';
                return;
            }
            $('ident').textContent = (vm.io.v3Mapping ? '[' + IO_V3_LABEL + ' · delivery is not detection'
                + (vm.io.buildStatus ? ' · build: ' + vm.io.buildStatus : '') + '] ' : '')
                + 'assay ' + vm.assay + ' · backend ' + vm.backend + (vm.graph ? ' (connectome graph)' : '')
                + ' · step ' + fmt(vm.step, 0) + ' · sim t ' + fmt(vm.simTime, 2) + ' s';
            table('srcTable', ['ID', 'source', 'x, y (mm)', 'emission'], vm.sources.length ? vm.sources.map((s) => ({cells: [
                {text: s.id}, {text: s.label + ' [' + s.key + ']'}, {text: fmt(s.pos[0], 1) + ', ' + fmt(s.pos[1], 1)}, {text: s.emission}]}))
                : [{cells: [{text: '–'}, {text: 'no source positions published for this assay'}, {text: '–'}, {text: '–'}]}]);
            table('concTable', ['key', 'what', 'value', 'sampled at'], vm.concentration.map((r) => ({cells: [
                {text: r.key}, {text: r.label}, {text: fmt(r.value)}, {text: r.where}]})).concat([
                {cells: [{text: 'L* / R*'}, {text: 'per-antenna odour'}, {text: NOT_PUBLISHED, cls: 'ns'}, {text: 'derived points drawn on the map only'}]}]));
            const w = vm.wind;
            $('wind').textContent = w.available
                ? 'vector (' + fmt(w.vx, 2) + ', ' + fmt(w.vy, 2) + ') mm/s from ' + w.key + ' · speed ' + fmt(w.speed, 2) + ' mm/s · toward '
                  + (w.towardDeg === null ? '– (calm)' : fmt(w.towardDeg, 0) + '° (0° = +x, CCW)') + '\npublished extras: '
                  + Object.entries(w.published).map(([k, v]) => k + ' ' + (v === null ? 'absent' : fmt(v, 3))).join(' · ') + '\n' + w.note
                : w.text;
            if (vm.input.available) {
                table('inputTable', ['probe', 'cells', 'current injected', 'status', 'delivery (as reported)'], vm.input.rows.map((r) => ({
                    attrs: {'data-probe': r.name, 'data-state': r.state}, cells: [
                        {text: r.name + (r.cellTypes ? ' (' + [].concat(r.cellTypes).join(', ') + ')' : '')},
                        {text: r.nCells === null ? '–' : String(r.nCells)},
                        {text: r.current === null ? UNAVAILABLE : fmt(r.current, 3), cls: r.current === null ? 'ns' : ''},
                        {text: r.status || '–'}, {text: r.text, cls: /NOT|UNAVAIL/.test(r.text) ? 'ns' : ''}]})));
            } else table('inputTable', ['receptor input to the connectome'], [{cells: [{text: vm.input.text, cls: 'ns'}]}]);
            const rows = vm.input.available ? vm.input.rows.map((r) => ({cells: [{text: r.name + ' cells spiking'}, {text: r.spiking === null ? UNAVAILABLE : String(r.spiking)}, {text: 'count this step'}]})) : [];
            if (vm.response.available) for (const [k, v] of vm.response.rates) rows.push({cells: [{text: 'DN ' + k}, {text: fmt(v, 2)}, {text: 'Hz'}]});
            table('respTable', ['neural response', 'value', 'unit'], rows.length ? rows : [{cells: [{text: vm.response.text, cls: 'ns'}, {text: '–'}, {text: '–'}]}]);
            $('looming').textContent = vm.looming ? 'θ ' + fmt(vm.looming.theta_deg, 2) + '° · expansion ' + fmt(vm.looming.expansion_rate_rad_s, 3)
                + ' rad/s · TTC ' + fmt(vm.looming.time_to_collision_s, 2) + ' s (abstract expanding disc; predator decoration never changes θ)'
                : 'no looming stimulus in this assay';
            $('predators').textContent = 'Predators: ' + vm.predators + '. Predator models in the gallery are decoration only.';
            $('caveats').textContent = vm.caveats.join('\n');
        }

        // ------------------------------------------------------------ controls
        for (const box of doc.querySelectorAll('input[data-layer]')) {
            box.checked = st.layers.has(box.dataset.layer);
            box.addEventListener('change', () => { if (box.checked) st.layers.add(box.dataset.layer); else st.layers.delete(box.dataset.layer); render(); });
        }
        $('connectBtn').addEventListener('click', () => {
            const v = $('daemonUrl').value.trim();
            const q = parseQuery('?daemon=' + encodeURIComponent(v));
            if (!q.daemon) { report('only a loopback daemon URL is accepted (http://127.0.0.1:PORT)'); return; }
            connect(q.daemon);
        });
        $('playBtn').addEventListener('click', () => { st.replayPaused = !st.replayPaused; $('playBtn').textContent = st.replayPaused ? 'play' : 'pause'; render(); });
        $('stepBtn').addEventListener('click', () => { if (st.frames.length) { st.frameIndex = (st.frameIndex + 1) % st.frames.length; st.pkt = st.frames[st.frameIndex]; render(); } });
        $('notSim').textContent = '';
        for (const n of NOT_SIMULATED_LIST) $('notSim').append(el('li', {}, n + ' — ' + NOT_SIMULATED));
        $('plumeEq').textContent = PLUME_EQUATION + '\n' + PLUME_ASSUMPTIONS;

        if (query.daemonRejected) report('daemon must be a loopback URL; ignored');
        if (query.src) { loadReplay(query.src); $('replayCtl').hidden = false; }
        else if (query.daemon) { $('daemonUrl').value = query.daemon; connect(query.daemon); }
        render();
        win.setInterval(() => {
            if (st.mode === 'replay' && !st.replayPaused && st.frames.length) {
                st.frameIndex = (st.frameIndex + 1) % st.frames.length; st.pkt = st.frames[st.frameIndex]; st.lastAt = now();
            }
            render();
        }, 500);
        return st;
    }

    return {
        STALE_AFTER_S, UNITS_ODOUR, NOT_SIMULATED, NOT_PUBLISHED, NOT_DELIVERED, UNAVAILABLE, ANTENNA_LENGTH_MM, ANTENNA_ANGLE_RAD,
        BRIDGE_RPC, SOURCE_KEYS, NOT_SIMULATED_LIST, PLUME_EQUATION, PLUME_ASSUMPTIONS,
        freshness, antennaPoints, sourcesOf, concentrationRows, windOf, inputRows, responseOf, caveats, viewModel,
        illustrativePlume, parseQuery, ioFeatures, IO_V3_LABEL, start
    };
}));
