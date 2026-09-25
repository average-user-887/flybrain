// Experiment studio page (roadmap P5). Talks to neurofly_studio.server on the same origin.
(() => {
    'use strict';
    const $ = (id) => document.getElementById(id);
    const esc = (value) => String(value ?? '').replace(/[&<>"']/g, (c) => ({
        '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
    const state = { catalog: null, runs: null, paradigm: null, tab: 'gallery', timer: null };

    async function api(path, options) {
        const response = await fetch(path, options);
        const body = await response.json().catch(() => ({}));
        if (!response.ok) throw new Error(body.error || ('HTTP ' + response.status));
        return body;
    }

    function showError(message) {
        const box = $('global-error');
        box.hidden = !message;
        box.textContent = message || '';
    }

    // ---- tabs -------------------------------------------------------------
    function selectTab(tab) {
        state.tab = tab;
        document.querySelectorAll('nav [data-tab]').forEach((b) => b.setAttribute('aria-selected', String(b.dataset.tab === tab)));
        document.querySelectorAll('main > section').forEach((s) => { s.hidden = s.id !== 'tab-' + tab; });
        refreshRuns();
    }
    document.querySelectorAll('nav [data-tab]').forEach((b) => b.addEventListener('click', () => selectTab(b.dataset.tab)));

    // ---- runs -------------------------------------------------------------
    const runKey = (run) => run.source + '/' + run.name;
    const fileUrl = (run, file) => '/api/studio/files/' + encodeURIComponent(run.source) + '/' + encodeURIComponent(run.name) + '/' + file;
    const replayUrl = (run) => 'embodied_replay.html?src=' + encodeURIComponent(fileUrl(run, 'body.nfbody'));
    function roleText(run) {
        const silenced = (run.silence || []).join(', ');
        if (run.role === 'intact') return 'Control: nothing silenced';
        if (run.role === 'output-disconnected') return 'Control: brain disconnected from legs' + (silenced ? ' (' + silenced + ' silenced)' : '');
        return silenced ? 'Silenced: ' + silenced : 'Intact fly';
    }
    const clampWarning = (run) => run.clamp_held === false
        ? ' <span class="badge state-failed" title="Some silenced neurons still spiked; the clamp did not hold for this run">clamp leaked</span>' : '';

    function paramText(run) {
        const p = run.parameters || {};
        const bits = [];
        if (p.world_angular_velocity_rad_s !== undefined) bits.push('rotation ' + p.world_angular_velocity_rad_s + ' rad/s');
        if (p.contrast !== undefined) bits.push('contrast ' + p.contrast);
        if (p.duration_s !== undefined) bits.push(p.duration_s + ' s');
        if (p.seed !== undefined) bits.push('seed ' + p.seed);
        if (run.controller === 'modular') bits.push('modular baseline controller');
        return bits.join(' · ');
    }

    function finishedRuns() {
        if (!state.runs) return [];
        return [...state.runs.curated, ...state.runs.queue.filter((r) => r.state === 'done')];
    }

    function runCard(run) {
        const pairRun = run.pair && finishedRuns().find((r) => r.source === run.source && r.name === run.pair);
        return '<div class="card" data-key="' + esc(runKey(run)) + '">' +
            '<div class="title">' + esc(run.title) + '</div>' +
            '<div><span class="badge Exploratory">' + esc(run.label || 'exploratory') + '</span> ' +
            '<span class="meta">' + esc(roleText(run)) + '</span>' + clampWarning(run) + '</div>' +
            (run.explanation ? '<div class="meta">' + esc(run.explanation) + '</div>' : '') +
            '<div class="meta">' + esc(paramText(run)) + '</div>' +
            '<div class="row">' +
            (run.has_recording ? '<a class="act" target="_blank" rel="noopener" href="' + esc(replayUrl(run)) + '">Watch</a>'
                : '<span class="meta">no 3D recording</span>') +
            (pairRun ? '<button class="act" data-compare="' + esc(runKey(run)) + '" data-with="' + esc(runKey(pairRun)) + '">Compare with pair</button>'
                : '<button class="act" data-compare="' + esc(runKey(run)) + '">Compare…</button>') +
            '</div></div>';
    }

    function renderGallery() {
        const curated = state.runs.curated;
        $('curated').innerHTML = curated.length ? curated.map(runCard).join('')
            : '<p class="empty">No curated runs are installed yet. They will ship with the first public release.</p>';
        const mine = state.runs.queue.filter((r) => r.state === 'done').reverse();
        $('finished').innerHTML = mine.length ? mine.map(runCard).join('')
            : '<p class="empty">Nothing has finished yet. Build an experiment to queue your first run.</p>';
    }

    function fmtTime(job) {
        if (job.state === 'done' || job.state === 'failed') {
            return job.wall_time_s != null ? (job.wall_time_s / 60).toFixed(1) + ' min' : '';
        }
        if (job.state === 'running' && job.started_at) {
            return 'since ' + new Date(job.started_at).toLocaleTimeString();
        }
        return job.added_at ? 'added ' + new Date(job.added_at).toLocaleTimeString() : '';
    }

    function renderQueue() {
        const jobs = state.runs.queue;
        $('queue-rows').innerHTML = jobs.length ? jobs.map((job) =>
            '<tr><td>' + esc(Number(job.order)) + '</td><td>' + esc(job.title) + '<div class="meta">' + esc(paramText(job)) + '</div></td>' +
            '<td>' + esc(roleText(job)) + clampWarning(job) + '</td><td>' + esc(job.seed ?? '') + '</td>' +
            '<td><span class="badge state-' + esc(job.state) + '">' + esc(job.state) + '</span>' +
            (job.state === 'failed' ? '<div class="meta">' + esc(job.error || ('exit ' + job.exit_status)) + '</div>' : '') + '</td>' +
            '<td>' + esc(fmtTime(job)) + '</td><td>' +
            (job.state === 'pending' ? '<button class="act" data-cancel="' + esc(job.name) + '">Cancel</button>' : '') +
            (job.state === 'done' && job.has_recording ? '<a class="act" target="_blank" rel="noopener" href="' + esc(replayUrl(job)) + '">Watch</a>' : '') +
            '</td></tr>').join('') : '<tr><td colspan="7" class="empty">The queue is empty.</td></tr>';
    }

    function renderCompareOptions() {
        const runs = finishedRuns();
        for (const id of ['cmp-a', 'cmp-b']) {
            const select = $(id);
            const current = select.value;
            select.innerHTML = '<option value="">Choose a finished run</option>' + runs.map((r) =>
                '<option value="' + esc(runKey(r)) + '">' + esc(r.title + ' · ' + roleText(r) + ' · seed ' + ((r.parameters || {}).seed ?? '?')) + '</option>').join('');
            if (runs.some((r) => runKey(r) === current)) select.value = current;
        }
    }

    async function refreshRuns() {
        try {
            state.runs = await api('/api/studio/runs');
            showError('');
        } catch (err) {
            showError('The studio server is not answering: ' + err.message);
            $('worker').textContent = 'offline';
            return;
        }
        const running = state.runs.queue.filter((j) => j.state === 'running').length;
        const pending = state.runs.queue.filter((j) => j.state === 'pending').length;
        $('worker').textContent = 'queue worker: ' + state.runs.worker + ' · ' + running + ' running · ' + pending + ' waiting';
        renderGallery();
        renderQueue();
        renderCompareOptions();
    }

    // ---- compare ----------------------------------------------------------
    const METRICS = [
        ['world_angular_velocity_rad_s', 'World rotation (rad/s)', 3],
        ['mean_body_yaw_velocity_rad_s', 'Mean turning speed of the fly (rad/s)', 3],
        ['turning_gain', 'Turning gain (fly ÷ world)', 3],
        ['net_yaw_change_deg', 'Net heading change (degrees)', 1],
        ['total_graph_spikes', 'Spikes in the whole brain', 0],
        ['simulated_s', 'Simulated time (s)', 2],
        ['silenced.targets', 'Silenced cell types', null],
        ['silenced.total_neurons', 'Silenced neurons', 0],
        ['silenced.spikes_total', 'Spikes from silenced neurons', 0],
        ['silenced.clamp_held', 'Silencing held (no silenced neuron spiked)', null],
    ];

    async function updateCompare() {
        const keys = [$('cmp-a').value, $('cmp-b').value];
        const runs = finishedRuns();
        const picked = keys.map((k) => runs.find((r) => runKey(r) === k));
        ['cmp-frame-a', 'cmp-frame-b'].forEach((id, i) => {
            const url = picked[i] && picked[i].has_recording ? replayUrl(picked[i]) : 'about:blank';
            if ($(id).dataset.url !== url) { $(id).dataset.url = url; $(id).src = url; }
        });
        const metrics = await Promise.all(picked.map((r) => r ? api('/api/studio/metrics/' + encodeURIComponent(r.source) + '/' + encodeURIComponent(r.name)).catch((e) => ({ error: e.message })) : null));
        const cell = (m, key, digits) => {
            if (!m) return '';
            if (m.error) return esc(m.error);
            const v = key.split('.').reduce((o, k) => (o == null ? o : o[k]), m);
            if (v === null || v === undefined) return '–';
            if (Array.isArray(v)) return esc(v.join(', '));
            if (typeof v === 'boolean') return v ? 'yes' : '<strong style="color:var(--warn)">no: silenced neurons spiked</strong>';
            return esc(Number(v).toFixed(digits));
        };
        $('cmp-table').innerHTML = picked.some(Boolean)
            ? '<thead><tr><th></th><th>' + esc(picked[0] ? picked[0].title + ' (' + roleText(picked[0]) + ')' : '') + '</th><th>' +
              esc(picked[1] ? picked[1].title + ' (' + roleText(picked[1]) + ')' : '') + '</th></tr></thead><tbody>' +
              METRICS.map(([key, label, digits]) => '<tr><th>' + esc(label) + '</th><td>' + cell(metrics[0], key, digits) + '</td><td>' + cell(metrics[1], key, digits) + '</td></tr>').join('') + '</tbody>'
            : '';
    }
    $('cmp-a').addEventListener('change', updateCompare);
    $('cmp-b').addEventListener('change', updateCompare);

    document.addEventListener('click', async (event) => {
        const compare = event.target.closest('[data-compare]');
        if (compare) {
            selectTab('compare');
            await refreshRuns();
            // Nothing-silenced run on the left; otherwise the experiment left, its control right.
            const runs = finishedRuns();
            const run = runs.find((r) => runKey(r) === compare.dataset.compare);
            const other = runs.find((r) => runKey(r) === compare.dataset.with);
            const leftFirst = (r) => (r.role === 'intact' ? 0 : r.role === 'experiment' && !(r.silence || []).length ? 0 : r.role === 'experiment' ? 1 : 2);
            const swap = run && other && leftFirst(other) < leftFirst(run);
            $('cmp-a').value = swap ? compare.dataset.with : compare.dataset.compare;
            $('cmp-b').value = swap ? compare.dataset.compare : (compare.dataset.with || '');
            updateCompare();
            return;
        }
        const cancel = event.target.closest('[data-cancel]');
        if (cancel) {
            cancel.disabled = true;
            try {
                await api('/api/studio/runs/' + encodeURIComponent(cancel.dataset.cancel) + '/cancel', {
                    method: 'POST', headers: { 'Content-Type': 'application/json' }, body: '{}' });
            } catch (err) { showError(err.message); }
            refreshRuns();
        }
    });

    // ---- build ------------------------------------------------------------
    function renderParadigms() {
        $('paradigms').innerHTML = state.catalog.paradigms.map((p) =>
            '<div class="card' + (state.paradigm && state.paradigm.id === p.id ? ' selected' : '') + '">' +
            '<div class="title">' + esc(p.number + '. ' + p.title) + '</div>' +
            '<div><span class="badge ' + esc(p.badge) + '" title="' + esc(p.connectome_status) + '">' + esc(p.badge) + '</span></div>' +
            '<div class="meta">' + esc(p.connectome_status.replace(/\*\*/g, '')) + '</div>' +
            '<div class="row">' + (p.buildable ? '<button class="act primary" data-build="' + esc(p.id) + '">Build</button>'
                : '<span class="meta">Not in the studio yet</span>') + '</div></div>').join('');
        document.querySelectorAll('[data-build]').forEach((b) => b.addEventListener('click', () => openBuilder(b.dataset.build)));
    }

    function fieldHtml(p) {
        const id = 'p-' + p.name;
        return '<div class="field"><label for="' + id + '">' + esc(p.label) + (p.unit ? ' (' + esc(p.unit) + ')' : '') + '</label>' +
            '<input type="range" data-range="' + esc(p.name) + '" min="' + p.min + '" max="' + p.max + '" step="' + p.step + '" value="' + p.default + '"' + (p.integer && p.max > 1e6 ? ' hidden' : '') + '>' +
            '<input type="number" id="' + id + '" data-param="' + esc(p.name) + '" min="' + p.min + '" max="' + p.max + '" step="' + p.step + '" value="' + p.default + '" required>' +
            (p.help ? '<div class="help">' + esc(p.help) + '</div>' : '') + '</div>';
    }

    function openBuilder(id) {
        const p = state.catalog.paradigms.find((x) => x.id === id);
        state.paradigm = p;
        renderParadigms();
        $('builder').hidden = false;
        $('b-title').innerHTML = esc(p.title) + ' <span class="badge ' + esc(p.badge) + '">' + esc(p.badge) + '</span>';
        $('b-explain').textContent = p.explanation || '';
        $('b-name').value = p.title;
        $('b-params').innerHTML = p.parameters.map(fieldHtml).join('');
        $('b-silence').hidden = !p.silence_groups.length;
        $('b-silence-list').innerHTML = p.silence_groups.map((g) =>
            '<div class="field"><label for="s-' + esc(g.id) + '">' + esc(g.label) + '</label>' +
            '<select id="s-' + esc(g.id) + '" data-silence="' + esc(g.id) + '"><option value="">Not silenced</option>' +
            '<option value="both">Silenced, both sides</option><option value="L">Silenced, left side only</option>' +
            '<option value="R">Silenced, right side only</option></select><span></span>' +
            '<div class="help">' + esc(g.explanation) + ' Cell types: ' + esc(g.cell_types.join(', ')) + '.</div></div>').join('');
        const controls = [['', 'No control run']].concat(Object.entries(p.controls));
        $('b-controls').innerHTML = '<div class="meta">Matched control run with the same seed:</div>' + controls.map(([key, text]) =>
            '<label class="check"><input type="radio" name="b-control" value="' + esc(key) + '"' + (key === 'intact' ? ' data-needs-silence' : '') + '><span>' + esc(text) + '</span></label>').join('');
        $('b-silence-list').querySelectorAll('[data-silence]').forEach((sel) => sel.addEventListener('change', () => syncControls(true)));
        syncControls(true);
        $('b-prereg').innerHTML = p.preregistered_specs.length
            ? 'Preregistered tests of this paradigm (read-only, never changed from here): ' + p.preregistered_specs.map((s) =>
                '<code>' + esc(s.id) + '</code> declared ' + esc(s.declared_at) + ' <span title="' + esc(s.sha256) + '">sha256 ' + esc(s.sha256.slice(0, 12)) + '…</span>').join('; ')
            : '';
        $('b-result').hidden = true;
        $('b-params').querySelectorAll('[data-range]').forEach((range) => {
            const number = $('p-' + range.dataset.range);
            range.addEventListener('input', () => { number.value = range.value; });
            number.addEventListener('input', () => { range.value = number.value; });
        });
        $('builder').scrollIntoView({ behavior: 'smooth', block: 'start' });
    }

    function silenceTargets() {
        const targets = [];
        $('b-silence-list').querySelectorAll('[data-silence]').forEach((sel) => {
            if (!sel.value) return;
            const group = state.paradigm.silence_groups.find((g) => g.id === sel.dataset.silence);
            group.cell_types.forEach((t) => targets.push(sel.value === 'both' ? t : t + ':' + sel.value));
        });
        return targets;
    }

    // Silencing something pairs it with the same fly un-silenced; otherwise with the disconnected brain.
    function syncControls(pickDefault) {
        const silenced = silenceTargets().length > 0;
        const radios = [...$('b-controls').querySelectorAll('input[name=b-control]')];
        radios.forEach((r) => { if (r.hasAttribute('data-needs-silence')) r.disabled = !silenced; });
        const current = radios.find((r) => r.checked);
        if (pickDefault || !current || current.disabled) {
            const want = silenced ? 'intact' : 'output-disconnected';
            const pick = radios.find((r) => r.value === want && !r.disabled) || radios[0];
            if (pick) pick.checked = true;
        }
    }

    $('builder').addEventListener('submit', async (event) => {
        event.preventDefault();
        const parameters = {};
        $('b-params').querySelectorAll('[data-param]').forEach((input) => { parameters[input.dataset.param] = Number(input.value); });
        const control = [...$('b-controls').querySelectorAll('input[name=b-control]')].find((c) => c.checked);
        const silence = silenceTargets();
        const experiment = {
            schema: 'neurofly-studio-experiment-v1', title: $('b-name').value.trim() || state.paradigm.title,
            paradigm: state.paradigm.id, parameters, repeats: Number($('b-repeats').value),
            control: control && control.value ? control.value : null,
        };
        if (silence.length) experiment.silence = silence;
        const result = $('b-result');
        $('b-submit').disabled = true;
        try {
            const reply = await api('/api/studio/experiments', {
                method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(experiment) });
            result.className = 'notice ok';
            result.textContent = 'Queued ' + reply.queued.length + ' run' + (reply.queued.length === 1 ? '' : 's') + '. Follow them in the Queue tab.';
            refreshRuns();
        } catch (err) {
            result.className = 'notice err';
            result.textContent = 'Not queued: ' + err.message;
        } finally {
            result.hidden = false;
            $('b-submit').disabled = false;
        }
    });

    // ---- start ------------------------------------------------------------
    (async () => {
        try {
            state.catalog = await api('/api/studio/catalog');
            renderParadigms();
        } catch (err) {
            showError('Could not load the paradigm catalog: ' + err.message);
        }
        await refreshRuns();
        state.timer = setInterval(() => { if (!document.hidden) refreshRuns(); }, 5000);
    })();
})();
