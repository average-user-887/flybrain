/*
 * Run replay and the Brain Activity panel.
 *
 * A recording (.nfrec, docs/RECORDING_FORMAT.md) is a gzip stream of JSON lines:
 * a header, one frame per recorded step, input events and an end record.  Frames
 * are the daemon's telemetry packets without wall-clock fields, so the player
 * re-adds those and hands each frame to DaemonBridgeClient.handleDaemonPacket:
 * the replay drives exactly the panels live mode drives.  Playback follows the
 * recorded simulated time (1x = real time of the simulated fly), independent of
 * how long the run took to compute.
 */
(function () {
    'use strict';

    const FORMAT = 'neurofly-run-recording';
    const VERSION = 1;
    const PATH_STEPS = 240;          // same trail window the daemon sends live
    const RASTER_FRAMES = 200;       // raster columns kept (4 s at 20 ms frames)

    const $ = (id) => document.getElementById(id);
    const report = (phase, err, ctx) => {
        try { window.neuroflyErrors?.report(phase, err, ctx || {}); } catch (e) { console.error(err); }
    };

    // ------------------------------------------------------------------ parsing
    async function inflate(bytes) {
        if (bytes.length < 2 || bytes[0] !== 0x1f || bytes[1] !== 0x8b) {
            throw new Error('not a gzip-compressed .nfrec recording');
        }
        const stream = new Blob([bytes]).stream().pipeThrough(new DecompressionStream('gzip'));
        return new Uint8Array(await new Response(stream).arrayBuffer());
    }

    function concat(chunks, total) {
        const out = new Uint8Array(total);
        let offset = 0;
        for (const c of chunks) { out.set(c, offset); offset += c.length; }
        return out;
    }

    async function sha256Hex(bytes) {
        if (!window.crypto?.subtle) return null;       // insecure origin: cannot verify here
        const digest = await window.crypto.subtle.digest('SHA-256', bytes);
        return Array.from(new Uint8Array(digest)).map((b) => b.toString(16).padStart(2, '0')).join('');
    }

    /** Parse and check a recording. Returns {header, frames, events, end, verified}. */
    async function parseRecording(compressed) {
        const raw = await inflate(compressed);
        const decoder = new TextDecoder();
        let header = null, end = null;
        const frames = [], events = [], frameBytes = [];
        let frameTotal = 0, start = 0;
        for (let i = 0; i <= raw.length; i++) {
            if (i < raw.length && raw[i] !== 10) continue;
            if (i > start) {
                const slice = raw.subarray(start, Math.min(i + 1, raw.length));
                const record = JSON.parse(decoder.decode(raw.subarray(start, i)));
                if (record.k === 'f') { frames.push(record); frameBytes.push(slice); frameTotal += slice.length; }
                else if (record.k === 'e') events.push(record);
                else if (record.k === 'header') header = record;
                else if (record.k === 'end') end = record;
            }
            start = i + 1;
        }
        if (!header || header.format !== FORMAT) throw new Error('missing neurofly-run-recording header');
        if (header.version !== VERSION) throw new Error(`unsupported recording version ${header.version}`);
        if (!end) throw new Error('recording is incomplete (no end record)');
        if (end.frames !== frames.length) throw new Error(`end record says ${end.frames} frames, file has ${frames.length}`);
        if (!frames.length) throw new Error('recording has no frames');
        const digest = await sha256Hex(concat(frameBytes, frameTotal));
        if (digest !== null && digest !== end.frames_sha256) throw new Error('frame digest mismatch: file is corrupt');
        return {header, frames, events, end, verified: digest !== null};
    }

    // ------------------------------------------------------------------ activity panel
    const ActivityPanel = {
        names: null,
        raster: [],            // [{step, pairs}] newest last
        rasterInfo: null,
        lastStep: null,

        reset() {
            this.raster = [];
            this.lastStep = null;
            this.draw();
        },

        update(pkt) {
            const activity = pkt.activity;
            const grouping = $('activityGrouping');
            const box = $('activityRegions');
            if (!box) return;
            if (!activity || !Array.isArray(activity.names)) {
                if (this.names !== null) { box.innerHTML = '<i>Not in this stream.</i>'; this.names = null; }
                if (grouping) grouping.textContent = '--';
            } else {
                const key = activity.names.join('|');
                if (key !== this.names) {
                    this.names = key;
                    box.innerHTML = activity.names.map((name, i) => {
                        const size = Array.isArray(activity.sizes) ? ` (${activity.sizes[i]} neurons)` : '';
                        return `<span class="name" title="${name}${size}">${name}</span><div class="bar"><div id="actBar${i}"></div></div><span class="val" id="actVal${i}">--</span>`;
                    }).join('');
                }
                if (grouping) grouping.textContent = `${activity.grouping} · ${activity.units}`;
                const rates = Array.isArray(activity.rates) ? activity.rates : null;
                const finite = rates ? rates.filter(Number.isFinite).map(Math.abs) : [];
                const scale = Math.max(1e-9, ...finite, activity.units === 'Hz' ? 1 : 1e-3);
                activity.names.forEach((_, i) => {
                    const v = rates ? rates[i] : null;
                    const bar = $(`actBar${i}`), val = $(`actVal${i}`);
                    if (bar) bar.style.width = Number.isFinite(v) ? `${Math.min(100, 100 * Math.abs(v) / scale)}%` : '0%';
                    if (val) val.textContent = Number.isFinite(v) ? (Math.abs(v) >= 100 ? v.toFixed(0) : v.toFixed(2)) : '--';
                });
            }
            this.pushRaster(pkt);
        },

        pushRaster(pkt) {
            const canvas = $('rasterCanvas'), note = $('rasterNote');
            if (!Array.isArray(pkt.spikes) || !pkt.raster) {
                if (canvas) canvas.hidden = true;
                if (note) note.textContent = pkt.raster === null
                    ? 'Spike raster: not recorded for this run (modular controller or raster off).'
                    : 'Spike raster: recorded graph runs only.';
                return;
            }
            this.rasterInfo = pkt.raster;
            this.addRasterColumn(pkt.step, pkt.spikes);
            if (canvas) canvas.hidden = false;
            if (note) {
                const spikes = this.raster.reduce((n, f) => { for (let j = 1; j < f.pairs.length; j += 2) n += f.pairs[j]; return n; }, 0);
                note.textContent = `Spike raster: ${pkt.raster.n} neurons (${pkt.raster.mode}), last ${this.raster.length} frames, ${spikes} spikes shown.`;
            }
            this.draw();
        },

        /** One raster column; the player also feeds the frames it skips at >1x or on a short seek. */
        addRasterColumn(step, pairs) {
            if (!Array.isArray(pairs)) return;
            if (this.lastStep !== null && step <= this.lastStep) this.raster = [];
            this.lastStep = step;
            this.raster.push({step, pairs});
            if (this.raster.length > RASTER_FRAMES) this.raster.splice(0, this.raster.length - RASTER_FRAMES);
        },

        draw() {
            const canvas = $('rasterCanvas');
            if (!canvas || canvas.hidden) return;
            const w = canvas.clientWidth || 300, h = canvas.clientHeight || 70;
            const dpr = window.devicePixelRatio || 1;
            if (canvas.width !== Math.round(w * dpr)) { canvas.width = Math.round(w * dpr); canvas.height = Math.round(h * dpr); }
            const ctx = canvas.getContext('2d');
            ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
            ctx.fillStyle = '#020617';
            ctx.fillRect(0, 0, w, h);
            const n = this.rasterInfo?.n || 1;
            const colW = w / RASTER_FRAMES;
            const rowH = Math.max(h / n, 0.5);
            ctx.fillStyle = '#e9d5ff';
            const offset = RASTER_FRAMES - this.raster.length;
            this.raster.forEach((frame, c) => {
                const x = (offset + c) * colW;
                for (let j = 0; j < frame.pairs.length; j += 2) {
                    ctx.globalAlpha = Math.min(1, 0.45 + 0.2 * frame.pairs[j + 1]);
                    ctx.fillRect(x, frame.pairs[j] * (h / n), Math.max(1, colW), Math.max(1, rowH));
                }
            });
            ctx.globalAlpha = 1;
        },
    };
    window.neuroflyActivityPanel = ActivityPanel;

    // ------------------------------------------------------------------ player
    class RecordingPlayer {
        constructor() {
            this.rec = null;
            this.playing = false;
            this.speed = 1;
            this.playhead = 0;
            this.applied = -1;
            this.lastWall = null;
            this.raf = null;
        }

        get bridge() { return window.neuroflyDiagnostics?.bridge || window.app?.hud?.daemonBridge || null; }

        async loadBytes(bytes, name) {
            const bridge = this.bridge;
            if (!bridge) throw new Error('dashboard is not initialised yet');
            const rec = await parseRecording(bytes);
            this.stop();
            this.rec = rec;
            this.name = name;
            this.times = rec.frames.map((f) => f.sim_time_s);
            this.buildIdentity();
            const prov = rec.header.provenance || {};
            bridge.enterReplay(`${prov.assay || '?'} · ${name}`);
            bridge.manifest = {replay: true, recording: name, header: rec.header};
            bridge.manifestRunId = this.identity.run_id;
            ActivityPanel.reset();
            this.playhead = this.times[0];
            this.applied = -1;
            this.showBar();
            this.apply(0);
            this.play();
            return rec;
        }

        buildIdentity() {
            const h = this.rec.header, prov = h.provenance || {}, graph = prov.graph || {};
            const digest = (this.rec.end.frames_sha256 || '').slice(0, 12);
            this.identity = {
                run_id: `recording-${digest}`, instance_id: 'recording', assay: prov.assay, backend: prov.backend,
                controller_version: prov.controller_version, graph_sha256: graph.graph_sha256 || null,
                neuron_map_sha256: graph.neuron_map_sha256 || null, io_map_sha256: graph.io_map_sha256 || null,
                synthetic: !!prov.synthetic, test_mode: !!prov.test_mode,
                label: `RECORDING ${this.name}${prov.label ? ' · ' + prov.label : ''}`,
                activation: 0, daemon_run_id: `recording-${digest}`,
            };
            const regions = h.channels?.regions || {};
            this.activityMeta = {grouping: regions.grouping, names: regions.names, sizes: regions.sizes || null,
                                 units: regions.grouping === 'modular-circuit' ? 'model units' : 'Hz'};
            const raster = h.channels?.raster;
            this.rasterMeta = raster ? {n: raster.n, mode: raster.mode, labels: raster.labels} : null;
        }

        packetFor(index) {
            const f = this.rec.frames[index];
            const path = [];
            for (let j = index; j >= 0 && index - j < PATH_STEPS; j--) {
                const g = this.rec.frames[j];
                if (g.segment_id !== f.segment_id) break;
                path.push([g.step, g.fly.x, g.fly.y]);
            }
            path.reverse();
            const dt = this.rec.header.provenance?.params?.dt_s ?? 0.02;
            const stepsInFrame = index > 0 ? f.step - this.rec.frames[index - 1].step : 0;
            const pkt = Object.assign({}, f, {
                type: 'telemetry', run_id: this.identity.run_id, brain_id: 'recording', identity: this.identity,
                timestamp: Date.now() / 1000, sim_speed: this.speed, path,
                timing: {requested_speed: this.speed, achieved_speed: this.speed, integration_dt_s: dt,
                         sim_time_s: f.sim_time_s, step: f.step, snapshot_seq: index, steps_in_frame: stepsInFrame,
                         overloaded: false, replay: true},
                activity: Object.assign({}, this.activityMeta, {rates: f.activity}),
                raster: this.rasterMeta,
            });
            delete pkt.k; delete pkt.i;
            return pkt;
        }

        apply(index) {
            const bridge = this.bridge;
            if (!bridge || !this.rec) return;
            if (index < this.applied) { bridge.resetReplayView(); ActivityPanel.reset(); }
            else if (this.rasterMeta && this.applied >= 0) {
                // Frames skipped between two drawn frames still belong in the raster.
                for (let j = Math.max(this.applied + 1, index - RASTER_FRAMES); j < index; j++) {
                    ActivityPanel.addRasterColumn(this.rec.frames[j].step, this.rec.frames[j].spikes);
                }
            }
            this.applied = index;
            const pkt = this.packetFor(index);
            try {
                bridge.handleDaemonPacket(pkt);
                bridge.lastValidDataTime = performance.now();
            } catch (e) {
                report('packet-apply', e, {run_id: pkt.run_id, step: pkt.step, assay: pkt.paradigm});
            }
        }

        indexAt(t) {
            let lo = 0, hi = this.times.length - 1;
            if (t <= this.times[0]) return 0;
            while (lo < hi) {
                const mid = (lo + hi + 1) >> 1;
                if (this.times[mid] <= t + 1e-9) lo = mid; else hi = mid - 1;
            }
            return lo;
        }

        loop(now) {
            this.raf = requestAnimationFrame((t) => this.loop(t));
            if (!this.rec) return;
            if (this.playing) {
                if (this.lastWall !== null) this.playhead += Math.min(0.25, (now - this.lastWall) / 1000) * this.speed;
                const last = this.times[this.times.length - 1];
                if (this.playhead >= last) { this.playhead = last; this.pause(); }
            }
            this.lastWall = now;
            const index = this.indexAt(this.playhead);
            if (index !== this.applied) this.apply(index);
            this.updateBar();
        }

        play() {
            if (!this.rec) return;
            if (this.playhead >= this.times[this.times.length - 1]) { this.playhead = this.times[0]; }
            this.playing = true;
            this.lastWall = null;
            if (this.raf === null) this.raf = requestAnimationFrame((t) => this.loop(t));
            this.updateButtons();
        }

        pause() { this.playing = false; this.updateButtons(); }
        toggle() { if (this.playing) this.pause(); else this.play(); }

        setSpeed(speed) {
            const v = Number(speed);
            if (!Number.isFinite(v) || v <= 0) return;
            this.speed = v;
            const sel = $('replaySpeed');
            if (sel && [...sel.options].some((o) => Number(o.value) === v)) sel.value = String(v);
            if (this.applied >= 0) this.apply(this.applied);
        }

        seekFraction(fraction) {
            if (!this.rec) return;
            const t0 = this.times[0], t1 = this.times[this.times.length - 1];
            this.playhead = t0 + Math.max(0, Math.min(1, fraction)) * (t1 - t0);
            const index = this.indexAt(this.playhead);
            if (index !== this.applied) this.apply(index);
            this.updateBar();
        }

        stop() {
            if (this.raf !== null) cancelAnimationFrame(this.raf);
            this.raf = null;
            this.playing = false;
        }

        exit() {
            this.stop();
            this.rec = null;
            $('replayBar').hidden = true;
            ActivityPanel.reset();
            this.bridge?.exitReplay();
            const pause = $('btnPauseToggle');
            if (pause) pause.textContent = 'Pause';
        }

        showBar() {
            const h = this.rec.header, prov = h.provenance || {};
            const graph = prov.graph?.graph_sha256 ? ` · graph ${prov.graph.graph_sha256.slice(0, 12)}` : '';
            const code = prov.code?.commit ? ` · code ${prov.code.commit.slice(0, 8)}${prov.code.dirty ? '+dirty' : ''}` : '';
            const info = `${prov.assay} · ${prov.backend}${prov.synthetic ? ' (SYNTHETIC TEST GRAPH)' : ''} · seed ${prov.seed}`
                + `${graph}${code} · ${this.rec.frames.length} frames · ${this.rec.events.length} inputs`
                + (this.rec.verified ? ' · digest OK' : ' · digest not checked (insecure origin)');
            const infoEl = $('replayInfo');
            infoEl.textContent = info;
            infoEl.title = JSON.stringify(prov, null, 1);
            $('replayBar').hidden = false;
            $('replayChooser').hidden = true;
            this.updateBar();
        }

        updateButtons() {
            const label = this.playing ? 'Pause' : 'Play';
            const btn = $('btnReplayPlay');
            if (btn) btn.textContent = label;
            const pause = $('btnPauseToggle');
            if (pause) pause.textContent = this.playing ? 'Pause' : 'Resume';
        }

        updateBar() {
            if (!this.rec) return;
            const t0 = this.times[0], t1 = this.times[this.times.length - 1];
            const seek = $('replaySeek');
            if (seek && document.activeElement !== seek) seek.value = String(Math.round(1000 * (t1 > t0 ? (this.playhead - t0) / (t1 - t0) : 0)));
            const time = $('replayTime');
            if (time) time.textContent = `${(this.playhead - t0).toFixed(2)} / ${(t1 - t0).toFixed(2)} s`;
        }
    }

    const player = new RecordingPlayer();
    window.neuroflyReplay = player;

    // ------------------------------------------------------------------ chooser
    async function loadFromUrl(url, name) {
        const res = await fetch(url);
        if (!res.ok) throw new Error(`HTTP ${res.status} for ${url}`);
        const bytes = new Uint8Array(await res.arrayBuffer());
        return player.loadBytes(bytes, name || decodeURIComponent(url.split('/').pop() || 'recording'));
    }

    function showLoadError(e, what) {
        const label = $('arenaRunState');
        if (label) label.textContent = `Replay not loaded (${what}): ${e.message}`;
        report('startup', e, {});
    }

    async function refreshList() {
        const list = $('replayList');
        const base = player.bridge?.activeUrl;
        if (!list) return;
        if (!base) { list.innerHTML = '<i>No daemon connected. Open a file instead.</i>'; return; }
        list.innerHTML = '<i>Loading…</i>';
        try {
            const res = await fetch(`${base}/api/recordings`, {signal: AbortSignal.timeout(4000)});
            const data = await res.json();
            const items = data.recordings || [];
            list.innerHTML = '';
            if (!items.length) { list.innerHTML = '<i>No recordings on this daemon yet.</i>'; return; }
            for (const r of items) {
                const btn = document.createElement('button');
                const mb = (r.bytes / 1e6).toFixed(r.bytes > 1e6 ? 1 : 2);
                btn.textContent = `${r.name} · ${r.assay || '?'} · ${r.backend || '?'}${r.synthetic ? ' (synthetic)' : ''} · ${r.duration_s ?? '?'} s · ${mb} MB`;
                btn.addEventListener('click', () => loadFromUrl(`${base}/api/recordings/${encodeURIComponent(r.name)}`, r.name)
                    .catch((e) => showLoadError(e, r.name)));
                list.appendChild(btn);
            }
        } catch (e) {
            list.innerHTML = `<i>Could not list recordings: ${e.message}</i>`;
        }
    }

    function wire() {
        $('btnReplay')?.addEventListener('click', () => {
            const chooser = $('replayChooser');
            chooser.hidden = !chooser.hidden;
            if (!chooser.hidden) refreshList();
        });
        $('btnReplayChooserClose')?.addEventListener('click', () => { $('replayChooser').hidden = true; });
        $('replayFileInput')?.addEventListener('change', async (event) => {
            const file = event.target.files?.[0];
            if (!file) return;
            try { await player.loadBytes(new Uint8Array(await file.arrayBuffer()), file.name); }
            catch (e) { showLoadError(e, file.name); }
            event.target.value = '';
        });
        $('btnReplayPlay')?.addEventListener('click', () => player.toggle());
        $('btnReplayExit')?.addEventListener('click', () => player.exit());
        $('replaySpeed')?.addEventListener('change', (e) => player.setSpeed(e.target.value));
        $('replaySeek')?.addEventListener('input', (e) => player.seekFraction(Number(e.target.value) / 1000));

        // ?replay=<url> opens a recording directly (e.g. a published open-data file).
        const param = new URLSearchParams(window.location.search || '').get('replay');
        if (param) {
            const started = performance.now();
            const tryLoad = () => {
                if (!player.bridge) {
                    if (performance.now() - started < 10000) setTimeout(tryLoad, 100);
                    return;
                }
                loadFromUrl(param).catch((e) => showLoadError(e, param));
            };
            tryLoad();
        }
    }

    if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', wire);
    else wire();

    window.neuroflyRecording = {parseRecording, loadFromUrl};
})();
