/*
 * 1x replay of an embodied FlyGym run (body.nfbody, neurofly_body/body_recording.py).
 *
 * The file is a gzip stream of JSON lines: a header (skeleton, provenance), one
 * frame per 1/fps s of simulated time and an end record carrying the frame count
 * and the SHA-256 of the frame lines, which is checked here when the browser
 * allows it.  Playback follows simulated time: 1x is the fly's real time,
 * however long the run took to compute.
 */
(function () {
    'use strict';

    const FORMAT = 'neurofly-embodied-recording';
    const $ = (id) => document.getElementById(id);
    const state = { rec: null, frame: 0, playing: false, clock: 0, last: null };

    function setStatus(text, bad) {
        const el = $('status');
        el.textContent = text;
        el.classList.toggle('bad', !!bad);
    }

    async function inflate(bytes) {
        if (bytes.length < 2 || bytes[0] !== 0x1f || bytes[1] !== 0x8b) {
            throw new Error('not a gzip-compressed .nfbody recording');
        }
        const stream = new Blob([bytes]).stream().pipeThrough(new DecompressionStream('gzip'));
        return new Uint8Array(await new Response(stream).arrayBuffer());
    }

    async function sha256Hex(bytes) {
        if (!window.crypto || !window.crypto.subtle) return null;
        const digest = await window.crypto.subtle.digest('SHA-256', bytes);
        return Array.from(new Uint8Array(digest)).map((b) => b.toString(16).padStart(2, '0')).join('');
    }

    async function parse(compressed) {
        const raw = await inflate(compressed);
        const text = new TextDecoder();
        let header = null, end = null, start = 0;
        const frames = [], frameRanges = [];
        for (let i = 0; i < raw.length; i++) {
            if (raw[i] !== 10) continue;
            const record = JSON.parse(text.decode(raw.subarray(start, i)));
            if (record.k === 'header') header = record;
            else if (record.k === 'f') { frames.push(record); frameRanges.push([start, i + 1]); }
            else if (record.k === 'end') end = record;
            start = i + 1;
        }
        if (!header || header.format !== FORMAT) throw new Error('not a ' + FORMAT + ' file');
        if (!end) throw new Error('recording is incomplete (no end record)');
        if (end.frames !== frames.length) throw new Error('frame count does not match the end record');
        let total = 0;
        for (const [a, b] of frameRanges) total += b - a;
        const joined = new Uint8Array(total);
        let offset = 0;
        for (const [a, b] of frameRanges) { joined.set(raw.subarray(a, b), offset); offset += b - a; }
        const digest = await sha256Hex(joined);
        if (digest !== null && digest !== end.frames_sha256) throw new Error('frame SHA-256 does not match the end record');
        return { header, frames, end, verified: digest !== null };
    }

    // ------------------------------------------------------------------ scene
    const view = $('view');
    const renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    view.appendChild(renderer.domElement);
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x060913);
    const camera = new THREE.PerspectiveCamera(40, 1, 0.05, 500);
    camera.up.set(0, 0, 1);                     // MuJoCo world frame: z up
    camera.position.set(-6, -6, 4);
    const controls = new THREE.OrbitControls(camera, renderer.domElement);
    controls.target.set(0, 0, 0.5);
    const grid = new THREE.GridHelper(200, 200, 0x3a4d75, 0x223052);
    grid.rotation.x = Math.PI / 2;              // GridHelper lies in XZ; the ground is XY
    scene.add(grid);
    scene.add(new THREE.AxesHelper(2));

    let bones = null, joints = null, trail = null, trailCount = 0;

    function buildBody(skeleton) {
        [bones, joints, trail].forEach((obj) => obj && scene.remove(obj));
        const n = skeleton.segments.length;
        const pairs = skeleton.parents.filter((p) => p >= 0).length;
        const boneGeom = new THREE.BufferGeometry();
        boneGeom.setAttribute('position', new THREE.BufferAttribute(new Float32Array(pairs * 6), 3));
        bones = new THREE.LineSegments(boneGeom, new THREE.LineBasicMaterial({ color: 0xd7e0f2 }));
        const jointGeom = new THREE.BufferGeometry();
        jointGeom.setAttribute('position', new THREE.BufferAttribute(new Float32Array(n * 3), 3));
        joints = new THREE.Points(jointGeom, new THREE.PointsMaterial({ color: 0x5fb3ff, size: 0.08 }));
        const trailGeom = new THREE.BufferGeometry();
        trailGeom.setAttribute('position', new THREE.BufferAttribute(new Float32Array(state.rec.frames.length * 3), 3));
        trail = new THREE.Line(trailGeom, new THREE.LineBasicMaterial({ color: 0xffb454 }));
        [bones, joints, trail].forEach((obj) => { obj.frustumCulled = false; scene.add(obj); });
        trailCount = 0;
    }

    function resize() {
        const w = view.clientWidth || 1, h = view.clientHeight || 1;
        renderer.setSize(w, h, false);
        camera.aspect = w / h;
        camera.updateProjectionMatrix();
    }
    window.addEventListener('resize', resize);

    // ------------------------------------------------------------------ frames
    function fmt(value, digits) { return Number.isFinite(value) ? value.toFixed(digits) : '–'; }

    function showFrame(index) {
        const rec = state.rec;
        if (!rec) return;
        index = Math.max(0, Math.min(rec.frames.length - 1, index));
        state.frame = index;
        const f = rec.frames[index], sk = rec.header.skeleton;
        const jp = joints.geometry.attributes.position.array;
        jp.set(f.pos);
        joints.geometry.attributes.position.needsUpdate = true;
        const bp = bones.geometry.attributes.position.array;
        let k = 0;
        sk.parents.forEach((p, i) => {
            if (p < 0) return;
            bp[k++] = f.pos[3 * i]; bp[k++] = f.pos[3 * i + 1]; bp[k++] = f.pos[3 * i + 2];
            bp[k++] = f.pos[3 * p]; bp[k++] = f.pos[3 * p + 1]; bp[k++] = f.pos[3 * p + 2];
        });
        bones.geometry.attributes.position.needsUpdate = true;
        const tp = trail.geometry.attributes.position.array;
        for (let i = 0; i <= index; i++) {
            const g = rec.frames[i].pos;
            tp[3 * i] = g[0]; tp[3 * i + 1] = g[1]; tp[3 * i + 2] = 0.01;
        }
        trailCount = index + 1;
        trail.geometry.setDrawRange(0, trailCount);
        trail.geometry.attributes.position.needsUpdate = true;
        if ($('follow').checked) {
            const dx = f.pos[0] - controls.target.x, dy = f.pos[1] - controls.target.y;
            controls.target.x += dx; controls.target.y += dy;
            camera.position.x += dx; camera.position.y += dy;
        }

        $('yaw').textContent = fmt(f.yaw, 3) + ' rad';
        $('xy').textContent = fmt(f.pos[0], 2) + ', ' + fmt(f.pos[1], 2) + ' mm';
        $('contacts').textContent = (f.contacts || []).map((c) => (c > 0 ? '●' : '○')).join(' ') || '–';
        const [l, r] = f.cmd;
        $('cmdL').textContent = fmt(l, 3); $('cmdR').textContent = fmt(r, 3);
        for (const [id, v] of [['barL', l], ['barR', r]]) {
            const pct = Math.min(50, Math.abs(v) / 1.2 * 50);
            const bar = $(id);
            bar.style.width = pct + '%';
            bar.style.left = v >= 0 ? '50%' : (50 - pct) + '%';
        }
        const dn = $('dn');
        dn.innerHTML = '';
        const entries = Object.entries(f.dn || {});
        if (!entries.length) dn.innerHTML = '<dt>not recorded</dt><dd></dd>';
        for (const [name, rate] of entries) {
            const dt = document.createElement('dt'); dt.textContent = name;
            const dd = document.createElement('dd'); dd.textContent = fmt(rate, 1);
            dn.append(dt, dd);
        }
        $('spikes').textContent = String(f.spikes);
        const events = [];
        for (let i = 0; i <= index; i++) (rec.frames[i].events || []).forEach((e) => events.push([rec.frames[i].t, e]));
        $('events').textContent = events.length
            ? events.slice(-5).map(([t, e]) => t.toFixed(3) + ' s ' + e.event + (e.body_action ? ' (' + e.body_action + ')' : '')).join('\n')
            : 'None yet';
        $('seek').value = String(index);
        $('clock').textContent = fmt(f.t, 3) + ' / ' + fmt(rec.frames[rec.frames.length - 1].t, 3) + ' s';
    }

    function tick(now) {
        if (state.playing && state.rec) {
            if (state.last !== null) state.clock += (now - state.last) / 1000 * Number($('speed').value);
            state.last = now;
            const fps = state.rec.header.fps;
            const first = state.rec.frames[0].t;
            const index = Math.floor((state.clock - first) * fps + 1e-9);
            if (index >= state.rec.frames.length - 1) {
                showFrame(state.rec.frames.length - 1);
                setPlaying(false);
            } else if (index !== state.frame) {
                showFrame(index);
            }
        }
        controls.update();
        renderer.render(scene, camera);
        requestAnimationFrame(tick);
    }

    function setPlaying(on) {
        state.playing = on && !!state.rec;
        state.last = null;
        if (state.playing) {
            if (state.frame >= state.rec.frames.length - 1) state.frame = 0;
            state.clock = state.rec.frames[state.frame].t;
        }
        $('play').textContent = state.playing ? 'Pause' : 'Play';
    }

    function describeRun(header) {
        const p = header.provenance || {}, cfg = p.config || {}, nb = p.neural_backend || {};
        const rows = [
            ['Controller', nb.controller_kind || (nb.graph_sha256 ? 'connectome' : '–')],
            ['Decoder', (p.decoder && p.decoder.name) || '–'],
            ['Mode', p.command_mode || cfg.mode || '–'],
            ['Seed', cfg.seed],
            ['Duration', cfg.duration_s + ' s'],
            ['Drum', cfg.world_angular_velocity_rad_s + ' rad/s'],
            ['Graph', nb.graph_sha256 ? nb.graph_sha256.slice(0, 12) : 'none'],
            ['Brain', nb.brain_backend || '–'],
            ['Frames', header.fps + ' fps'],
        ];
        const run = $('run');
        run.innerHTML = '';
        for (const [k, v] of rows) {
            const dt = document.createElement('dt'); dt.textContent = k;
            const dd = document.createElement('dd'); dd.textContent = v === undefined ? '–' : String(v);
            run.append(dt, dd);
        }
    }

    async function load(bytes, label) {
        setPlaying(false);
        setStatus('Loading…');
        try {
            const rec = await parse(bytes);
            if (!rec.frames.length) throw new Error('recording has no frames');
            state.rec = rec;
            buildBody(rec.header.skeleton);
            describeRun(rec.header);
            $('seek').max = String(rec.frames.length - 1);
            $('seek').disabled = false;
            $('play').disabled = false;
            $('drop').classList.add('hidden');
            $('source').textContent = label;
            const f0 = rec.frames[0].pos;
            controls.target.set(f0[0], f0[1], f0[2]);
            camera.position.set(f0[0] - 4, f0[1] - 4, f0[2] + 3);
            showFrame(0);
            setStatus(rec.frames.length + ' frames' + (rec.verified ? ', SHA-256 verified' : ', not verified (insecure origin)'));
            window.embodiedReplay = { frames: rec.frames.length, verified: rec.verified, get frame() { return state.frame; } };
        } catch (err) {
            setStatus('Could not open: ' + err.message, true);
            console.error(err);
        }
    }

    $('file').addEventListener('change', async (e) => {
        const file = e.target.files[0];
        if (file) await load(new Uint8Array(await file.arrayBuffer()), file.name);
    });
    view.addEventListener('dragover', (e) => e.preventDefault());
    view.addEventListener('drop', async (e) => {
        e.preventDefault();
        const file = e.dataTransfer.files[0];
        if (file) await load(new Uint8Array(await file.arrayBuffer()), file.name);
    });
    $('play').addEventListener('click', () => setPlaying(!state.playing));
    $('seek').addEventListener('input', (e) => { showFrame(Number(e.target.value)); if (state.playing) setPlaying(true); });
    $('speed').addEventListener('change', () => { if (state.playing) setPlaying(true); });
    window.addEventListener('keydown', (e) => {
        if (e.code === 'Space' && state.rec && e.target === document.body) { e.preventDefault(); setPlaying(!state.playing); }
    });

    resize();
    requestAnimationFrame(tick);
    const src = new URLSearchParams(location.search).get('src');
    if (src) {
        fetch(src).then((r) => {
            if (!r.ok) throw new Error('HTTP ' + r.status);
            return r.arrayBuffer();
        }).then((buf) => load(new Uint8Array(buf), src)).catch((err) => setStatus('Could not fetch ' + src + ': ' + err.message, true));
    }
})();
