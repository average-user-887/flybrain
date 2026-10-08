/*
 * Opt-in asset gallery (web/asset_gallery.html).  Presentation only.
 *
 * A standalone review page: nothing in the dashboard or the replay page loads this
 * file, so default UI behaviour is unchanged.  It reads a manifest
 * (asset_manifest.json, or ?manifest=<url>), loads each listed GLB with the vendored
 * three.js r128 GLTFLoader and shows it at close range with an LOD toggle, a floor
 * and bounds overlay, axes and a scale ruler, per-layer visibility and a contact
 * sheet.  Adding an asset means adding a manifest entry; no code change.
 *
 * No simulation runs here.  Every number on the page is measured from the loaded
 * geometry (vertex scans), never assumed.
 */
(function (root, factory) {
    'use strict';
    const api = factory();
    if (typeof module === 'object' && module.exports) module.exports = api;
    if (root) root.NeuroflyAssetGallery = api;
}(typeof window !== 'undefined' ? window : null, function () {
    'use strict';

    const MANIFEST_SCHEMA = 'neurofly-asset-manifest/1';
    const DEFAULT_MANIFEST_URL = 'asset_manifest.json';
    const STAND_MODES = ['tarsus', 'bbox', 'none'];
    const OVERLAYS = ['floor', 'bounds', 'axes', 'grid', 'ruler', 'labels'];
    const LEG_PREFIXES = ['lf', 'lm', 'lh', 'rf', 'rm', 'rh'];
    // Tolerance for "the lowest foot touches the floor": far below one LOD1 facet.
    const FLOOR_TOLERANCE = 1e-3;

    // ------------------------------------------------------------------ manifest
    function validateManifest(m) {
        const errors = [];
        const assets = [];
        if (!m || typeof m !== 'object') return {errors: ['manifest is not a JSON object'], assets, manifest: null};
        if (m.schema !== MANIFEST_SCHEMA) errors.push('schema is ' + JSON.stringify(m.schema) + ', expected ' + MANIFEST_SCHEMA);
        const layerSets = m.layer_sets && typeof m.layer_sets === 'object' ? m.layer_sets : {};
        const floorY = Number.isFinite(m.floor_y) ? m.floor_y : -0.02;
        const seen = new Set();
        for (const [i, a] of (Array.isArray(m.assets) ? m.assets : []).entries()) {
            const where = 'assets[' + i + ']';
            if (!a || typeof a.id !== 'string' || !a.id) { errors.push(where + ': missing id'); continue; }
            if (seen.has(a.id)) { errors.push(where + ': duplicate id ' + a.id); continue; }
            const lods = (Array.isArray(a.lods) ? a.lods : []).filter((l) => l && typeof l.url === 'string' && l.url);
            if (!lods.length) { errors.push(a.id + ': no LOD with a url'); continue; }
            if (lods.some((l) => /^[a-z]+:\/\//i.test(l.url) || l.url.startsWith('/'))) {
                errors.push(a.id + ': LOD urls must be relative to the page (no remote assets)'); continue;
            }
            const stand = STAND_MODES.includes(a.stand) ? a.stand : 'bbox';
            const layers = typeof a.layers === 'string' ? (layerSets[a.layers] || null)
                : (Array.isArray(a.layers) ? a.layers : null);
            if (typeof a.layers === 'string' && !layerSets[a.layers]) errors.push(a.id + ': unknown layer set ' + a.layers);
            seen.add(a.id);
            assets.push({
                id: a.id, title: String(a.title || a.id), category: String(a.category || 'other'),
                status: String(a.status || 'illustrative'), stand, node: a.node || null,
                validate: a.validate || null, notes: String(a.notes || ''),
                transform: a.transform || null, layers: compileLayers(layers || []),
                clips: (Array.isArray(a.clips) ? a.clips : []).filter((c) => c && typeof c.url === 'string' && !/^[a-z]+:\/\//i.test(c.url) && !c.url.startsWith('/'))
                    .map((c) => ({url: c.url, label: String(c.label || c.url), name: c.name || null, notes: String(c.notes || '')})),
                joints: typeof a.joints === 'string' && !/^[a-z]+:\/\//i.test(a.joints) && !a.joints.startsWith('/') ? a.joints : null,
                displayScale: a.display_scale === 'extras' || Number.isFinite(a.display_scale) ? a.display_scale : null,
                appearanceOnly: a.appearance_only === true,
                lods: lods.map((l, k) => ({level: Number.isFinite(l.level) ? l.level : k, label: String(l.label || ('LOD' + k)),
                    url: l.url, budget: Number.isFinite(l.budget_triangles) ? l.budget_triangles : null})),
                sprites: (Array.isArray(a.sprites) ? a.sprites : []).filter((s) => s && typeof s.url === 'string')
                    .map((s) => ({url: s.url, label: String(s.label || s.url)})),
                source: a.source || null
            });
        }
        if (!assets.length) errors.push('manifest lists no usable asset');
        const sheets = (Array.isArray(m.sheets) ? m.sheets : []).filter((x) => x && typeof x.url === 'string' && !/^[a-z]+:\/\//i.test(x.url))
            .map((x) => ({url: x.url, label: String(x.label || x.url)}));
        return {errors, assets, floorY, units: m.units || null, sheets, manifest: m};
    }

    function compileLayers(list) {
        return list.filter((l) => l && l.id).map((l) => ({
            id: String(l.id), label: String(l.label || l.id),
            match: (Array.isArray(l.match) ? l.match : []).map((p) => new RegExp(p))
        }));
    }

    // Layer of a named node, or null.  A mesh takes the layer of its nearest named
    // ancestor that matches (glTF multi-primitive meshes load as a group of meshes).
    function classifyName(name, layers) {
        if (!name) return null;
        for (const l of layers) if (l.match.some((re) => re.test(name))) return l.id;
        return null;
    }
    function layerOf(obj, layers, stopAt) {
        for (let o = obj; o && o !== stopAt; o = o.parent) {
            const id = classifyName(o.name, layers);
            if (id) return id;
        }
        return 'other';
    }

    // ------------------------------------------------------------------ geometry
    // Exact minimum / box over mesh vertices in the frame of `root`'s parent
    // (three r128's Box3.setFromObject uses transformed bounding boxes, which is loose).
    function vertexBounds(T, root, filter) {
        root.updateMatrixWorld(true);
        const box = new T.Box3();
        const v = new T.Vector3();
        root.traverse((m) => {
            const a = m.isMesh && m.geometry && m.geometry.attributes && m.geometry.attributes.position;
            if (!a || (filter && !filter(m))) return;
            for (let k = 0; k < a.count; k += 1) box.expandByPoint(v.fromBufferAttribute(a, k).applyMatrix4(m.matrixWorld));
        });
        return box.isEmpty() ? null : box;
    }

    function isLegMesh(m) {
        for (let o = m; o; o = o.parent) {
            if (/^(lf|lm|lh|rf|rm|rh)_(coxa|trochanterfemur|tibia|tarsus)$/.test(o.name || '')) return true;
        }
        return false;
    }

    function triangles(root) {
        let n = 0;
        root.traverse((o) => {
            if (!o.isMesh || !o.geometry) return;
            const g = o.geometry;
            n += (g.index ? g.index.count : (g.attributes.position ? g.attributes.position.count : 0)) / 3;
        });
        return Math.round(n);
    }

    // Classify the placed model against the floor.  gap = lowest point - floor.
    function floorStatus(lowestY, floorY, tol) {
        if (!Number.isFinite(lowestY)) return {gap: null, state: 'unknown', text: 'no geometry'};
        const t = Number.isFinite(tol) ? tol : FLOOR_TOLERANCE;
        const gap = lowestY - floorY;
        if (gap < -t) return {gap, state: 'penetrates', text: 'below floor by ' + (-gap).toFixed(4)};
        if (gap > t) return {gap, state: 'floating', text: 'above floor by ' + gap.toFixed(4)};
        return {gap, state: 'contact', text: 'on floor (|gap| ≤ ' + t + ')'};
    }

    // glTF root extras (nf_*), as the r128 GLTFLoader puts them on node userData.
    function rootExtras(model) {
        let found = null;
        if (model && typeof model.traverse === 'function') model.traverse((o) => {
            if (!found && o.userData && Object.keys(o.userData).some((k) => /^(nf|neurofly)_/.test(k))) found = o.userData;
        });
        return found || {};
    }
    // Uniform display scale for the whole root: manifest number, or nf_display_scale
    // from the GLB extras ('extras'), else 1.  Never per-part.
    function displayScaleOf(mode, extras) {
        if (Number.isFinite(mode) && mode > 0) return {scale: mode, source: 'manifest'};
        const v = extras && extras.nf_display_scale;
        if (mode === 'extras' && Number.isFinite(v) && v > 0 && v <= 10) return {scale: v, source: 'GLB extras nf_display_scale'};
        if (mode === 'extras') return {scale: 1, source: 'GLB has no nf_display_scale (treated as 1)'};
        return {scale: 1, source: 'none'};
    }

    // ------------------------------------------------------------------ animation
    // Illustrative clips only: never applied to recorded or live telemetry.
    const ANIMATION_LABEL = 'Illustrative animation, not simulated behaviour';
    const SPEEDS = [0.1, 0.25, 0.5, 1, 2];
    function trackNodes(clip) {
        const out = [];
        for (const t of (clip && clip.tracks) || []) {
            const name = String(t.name || '').split('.')[0];
            if (name && !out.includes(name)) out.push(name);
        }
        return out;
    }
    function clipSummary(animations) {
        return (animations || []).map((c) => ({name: c.name || '(unnamed)', duration: Number(c.duration) || 0,
            tracks: (c.tracks || []).length, nodes: trackNodes(c)}));
    }
    // Local rotation of a joint as Euler XYZ degrees, and the angle away from its bind pose.
    function jointReadout(T, obj, bindQuat) {
        const e = new T.Euler().setFromQuaternion(obj.quaternion, 'XYZ');
        const d = (r) => r * 180 / Math.PI;
        const delta = bindQuat ? d(2 * Math.acos(Math.min(1, Math.abs(obj.quaternion.dot(bindQuat))))) : null;
        return {x: d(e.x), y: d(e.y), z: d(e.z), delta};
    }

    // Signed angle (rad) of a joint rotation about its documented DOF axis (W2 rig v2:
    // bind = rest = 0; each DOF node rotates only about its own axis).
    function dofAngle(q, axis) {
        const n = Math.hypot(axis[0], axis[1], axis[2]) || 1;
        const s = (q.x * axis[0] + q.y * axis[1] + q.z * axis[2]) / n;
        let a = 2 * Math.atan2(s, q.w);
        if (a > Math.PI) a -= 2 * Math.PI;
        if (a < -Math.PI) a += 2 * Math.PI;
        return a;
    }
    const lerpTable = (xs, ys, x) => {
        if (x <= xs[0]) return ys[0];
        for (let i = 1; i < xs.length; i += 1) if (x <= xs[i]) return ys[i - 1] + (ys[i] - ys[i - 1]) * (x - xs[i - 1]) / (xs[i] - xs[i - 1]);
        return ys[ys.length - 1];
    };
    // W2 rig v2 joint contract: documented source ranges plus the measured display-safe
    // envelope.  A folded wing (sweep < 0.5) keeps pitch at 0; an open wing stays in the
    // intersection of the contract's measured pitch intervals.
    const WING_MIN_ELEVATE = [[0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 3.0], [0, 0, -0.1, -0.9, -0.8, -0.6, -0.3, 0, 0.2, 0.4, 0.4]];
    const OPEN_PITCH = [-0.07, 0.13];
    const ANTENNA_MAX_EXTEND = [[-0.4, 0.0, 0.1, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8], [0.4, 0.4, 0.35, 0.35, 0.3, 0.25, 0.2, 0.15, -0.05]];
    function envelopeViolations(angles, ranges, tol) {
        const t = Number.isFinite(tol) ? tol : 0.02;
        const out = [];
        for (const [node, a] of Object.entries(angles)) {
            const r = ranges && ranges[node];
            if (r && (a < r[0] - t || a > r[1] + t)) out.push(node + ' ' + a.toFixed(3) + ' outside source range [' + r[0] + ', ' + r[1] + ']');
        }
        for (const side of ['l', 'r']) {
            const sw = angles[side + '_wing_sweep'], el = angles[side + '_wing_elevate'], pi = angles[side + '_wing_pitch'];
            if (Number.isFinite(sw) && Number.isFinite(el)) {
                const lo = lerpTable(WING_MIN_ELEVATE[0], WING_MIN_ELEVATE[1], sw);
                if (el < lo - t) out.push(side + '_wing_elevate ' + el.toFixed(3) + ' below display-safe minimum ' + lo.toFixed(3) + ' at sweep ' + sw.toFixed(3));
                if (el > 0.8 + t) out.push(side + '_wing_elevate ' + el.toFixed(3) + ' above 0.8');
            }
            if (Number.isFinite(sw) && Number.isFinite(pi) && sw < 0.5 && Math.abs(pi) > t) out.push(side + '_wing_pitch ' + pi.toFixed(3) + ' while folded (sweep ' + sw.toFixed(3) + ' < 0.5)');
            // Open wing: the intersection of every measured safe pitch interval in the
            // contract, [-0.07, 0.13] (conservative; the sparse table is not interpolated).
            if (Number.isFinite(sw) && Number.isFinite(pi) && sw >= 0.5 && (pi < OPEN_PITCH[0] - t || pi > OPEN_PITCH[1] + t)) out.push(side + '_wing_pitch ' + pi.toFixed(3) + ' outside the open-wing safe pitch [' + OPEN_PITCH[0] + ', ' + OPEN_PITCH[1] + ']');
            const ab = angles[side + '_antenna_abduct'], ex = angles[side + '_antenna_extend'];
            if (Number.isFinite(ab) && Number.isFinite(ex)) {
                const hi = lerpTable(ANTENNA_MAX_EXTEND[0], ANTENNA_MAX_EXTEND[1], ab);
                if (ex > hi + t) out.push(side + '_antenna_extend ' + ex.toFixed(3) + ' above display-safe ' + hi.toFixed(3) + ' at abduct ' + ab.toFixed(3));
            }
        }
        return out;
    }

    // A plinth (stand: none) must stay wholly below the floor top: never a wall or cue.
    function plinthStatus(topY, floorY) {
        if (!Number.isFinite(topY)) return {gap: null, state: 'unknown', text: 'no geometry'};
        const gap = topY - floorY;
        return gap < 0 ? {gap, state: 'below-floor', text: 'top ' + (-gap).toFixed(4) + ' below floor (plinth, as intended)'}
            : {gap, state: 'pokes-through', text: 'top ' + gap.toFixed(4) + ' ABOVE floor: would read as a wall or cue'};
    }

    // Lift so the model stands on the floor: tarsus = lowest foot vertex (claws incl.),
    // bbox = lowest vertex of anything, none = manifest transform only.
    function standLift(mode, footMinY, bboxMinY, floorY) {
        if (mode === 'tarsus' && Number.isFinite(footMinY)) return floorY - footMinY;
        if (mode === 'bbox' && Number.isFinite(bboxMinY)) return floorY - bboxMinY;
        return 0;
    }

    // Grid / ruler step that suits the model size: 1, 10 or 100 units.
    function gridStep(extent) {
        if (!(extent > 0)) return 1;
        return Math.pow(10, Math.max(0, Math.floor(Math.log10(extent / 2))));
    }

    function parseQuery(search) {
        const q = new URLSearchParams(search || '');
        const list = (k) => (q.get(k) || '').split(',').map((s) => s.trim()).filter(Boolean);
        const lod = q.get('lod');
        return {
            manifest: q.get('manifest') || DEFAULT_MANIFEST_URL,
            asset: q.get('asset') || null,
            lod: lod !== null && /^\d+$/.test(lod) ? Number(lod) : null,
            hide: new Set(list('hide')),
            overlaysOff: new Set(list('off')),
            sheet: q.get('sheet') === '1',
            view: q.get('view') || null,
            lighting: q.get('lights') === 'neutral' ? 'neutral' : 'dashboard',
            compare: q.get('compare') || null,
            clip: q.get('clip') || null
        };
    }

    function viewDirection(name) {
        // Directions from the target to the camera, and the screen-up vector.
        switch (name) {
        case 'top': return {dir: [0, 1, 0], up: [0, 0, 1], ortho: true, label: 'top (head up)'};
        case 'side': return {dir: [1, 0, 0], up: [0, 1, 0], ortho: true, label: 'side (looking along −X)'};
        case 'front': return {dir: [0, 0, 1], up: [0, 1, 0], ortho: true, label: 'front (looking along −Z)'};
        case 'below': return {dir: [0.35, -0.6, 0.7], up: [0, 1, 0], ortho: false, label: 'from below the floor'};
        default: return {dir: [0.62, 0.48, 0.62], up: [0, 1, 0], ortho: false, label: '3/4 perspective'};
        }
    }

    // ------------------------------------------------------------------ browser app
    function start(env) {
        const T = env.THREE;
        const doc = env.document;
        const $ = (id) => doc.getElementById(id);
        const query = parseQuery(env.location.search);
        const errors = [];
        const state = {assets: [], floorY: -0.02, current: null, lod: 0, hidden: new Set(query.hide),
            overlays: new Set(OVERLAYS.filter((o) => !query.overlaysOff.has(o))), records: {}, manifestOk: false};
        state.compareId = query.compare;
        env.window.__gallery = state;   // read by the headless browser check

        function report(text) {
            errors.push(text);
            const box = $('errors');
            box.hidden = false;
            box.textContent = 'Problems (' + errors.length + '):\n' + errors.join('\n');
        }
        env.window.addEventListener('error', (e) => report('page error: ' + (e.message || e)));
        env.window.addEventListener('unhandledrejection', (e) => report('unhandled: ' + ((e.reason && e.reason.message) || e.reason)));

        let renderer;
        try {
            renderer = new T.WebGLRenderer({antialias: true, preserveDrawingBuffer: true});
        } catch (e) {
            report('WebGL unavailable (' + e.message + '): the 3D view and contact sheet cannot render. PNG sprites below still show.');
            renderer = null;
        }
        const view = $('view');
        const scene = new T.Scene();
        scene.background = new T.Color(0x060913);
        const camera = new T.PerspectiveCamera(35, 1, 0.01, 4000);
        camera.position.set(9, 7, 11);
        let controls = null;
        if (renderer) {
            renderer.setPixelRatio(Math.min(env.window.devicePixelRatio || 1, 2));
            view.appendChild(renderer.domElement);
            controls = new T.OrbitControls(camera, renderer.domElement);
            controls.enableDamping = true;
            controls.minDistance = 0.25;   // close range: a tarsal claw fills the view
            controls.maxDistance = 1500;
        }
        // The dashboard viewport's lights, so colours match the opt-in dashboard view.
        scene.add(new T.AmbientLight(0xffffff, 0.65));
        const key = new T.DirectionalLight(0x38bdf8, 1.2); key.position.set(20, 40, 20); scene.add(key);
        const back = new T.DirectionalLight(0xf59e0b, 0.6); back.position.set(-20, 20, -20); scene.add(back);
        const fill = new T.DirectionalLight(0xffffff, 0.35); fill.position.set(0, -30, 10); scene.add(fill);
        // Optional neutral white key, to judge surface colours without the dashboard's tint.
        function setLighting(mode) {
            const neutral = mode === 'neutral';
            key.color.setHex(neutral ? 0xffffff : 0x38bdf8); key.intensity = neutral ? 0.9 : 1.2;
            back.color.setHex(neutral ? 0xffffff : 0xf59e0b); back.intensity = neutral ? 0.4 : 0.6;
            state.lighting = neutral ? 'neutral' : 'dashboard';
        }

        const overlay = new T.Group(); overlay.name = 'gallery-overlays'; scene.add(overlay);
        const holder = new T.Group(); holder.name = 'gallery-asset'; scene.add(holder);

        const loader = renderer ? new T.GLTFLoader() : null;
        const cache = {};
        function loadGlb(url) {
            if (!loader) return Promise.reject(new Error('no WebGL / loader'));
            if (!cache[url]) {
                cache[url] = fetch(url, {method: 'HEAD'}).then((r) => {
                    if (!r.ok) throw new Error('HTTP ' + r.status + ' for ' + url + ' (not built or not staged: tools/assets/build_all.sh OUT --stage)');
                }).then(() => new Promise((resolve, reject) => loader.load(url, (g) => {
                    env.lib.matchRendererEncoding(T, g.scene, renderer);
                    resolve(g);
                }, undefined, (err) => reject(new Error('could not parse ' + url + ': ' + ((err && err.message) || err))))));
            }
            return cache[url];
        }

        // Build a placed instance of an asset LOD, measured from its own vertices.
        function instantiate(asset, lod) {
            return loadGlb(lod.url).then((gltf) => {
                let model;
                if (asset.node) {
                    const src = gltf.scene.getObjectByName(asset.node);
                    if (!src) throw new Error(asset.id + ': node ' + asset.node + ' missing in ' + lod.url);
                    model = src.clone();
                    model.position.set(0, 0, 0); model.quaternion.set(0, 0, 0, 1); model.scale.set(1, 1, 1);
                } else {
                    model = gltf.scene.clone();
                }
                let problem = null;
                if (asset.validate === 'fly') problem = env.lib.validateFly(T, env.lib.nodeIndex(model));
                const extras = rootExtras(model);
                const ds = displayScaleOf(asset.displayScale, extras);
                model.scale.setScalar(ds.scale);
                const wrap = new T.Group();
                wrap.add(model);
                if (asset.transform) {
                    const p = asset.transform.position, s = asset.transform.scale;
                    if (Array.isArray(p)) wrap.position.set(p[0], p[1], p[2]);
                    if (Array.isArray(s)) wrap.scale.set(s[0], s[1], s[2]);
                }
                const raw = vertexBounds(T, wrap);
                const footMin = asset.stand === 'tarsus' ? env.lib.lowestFootY(T, model) : null;
                // lowestFootY measures in the unscaled root frame; carry it through the
                // root's display scale and the wrapper transform (stand = floor - s * foot).
                const footPlaced = footMin === null ? NaN : footMin * ds.scale * wrap.scale.y + wrap.position.y;
                const lift = standLift(asset.stand, footPlaced, raw ? raw.min.y : NaN, state.floorY);
                wrap.position.y += lift;
                const placed = vertexBounds(T, wrap);
                const nonLeg = asset.stand === 'tarsus' ? vertexBounds(T, wrap, (m) => !isLegMesh(m)) : null;
                const layerCount = {};
                model.traverse((o) => { if (o.isMesh) { const id = layerOf(o, asset.layers, wrap); layerCount[id] = (layerCount[id] || 0) + 1; } });
                const rec = {
                    asset: asset.id, lod: lod.level, url: lod.url, tris: triangles(model), budget: lod.budget,
                    rawMinY: raw ? raw.min.y : null, footMinY: footMin, lift,
                    placedMinY: placed ? placed.min.y : null,
                    size: placed ? [placed.max.x - placed.min.x, placed.max.y - placed.min.y, placed.max.z - placed.min.z] : null,
                    floor: asset.stand === 'none' ? plinthStatus(placed ? placed.max.y : NaN, state.floorY)
                        : floorStatus(placed ? placed.min.y : NaN, state.floorY),
                    placedMaxY: placed ? placed.max.y : null,
                    bodyClearance: nonLeg ? nonLeg.min.y - state.floorY : null,
                    layers: layerCount, problem, displayScale: ds.scale, displayScaleSource: ds.source,
                    variant: extras.nf_variant || null, rig: extras.nf_rig || null,
                    appearanceOnly: extras.nf_appearance_only || null, scaleBasis: extras.nf_display_scale_basis || null,
                    declared: Object.entries(extras).filter(([k, v]) => /^neurofly_/.test(k) && typeof v === 'string')
                };
                state.records[asset.id + '@' + lod.level] = rec;
                rec.clips = clipSummary(gltf.animations);
                return {wrap, model, rec, box: placed, animations: gltf.animations || []};
            });
        }

        // ----------------------------------------------------------- overlays
        function textSprite(text, color, height) {
            const c = doc.createElement('canvas');
            const g = c.getContext('2d');
            const font = 'bold 40px monospace';
            g.font = font;
            c.width = Math.ceil(g.measureText(text).width) + 16; c.height = 56;
            g.font = font; g.fillStyle = 'rgba(6,9,19,0.78)'; g.fillRect(0, 0, c.width, c.height);
            g.fillStyle = color; g.textBaseline = 'middle'; g.fillText(text, 8, 28);
            const tex = new T.CanvasTexture(c);
            const s = new T.Sprite(new T.SpriteMaterial({map: tex, depthTest: false, transparent: true}));
            s.scale.set(height * c.width / c.height, height, 1);
            s.renderOrder = 10;
            return s;
        }
        function line(points, color, dashed) {
            const g = new T.BufferGeometry().setFromPoints(points.map((p) => new T.Vector3(p[0], p[1], p[2])));
            const m = dashed ? new T.LineDashedMaterial({color, dashSize: 0.2, gapSize: 0.12}) : new T.LineBasicMaterial({color});
            const l = new T.Line(g, m);
            if (dashed) l.computeLineDistances();
            return l;
        }
        function buildOverlays(box, extraBoxes) {
            while (overlay.children.length) overlay.remove(overlay.children[0]);
            if (!box) return;
            const own = box;
            box = box.clone();
            for (const b of extraBoxes || []) box.union(b);
            const size = box.getSize(new T.Vector3());
            const extent = Math.max(size.x, size.z, size.y, 1);
            const step = gridStep(extent);
            const half = Math.ceil(Math.max(Math.abs(box.min.x), Math.abs(box.max.x), Math.abs(box.min.z), Math.abs(box.max.z)) / step + 1) * step;
            const labelH = Math.max(0.25, extent * 0.045);
            const fy = state.floorY;
            const groups = {};
            const group = (id) => { const g = new T.Group(); g.name = 'overlay-' + id; g.userData.overlay = id; overlay.add(g); groups[id] = g; return g; };

            const floor = group('floor');
            const plane = new T.Mesh(new T.PlaneGeometry(2 * half, 2 * half), new T.MeshStandardMaterial(
                {color: 0x0b1220, roughness: 0.95, transparent: true, opacity: 0.72, side: T.DoubleSide, depthWrite: false}));
            plane.rotation.x = -Math.PI / 2; plane.position.y = fy; floor.add(plane);

            const grid = group('grid');
            const gh = new T.GridHelper(2 * half, Math.round(2 * half / step), 0x475569, 0x1e293b);
            gh.position.y = fy + 0.0005; grid.add(gh);

            const bounds = group('bounds');
            for (const b of [own].concat(extraBoxes || [])) bounds.add(new T.Box3Helper(b, 0xf472b6));
            // Dashed outline of the lowest point, drawn on the floor plane.
            bounds.add(line([[box.min.x, box.min.y, box.min.z], [box.max.x, box.min.y, box.min.z], [box.max.x, box.min.y, box.max.z],
                [box.min.x, box.min.y, box.max.z], [box.min.x, box.min.y, box.min.z]], 0xfacc15, true));

            const axes = group('axes');
            const len = Math.max(step, extent * 0.55);
            axes.add(new T.AxesHelper(len));
            const al = group('labels');
            const lab = (t, c, p) => { const s = textSprite(t, c, labelH); s.position.set(p[0], p[1], p[2]); al.add(s); };
            lab('+X', '#f87171', [len * 1.08, 0, 0]);
            lab('+Y up', '#4ade80', [0, len * 1.08, 0]);
            lab('+Z forward (fly head)', '#60a5fa', [0, 0, len * 1.08]);

            const ruler = group('ruler');
            const x0 = box.min.x, z0 = box.max.z + step * 0.6, y0 = fy + 0.002;
            ruler.add(line([[x0, y0, z0], [x0 + step, y0, z0]], 0xe2e8f0));
            for (let k = 0; k <= 10; k += 1) {
                const x = x0 + step * k / 10, h = (k % 5 === 0 ? 0.25 : 0.12) * step;
                ruler.add(line([[x, y0, z0], [x, y0, z0 - h]], 0xe2e8f0));
            }
            const rs = textSprite(step + ' unit' + (step === 1 ? '' : 's') + ' = ' + step + ' viewport-mm', '#e2e8f0', labelH);
            rs.position.set(x0 + step / 2, y0 + labelH, z0 + step * 0.35); ruler.add(rs);
            state.gridStep = step;
            applyOverlayVisibility();
        }
        function applyOverlayVisibility() {
            for (const g of overlay.children) {
                const id = g.userData.overlay;
                g.visible = state.overlays.has(id) && (id !== 'labels' || state.overlays.has('axes'));
            }
        }
        function applyLayerVisibility() {
            const cur = state.current;
            if (!cur) return;
            cur.model.traverse((o) => {
                if (o.isMesh) o.visible = !state.hidden.has(layerOf(o, cur.asset.layers, cur.wrap));
            });
            const cmp = state.compare;
            if (cmp) cmp.model.traverse((o) => {
                if (o.isMesh) o.visible = !state.hidden.has(layerOf(o, cmp.asset.layers, cmp.wrap));
            });
        }

        // Optional second asset placed beside the current one (+X), same floor rule.
        function loadCompare(mainBox) {
            state.compare = null;
            const other = state.assets.find((a) => a.id === state.compareId);
            if (!other || !mainBox || (state.current && other.id === state.current.asset.id)) return Promise.resolve(null);
            const lod = other.lods.find((l) => l.level === state.lod) || other.lods[0];
            return instantiate(other, lod).then((inst) => {
                const gap = Math.max(1, 0.15 * (mainBox.max.x - mainBox.min.x));
                const dx = mainBox.max.x + gap - inst.box.min.x;
                inst.wrap.position.x += dx;
                inst.box.translate(new T.Vector3(dx, 0, 0));
                state.compare = {asset: other, wrap: inst.wrap, model: inst.model, rec: inst.rec, box: inst.box};
                return state.compare;
            }, (err) => { report('compare ' + other.id + ': ' + err.message); return null; });
        }
        function renderCompareSelect() {
            const sel = $('compare');
            const keep = state.compareId || '';
            sel.textContent = '';
            sel.append(el('option', {value: ''}, 'none'));
            for (const a of state.assets) if (!state.current || a.id !== state.current.asset.id) sel.append(el('option', {value: a.id}, a.title));
            sel.value = keep;
        }

        // ----------------------------------------------------------- camera
        function frame(box, viewName, cam, aspect) {
            const v = viewDirection(viewName);
            const center = box.getCenter(new T.Vector3());
            const radius = Math.max(0.5, box.getSize(new T.Vector3()).length() / 2);
            const dir = new T.Vector3().fromArray(v.dir).normalize();
            if (cam.isOrthographicCamera) {
                const h = radius * 1.15, w = h * (aspect || 1);
                cam.left = -w; cam.right = w; cam.top = h; cam.bottom = -h; cam.near = -radius * 10; cam.far = radius * 10;
                cam.position.copy(center).addScaledVector(dir, radius * 3);
            } else {
                const dist = radius / Math.sin(T.MathUtils.degToRad(cam.fov / 2)) * 1.05;
                cam.position.copy(center).addScaledVector(dir, dist);
                cam.near = Math.max(0.005, dist / 500); cam.far = dist * 50;
            }
            cam.up.fromArray(v.up);
            cam.lookAt(center);
            cam.updateProjectionMatrix();
            return center;
        }
        function focus(viewName) {
            if (!state.current || !state.current.box) return;
            const box = state.current.box.clone();
            if (state.compare) box.union(state.compare.box);
            const c = frame(box, viewName || '34', camera, camera.aspect);
            if (controls) { controls.target.copy(c); controls.update(); }
        }

        // ----------------------------------------------------------- UI
        function el(tag, attrs, text) {
            const e = doc.createElement(tag);
            for (const [k, v] of Object.entries(attrs || {})) {
                if (k === 'class') e.className = v; else if (k.startsWith('on')) e.addEventListener(k.slice(2), v); else e.setAttribute(k, v);
            }
            if (text !== undefined) e.textContent = text;
            return e;
        }
        function renderAssetList() {
            const list = $('assetList');
            list.textContent = '';
            const byCat = {};
            for (const a of state.assets) (byCat[a.category] = byCat[a.category] || []).push(a);
            for (const [cat, assets] of Object.entries(byCat)) {
                list.append(el('div', {class: 'cat'}, cat));
                for (const a of assets) {
                    const b = el('button', {class: 'asset' + (state.current && state.current.asset.id === a.id ? ' on' : ''), 'data-id': a.id,
                        onclick: () => select(a.id, null)});
                    b.append(el('span', {}, a.title), el('span', {class: 'badge s-' + a.status}, a.status.toUpperCase()));
                    if (a.loadState) b.append(el('span', {class: 'badge l-' + a.loadState}, a.loadState));
                    list.append(b);
                }
            }
        }
        function renderLodButtons(asset) {
            const box = $('lods');
            box.textContent = '';
            for (const l of asset.lods) {
                const id = 'lod-' + l.level;
                const input = el('input', {type: 'radio', name: 'lod', id, value: String(l.level)});
                input.checked = l.level === state.lod;
                input.addEventListener('change', () => select(asset.id, l.level));
                const lab = el('label', {for: id});
                lab.append(input, doc.createTextNode(' ' + l.label));
                box.append(lab);
            }
        }
        function renderLayerToggles(asset) {
            const box = $('layers');
            box.textContent = '';
            const counts = (state.current && state.current.rec.layers) || {};
            const all = asset.layers.map((l) => [l.id, l.label]);
            if (counts.other) all.push(['other', 'Other (unclassified meshes)']);
            for (const [id, label] of all) {
                const input = el('input', {type: 'checkbox', id: 'layer-' + id, 'data-layer': id});
                input.checked = !state.hidden.has(id);
                input.addEventListener('change', () => {
                    if (input.checked) state.hidden.delete(id); else state.hidden.add(id);
                    applyLayerVisibility();
                });
                const lab = el('label', {for: 'layer-' + id});
                lab.append(input, doc.createTextNode(' ' + label + ' · ' + (counts[id] || 0) + ' mesh' + ((counts[id] || 0) === 1 ? '' : 'es')));
                box.append(lab);
            }
            if (!all.length) box.append(el('div', {class: 'muted'}, 'No layers declared for this asset.'));
        }
        function renderOverlayToggles() {
            const box = $('overlays');
            box.textContent = '';
            const names = {floor: 'Floor plane (y = ' + state.floorY + ')', bounds: 'Bounds box + lowest-point outline',
                axes: 'Axes (+X red, +Y green up, +Z blue head)', labels: 'Axis labels', grid: 'Grid', ruler: 'Scale ruler'};
            for (const id of OVERLAYS) {
                const input = el('input', {type: 'checkbox', id: 'ov-' + id, 'data-overlay': id});
                input.checked = state.overlays.has(id);
                input.addEventListener('change', () => {
                    if (input.checked) state.overlays.add(id); else state.overlays.delete(id);
                    applyOverlayVisibility();
                });
                const lab = el('label', {for: 'ov-' + id});
                lab.append(input, doc.createTextNode(' ' + names[id]));
                box.append(lab);
            }
        }
        const fmt = (v, d) => (v === null || v === undefined || !Number.isFinite(v) ? '–' : v.toFixed(d === undefined ? 4 : d));
        function renderFacts(asset, rec) {
            const dl = $('facts');
            dl.textContent = '';
            const row = (k, v, cls) => { dl.append(el('dt', {}, k)); dl.append(el('dd', cls ? {class: cls} : {}, v)); };
            row('asset', asset.id + ' · LOD' + rec.lod);
            row('file', rec.url);
            row('triangles', rec.tris.toLocaleString() + (rec.budget ? ' / budget ' + rec.budget.toLocaleString() + (rec.tris <= rec.budget ? ' (within)' : ' (OVER)') : ''),
                rec.budget && rec.tris > rec.budget ? 'bad' : '');
            row('size x·y·z', rec.size ? rec.size.map((s) => s.toFixed(3)).join(' · ') + ' units' : '–');
            if (rec.variant || rec.displayScale !== 1) {
                row('variant', (rec.variant || 'neutral') + (rec.rig ? ' · rig ' + rec.rig : ''));
                row('display scale', rec.displayScale.toFixed(4) + ' (' + rec.displayScaleSource + ')' + (rec.scaleBasis ? ' · ' + rec.scaleBasis : ''));
            }
            if (rec.appearanceOnly || asset.appearanceOnly) row('appearance only', rec.appearanceOnly || 'no brain dataset, physiology or behaviour is tied to this appearance');
            for (const [k, v] of rec.declared || []) row(k.replace(/^neurofly_/, ''), v, k === 'neurofly_status' ? 'bad' : '');
            row('stand mode', asset.stand + (asset.stand === 'tarsus' ? ' (lowest tarsus vertex, claws incl.)' : ''));
            if (rec.footMinY !== null) row('lowest foot (GLB root, unscaled)', fmt(rec.footMinY));
            row('raw lowest vertex', fmt(rec.rawMinY));
            row('lift applied', fmt(rec.lift));
            row('placed lowest vertex', fmt(rec.placedMinY));
            row('floor check', rec.floor.text, rec.floor.state === 'contact' || rec.floor.state === 'below-floor' ? '' : 'bad');
            if (rec.bodyClearance !== null) row('body clearance', fmt(rec.bodyClearance) + (rec.bodyClearance < 0 ? ' (body below floor!)' : ' above floor'), rec.bodyClearance < 0 ? 'bad' : '');
            row('loader check', rec.problem ? 'FAIL: ' + rec.problem : (asset.validate ? 'pass (' + env.lib.requiredFlyNodes().length + ' hook nodes)' : 'n/a'), rec.problem ? 'bad' : '');
            row('grid / ruler step', (state.gridStep || 1) + ' viewport-mm');
            $('notes').textContent = asset.notes;
        }
        function renderLodTable(asset) {
            const tb = $('lodTable');
            tb.textContent = '';
            const hdr = el('tr');
            for (const h of ['LOD', 'triangles', 'lowest foot', 'lift', 'placed min', 'floor', 'body clearance']) hdr.append(el('th', {}, h));
            tb.append(hdr);
            Promise.all(asset.lods.map((l) => instantiate(asset, l).then((r) => r.rec, (e) => ({lod: l.level, error: e.message}))))
                .then((recs) => {
                    for (const r of recs) {
                        const tr = el('tr', {'data-lod': String(r.lod)});
                        if (r.error) { tr.append(el('td', {}, 'LOD' + r.lod), el('td', {colspan: '6', class: 'bad'}, r.error)); tb.append(tr); continue; }
                        tr.append(el('td', {}, 'LOD' + r.lod), el('td', {}, r.tris.toLocaleString()), el('td', {}, fmt(r.footMinY)),
                            el('td', {}, fmt(r.lift)), el('td', {}, fmt(r.placedMinY)),
                            el('td', {class: r.floor.state === 'contact' || r.floor.state === 'below-floor' ? '' : 'bad'}, r.floor.state),
                            el('td', {}, fmt(r.bodyClearance)));
                        tb.append(tr);
                    }
                    const feet = recs.filter((r) => !r.error && Number.isFinite(r.footMinY)).map((r) => r.footMinY);
                    if (feet.length > 1) {
                        const spread = Math.max(...feet) - Math.min(...feet);
                        const tr = el('tr');
                        tr.append(el('td', {colspan: '7', class: spread > FLOOR_TOLERANCE ? 'bad' : ''},
                            'LOD agreement: lowest-foot spread ' + spread.toFixed(5) + (spread > FLOOR_TOLERANCE ? ' (LODs disagree)' : ' (LODs agree)')));
                        tb.append(tr);
                    }
                });
        }

        let token = 0;
        function select(id, lodLevel) {
            const asset = state.assets.find((a) => a.id === id) || state.assets[0];
            if (!asset) return Promise.resolve(null);
            const lod = asset.lods.find((l) => l.level === lodLevel) || asset.lods.find((l) => l.level === state.lod) || asset.lods[0];
            const assetChanged = !state.current || state.current.asset.id !== asset.id;
            state.lod = lod.level;
            const my = ++token;
            $('status').textContent = 'Loading ' + lod.url + '…';
            $('status').className = 'status';
            asset.loadState = 'loading';
            return instantiate(asset, lod).then((inst) => {
                if (my !== token) return null;
                state.current = {asset, wrap: inst.wrap, model: inst.model, rec: inst.rec, box: inst.box};
                return loadCompare(inst.box).then(() => inst);
            }).then((inst) => {
                if (!inst || my !== token) return null;
                const asset = state.current.asset;
                while (holder.children.length) holder.remove(holder.children[0]);
                holder.add(inst.wrap);
                if (state.compare) holder.add(state.compare.wrap);
                asset.loadState = inst.rec.problem ? 'invalid' : 'loaded';
                buildOverlays(inst.box, state.compare ? [state.compare.box] : []);
                renderCompareSelect();
                $('compareInfo').textContent = state.compare ? 'Beside it (+X): ' + state.compare.asset.id + ' LOD' + state.compare.rec.lod
                    + ' · display scale ' + state.compare.rec.displayScale.toFixed(4) + ' · floor ' + state.compare.rec.floor.state : '';
                applyLayerVisibility();
                renderLodButtons(asset); renderLayerToggles(asset); renderFacts(asset, inst.rec);
                setupAnimation(asset, inst);
                if (assetChanged || state.compare) focus(query.view || '34');
                if (assetChanged) renderLodTable(asset);
                renderAssetList();
                $('status').textContent = 'Loaded ' + lod.url + (inst.rec.problem ? ' · loader check FAILED: ' + inst.rec.problem : '');
                $('status').className = 'status' + (inst.rec.problem ? ' bad' : ' ok');
                $('view').dataset.asset = asset.id; $('view').dataset.lod = String(lod.level);
                return inst;
            }).catch((err) => {
                if (my !== token) return null;
                asset.loadState = 'missing';
                renderAssetList();
                $('status').textContent = 'Unavailable: ' + err.message;
                $('status').className = 'status bad';
                report(asset.id + ' LOD' + lod.level + ': ' + err.message);
                return null;
            });
        }

        // ----------------------------------------------------------- animation
        // The gallery is a standalone page with no telemetry: clips play on the gallery's
        // own model only.  The label below is persistent while an animated asset is shown.
        const anim = {mixer: null, clips: [], action: null, playing: false, speed: 1, loop: true, bind: new Map(), nodes: [], wanted: null};
        state.anim = anim;
        function clipLoad(asset, lod) {
            // Embedded animations plus any separate clip GLBs listed in the manifest
            // ('{lod}' in a clip url selects the clip file built for the shown LOD).
            return Promise.all(asset.clips.map((c) => ({...c, url: c.url.replace('{lod}', String(lod))})).map((c) => loadGlb(c.url).then((g) => (g.animations || []).map((a) => ({clip: a, label: (a.name || 'unnamed clip') + ' · ' + c.label, src: c.url})),
                (err) => { report(asset.id + ' clip ' + c.url + ': ' + err.message); return []; })));
        }
        function setupAnimation(asset, inst) {
            const prevName = anim.action ? anim.action.getClip().name : anim.wanted;
            const prevTime = anim.action ? anim.action.time : 0;
            const wasPlaying = anim.playing;
            if (anim.mixer) anim.mixer.stopAllAction();
            anim.mixer = null; anim.action = null; anim.clips = []; anim.bind = new Map(); anim.nodes = [];
            const embedded = (inst.animations || []).map((a) => ({clip: a, label: a.name || 'clip', src: inst.rec.url}));
            const jointsUrl = asset.joints ? asset.joints.replace('{lod}', String(inst.rec.lod)) : null;
            return Promise.all([clipLoad(asset, inst.rec.lod), loadJoints(jointsUrl)]).then(([lists, contract]) => {
                if (!state.current || state.current.model !== inst.model) return;
                anim.contract = contract;
                anim.clips = embedded.concat(...lists);
                const box = $('animPanel');
                box.hidden = !anim.clips.length;
                $('animLabel').hidden = !anim.clips.length;
                if (!anim.clips.length) { $('joints').textContent = ''; return; }
                inst.model.updateMatrixWorld(true);
                const rf = vertexBounds(T, inst.model, (m) => { for (let o = m; o; o = o.parent) if (/^(lf|lm|lh|rf|rm|rh)_tarsus$/.test(o.name || '')) return true; return false; });
                anim.restFoot = rf ? rf.min.y : NaN;
                anim.mixer = new T.AnimationMixer(inst.model);
                const sel = $('clip');
                sel.textContent = '';
                anim.clips.forEach((c, i) => sel.append(el('option', {value: String(i)}, c.label + ' (' + c.clip.duration.toFixed(2) + ' s)')));
                let idx = anim.clips.findIndex((c) => c.clip.name === prevName);
                if (idx < 0) idx = 0;
                sel.value = String(idx);
                chooseClip(idx, prevName === anim.clips[idx].clip.name ? prevTime : 0, wasPlaying);
            });
        }
        const jointCache = {};
        function loadJoints(url) {
            if (!url) return Promise.resolve(null);
            if (!jointCache[url]) jointCache[url] = fetch(url, {cache: 'no-store'}).then((r) => { if (!r.ok) throw new Error('HTTP ' + r.status); return r.json(); })
                .then((d) => {
                    const axes = {}, ranges = {};
                    for (const j of d.joints || []) if (Array.isArray(j.axis)) { axes[j.node] = j.axis; if (Array.isArray(j.range_rad)) ranges[j.node] = j.range_rad; }
                    return {axes, ranges, rig: d.rig || null};
                }).catch((e) => { report('joint contract ' + url + ': ' + e.message); return null; });
            return jointCache[url];
        }
        function currentAngles() {
            const out = {};
            if (!anim.contract) return out;
            for (const n of anim.nodes) {
                const ax = anim.contract.axes[n];
                const o = state.current.model.getObjectByName(n);
                if (ax && o) out[n] = dofAngle(o.quaternion, ax);
            }
            return out;
        }
        // Sample the whole clip on the gallery model against the documented envelope,
        // then restore the time it started from.
        const isTarsus = (m) => { for (let o = m; o; o = o.parent) if (/^(lf|lm|lh|rf|rm|rh)_tarsus$/.test(o.name || '')) return true; return false; };
        function scanEnvelope() {
            if (!anim.action || !anim.contract) return null;
            const clip = anim.action.getClip();
            const dur = clip.duration, keep = anim.action.time, n = Math.max(120, Math.ceil(dur * 60));
            const model = state.current.model;
            const movesLegs = trackNodes(clip).some((x) => /_(coxa|femur|tibia)_joint$/.test(x));
            const found = [];
            let lowest = Infinity;
            for (let k = 0; k <= n; k += 1) {
                anim.action.time = dur * k / n; anim.mixer.update(0);
                for (const v of envelopeViolations(currentAngles(), anim.contract.ranges)) found.push('t=' + (dur * k / n).toFixed(2) + ' s ' + v);
                if (movesLegs) { model.updateMatrixWorld(true); const b = vertexBounds(T, model, isTarsus); if (b) lowest = Math.min(lowest, b.min.y); }
            }
            anim.action.time = keep; anim.mixer.update(0);
            const notes = [];
            if (movesLegs && Number.isFinite(lowest) && Number.isFinite(anim.restFoot) && lowest < anim.restFoot - 1e-3)
                notes.push('feet go ' + (anim.restFoot - lowest).toFixed(3) + ' viewport-mm below the rest foot plane (through the floor)');
            const q0 = new T.Quaternion(), q1 = new T.Quaternion();
            for (const tr of clip.tracks) {
                if (!/\.quaternion$/.test(tr.name)) continue;
                for (let i = 1; i < tr.times.length; i += 1) {
                    q0.fromArray(tr.values, 4 * i - 4); q1.fromArray(tr.values, 4 * i);
                    const jump = 2 * Math.acos(Math.min(1, Math.abs(q0.dot(q1))));
                    if (jump > 1.0) notes.push('one-frame pop: ' + tr.name.split('.')[0] + ' key ' + (i - 1) + '→' + i + ' (t=' + tr.times[i - 1].toFixed(3) + ' s) jumps ' + jump.toFixed(2) + ' rad');
                }
            }
            return {samples: n + 1, violations: found, notes};
        }
        function chooseClip(i, time, play) {
            const c = anim.clips[i];
            if (!c || !anim.mixer) return;
            if (anim.action) anim.action.stop();
            // Bind pose for the readout: the joints as loaded, before the clip moves them.
            for (const [name, q] of anim.bind) { const o = state.current.model.getObjectByName(name); if (o) o.quaternion.copy(q); }
            anim.nodes = trackNodes(c.clip).filter((n) => state.current.model.getObjectByName(n));
            for (const n of anim.nodes) if (!anim.bind.has(n)) anim.bind.set(n, state.current.model.getObjectByName(n).quaternion.clone());
            anim.action = anim.mixer.clipAction(c.clip);
            applyLoop();
            anim.action.play();
            anim.action.time = Math.min(time || 0, c.clip.duration);
            anim.mixer.update(0);
            setPlaying(!!play);
            const missing = trackNodes(c.clip).length - anim.nodes.length;
            $('clipInfo').textContent = c.label + ' · ' + c.clip.duration.toFixed(2) + ' s · ' + c.clip.tracks.length + ' tracks on '
                + anim.nodes.length + ' joints' + (missing ? ' · ' + missing + ' track targets absent in this LOD' : '') + ' · ' + c.src;
            const scan = scanEnvelope();
            anim.envelope = scan;
            const env = $('envelope');
            if (!scan) { env.textContent = 'Joint envelope: not checked (no joint contract for this asset)'; env.className = 'status bad'; }
            else {
                const lines = [scan.violations.length ? 'Joint envelope (W2 contract): ' + scan.violations.length + ' violation(s) over ' + scan.samples + ' samples'
                    : 'Joint envelope (W2 contract): inside over ' + scan.samples + ' samples of the clip'];
                lines.push(...scan.violations.slice(0, 4), ...scan.notes.slice(0, 4).map((x) => 'Clip check: ' + x));
                env.textContent = lines.join('\n');
                env.className = 'status ' + (scan.violations.length || scan.notes.length ? 'bad' : 'ok');
            }
            renderJoints();
        }
        function applyLoop() {
            if (!anim.action) return;
            anim.action.setLoop(anim.loop ? T.LoopRepeat : T.LoopOnce, Infinity);
            anim.action.clampWhenFinished = true;
        }
        function setPlaying(on) {
            anim.playing = on;
            if (anim.action) anim.action.paused = !on;
            $('animPlay').textContent = on ? 'Pause' : 'Play';
            $('animPlay').setAttribute('aria-pressed', on ? 'true' : 'false');
            $('animState').textContent = on ? 'PLAYING' : 'PAUSED';
        }
        function renderJoints() {
            const t = $('joints');
            t.textContent = '';
            if (!anim.action) return;
            const hdr = el('tr');
            for (const h of ['joint', 'θ about axis', 'range', 'from bind', '']) hdr.append(el('th', {}, h));
            t.append(hdr);
            const angles = currentAngles();
            const bad = new Set(envelopeViolations(angles, anim.contract && anim.contract.ranges).map((v) => v.split(' ')[0]));
            for (const n of anim.nodes) {
                const o = state.current.model.getObjectByName(n);
                const r = jointReadout(T, o, anim.bind.get(n));
                const tr = el('tr', {'data-joint': n});
                const bar = el('span', {class: 'jbar'});
                bar.style.width = Math.min(48, Math.abs(r.delta || 0) / 2) + 'px';  // 2° per px, capped
                const cell = el('td'); cell.append(bar);
                const th = angles[n];
                const rg = anim.contract && anim.contract.ranges[n];
                tr.append(el('td', {}, n), el('td', bad.has(n) ? {class: 'bad'} : {}, th === undefined ? '–' : th.toFixed(3) + ' rad (' + (th * 180 / Math.PI).toFixed(1) + '°)' + (bad.has(n) ? ' ⚠' : '')),
                    el('td', {}, rg ? '[' + rg[0] + ', ' + rg[1] + ']' : '–'),
                    el('td', {}, (r.delta === null ? '–' : r.delta.toFixed(1) + '°')), cell);
                t.append(tr);
            }
            $('animTime').textContent = 't = ' + anim.action.time.toFixed(2) + ' / ' + anim.action.getClip().duration.toFixed(2) + ' s · speed ×' + anim.speed;
        }
        $('clip').addEventListener('change', () => { anim.wanted = null; chooseClip(Number($('clip').value), 0, anim.playing); });
        $('animPlay').addEventListener('click', () => {
            if (!anim.action) return;
            if (anim.atRest) { anim.atRest = false; anim.action.reset(); anim.action.play(); }
            if (!anim.loop && anim.action.time >= anim.action.getClip().duration - 1e-6) { anim.action.reset(); anim.action.play(); }
            setPlaying(!anim.playing);
        });
        $('animReset').addEventListener('click', () => {
            if (!anim.action) return;
            anim.atRest = false; anim.action.reset(); anim.action.play(); anim.action.time = 0; anim.mixer.update(0); setPlaying(false); renderJoints();
        });
        $('animRest').addEventListener('click', () => {
            if (!anim.action) return;
            anim.action.stop();          // deactivates the clip; Play or Reset restarts it
            for (const [name, q] of anim.bind) { const o = state.current.model.getObjectByName(name); if (o) o.quaternion.copy(q); }
            setPlaying(false);
            anim.atRest = true;
            $('animTime').textContent = 'rest (bind) pose · clip stopped';
            renderJointsAtRest();
        });
        function renderJointsAtRest() {
            const t = $('joints'); t.textContent = '';
            const hdr = el('tr'); for (const h of ['joint', 'θ about axis', 'range', 'from bind', '']) hdr.append(el('th', {}, h)); t.append(hdr);
            for (const n of anim.nodes) { const rg = anim.contract && anim.contract.ranges[n]; const tr = el('tr', {'data-joint': n}); tr.append(el('td', {}, n), el('td', {}, '0.000 rad (rest)'), el('td', {}, rg ? '[' + rg[0] + ', ' + rg[1] + ']' : '–'), el('td', {}, '0.0°'), el('td')); t.append(tr); }
        }
        $('animLoop').addEventListener('change', () => { anim.loop = $('animLoop').checked; applyLoop(); });
        for (const v of SPEEDS) $('animSpeed').append(el('option', {value: String(v)}, '×' + v));
        $('animSpeed').value = '1';
        $('animSpeed').addEventListener('change', () => { anim.speed = Number($('animSpeed').value); renderJoints(); });
        if (query.clip) anim.wanted = query.clip;
        let lastTick = null, lastJoints = 0;
        function animTick(nowMs) {
            const dt = lastTick === null ? 0 : Math.min(0.1, (nowMs - lastTick) / 1000);
            lastTick = nowMs;
            if (anim.mixer && anim.action && anim.playing) {
                anim.mixer.update(dt * anim.speed);
                if (!anim.loop && anim.action.time >= anim.action.getClip().duration - 1e-6) setPlaying(false);
                if (nowMs - lastJoints > 150) { lastJoints = nowMs; renderJoints(); }
            }
        }

        // ----------------------------------------------------------- contact sheet
        function contactSheet() {
            const out = $('sheet');
            out.textContent = '';
            const views = ['34', 'top', 'side', 'front'];
            const tile = 240;
            if (!renderer) { out.append(el('div', {class: 'bad'}, 'No WebGL: contact sheet unavailable.')); return Promise.resolve(null); }
            const off = new T.WebGLRenderer({antialias: true, preserveDrawingBuffer: true});
            off.setSize(tile, tile, false);
            const sheetScene = new T.Scene();
            sheetScene.background = new T.Color(0x0b1220);
            for (const l of scene.children) if (l.isLight) sheetScene.add(l.clone());
            const rows = [];
            for (const a of state.assets) for (const l of a.lods) rows.push([a, l]);
            const canvas = doc.createElement('canvas');
            canvas.width = tile * views.length; canvas.height = (tile + 22) * rows.length;
            const g = canvas.getContext('2d');
            g.fillStyle = '#060913'; g.fillRect(0, 0, canvas.width, canvas.height);
            let chain = Promise.resolve();
            rows.forEach(([a, l], r) => {
                chain = chain.then(() => instantiate(a, l).then((inst) => {
                    sheetScene.add(inst.wrap);
                    const floorG = new T.Mesh(new T.PlaneGeometry(1, 1), new T.MeshBasicMaterial({color: 0x1e293b, side: T.DoubleSide}));
                    const s = Math.max(inst.rec.size[0], inst.rec.size[2]) * 1.6 + 2;
                    floorG.scale.set(s, s, 1); floorG.rotation.x = -Math.PI / 2; floorG.position.y = state.floorY - 0.001;
                    if (a.stand !== 'none') sheetScene.add(floorG);
                    views.forEach((v, c) => {
                        const vd = viewDirection(v);
                        const cam = vd.ortho ? new T.OrthographicCamera(-1, 1, 1, -1, 0.01, 100) : new T.PerspectiveCamera(35, 1, 0.01, 4000);
                        frame(inst.box, v, cam, 1);
                        off.render(sheetScene, cam);
                        g.drawImage(off.domElement, c * tile, r * (tile + 22) + 22);
                        const top = r * (tile + 22) + 22;
                        g.fillStyle = 'rgba(6,9,19,0.7)'; g.fillRect(c * tile, top, tile, 16);
                        g.fillStyle = '#94a3b8'; g.font = '11px monospace';
                        g.fillText(vd.label, c * tile + 6, top + 12);
                        if (vd.ortho) {
                            // Scale bar from the orthographic frustum width.
                            const ppu = tile / (cam.right - cam.left);
                            let bar = gridStep(Math.max(...inst.rec.size));
                            while (bar > 1e-3 && ppu * bar > tile * 0.6) bar /= 10;
                            g.fillStyle = '#e2e8f0';
                            g.fillRect(c * tile + 8, top + tile - 10, ppu * bar, 3);
                            g.fillText(bar + ' viewport-mm', c * tile + 8, top + tile - 14);
                        }
                    });
                    g.fillStyle = '#e2e8f0'; g.font = 'bold 12px monospace';
                    g.fillText(a.title + ' [' + a.id + '] · LOD' + l.level + ' · ' + inst.rec.tris.toLocaleString() + ' triangles'
                        + (inst.rec.displayScale !== 1 ? ' · display ×' + inst.rec.displayScale.toFixed(4) : '') + ' · floor: '
                        + inst.rec.floor.state, 6, r * (tile + 22) + 15);
                    sheetScene.remove(inst.wrap); sheetScene.remove(floorG);
                }, (err) => {
                    g.fillStyle = '#fbbf24'; g.font = '12px monospace';
                    g.fillText(a.id + ' LOD' + l.level + ': ' + err.message.slice(0, 90), 8, r * (tile + 22) + 40);
                }));
            });
            return chain.then(() => {
                off.dispose();
                const img = el('img', {alt: 'contact sheet', id: 'sheetImg'});
                img.src = canvas.toDataURL('image/png');
                const a = el('a', {href: img.src, download: 'neurofly_asset_contact_sheet.png'}, 'Download contact sheet PNG');
                out.append(el('div', {class: 'muted'}, 'All layers drawn, gallery placement, ' + rows.length + ' asset·LOD rows. Ortho views carry a scale bar in viewport-mm.'), a, img);
                state.sheetReady = true;
                return canvas;
            });
        }

        function renderSprites() {
            const box = $('sprites');
            box.textContent = '';
            const list = [];
            for (const s of state.sheets || []) list.push(['legend', s]);
            for (const a of state.assets) for (const s of a.sprites) list.push([a.id, s]);
            for (const [aid, s] of list) {
                const a = {id: aid, title: aid};
                const fig = el('figure');
                const img = el('img', {alt: a.title + ' ' + s.label, loading: 'lazy'});
                const cap = el('figcaption', {}, a.id + ' · ' + s.label);
                img.onerror = () => { cap.textContent = a.id + ' · ' + s.label + ' · NOT BUILT (' + s.url + ')'; cap.className = 'bad'; img.remove(); };
                img.src = s.url;
                fig.append(img, cap); box.append(fig);
            }
        }

        function resize() {
            if (!renderer) return;
            const w = view.clientWidth || 1, h = view.clientHeight || 1;
            renderer.setSize(w, h, false); camera.aspect = w / h; camera.updateProjectionMatrix();
        }
        env.window.addEventListener('resize', resize);
        for (const b of doc.querySelectorAll('[data-view]')) b.addEventListener('click', () => focus(b.dataset.view));
        $('sheetBtn').addEventListener('click', () => contactSheet());
        $('compare').addEventListener('change', () => {
            state.compareId = $('compare').value || null;
            if (state.current) select(state.current.asset.id, state.lod);
        });
        $('lighting').addEventListener('change', () => setLighting($('lighting').value));
        if (query.lighting === 'neutral') { $('lighting').value = 'neutral'; setLighting('neutral'); }

        function boot(parsed, source) {
            state.assets = parsed.assets;
            state.sheets = parsed.sheets;
            state.floorY = parsed.floorY;
            state.manifestOk = !parsed.errors.length;
            $('manifestInfo').textContent = 'Manifest: ' + source + ' · ' + parsed.assets.length + ' assets'
                + (parsed.units ? ' · units: ' + parsed.units.name : '');
            $('unitsNote').textContent = parsed.units && parsed.units.note ? parsed.units.note : '';
            for (const e of parsed.errors) report('manifest: ' + e);
            renderOverlayToggles(); renderAssetList(); renderSprites(); resize();
            return select(query.asset || (state.assets[0] && state.assets[0].id), query.lod).then(() => {
                if (query.sheet) return contactSheet();
                return null;
            }).then(() => { state.ready = true; });
        }

        const manifestUrl = query.manifest;
        if (/^[a-z]+:\/\//i.test(manifestUrl) || manifestUrl.startsWith('//')) {
            report('manifest url must be relative to this page; ignoring ' + manifestUrl);
            state.ready = true;
        } else {
            fetch(manifestUrl, {cache: 'no-store'}).then((r) => {
                if (!r.ok) throw new Error('HTTP ' + r.status);
                return r.json();
            }).then((m) => boot(validateManifest(m), manifestUrl), (err) => {
                report('could not read manifest ' + manifestUrl + ' (' + err.message + ')');
                $('manifestInfo').textContent = 'Manifest unavailable: nothing to show.';
                $('status').textContent = 'No manifest.'; $('status').className = 'status bad';
                state.ready = true;
            });
        }

        (function tick(nowMs) {
            animTick(nowMs || 0);
            if (renderer) {
                if ($('spin').checked && state.current) state.current.wrap.rotation.y += 0.005;
                if (controls) controls.update();
                renderer.render(scene, camera);
            }
            env.window.requestAnimationFrame(tick);
        }());
        return state;
    }

    return {
        MANIFEST_SCHEMA, DEFAULT_MANIFEST_URL, OVERLAYS, FLOOR_TOLERANCE, LEG_PREFIXES,
        validateManifest, compileLayers, classifyName, layerOf, vertexBounds, isLegMesh, triangles,
        floorStatus, plinthStatus, standLift, dofAngle, envelopeViolations, ANIMATION_LABEL, SPEEDS, trackNodes, clipSummary, jointReadout, rootExtras, displayScaleOf, gridStep, parseQuery, viewDirection, start
    };
}));
