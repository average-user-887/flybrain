/*
 * Opt-in presentation assets (?assets=hq).  OFF by default.
 *
 * Without the URL flag every entry point returns null immediately: no script is
 * injected, nothing is fetched, no object, material or transform is touched, so
 * the dashboard and the replay page behave exactly as before.
 *
 * With the flag, the module loads assets/hq/fly_hq_lod0.glb (built by
 * tools/assets/build_fly.py) through the vendored three.js r128 GLTFLoader and
 * hangs each named mesh on the EXISTING pose groups of the procedural fly.  The
 * procedural meshes are hidden through their materials, never removed; contact
 * spheres stay visible because they carry measured stance/swing state.  Pose,
 * heading, joint angles, contacts, arena geometry, cues and every label are left
 * to the code that already owns them.  Any failure (missing loader, missing or
 * invalid file, missing node, absurd scale) keeps the procedural fly.
 *
 * See tools/assets/INTERFACE.md for frames, units and node names.
 */
(function (root, factory) {
    'use strict';
    const api = factory();
    if (typeof module === 'object' && module.exports) module.exports = api;
    if (root) { root.NeuroflyHQAssets = api.create({}); root.NeuroflyHQAssetsLib = api; }
}(typeof window !== 'undefined' ? window : null, function () {
    'use strict';

    const FLAG_VALUE = 'hq';
    const FLY_URL = 'assets/hq/fly_hq_lod0.glb';
    const ARENA_URL = 'assets/hq/arena_shell.glb';
    const LOADER_URL = 'vendor/GLTFLoader.js';
    // Viewport leg order (ArticulatedFly3DViewport.buildFlyMesh) -> FlyGym leg prefix.
    const VIEWPORT_LEGS = {L1: 'lf', L2: 'lm', L3: 'lh', R1: 'rf', R2: 'rm', R3: 'rh'};
    const LEG_PREFIXES = ['lf', 'lm', 'lh', 'rf', 'rm', 'rh'];
    const LEG_PARTS = {coxa: ['coxa'], femur: ['trochanterfemur'], tibia: ['tibia', 'tarsus']};
    const BODY_REQUIRED = ['c_thorax', 'c_head', 'l_eye', 'r_eye', 'c_abdomen12', 'l_wing', 'r_wing'];
    const BODY_OPTIONAL = ['c_scutum', 'c_thorax_bristles', 'c_head_bristles', 'c_proboscis',
        'l_pedicel', 'r_pedicel', 'l_funiculus', 'r_funiculus', 'l_arista', 'r_arista',
        'c_abdomen3', 'c_abdomen4', 'c_abdomen5', 'c_abdomen6', 'l_wing_veins', 'r_wing_veins',
        'l_haltere', 'r_haltere'];
    const VIEWPORT_LABEL = 'HQ asset preview (opt-in) · illustrative surface on the same pose stream'
        + ' · wings, halteres, antennae and bristles are static decoration, not simulated';
    const REPLAY_LABEL = 'HQ asset preview (opt-in) · illustrative surface stretched between recorded'
        + ' joint positions · body drawn rigid; wings, halteres and antennae not simulated';

    function flagEnabled(location) {
        try {
            const search = location && typeof location.search === 'string' ? location.search : '';
            return new URLSearchParams(search).get('assets') === FLAG_VALUE;
        } catch (e) {
            return false;
        }
    }

    function nodeIndex(sceneRoot) {
        const map = new Map();
        if (!sceneRoot || typeof sceneRoot.traverse !== 'function') return map;
        sceneRoot.traverse((obj) => {
            if (obj && obj.name && !map.has(obj.name)) map.set(obj.name, obj);
        });
        return map;
    }

    function requiredFlyNodes() {
        const names = BODY_REQUIRED.slice();
        for (const prefix of LEG_PREFIXES) {
            for (const parts of Object.values(LEG_PARTS)) for (const p of parts) names.push(prefix + '_' + p);
        }
        return names;
    }

    // Reject a fly whose thorax is not roughly viewport-sized (wrong units or axes).
    function validateFly(THREE, map) {
        const missing = requiredFlyNodes().filter((name) => !map.has(name));
        if (missing.length) return 'missing nodes: ' + missing.slice(0, 6).join(', ') + (missing.length > 6 ? ', ...' : '');
        if (THREE && THREE.Box3) {
            const thorax = map.get('c_thorax');
            const box = new THREE.Box3().setFromObject(thorax);
            const size = new THREE.Vector3();
            box.getSize(size);
            const longest = Math.max(size.x, size.y, size.z);
            if (!(longest > 0.5 && longest < 10)) return 'thorax size ' + longest + ' is outside 0.5..10 viewport-mm';
        }
        return null;
    }

    // glTF colours are linear; the dashboard renderers keep three r128's default linear
    // output, so convert once to keep the Blender colours instead of a darker cast.
    function matchRendererEncoding(T, root, renderer) {
        if (!T || !root || (renderer && renderer.outputEncoding === T.sRGBEncoding)) return;
        const seen = new Set();
        root.traverse((o) => {
            for (const m of [].concat(o.material || [])) {
                if (!m || seen.has(m)) continue;
                seen.add(m);
                if (m.color && m.color.convertLinearToSRGB) m.color.convertLinearToSRGB();
                if (m.emissive && m.emissive.convertLinearToSRGB) m.emissive.convertLinearToSRGB();
            }
        });
    }

    // Lowest foot point of a fly GLB in its own root frame (root at identity): the
    // minimum y over every vertex of the <leg>_tarsus meshes, claws included.  Used
    // only to stand the static preview model on a surface; never applied to poses.
    function lowestFootY(T, root) {
        if (!T || !root) return null;
        const saved = [root.position.clone(), root.quaternion.clone(), root.scale.clone()];
        root.position.set(0, 0, 0); root.quaternion.set(0, 0, 0, 1); root.scale.set(1, 1, 1);
        root.updateMatrixWorld(true);
        let lowest = Infinity;
        const v = new T.Vector3();
        for (const leg of LEG_PREFIXES) {
            const tarsus = root.getObjectByName(leg + '_tarsus');
            if (!tarsus) continue;
            tarsus.traverse((m) => {
                const a = m.isMesh && m.geometry && m.geometry.attributes.position;
                if (!a) return;
                for (let k = 0; k < a.count; k += 1) lowest = Math.min(lowest, v.fromBufferAttribute(a, k).applyMatrix4(m.matrixWorld).y);
            });
        }
        root.position.copy(saved[0]); root.quaternion.copy(saved[1]); root.scale.copy(saved[2]);
        root.updateMatrixWorld(true);
        return Number.isFinite(lowest) ? lowest : null;
    }

    function create(env) {
        const state = {status: 'off', reason: '', viewport: null, replay: null, restore: null};
        const location = () => env.location || (typeof window !== 'undefined' ? window.location : null);
        const enabled = () => flagEnabled(location());
        const THREEOf = () => env.THREE || (typeof THREE !== 'undefined' ? THREE : null); // eslint-disable-line no-undef

        function ensureLoader() {
            const T = THREEOf();
            if (T && T.GLTFLoader) return Promise.resolve(T.GLTFLoader);
            const doc = env.document || (typeof document !== 'undefined' ? document : null);
            if (!doc || !doc.createElement) return Promise.reject(new Error('GLTFLoader unavailable'));
            return new Promise((resolve, reject) => {
                const script = doc.createElement('script');
                script.src = env.loaderUrl || LOADER_URL;
                script.onload = () => (THREEOf() && THREEOf().GLTFLoader
                    ? resolve(THREEOf().GLTFLoader) : reject(new Error('GLTFLoader did not register')));
                script.onerror = () => reject(new Error('could not load ' + script.src));
                (doc.head || doc.body).appendChild(script);
            });
        }

        function loadGlb(url) {
            if (env.loadGlb) return env.loadGlb(url);
            return ensureLoader().then((Loader) => new Promise((resolve, reject) => {
                new Loader().load(url, resolve, undefined, (err) => reject(err instanceof Error ? err : new Error(String(err && err.message || err))));
            }));
        }

        function setLabel(host, text, warn) {
            const doc = env.document || (typeof document !== 'undefined' ? document : null);
            if (!host || !doc || !doc.createElement) return null;
            const el = doc.createElement('div');
            el.className = 'hq-assets-label';
            el.style.cssText = 'margin-top:5px;padding:5px 9px;border:1px solid ' + (warn ? '#f59e0b' : '#334155')
                + ';border-radius:5px;background:rgba(15,23,42,.9);color:' + (warn ? '#fbbf24' : '#94a3b8')
                + ';font:10px monospace';
            el.textContent = text;
            host.appendChild(el);
            return el;
        }

        function fallback(where, err) {
            state.status = 'fallback';
            state.reason = (err && err.message) || String(err);
            if (where && where.label) where.label.textContent = 'HQ assets unavailable (' + state.reason
                + ') · procedural fly shown';
            return null;
        }

        // ------------------------------------------------------------ dashboard viewport
        function applyToViewport(vp, map) {
            const contacts = new Set(vp.contactSpheres || []);
            const hidden = [];
            vp.flyGroup.traverse((obj) => {
                if (!obj.isMesh || contacts.has(obj) || !obj.material) return;
                const mats = Array.isArray(obj.material) ? obj.material : [obj.material];
                for (const m of mats) if (m.visible !== false && !hidden.some(([x]) => x === m)) hidden.push([m, m.visible]);
            });
            const attached = [];
            const attach = (parent, name) => {
                const obj = map.get(name);
                if (!obj) return;
                if (obj.parent) obj.parent.remove(obj);
                obj.position.set(0, 0, 0);
                obj.quaternion.set(0, 0, 0, 1);
                obj.scale.set(1, 1, 1);
                obj.traverse((o) => { if (o.isMesh) { o.castShadow = true; o.receiveShadow = true; } });
                parent.add(obj);
                attached.push(obj);
            };
            for (const name of BODY_REQUIRED.concat(BODY_OPTIONAL)) attach(vp.thoraxMesh, name);
            for (const leg of vp.legs || []) {
                const prefix = VIEWPORT_LEGS[leg.name];
                if (!prefix) continue;
                for (const [group, parts] of Object.entries(LEG_PARTS)) {
                    for (const p of parts) attach(leg[group], prefix + '_' + p);
                }
            }
            for (const [m] of hidden) m.visible = false;
            return () => {
                for (const obj of attached) if (obj.parent) obj.parent.remove(obj);
                for (const [m, visible] of hidden) m.visible = visible;
            };
        }

        function syncArenaShell(vp, template) {
            const T = THREEOf();
            if (!T || !vp.scene || !template) return;
            if (vp.hqArenaShell) { vp.scene.remove(vp.hqArenaShell); vp.hqArenaShell = null; }
            const d = vp.assayGeometryDescriptor;
            if (!d || !d.available || !(d.width > 0) || !(d.depth > 0)) return;
            const round = d.outline && d.outline.shape === 'circle' && !(d.walls && d.walls.length);
            const source = template.get(round ? 'shell_round' : 'shell_rect');
            if (!source) return;
            const shell = source.clone();
            const group = new T.Group();
            group.name = 'hq-arena-shell';
            let cx = 0, cz = 0, sx = d.width, sz = d.depth;
            if (round && Number.isFinite(d.outline.radius) && d.outline.center && d.bounds) {
                cx = d.outline.center.x - (d.bounds.minX + d.bounds.maxX) / 2;
                cz = -(d.outline.center.y - (d.bounds.minY + d.bounds.maxY) / 2);
                sx = sz = 2 * d.outline.radius;
            }
            const margin = 2 + 0.06 * Math.max(sx, sz);
            shell.position.set(0, 0, 0);
            shell.quaternion.set(0, 0, 0, 1);
            shell.scale.set(1, 1, 1);
            group.add(shell);
            // Top of the shell stays below the floor (-0.02): never a wall, never a cue.
            group.position.set(cx, -0.06, cz);
            group.scale.set(sx + 2 * margin, 0.6, sz + 2 * margin);
            vp.scene.add(group);
            vp.hqArenaShell = group;
        }

        function attachViewport(vp) {
            if (!enabled() || !vp || !vp.flyGroup || !vp.thoraxMesh) return null;
            state.status = 'loading';
            const where = {label: setLabel(vp.statusPanel || vp.container, 'HQ assets loading…', true)};
            state.viewport = where;
            const flyPromise = loadGlb(env.flyUrl || FLY_URL).then((gltf) => {
                const map = nodeIndex(gltf && (gltf.scene || (gltf.scenes && gltf.scenes[0])));
                const invalid = validateFly(THREEOf(), map);
                if (invalid) throw new Error(invalid);
                matchRendererEncoding(THREEOf(), gltf.scene || gltf.scenes[0], vp.renderer);
                state.restore = applyToViewport(vp, map);
                state.status = 'hq';
                if (where.label) {
                    where.label.textContent = VIEWPORT_LABEL;
                    where.label.style.borderColor = '#334155';
                    where.label.style.color = '#94a3b8';
                }
                return 'hq';
            }).catch((err) => fallback(where, err));
            // The shell is independent: if it is missing the fly still upgrades, and vice versa.
            const shellPromise = loadGlb(env.arenaUrl || ARENA_URL).then((gltf) => {
                const map = nodeIndex(gltf && (gltf.scene || (gltf.scenes && gltf.scenes[0])));
                if (!map.has('shell_rect')) throw new Error('arena shell has no shell_rect');
                matchRendererEncoding(THREEOf(), gltf.scene || gltf.scenes[0], vp.renderer);
                const original = vp.updateAssayGeometry;
                if (typeof original === 'function') {
                    vp.updateAssayGeometry = function () {
                        const changed = original.apply(this, arguments);
                        if (changed) syncArenaShell(this, map);
                        return changed;
                    };
                }
                syncArenaShell(vp, map);
                return 'shell';
            }).catch(() => null);
            return Promise.all([flyPromise, shellPromise]).then(([fly]) => fly);
        }

        // ------------------------------------------------------------ replay page
        // FlyGym body names in the recorded skeleton -> which HQ mesh spans which edge.
        function replayEdges(segments) {
            const idx = new Map(segments.map((name, i) => [String(name).split('/').pop(), i]));
            const edges = [];
            for (const leg of LEG_PREFIXES) {
                const chain = [['coxa', leg + '_coxa', leg + '_trochanterfemur'],
                    ['trochanterfemur', leg + '_trochanterfemur', leg + '_tibia'],
                    ['tibia', leg + '_tibia', leg + '_tarsus1'],
                    ['tarsus', leg + '_tarsus1', [5, 4, 3, 2].map((k) => leg + '_tarsus' + k).find((n) => idx.has(n))]];
                for (const [part, from, to] of chain) {
                    if (idx.has(from) && to && idx.has(to)) edges.push({mesh: leg + '_' + part, from: idx.get(from), to: idx.get(to)});
                }
            }
            const first = (...names) => { for (const n of names) if (idx.has(n)) return idx.get(n); return -1; };
            const thorax = first('c_thorax', 'Thorax');
            // FlyGym 2 records no c_head body; its eye bodies sit on the head joint.
            const head = first('c_head', 'Head', 'l_eye', 'r_eye');
            const femurs = LEG_PREFIXES.map((leg) => [first(leg + '_trochanterfemur'), first(leg + '_tibia')])
                .filter(([a, b]) => a >= 0 && b >= 0);
            return {edges, thorax, head, femurs};
        }

        // Unit wrapper: the mesh's proximal end at the origin, its long axis on -Y, length 1.
        function edgeWrapper(T, obj) {
            obj.position.set(0, 0, 0);
            obj.quaternion.set(0, 0, 0, 1);
            obj.scale.set(1, 1, 1);
            obj.updateMatrixWorld(true);
            const box = new T.Box3().setFromObject(obj);
            const length = Math.max(1e-6, box.max.y - box.min.y);
            const inner = new T.Group();
            obj.position.set(0, -box.max.y, 0);
            inner.add(obj);
            inner.scale.set(1, 1 / length, 1);
            const outer = new T.Group();
            outer.add(inner);
            return outer;
        }

        function placeEdge(T, wrapper, p0, p1, radial) {
            const a = new T.Vector3(p0[0], p0[1], p0[2]);
            const dir = new T.Vector3(p1[0], p1[1], p1[2]).sub(a);
            const length = dir.length();
            wrapper.visible = length > 1e-6;
            if (!wrapper.visible) return;
            wrapper.position.copy(a);
            wrapper.quaternion.setFromUnitVectors(new T.Vector3(0, -1, 0), dir.normalize());
            wrapper.scale.set(radial, length, radial);
        }

        function buildReplayRig(T, map, skeleton) {
            const {edges, thorax, head, femurs} = replayEdges(skeleton.segments || []);
            const rig = new T.Group();
            rig.name = 'hq-replay-rig';
            // The replay scene draws only unlit lines and points, so the opt-in rig brings
            // its own neutral lights; they light nothing else.
            const ambient = new T.AmbientLight(0xffffff, 0.65);
            const key = new T.DirectionalLight(0xffffff, 0.9);
            key.position.set(-20, -20, 40);
            const back = new T.DirectionalLight(0xfde68a, 0.35);
            back.position.set(20, 20, 20);
            const wrappers = edges.map((e) => {
                const src = map.get(e.mesh);
                const w = src ? edgeWrapper(T, src.clone()) : null;
                if (w) rig.add(w);
                return w;
            });
            let body = null;
            if (thorax >= 0) {
                body = new T.Group();
                for (const name of BODY_REQUIRED.concat(BODY_OPTIONAL)) {
                    const src = map.get(name);
                    if (src) { const c = src.clone(); c.position.set(0, 0, 0); c.quaternion.set(0, 0, 0, 1); c.scale.set(1, 1, 1); body.add(c); }
                }
                rig.add(body);
            }
            rig.add(ambient, key, back);
            return {rig, edges, wrappers, body, thorax, head, femurs};
        }

        // Size the illustrative body from the recording itself: median femur length over
        // the viewport femur (2.2 viewport-mm).  0.3 when no femur is recorded.
        const VIEWPORT_FEMUR = 2.2;
        const VIEWPORT_HEAD = [0, 0.25, 1.75];   // head centre in the thorax frame (build_fly.py)
        function replayScale(pos, femurs) {
            const lengths = (femurs || []).map(([a, b]) => Math.hypot(pos[3 * b] - pos[3 * a],
                pos[3 * b + 1] - pos[3 * a + 1], pos[3 * b + 2] - pos[3 * a + 2]))
                .filter((v) => Number.isFinite(v) && v > 0).sort((x, y) => x - y);
            if (!lengths.length) return 0.3;
            return Math.min(1, Math.max(0.05, lengths[Math.floor(lengths.length / 2)] / VIEWPORT_FEMUR));
        }

        function poseReplayRig(T, r, frame) {
            const pos = frame.pos || [];
            const at = (i) => new T.Vector3(pos[3 * i], pos[3 * i + 1], pos[3 * i + 2]);
            const s = replayScale(pos, r.femurs);
            r.edges.forEach((e, i) => {
                const w = r.wrappers[i];
                if (w) placeEdge(T, w, pos.slice(3 * e.from, 3 * e.from + 3), pos.slice(3 * e.to, 3 * e.to + 3), s);
            });
            if (r.body) {
                const c = at(r.thorax);
                const yawDir = () => new T.Vector3(Math.cos(frame.yaw || 0), Math.sin(frame.yaw || 0), 0);
                let fwd = r.head >= 0 ? at(r.head).sub(c) : yawDir();
                fwd.z = 0;                       // keep the drawn body level; no invented pitch/roll
                if (!(fwd.lengthSq() > 1e-12)) fwd = yawDir();
                fwd.normalize();
                const up = new T.Vector3(0, 0, 1);  // replay world is MuJoCo z-up
                const right = new T.Vector3().crossVectors(up, fwd).normalize();
                const basis = new T.Matrix4().makeBasis(right, up, fwd);
                r.body.quaternion.setFromRotationMatrix(basis);
                r.body.scale.set(s, s, s);
                // Anchor the drawn head on the recorded head point when there is one.
                if (r.head >= 0) {
                    const offset = new T.Vector3().fromArray(VIEWPORT_HEAD).multiplyScalar(s).applyQuaternion(r.body.quaternion);
                    r.body.position.copy(at(r.head)).sub(offset);
                } else {
                    r.body.position.copy(c);
                }
            }
            return s;
        }

        function attachReplay(ctx) {
            if (!enabled() || !ctx || !ctx.scene || !ctx.state) return null;
            const T = ctx.THREE || THREEOf();
            if (!T) return null;
            state.status = 'loading';
            const doc = env.document || (typeof document !== 'undefined' ? document : null);
            const host = ctx.labelHost || (doc && doc.body);
            const where = {label: setLabel(host, 'HQ assets loading…', true)};
            if (where.label) where.label.style.cssText += ';position:fixed;left:8px;bottom:56px;z-index:5;max-width:60ch';
            state.replay = where;
            return loadGlb(env.flyUrl || FLY_URL).then((gltf) => {
                const map = nodeIndex(gltf && (gltf.scene || (gltf.scenes && gltf.scenes[0])));
                const invalid = validateFly(T, map);
                if (invalid) throw new Error(invalid);
                matchRendererEncoding(T, gltf.scene || gltf.scenes[0], ctx.renderer);
                let rec = null, frameIndex = -1, rig = null, shown = null;
                const label = (text, warn) => {
                    if (!where.label) return;
                    where.label.textContent = text;
                    where.label.style.color = warn ? '#fbbf24' : '#94a3b8';
                    where.label.style.borderColor = warn ? '#f59e0b' : '#334155';
                };
                // Posed inside the replay's own render call (scene.onBeforeRender), so the
                // surface always shows the frame the skeleton shows; a separate animation
                // loop could lag it or stall in a throttled tab.
                const sync = () => {
                    const s = ctx.state;
                    if (s.rec !== rec) {
                        if (rig) ctx.scene.remove(rig.rig);
                        rec = s.rec; frameIndex = -1;
                        rig = rec && rec.header && rec.header.skeleton ? buildReplayRig(T, map, rec.header.skeleton) : null;
                        if (rig) {
                            rig.rig.traverse((o) => { if (o.isMesh) o.frustumCulled = false; });
                            ctx.scene.add(rig.rig);
                        }
                    }
                    if (rig && rec && rec.frames && rec.frames[s.frame] && s.frame !== frameIndex) {
                        frameIndex = s.frame;
                        poseReplayRig(T, rig, rec.frames[s.frame]);
                        rig.rig.updateMatrixWorld(true);
                    }
                    // The success label appears only once a posed mesh is in the scene.
                    const meshes = rig && frameIndex >= 0 && rig.rig.parent === ctx.scene
                        ? rig.wrappers.filter((w) => w && w.visible).length + (rig.body ? 1 : 0) : 0;
                    const next = !rec ? 'waiting' : meshes > 0 ? 'hq' : 'no-match';
                    if (next !== shown) {
                        shown = next;
                        state.status = next === 'hq' ? 'hq' : next === 'waiting' ? 'ready' : 'fallback';
                        state.reason = next === 'no-match' ? 'recording skeleton has no matching FlyGym body names' : '';
                        if (next === 'hq') label(REPLAY_LABEL, false);
                        else if (next === 'waiting') label('HQ assets loaded \u00b7 open a recording to see the surface', true);
                        else label('HQ surface unavailable for this recording (' + state.reason + ') \u00b7 skeleton shown', true);
                    }
                    state.replayFrame = frameIndex;
                };
                const previous = ctx.scene.onBeforeRender;
                ctx.scene.onBeforeRender = function () {
                    sync();
                    if (typeof previous === 'function') return previous.apply(this, arguments);
                    return undefined;
                };
                sync();
                return state.status;
            }).catch((err) => fallback(where, err));
        }

        return {
            enabled, attachViewport, attachReplay,
            get status() { return state.status; },
            get reason() { return state.reason; },
            get replayFrame() { return state.replayFrame; },
            _replay: {replayEdges, buildReplayRig, poseReplayRig},
            restoreViewport() { if (state.restore) { state.restore(); state.restore = null; state.status = 'off'; } }
        };
    }

    return {
        create, flagEnabled, nodeIndex, validateFly, requiredFlyNodes, matchRendererEncoding, lowestFootY,
        VIEWPORT_LEGS, FLY_URL, ARENA_URL, LOADER_URL, VIEWPORT_LABEL, REPLAY_LABEL
    };
}));
