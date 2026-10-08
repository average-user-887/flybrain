#!/usr/bin/env node
'use strict';
/*
 * Coordinate check: which anatomical side does the dashboard 3D viewport draw the
 * L legs on, and which way does its head point, compared with recorded FlyGym body
 * frames?  Read-only; changes nothing.  See docs/RENDERER_HANDEDNESS_20261008.md.
 *
 *   node scripts/check_viewport_handedness.js path/to/body.nfbody
 *
 * Recorded frame (MuJoCo, z up, mm, right-handed): forward f = horizontal part of
 * (eye midpoint - thorax); anatomical left = up x f.  A body part is on the
 * anatomical left when (part - thorax) . left > 0.
 *
 * Viewport (ArticulatedFly3DViewport.updatePose, unchanged): arena (x, y) is drawn at
 * world (x, ., -y), the root is rotated about Y by -heading + pi/2, and the L coxae sit
 * at local x = -1.  Mapping world back to arena gives (x, -z).
 */
const fs = require('node:fs');
const path = require('node:path');
const zlib = require('node:zlib');
const vm = require('node:vm');

const LEGS = ['lf', 'lm', 'lh', 'rf', 'rm', 'rh'];
const VIEWPORT = {L1: 'lf', L2: 'lm', L3: 'lh', R1: 'rf', R2: 'rm', R3: 'rh'};
const VIEWPORT_COXA = {L1: [-1, 2.0, 0.8], L2: [-1, 2.0, 0], L3: [-1, 2.0, -0.8],
    R1: [1, 2.0, 0.8], R2: [1, 2.0, 0], R3: [1, 2.0, -0.8]};

function loadThree() {
    const ctx = vm.createContext({});
    vm.runInContext('var self = globalThis;', ctx);
    vm.runInContext(fs.readFileSync(path.join(__dirname, '../web/vendor/three.min.js'), 'utf8'), ctx);
    return ctx.THREE;
}

function readNfbody(file) {
    const lines = zlib.gunzipSync(fs.readFileSync(file)).toString('utf8').split('\n').filter(Boolean).map((l) => JSON.parse(l));
    const header = lines[0];
    return {segments: header.skeleton.segments, frames: lines.slice(1).filter((r) => Array.isArray(r.pos))};
}

// Rotate a recorded frame rigidly about the world z axis (a proper rotation keeps handedness).
function rotateFrame(frame, angle) {
    const c = Math.cos(angle), s = Math.sin(angle);
    const pos = frame.pos.slice();
    for (let i = 0; i < pos.length; i += 3) {
        const x = pos[i], y = pos[i + 1];
        pos[i] = c * x - s * y; pos[i + 1] = s * x + c * y;
    }
    return {pos, yaw: frame.yaw + angle};
}

function recordedSides(segments, frame) {
    const at = (name) => { const i = segments.indexOf(name); return i < 0 ? null : frame.pos.slice(3 * i, 3 * i + 3); };
    const t = at('c_thorax'), le = at('l_eye'), re = at('r_eye');
    const fx = (le[0] + re[0]) / 2 - t[0], fy = (le[1] + re[1]) / 2 - t[1];
    const n = Math.hypot(fx, fy);
    const f = [fx / n, fy / n];
    const left = [-f[1], f[0]];                       // up x f with up = +z
    const side = {};
    for (const leg of LEGS) { const p = at(leg + '_coxa'); side[leg] = (p[0] - t[0]) * left[0] + (p[1] - t[1]) * left[1]; }
    // FlyGym 2 puts both eye bodies on the head joint, so the paired non-leg parts
    // checked are the wings, halteres and pedicels.
    for (const part of ['wing', 'haltere', 'pedicel']) {
        for (const s of ['l', 'r']) {
            const p = at(s + '_' + part);
            if (p) side[s + '_' + part] = (p[0] - t[0]) * left[0] + (p[1] - t[1]) * left[1];
        }
    }
    const yawDot = f[0] * Math.cos(frame.yaw) + f[1] * Math.sin(frame.yaw);
    return {forward: f, left, side, yawDot};
}

function viewportSides(THREE, heading) {
    const g = new THREE.Group();
    g.rotation.set(0, -heading + Math.PI / 2, 0);   // exactly updatePose
    g.updateMatrixWorld(true);
    const arena = (v) => [v.x, -v.z];                 // world -> arena
    const fwd = arena(new THREE.Vector3(0, 0, 1).transformDirection(g.matrixWorld));
    const trueFwd = [Math.cos(heading), Math.sin(heading)];
    const trueLeft = [-Math.sin(heading), Math.cos(heading)];
    const legSide = {};
    for (const [name, p] of Object.entries(VIEWPORT_COXA)) {
        const a = arena(new THREE.Vector3(...p).applyMatrix4(g.matrixWorld));
        legSide[name] = a[0] * trueLeft[0] + a[1] * trueLeft[1];
    }
    // Side relative to the head the viewport actually draws.
    const drawnLeft = [-fwd[1], fwd[0]];
    const legSideOfDrawnHead = {};
    for (const [name, p] of Object.entries(VIEWPORT_COXA)) {
        const a = arena(new THREE.Vector3(...p).applyMatrix4(g.matrixWorld));
        legSideOfDrawnHead[name] = a[0] * drawnLeft[0] + a[1] * drawnLeft[1];
    }
    return {headDot: fwd[0] * trueFwd[0] + fwd[1] * trueFwd[1], drawnHead: fwd, legSide, legSideOfDrawnHead};
}

function analyse(THREE, rec, headings = [0, 30, 60, 90, 135, 180, 225, 270, 315].map((d) => d * Math.PI / 180)) {
    const recorded = {frames: 0, lfLeft: 0, rightLegsRight: 0, pairedPartsOk: 0, yawAgrees: 0};
    for (const frame of rec.frames) {
        for (const extra of headings) {
            const r = recordedSides(rec.segments, rotateFrame(frame, extra));
            recorded.frames += 1;
            if (['lf', 'lm', 'lh'].every((l) => r.side[l] > 0)) recorded.lfLeft += 1;
            if (['rf', 'rm', 'rh'].every((l) => r.side[l] < 0)) recorded.rightLegsRight += 1;
            if (['wing', 'haltere', 'pedicel'].every((p) => r.side['l_' + p] > 0 && r.side['r_' + p] < 0)) recorded.pairedPartsOk += 1;
            if (r.yawDot > 0.95) recorded.yawAgrees += 1;
        }
    }
    const viewport = headings.map((h) => {
        const v = viewportSides(THREE, h);
        return {headingDeg: Math.round(h * 180 / Math.PI), headDot: +v.headDot.toFixed(4),
            L1sideOfTrueHeading: +v.legSide.L1.toFixed(4), L1sideOfDrawnHead: +v.legSideOfDrawnHead.L1.toFixed(4),
            R1sideOfDrawnHead: +v.legSideOfDrawnHead.R1.toFixed(4)};
    });
    return {recorded, viewport};
}

if (require.main === module) {
    const file = process.argv[2];
    if (!file) { console.error('usage: check_viewport_handedness.js body.nfbody'); process.exit(2); }
    console.log(JSON.stringify(analyse(loadThree(), readNfbody(file)), null, 1));
}

module.exports = {analyse, loadThree, readNfbody, recordedSides, viewportSides, rotateFrame, LEGS, VIEWPORT};
