'use strict';
// Characterises (does not fix) the dashboard 3D viewport's left/right and heading
// mapping against recorded FlyGym body frames.  The fix is a decision for the
// orchestrator: see docs/RENDERER_HANDEDNESS_20261008.md.  If the viewport mapping is
// changed on purpose, update these expectations together with that document.
const test = require('node:test');
const assert = require('node:assert/strict');
const path = require('node:path');
const fs = require('node:fs');
const check = require('../scripts/check_viewport_handedness.js');

const THREE = check.loadThree();
const rec = JSON.parse(fs.readFileSync(path.join(__dirname, 'fixtures/viewport_handedness_frames.json'), 'utf8'));
const app = fs.readFileSync(path.join(__dirname, '../web/app.js'), 'utf8');

test('recorded FlyGym frames: lf/lm/lh and l_* parts are on the anatomical left at every heading', () => {
    const r = check.analyse(THREE, rec).recorded;
    assert.ok(r.frames >= 45);
    assert.equal(r.lfLeft, r.frames);
    assert.equal(r.rightLegsRight, r.frames);
    assert.equal(r.pairedPartsOk, r.frames);
    assert.equal(r.yawAgrees, r.frames);       // recorded yaw = thorax->head direction
});

test('viewport as built: the drawn fly is a mirror image of the recorded one', () => {
    // The mapping under test is the unchanged updatePose code.
    assert.match(app, /this\.flyGroup\.rotation\.set\(0, -this\.lastHeading \+ Math\.PI \/ 2, 0\);/);
    assert.match(app, /\{ name: 'L1', side: -1, zOffset: 0\.8,/);
    for (const deg of [0, 30, 90, 180, 270]) {
        const v = check.viewportSides(THREE, deg * Math.PI / 180);
        // Head direction is the heading reflected about the arena x axis: cos(2h).
        assert.ok(Math.abs(v.headDot - Math.cos(2 * deg * Math.PI / 180)) < 1e-9, `${deg}`);
        // Relative to the head it draws, the L legs are always on the drawn fly's right.
        assert.ok(v.legSideOfDrawnHead.L1 < 0 && v.legSideOfDrawnHead.R1 > 0, `${deg}`);
    }
});
