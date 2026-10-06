'use strict';

const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');

const app = fs.readFileSync(path.resolve(__dirname, '../web/app.js'), 'utf8');

function declaration(prefix) {
    const start = app.indexOf(prefix);
    assert.notEqual(start, -1, `${prefix} missing`);
    const body = app.indexOf('{', start);
    let depth = 0;
    for (let i = body; i < app.length; i += 1) {
        if (app[i] === '{') depth += 1;
        if (app[i] === '}' && --depth === 0) return app.slice(start, i + 1);
    }
    throw new Error(`${prefix} is incomplete`);
}

const context = {window: {}, document: {getElementById: () => null}};
vm.createContext(context);
vm.runInContext([
    declaration('function orbitFrameValues('),
    declaration('const DAEMON_FRAME_OFFSET ='),
    declaration('function daemonFrameOffset('),
    declaration('function arenaPointFor3D('),
    declaration('function assayGeometry3DDescriptor('),
    declaration('function disposeThreeTree('),
    declaration('class ArticulatedFly3DViewport '),
    'this.Viewport = ArticulatedFly3DViewport;'
].join('\n'), context);

function vector(x, y, z) {
    return {x, y, z, set(nx, ny, nz) { this.x = nx; this.y = ny; this.z = nz; }};
}

function viewport() {
    const value = Object.create(context.Viewport.prototype);
    value.cameraMode = 'orbit';
    value.camera = {position: vector(0, 25, 45)};
    value.controls = {target: vector(0, 2, 0), enabled: true};
    value.flyGroup = {position: vector(35, 2, -28), rotation: vector(0, 0, 0)};
    value.lastPose = {xMm: 35, yMm: 28};
    value.pendingOrbitFrame = false;
    value.pendingOrbitOffset = null;
    value.savedOrbitOffset = null;
    value.orbitFrameIdentity = 'old-instance';
    value.arena = {activeParadigmId: 'wind-tunnel'};
    value.legs = [];
    return value;
}

function viewportWithHeldTelemetry() {
    const value = viewport();
    value.initialized = true;
    value.controls.update = () => {};
    value.arena = {
        activeParadigmId: 'wind-tunnel',
        awaitingDaemon: true,
        width: 100,
        height: 100,
        worldBounds: {minX: 0, maxX: 100, minY: 0, maxY: 100},
        fly: {x: 85, y: 78, heading: 0},
        remotePacket: {
            type: 'telemetry',
            paradigm: 'wind-tunnel',
            world_bounds: [0, 0, 100, 100],
            scene: {geometry: {schema:'neurofly.assay-geometry.v1', coordinate_frame:'arena-mm',
                source:'python-arena', bounds:[0, 0, 100, 100],
                containment:{kind:'rectangle', bounds:[0, 0, 100, 100]}, walls:[],
                surfaces:[{shape:'rectangle', bounds:[0, 0, 100, 100]}]}},
            segment_id: 's1',
            identity: {run_id: 'r', instance_id: 'i', activation: 2},
            fly: {heading: 0}
        }
    };
    value.cacheIdentity = value.packetIdentity(value.arena.remotePacket);
    value.orbitFrameIdentity = value.cameraIdentity(value.arena.remotePacket);
    value.lastHeading = null;
    value.lastBodyZ = null;
    value.lastJointAngles = null;
    value.lastLegContacts = null;
    value.poseStatus = null;
    return value;
}

test('view entry frames the current fly once and preserves later manual orbit', () => {
    const value = viewport();
    value.requestOrbitFrame();
    assert.equal(value.applyPendingOrbitFrame(), true);
    assert.deepEqual({...value.controls.target}, {x:35, y:2, z:-28, set:value.controls.target.set});
    assert.deepEqual([value.camera.position.x, value.camera.position.y, value.camera.position.z], [35, 25, 17]);

    value.controls.target.set(38, 3, -25);
    value.camera.position.set(42, 20, 9);
    assert.equal(value.applyPendingOrbitFrame(), false);
    assert.deepEqual([value.controls.target.x, value.controls.target.y, value.controls.target.z], [38, 3, -25]);
    assert.deepEqual([value.camera.position.x, value.camera.position.y, value.camera.position.z], [42, 20, 9]);
});

test('identity change waits for a valid new pose before framing', () => {
    const value = viewport();
    assert.equal(value.syncCameraIdentity({type:'telemetry', identity:{run_id:'r', instance_id:'new', activation:2}}), true);
    value.resetPoseCache('new-instance');
    assert.equal(value.applyPendingOrbitFrame(), false);
    assert.equal(value.pendingOrbitFrame, true);
    value.lastPose = {xMm: -12, yMm: -8};
    value.flyGroup.position.set(-12, 2, 8);
    assert.equal(value.applyPendingOrbitFrame(), true);
    assert.deepEqual([value.controls.target.x, value.controls.target.y, value.controls.target.z], [-12, 2, 8]);
});

test('a new segment clears pose channels without disturbing manual Orbit framing', () => {
    const value = viewport();
    const packet = {type:'telemetry', segment_id:'s1', identity:{run_id:'r', instance_id:'i', activation:2}};
    assert.equal(value.syncCameraIdentity(packet), true);
    value.pendingOrbitFrame = false;
    value.resetPoseCache(value.packetIdentity({...packet, segment_id:'s2'}));
    assert.equal(value.syncCameraIdentity({...packet, segment_id:'s2'}), false);
    assert.equal(value.pendingOrbitFrame, false);
});

test('returning from chase restores the last orbit offset around the current fly', () => {
    const value = viewport();
    value.controls.target.set(35, 2, -28);
    value.camera.position.set(45, 22, 12);
    value.setCameraMode('chase');
    value.camera.position.set(2, 10, 4);
    value.flyGroup.position.set(-20, 3, 15);
    value.setCameraMode('orbit');
    assert.equal(value.applyPendingOrbitFrame(), true);
    assert.deepEqual([value.controls.target.x, value.controls.target.y, value.controls.target.z], [-20, 3, 15]);
    assert.deepEqual([value.camera.position.x, value.camera.position.y, value.camera.position.z], [-10, 23, 55]);
});

test('view re-entry waits through a held pose and frames the first fresh pose', () => {
    const value = viewportWithHeldTelemetry();
    value.requestOrbitFrame();
    value.updatePose();
    assert.equal(value.pendingOrbitFrame, true);
    assert.deepEqual([value.controls.target.x, value.controls.target.y, value.controls.target.z], [0, 2, 0]);

    value.arena.awaitingDaemon = false;
    value.arena.fly = {x: 38, y: 57, heading: 0};
    value.updatePose();
    assert.equal(value.pendingOrbitFrame, false);
    assert.deepEqual([value.controls.target.x, value.controls.target.y, value.controls.target.z], [-12, 1.7, -7]);
});

test('Chase return keeps its saved offset pending until a fresh pose', () => {
    const value = viewportWithHeldTelemetry();
    value.controls.target.set(35, 2, -28);
    value.camera.position.set(45, 22, 12);
    value.setCameraMode('chase');
    value.camera.position.set(2, 10, 4);
    value.setCameraMode('orbit');
    value.updatePose();
    assert.equal(value.pendingOrbitFrame, true);
    assert.deepEqual([value.controls.target.x, value.controls.target.y, value.controls.target.z], [35, 2, -28]);

    value.arena.awaitingDaemon = false;
    value.arena.fly = {x: 30, y: 40, heading: 0};
    value.updatePose();
    assert.equal(value.pendingOrbitFrame, false);
    assert.deepEqual([value.controls.target.x, value.controls.target.y, value.controls.target.z], [-20, 1.7, 10]);
    assert.deepEqual([value.camera.position.x, value.camera.position.y, value.camera.position.z], [-10, 21.7, 50]);
});
