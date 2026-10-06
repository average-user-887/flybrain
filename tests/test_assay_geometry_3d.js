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
    declaration('class WallSegment '),
    declaration('class RectRegion '),
    declaration('class CircleRegion '),
    declaration('class UnionRegion '),
    declaration('class HoledRegion '),
    declaration('class ScientificBioArena '),
    declaration('const DAEMON_FRAME_OFFSET ='),
    declaration('function daemonFrameOffset('),
    declaration('function arenaPointFor3D('),
    declaration('function assayGeometry3DDescriptor('),
    declaration('function disposeThreeTree('),
    'this.Arena = ScientificBioArena;',
    'this.pointFor3D = arenaPointFor3D;',
    'this.descriptor = assayGeometry3DDescriptor;',
    'this.disposeTree = disposeThreeTree;'
].join('\n'), context);

function arenaFor(paradigm) {
    const arena = Object.create(context.Arena.prototype);
    arena.fly = {x: 0, y: 0, heading: 0, speed: 0, radius: 1.5, trail: []};
    arena.mb = {reset() {}};
    arena.cx = {};
    arena.currentWalls = [];
    arena.collisionNormals = [];
    arena.remotePacket = null;
    arena.remoteDriven = false;
    arena.awaitingDaemon = false;
    arena.initParadigm(paradigm);
    return arena;
}

const wallCounts = {
    'open-arena': 4,
    't-maze': 8,
    'y-maze': 9,
    'heat-maze': 32,
    'buridan': 32,
    'visual-operant': 0,
    'wind-tunnel': 4,
    'looming-escape': 0,
    'optomotor': 0,
    'gap-crossing': 4,
    'circadian-dam': 4,
    'courtship': 24,
    'labyrinth': 16,
    'multisensory-sandbox': 64
};

test('all 14 assays use their current wall set and finite configured bounds', () => {
    for (const [paradigm, expectedWalls] of Object.entries(wallCounts)) {
        const arena = arenaFor(paradigm);
        const geometry = context.descriptor(arena);
        assert.equal(geometry.available, true, paradigm);
        assert.equal(geometry.walls.length, expectedWalls, paradigm);
        assert.equal(geometry.kind, expectedWalls ? 'browser-preview-wall-segments' : 'browser-preview-bounds', paradigm);
        assert.match(geometry.status, /browser preview configuration/, paradigm);
        assert.ok(geometry.width > 0 && geometry.depth > 0, paradigm);
    }
});

test('fly and wall coordinates share the assay-centred transform', () => {
    const arena = arenaFor('wind-tunnel');
    const geometry = context.descriptor(arena);
    assert.deepEqual({...context.pointFor3D(0, 0, geometry.bounds)}, {x: -100, y: -30});
    assert.deepEqual({...context.pointFor3D(200, 60, geometry.bounds)}, {x: 100, y: 30});
    assert.deepEqual({...geometry.walls[0].p1}, {x: -100, y: -30});
    assert.deepEqual({...geometry.walls[0].p2}, {x: 100, y: -30});
});

test('geometry key is stable per assay geometry and changes with dimensions or walls', () => {
    const arena = arenaFor('open-arena');
    const first = context.descriptor(arena);
    assert.equal(context.descriptor(arena).key, first.key);
    arena.worldBounds = {...arena.worldBounds, maxX: arena.worldBounds.maxX + 1};
    const resized = context.descriptor(arena);
    assert.notEqual(resized.key, first.key);
    arena.currentWalls[0].p1[0] += 1;
    assert.notEqual(context.descriptor(arena).key, resized.key);
});

test('browser preview gap and DAM surfaces retain their configured dimensions', () => {
    const gap = arenaFor('gap-crossing');
    const narrow = context.descriptor(gap);
    assert.equal(narrow.surfaces.length, 2);
    assert.deepEqual({...narrow.surfaces[0]}, {shape:'rectangle', minX: 0, maxX: 45,
        minY: 7, maxY: 13, role:null});
    assert.equal(narrow.surfaces[1].minX, 48.5);
    gap.paradigmState.gapWidthMm = 5.0;
    const wide = context.descriptor(gap);
    assert.notEqual(wide.key, narrow.key);
    assert.equal(wide.surfaces[1].minX, 50);

    const dam = context.descriptor(arenaFor('circadian-dam'));
    assert.equal(dam.surfaces.length, 16);
    assert.deepEqual({...dam.surfaces[0]}, {shape:'rectangle', minX:0, maxX:65,
        minY:0, maxY:10, role:null});
    assert.deepEqual({...dam.surfaces[15]}, {shape:'rectangle', minX:0, maxX:65,
        minY:150, maxY:160, role:null});
});

function streamedPacket(paradigm, geometry) {
    return {paradigm, world_bounds: geometry.bounds,
        scene:{geometry:{schema:'neurofly.assay-geometry.v1', coordinate_frame:'arena-mm',
            source:'python-arena', containment:{kind:'rectangle', bounds:geometry.bounds},
            walls:[], surfaces:[{shape:'rectangle', bounds:geometry.bounds}], ...geometry}}};
}

test('current daemon geometry is authoritative and stale/missing geometry is unavailable', () => {
    const arena = arenaFor('open-arena');
    arena.remotePacket = streamedPacket('open-arena', {bounds:[0, 0, 300, 220]});
    const current = context.descriptor(arena);
    assert.deepEqual({...current.bounds}, {minX: -150, maxX: 150, minY: -110, maxY: 110});
    assert.equal(current.boundsSource, 'Python arena telemetry');
    assert.match(current.status, /Python arena telemetry/);
    arena.remotePacket = {paradigm: 't-maze', world_bounds: [0, 0, 1, 1]};
    const stale = context.descriptor(arena);
    assert.equal(stale.available, false);
    assert.match(stale.status, /awaiting current assay geometry telemetry/);
    arena.remotePacket = {paradigm: 'open-arena', world_bounds: [0, 0, 300, 220]};
    const missing = context.descriptor(arena);
    assert.equal(missing.available, false);
    assert.match(missing.status, /no valid geometry contract/);
});

test('streamed Buridan, labyrinth, and gap geometry overrides divergent browser previews', () => {
    const buridan = arenaFor('buridan');
    buridan.remotePacket = streamedPacket('buridan', {bounds:[10, 10, 110, 110],
        containment:{kind:'circle', center:[60, 60], radius:50},
        surfaces:[{shape:'circle', center:[60, 60], radius:50}], walls:[]});
    const platform = context.descriptor(buridan);
    assert.equal(platform.surfaces[0].radius, 50);
    assert.equal(platform.outline.radius, 50);
    assert.equal(platform.walls.length, 0);

    const labyrinth = arenaFor('labyrinth');
    labyrinth.remotePacket = streamedPacket('labyrinth', {bounds:[0, 0, 140, 100],
        walls:[{p1:[100, 70], p2:[100, 80]}]});
    const deadEnd = context.descriptor(labyrinth);
    assert.deepEqual({...deadEnd.walls[0].p1}, {x:30, y:20});
    assert.deepEqual({...deadEnd.walls[0].p2}, {x:30, y:30});

    const gap = arenaFor('gap-crossing');
    gap.remotePacket = streamedPacket('gap-crossing', {bounds:[5, 7.5, 95, 12.5],
        surfaces:[{shape:'rectangle', bounds:[0, 7.5, 45, 12.5]},
            {shape:'rectangle', bounds:[48.5, 7.5, 100, 12.5]}]});
    const tracks = context.descriptor(gap);
    assert.deepEqual({...tracks.surfaces[0]}, {shape:'rectangle', minX:0, minY:7.5,
        maxX:45, maxY:12.5, role:null});
    assert.deepEqual({...tracks.surfaces[1]}, {shape:'rectangle', minX:48.5, minY:7.5,
        maxX:100, maxY:12.5, role:null});
});

test('a live assay awaiting telemetry never substitutes browser preview walls', () => {
    for (const state of [{awaitingDaemon:true}, {remoteDriven:true}]) {
        const arena = arenaFor('labyrinth');
        Object.assign(arena, state, {remotePacket:null});
        const geometry = context.descriptor(arena);
        assert.equal(geometry.available, false);
        assert.match(geometry.status, /awaiting current assay geometry telemetry/);
    }
});

test('rapid assay switch-away and return recovers the original geometry identity', () => {
    const arena = arenaFor('t-maze');
    const first = context.descriptor(arena).key;
    arena.initParadigm('labyrinth');
    assert.notEqual(context.descriptor(arena).key, first);
    arena.initParadigm('t-maze');
    assert.equal(context.descriptor(arena).key, first);
});

test('unknown bounds are explicit and resource disposal covers geometry and material arrays', () => {
    const missing = context.descriptor({activeParadigmId: 'unknown', currentWalls: []});
    assert.equal(missing.available, false);
    assert.match(missing.status, /unavailable/);
    let geometryDisposed = 0;
    let materialsDisposed = 0;
    const objects = [
        {geometry: {dispose() { geometryDisposed += 1; }}, material: {dispose() { materialsDisposed += 1; }}},
        {material: [{dispose() { materialsDisposed += 1; }}, {dispose() { materialsDisposed += 1; }}]}
    ];
    context.disposeTree({traverse(callback) { objects.forEach(callback); }});
    assert.equal(geometryDisposed, 1);
    assert.equal(materialsDisposed, 3);
});
