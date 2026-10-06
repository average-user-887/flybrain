'use strict';
const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const source = fs.readFileSync('web/embodied_replay.js', 'utf8');

function harness(frames, frame) {
    const elements = new Map();
    const $ = id => {
        if (!elements.has(id)) elements.set(id, {
            checked: false, value: '', textContent: '', innerHTML: '', style: {},
            append() {}, addEventListener(name, callback) { this[name] = callback; }
        });
        return elements.get(id);
    };
    const geometry = size => ({attributes: {position: {array: new Float32Array(size)}}, setDrawRange() {}});
    const context = vm.createContext({
        $, document: {createElement() {return {}; }},
        state: {rec: frames === null ? null : {frames, header: {skeleton: {parents: [-1]}, provenance: {fixture: 'recorded'}}},
            frame, clock: 1.237, last: 765, playing: false},
        joints: {geometry: geometry(3)}, bones: {geometry: geometry(0)}, trail: {geometry: geometry(9)}, trailCount: 0,
        controls: {target: {x: 2, y: -3, z: .5}},
        camera: {position: {x: -4, y: -9, z: 4}, zoom: 1.75, quaternion: {x: .1, y: .2, z: .3, w: .9}}
    });
    const start = source.indexOf('    function fmt('), end = source.indexOf('    function tick(', start);
    assert.ok(start >= 0 && end > start);
    vm.runInContext(source.slice(start, end), context);
    const handlerStart = source.indexOf("    $('follow').addEventListener('change'");
    const handlerEnd = source.indexOf("    $('seek').addEventListener", handlerStart);
    assert.ok(handlerStart >= 0 && handlerEnd > handlerStart, 'production Follow handler exists');
    vm.runInContext(source.slice(handlerStart, handlerEnd), context);
    return {context, $};
}
const frames = [0, 5, 10].map((x, i) => ({pos: [x, x + 2, .5], t: i / 10, yaw: .3, cmd: [0, 0], spikes: 0}));

for (const [name, index] of [['paused middle', 1], ['paused end', 2]]) {
    test(`Follow ON immediately recenters ${name} without advancing or changing recorded evidence`, () => {
        const {context: c, $} = harness(frames, index);
        vm.runInContext('showFrame(state.frame)', c); // current frame rendered with Follow OFF
        const prior = JSON.stringify({state: c.state, seek: $('seek').value, clock: $('clock').textContent});
        const orientation = JSON.stringify(c.camera.quaternion);
        const offset = [c.camera.position.x - c.controls.target.x, c.camera.position.y - c.controls.target.y];
        $('follow').checked = true;
        $('follow').change(); // no animation tick or seek event
        assert.equal(c.controls.target.x, frames[index].pos[0]);
        assert.equal(c.controls.target.y, frames[index].pos[1]);
        assert.deepEqual([c.camera.position.x - c.controls.target.x, c.camera.position.y - c.controls.target.y], offset);
        assert.equal(c.camera.position.z, 4); assert.equal(c.controls.target.z, .5);
        assert.equal(c.camera.zoom, 1.75); assert.equal(JSON.stringify(c.camera.quaternion), orientation);
        assert.equal(JSON.stringify({state: c.state, seek: $('seek').value, clock: $('clock').textContent}), prior);
        $('follow').checked = false;
        const cameraBefore = JSON.stringify(c.camera);
        $('follow').change();
        assert.equal(JSON.stringify(c.camera), cameraBefore);
    });
}
for (const rec of [null, []]) {
    test(`Follow ON with ${rec === null ? 'no' : 'empty'} recording is a safe no-op`, () => {
        const {context: c, $} = harness(rec, 0);
        const before = JSON.stringify({state: c.state, camera: c.camera, target: c.controls.target});
        $('follow').checked = true;
        assert.doesNotThrow(() => $('follow').change());
        assert.equal(JSON.stringify({state: c.state, camera: c.camera, target: c.controls.target}), before);
    });
}
