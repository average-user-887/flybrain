'use strict';
// The embodied replay labels Reference fly (not connectome) recordings and only those.
const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const source = fs.readFileSync(path.resolve(__dirname, '..', 'web/embodied_replay.js'), 'utf8');

function referenceLabel() {
    const start = source.indexOf('    function referenceLabel('), end = source.indexOf('    function showFrame(', start);
    assert.ok(start >= 0 && end > start);
    const context = vm.createContext({});
    vm.runInContext(source.slice(start, end), context);
    return context.referenceLabel;
}

test('reference recordings are labelled, connectome and modular ones are not', () => {
    const label = referenceLabel();
    assert.match(label({provenance: {backend_id: 'reference-flygym'}}), /^Reference fly \(not connectome\) .* illustrative reference controller$/);
    assert.equal(label({provenance: {backend_id: 'reference-flygym', display_label: 'L'}}), 'L');
    assert.equal(label({provenance: {neural_backend: {graph_sha256: 'abc'}}}), null);
    assert.equal(label({provenance: {neural_backend: {controller_kind: 'modular-baseline'}}}), null);
    assert.equal(label({}), null);
});

test('run panel never prints a fabricated drum or duration value', () => {
    const start = source.indexOf('    function referenceLabel('), mid = source.indexOf('    function showFrame(', start);
    const dStart = source.indexOf('    function describeRun('), dEnd = source.indexOf('    async function load(', dStart);
    assert.ok(start >= 0 && dStart >= 0 && dEnd > dStart);
    const els = {};
    const el = () => ({textContent: '', hidden: false, innerHTML: '', children: [], append(...c) { this.children.push(...c); }});
    const context = vm.createContext({$: (id) => (els[id] = els[id] || el()), document: {createElement: el}, String, Number});
    vm.runInContext(source.slice(start, mid) + source.slice(dStart, dEnd), context);
    const rows = (header) => {
        context.describeRun(header);
        const out = {};
        const c = els.run.children;
        for (let i = 0; i < c.length; i += 2) out[c[i].textContent] = c[i + 1].textContent;
        els.run.children = [];
        return out;
    };
    const ref = rows({fps: 50, provenance: {backend_id: 'reference-flygym', config: {seed: 0, duration_s: 5}}});
    assert.equal(ref.Drum, 'not applicable (reference fly)');
    assert.equal(ref.Duration, '5 s');
    const missing = rows({fps: 50, provenance: {neural_backend: {graph_sha256: 'abc'}, config: {}}});
    assert.equal(missing.Drum, 'n/a (not recorded)');
    assert.equal(missing.Duration, '–');
    assert.equal(rows({fps: 50, provenance: {config: {world_angular_velocity_rad_s: 0.5, duration_s: 2}}}).Drum, '0.5 rad/s');
    for (const r of [ref, missing]) for (const v of Object.values(r)) assert.doesNotMatch(v, /undefined|NaN/);
});
