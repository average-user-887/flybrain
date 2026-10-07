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
