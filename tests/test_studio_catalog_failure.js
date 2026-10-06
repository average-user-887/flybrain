'use strict';
const test = require('node:test'), assert = require('node:assert/strict');
const fs = require('node:fs'), path = require('node:path'), vm = require('node:vm');
const root = process.env.NEUROFLY_STUDIO_SOURCE_ROOT || path.resolve(__dirname, '..');
const source = fs.readFileSync(path.join(root, 'web/studio.js'), 'utf8');

test('catalog 404 stays visible through healthy queue refreshes instead of leaving blank Build', async () => {
    const elements = new Map();
    const element = id => {
        if (!elements.has(id)) elements.set(id, {hidden: false, textContent: '', innerHTML: '',
            value: '', addEventListener() {}, querySelectorAll: () => []});
        return elements.get(id);
    };
    let refresh;
    const context = {window: {}, requestAnimationFrame: () => 1, document: {getElementById: element, querySelectorAll: () => [],
        addEventListener() {}}, setInterval: fn => {refresh = fn; return 1;},
        fetch: async url => url.endsWith('/catalog')
            ? {ok: false, status: 404, json: async () => ({error: 'not found: CAPABILITY_MATRIX.md'})}
            : {ok: true, json: async () => ({queue: [], curated: [], worker: 'running'})}};
    vm.createContext(context); vm.runInContext(source, context);
    await new Promise(setImmediate);
    assert.equal(element('global-error').hidden, false);
    assert.match(element('global-error').textContent, /Could not load the paradigm catalog.*CAPABILITY_MATRIX/);
    assert.match(element('worker').textContent, /running/);
    refresh(); await new Promise(setImmediate);
    assert.equal(element('global-error').hidden, false);
    assert.match(element('global-error').textContent, /Could not load the paradigm catalog/);
});
