'use strict';

const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');

const root = path.resolve(__dirname, '..');
const app = fs.readFileSync(path.join(root, 'web/app.js'), 'utf8');
const html = fs.readFileSync(path.join(root, 'web/index.html'), 'utf8');

function declaration(name) {
    const start = app.indexOf(`function ${name}(`);
    assert.notEqual(start, -1, `${name} missing`);
    const body = app.indexOf('{', start);
    let depth = 0;
    for (let i = body; i < app.length; i += 1) {
        if (app[i] === '{') depth += 1;
        if (app[i] === '}' && --depth === 0) return app.slice(start, i + 1);
    }
    throw new Error(`${name} is incomplete`);
}

const context = {};
vm.createContext(context);
vm.runInContext([
    declaration('validateRequestedSpeed'),
    declaration('requestedSpeedText'),
    declaration('deliveryBuildState'),
].join('\n'), context);

test('custom requested speed preserves 7.5 exactly and enforces daemon bounds', () => {
    assert.deepEqual({...context.validateRequestedSpeed('7.5')}, {ok:true, value:7.5});
    assert.equal(context.requestedSpeedText(7.5), '7.5x');
    for (const bad of ['', 'NaN', 'Infinity', '0.09', '100.1']) {
        const result = context.validateRequestedSpeed(bad);
        assert.equal(result.ok, false, bad);
        assert.match(result.message, /speed|Speed/);
    }
    assert.equal(context.validateRequestedSpeed('0.1').value, 0.1);
    assert.equal(context.validateRequestedSpeed('100').value, 100);
});

test('page/daemon handshake distinguishes stale and static-source pages', () => {
    assert.deepEqual({...context.deliveryBuildState('abc', {web_build:'abc'})},
                     {page:'abc', daemon:'abc', stale:false});
    assert.equal(context.deliveryBuildState('abc', {web_build:'def'}).stale, true);
    assert.deepEqual({...context.deliveryBuildState('__NEUROFLY_WEB_BUILD__', {web_build:'def'})},
                     {page:'local source', daemon:'def', stale:false});
});

test('dashboard exposes preset, custom, delivery and device controls', () => {
    assert.match(html, /<option value="15">15x<\/option>/);
    for (const id of ['customSpeed', 'btnApplyCustomSpeed', 'speedFeedback', 'deliveryBadge',
                      'deliveryBanner', 'btnReloadBuild', 'identDevice']) {
        assert.match(html, new RegExp(`id="${id}"`), id);
    }
});
