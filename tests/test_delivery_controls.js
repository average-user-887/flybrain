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

test('connected installed wheel without Git labels missing revision separately from delivery or liveness', () => {
    const elements = new Map();
    context.document = {
        querySelector: () => ({content:'956-installed-build'}),
        getElementById(id) {
            if (!elements.has(id)) elements.set(id, {textContent:'', title:'', style:{}});
            return elements.get(id);
        },
    };
    const start = app.indexOf('    renderDeliveryIdentity(status) {');
    const end = app.indexOf('    renderTiming(', start);
    assert.ok(start >= 0 && end > start);
    vm.runInContext(`renderer = {${app.slice(start, end)}}`, context);
    const bridge = {connected:true};
    const status = {delivery:{web_build:'956-installed-build', revision:null, source_dirty:false},
        compute:{device:'wgpu-amd'}};
    const before = JSON.stringify(status);
    const state = context.renderer.renderDeliveryIdentity.call(bridge, status);
    assert.equal(state.stale, false);
    assert.equal(elements.get('deliveryBadge').textContent, 'page 956-installed-build · daemon revision unavailable');
    assert.match(elements.get('deliveryBadge').title, /daemon web build 956-installed-build; daemon revision unavailable/);
    assert.equal(elements.get('deliveryBanner').style.display, 'none');
    assert.equal(bridge.connected, true);
    assert.equal(JSON.stringify(status), before);
    status.delivery.revision = 'known-commit';
    context.renderer.renderDeliveryIdentity.call(bridge, status);
    assert.equal(elements.get('deliveryBadge').textContent, 'page 956-installed-build · daemon revision known-commit');
});
