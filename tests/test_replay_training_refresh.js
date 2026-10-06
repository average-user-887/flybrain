'use strict';

const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');

const root = path.resolve(__dirname, '..');
const appSource = fs.readFileSync(path.join(root, 'web/app.js'), 'utf8');
const trainingSource = fs.readFileSync(path.join(root, 'web/training.js'), 'utf8');

function methodSource(name, nextName) {
    const start = appSource.indexOf(`    ${name}(`);
    const end = appSource.indexOf(`    ${nextName}(`, start + 1);
    assert.ok(start >= 0 && end > start, `${name} method source is present`);
    return appSource.slice(start, end);
}

test('replay entry and exit publish synchronous mode changes after changing state', () => {
    const enter = methodSource('enterReplay', 'resetReplayView');
    const exit = methodSource('exitReplay', 'markReadOnly');
    for (const [source, state] of [[enter, 'true'], [exit, 'false']]) {
        const assignment = source.indexOf(`this.replayMode = ${state}`);
        const notification = source.indexOf("window.dispatchEvent(new Event('neurofly-replay-mode-change'))");
        assert.ok(assignment >= 0 && notification > assignment);
    }
});

test('training controls refresh immediately and stay disabled for replay', () => {
    const listeners = new Map();
    const elements = new Map();
    const element = id => {
        if (!elements.has(id)) elements.set(id, {
            disabled: false, onclick: null, textContent: '', innerHTML: '', title: '',
            classList: {toggle() {}}, firstChild: {textContent: ''}, style: {},
            addEventListener() {}, setAttribute() {}, removeAttribute() {},
        });
        return elements.get(id);
    };
    const window = {
        app: {hud: {daemonBridge: {replayMode: false}}},
        addEventListener(name, callback) { listeners.set(name, callback); },
    };
    const context = {
        window,
        document: {getElementById: element},
        setInterval() {}, setTimeout() {}, clearTimeout() {},
        console,
    };
    vm.createContext(context);
    // Start from a writable live snapshot so exit behavior is independently visible.
    const source = trainingSource.replace(
        'let snapshot = null, writable = false,',
        'let snapshot = null, writable = true,');
    vm.runInContext(source, context);
    const notify = listeners.get('neurofly-replay-mode-change');
    assert.equal(typeof notify, 'function');

    const commandIds = ['trainingTeach', 'trainingReverse', 'trainingProbe', 'trainingFreeze', 'trainingSave'];
    notify();
    assert.ok(commandIds.every(id => element(id).disabled), 'unverified live owner cannot enable commands');

    window.app.hud.daemonBridge.replayMode = true;
    notify();
    assert.ok(commandIds.every(id => element(id).disabled), 'replay notification disables commands synchronously');
    assert.equal(window.neuroflyTrainingWriteAllowed({read_only:false, commands_require_token:false}), false,
        'a later writable status response remains read only while replay is active');

    window.app.hud.daemonBridge.replayMode = false;
    notify();
    assert.ok(commandIds.every(id => element(id).disabled), 'exit waits for verified live snapshot');
    assert.match(trainingSource, /if\(trainingReplayActive\(\)\)\{[\s\S]*Replay is read only/,
        'the command-time replay guard remains present');
});
