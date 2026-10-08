'use strict';
// Isolated transport fixtures only; no live daemon commands or simulation.
const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const http = require('node:http');
const app = fs.readFileSync(process.env.NF_ACK_APP_SOURCE || path.join(__dirname, '../web/app.js'), 'utf8');
const identity = {daemon_run_id: 'fixture-daemon', activation: 2, assay: 'fixture-assay', instance_id: 'fixture-instance'};
function receipt(id, action = 'set_paused') {
    return {status: 'ok', paused: true, command_id: 'fixture-1', client_command_id: id,
        ack: {action, applied: true, run_id: identity.daemon_run_id, identity: {...identity}, applied_step: 17}};
}
function harness(fetcher) {
    const warnings = [], timeouts = [];
    const ctx = vm.createContext({window: {}, document: {getElementById: () => null},
        console: {warn: (...args) => warnings.push(args)}, setTimeout, clearTimeout,
        AbortSignal: {timeout: (ms) => { timeouts.push(ms); return AbortSignal.timeout(ms); }}, fetch: fetcher});
    vm.runInContext(app.slice(app.indexOf('class DaemonBridgeClient {'), app.indexOf('const ASSAY_CONFIGS ='))
        + '\nthis.Bridge = DaemonBridgeClient;', ctx);
    const bridge = Object.create(ctx.Bridge.prototype);
    Object.assign(bridge, {connected: true, activeUrl: 'http://fixture', arena: {remotePacket: {identity}},
        pendingCommands: new Map(), commandAckCache: new Map(), hud: {}, commandAckTimeoutMs: 120000});
    return {bridge, warnings, timeouts};
}

test('delayed HTTP reply with early SSE ACK returns the applied result at the unchanged timeout', async (t) => {
    const commands = [], streams = [];
    let lateReply;
    const server = http.createServer(async (req, res) => {
        if (req.url === '/api/stream') {
            res.writeHead(200, {'Content-Type': 'text/event-stream'}); res.flushHeaders();
            streams.push(res); return;
        }
        if (req.url === '/api/command' && req.method === 'POST') {
            let body = ''; for await (const chunk of req) body += chunk;
            const cmd = JSON.parse(body); commands.push(cmd);
            const result = receipt(cmd.client_command_id, cmd.action);
            for (const stream of streams) stream.write('data: ' + JSON.stringify({command_acks: [result]}) + '\n\n');
            lateReply = setTimeout(() => { res.writeHead(200); res.end(JSON.stringify(result)); }, 2400);
            return;
        }
        res.writeHead(404); res.end();
    });
    await new Promise((resolve) => server.listen(0, '127.0.0.1', resolve));
    t.after(() => { clearTimeout(lateReply); server.closeAllConnections(); server.close(); });
    const h = harness(fetch); h.bridge.activeUrl = 'http://127.0.0.1:' + server.address().port;
    const stream = await fetch(h.bridge.activeUrl + '/api/stream');
    const reader = stream.body.getReader();
    const received = reader.read().then(({value}) => {
        const event = new TextDecoder().decode(value);
        h.bridge.resolveCommandAcks(JSON.parse(event.slice(6).trim()).command_acks);
    });
    const started = Date.now();
    const result = await h.bridge.sendCommand('set_paused', {paused: true}, null, {clientCommandId: 'fixture-request'});
    await received; await reader.cancel();
    assert.equal(result.status, 'ok'); assert.equal(result.ack.applied, true);
    assert.equal(result.client_command_id, 'fixture-request'); assert.equal(h.bridge.lastAck.applied_step, 17);
    assert.equal(commands.length, 1, 'the mutation is never resent');
    assert.deepEqual(h.timeouts, [2000]); assert.equal(h.warnings.length, 0);
    assert.ok(Date.now() - started >= 1900, 'the real HTTP timeout ran');
});

function unanswered(record, cache, mutate) {
    const requests = [];
    let h;
    h = harness(async (url, opts) => {
        requests.push({url, method: opts.method});
        if (opts.method === 'POST') {
            if (cache) h.bridge.resolveCommandAcks([cache]);
            mutate?.(h.bridge);
            const err = new Error('fixture timeout'); err.name = 'TimeoutError'; throw err;
        }
        return {ok: true, json: async () => record};
    });
    return {...h, requests};
}

test('read-only exact lookup recovers speed; an acknowledged rejection stays rejected', async () => {
    for (const rejected of [false, true]) {
        const r = receipt('fixture-request', 'set_speed'); r.sim_speed = 0.5;
        if (rejected) { r.status = 'error'; r.ack.applied = false; r.message = 'fixture refusal'; }
        const h = unanswered({state: 'acknowledged', daemon_run_id: identity.daemon_run_id, ack: r});
        const result = await h.bridge.sendCommand('set_speed', {speed: 0.5}, null, {clientCommandId: 'fixture-request'});
        assert.equal(result.status, rejected ? 'error' : 'ok'); assert.equal(result.ack.applied, !rejected);
        if (rejected) assert.equal(result.message, 'fixture refusal');
        assert.deepEqual(h.requests.map((r) => r.method), ['POST', 'GET']);
        assert.match(h.requests[1].url, /command_ack\?client_command_id=fixture-request$/);
        assert.deepEqual(h.timeouts, [2000, 2000]);
    }
});

for (const bad of ['missing', 'wrong-id', 'wrong-action', 'wrong-daemon', 'stale', 'pending', 'queued',
    'ok-unapplied', 'owner-changed', 'endpoint-changed', 'newer-activation', 'newer-applied-step']) {
    test('timeout recovery never accepts ' + bad + ' evidence', async () => {
        const r = receipt('fixture-request');
        let record = {state: 'acknowledged', daemon_run_id: identity.daemon_run_id, ack: r};
        let mutate;
        if (bad === 'missing') record = null;
        if (bad === 'wrong-id') r.client_command_id = 'other-request';
        if (bad === 'wrong-action') r.ack.action = 'set_speed';
        if (bad === 'wrong-daemon') r.ack.identity.daemon_run_id = 'other-daemon';
        if (bad === 'stale') r.ack.identity.activation = 1;
        if (bad === 'pending' || bad === 'queued') record.state = bad;
        if (bad === 'ok-unapplied') r.ack.applied = false;
        if (bad === 'owner-changed') mutate = (b) => { b.arena.remotePacket = {identity: {...identity, daemon_run_id: 'restarted'}}; };
        if (bad === 'endpoint-changed') mutate = (b) => { b.activeUrl = 'http://another-fixture'; };
        if (bad === 'newer-activation') mutate = (b) => { b.arena.remotePacket = {identity: {...identity, activation: 3}}; };
        if (bad === 'newer-applied-step') mutate = (b) => { b.lastAck = {...r.ack, applied_step: 18}; };
        const h = unanswered(record, null, mutate);
        const result = await h.bridge.sendCommand('set_paused', {paused: true}, null, {clientCommandId: 'fixture-request'});
        assert.equal(result.status, 'error'); assert.equal(result.request_unanswered, true);
        assert.equal(result.timed_out, true);
        assert.equal(h.bridge.lastAck?.applied_step, bad === 'newer-applied-step' ? 18 : undefined);
        assert.equal(h.requests.filter((r) => r.method === 'POST').length, 1);
        assert.ok(h.requests.every((r) => r.url.startsWith('http://fixture/')), 'never falls through to another endpoint');
    });
}
