'use strict';
// Local HTTP and real vendored GLTF parsing; no optional assets or simulation.
const test = require('node:test');
const assert = require('node:assert/strict');
const http = require('node:http');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const G = require('../web/asset_gallery.js');

test('gallery GET fallback for unsupported HEAD preserves real missing-file errors', async (t) => {
    const ctx = vm.createContext({TextDecoder, console, URL});
    vm.runInContext('var self = globalThis;', ctx);
    for (const name of ['three.min.js', 'GLTFLoader.js'])
        vm.runInContext(fs.readFileSync(path.join(__dirname, '../web/vendor', name), 'utf8'), ctx);
    const json = JSON.stringify({asset: {version: '2.0'}, scene: 0,
        scenes: [{nodes: [0]}], nodes: [{name: 'loaded-from-GET'}]});
    const chunk = Buffer.from(json.padEnd(Math.ceil(json.length / 4) * 4, ' '));
    const glb = Buffer.alloc(20 + chunk.length);
    glb.writeUInt32LE(0x46546c67, 0); glb.writeUInt32LE(2, 4); glb.writeUInt32LE(glb.length, 8);
    glb.writeUInt32LE(chunk.length, 12); glb.writeUInt32LE(0x4e4f534a, 16); chunk.copy(glb, 20);
    const requests = [];
    const server = http.createServer((req, res) => {
        requests.push([req.method, req.url]);
        const headStatus = req.url.startsWith('/405/') ? 405 : req.url.startsWith('/501/') ? 501 :
            req.url.endsWith('missing.glb') ? 404 : 200;
        if (req.method === 'HEAD') { res.writeHead(headStatus); return res.end(); }
        if (req.url.endsWith('missing.glb')) { res.writeHead(404); return res.end('missing'); }
        res.writeHead(200, {'Content-Type': 'model/gltf-binary'}); res.end(glb);
    });
    await new Promise((resolve) => server.listen(0, '127.0.0.1', resolve));
    t.after(() => new Promise((resolve) => { server.close(resolve); server.closeAllConnections(); }));
    const origin = 'http://127.0.0.1:' + server.address().port;
    // GET is the existing loader's responsibility after the HEAD probe; parse its real bytes.
    const loader = {load(url, done, progress, failed) {
        fetch(url).then((r) => {
            if (!r.ok) throw new Error('HTTP ' + r.status);
            return r.arrayBuffer();
        }).then((bytes) => new ctx.THREE.GLTFLoader().parse(bytes, '', done, failed), failed);
    }};
    for (const status of [200, 405, 501]) {
        const url = origin + '/' + status + '/present.glb';
        const loaded = [];
        const result = await G.loadGalleryGlb(url, loader, (g) => loaded.push(g));
        assert.equal(result.scene.getObjectByName('loaded-from-GET').name, 'loaded-from-GET');
        assert.deepEqual(loaded, [result]);
        assert.deepEqual(requests.splice(0), [['HEAD', '/' + status + '/present.glb'], ['GET', '/' + status + '/present.glb']]);
    }
    for (const status of [404, 405, 501]) {
        const url = origin + '/' + status + '/missing.glb';
        await assert.rejects(G.loadGalleryGlb(url, loader, () => assert.fail('missing asset loaded')), /HTTP 404/);
        assert.deepEqual(requests.splice(0), status === 404 ? [['HEAD', '/404/missing.glb']] :
            [['HEAD', '/' + status + '/missing.glb'], ['GET', '/' + status + '/missing.glb']]);
    }
});
