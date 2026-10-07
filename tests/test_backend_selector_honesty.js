'use strict';
// P1: the controller selector and identity banner reflect the daemon's ACTUAL backend.
const test=require('node:test');
const assert=require('node:assert/strict');
const fs=require('node:fs');
const vm=require('node:vm');
const path=require('node:path');
const app=fs.readFileSync(path.join(__dirname,'../web/app.js'),'utf8');
const html=fs.readFileSync(path.join(__dirname,'../web/index.html'),'utf8');
function slice(start,end){const a=app.indexOf(start),b=app.indexOf(end,a);assert.ok(a>=0&&b>a);return app.slice(a,b);}
function identity(backend){return {backend,activation:1,daemon_run_id:'daemon',run_id:'run1',instance_id:'instance1'};}
function ctx(){const c={window:{}};vm.createContext(c);vm.runInContext(slice('const SELECTABLE_BACKENDS =','function identityRejection('),c);return c;}
function live(backend){return {arena:{remoteDriven:true,awaitingDaemon:false,remotePacket:{run_id:'daemon',identity:identity(backend)}},
 bridge:{connected:true,activeUrl:'http://127.0.0.1:8769',readOnly:false,switchPending:false}};}

test('a connected daemon shows its actual backend, connectome or modular',()=>{
 const state=ctx().window.neuroflyBackendSelectorState;
 for(const backend of ['connectome-fixed','modular']){const {arena,bridge}=live(backend);
  const s=state(arena,bridge);assert.equal(s.backend,backend);assert.equal(s.allowed,true);}
});
for(const mode of ['disconnected','frozen','local'])test('no controller shown as selected when '+mode,()=>{
 const state=ctx().window.neuroflyBackendSelectorState;const {arena,bridge}=live('connectome-fixed');
 if(mode==='disconnected')bridge.connected=false;
 if(mode==='frozen')arena.awaitingDaemon=true;
 if(mode==='local')arena.remoteDriven=false;
 const s=state(arena,bridge);
 assert.equal(s.backend,'');assert.equal(s.allowed,false);assert.match(s.placeholder,/Not connected/);
});
test('modular is labelled a hand-built engineered preview, connectome-fixed is listed first and the reference fly last',()=>{
 assert.match(html,/<option value="modular">Modular: hand-built engineered preview \(not the connectome\)<\/option>/);
 const sel=html.slice(html.indexOf('id="selectBackend"'),html.indexOf('</select>',html.indexOf('id="selectBackend"')));
 const values=[...sel.matchAll(/<option value="([^"]*)"/g)].map(m=>m[1]);
 assert.deepEqual(values,['','connectome-fixed','connectome-plastic','connectome-with-trained-readout','modular','reference-flygym']);
 assert.match(sel,/<option value="reference-flygym">Reference fly \(not connectome\) — illustrative reference controller<\/option>/);
 assert.doesNotMatch(sel,/Legacy Heuristic/);
});

function banner(launch,pkt){
 const els={identityBanner:{textContent:'',style:{}}};
 const c={window:{neuroflyLaunchBackend:launch},document:{getElementById:id=>els[id]||null}};
 vm.createContext(c);
 vm.runInContext(slice('const MOTOR_SOURCE_NOTES =','function graphPanelCapabilities(')+slice('function renderIdentity(pkt) {','// -----------------------------------------------------------------------------\n// Structured error')+'\nwindow.renderIdentity=renderIdentity;',c);
 c.window.renderIdentity(pkt);return els.identityBanner;
}
test('the modular fallback of a fresh default launch is announced in the identity banner',()=>{
 const fallback={backend:'modular',source:'default',fallback:true,reason:'no verified MaleCNS graph found'};
 const b=banner(fallback,{identity:identity('modular')});
 assert.match(b.textContent,/NOT THE CONNECTOME/);assert.equal(b.style.display,'block');
 // An explicit modular choice, or a later switch to the connectome, carries no fallback note.
 assert.doesNotMatch(banner({backend:'modular',source:'requested',fallback:false},{identity:identity('modular')}).textContent,/NOT THE CONNECTOME/);
 assert.doesNotMatch(banner(fallback,{identity:identity('connectome-fixed')}).textContent,/NOT THE CONNECTOME/);
 assert.doesNotMatch(banner(null,{identity:identity('modular')}).textContent,/NOT THE CONNECTOME/);
});
