'use strict';
const test=require('node:test');
const assert=require('node:assert/strict');
const fs=require('node:fs');
const vm=require('node:vm');
const app=fs.readFileSync(require('node:path').join(__dirname,'../web/app.js'),'utf8');
const html=fs.readFileSync(require('node:path').join(__dirname,'../web/index.html'),'utf8');
function slice(start,end){const a=app.indexOf(start),b=app.indexOf(end,a);assert.ok(a>=0&&b>a);return app.slice(a,b);}
function identity(backend='modular',activation=1){return {backend,activation,daemon_run_id:'daemon',run_id:'run'+activation,instance_id:'instance'+activation};}
function final(target,{status='ok',applied=true,activation=2,message='',command_id='command1'}={}){
 return {status,message,command_id,ack:{applied,action:'switch_backend',identity:identity(target,activation),applied_step:10507}};
}
function harness(reply){
 const elements={selectBackend:{value:'',disabled:true,style:{},listeners:{},addEventListener(k,fn){this.listeners[k]=fn;}},backendCommandStatus:{textContent:'',style:{}},arenaRunState:{textContent:''}};
 const timers=new Map();let counter=0;const calls=[];
 const c={window:{},document:{getElementById:id=>elements[id]||null,activeElement:elements.selectBackend},console:{warn(){},error(){}},AbortSignal:{timeout(){return {}; }},
  setTimeout(fn){const id=++counter;timers.set(id,fn);return id;},clearTimeout(id){timers.delete(id);},
  fetch:async(url,opts)=>{calls.push({url,body:JSON.parse(opts.body)});return {ok:true,status:200,json:async()=>typeof reply==='function'?reply():reply};}};
 vm.createContext(c);
 vm.runInContext(slice('const SELECTABLE_BACKENDS =','function identityRejection('),c);
 vm.runInContext(slice('class DaemonBridgeClient {','// =============================================================================\n// 7C.')+'\nwindow.Bridge=DaemonBridgeClient;',c);
 vm.runInContext('class FocusHUD {\n'+slice('    reconcileBackendSelector() {','    setupCatalogEvents() {')+slice('    setupControlBarEvents() {','    setupToolEvents() {')+'}\nwindow.HUD=FocusHUD;',c);
 const hud=new c.window.HUD(),bridge=Object.create(c.window.Bridge.prototype);
 const arena={remoteDriven:true,awaitingDaemon:false,remotePacket:{run_id:'daemon',identity:identity()}};
 Object.assign(bridge,{hud,arena,connected:true,activeUrl:'http://selected-daemon:8879',readOnly:false,switchPending:false,pendingCommands:new Map(),commandAckCache:new Map(),commandAckTimeoutMs:120000});
 Object.assign(hud,{arena,daemonBridge:bridge});hud.setupControlBarEvents();
 return {hud,bridge,arena,elements,calls,timers,state:c.window.neuroflyBackendSelectorState};
}
async function flush(){for(let i=0;i<8;i++)await Promise.resolve();}
const queued={status:'queued',applied:false,command_id:'command1'};

test('declared trained readout has a truthful option and accessible persistent status',()=>{
 assert.match(html,/value="connectome-with-trained-readout">Fixed graph \+ external trained readout \(no synaptic learning\)/);
 assert.match(html,/id="selectBackend"[^>]*disabled[^>]*aria-label="Controller backend"[^>]*aria-describedby="backendCommandStatus"/);
 assert.match(html,/id="backendCommandStatus"[^>]*role="status"[^>]*aria-live="polite"[^>]*aria-atomic="true"/);
 assert.doesNotMatch(app,/fetch\('\/api\/controller'/);
});
test('actual selector change callback displays GraphUnavailable refusal and actual identity',async()=>{
 const h=harness(final('modular',{status:'error',applied:false,activation:1,message:'GraphUnavailable: missing graph'}));
 h.elements.selectBackend.value='connectome-fixed';
 assert.equal(await h.elements.selectBackend.listeners.change({target:h.elements.selectBackend}),false);
 assert.equal(h.calls.length,1);assert.equal(h.calls[0].url,'http://selected-daemon:8879/api/command');
 assert.equal(h.calls[0].body.action,'switch_backend');
 assert.equal(h.elements.selectBackend.value,'modular');assert.equal(h.elements.selectBackend.disabled,false);
 assert.match(h.elements.backendCommandStatus.textContent,/refused.*GraphUnavailable/);
 h.hud.reconcileBackendSelector();h.arena.remotePacket={run_id:'daemon',identity:identity()};h.hud.reconcileBackendSelector();
 assert.match(h.elements.backendCommandStatus.textContent,/refused.*GraphUnavailable/);
});
test('successful final ACK reconciles focused selector before target telemetry arrives',async()=>{
 const h=harness(final('connectome-with-trained-readout'));
 assert.equal(await h.hud.setBackend('connectome-with-trained-readout'),true);
 assert.equal(h.elements.selectBackend.value,'connectome-with-trained-readout');
 assert.match(h.elements.backendCommandStatus.textContent,/Applied.*10507/);
 assert.equal(h.bridge.lastSwitchAck.identity.activation,2);
 h.hud.reconcileBackendSelector();assert.equal(h.elements.selectBackend.value,'connectome-with-trained-readout');
 h.arena.remotePacket={run_id:'daemon',identity:identity('modular',3)};h.hud.reconcileBackendSelector();assert.equal(h.elements.selectBackend.value,'modular');
});
for(const mode of ['offline','replay','recorded','waiting','noidentity','unknownbackend','badowner','local','readOnly','switchPending'])test('backend unavailable without mutation: '+mode,async()=>{
 const h=harness(final('connectome-fixed'));
 if(mode==='offline')h.bridge.connected=false;
 if(mode==='replay')h.bridge.replayMode=true;
 if(mode==='recorded')h.arena.remotePacket.timing={replay:true};
 if(mode==='waiting')h.arena.awaitingDaemon=true;
 if(mode==='noidentity')delete h.arena.remotePacket.identity;
 if(mode==='unknownbackend')h.arena.remotePacket.identity.backend='pretend';
 if(mode==='badowner')h.arena.remotePacket.identity.daemon_run_id='other-daemon';
 if(mode==='local')h.arena.remoteDriven=false;
 if(mode==='readOnly')h.bridge.readOnly=true;
 if(mode==='switchPending')h.bridge.switchPending=true;
 assert.equal(await h.hud.setBackend('connectome-fixed'),false);
 assert.equal(h.calls.length,0);assert.equal(h.elements.selectBackend.disabled,true);
 assert.match(h.elements.backendCommandStatus.textContent,/refused/);
});
test('queued HTTP and nonfinal streamed ACK cannot claim applied',async()=>{
 const h=harness(queued);let resolved=false;
 const promise=h.hud.setBackend('connectome-fixed').then(v=>{resolved=true;return v;});await flush();
 assert.equal(resolved,false);assert.match(h.elements.backendCommandStatus.textContent,/queued; not yet applied/);
 assert.equal(h.elements.selectBackend.value,'modular');assert.equal(h.elements.selectBackend.disabled,true);
 h.bridge.resolveCommandAcks([queued]);await flush();assert.equal(resolved,false);
 h.bridge.resolveCommandAcks([final('modular',{status:'error',applied:false,activation:1,message:'GraphUnavailable'})]);
 assert.equal(await promise,false);assert.match(h.elements.backendCommandStatus.textContent,/refused.*GraphUnavailable/);
 assert.equal(h.elements.selectBackend.value,'modular');assert.equal(h.elements.selectBackend.disabled,false);
});
test('SSE final ACK before HTTP queued response is consumed exactly once',async()=>{
 let h;h=harness(()=>{h.bridge.resolveCommandAcks([final('connectome-fixed')]);return queued;});
 assert.equal(await h.hud.setBackend('connectome-fixed'),true);
 assert.equal(h.bridge.pendingCommands.size,0);assert.equal(h.bridge.commandAckCache.size,0);
 assert.match(h.elements.backendCommandStatus.textContent,/Applied/);
});
test('timeout remains unknown and late final ACK changes status and identity',async()=>{
 const h=harness(queued);const promise=h.hud.setBackend('connectome-fixed');await flush();
 for(const fn of [...h.timers.values()])fn();
 assert.equal(await promise,false);assert.match(h.elements.backendCommandStatus.textContent,/Timed out.*outcome is unknown/);
 assert.equal(h.elements.selectBackend.value,'modular');assert.equal(h.elements.selectBackend.disabled,true);
 h.bridge.resolveCommandAcks([final('connectome-fixed')]);
 assert.match(h.elements.backendCommandStatus.textContent,/Applied/);assert.equal(h.elements.selectBackend.disabled,false);
 assert.equal(h.elements.selectBackend.value,'connectome-fixed');assert.equal(h.bridge.lastSwitchAck.identity.activation,2);
});
test('foreign final ACK does not resolve the current backend command',async()=>{
 const h=harness(queued);const promise=h.hud.setBackend('connectome-fixed');await flush();
 h.bridge.resolveCommandAcks([final('connectome-plastic',{command_id:'other'})]);
 assert.match(h.elements.backendCommandStatus.textContent,/queued; not yet applied/);
 h.bridge.resolveCommandAcks([final('connectome-fixed')]);assert.equal(await promise,true);
});
test('status ok without applied true and matching identity never claims success',async()=>{
 for(const reply of [{status:'ok'},final('modular',{applied:false}),final('connectome-plastic')]){
  const h=harness(reply);assert.equal(await h.hud.setBackend('connectome-fixed'),false);
  assert.doesNotMatch(h.elements.backendCommandStatus.textContent,/^Applied/);
 }
});
test('replay selector follows recording identity rather than a newer live ACK',()=>{
 const h=harness(queued);h.bridge.lastBackendAck=final('connectome-fixed').ack;h.bridge.replayMode=true;
 h.hud.reconcileBackendSelector();assert.equal(h.elements.selectBackend.value,'modular');assert.equal(h.elements.selectBackend.disabled,true);
});

test('applied ACK from another daemon cannot promote success or identity',async()=>{
 const reply=final('connectome-fixed');reply.ack.identity.daemon_run_id='other-daemon';
 const h=harness(reply);assert.equal(await h.hud.setBackend('connectome-fixed'),false);
 assert.equal(h.elements.selectBackend.value,'modular');assert.doesNotMatch(h.elements.backendCommandStatus.textContent,/^Applied/);
});
test('HTTP timeout reports unknown outcome without same-origin fallback',async()=>{
 const h=harness(()=>{const err=new Error('timeout');err.name='TimeoutError';throw err;});
 assert.equal(await h.hud.setBackend('connectome-fixed'),false);
 assert.match(h.elements.backendCommandStatus.textContent,/timed out.*outcome is unknown/);
 assert.equal(h.calls.length,1);assert.equal(h.elements.selectBackend.value,'modular');
});
