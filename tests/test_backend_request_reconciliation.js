'use strict';
// CARD63 (8c3390d root browser run): a paused fixed->plastic switch showed
// "The command request timed out; its outcome is unknown." while the streamed
// identity became plastic, and nothing ever resolved it.  The HTTP request had no
// reply, so the dashboard had no daemon command id to match the acknowledgement.
// Every request now carries its own client_command_id; the outcome is matched only
// by that id (stream ack or GET /api/command_ack) and stays explicitly unknown
// until such evidence arrives.  The backend name in the stream is never evidence.
const test=require('node:test');
const assert=require('node:assert/strict');
const fs=require('node:fs');
const vm=require('node:vm');
const app=fs.readFileSync(require('node:path').join(__dirname,'../web/app.js'),'utf8');
function slice(start,end){const a=app.indexOf(start),b=app.indexOf(end,a);assert.ok(a>=0&&b>a);return app.slice(a,b);}
function identity(backend,activation,daemon='daemon'){return {backend,activation,daemon_run_id:daemon,run_id:'run'+activation,instance_id:'instance'+activation};}
function ack(target,{status='ok',applied=true,activation=3,message='',command_id='daemon-3',client_command_id=null,daemon='daemon'}={}){
 return {status,message,command_id,client_command_id,ack:{applied,action:'switch_backend',identity:identity(target,activation,daemon),applied_step:26}};
}
function timeoutError(){const e=new Error('timeout');e.name='TimeoutError';return e;}
function harness({post,lookup}){
 const elements={selectBackend:{value:'',disabled:true,style:{},listeners:{},addEventListener(k,fn){this.listeners[k]=fn;}},backendCommandStatus:{textContent:'',style:{}},arenaRunState:{textContent:''}};
 const timers=new Map();let counter=0;const posts=[],lookups=[];
 const c={window:{},document:{getElementById:id=>elements[id]||null,activeElement:elements.selectBackend},console:{warn(){},error(){}},AbortSignal:{timeout(){return {};}},
  setTimeout(fn){const id=++counter;timers.set(id,fn);return id;},clearTimeout(id){timers.delete(id);},
  fetch:async(url,opts)=>{
   if(url.includes('/api/command_ack')){const id=new URL(url).searchParams.get('client_command_id');lookups.push(id);
    const r=lookup?lookup(id):null;if(r instanceof Error)throw r;return {ok:true,status:200,json:async()=>r};}
   const body=JSON.parse(opts.body);posts.push(body);const r=post(body);if(r instanceof Error)throw r;
   return {ok:true,status:200,json:async()=>r};}};
 vm.createContext(c);
 vm.runInContext(slice('const SELECTABLE_BACKENDS =','function identityRejection('),c);
 vm.runInContext(slice('class DaemonBridgeClient {','// =============================================================================\n// 7C.')+'\nwindow.Bridge=DaemonBridgeClient;',c);
 vm.runInContext('class FocusHUD {\n'+slice('    reconcileBackendSelector() {','    setupCatalogEvents() {')+'}\nwindow.HUD=FocusHUD;',c);
 const hud=new c.window.HUD(),bridge=Object.create(c.window.Bridge.prototype);
 const arena={remoteDriven:true,awaitingDaemon:false,remotePacket:{run_id:'daemon',identity:identity('connectome-fixed',2)}};
 Object.assign(bridge,{hud,arena,connected:true,activeUrl:'http://isolated:8900',readOnly:false,switchPending:false,
  pendingCommands:new Map(),commandAckCache:new Map(),commandAckTimeoutMs:120000});
 Object.assign(hud,{arena,daemonBridge:bridge});
 const status=()=>elements.backendCommandStatus.textContent;
 // Run the reconciliation polls that are due (fake timers), letting each settle.
 const tick=async(n=1)=>{for(let i=0;i<n;i++){const due=[...timers.entries()];timers.clear();for(const [,fn] of due)await fn();await flush();}};
 return {hud,bridge,arena,elements,posts,lookups,timers,status,tick};
}
async function flush(){for(let i=0;i<10;i++)await Promise.resolve();}

test('timed-out request stays unknown when the stream merely shows the target backend',async()=>{
 const h=harness({post:()=>timeoutError(),lookup:()=>({daemon_run_id:'daemon',state:'unknown'})});
 assert.equal(await h.hud.setBackend('connectome-plastic'),false);
 const sent=h.posts[0];assert.equal(sent.action,'switch_backend');assert.match(sent.client_command_id,/^nf-[a-z0-9]+-[a-z0-9]+$/);
 assert.match(h.status(),/timed out; its outcome is unknown\. Waiting for the daemon's acknowledgement of this request/);
 h.arena.remotePacket={run_id:'daemon',identity:identity('connectome-plastic',3)};   // stream shows plastic
 h.hud.reconcileBackendSelector();await h.tick();
 assert.doesNotMatch(h.status(),/Applied/);assert.match(h.status(),/unknown|timed out/);
 assert.equal(h.elements.backendCommandStatus.style.color,'#fca5a5');
 assert.equal(h.lookups[0],sent.client_command_id);
});
test('late stream acknowledgement matched by request id resolves the timed-out request',async()=>{
 const h=harness({post:()=>timeoutError(),lookup:()=>({daemon_run_id:'daemon',state:'applying'})});
 await h.hud.setBackend('connectome-plastic');
 const id=h.posts[0].client_command_id;
 h.bridge.resolveCommandAcks([ack('connectome-plastic',{command_id:'other',client_command_id:'nf-other'})]);
 assert.doesNotMatch(h.status(),/Applied/);
 h.bridge.resolveCommandAcks([ack('connectome-plastic',{client_command_id:id})]);
 assert.match(h.status(),/^Applied controller connectome-plastic at step 26\./);
 assert.equal(h.bridge.lastSwitchAck.identity.activation,3);
 await h.tick(2);assert.match(h.status(),/^Applied/);       // reconciliation stopped
});
test('durable lookup resolves an applied outcome after the stream ack was missed',async()=>{
 let state='pending';
 const h=harness({post:()=>timeoutError(),lookup:id=>state==='pending'
  ?{daemon_run_id:'daemon',state:'pending',pending_control:{client_command_id:id,persistence_phase:'validating_target',elapsed_s:3.25}}
  :{daemon_run_id:'daemon',state:'acknowledged',ack:ack('connectome-plastic',{client_command_id:id})}});
 await h.hud.setBackend('connectome-plastic');await h.tick();
 assert.match(h.status(),/rebuilding \(validating target, 3\.3 s\); not yet applied \(request reply timed out/);
 assert.doesNotMatch(h.status(),/Applied/);
 state='done';await h.tick();
 assert.match(h.status(),/^Applied controller connectome-plastic at step 26\./);
});
test('a refusal found by lookup is shown as a refusal, not hidden',async()=>{
 const h=harness({post:()=>timeoutError(),lookup:id=>({daemon_run_id:'daemon',state:'acknowledged',
  ack:ack('connectome-fixed',{status:'error',applied:false,activation:2,client_command_id:id,message:'Target unavailable: GraphUnavailable: missing'})})});
 await h.hud.setBackend('connectome-plastic');await h.tick();
 assert.match(h.status(),/connectome-plastic refused: Target unavailable: GraphUnavailable/);
 assert.equal(h.elements.selectBackend.value,'connectome-fixed');
});
test('no evidence before the deadline leaves the outcome explicitly unknown',async()=>{
 const h=harness({post:()=>timeoutError(),lookup:()=>null});
 h.bridge.commandAckTimeoutMs=-1;
 await h.hud.setBackend('connectome-plastic');await h.tick();
 assert.match(h.status(),/no acknowledgement for it was found; its outcome is unknown/);
 assert.equal(h.timers.size,0);
});
test('a newer request is never overwritten by the older request\'s late acknowledgement',async()=>{
 const replies=[timeoutError(),ack('connectome-fixed',{activation:4,command_id:'daemon-4'})];
 const h=harness({post:body=>{const r=replies.shift();if(!(r instanceof Error))r.client_command_id=body.client_command_id;return r;},
  lookup:()=>({daemon_run_id:'daemon',state:'unknown'})});
 await h.hud.setBackend('connectome-plastic');const old=h.posts[0].client_command_id;
 const initialLookups=h.lookups.length; // shared recovery may do one immediate read-only lookup
 assert.equal(await h.hud.setBackend('connectome-fixed'),true);
 assert.match(h.status(),/^Applied controller connectome-fixed/);
 h.bridge.resolveCommandAcks([ack('connectome-plastic',{activation:3,client_command_id:old,command_id:'daemon-3'})]);
 await h.tick(2);
 assert.match(h.status(),/^Applied controller connectome-fixed/);
 assert.equal(h.bridge.lastSwitchAck.identity.activation,4);
 assert.equal(h.lookups.length,initialLookups);            // superseded request is no longer polled
});
test('a late HTTP reply for an older activation never replaces a newer acknowledgement',async()=>{
 const h=harness({post:()=>ack('connectome-plastic',{activation:3})});
 h.bridge.lastSwitchAck=ack('connectome-fixed',{activation:5}).ack;h.bridge.lastBackendAck=h.bridge.lastSwitchAck;
 await h.bridge.sendCommand('switch_backend',{backend:'connectome-plastic'});
 assert.equal(h.bridge.lastSwitchAck.identity.activation,5);assert.equal(h.bridge.lastBackendAck.identity.activation,5);
});
test('a daemon restart mid-switch is reported as unknown, never as applied',async()=>{
 const h=harness({post:()=>timeoutError(),lookup:()=>({daemon_run_id:'restarted',state:'unknown'})});
 await h.hud.setBackend('connectome-plastic');
 h.arena.remotePacket={run_id:'restarted',identity:identity('connectome-plastic',1,'restarted')};
 h.hud.reconcileBackendSelector();
 assert.match(h.status(),/outcome unknown\. The daemon process changed before this request was acknowledged; the new process reports controller connectome-plastic, which is not an acknowledgement/);
 assert.doesNotMatch(h.status(),/Applied/);
 await h.tick(2);assert.match(h.status(),/outcome unknown/);
 // and via the lookup when no new frame has arrived yet
 const g=harness({post:()=>timeoutError(),lookup:()=>({daemon_run_id:'restarted',state:'unknown'})});
 await g.hud.setBackend('connectome-plastic');await g.tick();
 assert.match(g.status(),/outcome unknown\. The daemon process changed/);
 assert.equal(g.timers.size,0);
});
test('queued request shows the daemon rebuild phase for that request only',async()=>{
 const h=harness({post:body=>({status:'queued',applied:false,command_id:'daemon-3',client_command_id:body.client_command_id})});
 const promise=h.hud.setBackend('connectome-plastic');await flush();
 h.arena.remotePacket={run_id:'daemon',identity:identity('connectome-fixed',2),
  observation_lifecycle:{pending_control:{command_id:'daemon-9',persistence_phase:'saving_prefix',elapsed_s:1}}};
 h.hud.reconcileBackendSelector();assert.match(h.status(),/queued; not yet applied/);
 h.arena.remotePacket.observation_lifecycle.pending_control={command_id:'daemon-3',persistence_phase:'preparing_transition',elapsed_s:7.5};
 h.hud.reconcileBackendSelector();
 assert.match(h.status(),/the daemon is rebuilding \(preparing transition, 7\.5 s\); not yet applied\./);
 h.bridge.resolveCommandAcks([ack('connectome-plastic',{command_id:'daemon-3'})]);
 assert.equal(await promise,true);assert.match(h.status(),/^Applied/);
});
