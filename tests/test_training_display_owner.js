'use strict';
const test=require('node:test'),assert=require('node:assert/strict'),fs=require('node:fs'),vm=require('node:vm');
const source=fs.readFileSync(require('node:path').join(__dirname,'../web/training.js'),'utf8');
const appSource=fs.readFileSync(require('node:path').join(__dirname,'../web/app.js'),'utf8');
const identityHelper=appSource.slice(appSource.indexOf('function identityRejection('),appSource.indexOf('window.neuroflyIdentityRejection = identityRejection;')+'window.neuroflyIdentityRejection = identityRejection;'.length);
function harness(){
 const elements={},listeners={},calls=[];let fetcher=async()=>({ok:true,json:async()=>data()}),exports=0;
 const identity={daemon_run_id:'daemon-a',activation:1,run_id:'run-a',instance_id:'instance-a',assay:'open-arena',backend:'modular'};
 const bridge={connected:true,activeUrl:'http://intended',replayMode:false,readOnly:false,lastOrderedPacket:{identity,brain_id:'brain-a',run_id:'daemon-a'},sendCommand:async()=>({status:'ok',ack:{applied:true}})};
 const hud={daemonBridge:bridge,arena:{activeParadigmId:'open-arena'}};
 function el(id){return elements[id]||=( {style:{},attrs:{},textContent:'',innerHTML:'',firstChild:{textContent:''},classList:{toggle(){}},addEventListener(){},setAttribute(k,v){this.attrs[k]=v;},removeAttribute(k){delete this.attrs[k];}} );}
 function data(){return {brain:{brain_id:'brain-a',paradigm:'open-arena',seed:1,steps:4,trials:0,learning_enabled:true,probe:{discrimination:0},weight_change_l2:0,history:[],weights:[]},telemetry:{identity:{...identity},brain_id:'brain-a'},brains:[],status:{stream:{},sim_speed:1}};}
 const c={window:{app:{hud},addEventListener(k,f){listeners[k]=f;}},document:{getElementById:el,createElement(){return {click(){exports++;}};}},AbortSignal:{timeout:()=>({})},fetch:async(...args)=>{calls.push(args);return fetcher(...args);},Blob,URL:{createObjectURL:()=>'',revokeObjectURL(){}},setTimeout(){},setInterval(){},Date,console};
 vm.createContext(c);vm.runInContext(identityHelper,c);vm.runInContext(source.replace(/\}\)\(\);\s*$/,'window.testOwner={refresh,reconcileTrainingOwner};})();'),c);
 return {bridge,hud,identity,el,data,listeners,calls,api:c.window.testOwner,fetch(fn){fetcher=fn;},exports:()=>exports};
}
function disabled(h){for(const x of ['Teach','Reverse','Probe','Freeze','Save','Export'])assert.equal(h.el('training'+x).disabled,true,x);}
test('replay immediately clears live weights/history/export and makes no observatory requests',async()=>{
 const h=harness();await h.api.refresh();assert.equal(h.el('trainingExport').disabled,false);
 h.bridge.replayMode=true;h.bridge.connected=false;h.bridge.lastOrderedPacket={identity:{assay:'t-maze'},brain_id:'recorded-brain:0'};h.listeners['neurofly-replay-mode-change']();disabled(h);
 assert.match(h.el('trainingTitle').textContent,/T-maze.*replay/);assert.match(h.el('trainingPhase').textContent,/not included/);assert.equal(h.el('trainingWeights').innerHTML,'');
 await h.el('trainingExport').onclick();await h.el('trainingTeach').onclick();await h.api.refresh();assert.equal(h.exports(),0);assert.equal(h.calls.length,1);
});
test('stale responses fenced on replay entry, endpoint change, assay change, activation and pending switch',async()=>{
 for(const change of [h=>{h.bridge.replayMode=true;h.listeners['neurofly-replay-mode-change']();},h=>h.bridge.activeUrl='http://other',h=>h.hud.arena.activeParadigmId='t-maze',h=>h.identity.activation++,h=>h.bridge.switchPending=true]){
  const h=harness();let resolve;const original=h.data();h.fetch(()=>new Promise(r=>resolve=r));const task=h.api.refresh();change(h);h.api.reconcileTrainingOwner();resolve({ok:true,json:async()=>original});await task;disabled(h);assert.doesNotMatch(h.el('trainingTitle').textContent,/retained brain/);
 }
});
test('new displayed assay disables old panel even before new observatory response; exit replay requires fresh snapshot',async()=>{
 const h=harness();await h.api.refresh();h.hud.arena.activeParadigmId='t-maze';h.api.reconcileTrainingOwner();disabled(h);
 h.hud.arena.activeParadigmId='open-arena';h.bridge.replayMode=true;h.listeners['neurofly-replay-mode-change']();h.bridge.replayMode=false;h.listeners['neurofly-replay-mode-change']();disabled(h);
 await h.api.refresh();assert.equal(h.el('trainingExport').disabled,false);assert.equal(h.el('trainingTeach').disabled,false);
});
test('self-consistent observatory response from another brain cannot enable controls or export',async()=>{
 const h=harness();h.fetch(async()=>{const d=h.data();d.brain.brain_id=d.telemetry.brain_id='other-brain';return {ok:true,json:async()=>d};});await h.api.refresh();disabled(h);assert.match(h.el('trainingMessage').textContent,/does not match/);
});
test('writes carry accepted owner and queued receipt waits for final applied ACK',async()=>{
 const h=harness();await h.api.refresh();let resolve,seen;h.bridge.sendCommand=(action,params,queued)=>{seen={action,params};queued();return new Promise(r=>resolve=r);};
 const task=h.el('trainingTeach').onclick();assert.match(h.el('trainingMessage').textContent,/awaiting final/);disabled(h);assert.equal(seen.params.expected_owner.brain_id,'brain-a');assert.equal(seen.params.expected_owner.activation,1);
 resolve({status:'error',ack:{applied:false},message:'stale owner'});await task;assert.match(h.el('trainingMessage').textContent,/stale owner/);assert.doesNotMatch(h.el('trainingMessage').textContent,/Applied|Recorded/);
 h.bridge.sendCommand=async()=>({status:'ok',ack:{applied:true,identity:{...h.identity}}});await h.el('trainingProbe').onclick();assert.match(h.el('trainingMessage').textContent,/Applied: probe brain/);
});
test('pending command completion cannot paint a new replay or displayed owner',async()=>{
 const h=harness();await h.api.refresh();let resolve;h.bridge.sendCommand=()=>new Promise(r=>resolve=r);const task=h.el('trainingSave').onclick();h.bridge.replayMode=true;h.listeners['neurofly-replay-mode-change']();resolve({status:'ok',ack:{applied:true}});await task;disabled(h);assert.doesNotMatch(h.el('trainingMessage').textContent,/Applied/);
});
test('actual reset handler blocks replay and preserves preview reset',()=>{
 const app=fs.readFileSync(require('node:path').join(__dirname,'../web/app.js'),'utf8');
 const fragment=app.match(/btnReset\.addEventListener\('click', ([^\n]+)\);/)[1];let clicks=0,handler;
 const ctx={btnReset:{addEventListener(_,f){handler=f;}},hud:{daemonBridge:{replayMode:true},arena:{resetTrial(){clicks++;}}}};vm.createContext(ctx);vm.runInContext(`(function(){btnReset.addEventListener('click',${fragment});}).call(hud)`,ctx);handler();assert.equal(clicks,0);ctx.hud.daemonBridge.replayMode=false;handler();assert.equal(clicks,1);
 assert.match(app,/reset\.disabled=!!this\.daemonBridge\?\.replayMode/);assert.match(app,/Exit replay to reset/);
});

test('switch/backend/reset ACK before telemetry refuses old reads and exports until accepted packet recovery',async()=>{
 for(const field of ['lastSwitchAck','lastBackendAck','lastAck']){
  const h=harness();await h.api.refresh();assert.equal(h.el('trainingExport').disabled,false);
  h.bridge.switchPending=true;h.api.reconcileTrainingOwner();disabled(h);
  h.bridge[field]={identity:{...h.identity,activation:2,run_id:'run-b',instance_id:'instance-b'},applied:true,action:field==='lastAck'?'reset_trial':'switch_backend'};
  h.bridge.switchPending=false;h.api.reconcileTrainingOwner();await h.api.refresh();disabled(h);await h.el('trainingExport').onclick();assert.equal(h.exports(),0);assert.equal(h.calls.length,1,'old telemetry cannot authorize a fresh old-owner read');
  Object.assign(h.identity,h.bridge[field].identity);h.api.reconcileTrainingOwner();await h.api.refresh();assert.equal(h.el('trainingExport').disabled,false);assert.equal(h.el('trainingTeach').disabled,false);h.el('trainingExport').onclick();assert.equal(h.exports(),1);
 }
});
test('delayed old observatory response after ACK cannot reacquire owner, and same-activation wrong instance remains fenced',async()=>{
 const h=harness();let resolve;const old=h.data();h.fetch(()=>new Promise(r=>resolve=r));const pending=h.api.refresh();
 h.bridge.lastSwitchAck={identity:{...h.identity,activation:2,run_id:'run-b',instance_id:'instance-b'}};
 resolve({ok:true,json:async()=>old});await pending;h.api.reconcileTrainingOwner();disabled(h);
 h.identity.activation=2;h.api.reconcileTrainingOwner();disabled(h);await h.api.refresh();assert.equal(h.calls.length,1);
 Object.assign(h.identity,h.bridge.lastSwitchAck.identity);h.fetch(async()=>({ok:true,json:async()=>h.data()}));await h.api.refresh();assert.equal(h.el('trainingExport').disabled,false);
});

test('command status is scoped to ownership generation; same-owner errors survive routine reconciliation',async()=>{
 const h=harness();await h.api.refresh();h.bridge.sendCommand=async()=>({status:'ok',ack:{applied:true,identity:{...h.identity}}});
 await h.el('trainingProbe').onclick();assert.match(h.el('trainingMessage').textContent,/Applied: probe brain/);
 h.bridge.replayMode=true;h.listeners['neurofly-replay-mode-change']();assert.doesNotMatch(h.el('trainingMessage').textContent,/Applied/);assert.match(h.el('trainingMessage').textContent,/Replay is read only/);
 h.el('trainingMessage').textContent='previous replay message';h.bridge.replayMode=false;h.listeners['neurofly-replay-mode-change']();assert.doesNotMatch(h.el('trainingMessage').textContent,/previous replay/);await h.api.refresh();
 await h.el('trainingProbe').onclick();assert.match(h.el('trainingMessage').textContent,/Applied/);h.identity.activation++;h.api.reconcileTrainingOwner();assert.doesNotMatch(h.el('trainingMessage').textContent,/Applied/);await h.api.refresh();
 h.el('trainingMessage').textContent='Current owner error';h.api.reconcileTrainingOwner();h.api.reconcileTrainingOwner();assert.equal(h.el('trainingMessage').textContent,'Current owner error');
});
