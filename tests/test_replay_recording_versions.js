'use strict';
const test=require('node:test');
const assert=require('node:assert/strict');
const fs=require('node:fs');
const path=require('node:path');
const vm=require('node:vm');
const {gzipSync}=require('node:zlib');
const {createHash,webcrypto}=require('node:crypto');
const {buildObservationDisplay}=require('../web/observation_display.js');
const fixtures=JSON.parse(fs.readFileSync(path.join(__dirname,'fixtures/observation_display_all14.json'))).fixtures;
const clone=x=>JSON.parse(JSON.stringify(x));
function harness(){
 const elements=new Map();
 const doc={readyState:'loading',addEventListener(){},getElementById(id){if(!elements.has(id))elements.set(id,{hidden:false,textContent:'',title:'',value:'0'});return elements.get(id);}};
 const c={window:{crypto:webcrypto,location:{search:''}},document:doc,Uint8Array,TextDecoder,Blob,Response,DecompressionStream,performance:{now:()=>0},console};
 vm.createContext(c);vm.runInContext(fs.readFileSync(path.join(__dirname,'../web/replay.js'),'utf8'),c);
 return {player:c.window.neuroflyReplay,parse:c.window.neuroflyRecording.parseRecording,elements};
}
function makeRecording(version,frames,projection){
 const frameLines=frames.map((f,i)=>JSON.stringify({...f,k:'f',i})+'\n').join('');
 const header={k:'header',format:'neurofly-run-recording',version,projection,provenance:{assay:frames[0].paradigm,backend:frames[0].identity?.backend||'modular',seed:0,params:{dt_s:.02}},channels:{}};
 const end={k:'end',frames:frames.length,events:0,frames_sha256:createHash('sha256').update(frameLines).digest('hex')};
 return {header,end,bytes:new Uint8Array(gzipSync(JSON.stringify(header)+'\n'+frameLines+JSON.stringify(end)+'\n'))};
}
function frameFromFixture(x,index=0){
 x=clone(x);const identity={...x.identity,daemon_run_id:'d0',run_id:'r'+index,instance_id:'b'+index,brain_id:'b'+index,activation:index};
 x.observation.identity=clone(identity);
 const segment='s'+index;
 x.observation.segment_id=segment;x.observation.config_id='segment:'+segment;x.observation.presentation_id=segment+':0';
 if(x.observation.override)x.observation.override.manifest_run_id=identity.run_id;
 // Preserve a historical terminal, even with receipt-like fields: the player must not certify it.
 x.observationPublication.last_terminal.recording_payload_sha256='a'.repeat(64);
 return {identity,brain_id:identity.brain_id,run_id:'r-daemon',segment_id:segment,paradigm:identity.assay,step:index+1,sim_time_s:(index+1)*.02,
  fly:{x:0,y:0},observation:x.observation,observation_publication:x.observationPublication,
  recording_context:{mode:'replay',identity_namespace:'recording-local/1',original_durable_evidence_verified:false},activity:null,spikes:null};
}
async function playerFor(version,frames){
 const h=harness(),made=makeRecording(version,frames,version===2?{identity_namespace:'recording-local/1'}:undefined);
 h.player.rec=await h.parse(made.bytes);h.player.name='bounded-fixture.nfrec';h.player.buildIdentity();
 h.player.times=frames.map(x=>x.sim_time_s);h.player.playhead=h.player.times[0];return h;
}
for(const x of fixtures)test('v2 typed provisional replay owner: '+x.identity.assay,async()=>{
 const frame=frameFromFixture(x);const before=JSON.stringify(frame);const h=await playerFor(2,[frame]);
 const pkt=clone(h.player.packetFor(0));
 assert.equal(pkt.identity.run_id,'r0');assert.equal(pkt.identity.instance_id,'b0');assert.equal(pkt.brain_id,'b0');
 assert.equal(pkt.run_id,pkt.identity.daemon_run_id);assert.deepEqual(pkt.observation.identity,pkt.identity);
 assert.equal(pkt.timing.replay,true);assert.equal(pkt.recording_context.original_durable_evidence_verified,false);
 assert.equal(pkt.observation_publication.last_terminal,null);
 const display=buildObservationDisplay({identity:pkt.identity,brainId:pkt.brain_id,observation:pkt.observation,observationPublication:pkt.observation_publication});
 assert.equal(display.live.state,'available',display.live.error);assert.equal(display.live.provisional,true);
 assert.equal(display.terminal.state,'unavailable');assert.deepEqual(pkt.observation.records,frame.observation.records);
 pkt.observation.records={};assert.equal(JSON.stringify(frame),before);
 assert.deepEqual(clone(h.player.rec.frames[0].observation.records),frame.observation.records);
 h.player.showBar();assert.match(h.elements.get('replayInfo').textContent,/REPLAY.*provisional.*saved terminal evidence unverified.*recorded frame digest OK/);
});
test('v2 transition changes only its recorded local owner and keeps source frames immutable',async()=>{
 const a=frameFromFixture(fixtures[0],0),b=frameFromFixture(fixtures[0],1);const h=await playerFor(2,[a,b]);
 const before=JSON.stringify(h.player.rec.frames);
 const first=clone(h.player.packetFor(0)),second=clone(h.player.packetFor(1));
 assert.equal(first.identity.run_id,'r0');assert.equal(second.identity.run_id,'r1');assert.equal(second.brain_id,'b1');
 assert.deepEqual(second.identity,second.observation.identity);assert.equal(second.path.length,1);
 assert.equal(JSON.stringify(h.player.rec.frames),before);
});
test('v2 missing/invalid owner is never repaired by header provenance or generic recording identity',async()=>{
 const frame=frameFromFixture(fixtures[0]);delete frame.identity;delete frame.brain_id;
 const h=await playerFor(2,[frame]);const pkt=clone(h.player.packetFor(0));
 assert.equal(pkt.identity,undefined);assert.equal(pkt.brain_id,undefined);
 assert.equal(buildObservationDisplay({identity:pkt.identity,brainId:pkt.brain_id,observation:pkt.observation}).state,'unavailable');
});
test('v1 compressed recordings remain playable with legacy identity/path behavior',async()=>{
 const frames=[{step:0,sim_time_s:0,paradigm:'t-maze',segment_id:'s0',fly:{x:0,y:0},metrics:{performance_index:0}},
  {step:1,sim_time_s:.02,paradigm:'t-maze',segment_id:'s0',fly:{x:1,y:0},metrics:{performance_index:0}}];
 const h=await playerFor(1,frames);const pkt=clone(h.player.packetFor(1));
 assert.match(pkt.identity.run_id,/^recording-/);assert.equal(pkt.brain_id,'recording');assert.equal(pkt.path.length,2);
 assert.equal(pkt.metrics.performance_index,0);assert.equal(pkt.timing.replay,true);assert.equal(h.player.rec.verified,true);
});
test('unsupported version or v2 namespace refuses before replaying',async()=>{
 const h=harness(),f=frameFromFixture(fixtures[0]);
 await assert.rejects(h.parse(makeRecording(3,[f]).bytes),/unsupported recording version/);
 await assert.rejects(h.parse(makeRecording(2,[f],{identity_namespace:'unknown'}).bytes),/unsupported recording identity namespace/);
});
test('frame digest corruption still refuses for both readable versions',async()=>{
 for(const version of [1,2]){
  const h=harness(),f=frameFromFixture(fixtures[0]),made=makeRecording(version,[f],{identity_namespace:'recording-local/1'});
  const bad=JSON.stringify(made.header)+'\n'+JSON.stringify({...f,k:'f',i:0,step:999})+'\n'+JSON.stringify(made.end)+'\n';
  await assert.rejects(h.parse(new Uint8Array(gzipSync(bad))),/frame digest mismatch/);
 }
});
test('actual Python v2 writer bytes pass parser and typed ownership without saved certification', {skip:!process.env.NEUROFLY_REPLAY_RECORDING}, async()=>{
 const h=harness();h.player.rec=await h.parse(new Uint8Array(fs.readFileSync(process.env.NEUROFLY_REPLAY_RECORDING)));
 assert.equal(h.player.rec.header.version,2);h.player.name='actual-cpu-fixture.nfrec';h.player.buildIdentity();
 const before=JSON.stringify(h.player.rec.frames),segments=new Set();
 for(let i=0;i<h.player.rec.frames.length;i++){
  const pkt=clone(h.player.packetFor(i));segments.add(pkt.segment_id);
  const current={...pkt.identity};if(!Object.hasOwn(current,'brain_id'))current.brain_id=pkt.brain_id;
  const display=buildObservationDisplay({identity:current,brainId:pkt.brain_id,observation:pkt.observation,observationPublication:pkt.observation_publication});
  assert.equal(display.live.state,'available',display.live.error);assert.equal(display.terminal.state,'unavailable');
  assert.equal(pkt.run_id,pkt.identity.daemon_run_id);assert.equal(pkt.recording_context.original_durable_evidence_verified,false);
 }
 assert.ok(segments.size>1,'scheduled reset must exercise a segment boundary');assert.equal(JSON.stringify(h.player.rec.frames),before);
});
