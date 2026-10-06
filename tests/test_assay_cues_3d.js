'use strict';
const test=require('node:test'),assert=require('node:assert/strict');
const fs=require('node:fs'),vm=require('node:vm');
const helper=require('../web/assay_cues_3d.js');
const app=fs.readFileSync(require('node:path').join(__dirname,'../web/app.js'),'utf8');
function declaration(prefix) {
 const start=app.indexOf(prefix),body=app.indexOf('{',start);let depth=0;
 for(let i=body;i<app.length;i++){if(app[i]==='{')depth++;if(app[i]==='}'&&!--depth)return app.slice(start,i+1);}
 throw Error(prefix);
}
const c={window:{},document:{getElementById:()=>null}};vm.createContext(c);
vm.runInContext(['class WallSegment ','class RectRegion ','class CircleRegion ','class UnionRegion ','class HoledRegion ','class ScientificBioArena ','const DAEMON_FRAME_OFFSET =','function daemonFrameOffset(','function arenaPointFor3D(','function assayGeometry3DDescriptor(','function identityRejection('].map(declaration).join('\n')+'\nthis.Arena=ScientificBioArena;this.geometry=assayGeometry3DDescriptor;',c);
const ids=['open-arena','t-maze','y-maze','heat-maze','buridan','visual-operant','wind-tunnel','looming-escape','optomotor','gap-crossing','circadian-dam','courtship','labyrinth','multisensory-sandbox'];
function arena(pid){const a=Object.create(c.Arena.prototype);Object.assign(a,{fly:{trail:[]},mb:{reset(){}},cx:{},currentWalls:[],collisionNormals:[],remotePacket:null,remoteDriven:false,awaitingDaemon:false});a.initParadigm(pid);if(pid==='open-arena'){a.foodItems=[{x:4,y:5}];a.alarms=[];a.predators=[];a.windVector=[0,0];}return a;}
const geometry={available:true,bounds:{minX:-100,maxX:100,minY:-50,maxY:50},width:200,depth:100};
test('all14 preview mappings are finite and disclose unavailable fields',()=>{
 for(const id of ids){const a=arena(id),d=helper.describe(a,c.geometry(a));assert.match(d.status,/standalone preview/);for(const q of d.cues){assert.ok(Number.isFinite(q.x)&&Number.isFinite(q.y),id);assert.ok(q.label);}assert.ok(d.cues.length||id==='gap-crossing'||d.unavailable.length,id);}
});
test('live cues use supplied packet coordinates and offset, not preview leftovers',()=>{
 const a=arena('open-arena');a.remoteDriven=true;a.remotePacket={paradigm:'open-arena',scene:{food:[[2,3]],hazards:[],predators:[]}};a.foodItems=[{x:999,y:999}];
 const d=helper.describe(a,geometry,[10,-5]);assert.equal(d.cues.length,1);assert.equal(d.cues[0].x,12);assert.equal(d.cues[0].y,-2);
 assert.ok(!d.unavailable.includes('Hazard / odor B positions'));
 delete a.remotePacket.scene.hazards;assert.ok(helper.describe(a,geometry).unavailable.includes('Hazard / odor B positions'));
});
test('mismatch, waiting, missing scene and malformed coordinates cannot borrow preview data',()=>{
 const a=arena('courtship');a.remoteDriven=true;a.remotePacket={paradigm:'t-maze',scene:{female_pos:[9,9]}};assert.equal(helper.describe(a,geometry).cues.length,0);
 a.remotePacket.paradigm='courtship';a.awaitingDaemon=true;assert.equal(helper.describe(a,geometry).cues.length,0);a.awaitingDaemon=false;a.remotePacket.scene={female_pos:[NaN,9]};assert.equal(helper.describe(a,geometry).cues.length,0);assert.ok(helper.describe(a,geometry).unavailable.includes('female position'));
});
test('live T arm annotation never fabricates cue coordinates',()=>{
 const a=arena('t-maze');a.remoteDriven=true;a.remotePacket={paradigm:'t-maze',scene:{cs_plus_arm:'arm_b',food:[],hazards:[],predators:[]}};
 const d=helper.describe(a,geometry);assert.equal(d.cues.length,1);assert.equal(d.cues[0].kind,'label');assert.match(d.cues[0].label,/shock coordinates unavailable/);
});
test('live no-default policy covers all14 assays',()=>{
 for(const id of ids){const a=arena(id);a.remoteDriven=true;a.remotePacket={paradigm:id,scene:{food:[],hazards:[],predators:[]}};const d=helper.describe(a,geometry);assert.equal(d.cues.length,0,id);assert.match(d.status,/packet/);}
});
test('zero contrast suppresses stripes; radius, angle and contrast remain explicit',()=>{
 const a=arena('buridan');a.remoteDriven=true;a.remotePacket={paradigm:'buridan',scene:{food:[],hazards:[],predators:[],landmarks:[[0,1],[2,3]],stripe_contrast:0}};assert.equal(helper.describe(a,geometry).cues.length,0);
 a.activeParadigmId='heat-maze';a.remotePacket={paradigm:'heat-maze',scene:{food:[],hazards:[],predators:[],refuge_pos:[1,2]}};assert.equal(helper.describe(a,geometry).cues.length,0);assert.ok(helper.describe(a,geometry).unavailable.includes('Cool refuge radius'));
});
test('canvas projection aligns XY with geometry bounds and clears prior owner pixels',()=>{
 const calls=[];const ctx=new Proxy({}, {get:(o,k)=>o[k]||(o[k]=(...args)=>calls.push([k,...args]))});
 helper.paint(ctx,{cues:[{kind:'marker',x:0,y:0,label:'Food',color:'#fff'}]},geometry,100);
 const ellipse=calls.find(v=>v[0]==='ellipse');assert.equal(ellipse[1],50);assert.equal(ellipse[2],50);assert.deepEqual(calls[0],['clearRect',0,0,100,100]);
 calls.length=0;helper.paint(ctx,{cues:[]},geometry,100);assert.deepEqual(calls,[['clearRect',0,0,100,100]]);
});
test('viewport reuses one plane/texture and clears/hides on switch and unavailable owner',()=>{
 const begin=app.indexOf('    updateAssayCues() {'),end=app.indexOf('\n    updateAssayGeometry()',begin);
 const allocations={plane:0,texture:0,paint:0};let descriptor={cues:[],status:'owner A'};
 const ctx={identityRejection:c.identityRejection,window:{NeuroFlyAssayCues3D:{describe:()=>descriptor,paint:()=>allocations.paint++}},document:{createElement:()=>({style:{},getContext:()=>({})})},daemonFrameOffset:p=>{assert.ok(p);return[0,0];},THREE:{DoubleSide:2,CanvasTexture:function(){allocations.texture++;},MeshBasicMaterial:function(){},PlaneGeometry:function(){allocations.plane++;},Mesh:function(){this.rotation={};this.position={};this.scale={set(){}};}}};vm.createContext(ctx);vm.runInContext('this.update=({'+app.slice(begin,end).trim()+'}).updateAssayCues;',ctx);
 const v={arena:{},assayGeometryDescriptor:geometry,scene:{add(){}},container:{appendChild(){}},packetIdentity:()=>v.owner,owner:'A'};
 ctx.update.call(v);ctx.update.call(v);assert.deepEqual(allocations,{plane:1,texture:1,paint:1});
 v.owner='B';descriptor={cues:[],status:'owner B'};ctx.update.call(v);assert.equal(allocations.paint,2);
 v.assayGeometryDescriptor={available:false};ctx.update.call(v);assert.equal(v.cuePlane.visible,false);
 v.assayGeometryDescriptor=geometry;v.owner='replay C';ctx.update.call(v);assert.equal(v.cuePlane.visible,true);assert.deepEqual(allocations,{plane:1,texture:1,paint:3});
});

test('marker projection has a fixed display bound with explicit omission disclosure',()=>{
 const a=arena('open-arena');a.remoteDriven=true;a.remotePacket={paradigm:'open-arena',scene:{food:Array.from({length:1000},(_,i)=>[i,0]),hazards:[],predators:[]}};
 const d=helper.describe(a,geometry);assert.equal(d.cues.length,256);assert.ok(d.unavailable.some(v=>v.includes('display limit256')));
});
test('sampled wind preserves vector direction and zero without inventing a spatial field',()=>{
 const a=arena('wind-tunnel');a.remoteDriven=true;a.remotePacket={paradigm:'wind-tunnel',scene:{food:[],hazards:[],predators:[]},sensory:{wind_x:0,wind_y:0}};
 let d=helper.describe(a,geometry);assert.equal(d.cues[0].kind,'arrow');assert.equal(d.cues[0].dx,0);assert.match(d.cues[0].label,/0.0 mm\/s/);
 a.remotePacket.sensory={wind_x:-3,wind_y:4};d=helper.describe(a,geometry);assert.equal(d.cues[0].dx,-3);assert.equal(d.cues[0].dy,4);assert.match(d.cues[0].label,/5.0 mm\/s/);
});

test('actual helper refuses absent/null/string/numeric remote operant polarity without invented colors',()=>{
 for(const bad of [undefined,null,'false','true',0,1]){
  const a=arena('visual-operant');a.remoteDriven=true;
  const scene={food:[],hazards:[],predators:[],drum_angle_deg:90};if(bad!==undefined)scene.invert_sectors=bad;
  a.remotePacket={paradigm:'visual-operant',scene};const before=JSON.stringify(a.remotePacket);
  const d=helper.describe(a,geometry);assert.ok(!d.cues.some(c=>c.kind==='ring'));assert.ok(d.unavailable.includes('operant sector polarity'));
  assert.equal(JSON.stringify(a.remotePacket),before);
 }
});
test('actual helper preserves both Boolean live/replay polarity and configured preview false',()=>{
 for(const replay of [false,true])for(const invert of [false,true]){
  const a=arena('visual-operant');a.remoteDriven=true;a.remotePacket={paradigm:'visual-operant',timing:{replay},scene:{food:[],hazards:[],predators:[],drum_angle_deg:90,invert_sectors:invert}};
  const d=helper.describe(a,geometry),ring=d.cues.find(c=>c.kind==='ring');assert.ok(ring);assert.equal(ring.invert,invert);assert.ok(!d.unavailable.includes('operant sector polarity'));
 }
 const d=helper.describe(arena('visual-operant'),geometry);assert.equal(d.cues.find(c=>c.kind==='ring').invert,false);
});
test('optomotor grating does not require unrelated operant polarity',()=>{
 const a=arena('optomotor');a.remoteDriven=true;a.remotePacket={paradigm:'optomotor',scene:{food:[],hazards:[],predators:[],drum_angle_deg:90},stimuli:{contrast:0}};
 const d=helper.describe(a,geometry),ring=d.cues.find(c=>c.kind==='ring');assert.ok(ring);assert.equal(ring.contrast,0);assert.equal(ring.operant,false);assert.ok(!d.unavailable.includes('operant sector polarity'));
});
test('optomotor 2D and 3D use exact authoritative phase and refuse unknown historical phase',()=>{
 const a=arena('optomotor');a.remoteDriven=true;
 const arcs=[],labels=[];const ctx={beginPath(){},moveTo(){},fill(){},arc(...args){arcs.push(args);},fillText(text){labels.push(text);}};
 a.worldToScreen=()=>({x:100,y:100});
 for(const angle of [undefined,null,'0',false]) {
  a.remotePacket={paradigm:'optomotor',scene:{food:[],hazards:[],predators:[],drum_angle_deg:angle},stimuli:{contrast:.9}};
  a.paradigmState.drumAngleDeg=77;arcs.length=0;labels.length=0;
  a.renderOptomotor(ctx);assert.equal(arcs.length,0);assert.ok(labels.includes('Grating phase unavailable'));
  const d=helper.describe(a,geometry);assert.ok(!d.cues.some(c=>c.kind==='ring'));assert.ok(d.unavailable.includes('drum angle'));
 }
 for(const angle of [0,359.5,.1]) {
  a.remotePacket.scene.drum_angle_deg=angle;arcs.length=0;
  a.renderOptomotor(ctx);assert.equal(arcs[0][3],angle*Math.PI/180);
  assert.equal(helper.describe(a,geometry).cues.find(c=>c.kind==='ring').angle,angle);
 }
 a.awaitingDaemon=true;arcs.length=0;a.renderOptomotor(ctx);assert.equal(arcs.length,0);
 a.awaitingDaemon=false;a.remotePacket.paradigm='t-maze';arcs.length=0;a.renderOptomotor(ctx);assert.equal(arcs.length,0);
 assert.doesNotMatch(app,/state\.drumAngleDeg=\(stimulus\.drum_velocity_deg_s\|\|0\)\*pkt\.trial_elapsed_s/);
});

test('same-assay pending and postACK retained old owner clear cues, matching new packet recovers',()=>{
 const a=arena('open-arena');a.remoteDriven=true;
 const ack={identity:{daemon_run_id:'daemon',run_id:'run-new',instance_id:'brain-new',activation:8}};
 a.remotePacket={type:'telemetry',run_id:'daemon',paradigm:'open-arena',identity:{run_id:'run-old',instance_id:'brain-old',activation:7},scene:{food:[[2,3]],hazards:[],predators:[]},sensory:{wind_x:0,wind_y:0}};
 let d=helper.describe(a,geometry,[0,0],{switchPending:true,ownerProblem:null});assert.equal(d.cues.length,0);assert.match(d.status,/switch pending/);
 const rejected=c.identityRejection(a.remotePacket,ack);assert.ok(rejected);
 d=helper.describe(a,geometry,[0,0],{switchPending:false,ownerProblem:rejected});assert.equal(d.cues.length,0);assert.match(d.status,/awaiting acknowledged owner/);
 a.remotePacket.identity={run_id:'run-new',instance_id:'brain-new',activation:8};a.remotePacket.observation=null;
 const accepted=c.identityRejection(a.remotePacket,ack);assert.equal(accepted,null);
 d=helper.describe(a,geometry,[0,0],{switchPending:false,ownerProblem:accepted});assert.ok(d.cues.some(q=>q.label==='Food / odor A'));assert.doesNotMatch(d.status,/awaiting acknowledged owner|switch pending/);
});
test('actual viewport forwards explicit pending state and existing ACK owner predicate',()=>{
 const begin=app.indexOf('    updateAssayCues() {'),end=app.indexOf('\n    updateAssayGeometry()',begin);let received;
 const ack={identity:{daemon_run_id:'daemon',run_id:'new',instance_id:'new-instance',activation:2}};
 const packet={run_id:'daemon',identity:{run_id:'old',instance_id:'old-instance',activation:1}};
 const bridge={switchPending:true,lastSwitchAck:ack};
 const ctx={window:{hud:{daemonBridge:bridge},NeuroFlyAssayCues3D:{describe(a,g,o,t){received=t;return {status:'unavailable',cues:[]};}}},identityRejection:c.identityRejection,daemonFrameOffset:()=>[0,0],document:{createElement:()=>({style:{}})}};
 vm.createContext(ctx);vm.runInContext('this.update=({'+app.slice(begin,end).trim()+'}).updateAssayCues;',ctx);
 const v={arena:{remotePacket:packet},assayGeometryDescriptor:{available:false},container:{appendChild(){}}};ctx.update.call(v);
 assert.equal(received.switchPending,true);assert.equal(received.ownerProblem,c.identityRejection(packet,ack));
 bridge.switchPending=false;packet.identity={run_id:'new',instance_id:'new-instance',activation:2};ctx.update.call(v);assert.equal(received.ownerProblem,null);assert.equal(received.switchPending,false);
});
test('current runtime Y arm annotations and labyrinth goal retain exact packet positions',()=>{
 const a=arena('y-maze');a.remoteDriven=true;a.remotePacket={paradigm:'y-maze',scene:{food:[],hazards:[],predators:[],arm_tips:[{name:'arm_2',position:[12.25,31.5]}]}};
 let d=helper.describe(a,geometry);assert.equal(d.cues[0].label,'arm_2');assert.equal(d.cues[0].x,12.25);assert.equal(d.cues[0].y,31.5);assert.ok(!d.unavailable.includes('arm-tip cue annotations'));
 a.activeParadigmId='labyrinth';a.remotePacket={paradigm:'labyrinth',scene:{food:[],hazards:[],predators:[],goal_pos:[20.25,10.5]}};
 d=helper.describe(a,geometry);assert.equal(d.cues[0].label,'Goal odor');assert.equal(d.cues[0].x,20.25);
});
test('sandbox cue positions use packet offset while spatial temperature remains unavailable',()=>{
 const a=arena('multisensory-sandbox');a.remoteDriven=true;a.remotePacket={paradigm:'multisensory-sandbox',sensory:{wind_x:0,wind_y:0},scene:{food:[],hazards:[],predators:[],food_pos:[45,45],repellent_pos:[-45,-45],pheromone_pos:[45,-45],hotspot_pos:[-40,40],cool_pos:[45,45],pillar_centers:[[35,35]]}};
 const d=helper.describe(a,geometry,[-80,-80]);const hot=d.cues.find(q=>q.label==='Hotspot location');assert.equal(hot.x,-120);assert.equal(hot.y,-40);
 assert.ok(d.cues.some(q=>q.label==='cVA pheromone'));assert.ok(d.cues.some(q=>q.label==='Visual pillar'));assert.ok(d.unavailable.includes('spatial temperature field'));
});
