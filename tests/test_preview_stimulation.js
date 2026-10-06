'use strict';
const test=require('node:test'),assert=require('node:assert/strict'),fs=require('node:fs'),vm=require('node:vm'),path=require('node:path');
const app=fs.readFileSync(path.join(__dirname,'../web/app.js'),'utf8'),html=fs.readFileSync(path.join(__dirname,'../web/index.html'),'utf8');
function harness(){
 const timers=new Map(),callbacks=[],durations=[],elements={},bridge={connected:false,replayMode:false,switchPending:false};let next=0;
 const c={window:{hud:{daemonBridge:bridge}},document:{getElementById:id=>elements[id]||null},setTimeout(fn,ms){durations.push(ms);callbacks.push(fn);timers.set(++next,fn);return next;},clearTimeout(id){timers.delete(id);}};
 vm.createContext(c);
 for(const [start,end] of [['class ScientificBioArena','const EXPERIMENT_GUIDES'],['const EXPERIMENT_GUIDES','class DaemonBridgeClient'],['const ASSAY_CONFIGS =','const LESION_INFO ='],['class ScientificHUD','// 9. APPLICATION INITIALIZATION']])vm.runInContext(app.slice(app.indexOf(start),app.indexOf(end,app.indexOf(start))),c);
 vm.runInContext('globalThis.A=ScientificBioArena;globalThis.H=ScientificHUD;globalThis.guides=EXPERIMENT_GUIDES;globalThis.tools=ASSAY_CONFIGS;',c);
 const arena=Object.assign(Object.create(c.A.prototype),{windVector:[-10,2],paradigmState:{},activeParadigmId:'multisensory-sandbox',fly:{heading:1},cpg:{},dn:{dnp01Gf:0}});
 return {c,arena,bridge,timers,callbacks,durations,elements,el(id){return elements[id]={addEventListener(k,f){this[k]=f;}};},fire(){const f=[...timers.values()][0];assert.ok(f);f();}};
}
test('unsupported manual DN and false sensory claims have no remaining UI/state/action binding',()=>{
 for(const id of ['btnToggleManualControl','labelControlMode','sliderStimDna02','sliderStimThrust','sliderStimMdn','btnFlareHeat','btnFlareOdor'])assert.ok(!html.includes(id)&&!app.includes(id),id);
 for(const field of ['manualActive','overrideDna02','overrideThrust','overrideMdn','overrideGf'])assert.ok(!app.includes(field),field);
 const h=harness(),actions=h.c.tools['multisensory-sandbox'].actions;assert.equal(actions.length,1);assert.equal(actions[0].label,'Rotate preview fly');actions[0].handler(h.arena);assert.equal(h.arena.fly.heading,1.4);h.bridge.connected=true;actions[0].handler(h.arena);assert.equal(h.arena.fly.heading,1.4);
});
test('gust restores exact previous vector and repeated click restarts duration without losing it',()=>{
 const h=harness();h.arena.startPreviewGust();assert.deepEqual([...h.arena.windVector],[-35,0]);assert.equal(h.timers.size,1);
 h.arena.startPreviewGust();assert.equal(h.timers.size,1);h.callbacks[0]();assert.deepEqual([...h.arena.windVector],[-35,0]);h.fire();assert.deepEqual([...h.arena.windVector],[-10,2]);assert.equal(h.timers.size,0);
});
test('stale assay or reset owner cannot restore into new state',()=>{
 for(const change of [h=>h.arena.paradigmState={},h=>h.arena.activeParadigmId='open-arena']){const h=harness();h.arena.startPreviewGust();change(h);h.arena.windVector=[-4,5];h.fire();assert.deepEqual([...h.arena.windVector],[-4,5]);assert.equal(h.arena.previewGust,null);}
});
test('remote/replay/waiting/switch transition denies restoration and new gust',()=>{
 for(const change of [h=>h.bridge.connected=true,h=>h.bridge.replayMode=true,h=>h.bridge.switchPending=true,h=>h.arena.awaitingDaemon=true,h=>h.arena.remoteDriven=true]){const h=harness();h.arena.startPreviewGust();change(h);h.arena.windVector=[-4,5];h.fire();h.arena.startPreviewGust();assert.deepEqual([...h.arena.windVector],[-4,5]);assert.equal(h.timers.size,0);}
});
test('in-place or replacement wind mutation survives the old timeout',()=>{
 for(const change of [h=>h.arena.windVector[1]=7,h=>h.arena.windVector=[-8,4]]){const h=harness();h.arena.startPreviewGust();change(h);const intended=[...h.arena.windVector];h.fire();assert.deepEqual([...h.arena.windVector],intended);}
});
test('actual preview wind flow controls cancel gust and retain their new setting',()=>{
 const h=harness();h.arena.startPreviewGust();h.c.guides['multisensory-sandbox'].params.find(s=>s.key==='windMagnitude').apply(h.arena,20);assert.equal(h.timers.size,0);h.callbacks[0]();assert.deepEqual([...h.arena.windVector],[-20,0]);
 h.arena.startPreviewGust();h.c.tools['wind-tunnel'].sliders.find(s=>s.key==='windVelocity').apply(h.arena,12);assert.equal(h.timers.size,0);h.callbacks.at(-1)();assert.equal(h.arena.windVector[0],-12);
});
test('retained preview callbacks refuse live/replay programmatic clicks',()=>{
 const h=harness();for(const id of ['sliderStimCpg','valStimCpg','btnFlareGf','btnFlareWind'])h.el(id);
 h.c.H.prototype.setupNeuroStimControls.call({arena:h.arena});h.elements.sliderStimCpg.input({target:{value:'9'}});assert.equal(h.arena.cpg.baseFreq,9);h.elements.btnFlareGf.click();assert.equal(h.arena.dn.dnp01Gf,1);
 h.bridge.replayMode=true;h.elements.sliderStimCpg.input({target:{value:'12'}});h.elements.btnFlareGf.click();h.elements.btnFlareWind.click();assert.equal(h.arena.cpg.baseFreq,9);assert.equal(h.arena.dn.dnp01Gf,1);assert.equal(h.timers.size,0);
});

function crosswind(h){h.c.tools['wind-tunnel'].actions.find(a=>a.label==='Turbulent Crosswind Gust').handler(h.arena);}
test('actual crosswind action retains x, lasts2500ms and restores exact prior vector',()=>{
 const h=harness();h.arena.activeParadigmId='wind-tunnel';h.arena.windVector=[-10,-0];vm.runInContext('Math.random=()=>0.75',h.c);
 crosswind(h);assert.deepEqual([...h.arena.windVector],[-10,7.5]);assert.equal(h.durations.at(-1),2500);h.fire();assert.deepEqual([...h.arena.windVector],[-10,-0]);
});
test('repeated and mixed actual gust callbacks share original baseline and fence every old timer',()=>{
 for(const sequence of [['cross','cross'],['cross','axial','cross'],['axial','cross','axial']]){
  const h=harness();vm.runInContext('Math.random=()=>0.75',h.c);
  for(const kind of sequence){if(kind==='cross')crosswind(h);else h.arena.startPreviewGust();assert.equal(h.timers.size,1);}
  const intended=[...h.arena.windVector];for(const stale of h.callbacks.slice(0,-1))stale();assert.deepEqual([...h.arena.windVector],intended);
  h.fire();assert.deepEqual([...h.arena.windVector],[-10,2]);assert.equal(h.timers.size,0);
 }
});
test('actual crosswind action cannot run or restore after owner/live/replay transitions',()=>{
 for(const change of [h=>h.arena.paradigmState={},h=>h.arena.activeParadigmId='open-arena',h=>h.bridge.connected=true,h=>h.bridge.replayMode=true,h=>h.bridge.switchPending=true,h=>h.arena.remoteDriven=true,h=>h.arena.awaitingDaemon=true]){
  const h=harness();crosswind(h);change(h);h.arena.windVector=[-4,5];h.fire();assert.deepEqual([...h.arena.windVector],[-4,5]);assert.equal(h.arena.previewGust,null);
  if(!h.arena.isStandalonePreview()){crosswind(h);assert.equal(h.timers.size,0);assert.deepEqual([...h.arena.windVector],[-4,5]);}
 }
});
test('crosswind in-place edits survive old timer and become mixed gust baseline',()=>{
 const h=harness();crosswind(h);h.arena.windVector[1]=7;h.arena.startPreviewGust();h.callbacks[0]();h.fire();assert.deepEqual([...h.arena.windVector],[-10,7]);
});
test('actual crosswind callback then actual canvas wind placement survives stale timers',()=>{
 const h=harness();let mouse;h.c.arenaToolCapabilities=()=>({remote:false,tools:{wind:{enabled:true}}});
 h.arena.canvas={addEventListener(k,fn){mouse=fn;},getBoundingClientRect(){return {left:0,top:0};}};
 h.arena.screenToWorld=()=>({x:4,y:3});h.arena.toolMode='wind';h.arena.setupMouseEvents();crosswind(h);mouse({clientX:10,clientY:20});
 const intended=[...h.arena.windVector];assert.equal(h.timers.size,0);for(const stale of h.callbacks)stale();assert.deepEqual([...h.arena.windVector],intended);
 crosswind(h);h.fire();assert.deepEqual([...h.arena.windVector],intended);
});
