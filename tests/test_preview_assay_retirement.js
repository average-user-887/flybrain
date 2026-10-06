'use strict';
const test=require('node:test'),assert=require('node:assert/strict'),fs=require('node:fs'),vm=require('node:vm'),path=require('node:path');
const app=fs.readFileSync(path.join(__dirname,'../web/app.js'),'utf8');
function harness(){
 const elements={},nodes={},calls=[],bridge={connected:false,replayMode:false,switchPending:false,sendCommand(){throw new Error('preview dispatched a daemon command');}};
 const ctx=new Proxy({},{get:(o,k)=>o[k]||(()=>{})});
 const canvas={getContext:()=>ctx,getBoundingClientRect:()=>({width:500,height:400}),addEventListener(){}};
 const panel={style:{},querySelector(selector){return nodes[selector]||=( {textContent:'',addEventListener(k,fn){this[k]=fn;}} );}};
 const c={window:{addEventListener(){},devicePixelRatio:1,hud:{daemonBridge:bridge}},document:{getElementById:id=>id==='arenaCanvas'?canvas:id==='assayToolsPanel'?panel:elements[id]||null},setTimeout(){return 1;},clearTimeout(){},console};
 vm.createContext(c);
 for(const [start,end]of [['class MushroomBodyCircuit','const EXPERIMENT_GUIDES'],['const EXPERIMENT_GUIDES','class DaemonBridgeClient'],['const ASSAY_CONFIGS =','const LESION_INFO ='],['class ScientificHUD','// 9. APPLICATION INITIALIZATION']])vm.runInContext(app.slice(app.indexOf(start),app.indexOf(end,app.indexOf(start))),c);
 vm.runInContext('this.A=ScientificBioArena;this.H=ScientificHUD;this.guides=EXPERIMENT_GUIDES;this.tools=ASSAY_CONFIGS;',c);
 const arena=new c.A('arenaCanvas');
 return {c,arena,panel,nodes,elements,bridge,calls,render(id){arena.initParadigm(id);c.H.prototype.renderAssayTools.call({arena,daemonBridge:bridge,renderObservationPanels(){}},id);}};
}
test('reported Y gates, Buridan stripe writes, thermal flash, reversal and false physical actions are retired',()=>{
 const h=harness();
 for(const [assay,labels]of Object.entries({'y-maze':['Gate Left Arm','Gate Right Arm'],'buridan':['Invert Contrast (Dark/Light)','Rotate Stripes (90°)'],'heat-maze':['Thermal Shock Flash (42°C)','Reset Spatial Map'],'t-maze':['Deliver Shock Pulse (PPL1)','Invert CS+/CS- Arms'],'visual-operant':['Laser Beam Pulse','Invert Heat Sectors (Reversal)'],'looming-escape':['Trigger Looming Disc','Direct Giant Fiber Spike'],'courtship':['Trigger Wing Vibration (Song)','Toggle Female Receptivity'],'gap-crossing':['Extend Foreleg Probe'],'labyrinth':['Bait Exit Chamber']})){
  h.render(assay);for(const label of labels)assert.ok(!h.panel.innerHTML.includes(label),label);
 }
 assert.ok(!app.includes('thermal_flash')&&!app.includes('gf_looming'));
});
test('preview T reward/punishment and escape state cannot invent reversal or voltage readouts',()=>{
 const h=harness();h.arena.initParadigm('t-maze');h.arena.paradigmState.csPlusArm='arm_b';
 const reward=h.c.tools['t-maze'].metrics.find(m=>m.label==='Preview reinforcement arms');assert.equal(reward.get(h.arena),'A reward / B punishment');
 const escape=h.c.tools['looming-escape'].metrics.find(m=>m.label==='Preview escape state');
 h.arena.dn.escapeActive=false;assert.equal(escape.get(h.arena),'INACTIVE');h.arena.dn.escapeActive=true;assert.equal(escape.get(h.arena),'ACTIVE');assert.ok(!app.includes('100% [SPIKE]')&&!app.includes('12% [SUBTHRESHOLD]'));
});
test('every fixed analytic preview diagram explicitly discloses that it is not measured data',()=>{
 const h=harness();
 for(const id of ['visual-operant','wind-tunnel','looming-escape','optomotor','gap-crossing','circadian-dam','courtship']){
  h.render(id);assert.match(h.panel.innerHTML,/Illustrative static diagram · not measurements/);assert.ok(!h.panel.innerHTML.includes('LIVE GRAPH'));
  const labels=[];const ctx=new Proxy({fillText(t){labels.push(t);}},{get:(o,k)=>o[k]||(()=>{})});h.c.tools[id].drawChart(ctx,380,110,h.arena,{});assert.match(labels[0],/Illustrative.*not measured/);
 }
});
test('preview readouts name available state and disclose fixed assumptions or retained history',()=>{
 const h=harness(),a=h.arena;
 a.initParadigm('wind-tunnel');a.windVector=[-3,4];assert.equal(h.c.tools['wind-tunnel'].metrics.find(m=>m.label==='Preview wind vector speed').get(a),'5.0 mm/s');assert.equal(h.c.tools['wind-tunnel'].sliders[0].unit,' mm/s');h.render('wind-tunnel');assert.match(h.panel.innerHTML,/plume transport and steering remain fixed/);
 a.initParadigm('looming-escape');a.fly.heading=Math.PI/2;a.fly.speed=7;assert.equal(h.c.tools['looming-escape'].metrics.find(m=>m.label==='Current preview heading').get(a),'90°');assert.equal(h.c.tools['looming-escape'].metrics.find(m=>m.label==='Current preview speed').get(a),'7.0 mm/s');
 a.initParadigm('optomotor');a.dn.dna02Diff=3;assert.equal(h.c.tools.optomotor.metrics.find(m=>m.label==='Preview descending steering drive').get(a),'3.00');assert.equal(h.c.tools.optomotor.metrics.find(m=>m.label==='Assumed preview gain').get(a),'0.88');
 a.initParadigm('y-maze');a.paradigmState.sar=.5;a.paradigmState.armSequence=[0,1,2];a.paradigmState.turnDirections=['L','R'];h.c.tools['y-maze'].actions[0].handler(a,h);assert.equal(a.paradigmState.armSequence.length,0);assert.equal(a.paradigmState.turnDirections.length,0);assert.equal(a.paradigmState.sar,.5);assert.match(h.c.tools['y-maze'].actions[0].label,/keeps last SAR/);
});
test('retained physical parameter callbacks affect the actual next-step or contact model',()=>{
 const h=harness(),a=h.arena;
 a.initParadigm('optomotor');h.c.tools.optomotor.sliders[0].apply(a,60);const old=a.paradigmState.drumAngleDeg;a.step(.01);assert.ok(Math.abs(a.paradigmState.drumAngleDeg-old-.6)<1e-8);
 a.initParadigm('gap-crossing');h.c.tools['gap-crossing'].sliders[0].apply(a,5);a.fly.x=43;a.step(.01);assert.equal(a.paradigmState.decisionOutcome,'ABORT');
 a.initParadigm('heat-maze');h.c.tools['heat-maze'].sliders.find(x=>x.key==='floorTemp').apply(a,42);a.fly.x=15;a.fly.y=15;a.step(.01);assert.ok(a.paradigmState.temp>36.5);
 // Inward motion has negative v·n; a zero normal impulse cannot exercise friction.
 a.initParadigm('labyrinth');const wall=a.currentWalls[0],before=a.coulombSlide(10,-2,[0,1],-2,wall.friction);h.c.tools.labyrinth.sliders[0].apply(a,.8);const after=a.coulombSlide(10,-2,[0,1],-2,wall.friction);assert.equal(before[1],0);assert.equal(after[1],0);assert.ok(Math.abs(after[0])<Math.abs(before[0]));
 a.initParadigm('multisensory-sandbox');h.c.guides['multisensory-sandbox'].params.find(x=>x.key==='cpgBaseFreq').apply(a,12);a.cpg.step(40,false,.01);assert.equal(a.cpg.steppingFreq,12);
});
test('retained local actions change the actual preview objects without remote dispatch',()=>{
 const h=harness(),a=h.arena;a.initParadigm('open-arena');const n=a.foodItems.length;h.c.tools['open-arena'].actions[0].handler(a,h);assert.equal(a.foodItems.length,n+1);
 const predators=a.predators.length;h.c.tools['open-arena'].actions[1].handler(a,h);assert.equal(a.predators.length,predators+1);
 a.initParadigm('t-maze');a.mb.w[0][0]=.5;h.c.tools['t-maze'].actions[0].handler(a,h);assert.equal(a.mb.w[0][0],0);
 a.initParadigm('optomotor');h.c.tools.optomotor.actions.find(x=>x.label==='Toggle preview grating drawing').handler(a,h);assert.equal(a.paradigmState.contrast,0);
 a.initParadigm('heat-maze');const old=[...a.paradigmState.refugePos];h.c.tools['heat-maze'].actions[0].handler(a,h);assert.notDeepEqual([...a.paradigmState.refugePos],old);
});
test('actual mounted preview listeners deny stale live/replay actions and sliders and never POST',()=>{
 const h=harness();h.render('open-arena');const action=h.nodes['#assay_btn_0'],slider=h.nodes['#assay_slider_wallRepulsion'];
 const original=h.arena.foodItems.length;action.click();assert.equal(h.arena.foodItems.length,original+1);slider.input({target:{value:'1.7'}});assert.equal(h.arena.wallRepulsion,1.7);
 for(const state of ['connected','replayMode','switchPending']){h.bridge[state]=true;action.click();slider.input({target:{value:'2.7'}});assert.equal(h.arena.foodItems.length,original+1);assert.equal(h.arena.wallRepulsion,1.7);h.bridge[state]=false;}
});
test('ineffective Guide callbacks are absent while empty sections disclose unavailable controls',()=>{
 const h=harness();for(const id of ['t-maze','y-maze','wind-tunnel','optomotor','circadian-dam','courtship'])assert.equal(h.c.guides[id].params.length,0,id);
 assert.deepEqual(Array.from(h.c.guides['heat-maze'].params,x=>x.key),['refugeRadius']);
 h.render('buridan');assert.match(h.panel.innerHTML,/No supported adjustable local preview parameters/);assert.match(h.panel.innerHTML,/local preview/);
 assert.match(h.c.guides.buridan.params[0].desc,/drawing only; containment and locomotion remain unchanged/);
});
test('retained Guide parameters feed the actual next-step model',()=>{
 const h=harness(),a=h.arena;
 a.initParadigm('visual-operant');h.c.guides['visual-operant'].params[0].apply(a,200);a.fly.yawRate=1;a.step(.01);assert.equal(a.paradigmState.drumAngleDeg,359);
 a.initParadigm('looming-escape');h.c.guides['looming-escape'].params[0].apply(a,50);a.step(.01);assert.ok(a.paradigmState.thetaDeg>2);
 a.initParadigm('heat-maze');a.fly.x=a.paradigmState.refugePos[0]+14;a.fly.y=a.paradigmState.refugePos[1];h.c.guides['heat-maze'].params[0].apply(a,15);a.step(.01);assert.equal(a.paradigmState.refugeReached,true);
 a.initParadigm('multisensory-sandbox');a.fly.x=a.paradigmState.hotspotPos[0];a.fly.y=a.paradigmState.hotspotPos[1];const hotspot=h.c.guides['multisensory-sandbox'].params.find(x=>x.key==='hotspotTemp');hotspot.apply(a,30);a.step(.01);const coolSurface=a.paradigmState.temp;a.fly.x=a.paradigmState.hotspotPos[0];a.fly.y=a.paradigmState.hotspotPos[1];hotspot.apply(a,38.5);a.step(.01);assert.ok(a.paradigmState.temp>coolSurface);a.fly.x=a.paradigmState.hotspotPos[0];a.fly.y=a.paradigmState.hotspotPos[1];hotspot.apply(a,45);a.step(.01);assert.equal(a.paradigmState.temp,42); // Existing surface-temperature clamp.
 a.initParadigm('labyrinth');h.c.guides.labyrinth.params[0].apply(a,.9);assert.ok(a.currentWalls.every(w=>w.friction===.9));assert.equal(a.coulombSlide(10,-2,[0,1],-2,a.currentWalls[0].friction)[0],8.2);
});
test('mounted Guide listener preserves local wind setting across live/replay/switch owners',()=>{
 const h=harness(),a=h.arena;a.initParadigm('open-arena');
 h.elements.guideTitle={textContent:''};h.elements.val_windVelocity={textContent:''};h.elements.dynamicSlidersContainer={appendChild(){}};
 h.c.document.createElement=()=>({querySelector(selector){return h.nodes[selector]={addEventListener(k,fn){this[k]=fn;}};}});
 h.c.H.prototype.renderExperimentGuide.call({arena:a},'open-arena');assert.equal(h.elements.guideTitle.textContent,h.c.guides['open-arena'].title);
 const slider=h.nodes['#slider_windVelocity'];slider.input({target:{value:'20'}});assert.deepEqual([...a.windVector],[-20,0]);
 for(const state of ['connected','replayMode','switchPending']){h.bridge[state]=true;slider.input({target:{value:'30'}});assert.deepEqual([...a.windVector],[-20,0]);h.bridge[state]=false;}
});
