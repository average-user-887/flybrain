'use strict';
const test=require('node:test');
const assert=require('node:assert/strict');
const fs=require('node:fs');
const vm=require('node:vm');
const path=require('node:path');
const app=fs.readFileSync(path.join(__dirname,'../web/app.js'),'utf8');
const html=fs.readFileSync(path.join(__dirname,'../web/index.html'),'utf8');
function harness(){
 const elements={};
 const c={document:{getElementById:id=>elements[id]||null,querySelectorAll:()=>[]},window:{location:{search:'?daemon=off'}},URLSearchParams,console};
 vm.createContext(c);
 for(const [start,end] of [['class ScientificBioArena','const EXPERIMENT_GUIDES'],['class DaemonBridgeClient','const ASSAY_CONFIGS'],['const LESION_INFO =','// 9. APPLICATION INITIALIZATION']]){
  const a=app.indexOf(start),b=app.indexOf(end,a);assert.ok(a>=0&&b>a);vm.runInContext(app.slice(a,b),c);
 }
 vm.runInContext('globalThis.A=ScientificBioArena;globalThis.B=DaemonBridgeClient;globalThis.H=ScientificHUD;',c);
 return {c,elements,el(id){return elements[id]={textContent:'',style:{},classList:{add(){},remove(){}},addEventListener(k,f){this[k]=f;}};}};
}
function disconnected(h,options={}){
 const arena={remoteDriven:false,awaitingDaemon:true};
 const bridge={arena,connected:false,activeUrl:null,showDaemonAddress(){},updateFreshness(){},...options};
 h.c.B.prototype.onDaemonDisconnected.call(bridge);return bridge;
}
test('dead T5 and wing state/controls are absent; general biological descriptions remain',()=>{
 for(const removed of ['DELTA_OFF','btnLesionOFF','overrideWings','overrideLegs','sliderStimWing','valStimWing']){assert.ok(!app.includes(removed),removed);assert.ok(!html.includes(removed),removed);}
 assert.match(html,/T4\/T5/);assert.match(html,/sliderStimCpg/);assert.match(html,/btnFlareGf/);
});
test('retained lesion buttons invoke actual supported model flags and WT restores them',()=>{
 const h=harness(),arena={cx:{},mb:{},setLesion:h.c.A.prototype.setLesion};
 for(const id of ['btnLesionWT','btnLesionMB','btnLesionCX','btnLesionGF','btnLesionJO'])h.el(id);
 h.c.H.prototype.setupLesionEvents.call({arena});
 for(const [id,check] of [['btnLesionMB',()=>arena.mb.plasticityEnabled===false],['btnLesionCX',()=>arena.cx.isLesioned],['btnLesionGF',()=>arena.gfLesioned],['btnLesionJO',()=>arena.joLesioned]]){h.elements[id].click();assert.ok(check(),id);h.elements.btnLesionWT.click();assert.equal(arena.cx.isLesioned,false);assert.equal(arena.mb.plasticityEnabled,true);assert.equal(arena.gfLesioned,false);assert.equal(arena.joLesioned,false);}
});
test('actual CPG cadence and GF escape callbacks remain functional',()=>{
 const h=harness();for(const id of ['sliderStimCpg','valStimCpg','btnFlareGf'])h.el(id);
 const arena={cpg:{},dn:{dnp01Gf:0},paradigmState:{},isStandalonePreview:h.c.A.prototype.isStandalonePreview};
 h.c.H.prototype.setupNeuroStimControls.call({arena});
 h.elements.sliderStimCpg.input({target:{value:'12.5'}});assert.equal(arena.cpg.baseFreq,12.5);assert.equal(h.elements.valStimCpg.textContent,'12.5');
 h.elements.btnFlareGf.click();assert.equal(arena.dn.escapeActive,true);assert.equal(arena.dn.escapeTimer,.4);assert.equal(arena.dn.dnp01Gf,1);
});
test('explicit off candidates and actual local disconnection update preview status',()=>{
 const h=harness();h.el('arenaRunState').textContent='Connecting to live experiment';
 const candidates=Object.getOwnPropertyDescriptor(h.c.B.prototype,'daemonCandidates').get.call({daemonPort:8769});assert.equal(candidates.auto.length,0);assert.equal(candidates.offer.length,0);
 const bridge=disconnected(h);assert.equal(bridge.arena.remoteDriven,false);assert.equal(bridge.arena.awaitingDaemon,false);assert.equal(h.elements.arenaRunState.textContent,'Standalone preview · local engine');
});
test('local status never overwrites frozen remote or replay scientific status',()=>{
 const h=harness();const label=h.el('arenaRunState');label.textContent='Prior remote frame';
 const bridge=disconnected(h,{activeUrl:'http://daemon',arena:{remoteDriven:true}});assert.equal(bridge.arena.awaitingDaemon,true);assert.equal(label.textContent,'Prior remote frame');
 disconnected(h,{replayMode:true});assert.equal(label.textContent,'Prior remote frame');
});
