'use strict';
const test=require('node:test'),assert=require('node:assert/strict'),fs=require('node:fs'),path=require('node:path'),vm=require('node:vm');
const renderer=require('../web/observation_renderer.js');
const app=fs.readFileSync(path.join(__dirname,'../web/app.js'),'utf8'),html=fs.readFileSync(path.join(__dirname,'../web/index.html'),'utf8');
const generated=JSON.parse(fs.readFileSync(process.env.NEUROFLY_DISPLAY_FIXTURES||path.join(__dirname,'fixtures/observation_display_all14.json')));
const pairs=[...html.matchAll(/data-paradigm="([^"]+)"[\s\S]*?id="(cardMetric[^"]+)"/g)];
function harness(){
 const cards={},ids={};
 for(const [,assay,id] of pairs){
  const label={textContent:'stale mode label'},row={title:'stale units/context',attributes:{},querySelector:()=>label,setAttribute(k,v){this.attributes[k]=v;}};
  const value={textContent:'stale number',parentElement:row};cards[assay]={label,row,value};ids[id]=value;
 }
 const c={window:{NeuroFlyObservationRenderer:renderer},document:{querySelectorAll:()=>Object.values(ids),getElementById:id=>ids[id]||null}};
 vm.createContext(c);
 vm.runInContext(app.slice(app.indexOf('function sandboxScoreView('),app.indexOf('function renderSandboxScorecard(')),c);
 for(const [start,end] of [['class ScientificBioArena','const EXPERIMENT_GUIDES'],['class ScientificHUD','// 9. APPLICATION INITIALIZATION']])vm.runInContext(app.slice(app.indexOf(start),app.indexOf(end,app.indexOf(start))),c);
 vm.runInContext('globalThis.A=ScientificBioArena;globalThis.H=ScientificHUD',c);
 const arena=Object.assign(Object.create(c.A.prototype),{activeParadigmId:'open-arena',mb:{netValence:-.79},paradigmState:{}});
 return {cards,c,arena,show(metric){c.H.prototype.updateCardMetric.call({arena},metric);},metric(){return arena.getCanonicalMetricInfo();}};
}
function check(card,metric){
 assert.equal(card.label.textContent,'Metric: '+metric.label);assert.equal(card.value.textContent,metric.value);
 assert.ok(card.row.title.includes(metric.sub));assert.equal(card.row.attributes['aria-label'],card.row.title);
 if(metric.unit)assert.ok(card.row.title.includes('unit: '+metric.unit.trim()));
}
function inactive(card){assert.equal(card.label.textContent,'Metric:');assert.equal(card.value.textContent,'Not selected');assert.equal(card.row.title,'No current metric · assay not selected');assert.equal(card.row.attributes['aria-label'],card.row.title);}
test('all14 markup starts neutral and actual update delegates canonical card binding',()=>{
 assert.equal(pairs.length,14);assert.equal((html.match(/data-card-metric-label>Metric:<\/span>/g)||[]).length,14);
 assert.match(app,/this\.updateCardMetric\(metric\)/);
});
test('actual preview canonical label corrects Distance/negative PI mismatch without changing value',()=>{
 const h=harness(),metric=h.metric();h.show(metric);check(h.cards['open-arena'],metric);
 assert.equal(metric.value,'-0.79');assert.match(h.cards['open-arena'].label.textContent,/Preference Index \(PI\).*local preview/);
 assert.doesNotMatch(h.cards['open-arena'].label.textContent,/Distance/);
 for(const [assay,card] of Object.entries(h.cards))if(assay!=='open-arena')inactive(card);
});
test('actual preview wind/sandbox definitions retain formatted units and context',()=>{
 const h=harness();h.arena.activeParadigmId='wind-tunnel';h.arena.paradigmState={surgeSteps:4,castSteps:2,upwindProgress:1,behavioralState:'SURGE'};
 let metric=h.metric();h.show(metric);check(h.cards['wind-tunnel'],metric);assert.equal(metric.value,'2.00');inactive(h.cards['open-arena']);
 h.arena.activeParadigmId='multisensory-sandbox';h.arena.paradigmState={compositeScore:0};metric=h.metric();h.show(metric);check(h.cards['multisensory-sandbox'],metric);
 assert.equal(metric.value,'0.0 / 100');assert.match(h.cards['multisensory-sandbox'].row.title,/heuristic body proxy/);inactive(h.cards['wind-tunnel']);
});
test('all14 actual canonical live/replay transitions replace label/value/unit/context and clear inactive cards',()=>{
 const h=harness();const bridge={connected:true,replayMode:false,switchPending:false};h.c.window.hud={daemonBridge:bridge};h.arena.remoteDriven=true;
 for(const replay of [false,true])for(const x of generated.fixtures){
  h.arena.activeParadigmId=x.identity.assay;bridge.replayMode=replay;
  h.arena.remotePacket={identity:x.identity,brain_id:x.brainId,observation:x.observation,observation_publication:x.observationPublication};
  const before=JSON.stringify(h.arena.remotePacket),metric=h.metric();h.show(metric);check(h.cards[x.identity.assay],metric);
  assert.match(metric.label,replay?/replay provisional/:/live provisional/);assert.equal(JSON.stringify(h.arena.remotePacket),before);
  for(const [assay,card] of Object.entries(h.cards))if(assay!==x.identity.assay)inactive(card);
 }
 h.arena.remoteDriven=false;h.arena.remotePacket=null;bridge.connected=false;bridge.replayMode=false;h.arena.activeParadigmId='open-arena';const metric=h.metric();h.show(metric);check(h.cards['open-arena'],metric);
 assert.match(metric.label,/local preview/);inactive(h.cards['multisensory-sandbox']);
});
test('pending/missing/disconnected/owner mismatch expose canonical unavailable context, not stale metric',()=>{
 for(const mode of ['pending','missing','disconnected','owner']){
  const h=harness(),x=generated.fixtures[0],bridge={connected:mode!=='disconnected',replayMode:false,switchPending:mode==='pending'};
  h.c.window.hud={daemonBridge:bridge};h.arena.remoteDriven=true;h.arena.activeParadigmId='open-arena';
  h.arena.remotePacket=mode==='missing'?null:{identity:{...x.identity,assay:mode==='owner'?'t-maze':'open-arena'},brain_id:x.brainId,observation:x.observation};
  const metric=h.metric();h.show(metric);check(h.cards['open-arena'],metric);assert.equal(metric.value,'Unavailable');
  for(const [assay,card] of Object.entries(h.cards))if(assay!=='open-arena')inactive(card);
 }
});
test('card tooltip exposes exact raw value and clears it on unavailable transition',()=>{
 const h=harness(),raw=26.417029773597136;
 h.show({label:'Distance · live provisional',value:'26.417 mm',unit:'mm',rawValue:raw,sub:'valid · presentation'});
 assert.ok(h.cards['open-arena'].row.title.includes('Raw value: '+raw));
 h.show({label:'Distance · live provisional',value:'Unavailable',unit:'',rawValue:null,sub:'Disconnected'});
 assert.doesNotMatch(h.cards['open-arena'].row.title,/Raw value|26\.417/);
});
