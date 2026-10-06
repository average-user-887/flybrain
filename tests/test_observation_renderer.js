'use strict';
const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const renderer = require('../web/observation_renderer.js');
const generated = JSON.parse(fs.readFileSync(process.env.NEUROFLY_DISPLAY_FIXTURES || path.join(__dirname,'fixtures/observation_display_all14.json')));
const clone = x => JSON.parse(JSON.stringify(x));
const packet = x => ({identity:x.identity, brain_id:x.brainId, observation:x.observation, observation_publication:x.observationPublication,
    metrics:{operant_learning_index:1,performance_index:12345}});
class Element {
    constructor(doc, tag) { this.ownerDocument=doc; this.tagName=tag; this.children=[]; this.ownText=''; }
    set textContent(text) { this.ownText=String(text); this.children=[]; }
    get textContent() { return this.ownText+this.children.map(x=>x.textContent).join(' '); }
    setAttribute(name,value) { this[name]=String(value); }
    appendChild(child) { this.children.push(child); }
    replaceChildren(...children) { this.ownText=''; this.children=children; }
    set innerHTML(_) { throw Error('Transported content must use DOM text'); }
}
const document = {createElement(tag) { return new Element(this,tag); }};
function rendered(d) { const el=new Element(document,'div'); renderer.render(el,d); return el; }
for (const x of generated.fixtures) test('all14 producer DOM binding: '+x.identity.assay,()=>{
    const d=renderer.view(packet(x),{connected:true}); const el=rendered(d);
    assert.equal(d.live.state,'available',d.live.error);
    assert.equal(d.terminal.state,'available',d.terminal.error);
    assert.match(el.textContent,/Live observation \(provisional\)/);
    assert.match(el.textContent,/Saved observation/);
    assert.match(el.textContent,/incomplete.*manual_reset/);
    for(const [name,r] of Object.entries(d.live.records)) {
        assert.ok(el.textContent.includes(name.replace(/_/g,' ')),name);
        assert.ok(el.textContent.includes(r.text),r.text);
        assert.ok(el.textContent.includes('unit: '+r.unit));
    }
    const p=renderer.primary(d), expected=d.live.records[renderer.PRIMARY[x.identity.assay][0]];
    assert.equal(p.value,expected.text);
    assert.equal(p.rawValue,expected.available && typeof expected.rawValue==='number'?expected.rawValue:null);
});
test('zero false unsupported and unavailable never become fabricated success',()=>{
    const all=generated.fixtures.flatMap(x=>Object.values(renderer.view(packet(x),{connected:true}).live.records));
    assert.ok(all.some(x=>x.available && x.rawValue===0 && x.valueText==='0'));
    all.push(...Object.values(renderer.view(packet(generated.deadlineFixture),{connected:true}).terminal.records));
    assert.ok(all.some(x=>x.available && x.rawValue===false && x.text==='false'));
    assert.ok(all.some(x=>!x.available && x.rawValue===null));
    const x=generated.fixtures.find(x=>x.identity.assay==='visual-operant');
    const d=renderer.view(packet(x),{connected:true});
    assert.match(renderer.primary(d).label,/Safe-zone occupancy/);
    assert.equal(d.live.records.operant_learning.reason,'unsupported');
    assert.doesNotMatch(renderer.primary(d).label,/Learning/);
});
test('missing envelope cannot fall back to graph scalar',()=>{
    const x=packet(generated.fixtures[0]); delete x.observation;
    const d=renderer.view(x,{connected:true});
    assert.equal(renderer.primary(d).value,'Unavailable');
    assert.equal(renderer.primary(d).rawValue,null);
    assert.match(rendered(d).textContent,/no_live_observation/);
});
test('daemon sibling brain owner is joined without concealing conflicting identity',()=>{
    const p=clone(packet(generated.fixtures[0]));
    delete p.identity.brain_id;
    const before=JSON.stringify(p);
    assert.equal(renderer.view(p,{connected:true}).live.state,'available');
    assert.equal(JSON.stringify(p),before);
    p.identity.brain_id='conflicting-owner';
    assert.equal(renderer.view(p,{connected:true}).live.state,'unavailable');
    p.identity.brain_id=null;
    assert.equal(renderer.view(p,{connected:true}).live.state,'unavailable');
    delete p.identity.brain_id; delete p.brain_id;
    assert.equal(renderer.view(p,{connected:true}).live.state,'unavailable');
});
test('disconnect pending selected-owner mismatch and reconnect suppress stale live values',()=>{
    const p=packet(generated.fixtures[0]);
    for(const options of [{connected:false},{connected:true,pending:true},{connected:true,assay:'visual-operant'}]) {
        const d=renderer.view(p,options); assert.equal(renderer.primary(d).rawValue,null);
        assert.equal(d.live.state,'unavailable');
    }
    assert.equal(renderer.view(p,{connected:false}).terminal.historical,null);
    assert.equal(renderer.view(p,{connected:true,pending:true}).terminal.state,'unavailable');
    assert.equal(renderer.view(p,{connected:true}).live.state,'available');
    assert.equal(renderer.view(packet(generated.fixtures[1]),{connected:true}).currentIdentity.assay,'t-maze');
});
test('invalid result values retain validity but never enter selected numerical plot',()=>{
    const x=clone(generated.fixtures[0]); x.observation.validity='invalidated';
    const d=renderer.view(packet(x),{connected:true});
    assert.equal(d.live.state,'available',d.live.error);
    assert.equal(renderer.primary(d).rawValue,null);
    assert.match(renderer.primary(d).value,/invalidated/);
});
test('historical receipts incomplete terminals and replay provenance stay explicit',()=>{
    const x=clone(generated.fixtures[0]);x.identity.activation++;x.identity.run_id='new-run';x.observation.identity=clone(x.identity);
    const d=renderer.view(packet(x),{connected:true});
    assert.equal(d.terminal.historical,true);
    assert.match(rendered(d).textContent,/Historical saved terminal.*incomplete.*manual_reset/);
    const replay=renderer.view(packet(x),{connected:true,replay:true});
    assert.equal(replay.terminal.state,'unavailable');
    assert.match(rendered(replay).textContent,/Recording replay · not live durable data/);
    const bad=clone(x);bad.observationPublication.last_terminal.receipt.durable=false;
    assert.match(rendered(renderer.view(packet(bad),{connected:true})).textContent,/rejected_transport/);
});
test('transported notes use text nodes, never HTML execution',()=>{
    const x=clone(generated.fixtures[0]);const r=Object.values(x.observation.records)[0];
    r.note='<img src=x onerror=alert(1)>';
    assert.ok(rendered(renderer.view(packet(x),{connected:true})).textContent.includes(r.note));
});
const app=fs.readFileSync(path.join(__dirname,'../web/app.js'),'utf8');
const ctx={window:{NeuroFlyObservationRenderer:renderer,hud:{daemonBridge:{connected:true}}}};
vm.createContext(ctx);
vm.runInContext(app.slice(app.indexOf('class ScientificBioArena'),app.indexOf('const EXPERIMENT_GUIDES'))+'\nglobalThis.methods=ScientificBioArena.prototype;',ctx);
test('actual arena adapter uses typed current packet and explicitly labels preview',()=>{
    const x=generated.fixtures.find(x=>x.identity.assay==='visual-operant');
    const a={remoteDriven:true,activeParadigmId:x.identity.assay,remotePacket:packet(x),
        getObservationDisplay:ctx.methods.getObservationDisplay};
    assert.match(ctx.methods.getCanonicalMetricInfo.call(a).label,/Safe-zone occupancy/);
    ctx.window.hud.daemonBridge.connected=false;
    assert.equal(ctx.methods.getCanonicalMetricInfo.call(a).value,'Unavailable');
    ctx.window.hud.daemonBridge.connected=true;
    assert.notEqual(ctx.methods.getCanonicalMetricInfo.call(a).value,'Unavailable');
    const preview={remoteDriven:false,awaitingDaemon:false,getPreviewMetricInfo:()=>({label:'Occupancy',value:'1'})};
    assert.match(ctx.methods.getCanonicalMetricInfo.call(preview).label,/local preview/);
});
test('scripts load validation before consumer; catalog has no initial made-up score',()=>{
    const html=fs.readFileSync(path.join(__dirname,'../web/index.html'),'utf8');
    const assets=['metric_records.js','observation_display.js','observation_renderer.js','app.js'];
    for(let i=1;i<assets.length;i++) assert.ok(html.indexOf('src="'+assets[i-1])<html.indexOf('src="'+assets[i]));
    assert.equal((html.match(/id="cardMetric[^\"]*">Not selected/g)||[]).length,14);
    assert.doesNotMatch(html,/Learning Index LI/);
});

const persistenceStart=app.indexOf('    updatePersistenceBanner() {');
const persistenceEnd=app.indexOf('    /** Seconds the daemon',persistenceStart);
vm.runInContext('globalThis.persistence = ({'+app.slice(persistenceStart,persistenceEnd)+'}).updatePersistenceBanner;',ctx);
test('recording halt uses recording recovery and unresolved required saves are never diagnostic',()=>{
    const el={style:{},textContent:''};ctx.document={getElementById:()=>el};
    const persistence={ok:false,reason:'disk full',failing:{recording:{required:true}}};
    ctx.persistence.call({connected:true,daemonPersistence:persistence,daemonMode:'scientific',
        daemonHalt:{detail:{failure_class:'persistence',channel:'recording',step:1945,
            recover:'Repair storage; restart with --record NEW_NAME and the same saved brains.'}}});
    assert.match(el.textContent,/STOPPED.*requested recording.*NEW_NAME/);
    assert.doesNotMatch(el.textContent,/then select the assay: it saves first/);
    ctx.persistence.call({connected:true,daemonPersistence:persistence,daemonMode:'scientific'});
    assert.match(el.textContent,/REQUIRED SAVE UNRESOLVED/);
    assert.doesNotMatch(el.textContent,/diagnostic log only|unaffected|run stopped/);
});
test('persistence banner preserves incomplete reason and uses only valid recovery steps',()=>{
    const el={style:{},textContent:''};ctx.document={getElementById:()=>el};
    for(const step of [undefined,null,'4',-1,1.5,NaN,Infinity]) {
        ctx.persistence.call({connected:true,daemonValidity:{state:'incomplete',incidents:[{step:3,reason:'save_failed',recovered_at:'time',recovered_step:step}]}});
        assert.match(el.textContent,/RESULT INCOMPLETE.*save_failed/);
        assert.doesNotMatch(el.textContent,/recovered at step|undefined/);
    }
    ctx.persistence.call({connected:true,daemonValidity:{state:'incomplete',incidents:[{step:3,reason:'save_failed',recovered_at:'time',recovered_step:0}]}});
    assert.match(el.textContent,/recovered at step 0/);
});

const hudStart=app.indexOf('class ScientificHUD');const hudEnd=app.indexOf('// 9. APPLICATION INITIALIZATION');
vm.runInContext(app.slice(hudStart,hudEnd)+'\nglobalThis.livePlot=ScientificHUD.prototype.renderLiveOutcome;',ctx);
test('actual plot discards old presentation and disconnected samples',()=>{
    const drawing={clearRect(){},fillText(){},beginPath(){},moveTo(){},lineTo(){},stroke(){}};
    let metric={rawValue:0,contextKey:'owner/presentation1',unit:'mm',value:'0 mm'};
    const hud={arena:{remotePacket:{step:1,sim_time_s:1},getCanonicalMetricInfo:()=>metric},curveCtx:drawing,curveCanvas:{clientWidth:400,clientHeight:100}};
    ctx.livePlot.call(hud);assert.equal(hud.liveOutcomeHistory[0].value,0);
    metric={rawValue:null,contextKey:'disconnected',unit:'mm',value:'Unavailable',sub:'Disconnected'};
    ctx.livePlot.call(hud);assert.equal(hud.liveOutcomeHistory.length,1);assert.equal(hud.liveOutcomeHistory[0].value,null);
    hud.arena.remotePacket={step:2,sim_time_s:2};metric={rawValue:3,contextKey:'owner/presentation2',unit:'mm'};
    ctx.livePlot.call(hud);assert.equal(hud.liveOutcomeHistory.length,1);assert.equal(hud.liveOutcomeHistory[0].value,3);
});

vm.runInContext('globalThis.hudMethods=ScientificHUD.prototype;',ctx);
vm.runInContext(fs.readFileSync(path.join(__dirname,'../web/live_assays.js'),'utf8'),ctx);
class InteractiveElement extends Element {
    constructor(doc,tag,id) { super(doc,tag);this.id=id;this.style={display:'none'};this.handlers={};this.classList={toggle(){}}; }
    addEventListener(name,callback) { this.handlers[name]=callback; }
    click() { this.handlers.click?.(); }
    querySelector(selector) { return this.ids?.[selector.slice(1)] || null; }
    querySelectorAll() { return []; }
    getContext() { return {clearRect(){},beginPath(){},moveTo(){},lineTo(){},stroke(){},fillText(){}}; }
    set innerHTML(markup) {
        if(this.id!=='assayToolsPanel') throw Error('Legacy measurement innerHTML write: '+this.id);
        this.ids={};
        for(const match of markup.matchAll(/id="([^"]+)"/g)) this.ids[match[1]]=new InteractiveElement(this.ownerDocument,'div',match[1]);
        this.children=Object.values(this.ids);
    }
}
test('actual tools tab, updater and remount have one typed owner with unchanged paused packet',()=>{
    const x=generated.fixtures.find(x=>x.identity.assay==='t-maze');
    const p={...packet(x),paused:true,paradigm:'t-maze',step:78214,sim_time_s:1,
        fly:{speed:0,state:'PAUSED'},descending:{dna02_yaw:0},
        live_assay:{parameters:[],actions:[],model_version:'test-ui-capability'},
        metrics:{legacy_scalar_must_not_render:8675309,performance_index:1}};
    const ids={};const doc={activeElement:null,createElement(tag){return new InteractiveElement(this,tag);},
        getElementById(id){return ids[id]||ids.assayToolsPanel?.querySelector('#'+id)||null;}};
    for(const id of ['assayToolsPanel','guideContent','limbDeckPanel','trainingPanel','tabGuide','tabAssayTools','tabLimbDeck','tabTraining'])
        ids[id]=new InteractiveElement(doc,'div',id);
    ctx.document=doc;ctx.window.neuroflyAssayLimitationForController=()=>'';
    const bridge={connected:true,readOnly:false,switchPending:false,replayMode:false};
    const arena={remoteDriven:true,activeParadigmId:'t-maze',remotePacket:p,getObservationDisplay:ctx.methods.getObservationDisplay};
    const hud={arena,daemonBridge:bridge};ctx.window.hud=hud;
    for(const name of ['renderObservationPanels','updateAssayTools','renderAssayTools','setupDeckTabs']) hud[name]=ctx.hudMethods[name];
    hud.renderAssayTools('t-maze');hud.setupDeckTabs();
    const measurements=()=>doc.getElementById('liveMetrics');
    const check=()=>{ assert.match(measurements().textContent,/Live observation \(provisional\)/);
        assert.match(measurements().textContent,/Saved observation/);
        assert.doesNotMatch(measurements().textContent,/legacy_scalar_must_not_render|8675309|performance_index/); };
    check();
    const before=measurements().textContent;
    ids.tabAssayTools.click();check();assert.equal(measurements().textContent,before);
    hud.activeAssayUpdater();check();assert.equal(measurements().textContent,before);
    for(const mode of ['disconnected','pending','ownerMismatch','replay','connected']) {
        arena.activeParadigmId=mode==='ownerMismatch'?'visual-operant':'t-maze';
        bridge.connected=mode!=='disconnected';bridge.switchPending=mode==='pending';bridge.replayMode=mode==='replay';
        ids.tabAssayTools.click();hud.activeAssayUpdater();check();
        if(['disconnected','pending','ownerMismatch'].includes(mode)) assert.match(measurements().textContent,/Unavailable/);
        if(mode==='replay') assert.match(measurements().textContent,/Recording replay · not live durable data/);
        hud.renderAssayTools('t-maze');check(); // Mount, without requiring a new packet/tick.
        ids.tabAssayTools.click();check();
    }
});


test('readable primary text keeps exact raw value and exposes it on observation DOM',()=>{
    const x=clone(generated.fixtures.find(x=>x.identity.assay==='open-arena'));
    const raw=26.417029773597136;
    Object.assign(x.observation.records.distance_mm,{value:raw,available:true,reason:null,capability:'measured'});
    const d=renderer.view(packet(x),{connected:true});
    assert.equal(d.live.state,'available',d.live.error);
    assert.equal(renderer.primary(d).value,'26.417 mm');assert.equal(renderer.primary(d).rawValue,raw);
    const root=rendered(d),all=[];function walk(el){all.push(el);el.children.forEach(walk);}walk(root);
    const value=all.find(el=>el.title==='Raw value: '+raw+' mm');assert.ok(value);
    assert.equal(value.textContent,'26.417 mm');assert.equal(value['aria-label'],'26.417 mm · Raw value: '+raw+' mm');
});
