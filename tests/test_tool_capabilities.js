'use strict';
const test=require('node:test'),assert=require('node:assert/strict'),fs=require('node:fs'),path=require('node:path'),vm=require('node:vm');
const app=fs.readFileSync(path.join(__dirname,'../web/app.js'),'utf8'),html=fs.readFileSync(path.join(__dirname,'../web/index.html'),'utf8');
function slice(a,b){const start=app.indexOf(a),end=app.indexOf(b,start);assert.ok(start>=0&&end>start);return app.slice(start,end);}
function harness(){
 const ids=['toolSelect','toolFood','toolAlarm','toolPredator','toolWind','arenaToolHelp','previewWallAssist','previewWallAssistLabel','previewWallAssistNote'];
 const elements={};for(const id of ids){const classes=new Set(id==='toolSelect'?['active']:[]);elements[id]={disabled:false,textContent:'',title:'',checked:true,listeners:{},attributes:{},classList:{toggle(c,value){if(value)classes.add(c);else classes.delete(c);},contains(c){return classes.has(c);}},setAttribute(k,v){this.attributes[k]=v;},addEventListener(k,fn){this.listeners[k]=fn;}};}
 const c={window:{},document:{getElementById:id=>elements[id]||null},Math,alert(){throw Error('unsupported command alert');},daemonFrameOffset:()=>[0,0]};vm.createContext(c);
 vm.runInContext(slice('const ARENA_TOOL_IDS =','const SELECTABLE_BACKENDS ='),c);
 vm.runInContext('class FocusHUD {\n'+slice('    reconcileToolCapabilities() {','    renderAssayTools(pid) {')+'}\nwindow.HUD=FocusHUD;',c);
 vm.runInContext('class FocusArena {\n'+slice('    setupMouseEvents() {','    setLesion(type) {')+'}\nwindow.Arena=FocusArena;',c);
 const commands=[],mouse={};const bridge={connected:true,activeUrl:'http://daemon',readOnly:false,replayMode:false,switchPending:false,sendCommand(action,params){commands.push({action,params});return Promise.resolve({status:'ok'});}};
 const arena=Object.assign(Object.create(c.window.Arena.prototype),{toolMode:'select',remoteDriven:true,awaitingDaemon:false,activeParadigmId:'open-arena',remotePacket:{paradigm:'open-arena'},previewWallAssist:true,
  canvas:{addEventListener(k,fn){mouse[k]=fn;},getBoundingClientRect(){return {left:0,top:0};}},screenToWorld:(x,y)=>({x,y}),foodItems:[],alarms:[],predators:[],fly:{x:0,y:0},windVector:[-7,0]});
 const hud=new c.window.HUD();Object.assign(hud,{arena,daemonBridge:bridge});c.window.hud=hud;hud.setupToolEvents();arena.setupMouseEvents();
 return {hud,arena,bridge,commands,elements,click:id=>elements[id].listeners.click(),mouse:()=>mouse.mousedown({clientX:10,clientY:0})};
}
function local(h){h.bridge.connected=false;h.arena.remoteDriven=false;h.arena.awaitingDaemon=false;h.bridge.replayMode=false;delete h.arena.remotePacket.timing;h.hud.reconcileToolCapabilities();}

test('disabled styles defeat enabled-looking hover/active styling and neutral labels are honest',()=>{
 assert.match(html,/button:disabled, input:disabled, select:disabled/);
 assert.match(html,/\.tool-btn:disabled:hover, \.tool-btn:disabled.active/);
 assert.match(html,/id="toolSelect"[^>]*>Neutral selection/);assert.doesNotMatch(html,/Inspect Fly|>Drag Wind</);
 assert.match(html,/id="previewWallAssist"[^>]*aria-describedby="previewWallAssistNote"/);
 assert.match(html,/id="arenaToolHelp"[^>]*role="status"/);
});
test('actual Alarm→T-maze transition resets mode, active class and aria pressed',()=>{
 const h=harness();h.click('toolAlarm');assert.equal(h.arena.toolMode,'alarm');assert.equal(h.elements.toolAlarm.classList.contains('active'),true);
 h.arena.activeParadigmId='t-maze';h.arena.remotePacket.paradigm='t-maze';h.hud.reconcileToolCapabilities();
 assert.equal(h.arena.toolMode,'select');assert.equal(h.elements.toolAlarm.disabled,true);assert.equal(h.elements.toolAlarm.classList.contains('active'),false);
 assert.equal(h.elements.toolSelect.classList.contains('active'),true);assert.equal(h.elements.toolSelect.attributes['aria-pressed'],'true');h.mouse();assert.equal(h.commands.length,0);
});
for(const mode of ['replay','recordingFrame','offline','pending','readOnly','waiting','ownerMismatch'])test('capability transition disarms spatial tools: '+mode,()=>{
 const h=harness();h.click('toolFood');
 if(mode==='replay')h.bridge.replayMode=true;
 if(mode==='recordingFrame')h.arena.remotePacket.timing={replay:true};
 if(mode==='offline')h.bridge.connected=false;
 if(mode==='pending')h.bridge.switchPending=true;
 if(mode==='readOnly')h.bridge.readOnly=true;
 if(mode==='waiting')h.arena.awaitingDaemon=true;
 if(mode==='ownerMismatch')h.arena.activeParadigmId='t-maze';
 h.hud.reconcileToolCapabilities();assert.equal(h.arena.toolMode,'select');assert.equal(h.elements.toolFood.disabled,true);
 assert.equal(h.elements.previewWallAssist.disabled,true);h.mouse();assert.equal(h.commands.length,0);
});
test('live wind and threat are visibly unavailable with persistent connected guidance',()=>{
 const h=harness();assert.equal(h.elements.toolWind.disabled,true);assert.equal(h.elements.toolPredator.disabled,true);
 assert.match(h.elements.toolWind.textContent,/Airflow/);assert.match(h.elements.toolWind.title,/unsupported.*Airflow/);
 assert.match(h.elements.toolPredator.title,/not implemented/);assert.match(h.elements.arenaToolHelp.textContent,/Threat placement unavailable live.*Airflow/);
 // Stale/programmatically armed unsupported modes cannot issue a real command.
 for(const mode of ['wind','predator']){h.arena.toolMode=mode;h.mouse();assert.equal(h.arena.toolMode,'select');}
 assert.equal(h.commands.length,0);assert.equal(h.arena.predators.length,0);assert.equal(h.arena.windVector[0],-7);
});
test('supported live Food and Alarm still use actual arena mouse request path',async()=>{
 const h=harness();for(const id of ['toolFood','toolAlarm']){h.click(id);h.mouse();}
 await Promise.resolve();assert.equal(h.commands.length,2);assert.equal(h.commands[0].action,'place_stimulus');assert.equal(h.commands[0].params.type,'food');assert.equal(h.commands[1].params.type,'alarm');
 assert.equal(h.commands[0].params.x,10);assert.equal(h.arena.foodItems.length,0);assert.equal(h.arena.alarms.length,0);
});
test('all existing standalone preview placement effects remain functional',()=>{
 const h=harness();local(h);
 for(const id of ['toolFood','toolAlarm','toolPredator','toolWind']){assert.equal(h.elements[id].disabled,false);h.click(id);h.mouse();}
 assert.equal(h.arena.foodItems.length,1);assert.equal(h.arena.alarms.length,1);assert.equal(h.arena.predators.length,1);
 assert.equal(h.arena.windVector[0],-15);assert.equal(h.commands.length,0);assert.match(h.elements.toolWind.textContent,/Click.*preview wind.*15 mm\/s/);
 assert.match(h.elements.arenaToolHelp.textContent,/toward the arena origin/);
});
test('preview assist is immutable live/replay and restores its prior local setting',()=>{
 const h=harness();h.arena.previewWallAssist=false;h.hud.reconcileToolCapabilities();
 assert.equal(h.elements.previewWallAssist.disabled,true);assert.match(h.elements.previewWallAssistNote.textContent,/Unavailable/);
 h.elements.previewWallAssist.listeners.change({target:{checked:true}});assert.equal(h.arena.previewWallAssist,false);
 local(h);assert.equal(h.elements.previewWallAssist.disabled,false);assert.equal(h.elements.previewWallAssist.checked,false);
 h.elements.previewWallAssist.listeners.change({target:{checked:true}});assert.equal(h.arena.previewWallAssist,true);
 h.bridge.replayMode=true;h.hud.reconcileToolCapabilities();h.elements.previewWallAssist.listeners.change({target:{checked:false}});
 assert.equal(h.arena.previewWallAssist,true);assert.equal(h.elements.previewWallAssistLabel.classList.contains('preview-control-unavailable'),true);
});
test('neutral selection has no inspector or arena placement side effects',()=>{
 const h=harness();local(h);h.click('toolSelect');h.mouse();assert.equal(h.arena.foodItems.length,0);assert.equal(h.arena.alarms.length,0);assert.equal(h.arena.predators.length,0);assert.equal(h.commands.length,0);
});
