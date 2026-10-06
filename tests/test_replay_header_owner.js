'use strict';
const test=require('node:test');
const assert=require('node:assert/strict');
const fs=require('node:fs');
const path=require('node:path');
const vm=require('node:vm');
const base=process.env.NEUROFLY_HEADER_SOURCE_ROOT||path.resolve(__dirname,'..');
const app=fs.readFileSync(path.join(base,'web/app.js'),'utf8');
const replay=fs.readFileSync(path.join(base,'web/replay.js'),'utf8');
function method(name,next){
 const start=app.indexOf(`    ${name}(`),end=app.indexOf(`    ${next}(`,start+1);
 assert.ok(start>=0&&end>start,`${name} exists`);return app.slice(start,end);
}
function declaration(name){
 const start=app.indexOf(`function ${name}(`),end=app.indexOf('\n}',start);
 assert.ok(start>=0&&end>start);return app.slice(start,end+2);
}
function harness(){
 const elements=new Map();
 const element=id=>{if(!elements.has(id))elements.set(id,{textContent:'',title:'',style:{},value:'1',hidden:false,
  options:[.5,1,100].map(value=>({value:String(value)})),querySelectorAll:()=>[],appendChild(option){this.options.push(option);}});return elements.get(id);};
 const context={window:{location:{search:''}},document:{readyState:'loading',activeElement:null,addEventListener(){},getElementById:id=>id==='activityCanvas'||id==='rasterCanvas'?null:element(id),createElement:()=>({dataset:{}})},
  requestAnimationFrame:()=>1,cancelAnimationFrame(){},performance:{now:()=>0},console};
 vm.createContext(context);vm.runInContext(replay,context);
 vm.runInContext([declaration('validateRequestedSpeed'),declaration('requestedSpeedText'),declaration('formatSimSpeed')].join('\n'),context);
 const playback=app.includes('    renderPlaybackState(')?method('renderPlaybackState','enterReplay'):
  `renderPlaybackState(pkt) {${app.slice(app.indexOf('            this.arena.paradigmStatus = pkt.error'),app.indexOf('            const panelCaps = graphPanelCapabilities(pkt);'))}}`;
 vm.runInContext(`window.headerMethods={${playback},${method('renderTiming','scheduleReconnect')},${method('speedFeedback','reconcileRequestedSpeed')},${method('reconcileRequestedSpeed','async setSpeed')},${method('async setSpeed','reconcileBackendSelector')}}`,context);
 const methods=context.window.headerMethods;
 const hud={simSpeed:100,speedOptions:[.5,1,100],...methods};
 const bridge={replayMode:true,connected:false,arena:{},hud,renderPlaybackState:methods.renderPlaybackState,renderTiming:methods.renderTiming,
  handleDaemonPacket(pkt){this.renderPlaybackState(pkt);this.renderTiming(pkt);},resetReplayView(){},exitReplay(){this.replayMode=false;}};
 hud.daemonBridge=bridge;context.window.neuroflyDiagnostics={bridge};
 const player=context.window.neuroflyReplay;
 player.rec={header:{version:1,provenance:{assay:'t-maze',params:{dt_s:.02}},channels:{}},end:{frames_sha256:'abc'},
  frames:[{step:0,sim_time_s:0,paused:true,paradigm:'t-maze',segment_id:'a',trial:1,fly:{x:0,y:0,state:'FORWARD'}},
   {step:133,sim_time_s:2.64,paused:false,paradigm:'t-maze',segment_id:'a',trial:1,fly:{x:1,y:0,state:'FORWARD'}}]};
 player.name='header-only-fixture';player.buildIdentity();player.times=[0,2.64];player.playhead=0;
 element('speedFeedback').textContent='Requested speed set to 100x.';
 return {player,bridge,hud,element};
}
function buttons(h,label){assert.equal(h.element('btnPauseToggle').textContent,label);assert.equal(h.element('btnReplayPlay').textContent,label);}

test('advancing playback owns header despite historical paused first frame; frames remain unchanged',()=>{
 const h=harness(),before=JSON.stringify(h.player.rec.frames);h.player.play();h.player.apply(0);
 buttons(h,'Pause');assert.equal(h.element('arenaRunState').textContent,'REPLAY · PLAYING');
 assert.equal(h.element('statAchieved').textContent,'1x');assert.match(h.element('statAchieved').title,/Replay.*not measured/);
 assert.equal(h.player.packetFor(0).paused,true);assert.equal(JSON.stringify(h.player.rec.frames),before);
 assert.match(app,/this\.renderPlaybackState\(pkt\);/);
});
test('end-of-playback and seek synchronize both controls without changing playback pause',()=>{
 const h=harness();h.player.play();h.player.playhead=2.63;h.player.lastWall=0;h.player.loop(100);
 assert.equal(h.player.playing,false);buttons(h,'Play');assert.equal(h.element('arenaRunState').textContent,'REPLAY · ENDED');
 assert.equal(h.element('statAchieved').textContent,'paused');
 h.player.seekFraction(0);buttons(h,'Play');assert.equal(h.element('arenaRunState').textContent,'REPLAY · PAUSED');
 h.player.seekFraction(1);buttons(h,'Play');assert.equal(h.element('arenaRunState').textContent,'REPLAY · ENDED');
 h.player.toggle();h.player.apply(0);buttons(h,'Pause');assert.equal(h.element('arenaRunState').textContent,'REPLAY · PLAYING');
 h.player.seekFraction(.5);buttons(h,'Pause');assert.equal(h.player.playing,true);
 h.player.toggle();buttons(h,'Play');assert.equal(h.element('arenaRunState').textContent,'REPLAY · PAUSED');
 assert.match(app,/if \(hud\.daemonBridge\?\.replayMode\) \{\s*window\.neuroflyReplay\?\.toggle\(\)/);
 assert.match(replay,/btnReplayPlay.*addEventListener\('click', \(\) => player\.toggle\(\)/);
});
test('replay speed surface replaces stale live feedback, header speed surface agrees, exit clears replay feedback',async()=>{
 const h=harness();h.player.play();h.player.apply(0);h.player.setSpeed(.5);
 assert.equal(h.hud.simSpeed,.5);assert.equal(h.element('selectSpeed').value,'0.5');
 assert.equal(h.element('statAchieved').textContent,'0.5x');assert.equal(h.element('speedFeedback').textContent,'Replay playback speed: 0.5x.');
 await h.hud.setSpeed(2);assert.equal(h.player.speed,2);assert.equal(h.hud.simSpeed,2);assert.match(h.element('speedFeedback').textContent,/Replay.*2x/);
 h.player.exit();assert.equal(h.bridge.replayMode,false);assert.equal(h.element('speedFeedback').textContent,'');
});
test('pending live speed acknowledgement cannot overwrite active replay controls or feedback',async()=>{
 const h=harness();let resolve;h.bridge.replayMode=false;h.bridge.connected=true;
 h.bridge.sendCommand=()=>new Promise(done=>{resolve=done;});const pending=h.hud.setSpeed(100);
 h.bridge.replayMode=true;h.player.play();h.player.apply(0);h.player.setSpeed(.5);
 resolve({status:'ok',sim_speed:100});assert.equal(await pending,false);
 assert.equal(h.player.speed,.5);assert.equal(h.hud.simSpeed,.5);assert.equal(h.element('speedFeedback').textContent,'Replay playback speed: 0.5x.');buttons(h,'Pause');
});
test('live pause status and acknowledged speed continue to use the daemon',async()=>{
 const h=harness();h.bridge.replayMode=false;h.bridge.connected=true;
 h.bridge.renderPlaybackState({paused:true});assert.equal(h.element('btnPauseToggle').textContent,'Resume');assert.equal(h.element('arenaRunState').textContent,'PAUSED');
 h.bridge.sendCommand=async()=>({status:'ok',sim_speed:7.5});assert.equal(await h.hud.setSpeed(7.5),true);
 assert.equal(h.hud.simSpeed,7.5);assert.equal(h.element('speedFeedback').textContent,'Requested speed set to 7.5x.');
});
