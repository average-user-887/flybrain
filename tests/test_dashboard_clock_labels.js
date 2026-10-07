'use strict';
// Navbar and trial clock readouts name the clock they show (daemon session,
// recorded session, preview, retained graph) and show Unknown only when the
// daemon's trial clock says so.
const test=require('node:test'),assert=require('node:assert/strict'),fs=require('node:fs'),path=require('node:path'),vm=require('node:vm');
const app=fs.readFileSync(path.join(__dirname,'../web/app.js'),'utf8');
const html=fs.readFileSync(path.join(__dirname,'../web/index.html'),'utf8');
const start=app.indexOf('const TRIAL_CLOCK_REASONS'),end=app.indexOf('class ScientificBioArena',start);
assert.ok(start>0&&end>start,'clock readout source block not found');
const c={};vm.createContext(c);vm.runInContext(app.slice(start,end)+';globalThis.readouts=clockReadouts;',c);
const view={simTime:0.02,stepCount:1,currentTrial:1,paradigmElapsedSec:0.02};
const known={schema:'neurofly.assay-trial-clock.v1',elapsed_s:4.36,current_trial:7,elapsed_known:true,trial_known:true,reason:null};
const legacy={schema:'neurofly.assay-trial-clock.v1',elapsed_s:0.02,current_trial:1,elapsed_known:false,trial_known:false,reason:'checkpoint_predates_trial_clock'};

test('after a restart the session clock is labelled as such and the retained brain time is shown apart',()=>{
 const pkt={sim_time_s:0.02,step:1,clocks:{session:{scope:'daemon_session',step:1,elapsed_s:0.02},trial:known,
  graph:{scope:'retained_graph_instance',step:380,elapsed_s:7.6}}};
 const r=c.readouts({...view,currentTrial:7,paradigmElapsedSec:4.36},pkt,true,false);
 assert.equal(r.timeLabel,'Session time');assert.equal(r.stepLabel,'Session step');
 assert.match(r.sessionTitle,/starts again from zero when the daemon restarts/);
 assert.equal(r.simTime,'0.02s');assert.equal(r.step,'1');
 assert.equal(r.graphVisible,true);assert.equal(r.graphTime,'7.60s');assert.match(r.graphTitle,/step 380/);
 assert.match(r.graphTitle,/not the trial clock/);
 assert.equal(r.trial,'#7');assert.equal(r.guideTrial,'TRIAL #7');assert.equal(r.trialTitle,'');
 assert.equal(r.elapsed,'4.36s');assert.equal(r.elapsedTitle,'');
});

test('a legacy unknown trial clock is shown as Unknown live and in a replay that recorded it',()=>{
 for(const replay of [false,true]){
  const r=c.readouts(view,{clocks:{trial:legacy}},true,replay);
  assert.equal(r.trial,'Unknown');assert.equal(r.guideTrial,'TRIAL UNKNOWN');assert.equal(r.elapsed,'Unknown');
  assert.match(r.trialTitle,/saved before trial timing was recorded/);
  assert.match(r.elapsedTitle,/unknown until a new trial starts/);assert.match(r.elapsedTitle,/At least 0\.02s/);
  assert.equal(r.timeLabel,replay?'Recorded session time':'Session time');
  assert.equal(r.stepLabel,replay?'Recorded session step':'Session step');
 }
 // Elapsed known again after a new trial; the ordinal stays unknown.
 const r=c.readouts(view,{clocks:{trial:{...legacy,elapsed_known:true}}},true,false);
 assert.equal(r.elapsed,'0.02s');assert.equal(r.trial,'Unknown');
});

test('packets and recordings without clocks keep their recorded values, labelled by source',()=>{
 const rec=c.readouts({...view,simTime:47.06,stepCount:2353,currentTrial:3,paradigmElapsedSec:4.34},{sim_time_s:47.06,trial_elapsed_s:4.34},true,true);
 assert.equal(rec.timeLabel,'Recorded session time');assert.match(rec.sessionTitle,/as recorded/);
 assert.equal(rec.simTime,'47.06s');assert.equal(rec.step,'2353');
 assert.equal(rec.elapsed,'4.34s');assert.equal(rec.trial,'#3');assert.equal(rec.graphVisible,false);
 const live=c.readouts(view,{sim_time_s:0.02},true,false);
 assert.equal(live.timeLabel,'Session time');assert.equal(live.elapsed,'0.02s');assert.equal(live.trial,'#1');
});

test('a preview is not labelled as a daemon or graph clock, even with a stale packet',()=>{
 const r=c.readouts(view,{clocks:{trial:legacy,graph:{step:1,elapsed_s:1}}},false,false);
 assert.equal(r.timeLabel,'Preview time');assert.equal(r.stepLabel,'Preview step');
 assert.equal(r.graphVisible,false);assert.equal(r.trial,'#1');assert.equal(r.elapsed,'0.02s');
});

test('the dashboard renders every readout from clockReadouts',()=>{
 for(const id of ['statSessionClock','statSessionStep','statSimTimeLabel','statStepLabel','statSimTime','statStep',
                  'statGraphTimeBox','statGraphTime','paradigmTrial','guideTrialBadge','paradigmElapsed'])
  assert.ok(html.includes(`id="${id}"`)||app.includes(`id="${id}"`),`missing element ${id}`);
 assert.ok(!/>Sim Time:/.test(html),'unscoped Sim Time label remains');
 const update=app.slice(app.indexOf('const clocks = clockReadouts('),app.indexOf("setText('paradigmElapsed'"));
 assert.ok(update.includes('this.daemonBridge?.replayMode'));
 for(const id of ['statSimTimeLabel','statStepLabel','statSimTime','statStep','statGraphTime','paradigmTrial','guideTrialBadge'])
  assert.ok(update.includes(`'${id}'`),`${id} is not rendered from clockReadouts`);
});

test('a restarted measurement window is shown apart from the continuing trial clock',()=>{
 const lineage={segment_id:'child',parent_segment_id:'parent',reason:'daemon_restart',
  parent_observation:'interrupted_incomplete_no_terminal',restored_trial_elapsed_s:0.08,
  observation_window:'restarted',metric_accumulators:'not_restored'};
 const pkt={clocks:{trial:{...known,elapsed_s:0.1,current_trial:1},
  observation:{scope:'observation_segment',segment_id:'child',segment_elapsed_s:0.02,effective_window_s:0.2,lineage}}};
 const r=c.readouts({...view,paradigmElapsedSec:0.1},pkt,true,false);
 assert.equal(r.elapsed,'0.10s');assert.equal(r.windowVisible,true);
 assert.equal(r.window,'Window 0.02s · restarted after daemon restart');
 assert.match(r.windowTitle,/interrupted and recorded as incomplete/);
 assert.match(r.windowTitle,/not carried over/);assert.match(r.windowTitle,/continues from 0\.08s/);
 const replay=c.readouts({...view,paradigmElapsedSec:0.1},pkt,true,true);
 assert.equal(replay.window,r.window);
 const switched=c.readouts(view,{clocks:{observation:{segment_elapsed_s:0,lineage:{...lineage,reason:'assay_reactivated'}}}},true,false);
 assert.match(switched.window,/restarted after assay switch/);assert.match(switched.windowTitle,/ended when you switched/);
 for(const p of [{clocks:{observation:{segment_elapsed_s:1,lineage:null}}},{}])
  assert.equal(c.readouts(view,p,true,false).windowVisible,false);
 assert.equal(c.readouts(view,pkt,false,false).windowVisible,false);
 assert.ok(html.includes('id="paradigmWindow"')&&html.includes('>Trial elapsed<'));
 assert.ok(app.includes("getElementById('paradigmWindow')"));
});
