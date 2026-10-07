'use strict';
(() => {
const $ = id => document.getElementById(id);
const notes = {
 'open-arena':['Open arena','Foraging & learned odor value','Does experience alter the value of a food-associated cue?','Reward and hazard encounters can train the odor pathway while the fly explores. Movement also depends on innate steering and hunger.','Measure held-out cue preference after conditioning. Distance traveled alone is not learning.','Freeze learning, match the seed and exposures, then compare cue readouts.'],
 't-maze':['T-maze','Differential odor conditioning','Can the brain distinguish a rewarded cue from a punished one?','The two odor identities enter separate input channels. Each brain retains its own associations when you leave this assay.','Read A and B separately. Their difference can change before a behavioral choice improves.','Reverse A+/B− to A−/B+ and watch whether the stored association changes direction.'],
 'y-maze':['Y-maze','Exploration & spontaneous alternation','How often does the fly choose a different arm?','Alternation is a movement statistic. The odor teaching chamber independently checks that this experiment’s mushroom body can store a cue association.','Do not interpret alternation as proof of associative memory; geometry and steering can produce it.','Compare the same geometry with learning frozen; retain all arm choices.'],
 'heat-maze':['Heat maze','Thermal refuge search','How quickly does the fly reach a cool refuge?','The assay combines spatial navigation and thermal conditions. The compact brain’s associative pathway currently encodes odors, not a validated visual place-memory representation.','Refuge latency and success are behavioral observations. A cue readout change does not prove place learning.','Keep thermal and spatial conditions fixed; compare fresh and conditioned brains.'],
 'buridan':['Buridan','Landmark fixation','Does the fly orient to opposing visual landmarks?','Visual fixation and wall avoidance contribute to paths on the circular platform. The separate odor probe reports this brain’s associative memory.','Stripe alignment may be an innate response. Look for consistent differences across matched runs.','Compare landmarks present versus absent with the same starting heading.'],
 'visual-operant':['Visual operant','Closed-loop yaw & heat','Does a closed-loop visual consequence change turning?','This is a tethered assay: position can stay fixed while heading or torque changes. A stationary fly here is not a containment failure.','Report yaw and heat exposure separately. Visual operant learning needs its own causal validation.','Compare coupled visual feedback with a replayed, uncoupled stimulus sequence.'],
 'wind-tunnel':['Wind tunnel','Odor plume tracking','How does odor contact organize surge and cast behavior?','Odor and wind engage sensorimotor navigation. Retained cue value can be inspected separately from plume-following reflexes.','Source approach may improve through steering alone. Log plume encounters and cue values alongside latency.','Freeze plasticity and keep the wind and plume configuration unchanged.'],
 'looming-escape':['Looming escape','Visual threat response','When does an expanding visual threat trigger escape?','The assay is primarily a sensorimotor reflex test. The common teaching chamber checks memory in its independent brain without claiming learned escape.','Escape timing and distance are not a general learning score.','Vary threat expansion while matching the starting pose and learning state.'],
 'optomotor':['Optomotor','Gaze stabilization','Does turning compensate for visual motion?','The fly may be tethered while optic flow drives yaw. The arena metric reflects stabilization; the cue probe reflects mushroom-body memory.','A strong reflex is not evidence of a learned association.','Compare opposite grating directions at matched speeds.'],
 'gap-crossing':['Gap crossing','Reachability & tactile probing','When does the fly attempt or abort a crossing?','The runway tests approach and crossing decisions. Its geometry is narrow, so wall contacts are expected.','Separate crossing success, aborts and timeouts. A stopped decision can be meaningful; a physics stall is a bug.','Compare gap widths and starting poses while keeping the brain checkpoint fixed.'],
 'circadian-dam':['Circadian DAM','Activity monitoring','How does activity vary over long periods?','The model tracks movement and beam-break style events. Short runs cannot demonstrate a circadian rhythm.','Use enough simulated hours and label the light schedule. Cue teaching temporarily pauses activity sampling.','Compare matched light schedules and record the full duration, including quiet intervals.'],
 'courtship':['Courtship','Social sensory response','How do social cues alter approach and courtship behavior?','Pheromone and social-response components are simplified. Odor calibration measures stored association, not validated courtship conditioning.','Keep rejection, approach and song-like outputs separate rather than treating one index as cognition.','Compare cue exposure and social state with a frozen-learning control.'],
 'labyrinth':['Labyrinth','Navigation & obstacles','Does experience improve reaching a goal through a maze?','The fly combines learned odor value with local steering and geometry. This model does not yet implement a validated multi-step planning algorithm.','Compare held-out goal latency and success across seeds. Do not substitute weight change for navigation success.','Freeze learning and keep maze, goal, seed and trial duration matched.'],
 'multisensory-sandbox':['Multisensory','Combined sensory benchmark','How do sensory and motor components interact?','This arena combines odor, visual, mechanosensory and locomotor components. Its composite score mixes multiple measures.','Inspect the raw submetrics. A rising composite score alone cannot establish learning.','Change one stimulus or component at a time and retain the before/after probe.']
};
const escapeText = value => String(value).replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
function lineChart(id, series, xlabel) {
 const svg=$(id), values=series.flatMap(s=>s.values).filter(Number.isFinite);
 if(!values.length){svg.innerHTML='<text x="280" y="105" text-anchor="middle" fill="#94a3b8" font-size="12">Awaiting recorded measurements</text>';return;}
 let low=Math.min(0,...values),high=Math.max(0,...values);if(high-low<.02){low-=.1;high+=.1;}const padding=(high-low)*.12;low-=padding;high+=padding;
 const count=Math.max(...series.map(s=>s.values.length));const X=i=>48+i/Math.max(1,count-1)*490,Y=v=>170-(v-low)/(high-low)*150;
 let html='';for(let i=0;i<4;i++){const v=low+(high-low)*i/3,y=Y(v);html+=`<path d="M48 ${y}H538" stroke="#233548"/><text x="40" y="${y+4}" text-anchor="end" font-size="10" fill="#94a3b8">${v.toFixed(2)}</text>`;}
 for(const s of series){let segment=[];for(let i=0;i<=s.values.length;i++){const v=s.values[i];if(Number.isFinite(v)){segment.push(`${X(i)},${Y(v)}`);html+=`<circle cx="${X(i)}" cy="${Y(v)}" r="2.8" fill="${s.color}"/>`;}else if(segment.length){html+=`<polyline points="${segment.join(' ')}" fill="none" stroke="${s.color}" stroke-width="2"/>`;segment=[];}}}
 html+=`<text x="48" y="190" font-size="10" fill="#94a3b8">1</text><text x="538" y="190" text-anchor="end" font-size="10" fill="#94a3b8">${count}</text><text x="290" y="206" text-anchor="middle" font-size="10" fill="#94a3b8">${escapeText(xlabel)}</text>`;svg.innerHTML=html;
}
let snapshot = null, writable = false, busy = false, refreshing = false, research = null, connected = false;
const number = (v, digits=3) => Number.isFinite(v) ? v.toFixed(digits) : '—';
const message = (text, error=false) => { $('trainingMessage').textContent=text; $('trainingMessage').classList.toggle('error',error); };
/** Caption and accessible name of the weight tile follow the data actually drawn in it. */
const weightsLabel = (aria, caption) => { $('trainingWeights').setAttribute('aria-label',aria); $('trainingWeightsCaption').textContent=caption; };
const trainingReplayActive = () => !!window.app?.hud?.daemonBridge?.replayMode;
const trainingWriteAllowed = (stream={}, replayMode=trainingReplayActive()) =>
 !replayMode&&!stream.read_only&&!stream.commands_require_token;
window.neuroflyTrainingWriteAllowed = trainingWriteAllowed;
// Displayed ownership, not the endpoint alone, authorizes measurements and actions.
const ownerFields=['daemon_run_id','activation','run_id','instance_id','assay','backend'];
let trainingOwner=null, trainingGeneration=0, snapshotOwner=null;
function displayedOwner() {
 const hud=window.app?.hud,b=hud?.daemonBridge,p=b?.lastOrderedPacket,i=p?.identity;
 const rejects=window.neuroflyIdentityRejection;
 // Final ACK can precede telemetry, including same-assay switches and trial resets.
 if(typeof rejects!=='function'||[b?.lastSwitchAck,b?.lastBackendAck,b?.lastAck].some(ack=>rejects(p,ack)))return null;
 if(!b?.connected||!b.activeUrl||b.replayMode||b.switchPending||b.switchQueueRunning||b.queuedParadigm||hud.backendCommandWaiting||!i
  ||ownerFields.some(k=>i[k]===undefined||i[k]===null)||!p.brain_id||i.assay!==hud.arena?.activeParadigmId)return null;
 return {bridge:b,url:b.activeUrl,identity:Object.fromEntries(ownerFields.map(k=>[k,i[k]])),brain_id:p.brain_id};
}
function sameOwner(a,b) {return !!a&&!!b&&a.bridge===b.bridge&&a.url===b.url&&a.brain_id===b.brain_id&&ownerFields.every(k=>a.identity[k]===b.identity[k]);}
function ownedSnapshot(){return !!snapshot&&sameOwner(snapshotOwner,displayedOwner());}
function unavailableTraining() {
 const replay=trainingReplayActive(), p=window.app?.hud?.daemonBridge?.lastOrderedPacket;
 $('trainingConnection').textContent=replay?'Replay · recorded provisional evidence · read only':'Training unavailable · waiting for a verified displayed live owner';
 $('trainingTitle').textContent=replay?`${notes[p?.identity?.assay]?.[0]||'Recorded assay'} · replay`:'Training owner unavailable';
 $('trainingBrainId').textContent=replay?(p?.brain_id||'Recording-local brain identity unavailable'):'Unavailable';
 $('trainingPhase').textContent=replay?'Weights, probes and training history are not included in this recording.':'No matching live snapshot available.';
 for(const id of ['Trials','Steps','Delta','Discrimination'])$('training'+id).textContent='Unavailable';
 for(const id of ['Question','Description','Interpretation','Control'])$('training'+id).textContent='';
 for(const id of ['ProbeChart','TrialChart'])lineChart('training'+id,[],'Unavailable for displayed owner');
 $('trainingWeightsTitle').textContent='Weights unavailable';$('trainingWeights').innerHTML='';
 weightsLabel('Weight readout unavailable','No weight readout is available for the displayed owner.');
 $('trainingBrains').innerHTML='';$('trainingEvents').innerHTML='';$('trainingMetric').textContent='No training history for the displayed owner.';
}
// The last command outcome for the displayed owner generation (CARD75). Poll refreshes show it
// instead of the generic capability notice; it is cleared at every owner or replay boundary.
let lastCommand=null;
function commandMessage(text,error=false){lastCommand={generation:trainingGeneration,text,error};message(text,error);}
function clearOwnerMessage(){lastCommand=null;message(trainingReplayActive()?'Replay is read only; no live training command result belongs to this recording.':'Awaiting training measurements for the displayed live owner.');}
function reconcileTrainingOwner() {
 const next=displayedOwner();
 if(!sameOwner(next,trainingOwner)&&(next||trainingOwner)) {trainingOwner=next;trainingGeneration++;refreshing=false;snapshot=null;snapshotOwner=null;writable=false;connected=false;clearOwnerMessage();}
 if(!ownedSnapshot()){writable=false;unavailableTraining();}
 buttons();return next;
}
window.neuroflyTrainingOwnerReconcile=reconcileTrainingOwner;
const capabilities = () => window.neuroflyGraphPanelCapabilities
 ? window.neuroflyGraphPanelCapabilities(snapshot?.telemetry || {identity:{backend:'modular'}})
 : {graph:false,modularMemory:true,teach:true,reverse:true,probe:true,learningControl:true,saveCheckpoint:true,label:'modular'};
function buttons() {
 reconcileResearchDownloads();
 const caps=capabilities(), allowed={Teach:caps.teach,Reverse:caps.reverse,Probe:caps.probe,Freeze:caps.learningControl,Save:caps.saveCheckpoint};
 const reasons={Teach:caps.modularReason,Reverse:caps.modularReason,Probe:caps.modularReason,
  Freeze:caps.graph&&!caps.learningControl?'Unavailable: this graph packet declares no measured plastic subset.':'',Save:''};
 for(const key of ['Teach','Reverse','Probe','Freeze','Save']){
  const button=$('training'+key);
  button.disabled=!ownedSnapshot()||!writable||trainingReplayActive()||busy||!allowed[key]||(!!snapshot?.brain.teaching&&['Teach','Reverse','Freeze'].includes(key));
  button.title=!allowed[key]?(reasons[key]||'Unsupported by the active controller.'):'';
 }
 $('trainingExport').disabled=!ownedSnapshot()||busy;
 $('trainingExport').title=ownedSnapshot()?'Export a JSON report for the displayed owner, not a restorable graph checkpoint.':'Unavailable: no training snapshot for the displayed owner.';
}
function endpoint() { return window.app?.hud?.daemonBridge?.activeUrl; }
async function request(path, body) {
 const api=endpoint(); if(!api) throw new Error('No daemon connection. Click the connection indicator to retry.');
 const response=await fetch(api+path,{method:body?'POST':'GET',headers:body?{'Content-Type':'application/json'}:undefined,body:body?JSON.stringify(body):undefined,signal:AbortSignal.timeout(4000)});
 const data=await response.json(); if(!response.ok||data.status==='error') throw new Error(data.message||data.error||`HTTP ${response.status}`);
 return data;
}
async function command(body) {
 const owner=reconcileTrainingOwner();
 if(!owner||!ownedSnapshot()||!writable||busy){message('Training commands require a verified displayed live owner.',true);return;}
 const generation=trainingGeneration;
 const caps=capabilities(), supported={teach_brain:caps.teach,probe_brain:caps.probe,set_learning:caps.learningControl,save_checkpoint:caps.saveCheckpoint};
 if(!supported[body.action]){message(caps.modularReason||'This action is unsupported by the active controller.',true);buttons();return;}
 if(trainingReplayActive()){
  writable=false;message('Replay is read only. Return to the live stream before sending daemon commands.',true);buttons();return;
 }
 busy=true;buttons();
 try {
  const {action,...params}=body;
  const result=await owner.bridge.sendCommand(action,{...params,expected_owner:{...owner.identity,brain_id:owner.brain_id}},()=>{if(generation===trainingGeneration)message('Queued · awaiting final durable acknowledgement.');});
  if(generation!==trainingGeneration||!sameOwner(owner,displayedOwner()))return;
  if(result?.status!=='ok'||result?.ack?.applied!==true||ownerFields.some(k=>result.ack.identity?.[k]!==owner.identity[k]))throw new Error(result?.message||'No applied final acknowledgement received.');
  const saved=action==='save_checkpoint'&&typeof result.checkpoint==='string'?' · '+result.checkpoint.split(/[\\/]/).pop():'';
  commandMessage('Applied: '+action.replaceAll('_',' ')+saved);
 }
 catch(error){if(generation===trainingGeneration)commandMessage(error.message,true);}finally{busy=false; await refresh();buttons();}
}
$('trainingTeach').onclick=()=>command({action:'teach_brain',pairs:8});
$('trainingReverse').onclick=()=>command({action:'teach_brain',pairs:8,reverse:true});
$('trainingProbe').onclick=()=>command({action:'probe_brain'});
$('trainingFreeze').onclick=()=>command({action:'set_learning',enabled:!snapshot?.brain.learning_enabled});
$('trainingSave').onclick=()=>command({action:'save_checkpoint',label:'instrument'});
$('trainingExport').onclick=()=>{
 reconcileTrainingOwner();if(!ownedSnapshot()||busy){message('Export unavailable for the displayed owner.',true);return;}
 const caps=capabilities();
 const scope=caps.graph?`${caps.label} controls the run. The embedded 120-KC brain snapshot is a retained modular helper and is not graph learning evidence.`:'Compact modular model. Arena outcomes and controlled odor probes are separate measurements.';
 const blob=new Blob([JSON.stringify({exported_at:new Date().toISOString(),scope,...snapshot,research},null,2)],{type:'application/json'});
 const url=URL.createObjectURL(blob),a=document.createElement('a');a.href=url;a.download=`neurofly-${snapshot.brain.paradigm}-${snapshot.brain.brain_id.slice(0,8)}.json`;a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);
};
function render() {
 const b=snapshot.brain,n=notes[b.paradigm];
 const caps=capabilities();
 if(caps.graph){renderGraph(n,caps);return;}
 $('trainingDeltaLabel').firstChild.textContent='Weight change (L2)';
 $('trainingDiscriminationLabel').firstChild.textContent='Cue A − B valence';
 $('trainingWeightsTitle').textContent='120 KC → 2 output synapses';
 weightsLabel('Saved weight heatmap of the 120-KC modular mushroom body',
  'Blue: below baseline · Amber: above baseline. Hover for exact weights. These are the daemon’s saved weights of the 120-KC modular mushroom body. The live cue matrix shows its current activity.');
 $('trainingTitle').textContent=n[0]+' · retained brain';
 $('trainingBrainId').textContent=b.brain_id+' · seed '+b.seed;
 $('trainingPhase').textContent=b.teaching ? `Teaching pair ${b.teaching.pair+1}/${b.teaching.pairs} · arena paused` : `Arena learning ${b.learning_enabled?'enabled':'frozen'}`;
 $('trainingTrials').textContent=b.trials; $('trainingSteps').textContent=b.steps.toLocaleString();
 $('trainingDelta').textContent=number(b.weight_change_l2);$('trainingDiscrimination').textContent=number(b.probe.discrimination);
 $('trainingFreeze').textContent=b.learning_enabled?'Freeze learning':'Enable learning';
 for(const [id,value] of [['Question',n[2]],['Description',n[3]],['Interpretation',n[4]],['Control',n[5]]])$('training'+id).textContent=value;
 const samples=b.history.filter(e=>e.probe),trials=b.history.filter(e=>e.kind==='trial');
 lineChart('trainingProbeChart',[{color:'#38bdf8',values:samples.map(e=>e.probe.A.valence)},{color:'#f472b6',values:samples.map(e=>e.probe.B.valence)}],'Recorded probe (recent window)');
 lineChart('trainingTrialChart',[{color:'#38bdf8',values:trials.map(e=>e.metric)}],'Completed arena trial (recent window)');
 $('trainingMetric').textContent=trials.length?`${trials.at(-1).metric_name||'No scalar outcome'} · end: ${trials.at(-1).reason}. Resets are trial boundaries, not movement.`:'No completed arena trial yet.';
 let heat='';for(let row=0;row<2;row++){
  heat+=`<text x="15" y="${28+row*72}" font-size="13" fill="#94a3b8">${row?'Avoidance':'Approach'}</text>`;
  b.weights.forEach((w,i)=>{const v=Math.max(0,Math.min(2,w[row])),c=v<1?`hsl(198 85% ${18+(1-v)*45}%)`:`hsl(38 90% ${18+(v-1)*45}%)`;
   heat+=`<rect x="${15+i*4.4}" y="${39+row*72}" width="4" height="45" fill="${c}"><title>KC ${i+1}: ${number(w[row],4)}</title></rect>`;});
 }$('trainingWeights').innerHTML=heat+'<text x="15" y="188" font-size="12" fill="#94a3b8">KC 1</text><text x="485" y="188" font-size="12" fill="#94a3b8">KC 120</text>';
 const max=Math.max(1,...snapshot.brains.map(x=>x.steps||0));
 $('trainingBrains').innerHTML=snapshot.brains.map(x=>`<div class="training-bar" title="${escapeText(x.brain_id||x.state)}"><span>${escapeText(notes[x.paradigm][0])}</span><span class="training-bar-track"><span class="training-bar-fill" style="display:block;width:${100*(x.steps||0)/max}%"></span></span><span>${x.steps?.toLocaleString()||'—'}</span></div>`).join('');
 $('trainingEvents').innerHTML=b.history.slice(-12).reverse().map(e=>`<tr><td>${new Date(e.timestamp*1000).toLocaleTimeString()}</td><td>${escapeText(e.kind.replaceAll('_',' '))}</td><td>${e.brain_step}</td></tr>`).join('');
 renderResearch();buttons();
}
function renderGraph(n,caps){
 const t=snapshot.telemetry||{},st=snapshot.status||{},wp6=t.plasticity?.wp6||t.connectome?.wp6;
 $('trainingTitle').textContent=`${n[0]} · ${caps.label}`;
 $('trainingBrainId').textContent=`${t.identity?.instance_id||'instance unavailable'} · run ${t.identity?.run_id||'unavailable'}`;
 $('trainingPhase').textContent=caps.wp6Measured
  ? `Measured WP6 subset · ${wp6.n_edges.toLocaleString()} ER→EPG edges · control ${snapshot.brain.learning_enabled?'enabled':'frozen'}`
  : 'No measured graph-learning subset is declared in the current packet.';
 $('trainingTrials').textContent=Number.isFinite(st.trials_completed)?st.trials_completed:'—';
 $('trainingSteps').textContent=Number.isFinite(st.total_steps)?st.total_steps.toLocaleString():'—';
 $('trainingDeltaLabel').firstChild.textContent='WP6 mean / max ΔW';
 $('trainingDelta').textContent=caps.wp6Measured?`${number(wp6.mean_delta,6)} / ${number(wp6.max_delta,6)}`:'Unavailable';
 $('trainingDiscriminationLabel').firstChild.textContent='Modular cue valence';
 $('trainingDiscrimination').textContent='Unavailable';
 $('trainingFreeze').textContent=snapshot.brain.learning_enabled?'Freeze declared plasticity':'Enable declared plasticity';
 for(const [id,value] of [['Question',n[2]],['Description',`${caps.label} controls this ${n[0]} run. This panel reports only graph-controller measurements declared in the current telemetry packet.`],
  ['Interpretation','Treat the observed arena outcome and graph packet channels as separate evidence. Legacy outcome labels shown elsewhere are not schema 1.2 metric validation.'],
  ['Control',caps.wp6Measured?'Freeze the declared WP6 subset and compare otherwise matched runs.':`Checkpoint saving remains available; ${caps.modularReason}`]]) $('training'+id).textContent=value;
 lineChart('trainingProbeChart',[],'Unavailable for graph controller');
 lineChart('trainingTrialChart',[],'Schema 1.2 observation history is not integrated in this panel');
 $('trainingMetric').textContent='Legacy training history belongs to the retained modular helper and is hidden for this graph run.';
 $('trainingWeightsTitle').textContent=caps.wp6Measured?'Measured WP6 ER→EPG subset':'Graph plasticity readout unavailable';
 $('trainingWeights').innerHTML=`<text x="280" y="100" text-anchor="middle" fill="#94a3b8" font-size="12">${caps.wp6Measured?`${wp6.n_edges.toLocaleString()} edges; mean ΔW ${number(wp6.mean_delta,6)}`:'No measured plastic subset in current telemetry'}</text>`;
 if(caps.wp6Measured) weightsLabel('Measured WP6 ER→EPG subset summary: edge count and mean weight change',
  'Summary of the WP6 ER→EPG subset declared in the current telemetry packet: edge count and mean weight change only. No per-synapse heatmap is drawn.');
 else weightsLabel('No graph weight readout available',
  `No saved weight readout is available: the current telemetry packet declares no measured plastic subset for ${String(caps.label).replace(/\.$/,'')}. The 120-KC modular weights are not shown because they do not belong to this graph run.`);
 $('trainingBrains').innerHTML=`<p style="color:#fbbf24">${caps.modularReason}</p>`;
 $('trainingEvents').innerHTML='<tr><td>—</td><td>Graph event journal not exposed by this panel</td><td>—</td></tr>';
 $('researchStatus').textContent='Background modular training cohorts are not graph-controller evidence.';
 $('researchOutcome').textContent='No graph learning claim is made here.';
 $('researchProtocol').textContent='Schema 1.2 observation history is a separate integration.';
 $('researchRows').innerHTML='';
 lineChart('researchChart',[],'No graph cohort connected');
 if(lastCommand?.generation===trainingGeneration)message(lastCommand.text,lastCommand.error);
 else message(caps.wp6Measured
  ? 'Teach, Reverse and Probe act only on the modular mushroom body and are disabled. Checkpoint save remains available; Freeze controls the measured WP6 subset.'
  : `Teach, Reverse, Probe and learning controls are disabled. ${caps.modularReason} Checkpoint save remains available.`);
 buttons();
}
function renderResearch() {
 if(!research)return;
 if(capabilities().graph){
  $('researchStatus').textContent='Background modular training cohorts are not graph-controller evidence.';
  $('researchOutcome').textContent='No graph learning claim is made here.';
  $('researchProtocol').textContent='Schema 1.2 observation history is a separate integration.';
  $('researchRows').innerHTML='';
  lineChart('researchChart',[],'No graph cohort connected');
  return;
 }
 const age=Date.now()/1000-research.updated_at;
 $('researchStatus').textContent=`${age>120?'Worker heartbeat stale':research.state} · ${research.completed} paired cohorts recorded · ${research.active||'waiting'} · updated ${new Date(research.updated_at*1000).toLocaleTimeString()}`;
 $('researchProtocol').textContent=research.protocol;
 const rows=research.recent||[],same=rows.filter(r=>r.paradigm===snapshot?.brain.paradigm && r.status==='complete');
 lineChart('researchChart',[{color:'#38bdf8',values:same.map(r=>r.trained.metric)},{color:'#f472b6',values:same.map(r=>r.frozen.metric)}],'Paired cohort in time order (blue trained, pink frozen)');
 $('researchOutcome').textContent=same.length?`${same.at(-1).metric_name} · ${same.length} paired observations across ${new Set(same.map(r=>r.seed)).size} seeds. Later rounds retain each brain’s memory; they are repeated measures. Differences are exploratory, with no significance test or learning claim.`:'No held-out cohort recorded for this assay yet.';
 $('researchRows').innerHTML=rows.slice(-14).reverse().map(r=>`<tr><td>${escapeText(r.paradigm)} / seed ${r.seed} / round ${r.round+1}</td><td>${r.status==='complete'?number(r.trained.metric):'FAILED'}</td><td>${r.status==='complete'?number(r.frozen.metric):escapeText(r.error||'')}</td></tr>`).join('');
}
async function refresh() {
 const owner=reconcileTrainingOwner();
 if(!owner||refreshing||busy)return;refreshing=true;
 const generation=trainingGeneration;
 const current=()=>generation===trainingGeneration&&sameOwner(owner,displayedOwner());
 try {
  const data=await request('/api/observatory');
  if(!current())return;
  const i=data.telemetry?.identity;
  if(data.brain?.brain_id!==owner.brain_id||data.telemetry?.brain_id!==owner.brain_id||data.brain?.paradigm!==owner.identity.assay
   ||ownerFields.some(k=>i?.[k]!==owner.identity[k]))throw new Error('Snapshot does not match the displayed owner; waiting for synchronization.');
  if(!connected)message('Verified live owner. Commands and report exports use this displayed owner.');
  connected=true;snapshot=data;snapshotOwner=owner;writable=!owner.bridge.readOnly&&trainingWriteAllowed(data.status?.stream);
  const st=data.status||{},live=st.liveness||{},notAdv=st.status==='error'||['stalled','dead'].includes(live.state);
  $('trainingConnection').textContent=(notAdv?'Daemon connected · SIMULATION NOT ADVANCING':st.paused?'Daemon connected · paused':`Live daemon · ${st.sim_speed}×`)+(st.status==='degraded'?` · NOT SAVING: ${st.persistence?.reason||'write failed'}`:'')+(writable?'':' · read only');
  render();
 }catch(error){if(current()){connected=false;snapshot=null;snapshotOwner=null;writable=false;unavailableTraining();message(error.message,true);buttons();}}
 finally{if(generation===trainingGeneration)refreshing=false;}
}
// Optional files belong to the selected live endpoint, never the page origin.
let researchOwner = null, researchGeneration = 0, researchInFlight = false;
let summaryDownload = {available:false, reason:'not checked'}, bundleDownload = {available:false, reason:'not checked'};
let bundleCheckedAt = null;
function currentResearchOwner() {
 const bridge=window.app?.hud?.daemonBridge;
 if(!bridge?.connected || !bridge.activeUrl || bridge.replayMode) return null;
 return {bridge, url:bridge.activeUrl.replace(/\/+$/, ''), run:bridge.lastOrderedPacket?.run_id || null};
}
function sameResearchOwner(a,b) {
 return !!a && !!b && a.bridge===b.bridge && a.url===b.url && a.run===b.run;
}
function reconcileResearchDownloads() {
 const owner=currentResearchOwner();
 if(!sameResearchOwner(owner,researchOwner) && (owner || researchOwner)) {
  researchOwner=owner;researchGeneration++;researchInFlight=false;bundleCheckedAt=null;
  summaryDownload={available:false,reason:'not checked'};bundleDownload={available:false,reason:'not checked'};
  research=null;
  $('researchRows').innerHTML='';$('researchProtocol').textContent='';
  $('researchOutcome').textContent='No research measurements from this endpoint yet.';
  lineChart('researchChart',[],'Awaiting endpoint measurements');
 }
 const unavailable=trainingReplayActive()?'Replay: optional live research downloads unavailable.'
  : !owner?'Offline/disconnected: optional research downloads unavailable.':null;
 for(const [id,state,file] of [['researchSummaryDownload',summaryDownload,'research-status.json'],['researchBundleDownload',bundleDownload,'research-latest.zip']]) {
  const link=$(id);if(!link)continue;
  const allowed=!!owner && !unavailable && state.available;
  if(allowed)link.setAttribute('href',owner.url+'/'+file);else link.removeAttribute('href');
  link.setAttribute('aria-disabled',String(!allowed));link.setAttribute('tabindex',allowed?'0':'-1');
  link.style.opacity=allowed?'1':'.55';link.style.color=allowed?'#38bdf8':'#94a3b8';
  link.title=allowed?'Verified resource at the selected live endpoint.':unavailable||state.reason;
 }
 const status=$('researchDownloadStatus');
 if(status)status.textContent=unavailable || `Summary: ${summaryDownload.available?'available':summaryDownload.reason}. ZIP: ${bundleDownload.available?'available':bundleDownload.reason}.`+(owner.bridge.readOnly?' Read-only connection: verified downloads remain readable.':'');
 if(!owner)$('researchStatus').textContent=unavailable;
 return owner;
}
function bindResearchDownloads() {
 for(const id of ['researchSummaryDownload','researchBundleDownload']) {
  const link=$(id);if(!link)continue;
  link.addEventListener('click',event=>{
   reconcileResearchDownloads();
   if(link.getAttribute('aria-disabled')!=='false'){event.preventDefault();}
  });
 }
}
async function pollResearch(){
 const owner=reconcileResearchDownloads();
 if(!owner || researchInFlight)return;
 const generation=researchGeneration;researchInFlight=true;
 const current=()=>generation===researchGeneration && sameResearchOwner(owner,currentResearchOwner());
 try {
  const r=await fetch(owner.url+'/research-status.json',{cache:'no-store',redirect:'error',signal:AbortSignal.timeout(3000)});
  if(!r.ok)throw new Error(`unavailable (HTTP ${r.status}); worker not reporting`);
  const data=await r.json();
  if(!Number.isFinite(data.updated_at)||typeof data.state!=='string'||!Number.isInteger(data.completed)||data.completed<0||!Array.isArray(data.recent))throw new Error('unavailable (invalid worker summary)');
  if(!current())return;
  research=data;summaryDownload={available:true};renderResearch();
 } catch(e) {
  if(!current())return;
  research=null;summaryDownload={available:false,reason:e.message||'unavailable; worker not reporting'};
  $('researchStatus').textContent='Research worker is not reporting yet.';
  $('researchRows').innerHTML='';$('researchProtocol').textContent='';$('researchOutcome').textContent='No current research measurements available.';
  lineChart('researchChart',[],'Awaiting recorded measurements');
 }
 try {
  if(!current())return;
  // No ZIP body is fetched for discovery. Servers without HEAD stay unverified.
  if(bundleCheckedAt===null || Date.now()-bundleCheckedAt>=60000) {
   bundleCheckedAt=Date.now();
   const r=await fetch(owner.url+'/research-latest.zip',{method:'HEAD',cache:'no-store',redirect:'error',signal:AbortSignal.timeout(3000)});
   if(!current())return;
   const length=Number(r.headers.get('Content-Length'));
   const type=(r.headers.get('Content-Type')||'').split(';')[0].trim().toLowerCase();
   bundleDownload=r.ok && Number.isSafeInteger(length) && length>0 && ['application/zip','application/x-zip-compressed','application/octet-stream'].includes(type)
    ? {available:true} : {available:false,reason:[405,501].includes(r.status)?'unverified (server does not support HEAD)':`unavailable or unverified (HTTP ${r.status})`};
  }
 } catch(e) {if(current())bundleDownload={available:false,reason:'unavailable (resource check failed)'};}
 finally {if(current()){researchInFlight=false;reconcileResearchDownloads();}}
}
bindResearchDownloads();
reconcileResearchDownloads();
window.addEventListener('neurofly-replay-mode-change',()=>{trainingGeneration++;snapshot=null;snapshotOwner=null;trainingOwner=null;refreshing=false;clearOwnerMessage();reconcileTrainingOwner();if(!trainingReplayActive())pollResearch();});
window.addEventListener('load',()=>{refresh();pollResearch();setInterval(refresh,1500);setInterval(pollResearch,10000);});
})();
