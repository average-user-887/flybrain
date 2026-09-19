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
function buttons() {
 for(const key of ['Teach','Reverse','Probe','Freeze','Save']) $('training'+key).disabled=!writable||busy||(!!snapshot?.brain.teaching&&['Teach','Reverse','Freeze'].includes(key));
 $('trainingExport').disabled=!snapshot;
}
function endpoint() { return window.app?.hud?.daemonBridge?.activeUrl; }
async function request(path, body) {
 const api=endpoint(); if(!api) throw new Error('No daemon connection. Click the connection indicator to retry.');
 const response=await fetch(api+path,{method:body?'POST':'GET',headers:body?{'Content-Type':'application/json'}:undefined,body:body?JSON.stringify(body):undefined,signal:AbortSignal.timeout(4000)});
 const data=await response.json(); if(!response.ok||data.status==='error') throw new Error(data.message||data.error||`HTTP ${response.status}`);
 return data;
}
async function command(body) {
 busy=true;buttons();
 try { await request('/api/command',body);message('Recorded: '+body.action.replaceAll('_',' ')); }
 catch(error){message(error.message,true);}finally{busy=false; await refresh();buttons();}
}
$('trainingTeach').onclick=()=>command({action:'teach_brain',pairs:8});
$('trainingReverse').onclick=()=>command({action:'teach_brain',pairs:8,reverse:true});
$('trainingProbe').onclick=()=>command({action:'probe_brain'});
$('trainingFreeze').onclick=()=>command({action:'set_learning',enabled:!snapshot.brain.learning_enabled});
$('trainingSave').onclick=()=>command({action:'save_checkpoint',label:'instrument'});
$('trainingExport').onclick=()=>{
 const blob=new Blob([JSON.stringify({exported_at:new Date().toISOString(),scope:'Compact modular model. Arena outcomes and controlled odor probes are separate measurements.',...snapshot,research},null,2)],{type:'application/json'});
 const url=URL.createObjectURL(blob),a=document.createElement('a');a.href=url;a.download=`neurofly-${snapshot.brain.paradigm}-${snapshot.brain.brain_id.slice(0,8)}.json`;a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);
};
function render() {
 const b=snapshot.brain,n=notes[b.paradigm];
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
function renderResearch() {
 if(!research)return;
 const age=Date.now()/1000-research.updated_at;
 $('researchStatus').textContent=`${age>120?'Worker heartbeat stale':research.state} · ${research.completed} paired cohorts recorded · ${research.active||'waiting'} · updated ${new Date(research.updated_at*1000).toLocaleTimeString()}`;
 $('researchProtocol').textContent=research.protocol;
 const rows=research.recent||[],same=rows.filter(r=>r.paradigm===snapshot?.brain.paradigm && r.status==='complete');
 lineChart('researchChart',[{color:'#38bdf8',values:same.map(r=>r.trained.metric)},{color:'#f472b6',values:same.map(r=>r.frozen.metric)}],'Independent seed cohort (blue trained, pink frozen)');
 $('researchOutcome').textContent=same.length?`${same.at(-1).metric_name} · ${same.length} recent paired seeds. Raw differences are exploratory; no learning claim or significance test is implied.`:'No held-out cohort recorded for this assay yet.';
 $('researchRows').innerHTML=rows.slice(-14).reverse().map(r=>`<tr><td>${escapeText(r.paradigm)} / ${r.seed}</td><td>${r.status==='complete'?number(r.trained.metric):'FAILED'}</td><td>${r.status==='complete'?number(r.frozen.metric):escapeText(r.error||'')}</td></tr>`).join('');
}
async function refresh() {
 if(refreshing||busy)return;refreshing=true;
 try {
  const data=await request('/api/observatory');
  if(data.brain.brain_id!==data.telemetry.brain_id) throw new Error('Inconsistent brain snapshot; retrying.');
  if (!connected) message('Connected. Live controls act on this retained brain; each experiment keeps its own memory.');
  connected=true; snapshot=data;writable=!data.status.stream?.read_only&&!data.status.stream?.commands_require_token;
  $('trainingConnection').textContent=`Live daemon · ${data.status.sim_speed}×${writable?'':' · read only'}`;
  render();
 }catch(error){connected=false;writable=false;$('trainingConnection').textContent='Disconnected · last measurements may be stale';message(error.message,true);buttons();}
 finally{refreshing=false;}
}
async function pollResearch(){
 try{const r=await fetch('research-status.json',{cache:'no-store',signal:AbortSignal.timeout(3000)});if(!r.ok)throw new Error();research=await r.json();renderResearch();}
 catch(e){$('researchStatus').textContent='Research worker is not reporting yet.';}
}
window.addEventListener('load',()=>{refresh();pollResearch();setInterval(refresh,1500);setInterval(pollResearch,10000);});
})();
