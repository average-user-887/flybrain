'use strict';
const $ = id => document.getElementById(id);
const api = (new URLSearchParams(location.search).get('daemon') || 'http://127.0.0.1:8781').replace(/\/+$/, '');
$('arenaLink').href = `index.html?daemon=${encodeURIComponent(api)}`;
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
let current = null, telemetry = null, catalog = [], trail = [], lastTrial = null, lastBrain = null, writable = false, busy = false;
const escapeText = value => String(value).replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const num = (v, n=3) => Number.isFinite(v) ? v.toFixed(n) : '—';
const message = (text, error=false) => { $('message').textContent=text; $('message').classList.toggle('error', error); };
async function request(path, body) {
 const res=await fetch(api+path, {method:body?'POST':'GET', headers:body?{'Content-Type':'application/json'}:undefined, body:body?JSON.stringify(body):undefined, signal:AbortSignal.timeout(5000)});
 const data=await res.json();
 if(!res.ok || data.status==='error') throw new Error(data.message || data.error || `HTTP ${res.status}`);
 return data;
}
function buttons() {
 for(const id of ['teach','reverse','probe','freeze','save']) $(id).disabled=!writable||busy||(!!current?.teaching&&['teach','reverse','freeze'].includes(id));
 $('export').disabled=!current;
 for(const button of $('experiments').querySelectorAll('button')) button.disabled=!writable||busy;
}
async function command(body) {
 busy=true;buttons();
 try {const response=await request('/api/command',body);message(body.action==='probe_brain' ? `Probe recorded. A − B = ${num(response.probe.discrimination)}. Weights were unchanged.` : `Recorded: ${body.action.replaceAll('_',' ')}.`);await refresh();}
 catch(e){message(e.message,true);}finally{busy=false;buttons();}
}
for(const [pid,n] of Object.entries(notes)) {
 const b=document.createElement('button');b.id=`select-${pid}`;b.textContent=n[0];
 const span=document.createElement('span');span.textContent=n[1];b.append(span);
 b.addEventListener('click',()=>command({action:'switch_paradigm',paradigm:pid}));$('experiments').append(b);
}
$('teach').onclick=()=>command({action:'teach_brain',pairs:8});
$('reverse').onclick=()=>command({action:'teach_brain',pairs:8,reverse:true});
$('probe').onclick=()=>command({action:'probe_brain'});
$('freeze').onclick=()=>command({action:'set_learning',enabled:!current.learning_enabled});
$('save').onclick=()=>command({action:'save_checkpoint',label:'research_ui'});
$('export').onclick=()=>{
 const blob=new Blob([JSON.stringify({exported_at:new Date().toISOString(),brain:current,telemetry,catalog},null,2)],{type:'application/json'});
 const url=URL.createObjectURL(blob),a=document.createElement('a');a.href=url;a.download=`neurofly-${current.paradigm}-${current.brain_id.slice(0,8)}.json`;a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);
};
function lineChart(id, series, xlabel) {
 const svg=$(id), values=series.flatMap(s=>s.values).filter(Number.isFinite);
 if(!values.length){svg.innerHTML='<text x="280" y="105" text-anchor="middle" fill="#71847a" font-size="12">Awaiting recorded measurements</text>';return;}
 let low=Math.min(0,...values),high=Math.max(0,...values);if(high-low<.02){low-=.1;high+=.1;}const padding=(high-low)*.12;low-=padding;high+=padding;
 const count=Math.max(...series.map(s=>s.values.length));const X=i=>48+i/Math.max(1,count-1)*490,Y=v=>170-(v-low)/(high-low)*150;
 let html='';for(let i=0;i<4;i++){const v=low+(high-low)*i/3,y=Y(v);html+=`<path d="M48 ${y}H538" stroke="#e5ebe6"/><text x="40" y="${y+4}" text-anchor="end" font-size="10" fill="#71847a">${v.toFixed(2)}</text>`;}
 for(const s of series){let segment=[];for(let i=0;i<=s.values.length;i++){const v=s.values[i];if(Number.isFinite(v)){segment.push(`${X(i)},${Y(v)}`);html+=`<circle cx="${X(i)}" cy="${Y(v)}" r="2.8" fill="${s.color}"/>`;}else if(segment.length){html+=`<polyline points="${segment.join(' ')}" fill="none" stroke="${s.color}" stroke-width="2"/>`;segment=[];}}}
 html+=`<text x="48" y="190" font-size="10" fill="#71847a">1</text><text x="538" y="190" text-anchor="end" font-size="10" fill="#71847a">${count}</text><text x="290" y="206" text-anchor="middle" font-size="10" fill="#71847a">${escapeText(xlabel)}</text>`;svg.innerHTML=html;
}
function render() {
 const b=current,n=notes[b.paradigm];$('title').textContent=n[0];$('subtitle').textContent=n[1]+' · One experiment, one retained learning memory.';
 for(const [id,text] of [['question',n[2]],['description',n[3]],['interpretation',n[4]],['control',n[5]]])$(id).textContent=text;
 $('brainId').textContent=b.brain_id.slice(0,12);$('brainId').title=b.brain_id;
 $('seed').textContent=`Seed ${b.seed} · ${b.restored?'restored from disk':'independent instance'}`;
 $('trials').textContent=b.trials;$('steps').textContent=b.steps.toLocaleString()+' simulation steps';$('delta').textContent=num(b.weight_change_l2);$('discrimination').textContent=num(b.probe.discrimination);
 $('phase').textContent=b.teaching ? `Teaching pair ${b.teaching.pair+1}/${b.teaching.pairs} · arena paused` : `Arena running · learning ${b.learning_enabled?'enabled':'frozen'}`;
 $('freeze').textContent=b.learning_enabled?'Freeze learning':'Enable learning';$('freeze').setAttribute('aria-pressed',String(!b.learning_enabled));
 for(const p of Object.keys(notes))$(`select-${p}`).classList.toggle('active',p===b.paradigm);
 const measurements=b.history.filter(e=>e.probe);
 lineChart('probeChart',[{color:'#087d68',values:measurements.map(e=>e.probe.A.valence)},{color:'#bd4e76',values:measurements.map(e=>e.probe.B.valence)}],'Recorded measurement (recent window)');
 const trials=b.history.filter(e=>e.kind==='trial');lineChart('trialsChart',[{color:'#087d68',values:trials.map(e=>e.metric)}],'Completed arena trial (recent window)');
 const last=trials.at(-1);$('metricCaption').textContent=last ? `${last.metric_name || 'No scalar metric available'} · Latest end reason: ${last.reason}. Separate teaching readouts from behavioral outcomes.` : 'No completed arena measurements yet. Teaching pairs are tracked separately.';
 let heat='';const colors=v=>v<1?`rgb(${Math.round(225-(1-v)*190)},${Math.round(237-(1-v)*104)},${Math.round(229-(1-v)*110)})`:`rgb(${Math.round(225+(v-1)*15)},${Math.round(237-(v-1)*100)},${Math.round(229-(v-1)*177)})`;
 for(let row=0;row<2;row++){heat+=`<text x="15" y="${28+row*72}" font-size="11" fill="#61736f">${row?'Avoidance output':'Approach output'}</text>`;b.weights.forEach((w,i)=>{heat+=`<rect x="${15+i*4.4}" y="${39+row*72}" width="4" height="45" fill="${colors(w[row])}"><title>KC ${i+1}, weight ${w[row].toFixed(4)}</title></rect>`;});}
 heat+='<text x="15" y="187" font-size="10" fill="#61736f">KC 1</text><text x="540" y="187" text-anchor="end" font-size="10" fill="#61736f">KC 120</text>';$('weights').innerHTML=heat;
 const max=Math.max(1,...catalog.map(x=>x.steps||0));$('brainBars').innerHTML=catalog.map(x=>`<div class="bar-row"><span>${escapeText(notes[x.paradigm][0])}</span><span class="bar-track"><span class="bar-fill" style="display:block;width:${100*(x.steps||0)/max}%"></span></span><span>${x.steps?.toLocaleString()||'—'}</span></div>`).join('');
 $('events').innerHTML=b.history.slice(-12).reverse().map(e=>`<tr><td>${new Date(e.timestamp*1000).toLocaleTimeString()}</td><td>${escapeText(e.kind.replaceAll('_',' '))}</td><td>${e.brain_step}</td><td>${e.probe ? 'A − B = '+num(e.probe.discrimination) : e.metric!==undefined ? num(e.metric) : e.learning_enabled===false ? 'Frozen control' : '—'}</td></tr>`).join('');
 renderTrajectory();buttons();
}
function renderTrajectory() {
 const b=current,t=telemetry;if(!t?.fly)return;
 if(lastBrain!==b.brain_id||lastTrial!==t.trial){trail=[];lastBrain=b.brain_id;lastTrial=t.trial;}
 trail.push([t.fly.x,t.fly.y]);trail=trail.slice(-400);
 const bounds=t.world_bounds||[0,0,140,100], [x0,y0,x1,y1]=bounds,scale=Math.min(560/(x1-x0),220/(y1-y0));
 const X=x=>300+(x-(x0+x1)/2)*scale,Y=y=>130-(y-(y0+y1)/2)*scale;
 let svg='<rect x="0" y="0" width="600" height="260" rx="8" fill="#f5f8f4"/>';
 for(let i=0;i<=8;i++)svg+=`<path d="M${20+i*70} 20V240" stroke="#e4ece4"/>`;
 for(const w of b.walls||[])svg+=`<path d="M${X(w[0][0])} ${Y(w[0][1])}L${X(w[1][0])} ${Y(w[1][1])}" stroke="#567a64" stroke-width="2"/>`;
 svg+=`<polyline points="${trail.map(([x,y])=>`${X(x)},${Y(y)}`).join(' ')}" fill="none" stroke="#57a388" stroke-width="1.8"/>`;
 const f=t.fly;svg+=`<g transform="translate(${X(f.x)},${Y(f.y)}) rotate(${-f.heading*180/Math.PI})"><ellipse rx="7" ry="4" fill="#173f33"/><path d="M4 0H13M0 -3L-3 -9M0 3L-3 9" stroke="#173f33" stroke-width="1.5"/></g>`;
 $('trajectory').innerHTML=svg;$('position').textContent=`x ${num(f.x,1)} · y ${num(f.y,1)} mm`;
}
async function refresh() {
 try{
 const [status,brain,t,all]=await Promise.all([request('/api/status'),request('/api/brain'),request('/api/telemetry'),request('/api/brains')]);
 if(t.brain_id && t.brain_id!==brain.brain_id)return;
 const first=!current;current=brain;telemetry=t;catalog=all.brains;writable=!status.stream?.read_only && !status.stream?.commands_require_token;
 $('connection').textContent=`Live · ${status.sim_speed}× simulated time${writable?'':' · read only'}`;$('signal').classList.add('live');
 if(first)message('Connected. Brains are isolated by experiment. Select an assay, teach its cues, then probe the retained memory.');render();
 }catch(e){writable=false;$('connection').textContent='Disconnected · measurements may be stale';$('signal').classList.remove('live');message(`Cannot reach ${api}. ${e.message}`,true);buttons();}
}
(async()=>{await refresh();setInterval(refresh,1200);})();
