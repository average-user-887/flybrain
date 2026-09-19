'use strict';
// Live controls and plots are generated from the daemon's actual capabilities.
// Preview-only buttons must never claim to alter a remotely controlled fly.
window.mountLiveAssay = function(hud, panel) {
 const arena=hud.arena, initial=arena.remotePacket, capability=initial.live_assay;
 const esc=s=>String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
 const card=(title,body)=>`<div class="hud-card" style="font-size:11px;line-height:1.5"><div class="hud-card-header">${title}</div>${body}</div>`;
 panel.innerHTML=card('LIVE MODEL · '+esc(capability.model_version),`<p style="color:#fbbf24;font-size:11px;line-height:1.5">${esc(capability.limitation)}</p>`)
 +card('Connected parameters',capability.parameters.map(p=>`<label class="param-slider-row">${esc(p.label)} <b id="live_value_${p.name}">${p.value}</b> ${esc(p.unit)}<input type="range" id="live_param_${p.name}" min="${p.min}" max="${p.max}" step="${p.step}" value="${p.value}"></label>`).join('')||'<p>No adjustable model parameters in this assay.</p>')
 +card('Live interventions',capability.actions.map(a=>`<button class="stim-btn" id="live_action_${a.name}">${esc(a.label)}</button>`).join('')+`<p id="liveCommandStatus" role="status">Applied interventions are recorded with brain ID and simulation time. Unconnected preview controls are unavailable here.</p>`)
 +card('Measured motor response','<p>Blue: speed (mm/s) · Pink: yaw (rad/s). Samples from the current segment; these curves are measurements.</p><canvas id="liveMotorChart" width="380" height="130" style="width:100%;height:130px"></canvas>')
 +card('Actual assay measurements','<div id="liveMetrics"></div>');
 const status=panel.querySelector('#liveCommandStatus');
 const send=async(action,params)=>{
  status.textContent='Applying…';
  const result=await hud.daemonBridge.sendCommand(action,params);
  status.textContent=result?.status==='ok'?'Applied and recorded: '+(result.applied||action):(result?.message||'Not applied: daemon unavailable or read only.');
  status.style.color=result?.status==='ok'?'#4ade80':'#fbbf24';
 };
 capability.parameters.forEach(p=>panel.querySelector('#live_param_'+p.name).addEventListener('change',e=>send('set_param',{name:p.name,value:Number(e.target.value)})));
 capability.actions.forEach(a=>panel.querySelector('#live_action_'+a.name).addEventListener('click',()=>send('assay_action',{name:a.name})));
 let lastStep=null,segment=null,samples=[];
 hud.activeAssayUpdater=()=>{
  const t=arena.remotePacket;if(!t?.live_assay)return;
  const readonly=!hud.daemonBridge.connected||hud.daemonBridge.readOnly||hud.daemonBridge.switchPending;
  panel.querySelectorAll('button,input').forEach(e=>e.disabled=readonly);
  for(const p of t.live_assay.parameters){
   const input=panel.querySelector('#live_param_'+p.name),value=panel.querySelector('#live_value_'+p.name);
   if(input&&document.activeElement!==input)input.value=p.value;
   if(value)value.textContent=p.value;
  }
  if(t.segment_id!==segment){segment=t.segment_id;samples=[];lastStep=null;}
  if(t.step!==lastStep){samples.push([t.fly.speed,t.descending.dna02_yaw]);samples=samples.slice(-120);lastStep=t.step;}
  const entries=Object.entries(t.metrics||{});
  panel.querySelector('#liveMetrics').innerHTML=(entries.length?entries:[['state',t.fly.state],['sim_time_s',t.sim_time_s]]).map(([k,v])=>`<div style="display:flex;justify-content:space-between;gap:8px;font-size:10px;padding:3px"><span>${esc(k)}</span><b>${v===null?'Not observed':esc(typeof v==='number'?Number(v.toFixed(4)):String(v))}</b></div>`).join('');
  if(panel.style.display==='none')return;
  const canvas=panel.querySelector('#liveMotorChart'),ctx=canvas.getContext('2d');ctx.clearRect(0,0,380,130);
  const max=Math.max(1,...samples.flat().map(Math.abs));
  ctx.strokeStyle='#334155';ctx.beginPath();ctx.moveTo(30,65);ctx.lineTo(375,65);ctx.stroke();
  ctx.fillStyle='#94a3b8';ctx.font='9px monospace';ctx.fillText('+'+max.toFixed(1),0,15);ctx.fillText('-'+max.toFixed(1),0,125);
  ['#38bdf8','#f472b6'].forEach((color,j)=>{ctx.strokeStyle=color;ctx.beginPath();samples.forEach((s,i)=>{const x=30+345*i/Math.max(1,samples.length-1),y=65-55*s[j]/max;i?ctx.lineTo(x,y):ctx.moveTo(x,y);});ctx.stroke();});
 };
};
