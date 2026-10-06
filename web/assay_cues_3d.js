/* Illustrative floor projection of supplied cue locations, never a physics field. */
(function(root) {
'use strict';
const point = p => Array.isArray(p) && p.length >= 2 && p.slice(0,2).every(Number.isFinite);
const finite = Number.isFinite;
function describe(arena, geometry, offset = [0,0], transport = {}) {
    const pid = arena.activeParadigmId, pkt = arena.remotePacket;
    const remote = !!(arena.remoteDriven || arena.awaitingDaemon || pkt);
    const current = pkt && (pkt.paradigm || '').toLowerCase().replace(/_/g,'-') === pid && !arena.awaitingDaemon;
    const scene = current ? pkt.scene || {} : {}, stim = current ? pkt.stimuli || {} : {};
    const p = remote ? {} : arena.paradigmState || {};
    if (transport.switchPending) return {cues:[], unavailable:['switch pending'], status:'3D cues unavailable · switch pending'};
    if (transport.ownerProblem) return {cues:[], unavailable:['acknowledged packet owner'],
        status:'3D cues unavailable · awaiting acknowledged owner · '+transport.ownerProblem};
    const cues = [], unavailable = [];
    const displayLimit = 256;
    const add = (kind, pos, label, color, extra={}) => {
        if (!point(pos)) return false;
        if(cues.length>=displayLimit) {if(!unavailable.includes('additional markers (display limit256)'))unavailable.push('additional markers (display limit256)');return false;}
        cues.push({kind, x:pos[0]+(remote?offset[0]:0), y:pos[1]+(remote?offset[1]:0), label, color, ...extra});
        return true;
    };
    const need = (ok, name) => {if(!ok)unavailable.push(name);};
    if (!geometry?.available || (remote && !current)) {
        return {cues:[], unavailable:['current assay cue data'], status:'3D cues unavailable · awaiting current assay data'};
    }
    const center = [(geometry.bounds.minX+geometry.bounds.maxX)/2,
        (geometry.bounds.minY+geometry.bounds.maxY)/2];
    const localCenter = remote ? center.map((v,i)=>v-offset[i]) : center;
    const size = Math.min(geometry.width,geometry.depth);
    const list = (values,label,color) => {
        if (!Array.isArray(values)) {unavailable.push(label+' positions');return;}
        if(values.length>displayLimit)unavailable.push(label+' additional positions (display limit256)');
        values.slice(0,displayLimit).forEach(v=>{if(!add('marker',v,label,color))unavailable.push('malformed '+label+' position');});
    };
    if (remote) {
        list(scene.food,'Food / odor A','#4ade80');
        list(scene.hazards,'Hazard / odor B','#fb7185');
        list(scene.predators,'Predator','#ef4444');
    } else if (pid==='open-arena') {
        list((arena.foodItems||[]).map(v=>[v.x,v.y]),'Food / odor A','#4ade80');
        list((arena.alarms||[]).filter(v=>v.life>0).map(v=>[v.x,v.y]),'Alarm / odor B','#fb7185');
        list((arena.predators||[]).filter(v=>v.active).map(v=>[v.x,v.y]),'Predator','#ef4444');
    }
    if(['open-arena','wind-tunnel','multisensory-sandbox'].includes(pid)) {
        const wind=remote?[pkt.sensory?.wind_x,pkt.sensory?.wind_y]:arena.windVector;
        if(point(wind))add('arrow',localCenter,`Sampled wind ${Math.hypot(...wind).toFixed(1)} mm/s`,'#38bdf8',
            {dx:wind[0],dy:wind[1],radius:size*.15});
        else unavailable.push('sampled wind');
    }
    switch(pid) {
    case 't-maze': {
        const arm=remote?scene.cs_plus_arm:p.csPlusArm;
        need(arm==='arm_a'||arm==='arm_b','CS+ arm');
        if(!remote && (arm==='arm_a'||arm==='arm_b')) {
            add('marker',[arm==='arm_a'?15:125,50],'CS+ odor','#4ade80');
            add('marker',[arm==='arm_a'?125:15,50],'CS− / shock region','#fb7185');
        } else if(arm==='arm_a'||arm==='arm_b') add('label',localCenter,`CS+ ${arm} · shock coordinates unavailable`,'#fbbf24');
        break;
    }
    case 'y-maze':
        if(remote) {
            if(!Array.isArray(scene.arm_tips))unavailable.push('arm-tip cue annotations');
            else {
                if(scene.arm_tips.length>displayLimit)unavailable.push('additional arm annotations (display limit256)');
                scene.arm_tips.slice(0,displayLimit).forEach(tip=>need(
                    typeof tip?.name==='string' && add('marker',tip.position,tip.name,'#38bdf8'),'arm-tip cue annotation'));
            }
        }
        else (p.armTips||[]).forEach((v,i)=>add('marker',v,`Arm ${i}`,'#38bdf8'));
        break;
    case 'heat-maze':
        need(add('disk',remote?scene.refuge_pos:p.refugePos,'Cool refuge','#22d3ee',
            {radius:remote?scene.refuge_radius:p.refugeRadius}),'refuge position');
        unavailable.push('spatial temperature field');break;
    case 'buridan': {
        const landmarks=remote?scene.landmarks:[[110,60],[10,60]];
        const contrast=remote?scene.stripe_contrast:1;
        if(!finite(contrast))unavailable.push('stripe contrast');
        else if(contrast!==0)list(landmarks,'Visual stripe','#e2e8f0');
        break;
    }
    case 'visual-operant':case 'optomotor': {
        const angle=remote?scene.drum_angle_deg:p.drumAngleDeg;
        const contrast=remote?stim.contrast:p.contrast;
        const invert=remote?scene.invert_sectors:false;
        const polarityAvailable=!remote || typeof invert==='boolean';
        need(finite(angle),'drum angle');
        if(pid==='visual-operant')need(polarityAvailable,'operant sector polarity');
        if(pid==='optomotor')need(finite(contrast),'grating contrast');
        if(pid==='optomotor' && finite(contrast) && !finite(angle))
            add('label',localCenter,`Grating contrast ${contrast} · phase unavailable`,'#e2e8f0');
        if(finite(angle)&&(pid!=='optomotor'||finite(contrast))&&(pid!=='visual-operant'||polarityAvailable))
            add('ring',localCenter,pid==='optomotor'?'Grating projection':'Operant sector projection','#e2e8f0',
                {radius:size*.35,angle,segments:pid==='optomotor'?24:4,contrast:pid==='optomotor'?contrast:1,invert:!!invert,operant:pid==='visual-operant'});
        break;
    }
    case 'wind-tunnel': {
        const pos=remote?scene.nozzle_pos:p.nozzlePos, sigma=remote?scene.filament_sigma:p.filamentSigma;
        need(add('marker',pos,'Odor nozzle','#4ade80'),'nozzle position');
        if(point(pos)&&finite(sigma)&&sigma>0)add('plume',pos,'Plume width ±σ schematic','#4ade80',{sigma});
        else unavailable.push('plume width');break;
    }
    case 'looming-escape': {
        const theta=remote?(stim.looming_angle_deg??stim.theta_deg):p.thetaDeg;
        need(finite(theta)&&theta>=0&&theta<180,'looming angle');
        if(finite(theta)&&theta>=0&&theta<180)add('disk',localCenter,`Loom θ ${theta.toFixed(1)}°`,'#fb7185',
            {radius:Math.min(size*.4,Math.tan(theta*Math.PI/360)*size*.3)});
        break;
    }
    case 'courtship':need(add('marker',remote?scene.female_pos:p.femalePos,
        `Female (${remote?scene.female_type||'type unavailable':p.femaleType||'type unavailable'})`,'#f472b6'),'female position');break;
    case 'labyrinth':need(add('marker',remote?scene.goal_pos:p.goalPos,'Goal odor','#4ade80'),'goal position');break;
    case 'multisensory-sandbox':
        if(remote) {
            [['food_pos','Food / odor A','#4ade80'],['repellent_pos','Repellent / odor B','#fb7185'],
                ['pheromone_pos','cVA pheromone','#c084fc'],['hotspot_pos','Hotspot location','#f97316'],['cool_pos','Cool location','#22d3ee']]
                .forEach(([key,label,color])=>need(add('marker',scene[key],label,color),label+' position'));
            list(scene.pillar_centers,'Visual pillar','#38bdf8');
            unavailable.push('spatial temperature field');
        }
        else {
            [['foodPos','Food / odor A','#4ade80'],['repellentPos','Repellent / odor B','#fb7185'],
                ['pheromonePos','cVA pheromone','#c084fc'],['hotspotPos','Hotspot','#f97316'],['coolPos','Cool refuge','#22d3ee']]
                .forEach(([key,label,color])=>need(add('marker',p[key],label,color),label+' position'));
            list(p.pillars,'Visual pillar','#38bdf8');
        }break;
    case 'circadian-dam':unavailable.push('beam positions and tube activity cue map');break;
    case 'gap-crossing':break; // Gap surfaces are already rendered from physical geometry.
    }
    // Invalid radii never inherit a preview/default physical footprint.
    const valid=cues.filter(c=>{if(c.kind==='disk'&&(!finite(c.radius)||c.radius<0)) {
        unavailable.push(c.label+' radius');return false;}return true;});
    return {cues:valid,unavailable,status:`3D cues · ${remote?'recorded/daemon packet':'standalone preview configuration'} · illustrative floor projection; marker/pattern sizes are display aids`
        +(unavailable.length?` · unavailable: ${[...new Set(unavailable)].join(', ')}`:'')};
}
function paint(ctx, descriptor, geometry, resolution=1024) {
    const b=geometry.bounds,w=b.maxX-b.minX,h=b.maxY-b.minY;
    ctx.clearRect(0,0,resolution,resolution);
    const xy=c=>[(c.x-b.minX)/w*resolution,(b.maxY-c.y)/h*resolution];
    for(const c of descriptor.cues) {
        const [x,y]=xy(c);ctx.fillStyle=c.color;ctx.strokeStyle=c.color;ctx.globalAlpha=.85;
        if(c.kind==='arrow') {
            const magnitude=Math.hypot(c.dx,c.dy);
            if(magnitude>0) {
                const ex=x+c.dx/magnitude*c.radius/w*resolution,ey=y-c.dy/magnitude*c.radius/h*resolution;
                ctx.lineWidth=3;ctx.beginPath();ctx.moveTo(x,y);ctx.lineTo(ex,ey);ctx.stroke();
                const angle=Math.atan2(ey-y,ex-x);ctx.beginPath();ctx.moveTo(ex,ey);
                ctx.lineTo(ex-9*Math.cos(angle-.4),ey-9*Math.sin(angle-.4));ctx.lineTo(ex-9*Math.cos(angle+.4),ey-9*Math.sin(angle+.4));ctx.closePath();ctx.fill();
            }
        } else if(c.kind==='ring') {
            const r=c.radius/w*resolution;
            ctx.save();ctx.translate(x,y);ctx.scale(1,w/h);
            for(let i=0;i<c.segments;i++) {
                const a=(c.angle+i*360/c.segments)*Math.PI/180;
                ctx.strokeStyle=c.contrast===0?'#64748b':c.operant?(((i%2===1)!==c.invert)?'#fb7185':'#4ade80'):(i%2?'#e2e8f0':'#020617');
                ctx.lineWidth=10;ctx.beginPath();ctx.arc(0,0,r,-a-2*Math.PI/c.segments,-a);ctx.stroke();
            }
            ctx.restore();
        } else if(c.kind==='plume') {
            ctx.globalAlpha=.2;const left=0, width=x-left;
            ctx.fillRect(left,y-c.sigma/h*resolution,width,2*c.sigma/h*resolution);
        } else if(c.kind!=='label') {
            const r=c.kind==='disk'?c.radius/w*resolution:8;
            ctx.globalAlpha=c.kind==='disk'?.3:.9;ctx.beginPath();ctx.ellipse(x,y,r,c.kind==='disk'?c.radius/h*resolution:r,0,0,2*Math.PI);ctx.fill();
        }
        ctx.globalAlpha=1;ctx.font='14px sans-serif';ctx.textAlign='center';ctx.fillStyle='#f8fafc';
        ctx.fillText(c.label,Math.max(90,Math.min(resolution-90,x)),Math.max(18,Math.min(resolution-8,y-12)));
    }
    ctx.globalAlpha=1;
}
root.NeuroFlyAssayCues3D={describe,paint};
if(typeof module!=='undefined')module.exports=root.NeuroFlyAssayCues3D;
})(typeof window!=='undefined'?window:globalThis);
