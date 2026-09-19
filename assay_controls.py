"""Validated live controls. Unsupported preview controls never acknowledge success."""
import math
from assay_response import MODEL_VERSION

LIMITS = {
    'open_arena': 'Odor navigation and feeding. The 2D motor model is not flight biomechanics.',
    't_maze': 'Odor associations affect steering. Continuous arm occupancy is not an independent choice trial.',
    'y_maze': 'Spontaneous exploration and alternation measurement. No learned working-memory task is implemented.',
    'heat_maze': 'Local thermal escape/search reflex. Visual place memory and landmark-based goal learning are not implemented.',
    'buridan': 'Innate stripe orientation and boundary avoidance. Fixation is not learned navigation.',
    'visual_operant': 'Closed-loop drum and heat-evoked yaw. Pattern-specific operant memory is not implemented.',
    'wind_tunnel': 'Bilateral odor sampling and upwind response. The default plume is stationary, not turbulent.',
    'looming_escape': 'A single expanding disk evokes a brief escape. Re-trigger the disk for another presentation.',
    'optomotor': 'Tethered yaw follows grating motion. Position should stay fixed.',
    'gap_crossing': 'Explicit threshold controller: pause, cross or turn away. This is not learned planning or leg reach biomechanics.',
    'circadian_dam': 'Activity/immobility monitor only. Light is recorded; no endogenous circadian oscillator is implemented.',
    'courtship': 'Simplified pheromone approach/avoidance. Courtship memory and a moving female are not implemented.',
    'labyrinth': 'Odor-guided exploration and wall avoidance. No stored map or multi-step planner.',
    'multisensory': 'Odor, thermal and wind inputs drive movement. Limb kinematics and the composite score remain illustrative.',
}


def describe(arena):
    p = arena.paradigm
    key = arena.paradigm_key(p) if p else 'open_arena'
    params, actions = [], []
    def param(name, label, value, low, high, step, unit):
        params.append(dict(name=name,label=label,value=value,min=low,max=high,step=step,unit=unit))
    def action(name, label):
        actions.append(dict(name=name,label=label))
    if key == 'open_arena':
        param('windStrength','Airflow toward −x',-arena.wind[0],0,40,1,'mm/s')
        action('food','Place food near fly')
    if key == 't_maze': action('reverse_arms','Reverse reward / shock arms')
    if key == 'heat_maze':
        param('floorTemp','Floor temperature',p.peltier.baseline_temp,24,44,.5,'°C')
    if key == 'buridan':
        action('rotate_stripes','Rotate stripes 90°')
        action('contrast','Toggle stripe contrast')
    if key == 'visual_operant':
        action('reverse_heat','Reverse heat sectors')
    if key == 'wind_tunnel':
        param('windVelocity','Airflow toward −x',-p.wind_flow[0],0,40,1,'mm/s')
        param('plumeWidth','Plume standard deviation',p.filament_sigma,1,15,.5,'mm')
        action('shift_plume','Shift plume laterally')
    if key == 'looming_escape': action('loom','Present looming disk again')
    if key == 'optomotor':
        param('patternSpeed','Grating velocity',p.drum_velocity_deg_s,-120,120,10,'°/s')
        action('reverse_grating','Reverse grating direction')
        action('contrast','Toggle grating contrast')
    if key == 'gap_crossing':
        param('gapWidth','Gap width (reset before changing near edge)',p.gap_width_mm,1.5,5.5,.25,'mm')
    if key == 'courtship': action('receptivity','Toggle virgin / mated female')
    return dict(model_version=MODEL_VERSION,limitation=LIMITS[key],parameters=params,actions=actions)


def set_parameter(arena, name, value):
    allowed = {p['name']:p for p in describe(arena)['parameters']}
    if name not in allowed: raise ValueError(f'Parameter {name!r} is not connected to this live assay')
    spec=allowed[name]
    value=float(value)
    if not math.isfinite(value) or not spec['min'] <= value <= spec['max']:
        raise ValueError(f'{name} must be between {spec["min"]} and {spec["max"]}')
    p=arena.paradigm
    if name=='windStrength': arena.wind=(-value,0.0)
    elif name=='floorTemp': p.peltier.baseline_temp=value
    elif name=='windVelocity': p.wind_flow=(-value,0.0)
    elif name=='plumeWidth': p.filament_sigma=value
    elif name=='patternSpeed': p.drum_velocity_deg_s=value
    elif name=='gapWidth':
        # Changing support while crossing would invalidate the trajectory.
        if 42 < arena.fly.pos.x < 45 + p.gap_width_mm + 2:
            raise ValueError('Reset the fly before changing the gap underneath it')
        p.gap_width_mm=value
        p.zones[1].bounds=(45.0,0.0,45.0+value,20.0)
        p.zones[2].bounds=(45.0+value,7.5,100.0,12.5)
        p.decision_outcome=None
    return value


def act(arena, name):
    if name not in {a['name'] for a in describe(arena)['actions']}:
        raise ValueError(f'Action {name!r} is not connected to this live assay')
    p=arena.paradigm
    if name=='food':
        from arena import Position
        f=arena.fly
        x=max(4,min(arena.width-4,f.pos.x+8*math.cos(f.heading)))
        y=max(4,min(arena.height-4,f.pos.y+8*math.sin(f.heading)))
        arena.food_positions.append(Position(x,y));arena.odor_a.add_source(x,y,1.0)
    elif name=='reverse_arms':
        p.cs_plus_arm='arm_b' if p.cs_plus_arm=='arm_a' else 'arm_a'
        for zone in p.zones:
            if zone.name in ('arm_a','arm_b'):
                zone.reward=float(zone.name==p.cs_plus_arm);zone.punishment=1-zone.reward
    elif name=='rotate_stripes':
        cx,cy=p.center
        for lm in p.landmarks:
            x,y=lm.pos;lm.pos=(cx-(y-cy),cy+(x-cx))
    elif name=='contrast':
        attr='stripe_contrast' if hasattr(p,'stripe_contrast') else 'contrast'
        setattr(p,attr,0.0 if getattr(p,attr)>0 else 1.0)
    elif name=='reverse_heat': p.invert_sectors=not p.invert_sectors
    elif name=='shift_plume':
        p.nozzle_pos=(180.0,45.0 if p.nozzle_pos[1]<=30 else 15.0)
        p.zones[0].bounds=(*p.nozzle_pos,6.0)
    elif name=='loom':
        # Restart the stimulus clock only. Pose, dataset segment and assay time stay intact.
        p.stimulus_started_ms=p.time_elapsed_ms;p.escape_initiated=False;p.gf_spike=False
    elif name=='reverse_grating': p.drum_velocity_deg_s=-p.drum_velocity_deg_s
    elif name=='receptivity': p.female_type='virgin' if p.female_type=='mated' else 'mated'
