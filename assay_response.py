"""Explicit, phenomenological assay reflexes for the compact modular model.

These rules connect measured stimuli to motor commands. They are not learned
policies, fitted biological circuits, spatial memory, or a full leg simulator.
Only local sensory measurements are used; no controller knows the goal position.
"""
import math

MODEL_VERSION = 'sensorimotor-v2'


def clamp(value, limit):
    return max(-limit, min(limit, value))


def respond(fly, stimuli, yaw, speed, state, dt):
    drives = {}
    # Opposing dark stripes: orient toward the nearest visible stripe.
    bearings = stimuli.get('stripe_bearings', [])
    contrast = float(stimuli.get('stripe_contrast', 1.0))
    if bearings and contrast > 0:
        drive = 1.5 * contrast * math.sin(min(bearings, key=abs))
        yaw += drive
        drives['stripe_yaw'] = drive

    bearing = stimuli.get('social_bearing_rad')
    if bearing is not None:
        attraction = float(stimuli.get('aphrodisiac_concentration', 0))
        aversion = float(stimuli.get('cva_concentration', 0))
        drive = 3 * (attraction - aversion) * math.sin(bearing)
        # A directly frontal aversive target needs a deterministic turn direction.
        if aversion > .1 and abs(math.sin(bearing)) < .05:
            drive = 1.2 * fly.wall_turn_dir
        yaw += drive
        drives['social_yaw'] = drive
        if max(attraction, aversion) > .08:
            speed = max(.8, speed)
            state = 'SOCIAL_APPROACH' if attraction > aversion else 'SOCIAL_AVOID'
        if attraction > .1 and stimuli.get('inter_fly_distance_mm', 999) < 2:
            speed, state = 0.0, 'COURTSHIP'

    # Heat elicits active search and local turning toward the cooler antenna.
    # Persistent reverse walking in a uniformly heated arena cannot find relief.
    temperature = float(stimuli.get('temperature', 25))
    if temperature > 28:
        gradient = float(stimuli.get('temperature_right', temperature)) - float(stimuli.get('temperature_left', temperature))
        if stimuli.get('laser_active'):
            yaw = 1.5 * fly.wall_turn_dir  # tethered yaw escape, not backward translation
        else:
            yaw = clamp(6 * gradient, 1.8) + .25 * fly.wall_turn_dir
        speed, state = 1.8, 'HEAT_ESCAPE'
        drives['thermal_yaw'] = yaw

    if 'gap_width_mm' in stimuli:
        distance = stimuli['dist_to_gap_mm']
        decision = stimuli.get('decision_outcome')
        if -stimuli['gap_width_mm'] - 2 < distance < 3:
            if stimuli.get('probing_duration_ms', 0) < 300 and distance > 0:
                yaw, speed, state = 0.0, 0.0, 'PROBE'
            elif decision == 'ABORT' and distance > 0:
                error = math.atan2(math.sin(math.pi - fly.heading), math.cos(math.pi - fly.heading))
                yaw = clamp(3 * error, 2.5)
                speed = .9 if abs(error) < .4 else 0.0
                state = 'ABORT'
            elif decision == 'CROSS':
                error = math.atan2(math.sin(-fly.heading), math.cos(-fly.heading))
                yaw, speed, state = clamp(3 * error, 2.5), 1.2, 'CROSS'
            drives['gap_decision'] = decision

    fly.sensorimotor_drives = drives
    return clamp(yaw, 2.5), speed, state
