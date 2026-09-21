"""Compiled, all-edge LIF simulation. Only incoming currents can drive neurons.

Three explicitly versioned dynamics live here; all are declared in
``docs/LIF_DYNAMICS_SPEC.md`` and in ``brainlab.graph_identity.DYNAMICS_VERSIONS``.

* ``advance``    -- v1, current-based. Same membrane/synapse constants as the
  reference Shiu-like probe (Shiu et al. 2024). Analytic subthreshold
  integration, threshold check each 0.1 ms, 1.8 ms transmission delay.
  Synaptic input is a voltage-equivalent current, so the membrane potential is
  unbounded: this is the defect WP5 recorded (-200 mV, runaway).
* ``advance_v2`` -- v2, conductance-based with reversal potentials
  (excitatory 0 mV, inhibitory/chloride -70 mV) and a hard floor/ceiling at
  those reversals. Exponential-Euler integration with the conductances frozen
  within each dt. Spike, delay, refractory and reset schedule identical to v1.
  ONE conductance quantum for both signs, calibrated on the excitatory PSP.
* ``advance_v3`` -- v3, the same conductance model with a PER-SIGN
  PSP-preserving calibration: the excitatory quantum reproduces the upstream
  unitary EPSP and the inhibitory quantum reproduces the upstream unitary IPSP
  (1/52 and 1/18 leak units per unit weight). Used together with the v3
  transmitter policy of ``brainlab.transmitter_policy``.

None of them models realistic ion channels, receptors, adaptation or learning.
Retina and lamina use a DECLARED coarse spiking approximation to graded cells.
"""
import math
import time
try:
    from numba import njit
except ImportError:
    def njit(*args, **kwargs):
        def decorator(fn):
            return fn
        return decorator

# Declared constants (docs/LIF_DYNAMICS_SPEC.md §1.1, §3.1).  numba treats
# module globals as compile-time constants, so graph_identity can import these
# and the declared manifest can never drift from the compiled engine.
V_REST_MV = -52.0
V_RESET_MV = -52.0
V_THRESHOLD_MV = -45.0
TAU_M_MS = 20.0
TAU_SYN_MS = 5.0
REFRACTORY_MS = 2.2
DELAY_MS = 1.8
# v2 only:
E_EXC_MV = 0.0            # nicotinic cation channels reverse near 0 mV (Lee & O'Dowd 1999)
E_INH_MV = -70.0          # chloride (Rdl / GluCl / HisCl); ENGINEERING ASSUMPTION
# Conductance quantum per unit of upstream weight, in leak-conductance units.
# Derived, not fitted: makes a unitary excitatory event at rest produce the same
# peak EPSP as v1 (0.275 mV per synapse, Shiu et al. 2024).
G_UNIT_PER_WEIGHT = 1.0 / (E_EXC_MV - V_REST_MV)
# v3 only: the SAME calibration principle applied to each sign separately, so
# that a unitary inhibitory event at rest reproduces the v1 IPSP just as the
# excitatory quantum reproduces the v1 EPSP (docs/LIF_DYNAMICS_SPEC.md §4.2).
# v2 used the excitatory quantum for both signs, which silently shrank the
# unitary IPSP to 18/52 of the value the upstream model specifies.
G_UNIT_EXC_V3 = 1.0 / (E_EXC_MV - V_REST_MV)      # 1/52
G_UNIT_INH_V3 = 1.0 / (V_REST_MV - E_INH_MV)      # 1/18


@njit(cache=True)
def advance(ptr,post,weight,v,g,refractory,drive,queue,queue_count,cursor,steps,dt,counts,active,active_flag,nactive):
    av=math.exp(-dt/20); ag=math.exp(-dt/5)
    coupling=(av-ag)/3
    delay_slots=queue.shape[0]
    for step in range(steps):
        # Delivery occurs after integration/threshold and before reset, matching the
        # reference schedule. A spike at tick t arrives at t+18 for dt=.1.
        slot=cursor%delay_slots
        for k in range(nactive[0]):
            i=active[k]
            if refractory[i]>0: refractory[i]-=1
            if refractory[i]==0:
                v[i]=-52+(v[i]+52)*av+drive[i]*(1-av)+g[i]*coupling
                g[i]*=ag
                if v[i]>-45:
                    counts[i]+=1
                    future=(cursor+int(round(1.8/dt)))%delay_slots
                    queue[future,queue_count[future]]=i
                    queue_count[future]+=1
        for q in range(queue_count[slot]):
            i=queue[slot,q]
            for e in range(ptr[i],ptr[i+1]):
                j=post[e]
                # Brian2's (unless refractory) makes g read-only, including
                # synaptic writes. Do not save arrivals for a later release.
                if refractory[j]>0: continue
                g[j]+=weight[e]
                if active_flag[j]==0:
                    active_flag[j]=1;active[nactive[0]]=j;nactive[0]+=1
        queue_count[slot]=0
        future=(cursor+int(round(1.8/dt)))%delay_slots
        for q in range(queue_count[future]):
            i=queue[future,q];v[i]=-52;g[i]=0;refractory[i]=int(round(2.2/dt))
        cursor+=1
    return cursor


@njit(cache=True)
def advance_v2(ptr,post,weight,v,g,refractory,drive,queue,queue_count,cursor,steps,dt,counts,active,active_flag,nactive,e_inh):
    """v2: conductance-based synapses with reversal potentials.

    ``g`` is (2, n): row 0 excitatory conductance, row 1 inhibitory, both in
    units of the leak conductance.  ``drive`` stays a per-neuron current in
    mV-equivalent units and is shunted by the total conductance, as a current
    injection physically is.  ``e_inh`` is the inhibitory reversal potential
    (default -70 mV; the declared sensitivity arm uses -56 mV).

    The schedule (threshold each dt, delivery 18 ticks later, reset after
    delivery) is identical to ``advance``, so every difference between the two
    versions comes from the synaptic model alone.
    """
    ag=math.exp(-dt/TAU_SYN_MS)
    delay_slots=queue.shape[0]
    delay_ticks=int(round(DELAY_MS/dt))
    refractory_ticks=int(round(REFRACTORY_MS/dt))
    for step in range(steps):
        slot=cursor%delay_slots
        for k in range(nactive[0]):
            i=active[k]
            if refractory[i]>0: refractory[i]-=1
            if refractory[i]==0:
                ge=g[0,i]; gi=g[1,i]
                gtot=1.0+ge+gi
                vinf=(V_REST_MV+ge*E_EXC_MV+gi*e_inh+drive[i])/gtot
                vi=vinf+(v[i]-vinf)*math.exp(-dt*gtot/TAU_M_MS)
                # Bounded by construction; the clamp only guards the external
                # current channel (encoder drive, Kir-like silencing clamp).
                if vi<e_inh: vi=e_inh
                elif vi>E_EXC_MV: vi=E_EXC_MV
                v[i]=vi
                g[0,i]=ge*ag; g[1,i]=gi*ag
                if vi>V_THRESHOLD_MV:
                    counts[i]+=1
                    future=(cursor+delay_ticks)%delay_slots
                    queue[future,queue_count[future]]=i
                    queue_count[future]+=1
        for q in range(queue_count[slot]):
            i=queue[slot,q]
            for e in range(ptr[i],ptr[i+1]):
                j=post[e]
                # Inherited from v1: arrivals at a refractory neuron are dropped.
                if refractory[j]>0: continue
                w=weight[e]
                if w>0.0:
                    g[0,j]+=w*G_UNIT_PER_WEIGHT
                else:
                    g[1,j]-=w*G_UNIT_PER_WEIGHT
                if active_flag[j]==0:
                    active_flag[j]=1;active[nactive[0]]=j;nactive[0]+=1
        queue_count[slot]=0
        future=(cursor+delay_ticks)%delay_slots
        for q in range(queue_count[future]):
            i=queue[future,q];v[i]=V_RESET_MV;g[0,i]=0.0;g[1,i]=0.0;refractory[i]=refractory_ticks
        cursor+=1
    return cursor



@njit(cache=True)
def advance_v3(ptr,post,weight,v,g,refractory,drive,queue,queue_count,cursor,steps,dt,counts,active,active_flag,nactive,e_inh,g_unit_exc,g_unit_inh):
    """v3: v2's conductance model with a per-sign PSP-preserving calibration.

    Identical to ``advance_v2`` in every equation, bound, schedule and reset.
    The only difference is that an excitatory and an inhibitory unit of weight
    open different conductances (``g_unit_exc``, ``g_unit_inh``), each
    calibrated so that the unitary PSP at rest equals the one the upstream
    current-based model specifies for that sign.  See
    ``docs/LIF_DYNAMICS_SPEC.md`` §4.
    """
    ag=math.exp(-dt/TAU_SYN_MS)
    delay_slots=queue.shape[0]
    delay_ticks=int(round(DELAY_MS/dt))
    refractory_ticks=int(round(REFRACTORY_MS/dt))
    for step in range(steps):
        slot=cursor%delay_slots
        for k in range(nactive[0]):
            i=active[k]
            if refractory[i]>0: refractory[i]-=1
            if refractory[i]==0:
                ge=g[0,i]; gi=g[1,i]
                gtot=1.0+ge+gi
                vinf=(V_REST_MV+ge*E_EXC_MV+gi*e_inh+drive[i])/gtot
                vi=vinf+(v[i]-vinf)*math.exp(-dt*gtot/TAU_M_MS)
                if vi<e_inh: vi=e_inh
                elif vi>E_EXC_MV: vi=E_EXC_MV
                v[i]=vi
                g[0,i]=ge*ag; g[1,i]=gi*ag
                if vi>V_THRESHOLD_MV:
                    counts[i]+=1
                    future=(cursor+delay_ticks)%delay_slots
                    queue[future,queue_count[future]]=i
                    queue_count[future]+=1
        for q in range(queue_count[slot]):
            i=queue[slot,q]
            for e in range(ptr[i],ptr[i+1]):
                j=post[e]
                if refractory[j]>0: continue
                w=weight[e]
                if w>0.0:
                    g[0,j]+=w*g_unit_exc
                else:
                    g[1,j]-=w*g_unit_inh
                if active_flag[j]==0:
                    active_flag[j]=1;active[nactive[0]]=j;nactive[0]+=1
        queue_count[slot]=0
        future=(cursor+delay_ticks)%delay_slots
        for q in range(queue_count[future]):
            i=queue[future,q];v[i]=V_RESET_MV;g[0,i]=0.0;g[1,i]=0.0;refractory[i]=refractory_ticks
        cursor+=1
    return cursor
