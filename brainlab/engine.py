"""Compiled, all-edge LIF simulation. Only incoming currents can drive neurons.

Same membrane/synapse constants as the reference Shiu-like probe. Analytic
subthreshold integration, threshold check each 0.1 ms, 1.8 ms transmission delay.
Retina and lamina use a DECLARED coarse spiking approximation to graded cells.
This does not model realistic ion channels, receptors, or learning.
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

