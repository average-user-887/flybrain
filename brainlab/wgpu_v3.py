"""Isolated optional fixed-weight v3 prototype; not a registered Brain backend.

Ordered delivery rounds each edge to float32 as CPU does, with no fixed-point
reduction. This serial delivery prototype makes no acceleration claim.
"""
from __future__ import annotations
import math
from pathlib import Path
import numpy as np

MAX_COMMAND_TICKS = 128
MAX_ADVANCE_STEPS = 1024
REQUIRED_FEATURES = ('shader-f64', 'shader-int64', 'shader-int64-atomic-all-ops')
FIELDS = ('v', 'ge', 'gi', 'refractory', 'queue', 'queue_count', 'counts',
          'active', 'nactive', 'active_flag', 'spiked', 'drive')
MATH_SOURCE = Path(__file__).with_name('wgpu_v3_math.wgsl').read_text()
DECL = r'''
struct U32 { values: array<u32> };
struct A32 { values: array<atomic<u32>> };
struct U64 { values: array<u64> };
@group(0) @binding(0) var<storage, read> p: U32;
@group(0) @binding(1) var<storage, read> c: U64;
@group(0) @binding(2) var<storage, read_write> s: A32;
@group(0) @binding(3) var<storage, read_write> stage: U64;
@group(0) @binding(4) var<storage, read> ptr: U64;
@group(0) @binding(5) var<storage, read> post: U32;
@group(0) @binding(6) var<storage, read> inc: U64;
fn state_load(o:u32,i:u32)->u32 { return atomicLoad(&s.values[p.values[o]+i]); }
fn put(o:u32,i:u32,v:u32) { atomicStore(&s.values[p.values[o]+i],v); }
fn ff(o:u32,i:u32)->f64 { return f64(bitcast<f32>(state_load(o,i))); }
fn fail() { atomicOr(&s.values[0],1u); }
fn bad(v:f64)->bool { return ((bitcast<u64>(v)>>52u)&0x7fflu)==0x7fflu; }
fn ok()->bool { return atomicLoad(&s.values[0])==0u; }
'''
# p: n,slots,delay,refractory, then twelve state offsets. c:dt,einh,ag.
SETUP = r'''
@compute @workgroup_size(1) fn main() {
 if (!ok()) { return; }
 let n=p.values[0];
 for(var i=0u;i<n;i=i+1u) {
  put(10u,i,0u); put(14u,i,0u);
  if ((state_load(15u,i)&0x7fffffffu)!=0u && state_load(13u,i)==0u) {
   let count=state_load(12u,0u); if(count>=n) { fail(); return; }
   put(11u,count,i);put(12u,0u,count+1u);put(13u,i,1u);
  }
 }
}
'''
PRODUCT = r'''
@compute @workgroup_size(64) fn main(@builtin(global_invocation_id) tid:vec3<u32>) {
 let i=tid.x; if(i>=p.values[0] || !ok()) { return; }
 let a=i*7u; for(var k=0u;k<7u;k=k+1u) { stage.values[a+k]=0lu; }
 if(state_load(13u,i)==0u) { return; }
 var r=bitcast<i32>(state_load(7u,i)); if(r>0) {r=r-1;put(7u,i,bitcast<u32>(r));}
 if(r!=0) {return;}
 let ge=ff(5u,i);let gi=ff(6u,i);let dt=bitcast<f64>(c.values[0]);
 let einh=bitcast<f64>(c.values[1]);
 let gtot=(f64(1.0)+ge)+gi;
 let numerator=((f64(-52.0)+ge*f64(0.0))+gi*einh)+ff(15u,i);
 let vinf=divide_normal_f64_rne(numerator,gtot);
 let x=divide_normal_f64_rne(-dt*gtot,f64(20.0));
 let ex=openlibm_exp_nonpositive(x);let difference=ff(4u,i)-vinf;
 let product=difference*ex;
 if(bad(vinf)||bad(x)||bad(ex)||bad(product)) {fail();return;}
 stage.values[a]=bitcast<u64>(gtot);stage.values[a+1u]=bitcast<u64>(numerator);
 stage.values[a+2u]=bitcast<u64>(vinf);stage.values[a+3u]=bitcast<u64>(x);
 stage.values[a+4u]=bitcast<u64>(ex);stage.values[a+5u]=bitcast<u64>(difference);
 stage.values[a+6u]=bitcast<u64>(product);
}
'''
ADD = r'''
@compute @workgroup_size(64) fn main(@builtin(global_invocation_id) tid:vec3<u32>) {
 let i=tid.x;if(i>=p.values[0] || !ok()) {return;}
 if(state_load(13u,i)==0u || bitcast<i32>(state_load(7u,i))!=0) {return;}
 let a=i*7u;let einh=bitcast<f64>(c.values[1]);
 var vi=bitcast<f64>(stage.values[a+2u])+bitcast<f64>(stage.values[a+6u]);
 if(vi<einh) {vi=einh;} else if(vi>f64(0.0)) {vi=f64(0.0);}
 let ag=bitcast<f64>(c.values[2]);
 put(4u,i,bitcast<u32>(f32(vi)));
 put(5u,i,bitcast<u32>(f32(ff(5u,i)*ag)));
 put(6u,i,bitcast<u32>(f32(ff(6u,i)*ag)));
 if(vi>f64(-45.0)) {put(14u,i,1u);}
}
'''
ENQUEUE = r'''
@compute @workgroup_size(1) fn main() {
 if(!ok()) {return;}
 let n=p.values[0];let future=(atomicLoad(&s.values[1])+p.values[2])%p.values[1];
 for(var k=0u;k<state_load(12u,0u);k=k+1u) {
  let i=state_load(11u,k);if(state_load(14u,i)==0u) {continue;}
  let count=state_load(9u,future);if(count>=n || state_load(10u,i)>=2147483647u) {fail();return;}
  put(8u,future*n+count,i);put(9u,future,count+1u);put(10u,i,state_load(10u,i)+1u);
 }
}
'''
DELIVER = r'''
@compute @workgroup_size(1) fn main() {
 if(!ok()) {return;}
 let n=p.values[0];let slot=atomicLoad(&s.values[1])%p.values[1];
 for(var q=0u;q<state_load(9u,slot);q=q+1u) {
  let i=state_load(8u,slot*n+q);
  for(var e=u32(ptr.values[i]);e<u32(ptr.values[i+1u]);e=e+1u) {
   let j=post.values[e];if(bitcast<i32>(state_load(7u,j))>0) {continue;}
   let increment=bitcast<f64>(inc.values[e]);var o=5u;
   if((inc.values[e] & 0x8000000000000000lu)!=0lu) {o=6u;}
   let v=ff(o,j)+abs(increment);let rounded=f32(v);
   if(bad(v) || ((bitcast<u32>(rounded)>>23u)&255u)==255u) {fail();return;}
   put(o,j,bitcast<u32>(rounded));
   if(state_load(13u,j)==0u) {
    let count=state_load(12u,0u);if(count>=n) {fail();return;}
    put(13u,j,1u);put(11u,count,j);put(12u,0u,count+1u);
   }
  }
 }
 put(9u,slot,0u);
}
'''
RESET = r'''
@compute @workgroup_size(1) fn main() {
 if(!ok()) {return;}
 let n=p.values[0];let cursor=atomicLoad(&s.values[1]);
 let future=(cursor+p.values[2])%p.values[1];
 for(var q=0u;q<state_load(9u,future);q=q+1u) {
  let i=state_load(8u,future*n+q);put(4u,i,bitcast<u32>(f32(-52.0)));
  put(5u,i,0u);put(6u,i,0u);put(7u,i,p.values[3]);
 }
 for(var i=0u;i<n;i=i+1u) {put(14u,i,0u);}
 atomicStore(&s.values[1],cursor+1u);
}
'''
PHASES = dict(setup=SETUP, product=PRODUCT, add=ADD, enqueue=ENQUEUE,
              deliver=DELIVER, reset=RESET)


def layout(n, slots=19):
    lengths = (n,n,n,n,slots*n,slots,n,n,1,n,n,n)
    offsets={};offset=4
    for name,length in zip(FIELDS,lengths,strict=True):
        offsets[name]=offset;offset+=length
    return offsets,offset


def validate_graph(ptr,post,weight,n):
    if not isinstance(n,int) or n<1 or n>2**31-1: raise ValueError('positive bounded n required')
    for a,dtype in ((ptr,np.int64),(post,np.int32),(weight,np.float32)):
        if not isinstance(a,np.ndarray) or a.dtype!=dtype or a.ndim!=1:raise ValueError('CSR dtype/shape')
    if ptr.shape!=(n+1,) or ptr[0]!=0 or ptr[-1]!=len(post) or len(post)!=len(weight):raise ValueError('CSR lengths')
    if len(post)>2**32-1 or np.any(ptr<0) or np.any(ptr>len(post)) or np.any(ptr[1:]<ptr[:-1]) or np.any(post<0) or np.any(post>=n):raise ValueError('CSR bounds')
    if not np.isfinite(weight).all():raise ValueError('finite weights required')


def memory_budget(n,edges,slots=19):
    _,words=layout(n,slots)
    buffers={'params':64,'constants':24,'state':words*4,'stage':n*7*8,
             'ptr':(n+1)*8,'post':max(4,edges*4),'increments':max(8,edges*8)}
    device=sum(buffers.values())
    # Conservative: original graph arrays + calibrated increments + full packed
    # upload copies + one readback snapshot + all device/staging buffer copies.
    original=(n+1)*8+edges*8+n*8
    host_peak=original+edges*8+device+buffers['state']+buffers['stage']
    return {'buffers_bytes':buffers,'device_bytes':device,'host_peak_bytes':host_peak,
            'peak_unified_upper_bytes':host_peak+2*device,'storage_bindings':7,
            'uniform_bytes':0,'staging_upper_bytes':device,'dtype_note':'CSR i64/i32; calibrated signed increments f64; state atomicu32 packed; seven f64 stage words/neuron'}


class WgpuV3State:
    """General graph state. wgpu loads only when a device is explicitly requested.

    Device failures poison the state: no continuation/export until a complete
    valid upload/reset. Input validation happens before submitting work.
    """
    def __init__(self,ptr,post,weight,*,n,e_inh=-70.,g_unit_exc=1/52,
                 g_unit_inh=1/18,dt=.1,delay_slots=19,device=None):
        validate_graph(ptr,post,weight,n)
        if not all(math.isfinite(v) for v in (e_inh,g_unit_exc,g_unit_inh,dt)):raise ValueError('finite constants')
        if (dt,e_inh,g_unit_exc,g_unit_inh,delay_slots)!=(.1,-70.,1/52,1/18,19):
            raise ValueError('prototype supports declared baseline v3 constants only')
        self.n=n;self.dt=dt;self.delay_slots=delay_slots;self.poisoned=False
        self.offsets,self.words=layout(n,delay_slots)
        if self.words>=2**32:raise ValueError('packed state indexing capacity')
        self.loaded=False
        self.cursor=0;self.total_spikes=0;self.sim_ms=0.
        self.budget=memory_budget(n,len(post),delay_slots)
        import wgpu
        if device is None:
            eligible=[a for a in wgpu.gpu.enumerate_adapters_sync()
                      if a.info.get('backend_type')=='Vulkan' and a.info.get('vendor_id')==0x1002
                      and a.info.get('adapter_type')!='CPU' and set(REQUIRED_FEATURES)<=set(a.features)]
            if len(eligible)!=1:raise RuntimeError('one AMD Vulkan f64/int64/atomic adapter required')
            device=eligible[0].request_device_sync(required_features=list(REQUIRED_FEATURES))
        if not set(REQUIRED_FEATURES)<=set(device.features):raise RuntimeError('device features absent')
        self.device=device
        limits=device.limits
        if limits['max-storage-buffers-per-shader-stage']<7:raise ValueError('storage binding limit')
        maxbytes=min(limits['max-buffer-size'],limits['max-storage-buffer-binding-size'])
        if max(self.budget['buffers_bytes'].values())>maxbytes:raise ValueError('buffer binding capacity')
        if (n+63)//64>limits['max-compute-workgroups-per-dimension']:raise ValueError('dispatch capacity')
        self.host=np.zeros(self.words,dtype=np.uint32)
        usage=wgpu.BufferUsage.STORAGE|wgpu.BufferUsage.COPY_SRC|wgpu.BufferUsage.COPY_DST
        params=np.array([n,delay_slots,18,22,*self.offsets.values()],dtype=np.uint32)
        constants=np.array([dt,e_inh,math.exp(-dt/5)],dtype=np.float64).view(np.uint64)
        increments=np.where(weight>0,weight.astype(np.float64)*g_unit_exc,-np.abs(weight.astype(np.float64)*g_unit_inh))
        data=[params,constants,self.host,np.zeros(n*7,dtype=np.uint64),ptr.view(np.uint64),
              post.view(np.uint32) if len(post) else np.zeros(1,dtype=np.uint32),
              increments.view(np.uint64) if len(post) else np.zeros(1,dtype=np.uint64)]
        self.buffers=[device.create_buffer_with_data(data=a,usage=usage) for a in data]
        self.pipelines={};self.groups={}
        for name,body in PHASES.items():
            module=device.create_shader_module(label='g4a-'+name,code=MATH_SOURCE+DECL+body)
            pipeline=device.create_compute_pipeline(label='g4a-'+name,layout='auto',compute={'module':module,'entry_point':'main'})
            # Explicit entries use only reachable globals, as inferred by layout.
            used={'setup':[0,2],'product':[0,1,2,3],'add':[0,1,2,3],
                  'enqueue':[0,2],'deliver':[0,2,4,5,6],'reset':[0,2]}[name]
            entries=[{'binding':i,'resource':{'buffer':self.buffers[i]}} for i in used]
            self.pipelines[name]=pipeline
            self.groups[name]=device.create_bind_group(layout=pipeline.get_bind_group_layout(0),entries=entries)

    def _put(self,name,value):
        a=np.asarray(value);o=self.offsets[name]
        if name in ('v','ge','gi','drive'):a=a.astype(np.float32).view(np.uint32)
        else:a=a.astype(np.int32).view(np.uint32)
        self.host[o:o+a.size]=a.ravel()

    def upload_state(self,v,g,refractory,queue,queue_count,counts,active_flag,
                     *,active=None,nactive=None,cursor=0,total_spikes=0,sim_ms=0.):
        arrays=(v,g,refractory,queue,queue_count,counts,active_flag)
        shapes=((self.n,),(2,self.n),(self.n,),(19,self.n),(19,),(self.n,),(self.n,))
        if any(np.asarray(a).shape!=shape for a,shape in zip(arrays,shapes,strict=True)):raise ValueError('state shape')
        dtypes=(np.float32,np.float32,np.int16,np.int32,np.int32,np.int32,np.uint8)
        if any(np.asarray(a).dtype!=dtype for a,dtype in zip(arrays,dtypes,strict=True)):raise ValueError('state dtype')
        if not np.isfinite(v).all() or not np.isfinite(g).all() or np.any(g<0):raise ValueError('state finite nonnegative g')
        if np.any(refractory<0) or np.any(refractory>22) or np.any(counts<0):raise ValueError('state counters')
        if np.any(queue_count<0) or np.any(queue_count>self.n):raise ValueError('queue capacity')
        if np.any((active_flag!=0)&(active_flag!=1)):raise ValueError('active flags')
        for slot,count in enumerate(queue_count):
            q=np.asarray(queue[slot,:int(count)])
            if np.any(q<0) or np.any(q>=self.n) or len(np.unique(q))!=len(q):raise ValueError('queue members')
        if active is None:active=np.flatnonzero(active_flag).astype(np.int32)
        if np.asarray(active).dtype!=np.int32 or np.asarray(active).ndim!=1 or len(active)>self.n:raise ValueError('active dtype/shape')
        if nactive is None:nactive=len(active)
        membership=np.asarray(active)[:int(nactive)]
        if not isinstance(nactive,(int,np.integer)):raise ValueError('integer active count')
        if int(nactive)<0 or int(nactive)>self.n or len(membership)!=nactive or len(np.unique(membership))!=nactive:
            raise ValueError('active capacity')
        if np.any(membership<0) or np.any(membership>=self.n) or set(map(int,membership))!=set(map(int,np.flatnonzero(active_flag))):raise ValueError('active membership')
        if not isinstance(cursor,int) or not 0<=cursor<=2**32-20:raise ValueError('cursor capacity')
        if not isinstance(total_spikes,int) or total_spikes<0 or not math.isfinite(sim_ms) or sim_ms<0:raise ValueError('state scalars')
        try:
            self.host.fill(0);self.host[1]=cursor
            for name,value in zip(FIELDS[:8],(v,g[0],g[1],refractory,queue,queue_count,counts,active),strict=True):self._put(name,value)
            self._put('nactive',[nactive]);self._put('active_flag',active_flag)
            self.device.queue.write_buffer(self.buffers[2],0,self.host)
            self.device.queue.write_buffer(self.buffers[3],0,np.zeros(self.n*7,dtype=np.uint64))
            self.cursor=cursor;self.total_spikes=total_spikes;self.sim_ms=sim_ms
            self.loaded=True;self.poisoned=False
        except BaseException:
            self.loaded=False;self.poisoned=True
            raise

    def _pass(self,encoder,name,groups=1):
        p=encoder.begin_compute_pass(label=name);p.set_pipeline(self.pipelines[name])
        p.set_bind_group(0,self.groups[name]);p.dispatch_workgroups(groups);p.end()

    def _read(self):
        try:
            self.host=np.frombuffer(self.device.queue.read_buffer(self.buffers[2]),dtype=np.uint32).copy()
            if self.host[0]:raise RuntimeError('device arithmetic/overflow failure; state poisoned')
        except BaseException:
            self.loaded=False;self.poisoned=True
            raise

    def advance(self,drive,cursor,steps,counts_out):
        if not self.loaded or self.poisoned:raise RuntimeError('valid complete upload required')
        if not isinstance(steps,int) or steps<1 or steps>MAX_ADVANCE_STEPS or not isinstance(cursor,int) or cursor!=self.cursor or cursor+steps>2**32-20:raise ValueError('steps/cursor capacity')
        a=np.asarray(drive)
        if a.dtype!=np.float32 or a.shape!=(self.n,) or not np.isfinite(a).all():raise ValueError('finite f32 drive required')
        if counts_out.dtype!=np.int32 or counts_out.shape!=(self.n,) or not counts_out.flags.writeable:raise ValueError('writable counts output required')
        try:
            self._put('drive',a);o=self.offsets['drive']
            self.device.queue.write_buffer(self.buffers[2],o*4,self.host[o:o+self.n])
            # Separate ended compute passes preserve each tick's phase ordering.
            # Bound both command size and queued buffers; setup/count reset occurs
            # once per advance call, including calls crossing command chunks.
            completed=0
            while completed<steps:
                chunk=min(MAX_COMMAND_TICKS,steps-completed)
                e=self.device.create_command_encoder()
                if completed==0:self._pass(e,'setup')
                for _ in range(chunk):
                    for name in ('product','add','enqueue','deliver','reset'):
                        self._pass(e,name,(self.n+63)//64 if name in ('product','add') else 1)
                self.device.queue.submit([e.finish()]);completed+=chunk
            self._read()
            self.cursor=int(self.host[1]);counts_out[:]=self._array('counts',np.int32)
            self.total_spikes+=int(counts_out.astype(np.int64).sum());self.sim_ms+=steps*self.dt
            return self.cursor
        except BaseException:
            self.loaded=False;self.poisoned=True
            raise

    def _array(self,name,dtype,length=None):
        if length is None:length=19*self.n if name=='queue' else 19 if name=='queue_count' else 1 if name=='nactive' else self.n
        o=self.offsets[name];return self.host[o:o+length].view(dtype).copy()

    def snapshot_state(self):
        if not self.loaded or self.poisoned:raise RuntimeError('valid state required')
        self._read()
        return {'v':self._array('v',np.float32),'g':np.array([self._array('ge',np.float32),self._array('gi',np.float32)]),
                'refractory':self._array('refractory',np.int32).astype(np.int16),
                'queue':self._array('queue',np.int32).reshape(19,self.n),'queue_count':self._array('queue_count',np.int32),
                'counts':self._array('counts',np.int32),'active':self._array('active',np.int32),
                'nactive':self._array('nactive',np.int32),'active_flag':self._array('active_flag',np.int32).astype(np.uint8),
                'cursor':self.cursor,'total_spikes':self.total_spikes,'sim_ms':self.sim_ms}

    def download_state(self,v,g,refractory,queue,queue_count,active_flag):
        """Validate all detached destinations before readback or any caller mutation.

        Exact writable aligned C-contiguous ndarrays are required. Destination
        overlap (including internal host storage) and ndarray subclasses refuse;
        callers must not concurrently change destination metadata/storage.
        """
        if not self.loaded or self.poisoned:raise RuntimeError('valid state required')
        names=('v','g','refractory','queue','queue_count','active_flag')
        targets=(v,g,refractory,queue,queue_count,active_flag)
        shapes=((self.n,),(2,self.n),(self.n,),(19,self.n),(19,),(self.n,))
        dtypes=(np.float32,np.float32,np.int16,np.int32,np.int32,np.uint8)
        for target,shape,dtype in zip(targets,shapes,dtypes,strict=True):
            if (type(target) is not np.ndarray or target.shape!=shape or target.dtype!=dtype
                    or not target.flags.writeable or not target.flags.c_contiguous or not target.flags.aligned):
                raise ValueError('exact writable aligned contiguous destination dtype/shape required')
        for i,target in enumerate(targets):
            if np.shares_memory(target,self.host) or any(np.shares_memory(target,prior) for prior in targets[:i]):
                raise ValueError('overlapping/internal destinations unsupported')
        state=self.snapshot_state()
        for name,target in zip(names,targets,strict=True):np.copyto(target,state[name],casting='no')

    def reset_state(self):
        self.upload_state(np.full(self.n,-52,dtype=np.float32),np.zeros((2,self.n),dtype=np.float32),
                          np.zeros(self.n,dtype=np.int16),np.zeros((19,self.n),dtype=np.int32),
                          np.zeros(19,dtype=np.int32),np.zeros(self.n,dtype=np.int32),np.zeros(self.n,dtype=np.uint8))

    def set_weights(self,*args):raise NotImplementedError('fixed-weight prototype; mutation unsupported')
    def update_edges(self,*args):raise NotImplementedError('fixed-weight prototype; mutation unsupported')
