'use strict';
// The graph gait proxy must not show a cadence number before the displayed owner has stepped.
const test=require('node:test');
const assert=require('node:assert/strict');
const fs=require('node:fs');
const path=require('node:path');
const vm=require('node:vm');
const base=process.env.NEUROFLY_GAIT_SOURCE_ROOT||path.resolve(__dirname,'..');
const app=fs.readFileSync(path.join(base,'web/app.js'),'utf8');
const start=app.indexOf('        // CPG Tripod Gait');
const end=app.indexOf('        // Clear sandbox-only displays',start);
assert.ok(start>=0&&end>start,'gait render block exists');
const block=app.slice(start,end);

function render(biomechanics,graph=true){
 const els=new Map();const el=id=>{if(!els.has(id))els.set(id,{textContent:'',title:''});return els.get(id);};
 const context={document:{getElementById:el},self:{arena:{remotePacket:{biomechanics},cpg:{steppingFreq:8}}},
  panelCaps:{graph,gaitLabel:graph?'Body gait proxy · model-derived from streamed pose':'Kuramoto tripod gait · modular model'}};
 vm.createContext(context);
 vm.runInContext(`(function(panelCaps){${block}}).call(self,panelCaps)`,context);
 return el('valCpgFreq');
}
const reason='No graph step for this assay since it was selected';

test('unstepped graph owner shows no cadence number',()=>{
 const shown=render({cadence_hz:null,cadence_unavailable:reason});
 assert.equal(shown.textContent,'No step yet');
 assert.match(shown.title,/No measurement yet/);
});

test('a measured graph cadence is unchanged, including zero',()=>{
 assert.equal(render({cadence_hz:0.8}).textContent,'0.8 Hz · proxy');
 assert.equal(render({cadence_hz:0}).textContent,'0.0 Hz · proxy');
 assert.match(render({cadence_hz:0.8}).title,/Model-derived body cadence/);
});

test('missing cadence without a reason keeps the old Unavailable text; modular unchanged',()=>{
 assert.equal(render({}).textContent,'Unavailable');
 assert.equal(render({cadence_hz:null,cadence_unavailable:reason},false).textContent,'8.0 Hz');
});
