'use strict';
// The Brain Activity panel must not present withheld graph rates as current values.
const test=require('node:test');
const assert=require('node:assert/strict');
const fs=require('node:fs');
const path=require('node:path');
const vm=require('node:vm');
const base=process.env.NEUROFLY_ACTIVITY_SOURCE_ROOT||path.resolve(__dirname,'..');
const replay=fs.readFileSync(path.join(base,'web/replay.js'),'utf8');

function panel(){
 const elements=new Map();
 const element=id=>{if(!elements.has(id))elements.set(id,{id,textContent:'',title:'',innerHTML:'',hidden:false,style:{}});return elements.get(id);};
 const context={window:{location:{search:''},dispatchEvent(){}},Event:function(){},
  document:{readyState:'loading',addEventListener(){},getElementById:id=>id==='rasterCanvas'||id==='activityCanvas'?null:element(id),createElement:()=>({dataset:{}})},
  requestAnimationFrame:()=>1,cancelAnimationFrame(){},performance:{now:()=>0},console};
 vm.createContext(context);vm.runInContext(replay,context);
 return {activity:context.window.neuroflyActivityPanel,element};
}
const names=['cb_intrinsic','vnc_motor'];
const packet=(activity)=>({step:10,activity:{grouping:'superclass',units:'Hz',names,sizes:[32164,10],...activity}});

test('withheld graph rates render as unavailable, not as numbers',()=>{
 const {activity,element}=panel();
 activity.update(packet({rates:null,unavailable:'No graph step for this assay since it was selected'}));
 assert.equal(element('activityGrouping').textContent,'superclass · Hz · no data for this assay yet');
 assert.equal(element('activityGrouping').title,'No graph step for this assay since it was selected');
 assert.equal(element('actVal0').textContent,'--');
 assert.equal(element('actVal1').textContent,'--');
 assert.equal(element('actBar0').style.width,'0%');
});

test('current rates still render normally and clear the unavailable note',()=>{
 const {activity,element}=panel();
 activity.update(packet({rates:null,unavailable:'No graph step for this assay since it was selected'}));
 activity.update(packet({rates:[8.55,5.65]}));
 assert.equal(element('activityGrouping').textContent,'superclass · Hz');
 assert.equal(element('activityGrouping').title,'');
 assert.equal(element('actVal0').textContent,'8.55');
 assert.equal(element('actVal1').textContent,'5.65');
});
