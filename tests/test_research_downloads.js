'use strict';
const test=require('node:test'),assert=require('node:assert/strict'),fs=require('node:fs'),vm=require('node:vm'),path=require('node:path');
const source=fs.readFileSync(path.join(__dirname,'../web/training.js'),'utf8');
const html=fs.readFileSync(path.join(__dirname,'../web/index.html'),'utf8');
const worker={updated_at:1,state:'waiting',completed:0,recent:[],protocol:'actual protocol'};
function response(status=200,data=worker,type='application/zip',length='123') {return {ok:status>=200&&status<300,status,json:async()=>data,headers:{get:n=>n==='Content-Length'?length:type}};}
function harness(fetcher){
 const elements={},listeners={},calls=[];let now=100;
 const bridge={connected:true,activeUrl:'http://intended:8879',lastOrderedPacket:{run_id:'daemon-a'},readOnly:false,replayMode:false};
 function el(id){return elements[id] ||= {attrs:{},style:{},textContent:'',innerHTML:'',firstChild:{textContent:''},classList:{toggle(){}},listeners:{},addEventListener(k,f){this.listeners[k]=f;},setAttribute(k,v){this.attrs[k]=String(v);},getAttribute(k){return this.attrs[k]??null;},removeAttribute(k){delete this.attrs[k];}};}
 const c={window:{app:{hud:{daemonBridge:bridge}},addEventListener(k,f){listeners[k]=f;}},document:{getElementById:el},AbortSignal:{timeout:()=>({})},Date:class extends Date{static now(){return now;}},setInterval(){},setTimeout(){},fetch:async(url,options)=>{calls.push({url,options});return fetcher(url,options,calls.length);}};
 vm.createContext(c);vm.runInContext(source.replace(/\}\)\(\);\s*$/, 'window.testResearch={pollResearch,reconcileResearchDownloads};})();'),c);
 return {c,bridge,calls,el,listeners,api:c.window.testResearch,tick(n){now+=n;}};
}
function disabled(h,id='researchSummaryDownload'){assert.equal(h.el(id).attrs['aria-disabled'],'true');assert.equal(h.el(id).attrs.href,undefined);}
test('markup has no initially enabled optional hrefs and has accessible status',()=>{
 for(const id of ['researchSummaryDownload','researchBundleDownload']){const tag=html.match(new RegExp('<a[^>]*id="'+id+'"[^>]*>'))[0];assert.ok(!tag.includes('href='));assert.match(tag,/aria-disabled="true"/);}
 assert.match(html,/id="researchDownloadStatus" role="status" aria-live="polite"/);
});
test('both validated resources enable only intended endpoint links; ZIP body never requested',async()=>{
 const h=harness(()=>response());disabled(h);await h.api.pollResearch();
 assert.equal(h.el('researchSummaryDownload').attrs.href,'http://intended:8879/research-status.json');assert.equal(h.el('researchBundleDownload').attrs.href,'http://intended:8879/research-latest.zip');assert.equal(h.calls[1].options.method,'HEAD');assert.equal(h.calls.length,2);
 await h.api.pollResearch();assert.equal(h.calls.filter(c=>c.options.method==='HEAD').length,1);h.tick(60000);await h.api.pollResearch();assert.equal(h.calls.filter(c=>c.options.method==='HEAD').length,2);
});
test('404 files disable both and explain missing worker',async()=>{
 const h=harness(()=>response(404));await h.api.pollResearch();disabled(h);disabled(h,'researchBundleDownload');assert.match(h.el('researchStatus').textContent,/not reporting/);assert.match(h.el('researchDownloadStatus').textContent,/HTTP 404/);
});
test('valid summary does not prove ZIP; HEAD unsupported is explicitly unverified',async()=>{
 const h=harness((url,o)=>response(o.method==='HEAD'?501:200));await h.api.pollResearch();assert.equal(h.el('researchSummaryDownload').attrs['aria-disabled'],'false');disabled(h,'researchBundleDownload');assert.match(h.el('researchDownloadStatus').textContent,/does not support HEAD/);
});
test('invalid worker JSON and bogus/empty ZIP success do not enable downloads',async()=>{
 for(const [type,length] of [['text/html','123'],['application/zip','0']]){const h=harness((url,o)=>response(200,o.method==='HEAD'?worker:{},type,length));await h.api.pollResearch();disabled(h);disabled(h,'researchBundleDownload');}
});
test('stale summary result cannot enable another endpoint; next poll checks new endpoint',async()=>{
 let resolve;const h=harness((u,o,n)=>n===1?new Promise(r=>resolve=r):response());const old=h.api.pollResearch();h.bridge.activeUrl='http://other:8880';h.api.reconcileResearchDownloads();resolve(response());await old;disabled(h);disabled(h,'researchBundleDownload');await h.api.pollResearch();assert.equal(h.el('researchSummaryDownload').attrs.href,'http://other:8880/research-status.json');assert.ok(!h.calls.some(c=>c.url==='http://intended:8879/research-latest.zip'));
});
test('stale ZIP result cannot enable changed daemon run at same URL',async()=>{
 let resolve;const h=harness((u,o)=>o.method==='HEAD'?new Promise(r=>resolve=r):response());const old=h.api.pollResearch();await new Promise(setImmediate);assert.equal(typeof resolve,'function');h.bridge.lastOrderedPacket.run_id='daemon-b';h.api.reconcileResearchDownloads();resolve(response());await old;disabled(h);disabled(h,'researchBundleDownload');
});
test('replay/offline/disconnect clear href and prevent clicks; live read-only permits verified downloads',async()=>{
 const h=harness(()=>response());await h.api.pollResearch();h.bridge.readOnly=true;h.api.reconcileResearchDownloads();assert.match(h.el('researchDownloadStatus').textContent,/Read-only/);assert.equal(h.el('researchBundleDownload').attrs['aria-disabled'],'false');
 h.bridge.replayMode=true;h.listeners['neurofly-replay-mode-change']();disabled(h);disabled(h,'researchBundleDownload');assert.match(h.el('researchDownloadStatus').textContent,/Replay/);
 h.bridge.replayMode=false;h.bridge.connected=false;h.api.reconcileResearchDownloads();disabled(h);assert.match(h.el('researchDownloadStatus').textContent,/Offline\/disconnected/);
 let blocked=false;h.el('researchSummaryDownload').listeners.click({preventDefault(){blocked=true;}});assert.equal(blocked,true);
});
test('click-time endpoint fence clears verified links before following them',async()=>{
 const h=harness(()=>response());await h.api.pollResearch();h.bridge.activeUrl='http://other';let blocked=false;h.el('researchBundleDownload').listeners.click({preventDefault(){blocked=true;}});assert.equal(blocked,true);disabled(h,'researchBundleDownload');
});
test('in-flight checks deduplicate and failed ZIP check is bounded by retry interval',async()=>{
 let resolve;const h=harness((u,o)=>o.method==='HEAD'?Promise.reject(new Error('CORS')):new Promise(r=>resolve=r));const a=h.api.pollResearch();await h.api.pollResearch();assert.equal(h.calls.length,1);resolve(response());await a;disabled(h,'researchBundleDownload');assert.equal(h.calls.length,2);
});
