// Test page event wiring without opening or controlling a browser.
const fs=require('fs'),vm=require('vm'),assert=require('assert');
const html=fs.readFileSync('learning.html','utf8');
const payload=JSON.parse(fs.readFileSync('learning-data.json','utf8'));
const elements=new Map([...html.matchAll(/id="([^"]+)"/g)].map(m=>[m[1],{value:'',textContent:'',innerHTML:'',disabled:false,setAttribute(k,v){this[k]=v}}]));
elements.get('split').value='evaluation';elements.get('seed').value='44';
const requests=[];
const context={document:{getElementById(id){assert(elements.has(id),`Missing ${id}`);return elements.get(id)}},
 fetch:async(url,opts)=>{requests.push({url,opts});return {ok:true,json:async()=>url==='/learning-data.json'?payload:url==='/api/run'?{started:true}:{running:false,exit_code:0,log:''}}},
 setInterval:()=>1,clearInterval:()=>{},setTimeout:()=>{},console};
vm.createContext(context);
vm.runInContext(html.match(/<script>([\s\S]*?)<\/script>/)[1],context);
(async()=>{
 await new Promise(r=>setImmediate(r));
 assert.equal(elements.get('trained').textContent,Math.round(payload.summary.learned_accuracy*100)+'%');
 const before=elements.get('trialLabel').textContent;
 elements.get('next').onclick();assert.notEqual(elements.get('trialLabel').textContent,before);
 elements.get('split').value='training';elements.get('split').onchange();
 assert(elements.get('trialLabel').textContent.includes('training'));
 elements.get('seek').value='4';elements.get('seek').oninput();assert(elements.get('trialLabel').textContent.startsWith('5 /'));
 await elements.get('run').onclick();assert(requests.some(r=>r.url==='/api/run'&&JSON.parse(r.opts.body).seed===44));
 console.log('Learning UI event checks passed');
})().catch(e=>{console.error(e);process.exitCode=1});
