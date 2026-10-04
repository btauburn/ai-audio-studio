'use strict';
const test=require('node:test'),assert=require('node:assert/strict');
const { ReviewClient }=require('../dubbing_service/webui/voice-lab.js');
const { mountLab }=require('./helpers/voice_lab_harness.cjs');
const tick=()=>new Promise(resolve=>setImmediate(resolve));
function clock(){let callback,id=0;return{clock:{setTimeout(fn){callback=fn;return++id},clearTimeout(){callback=null}},expire(){assert.ok(callback);callback()}};}
function deferred(){let resolve,reject;const promise=new Promise((yes,no)=>{resolve=yes;reject=no});return{promise,resolve,reject};}
const sha=id=>require('node:crypto').createHash('sha256').update(id).digest('hex');
function fixture(){return{roles:[{role_id:'r1',name:'Ada',status:'PENDING',round_revision:3,assignment_revision:7,candidates:[
 {candidate_id:'c1',key:'k1',label:'A',status:'ready',audio_url:'/a/c1.wav',audio_sha256:sha('c1'),duration_seconds:4,rating:5,favorite:true,sections:[]},
 {candidate_id:'c2',key:'k2',label:'B',status:'ready',audio_url:'/a/c2.wav',audio_sha256:sha('c2'),duration_seconds:4,rating:4,favorite:false,sections:[]}
]}]};}
test('a read that never receives headers expires and aborts, rather than hanging startup',async()=>{
 const timer=clock(),gate=deferred();let options;
 const client=new ReviewClient('/api/test',async(url,request)=>{options=request;return gate.promise},{readTimeoutMs:20000,clock:timer.clock});
 const pending=client.request(''),rejected=assert.rejects(pending,{code:'REVIEW_READ_TIMEOUT'});await tick();timer.expire();await rejected;
 assert.equal(options.signal.aborted,true);assert.equal(options.method,undefined);
 gate.resolve({ok:true,json:async()=>({late:true})});await tick();
});
test('the same read deadline includes a response body that never finishes',async()=>{
 const timer=clock(),gate=deferred();let options;
 const client=new ReviewClient('/api/test',async(url,request)=>{options=request;return{ok:true,status:200,json:()=>gate.promise}},{clock:timer.clock});
 const pending=client.request('casting'),rejected=assert.rejects(pending,/took too long/);await tick();timer.expire();await rejected;
 assert.equal(options.signal.aborted,true);gate.resolve({roles:[]});await tick();
});
test('read deadlines do not change the existing exact write or uncertain-save handling',async()=>{
 const timer=clock();let options;
 const client=new ReviewClient('/api/test',async(url,request)=>{options=request;return{ok:true,json:async()=>({saved:true})}},{clock:timer.clock});
 await client.request('preferences',{key:'k1',audio_sha256:sha('c1'),favorite:true});
 assert.equal(options.method,'POST');assert.equal(options.headers['X-Review-Request'],'1');assert.equal(options.signal,undefined);
 assert.deepEqual(JSON.parse(options.body),{key:'k1',audio_sha256:sha('c1'),favorite:true});
 assert.throws(()=>timer.expire());
});
test('pending project load is visible, keeps the old workspace hidden, and dispatches reads only',async()=>{
 const gate=deferred(),lab=await mountLab(fixture(),{startOnly:true,respond:action=>action===''?gate.promise:undefined});await lab.settle();
 assert.equal(lab.elements['voice-lab-startup'].hidden,false);assert.match(lab.elements['voice-lab-startup-status'].textContent,/Loading your voices/);
 assert.equal(lab.elements['voice-lab-startup-retry'].hidden,true);assert.equal(lab.elements.workspace.hidden,true);assert.equal(lab.media.paused,true);
 assert.ok(lab.requests.every(r=>r.method==='GET'));gate.resolve(lab.base);await lab.ready;
 assert.equal(lab.elements['voice-lab-startup'].hidden,true);assert.equal(lab.elements.workspace.hidden,false);assert.equal(lab.all('casting-circle').length,2);
});
test('failed project read stays failed, exposes manual retry, and never invents an empty project',async()=>{
 let reloads=0;const data=fixture(),original=JSON.stringify(data);
 const lab=await mountLab(data,{onReload:()=>reloads++,respond:action=>{if(action==='')throw new Error('offline')}});
 assert.equal(lab.state.startupFailed,true);assert.equal(lab.elements.workspace.hidden,true);assert.equal(lab.elements['voice-lab-startup-retry'].hidden,false);
 assert.match(lab.elements['voice-lab-startup-status'].textContent,/could not be loaded.*offline/);
 assert.equal(lab.all('casting-circle').length,0);assert.equal(JSON.stringify(data),original);assert.ok(lab.requests.every(r=>r.method==='GET'));
 await lab.click(lab.elements['voice-lab-startup-retry']);assert.equal(reloads,1);assert.ok(lab.requests.every(r=>r.method==='GET'));
});
test('pending casting read reports only the confirmed character count and keeps stale controls hidden',async()=>{
 const gate=deferred(),data=fixture(),lab=await mountLab(data,{startOnly:true,respond:action=>action==='casting'?gate.promise:undefined});await lab.settle();
 assert.match(lab.elements['voice-lab-startup-status'].textContent,/Found 1 characters.*Loading your audition choices/);
 assert.equal(lab.elements.workspace.hidden,true);assert.equal(lab.all('casting-circle').length,0);assert.ok(lab.requests.every(r=>r.method==='GET'));
 gate.resolve({schema:'casting-review-v1',revision:7,...data});await lab.ready;assert.equal(lab.all('casting-circle').length,2);
});
test('failed casting load shows retry; a healthy reopened page retains four/five-star voices and favorite metadata',async()=>{
 const data=fixture(),original=JSON.stringify(data);
 const failed=await mountLab(data,{respond:action=>{if(action==='casting')throw new Error('Voice Lab took too long to respond. Please retry.')}});
 assert.equal(failed.state.startupFailed,true);assert.equal(failed.elements.workspace.hidden,true);assert.equal(failed.elements['voice-lab-startup-retry'].hidden,false);
 assert.match(failed.elements['voice-lab-startup-status'].textContent,/took too long/);assert.equal(JSON.stringify(data),original);
 const restored=await mountLab(data);assert.equal(restored.state.startupFailed,undefined);assert.equal(restored.elements.workspace.hidden,false);
 assert.deepEqual(data.roles[0].candidates.map(c=>c.rating),[5,4]);assert.deepEqual(restored.all('casting-heart').map(h=>h.getAttribute('aria-pressed')),['true','false']);
 assert.ok([...failed.requests,...restored.requests].every(r=>r.method==='GET'));
});
