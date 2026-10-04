'use strict';
const {test} = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const path = require('node:path');

class Element {
  constructor(tag) { this.tagName=tag; this.children=[]; this.events={}; this.paused=true; this.ended=false; this.attrs={}; this.style={}; this.loads=0; }
  append(...items) { for (const item of items) item.parent=this; this.children.push(...items); }
  set textContent(value) { this.text=value;this.children=[]; }
  get textContent() { return this.text || ''; }
  setAttribute(k,v) { this.attrs[k]=v; }
  removeAttribute(k) { delete this[k]; }
  addEventListener(k,fn) { (this.events[k] ||= []).push(fn); }
  emit(k) { for(const fn of this.events[k] || []) fn(); }
  pause() { const was=!this.paused;this.paused=true;if(was)this.emit('pause'); }
  load() { this.loads++; }
  remove() { if(this.parent)this.parent.children=this.parent.children.filter(x=>x!==this); }
  querySelectorAll(tag) { return this.children.flatMap(c=>[...(c.tagName===tag?[c]:[]),...c.querySelectorAll(tag)]); }
  querySelector(tag) { return this.querySelectorAll(tag)[0] || null; }
}

function fixture(options = {}) {
  const rows=new Element('div'),load=new Element('button'),panel=new Element('details'),main=new Element('audio');main.id='media';
  panel.append(load,rows);panel.querySelector=s=>s==='button'?load:rows;
  const doc={readyState:'complete', documentElement:{dataset:{}}, getElementById:id=>id==='reference-transcripts'?panel:null,
    createElement:tag=>new Element(tag),querySelector:s=>s==='#media'?main:null,querySelectorAll:tag=>[main,...rows.querySelectorAll(tag)]};
  const draft={status:'draft',text:'Fixture words.',text_sha256:'b'.repeat(64),uncertainties:[],segments:[{start:0,end:3,text:'Fixture words.'}]};
  let correction=options.correction || null;
  const references=[{acquisition_id:'1'.repeat(32),acquisition_sha256:'a'.repeat(64),source:options.sourceA || {performer:'Fixture A',source_page:'javascript:alert(1)'},original_sha256:'4'.repeat(64),reference_audio_sha256:'5'.repeat(64),start_seconds:0,end_seconds:8,holds:['Listening review pending'],reference_text:options.providedText,source_review_method:options.providedText ? 'human_review' : null,transcript_evidence:options.noEmbeddedDraft ? null : draft},
    {acquisition_id:'2'.repeat(32),acquisition_sha256:'c'.repeat(64),source:{performer:'Fixture B',source_page:'https://commons.wikimedia.org/wiki/File:Fixture.wav'},holds:[],reference_text:options.providedText,source_review_method:options.providedText ? 'human_review' : null,transcript_evidence:options.noEmbeddedDraft ? null : draft}];
  const calls=[];
  const fetch=async(url,requestOptions={})=>{
    calls.push({url,headers:requestOptions.headers,credentials:requestOptions.credentials,body:requestOptions.body?JSON.parse(requestOptions.body):null});
    let result;
    if(url.endsWith('/prepare')) { result=options.prepareDeferred ? await options.prepareDeferred : options.prepareResult || {status:'hold',acquisition_id:'3'.repeat(32),acquisition_sha256:'e'.repeat(64),reference_text:'Prepared fixture words.',source_processing:{schema:'fixture'},technical_qa:{holds:['Transcript alignment needs review']}}; }
    else if(url.endsWith('/quality')) { result=options.qualityDeferred ? await options.qualityDeferred : {status:'pass', expected_text:JSON.parse(requestOptions.body).expected_text, checks:{music:{status:'unavailable',reason:'No validated classifier.'}},metrics:{word_error_rate:0}}; }
    else if(url.endsWith('/correction')) { correction={text:JSON.parse(requestOptions.body).text,text_sha256:'d'.repeat(64),status:'corrected_unreviewed'};result=correction; }
    else if(url.endsWith('/transcript')) result={...draft,...(correction?{correction}: {})};
    else result={references:references.map(r=>({...r,...(correction?{correction}: {})}))};
    if(url.endsWith('/prepare') && options.prepareError) return {ok:false,json:async()=>({detail:options.prepareError})};
    return {ok:true,json:async()=>result};
  };
  vm.runInNewContext(fs.readFileSync(path.join(__dirname,'../dubbing_service/webui/reference-transcripts.js'),'utf8'),
    {document:doc,location:{pathname:'/voice-lab',search:'?project=fixture'},fetch,URL,URLSearchParams,queueMicrotask});
  return {rows,load,doc,calls};
}

test('reference source links reject script URLs, switching audio preserves shortcuts guard, reload releases old clips',async()=>{
  const f=fixture();await f.load.onclick();
  const links=f.rows.querySelectorAll('a');assert.equal(links[0].href,undefined);assert.match(links[1].href,/^https:\/\/commons/);
  const [a,b]=f.rows.querySelectorAll('audio');
  a.paused=false;a.emit('play');await Promise.resolve();assert.equal(f.doc.documentElement.dataset.runtimeComparison,'playing');
  b.paused=false;b.emit('play');await Promise.resolve();assert.equal(a.paused,true);assert.equal(f.doc.documentElement.dataset.runtimeComparison,'playing');
  await f.load.onclick();await Promise.resolve();assert.equal(b.paused,true);assert.equal(b.loads,1);assert.equal(f.doc.documentElement.dataset.runtimeComparison,undefined);
  assert.ok(f.calls.every(c=>c.url.startsWith('/api/voice-lab/fixture/references')));
});

test('phone correction saves exact ASR binding and reloads saved words without changing voice ratings',async()=>{
  const f=fixture();await f.load.onclick();
  const card=f.rows.children[0];const editor=card.querySelector('textarea');editor.value='Correct fixture words.';
  const save=card.querySelectorAll('button').find(b=>b.textContent==='Save transcript correction');await save.onclick();
  const sent=f.calls.find(c=>c.url.endsWith('/correction'));
  assert.equal(sent.body.expected_text_sha256,'b'.repeat(64));assert.equal(sent.body.expected_correction_sha256,null);
  assert.equal(sent.body.text,'Correct fixture words.');assert.equal(Object.hasOwn(sent.body,'stars'),false);
  await f.load.onclick();assert.equal(f.rows.children[0].querySelector('textarea').value,'Correct fixture words.');
  assert.ok(f.rows.children[0].querySelectorAll('p').some(p=>/alignment review remain pending/.test(p.textContent)));
});


test('saved correction without embedded ASR remains visible after reload and requests fresh draft before editing',async()=>{
  const f=fixture({noEmbeddedDraft:true,correction:{text:'Previously corrected exact words.',text_sha256:'d'.repeat(64),status:'corrected_unreviewed'}});
  await f.load.onclick();
  const card=f.rows.children[0];assert.equal(card.querySelector('textarea').value,'Previously corrected exact words.');
  assert.equal(card.querySelectorAll('button').find(b=>b.textContent==='Save transcript correction').disabled,true);
  assert.equal(card.querySelectorAll('button').find(b=>b.textContent==='Generate transcript locally').disabled,undefined);
  assert.ok(card.querySelectorAll('p').some(p=>/review remain pending/.test(p.textContent)));
});

test('provided user-confirmed transcript is displayed without inventing an ASR draft', async()=>{
  const f=fixture({noEmbeddedDraft:true,providedText:'User-confirmed exact recording words.'}); await f.load.onclick();
  const card=f.rows.children[0]; assert.equal(card.querySelector('textarea').value,'User-confirmed exact recording words.');
  assert.ok(card.querySelectorAll('p').some(p=>/confirmed by the user/.test(p.textContent)));
  assert.equal(card.querySelectorAll('button').find(b=>b.textContent==='Save transcript correction').disabled,true);
});

test('quality POST binds transcript, preserves review guard, and cannot claim music/source approval', async()=>{
  const f=fixture();await f.load.onclick(); const card=f.rows.children[0];
  await card.querySelectorAll('button').find(b=>b.textContent==='Check reference locally').onclick();
  const call=f.calls.find(c=>c.url.endsWith('/quality'));
  assert.equal(call.headers['X-Review-Request'],'1'); assert.equal(call.credentials,'same-origin');
  assert.equal(call.body.expected_text,'Fixture words.'); assert.equal(Object.hasOwn(call.body,'stars'),false);
  assert.ok(card.querySelectorAll('p').some(p=>/Technical pass.*not assessed/.test(p.textContent)));
  const editor=card.querySelector('textarea');editor.value='New edited words.';editor.emit('input');
  assert.ok(card.querySelectorAll('p').some(p=>/Transcript changed/.test(p.textContent)));
});

test('quality request keeps editor responsive and discards a result after concurrent text edits', async()=>{
  let release;const deferred=new Promise(resolve=>release=resolve);const f=fixture({qualityDeferred:deferred});await f.load.onclick();
  const card=f.rows.children[0],button=card.querySelectorAll('button').find(b=>b.textContent==='Check reference locally');
  const pending=button.onclick();assert.equal(button.disabled,true);const editor=card.querySelector('textarea');
  assert.equal(editor.disabled,undefined);editor.value='Newer words while CPU works.';editor.emit('input');
  release({status:'pass',expected_text:'Fixture words.',checks:{},metrics:{}});await pending;
  assert.equal(button.disabled,false);assert.ok(card.querySelectorAll('p').some(p=>/changed during verification/.test(p.textContent)));
});

test('ASR programmatic replacement invalidates quality checked against a different edited transcript', async()=>{
  const f=fixture();await f.load.onclick();const card=f.rows.children[0];card.querySelector('textarea').value='Edited supplied words.';
  await card.querySelectorAll('button').find(b=>b.textContent==='Check reference locally').onclick();
  await card.querySelectorAll('button').find(b=>b.textContent==='Generate transcript locally').onclick();
  assert.equal(card.querySelector('textarea').value,'Fixture words.');
  assert.ok(card.querySelectorAll('p').some(p=>/Transcript changed/.test(p.textContent)));
});

test('local cleanup binds exact receipt and words, displays held processed clip paused and preserves original', async()=>{
  const f=fixture(); await f.load.onclick(); const card=f.rows.children[0];
  const original=card.querySelector('audio'), editor=card.querySelector('textarea');
  editor.value='Reviewed source words.';
  await card.querySelectorAll('button').find(b=>b.textContent==='Clean voices locally').onclick();
  const sent=f.calls.find(c=>c.url.endsWith('/prepare'));
  assert.equal(sent.body.acquisition_sha256,'a'.repeat(64)); assert.equal(sent.body.expected_text,'Reviewed source words.');
  assert.equal(sent.headers['X-Review-Request'],'1'); assert.equal(sent.credentials,'same-origin');
  assert.deepEqual(Object.keys(sent.body).sort(),['acquisition_sha256','expected_text']);
  assert.equal(editor.value,'Reviewed source words.'); assert.match(original.src,/1{32}\/audio/);
  const preview=card.querySelectorAll('audio')[1]; assert.match(preview.src,/3{32}\/audio\?acquisition_sha256=e{64}$/); assert.equal(preview.paused,true);
  assert.ok(card.querySelectorAll('p').some(p=>/needs review.*alignment/.test(p.textContent)));
  assert.ok(card.querySelectorAll('p').some(p=>p.textContent==='Processed reference · may affect timbre'));
  original.paused=false; original.emit('play'); preview.paused=false; preview.emit('play'); await Promise.resolve();
  assert.equal(original.paused,true); assert.equal(f.doc.documentElement.dataset.runtimeComparison,'playing');
});

test('cleanup suppresses concurrent submissions and retains newer unsaved words', async()=>{
  let release; const deferred=new Promise(resolve=>release=resolve); const f=fixture({prepareDeferred:deferred}); await f.load.onclick();
  const card=f.rows.children[0], button=card.querySelectorAll('button').find(b=>b.textContent==='Clean voices locally'), editor=card.querySelector('textarea');
  const pending=button.onclick(); await button.onclick(); assert.equal(f.calls.filter(c=>c.url.endsWith('/prepare')).length,1);
  assert.equal(button.disabled,true); assert.equal(editor.disabled,undefined); editor.value='Newer unsaved words.'; editor.emit('input');
  release({status:'preserved',reason:'Existing reference preserved.'}); await pending;
  assert.equal(editor.value,'Newer unsaved words.'); assert.equal(button.disabled,false);
  assert.ok(card.querySelectorAll('p').some(p=>/newer transcript edit is preserved/.test(p.textContent)));
  assert.equal(card.querySelectorAll('audio').length,1);
});

test('cleanup completing after reload cannot attach an old result to current reference cards', async()=>{
  let release; const f=fixture({prepareDeferred:new Promise(resolve=>release=resolve)}); await f.load.onclick();
  const old=f.rows.children[0], pending=old.querySelectorAll('button').find(b=>b.textContent==='Clean voices locally').onclick();
  await f.load.onclick(); release({status:'hold',acquisition_id:'3'.repeat(32),acquisition_sha256:'e'.repeat(64),technical_qa:{holds:[]}}); await pending;
  assert.equal(f.rows.children[0].querySelectorAll('audio').length,1); assert.equal(old.querySelectorAll('audio').length,1);
});

test('source-use or tool preparation hold remains visible and restores the cleanup control', async()=>{
  const f=fixture({prepareError:'Source-bound use evidence is missing'}); await f.load.onclick();
  const card=f.rows.children[0], editor=card.querySelector('textarea'), button=card.querySelectorAll('button').find(b=>b.textContent==='Clean voices locally');
  editor.value=''; await button.onclick();
  assert.equal(f.calls.find(c=>c.url.endsWith('/prepare')).body.expected_text,null);
  assert.equal(button.disabled,false); assert.equal(editor.value,''); assert.equal(card.querySelectorAll('audio').length,1);
  assert.ok(card.querySelectorAll('p').some(p=>/Preparation unavailable: Source-bound use evidence is missing/.test(p.textContent)));
});


test('pinned corpus preview displays hashes and pending delivery without starting synthesis',async()=>{
  const f=fixture({sourceA:{provider:'reviewed_dataset_recording',performer:'CSTR p226',source_page:'https://datashare.ed.ac.uk/handle/10283/3443',attribution:'CSTR VCTK',archive_member:'p226_021_mic1.flac'}});
  await f.load.onclick(); const card=f.rows.children[0];
  assert.equal(card.querySelector('a').href,'https://datashare.ed.ac.uk/handle/10283/3443');
  assert.ok(card.querySelectorAll('p').some(p=>/Source delivery review pending; this is not a generated audition/.test(p.textContent)));
  assert.ok(card.querySelectorAll('p').some(p=>p.textContent.includes('Original SHA-256 '+ '4'.repeat(64)) && p.textContent.includes('Reference SHA-256 '+ '5'.repeat(64))));
  assert.equal(f.calls.length,1);assert.ok(f.calls[0].url.endsWith('/references'));assert.equal(f.calls[0].body,null);
});

test('recorded corpus source links accept only exact primary pages',async()=>{
  for(const [url,allowed] of [
    ['https://github.com/dioco-group/jenny-tts-dataset',true],
    ['https://github.com/other/unreviewed',false],
    ['https://datashare.ed.ac.uk/handle/other',false],
    ['https://datashare.ed.ac.uk.evil.example/handle/10283/3443',false],
    ['http://datashare.ed.ac.uk/handle/10283/3443',false],
    ['javascript:alert(1)',false]]) {
    const f=fixture({sourceA:{provider:'reviewed_dataset_recording',performer:'Fixture',source_page:url}});await f.load.onclick();
    assert.equal(Boolean(f.rows.children[0].querySelector('a').href),allowed,url);
  }
});
