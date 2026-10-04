'use strict';
const test = require('node:test');
const assert = require('node:assert');
const { mountLab } = require('./helpers/voice_lab_harness.cjs');
const cand = (id, key) => ({ candidate_id: id, key, audio_sha256: 'h' + key, audio_url: '/audio/' + key + '.wav', status: 'ready', label: id.toUpperCase(), duration_seconds: 5, rating: null, favorite: false });
const role = (rid, cands) => ({ role_id: rid, name: rid, status: 'PENDING', required: true, round_revision: 1, assignment_revision: 0, candidates: cands, current_audio: null });
const fixture = () => ({ roles: [role('r', [cand('a', 'k1'), cand('b', 'k2')])] });
const hosts = lab => lab.all('voice-candidate-host');

test('collision-prone identities stay unique and encoded', async () => {
  const lab = await mountLab({ roles: [role('r-a', [cand('c', 'k')]), role('r', [cand('a-c', 'k')])] });
  const ids = hosts(lab).map(h => h.dataset.voiceIdentity);
  assert.strictEqual(ids.length, 2); assert.strictEqual(new Set(ids).size, 2);
  ids.forEach(id => assert.match(id, /^vc_[0-9a-f_-]+$/));
});

test('deferred transition whose ready rejects renders once and writes once', async () => {
  const calls = [];
  const lab = await mountLab(fixture(), { configure: doc => { doc.startViewTransition = cb => { calls.push(cb); return { ready: Promise.reject(new Error('anim')), finished: Promise.resolve(), updateCallbackDone: Promise.resolve() }; }; } });
  const before = hosts(lab)[0];
  await lab.click(lab.all('casting-heart')[0]); await lab.settle();
  assert.strictEqual(lab.requests.filter(r => r.path === 'preferences').length, 1);
  const mid = hosts(lab)[0]; assert.notStrictEqual(before, mid);
  calls[0](); await lab.settle();
  assert.strictEqual(hosts(lab)[0], mid);
});

test('reduced motion skips native transition but mutates once', async () => {
  let started = 0;
  const lab = await mountLab(fixture(), { configure: (doc, win) => { win.matchMedia = () => ({ matches: true }); doc.startViewTransition = () => { started++; }; } });
  await lab.click(lab.all('casting-heart')[0]); await lab.settle();
  assert.strictEqual(started, 0);
  assert.strictEqual(lab.all('casting-candidate').find(card=>card.dataset.candidateId==='a').dataset.favorite, 'true');
  assert.strictEqual(lab.requests.filter(r => r.path === 'preferences').length, 1);
});

test('compare uses the single primary play per candidate; media ticks never transition', async () => {
  let started = 0;
  const lab = await mountLab(fixture(), { configure: doc => { doc.startViewTransition = cb => { started++; cb(); return { ready: Promise.resolve(), finished: Promise.resolve(), updateCallbackDone: Promise.resolve() }; }; } });
  await lab.click(lab.all('casting-compare-toggle')[0]);
  await lab.click(lab.all('casting-compare-toggle')[1]);
  assert.strictEqual(started, 2);
  assert.strictEqual(lab.all('casting-compare-play').length, 2);
  assert.strictEqual(lab.all('casting-play').length, 2);
  assert.deepStrictEqual(lab.all('casting-compare-play').map(b => b.textContent), ['Play A', 'Play B']);
  lab.media.emit('timeupdate'); await lab.settle();
  assert.strictEqual(started, 2);
});

const has = (el, name) => String(el.className || '').split(/\s+/).includes(name);
const within = (el, name) => has(el, name) ? el : (el.children || []).map(child => within(child, name)).find(Boolean);
const exact = (lab, id) => lab.all('casting-candidate').find(card => card.dataset.candidateId === id);

test('comparison assignment follows a confirmed rating live without replacing its candidate or sibling', async () => {
  const data = fixture(); let release;
  const lab = await mountLab(data, { respond: (action, body) => action === 'casting/feedback' && body.action === 'save'
    ? new Promise(resolve => { release = () => resolve({ accepted: true, saved: true, event_id: body.event_id, round_revision: 1 }); }) : undefined });
  await lab.click(within(exact(lab, 'a'), 'casting-compare-toggle')); await lab.click(within(exact(lab, 'b'), 'casting-compare-toggle'));
  const card = exact(lab, 'a'), sibling = exact(lab, 'b'), assign = lab.all('casting-assign')[0];
  assert.strictEqual(assign.disabled, true);
  const pending = lab.click(lab.all('casting-star').filter(star=>docWithin(card,star)).find(star => star.attrs['aria-label']?.startsWith('Rate 4 stars'))); await lab.settle();
  assert.strictEqual(assign.disabled, true, 'pending rating does not unlock assignment');
  release(); await pending; await lab.settle();
  assert.strictEqual(assign.disabled, false, 'confirmed current 4 stars unlocks the same comparison control');
  assert.strictEqual(exact(lab, 'a'), card); assert.strictEqual(exact(lab, 'b'), sibling);
  assert.strictEqual(card.dataset.tier, 'shiny'); assert.strictEqual(card.dataset.favorite, 'false');
  const posts = lab.requests.filter(r => r.method === 'POST'); assert.strictEqual(posts.length, 1); assert.strictEqual(posts[0].body.action, 'save'); assert.strictEqual(posts[0].body.stars, 4);
  assert.strictEqual(data.roles[0].current_key, undefined, 'rating alone does not choose an assignment');
});

for (const afterCallback of [false, true]) test(`native transition throw ${afterCallback ? 'after' : 'before'} callback writes and renders once`, async () => {
  let callbacks = 0;
  const lab = await mountLab(fixture(), { configure: doc => { doc.startViewTransition = cb => { if(afterCallback){callbacks++;cb();} throw new Error('native animation failed'); }; } });
  await lab.click(within(exact(lab,'a'), 'casting-heart')); await lab.settle();
  assert.strictEqual(exact(lab,'a').dataset.favorite,'true'); assert.strictEqual(lab.requests.filter(r=>r.path==='preferences').length,1);
  assert.strictEqual(callbacks,afterCallback ? 1 : 0); assert.strictEqual(lab.all('voice-candidate-host').length,2);
});

test('finished rejection before deferred native callback also performs one render', async () => {
  let callback;
  const lab=await mountLab(fixture(),{configure:doc=>{doc.startViewTransition=cb=>{callback=cb;return{ready:new Promise(()=>{}),updateCallbackDone:new Promise(()=>{}),finished:Promise.reject(new Error('cancelled'))};};}});
  await lab.click(within(exact(lab,'a'),'casting-heart'));await lab.settle();
  const card=exact(lab,'a');assert.strictEqual(card.dataset.favorite,'true');callback();await lab.settle();assert.strictEqual(exact(lab,'a'),card);
  assert.strictEqual(lab.requests.filter(r=>r.path==='preferences').length,1);
});

test('native context disclosure has unique ids, Escape focus return and light-dismiss does not steal focus', async () => {
  const lab=await mountLab(fixture(),{configure:(doc,win)=>{
    const create=doc.createElement;
    doc.createElement=tag=>{const el=create(tag);el.focus=()=>{doc.activeElement=el;};el.matches=()=>!!el.open;
      el.showPopover=()=>{el.open=true;el.emit('toggle',{newState:'open'});};el.hidePopover=()=>{el.open=false;el.emit('toggle',{newState:'closed'});};return el;};
    win.CSS={supports:()=>true};
  }});
  const card=exact(lab,'a'),panel=within(card,'voice-candidate-panel'),trigger=within(card,'voice-candidate-more');
  assert.strictEqual(new Set(lab.all('voice-candidate-panel').map(e=>e.id)).size,4);
  await lab.click(trigger);assert.strictEqual(trigger.attrs['aria-expanded'],'true');assert.strictEqual(docWithin(panel,lab.doc.activeElement),true);
  await panel.emit('keydown',{key:'Escape'});panel.hidePopover();assert.strictEqual(lab.doc.activeElement,trigger);assert.strictEqual(trigger.attrs['aria-expanded'],'false');
  await lab.click(trigger);const outside={tagName:'BUTTON'};lab.doc.activeElement=outside;panel.hidePopover();assert.strictEqual(lab.doc.activeElement,outside);
  assert.strictEqual(lab.requests.filter(r=>r.method==='POST').length,0,'inspection has no feedback or generation writes');
});
function docWithin(root,target){return root===target||(root.children||[]).some(child=>docWithin(child,target));}

test('missing native transition result still applies confirmed UI projection once', async () => {
  const lab=await mountLab(fixture(),{configure:doc=>{doc.startViewTransition=()=>null;}});
  await lab.click(within(exact(lab,'a'),'casting-heart'));await lab.settle();
  assert.strictEqual(exact(lab,'a').dataset.favorite,'true');assert.strictEqual(lab.requests.filter(r=>r.path==='preferences').length,1);
});

test('asynchronous transition render failure remains visible without replaying a confirmed write', async () => {
  const lab=await mountLab(fixture(),{configure:doc=>{doc.startViewTransition=callback=>{const update=Promise.resolve().then(callback);return{ready:update,updateCallbackDone:update,finished:update};};}});
  const root=lab.elements['casting-roles'];let attempts=0;root.replaceChildren=()=>{attempts++;throw new Error('synthetic render failure');};
  await lab.click(within(exact(lab,'a'),'casting-heart'));await lab.settle();
  assert.strictEqual(attempts,1);assert.strictEqual(lab.requests.filter(r=>r.path==='preferences').length,1);
  assert.strictEqual(lab.elements.error.hidden,false);assert.match(lab.elements.error.textContent,/synthetic render failure/);
});

test('confirmed 4/5 rating immediately moves the exact host into kept without rebuilding any candidate', async () => {
  const data=fixture();let release;
  const lab=await mountLab(data,{respond:(action,body)=>action==='casting/feedback'&&body.action==='save'?new Promise(resolve=>{release=()=>resolve({accepted:true,saved:true,event_id:body.event_id,round_revision:1});}):undefined});
  const card=exact(lab,'a'),sibling=exact(lab,'b'),host=hosts(lab).find(el=>el.dataset.voiceCandidateId==='a');
  assert.strictEqual(host.dataset.voiceContext,'audition');
  const star=lab.all('casting-star').find(el=>docWithin(card,el)&&el.attrs['aria-label'].startsWith('Rate 4 stars'));
  const pending=lab.click(star);await lab.settle();assert.strictEqual(host.dataset.voiceContext,'audition','unacknowledged rating never moves the candidate');
  release();await pending;await lab.settle();
  assert.strictEqual(host.dataset.voiceContext,'kept');assert.strictEqual(exact(lab,'a'),card);assert.strictEqual(exact(lab,'b'),sibling);
  assert.strictEqual(lab.all('casting-candidate').length,2);assert.strictEqual(lab.all('casting-play').length,2);assert.strictEqual(hosts(lab).filter(el=>el===host).length,1);
  const kept=lab.all('voice-candidate-collection').find(el=>el.dataset.collection==='kept');assert.strictEqual(kept.hidden,false);assert.ok(kept.children.includes(host));
  assert.strictEqual(card.dataset.favorite,'false');assert.strictEqual(card.dataset.selected,'false');
  assert.strictEqual(lab.requests.filter(r=>r.method==='POST').length,1);
});
