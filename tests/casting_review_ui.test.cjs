'use strict';
const { test } = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const { mount } = require('../dubbing_service/webui/voice-lab.js');

class Element {
  constructor(tag = 'div') { this.tagName = tag.toUpperCase(); this.children = []; this.dataset = {}; this.events = {}; this.attrs = {}; this.value = ''; this.hidden = false; this.disabled = false; this.textContent = ''; this.classList = { toggle() {} }; }
  append(...children) { this.children.push(...children); }
  replaceChildren(...children) { this.children = children; }
  get options() { return this.children; }
  setAttribute(name, value) { this.attrs[name] = value; }
  addEventListener(name, fn) { (this.events[name] ||= []).push(fn); }
}
class Media {
  constructor() { this.events = {}; this.paused = true; this.readyState = 1; this.duration = 30; this.currentTime = 0; this.volume = 1; this.history = []; }
  addEventListener(name, fn) { (this.events[name] ||= []).push(fn); }
  emit(name) { for (const fn of this.events[name] || []) fn(); }
  pause() { this.paused = true; this.history.push(['pause', this.src]); }
  play() { this.paused = false; this.history.push(['play', this.src]); return Promise.resolve(); }
  load() { this.currentTime = 0; this.history.push(['load', this.src]); this.emit('loadedmetadata'); }
  removeAttribute(name) { delete this[name]; }
}
const hash = n => String(n).repeat(64).slice(0, 64);
const candidate = (id, rating = null) => ({ candidate_id: id, label: `Version ${id}`, key: `voice-${id}`, audio_url: `/audio/${id}`, audio_sha256: hash(id.charCodeAt(0) % 10 + 1), duration_seconds: 8, technical_qa: 'passed', status: 'ready', rating, identity_id: `identity-${id}`, sections: [] });
function fixture(options = {}) {
  const search = options.search || '?project=fixture';
  const voices = [candidate('a'), candidate('b'), candidate('c')];
  const casting = { schema: 'casting-review-v1', revision: 7, source_scope: 'Chapter One · 3 utterances', scope: { utterance_ids: ['u1', 'u2', 'u3'] }, roles: [
    { role_id: 'role-one', name: 'Mara', description: 'A patient guide.', status: 'NEEDS_REVIEW', reason: 'Choose a candidate.', assignment_revision: 5, round_revision: 3, round_status: 'awaiting_review', current_key: null, candidates: voices, direction: { requested: ['warm, close delivery'], applied: [], unsupported: ['emotional vector'], unverified: ['age impression'], retained: ['current role identity'] }, history: [] },
    { role_id: 'role-two', name: 'Approved guide', status: 'APPROVED', reason: 'Approved by Brian.', assignment_revision: 2, round_revision: 1, round_status: 'resolved', candidates: [], direction: {}, history: [] }
  ], productions: [], capabilities: { casting_review: true, round_requests: true } };
  const base = { project_name: 'Fixture', capabilities: { casting_review: true, casting_enabled: true }, voices: [], roles: [], automation: { paused: true }, issues: [] };
  const storage = new Map(); const calls = []; const intervals = []; let castingState = structuredClone(casting); let postFeedback = async body => ({ accepted: true, event_id: body.event_id, status: 'saved' }); let postRound = async body => {
    const next = structuredClone(castingState); const role = next.roles.find(row => row.role_id === body.role_id);
    role.round_revision = body.intent === 'generate' && !role.round_revision ? 1 : (role.round_revision || 0) + (body.intent === 'none_fit' ? 1 : 0);
    role.round_status = 'pending'; role.status = 'NEEDS_REVIEW';
    role.candidates = ['a', 'b', 'c'].map(id => ({ ...candidate(`${id}-next`), status: 'pending', key: null, audio_url: null, audio_sha256: null, rating: null }));
    castingState = next;
    return { casting: structuredClone(next), reason: body.intent === 'none_fit' ? 'Fresh voices requested for this character.' : 'Audio generation queued for this character.', receipt: { request_id: body.request_id, role_id: body.role_id, intent: body.intent, round_revision: role.round_revision, assignment_revision: role.assignment_revision, status: 'queued', created: true } };
  };
  const html = fs.readFileSync(path.join(__dirname, '../dubbing_service/webui/voice-lab.html'), 'utf8');
  const elements = Object.fromEntries([...html.matchAll(/<([a-z]+)[^>]* id="([^"]+)"/g)].map(m => [m[2], new Element(m[1])]));
  elements.media = new Media();
  const doc = new Element(); doc.getElementById = id => elements[id]; doc.createElement = tag => new Element(tag);
  const win = { location: { href: 'http://localhost/web/voice-lab.html' + search, origin: 'http://localhost', pathname: '/web/voice-lab.html', search }, crypto: { randomUUID: (() => { let i = 0; return () => `event-${++i}`; })() }, localStorage: { getItem: key => storage.get(key) || null, setItem: (key, value) => storage.set(key, value), removeItem: key => storage.delete(key) }, setInterval: () => 1, clearInterval: () => {}, fetch: async (url, request) => {
    calls.push({ url, request });
    if (url.endsWith('/casting') && !request?.method) return { ok: true, status: 200, json: async () => structuredClone(castingState) };
    if (url.endsWith('/casting/feedback')) { const body = JSON.parse(request.body); const response = await postFeedback(body); return { ok: response.ok !== false, status: response.status || 200, json: async () => response.body || response }; }
    if (url.endsWith('/casting/round')) { const body = JSON.parse(request.body); const response = await postRound(body); return { ok: response.ok !== false, status: response.status || 200, json: async () => response.body || response }; }
    if (url.endsWith('/casting/prepare')) return { ok: true, status: 200, json: async () => ({ ...structuredClone(castingState), request_id: JSON.parse(request.body).request_id, message: 'Preparation saved.' }) };
    if (url.endsWith('/casting/render')) return { ok: true, status: 200, json: async () => ({ request_id: JSON.parse(request.body).request_id, status: 'queued', message: 'Request queued.' }) };
    if (url.endsWith('/casting/resume')) {
      const requestId = JSON.parse(request.body).request_id;
      castingState = { ...structuredClone(castingState), productions: (castingState.productions || []).map(row => row.request_id === requestId ? { ...row, status: 'awaiting_review' } : row) };
      return { ok: true, status: 200, json: async () => structuredClone(castingState) };
    }
    if (url.endsWith('/position')) return { ok: true, status: 200, json: async () => ({ saved: true }) };
    return { ok: true, status: 200, json: async () => structuredClone(base) };
  }, setInterval: fn => { intervals.push(fn); return intervals.length; }, clearInterval: () => {} };
  return { doc, win, elements, calls, intervals, storage, base, get casting() { return castingState; }, set casting(value) { castingState = value; }, set postFeedback(value) { postFeedback = value; }, set postRound(value) { postRound = value; } };
}
function find(el, predicate) { if (!el || typeof el !== 'object') return null; if (predicate(el)) return el; for (const child of el.children || []) { const hit = find(child, predicate); if (hit) return hit; } return null; }
function findAll(el, predicate, out = []) { if (!el || typeof el !== 'object') return out; if (predicate(el)) out.push(el); for (const child of el.children || []) findAll(child, predicate, out); return out; }
const candidateCard = (env, id) => find(env.elements['casting-roles'], el => el.tagName === 'ARTICLE' && el.dataset.candidateId === id);
const buttonByText = (card, text) => find(card, el => el.tagName === 'BUTTON' && el.textContent === text);
const starButton = (card, value) => find(card, el => el.tagName === 'BUTTON' && el.className.includes('casting-star') && el.attrs['aria-label']?.startsWith(`Rate ${value} stars`));
const ratingStatus = card => find(card, el => el.className === 'casting-rating-status');
const click = el => el.events.click[0]();

test('inbox is read-only on load, keeps chosen portraits visible, and plays only on a candidate tap', async () => {
  const env = fixture(); const state = await mount(env.doc, env.win);
  // Three reads, and nothing else: where the book is, the project, then the casting review.
  assert.deepEqual(env.calls.map(row => [row.url, row.request?.method]), [
    ['/api/v1/projects/fixture/voice-lab/casting/book', undefined],
    ['/api/v1/projects/fixture/voice-lab', undefined], ['/api/v1/projects/fixture/voice-lab/casting', undefined]
  ]);
  assert.equal(env.elements['casting-inbox'].hidden, false);
  assert.equal(env.elements['casting-roles'].children.length, 2, 'chosen characters remain visible in the portrait view');
  assert.equal(state.player.media.paused, true);
  assert.deepEqual(env.calls.filter(row => row.request?.method === 'POST'), []);
  const play = buttonByText(candidateCard(env, 'a'), 'Play'); await click(play);
  assert.equal(state.player.current.key, 'voice-a'); assert.equal(state.player.media.paused, false); assert.equal(state.player.media.playbackRate, 1);
  assert.equal(buttonByText(candidateCard(env, 'a'), 'Pause').textContent, 'Pause');
  await click(buttonByText(candidateCard(env, 'a'), 'Pause'));
  state.player.media.currentTime = 2.4;
  const loadsBeforeResume = state.player.media.history.filter(row => row[0] === 'load').length;
  await click(buttonByText(candidateCard(env, 'a'), 'Play'));
  assert.equal(state.player.media.currentTime, 2.4, 'same-candidate Play resumes at the paused position');
  assert.equal(state.player.media.history.filter(row => row[0] === 'load').length, loadsBeforeResume);
  assert.equal(env.calls.filter(row => row.request?.method === 'POST').length, 0, 'playback does not dispatch synthesis or feedback');
  env.elements['show-approved-casts'].checked = true; env.elements['show-approved-casts'].onchange();
  assert.equal(env.elements['casting-roles'].children.length, 2);
});

test('Prepare is an explicit action and leaves utterance scope selection with the backend', async () => {
  const env = fixture(); await mount(env.doc, env.win);
  await env.elements['prepare-cast'].onclick();
  const call = env.calls.find(row => row.url.endsWith('/casting/prepare'));
  assert.deepEqual(JSON.parse(call.request.body), { request_id: 'event-1' });
  assert.match(env.elements['casting-status'].textContent, /Preparation saved/);
});

test('Generate audio is a per-character explicit request and never saves feedback or approval', async () => {
  const env = fixture();
  env.base.capabilities.casting_enabled = false;
  env.casting = { ...env.casting, roles: [{ ...env.casting.roles[0], status: 'NEW_CHARACTER', round_revision: 0, round_status: null, candidates: [] }] };
  await mount(env.doc, env.win);
  const roleCard = env.elements['casting-roles'].children[0];
  await click(buttonByText(roleCard, 'Generate audio'));
  const call = env.calls.find(row => row.url.endsWith('/casting/round'));
  const body = JSON.parse(call.request.body);
  assert.deepEqual(body, { request_id: 'event-1', role_id: 'role-one', expected_round_revision: 0, expected_assignment_revision: 5, intent: 'generate' });
  assert.equal(env.calls.filter(row => row.url.endsWith('/casting/feedback')).length, 0);
  assert.equal(env.casting.roles[0].round_status, 'pending');
  assert.equal(buttonByText(env.elements['casting-roles'].children[0], 'Generating audio…').disabled, true);
  assert.equal(env.elements.assign.hidden, true, 'direct role assignment is disabled after AutoCast generation enables casting review');
});

test('Try new voices is explicit even when every existing option is blocked and preserves chosen assignment', async () => {
  const env = fixture();
  env.casting = { ...env.casting, roles: [{ ...env.casting.roles[0], status: 'NEEDS_RECAST', current_key: 'voice-approved', round_status: 'blocked', round_revision: 3,
    candidates: [candidate('bad-a'), candidate('bad-b')].map(row => ({ ...row, key: null, audio_url: null, audio_sha256: null, status: 'blocked', rating: null })) }] };
  await mount(env.doc, env.win);
  await click(buttonByText(env.elements['casting-roles'].children[0], 'Try new voices'));
  const body = JSON.parse(env.calls.find(row => row.url.endsWith('/casting/round')).request.body);
  assert.equal(body.intent, 'none_fit'); assert.equal(body.role_id, 'role-one'); assert.equal(body.expected_round_revision, 3);
  assert.equal(env.casting.roles[0].current_key, 'voice-approved');
  assert.ok(env.casting.roles[0].candidates.every(row => row.rating == null));
  assert.equal(env.calls.filter(row => row.url.endsWith('/casting/feedback')).length, 0);
});

test('an uncertain round request is retried with the identical request ID and body', async () => {
  const env = fixture();
  env.casting = { ...env.casting, roles: [{ ...env.casting.roles[0], status: 'NEW_CHARACTER', round_revision: 0, round_status: null, candidates: [] }] };
  let count = 0; env.postRound = async body => { if (++count === 1) throw new Error('connection dropped after submit'); return { casting: env.casting, reason: 'Existing request receipt returned.', receipt: { request_id: body.request_id, status: 'queued', created: false } }; };
  await mount(env.doc, env.win);
  await click(buttonByText(env.elements['casting-roles'].children[0], 'Generate audio'));
  const storageKey = 'voice-lab-casting-round-v1:fixture:role-one';
  const saved = JSON.parse(env.storage.get(storageKey)).pendingRequest;
  assert.ok(saved.request_id); assert.equal(saved.intent, 'generate');
  await click(buttonByText(env.elements['casting-roles'].children[0], 'Retry request'));
  const bodies = env.calls.filter(row => row.url.endsWith('/casting/round')).map(row => JSON.parse(row.request.body));
  assert.deepEqual(bodies[0], bodies[1]);
  assert.equal(env.storage.has(storageKey), false, 'the durable request is cleared only after an acknowledged receipt');
});

test('rapid duplicate round clicks submit only one mutation', async () => {
  const env = fixture();
  env.casting = { ...env.casting, roles: [{ ...env.casting.roles[0], status: 'NEW_CHARACTER', round_revision: 0, round_status: null, candidates: [] }] };
  let release; env.postRound = body => new Promise(resolve => { release = () => resolve({ casting: env.casting, reason: 'Queued.', receipt: { request_id: body.request_id, status: 'queued' } }); });
  await mount(env.doc, env.win);
  const button = buttonByText(env.elements['casting-roles'].children[0], 'Generate audio');
  const first = click(button);
  await click(button);
  assert.equal(env.calls.filter(row => row.url.endsWith('/casting/round')).length, 1);
  release(); await first;
});

test('round CAS conflict reports reason, refreshes, and does not retry or mutate the round', async () => {
  const env = fixture();
  env.casting = { ...env.casting, roles: [{ ...env.casting.roles[0], status: 'NEW_CHARACTER', round_revision: 0, round_status: null, candidates: [] }] };
  const before = structuredClone(env.casting);
  env.postRound = async () => ({ ok: false, status: 409, body: { detail: 'This character changed; refresh before generating.' } });
  await mount(env.doc, env.win);
  await click(buttonByText(env.elements['casting-roles'].children[0], 'Generate audio'));
  assert.deepEqual(env.casting, before);
  assert.match(env.elements['casting-flash'].textContent, /character changed/);
  assert.equal(env.calls.filter(row => row.url.endsWith('/casting/round')).length, 1);
  assert.equal(env.storage.has('voice-lab-casting-round-v1:fixture:role-one'), false);
});

test('old server route rejection clears the uncertain request and keeps an actionable terminal message', async () => {
  const env = fixture();
  env.casting = { ...env.casting, roles: [{ ...env.casting.roles[0], status: 'NEW_CHARACTER', round_revision: 0, round_status: null, candidates: [] }] };
  env.postRound = async () => ({ ok: false, status: 404, body: { detail: 'Not found' } });
  await mount(env.doc, env.win);
  await click(buttonByText(env.elements['casting-roles'].children[0], 'Generate audio'));
  const storageKey = 'voice-lab-casting-round-v1:fixture:role-one';
  assert.equal(JSON.parse(env.storage.get(storageKey)).unsupported, true);
  env.elements['show-approved-casts'].checked = true; env.elements['show-approved-casts'].onchange();
  const role = env.elements['casting-roles'].children[0];
  assert.equal(buttonByText(role, 'Unavailable').disabled, true, 'rerender cannot revive an unsupported route');
  assert.match(find(role, el => el.className === 'casting-round-request-status').textContent, /Restart Studio and the private review service/);
});

test('a role with unresolved feedback cannot start a replacement round', async () => {
  const env = fixture(); await mount(env.doc, env.win);
  let release; env.postFeedback = body => new Promise(resolve => { release = () => resolve({ accepted: true, event_id: body.event_id }); });
  const card = env.elements['casting-roles'].children[0];
  const save = click(starButton(candidateCard(env, 'a'), 4));
  assert.equal(buttonByText(card, 'Try new voices').disabled, true);
  assert.equal(env.calls.some(row => row.url.endsWith('/casting/round')), false);
  release(); await save;
});

test('round refresh closes direction bound to the prior audition and cannot submit stale feedback', async () => {
  const env = fixture(); await mount(env.doc, env.win);
  const card = env.elements['casting-roles'].children[0];
  await click(buttonByText(candidateCard(env, 'a'), 'Add direction'));
  env.elements['casting-direction-input'].value = 'Keep the current vowel shape.';
  env.elements['casting-direction-input'].oninput();
  await click(buttonByText(card, 'Try new voices'));
  assert.equal(env.elements['casting-feedback'].hidden, true);
  await env.elements['casting-save'].onclick();
  assert.equal(env.calls.filter(row => row.url.endsWith('/casting/feedback')).length, 0);
  assert.equal(env.casting.roles[0].round_revision, 4);
});

test('star save freezes the exact candidate before async work and does not include optional direction text', async () => {
  const env = fixture(); await mount(env.doc, env.win);
  await click(buttonByText(candidateCard(env, 'b'), 'Add direction'));
  env.elements['casting-stars'].value = '4'; env.elements['casting-stars'].onchange();
  env.elements['casting-direction-input'].value = 'Keep the low, steady register.'; env.elements['casting-direction-input'].oninput();
  let release; env.postFeedback = body => new Promise(resolve => { release = () => resolve({ accepted: true, event_id: body.event_id, status: 'saved' }); });
  const save = click(starButton(candidateCard(env, 'b'), 4));
  assert.equal(env.elements['casting-feedback'].hidden, true, 'rating does not keep the optional direction form in the main flow');
  await click(buttonByText(candidateCard(env, 'a'), 'Play'));
  release(); await save;
  const sent = JSON.parse(env.calls.find(row => row.url.endsWith('/casting/feedback')).request.body);
  assert.equal(sent.candidate_id, 'b'); assert.equal(sent.key, 'voice-b'); assert.equal(sent.audio_sha256, candidate('b').audio_sha256);
  assert.equal(sent.raw_direction, ''); assert.equal(sent.stars, 4); assert.equal(sent.action, 'save');
  assert.deepEqual(sent.baseline_candidate_ids, ['a', 'b', 'c']);
  assert.equal(sent.round_revision, 3); assert.equal(sent.assignment_revision, 5);
});

test('uncertain star save keeps the exact idempotent request and retries it unchanged', async () => {
  const env = fixture(); await mount(env.doc, env.win);
  await click(buttonByText(candidateCard(env, 'c'), 'Add direction'));
  env.elements['casting-stars'].value = '3'; env.elements['casting-stars'].onchange();
  env.elements['casting-direction-input'].value = 'Softer consonants'; env.elements['casting-direction-input'].oninput();
  const key = 'voice-lab-casting-v1:fixture:role-one:3:c';
  env.postFeedback = async () => ({ ok: false, status: 409, body: { detail: 'The review round changed.' } });
  await click(starButton(candidateCard(env, 'c'), 3));
  const saved = JSON.parse(env.storage.get(key)); const original = saved.pendingRequest;
  assert.ok(original); assert.equal(original.raw_direction, ''); assert.equal(saved.raw_direction, 'Softer consonants');
  assert.match(ratingStatus(candidateCard(env, 'c')).textContent, /Save not confirmed/);
  env.postFeedback = async body => ({ accepted: true, event_id: body.event_id, status: 'saved' });
  await click(buttonByText(candidateCard(env, 'c'), 'Retry save'));
  const calls = env.calls.filter(row => row.url.endsWith('/casting/feedback')).map(row => JSON.parse(row.request.body));
  assert.deepEqual(calls[0], calls[1]); assert.equal(env.storage.has(key), true, 'unsent direction draft remains after rating acknowledgment');
  assert.equal(JSON.parse(env.storage.get(key)).raw_direction, 'Softer consonants');
});

test('Save & use requires 4–5 stars; a saved action keeps favorite outside its payload', async () => {
  const env = fixture(); await mount(env.doc, env.win);
  const card = candidateCard(env, 'a'); const use = buttonByText(card, 'Use this voice');
  assert.equal(use.hidden, true, 'unrated candidates do not offer approval');
  await click(starButton(card, 5));
  assert.equal(buttonByText(candidateCard(env, 'a'), 'Use this voice').hidden, false, 'Use appears only after the save acknowledgment');
  await click(buttonByText(candidateCard(env, 'a'), 'Use this voice'));
  const sent = env.calls.filter(row => row.url.endsWith('/casting/feedback')).map(row => JSON.parse(row.request.body));
  assert.equal(sent[0].action, 'save'); assert.equal(sent[0].stars, 5);
  assert.equal(sent[1].action, 'use'); assert.equal(sent[1].stars, 5); assert.equal(Object.hasOwn(sent[1], 'favorite'), false);
});

test('feedback conflict=true is acknowledged as historical and never invents current approval', async () => {
  const env = fixture(); await mount(env.doc, env.win);
  await click(buttonByText(candidateCard(env, 'a'), 'Play'));
  env.postFeedback = async () => ({ ok: false, status: 409, body: { conflict: true, reason: 'Saved historically; newer rating remains current.' } });
  await click(starButton(candidateCard(env, 'a'), 5));
  assert.equal(env.storage.has('voice-lab-casting-v1:fixture:role-one:3:a'), false);
  assert.equal(buttonByText(candidateCard(env, 'a'), 'Use this voice').hidden, true);
  assert.match(ratingStatus(candidateCard(env, 'a')).textContent, /Saved historically/);
  assert.match(env.elements['casting-flash'].textContent, /newer rating remains current/);
  env.doc.events.keydown[0]({ key: '4', target: null, preventDefault() {} });
  await new Promise(resolve => setImmediate(resolve));
  assert.equal(env.calls.filter(row => row.url.endsWith('/casting/feedback')).length, 1, 'stale playback shortcuts cannot resubmit historical feedback');
});

test('a pending feedback request prevents replacement actions but keeps playback available', async () => {
  const env = fixture(); await mount(env.doc, env.win);
  let release; env.postFeedback = body => new Promise(resolve => { release = () => resolve({ accepted: true, event_id: body.event_id }); });
  const card = candidateCard(env, 'a'); const save = click(starButton(card, 4));
  assert.equal(starButton(card, 2).disabled, true);
  assert.equal(buttonByText(card, 'Play').disabled, false);
  await click(buttonByText(card, 'Play'));
  assert.equal(env.calls.filter(row => row.url.endsWith('/casting/feedback')).length, 1);
  release(); await save;
});

test('keyboard ratings stay bound to a paused casting audition and never mutate library favorites', async () => {
  const env = fixture(); await mount(env.doc, env.win);
  await click(buttonByText(candidateCard(env, 'a'), 'Play'));
  await click(buttonByText(candidateCard(env, 'a'), 'Pause'));
  const fire = key => env.doc.events.keydown[0]({ key, target: null, preventDefault() {} });
  fire('4'); await new Promise(resolve => setImmediate(resolve));
  const feedback = env.calls.filter(row => row.url.endsWith('/casting/feedback')).map(row => JSON.parse(row.request.body));
  assert.equal(feedback.length, 1); assert.equal(feedback[0].candidate_id, 'a'); assert.equal(feedback[0].stars, 4);
  fire('f'); fire('ArrowLeft'); await new Promise(resolve => setImmediate(resolve));
  assert.equal(env.calls.some(row => row.url.endsWith('/preferences')), false);
});

test('reload restores a uniquely matched audition as character playback, preserving rating scope', async () => {
  const env = fixture();
  env.base.voices = [{ ...candidate('a'), name: 'Library nickname' }];
  env.base.review_position = { key: 'voice-a', audio_sha256: candidate('a').audio_sha256, seconds: 3 };
  const state = await mount(env.doc, env.win);
  assert.equal(state.player.current.casting_context.role_id, 'role-one');
  assert.equal(env.elements['now-playing'].textContent, 'Mara · Version a');
  assert.equal(state.player.media.paused, true);
  env.doc.events.keydown[0]({ key: '4', target: null, preventDefault() {} });
  await new Promise(resolve => setImmediate(resolve));
  assert.equal(env.calls.filter(row => row.url.endsWith('/casting/feedback')).length, 1);
  assert.equal(env.calls.some(row => row.url.endsWith('/preferences')), false);
});

test('only the exact approved current candidate is labeled chosen', async () => {
  const env = fixture();
  env.casting = { ...env.casting, roles: [
    { ...env.casting.roles[0], status: 'APPROVED', current_key: 'voice-a', candidates: [candidate('a', 5), candidate('b', 5)] }
  ] };
  await mount(env.doc, env.win);
  env.elements['show-approved-casts'].checked = true; env.elements['show-approved-casts'].onchange();
  assert.ok(find(candidateCard(env, 'a'), el => String(el.className || '').split(/\s+/).includes('casting-chosen-badge')));
  assert.equal(buttonByText(candidateCard(env, 'a'), 'Use this voice').hidden, true);
  assert.equal(find(candidateCard(env, 'b'), el => String(el.className || '').split(/\s+/).includes('casting-chosen-badge')), null);
  assert.equal(buttonByText(candidateCard(env, 'b'), 'Use this voice').hidden, false);
});

test('direction without stars can be saved or previewed, and status labels remain distinct', async () => {
  const env = fixture(); await mount(env.doc, env.win);
  assert.match(ratingStatus(candidateCard(env, 'a')).textContent, /Not rated yet/);
  const role = find(env.elements['casting-roles'], el => el.tagName === 'ARTICLE' && el.className === 'casting-role');
  assert.ok(role.children.some(el => el.tagName === 'DETAILS'), 'requested/applied/unsupported/unverified direction is exposed');
  await click(buttonByText(candidateCard(env, 'a'), 'Add direction'));
  env.elements['casting-direction-input'].value = 'More breath'; env.elements['casting-direction-input'].oninput();
  assert.equal(env.elements['casting-save'].disabled, false); assert.equal(env.elements['casting-recast'].disabled, false);
  await env.elements['casting-preview-direction'].onclick();
  const sent = JSON.parse(env.calls.find(row => row.url.endsWith('/casting/feedback')).request.body);
  assert.equal(sent.action, 'preview'); assert.equal(sent.raw_direction, 'More breath'); assert.equal(Object.hasOwn(sent, 'stars'), false);
});

test('render and resume are explicit and use the exact backend-returned scope', async () => {
  const env = fixture(); await mount(env.doc, env.win);
  assert.equal(env.elements['render-approved-cast'].disabled, false);
  await env.elements['render-approved-cast'].onclick();
  const render = JSON.parse(env.calls.find(row => row.url.endsWith('/casting/render')).request.body);
  assert.deepEqual(render.utterance_ids, ['u1', 'u2', 'u3']); assert.ok(render.request_id);
  const resume = env.elements['resume-cast-render'];
  resume.dataset.requestId = 'interrupted-7'; resume.hidden = false;
  await resume.onclick();
  const call = env.calls.find(row => row.url.endsWith('/casting/resume'));
  assert.deepEqual(JSON.parse(call.request.body), { request_id: 'interrupted-7' });
});

test('AutoCast disables legacy assignments and routes both library and shortlist actions to the inbox', async () => {
  const env = fixture();
  const voice = { key: 'voice-a', name: 'Cedar', role_id: 'role-one', role_name: 'Mara', version: 'v1',
    audio_url: '/audio/voice-a', audio_sha256: hash(7), duration_seconds: 5 };
  env.base.voices = [voice];
  env.base.roles = [{ id: 'role-one', name: 'Mara', current_identity_id: 'identity-current' }];
  env.base.shortlists = { 'role-one': { matches: [{ candidate_id: 'match-a', match_role_id: 'role-one' }] } };
  env.base.auditions = [{ role_id: 'role-one', candidate_id: 'match-a', status: 'succeeded', key: 'voice-a', request_id: 'audition-a' }];
  let navigated = 0;
  env.elements['casting-inbox'].scrollIntoView = () => { navigated++; };
  env.elements['casting-inbox'].focus = () => {};
  await mount(env.doc, env.win);

  assert.equal(env.elements.assign.hidden, true);
  assert.equal(env.elements.assign.disabled, true);
  assert.equal(env.elements['casting-approve-inbox'].hidden, false);
  assert.equal(env.elements['casting-approve-inbox'].textContent, 'Approve in casting inbox');
  await env.elements.assign.onclick(); // A stale/hidden activation still cannot call the legacy endpoint.
  assert.equal(navigated, 1);
  const shortlist = find(env.elements.matches, el => el.tagName === 'ARTICLE' && el.dataset.candidateId === 'match-a');
  const approve = buttonByText(shortlist, 'Approve in casting inbox');
  assert.ok(approve);
  await click(approve);
  assert.equal(navigated, 2);
  assert.ok(!env.calls.some(row => row.url.endsWith('/assign')));
});

test('casting inbox exposes the exact blocked scope reason without inventing an attribution route', async () => {
  const env = fixture();
  env.casting = { ...env.casting, scope_error: 'Unresolved speaker for seg_abc; review source attribution before casting' };
  await mount(env.doc, env.win);
  assert.equal(env.elements['casting-scope-error'].hidden, false);
  assert.equal(env.elements['casting-scope-error'].textContent, env.casting.scope_error);
  assert.equal(env.elements['casting-scope-error'].children.length, 0);
});

test('eight audition sections remain one candidate and seek the shared full-audio player', async () => {
  const env = fixture();
  const wholeHash = hash(8);
  const audition = { ...candidate('a'), label: 'A', audio_sha256: wholeHash, duration_seconds: 46.16,
    sections: Array.from({ length: 8 }, (_, index) => ({
      section_id: `part-${index + 1}`, label: `Section ${index + 1}`,
      start_seconds: index * 5.5, end_seconds: Math.min(46.16, index * 5.5 + 5),
      audio_sha256: hash(index + 1), technical_qa: 'passed',
      control_status: index === 4 ? { status: 'unsupported', reason: 'base renderer has no delivery-control input' } : undefined
    })) };
  const pending = { ...candidate('c'), status: 'pending', key: null, audio_sha256: null };
  env.casting = { ...env.casting, roles: [{ ...env.casting.roles[0], candidates: [audition, candidate('b'), pending] }] };
  const state = await mount(env.doc, env.win);
  const cards = findAll(env.elements['casting-roles'].children[0], el => el.tagName === 'ARTICLE' && !!el.dataset.candidateId);
  assert.equal(cards.length, 3, 'eight section records do not become duplicate A/B/C candidates');
  const cardA = candidateCard(env, 'a');
  assert.ok(cardA);
  // The heading is the stable accessory nickname derived from the candidate
  // ID; the exact ready option label sits beside it as its own labelled node.
  assert.match(find(cardA, el => el.tagName === 'H4').textContent, /^[A-Z][a-z]+ [A-Z][a-z]+$/);
  assert.equal(find(cardA, el => String(el.className || '').split(/\s+/).includes('casting-option-label')).textContent, 'Option A');
  assert.match(find(cardA, el => String(el.className || '').split(/\s+/).includes('casting-code')).textContent, /^N-[0-9A-Z]{1,4}$/);
  assert.match(find(cardA, el => String(el.className || '').split(/\s+/).includes('casting-readiness')).textContent, /Audio ready · 0:46/);
  const cues = find(cardA, el => String(el.className || '').split(/\s+/).includes('casting-section-help'));
  assert.ok(cues, 'section ownership explanation remains available with the seek controls');
  assert.match(cues.textContent, /Section cues jump/);
  assert.match(cues.textContent, /feedback you submit still applies to candidate A/);
  const sectionButtons = [];
  const gather = el => { if (el.tagName === 'BUTTON' && /^Play Section /.test(el.textContent)) sectionButtons.push(el); for (const child of el.children || []) gather(child); };
  gather(cardA);
  assert.equal(sectionButtons.length, 8);
  assert.equal(starButton(candidateCard(env, 'c'), 1).disabled, true, 'pending candidate cannot be rated');
  await click(sectionButtons[4]);
  assert.equal(state.player.current.audio_sha256, wholeHash, 'section hash never replaces the candidate audition hash');
  assert.equal(state.player.current.candidate_id, 'a');
  assert.equal(state.player.current.casting_context.candidate_id, 'a');
  assert.equal(state.player.current.name, 'Mara · A');
  assert.equal(env.elements['now-playing'].textContent, 'Mara · A', 'dock title follows the selected character audition');
  assert.equal(state.player.media.currentTime, audition.sections[4].start_seconds);
  assert.equal(state.player.media.playbackRate, 1);
  assert.match(find(cardA, el => el.textContent.includes('Controls unsupported')).textContent, /base renderer has no delivery-control input/);
  await click(starButton(cardA, 4));
  const body = JSON.parse(env.calls.find(row => row.url.endsWith('/casting/feedback')).request.body);
  assert.equal(body.candidate_id, 'a');
  assert.equal(body.key, 'voice-a');
  assert.equal(body.audio_sha256, wholeHash);
  assert.deepEqual(body.baseline_candidate_ids, ['a', 'b'], 'baseline includes only exact ready candidates');
  assert.equal(body.round_revision, 3, 'API revision remains an integer');
});

test('audition round status keeps casting inbox polling before production exists', async () => {
  const env = fixture();
  env.casting = { ...env.casting, productions: [], roles: [{ ...env.casting.roles[0], round_status: 'running' }] };
  await mount(env.doc, env.win);
  assert.equal(env.intervals.length, 1, 'running audition schedules a refresh without a production receipt');
});

test('unchanged generation polling preserves the actual rating controls and open details', async () => {
  const env = fixture(); env.casting.roles[0].round_status = 'running';
  await mount(env.doc, env.win);
  const card = candidateCard(env, 'a');
  const details = find(card, el => el.tagName === 'DETAILS'); details.open = true;
  env.intervals[0](); await new Promise(resolve => setImmediate(resolve));
  assert.equal(candidateCard(env, 'a'), card, 'a stable poll must preserve focusable DOM nodes');
  assert.equal(details.open, true);
});

test('a poll begun before feedback cannot roll back the acknowledged rating', async () => {
  const env = fixture(); env.casting.roles[0].round_status = 'running';
  const fetch = env.win.fetch; let delayPoll = false; let release;
  env.win.fetch = async (url, options) => {
    const response = await fetch(url, options);
    if (delayPoll && url.endsWith('/casting')) {
      const stale = await response.json();
      return { ok: true, status: 200, json: () => new Promise(resolve => { release = () => resolve(stale); }) };
    }
    return response;
  };
  await mount(env.doc, env.win); const card = candidateCard(env, 'a');
  delayPoll = true; env.intervals[0](); await new Promise(resolve => setImmediate(resolve));
  assert.equal(typeof release, 'function');
  await click(starButton(card, 4));
  release(); await new Promise(resolve => setImmediate(resolve));
  assert.equal(candidateCard(env, 'a'), card);
  assert.equal(starButton(card, 4).attrs['aria-pressed'], 'true');
  assert.equal(buttonByText(card, 'Use this voice').hidden, false);
});

test('backend blocked and source_changed production receipts expose explicit Resume', async () => {
  for (const status of ['blocked', 'source_changed']) {
    const env = fixture();
    const requestId = `${status}-receipt`;
    env.casting = { ...env.casting, productions: [{ request_id: requestId, status, reason: `Held: ${status}` }] };
    await mount(env.doc, env.win);
    const resume = env.elements['resume-cast-render'];
    assert.equal(resume.hidden, false, `${status} receipt is resumable by explicit action`);
    assert.equal(resume.dataset.requestId, requestId);
    await resume.onclick();
    const sent = JSON.parse(env.calls.find(row => row.url.endsWith('/casting/resume')).request.body);
    assert.deepEqual(sent, { request_id: requestId });
    assert.equal(env.elements['resume-cast-render'].hidden, true, `${status} response refreshes from canonical review state`);
  }
});

test('casting notification display reads role review-batch state and reason without claiming device delivery', async () => {
  const env = fixture();
  env.casting = { ...env.casting, productions: [], roles: [
    { ...env.casting.roles[0], notification: { state: 'accepted' } },
    { ...env.casting.roles[1], notification: { state: 'blocked', reason: 'Private review sharing is not authorized.' } }
  ] };
  await mount(env.doc, env.win);
  const status = env.elements['casting-notification-status'].textContent;
  assert.match(status, /Mara: accepted/);
  assert.match(status, /Provider accepted the notification; device delivery is unobserved/);
  assert.match(status, /Approved guide: blocked · Private review sharing is not authorized/);
  assert.doesNotMatch(status, /delivered to (the )?phone/i);
});

test('a notification deep link opens the exact character review without writing anything', async () => {
  const env = fixture({ search: '?project=fixture&character=role-one' });
  env.casting = { ...env.casting, roles: [
    { ...env.casting.roles[0], candidates: env.casting.roles[0].candidates.map((row, index) => ({ ...row, ready_at: `2026-09-28T0${index}:00:00+00:00` })) },
    env.casting.roles[1]
  ] };
  const state = await mount(env.doc, env.win);
  assert.equal(env.elements['casting-inbox'].hidden, false);
  assert.deepEqual(env.calls.filter(row => row.request?.method === 'POST'), [], 'opening a deep link never dispatches work');
  assert.equal(state.player.current?.key, undefined, 'a deep link prepares the review but does not autoplay audio');
  assert.equal(env.elements['casting-feedback-role'].textContent, 'Mara · Version c',
    'the newest reviewable candidate is the obvious one');
  assert.equal(candidateCard(env, 'c').dataset.audioState, 'ready');
});

test('a casting deep link lands on the review when the named character is gone', async () => {
  const env = fixture({ search: '?project=fixture&character=role-missing' });
  await mount(env.doc, env.win);
  assert.deepEqual(env.calls.filter(row => row.request?.method === 'POST'), []);
  assert.match(env.elements['casting-flash'].textContent, /no longer in the current casting review/);
  assert.equal(env.elements['casting-roles'].children.length, 2, 'chosen characters remain visible in the portrait view');
});

test('a casting overview deep link focuses the inbox without selecting a candidate', async () => {
  const env = fixture({ search: '?project=fixture&view=casting' });
  const state = await mount(env.doc, env.win);
  assert.deepEqual(env.calls.filter(row => row.request?.method === 'POST'), []);
  assert.equal(env.elements['casting-inbox'].hidden, false);
  assert.equal(state.player.current?.key, undefined);
});

test('every proposed candidate shows the reference performance and the reason it was proposed', async () => {
  const env = fixture();
  const anchor = { schema: 'casting-anchor-v1', performer: 'David Attenborough',
    performance: 'Planet Earth (2006), UK narration', matched_traits: ['british', 'male', 'mature', 'warm'],
    rationale: 'Matched vocal traits: british, male, mature, warm.', anchored: true };
  env.casting = { ...env.casting, roles: [{ ...env.casting.roles[0],
    candidates: env.casting.roles[0].candidates.map((row, index) => index === 0
      ? { ...row, anchor, inspiration: { performer: anchor.performer, performance: anchor.performance, source_urls: ['https://example.test/credits'] }, match_rationale: anchor.rationale }
      : index === 1
        ? { ...row, prior_match: true, provenance: 'prior_voice', library_voice_id: 'vl_example', library_version: 'v1',
            prior_evidence: { schema: 'preferred-voice-v1', prior_match: true, prior_rating: 5, match_percent: 91,
              match_reasons: ['presentation: masculine', 'perceived vocal age: older adult', 'accent: British'],
              previously_used: 'Elf_1_10 AutoCast Acceptance / Narrator', projects: ['ba9786ea'] },
            match_rationale: 'Previously approved voice · prior rating 5 stars · 91% match' }
        : { ...row, anchor: { anchored: false, unanchored_reason: 'No catalogue reference fits this character.' } }) }] };
  await mount(env.doc, env.win);

  const evidenceOf = id => {
    const block = find(candidateCard(env, id), el => el.className === 'casting-evidence');
    return (block?.children || []).map(row => row.textContent);
  };
  const generated = evidenceOf('a').join('\n');
  assert.match(generated, /Inspiration: David Attenborough, Planet Earth \(2006\), UK narration/);
  assert.match(generated, /Match rationale: british, male, mature, warm/);

  // A prior approved voice states its rating, its fit and where it was used, and
  // is still only a candidate: nothing about it is pre-selected or approved.
  const prior = evidenceOf('b').join('\n');
  assert.match(prior, /Previously approved voice · prior rating 5 stars · 91% match/);
  assert.match(prior, /Relevant traits: presentation: masculine, perceived vocal age: older adult, accent: British/);
  assert.match(prior, /Previously used in Elf_1_10 AutoCast Acceptance \/ Narrator/);
  const priorCard = candidateCard(env, 'b');
  assert.equal(buttonByText(priorCard, 'Use this voice').hidden, true);
  assert.equal(find(priorCard, el => String(el.className || '').includes('casting-chosen-badge')), null);

  assert.match(evidenceOf('c').join('\n'), /Unanchored: No catalogue reference fits this character/);
});

test('a blocked candidate states why no voice was generated instead of offering one', async () => {
  const env = fixture();
  env.casting = { ...env.casting, roles: [{ ...env.casting.roles[0],
    candidates: env.casting.roles[0].candidates.map((row, index) => index === 0
      ? { ...row, blocked: true, blocked_reason: 'No catalogue performance fits this candidate\'s casting concept.',
          anchor: { anchored: false, unanchored_reason: 'No catalogue performance fits this candidate\'s casting concept.' } }
      : row) }] };
  await mount(env.doc, env.win);
  const block = find(candidateCard(env, 'a'), el => el.className === 'casting-evidence');
  assert.match(block.children.map(row => row.textContent).join('\n'), /^Blocked: No catalogue performance fits/);
});

test('book-supported traits and casting inference are shown as two separate groups', async () => {
  const env = fixture();
  env.casting = { ...env.casting, roles: [{ ...env.casting.roles[0],
    candidates: env.casting.roles[0].candidates.map((row, index) => index === 0
      ? { ...row,
          supported_traits: [{ dimension: 'gender', value: 'male', level: 'SUPPORTED', evidence_ids: ['fact-gender'], basis: 'explicit gender evidence' }],
          inferred_traits: [{ dimension: 'accent', value: 'British (neutral RP)', level: 'CASTING_INFERENCE', evidence_ids: [], basis: 'Casting inference: the book establishes no accent' }] }
      : row) }] };
  await mount(env.doc, env.win);
  const card = candidateCard(env, 'a');
  // The groups now live in the card back (inspector), under CHARACTER and CASTING INFERENCE.
  const groups = findAll(candidateCard(env, 'a'), el => el.className === 'casting-traits');
  assert.equal(groups.length, 2);
  const labels = groups.map(group => group.children[0].textContent);
  assert.deepEqual(labels, ['Book-supported traits', 'Casting inference (not in the book)']);
  assert.deepEqual([...groups[0].children[1].children].map(row => row.textContent), ['gender: male']);
  assert.deepEqual([...groups[1].children[1].children].map(row => row.textContent), ['accent: British (neutral RP)']);
  // The inference is never rendered as though the book had said it.
  assert.doesNotMatch(groups[0].children[1].textContent, /accent/);
});

test('the acoustic characterization and the parameters sent to TTS are shown, labelled measured vs derived', async () => {
  const env = fixture();
  env.casting = { ...env.casting, roles: [{ ...env.casting.roles[0],
    candidates: env.casting.roles[0].candidates.map((row, index) => index === 0
      ? { ...row,
          reference: { performer: 'David Attenborough', performance: 'Planet Earth (2006), UK narration',
            characterization: 'inferred', characterization_method: 'descriptor-prior-v1',
            retrieval_state: 'not_ingested', retrieval_reason: 'No permitted reference audio is available for this anchor',
            measured_properties: [], inferred_properties: ['f0_median_hz', 'words_per_minute'] },
          conditioning: { parameters: { pitch_target_hz: 97.4, pitch_range_semitones: 6.0, rate_words_per_minute: 119,
            resonance: 'deep-chested', vocal_weight: 'heavy', age_impression: 'older adult', rasp: 0.18, breathiness: 0.14 } },
          acoustics: { shaping_plan: { semitones: -1.4, rate: 0.93 }, achieved: true,
            reach: { verdict: 'reached', reached: ['f0_median_hz', 'words_per_minute'], shortfall: [], out_of_scope: [], out_of_scope_count: 0 } } }
      : row) }] };
  await mount(env.doc, env.win);
  const block = find(candidateCard(env, 'a'), el => el.className === 'casting-acoustics');
  const text = block.children.map(row => row.textContent).join('\n');
  assert.match(text, /Acoustic characterization: inferred · descriptor-prior-v1/);
  assert.match(text, /Derived \(documented prior, not measured\): f0_median_hz, words_per_minute/);
  assert.match(text, /Generation parameters sent to TTS: 97 Hz, 6.0 st range, 119 wpm, deep-chested resonance, heavy weight, older adult impression, rasp 0\.18, breathiness 0\.14/);
  assert.match(text, /Acoustic shaping applied: -1\.40 semitones, x0\.930 rate/);
  assert.doesNotMatch(text, /Target not fully reached/);
  assert.doesNotMatch(text, /could not be measured/);
});

test('an unreached acoustic target is stated as a shortfall, and out-of-scope keys are not counted against it', async () => {
  const env = fixture();
  env.casting = { ...env.casting, roles: [{ ...env.casting.roles[0],
    candidates: env.casting.roles[0].candidates.map((row, index) => index === 0
      ? { ...row,
          acoustics: { shaping_plan: { semitones: 3.0, rate: 0.95 },
            reach: { verdict: 'shortfall', reached: ['f0_mean_hz'], shortfall: ['f0_median_hz'],
              out_of_scope: ['peak_dbfs', 'voiced_fraction'], out_of_scope_count: 2 } } }
      : row) }] };
  await mount(env.doc, env.win);
  const text = find(candidateCard(env, 'a'), el => el.className === 'casting-acoustics')
    .children.map(row => row.textContent).join('\n');
  // Four independent verdicts, each its own line: reached, shortfall and
  // out_of_scope are all named distinctly rather than folding out-of-scope
  // into a parenthetical count on the shortfall line.
  assert.match(text, /Target reached: f0_mean_hz/);
  assert.match(text, /Target not fully reached: f0_median_hz still outside tolerance/);
  assert.match(text, /Out of scope: peak_dbfs, voiced_fraction cannot be moved by any pitch\/rate transform/);
});

test('an unmeasurable acoustic target is never rendered as reached', async () => {
  const env = fixture();
  env.casting = { ...env.casting, roles: [{ ...env.casting.roles[0],
    candidates: env.casting.roles[0].candidates.map((row, index) => index === 0
      ? { ...row, acoustics: { shaping_plan: { semitones: 0, rate: 1.0 },
          reach: { verdict: 'could_not_measure', reached: [], shortfall: [], out_of_scope: ['peak_dbfs'], out_of_scope_count: 1 } } }
      : row) }] };
  await mount(env.doc, env.win);
  const text = find(candidateCard(env, 'a'), el => el.className === 'casting-acoustics')
    .children.map(row => row.textContent).join('\n');
  assert.match(text, /could not be measured: the comparison touched no key/);
  assert.doesNotMatch(text, /Target not fully reached/);
});

test('a blocking import of a prior voice is stated instead of an invented replacement', async () => {
  const env = fixture();
  env.casting = { ...env.casting, roles: [{ ...env.casting.roles[0],
    candidates: env.casting.roles[0].candidates.map((row, index) => index === 0
      ? { ...row, prior_match: true, library_voice_id: 'vl_example', library_version: 'v1',
          import_state: 'blocked', import_reason: 'Library voice version is unavailable',
          prior_evidence: { prior_match: true, prior_rating: 4, match_percent: 62, match_reasons: ['accent: British'] } }
      : row) }] };
  await mount(env.doc, env.win);
  const block = find(candidateCard(env, 'a'), el => el.className === 'casting-evidence');
  assert.match(block.children.map(row => row.textContent).join('\n'),
    /This voice is unavailable here: Library voice version is unavailable/);
});

test('a low rating never restarts the round and leaves the remaining options scoreable', async () => {
  const env = fixture();
  const ratings = new Map();
  env.postFeedback = async body => {
    ratings.set(body.candidate_id, body.stars);
    env.casting = { ...env.casting, roles: env.casting.roles.map(role => role.role_id !== 'role-one' ? role : {
      ...role, replacement_request: { schema: 'casting-replacement-v1', status: 'waiting_for_review', round_id: 'round-one',
        reason: 'Waiting for the remaining auditions in this round to be rated' },
      candidates: role.candidates.map(row => ratings.has(row.candidate_id)
        ? { ...row, rating: ratings.get(row.candidate_id),
            needs_replacement: ratings.get(row.candidate_id) <= 3 ? { rating: ratings.get(row.candidate_id) } : null }
        : row) }) };
    return { accepted: true, event_id: body.event_id, saved: true, round_revision: 3 };
  };
  const state = await mount(env.doc, env.win);
  await click(starButton(candidateCard(env, 'a'), 2));

  assert.equal(env.calls.filter(row => row.url.endsWith('/casting/round')).length, 0,
    'a low rating must never request a replacement round by itself');
  assert.equal(env.calls.filter(row => row.url.endsWith('/casting/feedback')).length, 1);
  const card = candidateCard(env, 'a');
  assert.equal(starButton(card, 2).attrs['aria-pressed'], 'true', 'the recorded rating stays visible');
  assert.match(find(card, el => String(el.className || '').includes('casting-replacement-badge')).textContent, /Replacement requested \(2 stars\)/);
  for (const id of ['b', 'c']) {
    const other = candidateCard(env, id);
    assert.equal(buttonByText(other, 'Play').disabled, false, `option ${id} stays playable`);
    assert.equal(starButton(other, 5).disabled, false, `option ${id} stays scoreable`);
  }
  assert.match(env.elements['casting-flash'].textContent, /Keep listening: the replacement arrives as a new round/);
  assert.match(find(env.elements['casting-roles'], el => el.className === 'casting-round-request-status').textContent,
    /Waiting for the remaining auditions in this round to be rated/);
  assert.equal(state.player.current, null, 'the rated option is not played for the human');
});

// ---- Visual fidelity pass: card identity, finishes and independent state layers ----
const textOf = (card, className) => find(card, el => String(className).split(/\s+/).every(token => String(el.className || '').split(/\s+/).includes(token)))?.textContent;
const withCandidates = (env, mapper) => { env.casting = { ...env.casting, roles: [{ ...env.casting.roles[0], candidates: env.casting.roles[0].candidates.map(mapper) }] }; };

test('category family comes from recorded presentation metadata and is neutral when unknown', async () => {
  const env = fixture();
  const trait = (dimension, value) => ({ supported_traits: [{ dimension, value }] });
  withCandidates(env, (row, index) => ({ ...row, ...[trait('presentation', 'masculine'), trait('presentation', 'feminine'), trait('presentation', 'androgynous')][index] }));
  await mount(env.doc, env.win);
  assert.deepEqual(['a', 'b', 'c'].map(id => candidateCard(env, id).dataset.family), ['male', 'female', 'unknown']);
  assert.match(textOf(candidateCard(env, 'a'), 'casting-code'), /^M-/); assert.match(textOf(candidateCard(env, 'b'), 'casting-code'), /^F-/); assert.match(textOf(candidateCard(env, 'c'), 'casting-code'), /^N-/);
  // Each card has a real portrait image derived from metadata, not a text glyph.
  const image = find(candidateCard(env, 'a'), el => String(el.className || '').split(/\s+/).includes('casting-circle-img'));
  assert.match(image.src, /^data:image\/svg\+xml/); assert.ok(image.alt.length > 0);
  const robot = fixture(); withCandidates(robot, row => ({ ...row, supported_traits: [{ dimension: 'species', value: 'robot' }] }));
  await mount(robot.doc, robot.win);
  assert.equal(candidateCard(robot, 'a').dataset.family, 'robot');
});

test('sibling candidates for one character never share a nickname', async () => {
  const env = fixture();
  const ids = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l'];
  env.casting = { ...env.casting, roles: [{ ...env.casting.roles[0], candidates: ids.map(id => candidate(id)) }] };
  await mount(env.doc, env.win);
  const names = ids.map(id => find(candidateCard(env, id), el => el.tagName === 'H4').textContent);
  assert.equal(new Set(names).size, ids.length, `nicknames must be unique: ${names.join(', ')}`);
});

test('only the current rating for this character drives the finish; a historical rating is supporting text', async () => {
  const env = fixture();
  withCandidates(env, (row, index) => index === 0
    ? { ...row, rating: null, prior_evidence: { prior_rating: 5, match_percent: 88, projects: ['p1'], previously_used: 'Book One', match_reasons: [] } }
    : index === 1 ? { ...row, rating: 4 } : { ...row, rating: 5 });
  await mount(env.doc, env.win);
  assert.deepEqual(['a', 'b', 'c'].map(id => candidateCard(env, id).dataset.tier), ['ordinary', 'shiny', 'holo']);
  const a = candidateCard(env, 'a');
  assert.match(textOf(a, 'casting-history-prior'), /Prior: rated 5 stars/);
  assert.equal(textOf(a, 'casting-history-current'), 'Current: not rated in this project');
  assert.equal(textOf(a, 'casting-rating-number'), 'Saved rating: not rated');
  assert.equal(buttonByText(a, 'Use this voice').hidden, true, 'a prior 5★ never pre-selects or unlocks assignment');
});

test('saving stars transforms the same card in place; it is never replaced or moved', async () => {
  const env = fixture(); await mount(env.doc, env.win);
  const card = candidateCard(env, 'a'); const siblings = findAll(env.elements['casting-roles'].children[0], el => el.tagName === 'ARTICLE' && !!el.dataset.candidateId);
  assert.equal(card.dataset.tier, 'ordinary');
  await click(starButton(card, 4));
  assert.equal(candidateCard(env, 'a'), card, 'the card node survives the save');
  assert.equal(card.dataset.tier, 'shiny'); assert.equal(textOf(card, 'casting-rating-number'), 'Saved rating: 4/5');
  await click(starButton(card, 5));
  assert.equal(card.dataset.tier, 'holo'); assert.equal(textOf(card, 'casting-rating-number'), 'Saved rating: 5/5');
  assert.equal(candidateCard(env, 'b'), siblings[1], 'siblings keep their nodes and order');
  assert.equal(buttonByText(card, 'Use this voice').hidden, false);
});

test('a low rating keeps the card visible with concise deferred-replacement wording', async () => {
  const env = fixture(); await mount(env.doc, env.win);
  // The backend records the rating and the deferred replacement request; the next refresh returns both.
  env.postFeedback = async body => {
    env.casting = { ...env.casting, revision: env.casting.revision + 1, roles: [{ ...env.casting.roles[0], candidates: env.casting.roles[0].candidates.map(row => row.candidate_id === 'b' ? { ...row, rating: 2, needs_replacement: { rating: 2, event_id: body.event_id } } : row) }] };
    return { accepted: true, event_id: body.event_id, status: 'saved', reason: 'A long backend explanation.' };
  };
  await click(starButton(candidateCard(env, 'b'), 2));
  assert.equal(textOf(candidateCard(env, 'b'), 'casting-rating-status'), 'Feedback saved · New option after this review');
  assert.equal(candidateCard(env, 'b').dataset.tier, 'ordinary');
  assert.equal(buttonByText(candidateCard(env, 'c'), 'Play').disabled, false, 'other candidates stay playable');
  assert.equal(starButton(candidateCard(env, 'c'), 3).disabled, false, 'other candidates stay rateable');
});

test('assignment is its own explicit ribbon and never implies a rating finish', async () => {
  const env = fixture();
  env.casting = { ...env.casting, roles: [{ ...env.casting.roles[0], status: 'APPROVED', current_key: 'voice-a', round_status: 'resolved' }] };
  await mount(env.doc, env.win);
  env.elements['show-approved-casts'].checked = true; env.elements['show-approved-casts'].onchange();
  const card = candidateCard(env, 'a');
  assert.equal(textOf(card, 'casting-chosen-badge'), 'Assigned voice');
  assert.equal(card.dataset.assigned, 'true'); assert.equal(card.dataset.tier, 'ordinary');
  assert.equal(candidateCard(env, 'b').dataset.assigned, 'false');
});

test('reuse evidence is truthful and never invents a MOST USED ranking', async () => {
  const env = fixture();
  withCandidates(env, (row, index) => index === 0 ? { ...row, prior_evidence: { prior_rating: 5, projects: ['p1', 'p2'], match_reasons: [] } }
    : index === 1 ? { ...row, prior_evidence: { prior_rating: 4, projects: ['p1'], match_reasons: [] } } : row);
  await mount(env.doc, env.win);
  assert.equal(textOf(candidateCard(env, 'a'), 'casting-reuse-badge'), 'Reused in 2 projects');
  assert.equal(textOf(candidateCard(env, 'b'), 'casting-reuse-badge'), 'Reused in 1 project');
  assert.equal(textOf(candidateCard(env, 'c'), 'casting-reuse-badge'), undefined);
  assert.equal(findAll(env.elements['casting-roles'], el => /MOST USED/i.test(el.textContent || '')).length, 0, 'reuse history cannot claim play-count popularity');
});

test('the card transport shows the real audition duration and follows real playback', async () => {
  const env = fixture(); const state = await mount(env.doc, env.win);
  const card = candidateCard(env, 'a');
  assert.equal(textOf(card, 'casting-circle-time'), '0:00 / 0:08');
  await click(buttonByText(card, 'Play'));
  state.player.media.currentTime = 2; state.player.media.emit('timeupdate');
  assert.match(textOf(candidateCard(env, 'a'), 'casting-circle-time'), /^0:02 \/ 0:\d\d$/);
  assert.equal(candidateCard(env, 'a').dataset.playing, 'true');
  assert.equal(candidateCard(env, 'b').dataset.playing, 'false', 'only the heard candidate shows the playing treatment');
});

test('A/B Prefer and Keep both are device notes: no rating, no assignment, no server call', async () => {
  const env = fixture(); await mount(env.doc, env.win);
  await click(buttonByText(candidateCard(env, 'a'), 'Compare')); await click(buttonByText(candidateCard(env, 'b'), 'Compare'));
  const roleCard = () => env.elements['casting-roles'].children[0];
  const slots = findAll(roleCard(), el => el.className === 'casting-compare-slot');
  assert.equal(slots.length, 2);
  await click(buttonByText(roleCard(), 'Prefer A'));
  assert.equal(env.calls.filter(row => row.request?.method === 'POST').length, 0, 'preference never contacts the server');
  assert.ok(env.storage.get('voice-lab-preference-v1:fixture:role-one'));
  assert.ok(buttonByText(roleCard(), '✓ Preferred A'));
  await click(buttonByText(roleCard(), 'Keep both'));
  assert.equal(env.calls.filter(row => row.request?.method === 'POST').length, 0);
  assert.equal(JSON.parse(env.storage.get('voice-lab-shortlist-v1:fixture:role-one')).length, 2);
  assert.equal(env.casting.roles[0].candidates.every(row => row.rating == null), true, 'no rating was fabricated');
  assert.equal(env.casting.roles[0].current_key, null, 'nothing was assigned');
  assert.equal(buttonByText(roleCard(), 'Assign this voice').disabled, true, 'assignment still needs its own saved 4–5★');
});

// ---- Audio reliability: polling must never starve playback, failures must be visible ----
test('casting polls never overlap and never run while audio is playing', async () => {
  const env = fixture();
  env.casting = { ...env.casting, roles: [{ ...env.casting.roles[0], round_status: 'pending' }, env.casting.roles[1]] };
  const state = await mount(env.doc, env.win);
  assert.ok(env.intervals.length >= 1, 'a pending round is polled');
  const tick = env.intervals[0];
  const gets = () => env.calls.filter(row => row.url.endsWith('/casting') && !row.request?.method).length;
  const base = gets();
  tick(); tick(); tick();
  assert.equal(gets() - base, 1, 'overlapping ticks are dropped instead of piling requests on the server');
  await new Promise(resolve => setImmediate(resolve)); await new Promise(resolve => setImmediate(resolve));
  state.player.media.paused = false;
  const whilePlaying = gets();
  tick();
  assert.equal(gets(), whilePlaying, 'no poll competes with audio that is playing');
  state.player.media.paused = true;
  tick();
  assert.equal(gets() - whilePlaying, 1, 'polling resumes once audio is paused');
});

test('a slow or failed audio load is shown, and Play retries by reloading', async () => {
  const env = fixture(); const state = await mount(env.doc, env.win);
  await click(buttonByText(candidateCard(env, 'a'), 'Play'));
  const loads = () => state.player.media.history.filter(row => row[0] === 'load').length;
  state.player.media.emit('waiting');
  assert.match(env.elements['player-status'].textContent, /Loading audio/);
  state.player.media.emit('canplay');
  assert.equal(env.elements['player-status'].textContent, '', 'the status clears once audio can play');
  const before = loads();
  state.player.media.error = { code: 4 }; state.player.media.emit('error');
  assert.match(env.elements['player-status'].textContent, /could not be loaded/i);
  assert.equal(textOf(candidateCard(env, 'a'), 'casting-circle-time'), 'Could not load · tap Play to retry');
  assert.equal(candidateCard(env, 'a').dataset.audioFailed, 'true');
  assert.ok(buttonByText(candidateCard(env, 'a'), 'Play'), 'a failed card offers Play, not Pause');
  await click(buttonByText(candidateCard(env, 'a'), 'Play'));
  assert.equal(loads(), before + 1, 'Play after a failure reloads the audio rather than toggling a dead element');
});

test('portraits use a defined colour for every character seed (no undefined fills)', async () => {
  const env = fixture();
  const template = env.casting.roles[0];
  env.casting = { ...env.casting, roles: Array.from({ length: 60 }, (_, index) => ({
    ...template, role_id: 'role-seed-' + index + '-' + 'z'.repeat(index % 9), name: 'Character ' + index,
    candidates: [candidate('seed' + index)] })) };
  await mount(env.doc, env.win);
  const images = findAll(env.elements['casting-roles'], el => String(el.className || '').split(/\s+/).includes('casting-circle-img') || el.className === 'casting-avatar');
  assert.ok(images.length >= 120, 'every card and every character strip has a portrait');
  for (const image of images) assert.ok(!decodeURIComponent(image.src).includes('undefined'), 'no fill is left undefined');
});


// ---- Where is my book? The page a push opens answers that first ----
const bookStatus = (overrides = {}) => ({ schema: 'casting-book-status-v1', project_id: 'fixture', project_name: 'Fixture', state: 'idle',
  voices: { waiting: [], generating: 0, approved: 0, total: 0 }, production: null, chapters: [], block: null, ...overrides });
const chapterRow = (number, status, extra = {}) => ({ chapter_id: `ch${number}`, title: `Chapter title ${number}`, number, status, lines_done: 0, lines_total: 10,
  ...(status === 'ready' ? { export_id: `cast_${String(number).repeat(28)}`, duration_seconds: 754, audio_sha256: hash(number),
    audio_url: `/api/v1/projects/fixture/voice-lab/casting/chapters/cast_${String(number).repeat(28)}/audio`,
    download_url: `/api/v1/projects/fixture/voice-lab/casting/chapters/cast_${String(number).repeat(28)}/audio?download=true` } : {}), ...extra });
function withBook(env, provider) {
  const original = env.win.fetch;
  env.win.fetch = async (url, request) => {
    if (url.endsWith('/casting/book')) { env.calls.push({ url, request }); return { ok: true, status: 200, json: async () => structuredClone(provider()) }; }
    if (url.endsWith('/casting/production/resolve')) {
      env.calls.push({ url, request }); const body = JSON.parse(request.body);
      return { ok: true, status: 200, json: async () => ({ decision: { request_id: body.request_id, mode: body.mode, resolved: 5, skipped: body.mode === 'auto' ? 3 : 0, still_unresolved: 0 },
        book: bookStatus({ state: 'processing', production: { phase: 'running', status: 'queued', lines_done: 0, lines_total: 40, chapters_ready: 0, chapters_total: 4, current_chapter: 'Chapter title 1', reason: null }, chapters: [chapterRow(1, 'rendering')] }) }) };
    }
    const decision = url.match(/\/casting\/production\/(skip|restore|policy)$/);
    if (decision) {
      env.calls.push({ url, request }); const body = JSON.parse(request.body);
      return { ok: true, status: 200, json: async () => ({ decision: { ...body, restored: body.reason ? 4 : 1, skipped: 1 }, book: structuredClone(provider()) }) };
    }
    if (url.endsWith('/casting/resume')) { env.calls.push({ url, request }); return { ok: true, status: 200, json: async () => ({}) }; }
    return original(url, request);
  };
}
// What the panel says, including what sits behind its info icons (shown on hover, so part of the page's text).
const panelText = env => { const out = []; (function walk(el) { if (el.textContent) out.push(el.textContent); if (el.attrs?.['data-tip']) out.push(el.attrs['data-tip']); for (const child of el.children || []) walk(child); })(env.elements['book-status']); return out.join(' | '); };

test('an idle or unrecognised status keeps the panel out of the way', async () => {
  const env = fixture(); await mount(env.doc, env.win);
  assert.equal(env.elements['book-status'].children.length, 0, 'nothing is drawn for a project with no book status');
  const idle = fixture(); withBook(idle, () => bookStatus()); await mount(idle.doc, idle.win);
  assert.equal(idle.elements['book-status'].children.length, 0);
});

test('voices waiting for the human are the first thing shown, each linking to its character', async () => {
  const env = fixture();
  withBook(env, () => bookStatus({ state: 'voices_waiting', voices: { waiting: [{ role_id: 'role-one', name: 'narrator', options: 3, replacement: false }], generating: 0, approved: 1, total: 2 } }));
  await mount(env.doc, env.win);
  const panel = env.elements['book-status'];
  assert.equal(panel.hidden, false); assert.equal(panel.dataset.state, 'voices_waiting');
  assert.match(panelText(env), /1 voice is waiting for you/);
  const chip = find(panel, el => el.className === 'book-voice-chip');
  assert.equal(chip.textContent, 'Narrator · 3 options'); assert.equal(chip.href, '#casting-role-role-one');
  assert.ok(buttonByText(panel, 'Choose voices'));
  assert.equal(find(panel, el => el.className === 'book-bar'), null, 'no progress bar before recording starts');
});

test('a book being recorded shows a real progress bar, the current chapter and finished chapters to play', async () => {
  const env = fixture(); let status = bookStatus({ state: 'processing',
    production: { phase: 'running', status: 'queued', lines_done: 412, lines_total: 1822, chapters_ready: 1, chapters_total: 3, current_chapter: 'Chapter title 2', reason: null },
    chapters: [chapterRow(1, 'ready'), chapterRow(2, 'rendering', { lines_done: 12, lines_total: 40 }), chapterRow(3, 'waiting')] });
  withBook(env, () => status); await mount(env.doc, env.win);
  const panel = env.elements['book-status']; assert.equal(panel.dataset.state, 'processing');
  assert.match(panelText(env), /Recording your book/); assert.match(panelText(env), /1 chapter of 3 ready to listen to\. Each finished chapter is sent to your phone/);
  const bar = find(panel, el => el.className === 'book-bar');
  assert.equal(bar.attrs['aria-valuenow'], '23'); assert.equal(bar.attrs.role, 'progressbar');
  assert.equal(bar.attrs['aria-valuetext'], '412 of 1,822 lines recorded · 1 of 3 chapters ready (23%)');
  const rows = findAll(panel, el => el.className === 'book-chapter');
  assert.deepEqual(rows.map(row => row.dataset.status), ['ready', 'rendering', 'waiting']);
  assert.match(textOf(rows[1], 'book-chapter-meta'), /Recording · 12 of 40 lines/);
  assert.equal(findAll(panel, el => el.tagName === 'BUTTON' && el.className === 'book-play').length, 1, 'only a finished chapter can be played');
  // It keeps itself current: the next poll shows the chapter that just finished.
  const poll = env.intervals.at(-1); assert.ok(poll, 'a book being recorded is polled');
  status = bookStatus({ ...status, production: { ...status.production, lines_done: 900, chapters_ready: 2 },
    chapters: [chapterRow(1, 'ready'), chapterRow(2, 'ready'), chapterRow(3, 'rendering')] });
  poll(); await new Promise(resolve => setImmediate(resolve)); await new Promise(resolve => setImmediate(resolve));
  assert.equal(findAll(env.elements['book-status'], el => el.tagName === 'BUTTON' && el.className === 'book-play').length, 2);
  assert.equal(find(env.elements['book-status'], el => el.className === 'book-bar').attrs['aria-valuenow'], '49');
});

test('a finished book shows every chapter, plays one on tap, and never saves a listening position for it', async () => {
  const env = fixture(); withBook(env, () => bookStatus({ state: 'finished',
    production: { phase: 'finished', status: 'succeeded', lines_done: 40, lines_total: 40, chapters_ready: 2, chapters_total: 2, current_chapter: null, reason: null },
    chapters: [chapterRow(1, 'ready'), chapterRow(2, 'ready')] }));
  const state = await mount(env.doc, env.win);
  const panel = env.elements['book-status']; assert.match(panelText(env), /Your book is ready/); assert.match(panelText(env), /2 chapters recorded/);
  assert.equal(find(panel, el => el.className === 'book-bar').attrs['aria-valuenow'], '100');
  const downloads = findAll(panel, el => el.tagName === 'A' && el.textContent === 'Download');
  assert.equal(downloads.length, 2); assert.match(downloads[0].href, /\/audio\?download=true$/);
  const [first] = findAll(panel, el => el.className === 'book-play');
  await click(first);
  assert.equal(state.player.current.book_chapter, true); assert.equal(state.player.media.paused, false);
  assert.match(state.player.media.src, /\/casting\/chapters\/cast_1{28}\/audio$/);
  assert.equal(first.textContent, 'Pause'); assert.match(first.attrs['aria-label'], /^Pause Chapter title 1$/);
  await click(first); assert.equal(state.player.media.paused, true); assert.equal(first.textContent, 'Play', 'tapping again pauses the same chapter');
  state.player.media.emit('pause'); await new Promise(resolve => setImmediate(resolve));
  assert.deepEqual(env.calls.filter(row => row.request?.method === 'POST'), [], 'playing a chapter never writes to the server');
  const second = findAll(env.elements['book-status'], el => el.className === 'book-play')[1];
  await click(second); assert.match(state.player.media.src, /cast_2{28}/); assert.equal(first.textContent, 'Play', 'only the chapter being heard shows Pause');
});

test('a link to one finished chapter opens it ready to play, without playing it', async () => {
  const env = fixture({ search: '?project=fixture&view=casting&chapter=ch2' });
  withBook(env, () => bookStatus({ state: 'processing',
    production: { phase: 'running', status: 'queued', lines_done: 20, lines_total: 40, chapters_ready: 2, chapters_total: 3, current_chapter: 'Chapter title 3', reason: null },
    chapters: [chapterRow(1, 'ready'), chapterRow(2, 'ready'), chapterRow(3, 'rendering')] }));
  const state = await mount(env.doc, env.win);
  assert.equal(state.player.current.key, `cast_${'2'.repeat(28)}`); assert.equal(state.player.media.paused, true, 'nothing autoplays');
  assert.equal(env.elements['now-playing'].textContent, 'Chapter title 2');
  assert.equal(findAll(env.elements['book-status'], el => el.className === 'book-chapter')[1].dataset.current, 'true');
  const missing = fixture({ search: '?project=fixture&view=casting&chapter=ch3' });
  withBook(missing, () => bookStatus({ state: 'processing', production: { phase: 'running', status: 'queued', lines_done: 1, lines_total: 4, chapters_ready: 0, chapters_total: 1, current_chapter: 'Chapter title 3', reason: null }, chapters: [chapterRow(3, 'rendering')] }));
  const other = await mount(missing.doc, missing.win); assert.equal(other.player.current, null, 'an unfinished chapter is not opened');
});

test('a book blocked on unattributed lines names them and offers only explicit choices', async () => {
  const env = fixture(); withBook(env, () => bookStatus({ state: 'blocked',
    production: { phase: 'blocked', status: 'blocked', lines_done: 0, lines_total: 40, chapters_ready: 0, chapters_total: 4, current_chapter: null, reason: '5 of 40 source utterance(s) have no speaker' },
    chapters: [chapterRow(1, 'waiting')], block: { unresolved: 5, suggested: 3, suggested_mappable: 2, without_evidence: 2, narrator: 'Narrator', actions: ['auto', 'suggested', 'narrator', 'skip'] } }));
  await mount(env.doc, env.win);
  const panel = env.elements['book-status'];
  assert.match(panelText(env), /Your book is waiting on you/);
  assert.match(textOf(panel, 'book-block-text'), /5 dialogue lines have no speaker yet: 2 have a suggested speaker, 3 could not be attributed\. Nothing changes until you choose\./);
  assert.ok(buttonByText(panel, 'Accept the 2 suggested speakers and skip the other 3')); assert.ok(buttonByText(panel, 'Read 5 lines with Narrator'));
  assert.equal(buttonByText(panel, 'Accept the 2 suggested speakers'), null, 'the old accept-only choice would leave the book blocked');
  assert.deepEqual(env.calls.filter(row => row.request?.method === 'POST'), [], 'nothing is decided on load');
  await click(buttonByText(panel, 'Read 5 lines with Narrator'));
  const posts = env.calls.filter(row => row.request?.method === 'POST');
  assert.equal(posts.length, 1); assert.ok(posts[0].url.endsWith('/casting/production/resolve'));
  const body = JSON.parse(posts[0].request.body);
  assert.equal(body.mode, 'narrator'); assert.match(body.request_id, /^event-/);
  assert.equal(env.elements['book-status'].dataset.state, 'processing', 'the panel follows the book once the choice is saved');
  assert.match(panelText(env), /Saved: 5 lines will be read by the narrator\. Recording starts shortly\./);
});

// ---- the chapter being recorded: its own bar, and the time until it is finished ----
const runningBook = (eta, chapterExtra = {}) => bookStatus({ state: 'processing',
  production: { phase: 'running', status: 'running', lines_done: 412, lines_total: 1822, chapters_ready: 1, chapters_total: 3, current_chapter: 'Chapter title 2', reason: null, eta },
  chapters: [chapterRow(1, 'ready'), chapterRow(2, 'rendering', { lines_done: 12, lines_total: 40, ...chapterExtra }), chapterRow(3, 'waiting')] });
const chapterBox = env => find(env.elements['book-status'], el => el.className === 'book-chapter-progress');
const chapterBar = env => find(chapterBox(env), el => el.attrs?.['aria-label'] === 'Chapter progress');
const settle = async () => { await new Promise(resolve => setImmediate(resolve)); await new Promise(resolve => setImmediate(resolve)); };

test('the chapter being recorded has its own progress bar and the time until it finishes', async () => {
  const env = fixture(); withBook(env, () => runningBook({ chapter_seconds: 372, book_seconds: 11100, measured_lines: 12 })); await mount(env.doc, env.win);
  const box = chapterBox(env); assert.ok(box, 'a chapter is being recorded, so it is shown');
  assert.equal(find(box, el => el.tagName === 'STRONG').textContent, 'Now recording “Chapter title 2”');
  assert.equal(textOf(box, 'book-eta'), 'about 6 minutes left');
  const bar = chapterBar(env);
  assert.equal(bar.attrs.role, 'progressbar'); assert.equal(bar.attrs['aria-valuenow'], '30', '12 of 40 lines');
  assert.equal(bar.attrs['aria-valuetext'], 'Now recording “Chapter title 2”: 12 of 40 lines recorded, about 6 minutes left');
  assert.equal(find(box, el => String(el.className).startsWith('book-chapter-lines')).textContent, '12 of 40 lines in this chapter');
  assert.equal(textOf(env.elements['book-status'], 'book-eta-book'), 'about 3 h 5 min left for the whole book');
  const [book] = findAll(env.elements['book-status'], el => el.className === 'book-bar');
  assert.equal(book.attrs['aria-valuenow'], '23', 'the whole-book bar is a separate bar');
});

test('the page says it is still measuring instead of inventing a time', async () => {
  const env = fixture(); withBook(env, () => runningBook({ chapter_seconds: null, book_seconds: null, measured_lines: 2 })); await mount(env.doc, env.win);
  assert.equal(textOf(chapterBox(env), 'book-eta'), 'Estimating time left…');
  assert.match(chapterBar(env).attrs['aria-valuetext'], /12 of 40 lines recorded, estimating time left…$/);
  assert.equal(textOf(env.elements['book-status'], 'book-eta-book'), '', 'no whole-book time until it can be measured');
  const bare = fixture(); withBook(bare, () => runningBook(undefined)); await mount(bare.doc, bare.win);
  assert.equal(textOf(chapterBox(bare), 'book-eta'), 'Estimating time left…', 'a server that sends no estimate is treated the same way');
});

test('time left is rounded the way a person would say it', async () => {
  const cases = [[0, 'less than a minute left'], [44, 'less than a minute left'], [45, 'about 1 minute left'], [90, 'about 2 minutes left'],
    [600, 'about 10 minutes left'], [5340, 'about 89 minutes left'], [5400, 'about 1 h 30 min left'], [7200, 'about 2 h left'], [45000, 'about 12 h 30 min left']];
  for (const [seconds, words] of cases) {
    const env = fixture(); withBook(env, () => runningBook({ chapter_seconds: seconds, book_seconds: seconds, measured_lines: 9 })); await mount(env.doc, env.win);
    assert.equal(textOf(chapterBox(env), 'book-eta'), words, `${seconds} s`);
  }
});

test('the time left is repainted in place, so a countdown never rebuilds the panel or steals focus', async () => {
  const env = fixture(); let status = runningBook({ chapter_seconds: 372, book_seconds: 11100, measured_lines: 12 });
  withBook(env, () => status); await mount(env.doc, env.win);
  const panel = env.elements['book-status']; const eta = find(chapterBox(env), el => el.className === 'book-eta'); const bar = chapterBar(env);
  const before = [...panel.children]; const [play] = findAll(panel, el => el.className === 'book-play');
  status = runningBook({ chapter_seconds: 291, book_seconds: 11020, measured_lines: 13 });
  env.intervals.at(-1)(); await settle();
  assert.equal(find(chapterBox(env), el => el.className === 'book-eta'), eta, 'the same element, not a rebuilt one');
  assert.equal(eta.textContent, 'about 5 minutes left'); assert.match(bar.attrs['aria-valuetext'], /about 5 minutes left$/);
  assert.equal(textOf(panel, 'book-eta-book'), 'about 3 h 4 min left for the whole book');
  assert.ok(panel.children.length === before.length && panel.children.every((child, index) => child === before[index]), 'nothing was rebuilt');
  assert.equal(findAll(panel, el => el.className === 'book-play')[0], play, 'a chapter player button keeps its identity, and so its focus');
  // A line finishing is something a person can see, so that does redraw the bar.
  status = runningBook({ chapter_seconds: 280, book_seconds: 11000, measured_lines: 14 }, { lines_done: 13 });
  env.intervals.at(-1)(); await settle();
  assert.equal(chapterBar(env).attrs['aria-valuenow'], '33'); assert.equal(textOf(chapterBox(env), 'book-eta'), 'about 5 minutes left');
});

test('no chapter time is shown when no chapter is being recorded', async () => {
  const done = fixture(); withBook(done, () => bookStatus({ state: 'finished', production: { phase: 'finished', status: 'succeeded', lines_done: 4, lines_total: 4, chapters_ready: 1, chapters_total: 1, current_chapter: null, reason: null, eta: null }, chapters: [chapterRow(1, 'ready')] }));
  await mount(done.doc, done.win); assert.equal(chapterBox(done), null);
  const blocked = fixture(); withBook(blocked, () => bookStatus({ state: 'blocked', production: { phase: 'blocked', status: 'blocked', lines_done: 0, lines_total: 4, chapters_ready: 0, chapters_total: 1, current_chapter: null, reason: 'x', eta: null }, block: null }));
  await mount(blocked.doc, blocked.win); assert.equal(chapterBox(blocked), null);
  const between = fixture(); withBook(between, () => bookStatus({ state: 'processing', production: { phase: 'running', status: 'running', lines_done: 40, lines_total: 80, chapters_ready: 0, chapters_total: 2, current_chapter: null, reason: null, eta: { chapter_seconds: null, book_seconds: 600, measured_lines: 9 } }, chapters: [chapterRow(1, 'waiting', { lines_done: 40, lines_total: 40 }), chapterRow(2, 'waiting')] }));
  await mount(between.doc, between.win); assert.equal(chapterBox(between), null, 'while a chapter is being assembled there is no chapter to time');
  assert.equal(textOf(between.elements['book-status'], 'book-eta-book'), 'about 10 minutes left for the whole book');
});

const blockedBook = (block, extra = {}) => bookStatus({ state: 'blocked',
  production: { phase: 'blocked', status: 'blocked', request_id: 'autocast_render', lines_done: 0, lines_total: 40, chapters_ready: 0, chapters_total: 4, current_chapter: null, reason: 'x' },
  chapters: [chapterRow(1, 'waiting')], block: { unresolved: 5, suggested: 3, suggested_mappable: 2, without_evidence: 2, narrator: 'Narrator', actions: ['auto', 'suggested', 'narrator', 'skip'], ...block }, ...extra });
const posts = env => env.calls.filter(row => row.request?.method === 'POST');

test('accepting the suggested speakers and skipping the rest is one tap, and says what it did', async () => {
  const env = fixture(); withBook(env, () => blockedBook()); await mount(env.doc, env.win);
  await click(buttonByText(env.elements['book-status'], 'Accept the 2 suggested speakers and skip the other 3'));
  assert.equal(posts(env).length, 1); assert.deepEqual({ ...JSON.parse(posts(env)[0].request.body), request_id: undefined }, { request_id: undefined, mode: 'auto' });
  assert.match(panelText(env), /Saved: 5 speakers accepted, 3 lines skipped\. Recording starts shortly\./);
});

test('the primary choice adapts to what the analyzer could and could not suggest', async () => {
  const none = fixture(); withBook(none, () => blockedBook({ suggested_mappable: 0, suggested: 0, without_evidence: 5, actions: ['narrator', 'skip'] })); await mount(none.doc, none.win);
  assert.ok(buttonByText(none.elements['book-status'], 'Skip the 5 lines with no speaker'), 'nothing to accept, so the choice is to skip');
  assert.equal(findAll(none.elements['book-status'], el => el.tagName === 'BUTTON' && /^Accept/.test(el.textContent)).length, 0);
  await click(buttonByText(none.elements['book-status'], 'Skip the 5 lines with no speaker')); assert.equal(JSON.parse(posts(none)[0].request.body).mode, 'skip');
  const all = fixture(); withBook(all, () => blockedBook({ suggested_mappable: 5, suggested: 5, without_evidence: 0 })); await mount(all.doc, all.win);
  assert.ok(buttonByText(all.elements['book-status'], 'Accept the 5 suggested speakers'), 'when every line has a suggestion there is nothing to skip');
});

const skippedSummary = () => ({ total: 4, groups: [
  { reason: 'unknown_speaker', label: 'No speaker could be found', count: 3, lines: [{ utterance_id: 'u1', text: 'Who said this?', chapter: 'The Gate', detail: null }, { utterance_id: 'u2', text: 'Nobody knows.', chapter: null, detail: null }], more: 1 },
  { reason: 'unspeakable', label: 'The voice model would not speak it', count: 1, lines: [{ utterance_id: 'u9', text: 'Table of Contents URL: https://example.test', chapter: 'Information', detail: 'reached its generation budget' }], more: 0 }] });

test('lines left out are counted, can be shown, and can be brought back a group at a time', async () => {
  const env = fixture(); withBook(env, () => blockedBook({}, { skipped: skippedSummary() })); await mount(env.doc, env.win);
  const panel = env.elements['book-status']; const box = find(panel, el => el.className === 'book-skipped');
  assert.equal(find(box, el => el.tagName === 'STRONG').textContent, '4 lines skipped'); assert.match(panelText(env), /and left out of the recording/);
  assert.equal(findAll(panel, el => el.className === 'book-skipped-group').length, 0, 'the list is tucked away until asked for');
  const toggle = buttonByText(box, 'Show which'); assert.equal(toggle.attrs['aria-expanded'], 'false');
  await click(toggle);
  const groups = findAll(env.elements['book-status'], el => el.className === 'book-skipped-group');
  assert.deepEqual(groups.map(group => group.dataset.reason), ['unknown_speaker', 'unspeakable']);
  const texts = findAll(groups[0], el => el.tagName === 'LI').map(el => el.textContent);
  assert.deepEqual(texts, ['“Who said this?” · The Gate', '“Nobody knows.”', 'and 1 more']);
  assert.equal(buttonByText(env.elements['book-status'], 'Hide').attrs['aria-expanded'], 'true');
  await click(buttonByText(groups[1], 'Bring back 1 line'));
  const body = JSON.parse(posts(env)[0].request.body); assert.equal(body.reason, 'unspeakable'); assert.match(body.request_id, /^event-/);
  assert.ok(posts(env)[0].url.endsWith('/casting/production/restore'));
  assert.match(panelText(env), /4 lines brought back\. Recording resumes shortly\./);
});

test('lines cannot be brought back while the book is being recorded, and that is said', async () => {
  const env = fixture(); withBook(env, () => runningBook({ chapter_seconds: 60, book_seconds: 600, measured_lines: 9 })); await mount(env.doc, env.win);
  const recording = fixture(); withBook(recording, () => ({ ...runningBook({ chapter_seconds: 60, book_seconds: 600, measured_lines: 9 }), skipped: skippedSummary() })); await mount(recording.doc, recording.win);
  await click(buttonByText(recording.elements['book-status'], 'Show which'));
  assert.equal(findAll(recording.elements['book-status'], el => el.tagName === 'BUTTON' && /^Bring back/.test(el.textContent)).length, 0);
  assert.match(panelText(recording), /You can bring lines back when recording pauses or finishes\./);
  assert.equal(find(env.elements['book-status'], el => el.className === 'book-skipped'), null, 'nothing skipped, nothing shown');
});

test('a chapter says how many of its lines were left out', async () => {
  const env = fixture(); withBook(env, () => bookStatus({ state: 'finished',
    production: { phase: 'finished', status: 'succeeded', request_id: 'autocast_render', lines_done: 40, lines_total: 40, chapters_ready: 2, chapters_total: 2, current_chapter: null, reason: null },
    chapters: [chapterRow(1, 'ready', { lines_skipped: 3 }), chapterRow(2, 'ready')] }));
  await mount(env.doc, env.win);
  const [first, second] = findAll(env.elements['book-status'], el => el.className === 'book-chapter');
  assert.equal(textOf(first, 'book-chapter-meta'), '12:34 · 3 lines skipped'); assert.equal(textOf(second, 'book-chapter-meta'), '12:34');
});

test('a line that stopped the book is shown, and can be skipped or tried again', async () => {
  const failed = { utterance_id: 'u7', text: 'Table of Contents URL: https://example.test', chapter: 'Information', error: 'reached its generation budget' };
  const env = fixture(); withBook(env, () => bookStatus({ state: 'waiting',
    production: { phase: 'waiting', status: 'awaiting_review', request_id: 'autocast_render', lines_done: 1, lines_total: 40, chapters_ready: 0, chapters_total: 4, current_chapter: null, reason: 'stopped' },
    chapters: [chapterRow(1, 'waiting')], failed_line: failed }));
  await mount(env.doc, env.win);
  const box = find(env.elements['book-status'], el => el.className === 'book-failed');
  assert.match(panelText(env), /Your book stopped at one line/); assert.equal(find(env.elements['book-status'], el => String(el.className).startsWith('book-reason')), null, 'the failure is not said twice');
  assert.match(textOf(box, 'book-failed-title'), /The book stopped at this line in “Information”:/);
  assert.equal(textOf(box, 'book-failed-text'), '“Table of Contents URL: https://example.test”');
  await click(buttonByText(box, 'Try again'));
  assert.ok(posts(env)[0].url.endsWith('/casting/resume')); assert.deepEqual(JSON.parse(posts(env)[0].request.body), { request_id: 'autocast_render' });
  await click(buttonByText(find(env.elements['book-status'], el => el.className === 'book-failed'), 'Skip this line'));
  const skip = posts(env)[1]; assert.ok(skip.url.endsWith('/casting/production/skip')); assert.deepEqual(JSON.parse(skip.request.body).utterance_ids, ['u7']);
});

test('automatic handling is visible, and can be turned off and on', async () => {
  const env = fixture(); let automatic = true; withBook(env, () => runningBook({ chapter_seconds: 60, book_seconds: 600, measured_lines: 9 }, {}) && { ...runningBook({ chapter_seconds: 60, book_seconds: 600, measured_lines: 9 }), automatic });
  await mount(env.doc, env.win);
  const row = find(env.elements['book-status'], el => el.className === 'book-auto');
  assert.match(textOf(row, 'book-auto-text muted small'), /^Automatic handling is on: lines with no speaker, and lines the voice model will not speak, are skipped/);
  automatic = false; await click(buttonByText(row, 'Turn off'));
  assert.deepEqual(JSON.parse(posts(env)[0].request.body), { enabled: false }, 'no request id: this is a setting, not a decision');
  assert.ok(posts(env)[0].url.endsWith('/casting/production/policy'));
  const after = find(env.elements['book-status'], el => el.className === 'book-auto');
  assert.match(textOf(after, 'book-auto-text muted small'), /^Automatic handling is off/); assert.ok(buttonByText(after, 'Turn on'));
  assert.match(panelText(env), /Automatic handling is off\. The book will wait for you\./);
  const none = fixture(); withBook(none, () => bookStatus({ state: 'voices_waiting', voices: { waiting: [{ role_id: 'r', name: 'Mara', options: 2, replacement: false }], generating: 0, approved: 0, total: 1 } })); await mount(none.doc, none.win);
  assert.equal(find(none.elements['book-status'], el => el.className === 'book-auto'), null, 'no setting to show before there is a book');
});

test('a book that is about to start by itself is followed, one waiting for a person is not', async () => {
  const waiting = fixture(); withBook(waiting, () => bookStatus({ state: 'waiting', production: { phase: 'waiting', status: 'awaiting_review', request_id: 'autocast_render', lines_done: 0, lines_total: 4, chapters_ready: 0, chapters_total: 1, current_chapter: null, reason: 'Your decision was saved; recording resumes.' }, chapters: [chapterRow(1, 'waiting')] }));
  await mount(waiting.doc, waiting.win); assert.equal(waiting.intervals.length, 1, 'a book that has just been re-planned is followed into recording');
  const auto = fixture(); withBook(auto, () => blockedBook({}, { automatic: true })); await mount(auto.doc, auto.win);
  assert.equal(auto.intervals.length, 1, 'a blocked book that handles itself will unblock');
  const manual = fixture(); withBook(manual, () => blockedBook({}, { automatic: false })); await mount(manual.doc, manual.win);
  assert.equal(manual.intervals.length, 0, 'a blocked book that waits for a person is not polled');
});

test('a book being recorded is polled only while it is changing', async () => {
  const done = fixture(); withBook(done, () => bookStatus({ state: 'finished', production: { phase: 'finished', status: 'succeeded', lines_done: 4, lines_total: 4, chapters_ready: 1, chapters_total: 1, current_chapter: null, reason: null }, chapters: [chapterRow(1, 'ready')] }));
  await mount(done.doc, done.win);
  assert.equal(done.intervals.length, 0, 'a finished book needs no polling');
  const blocked = fixture(); withBook(blocked, () => bookStatus({ state: 'blocked', production: { phase: 'blocked', status: 'blocked', lines_done: 0, lines_total: 4, chapters_ready: 0, chapters_total: 1, current_chapter: null, reason: 'x' }, block: null }));
  await mount(blocked.doc, blocked.win);
  assert.equal(blocked.intervals.length, 0, 'a book waiting on the human is not polled either');
});

test('a rating given while another option is still saving waits its turn instead of being dropped', async () => {
  const env = fixture(); await mount(env.doc, env.win);
  const releases = []; env.postFeedback = body => new Promise(resolve => { releases.push(() => resolve({ accepted: true, event_id: body.event_id, status: 'saved' })); });
  const tick = () => new Promise(resolve => setImmediate(resolve));
  const first = click(starButton(candidateCard(env, 'a'), 4));
  assert.equal(starButton(candidateCard(env, 'b'), 5).disabled, false, 'a sibling option stays clickable while this one saves');
  const second = click(starButton(candidateCard(env, 'b'), 5)); await tick();
  const sentSoFar = () => env.calls.filter(row => row.url.endsWith('/casting/feedback')).map(row => JSON.parse(row.request.body));
  assert.equal(sentSoFar().length, 1, 'one save at a time for a character');
  assert.match(ratingStatus(candidateCard(env, 'b')).textContent, /Waiting for the previous rating/, 'the click is acknowledged, not swallowed');
  releases[0](); await first; await tick();
  assert.equal(releases.length, 2, 'the waiting rating goes out as soon as the first one is confirmed');
  releases[1](); await second;
  assert.deepEqual(sentSoFar().map(row => [row.candidate_id, row.stars]), [['a', 4], ['b', 5]]);
  assert.match(ratingStatus(candidateCard(env, 'b')).textContent, /Saved 5 stars/);
});

test('a waiting rating is sent for the revisions the finished save refreshed, not the ones it was clicked under', async () => {
  const env = fixture(); await mount(env.doc, env.win);
  const releases = []; env.postFeedback = body => new Promise(resolve => { releases.push(() => resolve({ accepted: true, event_id: body.event_id, status: 'saved' })); });
  const tick = () => new Promise(resolve => setImmediate(resolve));
  const clickedUnder = env.casting.roles[0].assignment_revision;
  const first = click(starButton(candidateCard(env, 'a'), 2));            // a low rating: the view is refreshed afterwards
  const second = click(starButton(candidateCard(env, 'b'), 5)); await tick();
  env.casting.roles[0].assignment_revision = clickedUnder + 1;             // what the server moved to meanwhile
  releases[0](); await first; await tick();
  releases[1](); await second;
  const sent = env.calls.filter(row => row.url.endsWith('/casting/feedback')).map(row => JSON.parse(row.request.body));
  assert.equal(sent[0].assignment_revision, clickedUnder);
  assert.equal(sent[1].candidate_id, 'b');
  assert.equal(sent[1].assignment_revision, clickedUnder + 1, 'sent against the fresh revision, so the server does not file it as historical');
});

test('several ratings given while one is saving all go out, one at a time, in the order they were given', async () => {
  const env = fixture(); await mount(env.doc, env.win);
  const releases = []; env.postFeedback = body => new Promise(resolve => { releases.push(() => resolve({ accepted: true, event_id: body.event_id, status: 'saved' })); });
  const tick = () => new Promise(resolve => setImmediate(resolve));
  const clicks = [click(starButton(candidateCard(env, 'a'), 4)), click(starButton(candidateCard(env, 'b'), 5)), click(starButton(candidateCard(env, 'c'), 4))];
  await tick();
  for (let sent = 1; sent <= 3; sent++) {
    assert.equal(releases.length, sent, 'rating ' + sent + ' is sent once the one before it is confirmed, and not before');
    releases[sent - 1](); await tick(); await tick();
  }
  await Promise.all(clicks);
  const order = env.calls.filter(row => row.url.endsWith('/casting/feedback')).map(row => JSON.parse(row.request.body).candidate_id);
  assert.deepEqual(order, ['a', 'b', 'c']);
});

const listedCards = env => env.elements['casting-roles'].children.filter(child => child.tagName === 'ARTICLE');

test('the casting header counts the voices left and says so when the last one is chosen', async () => {
  const env = fixture(); env.casting.roles.forEach(role => { role.required = true; });
  env.casting.roles.push({ ...env.casting.roles[0], role_id: 'role-extra', name: 'Extra', required: false });
  await mount(env.doc, env.win);
  assert.match(env.elements['casting-status'].textContent, /^1 of 2 voices chosen · 1 left: Mara · /, 'a character the book does not need is not counted');
  const done = fixture(); done.casting.roles.forEach(role => { role.required = true; }); done.casting.roles[0].status = 'APPROVED';
  await mount(done.doc, done.win);
  assert.match(done.elements['casting-status'].textContent, /^All 2 voices chosen · your book starts recording by itself/);
});

test('a character whose approval stands is not listed as still to choose, even though a later round was reopened', async () => {
  const env = fixture(); env.casting.roles.forEach(role => { role.required = true; });
  Object.assign(env.casting.roles[0], { status: 'NEEDS_REVIEW', round_status: 'awaiting_review', chosen: true });
  await mount(env.doc, env.win);
  assert.equal(listedCards(env).length, 2, 'chosen character portraits stay visible');
  assert.match(env.elements['casting-status'].textContent, /^All 2 voices chosen/);
  env.elements['show-approved-casts'].checked = true; env.elements['show-approved-casts'].onchange();
  assert.equal(listedCards(env).length, 2, 'they are still there on request');
});

test('a confirmed choice stays visible and cannot be submitted twice when reloading fails', async () => {
  const env = fixture(); env.casting.roles.forEach(role => { role.required = true; });
  const real = env.win.fetch; let offline = false;   // the page binds fetch when it mounts, so the wrapper goes in first
  env.win.fetch = (url, request) => (offline && url.endsWith('/casting') && !request?.method) ? Promise.reject(new Error('offline')) : real(url, request);
  await mount(env.doc, env.win);
  await click(starButton(candidateCard(env, 'a'), 5));
  env.postFeedback = async body => { offline = true; return { accepted: true, event_id: body.event_id, status: 'saved', assignment_revision: 6, round_revision: 3 }; };
  await click(buttonByText(candidateCard(env, 'a'), 'Use this voice'));
  assert.equal(listedCards(env).length, 2, 'the confirmed choice stays visible even if reload fails');
  const check = find(env.elements['casting-roles'], el => (el.className || '').includes('casting-check') && el.attrs['aria-pressed'] === 'true');
  assert.ok(check, 'the confirmed choice has a visible checked control');
  const before = env.calls.filter(call => call.url.endsWith('/casting/feedback')).length;
  await click(check);
  assert.equal(env.calls.filter(call => call.url.endsWith('/casting/feedback')).length, before, 'checking the chosen voice again is a no-op');
  assert.match(env.elements['casting-flash'].textContent, /Mara chosen · that was the last voice/);
  assert.match(env.elements['casting-status'].textContent, /^All 2 voices chosen/);
});

test('choosing a voice says how many are left', async () => {
  const env = fixture(); env.casting.roles.forEach(role => { role.required = true; });
  env.casting.roles.push({ ...structuredClone(env.casting.roles[0]), role_id: 'role-three', name: 'Joss', round_revision: 1 });
  await mount(env.doc, env.win);
  await click(starButton(candidateCard(env, 'a'), 5));
  env.postFeedback = async body => { const row = env.casting.roles.find(role => role.role_id === body.role_id); Object.assign(row, { status: 'APPROVED', chosen: true, round_status: 'resolved' }); return { accepted: true, event_id: body.event_id, status: 'saved' }; };
  await click(buttonByText(candidateCard(env, 'a'), 'Use this voice'));
  assert.match(env.elements['casting-flash'].textContent, /Mara chosen · 1 left: Joss/);
});

// The page's capture listener sees a press before the control's own handler runs, as a browser dispatches it.
const press = (env, control) => (env.doc.events.click || []).forEach(listener => listener({ target: { closest: () => control } }));

test('the control a person pressed says it is working until its work is done', async () => {
  const env = fixture(); await mount(env.doc, env.win);
  let release; env.postFeedback = body => new Promise(resolve => { release = () => resolve({ accepted: true, event_id: body.event_id, status: 'saved' }); });
  const star = starButton(candidateCard(env, 'a'), 4);
  press(env, star); const save = click(star);
  assert.equal(star.attrs['aria-busy'], 'true', 'busy while the save is in flight');
  release(); await save;
  assert.equal(star.attrs['aria-busy'], 'false', 'and not once it is done');
});

test('work nobody pressed for never marks a control busy', async () => {
  const env = fixture(); await mount(env.doc, env.win);
  const first = starButton(candidateCard(env, 'a'), 4); const other = starButton(candidateCard(env, 'b'), 3);
  press(env, first); await click(first);
  let release; env.postFeedback = body => new Promise(resolve => { release = () => resolve({ accepted: true, event_id: body.event_id, status: 'saved' }); });
  const later = click(other);               // no press seen: a timer, a keyboard shortcut handled elsewhere
  assert.equal(first.attrs['aria-busy'], 'false', 'a press is used once, by the work it started');
  assert.equal(other.attrs['aria-busy'], undefined);
  release(); await later;
});

const fillWidth = bar => (/--p:([0-9.]+)%/.exec(bar.children[0].attrs.style || '') || [])[1];

test('a progress bar rebuilt by the next poll starts where it was and eases to the new value', async () => {
  const env = fixture(); const frames = []; env.win.requestAnimationFrame = fn => { frames.push(fn); return frames.length; };
  const flush = () => { while (frames.length) frames.shift()(); };
  let status = bookStatus({ state: 'processing',
    production: { phase: 'running', status: 'queued', lines_done: 412, lines_total: 1822, chapters_ready: 1, chapters_total: 3, current_chapter: 'Chapter title 2', reason: null },
    chapters: [chapterRow(1, 'ready'), chapterRow(2, 'rendering', { lines_done: 12, lines_total: 40 }), chapterRow(3, 'waiting')] });
  withBook(env, () => status); await mount(env.doc, env.win); flush();
  const book = () => find(env.elements['book-status'], el => el.className === 'book-bar');
  assert.equal(fillWidth(book()), '23', 'the first bar is drawn at its value');
  status = bookStatus({ ...status, production: { ...status.production, lines_done: 900 } });
  env.intervals.at(-1)(); await new Promise(resolve => setImmediate(resolve)); await new Promise(resolve => setImmediate(resolve));
  assert.equal(fillWidth(book()), '23', 'the rebuilt bar starts at the width it last showed');
  assert.match(book().children[0].attrs.style, /animation-delay:-\d+ms/, 'its glint carries its phase from the first frame');
  flush();
  assert.equal(fillWidth(book()), '49', 'and then moves to the new value, which the bar animates');
  assert.match(book().children[0].attrs.style, /animation-delay:-\d+ms/, 'its glint keeps its phase instead of restarting');
});

test('a bar with nothing recorded yet shows it is working rather than sitting empty', async () => {
  const env = fixture();
  withBook(env, () => bookStatus({ state: 'processing',
    production: { phase: 'running', status: 'queued', lines_done: 0, lines_total: 100, chapters_ready: 0, chapters_total: 2, current_chapter: 'Chapter title 1', reason: null },
    chapters: [chapterRow(1, 'rendering', { lines_done: 0, lines_total: 50 }), chapterRow(2, 'waiting')] }));
  await mount(env.doc, env.win);
  const [book, chapter] = findAll(env.elements['book-status'], el => (el.className || '').startsWith('book-bar') && el.attrs.role === 'progressbar');
  assert.equal(book.dataset.indeterminate, 'true'); assert.equal(chapter.dataset.indeterminate, 'true');
  assert.equal(book.attrs['aria-valuenow'], '0', 'still an honest 0% to assistive technology');
});

// ---- the section bar -----------------------------------------------------------------------------------------
function navDom(visible) {
  const make = (tag, id) => { const el = { tagName: tag, id, hidden: false, open: false, children: [], attrs: {}, listeners: {}, top: 9999, textContent: '',
    append(...kids) { this.children.push(...kids); }, addEventListener(type, fn) { (this.listeners[type] ||= []).push(fn); },
    setAttribute(k, v) { this.attrs[k] = v; }, removeAttribute(k) { delete this.attrs[k]; }, closest() { return this.hiddenAbove ? this : null; },
    getBoundingClientRect() { return { top: this.top }; } }; return el; };
  const ids = ['page-nav', 'book-status', 'casting-inbox', 'listening-notes', 'casting-extra-tools', 'finished-library', 'project-picker', 'book-creator', 'voice-lab'];
  const els = Object.fromEntries(ids.map(id => [id, make(id === 'casting-extra-tools' ? 'DETAILS' : 'DIV', id)]));
  for (const id of ids) if (id !== 'page-nav' && id !== 'voice-lab' && !visible.includes(id)) els[id].hiddenAbove = true;
  const doc = { getElementById: id => els[id], createElement: tag => make(tag.toUpperCase()), body: make('BODY') };
  const frames = []; const mutated = [];
  const win = { requestAnimationFrame: fn => frames.push(fn), addEventListener() {}, MutationObserver: class { constructor(fn) { mutated.push(fn); } observe() {} } };
  return { doc, win, els, frames, mutated };
}
const linkFor = (env, id) => env.els['page-nav'].children.find(link => link.href === `#${id}`);

test('the section bar lists only the sections that are showing', () => {
  const env = navDom(['casting-inbox', 'listening-notes']);
  require('../dubbing_service/webui/voice-lab.js').startSectionNav(env.doc, env.win);
  const shown = env.els['page-nav'].children.filter(link => !link.hidden).map(link => link.textContent);
  assert.deepEqual(shown, ['Voices', 'Notes']);
  assert.equal(env.els['page-nav'].hidden, false);
});

test('the section bar hides itself when nothing on the page has anything to link to', () => {
  const env = navDom([]);
  require('../dubbing_service/webui/voice-lab.js').startSectionNav(env.doc, env.win);
  assert.equal(env.els['page-nav'].hidden, true);
});

test('the section bar marks the section being read and follows the page as it changes', () => {
  const env = navDom(['casting-inbox', 'listening-notes']);
  const nav = require('../dubbing_service/webui/voice-lab.js').startSectionNav(env.doc, env.win);
  env.els['casting-inbox'].top = 20; env.els['listening-notes'].top = 900; nav.refresh();
  assert.equal(linkFor(env, 'casting-inbox').attrs['aria-current'], 'location');
  assert.equal(linkFor(env, 'listening-notes').attrs['aria-current'], undefined);
  env.els['listening-notes'].top = 100; nav.refresh();
  assert.equal(linkFor(env, 'listening-notes').attrs['aria-current'], 'location');
  assert.equal(linkFor(env, 'casting-inbox').attrs['aria-current'], undefined, 'only one section is marked');
  env.els['book-status'].hiddenAbove = false; env.mutated[0](); env.frames.shift()();
  assert.equal(linkFor(env, 'book-status').hidden, false, 'a section that appears is added to the bar');
});

test('choosing a collapsed section opens it', () => {
  const env = navDom(['casting-extra-tools']);
  require('../dubbing_service/webui/voice-lab.js').startSectionNav(env.doc, env.win);
  linkFor(env, 'casting-extra-tools').listeners.click[0]();
  assert.equal(env.els['casting-extra-tools'].open, true);
});

// ---- explanations sit behind info icons; the liquid settle after a click ----------------------------------------
const tipsIn = el => findAll(el, node => node.className === 'info-tip');

test('a book being recorded keeps its explanation behind an info icon instead of a paragraph', async () => {
  const env = fixture(); const status = bookStatus({ state: 'processing',
    production: { phase: 'running', status: 'queued', lines_done: 412, lines_total: 1822, chapters_ready: 1, chapters_total: 3, current_chapter: 'Chapter title 2', reason: null },
    chapters: [chapterRow(1, 'ready'), chapterRow(2, 'rendering', { lines_done: 12, lines_total: 40 }), chapterRow(3, 'waiting')] });
  withBook(env, () => status); await mount(env.doc, env.win);
  const panel = env.elements['book-status'];
  assert.equal(find(panel, el => el.className === 'book-detail'), null, 'no explanatory paragraph on the page');
  const [tip] = tipsIn(panel);
  assert.match(tip.attrs['data-tip'], /1 chapter of 3 ready to listen to/);
  assert.equal(tip.attrs['aria-label'], tip.attrs['data-tip']); assert.equal(tip.attrs.role, 'img'); assert.equal(tip.tabIndex, 0);
});

test('a book that is waiting on the person still says why, in plain sight', async () => {
  const env = fixture(); const status = bookStatus({ state: 'blocked', production: null });
  withBook(env, () => status); await mount(env.doc, env.win);
  const panel = env.elements['book-status'];
  assert.match(find(panel, el => el.className === 'book-detail').textContent, /some lines have no speaker yet/, 'what they must act on is not hidden behind an icon');
  assert.equal(tipsIn(panel).length, 0);
});

test('a click leaves its button settling for about two seconds, and a second click restarts that', async () => {
  const env = fixture(); const timers = [], cleared = [];
  env.win.setTimeout = (fn, ms) => { timers.push({ fn, ms }); return timers.length; }; env.win.clearTimeout = id => cleared.push(id);
  await mount(env.doc, env.win);
  const classes = new Set(); const control = { disabled: false, offsetWidth: 1, classList: { add: c => classes.add(c), remove: c => classes.delete(c) } };
  const click = () => (env.doc.events.click || []).forEach(listener => listener({ target: { closest: () => control } }));
  click();
  assert.ok(classes.has('settling'), 'the liquid starts moving the moment it is clicked');
  const settle = timers.find(timer => timer.ms === 2000); assert.ok(settle, 'and is given about two seconds');
  click(); assert.ok(cleared.includes(control.settleTimer - 1) || cleared.length > 0, 'a second click cancels the first timer, so it cannot cut the new settle short');
  timers.filter(timer => timer.ms === 2000).at(-1).fn(); assert.ok(!classes.has('settling'), 'when the time is up it is still again');
  control.disabled = true; classes.clear(); click(); assert.ok(!classes.has('settling'), 'a disabled button does not slosh');
});

// ---- the liquid flows only where it can be seen --------------------------------------------------------------
function liquidDom() {
  const make = (tag, isButton = true) => ({ tagName: tag, attrs: {}, children: [], nodeType: 1,
    toggleAttribute(name, on) { if (on) this.attrs[name] = ''; else delete this.attrs[name]; },
    matches() { return isButton; }, querySelectorAll() { return this.children; } });
  const buttons = [make('BUTTON'), make('BUTTON')];
  const classes = new Set();
  const root = { querySelectorAll: () => buttons, documentElement: { classList: { add: c => classes.add(c) } }, getElementById: () => ({}), body: {} };
  const observed = []; let report = null; let mutate = null;
  const win = {
    IntersectionObserver: class { constructor(fn) { report = fn; } observe(el) { observed.push(el); } },
    MutationObserver: class { constructor(fn) { mutate = fn; } observe() {} },
  };
  return { root, win, buttons, classes, observed, make, report: entries => report(entries), mutate: records => mutate(records) };
}
const { startLiquidViewport } = require('../dubbing_service/webui/voice-lab.js');

test('only buttons near the window are marked to flow, and they stop when they leave', () => {
  const env = liquidDom();
  startLiquidViewport(env.root, env.win);
  assert.ok(env.classes.has('liquid-gated'), 'the stylesheet is told the flow is gated');
  assert.deepEqual(env.observed, env.buttons, 'every button is watched');
  env.report([{ target: env.buttons[0], isIntersecting: true }, { target: env.buttons[1], isIntersecting: false }]);
  assert.ok('data-in-view' in env.buttons[0].attrs); assert.ok(!('data-in-view' in env.buttons[1].attrs));
  env.report([{ target: env.buttons[0], isIntersecting: false }]);
  assert.ok(!('data-in-view' in env.buttons[0].attrs), 'a button scrolled away stands still again');
});

test('buttons the page adds later are watched too, once each', () => {
  const env = liquidDom();
  startLiquidViewport(env.root, env.win);
  const card = env.make('DIV', false); const added = env.make('BUTTON'); card.children = [added];
  env.mutate([{ addedNodes: [card] }]);
  env.mutate([{ addedNodes: [card, { nodeType: 3 }] }]);
  assert.equal(env.observed.filter(el => el === added).length, 1);
});

test('without IntersectionObserver nothing is gated, so every button simply flows', () => {
  const env = liquidDom(); delete env.win.IntersectionObserver;
  assert.equal(startLiquidViewport(env.root, env.win), null);
  assert.ok(!env.classes.has('liquid-gated'));
});

// ---- an info tip stays on screen ------------------------------------------------------------------------------
test('an info tip near the right edge of a narrow window is moved left just enough to stay on screen', () => {
  const { startTipPlacement } = require('../dubbing_service/webui/voice-lab.js');
  const listeners = {}; const doc = { addEventListener: (type, fn) => { listeners[type] = fn; } };
  const tipAt = left => { const vars = {}; const tip = { getBoundingClientRect: () => ({ left }), style: { setProperty: (k, v) => { vars[k] = v; } } }; return { tip, vars, event: { target: { closest: () => tip } } }; };
  startTipPlacement(doc, { innerWidth: 400 });
  const near = tipAt(300); listeners.pointerover(near.event);
  assert.equal(near.vars['--tip-shift'], '230px', 'a 320px tip opened at 294 would end at 614; moved 230px it ends at 384, inside the 16px margin');
  const roomy = tipAt(20); listeners.focusin(roomy.event);
  assert.equal(roomy.vars['--tip-shift'], '0px', 'a tip with room is left where it is');
  const edge = tipAt(30); startTipPlacement(doc, { innerWidth: 200 }); listeners.pointerover(edge.event);
  assert.equal(edge.vars['--tip-shift'], '8px', 'and it is never pushed past the left edge');
});


test('character audio exposes six completed role samples with direct Play and no Queue or generation', async () => {
  const env = fixture();
  const names = ['Narrator', 'Harry', 'Hermione', 'Ron', 'Dumbledore', 'Snape'];
  env.base.roles = names.map(name => ({ id: name.toLowerCase(), name }));
  env.base.voices = names.map((name, i) => ({ ...candidate(String.fromCharCode(97 + i)), name, role_id: name.toLowerCase(), role_name: name, audio_available: true }));
  const state = await mount(env.doc, env.win);
  assert.equal(env.elements['character-audio'].hidden, false);
  assert.match(env.elements['character-audio-status'].textContent, /6\/6 characters have audio ready/);
  const cards = env.elements['character-audio-cards'];
  assert.equal(cards.children.length, 6);
  const play = buttonByText(cards, 'Play Harry');
  assert.ok(play); await click(play);
  assert.equal(state.player.current.role_id, 'harry');
  assert.equal(state.player.media.paused, false);
  assert.deepEqual(env.calls.filter(row => row.request?.method === 'POST' && !row.url.endsWith('/position')), []);
});

test('missing verified bytes cannot become Ready from a URL and checksum alone', async () => {
  const env = fixture();
  env.base.roles = [{ id: 'unrecorded', name: 'Unrecorded character' }];
  env.base.voices = [{ ...candidate('a'), role_id: 'unrecorded', audio_available: false, audio_error: 'missing' }];
  await mount(env.doc, env.win);
  const card = env.elements['character-audio-cards'].children[0];
  assert.equal(card.dataset.audioState, 'not_generated');
  assert.equal(findAll(card, el => el.tagName === 'BUTTON' && el.textContent.startsWith('Play')).length, 0);
  assert.match(env.elements['character-audio-status'].textContent, /0\/1 characters/);
});

test('new queued options are labeled Queued while the saved character audio stays playable', async () => {
  const env = fixture();
  env.base.roles = [{ id: 'role-one', name: 'Mara' }];
  env.base.voices = [{ ...candidate('saved'), role_id: 'role-one', name: 'Saved Mara', audio_available: true }];
  env.casting.roles[0].round_status = 'pending';
  env.casting.roles[0].candidates = [{ candidate_id: 'new', status: 'queued' }];
  const state = await mount(env.doc, env.win);
  const card = env.elements['character-audio-cards'].children[0];
  assert.equal(card.dataset.audioState, 'queued');
  assert.ok(find(card, el => el.textContent.includes('Queued for new audio')));
  await click(buttonByText(card, 'Play Mara'));
  assert.equal(state.player.media.paused, false);
  assert.ok(find(env.elements['casting-roles'], el => el.textContent.includes('Queued — audio not generated yet')));
});

test('running and failed requests display honest character generation state and reason', async () => {
  const env = fixture();
  env.base.roles = [{ id: 'running-role', name: 'Running' }, { id: 'failed-role', name: 'Failed' }];
  env.base.auditions = [{ role_id: 'running-role', status: 'running' }, { role_id: 'failed-role', status: 'interrupted', reason: 'CUDA unavailable' }];
  await mount(env.doc, env.win);
  const [running, failed] = env.elements['character-audio-cards'].children;
  assert.equal(running.dataset.audioState, 'generating');
  assert.ok(find(running, el => el.textContent.includes('Generating new audio')));
  assert.equal(failed.dataset.audioState, 'failed');
  assert.ok(find(failed, el => el.textContent.includes('CUDA unavailable')));
});


test('latest terminal receipt replaces a stale running receipt for the same request', async () => {
  const env = fixture();
  env.base.roles = [{ id: 'role-latest', name: 'Latest' }];
  env.base.auditions = [{ request_id: 'same', role_id: 'role-latest', candidate_id: 'x', status: 'running' }, { request_id: 'same', role_id: 'role-latest', candidate_id: 'x', status: 'interrupted', reason: 'Worker stopped' }];
  await mount(env.doc, env.win);
  const card = env.elements['character-audio-cards'].children[0];
  assert.equal(card.dataset.audioState, 'failed');
  assert.ok(find(card, el => el.textContent.includes('Worker stopped')));
});

test('a blocked round without an audition receipt displays Failed and its actual reason', async () => {
  const env = fixture();
  env.base.roles = [{ id: 'role-one', name: 'Mara' }];
  env.casting.roles[0].round_status = 'blocked';
  env.casting.roles[0].reason = 'Reference integrity failed';
  env.casting.roles[0].candidates = [];
  await mount(env.doc, env.win);
  const card = env.elements['character-audio-cards'].children[0];
  assert.equal(card.dataset.audioState, 'failed');
  assert.ok(find(card, el => el.textContent.includes('Failed — Reference integrity failed')));
});


test('star-free Save sends exact qualitative direction without inventing a rating', async () => {
  const env = fixture(); await mount(env.doc, env.win);
  await click(buttonByText(candidateCard(env, 'a'), 'Add direction'));
  env.elements['casting-direction-input'].value = 'Deeper, more like David Attenborough';
  env.elements['casting-direction-input'].oninput();
  assert.equal(env.elements['casting-save'].disabled, false);
  await env.elements['casting-save'].onclick();
  const sent = JSON.parse(env.calls.find(row => row.url.endsWith('/casting/feedback')).request.body);
  assert.equal(sent.action, 'save');
  assert.equal(sent.raw_direction, 'Deeper, more like David Attenborough');
  assert.equal(Object.hasOwn(sent, 'stars'), false);
  assert.equal(sent.candidate_id, 'a');
});
