const test = require('node:test');
const assert = require('node:assert/strict');
// ASSUMPTION: the existing production-mount fake DOM/media harness is adapted behind mountLab(fixture, { respond }).
// It must mount the real Voice Lab, return { requests:[{path,body}], all(className), click(el), settle(), media },
// and re-serve the (mutable) fixture for casting GETs. Adapt the require path to the repo's existing harness.
const { mountLab } = require('./helpers/voice_lab_harness.cjs');
const hash = id => require('node:crypto').createHash('sha256').update(id).digest('hex');

const ready = (id, label, extra = {}) => ({ candidate_id: id, label, status: 'ready', key: 'k_' + id, audio_sha256: hash(id), audio_url: '/a/' + id + '.wav', duration_seconds: 4, rating: null, favorite: false, sections: [], ...extra });
const fixture = () => ({ casting_enabled: true, roles: [
  { role_id: 'r1', name: 'Ada', status: 'PENDING', round_revision: 3, assignment_revision: 7, current_key: null, selected_candidate_id: null,
    candidates: [ready('c1', 'A', { rating: 2 }), ready('c2', 'B'), { candidate_id: 'c3', label: 'C', status: 'queued', key: '', audio_sha256: '', sections: [] }] },
  { role_id: 'r2', name: 'Bo', status: 'PENDING', round_revision: 1, assignment_revision: 1, candidates: [],
    current_audio: { key: 'cur', audio_sha256: hash('cur'), audio_url: '/a/cur.wav', audio_available: true, duration_seconds: 5, favorite: false, kind: 'original' } },
  { role_id: 'r3', name: 'Cy', status: 'PENDING', round_revision: 1, assignment_revision: 1, candidates: [] },
] });
const hasClass = (element, name) => String(element.className || '').split(/\s+/).includes(name);
const descend = (element, name) => hasClass(element, name) ? element : (element.children || []).map(child => descend(child, name)).find(Boolean);
const candidateControl = (lab, id, name) => { const card = lab.all('casting-candidate').find(node => node.dataset.candidateId === id); assert.ok(card, 'candidate ' + id + ' exists'); const control = descend(card, name); assert.ok(control, id + ' has ' + name); return control; };
const candidateStates = (lab, name) => ['c1', 'c2'].map(id => candidateControl(lab, id, name).getAttribute('aria-pressed'));
const paths = lab => lab.requests.map(r => r.path);
const noGeneration = lab => assert.ok(!paths(lab).some(p => /round|generate|approve|render/.test(p)), 'no generation or approval dispatched');

test('load alone never generates or writes', async () => {
  const lab = await mountLab(fixture());
  assert.ok(!paths(lab).includes('preferences'));
  assert.ok(!paths(lab).includes('casting/feedback'));
  noGeneration(lab);
});

test('hearts are independent, server-backed and multiple; removing keeps assignment', async () => {
  const data = fixture(); data.roles[0].current_key = 'k_c2'; data.roles[0].selected_candidate_id = 'c2'; data.roles[0].status = 'APPROVED';
  const lab = await mountLab(data, { respond: (path, body) => (path === 'preferences' ? { favorite: body.favorite } : undefined) });
  lab.click(candidateControl(lab, 'c1', 'casting-heart')); await lab.settle(); lab.click(candidateControl(lab, 'c2', 'casting-heart')); await lab.settle();
  const saves = lab.requests.filter(r => r.path === 'preferences');
  assert.deepEqual(saves[0].body, { key: 'k_c1', audio_sha256: hash('c1'), favorite: true });
  assert.deepEqual(Object.keys(saves[1].body).sort(), ['audio_sha256', 'favorite', 'key']);
  assert.deepEqual(candidateStates(lab, 'casting-heart'), ['true', 'true']);
  lab.click(candidateControl(lab, 'c2', 'casting-heart')); await lab.settle();
  assert.equal(lab.requests.filter(r => r.path === 'preferences').pop().body.favorite, false);
  assert.equal(data.roles[0].selected_candidate_id, 'c2');
  assert.equal(candidateControl(lab, 'c2', 'casting-check').getAttribute('aria-pressed'), 'true');
  assert.ok(!paths(lab).includes('casting/feedback'));
  noGeneration(lab);
});

test('failed heart save is honest and not claimed', async () => {
  const lab = await mountLab(fixture(), { respond: path => { if (path === 'preferences') throw Object.assign(new Error('offline'), { status: 500 }); } });
  lab.click(candidateControl(lab, 'c1', 'casting-heart')); await lab.settle();
  assert.equal(candidateControl(lab, 'c1', 'casting-heart').getAttribute('aria-pressed'), 'false');
  assert.ok(lab.all('casting-circle-note').some(n => n.textContent.includes('not saved')));
});

test('check sends exact use payload with selection_confirmed and no stars; old rating untouched', async () => {
  const data = fixture();
  const lab = await mountLab(data, { respond: (path, body) => {
    if (path === 'casting/feedback') { const r = data.roles[0]; r.selected_candidate_id = body.candidate_id; r.current_key = body.key; r.status = 'APPROVED'; return { ok: true, assignment_revision: 8 }; }
  } });
  lab.click(candidateControl(lab, 'c2', 'casting-check')); await lab.settle();
  const post = lab.requests.find(r => r.path === 'casting/feedback').body;
  assert.equal(post.action, 'use'); assert.strictEqual(post.selection_confirmed, true);
  assert.ok(!('stars' in post));
  assert.equal(post.role_id, 'r1'); assert.equal(post.candidate_id, 'c2'); assert.equal(post.key, 'k_c2'); assert.equal(post.audio_sha256, hash('c2'));
  assert.equal(post.round_revision, 3); assert.equal(post.assignment_revision, 7);
  assert.deepEqual(post.baseline_candidate_ids, ['c1', 'c2']); assert.ok(post.event_id);
  assert.equal(data.roles[0].candidates[0].rating, 2);
  assert.deepEqual(candidateStates(lab, 'casting-check'), ['false', 'true']);
  assert.ok(!lab.requests.some(r => r.path === 'preferences'));
  noGeneration(lab);
});

test('conflict or failed save never fakes a check, and retry reuses the exact request', async () => {
  let mode = 'conflict'; const data = fixture();
  const lab = await mountLab(data, { respond: path => {
    if (path !== 'casting/feedback') return undefined;
    if (mode === 'conflict') { mode = 'fail'; throw Object.assign(new Error('conflict'), { status: 409, body: { conflict: true, reason: 'newer decision' } }); }
    if (mode === 'fail') { mode = 'ok'; throw Object.assign(new Error('network'), { status: 500 }); }
    return { ok: true };
  } });
  lab.click(candidateControl(lab, 'c2', 'casting-check')); await lab.settle();
  assert.equal(candidateControl(lab, 'c2', 'casting-check').getAttribute('aria-pressed'), 'false');
  lab.click(candidateControl(lab, 'c2', 'casting-check')); await lab.settle();
  assert.equal(candidateControl(lab, 'c2', 'casting-check').getAttribute('aria-pressed'), 'false');
  lab.click(candidateControl(lab, 'c2', 'casting-check')); await lab.settle();
  const posts = lab.requests.filter(r => r.path === 'casting/feedback');
  assert.equal(posts[2].body.event_id, posts[1].body.event_id);
  assert.strictEqual(posts[2].body.selection_confirmed, true);
  assert.equal(data.roles[0].selected_candidate_id, null);
});

test('queued and missing audio disable play, heart and check with accessible reasons', async () => {
  const lab = await mountLab(fixture());
  const queued = { play: candidateControl(lab, 'c3', 'casting-play'), heart: candidateControl(lab, 'c3', 'casting-heart'), check: candidateControl(lab, 'c3', 'casting-check') };
  for (const control of Object.values(queued)) { assert.equal(control.disabled, true); assert.ok(/unavailable/i.test(control.getAttribute('aria-label'))); assert.ok(control.getAttribute('aria-label').includes('Queued')); }
  const missing = lab.all('casting-heart')[4];
  assert.equal(missing.disabled, true); assert.ok(/unavailable/i.test(missing.getAttribute('aria-label')));
  assert.equal(lab.all('casting-check').length, 3);
});

test('current_audio plays without a fabricated candidate, check or clone claim; heart uses exact key/hash', async () => {
  const data = fixture();
  const lab = await mountLab(data, { respond: (path, body) => (path === 'preferences' ? { favorite: body.favorite } : undefined) });
  const play = lab.all('casting-circle-play')[3];
  lab.click(play); await lab.settle();
  assert.ok(String(lab.media.src).includes('/a/cur.wav'));
  assert.equal(lab.currentContext().candidate_id, null);
  assert.equal(lab.currentContext().key, 'cur');
  assert.equal(play.getAttribute('aria-label').startsWith('Pause'), true);
  const text = lab.all('casting-circle-name').map(n => n.textContent).join('|');
  assert.ok(text.includes('Current original sample')); assert.ok(!/clone/i.test(text));
  lab.click(lab.all('casting-heart')[3]); await lab.settle();
  assert.deepEqual(lab.requests.find(r => r.path === 'preferences').body, { key: 'cur', audio_sha256: hash('cur'), favorite: true });
  assert.equal(lab.all('casting-check').length, 3);
  noGeneration(lab);
});

test('multiple confirmed favorites survive an authoritative reload and preserve old bookmarks', async () => {
  const data = fixture(); const storage = new Map([['voice-lab-shortlist-v1:portrait-fixture:r1', '["c2"]']]);
  const lab = await mountLab(data, { storage });
  await lab.click(candidateControl(lab, 'c1', 'casting-heart')); await lab.click(candidateControl(lab, 'c2', 'casting-heart')); await lab.settle();
  const restored = await mountLab(data, { storage });
  assert.deepEqual(candidateStates(restored, 'casting-heart'), ['true', 'true']);
  assert.equal(storage.get('voice-lab-shortlist-v1:portrait-fixture:r1'), '["c2"]');
});
test('a low historical rating is preserved when the person explicitly selects that voice', async () => {
  const data = fixture(); const lab = await mountLab(data);
  await lab.click(candidateControl(lab, 'c1', 'casting-check')); await lab.settle();
  assert.equal(data.roles[0].candidates[0].rating, 2);
  assert.equal(data.roles[0].selected_candidate_id, 'c1');
  assert.deepEqual(candidateStates(lab, 'casting-check'), ['true', 'false']);
  await lab.click(candidateControl(lab, 'c2', 'casting-check')); await lab.settle();
  assert.equal(data.roles[0].selected_candidate_id, 'c2');
  assert.deepEqual(candidateStates(lab, 'casting-check'), ['false', 'true']);
  assert.ok(lab.requests.filter(request => request.path === 'casting/feedback').every(request => request.body.selection_confirmed === true && !('stars' in request.body)));
  assert.ok(!lab.requests.some(request => request.path === 'preferences'));
});
test('current audio needs a positive verified-bytes flag and unknown provenance never claims an actor clone', async () => {
  const data = fixture(); delete data.roles[1].current_audio.audio_available; delete data.roles[1].current_audio.kind;
  const lab = await mountLab(data);
  assert.equal(lab.all('casting-circle-play')[3].disabled, true);
  assert.equal(lab.all('casting-heart')[3].disabled, true);
  assert.equal(lab.elements['character-audio'].hidden, true, 'new current_audio projection replaces the duplicate legacy card view');
  assert.ok(!lab.all('casting-circle-name').some(name => /clone/i.test(name.textContent)));
});
test('a poll begun before favorite acknowledgement cannot roll it back; later authoritative changes can', async () => {
  const data = fixture(); data.roles[0].round_status = 'pending';
  let arm = false, release; const stale = structuredClone(data);
  const lab = await mountLab(data, { respond: action => {
    if (arm && action === 'casting') { arm = false; return new Promise(resolve => { release = () => resolve(stale); }); }
  } });
  assert.ok(lab.intervals.length, 'queued work supplies a read-only poll');
  arm = true; lab.intervals.at(-1)(); await lab.settle(); assert.ok(release);
  await lab.click(candidateControl(lab, 'c1', 'casting-heart')); await lab.settle();
  release(); await lab.settle();
  assert.equal(candidateControl(lab, 'c1', 'casting-heart').getAttribute('aria-pressed'), 'true');
  data.roles[0].candidates[0].favorite = false;
  lab.intervals.at(-1)(); await lab.settle();
  assert.equal(candidateControl(lab, 'c1', 'casting-heart').getAttribute('aria-pressed'), 'false', 'fresh state from another device is not masked by a cached override');
});

test('rapid repeated heart taps submit only one exact favorite while its acknowledgement is pending', async () => {
  const data = fixture(); let release;
  const lab = await mountLab(data, { respond: action => action === 'preferences' ? new Promise(resolve => { release = () => resolve({ saved: true, favorite: true }); }) : undefined });
  const heart = candidateControl(lab, 'c1', 'casting-heart');
  const pending = lab.click(heart); await lab.settle();
  assert.equal(heart.disabled, true);
  assert.equal(heart.getAttribute('aria-pressed'), 'false', 'a pending save is not claimed');
  await lab.click(heart);
  assert.equal(lab.requests.filter(request => request.path === 'preferences').length, 1);
  release(); await pending; await lab.settle();
  assert.equal(heart.disabled, false);
  assert.equal(heart.getAttribute('aria-pressed'), 'true');
  assert.equal(data.roles[0].candidates[0].rating, 2);
  assert.equal(data.roles[0].selected_candidate_id, null);
});