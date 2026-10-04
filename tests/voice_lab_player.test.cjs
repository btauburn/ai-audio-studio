'use strict';
const { test } = require('node:test');
const assert = require('node:assert/strict');
const { Player, ReviewClient, shortcut, filtered, roleAt, matchedComparisons, comparisonGain, safeUrl, mount } = require('../dubbing_service/webui/voice-lab.js');

class Media {
  constructor() { this.events = {}; this.paused = true; this.readyState = 1; this.duration = 300; this.currentTime = 0; this.volume = 1; this.history = []; }
  addEventListener(name, fn) { (this.events[name] ||= []).push(fn); }
  emit(name) { for (const fn of this.events[name] || []) fn(); }
  pause() { this.paused = true; this.history.push(['pause', this.src]); }
  play() { this.paused = false; this.history.push(['play', this.src]); return Promise.resolve(); }
  load() { this.currentTime = 0; this.history.push(['load', this.src]); this.emit('loadedmetadata'); }
  removeAttribute(name) { delete this[name]; }
}
const recording = (key = 'voice-a', hash = 'hash-a') => ({ key, audio_url: '/audio/' + key, audio_sha256: hash });

test('one media element is paused before each source switch; playback never overlaps', async () => {
  const media = new Media(); const player = new Player(media);
  await player.load(recording(), { autoplay: true, seconds: 12 });
  await player.load(recording('voice-b', 'hash-b'), { autoplay: true });
  assert.deepEqual(media.history.map(row => row[0]), ['pause', 'load', 'play', 'pause', 'load', 'play']);
  assert.equal(player.media, media); assert.equal(player.current.audio_sha256, 'hash-b');
  await player.toggle(); assert.equal(media.paused, true);
  await player.load(null); assert.equal(media.src, undefined); assert.equal(media.paused, true);
});
test('opening a voice never autoplays; restoration and seek obey the audio duration', async () => {
  const media = new Media(); const player = new Player(media);
  await player.load(recording(), { seconds: 22 });
  assert.equal(media.currentTime, 22); assert.equal(media.paused, true);
  player.seek(999); assert.equal(media.currentTime, 300);
  player.seek(-100); assert.equal(media.currentTime, 0);
  await player.load({ key: 'design-only', audio_url: '/design.wav' }, { autoplay: true });
  assert.equal(media.src, undefined); assert.equal(media.paused, true);
});
test('stale aborted play does not fail or pause the newer selection', async () => {
  const media = new Media(); let reject;
  media.play = () => new Promise((resolve, no) => { reject = no; });
  const player = new Player(media); const old = player.load(recording(), { autoplay: true });
  await player.load(recording('new', 'new-hash')); reject(new Error('aborted old request')); await old;
  assert.equal(player.current.key, 'new');
});
test('review transport saves exact audio version and never permits synthesis routes', async () => {
  const calls = []; const client = new ReviewClient('/api/test', async (url, request) => { calls.push({ url, request }); return { ok: true, json: async () => ({ saved: true }) }; });
  await client.request('');
  for (const action of ['preferences', 'position', 'issues']) await client.request(action, { key: 'voice:v2', audio_sha256: 'exact-hash', seconds: 12 });
  assert.equal(calls.length, 4); assert.equal(calls[0].request.method, undefined);
  for (const { request } of calls.slice(1)) { assert.equal(request.headers['X-Review-Request'], '1'); assert.equal(JSON.parse(request.body).audio_sha256, 'exact-hash'); }
  await assert.rejects(client.request('synthesize', {}), /Unsupported/); assert.equal(calls.length, 4);
});
test('failed preference saves remain explicit errors', async () => {
  const client = new ReviewClient('/api/test', async () => ({ ok: false, status: 409, json: async () => ({ detail: 'The audio version changed' }) }));
  await assert.rejects(client.request('preferences', {}), /audio version changed/);
});
test('shortcuts do not intercept typing, controls, modifiers, or held keys', () => {
  const event = (key, target = { tagName: 'BODY' }, extra = {}) => ({ key, target, ...extra });
  for (const tagName of ['INPUT', 'TEXTAREA', 'SELECT', 'BUTTON', 'A', 'SUMMARY']) assert.equal(shortcut(event('1', { tagName })), null);
  assert.equal(shortcut(event('f', { isContentEditable: true })), null);
  assert.equal(shortcut(event('f', undefined, { ctrlKey: true })), null);
  assert.equal(shortcut(event('f', undefined, { repeat: true })), null);
  assert.equal(shortcut(event(' ')), 'play'); assert.equal(shortcut(event('F')), 'favorite'); assert.equal(shortcut(event('5')), 5);
  assert.equal(shortcut(event('ArrowLeft')), 'previous'); assert.equal(shortcut(event('ArrowRight')), 'next');
});
test('New means unrated, not disliked; favorites and ratings are independent', () => {
  const rows = [{ key: 'new', favorite: false, rating: null }, { key: 'heart', favorite: true, rating: null }, { key: 'rated', favorite: false, rating: 1 }];
  assert.deepEqual(filtered(rows, 'new').map(v => v.key), ['new']);
  assert.deepEqual(filtered(rows, 'favorites').map(v => v.key), ['heart']);
  assert.deepEqual(filtered(rows, 'rated').map(v => v.key), ['rated']);
});
test('source-to-audio map switches at precise contiguous boundaries', () => {
  const map = [{ utterance_id: 'a', role_id: 'narrator', start_seconds: 0, end_seconds: 5 }, { utterance_id: 'b', role_id: 'character', start_seconds: 5, end_seconds: 10 }];
  assert.equal(roleAt(map, 4.99).role_id, 'narrator'); assert.equal(roleAt(map, 5).role_id, 'character'); assert.equal(roleAt(map, 10), null);
});
test('A/B only accepts identical ordered utterances and measured loudness', () => {
  const excerpt = { map: [{ utterance_id: 'a' }, { utterance_id: 'b' }], comparisons: [
    { ...recording('a'), utterance_ids: ['a', 'b'], loudness_lufs: -16 },
    { ...recording('b'), utterance_ids: ['a', 'b'], loudness_lufs: -20 },
    { ...recording('reordered'), utterance_ids: ['b', 'a'], loudness_lufs: -20 },
    { ...recording('unmeasured'), utterance_ids: ['a', 'b'] },
    { ...recording('partial'), utterance_ids: ['a'], loudness_lufs: -20 }
  ] };
  const comparisons = matchedComparisons(excerpt); assert.equal(comparisons.length, 2);
  assert.equal(comparisonGain(comparisons[1], comparisons), 1);
  assert.ok(Math.abs(comparisonGain(comparisons[0], comparisons) - 0.630957344) < 1e-8);
  assert.equal(excerpt.comparisons[0].loudness_lufs, -16);
});
test('audio stays same-origin; reference links accept only http(s)', () => {
  const location = { href: 'https://private.example/voice-lab', origin: 'https://private.example' };
  assert.equal(safeUrl('/audio/a', location), 'https://private.example/audio/a');
  assert.equal(safeUrl('https://other.example/a', location), null);
  assert.equal(safeUrl('javascript:alert(1)', location, true), null);
  assert.equal(safeUrl('https://official.example/ref', location, true), 'https://official.example/ref');
});

// Minimal DOM adapter executes the production mount and handlers, not a duplicate
// model of their behavior. Browser/device audio decoding is verified separately.
class Element {
  constructor(tag = 'div') { this.tagName = tag.toUpperCase(); this.children = []; this.dataset = {}; this.events = {}; this.attrs = {}; this.value = ''; this.hidden = false; this.disabled = false; this.textContent = ''; this.classList = { toggle() {} }; }
  append(...children) { this.children.push(...children); }
  replaceChildren(...children) { this.children = children; }
  get options() { return this.children; }
  setAttribute(name, value) { this.attrs[name] = value; }
  removeAttribute(name) { delete this.attrs[name]; }
  addEventListener(name, fn) { (this.events[name] ||= []).push(fn); }
  emit(name, event) { for (const fn of this.events[name] || []) fn(event); }
}
function mountedEnvironment(data, failPreference = false) {
  const fs = require('node:fs'); const path = require('node:path');
  const html = fs.readFileSync(path.join(__dirname, '../dubbing_service/webui/voice-lab.html'), 'utf8');
  const elements = Object.fromEntries([...html.matchAll(/<([a-z]+)[^>]* id="([^"]+)"/g)].map(m => [m[2], new Element(m[1])]));
  elements.media = new Media();
  const doc = new Element(); doc.getElementById = id => elements[id]; doc.createElement = tag => new Element(tag);
  const calls = []; const win = { location: { href: 'http://localhost/web/voice-lab.html?project=test', origin: 'http://localhost', pathname: '/web/voice-lab.html', search: '?project=test' }, crypto: { randomUUID: () => 'request-id' }, fetch: async (url, request) => {
    calls.push({ url, request }); const fail = failPreference && url.endsWith('/preferences');
    return { ok: !fail, status: fail ? 409 : 200, json: async () => request?.method === 'POST' ? (fail ? { detail: 'This version changed' } : { saved: true }) : structuredClone(data) };
  } };
  return { doc, win, elements, calls };
}
const projectData = () => ({ project_name: 'Isolated test', voices: [
  { ...recording(), voice_id: 'a', version: '2', name: 'Cedar', role_id: 'r', favorite: false, rating: null },
  { ...recording('voice-b', 'hash-b'), voice_id: 'b', version: '1', name: 'Willow', role_id: 'r', favorite: false, rating: null }
], roles: [{ id: 'r', name: 'Narrator', current_identity_id: 'identity-r' }], automation: { paused: true }, issues: [], excerpt: { status: 'blocked_source', reason: 'Chapter 1 is unavailable.' } });

test('mounted page load, favorite, and navigation use only review endpoints and never fabricate blocked audio', async () => {
  const env = mountedEnvironment(projectData()); const state = await mount(env.doc, env.win);
  // The project itself, plus the read-only "where is my book" status.
  assert.equal(env.calls.length, 2); assert.ok(env.calls.every(call => call.request.method === undefined));
  assert.ok(env.calls.some(call => call.url.endsWith('/voice-lab/casting/book')));
  assert.equal(env.elements['play-excerpt'].disabled, true);
  assert.match(env.elements['excerpt-status'].textContent, /Chapter 1 is unavailable/);
  assert.equal(env.elements['voice-name'].textContent, 'Cedar');
  await env.elements.favorite.onclick();
  const favorite = env.calls.find(row => row.url.endsWith('/preferences'));
  assert.deepEqual(JSON.parse(favorite.request.body), { key: 'voice-a', audio_sha256: 'hash-a', favorite: true });
  assert.equal(env.elements.favorite.attrs['aria-pressed'], 'true');
  assert.equal(env.elements.next.disabled, false, 'Next reaches the sole remaining New voice after favoriting');
  await env.elements.next.onclick();
  assert.equal(state.player.current.key, 'voice-b');
  assert.ok(env.calls.every(row => !/design|synth|render|more-like/.test(row.url)));
});
test('mounted page save rejection leaves preference reversible and exposes failure', async () => {
  const env = mountedEnvironment(projectData(), true); await mount(env.doc, env.win);
  await env.elements.favorite.onclick();
  assert.equal(env.elements.favorite.attrs['aria-pressed'], 'false');
  assert.equal(env.elements.error.hidden, false); assert.match(env.elements.error.textContent, /version changed/);
  assert.equal(env.elements['save-status'].textContent, 'Not saved. Please retry.');
});

test('mounted library browse uses GET and import switches to the exact new recording paused', async () => {
  const data = projectData(); const env = mountedEnvironment(data);
  const imported = { ...recording('library-incoming', 'imported-hash'), voice_id: 'saved', version: '3', name: 'Imported Cedar', role_id: null, favorite: false, rating: null };
  let didImport = false;
  const requestHandler = env.win.fetch;
  env.win.fetch = async (url, request) => {
    if (url.endsWith('/library')) {
      env.calls.push({ url, request });
      return { ok: true, json: async () => ({ voices: [{ library_voice_id: 'vl_saved', name: 'Saved', versions: [{ version: '3', name: 'Imported Cedar' }] }] }) };
    }
    if (url.endsWith('/library/import')) {
      env.calls.push({ url, request }); didImport = true;
      return { ok: true, json: async () => ({ key: imported.key, imported: true }) };
    }
    if (didImport && !request.method) {
      env.calls.push({ url, request });
      return { ok: true, json: async () => ({ ...data, voices: [...data.voices, imported] }) };
    }
    return requestHandler(url, request);
  };
  const state = await mount(env.doc, env.win);
  await state.player.toggle(); assert.equal(state.player.media.paused, false);
  await env.elements['library-load'].onclick();
  const browse = env.calls.find(call => call.url.endsWith('/library'));
  assert.equal(browse.request.method, undefined);
  assert.equal(env.elements['library-versions'].children.length, 1);
  env.elements['library-versions'].value = env.elements['library-versions'].children[0].value;
  await env.elements['library-import'].onclick();
  const request = env.calls.find(call => call.url.endsWith('/library/import')).request;
  assert.equal(request.method, 'POST');
  assert.deepEqual(JSON.parse(request.body), { library_voice_id: 'vl_saved', version: '3' });
  assert.equal(state.player.current.key, imported.key);
  assert.equal(state.player.current.audio_sha256, 'imported-hash');
  assert.equal(state.player.media.paused, true);
  assert.equal(env.elements['voice-name'].textContent, 'Imported Cedar');
  assert.equal(env.elements.favorite.attrs['aria-pressed'], 'false');
  assert.ok(!env.calls.some(call => /design|synth|render/.test(call.url)));
});

test('empty project can browse saved voices without a selected voice or synthesis', async () => {
  const data = projectData(); data.voices = []; const env = mountedEnvironment(data);
  const requestHandler = env.win.fetch;
  env.win.fetch = async (url, request) => {
    if (url.endsWith('/library')) { env.calls.push({ url, request }); return { ok: true, json: async () => ({ voices: [] }) }; }
    return requestHandler(url, request);
  };
  const state = await mount(env.doc, env.win);
  assert.equal(env.elements['voice-detail'].hidden, true);
  await env.elements['library-load'].onclick();
  assert.match(env.elements['library-status'].textContent, /No reusable production versions/);
  assert.equal(state.player.current, null);
  assert.deepEqual(env.calls.map(call => call.request.method), [undefined, undefined, undefined]);
});

test('slow position save cannot restore an older selection after rapid navigation', async () => {
  const env = mountedEnvironment(projectData());
  const state = await mount(env.doc, env.win);
  let finishSave;
  state.client.fetcher = async (url) => {
    if (url.endsWith('/position')) await new Promise(resolve => { finishSave = resolve; });
    return { ok: true, json: async () => ({ saved: true }) };
  };
  await state.player.toggle();
  const first = env.elements.next.onclick();
  assert.equal(state.player.media.paused, false, 'navigation resumes playback without waiting for the bookmark POST');
  assert.equal(state.player.current.key, 'voice-b');
  const second = env.elements.previous.onclick();
  await second;
  assert.equal(state.player.current.key, 'voice-a');
  finishSave(); await first;
  assert.equal(state.player.current.key, 'voice-a');
  assert.equal(env.elements['voice-name'].textContent, 'Cedar');
});

test('chapter map selects the exact heard voice version and never treats pauses as cast', async () => {
  const { voiceForRow } = require('../dubbing_service/webui/voice-lab.js');
  const data = projectData();
  const line = { role_id: 'r', voice_id: 'b', voice_version: '1', start_seconds: 0, end_seconds: 5 };
  assert.equal(voiceForRow(data.voices, line).key, 'voice-b');
  assert.equal(voiceForRow(data.voices, { ...line, voice_version: 'missing' }), null);
  assert.equal(voiceForRow(data.voices, { role_id: null, kind: 'scene_break' }), null);
  data.excerpt = { ...recording('chapter', 'chapter-hash'), chapter_title: 'Chapter 258', map: [line,
    { role_id: null, kind: 'scene_break', start_seconds: 5, end_seconds: 6.25 }] };
  const env = mountedEnvironment(data); await mount(env.doc, env.win);
  assert.equal(env.elements.cast.children.length, 1);
  assert.equal(env.elements.cast.children[0].dataset.roleId, 'r');
});

function comparisonProject() {
  const data = projectData();
  const oldMap = [{ utterance_id: 'u', role_id: 'r', role_name: 'Narrator', voice_id: 'a', voice_version: '2', start_seconds: 3, end_seconds: 90 }];
  const newMap = [{ ...oldMap[0], voice_id: 'b', voice_version: '1', start_seconds: 5 }];
  const old = { ...recording('comparison-old', 'old-chapter-hash'), label: 'Older edition', utterance_ids: ['u'], map: oldMap, loudness_lufs: -16 };
  const fresh = { ...recording('comparison-new', 'new-chapter-hash'), label: 'New edition', utterance_ids: ['u'], map: newMap, loudness_lufs: -20 };
  data.excerpt = { ...recording('excerpt', 'new-chapter-hash'), schema: 'voice-lab-chapter-review-v1', chapter_title: 'Fixture chapter', map: newMap, comparisons: [old, fresh] };
  return { data, old, fresh };
}

test('active A/B cast feedback and Hear lines stay bound to the edition being heard', async () => {
  const { data, old, fresh } = comparisonProject(); const env = mountedEnvironment(data);
  const state = await mount(env.doc, env.win);
  await env.elements['play-excerpt'].onclick();
  assert.equal(state.player.current.audio_sha256, fresh.audio_sha256);
  assert.equal(env.elements.cast.children[0].children[1].textContent, 'Willow · v1');
  await env.elements.comparisons.children[0].events.click[0]();
  assert.equal(state.player.current.key, old.key);
  assert.equal(env.elements.cast.children[0].children[1].textContent, 'Cedar · v2');
  let card = env.elements.cast.children[0];
  await card.children.find(el => el.tagName === 'DIV').children[1].events.click[0]();
  let saved = env.calls.filter(call => call.url.endsWith('/preferences')).at(-1);
  assert.deepEqual(JSON.parse(saved.request.body), { key: 'voice-a', audio_sha256: 'hash-a', favorite: true });
  card = env.elements.cast.children[0];
  const rating = card.children.find(el => el.tagName === 'LABEL').children[0];
  rating.value = '4'; await rating.onchange();
  saved = env.calls.filter(call => call.url.endsWith('/preferences')).at(-1);
  assert.deepEqual(JSON.parse(saved.request.body), { key: 'voice-a', audio_sha256: 'hash-a', rating: 4 });
  card = env.elements.cast.children[0];
  const volume = state.player.media.volume;
  await card.children.find(el => el.tagName === 'DIV').children[0].events.click[0]();
  assert.equal(state.player.current.key, old.key);
  assert.equal(state.player.current.audio_sha256, old.audio_sha256);
  assert.equal(state.player.media.currentTime, 3);
  assert.equal(state.player.media.volume, volume);
  assert.equal(state.player.media.paused, false);
  await env.elements['play-excerpt'].onclick();
  assert.equal(state.player.current.audio_sha256, fresh.audio_sha256);
  assert.equal(env.elements.cast.children[0].children[1].textContent, 'Willow · v1');
  assert.ok(!env.calls.some(call => /design|synth|render/.test(call.url)));
});

test('refresh restores only the exact saved comparison with its map, time and matched gain paused', async () => {
  const { data, old } = comparisonProject();
  data.review_position = { key: old.key, audio_sha256: old.audio_sha256, seconds: 37, filter: 'all' };
  const env = mountedEnvironment(data); const state = await mount(env.doc, env.win);
  assert.equal(state.player.current.key, old.key);
  assert.equal(state.player.current.audio_sha256, old.audio_sha256);
  assert.equal(state.player.current.map[0].voice_id, 'a');
  assert.equal(state.player.media.currentTime, 37);
  assert.equal(state.player.media.paused, true);
  assert.ok(Math.abs(state.player.media.volume - Math.pow(10, -4 / 20)) < 1e-8);
  assert.equal(env.elements.cast.children[0].children[1].textContent, 'Cedar · v2');
  assert.equal(env.calls.length, 2, 'restoring a comparison only reads saved state (the project and the book status)');
  assert.ok(env.calls.every(call => call.request.method === undefined));
  data.review_position.audio_sha256 = 'superseded-unavailable-audio';
  const stale = mountedEnvironment(data); const fallback = await mount(stale.doc, stale.win);
  assert.notEqual(fallback.player.current.key, old.key);
  assert.equal(fallback.player.media.currentTime, 0);
  assert.equal(fallback.player.media.paused, true);
});

test('first chapter tap calls play within the gesture while previous-position POST is still pending', async () => {
  const { data } = comparisonProject(); const env = mountedEnvironment(data);
  const state = await mount(env.doc, env.win);
  state.player.media.currentTime = 17;
  let finishSave; const pendingSave = new Promise(resolve => { finishSave = resolve; });
  const positions = [];
  state.client.fetcher = async (url, request) => {
    if (url.endsWith('/position')) { positions.push(JSON.parse(request.body)); await pendingSave; }
    return { ok: true, json: async () => ({ saved: true }) };
  };
  let inGesture = true; const played = [];
  const play = state.player.media.play.bind(state.player.media);
  state.player.media.play = () => {
    played.push({ inGesture, source: state.player.media.src });
    if (!inGesture) return Promise.reject(new Error('Playback was called after the gesture'));
    return play();
  };
  const tapped = env.elements['play-excerpt'].onclick();
  inGesture = false;
  try {
    assert.equal(played.length, 1, 'the tap must call media.play before any network response');
    assert.equal(played[0].inGesture, true);
    assert.match(played[0].source, /\/audio\/excerpt$/);
    assert.equal(state.player.current.key, 'excerpt');
    assert.equal(state.player.media.paused, false);
    assert.deepEqual(positions, [{ key: 'voice-a', audio_sha256: 'hash-a', seconds: 17, filter: 'new' }]);
    await tapped;
    assert.equal(env.elements['now-playing'].textContent, 'Fixture chapter');
    assert.equal(env.elements.error.hidden, true);
  } finally { finishSave(); await tapped; }
});

test('failed background position save cannot block playback or replace a newer choice', async () => {
  const { data } = comparisonProject(); const env = mountedEnvironment(data);
  const state = await mount(env.doc, env.win);
  state.player.media.currentTime = 11;
  let failSave; const pendingSave = new Promise((resolve, reject) => { failSave = reject; });
  state.client.fetcher = async (url) => {
    if (url.endsWith('/position')) await pendingSave;
    return { ok: true, json: async () => ({ saved: true }) };
  };
  const first = env.elements['play-excerpt'].onclick();
  try {
    assert.equal(state.player.current.key, 'excerpt');
    assert.equal(state.player.media.paused, false);
    await env.elements.comparisons.children[0].events.click[0]();
    assert.equal(state.player.current.key, 'comparison-old');
    failSave(new Error('Offline bookmark service'));
    await first; await new Promise(resolve => setImmediate(resolve));
    assert.equal(state.player.current.key, 'comparison-old');
    assert.equal(state.player.media.paused, false);
    assert.equal(env.elements['now-playing'].textContent, 'Older edition');
    assert.match(env.elements.error.textContent, /Listening position was not saved: Offline bookmark service/);
  } finally { failSave(new Error('Offline bookmark service')); await first; }
});

function creatorEnvironment() {
  const env = mountedEnvironment({});
  env.win.location = { href: 'http://localhost/web/voice-lab.html', origin: 'http://localhost', pathname: '/web/voice-lab.html', search: '' };
  env.win.FormData = class { constructor() { this.fields = new Map(); } append(key, value) { this.fields.set(key, value); } get(key) { return this.fields.get(key); } };
  const version = { library_voice_id: 'library-one', version: 'version-one', name: 'Saved Narrator', audio_sha256: 'sample-hash', audio_url: '/api/v1/voice-library/versions/library-one/version-one/audio', duration_seconds: 10 };
  const output = { export_id: 'export-one', label: 'Chapter 1 MP3', chapter_ids: ['chapter-one'], duration_seconds: 121, audio_sha256: 'output-hash', audio_url: '/api/v1/book-creator/book-one/audio/export-one', download_url: '/api/v1/book-creator/book-one/audio/export-one?download=true', technical_qa: 'passed', human_review: 'pending' };
  const book = { project_id: 'book-one', name: 'Local fixture book', mode: 'single_narrator', source: { filename: 'fixture.epub', format: 'epub', sha256: 'source-hash' }, chapters: [{ id: 'chapter-one', title: 'ONE: Arrival', utterance_count: 3 }, { id: 'chapter-two', title: 'TWO: Home', utterance_count: 2 }], recordings: [], requests: [] };
  const server = { version, output, book, recordings: [], request: null, status: 'running', failImport: false };
  env.win.fetch = async (url, request = {}) => {
    env.calls.push({ url, request }); let result;
    if (url === '/api/v1/projects/summaries') result = [{ id: 'existing-book', name: 'Existing private book' }];
    else if (url === '/api/v1/voice-library/recordings') result = { recordings: server.recordings };
    else if (url === '/api/v1/book-creator') result = { books: [book], voices: [{ library_voice_id: version.library_voice_id, name: version.name, versions: [version] }] };
    else if (url.endsWith('/import')) { if (server.failImport) return { ok: false, status: 409, json: async () => ({ detail: 'Source validation failed' }) }; result = book; }
    else if (url.endsWith('/render')) { const body = JSON.parse(request.body); server.request = { ...body, project_id: book.project_id, job_id: 'job-one', status: server.status, completed: 1, total: 3, message: 'Fixture metadata only', outputs: [] }; result = server.request; }
    else if (url.includes('/requests/')) result = { ...server.request, status: server.status, completed: server.status === 'succeeded' ? 3 : 1, outputs: server.status === 'succeeded' ? [output] : [] };
    else if (url.endsWith('/book-one')) result = book;
    else throw new Error('Unexpected test-only request ' + url);
    return { ok: true, status: 200, json: async () => structuredClone(result) };
  };
  return { ...env, server };
}

test('desktop library landing loads existing books and saved voices with GET only', async () => {
  const env = creatorEnvironment(); const state = await mount(env.doc, env.win);
  assert.equal(env.elements['page-title'].textContent, 'Voice Library');
  assert.equal(env.elements['library-landing'].hidden, false);
  assert.equal(env.calls.filter(call => /\/projects\/.+\/voice-lab/.test(call.url)).length, 0);
  assert.equal(env.elements.projects.children[0].value, 'existing-book');
  assert.equal(env.elements['global-voices'].children.length, 1);
  assert.equal(env.elements['player-feedback'].hidden, true);
  assert.equal(state.player.current, null);
  assert.ok(env.calls.every(call => !call.request.method));
  env.elements.projects.value = 'existing-book'; env.elements['open-project'].onclick();
  assert.equal(env.win.location.href, '/web/voice-lab.html?project=existing-book');
});

test('file picker imports source, explicit narrator reading renders, then MP3 plays only on tap', async () => {
  const env = creatorEnvironment(); const state = await mount(env.doc, env.win);
  // AutoCast is the default reading setup; this test exercises the distinct
  // single-narrator reading path, so it selects that mode explicitly.
  env.elements['book-mode'].value = 'single_narrator'; env.elements['book-mode'].onchange();
  env.elements['book-file'].files = [{ name: 'fixture.epub', size: 42 }];
  await env.elements['book-file'].onchange();
  assert.ok(!env.calls.some(call => call.request.method === 'POST'));
  await env.elements['import-book'].onclick();
  const imported = env.calls.find(call => call.url.endsWith('/import'));
  assert.equal(imported.request.body.get('file').name, 'fixture.epub');
  assert.equal(imported.request.body.get('request_id'), 'request-id');
  assert.equal(imported.request.headers['Content-Type'], undefined, 'browser owns multipart boundary');
  assert.equal(env.elements['book-chapters'].children.length, 2);
  assert.equal(env.elements['create-mp3'].disabled, true, 'a narrator must be chosen explicitly');
  env.elements['book-narrator'].value = JSON.stringify({ library_voice_id: 'library-one', voice_version: 'version-one' });
  env.elements['book-narrator'].onchange();
  assert.equal(env.elements['create-mp3'].disabled, false);
  await env.elements['create-mp3'].onclick();
  const rendering = env.calls.find(call => call.url.endsWith('/render'));
  assert.deepEqual(JSON.parse(rendering.request.body), { chapter_ids: ['chapter-one'], library_voice_id: 'library-one', voice_version: 'version-one', request_id: 'request-id' });
  assert.equal(env.elements['book-narrator'].disabled, true);
  assert.equal(state.player.current, null, 'async generation does not autoplay');
  env.server.status = 'succeeded'; await env.elements['refresh-book-progress'].onclick();
  assert.equal(env.elements['book-progress-bar'].value, 3);
  assert.equal(env.elements['book-outputs'].children.length, 1);
  assert.equal(state.player.current, null);
  const controls = env.elements['book-outputs'].children[0].children.find(el => el.tagName === 'DIV');
  assert.equal(controls.children[1].href, env.server.output.download_url.replace(/^\//, 'http://localhost/'));
  await controls.children[0].events.click[0]();
  assert.equal(state.player.current.audio_sha256, 'output-hash');
  assert.equal(state.player.media.paused, false);
  assert.equal(env.calls.filter(call => call.url.endsWith('/render')).length, 1);
  assert.ok(!env.calls.some(call => /preferences|assign|audition/.test(call.url)));
});

test('drag/drop rejects unsupported files and import retries retain their request identity', async () => {
  const env = creatorEnvironment(); await mount(env.doc, env.win);
  let prevented = 0;
  await env.elements['book-drop'].ondrop({ preventDefault() { prevented++; }, dataTransfer: { files: [{ name: 'unexpected.exe' }] } });
  assert.equal(prevented, 1); assert.equal(env.elements['import-book'].disabled, true);
  assert.match(env.elements.error.textContent, /PDF, EPUB, or TXT/);
  await env.elements['book-drop'].ondrop({ preventDefault() {}, dataTransfer: { files: [{ name: 'local.txt' }] } });
  env.server.failImport = true; await env.elements['import-book'].onclick();
  assert.match(env.elements.error.textContent, /Source validation failed/);
  env.server.failImport = false; await env.elements['import-book'].onclick();
  const imports = env.calls.filter(call => call.url.endsWith('/import'));
  assert.equal(imports.length, 2); assert.equal(imports[0].request.body.get('request_id'), imports[1].request.body.get('request_id'));
  assert.equal(env.elements['book-setup'].hidden, false);
});

test('reopening an interrupted reading is read-only until explicit Resume sends the original intent', async () => {
  const env = creatorEnvironment();
  env.server.book.requests = [{ request_id: 'durable-old-request', status: 'interrupted', chapter_ids: ['chapter-two'], library_voice_id: 'library-one', voice_version: 'version-one', completed: 1, total: 2, outputs: [] }];
  await mount(env.doc, env.win);
  env.elements['creator-books'].value = 'book-one'; await env.elements['open-creator-book'].onclick();
  assert.equal(env.elements['resume-book'].hidden, false);
  assert.equal(env.elements['book-chapters'].children[1].children[0].checked, true);
  assert.ok(!env.calls.some(call => call.request.method === 'POST'));
  await env.elements['resume-book'].onclick();
  const body = JSON.parse(env.calls.find(call => call.url.endsWith('/render')).request.body);
  assert.deepEqual(body, { chapter_ids: ['chapter-two'], library_voice_id: 'library-one', voice_version: 'version-one', request_id: 'durable-old-request' });
});

test('persisted candidate role survives dropdown changes; preview then Play and Use are distinct actions', async () => {
  const data = projectData(); data.roles.push({ id: 'other', name: 'Other character', current_identity_id: 'other-identity' });
  data.shortlists = { r: { matches: [{ id: 'candidate-one', performer: 'Reference performer', performance: 'Verified performance' }] } };
  const env = mountedEnvironment(data); const original = env.win.fetch;
  const auditionVoice = { ...recording('audition-v1', 'audition-hash'), voice_id: 'preview-identity', version: 'preview-version', name: 'Original preview', role_id: 'r', duration_seconds: 10 };
  env.win.fetch = async (url, request) => {
    if (url.endsWith('/audition')) { env.calls.push({ url, request }); data.voices.push(auditionVoice); return { ok: true, json: async () => ({ request_id: 'request-id', status: 'succeeded', key: auditionVoice.key }) }; }
    return original(url, request);
  };
  const state = await mount(env.doc, env.win);
  env.elements['match-role'].value = 'other'; env.elements['match-role'].onchange();
  let card = env.elements.matches.children[0]; assert.equal(card.dataset.roleId, 'r');
  let actions = card.children.find(el => el.className === 'preview-actions');
  assert.match(actions.children[0].textContent, /Preview/); assert.equal(actions.children[1].disabled, true);
  await actions.children[0].events.click[0]();
  assert.deepEqual(JSON.parse(env.calls.find(call => call.url.endsWith('/audition')).request.body), { role_id: 'r', candidate_id: 'candidate-one', expected_identity_id: 'identity-r', request_id: 'request-id' });
  assert.equal(state.player.media.paused, true, 'generated preview waits for an explicit user Play tap');
  card = env.elements.matches.children[0]; actions = card.children.find(el => el.className === 'preview-actions');
  assert.match(actions.children[0].textContent, /^Play preview/);
  await actions.children[0].events.click[0]();
  assert.equal(state.player.current.key, 'audition-v1'); assert.equal(state.player.media.paused, false);
  await env.elements.favorite.onclick();
  assert.ok(!env.calls.some(call => call.url.endsWith('/assign')));
  await actions.children[1].events.click[0]();
  assert.deepEqual(JSON.parse(env.calls.find(call => call.url.endsWith('/assign')).request.body), { key: 'audition-v1', role_id: 'r', expected_identity_id: 'identity-r' });
  assert.equal(env.calls.filter(call => call.url.endsWith('/audition')).length, 1);
});

test('durable preview status refresh uses GET; conditioning-only records never become playable', async () => {
  const data = projectData();
  data.shortlists = { r: { matches: [{ id: 'candidate-one', performer: 'Reference' }] } };
  data.auditions = [{ role_id: 'r', candidate_id: 'candidate-one', expected_identity_id: 'identity-r', request_id: 'saved-preview', status: 'queued', key: 'design-only' }];
  data.voices.push({ key: 'design-only', role_id: 'r', name: 'Prepared reference', version: 'conditioning', audio_url: null });
  const env = mountedEnvironment(data); const original = env.win.fetch;
  env.win.fetch = async (url, request) => { if (url.endsWith('/auditions/saved-preview')) { env.calls.push({ url, request }); return { ok: true, json: async () => ({ ...data.auditions[0], status: 'interrupted', reason: 'Production lane was busy.' }) }; } return original(url, request); };
  await mount(env.doc, env.win);
  let actions = env.elements.matches.children[0].children.find(el => el.className === 'preview-actions');
  assert.match(actions.children[0].textContent, /Resume preview/); assert.equal(actions.children[1].disabled, true);
  await actions.children[2].events.click[0]();
  assert.equal(env.calls.find(call => call.url.endsWith('/auditions/saved-preview')).request.method, undefined);
  assert.ok(!env.calls.some(call => call.url.endsWith('/audition')));
  actions = env.elements.matches.children[0].children.find(el => el.className === 'preview-actions');
  assert.equal(actions.children[1].disabled, true);
  assert.match(env.elements.matches.children[0].children.at(-1).textContent, /Production lane/);
  assert.match(env.elements['voice-list'].children.at(-1).textContent, /reference only/);
});

test('private home uses only authorized proxy routes and shared-book navigation', async () => {
  const env = creatorEnvironment();
  env.win.location = { href: 'https://private.example/voice-lab?home=1', origin: 'https://private.example', pathname: '/voice-lab', search: '?home=1' };
  env.win.fetch = async (url, request = {}) => { env.calls.push({ url, request }); return { ok: true, json: async () => url === '/api/voice-lab-projects' ? [{ id: 'shared-book', name: 'Shared existing book' }] : { books: [env.server.book], voices: [] } }; };
  await mount(env.doc, env.win);
  assert.deepEqual(env.calls.map(call => call.url).sort(), ['/api/book-creator', '/api/voice-lab-projects', '/api/voice-library/recordings']);
  env.elements.projects.value = 'shared-book'; env.elements['open-project'].onclick();
  assert.equal(env.win.location.href, '/voice-lab?project=shared-book');
  assert.equal(env.elements['back-link'].href, '/');
  assert.equal(env.elements['player-feedback'].hidden, true);
});

test('private project retains feedback and links to private library home', async () => {
  const env = mountedEnvironment(projectData());
  env.win.location = { href: 'https://private.example/voice-lab?project=test', origin: 'https://private.example', pathname: '/voice-lab', search: '?project=test' };
  await mount(env.doc, env.win);
  assert.equal(env.elements['library-home'].href, '/voice-lab?home=1');
  assert.equal(env.elements['library-home'].hidden, false);
  assert.equal(env.elements['player-feedback'].hidden, false);
  assert.ok(env.calls.some(call => call.url === '/api/voice-lab/test'));
  assert.ok(env.calls.some(call => call.url === '/api/voice-lab/test/casting/book'));
  assert.ok(!env.calls.some(call => /book-creator/.test(call.url)));
});

test('late import completion cannot replace a newer selected reading', async () => {
  const env = creatorEnvironment(); const fetcher = env.win.fetch; let finishImport;
  env.win.fetch = async (url, request) => { if (url.endsWith('/import')) { env.calls.push({ url, request }); await new Promise(resolve => { finishImport = resolve; }); return { ok: true, json: async () => ({ ...env.server.book, project_id: 'late-import', name: 'Late imported book' }) }; } return fetcher(url, request); };
  await mount(env.doc, env.win);
  env.elements['book-file'].files = [{ name: 'later.txt' }]; await env.elements['book-file'].onchange();
  const importing = env.elements['import-book'].onclick();
  env.elements['creator-books'].value = 'book-one'; await env.elements['open-creator-book'].onclick();
  finishImport(); await importing;
  assert.equal(env.elements['book-title'].textContent, 'Local fixture book');
  assert.ok(!env.calls.some(call => call.url.endsWith('/render')));
});

test('opening a saved reading restores its exact fixed narrator instead of a previous picker choice', async () => {
  const env = creatorEnvironment(); const fetcher = env.win.fetch;
  const fixed = { library_voice_id: 'library-two', voice_version: 'version-two' };
  env.server.book.voice_selection = fixed;
  env.server.book.requests = [{ ...fixed, request_id: 'saved-reading', status: 'succeeded', chapter_ids: ['chapter-one'], completed: 3, total: 3 }];
  env.win.fetch = async (url, request) => {
    if (url === '/api/v1/book-creator') {
      const response = await fetcher(url, request); const result = await response.json();
      result.voices.push({ library_voice_id: 'library-two', name: 'Fixed narrator', versions: [{ version: 'version-two', audio_url: '/fixed', audio_sha256: 'fixed-hash' }] });
      return { ok: true, json: async () => result };
    }
    return fetcher(url, request);
  };
  await mount(env.doc, env.win);
  env.elements['book-narrator'].value = JSON.stringify({ library_voice_id: 'library-one', voice_version: 'version-one' });
  env.elements['book-narrator'].onchange();
  env.elements['creator-books'].value = 'book-one'; await env.elements['open-creator-book'].onclick();
  assert.deepEqual(JSON.parse(env.elements['book-narrator'].value), fixed);
  assert.equal(env.elements['book-narrator'].disabled, true);
  assert.ok(!env.calls.some(call => call.request.method === 'POST'));
});

test('late status from a prior request cannot replace a newer reading request in the same book', async () => {
  const env = creatorEnvironment(); const fetcher = env.win.fetch; let finishStatus;
  env.server.book.requests = [{ request_id: 'old-reading', status: 'succeeded', chapter_ids: ['chapter-one'], library_voice_id: 'library-one', voice_version: 'version-one', completed: 3, total: 3 }];
  env.win.crypto.randomUUID = () => 'new-reading';
  env.win.fetch = async (url, request) => {
    if (url.endsWith('/requests/old-reading')) { env.calls.push({ url, request }); await new Promise(resolve => { finishStatus = resolve; }); return { ok: true, json: async () => ({ ...env.server.book.requests[0], status: 'failed', message: 'Stale request status' }) }; }
    return fetcher(url, request);
  };
  await mount(env.doc, env.win);
  env.elements['creator-books'].value = 'book-one'; await env.elements['open-creator-book'].onclick();
  const refreshing = env.elements['refresh-book-progress'].onclick();
  await env.elements['create-mp3'].onclick();
  finishStatus(); await refreshing;
  assert.match(env.elements['book-progress'].textContent, /running/);
  assert.doesNotMatch(env.elements['book-progress'].textContent, /Stale request/);
  assert.equal(env.elements['resume-book'].hidden, true);
});

test('preview Resume preserves an explicitly null original identity token', async () => {
  const data = projectData(); data.shortlists = { r: { matches: [{ id: 'candidate-one', performer: 'Reference' }] } };
  data.auditions = [{ role_id: 'r', candidate_id: 'candidate-one', expected_identity_id: null, request_id: 'original-preview', status: 'interrupted' }];
  const env = mountedEnvironment(data); const fetcher = env.win.fetch;
  env.win.fetch = async (url, request) => {
    if (url.endsWith('/audition')) { env.calls.push({ url, request }); return { ok: true, json: async () => ({ ...data.auditions[0], status: 'queued' }) }; }
    return fetcher(url, request);
  };
  await mount(env.doc, env.win);
  const actions = env.elements.matches.children[0].children.find(el => el.className === 'preview-actions');
  await actions.children[0].events.click[0]();
  assert.deepEqual(JSON.parse(env.calls.find(call => call.url.endsWith('/audition')).request.body), { role_id: 'r', candidate_id: 'candidate-one', expected_identity_id: null, request_id: 'original-preview' });
});

function completedRecording(env, overrides = {}) {
  return { ...env.server.output, project_id: 'book-one', project_name: 'Snow fixture', review_url: '/web/voice-lab.html?home=1&book=book-one&output=export-one', kind: 'creator', created_at: '2026-09-27T00:00:00Z', ...overrides };
}
function runningReading(env, overrides = {}) {
  env.server.request = { request_id: 'long-reading', status: 'running', chapter_ids: ['chapter-one'], library_voice_id: 'library-one', voice_version: 'version-one', completed: 1, total: 853, outputs: [], ...overrides };
  env.server.book.requests = [env.server.request]; env.server.status = env.server.request.status;
}
function fakeTimers(env) {
  let nextId = 0; const pending = new Map(); const events = {};
  env.win.setTimeout = (fn, delay) => { const id = ++nextId; pending.set(id, { fn, delay }); return id; };
  env.win.clearTimeout = id => pending.delete(id);
  env.win.addEventListener = (name, fn) => { (events[name] ||= []).push(fn); };
  env.doc.hidden = false;
  const flush = () => new Promise(resolve => setImmediate(resolve));
  return { pending, flush, async tick() { const [id, timer] = pending.entries().next().value || []; assert.ok(timer, 'an active request must retain its next progress check'); pending.delete(id); timer.fn(); await flush(); return timer.delay; }, async windowEvent(name) { for (const fn of events[name] || []) fn(); await flush(); } };
}

test('finished library groups every output by book and search/play/download/review never render', async () => {
  const env = creatorEnvironment();
  env.server.recordings = [completedRecording(env, { label: 'Snow I', export_id: 'snow-one' }), completedRecording(env, { label: 'Snow II', export_id: 'snow-two' }), completedRecording(env, { label: 'Snow I + II', export_id: 'snow-combined', duration_seconds: 3688 }), completedRecording(env, { project_id: 'other-book', project_name: 'Other fixture', export_id: 'other', label: 'One chapter' })];
  const state = await mount(env.doc, env.win);
  assert.equal(env.elements['finished-recordings'].children.length, 2);
  const group = env.elements['finished-recordings'].children[0]; assert.equal(group.children.length, 4);
  assert.match(group.children[3].children[1].textContent, /1:01:28/);
  env.elements['recording-search'].value = 'snow ii'; env.elements['recording-search'].oninput();
  assert.equal(env.elements['finished-recordings'].children.length, 1);
  assert.equal(env.elements['finished-recordings'].children[0].children.length, 2);
  const card = env.elements['finished-recordings'].children[0].children[1]; const controls = card.children[2];
  assert.equal(controls.children[1].textContent, 'Download MP3'); assert.match(controls.children[1].href, /download=true/);
  assert.equal(controls.children[2].textContent, 'Open & review');
  await controls.children[0].events.click[0]();
  assert.equal(state.player.current.key, 'book-book-one:snow-two'); assert.equal(state.player.media.paused, false);
  assert.ok(env.calls.every(call => !call.request.method));
  env.elements['recording-search'].value = 'no matching book'; env.elements['recording-search'].oninput();
  assert.match(env.elements['recordings-status'].textContent, /No matches/);
});

test('book/output deep link opens the exact completed recording paused with GET only', async () => {
  const env = creatorEnvironment(); env.server.recordings = [completedRecording(env, { project_id: 'other-book', audio_sha256: 'wrong-project-hash' }), completedRecording(env)];
  env.server.book.recordings = [env.server.output];
  env.win.location.search = '?home=1&book=book-one&output=export-one';
  env.win.location.href += env.win.location.search;
  const state = await mount(env.doc, env.win);
  assert.equal(env.elements['book-title'].textContent, 'Local fixture book');
  assert.equal(state.player.current.audio_sha256, 'output-hash'); assert.equal(state.player.media.paused, true);
  assert.equal(env.elements['finished-recordings'].children[1].children[1].attrs['aria-current'], 'true');
  assert.match(env.elements['recordings-status'].textContent, /selected.*Press Play/);
  assert.ok(env.calls.every(call => !call.request.method));
});

test('legacy output deep link selects audio without treating a legacy project as a creator import', async () => {
  const env = creatorEnvironment(); env.server.recordings = [completedRecording(env, { project_id: 'legacy-book', kind: 'voice_lab' })];
  env.win.location.search = '?home=1&book=legacy-book&output=export-one';
  const state = await mount(env.doc, env.win);
  assert.equal(state.player.current.key, 'book-legacy-book:export-one'); assert.equal(state.player.media.paused, true);
  assert.ok(!env.calls.some(call => call.url === '/api/v1/book-creator/legacy-book'));
});

test('missing deep-linked output is explicit and cannot silently select another recording', async () => {
  const env = creatorEnvironment(); env.server.recordings = [completedRecording(env)];
  env.win.location.search = '?home=1&book=book-one&output=missing';
  const state = await mount(env.doc, env.win);
  assert.equal(state.player.current, null); assert.match(env.elements.error.textContent, /linked recording is not available/);
  assert.ok(env.calls.every(call => !call.request.method));
});

test('finished recording cards reject unsafe audio, download and review URLs', async () => {
  const env = creatorEnvironment(); env.server.recordings = [completedRecording(env, { audio_url: 'https://untrusted.example/audio', download_url: 'javascript:alert(1)', review_url: 'https://untrusted.example/review' })];
  await mount(env.doc, env.win);
  const controls = env.elements['finished-recordings'].children[0].children[1].children[2];
  assert.equal(controls.children.length, 1); assert.equal(controls.children[0].disabled, true);
});

test('long render polling remains live after 500 checks and publishes completed outputs without autoplay', async () => {
  const env = creatorEnvironment(); runningReading(env); const timers = fakeTimers(env);
  const state = await mount(env.doc, env.win);
  env.elements['creator-books'].value = 'book-one'; await env.elements['open-creator-book'].onclick();
  assert.equal([...timers.pending.values()][0].delay, 4000);
  for (let i = 0; i < 500; i++) await timers.tick();
  assert.equal(timers.pending.size, 1); assert.equal([...timers.pending.values()][0].delay, 30000);
  assert.match(env.elements['book-poll-status'].textContent, /Live progress/);
  env.server.status = 'succeeded'; env.server.recordings = [completedRecording(env)]; await timers.tick();
  assert.equal(timers.pending.size, 0); assert.equal(env.elements['finished-recordings'].children.length, 1);
  assert.equal(state.player.current, null); assert.ok(env.calls.every(call => !call.request.method));
});

test('hidden/pagehide pauses progress checks and return refreshes durable status immediately', async () => {
  const env = creatorEnvironment(); runningReading(env); const timers = fakeTimers(env);
  await mount(env.doc, env.win); env.elements['creator-books'].value = 'book-one'; await env.elements['open-creator-book'].onclick();
  env.doc.hidden = true; env.doc.emit('visibilitychange'); await timers.flush();
  assert.equal(timers.pending.size, 0); assert.match(env.elements['book-poll-status'].textContent, /pauses while this page is hidden/);
  const before = env.calls.filter(call => call.url.includes('/requests/')).length;
  env.doc.hidden = false; env.doc.emit('visibilitychange'); await timers.flush();
  assert.equal(env.calls.filter(call => call.url.includes('/requests/')).length, before + 1); assert.equal(timers.pending.size, 1);
  await timers.windowEvent('pagehide'); assert.equal(timers.pending.size, 0);
  await timers.windowEvent('pageshow'); assert.equal(timers.pending.size, 1);
  assert.ok(env.calls.every(call => !call.request.method));
});

test('temporary progress network failure backs off and automatically recovers without render dispatch', async () => {
  const env = creatorEnvironment(); runningReading(env); const timers = fakeTimers(env); const fetcher = env.win.fetch; let fail = true;
  env.win.fetch = async (url, request) => { if (url.includes('/requests/') && fail) { env.calls.push({ url, request }); throw new Error('Temporary offline fixture'); } return fetcher(url, request); };
  await mount(env.doc, env.win); env.elements['creator-books'].value = 'book-one'; await env.elements['open-creator-book'].onclick();
  await timers.tick(); assert.equal([...timers.pending.values()][0].delay, 10000); assert.match(env.elements['book-poll-status'].textContent, /Connection unavailable.*Retrying/);
  fail = false; await timers.tick(); assert.equal([...timers.pending.values()][0].delay, 4000); assert.equal(env.elements.error.hidden, true);
  assert.ok(env.calls.every(call => !call.request.method));
});

test('explicit recovery pause preserves progress and accepted push is never shown as confirmed phone delivery', async () => {
  const env = creatorEnvironment(); runningReading(env, { automation: { enabled: true, paused: false, retry_count: 1, max_retries: 2, reason: 'One transient retry used' }, notification: { status: 'accepted', accepted: true, device_delivery_confirmed: false } });
  const fetcher = env.win.fetch;
  env.win.fetch = async (url, request) => { if (url.endsWith('/automation')) { env.calls.push({ url, request }); env.server.request.automation.paused = JSON.parse(request.body).paused; return { ok: true, json: async () => structuredClone(env.server.request) }; } return fetcher(url, request); };
  await mount(env.doc, env.win); env.elements['creator-books'].value = 'book-one'; await env.elements['open-creator-book'].onclick();
  assert.match(env.elements['book-automation-status'].textContent, /1\/2 retries used/);
  assert.match(env.elements['book-notification-status'].textContent, /accepted.*provider.*not confirmed/);
  await env.elements['book-automation-toggle'].onclick();
  const mutations = env.calls.filter(call => call.request.method === 'POST');
  assert.equal(mutations.length, 1); assert.match(mutations[0].url, /\/long-reading\/automation$/);
  assert.deepEqual(JSON.parse(mutations[0].request.body), { paused: true });
  assert.match(env.elements['book-automation-status'].textContent, /Recovery paused/);
  assert.equal(env.elements['book-progress-bar'].value, 1);
});

test('scheduled retries remain visible and polling never performs the retry itself', async () => {
  const env = creatorEnvironment(); runningReading(env, { status: 'interrupted', automation: { enabled: true, paused: false, retry_count: 0, max_retries: 2, state: 'scheduled', next_retry_at: '2026-09-27T12:01:00Z', reason: 'Transient transport error' }, notification: { status: 'delivery_unknown', accepted: false } });
  const timers = fakeTimers(env); await mount(env.doc, env.win); env.elements['creator-books'].value = 'book-one'; await env.elements['open-creator-book'].onclick();
  assert.equal(timers.pending.size, 1); assert.match(env.elements['book-automation-status'].textContent, /Next retry: 2026-09-27T12:01:00Z/);
  assert.match(env.elements['book-notification-status'].textContent, /uncertain.*not be sent again automatically/);
  await timers.tick(); assert.equal(timers.pending.size, 1);
  assert.ok(env.calls.every(call => !call.request.method));
});

test('known notification failure with scheduled retry keeps status polling until provider acceptance', async () => {
  const env = creatorEnvironment(); runningReading(env, { status: 'succeeded', notification: { status: 'failed', next_retry_at: '2026-09-27T12:01:00Z', message: 'Retry scheduled' } });
  const timers = fakeTimers(env); await mount(env.doc, env.win); env.elements['creator-books'].value = 'book-one'; await env.elements['open-creator-book'].onclick();
  assert.equal(timers.pending.size, 1); assert.match(env.elements['book-notification-status'].textContent, /failed.*Retry scheduled/);
  env.server.request.notification = { status: 'accepted', accepted: true, next_retry_at: null, device_delivery_confirmed: false };
  await timers.tick(); assert.equal(timers.pending.size, 0);
  assert.match(env.elements['book-notification-status'].textContent, /accepted.*not confirmed/);
  assert.ok(env.calls.every(call => !call.request.method));
});

function previewPollEnvironment(auditions) {
  const data = projectData(); data.shortlists = { r: { matches: [{ id: 'candidate-one', performer: 'Reference' }] } }; data.auditions = auditions;
  const env = mountedEnvironment(data); const timers = fakeTimers(env); const original = env.win.fetch;
  env.win.document = env.doc;
  env.win.fetch = async (url, request) => { if (url.includes('/auditions/')) { env.calls.push({ url, request }); return { ok: true, json: async () => ({ status: 'running' }) }; } return original(url, request); };
  const asked = () => env.calls.filter(call => call.url.includes('/auditions/')).map(call => call.url.split('/').pop());
  const fireAll = async () => { for (const [id, timer] of [...timers.pending]) { timers.pending.delete(id); timer.fn(); } await timers.flush(); };
  return { env, timers, asked, fireAll };
}

test('saved preview receipts are not polled: waiting ones never, a running one every few seconds', async () => {
  const { env, timers, asked, fireAll } = previewPollEnvironment([
    { role_id: 'r', candidate_id: 'c1', request_id: 'waiting-1', status: 'queued' },
    { role_id: 'r', candidate_id: 'c2', request_id: 'waiting-2', status: 'queued' },
    { role_id: 'r', candidate_id: 'c3', request_id: 'live-1', status: 'running' }]);
  await mount(env.doc, env.win);
  await fireAll();
  assert.deepEqual(asked(), ['live-1'], 'a receipt that waits for someone to press Resume cannot change by itself, so it is not asked about');
  assert.ok([...timers.pending.values()].some(timer => timer.delay === 3000), 'the running one keeps being followed');
});

test('a tab that is not shown does not ask about running previews, and catches up when shown', async () => {
  const { env, timers, asked, fireAll } = previewPollEnvironment([{ role_id: 'r', candidate_id: 'c3', request_id: 'live-1', status: 'running' }]);
  await mount(env.doc, env.win);
  env.doc.hidden = true; await fireAll();
  assert.deepEqual(asked(), [], 'hidden: no request');
  assert.equal(timers.pending.size, 1, 'but the next look is still scheduled');
  env.doc.hidden = false; await fireAll();
  assert.deepEqual(asked(), ['live-1']);
});

test('a preview started on this page is still followed until it finishes', async () => {
  const { env, timers, asked, fireAll } = previewPollEnvironment([{ role_id: 'r', candidate_id: 'candidate-one', expected_identity_id: null, request_id: 'original-preview', status: 'interrupted' }]);
  const fetcher = env.win.fetch;
  env.win.fetch = async (url, request) => { if (url.endsWith('/audition')) { env.calls.push({ url, request }); return { ok: true, json: async () => ({ role_id: 'r', candidate_id: 'candidate-one', request_id: 'original-preview', status: 'queued' }) }; } return fetcher(url, request); };
  await mount(env.doc, env.win);
  await fireAll(); assert.deepEqual(asked(), [], 'nothing is followed before anyone asks for a preview');
  const actions = env.elements.matches.children[0].children.find(el => el.className === 'preview-actions');
  await actions.children[0].events.click[0]();
  await fireAll();
  assert.deepEqual(asked(), ['original-preview'], 'the request this page just made is followed');
});
