'use strict';
const { test } = require('node:test');
const assert = require('node:assert/strict');
const { mountLab } = require('./helpers/voice_lab_harness.cjs');
const hash = n => String(n).repeat(64).slice(0, 64);
const cand = (id, extra = {}) => ({ candidate_id: id, label: id.toUpperCase(), key: 'voice-' + id, audio_url: '/audio/' + id, audio_sha256: hash(id.charCodeAt(0) % 10), duration_seconds: 8, status: 'ready', rating: null, sections: [], ...extra });
const fixture = () => ({ roles: [{ role_id: 'r1', name: 'Mara', status: 'NEEDS_REVIEW', assignment_revision: 5, round_revision: 3, round_status: 'awaiting_review', current_key: null, candidates: [cand('a'), cand('b', { rating: 4 }), cand('c')], direction: {}, history: [] }] });
const has = (el, cls) => (el.className || '').split(/\s+/).includes(cls);
const within = (el, cls, out = []) => { if (has(el, cls)) out.push(el); for (const child of el.children || []) within(child, cls, out); return out; };
const article = (lab, id) => lab.all('casting-candidate').find(el => el.dataset.candidateId === id);

test('one article per candidate and no duplicate play controls', async () => {
  const lab = await mountLab(fixture());
  assert.equal(lab.all('casting-candidate').length, 3);
  assert.equal(lab.all('casting-play').length, 3);
  assert.equal(lab.requests.filter(r => r.method === 'POST').length, 0);
});

test('fallback panels are native details and unknown engine is honest', async () => {
  const lab = await mountLab(fixture());
  assert.ok(lab.all('voice-candidate-panel-fallback').length >= 6);
  assert.match(lab.all('voice-candidate-meta')[0].textContent, /Engine unknown/);
  assert.equal(lab.all('voice-candidate-badges').length, 3);
});

test('heart sends only the favorite preference and keeps one host', async () => {
  const lab = await mountLab(fixture());
  await lab.click(within(article(lab, 'a'), 'casting-heart')[0]); await lab.settle();
  const posts = lab.requests.filter(r => r.path === 'preferences');
  assert.deepEqual(posts.map(r => r.body), [{ key: 'voice-a', audio_sha256: hash(97 % 10), favorite: true }]);
  assert.equal(lab.all('casting-candidate').length, 3);
});

test('check sends confirmed use without stars', async () => {
  const lab = await mountLab(fixture());
  await lab.click(within(article(lab, 'b'), 'casting-check')[0]); await lab.settle();
  const body = lab.requests.find(r => r.path === 'casting/feedback').body;
  assert.equal(body.action, 'use'); assert.equal(body.selection_confirmed, true); assert.equal('stars' in body, false);
});

test('comparison moves hosts without duplicating candidates or players', async () => {
  const lab = await mountLab(fixture());
  await lab.click(within(article(lab, 'a'), 'casting-compare-toggle')[0]); await lab.settle();
  await lab.click(within(article(lab, 'c'), 'casting-compare-toggle')[0]); await lab.settle();
  assert.equal(lab.all('casting-compare').length, 1);
  assert.equal(lab.all('casting-compare-slot').length, 2);
  assert.equal(lab.all('casting-candidate').length, 3);
  assert.equal(lab.all('casting-play').length, 3);
  await lab.click(lab.all('casting-compare-play')[0]); await lab.settle();
  assert.equal(lab.currentContext().candidate_id, 'a');
});
