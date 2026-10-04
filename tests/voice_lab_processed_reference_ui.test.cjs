const test = require('node:test');
const assert = require('node:assert/strict');
const { mountLab } = require('./helpers/voice_lab_harness.cjs');
const hash = 'a'.repeat(64);
const candidate = { candidate_id: 'processed', status: 'ready', key: 'v_processed', audio_sha256: hash,
 audio_url: '/a/processed.wav', duration_seconds: 18.72, sections: [] };
const fixture = { casting_enabled: true, roles: [{ role_id: 'hermione', name: 'Hermione',
 status: 'NEEDS_REVIEW', round_revision: 1, assignment_revision: 1, candidates: [candidate] }] };
for (const exact of [true, false]) {
 test('processed-reference label requires exact recorded key and hash: ' + exact, async () => {
  const lab = await mountLab(structuredClone(fixture), { respond: action => action === '' ? {
   project_name: 'Reference fixture', casting_enabled: true, capabilities: { casting_enabled: true, casting_review: true },
   roles: [{ id: 'hermione', name: 'Hermione' }], issues: [], automation: { paused: true },
   voices: [{ key: candidate.key, audio_sha256: exact ? hash : 'b'.repeat(64),
    details: { provenance: { preprocessing: [{ operation: 'reference_vocals_isolation' }] } } }]
  } : undefined });
  assert.equal(lab.all('casting-circle-source-note').some(n => n.textContent.includes('Processed reference')), exact);
  assert.ok(!lab.requests.some(r => r.method === 'POST'));
 });
}
