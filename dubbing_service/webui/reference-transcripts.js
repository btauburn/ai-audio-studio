/* Explicit local transcription and correction; never selects or rates a voice. */
(function () {
  'use strict';
  const start = () => {
    const panel = document.getElementById('reference-transcripts');
    if (!panel) return;
    const project = new URLSearchParams(location.search).get('project');
    if (!project) return;
    const prefix = location.pathname === '/voice-lab'
      ? `/api/voice-lab/${encodeURIComponent(project)}` : `/api/v1/projects/${encodeURIComponent(project)}/voice-lab`;
    const rows = panel.querySelector('[data-references]');
    let loadEpoch = 0;
    const syncPlayback = () => { const playing = Array.from(document.querySelectorAll('audio')).some(a => a.id !== 'media' && !a.paused && !a.ended); if (playing) document.documentElement.dataset.runtimeComparison = 'playing'; else delete document.documentElement.dataset.runtimeComparison; };
    const request = async (suffix, payload) => {
      const response = await fetch(`${prefix}/${suffix}`, payload ? {method:'POST', headers:{'Content-Type':'application/json','X-Review-Request':'1'}, body:JSON.stringify(payload), credentials:'same-origin'} : {});
      const result = await response.json();
      if (!response.ok) throw new Error(result.detail || 'Reference request failed');
      return result;
    };
    panel.querySelector('button').onclick = async () => {
      const epoch = ++loadEpoch;
      rows.querySelectorAll('audio').forEach(a => { a.pause(); a.removeAttribute('src'); a.load(); }); syncPlayback();
      rows.textContent = 'Loading reference clips...';
      try {
        const data = await request('references'); if (epoch !== loadEpoch) return; rows.textContent = '';
        if (!data.references.length) rows.textContent = 'No recorded reference clips prepared yet.';
        for (const reference of data.references) {
          const card = document.createElement('article'); const name = document.createElement('h3');
          name.textContent = reference.source.performer || reference.source.title || 'Reference recording'; card.append(name);
          const source = document.createElement('a'); try { const url = new URL(reference.source.source_page); if (url.protocol === 'https:' && (url.hostname === 'commons.wikimedia.org' || (reference.source?.provider === 'reviewed_dataset_recording' && ((url.hostname === 'datashare.ed.ac.uk' && url.pathname === '/handle/10283/3443') || (url.hostname === 'github.com' && url.pathname === '/dioco-group/jenny-tts-dataset'))))) source.href = url.href; } catch (_) {} source.textContent = 'Recording source and licence'; source.target = '_blank'; source.rel = 'noopener'; card.append(source);
          const audio = document.createElement('audio'); audio.controls = true; audio.preload = 'none'; audio.src = `${prefix}/references/${reference.acquisition_id}/audio?acquisition_sha256=${reference.acquisition_sha256}`; audio.setAttribute('aria-label', `Actual reference for ${name.textContent}`); audio.style.maxWidth = '100%'; card.append(audio);
          audio.addEventListener('play', () => { document.querySelectorAll('audio').forEach(a => { if (a !== audio) a.pause(); }); syncPlayback(); queueMicrotask(syncPlayback); });
          for (const event of ['pause', 'ended']) audio.addEventListener(event, () => { syncPlayback(); queueMicrotask(syncPlayback); });
          document.querySelector('#media')?.addEventListener('play', () => audio.pause());
          const state = document.createElement('p'); state.textContent = reference.holds.join('; ') || 'Source prepared; synthetic audition listening review remains separate'; card.append(state);
          if (reference.source?.provider === 'reviewed_dataset_recording') {
            const provenance = document.createElement('details'); const title = document.createElement('summary'); title.textContent = 'Recorded source provenance'; provenance.append(title);
            const info = document.createElement('p'); info.textContent = `${reference.source.attribution || ''}. Original ${reference.source.archive_member || reference.source.source_revision || ''}. Crop ${reference.start_seconds ?? '?'}–${reference.end_seconds ?? '?'}s. Recording date unknown. Source delivery review pending; this is not a generated audition.`; provenance.append(info);
            const hashes = document.createElement('p'); hashes.style.overflowWrap = 'anywhere'; hashes.textContent = `Original SHA-256 ${reference.original_sha256 || 'unknown'} · Reference SHA-256 ${reference.reference_audio_sha256 || 'unknown'}`; provenance.append(hashes); card.append(provenance);
          }
          const transcript = document.createElement('textarea'); transcript.rows = 5; transcript.maxLength = 1000; transcript.setAttribute('aria-label', `Reference transcript for ${name.textContent}`); card.append(transcript);
          const status = document.createElement('p'); status.setAttribute('role', 'status'); card.append(status);
          const run = document.createElement('button'); run.textContent = 'Generate transcript locally'; card.append(run);
          const save = document.createElement('button'); save.textContent = 'Save transcript correction'; save.disabled = true; card.append(save);
          let result = reference.transcript_evidence || (reference.reference_text ? {status:'provided',text:reference.reference_text,segments:[],uncertainties:[],source_review_method:reference.source_review_method} : null); if (reference.correction) result = {...(result || {status:'correction_only',text:'',segments:[],uncertainties:[]}), correction:reference.correction};
          const check = document.createElement('button'); check.textContent = 'Check reference locally'; card.append(check);
          const clean = document.createElement('button'); clean.textContent = 'Clean voices locally'; card.append(clean);
          const cleanStatus = document.createElement('p'); cleanStatus.setAttribute('role', 'status'); card.append(cleanStatus);
          let cleaning = false, preparedDetails = null;
          clean.onclick = async () => {
            if (cleaning) return;
            cleaning = true; clean.disabled = true;
            const submitted = transcript.value;
            cleanStatus.textContent = 'Checking and cleaning this clip locally. The original recording is preserved...';
            try {
              const prepared = await request(`references/${reference.acquisition_id}/prepare`, {acquisition_sha256:reference.acquisition_sha256, expected_text:submitted.trim() || null});
              if (epoch !== loadEpoch) return;
              const holds = prepared.technical_qa?.holds || [];
              cleanStatus.textContent = prepared.reason || (prepared.status === 'preserved' ? 'Existing reference preserved.' : `${prepared.status === 'prepared' ? 'Reference prepared' : 'Reference needs review'}${holds.length ? ': ' + holds.join('; ') : '.'}`);
              if (transcript.value !== submitted) cleanStatus.textContent += ' Your newer transcript edit is preserved; this result used the submitted words.';
              if (prepared.status === 'preserved') return;
              if (!/^[a-f0-9]{32}$/.test(prepared.acquisition_id || '') || !/^[a-f0-9]{64}$/.test(prepared.acquisition_sha256 || '')) throw new Error('Prepared reference identity is unavailable. Reload reference clips to inspect the saved result.');
              if (preparedDetails) { preparedDetails.querySelectorAll('audio').forEach(a => { a.pause(); a.removeAttribute('src'); a.load(); }); preparedDetails.remove(); syncPlayback(); }
              preparedDetails = document.createElement('div');
              const label = document.createElement('p'); label.textContent = prepared.source_processing ? 'Processed reference · may affect timbre' : 'Prepared reference · original crop retained'; preparedDetails.append(label);
              const preview = document.createElement('audio'); preview.controls = true; preview.preload = 'none'; preview.style.maxWidth = '100%';
              preview.src = `${prefix}/references/${prepared.acquisition_id}/audio?acquisition_sha256=${prepared.acquisition_sha256}`;
              preview.setAttribute('aria-label', `Prepared reference for ${name.textContent}`); preparedDetails.append(preview);
              preview.addEventListener('play', () => { document.querySelectorAll('audio').forEach(a => { if (a !== preview) a.pause(); }); syncPlayback(); queueMicrotask(syncPlayback); });
              for (const event of ['pause', 'ended']) preview.addEventListener(event, () => { syncPlayback(); queueMicrotask(syncPlayback); });
              document.querySelector('#media')?.addEventListener('play', () => preview.pause());
              const words = document.createElement('p'); words.textContent = prepared.reference_text || 'Transcript unavailable; generate a draft after reloading reference clips.'; preparedDetails.append(words);
              const review = document.createElement('p'); review.textContent = 'Listen to the prepared clip. Voice selection and synthetic audition approval remain separate.'; preparedDetails.append(review);
              card.append(preparedDetails);
            } catch (e) { if (epoch === loadEpoch) cleanStatus.textContent = `Preparation unavailable: ${e.message}`; }
            finally { cleaning = false; clean.disabled = false; }
          };
          const qualityStatus = document.createElement('p'); qualityStatus.setAttribute('role', 'status'); card.append(qualityStatus);
          let quality = reference.quality_evidence, qualityDetails = null, asrDetails = null;
          const showQuality = () => {
            if (qualityDetails) { qualityDetails.remove(); qualityDetails = null; }
            if (!quality) { qualityStatus.textContent = 'Local technical checks have not run.'; return; }
            qualityStatus.textContent = quality.reason || `Technical ${quality.status}. Speaker overlap and music are not assessed. Identity, licence and synthetic audition approval remain separate.`;
            if (quality.checks) { qualityDetails = document.createElement('details'); const summary = document.createElement('summary'); summary.textContent = 'Reference check results and limits'; qualityDetails.append(summary); const detail = document.createElement('pre'); detail.style.whiteSpace = 'pre-wrap'; detail.textContent = Object.entries(quality.checks).map(([name, c]) => `${name}: ${c.status} — ${c.reason}`).join('\n') + '\n\n' + JSON.stringify(quality.metrics, null, 2); qualityDetails.append(detail); card.append(qualityDetails); }
          };
          transcript.addEventListener('input', () => { if (quality) { quality = {status:'stale', reason:'Transcript changed; run the local check again.'}; showQuality(); } });
          check.onclick = async () => { check.disabled = true; qualityStatus.textContent = 'Checking the exact reference on CPU...'; const expected = transcript.value.trim() || null; try { const evidence = await request(`references/${reference.acquisition_id}/quality`, {acquisition_sha256:reference.acquisition_sha256, expected_text:expected}); quality = (transcript.value.trim() || null) === expected && evidence.expected_text === expected ? evidence : {status:'stale',reason:'Transcript changed during verification; run the local check again.'}; showQuality(); } catch (e) { qualityStatus.textContent = `Check unavailable: ${e.message}`; } finally { check.disabled = false; } };
          const show = () => {
            const updatedText = result?.correction?.text || result?.text || '';
            if (quality?.expected_text !== undefined && (updatedText.trim() || null) !== quality.expected_text) { quality = {status:'stale',reason:'Transcript changed; run the local check again.'}; showQuality(); }
            transcript.value = updatedText;
            status.textContent = result?.status === 'draft' ? `Machine draft; listen and correct. ${result.uncertainties.length} uncertain passages. Word timestamps are available in the details.` : (result?.reason || 'Not transcribed yet.');
            if (result?.status === 'provided') status.textContent = result.source_review_method === 'human_review' ? 'Source transcript confirmed by the user. Synthetic audition approval remains separate.' : 'Provided source transcript; source review remains separate.';
            if (result?.correction) status.textContent = 'Saved correction; speaker, music and alignment review remain pending.';
            save.disabled = !result || result.status !== 'draft';
            if (asrDetails) { asrDetails.remove(); asrDetails = null; }
            if (result?.segments?.length) { const details = document.createElement('details'); asrDetails = details; const summary = document.createElement('summary'); summary.textContent = 'Timestamps and uncertainty'; details.append(summary); const text = document.createElement('pre'); text.style.whiteSpace = 'pre-wrap'; text.textContent = result.segments.map(s => `${s.start.toFixed(2)}-${s.end.toFixed(2)}s: ${s.text}`).join('\n'); details.append(text); card.append(details); }
          };
          run.onclick = async () => { run.disabled = true; status.textContent = 'Transcribing on CPU...'; try { result = await request(`references/${reference.acquisition_id}/transcript`, {acquisition_sha256:reference.acquisition_sha256}); show(); } catch(e) { status.textContent = e.message; } finally { run.disabled = false; } };
          save.onclick = async () => { save.disabled = true; const submitted = transcript.value; try { const corrected = await request(`references/${reference.acquisition_id}/transcript/correction`, {acquisition_sha256:reference.acquisition_sha256, expected_text_sha256:result.text_sha256 || null, expected_correction_sha256:result.correction?.text_sha256 || null, text:submitted}); result.correction = corrected; status.textContent = transcript.value !== submitted ? 'Correction saved; your newer edit is still unsaved.' : 'Correction saved separately. Speaker, music and alignment review remain pending.'; } catch(e) { status.textContent = e.message; } finally { save.disabled = false; } };
          show(); showQuality(); rows.append(card);
        }
      } catch (e) { if (epoch === loadEpoch) rows.textContent = e.message; }
    };
  };
  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', start, {once:true}); else start();
})();
