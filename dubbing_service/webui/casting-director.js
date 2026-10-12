/* Casting Director panel: explicit, additive review of ranked voice recommendations.
 *
 * Opening the panel is a read-only GET. Analysis and fresh auditions are separate
 * explicit POSTs. Nothing here synthesizes on load, navigation or playback, and
 * nothing here approves, rates, assigns or imports a voice: those stay with the
 * existing casting inbox and reusable-voice controls, which this panel links to. */
(function () {
  'use strict';
  const UNAVAILABLE = 'Casting Director is not available on this connection. Nothing was changed.';
  const ORIGIN_LABEL = {
    approved_library: 'Approved library voice', generic_library: 'Library voice',
    authorized_external: 'Authorized external source', model_design: 'Original synthetic direction',
  };
  const MAX_SCENES = 25;

  function text(value) { return value === null || value === undefined ? '' : String(value).trim(); }
  function list(value) { return Array.isArray(value) ? value : []; }
  function words(value) { return text(value).replaceAll('_', ' '); }
  function percent(value) { return typeof value === 'number' && Number.isFinite(value) ? Math.round(value * 100) + '%' : null; }
  function number(value) { return typeof value === 'number' && Number.isFinite(value) ? value.toFixed(2) : null; }
  function requestId() {
    const crypto = globalThis.crypto;
    if (crypto && typeof crypto.randomUUID === 'function') return 'cd-' + crypto.randomUUID();
    return 'cd-' + Date.now().toString(36) + Math.random().toString(36).slice(2, 10);
  }
  function safeAudio(value, location) {
    try {
      const url = new URL(value, location.href);
      return ['http:', 'https:'].includes(url.protocol) && url.origin === location.origin ? url.href : null;
    } catch (_) { return null; }
  }
  function safeImage(value, location) {
    const raw = text(value);
    if (/^data:image\/(?:png|jpeg|webp|svg\+xml)[;,]/i.test(raw)) return raw;
    return safeAudio(raw, location);
  }
  // Honest explanation text from the ranked candidate; tolerate a plain string.
  function explain(candidate) {
    const e = candidate.explanation;
    if (e && typeof e === 'object') return e;
    return { summary: text(e), strengths: [], concerns: [], unmeasured: [] };
  }

  function mount(doc, win) {
    const $ = id => doc.getElementById ? doc.getElementById(id) : doc.querySelector('#' + id);
    const host = $('casting-director');
    const open = $('casting-director-open');
    const analyzeButton = $('casting-director-analyze');
    const body = $('casting-director-body');
    const status = $('casting-director-status');
    const rolesBox = $('casting-director-roles');
    const scenesBox = $('casting-director-scenes');
    if (!host || !open || !analyzeButton || !body || !status || !rolesBox || !scenesBox) return null;
    host.hidden = false;

    const location = win.location;
    const project = new URLSearchParams(location.search || '').get('project');
    const privateRoute = location.pathname === '/voice-lab';
    const base = project ? (privateRoute
      ? '/api/voice-lab/' + encodeURIComponent(project)
      : '/api/v1/projects/' + encodeURIComponent(project) + '/voice-lab') + '/directors' : null;
    const fetcher = (...args) => win.fetch(...args);

    let epoch = 0;
    let busy = false;
    let report = null;
    let player = null;
    let setPlayerFlag = false;
    const requestIds = new Map();
    const receipts = new Map();
    const auditionBusy = new Set();

    const make = (tag, content, className) => {
      const element = doc.createElement(tag);
      if (className) element.className = className;
      if (content !== undefined && content !== null) element.textContent = content;
      return element;
    };
    const say = (message, tone) => { status.textContent = message; status.dataset.tone = tone || ''; };

    function stopPlayer() {
      if (!player) return;
      player.pause();
      if (typeof player.removeAttribute === 'function') player.removeAttribute('src');
      if (typeof player.load === 'function') player.load();
      clearFlag();
    }
    function clearFlag() {
      if (setPlayerFlag && doc.documentElement.dataset.runtimeComparison === 'playing') delete doc.documentElement.dataset.runtimeComparison;
      setPlayerFlag = false;
    }
    function ensurePlayer() {
      if (player) return player;
      player = doc.createElement('audio');
      player.controls = true;
      player.preload = 'none';
      player.className = 'casting-director-audio';
      // Reuse the existing playback-abstention flag so rating shortcuts and inbox
      // polling stay quiet while an audition from this panel is playing.
      player.addEventListener('play', () => {
        doc.querySelectorAll('audio').forEach(a => { if (a !== player) a.pause(); });
        doc.documentElement.dataset.runtimeComparison = 'playing'; setPlayerFlag = true;
      });
      player.addEventListener('pause', clearFlag);
      player.addEventListener('ended', clearFlag);
      const ordinary = doc.querySelector('#media');
      if (ordinary) ordinary.addEventListener('play', () => player.pause());
      return player;
    }

    async function request(action, init) {
      let response;
      try { response = await fetcher(base + action, init); } catch (_) { const e = new Error(UNAVAILABLE); e.unavailable = true; throw e; }
      let data = null;
      try { data = await response.json(); } catch (_) { /* non-JSON proxy page */ }
      if (!response.ok || !data || typeof data !== 'object') {
        // A proxy that does not expose the route answers 404/405/5xx or HTML: unavailable, not failure of the work.
        const detail = data && typeof data.detail === 'string' ? data.detail : null;
        const refused = [400, 403, 409, 422].includes(response.status);
        const e = new Error(refused ? (detail || 'Request was not accepted. Nothing was changed.') : UNAVAILABLE);
        e.unavailable = !refused;
        e.status = response.status;
        throw e;
      }
      return data;
    }
    const post = body => ({ method: 'POST', headers: { 'Content-Type': 'application/json', 'X-Review-Request': '1' }, credentials: 'same-origin', body: JSON.stringify(body || {}) });

    function badge(label, kind) { const b = make('span', label, 'cd-badge'); b.dataset.kind = kind; return b; }
    function pair(label, value) { const p = make('p', null, 'cd-fact'); p.append(make('strong', label + ' '), make('span', value)); return p; }
    function lines(title, items, className) {
      const rows = list(items).map(text).filter(Boolean);
      if (!rows.length) return null;
      const wrap = make('div', null, className || 'cd-list');
      wrap.append(make('p', title, 'cd-label'));
      const ul = make('ul'); rows.forEach(r => ul.append(make('li', r))); wrap.append(ul);
      return wrap;
    }

    function openInbox(roleId) {
      const card = doc.getElementById ? doc.getElementById('casting-role-' + roleId) : null;
      const target = card || $('casting-inbox');
      if (target) {
        if (typeof target.scrollIntoView === 'function') target.scrollIntoView({ block: 'start' });
        if (typeof target.focus === 'function') target.focus({ preventScroll: true });
      }
      return Boolean(card);
    }
    function openLibrary() {
      const section = $('reusable-voices');
      if (!section) return false;
      const tools = $('casting-extra-tools');
      if (tools) tools.open = true;
      section.open = true;
      if (typeof section.scrollIntoView === 'function') section.scrollIntoView({ block: 'start' });
      return true;
    }

    function renderCandidate(role, candidate, rank) {
      const card = make('li', null, 'cd-candidate');
      card.dataset.candidateId = text(candidate.candidate_id);
      card.dataset.rank = String(rank);
      const head = make('div', null, 'cd-candidate-head');
      head.append(make('span', '#' + rank, 'cd-rank'), make('h5', text(candidate.name) || text(candidate.candidate_id)));
      card.append(head);
      const badges = make('div', null, 'cd-badges');
      badges.append(badge(ORIGIN_LABEL[candidate.origin] || words(candidate.origin) || 'Unknown origin', 'origin'));
      if (candidate.reference_conditioned) badges.append(badge('Reference-conditioned', 'info'));
      if (candidate.protected) badges.append(badge('Protected', 'protected'));
      if (candidate.eligible === false) badges.append(badge('Not eligible', 'hold'));
      list(candidate.holds).forEach(h => badges.append(badge('Hold: ' + words(typeof h === 'object' ? (h.reason || h.code) : h), 'hold')));
      card.append(badges);

      const score = number(candidate.score), confidence = percent(candidate.confidence);
      card.append(pair('Match score', score ? `${score} (recommendation, not an approval)` : 'Not scored'));
      card.append(pair('Confidence', confidence || 'Not stated'));
      const explanation = explain(candidate);
      if (text(explanation.summary)) card.append(make('p', text(explanation.summary), 'cd-summary'));
      const strengths = lines('Matched evidence', explanation.strengths, 'cd-list cd-good');
      const concerns = lines('Concerns', explanation.concerns, 'cd-list cd-warn');
      const unmeasured = lines('Not measured', list(explanation.unmeasured).map(words), 'cd-list cd-muted');
      [strengths, concerns, unmeasured].forEach(n => { if (n) card.append(n); });

      const provenance = candidate.provenance && typeof candidate.provenance === 'object' ? candidate.provenance : {};
      card.append(pair('Rights status', text(provenance.rights_status) ? words(provenance.rights_status) : 'Not recorded (none assumed)'));
      const sources = list(provenance.source_urls).map(text).filter(Boolean);
      if (sources.length) card.append(pair('Sources', sources.length + ' recorded'));

      const actions = make('div', null, 'cd-actions');
      const audio = candidate.audio_sha256 ? safeAudio(candidate.audio_url, location) : null;
      if (audio) {
        const play = make('button', 'Play existing sample', 'secondary cd-play'); play.type = 'button';
        play.dataset.candidateId = text(candidate.candidate_id);
        play.addEventListener('click', () => {
          const element = ensurePlayer();
          if (element.parentNode !== card) card.append(element);
          doc.querySelectorAll('audio').forEach(a => { if (a !== element) a.pause(); });
          element.src = audio;
          const started = element.play();
          if (started && typeof started.catch === 'function') started.catch(() => say('This sample could not be played. Nothing was changed.', 'error'));
        });
        actions.append(play);
        card.append(make('p', 'Earlier sample for this exact voice' + (candidate.comparison_status ? '; a fresh matched-script audition is required before comparing.' : '.'), 'cd-muted small'));
      } else {
        card.append(make('p', 'No audio yet. A fresh audition must be requested for this character; nothing generates until you ask.', 'cd-muted small'));
      }
      if (candidate.library_voice_id) {
        const library = make('button', 'Open reusable voices', 'secondary cd-library'); library.type = 'button';
        library.addEventListener('click', () => { if (!openLibrary()) say('The reusable voices panel is not available here.', 'error'); });
        actions.append(library);
      }
      if (actions.children && actions.children.length) card.append(actions);
      return card;
    }

    function renderRole(role) {
      const card = make('article', null, 'cd-role');
      card.dataset.roleId = text(role.role_id);
      const head = make('header', null, 'cd-role-head');
      const existing = doc.getElementById ? doc.getElementById('casting-role-' + role.role_id) : null;
      const avatarSource = existing && existing.querySelector ? existing.querySelector('.casting-avatar') : null;
      const portrait = avatarSource ? safeImage(avatarSource.src, location) : null;
      if (portrait) { const img = make('img', null, 'cd-avatar'); img.src = portrait; img.alt = ''; head.append(img); }
      const identity = make('div', null, 'cd-identity');
      const analysis = role.analysis && typeof role.analysis === 'object' ? role.analysis : {};
      identity.append(make('h4', text(role.role_name) || text(analysis.role_name) || text(role.role_id)));
      const meta = [words(analysis.role_type), analysis.spoken_lines ? `${analysis.spoken_lines} spoken lines` : '',
        text(analysis.importance) ? `${words(analysis.importance)} role` : '', percent(analysis.confidence) ? `evidence confidence ${percent(analysis.confidence)}` : '']
        .filter(Boolean).join(' · ');
      if (meta) identity.append(make('p', meta, 'cd-muted small'));
      const states = make('div', null, 'cd-badges');
      if (role.protected) states.append(badge('Protected · cannot be recast here', 'protected'));
      if (role.needs_casting) states.append(badge('Needs casting', 'needs'));
      if (role.replacement_candidate) states.append(badge('Replacement candidate', 'warn'));
      if (text(role.coverage) === 'needs_casting') states.append(badge('No viable candidate', 'hold'));
      identity.append(states);
      head.append(identity);
      card.append(head);

      const current = role.current_voice && typeof role.current_voice === 'object' ? role.current_voice : {};
      const stars = typeof current.rating === 'number' ? `${current.rating} of 5 stars` : 'not rated';
      card.append(pair('Current voice', text(current.name) || text(current.key) || 'None chosen'));
      card.append(pair('Current rating', stars + (current.approved ? ' · approved by a person' : '')));
      if (role.policy_conflict) {
        card.append(make('p', 'The saved voice is excluded by this book’s cloned-only policy. Its approval remains saved; rendering is blocked until casting is explicitly reviewed.', 'cd-hold'));
      }

      const evidence = list(analysis.evidence).map(row => {
        const label = words(row.dimension), value = text(row.value);
        return label && value ? `${label}: ${value}${text(row.level) ? ` (${words(row.level)})` : ''}` : '';
      }).filter(Boolean);
      const evidenceBox = lines('What the book supports', evidence, 'cd-list');
      if (evidenceBox) card.append(evidenceBox);
      const missing = lines('No supporting evidence for', list(analysis.missing_evidence).map(words), 'cd-list cd-muted');
      if (missing) card.append(missing);
      const contrast = list(analysis.relationships).filter(r => r && r.needs_contrast).map(r => `Shares scenes with ${text(r.role_name) || text(r.role_id)}; choose a clearly different voice`);
      const contrastBox = lines('Needs contrast', contrast, 'cd-list cd-warn');
      if (contrastBox) card.append(contrastBox);

      const candidates = list(role.candidates);
      const recommendations = make('ol', null, 'cd-candidates');
      recommendations.setAttribute('aria-label', 'Ranked recommendations for ' + (text(role.role_name) || text(role.role_id)));
      candidates.forEach((candidate, index) => recommendations.append(renderCandidate(role, candidate, index + 1)));
      if (candidates.length) card.append(make('h5', 'Ranked recommendations', 'cd-subhead'), recommendations);
      else card.append(make('p', 'No viable candidate was found. This character still needs casting.', 'cd-muted'));

      const actions = make('div', null, 'cd-actions');
      const receiptNote = make('p', null, 'cd-receipt small'); receiptNote.setAttribute('role', 'status'); receiptNote.setAttribute('aria-live', 'polite');
      const request_ = make('button', 'Request fresh auditions', 'cd-request'); request_.type = 'button';
      const reason = role.protected ? 'Approved or highly rated casting is protected and cannot be recast here.'
        : !candidates.length ? 'There is no candidate to audition.' : !report.report_sha256 ? 'Analyze first.' : '';
      const known = receipts.get(role.role_id + ':' + report.report_sha256);
      request_.disabled = Boolean(reason) || Boolean(known);
      if (reason) request_.title = reason;
      const hint = make('p', reason || 'Queues new audio for up to five candidates on the GPU. Nothing generates until you press this.', 'cd-muted small');
      const open_ = make('button', 'Open in casting inbox', 'secondary cd-inbox'); open_.type = 'button';
      open_.addEventListener('click', () => { if (!openInbox(role.role_id)) say('That character is not shown in the casting inbox right now. Check “Show chosen voices”.', ''); });
      request_.addEventListener('click', () => audition(role, request_, receiptNote));
      actions.append(request_, open_);
      card.append(actions, hint, receiptNote);
      if (known) receiptNote.textContent = receiptText(known);
      return card;
    }

    function receiptText(receipt) {
      const holds = list(receipt.holds).length ? ` ${list(receipt.holds).length} candidate(s) held.` : '';
      return `Auditions ${words(receipt.status)}: ${receipt.candidate_count} candidate(s), round ${text(receipt.round_id)}. Rate, accept or reject them in the casting inbox.${holds}`;
    }

    function renderScenes(scenes) {
      scenesBox.replaceChildren();
      const data = scenes && typeof scenes === 'object' ? scenes : null;
      if (!data) return;
      scenesBox.append(make('h4', 'Scene direction', 'cd-subhead'),
        make('p', 'Recommendations bound to the book text. They are not applied to any voice or recording by this panel.', 'cd-muted small'));
      const requested = list(data.controls?.requested || data.requested_controls).map(words);
      const supported = data.controls?.supported ? Object.entries(data.controls.supported)
        .map(([name, capability]) => words(name) + ': ' + (capability.controls
          ? Object.entries(capability.controls).map(([control, detail]) => words(control) + ' ' + words(detail.state || 'unverified')).join('; ')
          : words(capability.state || capability.native_instruction_state || 'unverified')))
        : list(data.supported_controls).map(words);
      if (requested.length || supported.length) {
        scenesBox.append(pair('Controls requested', requested.join(', ') || 'none'), pair('Engine control capabilities', supported.join(' · ') || 'none confirmed'));
      }
      const directions = new Map();
      list(data.directions).forEach(d => {
        const id = text(d?.scene_id);
        if (!directions.has(id)) directions.set(id, []);
        directions.get(id).push(d);
      });
      const scenes_ = list(data.scenes);
      if (!scenes_.length) { scenesBox.append(make('p', 'No scenes were recommended.', 'cd-muted')); return; }
      const rows = make('ol', null, 'cd-scenes');
      scenes_.slice(0, MAX_SCENES).forEach((scene, index) => {
        const id = text(scene.scene_id || scene.id);
        const sceneDirections = directions.get(id) || [];
        const direction = Object.assign({}, scene.direction || {}, sceneDirections[0] || {});
        const item = make('li', null, 'cd-scene'); item.dataset.sceneId = id;
        item.append(make('strong', text(scene.label || scene.title) || `Scene ${index + 1}`));
        const value = field => typeof field === 'object' && field ? (field.value || field.hint || field.target || field.kind || '') : field;
        const bits = [['Emotion', value(direction.emotion || scene.emotion)], ['Intention', value(direction.intention || scene.intention)],
          ['Pace', value(direction.tempo || direction.pace || direction.pace_hint || scene.pace)], ['Boundary', value(direction.semantic_boundary || direction.boundary || scene.boundary)],
          ['Status', direction.status || direction.disposition]].filter(([, v]) => text(v));
        bits.forEach(([label, value]) => item.append(pair(label, words(value))));
        const emphasis = list(direction.emphasis || scene.emphasis).map(e => text(typeof e === 'object' && e ? (e.text || e.span) : e)).filter(Boolean);
        if (emphasis.length) item.append(pair('Emphasis', emphasis.join(', ')));
        rows.append(item);
      });
      scenesBox.append(rows);
      if (scenes_.length > MAX_SCENES) scenesBox.append(make('p', `Showing ${MAX_SCENES} of ${scenes_.length} scenes.`, 'cd-muted small'));
    }

    function render(data) {
      stopPlayer();
      report = data;
      rolesBox.replaceChildren(); scenesBox.replaceChildren();
      if (data.voice_policy === 'cloned_only') {
        rolesBox.append(make('p', 'Cloned voices only: recorded-reference clones are eligible. Synthetic designs and presets are held; existing approvals remain saved.', 'cd-muted'));
      }
      for (const hold of list(data.holds)) {
        rolesBox.append(make('p', 'Held: ' + words(typeof hold === 'object' ? (hold.reason || hold.code || JSON.stringify(hold)) : hold), 'cd-hold'));
      }
      if (data.status === 'needs_analysis') {
        say('This book has not been analyzed yet. Press “Analyze characters and scenes”; it reads the book and saves recommendations only.');
        return;
      }
      const roles = list(data.roles);
      if (!roles.length) { say('Analysis found no speaking characters to recommend voices for.'); return; }
      roles.forEach(role => rolesBox.append(renderRole(role)));
      renderScenes(data.scenes);
      say(`${roles.length} character(s) reviewed. Human review: ${data.human_review || 'pending'}. Recommendations never approve or replace a voice.`);
    }

    async function load() {
      if (busy) return;
      const token = ++epoch;
      if (!base) { say('Open a Voice Lab project first.', 'error'); return; }
      busy = true; open.disabled = analyzeButton.disabled = true;
      say('Loading saved recommendations…');
      try {
        const data = await request('', { credentials: 'same-origin' });
        if (token === epoch) render(data);
      } catch (e) { if (token === epoch) { stopPlayer(); rolesBox.replaceChildren(); scenesBox.replaceChildren(); report = null; say(e.message, 'error'); } }
      finally { busy = false; open.disabled = false; analyzeButton.disabled = false; }
    }

    async function analyze() {
      if (busy) return;
      const token = ++epoch;
      busy = true; open.disabled = analyzeButton.disabled = true;
      say('Analyzing characters and scenes. This reads the book only; it does not generate audio.');
      try {
        const data = await request('/analyze', post({}));
        if (token === epoch) render(data);
      } catch (e) { if (token === epoch) say(e.message, 'error'); }
      finally { busy = false; open.disabled = analyzeButton.disabled = false; }
    }

    async function audition(role, control, note) {
      const key = role.role_id + ':' + report.report_sha256;
      if (auditionBusy.has(role.role_id) || role.protected) return;
      auditionBusy.add(role.role_id); control.disabled = true;
      // One identity per role + evidence: a retry after an uncertain save replays the receipt instead of queuing twice.
      if (!requestIds.has(key)) requestIds.set(key, requestId());
      note.textContent = 'Requesting auditions…';
      const token = epoch;
      try {
        const receipt = await request('/auditions', post({ request_id: requestIds.get(key), role_id: role.role_id, report_sha256: report.report_sha256 }));
        receipts.set(key, receipt);
        note.textContent = receiptText(receipt);
      } catch (e) {
        note.textContent = e.unavailable ? e.message : e.message + ' Nothing was queued.';
        if (token === epoch) control.disabled = false;
      } finally { auditionBusy.delete(role.role_id); }
    }

    analyzeButton.disabled = !base;
    let opened = false;
    open.addEventListener('click', () => {
      const expanded = body.hidden;
      body.hidden = !expanded;
      open.setAttribute('aria-expanded', String(expanded));
      if (!expanded) { stopPlayer(); return; }
      if (!opened || !report) { opened = true; return load(); }
    });
    analyzeButton.addEventListener('click', analyze);
    return { load, analyze, state: () => ({ report, base }) };
  }

  if (typeof document !== 'undefined' && typeof window !== 'undefined') {
    const start = () => mount(document, window);
    if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', start, { once: true }); else start();
  }
  if (typeof module !== 'undefined' && module.exports) module.exports = { mount, safeAudio, explain };
})();
