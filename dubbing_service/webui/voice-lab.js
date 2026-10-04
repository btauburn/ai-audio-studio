/* One Studio review surface; one media element; no synthesis side effects. */
(function (root, factory) {
  'use strict';
  const api = factory();
  if (typeof module === 'object' && module.exports) module.exports = api;
  else { root.VoiceLab = api; const start = () => api.mount(document, window).catch(error => { const box = document.getElementById('error'); box.hidden = false; box.textContent = error.message; }); if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', start, { once: true }); else start(); }
})(typeof globalThis !== 'undefined' ? globalThis : this, function () {
  'use strict';
  // Design is an explicit user action, never part of any playback/review handler.
  const REVIEW_ACTIONS = new Set(['preferences', 'position', 'match', 'more-like', 'assign', 'automation', 'issues', 'design', 'audition', 'library', 'library/publish', 'library/import', 'casting', 'casting/book', 'casting/production/resolve', 'casting/production/skip', 'casting/production/restore', 'casting/production/policy', 'casting/prepare', 'casting/feedback', 'casting/round', 'casting/render', 'casting/resume']);
  const finite = value => Number.isFinite(Number(value)) ? Number(value) : 0;
  const time = seconds => { const n = Math.max(0, Math.floor(finite(seconds))); return n >= 3600 ? `${Math.floor(n / 3600)}:${String(Math.floor(n / 60) % 60).padStart(2, '0')}:${String(n % 60).padStart(2, '0')}` : `${Math.floor(n / 60)}:${String(n % 60).padStart(2, '0')}`; };
  const typing = target => !!(target && (target.isContentEditable || /^(INPUT|TEXTAREA|SELECT|BUTTON|A|SUMMARY)$/.test(target.tagName) || (target.closest && target.closest('[contenteditable="true"], [role="textbox"]'))));
  function shortcut(event) {
    if (typeof document !== 'undefined' && document.documentElement?.dataset?.runtimeComparison === 'playing') return null;
    if (event.altKey || event.ctrlKey || event.metaKey || event.repeat || typing(event.target)) return null;
    if (event.code === 'Space' || event.key === ' ') return 'play';
    if (event.key.toLowerCase() === 'f') return 'favorite';
    if (/^[1-5]$/.test(event.key)) return Number(event.key);
    if (event.key === 'ArrowLeft') return 'previous';
    if (event.key === 'ArrowRight') return 'next';
    return null;
  }
  function filtered(voices, filter) {
    return voices.filter(v => filter === 'favorites' ? v.favorite : filter === 'rated' ? v.rating != null : filter === 'new' ? !v.favorite && v.rating == null : true);
  }
  function roleAt(map, seconds) { return (map || []).find(row => finite(row.start_seconds) <= seconds && seconds < finite(row.end_seconds)) || null; }
  function voiceForRow(voices, row) {
    if (!row?.role_id) return null;
    return (voices || []).find(v => v.role_id === row.role_id &&
      (!row.voice_id || v.voice_id === row.voice_id) &&
      (!row.voice_version || v.version === row.voice_version)) || null;
  }
  function safeUrl(value, location, external = false) {
    try { const url = new URL(value, location.href); return ['http:', 'https:'].includes(url.protocol) && (external || url.origin === location.origin) ? url.href : null; } catch (_) { return null; }
  }
  function matchedComparisons(excerpt) {
    if (!excerpt) return [];
    const ids = (excerpt.map || []).map(row => row.utterance_id);
    if (!ids.length) return [];
    return (excerpt.comparisons || []).filter(row => {
      const other = row.utterance_ids || (row.map || []).map(item => item.utterance_id);
      return row.audio_url && row.audio_sha256 && typeof row.loudness_lufs === 'number' && Number.isFinite(row.loudness_lufs) && JSON.stringify(ids) === JSON.stringify(other);
    });
  }
  function comparisonGain(item, comparisons) {
    const target = Math.min(...comparisons.map(row => row.loudness_lufs));
    return Math.min(1, Math.max(0, Math.pow(10, (target - item.loudness_lufs) / 20)));
  }
  function castingDraftKey(project, role, candidate) {
    return `voice-lab-casting-v1:${project}:${role.role_id}:${role.round_revision}:${candidate.candidate_id}`;
  }
  function castingDirectionText(direction = {}) {
    return ['requested', 'applied', 'unsupported', 'unverified', 'retained']
      .flatMap(key => (direction[key] || []).map(value => ({ key, value })))
      .map(item => `${item.key}: ${item.value}`);
  }
  class ReviewClient {
    constructor(base, fetcher, { readTimeoutMs = 20000, clock = globalThis } = {}) { this.base = base; this.fetcher = fetcher; this.readTimeoutMs = readTimeoutMs; this.clock = clock; }
    async request(action, body) {
      const auditionStatus = /^auditions\/[A-Za-z0-9_-]{1,100}$/.test(action || '');
      if (action && !REVIEW_ACTIONS.has(action) && !auditionStatus) throw new Error('Unsupported review action');
      const write = action && action !== 'library' && action !== 'casting' && action !== 'casting/book' && !auditionStatus;
      // Bound reads through both headers and JSON body. Writes retain their existing uncertain-save/idempotency handling.
      const controller = !write && typeof AbortController === 'function' ? new AbortController() : null;
      const read = { credentials: 'same-origin', ...(controller ? { signal: controller.signal } : {}) };
      const perform = async () => {
        const response = await this.fetcher(this.base + (action ? '/' + action : ''), write ? { method: 'POST', headers: { 'Content-Type': 'application/json', 'X-Review-Request': '1' }, body: JSON.stringify(body), credentials: 'same-origin' } : read);
        let data; try { data = await response.json(); } catch (_) { throw new Error(`Review service returned ${response.status}. Please retry.`); }
        if (!response.ok) {
          const error = new Error(typeof data.detail === 'string' ? data.detail : (data.message || `Save failed (${response.status}). Your change was not saved.`));
          error.status = response.status; error.body = data; throw error;
        }
        return data;
      };
      if (write) return perform();
      let timer;
      const deadline = new Promise((resolve, reject) => {
        timer = this.clock.setTimeout(() => {
          const error = new Error('Voice Lab took too long to respond. Please retry.'); error.code = 'REVIEW_READ_TIMEOUT';
          reject(error); controller?.abort();
        }, this.readTimeoutMs);
      });
      try { return await Promise.race([perform(), deadline]); }
      finally { this.clock.clearTimeout(timer); }
    }
  }
  class Player {
    constructor(media) {
      this.media = media; this.current = null; this.pendingSeek = 0; this.generation = 0;
      media.addEventListener('loadedmetadata', () => { this.seek(this.pendingSeek); });
    }
    async load(recording, { seconds = 0, autoplay = false, gain = 1 } = {}) {
      const generation = ++this.generation;
      this.media.pause(); this.current = recording; this.pendingSeek = Math.max(0, finite(seconds));
      this.media.playbackRate = 1;
      this.media.volume = Math.min(1, Math.max(0, finite(gain)));
      if (!recording || !recording.audio_url || !recording.audio_sha256) { this.media.removeAttribute('src'); this.media.load(); return; }
      this.media.src = recording.audio_url; this.media.load();
      if (autoplay) { try { await this.media.play(); } catch (error) { if (generation === this.generation) throw error; } }
    }
    seek(seconds) { this.pendingSeek = Math.max(0, Math.min(finite(seconds), Number.isFinite(this.media.duration) ? this.media.duration : Infinity)); if (this.media.readyState > 0) this.media.currentTime = this.pendingSeek; }
    async toggle() { if (!this.current || !this.current.audio_url) return; if (this.media.paused) await this.media.play(); else this.media.pause(); }
  }
  function shortlistMatches(shortlists) {
    return (Array.isArray(shortlists) ? shortlists.map((row, i) => [row.role_id || 'general', row]) : Object.entries(shortlists || {}))
      .flatMap(([slot, row]) => (Array.isArray(row) ? row : (row.matches || [row])).map(match => ({ ...match,
        match_role_id: match.match_role_id || row.role_id || (slot === 'general' ? null : slot) })));
  }
  // ---- Shared card vocabulary (module scope: casting deck and voice library) ----
  // Category is explicit presentation metadata, never guessed from a
  // portrait: absent metadata stays neutral rather than inventing a family.
  const FAMILY_PATTERNS = [
    // The casting pipeline records presentation as "feminine" / "masculine" /
    // "androgynous"; only the first two map to a category family.
    ['female', /female|feminine|woman|girl(?!friend)/i],
    ['male', /\bmale\b|\bmasculine\b|\bman\b|\bboy\b/i],
    ['robot', /robot|android|synthetic construct|machine intelligence/i],
    ['alien', /\balien\b|xenomorph|extraterrestrial|non-human creature/i],
  ];
  const castingFamily = candidate => {
    const dimensions = [...(candidate?.supported_traits || []), ...(candidate?.inferred_traits || [])];
    for (const row of dimensions) {
      if (!/presentation|gender|species|kind/i.test(row?.dimension || '')) continue;
      const value = String(row?.value || '');
      const match = FAMILY_PATTERNS.find(([, pattern]) => pattern.test(value));
      if (match) return match[0];
    }
    return 'unknown';
  };
  // Stable accessory nicknames distinguish same-character candidates. Keyed
  // to the canonical candidate ID, never to list position, so a refreshed
  // sort or a later round never renames a voice the human already knows.
  const NICKNAMES = ['Red Hat', 'Blue Scarf', 'Round Glasses', 'Green Cap', 'Silver Pin', 'Amber Badge',
    'Gray Coat', 'Violet Ribbon', 'Copper Buckle', 'Ivory Brooch', 'Slate Hood', 'Teal Cravat'];
  const castingNickname = candidateId => {
    const id = String(candidateId || '');
    let hash = 0; for (let i = 0; i < id.length; i++) hash = (hash * 31 + id.charCodeAt(i)) >>> 0;
    return NICKNAMES[hash % NICKNAMES.length];
  };
  // Only the current-context rating drives the finish. A historical rating
  // from another project/character never activates shiny or holo here.
  // A character's display name: the pipeline's own labels are tidied for people.
  const castingRoleLabel = name => {
    const text = String(name || 'Character');
    const library = text.match(/^Library voice\s*\((.+)\)$/i);
    return /^narrator$/i.test(text) ? 'Narrator' : library ? library[1] : text;
  };
  const plural = (count, one, many = one + 's') => `${count} ${count === 1 ? one : many}`;
  const grouped = value => Number(value || 0).toLocaleString('en-US');
  // Time left, from the pace the worker has actually measured. `null` means it has not
  // recorded enough lines to know, and the page says so instead of inventing a number.
  function timeLeft(seconds) {
    if (!Number.isFinite(seconds)) return 'Estimating time left…';
    if (seconds < 45) return 'less than a minute left';
    const minutes = Math.max(1, Math.round(seconds / 60));
    if (minutes < 90) return `about ${plural(minutes, 'minute')} left`;
    const rest = minutes % 60;
    return `about ${Math.floor(minutes / 60)} h${rest ? ` ${rest} min` : ''} left`;
  }
  // What the human is looking at, in the order they care about it: voices waiting,
  // the book being recorded, then the finished book. Pure presentation of real state.
  function bookSummary(status) {
    const production = status?.production;
    const waiting = (status?.voices?.waiting || []).length;
    switch (status?.state) {
      case 'voices_waiting': return { title: waiting === 1 ? '1 voice is waiting for you' : `${waiting} voices are waiting for you`,
        detail: 'Listen, rate and choose. Your book starts recording as soon as every character has a voice.' };
      case 'processing': return { title: 'Recording your book',
        detail: `${plural(production.chapters_ready, 'chapter')} of ${production.chapters_total} ready to listen to. Each finished chapter is sent to your phone as soon as it is done.` };
      case 'finished': return { title: 'Your book is ready', detail: `${plural(production.chapters_total, 'chapter')} recorded. Play any chapter below.` };
      case 'blocked': return { title: 'Your book is waiting on you', detail: 'Every voice is chosen, but some lines have no speaker yet.' };
      case 'waiting':
        if (status.failed_line) return { title: 'Your book stopped at one line', detail: 'Skip that line or try it again, and recording carries on from there.' };
        return { title: 'Your book is waiting to start', detail: production?.reason || 'It starts as soon as the audio worker is free.' };
      case 'casting': return { title: 'Finding voices for your characters', detail: 'Auditions are being recorded. You will be notified when they are ready to hear.' };
      case 'starting': return { title: 'Starting your book', detail: 'Every voice is chosen. Recording begins in a moment.' };
      default: return null;
    }
  }
  const libraryFamily = voice => { const text = (voice?.tags || []).join(' '); const match = FAMILY_PATTERNS.find(([, pattern]) => pattern.test(text)); return match ? match[0] : 'unknown'; };
  const castingTier = candidate => candidate?.rating === 5 ? 'holo' : candidate?.rating === 4 ? 'shiny' : 'ordinary';
  const FAMILY_GLYPH = { female: '♀', male: '♂', robot: '⚙', alien: '✦', unknown: '?' };
  // Directing chips: a tap adds or removes the phrase from the direction the
  // human is composing; nothing is sent until they choose Save or Recast.
  const QUICK_DIRECTIONS = ['Deeper', 'Warmer', 'More expressive'];
  const DIRECTION_CHIPS = ['More British', 'Deeper', 'Less nasal', 'Warmer', 'Older', 'Younger', 'More expressive', 'Less robotic', 'Slower', 'Faster', 'More theatrical'];
  // Human-readable sound profile, from the generation parameters actually sent
  // to synthesis. Nothing here is a measurement claim.
  function castingSoundProfile(candidate) {
    const p = candidate?.conditioning?.parameters || {}; const out = [];
    if (p.pitch_target_hz) out.push(`Around ${Math.round(p.pitch_target_hz)} Hz`);
    if (p.rate_words_per_minute) out.push(`${Math.round(p.rate_words_per_minute)} words per minute`);
    if (p.resonance) out.push(`${capitalize(p.resonance)} resonance`);
    if (p.vocal_weight) out.push(`${capitalize(p.vocal_weight)} weight`);
    if (p.age_impression) out.push(`${capitalize(p.age_impression)} impression`);
    if (Number(p.breathiness) >= 0.2) out.push('Breathy');
    if (Number(p.rasp) >= 0.25) out.push('Raspy');
    return out;
  }
  const FAMILY_CODE = { male: 'M', female: 'F', robot: 'R', alien: 'A', unknown: 'N' };
  const FAMILY_LABEL = { male: 'Male family', female: 'Female family', robot: 'Robot family', alien: 'Alien family', unknown: 'Neutral family' };
  // Sibling candidates for one character must never share a nickname. Names
  // stay a pure function of the candidate ID; only an actual collision moves
  // the later ID (in sorted order) to the next free slot, so a voice keeps
  // its name across refreshes, rounds and screens.
  function castingNicknames(role) {
    const ids = (role?.candidates || []).map(row => row.candidate_id).filter(Boolean).sort();
    const taken = new Set(); const names = new Map();
    for (const id of ids) {
      let slot = NICKNAMES.indexOf(castingNickname(id));
      for (let step = 0; step < NICKNAMES.length && taken.has(slot); step++) slot = (slot + 1) % NICKNAMES.length;
      taken.add(slot); names.set(id, NICKNAMES[slot]);
    }
    return names;
  }
  // Short stable card code: family letter + the tail of the canonical ID.
  const castingCode = (candidateId, family) => `${FAMILY_CODE[family] || 'N'}-${String(candidateId || '').replace(/[^0-9a-z]/gi, '').slice(-4).toUpperCase().padStart(2, '0')}`;
  const capitalize = text => { const value = String(text || '').trim(); return value ? value[0].toUpperCase() + value.slice(1) : ''; };
  // Card-front inspiration: performer + performance and a short traits reason,
  // drawn only from the candidate's own recorded anchor / prior evidence.
  function castingInspiration(candidate) {
    const anchor = candidate?.anchor || {}; const inspiration = candidate?.inspiration || {}; const prior = candidate?.prior_evidence || null;
    const performer = inspiration.performer || anchor.performer || prior?.inspiration?.performer || '';
    const performance = inspiration.performance || anchor.performance || prior?.inspiration?.performance || '';
    const traits = (anchor.matched_traits || []).length ? anchor.matched_traits : (prior?.match_reasons || []).filter(row => !/^previously inspired/.test(row));
    return { performer, performance, reason: traits.map(capitalize).slice(0, 4).join(' · '), traits };
  }
  // ---- Character portraits ------------------------------------------------
  // Original synthetic characters, drawn from application metadata only: the
  // character (role) seeds the face so every candidate for that character
  // shares one identity, the category family picks the figure, and the stable
  // nickname picks the accessory that tells sibling voices apart.
  const SKIN = ['#f3cfae', '#e9b892', '#d99f74', '#c68a5b', '#a9714a', '#8a5a3b'];
  const HAIR = ['#2a1912', '#4b2d1b', '#6a4526', '#1d1d24', '#8d5c2d', '#b4682a'];
  const seedOf = value => { let h = 7; for (const ch of String(value ?? '')) h = (h * 33 + ch.charCodeAt(0)) >>> 0; return h; };
  const EYES = { human: [121, 179, 160], robot: [128, 172, 144], alien: [116, 184, 150] };
  function portraitAccessory(name, family, skin) {
    const eyes = family === 'robot' ? EYES.robot : family === 'alien' ? EYES.alien : EYES.human;
    const lift = family === 'alien' ? -34 : family === 'robot' ? -2 : 0;
    const cap = (a, b) => `<g transform="translate(0 ${lift})"><path d="M78 124C74 52 226 52 222 124C196 110 104 110 78 124Z" fill="${a}"/><path d="M72 122Q152 100 250 130Q238 150 152 137Q104 132 72 140Z" fill="${b}"/><circle cx="150" cy="63" r="6" fill="${b}"/><path d="M96 92C120 74 178 72 204 92" stroke="#fff" stroke-opacity=".28" stroke-width="5" fill="none" stroke-linecap="round"/></g>`;
    const map = {
      'Red Hat': { front: cap('#dc2f2f', '#a51c1c') },
      'Green Cap': { front: cap('#2fae5f', '#1c7c41') },
      'Blue Scarf': { front: '<path d="M96 240Q150 274 204 240L210 268Q150 300 90 268Z" fill="#2f74e0"/><path d="M168 272L204 266L214 336L176 326Z" fill="#1f56b8"/><path d="M100 252Q150 282 200 252M104 262Q150 292 196 262" stroke="#9cc4ff" stroke-width="3" fill="none" opacity=".7"/>' },
      'Round Glasses': { front: `<g fill="#bfe3ff" fill-opacity=".14" stroke="#1e2028" stroke-width="5"><circle cx="${eyes[0]}" cy="${eyes[2]}" r="24"/><circle cx="${eyes[1]}" cy="${eyes[2]}" r="24"/></g><path d="M${eyes[0] + 24} ${eyes[2]}H${eyes[1] - 24}M${eyes[0] - 24} ${eyes[2]}L84 ${eyes[2] - 4}M${eyes[1] + 24} ${eyes[2]}L216 ${eyes[2] - 4}" stroke="#1e2028" stroke-width="5" fill="none" stroke-linecap="round"/>` },
      'Silver Pin': { front: '<path d="M96 292l7 14 15 2-11 10 3 15-14-8-14 8 3-15-11-10 15-2z" fill="#dfe6ef" stroke="#8b98a9" stroke-width="2"/>' },
      'Amber Badge': { front: '<circle cx="206" cy="298" r="17" fill="#f5a524" stroke="#ffd27a" stroke-width="3"/><circle cx="206" cy="298" r="8" fill="#b4720d"/>' },
      'Gray Coat': { front: '<path d="M10 340C14 282 56 262 116 250L152 304L118 340Z" fill="#767e8c"/><path d="M290 340C286 282 244 262 184 250L148 304L182 340Z" fill="#666e7c"/><path d="M116 250L152 304L136 250Z" fill="#9aa3b1"/>' },
      'Violet Ribbon': { front: '<g transform="translate(98 94) rotate(-24)"><path d="M0 0L-30-18V18Z" fill="#8b5cf6"/><path d="M0 0L30-18V18Z" fill="#7c3aed"/><circle r="8" fill="#c4b5fd"/></g>' },
      'Copper Buckle': { front: '<path d="M52 262L214 340H262L96 250Z" fill="#5a3a26"/><rect x="132" y="284" width="34" height="30" rx="6" transform="rotate(28 149 299)" fill="#d98b4e" stroke="#ffc48a" stroke-width="3"/>' },
      'Ivory Brooch': { front: '<circle cx="150" cy="268" r="14" fill="#f5efe0" stroke="#d6c9a7" stroke-width="3"/><ellipse cx="150" cy="268" rx="6" ry="8" fill="#9db7d6"/>' },
      'Slate Hood': { back: '<path d="M30 340C22 246 40 150 74 104C104 58 196 58 226 104C260 150 278 246 270 340Z" fill="#4a5568"/><path d="M62 340C56 260 68 170 96 124" stroke="#6b778c" stroke-width="4" fill="none" opacity=".6"/>' },
      'Teal Cravat': { front: '<path d="M150 252L124 260L150 312L176 260Z" fill="#19b6a4"/><path d="M150 252L132 258L150 270L168 258Z" fill="#0d8f80"/>' },
    };
    return map[name] || {};
  }
  function portraitSvg(family, nickname, seed) {
    const h = seedOf(seed); const skin = SKIN[h % SKIN.length]; const hair = HAIR[(h >>> 4) % HAIR.length];
    const acc = portraitAccessory(nickname, family, skin);
    const shirt = { male: '#262a36', female: '#2b2634', robot: '#5e7391', alien: '#12322c', unknown: '#2a303c' }[family] || '#2a303c';
    const torso = `<path d="M12 340C16 282 58 262 118 250H182C242 262 284 282 288 340Z" fill="${shirt}"/>`;
    let body;
    if (family === 'robot') {
      body = `<defs><linearGradient id="g" x1="0" y1="0" x2="1" y2="1"><stop offset="0" stop-color="#e3edf9"/><stop offset="1" stop-color="#7f95b6"/></linearGradient></defs>
<path d="M12 340C16 282 58 262 118 250H182C242 262 284 282 288 340Z" fill="url(#g)"/><rect x="118" y="292" width="64" height="30" rx="10" fill="#0b1626" stroke="#3ed4ff" stroke-width="3"/><circle cx="150" cy="307" r="7" fill="#3ed4ff"/>
<rect x="132" y="206" width="36" height="46" rx="8" fill="#5e7391"/>
<rect x="56" y="126" width="24" height="50" rx="9" fill="#5e7391"/><rect x="220" y="126" width="24" height="50" rx="9" fill="#5e7391"/>
<rect x="74" y="76" width="152" height="136" rx="40" fill="url(#g)"/><path d="M150 76V48" stroke="#8ea5c6" stroke-width="6"/><circle cx="150" cy="42" r="10" fill="#3ed4ff"/><circle cx="147" cy="39" r="3" fill="#fff"/>
<rect x="94" y="108" width="112" height="82" rx="28" fill="#0b1626" stroke="#3ed4ff" stroke-width="3"/>
<circle cx="128" cy="144" r="13" fill="#3ed4ff"/><circle cx="172" cy="144" r="13" fill="#3ed4ff"/><circle cx="132" cy="139" r="4" fill="#fff"/><circle cx="176" cy="139" r="4" fill="#fff"/>
<path d="M126 172Q150 186 174 172" stroke="#3ed4ff" stroke-width="5" fill="none" stroke-linecap="round"/>`;
    } else if (family === 'alien') {
      body = `<defs><linearGradient id="g" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="#8ff5c8"/><stop offset="1" stop-color="#2fb886"/></linearGradient></defs>
${torso}<path d="M110 256L150 300L190 256" stroke="#22e69a" stroke-width="5" fill="none"/>
<rect x="134" y="214" width="32" height="44" rx="12" fill="#2fb886"/>
<ellipse cx="150" cy="140" rx="64" ry="92" fill="url(#g)"/>
<path d="M90 134C96 112 132 122 140 156C120 172 92 162 90 134Z" fill="#08201a"/><path d="M210 134C204 112 168 122 160 156C180 172 208 162 210 134Z" fill="#08201a"/>
<circle cx="112" cy="140" r="5" fill="#fff" opacity=".85"/><circle cx="188" cy="140" r="5" fill="#fff" opacity=".85"/>
<circle cx="144" cy="188" r="2.6" fill="#0b6b4c"/><circle cx="156" cy="188" r="2.6" fill="#0b6b4c"/><path d="M136 206Q150 214 164 206" stroke="#0b6b4c" stroke-width="4" fill="none" stroke-linecap="round"/>`;
    } else {
      const long = family === 'female';
      const back = long ? `<path d="M68 154C52 66 108 36 152 36C198 36 250 66 232 154C238 206 242 246 226 276H74C58 246 62 206 68 154Z" fill="${hair}"/>` : '';
      const front = long
        ? `<path d="M78 136C86 84 128 62 158 66C192 66 220 90 222 136C202 116 178 104 158 92C138 112 106 120 78 136Z" fill="${hair}"/><path d="M104 108C126 96 150 92 170 92" stroke="#fff" stroke-opacity=".16" stroke-width="5" fill="none" stroke-linecap="round"/>`
        : family === 'male'
          ? `<path d="M78 140C68 84 108 52 152 52C198 52 234 84 222 140C212 112 196 96 176 92C160 106 132 108 110 98C92 108 82 122 78 140Z" fill="${hair}"/><path d="M104 84C128 68 172 66 196 82" stroke="#fff" stroke-opacity=".16" stroke-width="5" fill="none" stroke-linecap="round"/>`
          : `<path d="M74 148C60 84 104 48 152 48C202 48 244 84 226 148C220 116 200 98 178 92C160 108 130 110 108 100C90 110 80 126 74 148Z" fill="${hair}"/><path d="M104 82C128 66 172 64 198 80" stroke="#fff" stroke-opacity=".16" stroke-width="5" fill="none" stroke-linecap="round"/>`;
      const lashes = long ? '<path d="M108 148l-9-4M112 144l-6-8M190 144l6-8M192 148l9-4" stroke="#231512" stroke-width="3" stroke-linecap="round"/>' : '';
      const mouth = long ? '<path d="M132 192Q150 208 168 192Q150 198 132 192Z" fill="#d1566a" stroke="#b03a52" stroke-width="3" stroke-linejoin="round"/>' : '<path d="M128 192Q150 214 172 192" stroke="#7b2f22" stroke-width="5" fill="none" stroke-linecap="round"/>';
      body = `${back}${acc.back || ''}${torso}<path d="M120 252L150 288L180 252Z" fill="${skin}"/><path d="M124 210h52v42c-14 12-38 12-52 0z" fill="${skin}"/><path d="M124 228c14 14 38 14 52 0v-16h-52z" fill="#000" opacity=".13"/>
<ellipse cx="80" cy="160" rx="12" ry="19" fill="${skin}"/><ellipse cx="220" cy="160" rx="12" ry="19" fill="${skin}"/><ellipse cx="150" cy="150" rx="71" ry="79" fill="${skin}"/>
<circle cx="102" cy="184" r="14" fill="#f27a8b" opacity=".22"/><circle cx="198" cy="184" r="14" fill="#f27a8b" opacity=".22"/>
<ellipse cx="121" cy="160" rx="9.5" ry="13" fill="#231512"/><ellipse cx="179" cy="160" rx="9.5" ry="13" fill="#231512"/><circle cx="124.5" cy="154" r="3.6" fill="#fff"/><circle cx="182.5" cy="154" r="3.6" fill="#fff"/>${lashes}
<path d="M104 138Q121 127 138 137M162 137Q179 127 196 138" stroke="${hair}" stroke-width="6" fill="none" stroke-linecap="round"/>
<path d="M150 166q-7 15 0 19q6 2 9-3" stroke="#000" stroke-opacity=".24" stroke-width="3" fill="none" stroke-linecap="round"/>${mouth}${front}`;
      return `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 300 340">${body}${acc.front || ''}</svg>`;
    }
    return `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 300 340">${acc.back || ''}${body}${acc.front || ''}</svg>`;
  }
  // The private review origin's CSP forbids inline style attributes, but CSSOM
  // writes are allowed. Fall back to the attribute only where no CSSOM exists.
  function setStyleVars(element, vars) {
    if (element.style && typeof element.style.setProperty === 'function') { for (const [name, value] of Object.entries(vars)) element.style.setProperty(name, value); return; }
    element.setAttribute('style', Object.entries(vars).map(([name, value]) => `${name}:${value}`).join(';'));
  }
  const portraitCache = new Map();
  function castingPortraitUri(family, nickname, seed) {
    const key = `${family}|${nickname}|${seed}`;
    if (!portraitCache.has(key)) portraitCache.set(key, `data:image/svg+xml;charset=utf-8,${encodeURIComponent(portraitSvg(family, nickname, seed))}`);
    return portraitCache.get(key);
  }
  // A sonic signature for the card's transport strip: bar heights are seeded
  // from the audio hash, so it is stable per recording. It is decorative; the
  // filled progress is the real playback position.
  const waveCache = new Map();
  function castingWaveUri(seedValue) {
    if (!waveCache.has(seedValue)) {
      let h = seedOf(seedValue) || 1; const bars = [];
      for (let i = 0; i < 44; i++) {
        h = (h * 1664525 + 1013904223) >>> 0;
        const envelope = 0.42 + 0.58 * Math.sin((i / 43) * Math.PI);
        const height = Math.max(4, Math.round(6 + ((h >>> 8) % 100) / 100 * 26 * envelope + 6 * envelope));
        bars.push(`<rect x="${i * 5 + 1}" y="${(40 - height) / 2}" width="3" height="${height}" rx="1.5"/>`);
      }
      waveCache.set(seedValue, `data:image/svg+xml;charset=utf-8,${encodeURIComponent(`<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 220 40" preserveAspectRatio="none">${bars.join('')}</svg>`)}`);
    }
    return waveCache.get(seedValue);
  }

  async function mountLanding(doc, win, { $, node, button, run, error }) {
    const player = new Player($('media')); let library = []; let catalogBooks = []; let recordings = []; let selectedOutput = null; let book = null; let request = null;
    let file = null; let importId = null; let importMode = 'autocast'; let chapterChoices = []; let narratorChoice = ''; let busy = false; let epoch = 0; let pollTimer = null; let pollCount = 0; let pollFailures = 0; let pollInFlight = null; let pageActive = true; let catalogGeneration = 0;
    const recordingsFilter = new URLSearchParams(win.location.search).get('recordings');
    const encode = encodeURIComponent;
    async function jsonFetch(url, options = {}) { const response = await win.fetch(url, { credentials: 'same-origin', ...options }); const result = await response.json(); if (!response.ok) throw new Error(typeof result.detail === 'string' ? result.detail : `Studio request failed (${response.status}).`); return result; }
    const isPrivate = win.location.pathname.startsWith('/voice-lab');
    const api = (path = '', options) => jsonFetch((isPrivate ? '/api/book-creator' : '/api/v1/book-creator') + path, options);
    const post = body => ({ method: 'POST', headers: { 'Content-Type': 'application/json', 'X-Review-Request': '1' }, body: JSON.stringify(body) });
    const versions = () => library.flatMap(voice => (voice.versions || []).map(version => ({ ...version, library_voice_id: version.library_voice_id || voice.library_voice_id, name: version.name || voice.name })));
    const voiceValue = v => JSON.stringify({ library_voice_id: v.library_voice_id, voice_version: v.version });
    const selectedChapters = () => chapterChoices.filter(row => row.input.checked).map(row => row.id);
    const active = () => request && ['queued', 'running'].includes(request.status);
    function updateControls() { $('import-book').disabled = !file || busy; $('create-mp3').disabled = !book || !selectedChapters().length || !$('book-narrator').value || busy || !!active(); }
    function updatePlayer() { const media = player.media; $('play').textContent = media.paused ? 'Play' : 'Pause'; $('elapsed').textContent = time(media.currentTime); const duration = Number.isFinite(media.duration) ? media.duration : finite(player.current?.duration_seconds); $('duration').textContent = time(duration); $('seek').max = duration; $('seek').value = media.currentTime || 0; }
    async function play(recording, autoplay = true) { const url = safeUrl(recording.audio_url, win.location); if (!recording.audio_url || !url || !recording.audio_sha256) throw new Error('No verified production sample is available for this version.'); const loading = player.load({ ...recording, audio_url: url }, { autoplay }); $('now-playing').textContent = recording.name || recording.label; $('play').disabled = $('seek').disabled = false; await loading; updatePlayer(); }
    $('page-title').textContent = 'Voice Library'; $('project-name').textContent = 'Choose a voice. Open a book. Create a narrator reading.';
    $('library-home').hidden = true; $('project-picker').hidden = false; $('library-landing').hidden = false;
    $('previous').disabled = $('next').disabled = true; $('player-feedback').hidden = true;
    $('play').onclick = () => run(() => player.toggle()); $('seek').oninput = () => { player.seek(Number($('seek').value)); updatePlayer(); };
    for (const event of ['loadedmetadata', 'timeupdate', 'play', 'pause', 'ended']) player.media.addEventListener(event, updatePlayer);
    player.media.addEventListener('error', () => error('The recording could not be played. Its saved version is unchanged.'));
    doc.addEventListener('keydown', event => { if (shortcut(event) === 'play') { event.preventDefault(); void run(() => player.toggle()); } });
    const outputKey = row => `${row.project_id}:${row.export_id}`;
    function outputControls(output, { finished = false } = {}) {
      const controls = node('div', null, 'button-row');
      const listen = button('Play MP3', () => { if (finished) { selectedOutput = outputKey(output); renderFinished(); } return play({ ...output, key: 'book-' + outputKey(output), name: output.label || output.project_name || book?.name }); });
      listen.disabled = !output.audio_url || !safeUrl(output.audio_url, win.location) || !output.audio_sha256; controls.append(listen);
      const rawUrl = output.download_url || output.audio_url; const url = rawUrl && safeUrl(rawUrl, win.location);
      if (url && output.audio_sha256) { const download = node('a', 'Download MP3', 'button secondary'); download.href = url; download.setAttribute('download', ''); controls.append(download); }
      const review = output.review_url && safeUrl(output.review_url, win.location);
      if (review) { const link = node('a', 'Open & review', 'button secondary'); link.href = review; controls.append(link); }
      return controls;
    }
    function renderFinished() {
      $('finished-recordings').replaceChildren(); const query = $('recording-search').value.trim().toLocaleLowerCase();
      const scoped = recordingsFilter ? recordings.filter(row => row.project_id === recordingsFilter) : recordings;
      const rows = scoped.filter(row => [row.project_name, row.label].some(value => String(value || '').toLocaleLowerCase().includes(query)));
      const scopedName = recordingsFilter && scoped[0]?.project_name;
      $('recordings-status').textContent = recordingsFilter
        ? (scoped.length ? `${rows.length} of ${scoped.length} finished recording${scoped.length === 1 ? '' : 's'} for ${scopedName || 'this book'}.`
          : 'No finished chapters yet for this book. Completed chapters appear here as soon as they render.')
        : recordings.length ? `${rows.length} of ${recordings.length} finished recordings${query && !rows.length ? ' · No matches. Try another book title.' : ''}` : 'No finished recordings yet. Completed chapters and combined books appear here.';
      const groups = new Map();
      for (const row of rows) { if (!groups.has(row.project_id)) groups.set(row.project_id, []); groups.get(row.project_id).push(row); }
      for (const [projectId, outputs] of groups) {
        const group = node('section', null, 'recording-book'); group.dataset.projectId = projectId;
        group.append(node('h3', outputs[0].project_name || projectId));
        for (const output of outputs) {
          const card = node('article', null, 'recording-card'); card.dataset.exportId = output.export_id; card.dataset.audioSha256 = output.audio_sha256;
          card.classList.toggle('selected', selectedOutput === outputKey(output)); if (selectedOutput === outputKey(output)) card.setAttribute('aria-current', 'true');
          card.append(node('h4', output.label || 'Finished recording'), node('p', `${time(output.duration_seconds)} · Technical QA: ${output.technical_qa || 'pending'} · Human review: ${output.human_review || 'pending'}`, 'muted small'), outputControls(output, { finished: true })); group.append(card);
        }
        $('finished-recordings').append(group);
      }
    }
    async function refreshRecordings() { const generation = ++catalogGeneration; try { const state = await jsonFetch(isPrivate ? '/api/voice-library/recordings' : '/api/v1/voice-library/recordings'); if (generation !== catalogGeneration) return; recordings = Array.isArray(state.recordings) ? state.recordings : []; renderFinished(); } catch (e) { if (generation === catalogGeneration) $('recordings-status').textContent = 'Finished recordings could not be refreshed. Your audio remains saved. Use Refresh recordings to retry.'; throw e; } }
    $('recording-search').oninput = renderFinished; $('refresh-recordings').onclick = () => run(refreshRecordings);
    function renderLibrary() {
      $('global-voices').replaceChildren(); const items = versions();
      $('global-library-status').textContent = items.length ? `${items.length} saved production voice ${items.length === 1 ? 'version' : 'versions'}.` : 'No saved production voices yet. Open an existing book, preview a voice, then save its version to the library.';
      const KIND_LABEL = { provider_builtin: 'Built-in voice', original_inspired: 'Original, performer-inspired', reference_clone: 'Reference voice', designed: 'Designed voice' };
      for (const v of items) {
        // A saved voice is a reusable card in the collection: the same portrait
        // language as the casting deck, with only the fields the library records.
        const family = libraryFamily(v); const nickname = castingNickname(v.library_voice_id);
        const card = node('article', null, 'cast-card library-card casting-candidate'); card.dataset.family = family; card.dataset.tier = 'ordinary'; card.dataset.score = '0'; card.dataset.libraryVoiceId = v.library_voice_id;
        const stage = node('div', null, 'casting-stage library-stage');
        const image = node('img', null, 'casting-portrait-img'); image.src = castingPortraitUri(family, nickname, v.library_voice_id); image.alt = `${nickname} portrait for ${v.name}`;
        const code = node('span', castingCode(v.library_voice_id, family), 'casting-code'); code.title = `Library voice ${v.library_voice_id}`;
        const plate = node('div', null, 'casting-nameplate'); const titleRow = node('div', null, 'casting-titlerow');
        titleRow.append(node('h3', v.name), node('span', nickname, 'casting-option-label')); plate.append(titleRow);
        stage.append(image, code, plate); card.append(stage);
        const facts = node('div', null, 'casting-profile library-facts');
        facts.append(node('span', KIND_LABEL[v.kind] || capitalize(String(v.kind || 'saved voice').replaceAll('_', ' ')), 'casting-profile-chip'));
        if (Number.isFinite(Number(v.duration_seconds)) && Number(v.duration_seconds) > 0) facts.append(node('span', `Sample ${time(v.duration_seconds)}`, 'casting-profile-chip'));
        for (const tag of (v.tags || []).slice(0, 3)) facts.append(node('span', String(tag), 'casting-profile-chip'));
        card.append(facts, node('p', `Version ${v.version}`, 'muted library-version'));
        const ref = v.inspiration || {};
        if (ref.performer) { const block = node('div', null, 'casting-inspired'); block.append(node('span', 'Inspired by', 'casting-inspired-label'), node('strong', ref.performer)); if (ref.performance) block.append(node('span', ref.performance, 'casting-inspired-work')); card.append(block); }
        const controls = node('div', null, 'button-row library-actions'); const preview = button('Play saved sample', () => play({ ...v, key: 'library-' + v.library_voice_id + '-' + v.version })); preview.disabled = !v.audio_url || !v.audio_sha256; preview.className = 'casting-compare-play';
        controls.append(preview, button('Use as narrator', () => { if (book?.requests?.length) throw new Error('This reading already has a fixed narrator. Import a new reading to choose another voice.'); narratorChoice = voiceValue(v); $('book-narrator').value = narratorChoice; $('import-status').textContent = book ? `${v.name} selected. Choose chapters, then Create MP3.` : `${v.name} selected. Import a book to create a narrator reading.`; updateControls(); }, 'casting-use-option'));
        card.append(controls); $('global-voices').append(card);
      }
      const current = $('book-narrator').value || narratorChoice; $('book-narrator').replaceChildren(); const empty = node('option', 'Choose a saved voice'); empty.value = ''; $('book-narrator').append(empty);
      for (const v of items) { const option = node('option', `${v.name} · ${String(v.version).slice(0, 8)}`); option.value = voiceValue(v); $('book-narrator').append(option); }
      $('book-narrator').value = items.some(v => voiceValue(v) === current) ? current : ''; updateControls();
    }
    async function refreshCatalog() { const state = await api(); library = state.voices || []; catalogBooks = state.books || []; renderLibrary(); const previous = $('creator-books').value; $('creator-books').replaceChildren(); const empty = node('option', 'Choose an imported book'); empty.value = ''; $('creator-books').append(empty); for (const item of catalogBooks) { const option = node('option', item.name || item.project_id); option.value = item.project_id; $('creator-books').append(option); } $('creator-books').value = catalogBooks.some(item => item.project_id === previous) ? previous : ''; }
    function renderOutputs() { $('book-outputs').replaceChildren(); const unique = new Map([...(book?.recordings || []), ...(request?.outputs || [])].map(row => [row.export_id, row])); for (const output of unique.values()) { const card = node('article', null, 'book-output'); card.dataset.exportId = output.export_id; card.append(node('h3', output.label || 'Narrator reading'), node('p', `${time(output.duration_seconds)} · Technical QA: ${output.technical_qa || 'pending'} · Human review: ${output.human_review || 'pending'}`, 'muted'), outputControls({ ...output, project_id: book.project_id })); $('book-outputs').append(card); } }
    function renderRequest() {
      $('book-progress-panel').hidden = !request; if (!request) return;
      const done = finite(request.completed ?? request.done ?? request.job?.done); const total = finite(request.total ?? request.job?.total);
      $('book-progress').textContent = `${String(request.status).replaceAll('_', ' ')} · ${done}/${total} utterances${request.message ? ' · ' + request.message : ''}`;
      $('book-progress-bar').max = Math.max(1, total); $('book-progress-bar').value = done;
      $('resume-book').hidden = !['queued', 'interrupted', 'failed'].includes(request.status); $('resume-book').disabled = busy;
      const automation = request.automation; const toggle = $('book-automation-toggle');
      toggle.hidden = !automation || ['succeeded', 'cancelled'].includes(request.status); toggle.disabled = busy;
      toggle.textContent = automation?.paused || !automation?.enabled ? 'Enable automatic recovery' : 'Pause automatic recovery';
      toggle.setAttribute('aria-pressed', String(!!automation?.paused));
      $('book-automation-status').textContent = automation ? `${automation.paused ? 'Recovery paused' : automation.enabled ? 'Automatic recovery enabled' : 'Automatic recovery disabled'} · ${finite(automation.retry_count)}/${finite(automation.max_retries)} retries used${automation.next_retry_at ? ' · Next retry: ' + automation.next_retry_at : ''}${automation.reason ? ' · ' + automation.reason : ''}` : 'Automatic recovery is not configured for this request.';
      const notification = request.notification; const status = notification?.status || notification?.state;
      $('book-notification-status').textContent = status === 'accepted' || notification?.accepted ? 'Completion push accepted by the notification provider. Display on your phone is not confirmed.' : status === 'delivery_unknown' ? 'Completion push acceptance is uncertain. It will not be sent again automatically.' : notification ? `Completion notification: ${String(status || 'pending').replaceAll('_', ' ')}${notification.message || notification.detail ? ' · ' + (notification.message || notification.detail) : ''}` : request.notify_on_completion ? 'Completion push is enabled for this reading.' : 'Completion push is not enabled for this request.';
      renderOutputs(); updateControls();
    }
    function shouldPoll() { const automation = request?.automation; const recovering = automation?.enabled && !automation.paused && (automation.next_retry_at || (['failed', 'interrupted'].includes(request.status) && finite(automation.retry_count) < finite(automation.max_retries) && !['blocked', 'exhausted'].includes(automation.state))); return !!request && (active() || recovering || !!request.notification?.next_retry_at || ['pending', 'waiting_for_exports'].includes(request.notification?.status || request.notification?.state)); }
    function clearPoll() { if (pollTimer != null && win.clearTimeout) win.clearTimeout(pollTimer); pollTimer = null; }
    function schedulePoll() {
      clearPoll(); if (!request) return;
      if (!pageActive || doc.hidden) { $('book-poll-status').textContent = 'Progress refresh pauses while this page is hidden and resumes when you return. Studio keeps the request.'; return; }
      if (!shouldPoll()) { $('book-poll-status').textContent = 'Progress is up to date. Refresh at any time.'; return; }
      const delay = pollFailures ? Math.min(60000, 5000 * 2 ** Math.min(pollFailures, 4)) : pollCount < 15 ? 4000 : pollCount < 60 ? 15000 : 30000;
      $('book-poll-status').textContent = pollFailures ? `Connection unavailable. Retrying progress in ${delay / 1000} seconds; saved work is unchanged.` : `Live progress · checking every ${delay / 1000} seconds while this page is visible.`;
      if (win.setTimeout) { const token = epoch; const requestId = request.request_id; pollTimer = win.setTimeout(() => { pollTimer = null; if (token === epoch && request?.request_id === requestId) void run(refreshRequest); }, delay); }
    }
    async function refreshRequest() {
      if (!book || !request) return; const token = epoch; const requestId = request.request_id; const flight = `${token}:${requestId}`;
      if (pollInFlight === flight) return; pollInFlight = flight;
      try {
        const result = await api(`/${encode(book.project_id)}/requests/${encode(requestId)}`); if (token !== epoch || request?.request_id !== requestId) return;
        const completedNow = request.status !== 'succeeded' && result.status === 'succeeded'; request = result; pollFailures = 0; pollCount++; renderRequest();
        if (completedNow) { try { await refreshRecordings(); } catch (e) { error(e.message); } }
      } catch (e) { if (token === epoch && request?.request_id === requestId) pollFailures++; throw e; }
      finally { if (pollInFlight === flight) pollInFlight = null; if (token === epoch && request?.request_id === requestId) schedulePoll(); }
    }
    function onVisibility() { if (doc.hidden || !pageActive) schedulePoll(); else { clearPoll(); if (request) void run(refreshRequest); void run(refreshRecordings); } }
    doc.addEventListener('visibilitychange', onVisibility);
    if (win.addEventListener) { win.addEventListener('pagehide', () => { pageActive = false; clearPoll(); }); win.addEventListener('pageshow', () => { pageActive = true; onVisibility(); }); }
    function showBook(state) { epoch++; pollCount = 0; pollFailures = 0; book = state; request = (state.requests || []).find(row => ['queued', 'running', 'interrupted'].includes(row.status)) || state.requests?.[0] || null; $('book-setup').hidden = false; $('book-title').textContent = state.name; $('book-source').textContent = [state.source?.filename, state.source?.format, 'Narrator reading · full selected text'].filter(Boolean).join(' · '); $('book-chapters').replaceChildren(); chapterChoices = (state.chapters || []).map((chapter, i) => { const label = node('label'); const input = node('input'); input.type = 'checkbox'; input.checked = request ? (request.chapter_ids || []).includes(chapter.id) : i === 0; input.onchange = updateControls; label.append(input, node('span', `${chapter.title} · ${chapter.utterance_count ?? 0} utterances`)); $('book-chapters').append(label); return { id: chapter.id, input }; }); const fixed = state.voice_selection || state.requests?.[0]; if (fixed) { narratorChoice = JSON.stringify({ library_voice_id: fixed.library_voice_id, voice_version: fixed.voice_version }); $('book-narrator').value = narratorChoice; } renderLibrary(); $('book-narrator').disabled = !!fixed; $('narrator-lock').textContent = fixed ? 'This reading keeps its saved narrator version. Import a new reading to choose another narrator.' : 'The chosen saved version stays fixed for this reading. No voice design is started.'; renderRequest(); renderOutputs(); updateControls(); schedulePoll(); }
    function chooseFile(files) { if (!files || files.length !== 1 || !/\.(pdf|epub|txt)$/i.test(files[0].name || '')) { file = null; updateControls(); throw new Error('Choose one PDF, EPUB, or TXT file.'); } file = files[0]; importId = win.crypto.randomUUID(); $('import-status').textContent = `${file.name} is ready to import. No audio has been generated.`; updateControls(); }
    $('book-file').onchange = () => run(() => chooseFile($('book-file').files));
    $('book-mode').value = importMode;
    $('book-mode').onchange = () => { importMode = $('book-mode').value === 'autocast' ? 'autocast' : 'single_narrator'; const help = importMode === 'autocast' ? 'AutoCast analyzes the source into character roles for your review. It does not generate audio or assign voices.' : 'Narrator reading uses one saved voice for all text, including dialogue.'; $('book-mode-help').setAttribute('data-tip', help); $('book-mode-help').setAttribute('aria-label', help); };
    $('book-drop').ondragover = event => { event.preventDefault(); $('book-drop').classList.toggle('drag-over', true); };
    $('book-drop').ondragleave = () => $('book-drop').classList.toggle('drag-over', false);
    $('book-drop').ondrop = event => { event.preventDefault(); $('book-drop').classList.toggle('drag-over', false); return run(() => chooseFile(event.dataTransfer?.files)); };
    $('import-book').onclick = () => run(async () => { if (!file || busy) return; const token = ++epoch; busy = true; updateControls(); $('import-status').textContent = importMode === 'autocast' ? 'Analyzing local source for speaking roles…' : 'Importing local source and reading chapter boundaries…'; try { const form = new (win.FormData || globalThis.FormData)(); form.append('file', file); form.append('request_id', importId); form.append('mode', importMode); if ($('book-name').value.trim()) form.append('name', $('book-name').value.trim()); const state = await api('/import', { method: 'POST', headers: { 'X-Review-Request': '1' }, body: form }); if (token !== epoch) return; if (importMode === 'autocast') { const target = state.review_url && safeUrl(state.review_url, win.location); if (!target) throw new Error('AutoCast completed, but its review link is invalid. Open the project from Voice Lab.'); win.location.href = state.review_url; $('import-status').textContent = state.status === 'blocked' ? 'Analysis needs attribution review. Opening the casting inbox…' : 'Analysis is ready. Opening the casting inbox…'; return; } showBook(state); $('import-status').textContent = 'Source imported. Select chapters and a saved narrator, then Create MP3.'; } catch (e) { if (token === epoch) $('import-status').textContent = 'Import did not complete. Your source remains selected for retry.'; throw e; } finally { busy = false; updateControls(); } });
    $('creator-books').onchange = () => { $('open-creator-book').disabled = !$('creator-books').value; };
    $('open-creator-book').onclick = () => run(async () => { if (!$('creator-books').value) return; const token = ++epoch; const state = await api('/' + encode($('creator-books').value)); if (token === epoch) showBook(state); });
    $('book-narrator').onchange = () => { narratorChoice = $('book-narrator').value; updateControls(); };
    async function submitReading(body) { if (busy || !book) return; const token = epoch; busy = true; updateControls(); try { const result = await api(`/${encode(book.project_id)}/render`, post(body)); if (token !== epoch) return; request = result; book.requests = [result, ...(book.requests || []).filter(row => row.request_id !== result.request_id)]; $('book-narrator').disabled = true; $('narrator-lock').textContent = 'This reading keeps this exact narrator version.'; renderRequest(); schedulePoll(); } finally { busy = false; renderRequest(); updateControls(); } }
    $('create-mp3').onclick = () => run(async () => { if (!book || !$('book-narrator').value || !selectedChapters().length) return; await submitReading({ chapter_ids: selectedChapters(), ...JSON.parse($('book-narrator').value), request_id: win.crypto.randomUUID() }); });
    $('resume-book').onclick = () => run(() => request && submitReading({ chapter_ids: request.chapter_ids, library_voice_id: request.library_voice_id, voice_version: request.voice_version, request_id: request.request_id }));
    $('book-automation-toggle').onclick = () => run(async () => {
      if (!book || !request || busy) return; const token = epoch; const requestId = request.request_id;
      const paused = !!request.automation?.enabled && !request.automation?.paused; busy = true; renderRequest();
      try { const result = await api(`/${encode(book.project_id)}/requests/${encode(requestId)}/automation`, post({ paused })); if (token !== epoch || request?.request_id !== requestId) return; request = result; renderRequest(); schedulePoll(); }
      finally { busy = false; renderRequest(); }
    });
    $('refresh-book-progress').onclick = () => run(refreshRequest); $('refresh-global-library').onclick = () => run(refreshCatalog);
    $('open-project').onclick = () => { if ($('projects').value) win.location.href = (isPrivate ? '/voice-lab?project=' : '/web/voice-lab.html?project=') + encode($('projects').value); };
    const initial = await Promise.allSettled([jsonFetch(isPrivate ? '/api/voice-lab-projects' : '/api/v1/projects/summaries'), refreshCatalog(), refreshRecordings()]);
    if (initial[0].status === 'fulfilled') { const result = initial[0].value; const items = Array.isArray(result) ? result : result.projects || result.items || []; for (const item of items) { const option = node('option', item.name || item.id); option.value = item.id; $('projects').append(option); } $('open-project').disabled = !items.length; } else error('Existing books could not be loaded: ' + initial[0].reason.message);
    if (initial[1].status === 'rejected') error('Book creation and saved voices could not be loaded: ' + initial[1].reason.message);
    if (initial[2].status === 'rejected') error('Finished recordings could not be loaded: ' + initial[2].reason.message);
    const query = new URLSearchParams(win.location.search); const linkedBook = query.get('book'); const linkedOutput = query.get('output');
    if (linkedBook && catalogBooks.some(row => row.project_id === linkedBook)) {
      const token = ++epoch;
      try { const state = await api('/' + encode(linkedBook)); if (token === epoch) { $('creator-books').value = linkedBook; $('open-creator-book').disabled = false; showBook(state); } } catch (e) { error('The linked book could not be opened: ' + e.message); }
    }
    if (linkedBook && linkedOutput && player.generation === 0) {
      const output = recordings.find(row => row.project_id === linkedBook && row.export_id === linkedOutput) || (book?.project_id === linkedBook && book.recordings?.find(row => row.export_id === linkedOutput));
      if (output) { selectedOutput = `${linkedBook}:${linkedOutput}`; renderFinished(); await play({ ...output, project_id: linkedBook, key: 'book-' + selectedOutput }, false); $('recordings-status').textContent = `${output.label || 'Recording'} selected. Press Play to listen.`; }
      else error('The linked recording is not available in this library. Refresh recordings to check again.');
    }
    return { player, refreshRequest, refreshRecordings };
  }
  // A bar under the header that lists only the sections that are showing, marks the one being read, and opens a
  // collapsed section when it is chosen. It is an aid: every section works the same without it.
  const NAV_SECTIONS = [['character-audio', 'Ready audio'], ['book-status', 'Book progress'], ['casting-inbox', 'Voices'], ['listening-notes', 'Notes'], ['casting-extra-tools', 'Library & tools'],
    ['finished-library', 'Finished recordings'], ['project-picker', 'Open a book'], ['book-creator', 'Create a book']];
  function startSectionNav(doc, win) {
    const nav = doc.getElementById('page-nav'); if (!nav || typeof doc.createElement !== 'function') return null;
    const links = new Map();
    for (const [id, label] of NAV_SECTIONS) {
      const link = doc.createElement('a'); link.href = `#${id}`; link.textContent = label; link.hidden = true;
      link.addEventListener('click', () => { const target = doc.getElementById(id); if (target && target.tagName === 'DETAILS') target.open = true; });
      links.set(id, link); nav.append(link);
    }
    const showing = el => !!el && !(el.closest && el.closest('[hidden]'));
    const refresh = () => {
      let any = false, current = null;
      for (const [id, link] of links) {
        const el = doc.getElementById(id), on = showing(el); link.hidden = !on; any = any || on;
        // The section being read is the last one whose top has reached the upper part of the window.
        if (on && el.getBoundingClientRect && el.getBoundingClientRect().top <= 140) current = id;
        link.removeAttribute('aria-current');
      }
      if (current) links.get(current).setAttribute('aria-current', 'location');
      nav.hidden = !any;
    };
    let queued = false;
    const soon = () => { if (queued) return; queued = true; (win.requestAnimationFrame || (fn => setTimeout(fn, 16)))(() => { queued = false; refresh(); }); };
    if (typeof win.MutationObserver === 'function') new win.MutationObserver(soon).observe(doc.getElementById('voice-lab') || doc.body, { attributes: true, attributeFilter: ['hidden'], subtree: true });
    win.addEventListener?.('scroll', soon, { passive: true });
    refresh();
    return { refresh };
  }

  // The liquid in every button flows all the time, but only where it can be seen: a button near the window is
  // marked, and the stylesheet stops the flow (and frees its layer) everywhere else. A page with every voice shown
  // has hundreds of buttons; letting them all flow off-screen would hold a GPU layer for each. Without
  // IntersectionObserver nothing is marked or gated, and every button simply flows.
  function startLiquidViewport(doc, win) {
    if (typeof win.IntersectionObserver !== 'function' || typeof win.MutationObserver !== 'function' || !doc.documentElement?.classList) return null;
    const SELECTOR = '#voice-lab button, #voice-lab .button';
    const seen = new WeakSet();
    const io = new win.IntersectionObserver(entries => {
      for (const entry of entries) entry.target.toggleAttribute('data-in-view', entry.isIntersecting);
    }, { rootMargin: '160px 0px' });
    const watch = root => {
      const found = root.matches?.(SELECTOR) ? [root] : [];
      for (const el of root.querySelectorAll ? root.querySelectorAll(SELECTOR) : []) found.push(el);
      for (const el of found) if (!seen.has(el)) { seen.add(el); io.observe(el); }
    };
    watch(doc);
    new win.MutationObserver(records => { for (const record of records) for (const added of record.addedNodes) if (added.nodeType === 1) watch(added); })
      .observe(doc.getElementById('voice-lab') || doc.body, { childList: true, subtree: true });
    doc.documentElement.classList.add('liquid-gated');
    return { watch };
  }

  // An info tip opens to the right of its icon; near the right edge of a narrow window it is moved left just
  // far enough to stay on screen (and never past the left edge).
  function startTipPlacement(doc, win) {
    const place = event => {
      const tip = event?.target?.closest?.('.info-tip');
      if (!tip?.getBoundingClientRect || !tip.style?.setProperty) return;
      const viewport = doc.documentElement?.clientWidth || win.innerWidth || 0, rect = tip.getBoundingClientRect();   // without the scrollbar
      const width = Math.min(320, viewport - 32);
      const overflow = rect.left - 6 + width - (viewport - 16);
      const shift = Math.max(0, Math.min(overflow, rect.left - 22));
      tip.style.setProperty('--tip-shift', `${Math.round(shift)}px`);
    };
    doc.addEventListener?.('pointerover', place, true);
    doc.addEventListener?.('focusin', place, true);
    return { place };
  }

  async function mount(doc, win) {
    const $ = id => doc.getElementById(id);
    try { startTipPlacement(doc, win); } catch (e) { /* a tip that is not moved still shows */ }
    try { startSectionNav(doc, win); } catch (e) { /* the bar is only an aid; the page must not depend on it */ }
    try { startLiquidViewport(doc, win); } catch (e) { /* without it every button simply flows */ }
    const node = (tag, text, className) => { const el = doc.createElement(tag); if (text != null) el.textContent = text; if (className) el.className = className; return el; };
    const button = (text, fn, className = 'secondary') => { const el = node('button', text, className); el.type = 'button'; el.addEventListener('click', () => run(fn)); return el; };
    // Explanations live behind a small info icon, shown on hover or keyboard focus, so the page itself stays quiet.
    const infoTip = text => { const el = node('span', 'i', 'info-tip'); el.tabIndex = 0; el.setAttribute('role', 'img'); el.setAttribute('aria-label', text); el.setAttribute('data-tip', text); return el; };
    const isPrivate = win.location.pathname.startsWith('/voice-lab');
    const project = new URLSearchParams(win.location.search).get('project');
    if (isPrivate) { $('back-link').href = '/'; $('back-link').textContent = 'Listening room'; $('library-home').href = '/voice-lab?home=1'; }
    // S10: connect this book's casting workspace to the existing completed-
    // chapters/downloads surface, scoped to this exact project.
    $('finished-chapters-link').href = (isPrivate ? '/voice-lab?home=1&recordings=' : '/web/voice-lab.html?recordings=') + encodeURIComponent(project || '');
    const error = message => { $('error').hidden = !message; $('error').textContent = message || ''; };
    // Whatever control a person just pressed says it is working until its work is done, however that
    // control was wired: on a busy server a click used to look like a click that did nothing. The press is
    // remembered only for the click being handled, so work started by a timer never borrows a button.
    // Work that finishes before the next frame (opening a sheet) never shows it at all.
    let pressed = null;
    doc.addEventListener?.('click', event => {
      pressed = event?.target?.closest?.('button, .button') || null;
      if (pressed && win.setTimeout) win.setTimeout(() => { pressed = null; }, 0);
    }, true);
    // A click leaves its button sloshing for about two seconds, like liquid in a glass that has just been tapped.
    // The motion is all CSS (a class for the length of the settle), so it costs nothing while nothing is touched.
    doc.addEventListener?.('click', event => {
      const control = event?.target?.closest?.('button, .button');
      if (!control?.classList?.add || control.disabled) return;
      control.classList.remove('settling'); void control.offsetWidth; control.classList.add('settling');
      win.clearTimeout?.(control.settleTimer); control.settleTimer = win.setTimeout?.(() => control.classList.remove('settling'), 2000);
    }, true);
    async function run(fn) {
      const control = pressed; pressed = null;
      control?.setAttribute?.('aria-busy', 'true');
      try { error(''); await fn(); } catch (e) { error(e.message || String(e)); }
      finally { control?.setAttribute?.('aria-busy', 'false'); }
    }
    if (!project) {
      if (isPrivate && new URLSearchParams(win.location.search).get('home') !== '1') throw new Error('No authorized Voice Lab project is selected. Open Voice Lab from your private listening room.');
      if ($('voice-lab-startup')) $('voice-lab-startup').hidden = true;
      if ($('voice-lab')) $('voice-lab').dataset.startupLoading = 'false';
      return mountLanding(doc, win, { $, node, button, run, error });
    }
    const client = new ReviewClient(isPrivate ? '/api/voice-lab/' + encodeURIComponent(project) : '/api/v1/projects/' + encodeURIComponent(project) + '/voice-lab', win.fetch.bind(win));
    const player = new Player($('media'));
    const startupPanel = $('voice-lab-startup'); const startupStatus = $('voice-lab-startup-status'); const startupRetry = $('voice-lab-startup-retry');
    const startup = (message, failed = false) => {
      if (startupPanel) { startupPanel.hidden = !message; startupPanel.setAttribute('aria-busy', String(!!message && !failed)); }
      if (startupStatus) startupStatus.textContent = message || '';
      if (startupRetry) startupRetry.hidden = !failed;
      if ($('voice-lab')) $('voice-lab').dataset.startupLoading = String(!!message);
    };
    if (startupRetry) startupRetry.addEventListener('click', () => win.location.reload?.());
    $('workspace').hidden = true; startup('Loading your voices…');
    // Loading and failure are shown, never silent: a slow or failed audio request
    // used to look exactly like a button that did nothing.
    let stallTimer = null;
    const setPlayerStatus = text => { const node = $('player-status'); if (node) node.textContent = text || ''; };
    const clearStall = () => { if (stallTimer != null) { win.clearTimeout?.(stallTimer); stallTimer = null; } };
    const watchLoading = () => {
      clearStall(); setPlayerStatus('Loading audio…');
      stallTimer = win.setTimeout?.(() => setPlayerStatus('Still loading. The review service is busy; audio starts as soon as it arrives.'), 8000) ?? null;
      updateCastingPlayLabels();
    };
    for (const name of ['loadstart', 'waiting']) player.media.addEventListener(name, watchLoading);
    for (const name of ['canplay', 'playing', 'loadeddata', 'pause', 'ended']) player.media.addEventListener(name, () => { clearStall(); setPlayerStatus(''); updateCastingPlayLabels(); });
    player.media.addEventListener('error', () => { clearStall(); setPlayerStatus('This audio could not be loaded. Press Play on the card to try again.'); updateCastingPlayLabels(); });

    // ---- Where is my book? ---------------------------------------------------
    // A push must land on an answer, not on a form: which voices still wait for the
    // human, how far the recording has got, or the finished chapters to play. It is
    // requested before (and independently of) the much larger casting review so it
    // appears at once, and it only ever reads real state.
    let bookStatus = null; let bookTimer = null; let bookRendered = ''; let bookEta = null; let bookSkippedOpen = false; let bookNote = ''; let bookPollBusy = false; let bookLinkedOpened = false;
    const bookPlayButtons = new Map();
    const bookPanel = $('book-status');
    const bookRecording = row => ({ key: row.export_id, audio_url: row.audio_url, audio_sha256: row.audio_sha256,
      duration_seconds: row.duration_seconds, name: row.title, book_chapter: true });
    function updateBookPlayLabels() {
      for (const [exportId, item] of bookPlayButtons) {
        const loaded = !!player.current?.book_chapter && player.current.key === exportId;
        const playing = loaded && !player.media.paused && !player.media.error;
        item.button.textContent = playing ? 'Pause' : 'Play';
        item.button.setAttribute('aria-label', `${playing ? 'Pause' : 'Play'} ${item.row.title}`);
        item.item.dataset.current = String(loaded);
      }
    }
    async function playBookChapter(row, autoplay = true) {
      if (autoplay && player.current?.book_chapter && player.current.key === row.export_id && !player.media.error) {
        await player.toggle(); updateBookPlayLabels(); updateTime(); return;
      }
      await loadRecording(bookRecording(row), { autoplay });
      updateBookPlayLabels(); updateTime();
    }
    // Every decision is one request that answers with the book as it now stands, and the
    // server restarts a blocked book itself, so the panel only has to show what was saved.
    async function bookAction(path, body, saved) {
      bookNote = 'Saving your choice…'; bookRendered = ''; if (bookStatus) renderBookStatus(bookStatus);
      try {
        const result = await client.request(path, body);
        bookNote = saved(result.decision || {});
        applyBookStatus(result.book || bookStatus); void refreshCasting(false);
      } catch (e) { bookNote = ''; bookRendered = ''; if (bookStatus) renderBookStatus(bookStatus); throw e; }
    }
    function savedNote(decision) {
      const parts = [];
      if (decision.resolved) parts.push(decision.mode === 'narrator' ? `${plural(decision.resolved, 'line')} will be read by the narrator` : `${plural(decision.resolved, 'speaker')} accepted`);
      if (decision.skipped) parts.push(`${plural(decision.skipped, 'line')} skipped`);
      return `Saved: ${parts.join(', ') || 'no change'}. Recording starts shortly.`;
    }
    const decideUnattributed = mode => bookAction('casting/production/resolve', { request_id: uuid(), mode }, savedNote);
    const setAutomatic = enabled => bookAction('casting/production/policy', { enabled },
      () => (enabled ? 'Automatic handling is on. Lines with no speaker are skipped so the book keeps going.' : 'Automatic handling is off. The book will wait for you.'));
    const skipLine = id => bookAction('casting/production/skip', { request_id: uuid(), utterance_ids: [id] }, () => 'Line skipped. Recording resumes shortly.');
    const restoreLines = reason => bookAction('casting/production/restore', { request_id: uuid(), reason },
      decision => `${plural(decision.restored || 0, 'line')} brought back. ${decision.restored ? 'Recording resumes shortly.' : ''}`.trim());
    async function tryAgain(requestId) {
      bookNote = 'Trying again…'; bookRendered = ''; if (bookStatus) renderBookStatus(bookStatus);
      try { await client.request('casting/resume', { request_id: requestId }); bookNote = 'Recording resumes shortly.'; await refreshBookStatus(); }
      catch (e) { bookNote = ''; bookRendered = ''; if (bookStatus) renderBookStatus(bookStatus); throw e; }
    }
    // The panel is rebuilt when the book moves on, and a bar rebuilt at its new width used to jump
    // there and restart its glint. Each new fill starts at the width its bar last showed and eases
    // to the new one, and its glint keeps the phase it had. A chapter's bar is its own: the next
    // chapter starts from its own progress, not from the end of the last one.
    const barWidths = new Map();
    function progressFill(key, percent, period) {
      const fill = node('span', null, 'book-bar-fill');
      const from = barWidths.has(key) ? barWidths.get(key) : percent;
      barWidths.set(key, percent);
      const phase = `-${Date.now() % period}ms`;
      setStyleVars(fill, { '--p': `${from}%`, 'animation-delay': phase });
      if (from !== percent) {
        const settle = () => setStyleVars(fill, { '--p': `${percent}%`, 'animation-delay': phase });
        if (win.requestAnimationFrame) win.requestAnimationFrame(() => win.requestAnimationFrame(settle)); else settle();
      }
      return fill;
    }
    // The time left changes on every poll, so it is repainted in place: the panel (and
    // any focus inside it) is only rebuilt when something a person acts on has changed.
    function paintBookEta(status) {
      if (!bookEta) return;
      const eta = status?.production?.eta || {};
      if (bookEta.book) bookEta.book.textContent = Number.isFinite(eta.book_seconds) ? `${timeLeft(eta.book_seconds)} for the whole book` : '';
      if (bookEta.chapter) {
        const left = timeLeft(eta.chapter_seconds);
        bookEta.chapter.textContent = left;
        bookEta.bar.setAttribute('aria-valuetext', `Now recording “${bookEta.title}”: ${bookEta.done} of ${bookEta.total} lines recorded, ${left.charAt(0).toLowerCase()}${left.slice(1)}`);
      }
    }
    function renderBookStatus(status) {
      if (!bookPanel) return;
      const summary = bookSummary(status);
      if (!summary) { bookPanel.hidden = true; return; }
      const signature = JSON.stringify(status, (key, value) => (key === 'eta' || key === 'eta_seconds' ? undefined : value)) + '|' + bookNote + '|' + bookSkippedOpen;
      if (signature === bookRendered) { paintBookEta(status); return; }
      bookRendered = signature;
      bookPanel.hidden = false; bookPanel.dataset.state = status.state;
      bookPlayButtons.clear(); bookEta = null;
      const production = status.production; const chapters = status.chapters || [];
      const head = node('div', null, 'book-head');
      const title = node('h2', summary.title); title.id = 'book-status-heading';
      const actionable = status.state === 'blocked' || status.state === 'waiting';
      const titleLine = node('div', null, 'title-line'); titleLine.append(title);
      if (!actionable) titleLine.append(infoTip(summary.detail));
      head.append(node('p', 'YOUR BOOK', 'eyebrow'), titleLine);
      if (actionable) head.append(node('p', summary.detail, 'book-detail'));
      bookPanel.setAttribute('aria-labelledby', 'book-status-heading');
      bookPanel.replaceChildren(head);
      const waiting = status.voices?.waiting || [];
      if (waiting.length) {
        const chips = node('div', null, 'book-voices');
        for (const voice of waiting) {
          const chip = node('a', `${castingRoleLabel(voice.name)} · ${plural(voice.options, 'option')}`, 'book-voice-chip');
          chip.href = `#casting-role-${voice.role_id}`;
          chip.setAttribute('aria-label', `Go to ${castingRoleLabel(voice.name)}, ${plural(voice.options, 'option')} to hear`);
          chips.append(chip);
        }
        bookPanel.append(chips, button('Choose voices', () => { $('casting-inbox')?.scrollIntoView?.({ behavior: 'smooth', block: 'start' }); }, 'book-cta'));
      }
      if (production && ['running', 'exporting', 'finished'].includes(production.phase)) {
        const total = Math.max(0, production.lines_total || 0); const done = production.phase === 'finished' ? total : Math.min(total, production.lines_done || 0);
        const percent = total ? Math.round(done / total * 100) : 0;
        const current = chapters.find(row => row.status === 'rendering');
        const text = production.phase === 'finished' ? `${plural(production.chapters_total, 'chapter')} · ${grouped(total)} lines`
          : `${grouped(done)} of ${grouped(total)} lines recorded · ${production.chapters_ready} of ${plural(production.chapters_total, 'chapter')} ready`;
        const bar = node('div', null, 'book-bar'); bar.setAttribute('role', 'progressbar'); bar.setAttribute('aria-label', 'Book progress');
        bar.setAttribute('aria-valuemin', '0'); bar.setAttribute('aria-valuemax', '100'); bar.setAttribute('aria-valuenow', String(percent));
        bar.setAttribute('aria-valuetext', `${text} (${percent}%)`);
        // Recording, with nothing in yet: a working bar rather than an empty one.
        bar.dataset.indeterminate = String(production.phase === 'running' && done === 0);
        bar.append(progressFill('book', percent, 2200));
        const line = node('p', null, 'book-progress'); line.append(node('strong', `${percent}%`), node('span', text));
        bookPanel.append(bar, line);
        if (production.phase === 'running') {
          bookEta = { book: node('span', null, 'book-eta-book') }; line.append(bookEta.book);
          if (current) {
            const lines = Math.max(0, current.lines_total || 0); const recorded = Math.min(lines, current.lines_done || 0);
            const share = lines ? Math.round(recorded / lines * 100) : 0;
            const box = node('div', null, 'book-chapter-progress');
            const head = node('p', null, 'book-chapter-progress-head'); const left = node('span', null, 'book-eta');
            head.append(node('strong', `Now recording “${current.title}”`), left);
            const chapterBar = node('div', null, 'book-bar chapter'); chapterBar.setAttribute('role', 'progressbar'); chapterBar.setAttribute('aria-label', 'Chapter progress');
            chapterBar.setAttribute('aria-valuemin', '0'); chapterBar.setAttribute('aria-valuemax', '100'); chapterBar.setAttribute('aria-valuenow', String(share));
            chapterBar.dataset.indeterminate = String(recorded === 0);
            chapterBar.append(progressFill(`chapter:${current.chapter_id || current.title}`, share, 1600));
            box.append(head, chapterBar, node('p', `${grouped(recorded)} of ${grouped(lines)} lines in this chapter`, 'book-chapter-lines muted small'));
            bookPanel.append(box);
            Object.assign(bookEta, { chapter: left, bar: chapterBar, title: current.title, done: recorded, total: lines });
          }
          paintBookEta(status);
        }
      }
      const block = status.block;
      if (block && ['blocked', 'waiting'].includes(status.state)) {
        const box = node('div', null, 'book-block');
        box.append(node('p', `${plural(block.unresolved, 'dialogue line')} ${block.unresolved === 1 ? 'has' : 'have'} no speaker yet: ${block.suggested_mappable} ${block.suggested_mappable === 1 ? 'has' : 'have'} a suggested speaker, ${block.unresolved - block.suggested_mappable} could not be attributed. Nothing changes until you choose.`, 'book-block-text'));
        const actions = node('div', null, 'book-actions'); const can = name => (block.actions || []).includes(name);
        const rest = block.unresolved - block.suggested_mappable;
        if (can('auto')) actions.append(button(rest > 0 ? `Accept the ${plural(block.suggested_mappable, 'suggested speaker')} and skip the other ${rest}` : `Accept the ${plural(block.suggested_mappable, 'suggested speaker')}`, () => decideUnattributed('auto'), 'book-cta'));
        else if (can('skip')) actions.append(button(`Skip the ${plural(block.unresolved, 'line')} with no speaker`, () => decideUnattributed('skip'), 'book-cta'));
        if (can('narrator')) actions.append(button(`Read ${plural(block.unresolved, 'line')} with ${castingRoleLabel(block.narrator)}`, () => decideUnattributed('narrator'), 'book-cta secondary'));
        box.append(actions);
        bookPanel.append(box);
      }
      const failed = status.failed_line;
      if (failed) {
        const box = node('div', null, 'book-failed');
        box.append(node('p', `The book stopped at this line${failed.chapter ? ` in “${failed.chapter}”` : ''}:`, 'book-failed-title'), node('p', `“${failed.text}”`, 'book-failed-text'), node('p', failed.error, 'book-failed-error muted small'));
        const actions = node('div', null, 'book-actions');
        actions.append(button('Skip this line', () => skipLine(failed.utterance_id), 'book-cta'), button('Try again', () => tryAgain(production.request_id), 'book-cta secondary'));
        box.append(actions); bookPanel.append(box);
      }
      if (production?.reason && ['blocked', 'waiting'].includes(status.state) && !failed) bookPanel.append(node('p', production.reason, 'book-reason muted small'));
      if (bookNote) { const note = node('p', bookNote, 'book-note'); note.setAttribute('role', 'status'); bookPanel.append(note); }
      const skipped = status.skipped;
      if (skipped) {
        const recording = production?.phase === 'running';
        const box = node('div', null, 'book-skipped');
        const head = node('p', null, 'book-skipped-head');
        head.append(node('strong', `${plural(skipped.total, 'line')} skipped`), node('span', ' and left out of the recording.', 'muted'));
        const toggle = button(bookSkippedOpen ? 'Hide' : 'Show which', () => { bookSkippedOpen = !bookSkippedOpen; bookRendered = ''; renderBookStatus(bookStatus); }, 'book-skipped-toggle secondary');
        toggle.setAttribute('aria-expanded', String(bookSkippedOpen));
        if (recording) head.append(' ', infoTip('You can bring lines back when recording pauses or finishes.'));
        box.append(head, toggle);
        if (bookSkippedOpen) {
          for (const group of skipped.groups || []) {
            const section = node('div', null, 'book-skipped-group'); section.dataset.reason = group.reason;
            const title = node('p', null, 'book-skipped-group-head'); title.append(node('strong', group.label), node('span', ` · ${grouped(group.count)}`, 'muted'));
            section.append(title);
            if (!recording) section.append(button(`Bring back ${plural(group.count, 'line')}`, () => restoreLines(group.reason), 'book-restore secondary'));
            const list = node('ul', null, 'book-skipped-lines');
            for (const line of group.lines || []) list.append(node('li', `“${line.text}”${line.chapter ? ` · ${line.chapter}` : ''}`, 'book-skipped-line'));
            if (group.more) list.append(node('li', `and ${grouped(group.more)} more`, 'book-skipped-line muted'));
            section.append(list); box.append(section);
          }
        }
        bookPanel.append(box);
      }
      if (production) {
        const auto = node('div', null, 'book-auto');
        auto.append(node('span', status.automatic ? 'Automatic handling is on: lines with no speaker, and lines the voice model will not speak, are skipped so the book keeps going.'
          : 'Automatic handling is off: the book waits for you when a line has no speaker.', 'book-auto-text muted small'));
        auto.append(button(status.automatic ? 'Turn off' : 'Turn on', () => setAutomatic(!status.automatic), 'book-auto-toggle secondary'));
        bookPanel.append(auto);
      }
      if (chapters.length) {
        const list = node('ol', null, 'book-chapters');
        for (const row of chapters) {
          const item = node('li', null, 'book-chapter'); item.dataset.status = row.status; item.dataset.chapterId = row.chapter_id;
          const label = node('div', null, 'book-chapter-label');
          label.append(node('span', String(row.number), 'book-chapter-number'), node('strong', row.title));
          const base = row.status === 'ready' ? time(row.duration_seconds) : row.status === 'rendering' ? `Recording · ${row.lines_done} of ${row.lines_total} lines`
            : row.status === 'failed' ? `Needs another try${row.reason ? `: ${row.reason}` : ''}` : 'Waiting';
          const meta = row.lines_skipped ? `${base} · ${plural(row.lines_skipped, 'line')} skipped` : base;
          label.append(node('span', meta, 'book-chapter-meta'));
          item.append(label);
          if (row.status === 'ready') {
            const play = button('Play', () => playBookChapter(row), 'book-play');
            bookPlayButtons.set(row.export_id, { button: play, row, item });
            const controls = node('div', null, 'book-chapter-actions'); controls.append(play);
            if (row.download_url) { const download = node('a', 'Download', 'button secondary'); download.href = row.download_url; download.setAttribute('download', ''); controls.append(download); }
            item.append(controls);
          }
          list.append(item);
        }
        bookPanel.append(list);
      }
      updateBookPlayLabels();
    }
    function scheduleBookPoll() {
      const active = !!bookStatus && (['processing', 'starting', 'casting', 'waiting'].includes(bookStatus.state) || (bookStatus.state === 'blocked' && bookStatus.automatic === true));
      if (active && !bookTimer && win.setInterval) {
        bookTimer = win.setInterval(() => {
          if (bookPollBusy || win.document?.hidden) return;
          bookPollBusy = true; void refreshBookStatus().finally(() => { bookPollBusy = false; });
        }, 5000);
      }
      if (!active && bookTimer && win.clearInterval) { win.clearInterval(bookTimer); bookTimer = null; }
    }
    function applyBookStatus(status) {
      if (status?.schema !== 'casting-book-status-v1') return;
      bookStatus = status; renderBookStatus(status); scheduleBookPoll();
    }
    async function refreshBookStatus() {
      try { applyBookStatus(await client.request('casting/book')); } catch (_) { /* the panel keeps what it last knew */ }
    }
    const bookReady = refreshBookStatus();
    async function openLinkedBookChapter() {
      const linked = new URLSearchParams(win.location.search).get('chapter');
      if (bookLinkedOpened || !linked || !bookStatus) return;
      const row = (bookStatus.chapters || []).find(item => item.chapter_id === linked && item.status === 'ready');
      if (!row) return;
      bookLinkedOpened = true;
      await loadRecording(bookRecording(row), { autoplay: false });
      updateBookPlayLabels(); updateTime();
      bookPanel?.scrollIntoView?.({ behavior: 'smooth', block: 'start' });
    }
    let data;
    try {
      data = await client.request('');
      if (!data || typeof data !== 'object' || Array.isArray(data)) throw new Error('The review service returned an invalid project response. Please retry.');
    } catch (e) { startup(`Your voices could not be loaded. ${e.message}`, true); return { player, client, startupFailed: true }; }
    let castingEnabled = data.casting_enabled === true || data.capabilities?.casting_enabled === true; let selectedKey = null; let filter = data.review_position?.filter || 'new'; let hideLabels = false; let lastPosition = 0; let savingPosition = false; let currentMatches = []; let loadToken = 0;
    const auditionState = new Map(); const auditionBusy = new Set(); const auditionTimers = new Map(); const candidateRoles = new Map();
    const pending = new Set();
    let castingPollBusy = false; let casting = null; let castingActive = null; let castingDirectionOpen = false; let showApprovedCasts = false; let castingTimer = null; let castingRequestId = null;
    const castingRoleBusy = new Set(); const castingRoundBusy = new Set(); const castingRoundControls = new Map(); const castingCardControls = new Map(); const castingCardStates = new Map(); const castingPlayButtons = new Map();
    // Scoped strictly to the one candidate whose feedback is in flight, so a
    // save never disables a sibling candidate's own stars/actions.
    const castingCandidateBusy = new Set();
    // Exact-version favorites: one in-flight toggle per key|hash, plus the last server-confirmed value so a stale poll cannot erase it.
    const castingFavoriteBusy = new Set(); const castingFavoriteSyncs = new Map(); const castingFavoriteNotes = new Map();
    // Ratings given for a character while another of its ratings is still being saved.
    const castingFeedbackWaiting = new Map();
    let castingWriteGeneration = 0;
    const uuid = () => win.crypto?.randomUUID?.() || `ui_${Date.now().toString(36)}_${Math.random().toString(36).slice(2)}`;
    const clone = value => JSON.parse(JSON.stringify(value));
    function openCastingInbox() {
      const inbox = $('casting-inbox');
      inbox.scrollIntoView?.({ behavior: 'smooth', block: 'start' });
      inbox.focus?.();
    }
    // A Pushover deep link names the exact review the human was asked to do.
    // This is a pure read of already-loaded state: opening it never writes
    // casting state and never causes a notification.
    const deepLink = new URLSearchParams(win.location.search);
    const linkedCharacterId = deepLink.get('character');
    const linkedCastingView = deepLink.get('view') === 'casting';
    function focusLinkedCastingReview() {
      if (!casting) return false;
      if (!linkedCharacterId && bookStatus && bookStatus.state !== 'idle' && bookPanel && !bookPanel.hidden) {
        bookPanel.scrollIntoView?.({ behavior: 'smooth', block: 'start' }); bookPanel.focus?.();
        return true;
      }
      openCastingInbox();
      if (!linkedCharacterId) return true;
      const role = (casting.roles || []).find(item => item.role_id === linkedCharacterId);
      if (!role) { setCastingFlash('This character is no longer in the current casting review. Showing every character instead.'); return false; }
      const ready = castingRows(role).filter(candidateReady);
      if (!ready.length) { setCastingFlash(`${castingRoleName(role)} has no playable audition yet; its status is shown below.`); }
      const newest = [...ready].sort((left, right) => String(right.ready_at || '').localeCompare(String(left.ready_at || '')))[0];
      if (newest) { activateCasting(role, newest); renderCasting(); }
      const card = doc.getElementById(`casting-role-${role.role_id}`);
      card?.scrollIntoView?.({ behavior: 'smooth', block: 'start' });
      card?.focus?.();
      if (newest) {
        const prefix = `${role.role_id}:${newest.candidate_id}:${newest.key}:${newest.audio_sha256}:`;
        for (const [key, control] of castingPlayButtons) if (key.startsWith(prefix)) { control.button?.focus?.(); break; }
      }
      return true;
    }
    function loadCastingDraft(role, candidate) {
      const key = castingDraftKey(project, role, candidate);
      try { return { key, draft: JSON.parse(win.localStorage?.getItem(key) || 'null') || { stars: null, raw_direction: '' } }; }
      catch (_) { return { key, draft: { stars: null, raw_direction: '' } }; }
    }
    function saveCastingDraft(key, draft) {
      try { win.localStorage?.setItem(key, JSON.stringify(draft)); } catch (_) { /* in-memory editing remains available */ }
    }
    function clearCastingDraft(key) { try { win.localStorage?.removeItem(key); } catch (_) { /* server remains authoritative */ } }
    const castingRoundDraftKey = role => `voice-lab-casting-round-v1:${project}:${role.role_id}`;
    function castingRoundState(role) {
      if (castingRoundControls.has(role.role_id)) return castingRoundControls.get(role.role_id).state;
      let saved = null; try { saved = JSON.parse(win.localStorage?.getItem(castingRoundDraftKey(role)) || 'null'); } catch (_) { /* invalid local request is ignored */ }
      return { pendingRequest: saved?.pendingRequest || null, unsupported: saved?.unsupported === true, busy: false };
    }
    function persistCastingRoundState(role, state) {
      const key = castingRoundDraftKey(role);
      try {
        if (state.pendingRequest || state.unsupported) win.localStorage?.setItem(key, JSON.stringify({ pendingRequest: state.pendingRequest || null, unsupported: !!state.unsupported }));
        else win.localStorage?.removeItem(key);
      } catch (_) { /* server request remains idempotent for this mounted page */ }
    }
    function hasPendingFeedback(role) {
      return (role.candidates || []).some(candidate => {
        const state = castingCardStates.get(castingCandidateKey(role, candidate));
        return !!(state?.pendingRequest || loadCastingDraft(role, candidate).draft.pendingRequest);
      });
    }
    // Shortlist is a local device bookmark, deliberately not a backend call:
    // it organizes review without ever promoting a voice or awarding a star.
    function shortlistKey(role) { return `voice-lab-shortlist-v1:${project}:${role.role_id}`; }
    function loadShortlist(role) {
      try { return new Set(JSON.parse(win.localStorage?.getItem(shortlistKey(role)) || '[]')); }
      catch (_) { return new Set(); }
    }
    function toggleShortlist(role, candidate) {
      const set = loadShortlist(role);
      if (set.has(candidate.candidate_id)) set.delete(candidate.candidate_id); else set.add(candidate.candidate_id);
      try { win.localStorage?.setItem(shortlistKey(role), JSON.stringify([...set])); } catch (_) { /* shortlist stays session-only on this device */ }
      castingPendingFocus = voiceFocusKey(role, candidate, 'shortlist');
      voiceTransition(() => renderCasting());
    }
    // Pairwise A/B selection is session-local UI state, at most two candidates
    // per character. Assignment still goes through the exact existing
    // explicit "Use this voice" contract; comparison never assigns by itself.
    const castingCompareSelection = new Map();
    // "New round ready" is announced, never entered: the round a human is
    // hearing stays put until they choose to look, and nothing autoplays.
    const castingCompareBoxes = new Map(); const castingSeenRounds = new Map(); const castingNewRound = new Set();
    // A/B preference is a private on-device note (like the shortlist). It is
    // never a rating and never an assignment.
    const preferenceKey = role => `voice-lab-preference-v1:${project}:${role.role_id}`;
    function loadPreference(role) { try { return win.localStorage?.getItem(preferenceKey(role)) || null; } catch (_) { return null; } }
    function togglePreference(role, candidate) {
      const current = loadPreference(role);
      try { if (current === candidate.candidate_id) win.localStorage?.removeItem(preferenceKey(role)); else win.localStorage?.setItem(preferenceKey(role), candidate.candidate_id); } catch (_) { /* preference stays unsaved */ }
      voiceTransition(() => renderCasting());
    }
    function compareSelected(role) { return castingCompareSelection.get(role.role_id) || []; }
    function toggleCompare(role, candidate) {
      const current = compareSelected(role);
      const index = current.findIndex(c => c.candidate_id === candidate.candidate_id);
      const next = index >= 0 ? current.filter((_, i) => i !== index)
        : current.length >= 2 ? [current[1], clone(candidate)] : [...current, clone(candidate)];
      castingCompareSelection.set(role.role_id, next);
      castingPendingFocus = voiceFocusKey(role, candidate, 'compare');
      voiceTransition(() => renderCasting());
    }
    const castingRows = role => (role.candidates || []).map(candidate => ({
      ...candidate,
      // Sections are seek points inside the whole audition, never candidates.
      // Keep the parent candidate's key/hash/ID as the review identity.
      sections: Array.isArray(candidate.sections) ? candidate.sections : []
    }));
    // The proposal's own evidence, so a candidate is reviewed as a deliberate
    // casting choice rather than an anonymous synthetic voice.
    const traitLine = row => `${row.dimension.replace(/_/g, ' ')}: ${row.value}`;
    const castingTraits = candidate => {
      // What the book establishes and what casting inferred are never mixed: a
      // reviewer must be able to see which dimensions the text supports.
      const supported = Array.isArray(candidate?.supported_traits) ? candidate.supported_traits : [];
      const inferred = Array.isArray(candidate?.inferred_traits) ? candidate.inferred_traits : [];
      const groups = [];
      if (supported.length) groups.push(['Book-supported traits', supported.map(traitLine)]);
      if (inferred.length) groups.push(['Casting inference (not in the book)', inferred.map(traitLine)]);
      return groups;
    };
    const castingAcoustics = candidate => {
      // The acoustic layer, reported as it happened: what was characterized,
      // whether it was measured or described, and the parameters that reached TTS.
      const reference = candidate?.reference || null;
      const acoustics = candidate?.acoustics || null;
      const conditioning = candidate?.conditioning || null;
      const lines = [];
      const add = (text, verdict = null) => lines.push({ text, verdict });
      if (reference) {
        if (reference.source_type) add(`Reference evidence: ${reference.source_type}`);
        if (reference.retrieval_state) {
          add(`Reference audio retrieval: ${reference.retrieval_state}${reference.retrieval_reason ? ` (${reference.retrieval_reason})` : ''}`);
        }
        if (reference.characterization_method) add(`Acoustic characterization: ${reference.characterization} · ${reference.characterization_method}`);
        const measured = reference.measured_properties || [];
        const derived = reference.inferred_properties || [];
        if (measured.length) add(`Measured properties: ${measured.join(', ')}`);
        if (derived.length) add(`Derived (documented prior, not measured): ${derived.join(', ')}`);
      }
      const parameters = conditioning?.parameters || {};
      const bits = [];
      if (parameters.pitch_target_hz) bits.push(`${Math.round(parameters.pitch_target_hz)} Hz`);
      if (parameters.pitch_range_semitones) bits.push(`${parameters.pitch_range_semitones.toFixed(1)} st range`);
      if (parameters.rate_words_per_minute) bits.push(`${Math.round(parameters.rate_words_per_minute)} wpm`);
      if (parameters.resonance) bits.push(`${parameters.resonance} resonance`);
      if (parameters.vocal_weight) bits.push(`${parameters.vocal_weight} weight`);
      if (parameters.age_impression) bits.push(`${parameters.age_impression} impression`);
      if (parameters.rasp !== null && parameters.rasp !== undefined) bits.push(`rasp ${parameters.rasp.toFixed(2)}`);
      if (parameters.breathiness !== null && parameters.breathiness !== undefined) bits.push(`breathiness ${parameters.breathiness.toFixed(2)}`);
      if (bits.length) add(`Generation parameters sent to TTS: ${bits.join(', ')}`);
      if (acoustics?.shaping_plan) {
        add(`Acoustic shaping applied: ${acoustics.shaping_plan.semitones >= 0 ? '+' : ''}${acoustics.shaping_plan.semitones.toFixed(2)} semitones, x${acoustics.shaping_plan.rate.toFixed(3)} rate`);
      }
      // Four independent verdicts, never one escalating badge: reached,
      // shortfall, out_of_scope and could_not_measure are rendered as their
      // own distinct lines. Only what the backend actually reported appears;
      // an empty comparison never renders as a pass.
      const reach = acoustics?.reach;
      if (reach) {
        if ((reach.reached || []).length) add(`Target reached: ${reach.reached.join(', ')}.`, 'reached');
        if ((reach.shortfall || []).length) {
          add(`Target not fully reached: ${reach.shortfall.join(', ')} still outside tolerance after the bounded shaping pass.`, 'shortfall');
        }
        if ((reach.out_of_scope || []).length || reach.out_of_scope_count) {
          const keys = (reach.out_of_scope || []).join(', ');
          add(`Out of scope: ${keys || `${reach.out_of_scope_count} target key(s)`} cannot be moved by any pitch/rate transform.`, 'out_of_scope');
        }
        if (reach.verdict === 'could_not_measure') {
          add('Target reach could not be measured: the comparison touched no key the acoustic shaping pass can move.', 'could_not_measure');
        }
      }
      return lines;
    };
    const castingEvidence = candidate => {
      const anchor = candidate?.anchor || {};
      const inspiration = candidate?.inspiration || {};
      const prior = candidate?.prior_evidence || null;
      const performer = inspiration.performer || anchor.performer;
      const performance = inspiration.performance || anchor.performance;
      const lines = [];
      if (prior) {
        lines.push(`Previously approved voice · prior rating ${prior.prior_rating} stars · ${prior.match_percent}% match`);
        if (prior.inspiration?.performer) lines.push(`Originally inspired by ${prior.inspiration.performer}${prior.inspiration.performance ? `, ${prior.inspiration.performance}` : ''}`);
        if (prior.previously_used) lines.push(`Previously used in ${prior.previously_used}`);
        const reasons = (prior.match_reasons || []).filter(row => !row.startsWith('previously inspired'));
        if (reasons.length) lines.push(`Relevant traits: ${reasons.join(', ')}`);
      } else if (candidate?.blocked) {
        lines.push(`Blocked: ${candidate.blocked_reason || anchor.unanchored_reason || 'this candidate has no performer reference'}`);
      } else if (performer) {
        lines.push(`Inspiration: ${performer}${performance ? `, ${performance}` : ''}`);
        const matched = (anchor.matched_traits || []).join(', ');
        lines.push(`Match rationale: ${matched || anchor.rationale || 'exploratory fit from the performance catalogue'}`);
      } else if (anchor.unanchored_reason) {
        lines.push(`Unanchored: ${anchor.unanchored_reason}`);
      }
      if (candidate?.concept?.emphasis) lines.push(`Casting concept: ${candidate.concept.label || candidate.concept.key} — ${candidate.concept.emphasis}`);
      if (candidate?.import_state === 'blocked') lines.push(`This voice is unavailable here: ${candidate.import_reason || 'the approved voice could not be imported'}`);
      else if (candidate?.import_state === 'waiting') lines.push('Preparing this approved voice for listening…');
      return lines;
    };
    const candidateReady = candidate => candidate?.status === 'ready' && !!candidate.key && !!candidate.audio_sha256 && !!candidate.audio_url && !!safeUrl(candidate.audio_url, win.location);
    const castingRoleName = role => castingRoleLabel(role?.name || role?.role_id);
    const castingCandidateKey = (role, candidate) => castingDraftKey(project, role, candidate);
    function audioReadiness(candidate) {
      if (candidateReady(candidate)) return `Audio ready · ${time(candidate.duration_seconds)}`;
      if (['blocked', 'failed', 'unavailable', 'source_changed'].includes(candidate?.status)) return `Failed — ${candidate.reason || 'audio unavailable'}`;
      if (['running', 'generating'].includes(candidate?.status)) return 'Generating audio';
      if (['pending', 'queued'].includes(candidate?.status)) return 'Queued — audio not generated yet';
      return 'Not generated';
    }
    function cardControls(role, candidate) { return castingCardControls.get(castingCandidateKey(role, candidate)); }
    function updateCastingPlayLabels() {
      const current = player.current?.casting_context;
      const media = player.media;
      for (const item of castingPlayButtons.values()) {
        const loaded = !!current && current.role_id === item.roleId
          && current.candidate_id === item.candidateId && current.key === item.key && current.audio_sha256 === item.audioSha256;
        const active = loaded && !media.paused && !media.error;
        const slot = item.slot ? ` ${item.slot}` : '';
        item.button.textContent = `${active ? 'Pause' : 'Play'}${slot}`;
        item.button.setAttribute('aria-label', item.nameFor ? `${active ? 'Pause' : 'Play'}${slot} ${item.nameFor}` : `${active ? 'Pause' : 'Play'}${slot} ${item.roleName}, option ${item.label}`);
        if (item.host?.dataset && item.host.dataset.wide !== String(loaded)) item.host.dataset.wide = String(loaded);
        // Real playback only: position and duration come from the one media element.
        const total = loaded && Number.isFinite(media.duration) && media.duration > 0 ? media.duration : finite(item.duration);
        const position = loaded ? finite(media.currentTime) : 0;
        const progress = total > 0 ? Math.min(100, position / total * 100).toFixed(1) : '0.0';
        const failed = loaded && !!media.error;
        const loading = loaded && !failed && !media.paused && media.networkState === 2 && media.readyState < 3;
        const stamp = `${loaded}|${active}|${failed}|${loading}|${time(position)}|${time(total)}|${progress}`;
        if (stamp === item.lastProgress) continue;
        item.lastProgress = stamp;
        item.card.dataset.playing = String(active && !loading); item.card.dataset.loaded = String(loaded);
        item.card.dataset.audioLoading = String(loading); item.card.dataset.audioFailed = String(failed);
        item.timeNode.textContent = failed ? 'Could not load · tap Play to retry' : loading ? 'Loading audio…' : `${time(position)} / ${time(total)}`;
        if (item.wave.style && typeof item.wave.style.setProperty === 'function') item.wave.style.setProperty('--p', `${progress}%`);
        else setStyleVars(item.wave, { ...item.waveVars, '--p': `${progress}%` });
        if (item.progress) {
          if (total > 0) {
            item.progress.setAttribute('aria-valuemax', String(Number(total.toFixed(1))));
            item.progress.setAttribute('aria-valuenow', String(Number(Math.min(position, total).toFixed(1))));
            item.progress.setAttribute('aria-valuetext', `${time(position)} of ${time(total)}`);
          } else {
            item.progress.removeAttribute('aria-valuemax'); item.progress.removeAttribute('aria-valuenow');
            item.progress.setAttribute('aria-valuetext', 'Duration unknown');
          }
          item.progress.dataset.active = String(active && !loading);
        }
      }
    }
    function setCastingFlash(message) {
      const flash = $('casting-flash') || $('casting-status');
      flash.textContent = message;
      if (flash === $('casting-flash')) flash.hidden = !message;
    }
    async function requestCastingRound(role, requestedIntent) {
      const control = castingRoundControls.get(role.role_id);
      const state = control?.state || castingRoundState(role);
      if (state.busy || castingRoundBusy.has(role.role_id)) return;
      if (state.refreshRequired) {
        const fresh = await refreshCasting(false);
        if (fresh) { state.refreshRequired = false; return; }
        if (control?.button) { control.button.disabled = false; control.button.textContent = 'Refresh character'; }
        if (control?.statusNode) control.statusNode.textContent = 'The request was not sent again. Refresh casting status before trying another action.';
        return;
      }
      if (castingRoleBusy.has(role.role_id) || hasPendingFeedback(role)) {
        const reason = 'Retry or finish the saved feedback request for this character before requesting new audio.';
        if (control?.statusNode) control.statusNode.textContent = reason;
        setCastingFlash(reason); return;
      }
      if (!state.pendingRequest && ['pending', 'running'].includes(role.round_status)) return;
      const body = state.pendingRequest ? clone(state.pendingRequest) : {
        request_id: uuid(), role_id: role.role_id,
        expected_round_revision: role.round_revision,
        expected_assignment_revision: role.assignment_revision,
        intent: requestedIntent
      };
      state.pendingRequest = clone(body); state.busy = true;
      persistCastingRoundState(role, state); castingRoundBusy.add(role.role_id); castingWriteGeneration++;
      updateRoleCastingControls(role.role_id); updateCastingFeedbackControls();
      if (control?.button) { control.button.disabled = true; control.button.textContent = 'Sending request…'; }
      if (control?.statusNode) control.statusNode.textContent = 'Sending this character’s exact request…';
      try {
        const result = await client.request('casting/round', body);
        state.pendingRequest = null; state.busy = false; persistCastingRoundState(role, state);
        const updatedRole = result.casting?.roles?.find(item => item.role_id === role.role_id);
        if (castingActive?.role.role_id === role.role_id && updatedRole
          && updatedRole.round_revision !== castingActive.role.round_revision) {
          castingActive = null; castingDirectionOpen = false; $('casting-feedback').hidden = true;
        }
        castingEnabled = true; data.casting_enabled = true;
        data.capabilities = { ...(data.capabilities || {}), casting_enabled: true };
        renderVoices();
        if (result.casting) { casting = result.casting; renderCasting(); }
        else await refreshCasting(false);
        const reason = result.reason || result.receipt?.reason || (body.intent === 'none_fit'
          ? 'A fresh set of voices was requested. Existing ratings and the chosen voice were left unchanged.'
          : 'Audio generation was requested for this character.');
        setCastingFlash(reason);
        return result;
      } catch (error) {
        if (error.status === 409) {
          state.pendingRequest = null; state.busy = false; persistCastingRoundState(role, state);
          state.refreshRequired = true;
          setCastingFlash(error.body?.detail || error.body?.reason || error.message || 'The character changed before this request. Refresh and try again.');
          const fresh = await refreshCasting(false);
          if (fresh) state.refreshRequired = false;
          else {
            if (control?.button) { control.button.disabled = false; control.button.textContent = 'Refresh character'; }
            if (control?.statusNode) control.statusNode.textContent = 'Could not refresh the character after a revision conflict. Use Refresh character before trying again.';
          }
          return null;
        }
        if ([404, 405, 422].includes(error.status)) {
          state.pendingRequest = null; state.busy = false; persistCastingRoundState(role, state);
          state.unsupported = true; persistCastingRoundState(role, state);
          if (control?.button) { control.button.disabled = true; control.button.textContent = 'Unavailable'; }
          if (control?.statusNode) control.statusNode.textContent = 'This Studio session does not support on-demand character audio. Restart Studio and the private review service to load the updated casting controls.';
          setCastingFlash('On-demand character audio is not available in this Studio session. Restart Studio and the private review service to load the updated casting controls.');
          return null;
        }
        state.busy = false;
        if (control?.button) { control.button.disabled = false; control.button.textContent = 'Retry request'; }
        if (control?.statusNode) control.statusNode.textContent = `Request not confirmed. Retry uses the same request ID. ${error.message}`;
        setCastingFlash(`Request not confirmed. Retry is safe and will reuse the exact request. ${error.message}`);
        return null;
      } finally {
        state.busy = false; castingRoundBusy.delete(role.role_id);
        updateRoleCastingControls(role.role_id); updateCastingFeedbackControls();
      }
    }
    // A voice counts as chosen while a valid approval stands, even if a later round was reopened
    // for the character (which reads NEEDS_REVIEW): only the book's own count decides what is left.
    const castingFavoriteId = item => `${item?.key || ''}|${item?.audio_sha256 || ''}`;
    function castingIsFavorite(item) { return item?.favorite === true; }
    function syncCastingFavorites(id) { for (const sync of castingFavoriteSyncs.get(id) || []) sync(); }
    const castingStateLabel = c => candidateReady(c) ? 'Ready'
      : ['blocked', 'failed', 'unavailable', 'source_changed'].includes(c?.status) ? 'Failed'
        : ['running', 'generating'].includes(c?.status) ? 'Generating'
          : ['pending', 'queued'].includes(c?.status) ? 'Queued' : 'Missing';
    // Heart: confirmed server write only. Never sends selection, stars or a round request, and never touches the assignment.
    async function toggleCastingFavorite(audio, next, note, sync) {
      const id = castingFavoriteId(audio);
      if (!audio.key || !audio.audio_sha256 || castingFavoriteBusy.has(id)) return false;
      let confirmed = false;
      castingWriteGeneration++; castingFavoriteBusy.add(id); castingFavoriteNotes.set(id, 'Saving favorite.'); syncCastingFavorites(id);
      try {
        const result = await client.request('preferences', { key: audio.key, audio_sha256: audio.audio_sha256, favorite: next });
        const saved = typeof result?.favorite === 'boolean' ? result.favorite : next;
        for (const role of casting?.roles || []) {
          for (const row of role.candidates || []) if (row.key === audio.key && row.audio_sha256 === audio.audio_sha256) row.favorite = saved;
          if (role.current_audio && role.current_audio.key === audio.key && role.current_audio.audio_sha256 === audio.audio_sha256) role.current_audio.favorite = saved;
        }
        for (const voice of data.voices || []) if (voice.key === audio.key && voice.audio_sha256 === audio.audio_sha256) voice.favorite = saved;
        castingFavoriteNotes.set(id, saved ? 'Favorite saved.' : 'Favorite removed.');
        confirmed = true;
      } catch (error) { castingFavoriteNotes.set(id, `Favorite not saved. ${error.message}`); }
      finally { castingWriteGeneration++; castingFavoriteBusy.delete(id); syncCastingFavorites(id); }
      return confirmed;
    }    function useCastingSelection(role, candidate, statusNode) {
      activateCasting(role, candidate, false);
      return submitCastingFeedback(role, candidate, { stars: null, rawDirection: '', action: 'use', selectionConfirmed: true }, statusNode, { preserveDraftDirection: true });
    }
    async function playCastingExact(role, displayName, audio, candidate, name) {
      const context = { role_id: role.role_id, candidate_id: candidate ? candidate.candidate_id : null, key: audio.key, audio_sha256: audio.audio_sha256, round_revision: role.round_revision, assignment_revision: role.assignment_revision };
      const loaded = player.current?.casting_context;
      const same = !!loaded && loaded.role_id === context.role_id && loaded.candidate_id === context.candidate_id && loaded.key === context.key && loaded.audio_sha256 === context.audio_sha256 && !player.media.error;
      if (same) { await player.toggle(); updateCastingPlayLabels(); updateTime(); return; }
      const exact = clone(audio);
      if (candidate) activateCasting(role, exact, false);
      $('casting-feedback').hidden = true;
      await loadRecording({ ...exact, name: `${displayName} · ${name}`, casting_context: context }, { autoplay: true, seconds: 0, gain: 1 });
      updateCastingPlayLabels(); updateTime();
    }
    let castingPendingFocus = null; let castingActiveTransition = null;
    const vcEnc = s => { const t = String(s ?? ''); const out = []; for (let i = 0; i < t.length; i++) out.push(t.charCodeAt(i).toString(16)); return out.join('-'); };
    const vcId = parts => 'vc_' + parts.map(vcEnc).join('_');
    const voiceFocusKey = (role, c, action) => `${castingFavoriteId(c)}|${c.candidate_id}|${role.role_id}|${action}`;
    // Cosmetic only: update() (business render) runs exactly once on every route; never retries or delays writes.
    function voiceTransition(update) {
      let done = false, failure = null, reported = false;
      const report = e => { if (reported) return; reported = true; void run(() => { throw e; }); };
      const once = () => { if (done) return; done = true; try { return update(); } catch (e) { failure = e; throw e; } };
      let reduced = false; try { reduced = !!win.matchMedia?.('(prefers-reduced-motion: reduce)')?.matches; } catch (_) { reduced = false; }
      if (reduced || typeof doc.startViewTransition !== 'function') return once();
      try { castingActiveTransition?.skipTransition?.(); } catch (_) { /* animation only */ }
      let t;
      try { t = doc.startViewTransition(once); } catch (e) { if (failure) throw failure; if (!done) return once(); return; }
      castingActiveTransition = t;
      if (!t) return once();
      const guard = p => Promise.resolve(p).catch(() => {
        if (!done) { try { once(); } catch (e) { report(e); } }
        else if (failure) report(failure);
      });
      guard(t.ready); guard(t.updateCallbackDone);
      Promise.resolve(t.finished).catch(() => { if (!done) { try { once(); } catch (e) { report(e); } } else if (failure) report(failure); }).then(() => { if (castingActiveTransition === t) castingActiveTransition = null; });
    }
    function contextPanel(title, contentNodes, safe) {
      const probe = node('div', null, 'voice-candidate-panel');
      const native = typeof probe.showPopover === 'function' && typeof probe.hidePopover === 'function';
      if (!native) {
        const details = node('details', null, 'voice-candidate-panel voice-candidate-panel-fallback'); details.dataset.panelMode = 'details';
        details.append(node('summary', title, 'voice-candidate-more'), ...contentNodes);
        return { trigger: null, panel: details };
      }
      const panel = probe; panel.id = `voice-panel-${safe}`; panel.dataset.panelMode = 'popover'; panel.dataset.positionFallback = 'false';
      panel.setAttribute('popover', 'auto'); panel.setAttribute('role', 'dialog'); panel.setAttribute('aria-label', title);
      const anchor = `--voice-anchor-${safe}`;
      let returnFocus = false, listening = false;
      const place = () => {
        panel.dataset.positionFallback = 'false';
        let r = null; try { r = panel.getBoundingClientRect?.(); } catch (_) { r = null; }
        if (!r) return;
        const w = win.innerWidth || 0, h = win.innerHeight || 0;
        if (r.left < 8 || r.right > w - 8 || r.top < 8 || r.bottom > h - 8) panel.dataset.positionFallback = 'true';
      };
      const listen = on => {
        try {
          if (on && !listening) { win.addEventListener?.('resize', place); listening = true; }
          else if (!on && listening) { win.removeEventListener?.('resize', place); listening = false; }
        } catch (_) { listening = false; }
      };
      const trigger = button(title, () => {
        let open = false;
        try { open = panel.matches(':popover-open'); } catch (_) { open = trigger.getAttribute('aria-expanded') === 'true'; }
        if (open) { returnFocus = true; return panel.hidePopover(); }
        try { panel.showPopover({ source: trigger }); } catch (_) { panel.showPopover(); }
      }, 'voice-candidate-more');
      trigger.setAttribute('aria-haspopup', 'dialog'); trigger.setAttribute('aria-controls', panel.id); trigger.setAttribute('aria-expanded', 'false');
      trigger.style?.setProperty?.('anchor-name', anchor); panel.style?.setProperty?.('position-anchor', anchor);
      const heading = node('h5', title); heading.setAttribute('tabindex', '-1');
      panel.addEventListener('toggle', event => {
        const open = event.newState === 'open';
        trigger.setAttribute('aria-expanded', String(open));
        if (open) { place(); listen(true); heading.focus?.({ preventScroll: true }); }
        else { panel.dataset.positionFallback = 'false'; listen(false); if (returnFocus) { returnFocus = false; trigger.focus?.(); } }
      });
      panel.addEventListener('keydown', event => { if (event.key === 'Escape') returnFocus = true; });
      const head = node('div', null, 'voice-candidate-panel-head');
      const close = button('Close', () => { returnFocus = true; panel.hidePopover(); }, 'voice-candidate-close'); close.setAttribute('aria-label', `Close ${title}`);
      head.append(heading, close);
      const body = node('div', null, 'voice-candidate-panel-body'); body.append(...contentNodes);
      panel.append(head, body);
      return { trigger, panel };
    }
    function createVoiceCandidate({ role, displayName, audio, candidate, name, family, state, ready, context = 'audition' }) {
      const host = node('div', null, 'voice-candidate-host');
      const identity = vcId([role.role_id, candidate ? 'candidate' : 'current', candidate ? candidate.candidate_id : '', audio.key || '', audio.audio_sha256 || '']);
      host.voiceSyncHooks = new Set();
      host.dataset.voiceIdentity = identity;
      host.dataset.voiceContext = context; host.dataset.voiceRoleId = String(role.role_id ?? '');
      host.dataset.voiceCandidateKind = candidate ? 'candidate' : 'current_audio';
      if (candidate?.candidate_id) host.dataset.voiceCandidateId = String(candidate.candidate_id);
      if (audio.key) host.dataset.voiceKey = String(audio.key);
      if (audio.audio_sha256) host.dataset.voiceAudioSha256 = String(audio.audio_sha256);
      const circle = node('article', null, `${candidate ? 'casting-candidate ' : ''}casting-circle voice-candidate`);
      circle.dataset.state = state.toLowerCase(); circle.dataset.playing = 'false'; circle.dataset.selected = 'false'; circle.dataset.family = family;
      circle.dataset.tier = candidate ? castingTier(candidate) : 'plain';
      if (candidate) {
        circle.dataset.candidateId = candidate.candidate_id; circle.dataset.key = candidate.key || '';
        circle.dataset.audioState = ready ? 'ready' : ['blocked', 'failed', 'unavailable', 'source_changed'].includes(candidate.status) ? 'unavailable' : ['pending', 'running', 'generating', 'queued'].includes(candidate.status) ? 'generating' : 'none';
      }
      const playName = candidate ? `${name}, option ${candidate.label || 'Option'}, ${displayName}` : `${name}, ${displayName}`;
      circle.setAttribute('aria-label', `${playName}: ${state}`);
      const face = node('img', null, 'casting-circle-img'); face.src = castingPortraitUri(family, candidate ? name : '', role.role_id); face.alt = `Generic voice portrait for ${name}`;
      const nameNode = node('h4', name, 'casting-circle-name');
      const stateNode = node('span', state, 'casting-circle-state'); stateNode.dataset.state = state.toLowerCase();
      const timeNode = node('span', ready ? time(audio.duration_seconds) : '', 'casting-circle-time casting-time');
      let vtOk = false; try { vtOk = !!win.CSS?.supports?.('view-transition-name', 'none'); } catch (_) { vtOk = false; }
      if (vtOk) [[host, 'root'], [circle, 'bg'], [face, 'portrait'], [nameNode, 'name'], [stateNode, 'status']].forEach(([el, part]) => el.style?.setProperty?.('view-transition-name', `${identity}_${part}`));
      const exactVoice = (data.voices || []).find(v => v.key === audio.key && v.audio_sha256 === audio.audio_sha256);
      const processed = (exactVoice?.details?.provenance?.preprocessing || []).some(step => step.operation === 'reference_vocals_isolation');
      const sourceNote = node('span', processed ? 'Processed reference · may affect timbre' : '', 'casting-circle-note casting-circle-source-note');
      const prior = candidate?.prior_evidence || null;
      const engine = candidate?.engine ?? audio.engine ?? exactVoice?.engine;
      const meta = node('span', '', 'voice-candidate-meta');
      const badges = node('div', null, 'voice-candidate-badges');
      const fill = node('span', null, 'voice-candidate-progress-fill');
      const progress = node('span', null, 'voice-candidate-progress'); progress.append(fill);
      if (ready) { progress.setAttribute('role', 'progressbar'); progress.setAttribute('aria-label', `Playback progress for ${playName}`); progress.setAttribute('aria-valuemin', '0'); }
      else progress.setAttribute('aria-hidden', 'true');
      const note = node('span', '', 'casting-circle-note'); note.setAttribute('role', 'status'); note.setAttribute('aria-live', 'polite');
      const favoriteNote = node('span', '', 'casting-circle-note casting-circle-favorite-note'); favoriteNote.setAttribute('role', 'status'); favoriteNote.setAttribute('aria-live', 'polite');
      const reason = ready ? '' : `Unavailable: ${state}${candidate?.reason ? `, ${candidate.reason}` : ''}`;
      const favSource = candidate ? ((role.candidates || []).find(item => item.candidate_id === candidate.candidate_id && item.key === candidate.key) || candidate) : (role.current_audio || audio);
      const stateKey = candidate ? castingCandidateKey(role, candidate) : '';
      const favoriteId = castingFavoriteId(audio);
      const focusId = `${favoriteId}|${candidate ? candidate.candidate_id : 'current'}|${role.role_id}`;
      const focusables = {};
      const isSelected = () => !!candidate && role.selected_candidate_id === candidate.candidate_id && !!role.current_key && role.current_key === candidate.key;
      const isAssigned = () => !!candidate && role.status === 'APPROVED' && role.current_key === candidate.key;
      const play = button('Play', () => playCastingExact(role, displayName, audio, candidate, name), 'casting-play casting-circle-play');
      host.voicePlay = play;
      play.disabled = !ready; play.setAttribute('aria-label', ready ? `Play ${playName}` : `Play ${playName} unavailable. ${reason}`);
      if (ready) castingPlayButtons.set(`${role.role_id}:${candidate ? candidate.candidate_id : 'current'}:${audio.key}:${audio.audio_sha256}:circle`, { button: play, card: circle, host, slot: '', wave: fill, waveVars: {}, progress, timeNode, duration: audio.duration_seconds, lastProgress: '', roleId: role.role_id, roleName: displayName, nameFor: playName, candidateId: candidate ? candidate.candidate_id : null, key: audio.key, audioSha256: audio.audio_sha256, label: name });
      else play.setAttribute('title', reason);
      const heart = button('♡', async () => {
        const saved = await toggleCastingFavorite(audio, !castingIsFavorite(favSource), favoriteNote, sync);
        if (saved === true) { castingPendingFocus = `${focusId}|heart`; voiceTransition(() => renderCasting()); }
      }, 'casting-circle-btn casting-heart');
      focusables.heart = heart;
      const check = candidate ? button('✓', () => {
        if (isSelected()) return;
        const saved = castingCardStates.get(stateKey);
        if (saved?.pendingRequest?.selection_confirmed === true) return retryCastingFeedback(role, candidate, note);
        return useCastingSelection(role, candidate, note);
      }, 'casting-circle-btn casting-check') : null;
      focusables.check = check;

      let updateReview = () => {};
      const actions = node('div', null, 'voice-candidate-actions');
      const line = (text, cls = 'muted small') => node('p', text, cls);
      const list = rows => { const ul = node('ul', null, 'casting-traits-list'); rows.forEach(row => ul.append(node('li', row))); return ul; };
      const section = (label, ...nodes) => { const box = node('section', null, 'casting-back-section'); box.append(node('p', label, 'casting-back-label'), ...nodes); return box; };
      const asText = value => typeof value === 'string' ? value : JSON.stringify(value);
      const BADGE_CLASS = { assigned: 'casting-chosen-badge', replacement: 'casting-replacement-badge', reuse: 'casting-reuse-badge' };
      const badgeData = () => {
        const rows = [];
        if (isAssigned()) rows.push(['Assigned voice', 'This exact audio is the approved voice for this character.', 'assigned']);
        if (castingIsFavorite(favSource)) rows.push(['Favorite', 'Saved to your server favorites for this exact audio.', 'favorite']);
        const saved = Number(candidate?.rating);
        if (Number.isInteger(saved) && saved >= 1 && saved <= 5) rows.push([`Saved ${saved}/5`, 'Your saved rating for this character’s option.', 'rating']);
        const projects = [...new Set(prior?.projects || [])];
        if (projects.length) rows.push([`Reused in ${projects.length} ${projects.length === 1 ? 'project' : 'projects'}`, 'Prior approval evidence lists these distinct projects. This is evidence of reuse, not a popularity ranking; no usage count was reported.', 'reuse']);
        if (candidate?.needs_replacement) rows.push([`Replacement requested (${candidate.needs_replacement.rating} stars)`, `A replacement was requested after ${candidate.needs_replacement.rating} stars.`, 'replacement']);
        if (candidate && loadShortlist(role).has(candidate.candidate_id)) rows.push(['On shortlist', 'Kept on this device only. It does not rate or assign.', 'shortlist']);
        if (processed) rows.push(['Processed reference', 'Reference vocals were isolated first, which may affect timbre.', 'processed']);
        return rows;
      };
      const badgeExplain = node('div', null, 'voice-candidate-badge-explain');

      const why = [];
      const readiness = node('p', candidate ? audioReadiness(candidate) : ready ? `Audio ready · ${time(audio.duration_seconds)}` : 'Audio unavailable', 'muted small casting-readiness');
      let identityRow = null;
      if (candidate) {
        identityRow = node('p', null, 'muted small casting-candidate-code');
        identityRow.append(node('span', candidate.label ? `Option ${candidate.label}` : 'Option', 'casting-option-label'), ' · ', node('span', castingCode(candidate.candidate_id, family), 'casting-code'));
      }
      why.push(section('Identity', ...(candidate ? [identityRow, readiness] : [readiness])));
      const groups = castingTraits(candidate);
      const supportedGroup = groups.find(([label]) => /^Book-supported/.test(label));
      const inferredGroup = groups.find(([label]) => /^Casting inference/.test(label));
      const traitBox = group => { const box = node('div', null, 'casting-traits'); box.append(node('p', group[0], 'casting-traits-label'), list(group[1])); return box; };
      if (!candidate) why.push(line('This is the current sample, not a casting candidate. Rating, direction and comparison are unavailable for it.'));
      why.push(section('Why this voice', supportedGroup ? traitBox(supportedGroup) : line('No book-supported traits are recorded for this voice.'), inferredGroup ? traitBox(inferredGroup) : line('No casting inference is recorded for this voice.')));
      const evidence = castingEvidence(candidate);
      const evidenceBox = node('div', null, 'casting-evidence');
      if (evidence.length) evidence.forEach(text => evidenceBox.append(node('p', text, 'casting-evidence-line')));
      else evidenceBox.append(line('Performer match unavailable: no performer evidence was reported.'));
      why.push(section('Performer match and evidence', evidenceBox));
      const acoustics = castingAcoustics(candidate);
      const modelBox = node('div', null, 'casting-acoustics'); modelBox.append(line(`Model/engine: ${engine ? String(engine) : 'unknown'}`));
      if (acoustics.length) acoustics.forEach(row => { const p = node('p', row.text, 'casting-acoustics-line'); if (row.verdict) p.dataset.verdict = row.verdict; modelBox.append(p); });
      else modelBox.append(line('No generation settings or conditioning were reported for this voice.'));
      why.push(section('Model and settings', modelBox));
      const profile = castingSoundProfile(candidate);
      if (profile.length) { const chips = node('div', null, 'casting-profile'); profile.forEach(item => chips.append(node('span', item, 'casting-profile-chip'))); why.push(section('Sound profile', chips)); }
      const pron = candidate?.pronunciation ?? role.pronunciation;
      const pronRows = Array.isArray(pron) ? pron.map(asText) : pron ? [asText(pron)] : [];
      why.push(section('Pronunciation', pronRows.length ? list(pronRows) : line('No pronunciation notes were reported.')));
      const dir = role.direction || {};
      const dirRows = Object.entries(dir).filter(([, value]) => Array.isArray(value) && value.length).map(([key, value]) => `${capitalize(key)}: ${value.map(asText).join('; ')}`);
      const adjust = (Array.isArray(role.history) ? role.history : []).map(asText);
      why.push(section('Direction and adjustment history', ...(dirRows.length || adjust.length ? [list([...dirRows, ...adjust])] : [line('No direction or adjustment history for this character.')])));
      const ratingNumber = node('p', '', 'muted small casting-rating-number');
      const histCurrent = node('p', '', 'muted small casting-history-current');
      const histPrior = node('p', prior ? `Prior: rated ${prior.prior_rating} stars${prior.highest_rating ? ` (highest ${prior.highest_rating})` : ''}` : 'Prior: no earlier rating recorded.', 'muted small casting-history-prior');
      if (candidate) {
        const historyRows = [];
        if (prior) {
          if (prior.previously_used) historyRows.push(`Previously used in ${prior.previously_used}`);
          if ((prior.projects || []).length) historyRows.push(`Projects: ${prior.projects.join(', ')}`);
          if (prior.library_voice_id) historyRows.push(`Voice identity ${prior.library_voice_id}${prior.library_version ? ` · version ${String(prior.library_version).slice(0, 10)}` : ''}`);
        }
        if (candidate.needs_replacement) historyRows.push(`Replacement requested after ${candidate.needs_replacement.rating} stars${candidate.needs_replacement.feedback ? `: “${candidate.needs_replacement.feedback}”` : ''}`);
        historyRows.push(`Candidate ${candidate.candidate_id}`);
        why.push(section('History', ratingNumber, histCurrent, histPrior, ...historyRows.map(row => line(row))));
      }
      why.push(section('Badges', badgeExplain));
      const technical = node('details', null, 'casting-technical'); technical.append(node('summary', 'Technical details'));
      if (candidate?.reason) technical.append(line(candidate.reason));
      if (candidate?.technical_qa != null) technical.append(line(`Technical QA: ${asText(candidate.technical_qa)}`));
      if (candidate?.technical_qa && typeof candidate.technical_qa === 'object') technical.append(node('pre', JSON.stringify(candidate.technical_qa, null, 2), 'small'));
      if (audio.duration_seconds != null) technical.append(line(`Audition duration: ${time(audio.duration_seconds)} · Playback at normal speed.`));
      if (processed) technical.append(line('Reference vocals were isolated before conditioning; this may affect timbre.'));
      if (technical.children.length > 1) why.push(technical);

      const reviewNodes = [];
      let cs = null, cardStatus = null;
      if (candidate) {
        const persisted = loadCastingDraft(role, candidate).draft;
        cs = castingCardStates.get(stateKey) || { status: persisted.pendingRequest ? 'failed' : candidate.rating == null ? 'unrated' : 'saved', pendingRequest: persisted.pendingRequest || null };
        castingCardStates.set(stateKey, cs);
        cardStatus = node('p', cs.pendingRequest ? 'Save not confirmed. Retry the exact rating.'
          : cs.status === 'historical' ? (cs.reason || 'Saved as historical feedback. A newer casting decision remains current.')
            : candidate.rating == null ? 'Not rated yet.' : candidate.needs_replacement ? 'Feedback saved · New option after this review' : `Saved ${candidate.rating} stars.`, 'casting-rating-status');
        cardStatus.setAttribute('role', 'status'); cardStatus.setAttribute('aria-live', 'polite');
        const optionLabel = candidate.label || 'Option';
        const retry = button('Retry save', () => retryCastingFeedback(role, candidate, cardStatus), 'casting-retry'); retry.hidden = !cs.pendingRequest;
        const stars = node('div', null, 'casting-stars'); stars.setAttribute('role', 'group'); stars.setAttribute('aria-label', `Rate ${displayName}, option ${optionLabel}`);
        const starButtons = [];
        for (let value = 1; value <= 5; value++) {
          const star = button(value <= (candidate.rating || 0) ? '★' : '☆', () => { if (doc.activeElement === star) host.voiceRatingFocus = star; return saveCastingRating(role, candidate, value, cardStatus); }, 'casting-star');
          star.setAttribute('aria-label', `Rate ${value} stars for ${displayName}, option ${optionLabel}`);
          star.setAttribute('aria-pressed', String(candidate.rating === value)); starButtons.push(star); stars.append(star);
        }
        const addDirection = button('Add direction', () => openCastingDirection(role, candidate), 'casting-add-direction');
        const use = button('Use this voice', () => useCastingCandidate(role, candidate, cardStatus), 'casting-use-option');
        const chipRow = node('div', null, 'casting-quick'); chipRow.setAttribute('role', 'group'); chipRow.setAttribute('aria-label', `Quick direction for ${name}`);
        const chipButtons = QUICK_DIRECTIONS.map(phrase => { const chip = button(phrase, () => addDirectionPhrase(role, candidate, phrase), 'casting-chip'); chip.setAttribute('aria-label', `Direct ${name}: ${phrase}`); chipRow.append(chip); return chip; });
        const compareBtn = button('Compare', () => toggleCompare(role, candidate), 'casting-compare-toggle');
        const shortlistBtn = button('♡', () => toggleShortlist(role, candidate), 'casting-shortlist-toggle');
        const foot = node('div', null, 'casting-cardfoot'); foot.append(compareBtn, shortlistBtn, addDirection, use);
        focusables.compare = compareBtn; focusables.shortlist = shortlistBtn;
        reviewNodes.push(stars, cardStatus, retry, chipRow, foot);
        updateReview = () => {
          const busy = castingCandidateBusy.has(stateKey) || castingRoundBusy.has(role.role_id);
          const pending = !!cs.pendingRequest; const ok = candidateReady(candidate);
          starButtons.forEach((star, index) => { star.disabled = !ok || busy || pending; star.textContent = index + 1 <= (candidate.rating || 0) ? '★' : '☆'; star.classList?.toggle('filled', index + 1 <= (candidate.rating || 0)); star.setAttribute('aria-pressed', String(candidate.rating === index + 1)); });
          const tier = castingTier(candidate);
          if (circle.dataset.tier !== tier) circle.dataset.reveal = tier;
          circle.dataset.tier = tier; circle.dataset.saving = String(busy); circle.dataset.assigned = String(isAssigned());
          circle.classList?.toggle('active', castingActive?.role.role_id === role.role_id && castingActive?.candidate.candidate_id === candidate.candidate_id && castingActive?.candidate.key === candidate.key);
          addDirection.disabled = !ok || busy || pending;
          chipButtons.forEach(chip => { chip.disabled = !ok || busy || pending; });
          use.hidden = !(Number(candidate.rating) >= 4) || isAssigned(); use.disabled = !ok || busy || pending;
          retry.hidden = !pending; retry.disabled = busy;
          const comparing = compareSelected(role).some(c => c.candidate_id === candidate.candidate_id);
          compareBtn.textContent = comparing ? '✓ Compare' : 'Compare'; compareBtn.setAttribute('aria-pressed', String(comparing)); circle.dataset.compare = String(comparing);
          compareBtn.disabled = !ok && !comparing;
          compareBtn.setAttribute('aria-label', `${comparing ? 'Remove' : 'Add'} ${name}, option ${optionLabel}, ${comparing ? 'from' : 'to'} the A/B comparison for ${displayName}`);
          const listed = loadShortlist(role).has(candidate.candidate_id);
          shortlistBtn.textContent = listed ? '♥' : '♡'; shortlistBtn.setAttribute('aria-pressed', String(listed)); circle.dataset.shortlisted = String(listed);
          shortlistBtn.setAttribute('aria-label', `${listed ? 'Remove' : 'Add'} ${name}, option ${optionLabel}, ${listed ? 'from' : 'to'} your shortlist for ${displayName}`);
          const n = Number(candidate.rating); const has = Number.isInteger(n) && n >= 1 && n <= 5;
          ratingNumber.textContent = has ? `Saved rating: ${n}/5` : 'Saved rating: not rated';
          histCurrent.textContent = has ? `Current: rated ${n}/5 in this project` : 'Current: not rated in this project';
        };
        const playFrom = seconds => run(async () => {
          const exact = clone(candidate);
          activateCasting(role, exact, false); $('casting-feedback').hidden = true;
          await loadRecording({ ...exact, name: `${displayName} · ${exact.label || 'Option'}`, casting_context: { role_id: role.role_id, candidate_id: exact.candidate_id, key: exact.key, audio_sha256: exact.audio_sha256, round_revision: role.round_revision, assignment_revision: role.assignment_revision } }, { autoplay: true, seconds, gain: 1 });
          updateCastingPlayLabels(); updateTime();
        });
        const cues = Array.isArray(candidate.sections) ? candidate.sections : [];
        if (cues.length) {
          const sectionHelp = node('p', `Section cues jump within this audition; feedback you submit still applies to candidate ${candidate.label || 'Option'}.`, 'muted small casting-section-help');
          const sectionList = node('details', null, 'casting-section-list'); sectionList.append(node('summary', `${cues.length} section cues`));
          cues.forEach((cue, index) => {
            const label = cue.label || cue.title || `Section ${index + 1}`;
            const hasStart = Number.isFinite(cue.start_seconds); const hasEnd = Number.isFinite(cue.end_seconds);
            const range = hasStart && hasEnd ? `${time(cue.start_seconds)}–${time(cue.end_seconds)}` : hasStart ? `starts ${time(cue.start_seconds)}` : 'time unavailable';
            const control = typeof cue.control_status === 'string' ? cue.control_status : cue.control_status?.status || cue.control_status?.state || '';
            const controlReason = typeof cue.control_status === 'object' ? cue.control_status?.reason : '';
            const controlText = control === 'unsupported' ? `Controls unsupported${controlReason ? `: ${controlReason}` : ''}` : control ? `Controls ${String(control).replaceAll('_', ' ')}` : '';
            const row = node('div', null, 'casting-section-cue');
            row.append(node('span', `${label} · ${[range, cue.technical_qa || null, controlText || null].filter(Boolean).join(' · ')}`, 'muted small'));
            const seek = button(`Play ${label}`, () => playFrom(hasStart ? cue.start_seconds : 0));
            seek.disabled = !candidateReady(candidate) || !candidate.audio_url || !hasStart;
            if (hasStart) seek.setAttribute('aria-label', `Play ${label} from ${time(cue.start_seconds)} in this audition`);
            row.append(seek); sectionList.append(row);
          });
          why.push(sectionHelp, sectionList);
        }
      }      const reviewPanel = candidate ? contextPanel('Rate and direct', reviewNodes, vcId([role.role_id, candidate.candidate_id, audio.key || '', audio.audio_sha256 || '', 'review'])) : null;
      const whyPanel = contextPanel('Why this voice', why, vcId([role.role_id, candidate ? candidate.candidate_id : '', audio.key || '', audio.audio_sha256 || '', candidate ? 'why' : 'why-current']));
      if (reviewPanel?.trigger) focusables.compare = focusables.shortlist = reviewPanel.trigger;
      actions.append(...[reviewPanel?.trigger, reviewPanel?.panel, whyPanel.trigger, whyPanel.panel].filter(Boolean));

      const sync = () => {
        if (host.isConnected === false && host.dataset.seen === 'true') { castingFavoriteSyncs.get(favoriteId)?.delete(sync); return; }
        updateReview();
        host.voiceSyncHooks.forEach(fn => fn());
        const fav = castingIsFavorite(favSource); const favOk = ready && !!audio.key && !!audio.audio_sha256;
        heart.textContent = fav ? '♥' : '♡'; heart.setAttribute('aria-pressed', String(fav));
        heart.disabled = !favOk || castingFavoriteBusy.has(favoriteId);
        const heartLabel = favOk ? `${fav ? 'Remove' : 'Add'} ${playName} ${fav ? 'from' : 'to'} favorites` : `Favorite ${playName} unavailable. ${reason || 'Verified audio is required.'}`;
        heart.setAttribute('aria-label', heartLabel); if (!favOk) heart.setAttribute('title', heartLabel);
        circle.dataset.favorite = String(fav);
        favoriteNote.textContent = castingFavoriteNotes.get(favoriteId) || '';
        const savedRating = Number(candidate?.rating);
        meta.textContent = [candidate ? (Number.isInteger(savedRating) && savedRating >= 1 && savedRating <= 5 ? `Saved rating ${savedRating}/5` : 'Not rated') : '', `Engine ${engine ? String(engine) : 'unknown'}`].filter(Boolean).join(' · ');
        const rows = badgeData();
        badges.replaceChildren(...rows.map(row => { const b = node('span', row[0], `voice-candidate-badge ${BADGE_CLASS[row[2]] || ''}`.trim()); b.dataset.badge = row[2]; return b; }));
        badgeExplain.replaceChildren(...(rows.length ? rows.map(row => line(`${row[0]}: ${row[1]}`)) : [line('No badges: no saved rating, favorite, assignment or reuse evidence yet.')]));
        if (!check) return;
        const selected = isSelected(); const saved = castingCardStates.get(stateKey);
        const busy = castingCandidateBusy.has(stateKey) || castingRoleBusy.has(role.role_id) || castingRoundBusy.has(role.role_id);
        const blocked = !!saved?.pendingRequest && saved.pendingRequest.selection_confirmed !== true;
        check.setAttribute('aria-pressed', String(selected)); circle.dataset.selected = String(selected);
        check.disabled = !ready || busy || blocked;
        check.setAttribute('aria-label', !ready ? `Choose ${playName} unavailable. ${reason}` : selected ? `${playName} is the chosen voice` : blocked ? `Choose ${playName} unavailable. Retry the saved rating request first` : saved?.pendingRequest ? `Retry saving ${playName} as the chosen voice` : `Choose ${playName} as the voice for ${displayName}`);
      };
      if (candidate) castingCardControls.set(stateKey, { roleId: role.role_id, statusNode: cardStatus, update: () => sync() });
      const favoriteSyncs = castingFavoriteSyncs.get(favoriteId) || new Set(); favoriteSyncs.add(sync); castingFavoriteSyncs.set(favoriteId, favoriteSyncs);
      sync();
      if (typeof castingPendingFocus === 'string' && castingPendingFocus.startsWith(`${focusId}|`)) {
        const target = focusables[castingPendingFocus.slice(focusId.length + 1)]; castingPendingFocus = null;
        Promise.resolve().then(() => target?.focus?.());
      }
      Promise.resolve().then(() => { if (host.isConnected) host.dataset.seen = 'true'; });
      const controls = node('div', null, 'casting-circle-controls'); controls.append(...[play, heart, check].filter(Boolean));
      circle.append(face, nameNode, controls, stateNode, meta, badges, timeNode, progress, sourceNote, note, favoriteNote, actions);
      host.append(circle);
      return host;
    }

    function castingCircleRow(role, candidates, nicknames, displayName, roleFamily, roleReady = 0) {
      const board = node('div', null, 'casting-circle-row voice-candidate-board'); board.setAttribute('role', 'group'); board.setAttribute('aria-label', `${displayName} voice options`);
      const make = options => createVoiceCandidate({ role, displayName, context: 'audition', ...options });
      const seenIds = new Set(); candidates = candidates.filter(c => !seenIds.has(c.candidate_id) && seenIds.add(c.candidate_id));
      const names = new Map(); const hosts = new Map();
      for (const candidate of candidates) {
        const name = nicknames.get(candidate.candidate_id) || castingNickname(candidate.candidate_id); names.set(candidate.candidate_id, name);
        hosts.set(candidate.candidate_id, make({ audio: candidate, candidate, name, family: castingFamily(candidate), state: castingStateLabel(candidate), ready: candidateReady(candidate) }));
      }
      let currentHost = null;
      const current = role.current_audio;
      if (current) {
        const audio = { ...current, duration_seconds: current.duration_seconds ?? current.duration };
        const verified = !!audio.key && !!audio.audio_sha256 && !!audio.audio_url && audio.audio_available === true && !!safeUrl(audio.audio_url, win.location);
        const duplicate = verified && candidates.some(item => item.key === audio.key && item.audio_sha256 === audio.audio_sha256);
        const original = audio.kind === 'original' || audio.source_provenance?.source_type === 'original_synthetic' || audio.provenance?.kind === 'original_synthetic';
        if (!duplicate) currentHost = make({ audio, candidate: null, name: verified ? (original ? 'Current original sample' : 'Current sample') : 'No candidate audio', family: roleFamily, state: verified ? 'Ready' : 'Missing', ready: verified });
      } else if (!candidates.length) currentHost = make({ audio: {}, candidate: null, name: 'No candidate audio', family: roleFamily, state: 'Missing', ready: false });
      if (!candidates.length) board.append(node('p', 'No options yet. Generate audio for this character when the on-demand control is available.', 'muted small'));
      else if (!roleReady) board.append(node('p', 'Audio is not ready to play yet. Check each option’s status below.', 'muted small'));
      const loaded = player.current?.casting_context;
      const pair = compareSelected(role).map(p => candidates.find(c => c.candidate_id === p.candidate_id) || p).filter(c => hosts.has(c.candidate_id));
      const comparing = pair.length === 2 && pair.every(candidateReady);
      const inPair = id => comparing && pair.some(c => c.candidate_id === id);
      const keeper = c => castingIsFavorite(c) || Number(c.rating) >= 4 || (role.status === 'APPROVED' && role.current_key === c.key);
      const collection = (kind, label) => { const box = node('section', null, 'voice-candidate-collection'); box.dataset.collection = kind; box.setAttribute('aria-label', label); box.append(node('h4', label, 'voice-candidate-collection-title')); return box; };
      if (comparing) {
        const scriptSig = c => c.script_hash || c.script_sha256 || c.text_hash || c.text_sha256 || (Array.isArray(c.sections) && c.sections.length && c.sections.every(s => typeof s.text === 'string') ? JSON.stringify(c.sections.map(s => [s.label || s.title || '', s.text])) : null);
        const sigA = scriptSig(pair[0]); const sameScript = !!sigA && sigA === scriptSig(pair[1]);
        const compareBox = node('div', null, 'casting-compare');
        const bar = node('div', null, 'casting-compare-bar');
        bar.append(node('span', sameScript ? `Comparing ${displayName}: A vs B, same script. Assignment stays explicit.` : `Comparing ${displayName}: A vs B. Compare delivery; script alignment is unverified. Assignment stays explicit.`, 'muted small'), button('Clear comparison', () => { castingCompareSelection.delete(role.role_id); voiceTransition(() => renderCasting()); }, 'quiet'));
        compareBox.append(bar);
        const view = node('div', null, 'casting-compare-view'); const preferred = loadPreference(role);
        pair.forEach((cand, slotIndex) => {
          const slotLabel = slotIndex === 0 ? 'A' : 'B'; const slotName = names.get(cand.candidate_id);
          const orig = candidates.find(c => c.candidate_id === cand.candidate_id) || cand;
          const slotNode = node('div', null, 'casting-compare-slot'); slotNode.dataset.slot = slotLabel.toLowerCase(); slotNode.dataset.preferred = String(preferred === cand.candidate_id);
          const host = hosts.get(cand.candidate_id); host.dataset.voiceContext = 'compare'; slotNode.append(host);
          const entry = castingPlayButtons.get(`${role.role_id}:${cand.candidate_id}:${cand.key}:${cand.audio_sha256}:circle`);
          if (entry) entry.slot = slotLabel;
          if (host.voicePlay) { host.voicePlay.classList?.add?.('casting-compare-play'); host.voicePlay.dataset.slot = slotLabel.toLowerCase(); }
          const reasonNode = node('p', '', 'muted small');
          const preferBtn = button(preferred === cand.candidate_id ? `✓ Preferred ${slotLabel}` : `Prefer ${slotLabel}`, () => togglePreference(role, cand), 'casting-prefer');
          preferBtn.setAttribute('aria-pressed', String(preferred === cand.candidate_id)); preferBtn.setAttribute('aria-label', `Prefer ${slotLabel}, ${slotName}. A note on this device only; it does not rate or assign.`);
          const assignSlot = button('Assign this voice', () => useCastingCandidate(role, orig, reasonNode), 'casting-assign');
          const gateKey = castingCandidateKey(role, orig); let lastGate = null;
          const gate = () => {
            const rated = Number(orig.rating) >= 4; const assigned = role.status === 'APPROVED' && role.current_key === orig.key;
            const busy = castingCandidateBusy.has(gateKey) || castingRoundBusy.has(role.role_id) || castingRoleBusy.has(role.role_id);
            const pending = !!castingCardStates.get(gateKey)?.pendingRequest;
            assignSlot.disabled = !rated || assigned || busy || pending || !candidateReady(orig); assignSlot.dataset.assigned = String(assigned);
            assignSlot.textContent = assigned ? 'Already assigned' : 'Assign this voice';
            const text = assigned ? 'This option is already the assigned voice.' : pending ? 'Retry the saved request before assigning.' : busy ? 'Saving… assignment is blocked until this finishes.' : rated ? '' : 'Save a 4- or 5-star rating on this option before it can be assigned.';
            if (text !== lastGate) { lastGate = text; reasonNode.textContent = text; }
          };
          host.voiceSyncHooks.add(gate); gate();
          const slotActions = node('div', null, 'casting-compare-actions'); slotActions.append(preferBtn, assignSlot);
          slotNode.append(slotActions, reasonNode); view.append(slotNode);
        });
        compareBox.append(view);
        const both = button('Keep both', () => { const set = loadShortlist(role); pair.forEach(row => set.add(row.candidate_id)); try { win.localStorage?.setItem(shortlistKey(role), JSON.stringify([...set])); } catch (_) { } voiceTransition(() => renderCasting()); }, 'casting-keep-both');
        both.setAttribute('aria-label', `Keep both ${displayName} options on your shortlist. Does not rate or assign.`);
        const foot = node('div', null, 'casting-compare-foot'); foot.append(both, node('p', 'Preferring and keeping are notes on this device. They never rate or assign a voice.', 'muted small casting-compare-note'));
        compareBox.append(foot); board.append(compareBox);
      }
      const keptRows = candidates.filter(c => !inPair(c.candidate_id) && keeper(c));
      const auditionRows = candidates.filter(c => !inPair(c.candidate_id) && !keeper(c));
      const audition = collection('audition', 'Auditions');
      const kept = collection('kept', 'Kept voices');
      for (const c of auditionRows) {
        const host = hosts.get(c.candidate_id); host.dataset.voiceContext = 'audition';
        host.dataset.wide = String(!!loaded && loaded.role_id === role.role_id && loaded.candidate_id === c.candidate_id && loaded.key === c.key && loaded.audio_sha256 === c.audio_sha256);
        audition.append(host);
      }
      if (currentHost) audition.append(currentHost);
      for (const c of keptRows) { const host = hosts.get(c.candidate_id); host.dataset.voiceContext = 'kept'; kept.append(host); }
      audition.hidden = audition.children.length <= 1; kept.hidden = kept.children.length <= 1;
      board.append(audition, kept);
      // A confirmed rating changes the collection immediately, preserving the exact host/card/player.
      // Atomic DOM moves preserve native focus/popover state. Older browsers restore those states explicitly.
      for (const c of candidates.filter(row => !inPair(row.candidate_id))) {
        const host = hosts.get(c.candidate_id), stateKey = castingCandidateKey(role, c);
        host.voiceSyncHooks.add(() => {
          if (castingCandidateBusy.has(stateKey) || castingRoleBusy.has(role.role_id) || castingRoundBusy.has(role.role_id)
            || castingCardStates.get(stateKey)?.pendingRequest) return;
          const next = keeper(c) ? 'kept' : 'audition';
          const currentFocus = doc.activeElement;
          const savedFocus = host.voiceRatingFocus;
          host.voiceRatingFocus = null;
          const active = host.contains?.(currentFocus) ? currentFocus
            : savedFocus && (!currentFocus || currentFocus === doc.body || currentFocus === doc.documentElement) ? savedFocus : null;
          const restoreFocus = !!active;
          if (host.dataset.voiceContext === next) { if (restoreFocus) active.focus?.({ preventScroll: true }); return; }
          voiceTransition(() => {
            const prior = host.dataset.voiceContext === 'kept' ? kept : audition, target = next === 'kept' ? kept : audition;
            target.hidden = false;

            let movedAtomically = false;
            if (typeof target.moveBefore === 'function' && host.parentNode && target.isConnected && host.isConnected) {
              try { target.moveBefore(host, null); movedAtomically = true; } catch (_) { /* ordinary DOM move below */ }
            }
            if (!movedAtomically) {
              const openPanels = typeof host.querySelectorAll === 'function' ? [...host.querySelectorAll('[popover]')].filter(panel => { try { return panel.matches(':popover-open'); } catch (_) { return false; } }) : [];
              // The minimal production-mount harness stores children in an array; real DOM append moves them itself.
              if (Array.isArray(prior.children)) prior.children = prior.children.filter(child => child !== host);
              target.append(host);
              for (const panel of openPanels) {
                try { if (!panel.matches(':popover-open')) panel.showPopover({ source: host.querySelector?.('[aria-controls="' + panel.id + '"]') }); } catch (_) { /* content remains available from its trigger */ }
              }
              if (restoreFocus) { const focus = () => active.focus?.({ preventScroll: true }); if (typeof win.setTimeout === 'function') win.setTimeout(focus, 0); else Promise.resolve().then(focus); }
            }
            if (movedAtomically && restoreFocus) active.focus?.({ preventScroll: true });
            host.dataset.voiceContext = next;
            prior.hidden = prior.children.length <= 1;
          });
        });
      }
      updateCastingPlayLabels();
      return board;
    }
    function castingChosen(role) { return role.status === 'APPROVED' || role.chosen === true; }
    function voiceList(rows) { const names = rows.map(castingRoleName); return names.length > 4 ? `${names.slice(0, 4).join(', ')} and ${names.length - 4} more` : names.join(', '); }
    function voiceChosenMessage(roleValue) {
      const left = (casting?.roles || []).filter(row => row.required && !castingChosen(row));
      return left.length ? `${castingRoleName(roleValue)} chosen · ${left.length} left: ${voiceList(left)}`
        : `${castingRoleName(roleValue)} chosen · that was the last voice, so your book starts recording by itself`;
    }
    // The server has confirmed the approval, so the character leaves the list now. Waiting for the
    // whole casting view to be fetched again left the card on screen for seconds (or for good, when
    // that fetch failed), and it was clicked again, which the server filed as historical feedback.
    function markVoiceChosen(roleId, candidateValue, result) {
      const current = (casting?.roles || []).find(item => item.role_id === roleId); if (!current) return;
      current.status = 'APPROVED'; current.chosen = true; current.current_key = candidateValue.key; current.selected_candidate_id = candidateValue.candidate_id;
      if (Number.isInteger(result.assignment_revision)) current.assignment_revision = result.assignment_revision;
      if (Number.isInteger(result.round_revision)) current.round_revision = result.round_revision;
      castingPendingFocus = voiceFocusKey(current, candidateValue, 'check');
      voiceTransition(() => renderCasting());
    }
    function renderCasting() {
      if (!casting) return;
      renderCharacterAudio();
      const roles = casting.roles || [];
      const visible = roles; // primary portrait view keeps chosen characters visible so their check stays visible
      const readyTotal = visible.reduce((sum, role) => sum + (role.candidates || []).filter(candidateReady).length, 0);
      const pipeline = casting.pipeline || {};
      const discovered = Number(pipeline.characters ?? visible.length) || 0;
      const attentionCount = Number(pipeline.needs_attention) || 0;
      const owed = roles.filter(role => role.required); const remaining = owed.filter(role => !castingChosen(role));
      const listening = `${readyTotal} ${readyTotal === 1 ? 'ready option' : 'ready options'} to play`;
      $('casting-status').textContent = !owed.length ? `${discovered} ${discovered === 1 ? 'character' : 'characters'} · ${listening}`
        : remaining.length ? `${owed.length - remaining.length} of ${owed.length} voices chosen · ${remaining.length} left: ${voiceList(remaining)} · ${listening}`
        : `All ${owed.length} voices chosen · your book starts recording by itself`;
      const pipelineNode = $('casting-pipeline');
      if (pipelineNode) {
        const parts = [pipeline.label || ''];
        if (attentionCount) parts.push(`${attentionCount} utterance${attentionCount === 1 ? '' : 's'} need speaker review`);
        if (pipeline.run) parts.push(`Auto Cast: ${String(pipeline.run.status || '').replaceAll('_', ' ')}`);
        pipelineNode.hidden = !parts.filter(Boolean).length;
        pipelineNode.textContent = parts.filter(Boolean).join(' · ');
      }
      const scopeError = typeof casting.scope_error === 'string' ? casting.scope_error : '';
      $('casting-scope-error').hidden = !scopeError;
      $('casting-scope-error').textContent = scopeError;
      const scope = casting.source_scope || casting.scope?.label || 'all current speaking utterances';
      $('casting-source-scope').textContent = `Preparation scope: ${scope}.`;
      const notifications = roles.flatMap(role => {
        const notice = role.notification;
        const state = notice?.state || notice?.status;
        if (!state) return [];
        const detail = notice.reason || (state === 'accepted' ? 'Provider accepted the notification; device delivery is unobserved.'
          : state === 'delivery_unknown' ? 'Provider acceptance is unknown; automatic resend is blocked.' : '');
        return [`${role.name || role.role_id}: ${String(state).replaceAll('_', ' ')}${detail ? ` · ${detail}` : ''}`];
      });
      $('casting-notification-status').textContent = notifications.join(' · ');
      const container = $('casting-roles'); container.classList?.add?.('casting-portrait-view'); container.replaceChildren(); castingFavoriteSyncs.clear();
      const root = $('voice-lab'); if (root) root.dataset.portraitCasting = 'true';
      const legacyAudio = $('character-audio'); if (legacyAudio && roles.some(role => Object.prototype.hasOwnProperty.call(role, 'current_audio'))) legacyAudio.hidden = true; castingCardControls.clear(); castingPlayButtons.clear(); castingRoundControls.clear();
      const jump = $('casting-jump'); if (jump) jump.replaceChildren();
      if (!visible.length) { container.append(node('p', roles.length ? 'All current speaking characters are approved.' : (pipeline.label || 'No speaking characters are in this project yet.'), 'casting-empty')); }
      for (const role of visible) {
        const displayName = castingRoleName(role);
        const candidates = castingRows(role);
        const roleReady = candidates.filter(candidateReady).length;
        const nicknames = castingNicknames(role);
        const roleFamily = candidates.map(castingFamily).find(value => value !== 'unknown') || 'unknown';
        const card = node('article', null, 'casting-role'); card.id = `casting-role-${role.role_id}`; card.dataset.roleId = role.role_id; card.dataset.status = role.status || ''; card.dataset.family = roleFamily; card.tabIndex = -1;
        const seenRound = castingSeenRounds.get(role.role_id);
        if (seenRound !== undefined && Number(role.round_revision) > seenRound) castingNewRound.add(role.role_id);
        castingSeenRounds.set(role.role_id, Number(role.round_revision) || 0);
        const description = String(role.description || '').trim();
        const hasUsefulDescription = description && !/^(source voice description is not yet established|no explicit vocal description in the current source horizon)\.?$/i.test(description);
        // Character identity strip: one recognizable face for the character,
        // its name, what the book says about it and how far casting has got.
        const roleHead = node('header', null, 'casting-role-head');
        const avatar = node('img', null, 'casting-avatar'); avatar.src = castingPortraitUri(roleFamily, '', role.role_id); avatar.alt = '';
        const identity = node('div', null, 'casting-identity');
        identity.append(node('h3', displayName));
        identity.append(node('p', hasUsefulDescription ? description : FAMILY_LABEL[roleFamily], 'casting-role-sub'));
        const roleMeta = node('p', `${candidates.length} ${candidates.length === 1 ? 'option' : 'options'} · ${roleReady} audio ready`, 'casting-option-meta');
        identity.append(roleMeta);
        const statusText = role.status === 'APPROVED' ? 'Voice chosen' : roleReady ? 'Ready to hear' : role.status === 'BLOCKED' ? 'Audio needs attention' : 'Waiting for audio';
        const statusNode = node('p', statusText, 'quality'); statusNode.dataset.status = role.status || '';
        identity.append(statusNode);
        roleHead.append(avatar, identity);
        const roundState = castingRoundState(role);
        const roundSupported = casting.capabilities?.round_requests === true;
        const roundActive = ['pending', 'running'].includes(role.round_status);
        const feedbackPending = hasPendingFeedback(role);
        const actionIntent = Number(role.round_revision || 0) > 0 || candidates.length ? 'none_fit' : 'generate';
        const actionLabel = roundState.pendingRequest ? 'Retry request' : roundActive ? 'Generating audio…'
          : actionIntent === 'none_fit' ? 'Try new voices' : 'Generate audio';
        const roundButton = button(actionLabel, () => requestCastingRound(role, actionIntent), 'casting-round-action');
        roundButton.hidden = !roundSupported;
        roundButton.disabled = roundState.busy || castingRoundBusy.has(role.role_id) || (roundActive && !roundState.pendingRequest) || feedbackPending;
        const roundHelp = roundState.pendingRequest ? 'Request status is not confirmed. Retry safely; the same request ID and content will be sent.'
          : roundActive ? 'Audio generation is in progress for this character.'
            : feedbackPending ? 'Retry the saved feedback request before changing this character’s audition round.'
              : actionIntent === 'none_fit' ? 'If these options do not fit, request one fresh set. Ratings and the chosen assignment stay separate.'
                : 'Generate this character’s audio options on demand. This does not rate or approve any voice.';
        const roundStatus = node('p', roundHelp, 'casting-round-request-status'); roundStatus.setAttribute('role', 'status'); roundStatus.setAttribute('aria-live', 'polite');
        const request = role.replacement_request || null;
        if (request && ['waiting_for_review', 'queued'].includes(request.status)) {
          roundStatus.textContent = request.status === 'queued'
            ? 'A replacement round was created from your rating and feedback.'
            : `${request.reason || 'A replacement is requested'}. Keep rating the options here; the replacement arrives as a new round.`;
        }
        const roundActions = node('div', null, 'casting-actions casting-role-actions'); roundActions.append(roundButton);
        roundActions.hidden = !roundSupported;
        if (!roundSupported || roundState.unsupported) roundStatus.textContent = 'On-demand character audio is not available in this Studio session. Restart Studio and the private review service to load the updated casting controls.';
        if (roundState.unsupported) { roundButton.disabled = true; roundButton.textContent = 'Unavailable'; }
        roleHead.append(roundActions);
        card.append(roleHead);
        // Background work is one compact strip, never a blocking screen.
        const jobs = [];
        if (request?.status === 'waiting_for_review') jobs.push(['waiting', 'Waiting for review']);
        else if (request?.status === 'queued') jobs.push(['queued', 'Queued']);
        if (roundActive) jobs.push(['generating', 'Generating']);
        if (['failed', 'blocked'].includes(role.round_status)) jobs.push(['failed', 'Failed']);
        if (request?.status === 'superseded' || role.round_status === 'superseded') jobs.push(['superseded', 'Superseded']);
        const jobStrip = node('div', null, 'casting-jobstrip');
        for (const [kind, label] of jobs) { const pill = node('span', label, 'casting-job-pill'); pill.dataset.job = kind; jobStrip.append(pill); }
        if (castingNewRound.has(role.role_id)) {
          const fresh = button(`New round ready · round ${role.round_revision}`, () => { castingNewRound.delete(role.role_id); renderCasting(); }, 'casting-job-pill casting-new-round');
          fresh.dataset.job = 'ready'; fresh.setAttribute('aria-label', `New round ${role.round_revision} is ready for ${displayName}. Dismiss this notice; nothing plays automatically.`);
          jobStrip.append(fresh);
        }
        jobStrip.hidden = !jobStrip.children.length;
        card.append(jobStrip, roundStatus);
        castingRoundControls.set(role.role_id, { role, state: roundState, button: roundButton, statusNode: roundStatus });
        if (jump) {
          const chip = node('a', `${displayName} · ${roleReady} ready`, 'casting-jump-link'); chip.href = `#casting-role-${role.role_id}`;
          chip.setAttribute('aria-label', `Jump to ${displayName}, ${roleReady} options ready`); jump.append(chip);
        }
        card.append(castingCircleRow(role, candidates, nicknames, displayName, roleFamily, roleReady));
        if (role.reason || description) {
          const details = node('details', null, 'casting-role-details'); details.append(node('summary', 'Character notes and review details'));
          if (description) details.append(node('p', description, 'muted small'));
          if (role.reason) details.append(node('p', role.reason, 'muted small'));
          card.append(details);
        }
        const direction = castingDirectionText(role.direction).map(line => node('div', line));
        if (direction.length) { const details = node('details'); details.append(node('summary', 'Direction status')); const box = node('div', null, 'casting-direction'); box.append(...direction); details.append(box); card.append(details); }
        if (Array.isArray(role.history) && role.history.length) { const details = node('details'); details.append(node('summary', 'Earlier review decisions')); for (const item of role.history) details.append(node('p', item.summary || item.action || item.status || 'Review event', 'muted small')); card.append(details); }
        container.append(card);
      }
      $('casting-feedback').hidden = !castingActive || !castingDirectionOpen;
      const scopeIds = casting.scope?.utterance_ids || casting.utterance_ids || (casting.productions || []).flatMap(row => row.utterance_ids || []);
      $('render-approved-cast').disabled = !Array.isArray(scopeIds) || !scopeIds.length;
      const resumable = (casting.productions || []).find(row => ['interrupted', 'failed', 'waiting', 'blocked', 'source_changed'].includes(row.status) && row.request_id);
      $('resume-cast-render').hidden = !resumable;
      if (resumable) $('resume-cast-render').dataset.requestId = resumable.request_id;
      const poll = (casting.productions || []).some(row => ['queued', 'running', 'pending', 'preparing', 'rendering'].includes(row.status))
        || roles.some(role => ['pending', 'running'].includes(role.round_status));
      // One poll at a time, and never while audio is playing or loading: the casting
      // view is several megabytes, and overlapping polls used to starve the audio
      // request in the single Studio process.
      if (poll && !castingTimer && win.setInterval) castingTimer = win.setInterval(() => {
        const media = player.media;
        if (castingPollBusy || win.document?.hidden || doc.documentElement?.dataset?.runtimeComparison === 'playing' || !media.paused || (player.current && media.networkState === 2 && media.readyState < 3)) return;
        castingPollBusy = true;
        void refreshCasting(false, true).finally(() => { castingPollBusy = false; });
      }, 6000);
      if (!poll && castingTimer && win.clearInterval) { win.clearInterval(castingTimer); castingTimer = null; }
      const hasReadyCandidate = visible.some(role => (role.candidates || []).some(candidateReady));
      const toolbar = $('casting-toolbar'); if (toolbar && !hasReadyCandidate) toolbar.open = true;
    }
    async function refreshCasting(showErrors = true, polling = false) {
      if (polling && (castingRoleBusy.size || castingRoundBusy.size || castingFavoriteBusy.size)) return casting;
      const writeGeneration = castingWriteGeneration;
      try {
        const fresh = await client.request('casting');
        if (polling && (castingRoleBusy.size || castingRoundBusy.size || castingFavoriteBusy.size || writeGeneration !== castingWriteGeneration)) return casting;
        // Stable polling must not replace focused controls or close details.
        if (polling && JSON.stringify(fresh) === JSON.stringify(casting)) return casting;
        if (casting && Number.isInteger(fresh.revision) && Number.isInteger(casting.revision) && fresh.revision < casting.revision) return casting;
        casting = fresh; renderCasting(); return casting;
      }
      catch (e) { if (showErrors) $('casting-status').textContent = `Casting review is unavailable: ${e.message}`; return null; }
    }
    function activateCasting(role, candidate, showDirection = false) {
      castingActive = { role: clone(role), candidate: clone(candidate) };
      castingDirectionOpen = showDirection;
      $('casting-feedback').hidden = !showDirection;
      const { draft } = loadCastingDraft(role, candidate);
      $('casting-feedback-role').textContent = `${castingRoleName(role)} · ${candidate.label || 'Option'}`;
      $('casting-stars').value = draft.stars == null ? '' : String(draft.stars);
      $('casting-direction-input').value = draft.raw_direction || '';
      $('casting-feedback-status').textContent = `Draft saved on this device for review round ${role.round_revision}.`;
      updateCastingFeedbackControls(); syncDirectionChips();
    }
    function openCastingDirection(role, candidate) {
      activateCasting(role, candidate, true);
      $('casting-feedback').hidden = false;
      $('casting-feedback').scrollIntoView?.({ behavior: 'smooth', block: 'center' });
      $('casting-direction-input').focus?.();
    }
    function syncDirectionChips() {
      const box = $('casting-chips'); if (!box) return;
      const chosen = new Set($('casting-direction-input').value.split(/[,;\n]+/).map(row => row.trim().toLowerCase()).filter(Boolean));
      for (const chip of box.children) chip.setAttribute('aria-pressed', String(chosen.has(chip.textContent.toLowerCase())));
    }
    function addDirectionPhrase(role, candidate, phrase) {
      if (!castingActive || castingActive.role.role_id !== role.role_id || castingActive.candidate.candidate_id !== candidate.candidate_id || !castingDirectionOpen) openCastingDirection(role, candidate);
      const input = $('casting-direction-input');
      const rows = input.value.split(/[,;\n]+/).map(row => row.trim()).filter(Boolean);
      const at = rows.findIndex(row => row.toLowerCase() === phrase.toLowerCase());
      if (at >= 0) rows.splice(at, 1); else rows.push(phrase);
      input.value = rows.join(', ');
      input.oninput?.(); syncDirectionChips();
    }
    function updateRoleCastingControls(roleId) {
      for (const control of castingCardControls.values()) if (control.roleId === roleId) control.update();
      const roundControl = castingRoundControls.get(roleId);
      const role = (casting?.roles || []).find(item => item.role_id === roleId) || roundControl?.role;
      if (!roundControl || !role) return;
      const active = ['pending', 'running'].includes(role.round_status);
      const pendingFeedback = hasPendingFeedback(role);
      roundControl.button.disabled = roundControl.state.busy || castingRoundBusy.has(roleId) || castingRoleBusy.has(roleId)
        || roundControl.state.unsupported || (active && !roundControl.state.pendingRequest) || pendingFeedback;
      if (roundControl.state.pendingRequest) roundControl.button.textContent = 'Retry request';
      else if (roundControl.state.unsupported) roundControl.button.textContent = 'Unavailable';
      else if (active) roundControl.button.textContent = 'Generating audio…';
      if (pendingFeedback) roundControl.statusNode.textContent = 'Retry the saved feedback request before changing this character’s audition round.';
    }
    function updateCastingFeedbackControls() {
      const hasStars = $('casting-stars').value !== '';
      const hasDirection = $('casting-direction-input').value.trim().length > 0;
      const ready = candidateReady(castingActive?.candidate);
      const busy = !!castingActive && (castingRoleBusy.has(castingActive.role.role_id) || castingRoundBusy.has(castingActive.role.role_id));
      $('casting-save').disabled = !ready || busy || (!hasStars && !hasDirection);
      $('casting-use').disabled = !ready || busy || !['4', '5'].includes($('casting-stars').value);
      $('casting-recast').disabled = !ready || busy || (!hasStars && !hasDirection);
      $('casting-preview-direction').disabled = !ready || busy || !hasDirection;
      const pending = castingActive && castingCardStates.get(castingCandidateKey(castingActive.role, castingActive.candidate))?.pendingRequest;
      $('casting-save').disabled ||= !!pending; $('casting-use').disabled ||= !!pending;
      $('casting-recast').disabled ||= !!pending; $('casting-preview-direction').disabled ||= !!pending;
    }
    function frozenFeedback(roleValue, candidateValue, { stars = null, rawDirection = '', action = 'save', pendingRequest = null, selectionConfirmed = false } = {}) {
      const role = clone(roleValue); const candidate = clone(candidateValue);
      if (!candidateReady(candidate)) throw new Error('Choose an option with ready, verified audio before saving feedback.');
      const baseline = [...new Set((role.candidates || []).filter(candidateReady).map(row => row.candidate_id).filter(Boolean))];
      const payload = { role_id: role.role_id, candidate_id: candidate.candidate_id, round_revision: role.round_revision,
        assignment_revision: role.assignment_revision, key: candidate.key, audio_sha256: candidate.audio_sha256,
        raw_direction: rawDirection, action, baseline_candidate_ids: baseline };
      if (selectionConfirmed === true) payload.selection_confirmed = true;
      if (stars != null) payload.stars = stars;
      if (pendingRequest) return { role, candidate, body: clone(pendingRequest), signature: null };
      const signature = JSON.stringify(payload);
      const { draft } = loadCastingDraft(role, candidate);
      const eventId = draft.event_signature === signature && draft.event_id ? draft.event_id : uuid();
      return { role, candidate, body: { ...payload, event_id: eventId }, signature };
    }
    function acknowledgeCastingDraft(role, candidate, submitted) {
      const { key, draft } = loadCastingDraft(role, candidate);
      if (draft.pendingRequest?.event_id !== submitted.event_id) return;
      const latestText = String(draft.raw_direction || '');
      const submittedText = String(submitted.raw_direction || '');
      const keepText = latestText !== submittedText ? latestText : '';
      if (keepText) saveCastingDraft(key, { stars: null, raw_direction: keepText });
      else clearCastingDraft(key);
    }
    async function submitCastingFeedback(roleValue, candidateValue, options, cardStatus, { retry = false, preserveDraftDirection = false } = {}) {
      const initialKey = castingCandidateKey(roleValue, candidateValue);
      const prior = castingCardStates.get(initialKey) || { status: 'unrated', pendingRequest: null };
      if (castingRoundBusy.has(roleValue.role_id)) return;
      if (castingRoleBusy.has(roleValue.role_id)) {
        // Only the card being saved is locked, so the character's other options stay clickable.
        // A rating given while another is still being saved used to be dropped without a word;
        // it now waits for that save and goes out against the character's fresh revisions.
        if (cardStatus) cardStatus.textContent = 'Waiting for the previous rating to finish saving…';
        return new Promise((resolve, reject) => {
          const line = castingFeedbackWaiting.get(roleValue.role_id) || [];
          line.push({ roleValue, candidateValue, options, cardStatus, mode: { retry, preserveDraftDirection }, resolve, reject });
          castingFeedbackWaiting.set(roleValue.role_id, line);
        });
      }
      if (prior.pendingRequest && !retry) throw new Error('Save not confirmed. Retry the exact saved request before starting another action.');
      castingWriteGeneration++;
      const frozen = frozenFeedback(roleValue, candidateValue, { ...options, pendingRequest: retry ? prior.pendingRequest : null });
      const { role, candidate, body } = frozen;
      const submittedStars = Number.isInteger(body.stars) ? body.stars : null;
      const submittedAction = body.action;
      const { key: draftKey, draft } = loadCastingDraft(role, candidate);
      const newDraft = { ...draft, stars: options.stars == null ? draft.stars : options.stars,
        raw_direction: preserveDraftDirection ? (draft.raw_direction || '') : options.rawDirection };
      newDraft.pendingRequest = clone(body);
      if (frozen.signature) { newDraft.event_signature = frozen.signature; newDraft.event_id = body.event_id; }
      saveCastingDraft(draftKey, newDraft);
      prior.status = 'saving'; prior.pendingRequest = clone(body); castingCardStates.set(initialKey, prior);
      castingRoleBusy.add(role.role_id); castingCandidateBusy.add(initialKey); if (cardStatus) cardStatus.textContent = body.selection_confirmed === true ? 'Saving choice…' : submittedStars == null ? 'Saving direction…' : `Saving ${submittedStars}-star rating…`;
      updateRoleCastingControls(role.role_id); updateCastingFeedbackControls();
      try {
        let result;
        try { result = await client.request('casting/feedback', body); }
        catch (error) {
          if (error.status === 409 && error.body?.conflict === true) result = error.body;
          else throw error;
        }
        acknowledgeCastingDraft(role, candidate, body);
        prior.pendingRequest = null; prior.status = result.conflict ? 'historical' : 'saved';
        prior.reason = result.conflict ? (result.reason || result.message || 'Saved as historical feedback. A newer casting decision remains current.') : '';
        const currentRole = (casting?.roles || []).find(item => item.role_id === role.role_id);
        const currentCandidate = (currentRole?.candidates || []).find(item => item.candidate_id === candidate.candidate_id && item.key === candidate.key && item.audio_sha256 === candidate.audio_sha256);
        if (!result.conflict && submittedStars != null) {
          candidateValue.rating = submittedStars;
          if (currentCandidate) currentCandidate.rating = submittedStars;
        }
        if (cardStatus) cardStatus.textContent = (!result.conflict && submittedStars != null && submittedStars <= 3) ? 'Feedback saved · New option after this review' : result.reason || result.message || (result.conflict
          ? 'Saved as historical feedback. A newer casting decision remains current.'
          : body.selection_confirmed === true ? 'Choice saved.' : submittedStars == null ? 'Direction saved.' : `Saved ${submittedStars} stars. Use is a separate action.`);
        updateRoleCastingControls(role.role_id);
        $('casting-feedback-status').textContent = result.reason || result.message || 'Feedback saved.';
        const chosenNow = submittedAction === 'use' && !result.conflict;
        if (chosenNow) markVoiceChosen(role.role_id, candidate, result);
        const lowRating = !result.conflict && submittedStars != null && submittedStars <= 3;
        const roundChanged = !result.conflict && Number.isInteger(result.round_revision) && result.round_revision !== role.round_revision;
        if (lowRating) {
          // A low rating must not interrupt the session. The option keeps its
          // rating, the batch stays exactly as it was heard, and the replacement
          // is a durable request that arrives later as its own round.
          setCastingFlash(result.reason || `Saved ${submittedStars} stars for ${castingRoleName(role)} · ${candidate.label || 'Option'}. Keep listening: the replacement arrives as a new round.`);
          await refreshCasting(false);
        } else if (roundChanged) {
          castingActive = null; castingDirectionOpen = false; $('casting-feedback').hidden = true;
          setCastingFlash(result.reason || 'Casting round updated.');
          await refreshCasting(false);
        } else if (submittedAction === 'use' || result.conflict) {
          castingActive = null; castingDirectionOpen = false; $('casting-feedback').hidden = true;
          setCastingFlash(chosenNow ? voiceChosenMessage(role) : result.reason || (result.conflict ? 'Saved as historical feedback; the newer decision remains current.' : 'Voice approved and assigned for future Takes.'));
          await refreshCasting(false);
        }
        return result;
      } catch (error) {
        prior.status = 'failed'; prior.pendingRequest = clone(body); castingCardStates.set(initialKey, prior);
        if (cardStatus) cardStatus.textContent = `Save not confirmed. Your draft and exact request are on this device. Retry is safe. ${error.message}`;
        $('casting-feedback-status').textContent = `Not acknowledged; this draft is saved on this device. ${error.message}`;
        throw error;
      } finally {
        castingRoleBusy.delete(role.role_id); castingCandidateBusy.delete(initialKey); updateRoleCastingControls(role.role_id); updateCastingFeedbackControls();
        runNextWaitingFeedback(role.role_id);
      }
    }
    function runNextWaitingFeedback(roleId) {
      const line = castingFeedbackWaiting.get(roleId);
      const next = line && line.shift();
      if (line && !line.length) castingFeedbackWaiting.delete(roleId);
      if (!next) return;
      // The save that just finished may have refreshed the view, so the rating is sent for the
      // option as it is now, not as it was when it was clicked.
      const role = (casting?.roles || []).find(item => item.role_id === roleId) || next.roleValue;
      const wanted = next.candidateValue;
      const candidate = (role.candidates || []).find(item => item.candidate_id === wanted.candidate_id && item.key === wanted.key && item.audio_sha256 === wanted.audio_sha256) || wanted;
      submitCastingFeedback(role, candidate, next.options, next.cardStatus, next.mode).then(next.resolve, next.reject);
    }
    function saveCastingRating(role, candidate, stars, statusNode) {
      activateCasting(role, candidate, false);
      return submitCastingFeedback(role, candidate, { stars, rawDirection: '', action: 'save' }, statusNode, { preserveDraftDirection: true });
    }
    function retryCastingFeedback(role, candidate, statusNode) {
      return submitCastingFeedback(role, candidate, { stars: null, rawDirection: '', action: 'save' }, statusNode, { retry: true, preserveDraftDirection: true });
    }
    function useCastingCandidate(role, candidate, statusNode) {
      if (!Number.isInteger(candidate.rating) || candidate.rating < 4) throw new Error('Save a 4- or 5-star rating before using this voice.');
      activateCasting(role, candidate, false);
      return submitCastingFeedback(role, candidate, { stars: candidate.rating, rawDirection: '', action: 'use' }, statusNode, { preserveDraftDirection: true });
    }
    async function castingFeedback(action) {
      if (!castingActive) return;
      const { role, candidate } = castingActive;
      const rawDirection = $('casting-direction-input').value;
      const selectedStars = $('casting-stars').value === '' ? null : Number($('casting-stars').value);
      const hasDirection = rawDirection.trim().length > 0;
      if (selectedStars == null && !hasDirection) throw new Error('Add a rating or casting direction before saving.');
      if (action === 'use' && !(selectedStars >= 4)) throw new Error('Use this voice requires a saved 4- or 5-star rating.');
      const bodyAction = action === 'preview' ? 'preview' : action === 'recast' ? 'recast' : action;
      return submitCastingFeedback(role, candidate, { stars: selectedStars, rawDirection, action: bodyAction }, $('casting-feedback-status'));
    }
    function bindCastingUI() {
      $('show-approved-casts').onchange = () => { showApprovedCasts = $('show-approved-casts').checked; renderCasting(); };
      const chipBox = $('casting-chips');
      if (chipBox) {
        chipBox.replaceChildren();
        for (const phrase of DIRECTION_CHIPS) {
          const chip = node('button', phrase, 'casting-chip'); chip.type = 'button'; chip.setAttribute('aria-pressed', 'false');
          chip.addEventListener('click', () => {
            if (!castingActive) return;
            const input = $('casting-direction-input'); const rows = input.value.split(/[,;\n]+/).map(row => row.trim()).filter(Boolean);
            const at = rows.findIndex(row => row.toLowerCase() === phrase.toLowerCase());
            if (at >= 0) rows.splice(at, 1); else rows.push(phrase);
            input.value = rows.join(', '); input.oninput?.(); syncDirectionChips();
          });
          chipBox.append(chip);
        }
      }
      const closeSheet = $('casting-feedback-close');
      if (closeSheet) closeSheet.onclick = () => { castingDirectionOpen = false; $('casting-feedback').hidden = true; };
      $('casting-stars').onchange = () => { if (!castingActive) return; const { key, draft } = loadCastingDraft(castingActive.role, castingActive.candidate); draft.stars = $('casting-stars').value === '' ? null : Number($('casting-stars').value); saveCastingDraft(key, draft); updateCastingFeedbackControls(); };
      $('casting-direction-input').oninput = () => { if (!castingActive) return; const { key, draft } = loadCastingDraft(castingActive.role, castingActive.candidate); draft.raw_direction = $('casting-direction-input').value; saveCastingDraft(key, draft); updateCastingFeedbackControls(); syncDirectionChips(); };
      $('casting-save').onclick = () => run(() => castingFeedback('save'));
      $('casting-use').onclick = () => run(() => castingFeedback('use'));
      $('casting-recast').onclick = () => run(() => castingFeedback('recast'));
      $('casting-preview-direction').onclick = () => run(() => castingFeedback('preview'));
      $('prepare-cast').onclick = () => run(async () => {
        castingRequestId ||= uuid(); $('casting-status').textContent = 'Preparing candidates for all current speaking utterances…';
        const result = await client.request('casting/prepare', { request_id: castingRequestId });
        casting = result.casting || result; castingRequestId = null; renderCasting();
        $('casting-status').textContent = result.message || 'Cast preparation was recorded. Review is ready when candidates arrive.';
      });
      $('render-approved-cast').onclick = () => run(async () => {
        const scope = casting?.scope?.utterance_ids || casting?.utterance_ids || (casting?.productions || []).flatMap(row => row.utterance_ids || []);
        if (!Array.isArray(scope) || !scope.length) throw new Error('Prepare the cast first so Studio can return the exact source scope.');
        const requestId = uuid(); $('casting-render-status').textContent = 'Submitting the approved source scope…';
        const result = await client.request('casting/render', { request_id: requestId, utterance_ids: [...scope] });
        casting = result.casting || { ...(casting || {}), productions: [...(casting?.productions || []), result] };
        $('casting-render-status').textContent = result.message || result.status || 'Approved scope render request saved.'; renderCasting();
      });
      $('resume-cast-render').onclick = () => run(async () => {
        const requestId = $('resume-cast-render').dataset.requestId; if (!requestId) return;
        const result = await client.request('casting/resume', { request_id: requestId });
        casting = result.schema === 'casting-review-v1' ? result
          : result.casting || { ...(casting || {}), productions: (casting?.productions || []).map(row => row.request_id === requestId ? { ...row, ...result } : row) };
        $('casting-render-status').textContent = result.message || result.status || 'Existing render request resumed.'; renderCasting();
      });
    }
    const selected = () => (data.voices || []).find(v => v.key === selectedKey);
    const heardVoice = () => { const current = player.current; const row = roleAt(current?.map || [], player.media.currentTime || 0); return row ? voiceForRow(data.voices, row) : (data.voices || []).find(v => v.key === current?.key && v.audio_sha256 === current?.audio_sha256); };
    const voices = () => filtered(data.voices || [], filter);
    const readableStatus = value => ({ blocked_source: 'Source needed', not_run_source_unavailable: 'waiting for source', selected_pending_synthesis: 'Ready to render', conditioning_only: 'reference prepared', waiting_for_production: 'waiting for production', bounded_cycle_complete: 'discovery cycle complete', interrupted_resume_available: 'resume available' }[value] || String(value || '').replaceAll('_', ' '));
    const qaText = recording => { const qa = recording.technical_qa; const status = typeof qa === 'object' && qa ? (qa.status || 'not assessed') : (qa || 'not assessed'); return `Technical QA: ${readableStatus(status)} · Human review: ${recording.human_review || 'pending'}`; };
    const inspiration = v => hideLabels ? 'Inspiration hidden for listening' : [v.inspiration?.performer, v.inspiration?.performance].filter(Boolean).join(' · ');
    function recordingOf(v) { const url = safeUrl(v.audio_url || '', win.location); return v.audio_available !== false && v.audio_url && url && v.audio_sha256 ? { ...v, audio_url: url } : null; }
    function sourceLinks(container, urls) { container.replaceChildren(); for (const value of urls || []) { const url = safeUrl(value, win.location, true); if (!url) continue; const a = node('a', url, 'source-link'); a.href = url; a.target = '_blank'; a.rel = 'noopener noreferrer'; container.append(a); } }
    function renderCharacterAudio() {
      const panel = $('character-audio'); if (!panel) return;
      // The new casting projection owns current verified samples; keep the legacy view for older routes only.
      if ((casting?.roles || []).some(role => Object.prototype.hasOwnProperty.call(role, 'current_audio'))) { panel.hidden = true; return; }
      const roles = data.roles || [];
      panel.hidden = !roles.length;
      const cards = $('character-audio-cards'); cards.replaceChildren();
      let readyCharacters = 0;
      const stored = Array.isArray(data.auditions) ? data.auditions : Object.values(data.auditions || {});
      const byRequest = new Map();
      for (const receipt of [...stored, ...auditionState.values()]) byRequest.set(receipt.request_id || receipt.job_id || `${receipt.role_id}:${receipt.candidate_id}`, receipt);
      const receipts = [...byRequest.values()];
      for (const role of roles) {
        const id = role.id || role.role_id; const name = role.name || role.role_name || id;
        const recordings = (data.voices || []).filter(v => v.role_id === id && recordingOf(v));
        const round = (casting?.roles || []).find(r => r.role_id === id);
        const options = round?.candidates || [];
        const roundReady = options.filter(candidateReady);
        const active = receipts.filter(r => r.role_id === id && ['queued', 'running'].includes(r.status));
        const failed = receipts.filter(r => r.role_id === id && ['failed', 'interrupted'].includes(r.status)).at(-1);
        const generating = active.some(r => r.status === 'running') || options.some(c => ['running', 'generating'].includes(c.status)) || round?.round_status === 'running';
        const queued = active.some(r => r.status === 'queued') || options.some(c => ['pending', 'queued'].includes(c.status)) || round?.round_status === 'pending';
        const ready = recordings.length > 0 || roundReady.length > 0;
        if (ready) readyCharacters++;
        const state = generating ? 'generating' : queued ? 'queued' : ready ? 'ready' : failed || ['failed', 'blocked'].includes(round?.round_status) ? 'failed' : 'not_generated';
        const card = node('article', null, 'cast-card'); card.dataset.roleId = id; card.dataset.audioState = state;
        card.append(node('h3', name));
        const status = node('p', ready ? `Ready — ${recordings.length || roundReady.length} ${Math.max(recordings.length, roundReady.length) === 1 ? 'recording' : 'recordings'} to play` : 'Not generated', 'quality');
        status.setAttribute('role', 'status'); status.setAttribute('aria-live', 'polite'); card.append(status);
        if (generating || queued) {
          const count = options.length ? ` — ${roundReady.length}/${options.length} new options ready` : '';
          card.append(node('p', `${generating ? 'Generating new audio' : 'Queued for new audio'}${count}${ready ? '. Existing recordings remain playable.' : '.'}`, 'muted'));
        } else if (state === 'failed' && !ready) status.textContent = `Failed — ${failed?.reason || round?.reason || options.find(c => c.reason)?.reason || 'retry available'}`;
        if (failed && !active.length) card.append(node('p', `Unsuccessful attempt: ${failed.reason || 'Generation did not complete. Retry the same request below.'}`, 'notice error'));
        const actions = node('div', null, 'button-row');
        const savedRecordings = recordings.filter(v => !roundReady.some(c => c.key === v.key));
        savedRecordings.forEach((v, index) => {
          const label = savedRecordings.length === 1 ? `Play ${name}` : `Play ${name} saved recording ${index + 1}`;
          const play = button(label, () => selectVoice(v.key, true));
          play.setAttribute('aria-label', `${label}, ${time(v.duration_seconds)}`); actions.append(play);
        });
        if (roundReady.length) card.append(node('p', `${roundReady.length}/${options.length} new audition options ready`, 'quality'));
        roundReady.forEach((c, index) => actions.append(button(`Play new ${name} option ${index + 1}`, async () => {
          activateCasting(round, c, false);
          await loadRecording({ ...c, name: `${name} — ${c.label || 'Option'}`, casting_context: { role_id: id, candidate_id: c.candidate_id, key: c.key, audio_sha256: c.audio_sha256, round_revision: round.round_revision, assignment_revision: round.assignment_revision } }, { autoplay: true });
        })));
        if (!ready && !generating && !queued && round && casting.capabilities?.round_requests) actions.append(button(failed ? `Retry ${name} audio` : `Generate ${name} audio`, () => requestCastingRound(round, 'generate')));
        card.append(actions); cards.append(card);
      }
      $('character-audio-status').textContent = `${readyCharacters}/${roles.length} characters have audio ready. New options appear separately from existing recordings.`;
      const sampler = $('character-audio-sampler'); sampler.replaceChildren();
      if (data.excerpt?.audio_available !== false && data.excerpt?.book_not_supplied && data.excerpt?.audio_url && data.excerpt?.audio_sha256) sampler.append(button('Play labeled cast sampler', () => playExcerpt()));
    }
    function renderVoices() {
      renderCharacterAudio();
      $('assign').hidden = castingEnabled; $('casting-approve-inbox').hidden = !castingEnabled;
      $('casting-approve-inbox').textContent = 'Approve in casting inbox';
      $('assign').disabled = castingEnabled || !(data.roles || []).length;
      $('filter').value = filter; $('voice-list').replaceChildren(); const rows = voices();
      $('empty-voices').hidden = rows.length > 0; $('empty-voices').textContent = (data.voices || []).length ? 'No voices in this filter. Try All voices.' : 'No production voice recordings are available for this project yet.';
      for (const v of rows) { const b = button(v.name + (recordingOf(v) ? '' : ' · reference only') + (v.favorite ? ' ♥' : ''), () => selectVoice(v.key), 'voice-choice' + (selectedKey === v.key ? ' selected' : '')); b.setAttribute('aria-pressed', String(v.key === selectedKey)); $('voice-list').append(b); }
      const v = selected(); const noNext = !rows.length || (rows.length === 1 && rows[0].key === selectedKey); $('voice-detail').hidden = !v; $('previous').disabled = noNext; $('next').disabled = noNext;
      if (!v) return;
      const playable = !!recordingOf(v); const busy = pending.has(v.key);
      $('voice-name').textContent = v.name; $('voice-role').textContent = `${v.role_name || 'Unassigned voice'} · ${v.synthetic === false ? 'Licensed voice' : 'Synthetic voice'} · v${String(v.version).slice(0,8)}`;
      $('voice-inspiration').textContent = inspiration(v); $('voice-explanation').textContent = v.explanation || '';
      $('voice-tags').replaceChildren(...(v.tags || []).slice(0, 3).map(tag => node('span', tag)));
      $('voice-qa').textContent = playable ? qaText(v) : 'Production-path audio is not available. This version cannot be favorited or rated yet.';
      $('favorite').textContent = v.favorite ? '♥ Favorited' : '♡ Favorite'; $('favorite').setAttribute('aria-pressed', String(!!v.favorite)); $('favorite').disabled = !playable || busy;
      $('rating-stars').replaceChildren();
      for (let n = 1; n <= 5; n++) { const b = button(`${n} ★`, () => preference(v, { rating: n })); b.setAttribute('aria-label', `Rate ${n} out of 5`); b.setAttribute('aria-pressed', String(v.rating === n)); b.disabled = !playable || busy; $('rating-stars').append(b); }
      $('clear-rating').disabled = !playable || v.rating == null || busy;
      const priorRole = $('assign-role').value; $('assign-role').replaceChildren();
      for (const role of data.roles || []) { const option = node('option', role.name || role.role_name || role.id); option.value = role.id || role.role_id; $('assign-role').append(option); }
      const targetRole = priorRole || v.role_id; if ([...$('assign-role').options].some(o => o.value === targetRole)) $('assign-role').value = targetRole;
      $('assign').hidden = castingEnabled; $('casting-approve-inbox').hidden = !castingEnabled;
      $('casting-approve-inbox').textContent = 'Approve in casting inbox';
      $('assign').disabled = castingEnabled || !(data.roles || []).length || busy; $('more-like').disabled = busy;
      renderSuitability(); $('suitability').disabled = !playable || busy || !$('assign-role').value;
      sourceLinks($('voice-sources'), v.inspiration?.source_urls);
      $('voice-details').textContent = JSON.stringify({ voice_id: v.voice_id, version: v.version, kind: v.kind, audio_sha256: v.audio_sha256, ...(v.details || {}) }, null, 2);
    }
    function renderSuitability() { const v = selected(); const value = v?.suitability; $('suitability').value = value && typeof value === 'object' ? (value[$('assign-role').value] ?? '') : (value ?? ''); }
    async function preference(v, patch) {
      if (!v || !recordingOf(v) || pending.has(v.key)) return;
      pending.add(v.key); renderVoices(); renderCast(); $('save-status').textContent = 'Saving…';
      try {
        await client.request('preferences', { key: v.key, audio_sha256: v.audio_sha256, ...patch });
        if ('suitability' in patch) v.suitability = { ...(typeof v.suitability === 'object' ? v.suitability : {}), [patch.role_id]: patch.suitability };
        if ('favorite' in patch) v.favorite = patch.favorite; if ('rating' in patch) v.rating = patch.rating;
        $('save-status').textContent = 'Saved to this voice version.';
      } catch (e) { $('save-status').textContent = 'Not saved. Please retry.'; throw e; }
      finally { pending.delete(v.key); renderVoices(); renderCast(); }
    }
    function renderCast() {
      const container = $('cast'); container.replaceChildren();
      const recording = player.current?.map?.length ? player.current : data.excerpt;
      const map = recording?.map || []; const gain = recording === player.current ? player.media.volume : 1;
      const ids = [...new Set(map.map(row => row.role_id).filter(Boolean))];
      for (const id of ids) {
        const line = map.find(row => row.role_id === id); const v = voiceForRow(data.voices, line); const card = node('article', null, 'cast-card'); card.dataset.roleId = id;
        card.append(node('h3', line.role_name || id), node('p', v ? `${v.name} · v${v.version}` : 'Voice not registered', 'muted'));
        if (v) card.append(node('p', inspiration(v), 'muted'));
        const controls = node('div', null, 'button-row'); const jump = button('Hear lines', async () => { await loadRecording({ ...recording, key: recording.key || 'excerpt', name: recording.name || recording.chapter_title || 'Listening excerpt' }, { seconds: line.start_seconds, autoplay: true, gain }); }); jump.disabled = !recordingOf(recording || {}); controls.append(jump);
        if (v) { const heart = button(v.favorite ? '♥' : '♡', () => preference(v, { favorite: !v.favorite })); heart.setAttribute('aria-label', `Favorite ${v.name}`); heart.setAttribute('aria-pressed', String(!!v.favorite)); heart.disabled = !recordingOf(v) || pending.has(v.key); controls.append(heart); }
        card.append(controls);
        if (v) { const label = node('label', 'Voice rating'); const rating = node('select'); rating.setAttribute('aria-label', `Rate ${v.name}`); const clear = node('option', 'Not rated'); clear.value = ''; rating.append(clear); for (let i = 1; i <= 5; i++) { const option = node('option', `${i} ${'★'.repeat(i)}`); option.value = String(i); rating.append(option); } rating.value = v.rating ?? ''; rating.disabled = !recordingOf(v) || pending.has(v.key); rating.onchange = () => run(() => preference(v, { rating: rating.value ? Number(rating.value) : null })); label.append(rating); card.append(label); }
        container.append(card);
      }
      updateTime();
    }
    function renderExcerpt() {
      const e = data.excerpt; $('excerpt-heading').textContent = e?.chapter_title || (data.scope === 'isolated_listening_test' ? 'Chapter 1 listening test' : 'Continuous listening');
      $('play-excerpt').textContent = e?.schema === 'voice-lab-chapter-review-v1' ? 'Listen to chapter' : 'Listen to excerpt';
      $('excerpt-status').textContent = e ? [readableStatus(e.status), e.reason, e.available_chapters?.length ? `Available chapters: ${e.available_chapters.join(', ')}` : '', e.duration_seconds ? `${time(e.duration_seconds)} at natural playback speed` : ''].filter(Boolean).join(' · ') : 'No continuous listening file is available for this project yet.';
      $('play-excerpt').disabled = !recordingOf(e || {}); $('excerpt-qa').textContent = e ? qaText(e) : 'Technical QA: not assessed · Human review: pending';
      const comparisons = matchedComparisons(e); $('comparison-panel').hidden = comparisons.length < 2; $('comparisons').replaceChildren();
      for (const [i, comparison] of comparisons.entries()) { const b = button(hideLabels ? `Recording ${String.fromCharCode(65 + i)}` : (comparison.label || `Recording ${i + 1}`), () => loadRecording({ ...comparison, key: comparison.key || 'comparison-' + i, name: hideLabels ? `Recording ${String.fromCharCode(65 + i)}` : (comparison.label || `Recording ${i + 1}`), map: comparison.map || e.map }, { autoplay: true, seconds: player.media.currentTime || 0, gain: comparisonGain(comparison, comparisons) })); $('comparisons').append(b); }
      $('comparison-status').textContent = comparisons.length >= 2 ? 'Measured loudness is matched using playback attenuation only.' : 'A/B needs two existing recordings with verified matching utterances and measured loudness.';
      renderCast();
    }
    function renderAutomation() { const a = data.automation || {}; $('automation-status').textContent = `${a.paused ? 'Paused' : 'Active'} · ${readableStatus(a.status) || 'Bounded discovery'}${a.cursor ? ' · Completed cycles: ' + a.cursor : ''}`; $('automation-toggle').textContent = a.paused ? 'Resume discovery' : 'Pause discovery'; }
    function renderIssues() { $('issues').replaceChildren(); for (const issue of data.issues || []) { const item = node('div', null, 'issue-row'); item.append(node('strong', `${time(issue.seconds)} · ${issue.category || 'Note'}`), node('p', issue.text), node('small', `Audio ${(issue.audio_sha256 || '').slice(0, 12)}`)); $('issues').append(item); } }
    async function savePosition(force = false) {
      if (!player.current || player.current.book_chapter || savingPosition || (!force && Date.now() - lastPosition < 5000)) return;
      savingPosition = true;
      try { await client.request('position', { key: player.current.key, audio_sha256: player.current.audio_sha256, seconds: player.media.currentTime || 0, filter }); lastPosition = Date.now(); }
      catch (e) { error('Listening position was not saved: ' + e.message); }
      finally { savingPosition = false; }
    }
    async function loadRecording(recording, options = {}) {
      const token = ++loadToken;
      const safe = recordingOf(recording || {});
      if (!safe?.casting_context) { castingActive = null; $('casting-feedback').hidden = true; }
      player.media.pause();
      // Capture the old recording's key/hash/time before loading the next one.
      // Saving handles its own failures; a network wait must not move play()
      // outside the user's tap, where mobile browsers can reject playback.
      void savePosition(true);
      if (token !== loadToken) return;
      await player.load(safe, options);
      if (token !== loadToken) return;
      $('play').disabled = !safe; $('seek').disabled = !safe; $('save-issue').disabled = !safe; $('now-playing').textContent = safe?.name || 'Production audio is not available'; renderCast();
    }
    async function selectVoice(key, autoplay = false, seconds = 0) { selectedKey = key; renderVoices(); const v = selected(); await loadRecording(v, { autoplay, seconds }); }
    async function move(delta) { const rows = voices(); if (!rows.length) return; const index = rows.findIndex(v => v.key === selectedKey); const next = index < 0 ? (delta > 0 ? 0 : rows.length - 1) : (index + delta + rows.length) % rows.length; await selectVoice(rows[next].key, !player.media.paused); }
    async function playExcerpt(seconds = 0, autoplay = true) { const e = data.excerpt; if (!e) return; await loadRecording({ ...e, key: e.key || 'excerpt', name: e.chapter_title || 'Listening excerpt' }, { autoplay, seconds }); }
    function updateTime() {
      const media = player.media; const current = player.current; const duration = Number.isFinite(media.duration) ? media.duration : finite(current?.duration_seconds);
      $('elapsed').textContent = time(media.currentTime); $('duration').textContent = time(duration); $('seek').max = duration; $('seek').value = media.currentTime || 0; $('play').textContent = media.paused ? 'Play' : 'Pause';
      const castingPlayback = !!current?.casting_context;
      $('previous').disabled = $('next').disabled = castingPlayback || !voices().length || (voices().length === 1 && voices()[0].key === selectedKey);
      updateCastingPlayLabels(); updateBookPlayLabels();
      const row = roleAt(current?.map || [], media.currentTime || 0); $('current-role').textContent = row ? (row.role_name || row.role_id) : (current?.role_name || '');
      for (const card of $('cast').children) { const active = !!row && card.dataset.roleId === row.role_id; card.classList.toggle('current', active); if (active) card.setAttribute('aria-current', 'true'); else card.removeAttribute('aria-current'); }
      $('issue-version').textContent = current?.audio_sha256 ? `${time(media.currentTime)} · ${current.name || 'Recording'} · Audio ${current.audio_sha256.slice(0, 12)}` : 'Choose a recording to leave a timestamped note.';
    }
    const auditionSlot = (roleId, candidateId) => roleId + ':' + candidateId;
    function auditionFor(roleId, candidateId) { const slot = auditionSlot(roleId, candidateId); const stored = Array.isArray(data.auditions) ? data.auditions : Object.values(data.auditions || {}); return auditionState.get(slot) || stored.find(row => row.role_id === roleId && row.candidate_id === candidateId); }
    async function refreshAudition(receipt, cycles = 0) {
      const slot = auditionSlot(receipt.role_id, receipt.candidate_id);
      const result = await client.request('auditions/' + encodeURIComponent(receipt.request_id));
      if (auditionState.has(slot) && auditionState.get(slot).request_id !== receipt.request_id) return;
      const latest = { ...receipt, ...result }; auditionState.set(slot, latest);
      if (latest.status === 'succeeded') { data = await client.request(''); renderVoices(); renderCast(); }
      renderCharacterAudio(); renderMatches(currentMatches); scheduleAudition(latest, cycles + 1);
    }
    function scheduleAudition(receipt, cycles = 0) {
      const slot = auditionSlot(receipt.role_id, receipt.candidate_id); if (auditionTimers.has(slot) && win.clearTimeout) win.clearTimeout(auditionTimers.get(slot));
      if (['queued', 'running'].includes(receipt.status) && cycles < 180 && win.setTimeout) auditionTimers.set(slot, win.setTimeout(() => {
        // A tab nobody is looking at asks for nothing; it catches up when it is shown again.
        if (win.document?.hidden) { scheduleAudition(receipt, cycles); return; }
        void run(() => refreshAudition(receipt, cycles));
      }, 3000));
    }
    async function previewCandidate(match, roleId) {
      const candidateId = match.candidate_id || match.id; const slot = auditionSlot(roleId, candidateId);
      if (auditionBusy.has(slot)) return; const role = (data.roles || []).find(row => (row.id || row.role_id) === roleId);
      if (!role) throw new Error('Choose a character for this preview.');
      const previous = auditionFor(roleId, candidateId);
      const payload = { role_id: roleId, candidate_id: candidateId, expected_identity_id: previous && Object.hasOwn(previous, 'expected_identity_id') ? previous.expected_identity_id : role.current_identity_id ?? null, request_id: previous?.request_id || win.crypto.randomUUID() };
      auditionBusy.add(slot); auditionState.set(slot, { ...payload, status: 'running', reason: 'Preparing one production-path preview…' }); renderCharacterAudio(); renderMatches(currentMatches);
      try { const result = await client.request('audition', payload); const receipt = { ...payload, ...result }; auditionState.set(slot, receipt); if (receipt.status === 'succeeded') { data = await client.request(''); renderVoices(); renderCast(); } renderCharacterAudio(); scheduleAudition(receipt); }
      catch (e) { auditionState.set(slot, { ...payload, status: 'interrupted', reason: e.message }); renderCharacterAudio(); throw e; }
      finally { auditionBusy.delete(slot); renderMatches(currentMatches); }
    }
    function renderMatches(matches) {
      currentMatches = matches || []; $('matches').replaceChildren();
      for (const match of currentMatches) {
        const card = node('article', null, 'match-card'); const ref = match.inspiration || match;
        card.append(node('h3', [ref.performer || match.performer_name || match.name, ref.performance || match.performance_title].filter(Boolean).join(' · ')), node('p', match.explanation || match.rationale || (match.reasons || []).join(' · ')), node('p', match.evidence_label || 'Exploratory metadata match · not an audio measurement', 'quality'));
        const details = node('details'); details.append(node('summary', 'Evidence and creative direction'));
        for (const fact of match.verified_facts || []) details.append(node('p', fact));
        if (match.creative_direction) details.append(node('p', 'Original voice direction: ' + match.creative_direction));
        details.append(node('p', match.creative_interpretation || 'Performance references inform an original synthetic voice. They do not authorize cloning the performer.', 'muted'));
        const links = node('div'); sourceLinks(links, ref.source_urls || match.source_urls || (match.source_url ? [match.source_url] : [])); details.append(links); card.append(details);
        const candidateId = match.candidate_id || match.id;
        const roleId = match.match_role_id || candidateRoles.get(candidateId) || null;
        const role = (data.roles || []).find(r => (r.id || r.role_id) === roleId);
        card.dataset.roleId = roleId || ''; card.dataset.candidateId = candidateId || '';
        card.append(node('p', role ? 'For ' + (role.name || role.role_name) : 'General direction · choose a character to preview', 'match-context'));
        if (!match.match_role_id) { const label = node('label', 'Preview for character'); const picker = node('select'); const empty = node('option', 'Choose a character'); empty.value = ''; picker.append(empty); for (const item of data.roles || []) { const option = node('option', item.name || item.role_name || item.id); option.value = item.id || item.role_id; picker.append(option); } picker.value = roleId || ''; picker.onchange = () => { candidateRoles.set(candidateId, picker.value); renderMatches(currentMatches); }; label.append(picker); card.append(label); }
        if (candidateId) {
          const receipt = role ? auditionFor(roleId, candidateId) : null;
          const voice = receipt?.status === 'succeeded' && receipt.key ? (data.voices || []).find(v => v.key === receipt.key && recordingOf(v)) : null;
          const slot = auditionSlot(roleId, candidateId); const busy = auditionBusy.has(slot) || receipt?.status === 'running';
          const actions = node('div', null, 'preview-actions');
          const preview = button(voice ? `Play preview · ${time(voice.duration_seconds)}` : busy ? 'Generating preview…' : receipt ? 'Resume preview' : 'Preview · ~10 seconds', () => voice ? selectVoice(voice.key, true) : previewCandidate(match, roleId)); preview.disabled = !role || busy; actions.append(preview);
          const assign = castingEnabled
            ? button('Approve in casting inbox', openCastingInbox)
            : button('Use for this character', async () => { const currentRole = (data.roles || []).find(r => (r.id || r.role_id) === roleId); await client.request('assign', { key: voice.key, role_id: roleId, expected_identity_id: currentRole.current_identity_id ?? null }); data = await client.request(''); renderVoices(); renderCast(); renderMatches(currentMatches); $('save-status').textContent = `${voice.name} assigned to ${currentRole.name || currentRole.role_name} for future Takes.`; });
          assign.disabled = !role || (!castingEnabled && (!voice || busy)); actions.append(assign);
          if (receipt?.request_id && !voice) actions.append(button('Refresh preview status', () => refreshAudition(receipt)));
          card.append(actions, node('p', voice ? `${voice.name} · version ${String(voice.version).slice(0, 8)} · ${time(voice.duration_seconds || receipt.duration_seconds)}. Ready to play; casting is unchanged until you choose Use.` : receipt?.reason || (receipt ? `Preview ${readableStatus(receipt.status)}. A reference alone is not a playable audition.` : 'One original synthetic preview. No celebrity recording is cloned without eligibility.'), 'preview-status'));
        }
        $('matches').append(card);
      }
    }
    $('filter').onchange = () => run(async () => { filter = $('filter').value; const rows = voices(); if (!rows.some(v => v.key === selectedKey)) { if (rows.length) await selectVoice(rows[0].key); else { selectedKey = null; await loadRecording(null); } } renderVoices(); await client.request('position', { key: player.current?.key || null, audio_sha256: player.current?.audio_sha256 || null, seconds: player.media.currentTime || 0, filter }); });
    $('play').onclick = () => run(() => player.toggle()); $('previous').onclick = () => run(() => move(-1)); $('next').onclick = () => run(() => move(1));
    $('favorite').onclick = () => run(() => { const v = selected(); return preference(v, { favorite: !v?.favorite }); });
    $('clear-rating').onclick = () => run(() => preference(selected(), { rating: null }));
    $('assign-role').onchange = renderSuitability;
    $('suitability').onchange = () => run(() => preference(selected(), { role_id: $('assign-role').value, suitability: $('suitability').value ? Number($('suitability').value) : null }));
    $('assign').onclick = () => run(async () => { if (castingEnabled) { openCastingInbox(); return; } const v = selected(); const role = (data.roles || []).find(r => (r.id || r.role_id) === $('assign-role').value); if (!v || !role) return; await client.request('assign', { key: v.key, role_id: role.id || role.role_id, expected_identity_id: role.current_identity_id ?? null }); data = await client.request(''); renderVoices(); $('save-status').textContent = 'Voice assigned for future Takes.'; });
    $('casting-approve-inbox').onclick = openCastingInbox;
    $('more-like').onclick = () => run(async () => { const v = selected(); if (!v) return; const result = await client.request('more-like', { key: v.key, request_id: win.crypto.randomUUID() }); $('save-status').textContent = result.message || 'Related directions saved. Audio replenishment remains idle.'; });
    $('library-publish').onclick = () => run(async () => { const v = selected(); if (!v || !recordingOf(v)) throw new Error('Choose a voice with production audio first.'); await client.request('library/publish', { key: v.key }); $('library-status').textContent = 'Exact voice version saved to your existing Studio library.'; });
    $('library-load').onclick = () => run(async () => { const result = await client.request('library'); const picker = $('library-versions'); picker.replaceChildren(); for (const voice of result.voices || []) for (const version of voice.versions || []) { const option = node('option', `${version.name || voice.name} · ${String(version.version).slice(0,8)}`); option.value = JSON.stringify({library_voice_id:voice.library_voice_id, version:version.version}); picker.append(option); } $('library-status').textContent = picker.children.length ? 'Choose a saved version to import.' : 'No reusable production versions have been saved yet.'; });
    $('library-import').onclick = () => run(async () => { const value = $('library-versions').value; if (!value) throw new Error('Browse and choose a saved version first.'); const result = await client.request('library/import', JSON.parse(value)); data = await client.request(''); filter = 'all'; await selectVoice(result.key, false); $('library-status').textContent = 'Version imported. Character casting and preferences are unchanged.'; });
    $('automation-toggle').onclick = () => run(async () => { await client.request('automation', { paused: !data.automation?.paused }); data = await client.request(''); renderAutomation(); });
    $('play-excerpt').onclick = () => run(() => playExcerpt());
    bindCastingUI();
    $('hide-labels').onchange = () => { hideLabels = $('hide-labels').checked; renderVoices(); renderExcerpt(); };
    $('seek').oninput = () => { player.seek(Number($('seek').value)); updateTime(); }; $('seek').onchange = () => run(() => savePosition(true));
    $('match-role').onchange = () => renderMatches(currentMatches);
    $('match-form').onsubmit = event => { event.preventDefault(); const roleId = $('match-role').value || null; return run(async () => { $('match-status').textContent = 'Finding performance references…'; try { const result = await client.request('match', { description: $('match-description').value.trim(), role_id: roleId, limit: 4 }); const matches = (result.matches || []).map(match => ({ ...match, match_role_id: roleId })); renderMatches(matches); $('match-status').textContent = `${matches.length} performance ${matches.length === 1 ? 'reference' : 'references'}. Choose Preview to hear a candidate.`; } catch (e) { $('match-status').textContent = 'Discovery did not complete. Please retry.'; throw e; } }); };
    $('issue-form').onsubmit = event => { event.preventDefault(); run(async () => { const current = player.current; const text = $('issue-text').value.trim(); if (!current?.audio_sha256 || !text) return; const issue = { audio_sha256: current.audio_sha256, seconds: player.media.currentTime || 0, text, category: $('issue-category').value }; await client.request('issues', issue); data.issues = [...(data.issues || []), issue]; $('issue-text').value = ''; renderIssues(); $('save-status').textContent = 'Timestamped note saved. No repair was started.'; }); };
    for (const type of ['timeupdate', 'loadedmetadata', 'play', 'pause', 'ended']) player.media.addEventListener(type, () => { updateTime(); if (type === 'timeupdate' && !player.media.paused) void savePosition(); if (type === 'pause' || type === 'ended') void savePosition(true); });
    player.media.addEventListener('error', () => error('This recording could not be played. Its saved voice and feedback have not changed.'));
    doc.addEventListener('keydown', event => {
      const action = shortcut(event); if (!action || $('workspace').hidden) return; event.preventDefault();
      if (player.current?.book_chapter) { if (action === 'play') run(async () => { await player.toggle(); updateBookPlayLabels(); }); return; }
      const castingPlayback = player.current?.casting_context;
      if (castingPlayback) {
        if (typeof action === 'number') {
          const active = castingActive;
          if (active?.role.role_id === castingPlayback.role_id && active.candidate.candidate_id === castingPlayback.candidate_id
            && active.candidate.key === castingPlayback.key && active.candidate.audio_sha256 === castingPlayback.audio_sha256) {
            run(async () => { const status = cardControls(active.role, active.candidate); await saveCastingRating(active.role, active.candidate, action, status?.statusNode); });
          } else setCastingFlash('This option is no longer current. Refresh the casting inbox before rating it.');
        } else if (action === 'favorite') setCastingFlash('Favorites apply to saved library voices; this audition rating is separate.');
        else if (action === 'previous' || action === 'next') return;
        else run(() => player.toggle());
        return;
      }
      const heard = heardVoice(); run(() => typeof action === 'number' ? preference(heard, { rating: action }) : action === 'favorite' ? preference(heard, { favorite: !heard?.favorite }) : action === 'previous' ? move(-1) : action === 'next' ? move(1) : player.toggle());
    });
    doc.addEventListener('visibilitychange', () => { if (doc.hidden) void savePosition(true); });
    $('project-name').textContent = data.project_name || project; renderExcerpt(); renderAutomation(); renderIssues();
    const castingAvailable = data.capabilities?.casting_review === true || data.capabilities?.casting_enabled === true || data.casting_enabled === true;
    $('casting-inbox').hidden = !castingAvailable;
    if (!castingAvailable && $('casting-extra-tools')) $('casting-extra-tools').open = true;
    if (castingAvailable) {
      const roleCount = Array.isArray(data.roles) ? data.roles.length : 0;
      startup(roleCount ? `Found ${roleCount} characters. Loading your audition choices…` : 'Loading your character options…');
      if (!await refreshCasting()) {
        startup($('casting-status').textContent || 'Your character options could not be loaded. Please retry.', true);
        return { player, client, startupFailed: true };
      }
    }
    $('workspace').hidden = false; startup('');
    for (const role of data.roles || []) { const option = node('option', role.name || role.role_name || role.id); option.value = role.id || role.role_id; $('match-role').append(option); }
    const saved = data.review_position || {}; const savedVoice = (data.voices || []).find(v => v.key === saved.key && v.audio_sha256 === saved.audio_sha256);
    const savedCasting = (casting?.roles || []).flatMap(role => (role.candidates || [])
      .filter(candidate => candidateReady(candidate) && candidate.key === saved.key && candidate.audio_sha256 === saved.audio_sha256)
      .map(candidate => ({ role, candidate })));
    const comparisons = matchedComparisons(data.excerpt); const savedComparison = comparisons.find(v => v.key === saved.key && v.audio_sha256 === saved.audio_sha256);
    if (savedCasting.length === 1) {
      const { role, candidate } = savedCasting[0]; activateCasting(role, candidate); renderVoices();
      await loadRecording({ ...candidate, name: `${castingRoleName(role)} · ${candidate.label || 'Option'}`,
        casting_context: { role_id: role.role_id, candidate_id: candidate.candidate_id, key: candidate.key,
          audio_sha256: candidate.audio_sha256, round_revision: role.round_revision, assignment_revision: role.assignment_revision } },
        { autoplay: false, seconds: saved.seconds });
    }
    else if (savedVoice) await selectVoice(savedVoice.key, false, saved.seconds);
    else if (savedComparison) { renderVoices(); await loadRecording({ ...savedComparison, name: savedComparison.label || 'Comparison recording', map: savedComparison.map || data.excerpt.map }, { seconds: saved.seconds, autoplay: false, gain: comparisonGain(savedComparison, comparisons) }); }
    else if (data.excerpt?.audio_sha256 && saved.audio_sha256 === data.excerpt.audio_sha256) { renderVoices(); await playExcerpt(saved.seconds, false); }
    else if (voices().length) await selectVoice(voices()[0].key); else renderVoices();
    // The deep link wins over the last-review position: the human was sent here
    // for one exact review, so that is what must be on screen and focused.
    await bookReady;
    if (castingAvailable && (linkedCharacterId || linkedCastingView)) focusLinkedCastingReview();
    await openLinkedBookChapter();
    renderMatches(shortlistMatches(data.shortlists));
    // The receipts in this payload already carry their current status. Only one that is running can
    // change by itself: a queued one waits for someone to press Resume. Polling every saved receipt
    // every three seconds, as this did, kept Studio at full load for as long as any page was open
    // once a few dozen stale ones had piled up. A preview started on this page still polls to its end.
    for (const receipt of Array.isArray(data.auditions) ? data.auditions : Object.values(data.auditions || {})) if (receipt.request_id && receipt.role_id && receipt.candidate_id && receipt.status === 'running') scheduleAudition(receipt);
    return { player, client };
  }
  return { Player, ReviewClient, shortcut, typing, filtered, roleAt, voiceForRow, matchedComparisons, comparisonGain, safeUrl, shortlistMatches, mount, startSectionNav, startLiquidViewport, startTipPlacement };
});
