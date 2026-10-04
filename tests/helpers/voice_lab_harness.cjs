'use strict';
const fs = require('node:fs');
const path = require('node:path');
const { mount } = require('../../dubbing_service/webui/voice-lab.js');
class Element {
  constructor(tag = 'div') {
    this.tagName = tag.toUpperCase(); this.children = []; this.dataset = {}; this.events = {}; this.attrs = {};
    this.value = ''; this.hidden = false; this.disabled = false; this.textContent = ''; this.className = '';
    this.classList = { toggle: (name, value) => { const names = new Set(this.className.split(/\s+/).filter(Boolean)); const add = value ?? !names.has(name); if (add) names.add(name); else names.delete(name); this.className = [...names].join(' '); }, add: name => { if (!this.className.split(/\s+/).includes(name)) this.className += ' ' + name; } };
    this.style = { setProperty: (name, value) => { this.style[name] = value; } };
  }
  append(...children) { this.children.push(...children); }
  replaceChildren(...children) { this.children = children; }
  setAttribute(name, value) { this.attrs[name] = String(value); }
  getAttribute(name) { return this.attrs[name] ?? null; }
  removeAttribute(name) { delete this.attrs[name]; }
  addEventListener(name, fn) { (this.events[name] ||= []).push(fn); }
  emit(name, event) { return Promise.all((this.events[name] || []).map(fn => fn(event))); }
  get options() { return this.children; }
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
const clone = object => structuredClone(object);
const allNodes = (element, className, result = []) => { if (element?.className?.split(/\s+/).includes(className)) result.push(element); for (const child of element?.children || []) allNodes(child, className, result); return result; };
async function mountLab(fixture, options = {}) {
  const html = fs.readFileSync(path.join(__dirname, '../../dubbing_service/webui/voice-lab.html'), 'utf8');
  const elements = Object.fromEntries([...html.matchAll(/<([a-z]+)[^>]* id="([^"]+)"/g)].map(match => [match[2], new Element(match[1])]));
  const doc = new Element(); doc.getElementById = id => elements[id]; doc.createElement = tag => new Element(tag);
  elements.media = new Media();
  const storage = options.storage || new Map(), requests = [], intervals = [];
  const base = { project_name: 'Portrait fixture', capabilities: { casting_enabled: true, casting_review: true }, casting_enabled: true,
    voices: fixture.roles.flatMap(role => role.current_audio ? [{ ...role.current_audio, role_id: role.role_id }] : []), roles: fixture.roles.map(role => ({ id: role.role_id, name: role.name })), automation: { paused: true }, issues: [] };
  const win = { location: { href: 'http://localhost/web/voice-lab.html?project=portrait-fixture', origin: 'http://localhost', pathname: '/web/voice-lab.html', search: '?project=portrait-fixture', reload: () => options.onReload?.() },
    crypto: { randomUUID: (() => { let next = 0; return () => 'fixture-event-' + ++next; })() },
    localStorage: { getItem: key => storage.get(key) || null, setItem: (key, value) => storage.set(key, value), removeItem: key => storage.delete(key) },
    setInterval: fn => { intervals.push(fn); return intervals.length; }, clearInterval() {},
    fetch: async (url, request) => {
      const action = url.split('/voice-lab')[1]?.replace(/^\//, '') || '';
      const body = request?.method === 'POST' ? JSON.parse(request.body) : undefined;
      const entry = { path: action, body, method: request?.method || 'GET' }; requests.push(entry);
      let response = await options.respond?.(action, body, entry);
      if (body && action === 'preferences') {
        const value = typeof response?.favorite === 'boolean' ? response.favorite : body.favorite;
        for (const role of fixture.roles) for (const row of [...(role.candidates || []), role.current_audio].filter(Boolean)) if (row.key === body.key && row.audio_sha256 === body.audio_sha256) row.favorite = value;
        response ??= { saved: true, favorite: value };
      }
      if (response === undefined && body && action === 'casting/feedback') {
        const role = fixture.roles.find(row => row.role_id === body.role_id);
        const row = role?.candidates.find(candidate => candidate.candidate_id === body.candidate_id && candidate.key === body.key && candidate.audio_sha256 === body.audio_sha256);
        if (!row || body.action !== 'use' || body.selection_confirmed !== true || 'stars' in body || body.round_revision !== role.round_revision || body.assignment_revision !== role.assignment_revision) throw new Error('Fixture received an invalid selection payload');
        role.selected_candidate_id = row.candidate_id; role.current_key = row.key; role.status = 'APPROVED'; role.chosen = true; role.assignment_revision++;
        response = { accepted: true, assignment_revision: role.assignment_revision, round_revision: role.round_revision };
      }
      if (response === undefined) {
        if (action === '') response = base;
        else if (action === 'casting') response = { schema: 'casting-review-v1', revision: fixture.revision || 7, ...fixture };
        else if (action === 'casting/book') response = { schema: 'casting-book-status-v1', state: 'idle', voices: { waiting: [], approved: 0, total: fixture.roles.length }, chapters: [], production: null };
        else if (action === 'position') response = { saved: true };
        else if (body) throw new Error('Unexpected fixture mutation: ' + action);
        else response = {};
      }
      return { ok: true, status: 200, json: async () => clone(response) };
    }
  };
  options.configure?.(doc, win);
  const statePromise = mount(doc, win);  const lab = { state: null, base, elements, doc, win, storage, intervals, requests, media: elements.media,
    all: name => allNodes(elements['casting-roles'], name), currentContext: () => lab.state?.player.current?.casting_context,
    click: async element => { if (!element.disabled) await element.emit('click'); },
    settle: async () => { for (let turn = 0; turn < 12; turn++) await new Promise(resolve => setImmediate(resolve)); }
  };
  lab.ready = statePromise.then(state => { lab.state = state; return state; });
  if (!options.startOnly) await lab.ready;
  return lab;
}
module.exports = { mountLab };
