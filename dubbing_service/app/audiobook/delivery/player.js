'use strict';
const batchId = location.pathname.split('/').filter(Boolean).pop();
const api = `/api/batches/${encodeURIComponent(batchId)}`;
const players = [];
let batch;
const $ = id => document.getElementById(id);
const readLocal = key => { try { return localStorage.getItem(key); } catch { return null; } };
const writeLocal = (key, value) => { try { localStorage.setItem(key, value); } catch {} };
const time = seconds => { const n = Math.max(0, Math.floor(Number.isFinite(seconds) ? seconds : 0)); return `${Math.floor(n / 60)}:${String(n % 60).padStart(2, '0')}`; };
const node = (tag, cls, text) => { const el = document.createElement(tag); if (cls) el.className = cls; if (text !== undefined) el.textContent = text; return el; };

async function json(url, options) {
  const response = await fetch(url, {credentials:'same-origin', ...options});
  if (!response.ok) throw new Error(response.status === 403 ? 'Connect Tailscale on your phone, then try again.' : 'Could not connect. Keep Tailscale connected and try again. Your draft is saved.');
  return response.json();
}
async function submit(payload) {
  return json(`${api}/feedback`, {method:'POST',headers:{'Content-Type':'application/json','X-Review-Request':'1'},body:JSON.stringify(payload)});
}
function buildChapter(chapter) {
  const card = node('article', 'card');
  const heading = node('div', 'chapter-heading');
  heading.append(node('h2', '', `Chapter ${chapter.chapter_number}`), node('span', 'duration', time(chapter.duration_seconds)));
  card.append(heading);
  if (chapter.title && !new RegExp(`^(chapter\\s*)?${chapter.chapter_number}$`, 'i').test(chapter.title.trim())) card.append(node('p', 'title', chapter.title));
  const audio = node('audio'); audio.controls = true; audio.preload = 'metadata'; audio.setAttribute('playsinline', '');
  audio.setAttribute('aria-label', `Play Chapter ${chapter.chapter_number}`);
  audio.src = `/audio/${encodeURIComponent(batchId)}/${encodeURIComponent(chapter.chapter_id)}`;
  const placeKey = `audiobook-place-${chapter.audio_sha256}`;
  const draftKey = `audiobook-note-${chapter.audio_sha256}`;
  audio.addEventListener('loadedmetadata', () => {const saved = Number(readLocal(placeKey)); if (Number.isFinite(saved) && saved > 0 && saved < audio.duration - 2) audio.currentTime = saved;});
  audio.addEventListener('timeupdate', () => {if (Number.isFinite(audio.currentTime)) writeLocal(placeKey, String(audio.currentTime));});
  audio.addEventListener('play', () => {
    players.forEach(other => { if (other !== audio) other.pause(); });
    if ('mediaSession' in navigator && 'MediaMetadata' in window) navigator.mediaSession.metadata = new MediaMetadata({title:`Chapter ${chapter.chapter_number}`,artist:batch.book_title,album:batch.book_title});
  });
  audio.addEventListener('error', () => {hint.textContent = 'Audio unavailable. Check Tailscale, then refresh this page to reconnect.';});
  players.push(audio); card.append(audio);
  const actions = node('div', 'actions');
  if (batch.allow_download) {const link = node('a','button secondary','Download'); link.href = `${audio.src}?download=true`; link.download = `Chapter_${chapter.chapter_number}.mp3`; actions.append(link);}
  const hint = node('p', 'status'); hint.setAttribute('aria-live','polite');
  if (batch.allow_feedback) {
    const open = node('button', '', 'Leave feedback'); open.type = 'button'; open.setAttribute('aria-expanded','false'); actions.append(open);
    const form = node('form', 'feedback-form'); form.hidden = true;
    const row = node('div','time-row');
    const label = node('label','','At'); const timestamp = node('input'); timestamp.type = 'text'; timestamp.inputMode = 'decimal'; timestamp.value = '0:00'; timestamp.id = `time-${chapter.chapter_id}`; timestamp.setAttribute('aria-label','Playback timestamp in minutes and seconds'); label.htmlFor = timestamp.id;
    const current = node('button','secondary','Use current time'); current.type = 'button'; current.addEventListener('click', () => {timestamp.value = time(audio.currentTime); saveDraft();}); row.append(label,timestamp,current);
    const text = node('textarea'); text.maxLength = 4000; text.rows = 3; text.placeholder = 'What did you notice here?'; text.setAttribute('aria-label',`Feedback for Chapter ${chapter.chapter_number}`);
    try {const draft = JSON.parse(readLocal(draftKey)); if (draft) {text.value = draft.text || ''; timestamp.value = draft.time || '0:00';}} catch {}
    function saveDraft() {writeLocal(draftKey, JSON.stringify({text:text.value,time:timestamp.value}));}
    text.addEventListener('input',saveDraft); timestamp.addEventListener('input',saveDraft);
    const save = node('button','','Save feedback'); save.type = 'submit'; const status = node('p','status');status.setAttribute('aria-live','polite');
    form.append(row,text,save,status);
    open.addEventListener('click', () => {form.hidden = !form.hidden;open.setAttribute('aria-expanded',String(!form.hidden));if(!form.hidden){if(!text.value) timestamp.value = time(audio.currentTime);text.focus();}});
    form.addEventListener('submit', async event => {
      event.preventDefault(); const parts = timestamp.value.trim().match(/^(\d+):([0-5]\d)$/);
      if (!parts || !text.value.trim()) {status.textContent = 'Add a note and a time like 12:43.';return;}
      const seconds = Number(parts[1])*60+Number(parts[2]); if (seconds > chapter.duration_seconds) {status.textContent = 'That time is past the end of this chapter.';return;}
      save.disabled = true; status.textContent = 'Saving…';
      try {await submit({chapter_id:chapter.chapter_id,playback_seconds:seconds,text:text.value.trim()}); text.value = '';saveDraft();status.textContent = `Saved at ${time(seconds)}. Thank you.`;await refreshFeedback();}
      catch(error){status.textContent = error.message;} finally {save.disabled = false;}
    });
    card.append(actions,hint,form);
  } else card.append(actions,hint);
  return card;
}
async function refreshFeedback() {
  const result = await json(`${api}/feedback`); $('saved-list').replaceChildren(); $('saved').hidden = !result.feedback.length;
  for (const feedback of result.feedback.slice().reverse()) {
    const chapter = batch.chapters.find(item => item.chapter_id === feedback.chapter_id);
    const label = chapter ? `Chapter ${chapter.chapter_number}${feedback.playback_seconds === null ? '' : ' · '+time(feedback.playback_seconds)}` : 'Overall feedback';
    const note = node('div','saved-note');note.append(node('span','saved-label',label),document.createTextNode(feedback.text)); $('saved-list').append(note);
  }
}
$('reload').addEventListener('click', () => location.reload());
(async () => {
  try {
    batch = await json(api); document.title = `${batch.book_title} · Listen & review`; $('book').textContent = batch.book_title.toUpperCase();
    const minutes = Math.round(batch.chapters.reduce((sum,item)=>sum+item.duration_seconds,0)/60);
    $('summary').textContent = `${batch.chapters.length} ${batch.chapters.length === 1 ? 'chapter' : 'chapters'} ready · ${minutes} minutes`;
    batch.chapters.forEach(chapter => $('chapters').append(buildChapter(chapter)));
    if(batch.allow_feedback){
      $('overall').hidden = false; const key = `audiobook-overall-${batchId}`;$('overall-text').value = readLocal(key)||'';
      $('overall-text').addEventListener('input',()=>writeLocal(key,$('overall-text').value));
      $('overall-form').addEventListener('submit',async event=>{event.preventDefault();const button=event.submitter;if(!$('overall-text').value.trim())return;button.disabled=true;$('overall-status').textContent='Saving…';try{await submit({text:$('overall-text').value.trim()});$('overall-text').value='';writeLocal(key,'');$('overall-status').textContent='Saved. Thank you.';await refreshFeedback();}catch(error){$('overall-status').textContent=error.message;}finally{button.disabled=false;}});
      await refreshFeedback();
    }
  } catch(error) {$('error').hidden=false;$('error-text').textContent=error.message;$('summary').textContent='Your private audiobook review';}
})();
