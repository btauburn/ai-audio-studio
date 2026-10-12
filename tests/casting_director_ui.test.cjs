'use strict';
const {test}=require('node:test');
const assert=require('node:assert/strict');
const fs=require('node:fs');
const path=require('node:path');
const webui=path.join(__dirname,'../dubbing_service/webui');
const {mount}=require(path.join(webui,'casting-director.js'));

class Element{
  constructor(tag){this.tag=tag;this.children=[];this.parentNode=null;this.events={};this.dataset={};this.attrs={};this.hidden=false;this.disabled=false;this.className='';this._text='';this.scrolled=0;this.focused=0;}
  addEventListener(name,fn){(this.events[name]||=[]).push(fn);}
  async emit(name){for(const fn of this.events[name]||[])await fn();}
  setAttribute(k,v){this.attrs[k]=v;}
  append(...nodes){for(const n of nodes){n.parentNode?.children.splice(n.parentNode.children.indexOf(n),1);n.parentNode=this;this.children.push(n);}}
  replaceChildren(){this.children.forEach(c=>{c.parentNode=null;});this.children=[];}
  set textContent(v){this.replaceChildren();this._text=String(v);}
  get textContent(){return this._text+this.children.map(c=>c.textContent).join('');}
  scrollIntoView(){this.scrolled++;}
  focus(){this.focused++;}
  walk(){return [this,...this.children.flatMap(c=>c.walk())];}
  querySelector(sel){const cls=sel.startsWith('.')?sel.slice(1):null;return this.walk().find(n=>cls&&n.className.split(' ').includes(cls))||null;}
}
class Audio extends Element{
  constructor(){super('audio');this.paused=true;this.history=[];}
  pause(){if(this.paused)return;this.paused=true;this.history.push('pause');void this.emit('pause');}
  play(){this.paused=false;this.history.push('play');void this.emit('play');return Promise.resolve();}
  removeAttribute(name){delete this[name];}
  load(){this.paused=true;this.history.push('load');}
}
function json(status,body){return {ok:status>=200&&status<300,status,json:async()=>{if(body===undefined)throw Error('html');return body;}};}
function ranked(over={}){
  return {candidate_id:'lib:a',key:'voice-a',name:'Warm Alto',origin:'approved_library',library_voice_id:'lib',library_version:1,
    reference_conditioned:true,protected:false,eligible:true,holds:[],score:0.7312,confidence:0.64,
    explanation:{summary:'Warm Alto: fit 0.80. Recommendation only; a human must audition and approve.',strengths:['matches age'],concerns:['sounds similar to Mara'],unmeasured:['fidelity','human_rating']},
    provenance:{source_urls:['https://example.test/a']},audio_url:'/api/v1/projects/book/voice-lab/audio/voice-a',audio_sha256:'a'.repeat(64),
    audition_status:'historical_sample',comparison_status:'fresh_matched_script_required',...over};
}
function role(id,over={}){
  return {role_id:id,role_name:id==='r1'?'Elena':'Mara',protected:false,needs_casting:true,replacement_candidate:false,coverage:'recommendations',
    current_voice:{key:'cur',name:'Current Voice',rating:2,approved:false,protected:false,audio_sha256:null},
    analysis:{role_id:id,role_name:'Elena',role_type:'character',spoken_lines:30,importance:'major',confidence:0.8,
      evidence:[{dimension:'age',value:'elderly',level:'explicit',basis:'chapter 2',evidence_ids:['e1']}],missing_evidence:['accent'],
      relationships:[{role_id:'r2',role_name:'Mara',needs_contrast:true}]},
    candidates:[ranked(),ranked({candidate_id:'lib:b',key:'voice-b',name:'Quiet Tenor',audio_url:null,audio_sha256:undefined,provenance:{rights_status:'licensed_use_recorded'}})],...over};
}
const REPORT={schema:'casting-director-v1',project_id:'book',report_sha256:'f'.repeat(64),human_review:'pending',holds:[],
  roles:[role('r1'),role('r2',{protected:true,needs_casting:false,replacement_candidate:true})],
  scenes:{requested_controls:['emotion','pace'],supported_controls:['pace'],scenes:[{scene_id:'s1',label:'Scene 1'}],
    directions:[{scene_id:'s1',emotion:'wary',intention:'deflect',pace_hint:'quick',emphasis:['never'],semantic_boundary:'after "never"'}]}};

function surface({search='?project=book',pathname='/voice-lab-ui',routes={},missingProject=false}={}){
  const ids={};const mk=(id,tag='div')=>{const e=new Element(tag);e.id=id;ids[id]=e;return e;};
  const host=mk('casting-director','section');host.hidden=true;
  const open=mk('casting-director-open','button'),analyze=mk('casting-director-analyze','button'),body=mk('casting-director-body');body.hidden=true;
  const status=mk('casting-director-status','p'),roles=mk('casting-director-roles'),scenes=mk('casting-director-scenes');
  const ordinary=new Audio(),other=new Audio();const inbox=mk('casting-inbox','section'),card1=mk('casting-role-r1','article');
  const avatar=new Element('img');avatar.className='casting-avatar';avatar.src='data:image/png;base64,AAA';card1.append(avatar);
  const library=mk('reusable-voices','details'),tools=mk('casting-extra-tools','details');
  const players=[];
  const doc={readyState:'complete',documentElement:{dataset:{}},getElementById:id=>ids[id]||null,
    querySelector:q=>q==='#media'?ordinary:null,
    querySelectorAll:q=>q==='audio'?[ordinary,other,...players]:[],
    createElement:tag=>{if(tag==='audio'){const a=new Audio();players.push(a);return a;}return new Element(tag);}};
  const calls=[];
  const fetch=async(url,init)=>{calls.push({url,method:init?.method||'GET',headers:init?.headers,body:init?.body?JSON.parse(init.body):undefined});
    const route=routes[(init?.method||'GET')+' '+url.replace(/^.*\/directors/,'')];
    if(typeof route==='function')return route();return route||json(404);};
  const win={location:{href:'https://studio.test/voice-lab-ui'+search,origin:'https://studio.test',search,pathname},fetch};
  const api=missingProject?mount(doc,{...win,location:{...win.location,search:''}}):mount(doc,win);
  const texts=()=>roles.walk().concat(scenes.walk(),[status]).map(n=>n._text).filter(Boolean);
  return {api,doc,host,open,analyze,body,status,roles,scenes,ordinary,other,players,calls,inbox,card1,library,tools,ids,texts,
    all:()=>texts().join('\n'),
    button:label=>roles.walk().filter(n=>n.tag==='button'&&n._text===label)};
}
const GET_OK={'GET ':json(200,REPORT)};

test('mount reveals the panel but never fetches or generates',()=>{
  const s=surface({routes:GET_OK});
  assert.equal(s.host.hidden,false);assert.equal(s.calls.length,0);assert.equal(s.players.length,0);
  assert.equal(s.body.hidden,true);assert.equal(s.analyze.disabled,false);
});

test('opening performs one read-only GET and renders evidence, state and ranked recommendations',async()=>{
  const s=surface({routes:GET_OK});await s.open.emit('click');
  assert.deepEqual(s.calls.map(c=>[c.method,c.url]),[['GET','/api/v1/projects/book/voice-lab/directors']]);
  assert.equal(s.calls[0].body,undefined);assert.equal(s.body.hidden,false);assert.equal(s.open.attrs['aria-expanded'],'true');
  const all=s.all();
  for(const expected of ['Elena','age: elderly (explicit)','Current Voice','2 of 5 stars','Needs casting','Replacement candidate','Protected · cannot be recast here','Warm Alto','Quiet Tenor','Approved library voice','Reference-conditioned','sounds similar to Mara','matches age','accent','Shares scenes with Mara; choose a clearly different voice'])
    assert.ok(all.includes(expected),'missing: '+expected);
  assert.ok(all.includes('0.73'));assert.ok(all.includes('64%'));
  assert.ok(all.includes('Recommendations never approve or replace a voice'));
  assert.ok(all.includes('Human review: pending'));
  const avatar=s.roles.walk().find(n=>n.tag==='img');assert.equal(avatar.src,'data:image/png;base64,AAA');
  assert.equal(s.calls.length,1);
});

test('rights are never invented: absent rights_status is shown as not recorded',async()=>{
  const s=surface({routes:GET_OK});await s.open.emit('click');
  const facts=s.roles.walk().filter(n=>n.className==='cd-fact').map(n=>n.textContent);
  assert.ok(facts.some(t=>t.includes('Rights status Not recorded (none assumed)')));
  assert.ok(facts.some(t=>t.includes('Rights status licensed use recorded')));
});

test('scene recommendations are shown as source-bound, unapplied direction',async()=>{
  const s=surface({routes:GET_OK});await s.open.emit('click');
  const all=s.all();
  for(const expected of ['Scene 1','wary','deflect','quick','never','after "never"','Engine control capabilities','not applied to any voice or recording'])
    assert.ok(all.includes(expected),'missing: '+expected);
});

test('a present capability record does not claim that default instructions are forwarded',async()=>{
  const report={...REPORT,scenes:{...REPORT.scenes,controls:{requested:['native_instruction'],supported:{
    'faster-base-032':{available:true,controls:{native_instruction:{state:'available_not_forwarded'},emotion_label:{state:'unsupported'}}},
    'faster-base-032-direction-x1':{available:true,controls:{native_instruction:{state:'forwarded_experimental'}}}
  }}}};
  const s=surface({routes:{'GET ':json(200,report)}});await s.open.emit('click');
  const all=s.all();
  assert.ok(all.includes('faster-base-032: native instruction available not forwarded'));
  assert.ok(all.includes('emotion label unsupported'));
  assert.ok(all.includes('native instruction forwarded experimental'));
  assert.equal(s.calls.length,1);
});

test('samples play only on click, one player at a time, and unverified or foreign audio is not playable',async()=>{
  const foreign=ranked({candidate_id:'x',audio_url:'https://evil.test/a.wav'});
  const report={...REPORT,roles:[role('r1',{candidates:[ranked(),ranked({candidate_id:'nohash',audio_sha256:undefined}),foreign]})]};
  const s=surface({routes:{'GET ':json(200,report)}});await s.open.emit('click');
  assert.equal(s.players.length,0);assert.equal(s.calls.length,1);
  const plays=s.roles.walk().filter(n=>n.className.includes('cd-play'));assert.equal(plays.length,1);
  s.other.paused=false;s.ordinary.paused=false;
  await plays[0].emit('click');
  const player=s.players[0];assert.equal(player.src,'https://studio.test/api/v1/projects/book/voice-lab/audio/voice-a');
  assert.equal(player.paused,false);assert.equal(s.other.paused,true);assert.equal(s.ordinary.paused,true);
  assert.equal(s.doc.documentElement.dataset.runtimeComparison,'playing');
  await s.ordinary.emit('play');assert.equal(player.paused,true);assert.equal(s.doc.documentElement.dataset.runtimeComparison,undefined);
  assert.equal(s.calls.length,1,'playback never calls the service');
});

test('closing or reloading stops audition playback',async()=>{
  const s=surface({routes:GET_OK});await s.open.emit('click');
  await s.roles.walk().find(n=>n.className.includes('cd-play')).emit('click');const player=s.players[0];
  await s.open.emit('click');assert.equal(s.body.hidden,true);assert.equal(player.paused,true);
  assert.equal(s.doc.documentElement.dataset.runtimeComparison,undefined);
});

test('analyze is an explicit POST with the review header and renders the result',async()=>{
  const s=surface({routes:{'GET ':json(200,{status:'needs_analysis',roles:[],scenes:null,holds:[],human_review:'pending'}),'POST /analyze':json(200,REPORT)}});
  await s.open.emit('click');assert.ok(s.all().includes('has not been analyzed yet'));
  assert.deepEqual(s.calls.map(c=>c.method),['GET']);
  await s.analyze.emit('click');
  const post=s.calls[1];assert.equal(post.method,'POST');assert.equal(post.url,'/api/v1/projects/book/voice-lab/directors/analyze');
  assert.equal(post.headers['X-Review-Request'],'1');assert.ok(s.all().includes('Elena'));
});

test('requesting auditions sends role, evidence hash and a stable request_id once',async()=>{
  let attempt=0;
  const s=surface({routes:{'GET ':json(200,REPORT),'POST /auditions':()=>++attempt===1?json(503,undefined):json(200,{project_id:'book',role_id:'r1',request_id:'x',round_id:'rnd-1',candidate_count:2,status:'queued',holds:[],human_review:'pending'})}});
  await s.open.emit('click');
  const [request]=s.button('Request fresh auditions');
  await request.emit('click');
  assert.ok(s.all().includes('not available')||s.roles.textContent.includes('not available'));
  assert.equal(request.disabled,false,'an unavailable service leaves retry possible');
  await request.emit('click');
  const posts=s.calls.filter(c=>c.url.endsWith('/auditions'));
  assert.equal(posts.length,2);
  assert.deepEqual(Object.keys(posts[0].body).sort(),['report_sha256','request_id','role_id']);
  assert.equal(posts[0].body.role_id,'r1');assert.equal(posts[0].body.report_sha256,REPORT.report_sha256);
  assert.match(posts[0].body.request_id,/^[A-Za-z0-9_-]{1,100}$/);
  assert.equal(posts[1].body.request_id,posts[0].body.request_id,'retry replays the same identity');
  assert.equal(posts[0].headers['X-Review-Request'],'1');
  assert.ok(s.roles.textContent.includes('Auditions queued: 2 candidate(s), round rnd-1'));
  assert.equal(request.disabled,true,'a queued receipt cannot be queued twice');
});

test('simultaneous clicks queue one request',async()=>{
  let release;const gate=new Promise(r=>{release=r;});
  const s=surface({routes:{'GET ':json(200,REPORT),'POST /auditions':()=>({ok:true,status:200,json:async()=>{await gate;return {round_id:'r',candidate_count:2,status:'queued',holds:[]};}})}});
  await s.open.emit('click');const [request]=s.button('Request fresh auditions');
  const first=request.emit('click'),second=request.emit('click');release();await Promise.all([first,second]);
  assert.equal(s.calls.filter(c=>c.url.endsWith('/auditions')).length,1);
});

test('protected roles cannot request auditions and show why',async()=>{
  const s=surface({routes:GET_OK});await s.open.emit('click');
  const requests=s.button('Request fresh auditions');assert.equal(requests.length,2);
  assert.equal(requests[1].disabled,true);assert.match(requests[1].title,/protected/);
  await requests[1].emit('click');assert.equal(s.calls.filter(c=>c.url.endsWith('/auditions')).length,0);
});

test('refused requests surface the service reason and say nothing was queued',async()=>{
  const s=surface({routes:{'GET ':json(200,REPORT),'POST /auditions':json(409,{detail:'Casting evidence changed; refresh before requesting auditions'})}});
  await s.open.emit('click');const [request]=s.button('Request fresh auditions');await request.emit('click');
  assert.ok(s.roles.textContent.includes('Casting evidence changed; refresh before requesting auditions Nothing was queued.'));
});

test('unavailable endpoint degrades gracefully with no writes',async()=>{
  for(const routes of [{},{'GET ':json(502,undefined)},{'GET ':()=>{throw Error('network');}}]){
    const s=surface({routes});await s.open.emit('click');
    assert.ok(s.status.textContent.includes('not available on this connection'));
    assert.equal(s.status.dataset.tone,'error');assert.deepEqual(s.calls.map(c=>c.method),['GET']);
    assert.equal(s.open.disabled,false);assert.equal(s.roles.children.length,0);
  }
});

test('private route uses the proxied voice-lab base and encodes the project',async()=>{
  const s=surface({search:'?project=my%20book',pathname:'/voice-lab',routes:GET_OK});await s.open.emit('click');
  assert.equal(s.calls[0].url,'/api/voice-lab/my%20book/directors');
});

test('without a project nothing is fetched and the user is told to open one',async()=>{
  const s=surface({missingProject:true,routes:GET_OK});await s.open.emit('click');
  assert.equal(s.calls.length,0);assert.ok(s.status.textContent.includes('Open a Voice Lab project first'));assert.equal(s.analyze.disabled,true);
});

test('casting controls are reused: the inbox button navigates without any request',async()=>{
  const s=surface({routes:GET_OK});await s.open.emit('click');
  const [inbox]=s.button('Open in casting inbox');await inbox.emit('click');
  assert.equal(s.card1.scrolled,1);assert.equal(s.card1.focused,1);
  const [,second]=s.button('Open in casting inbox');await second.emit('click');
  assert.equal(s.inbox.scrolled,1,'falls back to the inbox when the card is hidden');
  const library=s.roles.walk().find(n=>n.className.includes('cd-library'));await library.emit('click');
  assert.equal(s.library.open,true);assert.equal(s.tools.open,true);
  assert.equal(s.calls.length,1);
  const labels=s.roles.walk().filter(n=>n.tag==='button').map(n=>n._text).join('|');
  assert.doesNotMatch(labels,/approve|accept|reject|assign|rate|favorite/i,'no duplicate mutation controls');
});

test('only the three Casting Director endpoints are ever used',async()=>{
  const s=surface({routes:{...GET_OK,'POST /analyze':json(200,REPORT),'POST /auditions':json(200,{round_id:'r',candidate_count:1,status:'queued',holds:[]})}});
  await s.open.emit('click');await s.analyze.emit('click');await s.button('Request fresh auditions')[0].emit('click');
  for(const call of s.calls)assert.match(call.url,/^\/api\/v1\/projects\/book\/voice-lab\/directors(\/analyze|\/auditions)?$/);
});

test('page wiring is additive, hidden until mounted, and rendering never uses innerHTML',()=>{
  const html=fs.readFileSync(path.join(webui,'voice-lab.html'),'utf8');
  assert.match(html,/<section id="casting-director"[^>]*\shidden>/);
  assert.match(html,/id="casting-director-open"/);assert.match(html,/id="casting-director-analyze"/);
  assert.match(html,/<script src="\/web\/casting-director\.js" defer><\/script>/);
  assert.match(html,/<link rel="stylesheet" href="\/web\/casting-director\.css">/);
  assert.match(html,/<script defer src="\/web\/voice-lab\.js"><\/script>/);
  assert.match(html,/id="reusable-voices"/);
  const source=fs.readFileSync(path.join(webui,'casting-director.js'),'utf8');
  assert.doesNotMatch(source,/innerHTML|outerHTML|insertAdjacentHTML|eval\(|new Function/);
  assert.ok(fs.statSync(path.join(webui,'casting-director.css')).size>0);
});

test('cloned-only book policy is visible without generating or changing approvals',async()=>{
  const report={...REPORT,voice_policy:'cloned_only'};
  const s=surface({routes:{'GET ':json(200,report)}});
  await s.open.emit('click');
  const copy=s.roles.walk().map(n=>n._text).join(' ');
  assert.match(copy,/Cloned voices only/);
  assert.match(copy,/existing approvals remain saved/);
  assert.equal(s.calls.length,1);
  assert.equal(s.calls[0].method,'GET');
});
