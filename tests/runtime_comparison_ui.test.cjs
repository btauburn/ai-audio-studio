'use strict';
const {test}=require('node:test');
const assert=require('node:assert/strict');
const vm=require('node:vm');
const fs=require('node:fs');
const path=require('node:path');
function surface(options={}){
  let detachedPlaying=0;const all=[];
  class Element{
    constructor(tag){this.tag=tag;this.children=[];this.events={};this.dataset={};this.disabled=false;this._text='';this.attached=true;all.push(this);}
    addEventListener(name,fn){(this.events[name]||=[]).push(fn);}
    async emit(name){for(const fn of this.events[name]||[])await fn();}
    append(...children){this.children.push(...children);}
    detach(){if(this.tag==='audio'&&!this.paused)detachedPlaying++;this.attached=false;this.children.forEach(c=>c.detach());}
    replaceChildren(){this.children.forEach(c=>c.detach());this.children=[];}
    set textContent(v){this.replaceChildren();this._text=v;}
    get textContent(){return this._text;}
  }
  class Audio extends Element{
    constructor(){super('audio');this.paused=true;this.history=[];}
    pause(){this.paused=true;this.history.push('pause');void this.emit('pause');}
    play(){this.paused=false;this.history.push('play');void this.emit('play');return Promise.resolve();}
    removeAttribute(name){delete this[name];}
    load(){this.paused=true;this.history.push('load');}
  }
  const host=new Element('details'),load=new Element('button'),box=new Element('div'),ordinary=new Audio();
  host.querySelector=q=>q==='button'?load:box;
  const doc={readyState:'complete',documentElement:{dataset:{}},querySelector:q=>q==='#runtime-comparison'?host:q==='#media'?ordinary:null,querySelectorAll:q=>all.filter(x=>x.tag==='audio'&&x.attached),createElement:tag=>tag==='audio'?new Audio():new Element(tag)};
  const calls=[];let failure=false;
  const item={audio_url:'/api/voice-lab/book/runtime-comparison/audio/short/baseline',wall_seconds:4,audio_seconds:8,x_realtime:2,gpu_peak_allocated_mib:6000};
  if(options.flags)item.asr_flags=['00:33–00:49: possible inserted final word'];
  const fetch=async(url,options)=>{calls.push({url,options});if(failure)throw Error('broken');return {ok:true,json:async()=>({status:'experimental',human_review:'pending',pairs:[{id:'short',label:'Approved narrator',text:'Hello.',baseline:item,candidate:{...item,audio_url:item.audio_url.replace('baseline','candidate')}}]})};};
  vm.runInNewContext(fs.readFileSync(path.join(__dirname,'../dubbing_service/webui/runtime-comparison.js'),'utf8'),{document:doc,location:{href:'https://desktop.example/voice-lab?project=book',origin:'https://desktop.example',pathname:'/voice-lab',search:'?project=book'},URL,URLSearchParams,fetch});
  return {doc,load,box,ordinary,all,calls,fail:()=>{failure=true;},detached:()=>detachedPlaying,button:label=>all.filter(x=>x.tag==='button'&&x.textContent===label&&x.attached).at(-1),player:()=>all.filter(x=>x.tag==='audio'&&x!==ordinary&&x.attached).at(-1)};
}
test('same comparison player switches lanes and never generates or writes ratings',async()=>{
  const s=surface();await s.load.emit('click');const p=s.player();
  s.ordinary.paused=false;await s.button('Play current Qwen').emit('click');assert.equal(s.ordinary.paused,true);assert.equal(p.paused,false);
  await s.button('Play Faster Qwen').emit('click');assert.equal(s.player(),p);assert.match(p.src,/candidate$/);
  assert.equal(s.doc.documentElement.dataset.runtimeComparison,'playing');
  assert.deepEqual(s.calls,[{url:'/api/voice-lab/book/runtime-comparison',options:undefined}]);
  assert.equal(s.detached(),0);
});
test('reload and failure stop the prior player before detaching it',async()=>{
  const s=surface();await s.load.emit('click');await s.button('Play current Qwen').emit('click');const first=s.player();
  await s.load.emit('click');assert.equal(first.paused,true);assert.equal(s.detached(),0);assert.notEqual(s.player(),first);
  await s.button('Play Faster Qwen').emit('click');s.fail();await s.load.emit('click');assert.equal(s.detached(),0);assert.equal(s.doc.documentElement.dataset.runtimeComparison,undefined);
  assert.equal(s.ordinary.events.play.length,1);
});
test('ordinary playback pauses comparison and restores ordinary shortcuts',async()=>{
  const s=surface();await s.load.emit('click');await s.button('Play Faster Qwen').emit('click');await s.ordinary.emit('play');
  assert.equal(s.player().paused,true);assert.equal(s.doc.documentElement.dataset.runtimeComparison,undefined);
});
test('native comparison controls also stop ordinary playback',async()=>{
  const s=surface();await s.load.emit('click');await s.button('Play Faster Qwen').emit('click');
  await s.ordinary.play();assert.equal(s.player().paused,true);
  await s.player().play();assert.equal(s.ordinary.paused,true);assert.equal(s.player().paused,false);
  assert.equal(s.doc.documentElement.dataset.runtimeComparison,'playing');
});
test('voice-rating shortcuts abstain while comparison plays',()=>{
  const {shortcut}=require('../dubbing_service/webui/voice-lab.js');
  const previous=global.document;global.document={documentElement:{dataset:{runtimeComparison:'playing'}}};
  try{assert.equal(shortcut({key:'5'}),null);assert.equal(shortcut({key:' ',code:'Space'}),null);}finally{global.document=previous;}
});
test('ASR flags remain diagnostic and do not trigger playback or writes',async()=>{
  const s=surface({flags:true});await s.load.emit('click');
  const notes=s.all.filter(x=>x.tag==='p'&&x.textContent.startsWith('Offline ASR flags'));
  assert.equal(notes.length,2);assert.match(notes[0].textContent,/verify by listening/);assert.match(notes[0].textContent,/Recognition errors/);
  assert.equal(s.player().paused,true);assert.deepEqual(s.calls,[{url:'/api/voice-lab/book/runtime-comparison',options:undefined}]);
});
