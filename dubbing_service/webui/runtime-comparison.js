/* Explicit advanced review: one media element, no generation or casting writes. */
(function(){'use strict';
  function mount(){
    const host=document.querySelector('#runtime-comparison');if(!host)return;
    const load=host.querySelector('button');let player=null;
    document.querySelector('#media')?.addEventListener('play',()=>player?.pause());
    load.addEventListener('click',async()=>{
      if(player){player.pause();player.removeAttribute('src');player.load();player=null;}
      delete document.documentElement.dataset.runtimeComparison;
      load.disabled=true;const box=host.querySelector('[data-pairs]');box.textContent='Loading paired recordings…';
      try{
        const project=new URLSearchParams(location.search).get('project');if(!project)throw Error('Open an existing Voice Lab project first.');
        const privateRoute=location.pathname==='/voice-lab';
        const origin=privateRoute?`/api/voice-lab/${encodeURIComponent(project)}`:`/api/v1/projects/${encodeURIComponent(project)}/voice-lab`;
        const response=await fetch(origin+'/runtime-comparison');if(!response.ok)throw Error('Runtime comparison is unavailable.');
        const data=await response.json();box.replaceChildren();
        const status=document.createElement('p');status.textContent=`${data.status || 'experimental'} · Human voice review: ${data.human_review || 'pending'}`;box.append(status);
        if(!data.pairs?.length){status.textContent=data.message || 'No completed pairs for this project.';return;}
        player=document.createElement('audio');player.controls=true;player.preload='none';box.append(player);
        player.addEventListener('play',()=>{document.querySelectorAll('audio').forEach(a=>{if(a!==player)a.pause();});document.documentElement.dataset.runtimeComparison='playing';});
        player.addEventListener('pause',()=>{delete document.documentElement.dataset.runtimeComparison;});
        player.addEventListener('ended',()=>{delete document.documentElement.dataset.runtimeComparison;});
        for(const pair of data.pairs){
          const section=document.createElement('section'), title=document.createElement('h3');title.textContent=pair.label;section.append(title);
          const text=document.createElement('p');text.textContent=pair.text;section.append(text);
          for(const lane of ['baseline','candidate']){
            const item=pair[lane], button=document.createElement('button');button.type='button';button.textContent=lane==='baseline'?'Play current Qwen':'Play Faster Qwen';button.disabled=!item?.audio_url;
            button.addEventListener('click',()=>{const url=new URL(item.audio_url,location.href);if(url.origin!==location.origin)return;document.querySelectorAll('audio').forEach(a=>{if(a!==player)a.pause();});player.pause();player.src=url.href;player.play().catch(()=>{});});section.append(button);
            const info=document.createElement('p');info.className='muted';info.textContent=item?`${lane}: ${item.wall_seconds.toFixed(2)}s generation · ${item.audio_seconds.toFixed(2)}s audio · ${item.x_realtime.toFixed(2)}× realtime · ${item.gpu_peak_allocated_mib.toFixed(0)} MiB peak allocated VRAM`:lane+': failed or incomplete';section.append(info);
            if(item?.asr_flags?.length){const flags=document.createElement('p');flags.className='muted';flags.textContent='Offline ASR flags; verify by listening: '+item.asr_flags.join('; ')+'. Recognition errors can resemble speech errors.';section.append(flags);}
          }
          box.append(section);
        }
        const notes=document.createElement('p');notes.textContent='Listen for identity, accent, age, timbre, emotion, pauses, buzzing, repeats, omitted words and complete endings. Long recordings also need chunk-boundary review. Your voice ratings and cast remain unchanged. This experiment cannot promote itself.';box.append(notes);
      }catch(e){box.textContent=e.message;}finally{load.disabled=false;}
    });
  }
  if(document.readyState==='loading')document.addEventListener('DOMContentLoaded',mount,{once:true});else mount();
})();
