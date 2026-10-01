"""Pinned serial Base runtime; voice identity stays in CharacterVoiceRuntime.

The wrapper is deliberately private to the experimental worker. It accepts the
existing persisted prompt and production controls, never an unverified path
cached reference, new checkpoint, altered precision or fallback voice.
"""
from __future__ import annotations
import hashlib
import importlib.metadata as md
import inspect
import threading
import time
from pathlib import Path
from typing import Any

FASTER_VERSION='0.3.2'
GRAPH_MAX_SEQ_LEN=2048
# Reviewed registry wheel source. Checked before any interception.
GENERATE_SHA256='972d167ff41cb10593fe5dcfa9a4fd07d7a0ec680e86d86b9fd394bf2280c35f'

class GenerationCancelled(RuntimeError): pass
class GenerationTruncated(RuntimeError): pass

def classify_completion(steps: int, budget: int, prefill: int, capacity: int) -> str:
    if any(type(x) is not int for x in (steps,budget,prefill,capacity)):
        raise TypeError('codec limits must be integers')
    if steps<=0 or budget<=0 or prefill<0 or capacity<=1: raise ValueError('invalid codec limits')
    if prefill>=capacity or steps>=budget or steps>=capacity-prefill:
        raise GenerationTruncated(f'No confirmed EOS: frames={steps}, budget={budget}, prefill={prefill}, capacity={capacity}')
    return 'eos'

class FasterBaseAdapter:
    def __init__(self, fast: Any):
        self.fast=fast
        self.last_metadata: dict[str,Any]={}
        self.cancel_event=threading.Event()
        self.lock=threading.Lock()

    @classmethod
    def load(cls,path: Path):
        import torch
        if not path.is_absolute() or not (path/'model.safetensors').is_file():
            raise ValueError('Faster requires a verified absolute local Base snapshot')
        if not torch.cuda.is_available(): raise RuntimeError('Faster Base requires CUDA; no CPU fallback')
        if md.version('faster-qwen3-tts')!=FASTER_VERSION or md.version('qwen-tts')!='0.1.1' or md.version('transformers')!='4.57.3' or md.version('torch')!='2.10.0+cu130':
            raise RuntimeError('Faster Base requires 0.3.2 / qwen-tts 0.1.1 / Transformers 4.57.3 / torch 2.10.0+cu130')
        try: md.version('qwen-tts-hf')
        except md.PackageNotFoundError: pass
        else: raise RuntimeError('qwen-tts-hf and qwen-tts must never share this environment')
        from faster_qwen3_tts import FasterQwen3TTS
        from faster_qwen3_tts import generate
        digest=hashlib.sha256(Path(inspect.getfile(generate)).read_bytes()).hexdigest()
        if digest!=GENERATE_SHA256: raise RuntimeError('Pinned Faster decode source changed')
        # Loader ignores local_files_only in its Torch branch. Offline environment
        # and exact materialized absolute checkpoint prevent Hub resolution.
        fast=FasterQwen3TTS.from_pretrained(str(path),device='cuda:0',dtype=torch.bfloat16,attn_implementation='sdpa',max_seq_len=GRAPH_MAX_SEQ_LEN)
        fast.predictor_graph.do_sample=True;fast.predictor_graph.top_k=50
        fast.predictor_graph.top_p=1.0;fast.predictor_graph.temperature=.9
        start=time.perf_counter();fast.warmup(prefill_len=100)
        adapter=cls(fast);adapter.warmup_seconds=time.perf_counter()-start
        return adapter

    def create_voice_clone_prompt(self,**kwargs):
        return self.fast.model.create_voice_clone_prompt(**kwargs)

    def generate_voice_clone(self, *, text, language, voice_clone_prompt, **controls):
        import numpy as np
        import torch
        from faster_qwen3_tts.generate import fast_generate
        if not isinstance(text,str) or not text.strip() or len(text)>10000: raise ValueError('invalid synthesis text')
        if not isinstance(voice_clone_prompt,list) or len(voice_clone_prompt)!=1:raise ValueError('one persisted identity is required')
        expected={'do_sample':True,'top_k':50,'top_p':1.0,'temperature':.9,'repetition_penalty':1.05,'subtalker_dosample':True,'subtalker_top_k':50,'subtalker_top_p':1.0,'subtalker_temperature':.9,'non_streaming_mode':True}
        if any(controls.get(k)!=v for k,v in expected.items()): raise ValueError('unsupported sampling controls; predictor graph is pinned')
        budget=controls.get('max_new_tokens')
        if type(budget) is not int or not 1<=budget<=2048: raise ValueError('invalid token budget')
        if not self.lock.acquire(blocking=False):raise RuntimeError('Faster supports one serial GPU request')
        try:
            if self.cancel_event.is_set(): raise GenerationCancelled('cancelled before preparation')
            with torch.inference_mode():
                m,talker,config,tie,tam,tth,tpe,ref_codes=self.fast._prepare_generation(text=text,language=language,voice_clone_prompt=voice_clone_prompt,non_streaming_mode=True,append_silence=False)
                prefill=int(tie.shape[1])
                if prefill+budget>=GRAPH_MAX_SEQ_LEN:raise ValueError('text/prompt budget exceeds static cache capacity 2048; use frozen production segmentation')
                graph=self.fast.talker_graph;predictor=self.fast.predictor_graph
                run_talker=graph.run;run_predictor=predictor.run
                def checked_talker(*a,**k):
                    if self.cancel_event.is_set():raise GenerationCancelled('cancelled during talker decode')
                    return run_talker(*a,**k)
                def checked_predictor(*a,**k):
                    if self.cancel_event.is_set():raise GenerationCancelled('cancelled during predictor decode')
                    return run_predictor(*a,**k)
                graph.run=checked_talker;predictor.run=checked_predictor
                try:
                    start=time.perf_counter()
                    codec,timing=fast_generate(talker=talker,talker_input_embeds=tie,attention_mask=tam,trailing_text_hiddens=tth,tts_pad_embed=tpe,config=config,predictor_graph=predictor,talker_graph=graph,max_new_tokens=budget,min_new_tokens=2,temperature=.9,top_k=50,top_p=1.,do_sample=True,repetition_penalty=1.05,subtalker_dosample=True,subtalker_top_k=50,subtalker_top_p=1.,subtalker_temperature=.9)
                finally: graph.run=run_talker;predictor.run=run_predictor
                if codec is None: raise ValueError('Faster returned no codec frames')
                steps=int(codec.shape[0]);reason=classify_completion(steps,budget,prefill,GRAPH_MAX_SEQ_LEN)
                if self.cancel_event.is_set():raise GenerationCancelled('cancelled before waveform decode')
                decode_codes=torch.cat([ref_codes.to(codec.device),codec],dim=0) if ref_codes is not None else codec
                waves,rate=m.speech_tokenizer.decode({'audio_codes':decode_codes.unsqueeze(0)})
                ref_len=int(ref_codes.shape[0]) if ref_codes is not None else 0
                arrays=[]
                for wave in waves:
                    a=wave.flatten().cpu().numpy() if hasattr(wave,'cpu') else np.asarray(wave).flatten()
                    if ref_len:a=a[int(ref_len/len(decode_codes)*len(a)):]
                    if not len(a) or not np.isfinite(a).all():raise ValueError('Faster returned invalid waveform')
                    arrays.append(a)
                if self.cancel_event.is_set():raise GenerationCancelled('cancelled after waveform decode')
                self.last_metadata={'runtime_profile':'faster-base-032','faster_version':FASTER_VERSION,'termination_reason':reason,'generated_codec_steps':steps,'prefill_tokens':prefill,'graph_capacity':GRAPH_MAX_SEQ_LEN,'append_silence':False,'warmup_seconds':getattr(self,'warmup_seconds',None),'generation_seconds':time.perf_counter()-start,'timing':timing,'fallback':None,'streaming':False}
                return arrays,rate
        finally:self.lock.release()
