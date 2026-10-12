"""Explicit, isolated chapter review using existing recorded clones and Takes.

No approval is manufactured. Source text, supplied attribution and library
identities are frozen; review audio is separate from production selections.
Reuses QA-passed Takes from the approved Faster Base route. No GPU work,
+startup, download or model switch.
"""
from __future__ import annotations
import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import re
import sys

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO))


def digest(data):
    return hashlib.sha256(data).hexdigest()


def json_bytes(value):
    return (json.dumps(value,ensure_ascii=False,sort_keys=True,indent=2,allow_nan=False)+'\n').encode()


def publish(path, data):
    path.parent.mkdir(parents=True,exist_ok=True)
    try:
        with path.open('xb') as f:
            f.write(data); f.flush(); os.fsync(f.fileno())
    except FileExistsError:
        if path.read_bytes()!=data: raise ValueError(f'Immutable artifact conflict: {path}')


def partition_chapter(chapter, mapping):
    """Exact nonblank spans, with no fabricated speaker or dropped prose."""
    from dubbing_service.app.audiobook.book_creator import _utterances
    text=chapter['text']
    if mapping['text_sha256']!=digest(text.encode()): raise ValueError('Attribution is bound to different source text')
    quotes=quote_spans(text)
    supplied=mapping['quotes']
    if len(quotes)!=len(supplied): raise ValueError('Every quoted span requires an explicit classification')
    rows=[];cursor=0
    def narration(start,end):
        for item in _utterances(text[start:end],0):
            rows.append({'text':item.text,'start_offset':start+item.start_pos,'end_offset':start+item.end_pos,
                         'speaker':'Narrator','kind':'narration','confidence':1.0,'reason':'Exact unquoted source span'})
    for i,(quote,decision) in enumerate(zip(quotes,supplied)):
        if (decision['quote_index']!=i or decision['start_offset']!=quote.start()
                or decision['end_offset']!=quote.end()): raise ValueError('Attribution quote offsets/order changed')
        narration(cursor,quote.start())
        speaker=decision.get('speaker');kind=decision.get('kind');confidence=decision.get('confidence')
        if kind not in {'dialogue','narrative_quote','thought'}: raise ValueError('Unknown quote classification')
        if not isinstance(speaker,str) or not speaker.strip(): raise ValueError('Unresolved spoken attribution: no narrator fallback')
        if not isinstance(confidence,(float,int)) or isinstance(confidence,bool) or not 0<=confidence<=1:
            raise ValueError('Attribution confidence must be explicit')
        if kind!='dialogue' and speaker!='Narrator': raise ValueError('Narrative quotation must use the narrator')
        rows.append({'text':quote.group(),'start_offset':quote.start(),'end_offset':quote.end(),
                     'speaker':speaker,'kind':kind,'confidence':confidence,'reason':decision.get('reason')})
        cursor=quote.end()
    narration(cursor,len(text))
    cursor=0
    for i,row in enumerate(rows):
        if text[cursor:row['start_offset']].strip() or text[row['start_offset']:row['end_offset']]!=row['text']:
            raise ValueError('Segmentation lost or altered source prose')
        row['index']=i
        row['paragraph_index']=text[:row['start_offset']].count('\n\n')
        cursor=row['end_offset']
    if text[cursor:].strip(): raise ValueError('Segmentation omitted the end of the chapter')
    return rows


def quote_spans(text):
    """Line-bounded curly/straight dialogue, including a source-open final quote."""
    spans=[]
    for line in re.finditer(r'[^\r\n]+',text):
        start=None
        for i,char in enumerate(line.group()):
            if start is None and char in '“"':start=i
            elif start is not None and char in '”"':
                spans.append((line.start()+start,line.start()+i+1));start=None
        if start is not None:spans.append((line.start()+start,line.end()))
    class Span:
        def __init__(self,start,end):self.begin,self.finish=start,end
        def start(self):return self.begin
        def end(self):return self.finish
        def group(self):return text[self.begin:self.finish]
    return [Span(start,end) for start,end in spans]










def rebuild_approved(args):
    """Render new timed exports from approved immutable Takes; no synthesis."""
    import numpy as np
    import soundfile as sf
    from dubbing_service.app.audiobook.epub_chapter_selection import select_epub_chapters
    from dubbing_service.app.audiobook import voice_lab_chapter_export as exporter, performance_boundaries as boundaries, takes
    from dubbing_service.app.audiobook.cloned_voice_policy import require_record
    from dubbing_service.app.audiobook.voice_lab_render import _mp3,_loudness,_decoded_duration
    from dubbing_service.app.audiobook.voice_lab_library import _read_verified,_safe_path
    from dubbing_service.app.core.config import safe_project_root
    from dubbing_service.app.audiobook.runtime_binding import take_runtime_profiles
    from dubbing_service.app.audiobook import casting_director
    analysis_cache=args.output_dir/'analysis-cache';analysis_cache.mkdir(parents=True,exist_ok=True)
    book=select_epub_chapters(args.epub.read_bytes(),args.epub.name,chapter_numbers=[1,2,3])
    if len(args.reuse_projects)!=3:raise ValueError('Three approved chapter projects must be supplied in order')
    output=args.output_dir/'exports';output.mkdir(parents=True,exist_ok=True)
    for chapter,pid in zip(book['chapters'],args.reuse_projects):
        from dubbing_service.app.storage.projects import load_project
        project=load_project(pid)
        canonical=next((c for c in project.chapters if c.title==chapter['title']),None)
        if canonical is None:raise ValueError('Approved project does not contain the exact numbered chapter')
        root=safe_project_root(pid); before={p.name:digest(p.read_bytes()) for p in [root/'project.json',root/'segments.json']}
        project,segments,source=exporter._source_snapshot(pid,canonical.id)
        if not any(r.sha256==book['source_sha256'] for r in project.source_revisions):
            raise ValueError('Approved chapter is not bound to this exact EPUB')
        if re.sub(r'\s+','',chapter['text'])!=re.sub(r'\s+','',''.join(s.text for s in segments)):
            raise ValueError('Canonical chapter does not cover every EPUB prose character')
        roles={r.id:r for r in project.roles}
        for rid in {s.role_id for s in segments if s.id in source['speech_utterance_ids']}:
            role=roles[rid];require_record(role.voice_identity,mode='cloned_only')
            if role.tts_backend!='qwen3tts':raise ValueError('Cloned-only review requires Qwen Base assignments')
            for field,hfield in [('reference_audio','reference_audio_sha256'),('prompt_artifact','prompt_sha256')]:
                _read_verified(_safe_path(root,getattr(role.voice_identity,field)),getattr(role.voice_identity,hfield))
        paths,selections,evidence=exporter._take_snapshot(project,segments,source)
        speech=[s for s in segments if s.id in source['speech_utterance_ids']]
        profiles=set().union(*(take_runtime_profiles(takes.selected_take(s)) for s in speech))
        if profiles!={'faster-base-032'}:raise ValueError('Chapter Takes must prove the approved Faster Base route: '+str(profiles))
        frozen={'source_sha256':book['source_sha256'],'chapter_text_sha256':chapter['text_sha256'],
                'chapter_member':chapter['member'],'source_project_id':pid,'state_sha256':evidence['state_sha256'],
                'selections':selections,'timing_policy':boundaries.default_policy(),'voice_policy':'cloned_only'}
        n=chapter['chapter_number'];receipt=output/f'chapter-{n:03}.json'
        if receipt.exists():
            saved=json.loads(receipt.read_bytes())
            if saved['frozen']!=frozen:raise ValueError('Approved cast/source/Take state changed since export')
            for kind in ['master','mp3']:
                if digest((args.output_dir/saved[kind+'_file']).read_bytes())!=saved[kind+'_sha256']:raise ValueError('Export audio changed')
            print(json.dumps({'chapter':n,'reused_export':True}),flush=True);continue
        chunks=[];mapping=[];elapsed=0;rate=None;previous=None;previous_class=None
        for segment,path,selection in zip(speech,paths,selections):
            pcm,sr=sf.read(path,dtype='float32',always_2d=True)
            if rate is not None and rate!=sr:raise ValueError('Mixed PCM sample domains')
            rate=sr;measure=boundaries.measure_take(path,selection['audio_sha256'],text=segment.text)
            anchor=segment.source_anchor
            line={'utterance_id':segment.id,'text':segment.text,'role_id':segment.role_id,
                  'source_revision_id':anchor.source_revision_id,'start_offset':anchor.start_offset,'end_offset':anchor.end_offset}
            timing=None
            if previous:
                left,left_measure=previous
                line['previous_tag']=previous_class
                classification=boundaries.classify_boundary(left,line,role_types={rid:r.role_type.value for rid,r in roles.items()},
                                                            aliases=project.character_cast.role_aliases)
                klass=classification['class'];target=boundaries.default_policy()['targets_ms'].get(klass,350)
                # Canonical scene separators keep their recorded 1.25 s target.
                left_i=next(s.index for s in segments if s.id==left['utterance_id'])
                if any(left_i<b['index']<segment.index for b in source['breaks']):target=max(target,1250)
                existing=(left_measure['trailing']['silent_frames']+measure['leading']['silent_frames'])*1000/rate
                padding=max(0,round((target-existing)*rate/1000));chunks.append(np.zeros((padding,pcm.shape[1]),dtype='float32'));elapsed+=padding/rate
                timing={'classification':classification,'target_ms':target,'existing_silence_ms':existing,'added_frames':padding,'trimming':False}
                previous_class={'class':klass,'role_id':left['role_id']}
            mapping.append({**selection,'role_name':roles[segment.role_id].name,'source_anchor':anchor.model_dump(mode='json'),
                            'start_seconds':elapsed,'end_seconds':elapsed+len(pcm)/rate,'boundary':timing})
            chunks.append(pcm);elapsed+=len(pcm)/rate;previous=(line,measure)
        master=output/f'chapter-{n:03}.wav';mp3=output/f'chapter-{n:03}.mp3'
        buffer=io.BytesIO();sf.write(buffer,np.concatenate(chunks),rate,format='WAV',subtype='PCM_16');publish(master,buffer.getvalue())
        if not mp3.exists():_mp3(master,mp3)
        # Analysis is on a private projection: no saved identity, rating or approval changes.
        projection=project.model_copy(deep=True);projection.metadata['director_voice_policy']={'mode':'cloned_only'}
        analysis=casting_director._compute(projection,segments,casting_director._binding(projection,segments),
                                           scene_cache_root=analysis_cache)
        publish(args.output_dir/f'chapter-{n:03}-director-analysis.json',json_bytes(analysis))
        result={'schema':'cloned-chapter-export-v2','frozen':frozen,'chapter_number':n,
                'master_file':master.relative_to(args.output_dir).as_posix(),'master_sha256':digest(master.read_bytes()),
                'mp3_file':mp3.relative_to(args.output_dir).as_posix(),'mp3_sha256':digest(mp3.read_bytes()),
                'duration_seconds':_decoded_duration(mp3),'integrated_lufs':_loudness(mp3),'source_to_audio':mapping,
                'reused_immutable_take_count':len(selections),'new_synthesis_count':0,'human_review':'pending',
                'native_instruction_forwarding':False,'quality_improvement':'not_established',
                'technical_qa':'canonical_source_cast_release_and_take_hash_checks_passed'}
        after={p.name:digest(p.read_bytes()) for p in [root/'project.json',root/'segments.json']}
        if after!=before:raise ValueError('Canonical project/Take metadata changed during review; output not finalized')
        result['canonical_metadata_unchanged']=True
        publish(receipt,json_bytes(result));print(json.dumps({'chapter_export':n,'duration_seconds':result['duration_seconds'],'reused_takes':len(selections)}),flush=True)




def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--epub',type=Path,required=True)
    parser.add_argument('--reuse-approved-root',type=Path,required=True)
    parser.add_argument('--reuse-projects',nargs=3,required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    args=parser.parse_args()
    args.output_dir=args.output_dir.resolve();args.epub=args.epub.resolve()
    source_root=args.reuse_approved_root.resolve()
    if args.output_dir==source_root or args.output_dir.is_relative_to(source_root):
        parser.error('Output must be outside the approved source store')
    os.environ['DUBBING_PROJECTS_ROOT']=str(source_root)
    from dubbing_service.app.core.config import clear_config_caches
    clear_config_caches();rebuild_approved(args)


if __name__=='__main__':main()
