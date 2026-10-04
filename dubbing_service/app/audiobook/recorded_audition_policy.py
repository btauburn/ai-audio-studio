"""Exact screened recordings may form unapproved drafts, never implicit selections.

The user approved this bounded policy on2026-10-03. Unknown source delivery is
retained. Real source/technical rejects remain blocking. Production requires
an exact current human casting event; favorites/assignment alone never qualify.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import re
import numpy as np
from . import official_recording_sources, reference_quality, reference_source_qc

SCHEMA = 'recorded-audition-draft-v1'
EVIDENCE_SHA256 = 'a690014f78cade9032efeac5b2fa38f5b96c574e4298e0707e215b9a30a07012'
AUTHORIZATION = 'Sentinel_b862b312ba848191ba41f987ce2c4dfe'
_REQUIRED = {'decode','clipping','signal','speech','asr','transcript_alignment','crop_coverage'}

def _sha(raw):
    return hashlib.sha256(raw).hexdigest()

def _records():
    raw=Path(__file__).with_name('reviewed_recorded_drafts.json').read_bytes()
    if _sha(raw)!=EVIDENCE_SHA256:
        raise ValueError('Recorded audition evidence changed; explicit revalidation required')
    return json.loads(raw)['records']

def reference_record(wave, rate):
    pcm=_sha(wave.tobytes())
    return next((r for r in _records() if r['input_pcm_sha256']==pcm and r['sample_rate']==rate),None)

def _check_evidence(row):
    quality=row['quality'];classifier=row['source_classifier'];pipeline=quality['pipeline']
    root=Path(__file__).resolve().parent
    if (quality.get('status')!='pass' or any(quality.get('checks',{}).get(k,{}).get('status')!='pass' for k in _REQUIRED)
            or quality.get('expected_text')!=row['transcript'] or quality.get('expected_text_sha256')!=row['transcript_sha256']
            or quality.get('expected_text_origin')!='provided'
            or pipeline.get('source_sha256')!=_sha((root/'reference_quality.py').read_bytes())
            or pipeline.get('asr_source_sha256')!=_sha((root/'reference_transcripts.py').read_bytes())):
        raise ValueError('Recorded draft has unavailable/stale independent transcript or technical QC')
    expected=reference_quality._tokens(row['transcript']);publisher=reference_quality._tokens(row['publisher_transcript'])
    if not expected or publisher[:len(expected)]!=expected:
        raise ValueError('Recorded draft text is not bound to independent publisher words')
    if (classifier.get('status')!='pass' or classifier.get('contaminated') is not False
            or classifier.get('input_pcm_sha256')!=row['input_pcm_sha256']
            or classifier.get('sample_rate')!=row['sample_rate']
            or classifier.get('pipeline',{}).get('models')!=reference_source_qc.MODEL_FILES
            or classifier.get('pipeline',{}).get('settings')!=reference_source_qc.SETTINGS
            or classifier.get('pipeline',{}).get('source_sha256')!=reference_source_qc.SOURCE_SHA
            or reference_source_qc.evaluate_windows(classifier.get('windows',[])).get('status')!='pass'):
        raise ValueError('Recorded draft music/noise/overlap/speaker evidence is not passing')
    if (row.get('scope')!='unapproved_recorded_audition_draft' or row.get('authorization_event')!=AUTHORIZATION
            or row.get('human_verified') is not False or row.get('production_approved') is not False
            or row.get('source_delivery_fit')!='unknown'):
        raise ValueError('Recorded draft policy cannot invent human or production approval')

def policy_for(wave,rate,*,source,original_sha256,transcript,start_seconds,end_seconds):
    if (official_recording_sources.reviewed(source) is None
            or source.get('provider')!='reviewed_dataset_recording'):
        raise ValueError('Recorded draft requires an exact trusted corpus original')
    row=reference_record(wave,rate)
    if row is None or any(row[k]!=v for k,v in {
        'source_id':source['source_id'],'original_sha256':original_sha256,
        'transcript':transcript,'transcript_sha256':_sha(transcript.encode()),
        'start_seconds':start_seconds,'end_seconds':end_seconds}.items()):
        raise ValueError('Recorded draft does not match the exact screened PCM/source/text/span')
    _check_evidence(row)
    return {'schema':SCHEMA,'draft_id':row['draft_id'],'evidence_sha256':EVIDENCE_SHA256,
        'authorization_event':AUTHORIZATION,'source_delivery_fit':'unknown','human_verified':False,
        'production_approved':False,'selection_requires_human_casting':True}

def verify_policy(policy,wave,rate,**kwargs):
    expected=policy_for(wave,rate,**kwargs)
    if policy!=expected:
        raise ValueError('Recorded audition draft policy differs from pinned objective evidence')
    return expected

def _json(path):
    if path.is_symlink() or path.stat().st_size>8_000_000:
        raise ValueError('Recorded draft owner metadata is outside bounds')
    return json.loads(path.read_text('utf-8'))

def _safe(root,relative):
    root=root.resolve();path=(root/relative).resolve();path.relative_to(root)
    if path.is_symlink():raise ValueError('Recorded draft artifact cannot be a symlink')
    return path

def audition_context(root,identity_id,acquisition_sha256,text):
    """Verify both owner links and the exact fixed source before model loading."""
    try:
        from . import voice_lab as lab, voice_lab_audition as audition
        from dubbing_service.app.models.project import Project
        child_raw=_json(root/'project.json');child=Project.model_validate(child_raw)
        meta=child.metadata.get('voice_lab_audition',{})
        parent_id=meta.get('parent_project_id','');intent_hash=meta.get('intent','')
        if not re.fullmatch(r'[A-Za-z0-9_-]{1,100}',parent_id) or not re.fullmatch(r'[a-f0-9]{64}',intent_hash):return False
        parent=_json(root.parent/parent_id/'project.json')
        intent=parent.get('metadata',{}).get('voice_lab_auditions',{}).get('intents',{}).get(intent_hash,{})
        binding=intent.get('binding',{})
        if intent.get('audition_project_id')!=root.name or lab.digest(binding)!=intent_hash or binding.get('reuse_key')!=intent.get('reuse_key'):return False
        record=parent.get('voice_lab',{}).get('voices',{}).get(intent.get('reuse_key'),{})
        identity=record.get('identity') or {}
        if (identity.get('id')!=identity_id or identity.get('provenance',{}).get('acquisition_sha256')!=acquisition_sha256
                or not identity.get('provenance',{}).get('audition_draft_policy')
                or len(child.roles)!=1 or not child.roles[0].voice_identity or child.roles[0].voice_identity.id!=identity_id):return False
        rows=audition._verified_preview_source(child,binding)
        return any(row.text==text for row in rows)
    except (OSError,ValueError,TypeError,KeyError):
        return False

def human_casting_context(root,identity_id,acquisition_sha256):
    """Current positive human audition selection, with immutable sample binding."""
    try:
        project=_json(root/'project.json');state=project.get('voice_lab',{}).get('casting',{})
        for role_id,item in state.get('roles',{}).items():
            assignment=item.get('assignment') or {};key=assignment.get('key')
            record=project.get('voice_lab',{}).get('voices',{}).get(key,{})
            identity=record.get('identity') or {};event=state.get('events',{}).get(assignment.get('event_id'),{})
            role=next((r for r in project.get('roles',[]) if r.get('id')==role_id),{})
            if (assignment.get('approved') is not True or assignment.get('rejected') is True
                    or identity.get('id')!=identity_id or record.get('voice_id')!=identity_id
                    or (role.get('voice_identity') or {}).get('id')!=identity_id
                    or assignment.get('voice_revision')!=record.get('version')
                    or identity.get('provenance',{}).get('acquisition_sha256')!=acquisition_sha256
                    or event.get('origin')!='human' or event.get('actor')!='brian'
                    or event.get('action')!='use' or event.get('role_id')!=role_id or event.get('key')!=key
                    or event.get('voice_id')!=identity_id or event.get('voice_version')!=record.get('version')
                    or event.get('audio_sha256')!=record.get('audio_sha256')):continue
            stars=event.get('stars')
            positive=event.get('selection_confirmed') is True or (type(stars) is int and stars>=4)
            if positive and record.get('audio_file') and _sha(_safe(root,record['audio_file']).read_bytes())==record.get('audio_sha256'):
                return True
        return False
    except (OSError,ValueError,TypeError,KeyError):
        return False
