"""Explicit human performer requests use verified recordings, never name-only fallback."""
from __future__ import annotations
import re


def interpret(raw, previous=None):
    from .performer_discovery import CATALOG
    previous = previous or {}
    requested = previous.get('requested_performer')
    names = sorted({row.get('performer') for row in CATALOG if row.get('performer')}, key=len, reverse=True)
    for name in names:
        if re.search(r'(?<!\w)' + re.escape(name) + r'(?!\w)', raw, re.I):
            requested = name
            break
    if not requested:
        named = re.search(r"(?i:more\s+like|sound(?:s)?\s+like|clone(?:\s+the\s+voice\s+of)?)\s+([A-Z][\w’\-]+(?:\s+[A-Z][\w’\-]+){1,3})", raw)
        if named:
            requested = named.group(1).strip()
    dissatisfied = bool(re.search(r"not (?:great|good|right)|wasn['’]t (?:great|good|right)|isn['’]t (?:great|good|right)|too (?:high|low|nasal)|wrong voice", raw, re.I))
    actionable = bool(raw.strip() and (requested or dissatisfied or re.search(r'\b(?:more|less|should|deeper|warmer|slower|faster|British|male|female)\b', raw, re.I)))
    return {'requested_performer':requested, 'reference_mode':'verified_recording' if requested else None,
            'dissatisfied':dissatisfied, 'actionable':actionable}


def requested_performer(project, role_id, retained="", explicit=None):
    """An explicit owner source target survives later rounds and blank direction."""
    performer = explicit or interpret(retained).get("requested_performer")
    if performer:
        return performer
    targets = project.metadata.get("voice_lab_source_targets") or {}
    target = targets.get(role_id) if isinstance(targets, dict) else None
    if not isinstance(target, dict) or not target.get("requested_by_event"):
        return None
    name = target.get("performer")
    return name.strip() if isinstance(name, str) and 0 < len(name.strip()) <= 100 else None


def proposals(project, role_id, performer):
    """Exact eligible prepared references may audition; a missing source stays explicit."""
    from . import voice_lab as lab
    from .voice_lab_audition import _reuse_record
    from dubbing_service.app.core.config import safe_project_root
    from service.qwen_character_voice import _eligible_acquisition
    rows = (project.metadata.get('voice_lab_named_references') or {}).get(role_id) or []
    ready = []
    for row in rows[:3]:
        if str(row.get('performer') or '').casefold() != performer.casefold():
            continue
        try:
            record = _reuse_record(project, row['key'])
            provenance = record['identity']['provenance']
            if provenance.get('origin') != 'eligible_reference_clone':
                raise ValueError('Requested recording is not an imported reference clone')
            receipt, _, _ = _eligible_acquisition(safe_project_root(project.id), row['acquisition_id'], row['acquisition_sha256'])
            if (provenance.get('acquisition_id') != row['acquisition_id'] or provenance.get('acquisition_sha256') != row['acquisition_sha256']
                    or record['identity']['reference_audio_sha256'] != receipt['reference_audio_sha256']):
                raise ValueError('Prepared clone does not match selected source bytes')
        except (ValueError, KeyError, OSError, lab.VoiceLabError):
            continue
        ready.append({'kind':'reference_clone', 'reuse_key':row['key'], 'rationale':f'Explicit human request for {performer}; exact imported recording reference.',
            'design_instruction':'Use the exact eligible imported reference; do not generate VoiceDesign conditioning.',
            'inspiration':{'performer':performer,'performance':row.get('performance') or 'Verified recorded voice','source_urls':[receipt['source']['source_page']]},
            'reference':{'performer':performer,'source_type':'eligible_reference_clone','characterization':'actual recording','acquisition_id':row['acquisition_id'],'acquisition_sha256':row['acquisition_sha256']},
            'blocked':False, 'tags':[], 'source_urls':[receipt['source']['source_page']]})
    if ready:
        return ready[:2]
    # Prepared source audio is separate from a synthetic audition and never counts as a heard candidate.
    sources = (project.metadata.get('voice_lab_named_reference_sources') or {}).get(role_id) or []
    for selected in sources[:3]:
        if str(selected.get('performer') or '').casefold() != performer.casefold():
            continue
        try:
            from .reference_transcripts import acquisition_audio
            from .voice_lab_audition import _read_verified, _safe_path
            import json
            source_audio, source_sha = acquisition_audio(project.id, selected['acquisition_id'], selected['acquisition_sha256'])
            _read_verified(source_audio, source_sha)
            directory = safe_project_root(project.id) / 'voice_lab/discovery'
            receipt = json.loads(_read_verified(_safe_path(directory, f"acquisitions/{selected['acquisition_id']}/acquisition.json"), selected['acquisition_sha256']))
            if str(receipt['source'].get('performer') or '').casefold() != performer.casefold():
                raise ValueError('Selected performer differs from reference source')
        except (ValueError, KeyError, OSError, TypeError):
            continue
        reason = f'Actual {performer} recording is prepared. Speaker, music, noise and exact transcript review remain pending before a synthetic audition; no unrelated VoiceDesign substitute is queued.'
        return [{'kind':'reference_clone', 'reuse_key':None, 'rationale':f'Explicit request for {performer}; actual recorded source prepared.',
                 'design_instruction':'', 'inspiration':{'performer':performer, 'performance':'Actual recorded reference prepared; listening review pending', 'source_urls':[receipt['source']['source_page']]},
                 'reference':{'performer':performer, 'source_type':'recording_prepared_review_pending', 'characterization':'actual recording; synthetic audition pending',
                              'acquisition_id':selected['acquisition_id'], 'acquisition_sha256':selected['acquisition_sha256']},
                 'blocked':True, 'blocked_reason':reason, 'tags':[], 'source_urls':[receipt['source']['source_page']]}]
    reason = f'The requested {performer} recording needs a verified acquisition, source-use evidence and audio review. Existing recordings remain playable; no unrelated VoiceDesign substitute is queued.'
    return [{'kind':'reference_clone','reuse_key':None,'rationale':f'Explicit human request for {performer}; recording pending.',
        'design_instruction':'','inspiration':{'performer':performer,'performance':'Requested recording; not loaded','source_urls':[]},
        'reference':{'performer':performer,'source_type':'recording_required','characterization':'not_loaded'},
        'blocked':True,'blocked_reason':reason,'tags':[],'source_urls':[]}]
