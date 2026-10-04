"""Exact acquired-reference bundles for versioned cross-project voice reuse.

Bundles append beside immutable Library versions. They never change the version,
identity, reference, cached prompt, ratings or source project. No TTS model/network; existing CPU source-processing recipes may replay during eligibility validation.
"""
from __future__ import annotations

import hashlib
import copy
import io
import json
import re
import zipfile
from pathlib import Path
from threading import Lock

import numpy as np

SCHEMA = 'voice-reference-portability-v1'
_LOCK = Lock()


def _source_draft_grant(entry, source_root):
    """Bind review staging to this exact version's current source selection.

    The source decision is import provenance, never a target approval or a
    narration-delivery claim. Existing worker checks still own generation.
    """
    from . import recorded_audition_policy as draft, voice_lab_library as shelf
    from .fingerprint import voice_revision_for_role
    from dubbing_service.app.models.project import Role
    from dubbing_service.app.storage.projects import project_meta_lock
    record = entry['record']
    identity = record['identity']
    acquisition_sha = identity['provenance']['acquisition_sha256']
    root = Path(source_root)
    if root.name != entry['source_project_id']:
        raise ValueError('Recorded draft source project differs from this Library version')
    with project_meta_lock(entry['source_project_id']):
        source = draft._json(shelf._safe_path(root, 'project.json'))
        if (source.get('id') != entry['source_project_id']
                or not draft.human_casting_context(root, identity['id'], acquisition_sha)):
            raise ValueError('Recorded draft review import requires current exact source human selection')
        state = (source.get('voice_lab') or {}).get('casting') or {}
        saved = (source.get('voice_lab') or {}).get('voices', {}).get(record['key']) or {}
        if (saved.get('key') != record['key'] or saved.get('version') != record['version']
                or saved.get('voice_id') != record['voice_id']
                or saved.get('audio_sha256') != record['audio_sha256']
                or shelf._profile_fingerprint(saved) != shelf._profile_fingerprint(record)):
            raise ValueError('Recorded draft source selection differs from this exact Library version')
        shelf._read_verified(shelf._safe_path(root, saved.get('audio_file', '')), record['audio_sha256'])
        for role_id, item in state.get('roles', {}).items():
            assignment = item.get('assignment') or {}
            event_id = assignment.get('event_id')
            event = state.get('events', {}).get(event_id) or {}
            role = next((r for r in source.get('roles', []) if r.get('id') == role_id), {})
            stars = event.get('stars')
            positive = event.get('selection_confirmed') is True or (type(stars) is int and stars >= 4)
            if (assignment.get('approved') is not True or assignment.get('rejected') is True
                    or assignment.get('key') != record['key']
                    or assignment.get('voice_revision') != record['version']
                    or (role.get('voice_identity') or {}).get('id') != identity['id']
                    or voice_revision_for_role(Role.model_validate(role)) != record['version']
                    or event.get('actor') != 'brian' or event.get('origin') != 'human'
                    or event.get('action') != 'use' or not positive
                    or event.get('role_id') != role_id or event.get('key') != record['key']
                    or event.get('voice_id') != identity['id'] or event.get('voice_version') != record['version']
                    or event.get('audio_sha256') != record['audio_sha256']):
                continue
            return {'schema': 'recorded-draft-source-selection-v1',
                    'source_project_id': entry['source_project_id'], 'source_role_id': role_id,
                    'key': record['key'], 'version': record['version'], 'voice_id': identity['id'],
                    'audio_sha256': record['audio_sha256'], 'acquisition_sha256': acquisition_sha,
                    'event_id': event_id, 'event': {k: event.get(k) for k in
                        ('actor', 'origin', 'action', 'role_id', 'key', 'voice_id', 'voice_version',
                         'audio_sha256', 'selection_confirmed', 'stars')}}
    raise ValueError('Recorded draft source selection is unavailable or no longer current')


def _draft_server_assets(source_root, identity, meta, acquired, original, reference):
    """Preserve the exact server receipt needed by the unchanged draft worker."""
    from . import voice_lab_library as shelf
    prefix = f'casting/voices/{identity["id"]}/'
    relative = prefix + 'reference.json'
    path = shelf._safe_path(Path(source_root), relative)
    if not path.is_file() or path.stat().st_size > 256_000:
        raise ValueError('Recorded draft server reference receipt is missing or exceeds its bound')
    raw = path.read_bytes()
    server = json.loads(raw)
    provenance = server.get('provenance') or {}
    request_hash = hashlib.sha256(json.dumps([acquired['acquisition_id'], meta['acquisition_sha256']]).encode()).hexdigest()
    if (server.get('identity_id') != identity['id'] or server.get('mechanism') != 'qwen_reference'
            or server.get('backend') != 'qwen3tts' or server.get('request_hash') != request_hash
            or server.get('reference_audio') != prefix + 'reference.wav'
            or server.get('reference_audio_sha256') != identity['reference_audio_sha256']
            or server.get('reference_text') != meta['ref_text']
            or server.get('original_audio') != prefix + 'original.audio'
            or server.get('original_audio_sha256') != acquired['original_sha256']
            or provenance.get('origin') != 'eligible_reference_clone'
            or provenance.get('acquisition_id') != acquired['acquisition_id']
            or provenance.get('acquisition_sha256') != meta['acquisition_sha256']
            or provenance.get('audition_draft_policy') != acquired.get('audition_draft_policy')
            or provenance.get('audition_draft_policy') != meta.get('audition_draft_policy')
            or provenance.get('source') != acquired['source']
            or provenance.get('eligibility') != acquired['eligibility']
            or provenance.get('preprocessing') != acquired['preprocessing']
            or provenance.get('technical_qa') != acquired['technical_qa']
            or provenance.get('production_approved') is not False
            or meta.get('x_vector_only_mode') is not False or meta.get('icl_mode') is not True):
        raise ValueError('Recorded draft server receipt differs from the exact cached prompt/acquisition')
    canonical_original = shelf._safe_path(Path(source_root), prefix + 'original.audio')
    canonical_reference = shelf._safe_path(Path(source_root), prefix + 'reference.wav')
    shelf._read_verified(canonical_original, acquired['original_sha256'])
    shelf._read_verified(canonical_reference, acquired['reference_audio_sha256'])
    return {relative: (path, hashlib.sha256(raw).hexdigest()),
            prefix + 'original.audio': (canonical_original, acquired['original_sha256']),
            prefix + 'reference.wav': (canonical_reference, acquired['reference_audio_sha256'])}


def _prompt_metadata(raw):
    # Bound decompression/header claims before parsing an immutable NPZ's metadata.
    try:
        with zipfile.ZipFile(io.BytesIO(raw)) as archive:
            entries = archive.infolist()
            if (len(entries) != 3 or {item.filename for item in entries}
                    != {'ref_code.npy','ref_spk_embedding.npy','metadata.npy'}
                    or sum(item.file_size for item in entries) > 32*1024*1024
                    or any(item.compress_type not in {zipfile.ZIP_STORED,zipfile.ZIP_DEFLATED}
                           or item.flag_bits & 1 for item in entries)):
                raise ValueError('External prompt ZIP is outside its fixed schema/budget')
            entry = archive.getinfo('metadata.npy')
            if entry.file_size > 256_000:
                raise ValueError('External prompt metadata exceeds budget')
            stream = io.BytesIO(archive.read(entry))
        version = np.lib.format.read_magic(stream)
        if version == (1,0):
            shape, fortran, dtype = np.lib.format.read_array_header_1_0(stream)
        elif version == (2,0):
            shape, fortran, dtype = np.lib.format.read_array_header_2_0(stream)
        else:
            raise ValueError('External prompt metadata array version unsupported')
        if len(shape) != 1 or not 1 <= shape[0] <= 250_000 or dtype != np.dtype('uint8') or fortran:
            raise ValueError('External prompt metadata shape/dtype invalid')
        payload = stream.read()
        if len(payload) != shape[0]:
            raise ValueError('External prompt metadata length differs from header')
        result = json.loads(payload.decode('utf-8'))
        if not isinstance(result,dict):
            raise ValueError('External prompt metadata is not an object')
        return result
    except (OSError,KeyError,UnicodeError,zipfile.BadZipFile) as exc:
        raise ValueError('External prompt metadata cannot be verified') from exc


def prepare(entry, library_voice_id, prompt_bytes, library_root, source_root):
    """Preserve one acquired source; selected drafts stage unassigned review only.

    Ordinary references retain their existing strict, self-contained bundles.
    A screened draft additionally depends on its exact current source decision;
    it never gains unrestricted portability or a copied target approval.
    """
    from . import voice_lab_library as shelf
    from service.qwen_character_voice import _eligible_acquisition
    record = entry['record']
    identity = record.get('identity') or {}
    provenance = identity.get('provenance') or {}
    if provenance.get('origin') != 'eligible_reference_clone':
        return None
    meta = _prompt_metadata(prompt_bytes)
    acquisition_id, receipt_sha = provenance.get('acquisition_id'), provenance.get('acquisition_sha256')
    if (meta.get('schema') != 'qwen-original-prompt-v1'
            or not isinstance(acquisition_id,str) or not re.fullmatch(r'[a-f0-9]{32}',acquisition_id)
            or not isinstance(receipt_sha,str) or not re.fullmatch(r'[a-f0-9]{64}',receipt_sha)
            or meta.get('origin') != 'eligible_reference_clone' or meta.get('identity_id') != identity.get('id')
            or meta.get('reference_audio_sha256') != identity.get('reference_audio_sha256')
            or meta.get('acquisition_id') != acquisition_id or meta.get('acquisition_sha256') != receipt_sha
            or meta.get('ref_text') != identity.get('reference_text')):
        raise shelf.VoiceLibraryError('Cached external prompt differs from its acquired reference identity')
    review = meta.get('audition_draft_policy') is not None or provenance.get('audition_draft_policy') is not None
    source_grant = None
    source_server_assets = {}
    if review:
        if (not isinstance(meta.get('audition_draft_policy'), dict)
                or meta['audition_draft_policy'] != provenance.get('audition_draft_policy')):
            raise ValueError('Recorded draft review requires the unchanged prompt/provenance policy')
        # This is the existing pinned policy/QC validator, not a delivery waiver.
        acquired, original, reference = _eligible_acquisition(Path(source_root), acquisition_id, receipt_sha,
                                                             allow_draft=True)
        _identity_matches(acquired, identity)
        source_grant = _source_draft_grant(entry, source_root)
        source_server_assets = _draft_server_assets(source_root, identity, meta, acquired, original, reference)
    binding = {'schema':SCHEMA, 'library_voice_id':library_voice_id, 'version':record['version'],
        'voice_id':record['voice_id'], 'identity_id':identity['id'], 'source_project_id':entry['source_project_id'],
        'profile_sha256':shelf._profile_fingerprint(record),
        'sample_audio_sha256':record['audio_sha256'], 'reference_audio_sha256':identity['reference_audio_sha256'],
        'prompt_sha256':identity['prompt_sha256'], 'acquisition_id':acquisition_id,
        'acquisition_sha256':receipt_sha}
    if review:
        binding['recorded_draft_review'] = {
            'schema': 'recorded-draft-review-staging-v1',
            'source_grant': source_grant, 'source_grant_sha256': shelf.digest(source_grant),
            'target_selection_required': True, 'source_delivery_approval_inferred': False,
            'portability_status': 'guarded_review_material'}
    bundle_id = shelf.digest(binding)[:32]
    root = Path(library_root)
    workspace_relative = f'voice_lab/portability/{bundle_id}/workspace'
    workspace = shelf._safe_path(root,workspace_relative)
    manifest_relative = f'voice_lab/portability/{bundle_id}/manifest.json'
    manifest_path = shelf._safe_path(root,manifest_relative)
    receipt_relative = f'voice_lab/discovery/acquisitions/{acquisition_id}/acquisition.json'
    with _LOCK:
        if not manifest_path.exists():
            acquired, original, reference = _eligible_acquisition(Path(source_root),acquisition_id,receipt_sha,
                                                                 **({'allow_draft': True} if review else {}))
            _identity_matches(acquired,identity)
            prefix = 'voice_lab/discovery/'
            assets = {prefix+acquired['original_path']:(original,acquired['original_sha256']),
                      prefix+acquired['reference_path']:(reference,acquired['reference_audio_sha256']),
                      receipt_relative:(shelf._safe_path(Path(source_root),receipt_relative),receipt_sha)}
            assets.update(source_server_assets)
            workspace.mkdir(parents=True,exist_ok=True)
            # Receipt commits after its exact original/reference assets.
            for relative,(path,checksum) in assets.items():
                shelf._publish_blob(workspace,relative,shelf._read_verified(path,checksum),checksum)
        acquired,original,reference = _eligible_acquisition(workspace,acquisition_id,receipt_sha,
                                                           **({'allow_draft': True} if review else {}))
        _identity_matches(acquired,identity)
        files = {f'voice_lab/discovery/{acquired["original_path"]}':acquired['original_sha256'],
                 f'voice_lab/discovery/{acquired["reference_path"]}':acquired['reference_audio_sha256'],
                 receipt_relative:receipt_sha}
        if review:
            # The worker receipt is checked again in the copied evidence scope.
            replayed_server_assets = _draft_server_assets(workspace, identity, meta, acquired, original, reference)
            files.update({relative: checksum for relative, (_, checksum) in replayed_server_assets.items()})
            if _source_draft_grant(entry, source_root) != source_grant:
                raise ValueError('Recorded draft source selection changed during review staging')
        manifest = {**binding,'bundle_id':bundle_id,'assets':files,'source_project_unchanged':True,
                    'cached_prompt_reused':True,'new_permission_inferred':False,'tts_model_called':False,
                    'source_processing_revalidation': 'existing_cpu_recipe_replay' if acquired.get('source_processing') else 'exact_crop_decode'}
        raw = json.dumps(manifest,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()
        checksum = hashlib.sha256(raw).hexdigest()
        shelf._publish_blob(root,manifest_relative,raw,checksum)
        result = {'manifest':manifest,'manifest_sha256':checksum,'workspace':workspace,
                  'manifest_relative':manifest_relative,'assets':files}
        if review:
            result['_draft_review_source'] = {'root': Path(source_root), 'entry': copy.deepcopy(entry),
                                              'prompt_metadata': copy.deepcopy(meta)}
        return result


def _identity_matches(acquired,identity):
    provenance = identity['provenance']
    if (acquired['reference_audio_sha256'] != identity['reference_audio_sha256']
            or acquired['reference_text'] != identity['reference_text']
            or acquired['source'] != provenance.get('source')
            or acquired['eligibility'].get('intended_use_evidence')
               != (provenance.get('eligibility') or {}).get('intended_use_evidence')):
        raise ValueError('Portability evidence differs from the immutable source/reference/use identity')


def install(bundle,target_root,library_voice_id,version,*,retain_lineage=True):
    """Copy exact checked evidence into the worker's existing project scope."""
    if bundle is None:
        return None
    from . import voice_lab_library as shelf
    from service.qwen_character_voice import _eligible_acquisition
    target_root = Path(target_root)
    manifest = bundle['manifest']
    review = manifest.get('recorded_draft_review')
    if review is not None:
        context = bundle.get('_draft_review_source') or {}
        record = context['entry']['record']
        raw_manifest = json.dumps(manifest, sort_keys=True, separators=(',', ':'), ensure_ascii=False).encode()
        if (manifest.get('library_voice_id') != library_voice_id
                or manifest.get('version') != version or version != record['version']
                or manifest.get('source_project_id') != context['entry']['source_project_id']
                or manifest.get('identity_id') != record['identity']['id']
                or manifest.get('voice_id') != record['voice_id']
                or manifest.get('profile_sha256') != shelf._profile_fingerprint(record)
                or manifest.get('sample_audio_sha256') != record['audio_sha256']
                or hashlib.sha256(raw_manifest).hexdigest() != bundle.get('manifest_sha256')):
            raise ValueError('Recorded draft review bundle differs from the requested exact Library version')
        source_acquired, source_original, source_reference = _eligible_acquisition(context['root'],
            manifest['acquisition_id'], manifest['acquisition_sha256'], allow_draft=True)
        _identity_matches(source_acquired, context['entry']['record']['identity'])
        _draft_server_assets(context['root'], context['entry']['record']['identity'],
                             context['prompt_metadata'], source_acquired, source_original, source_reference)
        source_grant = _source_draft_grant(context['entry'], context['root'])
        if (review.get('schema') != 'recorded-draft-review-staging-v1'
                or review.get('target_selection_required') is not True
                or review.get('source_delivery_approval_inferred') is not False
                or review.get('portability_status') != 'guarded_review_material'
                or review.get('source_grant') != source_grant
                or review.get('source_grant_sha256') != shelf.digest(source_grant)):
            raise ValueError('Recorded draft source selection changed before review installation')
        acquired, original, reference = _eligible_acquisition(bundle['workspace'], manifest['acquisition_id'],
            manifest['acquisition_sha256'], allow_draft=True)
        _identity_matches(acquired, context['entry']['record']['identity'])
        _draft_server_assets(bundle['workspace'], context['entry']['record']['identity'],
                             context['prompt_metadata'], acquired, original, reference)
    for relative,checksum in bundle['assets'].items():
        content = shelf._read_verified(shelf._safe_path(bundle['workspace'],relative),checksum)
        shelf._publish_blob(target_root,relative,content,checksum)
    _eligible_acquisition(target_root,manifest['acquisition_id'],manifest['acquisition_sha256'],
                          **({'allow_draft': True} if review is not None else {}))
    if not retain_lineage:
        return None
    relative = f'voice_lab/library/{library_voice_id}/{shelf.digest([library_voice_id,version])[:24]}/reference-portability-{bundle["manifest_sha256"]}.json'
    raw = json.dumps(manifest,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()
    shelf._publish_blob(target_root,relative,raw,bundle['manifest_sha256'])
    return {'schema':SCHEMA,'manifest_file':relative,'manifest_sha256':bundle['manifest_sha256'],
            'bundle_id':manifest['bundle_id'],'source_project_id':manifest['source_project_id'],
            'acquisition_id':manifest['acquisition_id'],'acquisition_sha256':manifest['acquisition_sha256'],
            'cached_prompt_reused':True,'new_permission_inferred':False}
