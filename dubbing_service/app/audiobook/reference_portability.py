"""Exact acquired-reference bundles for versioned cross-project voice reuse.

Bundles append beside immutable Library versions. They never change the version,
identity, reference, cached prompt, ratings or source project. No TTS model/network; existing CPU source-processing recipes may replay during eligibility validation.
"""
from __future__ import annotations

import hashlib
import io
import json
import re
import zipfile
from pathlib import Path
from threading import Lock

import numpy as np

SCHEMA = 'voice-reference-portability-v1'
_LOCK = Lock()


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
    """Validate/preserve one acquired source in a self-contained append-only bundle."""
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
    binding = {'schema':SCHEMA, 'library_voice_id':library_voice_id, 'version':record['version'],
        'voice_id':record['voice_id'], 'identity_id':identity['id'], 'source_project_id':entry['source_project_id'],
        'profile_sha256':shelf._profile_fingerprint(record),
        'sample_audio_sha256':record['audio_sha256'], 'reference_audio_sha256':identity['reference_audio_sha256'],
        'prompt_sha256':identity['prompt_sha256'], 'acquisition_id':acquisition_id,
        'acquisition_sha256':receipt_sha}
    bundle_id = shelf.digest(binding)[:32]
    root = Path(library_root)
    workspace_relative = f'voice_lab/portability/{bundle_id}/workspace'
    workspace = shelf._safe_path(root,workspace_relative)
    manifest_relative = f'voice_lab/portability/{bundle_id}/manifest.json'
    manifest_path = shelf._safe_path(root,manifest_relative)
    receipt_relative = f'voice_lab/discovery/acquisitions/{acquisition_id}/acquisition.json'
    with _LOCK:
        if not manifest_path.exists():
            acquired, original, reference = _eligible_acquisition(Path(source_root),acquisition_id,receipt_sha)
            _identity_matches(acquired,identity)
            prefix = 'voice_lab/discovery/'
            assets = {prefix+acquired['original_path']:(original,acquired['original_sha256']),
                      prefix+acquired['reference_path']:(reference,acquired['reference_audio_sha256']),
                      receipt_relative:(shelf._safe_path(Path(source_root),receipt_relative),receipt_sha)}
            workspace.mkdir(parents=True,exist_ok=True)
            # Receipt commits after its exact original/reference assets.
            for relative,(path,checksum) in assets.items():
                shelf._publish_blob(workspace,relative,shelf._read_verified(path,checksum),checksum)
        acquired,original,reference = _eligible_acquisition(workspace,acquisition_id,receipt_sha)
        _identity_matches(acquired,identity)
        files = {f'voice_lab/discovery/{acquired["original_path"]}':acquired['original_sha256'],
                 f'voice_lab/discovery/{acquired["reference_path"]}':acquired['reference_audio_sha256'],
                 receipt_relative:receipt_sha}
        manifest = {**binding,'bundle_id':bundle_id,'assets':files,'source_project_unchanged':True,
                    'cached_prompt_reused':True,'new_permission_inferred':False,'tts_model_called':False,
                    'source_processing_revalidation': 'existing_cpu_recipe_replay' if acquired.get('source_processing') else 'exact_crop_decode'}
        raw = json.dumps(manifest,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()
        checksum = hashlib.sha256(raw).hexdigest()
        shelf._publish_blob(root,manifest_relative,raw,checksum)
        return {'manifest':manifest,'manifest_sha256':checksum,'workspace':workspace,
                'manifest_relative':manifest_relative,'assets':files}


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
    for relative,checksum in bundle['assets'].items():
        content = shelf._read_verified(shelf._safe_path(bundle['workspace'],relative),checksum)
        shelf._publish_blob(target_root,relative,content,checksum)
    manifest = bundle['manifest']
    _eligible_acquisition(target_root,manifest['acquisition_id'],manifest['acquisition_sha256'])
    if not retain_lineage:
        return None
    relative = f'voice_lab/library/{library_voice_id}/{shelf.digest([library_voice_id,version])[:24]}/reference-portability-{bundle["manifest_sha256"]}.json'
    raw = json.dumps(manifest,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()
    shelf._publish_blob(target_root,relative,raw,bundle['manifest_sha256'])
    return {'schema':SCHEMA,'manifest_file':relative,'manifest_sha256':bundle['manifest_sha256'],
            'bundle_id':manifest['bundle_id'],'source_project_id':manifest['source_project_id'],
            'acquisition_id':manifest['acquisition_id'],'acquisition_sha256':manifest['acquisition_sha256'],
            'cached_prompt_reused':True,'new_permission_inferred':False}
