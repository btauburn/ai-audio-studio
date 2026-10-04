"""Read-only pinned experiment artifacts, never canonical voice metadata."""
from __future__ import annotations
import hashlib,json,re
from pathlib import Path

ROOT=Path(__file__).resolve().parents[3]
BUNDLE=ROOT/'outputs/2026-10-01-faster-qwen-base'
def load_comparison(project_id: str) -> dict:
    from dubbing_service.app.storage.projects import load_project
    load_project(project_id)
    path=BUNDLE/'comparison.json'
    if not path.is_file():return {'status':'pending','human_review':'pending','pairs':[],'message':'The pinned runtime experiment is still running.'}
    data=json.loads(path.read_text('utf-8'))
    # Experiment is explicitly scoped to the existing private review book.
    if project_id!=data.get('review_project_id'):return {'status':'unavailable','human_review':'pending','pairs':[]}
    for pair in data['pairs']:
        for lane in ('baseline','candidate'):
            item=pair.get(lane)
            if item and item.get('audio_file'):
                item['audio_url']=f'/api/v1/projects/{project_id}/voice-lab/runtime-comparison/audio/{pair["id"]}/{lane}'
    return data
def comparison_audio(project_id: str,pair_id: str,lane: str):
    if not re.fullmatch(r'[a-z0-9-]{1,100}',pair_id) or lane not in ('baseline','candidate'):raise ValueError('Invalid comparison selection')
    data=load_comparison(project_id)
    pair=next((p for p in data.get('pairs',[]) if p['id']==pair_id),None)
    if not pair or not pair.get(lane):raise FileNotFoundError('Comparison audio is unavailable')
    item=pair[lane];relative=Path(item['audio_file'])
    if relative.is_absolute() or '..' in relative.parts:raise ValueError('Invalid comparison artifact path')
    path=BUNDLE/relative
    if path.is_symlink() or not path.resolve().is_relative_to(BUNDLE.resolve()):raise ValueError('Linked comparison audio is refused')
    if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest()!=item['audio_sha256']:raise ValueError('Comparison audio checksum changed')
    return path,item['audio_sha256']
