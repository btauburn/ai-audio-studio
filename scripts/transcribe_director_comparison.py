"""Optional offline CPU ASR diagnostics; human listening still decides quality."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import re
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))


def words(text):
    return re.findall(r"[\w]+(?:['’][\w]+)?", text.casefold().replace('’', "'"))


def word_errors(expected, heard):
    a, b = words(expected), words(heard)
    table = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]
    for i in range(len(a) + 1): table[i][0] = i
    for j in range(len(b) + 1): table[0][j] = j
    for i in range(1, len(a) + 1):
        for j in range(1, len(b) + 1):
            table[i][j] = min(table[i-1][j-1] + (a[i-1] != b[j-1]), table[i-1][j] + 1, table[i][j-1] + 1)
    i, j, edits = len(a), len(b), []
    while i or j:
        if i and j and a[i-1] == b[j-1] and table[i][j] == table[i-1][j-1]:
            i -= 1; j -= 1
        elif i and j and table[i][j] == table[i-1][j-1] + 1:
            edits.append({'kind': 'substitution', 'expected': a[i-1], 'heard': b[j-1]}); i -= 1; j -= 1
        elif i and table[i][j] == table[i-1][j] + 1:
            edits.append({'kind': 'potential_omission', 'expected': a[i-1]}); i -= 1
        else:
            edits.append({'kind': 'insertion', 'heard': b[j-1]}); j -= 1
    return {'expected_words': len(a), 'edit_distance': table[-1][-1],
            'wer': table[-1][-1] / len(a) if a else None, 'edits': list(reversed(edits)),
            'tail_mismatch': a[-min(3, len(a)):] != b[-min(3, len(a)):] if a else None,
            'interpretation': 'ASR disagreement, not proof of an audible omission or clipped ending.'}


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''): h.update(block)
    return h.hexdigest()


def write_once(path, data):
    content = json.dumps(data, indent=2, sort_keys=True, ensure_ascii=False).encode()
    try:
        with path.open('xb') as f: f.write(content)
    except FileExistsError:
        if path.read_bytes() != content: raise ValueError('Existing ASR evidence differs; choose a new output directory')


def normalize_manifest(manifest):
    """Accept the paired scene contract as well as the original four-arm export."""
    if manifest.get('schema_version') != 'scene-dialogue-comparison-v1':
        return manifest
    from dubbing_service.app.audiobook.scene_dialogue_evaluation import _SafeSources
    sources = _SafeSources(Path(manifest['base_dir']))
    lines = {line['utterance_id']: line for line in manifest['lines']}
    arms = {}
    for arm, spec in manifest['arms'].items():
        samples = []
        for uid, sample in spec['samples'].items():
            if sample['text'] != lines[uid]['text']:
                raise ValueError('ASR sample text differs from the frozen source')
            samples.append({**sample, 'utterance_id': uid,
                            'audio_path': str(sources.resolve(sample['audio_file'], f'{arm}.{uid}.audio_file'))})
        arms[arm] = {**spec, 'samples': samples}
    return {**manifest, 'arms': arms}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('manifest', type=Path)
    p.add_argument('--model-dir', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    args = p.parse_args()
    model_dir = args.model_dir.resolve(strict=True)
    required = ['model.bin', 'config.json', 'tokenizer.json', 'vocabulary.txt']
    hashes = {name: file_sha(model_dir / name) for name in required}
    model_sha = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()
    manifest_bytes = args.manifest.read_bytes()
    manifest = normalize_manifest(json.loads(manifest_bytes))
    output = args.output_dir.resolve()
    if any(Path(s['audio_path']).resolve().is_relative_to(output)
           for arm in manifest['arms'].values() for s in arm['samples']):
        raise ValueError('ASR output may not contain immutable input samples')
    output.mkdir(parents=True, exist_ok=True)
    from faster_whisper import WhisperModel
    model = None
    results = {}
    rows = []
    for arm, spec in manifest['arms'].items():
        for sample in spec['samples']:
            path = Path(sample['audio_path'])
            audio_sha = file_sha(path)
            if audio_sha != sample['audio_sha256']: raise ValueError('Audio hash changed before ASR')
            cached = output / f'{audio_sha}-{model_sha[:16]}.json'
            if cached.exists():
                heard = json.loads(cached.read_bytes())
                if heard['audio_sha256'] != audio_sha or heard['model_sha256'] != model_sha: raise ValueError('ASR cache mismatch')
            else:
                if model is None:
                    model = WhisperModel(str(model_dir), device='cpu', compute_type='int8', cpu_threads=4,
                                         num_workers=1, local_files_only=True)
                segments, info = model.transcribe(str(path), language='en', beam_size=5,
                                                   condition_on_previous_text=False, vad_filter=False, word_timestamps=True)
                segments = list(segments)
                heard = {'audio_sha256': audio_sha, 'model_sha256': model_sha, 'model_files': hashes,
                         'device': 'cpu', 'compute_type': 'int8', 'text': ' '.join(s.text.strip() for s in segments),
                         'words': [{'word': w.word, 'start': w.start, 'end': w.end, 'probability': w.probability}
                                   for s in segments for w in (s.words or [])], 'human_verified': False}
                write_once(cached, heard)
            if file_sha(path) != audio_sha: raise ValueError('Input audio changed during ASR')
            row = {'arm': arm, 'utterance_id': sample['utterance_id'], 'audio_sha256': audio_sha,
                   'text_sha256': hashlib.sha256(sample['text'].encode()).hexdigest(), 'expected_text': sample['text'],
                   'recognized_text': heard['text'], 'asr_model_sha256': model_sha,
                   **word_errors(sample['text'], heard['text'])}
            rows.append(row)
            print(arm, sample['utterance_id'], 'WER', round(row['wer'], 3), flush=True)
    for arm in manifest['arms']:
        arm_rows = [r for r in rows if r['arm'] == arm]
        results[arm] = {'words': sum(r['expected_words'] for r in arm_rows),
                        'edit_distance': sum(r['edit_distance'] for r in arm_rows),
                        'potential_omissions': sum(e['kind'] == 'potential_omission' for r in arm_rows for e in r['edits']),
                        'tail_mismatches': sum(bool(r['tail_mismatch']) for r in arm_rows)}
        results[arm]['wer'] = results[arm]['edit_distance'] / results[arm]['words']
    report = {'schema': 'director-asr-diagnostics-v1', 'manifest_sha256': hashlib.sha256(manifest_bytes).hexdigest(),
              'model_sha256': model_sha, 'device': 'cpu', 'model_downloads': 0, 'rows': rows, 'summary': results,
              'human_review': 'pending', 'quality_improvement': 'not_established'}
    write_once(output / 'asr-diagnostics.json', report)
    if args.manifest.read_bytes() != manifest_bytes: raise ValueError('Manifest changed during ASR')
    print(json.dumps(results, indent=2), flush=True)


if __name__ == '__main__': main()
