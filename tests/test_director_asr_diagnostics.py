import pytest
from dubbing_service.app.audiobook.director_evaluation import EvaluationError
from scripts.transcribe_director_comparison import word_errors, normalize_manifest


def test_scene_manifest_retains_exact_text_and_audio_binding(tmp_path):
    (tmp_path/'sample.wav').write_bytes(b'fixture')
    manifest={'schema_version':'scene-dialogue-comparison-v1','base_dir':str(tmp_path),
              'lines':[{'utterance_id':'u1','text':'Wait here.'}],
              'arms':{'baseline':{'samples':{'u1':{'text':'Wait here.','audio_file':'sample.wav','audio_sha256':'fixture'}}}}}
    result=normalize_manifest(manifest)
    assert result['arms']['baseline']['samples'][0]['utterance_id']=='u1'
    assert result['arms']['baseline']['samples'][0]['audio_path']==str(tmp_path/'sample.wav')
    manifest['arms']['baseline']['samples']['u1']['audio_file']='../outside.wav'
    with pytest.raises(EvaluationError, match='path_refused'):normalize_manifest(manifest)


def test_scene_manifest_rejects_changed_source_text(tmp_path):
    manifest={'schema_version':'scene-dialogue-comparison-v1','base_dir':str(tmp_path),
              'lines':[{'utterance_id':'u1','text':'Wait here.'}],
              'arms':{'baseline':{'samples':{'u1':{'text':'Stay here.','audio_file':'sample.wav'}}}}}
    with pytest.raises(ValueError, match='frozen source'):normalize_manifest(manifest)


def test_omission_and_final_word_disagreement_are_diagnostics():
    result = word_errors('Listen to that clock.', 'Listen to that.')
    assert result['wer'] == 0.25
    assert result['edits'] == [{'kind': 'potential_omission', 'expected': 'clock'}]
    assert result['tail_mismatch'] is True
    assert 'not proof' in result['interpretation']


def test_punctuation_and_case_do_not_manufacture_omissions():
    result = word_errors('Don’t be AFRAID!', "don't be afraid")
    assert result['wer'] == 0
    assert result['edits'] == []
    assert result['tail_mismatch'] is False


def test_substitution_and_insertion_counts():
    result = word_errors('We should wait.', 'We could wait here.')
    assert result['edit_distance'] == 2
    assert {e['kind'] for e in result['edits']} == {'substitution', 'insertion'}


def test_empty_reference_does_not_claim_intelligibility():
    assert word_errors('', 'hello')['wer'] is None
