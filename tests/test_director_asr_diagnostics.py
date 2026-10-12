from scripts.transcribe_director_comparison import word_errors


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
