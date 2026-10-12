"""Three actual production boundaries refuse experimental Takes; review stays labeled."""
import copy
from types import SimpleNamespace

import pytest
from dubbing_service.app.audiobook import casting_assembly, autocast_book, book_creator, runtime_binding as binding
from dubbing_service.app.models.project import SegmentTake
from dubbing_service.app.storage.projects import save_project
from dubbing_service.app.storage.segments import save_segments, load_segments
from tests.audiobook_helpers import use_projects_root, build_project, build_segments
from tests.test_voice_lab_chapter_export import chapter


def experimental_metadata(profile=binding.DIRECTION_RUNTIME_PROFILE):
    return {'runtime_binding': {'contract': {'worker': {'contract': {'runtime_profile': profile}}}}}


@pytest.fixture
def selected(tmp_path, monkeypatch):
    use_projects_root(monkeypatch, tmp_path)
    project = build_project(name='Synthetic experimental exclusion')
    segments = build_segments(project, ['Synthetic exclusion line.'])
    take = SegmentTake(id='exp-take', audio_file='audio/synthetic.wav', generation_metadata=experimental_metadata())
    segments[0].takes = [take]
    segments[0].selected_take_id = take.id
    save_segments(project.id, segments)
    return project, segments


def test_frozen_assembly_refuses_before_legacy_early_return(selected):
    project, segments = selected
    with pytest.raises(casting_assembly.CastingAssemblyError, match='review-only'):
        casting_assembly.current_frozen_binding(project, segments)


def test_book_creator_refuses_before_coverage_or_audio_lookup(selected):
    with pytest.raises(book_creator.BookCreatorError, match='review-only'):
        book_creator._selected_audio(*selected)


def test_autocast_export_refuses_before_coverage_or_audio_lookup(selected):
    project, segments = selected
    with pytest.raises(autocast_book.BookError, match='review-only'):
        autocast_book.export_chapter(project.id, 'synthetic-request', 'chap0')


@pytest.mark.parametrize('metadata', [experimental_metadata('unknown-profile'),
    {'provider_calls': [{'worker_runtime_identity': {'contract': {'runtime_profile': binding.DIRECTION_RUNTIME_PROFILE}}}]},
    {'provider_calls': [{'generation': {'direction': {'policy': 'native-instruct-icl-x1'}}}]}])
def test_unknown_or_nested_experimental_provenance_never_becomes_production(metadata):
    take = SimpleNamespace(generation_metadata=metadata)
    with pytest.raises(binding.RuntimeBindingError):
        binding.require_production_take(take)


@pytest.mark.parametrize('profile', [None, 'native-base', 'faster-base-032'])
def test_legacy_or_known_production_takes_keep_the_old_path(profile):
    binding.require_production_take(SimpleNamespace(generation_metadata=experimental_metadata(profile) if profile else {}))


def test_review_chapter_is_available_and_labels_experimental_runtime(chapter):
    from dubbing_service.app.audiobook import voice_lab_chapter_export as review
    rows = load_segments(chapter[0])
    rows[0].takes[0].generation_metadata.update(experimental_metadata())
    save_segments(chapter[0], rows)
    result = review.export_chapter(chapter[0], 'chap0')
    assert result['experimental_runtime'] is True
    assert result['human_review'] == 'pending'


def test_review_opt_in_keeps_the_strict_cast_guard(selected):
    project, segments = selected
    project.voice_lab.casting['enabled'] = True
    with pytest.raises(casting_assembly.CastingAssemblyError, match='current frozen cast'):
        casting_assembly.current_frozen_binding(project, segments, allow_experimental=True)
