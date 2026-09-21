"""Phases 1 and 4: source snapshots, spoken-text revisions and attribution."""

from __future__ import annotations

import pytest

from dubbing_service.app.audiobook import provenance
from dubbing_service.app.models.production import PronunciationEntry
from dubbing_service.app.storage.production import load_settings, save_settings
from tests.audiobook_helpers import build_project, build_segments, reload, use_projects_root


@pytest.fixture()
def studio(monkeypatch, tmp_path):
    root = use_projects_root(monkeypatch, tmp_path)
    project = build_project()
    segments = build_segments(project, ["The quick brown fox.", "Second utterance."])
    return root, project, segments


def test_source_snapshot_is_immutable_and_deduplicated(studio):
    root, project, _segments = studio
    revision, created = provenance.register_source_revision(
        project.id,
        project,
        uri="book.epub",
        data="chapter one\ntext",
        media_type="application/epub+zip",
    )
    assert created is True
    assert revision.sha256
    assert provenance.read_source_snapshot(project.id, revision) == "chapter one\ntext"

    again, created_again = provenance.register_source_revision(
        project.id, project, uri="book.epub", data="chapter one\ntext"
    )
    assert created_again is False
    assert again.id == revision.id

    _, stored = reload(project)
    assert stored  # segments still readable through the same root


def test_tampered_snapshot_is_refused(studio):
    root, project, _segments = studio
    revision, _ = provenance.register_source_revision(
        project.id, project, uri="book.txt", data="original"
    )
    path = root / project.id / revision.snapshot_file
    path.write_text("tampered", encoding="utf-8")
    with pytest.raises(provenance.ProvenanceError):
        provenance.read_source_snapshot(project.id, revision)


def test_spoken_text_revision_never_mutates_canonical_text(studio):
    _root, project, segments = studio
    segment = segments[0]
    canonical_before = segment.text

    revision = provenance.set_spoken_text(
        project,
        segment,
        spoken_text="The kwik brown fox.",
        pronunciation_revision="pron-1",
        reason="phoneme override for pytest",
    )

    assert segment.text == canonical_before
    assert segment.spoken_text_revision_id == revision.id
    assert revision.source_text_hash == provenance.canonical_source_hash(segment)
    assert revision.spoken_text_hash != revision.source_text_hash
    assert provenance.spoken_text_for_segment(project, segment) == "The kwik brown fox."


def test_spoken_text_revisions_are_content_addressed(studio):
    _root, project, segments = studio
    segment = segments[0]
    first = provenance.set_spoken_text(project, segment, spoken_text="Spelling one.")
    repeat = provenance.set_spoken_text(project, segments[1], spoken_text="Spelling one.")
    # same source text + same spoken text + same dictionary revision reuses the entry
    assert repeat.id != first.id
    idempotent = provenance.set_spoken_text(project, segment, spoken_text="Spelling one.")
    assert idempotent.id == first.id

    third = provenance.set_spoken_text(project, segment, spoken_text="Spelling two.")
    assert third.id != first.id
    assert third.previous_revision_id == first.id
    assert provenance.get_spoken_text_revision(project, first.id).spoken_text == "Spelling one."


def test_human_correction_is_authoritative_over_later_analysis(studio):
    _root, project, segments = studio
    segment = segments[0]

    provenance.apply_attribution_proposal(
        segment, speaker_id="narrator", source="nlp", confidence=0.42
    )
    human = provenance.human_correction(segment, speaker_id="maria", note="confirmed")
    assert human.source == "user"
    assert human.confidence == 1.0
    assert human.human_authoritative is True

    kept, applied = provenance.apply_attribution_proposal(
        segment, speaker_id="narrator", source="nlp", confidence=0.99
    )
    assert applied is False
    assert kept.speaker_id == "maria"
    assert kept.source == "user"
    assert kept.confidence == 1.0


def test_automatic_proposals_can_still_be_refined(studio):
    _root, _project, segments = studio
    segment = segments[0]
    provenance.apply_attribution_proposal(
        segment, speaker_id="unknown", source="unknown", confidence=0.1
    )
    updated, applied = provenance.apply_attribution_proposal(
        segment, speaker_id="maria", source="tag", confidence=0.9
    )
    assert applied is True
    assert updated.speaker_id == "maria"
    assert updated.source == "tag"


def test_attribution_survives_save_and_load(studio):
    _root, project, segments = studio
    provenance.human_correction(segments[1], speaker_id="maria")
    from dubbing_service.app.storage.segments import save_segments

    save_segments(project.id, segments)
    _project, stored = reload(project)
    attribution = next(item.attribution for item in stored if item.id == segments[1].id)
    assert attribution is not None
    assert attribution.speaker_id == "maria"
    assert attribution.source == "user"


def test_pronunciation_dictionary_revision_tracks_settings(studio):
    _root, project, _segments = studio
    before = provenance.pronunciation_revision_id(project.id)

    settings = load_settings(project.id)
    settings.pronunciation_dictionary.append(
        PronunciationEntry(id="p1", source="pytest", pronunciation="pie-test")
    )
    save_settings(project.id, settings)

    after = provenance.pronunciation_revision_id(project.id)
    assert after != before
    assert provenance.apply_pronunciations(project.id, "pytest rocks") == "pie-test rocks"


def test_pronunciation_application_is_a_copy_not_a_rewrite(studio):
    _root, project, segments = studio
    settings = load_settings(project.id)
    settings.pronunciation_dictionary.append(
        PronunciationEntry(id="p1", source="OmniVoice", pronunciation="omni voice")
    )
    save_settings(project.id, settings)

    segment = segments[0]
    original = segment.text
    spoken = provenance.apply_pronunciations(project.id, original)
    provenance.set_spoken_text(project, segment, spoken_text=spoken)

    assert segment.text == original
    assert provenance.spoken_text_for_segment(project, segment) == spoken
