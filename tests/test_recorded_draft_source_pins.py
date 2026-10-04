"""Frozen acquisition evidence must remain usable against its exact source policy."""
import pytest

from dubbing_service.app.audiobook import recorded_audition_policy as drafts
from dubbing_service.app.audiobook import reference_source_qc


@pytest.mark.parametrize('row', drafts._records(), ids=lambda row: row['draft_id'])
def test_pinned_real_catalog_evidence_matches_current_source(row):
    # No model, live voice library, source audio or human approval is fabricated.
    # This exercises actual checked-in QC receipts, including their source pins.
    drafts._check_evidence(row)


def test_changed_classifier_source_still_fails_closed(monkeypatch):
    monkeypatch.setattr(reference_source_qc, 'SOURCE_SHA', '0' * 64)
    with pytest.raises(ValueError, match='music/noise/overlap/speaker evidence'):
        drafts._check_evidence(drafts._records()[0])
