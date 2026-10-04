"""Exact reviewed sources do not grant general host or personality access."""
import pytest

from dubbing_service.app.audiobook import official_recording_sources as official
from dubbing_service.app.audiobook import performer_discovery as discovery


def source():
    return official.records_for("John Lithgow")[0]


def permission(row):
    return {"source_id": row["source_id"], "status": "approved", "use": "private_synthetic_voice",
            "authority": "owner_private_use_request", "evidence_ref": "explicit-private-test-request"}


def test_exact_official_recording_download_does_not_confer_synthetic_approval():
    row = source()
    result = discovery.reference_eligibility(row)
    assert result["download_eligible"]
    assert not result["intended_use_eligible"] and not result["clone_ready"]
    approved = discovery.reference_eligibility(row, intended_use_evidence=permission(row))
    assert approved["acquisition_eligible"] and not approved["clone_ready"]
    assert approved["intended_use_evidence"]["authority"] != "performer_consent"


@pytest.mark.parametrize("field", ["source_id", "performer", "source_page", "media_url", "license_url",
                                    "reviewed_original_sha256", "size_bytes", "duration_seconds"])
def test_modified_record_or_other_government_asset_stays_held(field):
    row = source()
    row[field] = 42 if isinstance(row[field], (int, float)) else str(row[field]) + "modified"
    assert official.reviewed(row) is None
    assert not discovery.reference_eligibility(row, intended_use_evidence=permission(row))["acquisition_eligible"]


def test_url_requires_exact_asset_and_copies_cannot_mutate_registry():
    row = source()
    assert official.reviewed_url(row["media_url"])
    assert official.reviewed_url(row["media_url"] + "?download=1") is None
    row["performer"] = "Someone else"
    assert source()["performer"] == "John Lithgow"
    assert official.records_for("Other Person") == []


def test_cached_or_downloaded_bytes_must_match_observed_original():
    with pytest.raises(ValueError, match="immutable original"):
        official.verify_original(source(), b"replacement audio")
    with pytest.raises(ValueError, match="reviewed exact source"):
        official.verify_original(source() | {"performer": "Someone else"}, b"replacement")


def test_source_bound_private_request_not_accepted_for_unreviewed_commons():
    row = {"source_id": "commons-test", "media_url": "https://upload.wikimedia.org/wikipedia/commons/a/ab/Test.wav",
           "source_page": "https://commons.wikimedia.org/wiki/File:Test.wav", "terms_url": discovery.TERMS_URL,
           "license_url": "https://creativecommons.org/licenses/by/3.0/", "attribution": "Test",
           "language": "en", "duration_seconds": 5}
    assert not discovery.reference_eligibility(row, intended_use_evidence=permission(row))["intended_use_eligible"]


def test_official_lookup_needs_no_external_model_or_arbitrary_fetch():
    class NoNetwork:
        limits = discovery.DiscoveryLimits()
        def get(self, *args, **kwargs):
            pytest.fail("Reviewed record should not make a network metadata request")
    records = discovery._performer_sources(NoNetwork(), {"performer": "John Lithgow"})
    assert records[0]["source_id"] == source()["source_id"]
    assert records[0]["download_eligible"] and not records[0]["intended_use_eligible"]


@pytest.mark.parametrize("field", ["rights_scope", "recording_license_evidence", "performance", "performance_linkage"])
def test_altered_identity_or_rights_claim_cannot_retain_reviewed_status(field):
    row = source() | {field: "invented performer consent or Dumbledore performance"}
    assert official.reviewed(row) is None
    assert not discovery.reference_eligibility(row, intended_use_evidence=permission(row))["acquisition_eligible"]


def test_worker_rejects_self_consistent_replacement_original_before_decode(tmp_path):
    import hashlib
    import json
    from service.qwen_character_voice import _eligible_acquisition
    acquisition_id = "a" * 32
    directory = tmp_path / "voice_lab/discovery"
    original = directory / "downloads" / ("b" * 64 + ".audio")
    reference = directory / "acquisitions" / acquisition_id / "reference.wav"
    original.parent.mkdir(parents=True)
    reference.parent.mkdir(parents=True)
    original.write_bytes(b"self-consistent replacement recording")
    reference.write_bytes(b"self-consistent replacement reference")
    row = source()
    receipt = {"schema": "voice-lab-acquisition-v1", "acquisition_id": acquisition_id,
               "source": row, "eligibility": {"intended_use_evidence": permission(row)},
               "clone_ready": True, "technical_qa": {"status": "pass"},
               "original_path": "downloads/" + original.name,
               "reference_path": "acquisitions/" + acquisition_id + "/reference.wav",
               "original_sha256": hashlib.sha256(original.read_bytes()).hexdigest(),
               "reference_audio_sha256": hashlib.sha256(reference.read_bytes()).hexdigest()}
    receipt_path = reference.with_name("acquisition.json")
    receipt_path.write_text(json.dumps(receipt), "utf-8")
    with pytest.raises(ValueError, match="immutable original"):
        _eligible_acquisition(tmp_path, acquisition_id, hashlib.sha256(receipt_path.read_bytes()).hexdigest())
