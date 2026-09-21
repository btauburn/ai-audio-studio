"""License and engine-identity invariants.

The brief's model-license policy is a code contract, not prose: a permissive
repository license must never be read as permission for weights or for a
cloned voice, and the non-commercial engine must never become the default.
"""

from __future__ import annotations

import pytest

from dubbing_service.app.audiobook import engines
from dubbing_service.app.models.audiobook import LicenseScope

ENGINE_IDS = sorted(engines._LICENSES)


@pytest.mark.parametrize("engine_id", ENGINE_IDS)
def test_every_engine_tracks_the_four_license_scopes_separately(engine_id: str) -> None:
    record = engines.license_record(engine_id)

    assert record is not None
    assert record.engine_id == engine_id
    assert record.display_name
    for field in (
        "platform_license",
        "engine_source_license",
        "model_weight_license",
        "voice_rights",
    ):
        assert getattr(record, field).strip(), f"{engine_id}.{field} is empty"
    assert record.commercial_use in set(LicenseScope.__args__)


@pytest.mark.parametrize("engine_id", ENGINE_IDS)
def test_commercial_use_is_never_inherited_from_the_platform_license(
    engine_id: str,
) -> None:
    record = engines.license_record(engine_id)

    assert record is not None
    assert record.platform_license
    if record.commercial_use == "allowed":
        # "Allowed" is only justified by the weights and the voice rights.
        assert "n/a" not in record.model_weight_license
        assert record.voice_rights.strip()


def test_the_non_commercial_engine_is_never_the_default() -> None:
    default = engines.default_engine_id()

    assert default not in engines.NON_DEFAULT_ENGINES
    assert default == "qwen3tts_customvoice"
    assert engines.license_record(default).commercial_use == "allowed"

    omnivoice = engines.license_record("omnivoice")
    assert omnivoice.commercial_use == "not_allowed"
    assert "commercial" in omnivoice.model_weight_license
    assert omnivoice.is_default is False


def test_exactly_one_engine_is_marked_default() -> None:
    defaults = [record for record in engines.license_records() if record.is_default]

    assert [record.engine_id for record in defaults] == [engines.default_engine_id()]


@pytest.mark.parametrize(
    "backend_id", ["indextts", "cosyvoice3", "qwen3tts", "qwen3tts_cv"]
)
def test_each_backend_maps_to_a_licensed_engine(backend_id: str) -> None:
    engine_id = engines.engine_id_for_backend(backend_id)

    assert engines.license_record(engine_id) is not None
    assert engine_id not in engines.NON_DEFAULT_ENGINES
    assert backend_id in engines.license_record(engine_id).backend_ids


def test_a_backend_without_a_license_record_is_refused() -> None:
    """An unknown backend must not silently render with the default engine."""

    with pytest.raises(engines.UnknownBackendError):
        engines.engine_id_for_backend("no_such_backend")


def test_no_engine_claims_rights_over_a_users_voice() -> None:
    for record in engines.license_records():
        if record.commercial_use == "allowed":
            assert "user must hold rights" not in record.voice_rights


def test_the_dev_fixture_engine_can_never_be_shipped_as_product_audio() -> None:
    fixture = engines.license_record("fixture")

    assert fixture.commercial_use == "not_allowed"
    assert fixture.is_default is False
    assert not fixture.backend_ids or fixture.backend_ids == ["fixture"]
