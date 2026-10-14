"""SJB intake preserves measured-condition semantics and frozen prior inputs."""

import sys
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from extend_endpoint_alignment_round9 import (
    BASE, ROOT, SJB, SOURCE_ID, PROTOCOL_ID, add_sjb, builder_from_previous,
    read_sjb_means, run, validate_mean_rows,
)
from lung.data.endpoint_alignment import aggregate_units, load_alignment


def test_primary_means_and_pixel_ranges_are_read_in_percent():
    rows, evidence = read_sjb_means()
    assert rows.value.tolist() == [3.6, 8.3, 9.5, 20.1]
    assert rows.readout_lower.tolist() == [3.3, 8.0, 9.2, 19.8]
    assert rows.readout_upper.tolist() == [4.0, 8.8, 9.9, 20.6]
    assert any(p.name == "cimb-47-00155_p08_image2_Im6.jpg" for p in evidence)


@pytest.mark.parametrize("field,bad", [
    ("duration_h", 48), ("unit", "fraction"), ("dose_uM", 6.25),
    ("value_origin", "author_printed_flow_plot_percentage"),
    ("uncertainty_type", "SD"), ("denominator", "all_seeded_cells"),
    ("replicate_kind", "independent_experiment"),
])
def test_invalid_source_semantics_are_rejected(field, bad):
    rows, _ = read_sjb_means()
    altered = rows.copy()
    altered.loc[1, field] = bad
    with pytest.raises(ValueError):
        validate_mean_rows(altered)


def test_published_n9_is_not_expanded_to_nine_replicates_or_same_cell_RNA():
    previous = load_alignment(BASE / "curated-20261010-r4", ROOT)
    builder = builder_from_previous(previous)
    rows, evidence = read_sjb_means()
    add_sjb(builder, rows, evidence)
    conditions = pd.DataFrame(builder.conditions.values())
    new = conditions[conditions.source_id.eq(SOURCE_ID)]
    observations = pd.DataFrame(builder.observations)
    new_obs = observations[observations.condition_id.isin(new.condition_id)]
    assert len(new_obs) == 4
    np.testing.assert_allclose(new_obs.value, [.036, .083, .095, .201], atol=1e-15)
    assert new_obs.replicate_kind.eq("published_mean_not_raw_replicate").all()
    assert new_obs.source_file.str.endswith("Fig4A_digitized_apoptosis_means.csv").all()
    summary = aggregate_units(new, new_obs)
    assert summary.available_numeric_records.eq(1).all()
    assert summary.measured_replicates.eq(0).all()
    assert summary.sd_fraction.isna().all()
    assert new.initial_expression_sample_id.isna().all()
    assert not new.same_sample_expression_endpoint.any()
    assert "upstream gates" in builder.protocols[PROTOCOL_ID]["sampling_gate"]
    with pytest.raises(ValueError, match="already present"):
        add_sjb(builder, rows, evidence)


def test_full_additive_build_keeps_previous_records_and_recomputes_all_bits():
    # Production provenance uses project-relative paths. Keep only this test's
    # generated artifacts in an owned project directory; preserve source files.
    owned = SJB / "chemical-review-v1" / "_test-builds"
    owned.mkdir(parents=True, exist_ok=True)
    try:
        with TemporaryDirectory(prefix="sjb-", dir=owned) as directory:
            work = Path(directory).resolve()
            assert work.parent == owned.resolve() and work.is_relative_to(ROOT)
            result = run(work / "r5", work / "v6", work / "chemical")
            assert result["conditions"] == 308
            assert result["observations"] == 846
            assert result["strict_A549_single_24h_Annexin_before"] == 14
            assert result["strict_A549_single_24h_Annexin_after"] == 17
            assert result["all_fingerprints_recomputed_bitwise"]
            assert result["RNA_samples_or_features_generated"] == 0
            with pytest.raises(FileExistsError, match="empty new output"):
                run(work / "r5", work / "v6", work / "chemical")
    finally:
        if owned.exists() and not any(owned.iterdir()):
            owned.rmdir()
