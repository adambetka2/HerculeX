"""Sheet detection keeps a scan's units to the brick rows they were cut to, so units of two sizes never overlap."""

import json
from concurrent.futures import Future

import pytest

from unwrapping.detection import detect_nodes

ROWS = 40


@pytest.fixture
def scans(tmp_path, monkeypatch):
    """A catalog of 40-brick-row scans whose nodes go under tmp_path, and the planned unit names for a unit size."""
    monkeypatch.setattr(
        detect_nodes, "chosen_scans", lambda samples: [{"sample": s, "volume": "v", "voxel_um": 7.91} for s in samples]
    )
    monkeypatch.setattr(detect_nodes, "sheet_level", lambda voxel_um: 0)
    monkeypatch.setattr(detect_nodes, "level_shape", lambda row, level: [ROWS * 128, 256, 256])
    monkeypatch.setattr(detect_nodes, "central_row", lambda row, level: None)
    monkeypatch.setattr(detect_nodes, "scan_folder", lambda sample, volume, route: tmp_path / sample / volume)

    def planned(sample, unit_rows, band=False):
        monkeypatch.setattr(detect_nodes, "CHUNK_ROWS", unit_rows)
        monkeypatch.setattr(detect_nodes, "BAND_ROWS", unit_rows)
        return [job["out"].name for job in detect_nodes.jobs({sample}, band)]

    return planned


def finished(tmp_path, sample, name, **record):
    unit = detect_nodes.stage_folder(tmp_path / sample / "v", "nodes") / name
    unit.mkdir(parents=True)
    (unit / "record.json").write_text(json.dumps({"method": detect_nodes.METHOD, **record}))


def test_units_from_before_records_named_their_rows_count_as_twelve(scans, tmp_path):
    whole = scans("A", 12)
    assert whole == ["rows-0000-0012", "rows-0011-0023", "rows-0022-0034", "rows-0033-0040"]
    finished(tmp_path, "A", "rows-0000-0012")
    assert scans("A", 12) == whole[1:]
    with pytest.raises(ValueError, match="rows-0000-0012"):
        scans("A", 6)


def test_a_scan_cut_smaller_resumes_at_its_size_and_refuses_the_default(scans, tmp_path):
    whole = scans("B", 6)
    assert whole[:2] == ["rows-0000-0006", "rows-0005-0011"] and whole[-1].endswith("-0040")
    finished(tmp_path, "B", whole[0], unit_rows=6)
    (detect_nodes.stage_folder(tmp_path / "B" / "v", "nodes") / "rows-0000-0012.building").mkdir()
    assert scans("B", 6) == whole[1:]
    with pytest.raises(ValueError, match="HERCULEX_UNIT_ROWS=12"):
        scans("B", 12)


def test_a_band_of_another_size_is_refused(scans, tmp_path):
    (band,) = scans("C", 12, band=True)
    assert band == "band-0014-0026"
    finished(tmp_path, "C", band)
    assert scans("C", 12, band=True) == []
    with pytest.raises(ValueError, match=band):
        scans("C", 6, band=True)


def test_a_unit_records_the_rows_it_was_cut_to(scans, tmp_path, monkeypatch):
    seen = {}

    def stand_in_detection(level, out, record, cell_planes):
        seen.update(record)
        return record | {"nodes": 0, "seconds": 0.0}

    monkeypatch.setattr(detect_nodes, "detect_store", stand_in_detection)
    monkeypatch.setattr(detect_nodes, "work_folder", lambda job: tmp_path / "work")
    scans("D", 6)
    job = next(detect_nodes.jobs({"D"}, False))
    fetched = Future()
    fetched.set_result(tmp_path / "work")
    assert detect_nodes.detect_chunk(job, fetched) == 0
    assert seen["unit_rows"] == 6 and seen["rows"] == [0, 6]
