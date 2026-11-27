"""Studio is a model kind of its own: any .safetensors checkpoint of its network is checked, listed and read."""

import json
from pathlib import Path

import numpy as np
import pytest
import torch
from safetensors.torch import save_file

from ink.model import models
from ink.model import reader as lanes
from ink.model import studio_reader as studio
from ink.reading import piece_reading


class Tiny(torch.nn.Module):
    """A stand-in for Studio's network with the same constructor."""

    def __init__(self, deeper=False, width=3):
        super().__init__()
        self.conv = torch.nn.Conv3d(1, width, 3)


def stand_in(monkeypatch, stride, value=0.0):
    reader = object.__new__(studio.StudioReader)
    reader.torch, reader.device, reader.half, reader.kernel = torch, torch.device("cpu"), False, studio.hann()
    reader.model = lambda image: torch.full((image.shape[0], 1, studio.WINDOW, studio.WINDOW), value)
    monkeypatch.setattr(studio, "STRIDE", stride)
    return reader


@pytest.mark.parametrize("stride", [96, 64])
def test_every_covered_pixel_is_read_once_over_and_an_empty_one_stays_zero(monkeypatch, stride):
    stack = np.full((16, 200, 250), 90, dtype=np.uint8)
    stack[:, :, :20] = 0
    read = stand_in(monkeypatch, stride).read(stack)
    assert read.shape == (200, 250) and np.allclose(read[:, 20:], 0.5, atol=1e-5) and not read[:, :20].any()


def test_a_window_past_the_edge_is_reflected_not_zero_filled():
    stack = np.arange(16 * 10 * 10, dtype=np.uint8).reshape(16, 10, 10)
    window = studio.window_at(stack, -3, -3)
    assert window.shape == (16, 128, 128) and np.array_equal(window[:, 3:13, 3:13], stack)
    assert np.array_equal(window[:, 2, 3:13], stack[:, 1])


def test_a_device_failure_stops_the_read_but_a_bad_stack_is_returned(monkeypatch):
    reader = stand_in(monkeypatch, 96)
    good = np.full((16, 40, 40), 90, dtype=np.uint8)
    maps = list(reader.maps([(good, None), (good[:5], None)]))
    assert maps[0].shape == (40, 40) and isinstance(maps[1], ValueError)
    reader.read = lambda stack: (_ for _ in ()).throw(RuntimeError("CUDA error: unknown error"))
    with pytest.raises(RuntimeError, match="CUDA"):
        list(reader.maps([(good, None)]))


def test_a_checkpoint_is_checked_against_the_networks_tensors(monkeypatch, tmp_path):
    monkeypatch.setattr(studio, "network", lambda: Tiny)
    good, wide = tmp_path / "good.safetensors", tmp_path / "wide.safetensors"
    save_file(Tiny().state_dict(), str(good))
    save_file(Tiny(width=5).state_dict(), str(wide))
    studio.check(good)
    with pytest.raises(SystemExit, match="2 of the wrong shape"):
        studio.check(wide)


def test_a_dropped_in_safetensors_is_listed_as_a_studio_model_and_a_pth_stays_reader_v3(monkeypatch, tmp_path):
    monkeypatch.setattr(models, "LISTED", tmp_path / "models.json")
    monkeypatch.setattr(models, "INK_WEIGHTS", tmp_path)
    monkeypatch.setattr(studio, "check", lambda path: None)
    mine = tmp_path / "mine.safetensors"
    mine.write_bytes(b"weights")
    models.add("mine", "my Studio fine-tune", mine)
    assert json.loads((tmp_path / "models.json").read_text())["mine"]["file"] == "mine/studio.safetensors"
    found = models.models()["mine"]
    assert (found.kind, found.page_suffix, found.weights.read_bytes()) == ("studio", "-mine", b"weights")
    assert models.models()["reader-v3"].kind == "reader-v3"
    assert lanes.sources_of(found.weights) == (Path(studio.__file__),)
    assert lanes.sources_of(tmp_path / "tile-reader.pth") == lanes.SOURCES


def test_coverage_counts_the_ink_maps_of_the_model_that_reads_not_only_stocks(monkeypatch, tmp_path):
    from website import survey_status

    piece = {"unit": "rows-0", "piece": 1, "area": 100.0, "shape": [4, 4], "stride": 16, "group": 0}
    sheet = tmp_path / "S" / "v" / "sheet"
    sheet.mkdir(parents=True)
    (sheet / "layout.json").write_text(json.dumps({"pieces": [piece], "record": {"joints": 0, "moved": 0}}))
    own = tmp_path / "S" / "v" / "faces" / "level-0-mine" / "rows-0"
    own.mkdir(parents=True)
    for name in ("piece-0001-layer0.png", "piece-0001-taught.png"):
        (own / name).write_bytes(b"png")
    monkeypatch.setattr(survey_status, "PRODUCTION_FACES", "level-0-mine")
    assert survey_status.snapshot("S", "v", use_review=False, root=tmp_path)["ink_pieces"] == 1
    monkeypatch.setattr(survey_status, "PRODUCTION_FACES", "level-0")
    assert survey_status.snapshot("S", "v", use_review=False, root=tmp_path)["ink_pieces"] == 0


def test_studio_reads_sixteen_planes_of_the_stack_and_one_stack_per_depth_shift(monkeypatch):
    monkeypatch.setattr(piece_reading, "REACH", 2)
    monkeypatch.setattr(piece_reading, "SHIFTS", (-2, 0, 2))
    cube = np.arange(35, dtype=np.uint8)[:, None, None] * np.ones((35, 2, 2), dtype=np.uint8)
    kept = piece_reading.studio_planes(cube, 2)
    assert kept[:, 0, 0].tolist() == list(range(8, 28))
    reads = piece_reading.studio_reads(kept)
    assert [r[0][:, 0, 0].tolist()[0] for r in reads] == [10, 8, 12]
    assert all(r[0].shape == (16, 2, 2) and r[1] is None for r in reads)
    assert piece_reading.studio_planes(cube[2:33])[:, 0, 0].tolist() == list(range(10, 26))
