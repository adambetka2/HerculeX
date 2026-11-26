"""End to end on synthetic pieces."""

import io
import itertools
import json
import os
import subprocess
import sys
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer

import numpy as np
import pytest
from PIL import Image

from ink.model.stock_model import folded
from runner import pipeline, sheet_stage
from sampling.column_sampling import bridged_mesh, smoothed_mesh
from sampling.piece_images import ink_window
from sampling.sheet_images import across_sheet, extended
from storage.brick_store import BrickStore
from unwrapping.layout.mosaic_layout import agreed_turns, flips, mosaics
from unwrapping.layout.roll_placement import outward_grid
from website import site_server
from website.sheet_tiles import (
    SCAN_UNDER_TAUGHT,
    paint_strip,
    parted,
    piece_layer,
    seen_from_written_face,
)
from website.site_pages import shelves_of
from website.window_proposals import cells_of, looks_of, reads_as_written, windows


def test_a_mosaic_stops_where_a_joint_would_land_it_on_itself():
    flat = {}
    keys = [(0, n) for n in range(4)]
    for k in keys:
        flat[k] = {
            "mask": np.ones((20, 20), dtype=bool),
            "area": 400.0 - k[1],
            "shape": [100, 100],
            "side": 0.2,
            "flip_rows": False,
            "flip_cols": False,
        }

    def joint(a, b, ca, cb):
        return {"a": keys[a], "b": keys[b], "edges": 20, "columns": [ca, cb], "rows": [50.0, 50.0]}

    joints = [joint(0, 1, 99.0, 0.0), joint(1, 2, 99.0, 0.0), joint(2, 3, 0.0, 99.0)]
    _groups, group, place = mosaics(keys, flat, joints)
    assert place[keys[1]][0] == 99 * 0.2 and place[keys[2]][0] == 2 * 99 * 0.2
    assert group[keys[0]] == group[keys[1]] == group[keys[2]]
    assert group[keys[3]] != group[keys[0]]


def piece(tmp_path, pixels, n=0, x_mm=0.0):
    level = tmp_path / "level-0" / "unit"
    level.mkdir(parents=True, exist_ok=True)
    Image.fromarray(pixels).save(level / f"piece-{n:04d}.png")
    h, w = pixels.shape
    return {
        "unit": "unit",
        "piece": n,
        "area": 30.0 - n,
        "um": 0.01,
        "stride": 1,
        "x_mm": x_mm + 0.02,
        "y_mm": 0.02,
        "width_mm": w * 0.01,
        "height_mm": h * 0.01,
        "flip_rows": False,
        "flip_cols": False,
        "group": 1,
    }


def painted(tmp_path, pieces, layer="sheet"):
    tiles = {(c, r): tile for c, r, tile in paint_strip((0, pieces, tmp_path / "level-0", layer, 0.01, 1, 2))}
    return np.asarray(Image.open(io.BytesIO(tiles[0, 0]))).astype(np.float32)


def test_smoothing_takes_the_cell_jitter_off_a_mesh_and_leaves_its_holes():
    rows, columns = np.mgrid[0:40, 0:40].astype(np.float32) * 16
    flat = np.stack([columns + 100, rows + 100, np.full_like(rows, 200.0)])
    shaken = flat + np.random.default_rng(1).normal(0, 3, flat.shape).astype(np.float32)
    shaken[:, 18:22, 18:22] = -1
    smooth = smoothed_mesh(shaken)
    inner = np.zeros(rows.shape, dtype=bool)
    inner[4:-4, 4:-4] = True
    inner[14:26, 14:26] = False
    assert (smooth[0] < 0).sum() == 16 and np.abs(smooth - flat)[:, inner].std() < 0.5 * np.abs(shaken - flat)[
        :, inner
    ].std()


def test_a_gap_is_bridged_along_the_sheet_but_never_across_a_fold():
    rows, columns = np.mgrid[0:40, 0:60].astype(np.float32) * 16
    flat = np.stack([columns + 100, rows + 100, np.full_like(rows, 200.0)])
    flat[:, 15:25, 25:35] = -1
    mesh = bridged_mesh(flat, 16)
    assert (
        (mesh[0] > 0).all()
        and abs(float(mesh[2, 20, 30]) - 200.0) < 1e-3
        and abs(float(mesh[0, 20, 30]) - (100 + 30 * 16)) < 8
    )
    folded = flat.copy()
    folded[2, :, 35:] += 400.0
    assert (bridged_mesh(folded, 16)[0, 15:25, 25:35] <= 0).any()


def test_the_scan_under_taught_ink_is_the_sheet_laid_as_that_map_lies(tmp_path):
    p = piece(tmp_path, np.tile(np.arange(100, dtype=np.uint8), (100, 1))) | {"flip_cols": True}
    image, seen, _ = piece_layer(tmp_path / "level-0", p, SCAN_UNDER_TAUGHT)
    assert image[0, 0] == 99 and image[0, -1] == 0 and not seen[:, -1].any() and seen[:, :-1].all()


def test_a_neighbours_ink_never_shows_through_blank_papyrus(tmp_path):
    first, second = (piece(tmp_path, np.full((100, 100), 120, dtype=np.uint8), n) for n in (0, 1))
    level = tmp_path / "level-0" / "unit"
    Image.fromarray(np.zeros((100, 100), dtype=np.uint8)).save(level / "piece-0000-taught.png")
    Image.fromarray(np.full((100, 100), 255, dtype=np.uint8)).save(level / "piece-0001-taught.png")
    assert painted(tmp_path, [first, second], "taught")[10:90, 10:90].max() < 20


def test_air_inside_a_sheet_cannot_print_dark(tmp_path):
    stack = np.full((31, 128, 128), 120, dtype=np.uint8)
    stack[14:17, 40:60, 40:60] = 60
    store = BrickStore(tmp_path / "stack", stack.shape, stack.shape)
    store.put((0, 0, 0), stack.tobytes())
    store.save()
    sheet, layers = across_sheet(store, 9.4)
    assert layers[0][50, 50] == 60 and sheet[50, 50] == 120 and sorted(layers) == [-3, -2, -1, 0, 1, 2, 3]


def test_every_voxel_reaches_the_sheet_as_it_is(tmp_path):
    rng = np.random.default_rng(5)
    pixels = rng.integers(90, 150, (300, 300)).astype(np.uint8)
    pixels[150, 50:250] = 230
    tile = painted(tmp_path, [piece(tmp_path, pixels)])
    assert tile[150, 60:240].mean() > tile[147, 60:240].mean() + 80
    assert np.corrcoef(tile[:300, :300].ravel(), pixels.ravel())[0, 1] > 0.98


def test_where_there_is_no_scan_stays_black(tmp_path):
    pixels = np.full((300, 300), 140, dtype=np.uint8)
    pixels[100:140, 100:140] = 0
    tile = painted(tmp_path, [piece(tmp_path, pixels)])
    assert tile[110:130, 110:130].max() < 4 and tile[200:220, 200:220].min() > 130 and tile[310:, 310:].max() < 4


def test_a_mesh_runs_on_past_its_edge_and_into_small_holes():
    y, x = np.mgrid[0:20, 0:30].astype(np.float32)
    xyz = np.stack([x + 100, y + 100, x * 0 + 100])
    xyz[:, 8:10, 8:10] = -1.0
    grid = extended(xyz)
    assert grid.shape == (3, 24, 34) and (grid[0, 2:22, 2:32] > 0).all()
    assert abs(grid[0, 2, 0] - 98) < 1e-3 and abs(grid[1, 10, 10] - 108) < 1e-3


def test_a_piece_own_voxels_win_over_a_neighbours_margin(tmp_path):
    first, second = np.full((100, 100), 100, dtype=np.uint8), np.full((100, 100), 200, dtype=np.uint8)
    tile = painted(tmp_path, [piece(tmp_path, first), piece(tmp_path, second, 1, 0.96)])
    assert tile[50, 95] < 110 and tile[50, 98] > 190


def test_the_model_reads_the_written_face():
    stack = np.full((31, 16, 16), 110, dtype=np.uint8)
    stack[12:18] = 140
    window = ink_window(stack, 17)
    assert window.shape[0] == 17 and window[10].mean() == 140 and window[11].mean() == 110


def test_the_folded_convolution_is_the_same_sum():
    import torch

    conv = torch.nn.Conv3d(4, 6, 3, padding=1, bias=False)
    x = torch.randn(2, 4, 7, 16, 16)
    assert torch.allclose(folded(conv)(x), conv(x), atol=1e-5)


def test_proposals_rank_a_window_by_its_strong_ink_away_from_rims():
    strong, sheet = np.zeros((60, 60)), np.ones((60, 60), dtype=bool)
    strong[30:50, 30:50] = 1.0
    best = max(windows(strong, sheet, 20, 5), key=lambda row: row[2])
    assert best[:2] == (30, 30) and best[2] > 0.9
    assert cells_of(np.zeros((60, 60), dtype=np.uint8), np.full((60, 60), 90, dtype=np.uint8), 6)[1].all()


def test_only_pieces_sure_on_their_written_face_are_proposed(tmp_path):
    (tmp_path / "unit").mkdir()
    for n, near, far, written in (
        (0, 0.15, 0.03, "near"),
        (1, 0.02, 0.14, "near"),
        (2, 0.15, 0.12, "near"),
        (3, 0.04, 0.0, "near"),
        (4, 0.02, 0.14, "far"),
    ):
        (tmp_path / "unit" / f"piece-{n:04d}-taught-side.json").write_text(
            json.dumps({"near": near, "far": far, "written": written})
        )
    (tmp_path / "unit" / "piece-0005-taught-side.json").write_text(
        json.dumps({"near": 0.15, "far": 0.03, "written": "near", "scrambled": 0.13})
    )
    assert [reads_as_written(looks_of(tmp_path, {"unit": "unit", "piece": n})) for n in range(6)] == [
        True,
        False,
        False,
        False,
        True,
        False,
    ]


def test_pieces_never_joined_keep_a_line_of_black_between_them():
    covered = np.ones((100, 100), dtype=bool)
    owner = np.ones(covered.shape, dtype=np.int32)
    owner[52:] = 2
    parted(covered, owner, 5)
    assert not covered[50:54].any() and covered[:40].all() and covered[60:].all()


def test_a_long_sheet_stacks_into_shelves_cut_between_pieces(tmp_path, monkeypatch):
    pieces = [
        {"x_mm": 100.0 * n, "y_mm": 0.0, "width_mm": 90.0, "height_mm": 180.0, "area": 16200.0} for n in range(95)
    ]
    monkeypatch.setattr("website.site_pages.layout_of", lambda sample, volume: pieces)
    shelves = shelves_of("sample", "volume")
    assert len(shelves) == 5 and shelves[0]["x"] == 0 and shelves[-1]["x"] + shelves[-1]["w"] == 9490
    assert all(a["x"] + a["w"] == b["x"] and a["y"] + a["h"] < b["y"] for a, b in itertools.pairwise(shelves))
    assert all(s["x"] % 100 >= 90 for s in shelves[1:])


class Finished:
    def wait(self, *_):
        return 0


def test_full_pipeline_stops_the_other_stages_when_one_fails(tmp_path, monkeypatch):
    launch = subprocess.Popen
    children = {}

    def start(command, cwd):
        stage = command[4]
        code = "raise SystemExit(7)" if stage == "pieces" else "import time; time.sleep(2)"
        children[stage] = launch([sys.executable, "-c", code], cwd=cwd)
        return children[stage]

    monkeypatch.setattr(pipeline, "ROUTE_OUTPUTS", tmp_path)
    monkeypatch.setattr(pipeline.subprocess, "Popen", start)
    assert pipeline.run_all(["PHerc0000"], band=True) == 7
    assert all(children[stage].returncode < 0 for stage in ("sheets", "sheet", "fetch"))


def test_every_scan_is_laid_out_read_and_published_and_a_second_pass_only_republishes(tmp_path, monkeypatch):
    from runner import review_page

    base = tmp_path / "survey" / "PHerc0000" / "1"
    faces = base / "faces" / "level-0" / "rows-0"
    for folder in (
        base / "pieces" / "rows-0",
        base / "pieces" / "rows-1",
        base / "pieces" / "joints",
        base / "pieces" / "frame",
        base / "nodes" / "rows-0",
        base / "nodes" / "rows-1",
        faces,
    ):
        folder.mkdir(parents=True)
    for unit in ("rows-0", "rows-1", "joints", "frame"):  # the frame's record last, newer than its units'
        (base / "pieces" / unit / "record.json").write_text(json.dumps({"voxel_um": 9.0}))
    pieces = [
        {"unit": "rows-0", "piece": n, "area": 10.0 - n, "group": n, "moved_from_joint_mm": 0.0} for n in range(2)
    ]
    calls = []

    def run_step(module, *arguments, **settings):
        calls.append(module.rsplit(".", 1)[1])
        if module.endswith("scroll_layout"):
            (base / "sheet").mkdir(exist_ok=True)
            (base / "sheet" / "layout.json").write_text(json.dumps({"pieces": pieces}))
        return Finished()

    monkeypatch.setattr(sheet_stage, "ROUTE_OUTPUTS", tmp_path / "survey")
    monkeypatch.setattr(sheet_stage, "started", run_step)
    monkeypatch.setattr(review_page, "publish", lambda sample, volume, url: calls.append(f"publish {sample} {volume}"))
    sheet_stage.run()
    assert calls == ["scroll_layout", "piece_reading", "publish PHerc0000 1"]
    calls.clear()
    sheet_stage.run()
    assert calls == ["piece_reading", "publish PHerc0000 1"]
    calls.clear()
    sheet_stage.run(["PHerc0000"], volume="1", mode="ink", require_ready=True)
    assert calls == ["piece_reading"]
    with pytest.raises(RuntimeError, match="no ready pieces"):
        sheet_stage.run(["PHerc0000"], volume="2", mode="ink", require_ready=True)


def test_level_publication_waits_for_and_uses_the_exact_frame(tmp_path, monkeypatch):
    sample, volume = "PHerc0000", "1"
    base = tmp_path / sample / volume
    monkeypatch.setattr(sheet_stage, "ROUTE_OUTPUTS", tmp_path)
    for name in ("rows-0", "rows-1"):
        (base / "nodes" / name).mkdir(parents=True)
        unit = base / "pieces" / name
        unit.mkdir(parents=True)
        (unit / "record.json").write_text("{}")
        os.utime(unit / "record.json", (10, 10))
    joints = base / "pieces" / "joints"
    joints.mkdir()
    (joints / "record.json").write_text("{}")
    assert list(sheet_stage.ready({sample})) == []

    frame = base / "pieces" / "frame"
    frame.mkdir()
    (frame / "record.json").write_text("{}")
    os.utime(frame / "record.json", (30, 30))
    assert list(sheet_stage.ready({sample})) == [(sample, volume)]

    out = base / "sheet"
    out.mkdir()
    (out / "layout.json").write_text('{"pieces": []}')
    os.utime(out / "layout.json", (20, 20))
    (out / "joints.json").write_text("[]")
    calls = []
    monkeypatch.setattr(sheet_stage, "step", lambda *args: calls.append(args))
    assert sheet_stage.laid_out(sample, volume) == []
    assert calls == [("unwrapping.level.scroll_layout", sample, volume)]
    assert (out / "joints.json").read_text() == "[]"
    os.utime(base / "pieces" / "rows-1" / "record.json", (40, 40))
    assert list(sheet_stage.ready({sample})) == []


def asked(port, path, body=None, kind="application/json"):
    data = None if body is None else json.dumps(body).encode()
    request = urllib.request.Request(
        f"http://127.0.0.1:{port}{path}", data=data, headers={"Content-Type": kind} if data else {}
    )
    with urllib.request.urlopen(request) as reply:
        return json.loads(reply.read())


def test_marks_are_kept_through_a_restart_and_only_marks_are_accepted(tmp_path, monkeypatch):
    sheet = tmp_path / "PHerc0000" / "1" / "sheet"
    sheet.mkdir(parents=True)
    (sheet / "layout.json").write_text(json.dumps({"pieces": []}))
    monkeypatch.setattr(site_server, "LEVEL_ROUTE", tmp_path)
    monkeypatch.setattr(site_server, "HAND_MARKS", tmp_path / "hand_marks")

    def running():
        server = ThreadingHTTPServer(("127.0.0.1", 0), site_server.Site)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        return server, server.server_address[1]

    server, port = running()
    stroke = {
        "scan": "level/PHerc0000/1",
        "kind": "ink",
        "width_mm": 0.4,
        "layer": "ink",
        "points": [[10.5, 20.25], [11.0, 21.0]],
    }
    first = asked(port, "/marks", stroke)
    second = asked(port, "/marks", stroke | {"kind": "blank"})
    asked(port, "/marks", {"scan": "level/PHerc0000/1", "remove": first["id"]})
    server.shutdown()
    server, port = running()
    kept = asked(port, "/marks?scan=level/PHerc0000/1")
    assert (
        [k["id"] for k in kept] == [second["id"]]
        and kept[0]["points"] == stroke["points"]
        and kept[0]["kind"] == "blank"
    )
    assert (
        tmp_path / "hand_marks" / "level_route" / "PHerc0000" / "1" / "layouts" / f"{kept[0]['layout']}.json"
    ).exists()
    for body, kind in (
        (stroke, "text/plain"),
        (stroke | {"scan": "../../etc/1"}, "application/json"),
        (stroke | {"kind": "other"}, "application/json"),
    ):
        with pytest.raises(urllib.error.HTTPError):
            asked(port, "/marks", body, kind)
    server.shutdown()


def test_pieces_joined_into_a_sheet_read_one_face_the_larger_area_keeps_its_own():
    a, b, c, d = (0, 1), (1, 1), (2, 1), (3, 1)
    areas = {a: 100.0, b: 90.0, c: 5.0, d: 7.0}
    turns = agreed_turns(areas, [(a, b, 50, 1), (b, c, 30, -1), (c, d, 20, 1)])
    assert turns == {a: 1, b: 1, c: -1, d: -1}
    assert agreed_turns({a: 3.0, b: 9.0}, [(a, b, 10, -1)]) == {a: -1, b: 1}


def test_a_piece_seen_from_the_middle_is_never_mirrored_and_the_curl_only_sets_its_top():
    centre = np.array([5000.0, 5000.0])
    theta = np.linspace(0.3, 0.9, 60)
    z = 500.0 + 16.0 * np.arange(40)[:, None] + 0 * theta
    arc = np.stack([centre[0] + 2000.0 * np.cos(theta) + 0 * z, centre[1] + 2000.0 * np.sin(theta) + 0 * z, z])
    for stored, curl in itertools.product((arc, arc[:, ::-1], arc[:, :, ::-1], arc[:, ::-1, ::-1]), (1, -1)):
        p = {"rows_up": bool(np.diff(stored[2], axis=0).mean() > 0), "outward": outward_grid(stored, centre)}
        flip_rows, flip_cols = flips(p, curl)
        shown = stored[:, ::-1] if flip_rows else stored
        shown = shown[:, :, ::-1] if flip_cols else shown
        forward = np.r_[shown[:2, 20, 30] - centre, 0.0]
        up = np.array([0.0, 0.0, float(curl)])
        right = np.cross(forward, up)
        assert (shown[:, 20, 31] - shown[:, 20, 30]) @ right > 0
        assert (shown[:, 21, 30] - shown[:, 20, 30]) @ up < 0


def test_a_sheet_written_on_its_far_side_turns_over_as_one(tmp_path):
    level = tmp_path / "unit"
    level.mkdir()

    def placed(n, x, group, side):
        (level / f"piece-{n:04d}-side.json").write_text(json.dumps({"written": side}))
        return {
            "unit": "unit",
            "piece": n,
            "area": 10.0,
            "x_mm": x,
            "width_mm": 10.0,
            "flip_cols": False,
            "group": group,
            "moved_from_joint_mm": 0.0,
        }

    pieces = [
        placed(0, 100.0, 1, "far"),
        placed(1, 106.0, 1, "far"),
        placed(2, 300.0, 2, "near"),
        placed(3, 500.0, 3, "far"),
        placed(4, 510.0, 3, "near"),
        placed(5, 519.0, 3, "near"),
    ]
    seen_from_written_face(pieces, tmp_path)
    assert [p["x_mm"] for p in pieces[:2]] == [106.0, 100.0] and all(p["flip_cols"] for p in pieces[:2])
    assert pieces[2]["x_mm"] == 300.0 and not pieces[2]["flip_cols"]
    assert pieces[3]["x_mm"] == 500.0 and pieces[3]["flip_cols"] and not pieces[4]["flip_cols"]
