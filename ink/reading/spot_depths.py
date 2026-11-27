"""Places circled on a sheet with the viewer's Spot tool, read again at nearby depths."""

import calendar
import itertools
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from ink.model.input_stack import DEPTH, PAST_FACE, face_layer_of
from ink.model.reader import lane_reader
from ink.model.stock_model import SURE
from ink.reading.piece_reading import STUDIO, WEIGHTS, digest
from sampling import column_sampling as sampler
from sampling.piece_images import (
    axis_of_pieces,
    axis_points,
    fetch_bricks,
    shown,
    surface_of,
)
from sampling.sheet_images import MARGIN_CELLS
from storage.brick_store import BrickStore
from storage.data_roots import ROUTE_OUTPUTS, SHEETS, calibration_output, ct_bricks, stage_folder
from storage.tile_store import TileStore, TileWriter
from website import panel_pages
from website.sheet_tiles import PARTING_MM, STRIP, TILE, parted, upright
from website.site_pages import scan_of
from website.site_server import marks_of

READ_LAYERS = 31
EXTRA = 2
OFFSETS = (-2, 0, 2)
CLICK_MM, CLICK_BOX_MM = 2.0, 10.0
MOST_BOX_MM = 40.0
SHOWN_MARGIN_MM = 1.0
TILE_AGREES, INKISH = 12, 40
MOST_UNSURE = 6
PIECES_A_STORE = 10
PANEL_PX = 640
SPOT_GREEN = (57, 255, 20)
TRY_LATER = 75


def work_of(sample):
    return calibration_output("spot_depths") / sample


def seconds_of(stamp):
    return calendar.timegm(time.strptime(stamp, "%Y-%m-%dT%H:%M:%SZ"))


def taught_layer(sheet):
    return next(layer for layer in json.loads((sheet / "layers.json").read_text()) if layer["path"] == "taught")


def compositions(sample, volume, name):
    """Every composed taught layer kept for a sheet, oldest first, as (stamp, placement written, folder)."""
    root, sheet = work_of(sample) / "compositions", SHEETS / name
    stamp = taught_layer(sheet)["stamp"]
    placement = stage_folder(ROUTE_OUTPUTS / sample / volume, "sheet") / "placed-taught.json"
    here = root / f"taught-{stamp}"
    if not (here / "composition.json").exists():
        if placement.stat().st_mtime > stamp + 1:
            print(
                "the sheet is being composed again: its placement is newer than its published tiles; trying later",
                flush=True,
            )
            sys.exit(TRY_LATER)
        part = root / f"taught-{stamp}-part"
        shutil.rmtree(part, ignore_errors=True)
        part.mkdir(parents=True)
        with TileWriter(part) as kept:
            kept.carry(TileStore(sheet), "taught", levels=1)
        shutil.copy2(placement, part / "placed-taught.json")
        shutil.copy2(sheet / "layers.json", part / "layers.json")
        (part / "composition.json").write_text(
            json.dumps({"stamp": stamp, "placement_written": placement.stat().st_mtime})
        )
        shutil.rmtree(here, ignore_errors=True)
        part.rename(here)
    found = []
    for folder in root.glob("taught-*"):
        if (folder / "composition.json").exists():
            meta = json.loads((folder / "composition.json").read_text())
            found.append((meta["stamp"], meta["placement_written"], folder))
    return sorted(found)


def composition_of(mark, kept):
    """The composition on the screen when a mark was drawn."""
    at = seconds_of(mark["time"])
    before = [c for c in kept if c[0] <= at]
    return before[-1] if before else kept[0]


def spot_box(mark, pixel_mm):
    """A spot's box (y0, y1, x0, x1) in sheet pixels, with the margin shown round it, and its outline as sheet pixels."""
    points = np.array(mark["points"], dtype=np.float64)
    low, high = points.min(axis=0), points.max(axis=0)
    middle = (low + high) / 2
    if (high - low).max() < CLICK_MM:
        half = CLICK_BOX_MM / 2
        outline = middle + np.array([[-half, -half], [half, -half], [half, half], [-half, half]])
    else:
        outline = points
    low, high = outline.min(axis=0), outline.max(axis=0)
    if (high - low).max() > MOST_BOX_MM:
        low, high = np.maximum(low, middle - MOST_BOX_MM / 2), np.minimum(high, middle + MOST_BOX_MM / 2)
    low, high = low - SHOWN_MARGIN_MM, high + SHOWN_MARGIN_MM
    x0, y0 = (np.floor(low / pixel_mm)).astype(int)
    x1, y1 = (np.ceil(high / pixel_mm)).astype(int)
    return (max(int(y0), 0), int(y1), max(int(x0), 0), int(x1)), outline / pixel_mm


def inside(outline, box):
    """Pixels of a box inside a spot's outline."""
    y0, y1, x0, x1 = box
    mask = Image.new("L", (x1 - x0, y1 - y0), 0)
    if len(outline) >= 3:
        ImageDraw.Draw(mask).polygon([(float(x - x0), float(y - y0)) for x, y in outline], fill=1)
    return np.asarray(mask, dtype=bool)


def faces_folder(sample, volume):
    return stage_folder(ROUTE_OUTPUTS / sample / volume, "faces") / "level-0"


def piece_png(faces, p, layer=""):
    return faces / p["unit"] / f"piece-{p['piece']:04d}{layer}.png"


def placed_extent(p, pixel_mm):
    """The sheet pixels (y0, y1, x0, x1) a piece's image covers, margins included."""
    margin = round(MARGIN_CELLS * p["stride"] * p["um"] / pixel_mm)
    rows, columns = p["shape"]
    height, width = (rows + 2 * MARGIN_CELLS - 1) * p["stride"] + 1, (columns + 2 * MARGIN_CELLS - 1) * p["stride"] + 1
    py, px = round(p["y_mm"] / pixel_mm) - margin, round(p["x_mm"] / pixel_mm) - margin
    return py, py + height, px, px + width


def painted(members, faces, box, pixel_mm, strip):
    """Which member the composed taught layer shows at each pixel of a box inside one strip of tiles, -1 for none."""
    width = max(3, round(PARTING_MM / pixel_mm))
    y0, y1, x0, x1 = box
    Y0, Y1, X0, X1 = max(y0 - width, 0), y1 + width, max(x0 - width, strip[0]), min(x1 + width, strip[1])
    covered = np.zeros((Y1 - Y0, X1 - X0), dtype=bool)
    owner = np.full(covered.shape, -1, dtype=np.int32)
    mosaic = np.zeros(covered.shape, dtype=np.int32)
    mosaics, held = {}, []
    for k, p in sorted(enumerate(members), key=lambda kp: -kp[1]["area"]):
        py, _, px, _ = placed_extent(p, pixel_mm)
        seen = upright(np.asarray(Image.open(piece_png(faces, p))) > 0, p)
        h, w = seen.shape
        a0, a1, b0, b1 = max(py, Y0), min(py + h, Y1), max(px, X0), min(px + w, X1)
        if a1 <= a0 or b1 <= b0:
            continue
        margin = round(MARGIN_CELLS * p["stride"] * p["um"] / pixel_mm)
        top, bottom, left, right = p.get("overlap", (0, 0, 0, 0))
        top, bottom = (bottom, top) if p["flip_rows"] else (top, bottom)
        left, right = (right, left) if p["flip_cols"] else (left, right)
        cells = p["stride"] * p["um"] / pixel_mm
        own = np.zeros((h, w), dtype=bool)
        own[
            margin + round(top * cells) : h - margin - round(bottom * cells),
            margin + round(left * cells) : w - margin - round(right * cells),
        ] = True
        number = mosaics.setdefault((p.get("group"), round(p.get("moved_from_joint_mm", 0.0))), len(mosaics) + 1)
        held.append(
            (
                (slice(a0 - Y0, a1 - Y0), slice(b0 - X0, b1 - X0)),
                seen[a0 - py : a1 - py, b0 - px : b1 - px],
                own[a0 - py : a1 - py, b0 - px : b1 - px],
                k,
                number,
            )
        )
    for margins in (False, True):
        for here, seen, own, k, number in held:
            free = seen & ~covered[here] & (own | margins)
            covered[here] |= free
            owner[here][free] = k
            mosaic[here][free] = number
    parted(covered, mosaic, width)
    owner[~covered] = -1
    return owner[y0 - Y0 : y1 - Y0, max(x0, strip[0]) - X0 : min(x1, strip[1]) - X0]


def owners_of(pieces, faces, box, pixel_mm):
    """Which piece the composed taught layer shows at each pixel of a box, strip by strip as it was painted."""
    y0, y1, x0, x1 = box
    owner = np.full((y1 - y0, x1 - x0), -1, dtype=np.int32)
    span = STRIP * TILE
    for s0 in range(x0 // span * span, x1, span):
        strip = (s0, s0 + span)
        lo, hi = strip[0] * pixel_mm, strip[1] * pixel_mm
        members = [k for k, p in enumerate(pieces) if p["x_mm"] < hi and p["x_mm"] + p["width_mm"] > lo]
        if not members:
            continue
        part = painted([pieces[k] for k in members], faces, box, pixel_mm, strip)
        a, b = max(x0, strip[0]) - x0, min(x1, strip[1]) - x0
        owner[:, a:b] = np.where(part >= 0, np.array(members, dtype=np.int32)[np.maximum(part, 0)], -1)
    return owner


def composed(folder, box):
    """A box of a kept composition's taught layer, as its tiles hold it."""
    y0, y1, x0, x1 = box
    out = np.zeros((y1 - y0, x1 - x0), dtype=np.uint8)
    store = TileStore(folder)
    for r in range(y0 // TILE, (y1 - 1) // TILE + 1):
        for c in range(x0 // TILE, (x1 - 1) // TILE + 1):
            tile = store.image("taught", 0, c, r)
            if tile is not None:
                a0, a1, b0, b1 = max(y0, r * TILE), min(y1, (r + 1) * TILE), max(x0, c * TILE), min(x1, (c + 1) * TILE)
                out[a0 - y0 : a1 - y0, b0 - x0 : b1 - x0] = tile[
                    a0 - r * TILE : a1 - r * TILE, b0 - c * TILE : b1 - c * TILE
                ]
    return out


def laid(pieces, owner, box, images, pixel_mm):
    """One picture of a box from each owner's upright image, black where the sheet shows nothing."""
    y0, _y1, x0, _x1 = box
    out = np.zeros(owner.shape, dtype=np.float32)
    for k in np.unique(owner[owner >= 0]):
        p = pieces[k]
        image = upright(images[k], p)
        py, _, px, _ = placed_extent(p, pixel_mm)
        here = owner == k
        rows, columns = np.nonzero(here)
        out[rows, columns] = image[rows + y0 - py, columns + x0 - px]
    return out


def agreement(screen, predicted):
    """Share of inky pixels where a traced picture matches the tiles that were on the screen."""
    judged = (screen > INKISH) | (predicted > INKISH)
    if judged.sum() < 500:
        return None
    return float((np.abs(screen.astype(np.int32) - predicted.astype(np.int32)) <= TILE_AGREES)[judged].mean())


def traced(mark, kept, sample, volume, pixel_mm):
    """A spot's box, outline, composition, the pieces under it and which shows where, and how well that matches the tiles of the time."""
    stamp, written, folder = composition_of(mark, kept)
    placed = json.loads((folder / "placed-taught.json").read_text())["pieces"]
    faces = faces_folder(sample, volume)
    box, outline = spot_box(mark, pixel_mm)
    reach = max(3, round(PARTING_MM / pixel_mm))
    y0, y1, x0, x1 = box[0] - reach, box[1] + reach, box[2] - reach, box[3] + reach
    sure, unsure = [], []
    for p in placed:
        py, py1, px, px1 = placed_extent(p, pixel_mm)
        path = piece_png(faces, p, "-taught")
        if py >= y1 or py1 <= y0 or px >= x1 or px1 <= x0 or not path.exists() or not piece_png(faces, p).exists():
            continue
        made = path.stat().st_mtime
        if made <= written:
            sure.append(p)
        elif made <= stamp:
            unsure.append(p)
    screen = composed(folder, box)
    best = None
    choices = (
        [list(c) for n in range(len(unsure) + 1) for c in itertools.combinations(unsure, n)]
        if len(unsure) <= MOST_UNSURE
        else [unsure, []]
    )
    for chosen in choices:
        pieces = sure + chosen
        owner = owners_of(pieces, faces, box, pixel_mm)
        maps = {k: np.asarray(Image.open(piece_png(faces, pieces[k], "-taught"))) for k in np.unique(owner[owner >= 0])}
        score = agreement(screen, laid(pieces, owner, box, maps, pixel_mm).astype(np.uint8))
        if best is None or (score or 0) > (best[2] or 0):
            best = (pieces, owner, score)
    pieces, owner, score = best
    within = inside(outline, box)
    shown_here = sorted({int(k) for k in np.unique(owner[owner >= 0])})
    return {
        "box": box,
        "outline": outline,
        "composition": stamp,
        "screen": screen,
        "pieces": [pieces[k] for k in shown_here],
        "owner": np.searchsorted(shown_here, owner).astype(np.int32) * (owner >= 0) - (owner < 0),
        "agreement": score,
        "within": within,
    }


def piece_key(p):
    return f"{p['unit']}-piece-{p['piece']:04d}"


def axis_of(sample, volume):
    """The axis the read took."""
    axis = axis_points(sample, volume)
    return axis_of_pieces(sorted((ROUTE_OUTPUTS / sample / volume / "pieces").glob("rows-*"))) if axis is None else axis


def read_pieces(sample, volume, pieces, model):
    """Each piece sampled as the full read samples it, two layers deeper either side, and read with its window where the read put it and moved by OFFSETS; kept, so a piece under two spots is read once."""
    if STUDIO:
        raise SystemExit("spot depths feed Reader v3's 17-layer windows; HERCULEX_INK_MODEL names a Studio model")
    store_root = work_of(sample) / "pieces"
    store_root.mkdir(parents=True, exist_ok=True)
    wanted = [
        p for p in {piece_key(p): p for p in pieces}.values() if not (store_root / f"{piece_key(p)}.npz").exists()
    ]
    if not wanted:
        return
    sampler.LAYERS = READ_LAYERS + 2 * EXTRA
    base = ROUTE_OUTPUTS / sample / volume
    axis, faces = axis_of(sample, volume), faces_folder(sample, volume)
    reader = None
    for start in range(0, len(wanted), PIECES_A_STORE):
        batch = wanted[start : start + PIECES_A_STORE]
        folder = ct_bricks(sample, "spots")
        shutil.rmtree(folder, ignore_errors=True)
        jobs = [(base / "pieces" / p["unit"] / f"piece-{p['piece']:04d}", None, axis) for p in batch]
        fetch_bricks(sample, volume, jobs, folder)
        store = BrickStore(folder)
        reader = reader or lane_reader(WEIGHTS)
        for p, job in zip(batch, jobs):
            began = time.time()
            side = json.loads(piece_png(faces, p, "-taught-side").with_suffix(".json").read_text())
            xyz, meta = sampler.load_surface(job[0])
            margined, normal = surface_of(xyz, meta, axis)
            turn = side.get("turn", p.get("turn", 1))
            cube = sampler.flatten(margined, meta["stride"], normal * turn, store, layers=sampler.LAYERS).cube
            layer0 = np.asarray(Image.open(piece_png(faces, p, "-layer0")))
            same_layer = (
                float((cube[EXTRA + READ_LAYERS // 2] == layer0).mean()) if layer0.shape == cube.shape[1:] else None
            )
            face = cube if side["written"] == "near" else cube[::-1]
            first = min(face_layer_of(face[EXTRA : EXTRA + READ_LAYERS]) + PAST_FACE, READ_LAYERS - DEPTH)
            maps = {}
            for offset in OFFSETS:
                at = first + EXTRA + offset
                maps[f"d{offset:+d}"] = reader.read(np.ascontiguousarray(face[at : at + DEPTH])).astype(np.float16)
            sheet = np.asarray(Image.open(piece_png(faces, p)))
            stored = np.asarray(Image.open(piece_png(faces, p, "-taught")))
            again = shown(maps["d+0"].astype(np.float32), sheet)
            on = sheet > 0
            same_map = (
                float((np.abs(again.astype(np.int32) - stored.astype(np.int32)) <= 2)[on].mean())
                if stored.shape == again.shape and on.any()
                else None
            )
            part = store_root / f"{piece_key(p)}.part.npz"
            np.savez_compressed(
                part,
                **maps,
                first=first,
                written=side["written"],
                model=model,
                read_model=side.get("model", ""),
                same_layer=same_layer if same_layer is not None else -1.0,
                same_map=same_map if same_map is not None else -1.0,
            )
            part.rename(store_root / f"{piece_key(p)}.npz")
            print(
                piece_key(p),
                "window",
                first,
                side["written"],
                "layer 0 as the read's",
                same_layer,
                "map at depth 0 as the read's",
                same_map,
                round(time.time() - began, 1),
                "s",
                flush=True,
            )
        shutil.rmtree(folder, ignore_errors=True)


def contrast(image, low, high):
    return np.clip((image.astype(np.float32) - low) * (255.0 / max(high - low, 1)), 0, 255).astype(np.uint8)


def outlined(gray, outline, box, scale=1.0):
    """A grey panel in colour with the spot's outline drawn on it."""
    y0, _, x0, _ = box
    image = Image.fromarray(gray).convert("RGB")
    if scale != 1.0:
        image = image.resize((max(1, round(image.width * scale)), max(1, round(image.height * scale))), Image.LANCZOS)
    if len(outline) >= 3:
        ImageDraw.Draw(image).line(
            [((x - x0) * scale, (y - y0) * scale) for x, y in [*outline, outline[0]]], fill=SPOT_GREEN, width=2
        )
    return image


def laid_out(sample, volume, name, mark, trace, out, pixel_mm=None, scan_range=None):
    """A spot's panels at full size and as one strip, and its numbers."""
    faces = faces_folder(sample, volume)
    pieces, owner, box, within = trace["pieces"], trace["owner"], trace["box"], trace["within"]
    kept = {piece_key(p): np.load(work_of(sample) / "pieces" / f"{piece_key(p)}.npz") for p in pieces}
    scan_range = scan_range or next(
        (layer for layer in json.loads((SHEETS / name / "layers.json").read_text()) if layer["path"] == "taught-scan"),
        {"low": 0, "high": 255},
    )
    pixel_mm = pixel_mm or taught_layer(SHEETS / name)["mm"]
    scan = laid(
        pieces, owner, box, {k: np.asarray(Image.open(piece_png(faces, p))) for k, p in enumerate(pieces)}, pixel_mm
    )
    panels = {"scan": contrast(scan, scan_range["low"], scan_range["high"]), "on the site": trace["screen"]}
    reads = {}
    for offset in OFFSETS:
        key = f"d{offset:+d}"
        reads[key] = laid(
            pieces, owner, box, {k: kept[piece_key(p)][key].astype(np.float32) for k, p in enumerate(pieces)}, pixel_mm
        )
    reads["mean"] = np.mean([reads[f"d{offset:+d}"] for offset in OFFSETS], axis=0)
    titles = {f"d{offset:+d}": f"depth {offset:+d}" for offset in OFFSETS} | {"mean": "mean of the three"}
    on = owner >= 0
    for key, read in reads.items():
        panels[titles[key]] = np.where(on, shown(read, np.ones_like(read)), 0).astype(np.uint8)
    out.mkdir(parents=True, exist_ok=True)
    files = {}
    for k, (title, gray) in enumerate(panels.items()):
        files[title] = f"{k}-{title.replace(' ', '-')}.png"
        outlined(gray, trace["outline"], box).save(out / files[title], compress_level=3)
    scale = min(1.0, PANEL_PX / max(gray.shape))
    strip = [outlined(gray, trace["outline"], box, scale) for gray in panels.values()]
    sheet = Image.new("RGB", (sum(s.width for s in strip) + 8 * (len(strip) - 1), strip[0].height + 22), (17, 17, 17))
    draw, x = ImageDraw.Draw(sheet), 0
    for title, image in zip(panels, strip):
        sheet.paste(image, (x, 22))
        draw.text((x + 4, 5), title, fill=(255, 255, 255))
        x += image.width + 8
    sheet.save(out / "strip.jpg", quality=92)
    judged = within & on
    numbers = {
        key: {
            "sure": round(float((read[judged] >= SURE).mean()), 4) if judged.any() else None,
            "mean": round(float(read[judged].mean()), 4) if judged.any() else None,
        }
        for key, read in reads.items()
    }
    pairs = {
        f"{a} against {b}": round(float(np.corrcoef(reads[a][judged], reads[b][judged])[0, 1]), 3)
        for a, b in (("d-2", "d+0"), ("d+2", "d+0"), ("d-2", "d+2"))
        if judged.sum() > 100
    }
    checks = {
        piece_key(p): {
            "window": int(kept[piece_key(p)]["first"]),
            "written": str(kept[piece_key(p)]["written"]),
            "layer 0 same": round(float(kept[piece_key(p)]["same_layer"]), 4),
            "map same": round(float(kept[piece_key(p)]["same_map"]), 4),
            "same model": str(kept[piece_key(p)]["model"]) == str(kept[piece_key(p)]["read_model"]),
        }
        for p in pieces
    }
    row = {
        "id": mark["id"],
        "time": mark["time"],
        "layer": mark.get("layer", ""),
        "x_mm": round(float(np.mean([q[0] for q in mark["points"]])), 2),
        "y_mm": round(float(np.mean([q[1] for q in mark["points"]])), 2),
        "box_px": list(map(int, box)),
        "area_cm2": round(float(within.sum()) * pixel_mm**2 / 100, 3),
        "composition": trace["composition"],
        "tracing agrees with the screen": trace["agreement"],
        "pieces": checks,
        "within the circle": numbers,
        "depths agree": pairs,
        "files": files,
    }
    (out / "spot.json").write_text(json.dumps(row, indent=1))
    return row


def run(sample, volume, name, marks_path=None, page_name=None):
    """Every standing spot of a sheet, read once."""
    scan = scan_of(sample, volume)
    marks = json.loads(Path(marks_path).read_text()) if marks_path else marks_of(scan)
    spots = [m for m in marks if m.get("kind") == "spot"]
    page_name = page_name or name.replace("sheet-", "spots-", 1)
    out = SHEETS / page_name
    print(len(spots), "spots on", scan, flush=True)
    kept = compositions(sample, volume, name)
    pixel_mm = taught_layer(SHEETS / name)["mm"]
    model = digest()
    traces = {}
    for mark in spots:
        if (out / f"spot-{mark['id'][:8]}" / "spot.json").exists():
            continue
        traces[mark["id"]] = traced(mark, kept, sample, volume, pixel_mm)
        t = traces[mark["id"]]
        print(
            "spot",
            mark["id"][:8],
            mark["time"],
            "box",
            t["box"],
            "pieces",
            [piece_key(p) for p in t["pieces"]],
            "tracing agrees",
            t["agreement"],
            flush=True,
        )
    read_pieces(sample, volume, [p for t in traces.values() for p in t["pieces"]], model)
    for mark in spots:
        if mark["id"] in traces and traces[mark["id"]]["pieces"]:
            laid_out(sample, volume, name, mark, traces[mark["id"]], out / f"spot-{mark['id'][:8]}")
    standing = {f"spot-{m['id'][:8]}" for m in spots}
    places = [place for place in panel_pages.places_of(out) if place["name"] in standing]
    panel_pages.write(out, f"{sample}: circled spots at three depths", places=places)
    print("page", out / "index.html", len(places), "spots", flush=True)


if __name__ == "__main__":
    run(*sys.argv[1:6])
