"""Pieces of a sheet read again from freshly sampled stacks by several models, at the read's depth and as a mean over nearby depths, beside the site's map."""

import json
import shutil
import sys
from pathlib import Path

import numpy as np
from PIL import Image

from ink.model import models as ink_models
from ink.model.input_stack import DEPTH, PAST_FACE, face_layer_of, scrambled
from ink.model.reader import lane_reader
from ink.model.stock_model import SURE
from ink.reading import spot_depths
from sampling import column_sampling as sampler
from sampling.piece_images import fetch_bricks, shown, surface_of
from storage.brick_store import BrickStore
from storage.data_roots import ROUTE_OUTPUTS, SHEETS, calibration_output, ct_bricks
from website import panel_pages

DEPTHS = tuple(range(-4, 5))
EXTRA = 4
PANEL_SIDE = 1536
STRETCH_TOP = 0.995
SCRAMBLE_SEED = 5
CONTROL_KEYS = ("written", "written depth mean", "back face", "scrambled")


def work_of(sample, model):
    return calibration_output("passage_reads") / sample / model


def read_all(sample, volume, pieces, names):
    """Each piece sampled once from its fetched bricks and read by every model named at every depth of DEPTHS, kept per
    model."""
    known = ink_models.models()
    if studios := [name for name in names if known[name].kind == "studio"]:
        raise SystemExit(f"passage reads feed Reader v3's 17-layer windows; {', '.join(studios)} is a Studio model")
    models = {name: known[name].weights for name in names}
    wanted = [
        p
        for p in {spot_depths.piece_key(p): p for p in pieces}.values()
        if not all(read_kept(work_of(sample, m) / "pieces" / f"{spot_depths.piece_key(p)}.npz") for m in models)
    ]
    if not wanted:
        return
    sampler.LAYERS = spot_depths.READ_LAYERS + 2 * EXTRA
    base = ROUTE_OUTPUTS / sample / volume
    axis, faces = spot_depths.axis_of(sample, volume), spot_depths.faces_folder(sample, volume)
    readers = {m: lane_reader(w) for m, w in models.items()}
    for start in range(0, len(wanted), spot_depths.PIECES_A_STORE):
        batch = wanted[start : start + spot_depths.PIECES_A_STORE]
        folder = ct_bricks(sample, "passages")
        shutil.rmtree(folder, ignore_errors=True)
        jobs = [(base / "pieces" / p["unit"] / f"piece-{p['piece']:04d}", None, axis) for p in batch]
        fetch_bricks(sample, volume, jobs, folder)
        store = BrickStore(folder)
        for p, job in zip(batch, jobs):
            side = json.loads(spot_depths.piece_png(faces, p, "-taught-side").with_suffix(".json").read_text())
            xyz, meta = sampler.load_surface(job[0])
            margined, normal = surface_of(xyz, meta, axis)
            turn = side.get("turn", 1)
            cube = sampler.flatten(margined, meta["stride"], normal * turn, store, layers=sampler.LAYERS).cube
            face = cube if side["written"] == "near" else cube[::-1]
            extra, layers = EXTRA, spot_depths.READ_LAYERS
            first = min(face_layer_of(face[extra : extra + layers]) + PAST_FACE, layers - DEPTH)
            back = face[::-1]
            back_first = min(face_layer_of(back[extra : extra + layers]) + PAST_FACE, layers - DEPTH)
            written = np.ascontiguousarray(face[first + extra : first + extra + DEPTH])
            controls = {
                "back": np.ascontiguousarray(back[back_first + extra : back_first + extra + DEPTH]),
                "scrambled": scrambled(written, np.random.default_rng(SCRAMBLE_SEED)),
            }
            for model, reader in readers.items():
                maps = {
                    f"d{d:+d}": reader.read(
                        np.ascontiguousarray(face[first + extra + d : first + extra + d + DEPTH])
                    ).astype(np.float16)
                    for d in DEPTHS
                } | {name: reader.read(stack).astype(np.float16) for name, stack in controls.items()}
                out = work_of(sample, model) / "pieces" / f"{spot_depths.piece_key(p)}.npz"
                out.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(out.with_suffix(".part.npz"), **maps, first=first, written=side["written"])
                out.with_suffix(".part.npz").rename(out)
            print(spot_depths.piece_key(p), "window", first, side["written"], "read by", len(readers), flush=True)
        shutil.rmtree(folder, ignore_errors=True)


def read_kept(path):
    """Whether a piece's reads are kept with their controls."""
    if not path.exists():
        return False
    with np.load(path) as kept:
        return "back" in kept.files


def controls_of(sample, volume, places, models):
    """Per model: the share of the sheet called sure ink on the written face, its depth mean, the back face and scrambled papyrus."""
    faces = spot_depths.faces_folder(sample, volume)
    found = {}
    for model in models:
        totals = np.zeros(5)
        for p in places:
            kept = np.load(work_of(sample, model) / "pieces" / f"{spot_depths.piece_key(p)}.npz")
            sheet = np.asarray(Image.open(spot_depths.piece_png(faces, p))) > 0
            reads = [kept[f"d{d:+d}"].astype(np.float32) for d in DEPTHS]
            parts = [reads[DEPTHS.index(0)], np.mean(reads, axis=0), kept["back"], kept["scrambled"]]
            on = sheet[: parts[0].shape[0], : parts[0].shape[1]]
            totals += [on.sum()] + [float((part[: on.shape[0], : on.shape[1]][on] >= SURE).sum()) for part in parts]
        shares = totals[1:] / max(totals[0], 1)
        found[model] = {key: round(float(v), 4) for key, v in zip(CONTROL_KEYS, shares)} | {
            "written over back": round(float(shares[0] / max(shares[2], 1e-6)), 2)
        }
        print(sample, model, found[model], flush=True)
    return found


def ink_panel(read, sheet):
    """A read as the site shows it, ink dark on light papyrus, black off the sheet."""
    return np.where(sheet > 0, 255 - shown(read, sheet), 0).astype(np.uint8)


def stretched_panel(read, sheet):
    """A read stretched linearly from its median on the sheet to its top half-percent, the same ink shape at the site's contrast."""
    on = read[sheet > 0]
    if not on.size:
        return ink_panel(read, sheet)
    low, high = np.median(on), np.quantile(on, STRETCH_TOP)
    level = np.clip((read - low) / max(high - low, 1e-6), 0, 1)
    return np.where(sheet > 0, 255 - level * 255, 0).astype(np.uint8)


def panels_of(sample, volume, p, models):
    """The scan, the site's map and each model's single and depth-mean reads of one piece, cropped to its passage."""
    faces = spot_depths.faces_folder(sample, volume)
    sheet = np.asarray(Image.open(spot_depths.piece_png(faces, p)))
    site = np.asarray(Image.open(spot_depths.piece_png(faces, p, "-taught")))
    y, x = p.get("y", 0), p.get("x", 0)
    h, w = p.get("h", PANEL_SIDE), p.get("w", PANEL_SIDE)
    crop = (slice(y, y + h), slice(x, x + w))
    panels = {"scan": sheet[crop], "site": np.where(sheet > 0, 255 - site, 0).astype(np.uint8)[crop]}
    for model in models:
        kept = np.load(work_of(sample, model) / "pieces" / f"{spot_depths.piece_key(p)}.npz")
        reads = [kept[f"d{d:+d}"].astype(np.float32) for d in DEPTHS]
        panels[f"{model} depth +0"] = ink_panel(reads[DEPTHS.index(0)], sheet)[crop]
        panels[f"{model} depth mean"] = ink_panel(np.mean(reads, axis=0), sheet)[crop]
        panels[f"{model} depth mean stretched"] = stretched_panel(np.mean(reads, axis=0), sheet)[crop]
        if "back" in kept.files:
            panels[f"{model} other face mirrored"] = stretched_panel(kept["back"].astype(np.float32), sheet)[crop][
                :, ::-1
            ]
    return panels


def run(sample, volume, places_path, page_name, *names):
    """Every listed piece read by every model named (ink.model.models), then one page of panels."""
    places = json.loads(Path(places_path).read_text())
    read_all(sample, volume, places, names)
    models = list(names)
    out = SHEETS / page_name
    for p in places:
        folder = out / spot_depths.piece_key(p)
        shutil.rmtree(folder, ignore_errors=True)
        folder.mkdir(parents=True)
        panels = panels_of(sample, volume, p, models)
        for k, (title, gray) in enumerate(panels.items()):
            Image.fromarray(gray).save(folder / f"{k}-{title.replace(' ', '-')}.png", compress_level=3)
        print(folder.name, "panels", len(panels), flush=True)
    panel_pages.write(out, f"{sample}: passages read again")
    print("page", out / "index.html", flush=True)
    (work_of(sample, "controls") / f"{page_name}.json").parent.mkdir(parents=True, exist_ok=True)
    (work_of(sample, "controls") / f"{page_name}.json").write_text(
        json.dumps(controls_of(sample, volume, places, models), indent=1)
    )


if __name__ == "__main__":
    run(*sys.argv[1:])
