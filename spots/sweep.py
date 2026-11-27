"""Every rendered surface of a spot read by several ink models at every depth from -15 to +15, from both sides.

A model reads a 17-layer window centred on depth d, layers 33 + d - 8 .. 33 + d + 8 of the 67-layer render, as the
PHerc0343 First Letters sweep did (run_depth_sweep.sh, d = -15..+15, both directions). The render's layers run
outward, away from the scroll axis; a model is shown the window in that order ("outward") and reversed
("inward", villa's "reverse", the reading side of the PHerc0343 meshes). Which side holds the writing is decided
from the images, both kept.

The models are every ink model of the library (ink.model.models): stock Reader v3 ("reader-v3") and the checkpoints of
the same network listed in weights/ink/models.json; the original surface is read by stock Reader v3 alone.

Maps are kept as float16 (side, depth, row, column) under spots/SPOT/reads/SURFACE/MODEL/PIECE.npy.
"""

import json
import time

import numpy as np

from ink.model import models
from ink.model.reader import lane_reader
from spots.surfaces import LAYERS, MIDDLE, work_of

DEPTHS = tuple(range(-15, 16))
SIDES = ("inward", "outward")
WINDOW = 17
HALF = WINDOW // 2
WEIGHTS = {
    name: model.weights for name, model in models.models().items() if model.kind == "reader-v3"
}  # 17-layer windows
MODELS = tuple(WEIGHTS)
SURFACES = ("grown", "m7", "original")
READ_BY = {
    "grown": MODELS,
    "m7": MODELS,
    "original": ("reader-v3",),
}  # the original surface is the baseline, read by stock alone


def reader_of(name):
    return lane_reader(WEIGHTS[name])


def window(stack, depth, side):
    begin = MIDDLE + depth - HALF
    assert begin >= 0 and begin + WINDOW <= LAYERS, depth
    part = stack[begin : begin + WINDOW]
    return np.ascontiguousarray(part[::-1] if side == "inward" else part)


def read_spot(spot, records, models=MODELS, surfaces=SURFACES):
    work = work_of(spot["spot_id"])
    for name in models:
        wanted = [
            (surface, r["name"])
            for surface in surfaces
            for r in records
            if name in READ_BY[surface]
            and (work / "stacks" / surface / f"{r['name']}.npy").exists()
            and not (work / "reads" / surface / name / f"{r['name']}.npy").exists()
        ]
        if not wanted:
            continue
        reader = reader_of(name)
        for surface, piece in wanted:
            began = time.time()
            stack = np.load(work / "stacks" / surface / f"{piece}.npy", mmap_mode="r")
            maps = np.zeros((len(SIDES), len(DEPTHS), *stack.shape[1:]), dtype=np.float16)
            for s, side in enumerate(SIDES):
                for k, depth in enumerate(DEPTHS):
                    maps[s, k] = reader.read(window(stack, depth, side)).astype(np.float16)
            out = work / "reads" / surface / name / f"{piece}.npy"
            out.parent.mkdir(parents=True, exist_ok=True)
            part = out.with_suffix(".part.npy")
            np.save(part, maps)
            part.rename(out)
            print(
                spot["spot_id"],
                surface,
                name,
                piece,
                "read",
                maps.shape,
                round(time.time() - began, 1),
                "s",
                flush=True,
            )
        del reader
    (work / "reads" / "recipe.json").write_text(
        json.dumps(
            {"depths": DEPTHS, "sides": SIDES, "window": WINDOW, "middle": MIDDLE, "layers": LAYERS, "read_by": READ_BY}
        )
    )
