"""Every piece of a scan placed on one sheet by the scan's one frame."""

import json
import sys
import time

import numpy as np

from runner import stage_queue
from sampling.piece_images import axis_of_pieces, axis_points
from storage.data_roots import ROUTE_OUTPUTS, stage_folder
from unwrapping.layout import mosaic_layout as ml
from unwrapping.level.piece_joiner import boundary_labels, joins_of, seam_field

BLOCK_EDGES = 10**6
TURN_BIN = 0.25
LEAST_SPAN = 0.1
SMOOTHING_BINS = 9


def source_of(p):
    return p["row"].get("source", p["row"]["piece"])


def turn_length(place, width, span, weight):
    """Length of sheet per turn along the frame, in mm."""
    long = np.abs(span) >= LEAST_SPAN
    edges = np.arange(np.floor(place.min()) - TURN_BIN, np.ceil(place.max()) + 2 * TURN_BIN, TURN_BIN)
    index = np.digitize(place[long], edges) - 1
    speed = width[long] / np.abs(span[long])
    measured = np.full(len(edges), np.nan)
    for b in np.unique(index):
        mine = index == b
        order = np.argsort(speed[mine])
        measured[b] = speed[mine][order][
            np.searchsorted(np.cumsum(weight[long][mine][order]), weight[long][mine].sum() / 2)
        ]
    known = np.flatnonzero(np.isfinite(measured))
    length = np.interp(np.arange(len(edges)), known, measured[known])
    pad = SMOOTHING_BINS // 2
    length = np.convolve(np.pad(length, pad, mode="edge"), np.ones(SMOOTHING_BINS) / SMOOTHING_BINS, mode="valid")
    return edges, length, np.concatenate([[0.0], np.cumsum(length[:-1] * TURN_BIN)])


def seam_joints(units, k, pieces_of, chunks, blocks_of):
    """Joints across one seam."""
    seam = stage_queue.record_of(chunks[k])["cell_planes"][1]
    piece, side, edges, pairs, certainty, _, nodes = seam_field(
        chunks[k : k + 2], seam, [boundary_labels(units[k]), boundary_labels(units[k + 1])]
    )
    joins = joins_of(piece, side, edges, pairs, certainty)
    if not joins:
        return []
    a, b = edges["a"], edges["b"]
    across = (side[a] != side[b]) & (piece[a] >= 0) & (piece[b] >= 0)
    lower, upper = np.where(side[a] == 0, a, b)[across], np.where(side[a] == 0, b, a)[across]
    keys = piece[lower].astype(np.int64) * 2**31 + piece[upper]
    position = np.stack([nodes["x"], nodes["y"], nodes["z"]], axis=1).astype(np.float64) * ml.NODE_SCALE
    um = next(iter(pieces_of[k].values()))["um"]
    z_mid = float(np.median(position[lower, 2])) * um
    cells = {}

    def cells_of(level, n):
        if (level, n) not in cells:
            cells[(level, n)] = ml.near_seam(
                units[level], pieces_of[level][n], z_mid - ml.SEAM_REACH_MM, z_mid + ml.SEAM_REACH_MM
            )
        return cells[(level, n)]

    out = []
    for lo, hi, _ in joins:
        chosen = keys == lo * 2**31 + hi
        best = None
        for na in blocks_of[k].get(lo, ()):
            ta = cells_of(k, na)
            if ta is None:
                continue
            da, ia = ta[0].query(position[lower[chosen]])
            for nb in blocks_of[k + 1].get(hi, ()):
                tb = cells_of(k + 1, nb)
                if tb is None:
                    continue
                db, ib = tb[0].query(position[upper[chosen]])
                close = (da < 3 * pieces_of[k][na]["stride"]) & (db < 3 * pieces_of[k + 1][nb]["stride"])
                if close.sum() >= 3 and (best is None or close.sum() > best[0]):
                    best = (int(close.sum()), na, nb, ia[close], ib[close], ta, tb)
        if best:
            n, na, nb, ia, ib, ta, tb = best
            out.append(
                {
                    "a": (k, na),
                    "b": (k + 1, nb),
                    "edges": n,
                    "columns": [float(np.median(ta[1][ia])), float(np.median(tb[1][ib]))],
                    "rows": [float(np.median(ta[2][ia])), float(np.median(tb[2][ib]))],
                }
            )
    return out


def block_joints(pieces_of):
    """Each block of a piece held to the block before it at its exact offset in the piece's own grid."""
    out = []
    for k, group in enumerate(pieces_of):
        by_source = {}
        for n, p in group.items():
            by_source.setdefault(source_of(p), []).append(n)
        for members in by_source.values():
            first = members[0]
            r0, c0 = group[first]["row"].get("block", [0, 0])
            for n in members[1:]:
                r, c = group[n]["row"]["block"]
                out.append(
                    {
                        "a": (k, first),
                        "b": (k, n),
                        "edges": BLOCK_EDGES,
                        "columns": [float(c - c0), 0.0],
                        "rows": [float(r - r0), 0.0],
                    }
                )
    return out


def cored(p):
    """A block's occupancy without the edge it borrows from its neighbours, so neighbouring blocks never seem to overlap."""
    top, bottom, left, right = p["row"].get("overlap", (0, 0, 0, 0))
    rows, columns = p["shape"]
    step = ml.OCCUPANCY_MM / p["side"]
    r = np.minimum((np.arange(p["mask"].shape[0]) * step).astype(np.int64), rows - 1)
    c = np.minimum((np.arange(p["mask"].shape[1]) * step).astype(np.int64), columns - 1)
    p["mask"] = p["mask"] & ((r >= top) & (r < rows - bottom))[:, None] & ((c >= left) & (c < columns - right))[None, :]
    return p


def run(sample, volume):
    start = time.time()
    base = ROUTE_OUTPUTS / sample / volume
    chunks = sorted(stage_folder(base, "nodes").glob("rows-*"))
    units = [base / "pieces" / c.name for c in chunks]
    axis = axis_points(sample, volume)
    axis = axis_of_pieces(units) if axis is None else axis
    with np.load(base / "pieces" / "frame" / "levels.npz") as frame:
        curl, first, levels = int(frame["curl"]), frame["first"], frame["level"]
        held = {(int(a), int(b)) for a, b in zip(frame["held_p"], frame["held_q"])}
    turns = []
    for unit in units:
        with np.load(unit / "levels.npz") as saved:
            turns.append(saved["turns"])
    pieces_of, blocks_of = [], []
    for unit in units:
        rows = [r for r in stage_queue.record_of(unit)["pieces"] if r.get("written")]
        pieces_of.append({r["piece"]: cored(ml.summary(unit, r, axis) | {"row": r}) for r in rows})
        blocks = {}
        for r in rows:
            blocks.setdefault(r.get("source", r["piece"]), []).append(r["piece"])
        blocks_of.append(blocks)
        print(unit.name, len(rows), "blocks of", len(blocks), "pieces", round(time.time() - start), flush=True)
    keys = [(k, n) for k, group in enumerate(pieces_of) for n in group]
    flat = {(k, n): pieces_of[k][n] for k, n in keys}
    level = {key: int(levels[first[key[0]] + source_of(p)]) for key, p in flat.items()}
    out = stage_folder(base, "sheet")
    out.mkdir(parents=True, exist_ok=True)
    cache = out / "joints.json"
    if cache.exists():
        seams = [j | {"a": tuple(j["a"]), "b": tuple(j["b"])} for j in json.loads(cache.read_text())]
    else:
        seams = []
        for k in range(len(units) - 1):
            found = seam_joints(units, k, pieces_of, chunks, blocks_of)
            seams += found
            print("seam", k, len(found), round(time.time() - start), flush=True)
        cache.write_text(json.dumps(seams))
    refused = sum(
        (first[j["a"][0]] + source_of(flat[j["a"]]), first[j["b"][0]] + source_of(flat[j["b"]])) not in held
        for j in seams
    )
    standing = seams
    blocks = block_joints(pieces_of)
    for p in flat.values():
        p["turn"] = 1
    facing = {}
    for k, p in sorted(flat.items(), key=lambda kv: -kv[1]["area"]):
        facing.setdefault((k[0], source_of(p)), ml.flips(p, curl))
    for k, p in flat.items():
        p["flip_rows"], p["flip_cols"] = facing[(k[0], source_of(p))]
        p["mask"] = p["mask"][::-1] if p["flip_rows"] else p["mask"]
        p["mask"] = p["mask"][:, ::-1] if p["flip_cols"] else p["mask"]
    groups, group_of, local = ml.mosaics(keys, flat, blocks + standing)
    area = np.array([sum(flat[k]["area"] for k in m) for m in groups])
    sources = sorted({(k[0], source_of(p)) for k, p in flat.items()})
    extent = {}
    for k, p in flat.items():
        _r0, c0 = p["row"].get("block", [0, 0])
        key = (k[0], source_of(p))
        extent[key] = max(extent.get(key, 0.0), c0 * p["side"] + p["shape"][1] * p["side"])
    place = np.array(
        [level[(k, next(n for n in blocks_of[k][s] if (k, n) in flat))] + curl * turns[k][s] for k, s in sources]
    )
    width = np.array([extent[key] for key in sources])
    span = np.array(
        [
            flat[(k, blocks_of[k][s][0])]["row"]["turns_spanned"] if (k, blocks_of[k][s][0]) in flat else 0.0
            for k, s in sources
        ]
    )
    weight = np.array([sum(flat[(k, n)]["area"] for n in blocks_of[k][s] if (k, n) in flat) for k, s in sources])
    edges, _length, arc = turn_length(place, width, span, weight)
    roll_mm = float(arc.max())
    along = dict(zip(sources, roll_mm - np.interp(place, edges, arc)))
    shift = [
        float(
            np.median(
                [
                    along[(k[0], source_of(flat[k]))] - local[k][0] - flat[k]["shape"][1] * flat[k]["side"] / 2
                    for k in members
                ]
            )
        )
        for members in groups
    ]
    z_top, z_bottom = max(p["z_high"] for p in flat.values()), min(p["z_low"] for p in flat.values())
    items, places = [], np.zeros((len(keys), 2))
    for n, k in enumerate(keys):
        p, g = flat[k], group_of[k]
        seed = groups[g][0]
        height = (z_top - flat[seed]["z_high"]) if curl > 0 else (flat[seed]["z_low"] - z_bottom)
        places[n] = (shift[g] + local[k][0], height + local[k][1])
        items.append(
            {
                "unit": p["unit"],
                "piece": p["piece"],
                "source": int(source_of(p)),
                "turn": p["turn"],
                "area": p["area"],
                "group": int(g),
                "group_area": float(area[g]),
                "width_mm": p["shape"][1] * p["side"],
                "height_mm": p["shape"][0] * p["side"],
                "flip_rows": bool(p["flip_rows"]),
                "flip_cols": bool(p["flip_cols"]),
                "um": p["um"],
                "shape": p["shape"],
                "stride": p["stride"],
                "level": level[k],
                "overlap": p["row"].get("overlap", [0, 0, 0, 0]),
            }
        )
    places[:, 0] -= places[:, 0].min()
    places[:, 1] -= places[:, 1].min()
    joined = places.copy()
    moved = ml.packed(items, [flat[k]["mask"] for k in keys], places, float(places[:, 1].max()) + 120)
    places[:, 0] -= places[:, 0].min()
    for n, item in enumerate(items):
        item["x_mm"], item["y_mm"], item["moved_from_joint_mm"] = (
            float(places[n, 0]),
            float(places[n, 1]),
            float(places[n, 0] - joined[n, 0]),
        )
    top = np.sort(area)[::-1][:6] / 100
    record = {
        "sample": sample,
        "volume": volume,
        "chirality": curl,
        "joints": len(standing),
        "joints the frame does not hold": refused,
        "pieces": len(sources),
        "blocks": len(items),
        "mosaics": len(groups),
        "largest_mosaics_cm2": top.tolist(),
        "in_mosaics_of_4cm2": float(area[area >= 400].sum() / 100),
        "area_cm2": float(area.sum() / 100),
        "turns": [float(place.min()), float(place.max())],
        "roll_mm": roll_mm,
        "moved": moved,
        "seconds": round(time.time() - start),
    }
    (out / "layout.json").write_text(json.dumps({"record": record, "pieces": items}))
    print(record, flush=True)


if __name__ == "__main__":
    run(sys.argv[1], sys.argv[2])
