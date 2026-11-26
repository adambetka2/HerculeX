"""Layout stage: every piece of a scan placed on one sheet."""

import heapq
import itertools
import json
import sys
import time

import numpy as np
from scipy.spatial import cKDTree

from sampling import column_sampling as sampler
from sampling.piece_images import axis_of_pieces, axis_points, centre_at, surface_of
from sampling.sheet_images import MARGIN_CELLS
from storage.data_roots import ROUTE_OUTPUTS, stage_folder
from unwrapping.layout.roll_placement import (
    loaded,
    outward_grid,
    placed,
    scroll_centre,
)
from unwrapping.level.piece_joiner import boundary_labels, joins_of, seam_field

SEAM_REACH_MM = 1.5
SHEET_SHARE = 0.5
OCCUPANCY_MM = 1.0
ALLOWED_OVERLAP = 0.1
SHIFT_STEP_MM = 5.0
MOST_SHIFT_MM = 1500.0
NODE_SCALE = 2.0
MOSAIC_SPAN_MM = 2000.0
MIN_INNER_EDGES = 5


def scan_chirality(units):
    """One curl for the whole scan, by the costs of all its units."""
    total = {"1": 0.0, "-1": 0.0}
    for unit in units:
        costs = json.loads((unit / "record.json").read_text())["curl_costs"]
        for key in total:
            total[key] += costs[key]
    return 1 if total["1"] < total["-1"] else -1


def summary(unit, row, axis):
    """What layout needs of one piece."""
    folder = unit / f"piece-{row['piece']:04d}"
    xyz, meta = sampler.load_surface(folder)
    good = xyz[0] > 0
    um = meta["voxel_um"] / 1000
    side = meta["stride"] * um
    i, j = np.nonzero(good)
    pick = slice(None, None, max(1, len(i) // 4000))
    z = xyz[2, i, j]
    a = np.linalg.lstsq(np.c_[i[pick], j[pick], np.ones(len(i[pick]))], z[pick], rcond=None)[0]
    dx = xyz[0, i, j][pick] - np.interp(z[pick], axis[:, 0], axis[:, 1])
    dy = xyz[1, i, j][pick] - np.interp(z[pick], axis[:, 0], axis[:, 2])
    middle = np.median(xyz[:, good], axis=1)
    rows = np.minimum(
        (np.arange(int(good.shape[0] * side / OCCUPANCY_MM) + 1) * OCCUPANCY_MM / side).astype(np.int64),
        good.shape[0] - 1,
    )
    columns = np.minimum(
        (np.arange(int(good.shape[1] * side / OCCUPANCY_MM) + 1) * OCCUPANCY_MM / side).astype(np.int64),
        good.shape[1] - 1,
    )
    return {
        "mask": good[np.ix_(rows, columns)],
        "unit": unit.name,
        "piece": row["piece"],
        "area": row["area_mm2"],
        "side": side,
        "um": um,
        "shape": list(good.shape),
        "rows_up": bool(a[0] > 0),
        "stride": meta["stride"],
        "outward": outward_grid(xyz.astype(np.float64), centre_at(axis, middle[2], middle)),
        "radius": float(np.median(np.hypot(dx, dy)) * um),
        "z_low": float(z.min() * um),
        "z_high": float(z.max() * um),
    }


def flips(p, chirality):
    """Rows and columns turned so the piece stands upright seen from the written face, up being +z when the chirality says so."""
    flip_rows = p["rows_up"] == (chirality > 0)
    return flip_rows, flip_rows == p["outward"]


def frame(p, chirality):
    p["flip_rows"], p["flip_cols"] = flips(p, chirality)
    p["mask"] = p["mask"][::-1] if p["flip_rows"] else p["mask"]
    p["mask"] = p["mask"][:, ::-1] if p["flip_cols"] else p["mask"]


def near_seam(unit, p, z_low, z_high):
    """Cells of a piece between two heights."""
    xyz, _ = sampler.load_surface(unit / f"piece-{p['piece']:04d}")
    i, j = np.nonzero((xyz[0] > 0) & (xyz[2] * p["um"] >= z_low) & (xyz[2] * p["um"] <= z_high))
    return None if len(i) == 0 else (cKDTree(xyz[:, i, j].T.astype(np.float64)), j, i)


def seam_joints(units, k, pieces_of, chunks):
    """Per join across one seam."""
    seam = json.loads((chunks[k] / "record.json").read_text())["cell_planes"][1]
    labels = [boundary_labels(units[k]), boundary_labels(units[k + 1])]
    piece, side, edges, pairs, certainty, _, nodes = seam_field(chunks[k : k + 2], seam, labels)
    joins = joins_of(piece, side, edges, pairs, certainty)
    if not joins:
        return []
    a, b = edges["a"], edges["b"]
    across = (side[a] != side[b]) & (piece[a] >= 0) & (piece[b] >= 0)
    lower, upper = np.where(side[a] == 0, a, b)[across], np.where(side[a] == 0, b, a)[across]
    keys = piece[lower].astype(np.int64) * 2**31 + piece[upper]
    position = np.stack([nodes["x"], nodes["y"], nodes["z"]], axis=1).astype(np.float64) * NODE_SCALE
    um = next(iter(pieces_of[k].values()))["um"]
    z_mid = float(np.median(position[lower, 2])) * um
    cells = {}

    def cells_of(level, n):
        if (level, n) not in cells:
            cells[(level, n)] = near_seam(
                units[level], pieces_of[level][n], z_mid - SEAM_REACH_MM, z_mid + SEAM_REACH_MM
            )
        return cells[(level, n)]

    out = []
    for lo, hi, _ in joins:
        A, B = pieces_of[k].get(lo), pieces_of[k + 1].get(hi)
        if A is None or B is None:
            continue
        ta, tb = cells_of(k, lo), cells_of(k + 1, hi)
        if ta is None or tb is None:
            continue
        chosen = keys == lo * 2**31 + hi
        da, ia = ta[0].query(position[lower[chosen]])
        db, ib = tb[0].query(position[upper[chosen]])
        close = (da < 3 * A["stride"]) & (db < 3 * B["stride"])
        if close.sum() < 3:
            continue
        ia, ib = ia[close], ib[close]
        out.append(
            {
                "a": (k, lo),
                "b": (k + 1, hi),
                "edges": int(close.sum()),
                "columns": [float(np.median(ta[1][ia])), float(np.median(tb[1][ib]))],
                "rows": [float(np.median(ta[2][ia])), float(np.median(tb[2][ib]))],
            }
        )
    return out


def inner_joints(base, unit, k, pieces):
    """Joints between pieces of one unit, from the sheet edges the unwrapper found between them and could not merge."""
    path = stage_folder(base, "sheet") / f"between-{unit.name}.npz"
    if not path.exists():
        return []
    data = np.load(path)
    a, b = np.minimum(data["a"], data["b"]).astype(np.int64), np.maximum(data["a"], data["b"]).astype(np.int64)
    swapped = data["a"] > data["b"]
    pa = np.where(swapped[:, None], data["pb"], data["pa"]).astype(np.float64) * NODE_SCALE
    pb = np.where(swapped[:, None], data["pa"], data["pb"]).astype(np.float64) * NODE_SCALE
    keys = a * 2**31 + b
    order = np.argsort(keys, kind="stable")
    keys, pa, pb = keys[order], pa[order], pb[order]
    starts = np.r_[0, np.flatnonzero(np.diff(keys)) + 1, len(keys)]
    trees = {}

    def tree(n):
        if n not in trees:
            xyz, _ = sampler.load_surface(unit / f"piece-{n:04d}")
            i, j = np.nonzero(xyz[0] > 0)
            trees[n] = (cKDTree(xyz[:, i, j].T.astype(np.float64)), j, i)
        return trees[n]

    out = []
    for s, e in itertools.pairwise(starts):
        lo, hi = int(keys[s] // 2**31), int(keys[s] % 2**31)
        if e - s < MIN_INNER_EDGES or lo not in pieces or hi not in pieces:
            continue
        ta, tb = tree(lo), tree(hi)
        da, ia = ta[0].query(pa[s:e])
        db, ib = tb[0].query(pb[s:e])
        close = (da < 3 * pieces[lo]["stride"]) & (db < 3 * pieces[hi]["stride"])
        if close.sum() < MIN_INNER_EDGES:
            continue
        ia, ib = ia[close], ib[close]
        out.append(
            {
                "a": (k, lo),
                "b": (k, hi),
                "edges": int(close.sum()),
                "columns": [float(np.median(ta[1][ia])), float(np.median(tb[1][ib]))],
                "rows": [float(np.median(ta[2][ia])), float(np.median(tb[2][ib]))],
            }
        )
    return out


def joint_relations(units, joints, axis):
    """For every joint, whether its two pieces' normals agree where they meet."""
    ordered = sorted(joints, key=lambda j: (j["a"], j["b"]))
    endpoints = {}
    for i, joint in enumerate(ordered):
        for side, name in enumerate(("a", "b")):
            row = int(joint["rows"][side]) + MARGIN_CELLS
            column = int(joint["columns"][side]) + MARGIN_CELLS
            endpoints.setdefault(joint[name], []).append((i, side, row, column))
    normals = np.empty((len(ordered), 2, 3), dtype=np.float32)
    for (unit, piece), points in endpoints.items():
        xyz, meta = sampler.load_surface(units[unit] / f"piece-{piece:04d}")
        normal = surface_of(xyz, meta, axis, 0, 0)[1]
        _, height, width = normal.shape
        for i, side, row, column in points:
            normals[i, side] = normal[:, min(row, height - 1), min(column, width - 1)]
    return [(j["a"], j["b"], j["edges"], 1 if a @ b > 0 else -1) for j, (a, b) in zip(ordered, normals)]


def agreed_turns(areas, relations):
    """+1 or -1 per joined piece, so that pieces joined into one sheet read one face."""
    around = {}
    for a, b, edges, relation in relations:
        around.setdefault(a, []).append((edges, b, relation))
        around.setdefault(b, []).append((edges, a, relation))
    turns = {}
    for seed in sorted(around, key=lambda k: -areas[k]):
        if seed in turns:
            continue
        sheet = {seed: 1}
        heap = [(-e, t, r) for e, t, r in around[seed]]
        heapq.heapify(heap)
        while heap:
            _, k, sign = heapq.heappop(heap)
            if k not in sheet:
                sheet[k] = sign
                for e, t, r in around[k]:
                    if t not in sheet:
                        heapq.heappush(heap, (-e, t, sign * r))
        kept = 1 if sum(areas[k] * s for k, s in sheet.items()) >= 0 else -1
        turns |= {k: s * kept for k, s in sheet.items()}
    return turns


def offset_of(j, flat):
    """Where the second piece of a joint sits against the first, in mm of their upright frames."""
    A, B = flat[j["a"]], flat[j["b"]]
    ua = ((A["shape"][1] - 1 - j["columns"][0]) if A["flip_cols"] else j["columns"][0]) * A["side"]
    ub = ((B["shape"][1] - 1 - j["columns"][1]) if B["flip_cols"] else j["columns"][1]) * B["side"]
    va = ((A["shape"][0] - 1 - j["rows"][0]) if A["flip_rows"] else j["rows"][0]) * A["side"]
    vb = ((B["shape"][0] - 1 - j["rows"][1]) if B["flip_rows"] else j["rows"][1]) * B["side"]
    return ua - ub, va - vb


def mosaics(keys, flat, joints):
    """Groups of pieces with their places in mm, grown from the largest piece along the strongest joints."""
    around = {k: [] for k in keys}
    for j in joints:
        dx, dy = offset_of(j, flat)
        around[j["a"]].append((j["edges"], j["b"], dx, dy))
        around[j["b"]].append((j["edges"], j["a"], -dx, -dy))
    half_w, half_h = int(MOSAIC_SPAN_MM / OCCUPANCY_MM), int(MOSAIC_SPAN_MM / 8 / OCCUPANCY_MM)
    taken = np.zeros((2 * half_h, 2 * half_w), dtype=bool)
    group, place, out = {}, {}, []
    for seed in sorted(keys, key=lambda k: -flat[k]["area"]):
        if seed in group:
            continue
        taken[:] = False
        members = []

        def add(k, x, y):
            m = flat[k]["mask"]
            a, b = int(y / OCCUPANCY_MM) + half_h, int(x / OCCUPANCY_MM) + half_w
            if a < 0 or b < 0 or a + m.shape[0] >= taken.shape[0] or b + m.shape[1] >= taken.shape[1]:
                return False
            if members and taken[a : a + m.shape[0], b : b + m.shape[1]][m].mean() > ALLOWED_OVERLAP:
                return False
            taken[a : a + m.shape[0], b : b + m.shape[1]] |= m
            group[k], place[k] = len(out), (x, y)
            members.append(k)
            return True

        add(seed, 0.0, 0.0)
        heap = [(-e, t, place[seed][0] + dx, place[seed][1] + dy) for e, t, dx, dy in around[seed]]
        heapq.heapify(heap)
        while heap:
            _, k, x, y = heapq.heappop(heap)
            if k not in group and add(k, x, y):
                for e, t, dx, dy in around[k]:
                    if t not in group:
                        heapq.heappush(heap, (-e, t, x + dx, y + dy))
        out.append(members)
    return out, group, place


def covered(taken, y, x, mask, cells):
    """Share of a piece's cells that would land on sheet already placed."""
    return np.count_nonzero(taken[y : y + mask.shape[0], x : x + mask.shape[1]] & mask) / cells if cells else 0.0


def packed(items, masks, places, height_mm):
    """Groups of joined pieces held rigid, largest first, each slid along the roll to where it collides least."""
    scale = 1.0 / OCCUPANCY_MM
    margin = int(MOST_SHIFT_MM * scale) + 10
    W = int(places[:, 0].max() * scale) + 6000 + 2 * margin
    H = int(height_mm * scale) + 4 + max(m.shape[0] for m in masks)
    taken = np.zeros((H, W), dtype=bool)
    cells = [np.count_nonzero(mask) for mask in masks]
    groups = {}
    for n, p in enumerate(items):
        groups.setdefault(p["group"], []).append(n)

    def spot(n, dx=0):
        x = int(places[n, 0] * scale) + margin + dx
        return min(max(int(places[n, 1] * scale), 0), H - masks[n].shape[0] - 1), x

    loose = []
    for members in sorted(groups.values(), key=lambda m: -items[m[0]]["group_area"]):
        if len(members) < 2:
            loose += members
            continue
        best, least = 0, None
        for step in range(0, int(MOST_SHIFT_MM / SHIFT_STEP_MM), 8):
            for sign in (1, -1) if step else (1,):
                dx = sign * int(step * SHIFT_STEP_MM * scale)
                hit = 0.0
                for n in members:
                    y, x = spot(n, dx)
                    if x >= 0 and x + masks[n].shape[1] < W:
                        hit += covered(taken, y, x, masks[n], cells[n]) * items[n]["area"]
                if least is None or hit < least - 1e-6:
                    best, least = dx, hit
            if least == 0.0:
                break
        for n in sorted(members, key=lambda n: -items[n]["area"]):
            places[n, 0] += best / scale
            y, x = spot(n)
            if x >= 0 and x + masks[n].shape[1] < W and covered(taken, y, x, masks[n], cells[n]) <= ALLOWED_OVERLAP:
                taken[y : y + masks[n].shape[0], x : x + masks[n].shape[1]] |= masks[n]
            else:
                loose.append(n)
    moved = 0
    kept = len(items) - len(loose)
    for n in sorted(loose, key=lambda n: -items[n]["area"]):
        y, x0 = spot(n)
        for step in range(4 * W):
            found = None
            for sign in (1, -1) if step else (1,):
                x = x0 + sign * int(step * SHIFT_STEP_MM * scale)
                if x >= 0 and x + masks[n].shape[1] < W and covered(taken, y, x, masks[n], cells[n]) <= ALLOWED_OVERLAP:
                    found = x
                    break
            if found is not None:
                taken[y : y + masks[n].shape[0], found : found + masks[n].shape[1]] |= masks[n]
                shift = (found - x0) / scale
                places[n, 0] += shift
                items[n]["shifted_mm"] = shift
                moved += abs(shift) > 2 * SHIFT_STEP_MM
                break
    print("placed by joints", kept, "of", len(items), "moved to free sheet", moved, flush=True)
    return moved


def run(sample, volume):
    start = time.time()
    base = ROUTE_OUTPUTS / sample / volume
    axis = axis_points(sample, volume)
    chunks = sorted(stage_folder(base, "nodes").glob("rows-*"))
    units = [base / "pieces" / c.name for c in chunks]
    if axis is None:
        axis = axis_of_pieces(units)
    pieces_of = []
    for unit in units:
        rows = [r for r in json.loads((unit / "record.json").read_text())["pieces"] if r.get("written")]
        pieces_of.append({r["piece"]: summary(unit, r, axis) for r in rows})
        print(unit.name, len(rows), round(time.time() - start), flush=True)
    keys = [(k, n) for k, group in enumerate(pieces_of) for n in group]
    flat = {(k, n): pieces_of[k][n] for k, n in keys}
    out = stage_folder(base, "sheet")
    out.mkdir(parents=True, exist_ok=True)
    cache = out / "joints.json"
    if cache.exists():
        joints = json.loads(cache.read_text())
        for j in joints:
            j["a"], j["b"] = tuple(j["a"]), tuple(j["b"])
    else:
        joints = []
        for k in range(len(units) - 1):
            found = seam_joints(units, k, pieces_of, chunks)
            joints += found
            print("seam", k, len(found), round(time.time() - start), flush=True)
        cache.write_text(json.dumps(joints))
    inner = []
    for k, unit in enumerate(units):
        inner += inner_joints(base, unit, k, pieces_of[k])
    print("seam joints", len(joints), "joints inside units", len(inner), round(time.time() - start), flush=True)
    joints = joints + inner
    chirality = scan_chirality(units)
    turns = agreed_turns({k: p["area"] for k, p in flat.items()}, joint_relations(units, joints, axis))
    print(
        "pieces turned to read their sheet's face",
        sum(t < 0 for t in turns.values()),
        "of",
        len(turns),
        "joined,",
        round(sum(flat[k]["area"] for k, t in turns.items() if t < 0) / 100),
        "cm2",
        flush=True,
    )
    for k, p in flat.items():
        p["turn"] = turns.get(k, 1)
        p["outward"] = p["outward"] != (p["turn"] < 0)
        frame(p, chirality)
    groups, group_of, local = mosaics(keys, flat, joints)
    area = np.array([sum(flat[k]["area"] for k in m) for m in groups])
    top = np.sort(area)[::-1][:6] / 100
    print(
        "mosaics",
        len(groups),
        "with two or more pieces",
        sum(len(m) > 1 for m in groups),
        "largest cm2",
        top.round(1).tolist(),
        "in mosaics of 4 cm2 or more",
        round(float(area[area >= 400].sum() / 100)),
        "of",
        round(float(area.sum() / 100)),
        round(time.time() - start),
        flush=True,
    )
    radius = np.array(
        [np.average([flat[k]["radius"] for k in m], weights=[flat[k]["area"] for k in m]) for m in groups]
    )
    order = np.argsort(radius)
    z_top, z_bottom = max(p["z_high"] for p in flat.values()), min(p["z_low"] for p in flat.values())
    arc = np.zeros(len(groups))
    arc[order] = np.cumsum(area[order]) / (z_top - z_bottom) / SHEET_SHARE
    roll_mm = float(arc.max())
    items, places = [], np.zeros((len(keys), 2))
    for n, k in enumerate(keys):
        p = flat[k]
        g = group_of[k]
        members = groups[g]
        seed = members[0]
        height = (z_top - flat[seed]["z_high"]) if chirality > 0 else (flat[seed]["z_low"] - z_bottom)
        places[n] = (roll_mm - arc[g] + local[k][0], height + local[k][1])
        items.append(
            {
                "unit": p["unit"],
                "piece": p["piece"],
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
            }
        )
    places[:, 0] -= places[:, 0].min()
    places[:, 1] -= places[:, 1].min()
    joined = places.copy()
    moved = packed(items, [flat[k]["mask"] for k in keys], places, float(places[:, 1].max()) + 120)
    places[:, 0] -= places[:, 0].min()
    for n, item in enumerate(items):
        item["x_mm"], item["y_mm"], item["moved_from_joint_mm"] = (
            float(places[n, 0]),
            float(places[n, 1]),
            float(places[n, 0] - joined[n, 0]),
        )
    record = {
        "sample": sample,
        "volume": volume,
        "chirality": chirality,
        "joints": len(joints),
        "pieces": len(items),
        "mosaics": len(groups),
        "largest_mosaics_cm2": top.tolist(),
        "mosaic_cm2_over_4": float(area[area >= 400].sum() / 100),
        "moved": moved,
        "width_mm": float(places[:, 0].max()),
        "seconds": round(time.time() - start),
    }
    (out / "layout.json").write_text(json.dumps({"record": record, "pieces": items}))
    print(record, flush=True)


def run_band(sample, volume):
    """A scan unwrapped as one band has no seams."""
    base = ROUTE_OUTPUTS / sample / volume
    unit = min((base / "pieces").glob("band-*"))
    record = json.loads((unit / "record.json").read_text())
    pieces = loaded(unit)
    centre = scroll_centre(pieces)
    placed(pieces, centre, record["chirality"])
    end = max(p["arc"] for p in pieces) + 40
    top = max(float(p["xyz"][2][p["good"]].max()) * p["meta"]["voxel_um"] / 1000 for p in pieces)
    low = min(float(p["xyz"][2][p["good"]].min()) * p["meta"]["voxel_um"] / 1000 for p in pieces)
    items, places = [], np.zeros((len(pieces), 2))
    for n, p in enumerate(pieces):
        h, w = p["good"].shape
        side = p["side"]
        flip_rows, flip_cols = bool(p["up"][1] > 0), bool(p["right"][0] < 0)
        anchor_x = ((w - 1 - p["anchor"][0]) if flip_cols else p["anchor"][0]) * side
        um = p["meta"]["voxel_um"] / 1000
        z_high, z_low = float(p["xyz"][2][p["good"]].max()) * um, float(p["xyz"][2][p["good"]].min()) * um
        places[n] = (end - p["arc"] - anchor_x, (top - z_high) if record["chirality"] > 0 else (z_low - low))
        items.append(
            {
                "unit": unit.name,
                "piece": p["row"]["piece"],
                "area": p["row"]["area_mm2"],
                "group": n,
                "group_area": p["row"]["area_mm2"],
                "width_mm": w * side,
                "height_mm": h * side,
                "flip_rows": flip_rows,
                "flip_cols": flip_cols,
                "um": um,
                "shape": [h, w],
                "stride": p["meta"]["stride"],
            }
        )
    places[:, 0] -= places[:, 0].min()
    joined = places.copy()
    masks = []
    for p, item in zip(pieces, items):
        step = max(1, int(round(OCCUPANCY_MM / p["side"])))
        m = p["good"][::step, ::step]
        masks.append(m[::-1][:, ::1] if item["flip_rows"] else m)
        masks[-1] = masks[-1][:, ::-1] if item["flip_cols"] else masks[-1]
    moved = packed(items, masks, places, float(places[:, 1].max()) + 60)
    places[:, 0] -= places[:, 0].min()
    for n, item in enumerate(items):
        item["x_mm"], item["y_mm"], item["moved_from_joint_mm"] = (
            float(places[n, 0]),
            float(places[n, 1]),
            float(places[n, 0] - joined[n, 0]),
        )
    out = stage_folder(base, "sheet")
    out.mkdir(parents=True, exist_ok=True)
    summary_row = {
        "sample": sample,
        "volume": volume,
        "chirality": record["chirality"],
        "joints": 0,
        "pieces": len(items),
        "moved": moved,
        "width_mm": float(places[:, 0].max()),
    }
    (out / "layout.json").write_text(json.dumps({"record": summary_row, "pieces": items}))
    print(summary_row, flush=True)


if __name__ == "__main__":
    (run_band if len(sys.argv) > 3 else run)(sys.argv[1], sys.argv[2])
