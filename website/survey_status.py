"""Measured coverage and reading products of every surveyed scan."""

import json
from datetime import UTC, datetime

from sampling.sheet_images import mosaics_of
from storage.data_roots import PRODUCTION_FACES, ROUTE_OUTPUTS, stage_folder
from website.sheet_tiles import written_on_far_side


def snapshot(sample, volume, *, use_review=True, root=ROUTE_OUTPUTS):
    base = root / sample / volume
    sheet = stage_folder(base, "sheet")
    layout = json.loads((sheet / "layout.json").read_text())
    kept = sheet / "review.json"
    if use_review and kept.is_file() and kept.stat().st_mtime >= (sheet / "layout.json").stat().st_mtime:
        return json.loads(kept.read_text())["coverage"]
    pieces = layout["pieces"]
    groups = mosaics_of(pieces)
    areas = sorted((sum(p["area"] for p in group) / 100 for group in groups), reverse=True)
    records = [json.loads(path.read_text()) for path in sorted((base / "pieces").glob("rows-*/record.json"))]
    if not records:
        records = [json.loads(path.read_text()) for path in sorted((base / "pieces").glob("band-*/record.json"))]
    image_root = stage_folder(base, "faces") / PRODUCTION_FACES  # the faces of the model that reads, stock or not
    raw, ink, ink_bytes, raw_bytes, rendered_pixels = [], [], 0, 0, 0
    for p in pieces:
        stem = image_root / p["unit"] / f"piece-{p['piece']:04d}"
        raw_path, ink_path = stem.with_name(stem.name + "-layer0.png"), stem.with_name(stem.name + "-taught.png")
        if raw_path.exists():
            raw.append(p)
            raw_bytes += raw_path.stat().st_size
            rendered_pixels += ((p["shape"][0] + 3) * p["stride"] + 1) * ((p["shape"][1] + 3) * p["stride"] + 1)
        if ink_path.exists():
            ink.append(p)
            ink_bytes += ink_path.stat().st_size
    return {
        "sample": sample,
        "volume": volume,
        "updated": datetime.now(UTC).isoformat(timespec="seconds"),
        "voxel_um": records[0]["voxel_um"] if records else None,
        "geometry_level": records[0]["level"] if records else None,
        "unwrapped_cm2": sum(r["area_cm2"] for r in records),
        "laid_cm2": sum(p["area"] for p in pieces) / 100,
        "pieces": len(pieces),
        "logical_mosaics": layout["record"].get("mosaics", len(pieces)),
        "display_mosaics": len(groups),
        "joints": layout["record"]["joints"],
        "relocated_pieces": layout["record"]["moved"],
        "largest_cm2": areas[0] if areas else 0,
        "large_mosaics": sum(a >= 4 for a in areas),
        "large_cm2": sum(a for a in areas if a >= 4),
        "raw_pieces": len(raw),
        "ink_pieces": len(ink),
        "raw_cm2": sum(p["area"] for p in raw) / 100,
        "ink_cm2": sum(p["area"] for p in ink) / 100,
        "raw_png_bytes": raw_bytes,
        "ink_png_bytes": ink_bytes,
        "raw_rendered_pixels": rendered_pixels,
        "sides_checked": sum((image_root / p["unit"] / f"piece-{p['piece']:04d}-side.json").exists() for p in pieces),
        "written_on_far_side": sum(written_on_far_side(image_root, p) for p in pieces),
        "turned_to_their_sheet": sum(p.get("turn", 1) < 0 for p in pieces),
        "layout_pixels": sum(
            ((p["shape"][0] + 3) * p["stride"] + 1) * ((p["shape"][1] + 3) * p["stride"] + 1) for p in pieces
        ),
        "caveat": "Areas follow the current mesh layout. Global scroll order and wrong-wrap rate are unverified; ink maps are predictions.",
    }


if __name__ == "__main__":
    import sys

    print(json.dumps(snapshot(*sys.argv[1:3]), indent=1))
