"""Sheet detection stage: CT bricks to sheet nodes with normals."""

import json
import os
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from http.client import HTTPException
from pathlib import Path
from urllib.request import urlopen

import numpy as np

from runner import stage_queue
from scans.catalog.scan_catalog import load, survey_volumes
from scans.catalog.scan_choice import chosen_scans, sheet_level
from scans.transfer.fetch_volume import fetch_level
from storage.brick_store import UNKNOWN, BrickStore
from storage.data_roots import NODE_ROUTE, RAW_SCANS, UNWRAP_OUTPUTS, scan_folder, stage_folder
from storage.node_store import pack_chunk
from unwrapping.detection import cpu_nodes, gpu_nodes
from unwrapping.detection.kernel_bindings import CELL, GPU_SOURCES

HERE = Path(__file__).parent
METHOD = stage_queue.method_of(
    *(HERE / source for source in GPU_SOURCES), HERE / "gpu_nodes.py", HERE / "kernel_bindings.py"
)
SAME_OUTPUT = (HERE / "same_output_methods.txt").read_text().split()
# brick rows a unit solves in one piece; fewer need less memory, and a scan keeps the number its units were cut to
CHUNK_ROWS = int(os.environ.get("HERCULEX_UNIT_ROWS", "12"))
OLD_UNIT_ROWS = 12  # units recorded before records named their rows were all cut to this
BAND_ROWS = CHUNK_ROWS
PER_ROW = 128 // CELL
EDGE = 2
SEAM = PER_ROW // 2
CONNECTIONS = int(os.environ.get("HERCULEX_FETCH_CONNECTIONS", "16"))
PASSES = 3
FLOOR_GIB = 30
NO_ROOM = 3


def chunks(height):
    """Brick rows of each chunk and the cell planes it owns."""
    rows, top = -(-height // 128), -(-height // CELL) - EDGE
    planned, first = [], 0
    while True:
        last = min(rows, first + CHUNK_ROWS)
        low = EDGE if first == 0 else first * PER_ROW + SEAM
        high = top if last == rows else (last - 1) * PER_ROW + SEAM
        if min(high, top) > min(low, top):
            planned.append((first, last, min(low, top), min(high, top)))
        if last == rows:
            return planned
        first = last - 1


def band_planes(first, last, height):
    """Cell planes a band of brick rows owns, kept clear of the scan's ends."""
    top = -(-height // CELL) - EDGE
    return max(first * PER_ROW + 2, EDGE), min(last * PER_ROW - 2, top)


def central_row(row, level):
    """Brick row at the middle of the papyrus, from the atlas when it holds this very scan."""
    for folder in sorted(RAW_SCANS.glob(f"{row['sample']}/atlas_*/L[35]")):
        store = BrickStore(folder)
        if row["volume"] not in (store.source or ""):
            continue
        present = (store.offsets >= 0).sum(axis=(1, 2)).astype(float)
        return float(np.searchsorted(np.cumsum(present) / present.sum(), 0.5) + 0.5) * 2 ** (
            int(folder.name[1:]) - level
        )
    return None


def level_shape(row, level):
    path, root = row["products"]["ome-zarr"], row["roots"]["ome-zarr"]
    return json.load(urlopen(f"{root}/{path.strip('/')}/{level}/.zarray", timeout=60))["shape"]


def other_sized_units(out):
    """Finished units in a nodes folder that were cut to another number of brick rows than CHUNK_ROWS."""
    return sorted(
        unit.name
        for unit in [*out.glob("rows-*"), *out.glob("band-*")]
        if not unit.name.endswith(".building")
        and (unit / stage_queue.RECORD).is_file()
        and stage_queue.record_of(unit).get("unit_rows", OLD_UNIT_ROWS) != CHUNK_ROWS
    )


def jobs(samples, band, volume=None):
    """Every chunk still to do, in the prizes' order."""
    selected = (
        chosen_scans(samples)
        if volume is None
        else [
            row
            for row in survey_volumes(load())
            if row["sample"] in samples
            and row["volume"] == volume
            and row["products"].get("ome-zarr")
            and row["voxel_um"] is not None
        ]
    )
    missing = set(samples) - {row["sample"] for row in selected}
    if missing:
        raise ValueError(f"no survey scan in the local catalog for {', '.join(sorted(missing))}")
    for row in selected:
        level = sheet_level(row["voxel_um"])
        shape = level_shape(row, level)
        out = stage_folder(scan_folder(row["sample"], row["volume"], route=NODE_ROUTE), "nodes")
        out.mkdir(parents=True, exist_ok=True)
        others = other_sized_units(out)
        if others:
            raise ValueError(
                f"{row['sample']} {row['volume']}: {out} holds units not cut to HERCULEX_UNIT_ROWS={CHUNK_ROWS} "
                f"brick rows ({', '.join(others[:3])}), and new units would overlap them. Run with the number they "
                "were cut to, or remove this scan's nodes and pieces first."
            )
        if band:
            rows = -(-shape[0] // 128)
            first = max(0, int((central_row(row, level) or rows / 2) - BAND_ROWS / 2))
            last = min(rows, first + BAND_ROWS)
            planned = [(first, last, *band_planes(first, last, shape[0]))]
        else:
            planned = chunks(shape[0])
        for first, last, low, high in planned:
            folder = out / f"{'band' if band else 'rows'}-{first:04d}-{last:04d}"
            if not stage_queue.done(folder, METHOD, SAME_OUTPUT):
                yield {
                    "row": row,
                    "level": level,
                    "shape": shape,
                    "rows": (first, last),
                    "planes": (low, high),
                    "out": folder,
                }


def work_folder(job):
    row, level, (first, last) = job["row"], job["level"], job["rows"]
    return UNWRAP_OUTPUTS / "sheet_detector_work" / row["sample"] / row["volume"] / f"L{level}-{first:04d}-{last:04d}"


def fetch_chunk(job):
    """A brick store resumes, so a restart keeps the bricks a killed run already fetched."""
    row, (first, last) = job["row"], job["rows"]
    store = fetch_level(
        row["products"]["ome-zarr"],
        job["level"],
        work_folder(job),
        connections=CONNECTIONS,
        root=row["roots"]["ome-zarr"],
        z_bricks=(first, last),
        list_bricks=os.environ.get("HERCULEX_LIST_CT_BRICKS") == "1",
    )
    unknown = int((store.offsets[first:last] == UNKNOWN).sum())
    if unknown:
        raise OSError(f"{unknown} bricks still unknown")
    return work_folder(job)


def detect_store(level, out, record, cell_planes=None, device="gpu"):
    """Packed nodes of a brick store's cell planes, with their record written last."""
    start = time.time()
    store = BrickStore(level)
    Z, Y, X = store.shape
    grid = [-(-Z // CELL), -(-Y // CELL), -(-X // CELL)]
    planes = cell_planes or (0, grid[0])
    with stage_queue.building(out) as part:
        with open(part / "nodes.bin", "wb") as nodes, open(part / "cells.bin", "wb") as cells:
            total, gpu = (
                gpu_nodes.detect(store, planes, nodes, cells)
                if device == "gpu"
                else (cpu_nodes.detect(store, planes, nodes, cells), 0.0)
            )
        pack_chunk(part)
        record = record | {
            "shape": list(store.shape),
            "cell": CELL,
            "grid": grid,
            "cell_planes": list(planes),
            "nodes": total,
            "gpu_seconds": round(gpu, 1),
            "seconds": round(time.time() - start, 1),
            "method": METHOD,
        }
        stage_queue.finish(part, record)
    return record


def detect_chunk(job, fetched):
    """Nodes of a fetched chunk."""
    row = job["row"]
    try:
        record = detect_store(
            fetched.result(),
            job["out"],
            {
                "sample": row["sample"],
                "volume": row["volume"],
                "scan_voxel_um": row["voxel_um"],
                "level": job["level"],
                "voxel_um": row["voxel_um"] * 2 ** job["level"],
                "rows": list(job["rows"]),
                "unit_rows": CHUNK_ROWS,
            },
            job["planes"],
        )
    except (OSError, ValueError, RuntimeError, HTTPException) as error:
        print("failed", row["sample"], job["rows"], str(error)[:160], flush=True)
        return 1
    shutil.rmtree(work_folder(job), ignore_errors=True)
    print(row["sample"], job["out"].name, record["nodes"], "nodes", record["seconds"], "s", flush=True)
    return 0


def one_pass(samples, band, volume=None):
    """Fetch each chunk while the one before it is detected."""
    failed, pending = 0, None
    with ThreadPoolExecutor(max_workers=1) as fetcher:
        for job in jobs(samples, band, volume):
            if shutil.disk_usage(NODE_ROUTE).free / 2**30 < FLOOR_GIB:
                print("stopped: under", FLOOR_GIB, "GiB free on the data disc", flush=True)
                if pending:
                    detect_chunk(*pending)
                sys.exit(NO_ROOM)
            following = (job, fetcher.submit(fetch_chunk, job))
            if pending:
                failed += detect_chunk(*pending)
            pending = following
        if pending:
            failed += detect_chunk(*pending)
    return failed


def run(samples=(), band=False, *, volume=None):
    gpu_nodes.gpu_library()  # Metal on a Mac, CUDA elsewhere: without it the stage stops before fetching any chunk
    with stage_queue.stage_lock(NODE_ROUTE / "sheet-detector.lock"):
        for _ in range(PASSES):
            if not (failed := one_pass(set(samples), band, volume)):
                return
    raise RuntimeError(f"sheet detection failed on {failed} chunk(s). Rerun the same command to resume.")


if __name__ == "__main__":
    if sys.argv[1:] in (["-h"], ["--help"]):
        print("Usage: ./run.sh unwrapping.detection.detect_nodes {run,band} [SCROLL ...]")
    else:
        run(sys.argv[2:], band=sys.argv[1] == "band")
