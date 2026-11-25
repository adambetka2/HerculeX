"""Where a scan's CT volume lives, cached in arrays.sqlite. The chunk store reads nothing else of a scan.

The open-data bucket is listed first. Scans it does not hold (the 7.91 µm DLS scans of PHerc1667 and PHercParis4 are
only on data.aws.ash2txt.org) are found through the HerculeX catalogue, whose access roots name the server.
"""

import json

from scans.catalog.scan_catalog import BUCKET
from scans.chunks import array_registry, bucket
from storage.data_roots import SCAN_CATALOG

CATALOG = SCAN_CATALOG / "metadata.json"
KINDS = ("ct",)
SERVERS = {"s3://vesuvius-challenge-open-data": BUCKET}


def base_name(key):
    return key.rstrip("/").split("/")[-1]


def pick_ct(folders):
    """The plain masked zarr v2 volume. Since November 2026 the bucket also holds a zarr v3 copy of each scan
    (`...-masked-fast-q1-c64-v0.3.2.vcd3d.zarr`), which sorts first but which the chunk store cannot read."""
    readable = [f for f in folders if ".vcd3d." not in f] or folders
    masked = [f for f in readable if f.rstrip("/").endswith("-masked.zarr")]
    masked = masked or [f for f in readable if "-masked" in f]
    return (masked or readable)[0]


def from_listing(sample, scan):
    _, folders = bucket.list_prefix(f"{sample}/volumes/")
    ct = sorted(f for f in folders if base_name(f).startswith(f"{scan}-") and f.endswith(".zarr/"))
    return {"ct": (f"{BUCKET}/{pick_ct(ct).rstrip('/')}", 0, ct)} if ct else {}


def from_catalog(sample, scan):
    """The scan's OME-Zarr origins in the catalogue, on the server of each access root (the bucket first)."""
    if not CATALOG.exists():
        return {}
    volume = json.loads(CATALOG.read_text())["samples"].get(sample, {}).get("volumes", {}).get(scan, {})
    roots = [
        f"{SERVERS.get(server['url'], server['url']).rstrip('/')}/{origin['path'].strip('/')}"
        for data in volume.get("data", [])
        if data.get("type") == "ome-zarr"
        for origin in data.get("origins", [])
        for server in origin.get("access_roots", [])
    ]
    roots = sorted(r for r in roots if r.endswith(".zarr"))
    roots.sort(key=lambda r: not r.startswith(BUCKET))
    return {"ct": (pick_ct(roots), 0, roots)} if roots else {}


def resolve(sample, scan, refresh=False):
    """kind -> (url, level) for the kinds the chunk store reads: 'ct', the scan's (masked) OME-Zarr root at level 0."""
    if not refresh:
        cached = {k: v for k, v in array_registry.resolved(sample, scan).items() if k in KINDS}
        if cached and not any(".vcd3d." in url for url, _ in cached.values()):
            return cached
    answers, source = from_listing(sample, scan), "listing"
    if not answers:
        answers, source = from_catalog(sample, scan), "catalog"
    if not answers:
        raise KeyError(f"no CT volume published for {sample} scan {scan}")
    array_registry.save_resolved(sample, scan, answers, source)
    return {kind: (url, level) for kind, (url, level, _) in answers.items()}


def array_root(sample, scan, kind="ct"):
    """(zarr group root URL, default level) of one kind."""
    answers = resolve(sample, scan)
    if kind not in answers:
        raise KeyError(f"the chunk store reads only {', '.join(KINDS)} of a scan, not '{kind}'")
    return answers[kind]
