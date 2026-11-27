"""The Studio ink model, read wherever PyTorch runs: any checkpoint of its network, kept as a .safetensors file.

./run.sh ink.model.studio_reader fetch          download Studio's public weights and code, and list the weights as model `studio`
./run.sh ink.model.studio_reader check FILE     say whether FILE holds the network's tensors
./run.sh ink.model.models add NAME TITLE FILE   list your own Studio checkpoint (a .safetensors file) as a model

Studio (https://github.com/priyankaranke/vesuvius-studio-multiscroll-cpt, code MIT; weights CC BY-NC 4.0, the licence of
the scan data) is a 3-D residual U-Net over 16 planes of the sheet, planes 8 to 23 of a 31-plane stack centred on the
mesh, in this pipeline's order (the model's own notes list them as 23 down to 8 of a Volume Cartographer render, whose
planes run the other way). A 128-pixel window is read with the planes clipped to 0..200 and divided by 255, and its
central 96 pixels are kept. Windows every 64 pixels are blended by a Hann window (the tiling of the model's submission,
smooth and without tile seams), or with HERCULEX_STUDIO_STRIDE=96 every 96 with no overlap (its default tiling, 2.25
times less work, with a visible grid of tile seams). Pixels with no CT in any plane are 0. This follows the repository's
own inference.py, which it matches to bf16 rounding. The network's code is fetched at a pinned commit and checked by
sha256 the first time a Studio model is added or read.
"""

import hashlib
import importlib.util
import os
import sys
import urllib.request
from itertools import islice
from pathlib import Path

import numpy as np

from storage.data_roots import EXTERNAL_TOOLS, INK_WEIGHTS

COMMIT = "235e3be8a06749344862c89a1e52c81d90b15e0e"
REPOSITORY = "priyankaranke/vesuvius-studio-multiscroll-cpt"
WEIGHTS_URL = f"https://github.com/{REPOSITORY}/releases/download/studio-softloss-20261023/studio-softloss-20261023-ema.safetensors"
WEIGHTS_SHA256 = "96b326286e38de99047f7cab36170c53749a18d5f43e5fde0c7c20f2980b0e90"
CODE_URL = f"https://raw.githubusercontent.com/{REPOSITORY}/{COMMIT}/studio-multiscroll-cpt-20261023/hf/model.py"
CODE_SHA256 = "efbb390b6208c480fa8e8681137690b7df897c3e406956b9e804fa66645316d7"
CODE = EXTERNAL_TOOLS / "studio-multiscroll-cpt" / "model.py"
PUBLIC_NAME, PUBLIC_TITLE = "studio", "Studio multi-scroll (soft loss)"
PUBLIC_WEIGHTS = INK_WEIGHTS / PUBLIC_NAME / "studio.safetensors"
SUFFIX = ".safetensors"
PLANE_COUNT, FIRST_PLANE, STACK_LAYERS = 16, 8, 31  # planes 8 to 23 of the 31 of a stack centred on the mesh
WINDOW, KEEP, EDGE = 128, 96, 16
# 64 is its submission's smooth tiling, 96 its no-overlap one
STRIDE = int(os.environ.get("HERCULEX_STUDIO_STRIDE", "64"))
BATCH = int(os.environ.get("HERCULEX_STUDIO_BATCH", "4"))


def is_studio(path):
    """Whether a model file is read as Studio: by its suffix, as Reader v3's checkpoints are .pth files."""
    return Path(path).suffix == SUFFIX


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1 << 22), b""):
            digest.update(block)
    return digest.hexdigest()


def download(url, path, digest):
    """A file fetched once, written whole and checked against its sha256."""
    if path.is_file() and sha256(path) == digest:
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_name(path.name + ".part")
    print(f"fetching {url}", flush=True)
    urllib.request.urlretrieve(url, part)
    if sha256(part) != digest:
        part.unlink()
        raise SystemExit(f"{url} does not match its sha256; try again")
    part.replace(path)
    return path


def network():
    """The pinned network class, from its checked source file."""
    download(CODE_URL, CODE, CODE_SHA256)
    spec = importlib.util.spec_from_file_location("studio_model", CODE)
    module = importlib.util.module_from_spec(spec)
    sys.modules["studio_model"] = module
    spec.loader.exec_module(module)
    return module.DeepResidualUNet


def check(path):
    """Raises SystemExit unless the file holds exactly the network's tensors, with their shapes."""
    import torch
    from safetensors import safe_open

    with torch.device("meta"):
        expected = {name: tuple(t.shape) for name, t in network()(deeper=False).state_dict().items()}
    with safe_open(str(path), "pt") as file:
        found = {name: tuple(file.get_slice(name).get_shape()) for name in file.keys()}  # noqa: SIM118
    missing, extra = sorted(expected.keys() - found.keys()), sorted(found.keys() - expected.keys())
    wrong = sorted(name for name in expected.keys() & found.keys() if expected[name] != found[name])
    if missing or extra or wrong:
        raise SystemExit(
            f"{path} is not a Studio checkpoint: {len(missing)} tensors missing, {len(extra)} unexpected, "
            f"{len(wrong)} of the wrong shape (first: {(missing + extra + wrong)[0]})"
        )


def fetch():
    """The network's code and the public weights, the weights listed as model `studio`."""
    from ink.model import models

    download(CODE_URL, CODE, CODE_SHA256)
    download(WEIGHTS_URL, PUBLIC_WEIGHTS, WEIGHTS_SHA256)
    models.list_model(PUBLIC_NAME, PUBLIC_TITLE, PUBLIC_WEIGHTS)
    print(f"Studio ready: {PUBLIC_WEIGHTS}", flush=True)


def planes_of(stack):
    """The 16 planes the model reads from a 31-plane stack, in this pipeline's depth order (checked on hand-drawn ink)."""
    return np.ascontiguousarray(stack[FIRST_PLANE : FIRST_PLANE + PLANE_COUNT])


def window_at(volume, y, x):
    """A 128-pixel window with its corner at (y, x), reflected where it runs past the stack."""
    h, w = volume.shape[-2:]
    tile = volume[:, max(0, y) : min(h, y + WINDOW), max(0, x) : min(w, x + WINDOW)]
    rows, cols = (max(0, -y), max(0, y + WINDOW - h)), (max(0, -x), max(0, x + WINDOW - w))
    return np.pad(tile, ((0, 0), rows, cols), mode="reflect") if any(rows) or any(cols) else tile


def hann():
    w = 0.5 - 0.5 * np.cos(2 * np.pi * (np.arange(KEEP) + 0.5) / KEEP)
    return (w[:, None] * w[None, :]).astype(np.float32)


class StudioReader:
    def __init__(self, weights, device=None):
        import torch
        from safetensors.torch import load_file

        from runner.lane import torch_device

        self.torch, self.device = torch, torch.device(device or torch_device())
        model = network()(deeper=False)
        model.load_state_dict(load_file(str(weights), device="cpu"), strict=True)
        self.model = model.eval().requires_grad_(False).to(self.device)
        self.half = self.device.type == "cuda" and os.environ.get("HERCULEX_PRECISION", "bfloat16") != "float32"
        self.kernel = hann()

    def read(self, stack):
        """Probability of ink of a (16, height, width) uint8 stack."""
        torch = self.torch
        if stack.ndim != 3 or stack.shape[0] != PLANE_COUNT or stack.dtype != np.uint8:
            raise ValueError(f"expected a uint8 stack of {PLANE_COUNT} planes")
        h, w = stack.shape[1:]
        total, weight = np.zeros((h, w), np.float32), np.zeros((h, w), np.float32)
        ys = sorted({*range(0, h, STRIDE), max(h - KEEP, 0)})
        xs = sorted({*range(0, w, STRIDE), max(w - KEEP, 0)})
        covered = stack.any(axis=0)
        spots = (
            (y, x)
            for y in ys
            for x in xs
            if covered[max(y - EDGE, 0) : y - EDGE + WINDOW, max(x - EDGE, 0) : x - EDGE + WINDOW].any()
        )
        with torch.inference_mode():
            while batch := list(islice(spots, BATCH)):
                raw = np.stack([window_at(stack, y - EDGE, x - EDGE) for y, x in batch])
                image = torch.from_numpy(raw[:, None]).to(self.device, torch.float32).clamp_(0, 200).div_(255)
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.half):
                    logits = self.model(image)
                kept = logits[:, 0, EDGE : EDGE + KEEP, EDGE : EDGE + KEEP].float().sigmoid().cpu().numpy()
                for (y, x), p in zip(batch, kept):
                    a, b = min(KEEP, h - y), min(KEEP, w - x)
                    total[y : y + a, x : x + b] += p[:a, :b] * self.kernel[:a, :b]
                    weight[y : y + a, x : x + b] += self.kernel[:a, :b]
        result = np.where(weight > 0, total / np.maximum(weight, 1e-12), 0).astype(np.float32)
        result[~covered] = 0
        return result

    def maps(self, stacks):
        """TileReader.maps: each stack, prepared or raw, read alone; a failure is returned in place of its map."""
        for stack, _ in stacks:
            try:
                yield self.read(np.asarray(stack.values if hasattr(stack, "values") else stack))
            except Exception as error:
                if "CUDA" in str(error):
                    raise  # a failed device fails every later window too; stop so a fresh process can resume
                yield error


def main():
    command = sys.argv[1:2]
    if command == ["fetch"]:
        fetch()
    elif command == ["check"] and len(sys.argv) == 3:
        check(sys.argv[2])
        print(f"{sys.argv[2]}: a Studio checkpoint")
    else:
        sys.exit(__doc__)


if __name__ == "__main__":
    main()
