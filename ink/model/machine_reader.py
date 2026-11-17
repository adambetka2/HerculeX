"""The ink reader this machine runs, kept out of tile_reader.py, whose bytes are part of the Mac read digest."""

from ink.model.portable_reader import PortableReader
from ink.model.tile_reader import TileReader
from runner.compiled_kernels import APPLE


def reader(weights):
    """Metal and the Neural Engine on a Mac, else the same arithmetic in PyTorch on a CUDA GPU, never on the CPU."""
    if APPLE:
        return TileReader(weights)
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("off macOS the ink reader needs a CUDA GPU, and PyTorch sees none")
    return PortableReader(weights, "cuda")
