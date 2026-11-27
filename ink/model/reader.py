"""The ink reader of this machine's lane, and the files that decide what it reads, for the read's digest."""

from pathlib import Path

from ink.model import studio_reader
from runner.lane import APPLE

HERE = Path(__file__).parent
SHARED = (HERE / "tiles.py", HERE / "tile_filling.c")
if APPLE:
    LANE_SOURCES = tuple(
        HERE / "reader_apple" / name for name in ("tile_reader.py", "depth_stem.metal", "neural_engine.py")
    )
else:
    LANE_SOURCES = (HERE / "reader_cuda" / "tile_reader.py",)
SOURCES = (*LANE_SOURCES, *SHARED)


def sources_of(weights):
    """The files whose bytes make up a read's digest: Studio's reader for its checkpoints, the lane's for Reader v3's."""
    return (Path(studio_reader.__file__),) if studio_reader.is_studio(weights) else SOURCES


def lane_reader(weights):
    """Studio's PyTorch reader for a .safetensors checkpoint; else Metal and the Neural Engine on the Apple lane, PyTorch on the GPU on the CUDA lane, which stops without one."""
    if studio_reader.is_studio(weights):
        return studio_reader.StudioReader(weights)
    if APPLE:
        from ink.model.reader_apple.tile_reader import TileReader

        return TileReader(weights)
    import torch

    from ink.model.reader_cuda.tile_reader import TileReader

    if not torch.cuda.is_available():
        raise RuntimeError("the CUDA lane reads ink on an NVIDIA GPU, and PyTorch sees none")
    return TileReader(weights, "cuda")
