# ink/model

These modules fetch Reader v3 and run it on Apple silicon: the 3D stem on the GPU with Metal and the 2D body on the
Neural Engine.

| File | Description |
| --- | --- |
| `weights.py` | Downloads Reader v3 at its pinned revision, checks its SHA-256 digest, and converts it for the tile reader. |
| `stock_model.py` | Builds the `ink_9um` network from the Vesuvius Challenge `villa` code at a pinned commit, and fetches that code and the checkpoint that configures the network on first use. |
| `tile_reader.py` | Reads a stack in 512-pixel tiles every 384 pixels and blends them with a Hann window. It fixes the stem's normalization to statistics measured once per stack. |
| `depth_stem.metal` | Runs the 3D stem as one Metal kernel, on its own thread, ahead of the Neural Engine. |
| `neural_engine.py` | Compiles the 2D body for the Neural Engine in half precision, once per checkpoint. |
| `engine_calls.m` | Calls Core ML from C, so a thread that waits on the Neural Engine doesn't hold Python's global interpreter lock. |
| `input_stack.py` | Defines the input: a 17-layer window placed from the stack's written face, with raw values scaled. |
| `tile_filling.c` | Fills a tile's empty columns from shifted, flipped copies of its own stack. |
| `portable_reader.py` | Runs the same arithmetic in PyTorch, in float32, on a CPU or a CUDA GPU. Off macOS, `machine_reader.reader()` gives this reader to the ink stage. Start from this file to port the reader to other hardware. |
| `machine_reader.py` | Gives the reader this machine runs: the tile reader on a Mac, the portable reader on a CUDA GPU elsewhere (it stops when there is none). It sits apart from `tile_reader.py`, whose bytes are part of the Mac read digest. |

The tile reader fills empty columns so that per-tile normalization doesn't read the edge of a hole as ink.
`tests/test_tile_reader_e2e.py` checks the Metal stem and the half-precision Neural Engine body against the float32
network. `tests/test_portable_reader_e2e.py` checks the PyTorch reader's stream of stacks, on CUDA against the CPU.
