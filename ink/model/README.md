# ink/model

These modules fetch Reader v3 and run it in either lane: on the apple lane, the 3D stem on the GPU with Metal and the
2D body on the Neural Engine; on the cuda lane, the whole network in PyTorch on an NVIDIA GPU. Both lanes read the same
tiles with the same arithmetic.

| File | Description |
| --- | --- |
| `models.py` | Lists the ink models by name: stock Reader v3 (`reader-v3`), built in, and the checkpoints that a library lists in `weights/ink/models.json`. A `.safetensors` file is a Studio checkpoint and anything else a Reader v3 one. `HERCULEX_INK_MODEL` names the one every read uses; `./run.sh ink.model.models` lists them with their kind, and `add NAME TITLE FILE` adds one. |
| `studio_reader.py` | Reads with the [Studio](https://github.com/priyankaranke/vesuvius-studio-multiscroll-cpt) network, or any checkpoint of it: a 3-D residual U-Net over 16 planes of the stack in 128-pixel windows, blended by a Hann window, in PyTorch on this lane's GPU. `fetch` downloads its pinned code and public weights and lists them as `studio`; `check FILE` and `models add` check a file's tensors against the network. The code is MIT and the public weights CC BY-NC 4.0. |
| `weights.py` | Downloads Reader v3 at its pinned revision, checks its SHA-256 digest, and converts it, or any training checkpoint of the network, into the tile readers' state, `weights/ink/NAME/tile-reader.pth`. |
| `stock_model.py` | Builds the `ink_9um` network from the Vesuvius Challenge `villa` code at a pinned commit, and fetches that code and the checkpoint that configures the network on first use. |
| `input_stack.py` | Defines the input: a 17-layer window placed from the stack's written face, with raw values scaled. |
| `tiles.py` | Holds what both lanes' readers share: the statistics that normalize a stack, measured once per stack, the 512-pixel tiles every 384 pixels, and the Hann window that blends their maps. |
| `tile_filling.c` | Fills a tile's empty columns from shifted, flipped copies of its own stack. |
| `reader.py` | Gives the ink stage the reader of a checkpoint: Studio's for a `.safetensors` file, the lane's for Reader v3's. It lists the files whose bytes make up the read's digest. |
| `reader_apple/` | The apple lane's reader. `tile_reader.py` runs the stem as one Metal kernel (`depth_stem.metal`) on its own thread, ahead of the body, which `neural_engine.py` compiles for the Neural Engine in half precision once per checkpoint. `engine_calls.m` calls Core ML from C, so a thread that waits on the Neural Engine doesn't hold Python's global interpreter lock. |
| `reader_cuda/` | The cuda lane's reader. `tile_reader.py` runs the stem as two 3D convolutions with its norms folded in, then the body, in half precision on the GPU, with the 512-pixel tile compiled once. On a CPU, it runs in float32. |

The readers fill empty columns so that per-tile normalization doesn't read the edge of a hole as ink.
`tests/test_reader_apple_e2e.py` checks the Metal stem and the half-precision Neural Engine body against the float32
network. `tests/test_reader_cuda_e2e.py` checks the cuda lane's reader on a stream of stacks, on CUDA against the CPU.
