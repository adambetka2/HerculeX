# Linux and CUDA

HerculeX runs on Linux with an NVIDIA GPU. Sheet detection uses a CUDA kernel, the level solve uses OpenMP, and ink is
read with the PyTorch reader. I tested it on WSL 2 (Ubuntu) with an RTX 5090.

Sheet detection and the ink reader need a CUDA GPU. Without one, the `sheets` and `sheet` jobs stop and say why; they
don't fall back to the CPU.

## How it works

| Mac | Linux |
| --- | --- |
| `gpu_nodes.metal` (Metal kernel) | `gpu_nodes.cu` (CUDA kernel) |
| `gpu_nodes_host.m` (Objective-C host) | `gpu_nodes_host.c` (C host, NVRTC) |
| `dispatch_apply` | OpenMP, through `parallel_blocks.h` |
| `TileReader` (Metal and Neural Engine) | `PortableReader` (PyTorch on CUDA) |
| `libproc`, `vm_stat` | `/proc`, `nvidia-smi` |

## Install

You need an NVIDIA driver and GCC. The kernel is compiled with NVRTC, which comes with PyTorch.

```sh
sudo apt install build-essential git curl
curl -LsSf https://astral.sh/uv/install.sh | sh
git clone https://github.com/adambetka2/HerculeX.git && cd HerculeX
UV_PROJECT_ENVIRONMENT=runtime/python uv sync --locked --extra ink
./run.sh ink.model.weights fetch
./run.sh check
```

`./run.sh check` includes two GPU tests: the CUDA kernel against the C kernel, and the reader on CUDA against the
reader on the CPU. I got 117 passed and 4 skipped (the Mac-only tests).

## Running

Use the same commands as on a Mac (see the [README](../README.md#unwrap-a-scroll)). On the job board, set
`HERCULEX_BOARD_CORES` to your core count (the default is 10).

## Memory

The level solve takes a whole unit of 12 brick rows at once. On a PHerc0841 band it peaked at 6.6 GB. With less memory,
`HERCULEX_UNIT_ROWS` sets a smaller unit on the `sheets` job. Set it before a scan's first `sheets` job and keep it: units
record their size, and the stage refuses to add units of another size, which would overlap them.

## WSL 2

Keep the checkout on the Linux disk. On a Windows drive (`/mnt/c`), temporary files fail and imports are
slow. If you have to keep it there, link the scratch folder to the Linux disk:

```sh
mkdir -p ~/herculex-scratch && ln -s ~/herculex-scratch runtime/scratch
```

## Differences from a Mac

The CUDA node count matches the C reference's to 0.0004%, but the level solve is sensitive to small differences. On a
PHerc0139 band, the CUDA nodes gave 240 and 256 pieces where the C nodes gave 262 and 282. So Mac and Linux runs of the
same scroll won't give the same pieces.

## Numbers

On one band of PHerc0841 (9.4 µm, 17.2 million nodes): sheet detection took 36 s on the GPU and 255 s with the C
kernel on six threads, and 99.9% of nodes lie within 0.003 voxel of the C reference's. The level solve took 9 minutes.
