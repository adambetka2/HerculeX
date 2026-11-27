# ink/reading

These modules run the ink stage over every laid-out piece, and read selected places again at other depths.

| File | Description |
| --- | --- |
| `piece_reading.py` | Runs the ink stage. For each piece, it samples the layer stack, writes the scan images, finds the written face, and reads the ink map. It fetches the next CT store while it reads the current one. |
| `unit_fetch.py` | Fetches each unit's CT stores for the read as soon as the pieces stage writes the unit, so the link works while later units are unwrapped. A store is locked while one process fetches into it and stamped with its unit's record, so a rebuilt unit fetches again. |
| `spot_depths.py` | Reads the places circled with the viewer's Spot tool again at three nearby depths, and writes a page of them. |
| `passage_reads.py` | Reads listed pieces again from freshly sampled stacks with each ink model you name, at depths -4 to +4 and as a mean over depths, and writes a page beside the site's map. |
| `same_output_models.txt` | Lists, one line per checkpoint, its current read digest followed by earlier digests that give byte-identical maps. |

## Commands

```sh
# The pipeline's ink stage: lay the pieces out again if they changed, then read them.
./run.sh runner.pipeline run ink SCROLL [--scan ID]

# Read one scan directly. ORDER.json optionally lists the stores and pieces to read first.
./run.sh ink.reading.piece_reading SCROLL SCAN [ORDER.json]

# Fetch every unit's CT as the pieces stage writes it, until the layout is written (the pipeline's fetch stage).
./run.sh ink.reading.unit_fetch SCROLL SCAN

# Fetch the CT of a read's first STORES stores without reading, for example while the GPU is busy.
./run.sh ink.reading.piece_reading ahead SCROLL SCAN STORES [ORDER.json]

# Read every circled spot on a published sheet at three depths.
./run.sh ink.reading.spot_depths SCROLL SCAN SHEET_NAME [MARKS.json] [PAGE_NAME]

# Read the pieces listed in PLACES.json again with one or more checkpoints.
./run.sh ink.reading.passage_reads SCROLL SCAN PLACES.json PAGE_NAME MODEL...
```

For a whole scroll, run the read as a job on the [job board](../../runner/README.md#run-a-scroll).

## Outputs

The ink stage writes each piece's images and ink map to `faces/level-0/UNIT/` in the scan's folder
(`unwrapping/level/SCROLL/SCAN/` in the outputs). It keeps each piece's layer stack in `stacks/level/SCROLL/SCAN/` in
the outputs, and deletes each unit's fetched CT store after the unit is read. If a read fails, the CT it fetched stays
in place for the next attempt.

## Settings

| Variable | Default | Effect |
| --- | --- | --- |
| `HERCULEX_INK_MODEL` | `reader-v3` | Reads with another ink model of `./run.sh ink.model.models`, into its own faces folder. A Studio model reads 16 planes of each stack as they are, without Reader v3's filling of empty columns, and keeps its own layer stacks; spot depths and passage reads take Reader v3 models only. |
| `HERCULEX_STUDIO_STRIDE` | `64` | A Studio read's window step: 64 blends overlapping windows with a Hann window, 96 reads each 96-pixel square once, 2.25 times faster, with a grid of tile seams. It is part of the read's digest. |
| `HERCULEX_STUDIO_BATCH` | `4` | The windows of a Studio forward pass. |
| `HERCULEX_PRECISION` | `bfloat16` | A Studio read's precision on CUDA; `float32` turns half precision off. |
| `HERCULEX_SAMPLERS` | `4` | Sets the number of processes that build stacks ahead of the reader. |
| `HERCULEX_WINDOWS` | `stacks/level/` in the outputs | Sets the folder for kept layer stacks. |
| `HERCULEX_KEEP_STACKS` | `1` | Set to `0` to skip keeping layer stacks. |
| `HERCULEX_KEEP_BRICKS` | unset | Set to `1` to keep each unit's CT store after its read, so a later read of the same units fetches only the bricks that new pieces add. |
| `HERCULEX_FACES` | `level-0` | Writes a separate read to `faces/NAME/` instead of the published read. |
| `HERCULEX_DEPTH_SHIFTS` | `0` | Averages reads at these comma-separated layer shifts, for example `-4,-2,0,2,4`. |
| `HERCULEX_UNITS` | every unit | Reads only these comma-separated units. |

A separate read uses the same written face as the published read. Set `HERCULEX_FACES` whenever you set
`HERCULEX_DEPTH_SHIFTS`, so that the separate read doesn't replace the published one. `runner.review_page` publishes
only a full read in `faces/level-0/`; to view a separate read, use `website.sheet_tiles` and `website.site_pages part`.
