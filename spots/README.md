# spots

The `spots` package turns each circle that you draw with the viewer's Spot tool into a depth sweep that you can judge.
For each circle, it narrows the sheet to the pieces under the circle, rebuilds them as continuous surfaces, reads them
at every depth from -15 to +15, and stores the results in a Parquet database. Use it to find and judge candidates for
a First Letters submission (10 legible letters in 4 cm²).

| File | Description |
| --- | --- |
| `sources.py` | Finds the scroll, scan, page and pieces that each marks log belongs to. |
| `catalogue.py` | Locates every standing Spot circle on its pieces and in the scan, writes the `spots` table, and drops circles erased in the viewer. |
| `surfaces.py` | Narrows a spot's pieces to the circle, rebuilds them on the m7 surface prediction, and renders 67 layers along the normal. |
| `sweep.py` | Reads every rendered surface at every depth, in both stack orders, with Reader v3 and any readers you list. |
| `images.py` | Lays the sweep out as the page showed it, stretches each model's reads once, builds a z-score ensemble, and writes a page per spot. |
| `pilot.py` | Runs one spot through its stages; each stage resumes where it stopped. |
| `database.py` | Keeps the Parquet tables `spots`, `surfaces`, `reads` and `verdicts`, and records verdicts. |
| `flatten.py` | Shifts each pixel's column of a rendered stack along the normal so that the stack follows its own papyrus layers. |
| `reflatten.py` | Gives a mesh grid a near-isometric flattening, laid out by its 3D edge lengths and resampled on a regular grid. |

## Commands

```sh
# Build the spots table from the live marks and print each spot's ID.
./run.sh spots.catalogue

# Run the named stages for one spot, or all of them in order.
./run.sh spots.pilot SPOT [build] [grow] [render] [read] [images]

# Show the table sizes, or print one table as JSON lines.
./run.sh spots.database [TABLE]

# Record a verdict.
./run.sh spots.database verdict SPOT SURFACE --by NAME --readable no|traces|letters|passage [--letters N] \
  [--stack ensemble] [--side inward] [--depth D] [--reading TEXT] [--note TEXT]
```

On the job board, each spot is two jobs: a build on the network link and a read on the GPU and the Neural Engine.

```sh
./run.sh board add spot-SPOT-build --link --memory 3 -- ./run.sh spots.pilot SPOT build render
./run.sh board add spot-SPOT-read --gpu --engine --memory 5 --after spot-SPOT-build -- ./run.sh spots.pilot SPOT read images
```

The pages are at `http://localhost:8765/sheets/spot-candidates/index.html`. The index lists every circle and links to
the sweep page of each spot that has one. On a sweep page, the arrow keys step through the depths and the stacks.

## Choose the circles

Every standing circle is a target. To target only recent circles, set `HERCULEX_SPOTS_SINCE=YYYY-MM-DD`; older circles
stay in the table, marked as not targets.

The tools find circles on any page that the pipeline published. For a page made another way, such as a rebuilt
surface, add an entry to the library's `marks/sheets.json` in the form `{key, scroll, scan, route, page, pieces}`, with
`pieces` relative to the outputs folder.

## How a spot is read

The method follows the published code of the PHerc0343 First Letters entry.

- **Surfaces.** `original` is the piece as the pipeline reads it. `m7` moves every node along its normal onto the
  nearest sheet in the published m7 surface prediction for the scan. `grown`, made by the optional `grow` stage,
  extends the `m7` surface past its edges and into its holes wherever the prediction holds a sheet within 6 voxels
  along the normal, and is cut to the circle's box and margin.
- **Render.** Each surface is rendered as 67 layers one voxel apart. Depth `d` is the layer number minus 33, so `d = 0`
  is the surface and positive depths point away from the scroll axis.
- **Sweep.** A reader reads a 17-layer window centered on every `d` from -15 to +15, in both stack orders: `outward`
  and `inward` (reversed). You decide from the images which side holds the writing.
- **Readers.** The sweep reads with every Reader v3 ink model of the library (`./run.sh ink.model.models`): Reader v3 and the
  checkpoints of the same network listed in `weights/ink/models.json`; Studio models are left out, as they read 16 planes. Reader v3 alone reads the `original` surface.
- **Images.** Each model's reads are stretched once over all depths, never per depth, from their median to their 99.9th
  percentile; the CT is stretched from its 1st to its 99.5th percentile. The ensemble averages the models' z-scores,
  and `ensemble 5 depths` also averages it over `d-2` to `d+2`. A page opens at the depth where the ensemble's 97th
  percentile inside the circle peaks. You can mirror or turn the view.

Most circles cover several pieces, often from different windings, that the layout places side by side. Each piece is
rebuilt and read on its own as one continuous surface, and the pieces are laid out together only for the images. As a
result, letters that cross two pieces of different windings don't join up.

## Data

All spot data is stored in `SPOT_CANDIDATES`, the library's `studies/shared/spot-candidates/` folder:

| Folder | Contents |
| --- | --- |
| `db/` | The Parquet tables. |
| `m7/` | The fetched chunks of the m7 prediction. |
| `spots/SPOT/` | The surfaces, CT bricks, stacks and reads of each spot. |
