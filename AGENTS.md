# HerculeX for agents

HerculeX unwraps Herculaneum scrolls from X-ray CT and reads their ink. A person reads the results in a local viewer
while you run the pipeline and act on what they label. This page is your map: the rules, how the modules fit together,
the commands for each task, and what failures mean. Each folder's README lists its files and settings.

## Rules

- Run every module from the repository root as `./run.sh MODULE ARGS`. `./run.sh list` shows every module and what it
  does.
- Get data paths only from `storage/data_roots.py`; never write absolute paths into code. `HERCULEX_DATA`,
  `HERCULEX_LIBRARY` and `HERCULEX_OUTPUTS` move the data folders.
- Keep code that only one lane runs in that lane's folder (`*_apple`, `*_cuda`) and shared code outside. When you
  change one lane's kernel or reader, make the same change in the other lane's.
- A unit's `record.json` names the digest of the code that made it, so editing that code makes its units stale. If you
  prove with a byte comparison that the new code gives the same output, add the old digest to the module's
  `same_output_methods.txt` (or `ink/reading/same_output_models.txt`). Never add one without that proof.
- On the apple lane, put anything that runs for more than a few minutes or uses the GPU, the Neural Engine or the
  network on the job board, and run `./run.sh board room GB` before a test or a quick GPU check.
- Treat the label logs in `data/library/marks/` as the person's data: append-only, never rewritten or deleted.
- Never edit the viewer's pages by hand. The pipeline publishes scroll pages; `./run.sh website.site_pages card PAGE`
  puts any other page on the start page. Titles read `scroll · ink model · surface · date`.
- Write no comments in code and one-sentence docstrings. Before you finish, run `./run.sh check cpu`, also
  `./run.sh check` if you touched a GPU path or the reader, and `./run.sh check lint` (`--fix` formats first).
- When you report a number, name the data it was measured on.

## How it fits together

```text
scans/catalog          which scan of which scroll, from the open data's metadata
  │ scans/transfer       CT bricks fetched by byte range into brick stores
  ▼
unwrapping/detection   sheets stage: sheet nodes per unit of brick rows             unwrapping/nodes/SCROLL/SCAN/nodes
  ▼
unwrapping/level       pieces stage: exact winding levels, pieces, joins, the frame  unwrapping/level/SCROLL/SCAN/pieces
  ▼
unwrapping/layout      every piece placed on one sheet                               unwrapping/level/SCROLL/SCAN/sheet
  ▼
sampling + ink         sheet stage: layer stacks, the written face, ink maps         unwrapping/level/SCROLL/SCAN/faces
  ▼
website                tile pyramid, viewer page, start-page card                    website/sheets/sheet-PHerc0826-113keV-level
  ▲
spots                  the person's circles rebuilt and read at every depth          website/sheets/spot-candidates
```

The right-hand paths are in `data/outputs`. A read by another ink model than Reader v3 writes `faces/level-0-MODEL`
and a page whose name ends in `-MODEL`.

| Module | What it holds |
| --- | --- |
| `runner/` | The stages (`pipeline.py`), each lane's unwrap and machine probes (`machine_apple/`, `machine_cuda/`), the job board, kernel builds, the test and lint runners. |
| `storage/` | Data roots, brick stores, node stores, tile stores. |
| `scans/` | The scan catalog and CT transfer. |
| `unwrapping/` | Detection (`nodes_apple/`, `nodes_cuda/`), the level fit (`threads_apple/`, `threads_cuda/`), layout. |
| `sampling/` | Layer stacks along piece normals and the written face. |
| `ink/` | Reader v3 and the other ink models (`model/`: `reader_apple/`, `reader_cuda/`, `models.py`), the ink stage (`reading/`). |
| `website/` | Tiles, the viewer templates in `assets/`, the start page, the local server and its mark store. |
| `spots/` | Circled spots to rebuilt surfaces, depth sweeps and verdicts. |
| `calibration/` | The bench that scores fits against published meshes (`bench/`), and earlier unwrap studies. |
| `tests/` | `*_e2e.py` run a stage on synthetic inputs; `*_golden.py` hold fast paths to a reference. |

Two lanes run the same stages: `apple` on a Mac (Metal and the Neural Engine) and `cuda` on Linux with NVIDIA GPUs
(CUDA and PyTorch). `runner/lane.py` picks the lane from the machine. Tests marked `apple` or `cuda` skip on the other.

## Unwrap and publish a scroll

```sh
./run.sh setup                      # once: the lane's packages and Reader v3
./run.sh unwrap PHerc0826           # SCROLL[:SCAN] ..., CT to published page
```

- On the apple lane, `unwrap` queues four board jobs per scroll: `SCROLL-sheets`, `SCROLL-pieces`, `SCROLL-fetch`
  (the CT for the read, fetched while units are unwrapped) and `SCROLL-sheet` (read and publish). Follow them with
  `./run.sh board status` and `./run.sh board log NAME`; after a fix, `./run.sh board retry NAME`.
- On the cuda lane, `unwrap` runs every scroll named in the foreground until its page is published and prints why any
  scroll stopped. Job logs go to `logs/unwrap/`. Rerun the same command after a fix: it skips finished work.
- `./run.sh runner.pipeline status SCROLL` shows what is done. `./run.sh runner.pipeline run STAGE SCROLL` runs one
  stage: `sheets`, `pieces`, `fetch`, `layout`, `ink` (read without publishing) or `sheet`. Add `--scan ID` for another
  scan than the survey scan.
- When a page is published, start the viewer (`./run.sh website.site_server`), check that the page loads at
  `http://localhost:8765/sheets/index.html`, and tell the person.

| Task | Command |
| --- | --- |
| Read with another ink model | `./run.sh ink.model.models` lists them; set `HERCULEX_INK_MODEL=NAME` on `unwrap` or the `sheet` job. Its faces and page are its own. |
| Add a checkpoint of the Reader v3 network | `./run.sh ink.model.models add NAME "TITLE" FILE` with a `.pth` |
| Add a Studio checkpoint | `./run.sh ink.model.models add NAME "TITLE" FILE.safetensors` checks the file against Studio's network and lists it; the public weights are `./run.sh ink.model.studio_reader fetch`, listed as `studio`. Read with it like any model. |
| Read apart from the published ink, such as at several depths | `HERCULEX_FACES=NAME HERCULEX_DEPTH_SHIFTS=-4,-2,0,2,4`, then `website.sheet_tiles` and `website.site_pages part` |
| Republish a read scan | `./run.sh runner.review_page SCROLL SCAN` |
| Draw a scroll as one picture | `./run.sh website.overview PAGE OUT.jpg` |
| See where a run's time went | `./run.sh runner.timeline "SCROLL-*"` |
| Find space to free | `./run.sh runner.disk` (it lists, never deletes) |
| Score pieces against the bench's truth | `./run.sh calibration.bench.score CONTROL PATH/TO/pieces/UNIT` |

## Act on the person's labels

The viewer saves each label as one JSON line in `data/library/marks/ROUTE/SCROLL/SCAN/marks.jsonl`: `kind` is `ink`,
`blank` or `spot`, `points` are millimeters on the unrolled sheet, and `layout` names the piece layout it was drawn on
(saved in `layouts/` beside the log). A line with `remove` withdraws an earlier label.

1. `./run.sh spots.catalogue` maps every standing circle to its pieces, windings and box in scan voxels.
2. For each spot, queue a build and a read:
   `./run.sh board add spot-SPOT-build --link --memory 3 -- ./run.sh spots.pilot SPOT build render` and
   `./run.sh board add spot-SPOT-read --gpu --engine --memory 5 --after spot-SPOT-build -- ./run.sh spots.pilot SPOT read images`.
   Each spot gets a page under `/sheets/spot-candidates/` with reads at every depth from -15 to +15.
3. The person judges the letters. Record each verdict with
   `./run.sh spots.database verdict SPOT SURFACE --by NAME --readable no|traces|letters|passage`.

## When something fails

| You see | It means | Do |
| --- | --- | --- |
| `unwrapping failed on N unit(s)` | Some units of the pieces stage failed; the others finished and joined. | Read the error above it, fix the cause, rerun. |
| `no finished sheet-node units` or `no ready pieces` | The stage before isn't done for that scan. | Run or finish that stage first. |
| `holds units not cut to HERCULEX_UNIT_ROWS` | The scan's units were cut to another number of brick rows. | Use the number they were cut to. |
| Units rerun that you didn't change | Their recorded digest isn't the current one or in the same-output list. | Find the edit that changed the digest; prove same output before listing it. |
| `this machine runs the apple lane` | `HERCULEX_LANE` names the other lane. | Unset it. |
| `the CUDA lane reads ink on an NVIDIA GPU, and PyTorch sees none` or `needs libcuda.so.1` | No usable NVIDIA driver or GPU. | Check `nvidia-smi`. |
| `HERCULEX_INK_MODEL=NAME: the ink models are ...` or `no checkpoint at` | The model isn't listed, or its weights are missing. | `./run.sh ink.model.models`, then `./run.sh setup` or `add`. |
| `HerculeX data folder is unavailable` | The data drive isn't mounted. | Mount it or set `HERCULEX_DATA`. |
| `pieces still need ink` when publishing | The read isn't finished. | Run the `ink` stage. |
| A board job waits for room, or is stopped under swap | Memory is short beside the other jobs. | Let it wait; never raise its reservation to force it. |
| A first Neural Engine read is slow | The body is compiled once per checkpoint. | Wait. |
| A fetch fails or a disc fills | The CT server or the disc. | Rerun to resume; `./run.sh runner.disk` for space. |
| `./run.sh check lint` fails | A file isn't formatted, or a linter found a bug. | `./run.sh check lint --fix`, then fix what remains by hand. |

## Where things are

```text
README.md                    what HerculeX is, results, setup and the viewer
runner/README.md             stages, the job board, each lane's unwrap, settings
unwrapping/*/README.md       detection and the level fit, their kernels and settings
ink/model/README.md          the readers and the ink models
spots/README.md              from circled spots to depth sweeps and verdicts
storage/README.md            data folders and file formats
calibration/bench/README.md  the bench that scores the level fit against published meshes
tests/README.md              the kinds of tests
```
