# runner

The `runner` package runs the pipeline on one Mac. It defines the stages, makes each stage resumable, and uses a job
board to share the GPU, the Neural Engine, the network link and memory between long jobs.

| File | Description |
| --- | --- |
| `pipeline.py` | Defines the pipeline's stages and runs or resumes each one from its records. |
| `sheet_stage.py` | Takes a scan's pieces to the website: it lays them out, reads every piece, and then publishes the scroll's page. |
| `review_page.py` | Publishes a read scan as its whole-scroll page and puts its card on the start page. |
| `stage_queue.py` | Provides what every stage shares: code digests, atomic units, unit claims and stage locks. |
| `job_board.py` | Runs one queue for every long job, with resource slots, memory limits, priorities and a swap guard. |
| `wait_for_queue.py` | Exits once none of the named board jobs can still run, however they end. |
| `memory_trace.py` | Runs a module or script and reports where its memory footprint peaks, line by line, including C and GPU memory. |
| `compiled_kernels.py` | Compiles the C and Objective-C kernels once per source digest into `runtime/kernels/` (with `cc` and OpenMP off macOS). |

## Run a scroll

Run every command from the repository root. Fetch Reader v3 once with `./run.sh ink.model.weights fetch`. Then queue
three board jobs for the scroll:

```sh
S=PHerc0826
./run.sh board add $S-sheets --gpu --link --share --memory 5 --setting HERCULEX_LIST_CT_BRICKS=1 -- \
  ./run.sh runner.pipeline run sheets $S
./run.sh board add $S-pieces --memory 5 --cores 3 --restartable --after $S-sheets -- \
  ./run.sh runner.pipeline run pieces $S
./run.sh board add $S-sheet --gpu --engine --link --memory 5 --cores 5 --after $S-pieces --setting HERCULEX_KEEP_STACKS=0 -- \
  ./run.sh runner.pipeline run sheet $S
./run.sh board log $S-sheet
```

Each job starts when the one before it finishes. The last job reads every piece and publishes the scroll's page, with
its card on the start page.

| Stage | What it does | Uses |
| --- | --- | --- |
| `sheets SCROLL` | Detects sheet nodes in the scroll's survey scan and fetches the CT as it goes. | GPU, link |
| `pieces SCROLL` | Unwraps the nodes into pieces, joins them across unit seams, and fits the scan's frame. | CPU |
| `layout SCROLL` | Places the pieces on one sheet. | CPU |
| `ink SCROLL` | Lays the pieces out again if they changed, then reads every piece into an ink map. | GPU, Neural Engine, link |
| `sheet SCROLL` | Runs `ink`, then publishes the scroll's page and its card on the start page. | GPU, Neural Engine, link |
| `all SCROLL` | Runs `sheets`, `pieces` and `sheet` at once, each following the one before, without the board. | All |

Every stage resumes from its finished units. If a job fails, fix the cause and run `./run.sh board retry NAME`. Add
`--scan ID` to a single stage to use a scan other than the survey scan, or `--band` to `sheets`, `pieces` or `all` to
process only a central band of the scan. To see what is done, run `./run.sh runner.pipeline status [SCROLL]`; to
rebuild the index of finished units, run `./run.sh runner.pipeline index`.

The `sheet` stage ends with `runner.review_page SCROLL SCAN`. It composes the tiles of the scan and the ink, writes the
viewer page and its card on the start page, and, if the local site is running, checks the tiles and marks over HTTP.
To publish a read scan again, run `./run.sh runner.review_page SCROLL SCAN`.

## Settings

| Variable | Default | Effect |
| --- | --- | --- |
| `HERCULEX_WEIGHTS` | Reader v3 | Reads with another checkpoint of the same network. |
| `HERCULEX_MODEL_NAME` | `Reader v3`, or the checkpoint's file name | Names the ink model on the page and its card. |
| `HERCULEX_KEEP_STACKS` | `1` | Set to `0` to skip keeping each piece's layer stack. |
| `HERCULEX_BRICKS` | `cache/bricks/` in the library | Puts the CT brick cache on another drive. |
| `HERCULEX_LIST_CT_BRICKS` | unset | Set to `1` on `sheets` to list the published bricks first and skip masked-out ones. |
| `HERCULEX_SITE` | `http://127.0.0.1:8765` | Sets the local site that the `sheet` stage checks after it publishes. |
| `HERCULEX_BOARD` | `runtime/board/` | Sets the folder for the board's database, logs and dispatcher lock. |
| `HERCULEX_BOARD_MARGIN_GB` | `1.0` | Sets the memory, in GB, that the board leaves free. |
| `HERCULEX_BOARD_CORES` | `10` | Sets the number of CPU cores the board shares out. |
| `HERCULEX_UNIT_ROWS` | `12` | Sets the brick rows of each sheet-detection unit, which the level solve takes whole; fewer need less memory. Set it before a scan's first `sheets` job: the stage refuses to add units of another size beside the ones a scan has. |

## Job board

```sh
./run.sh board add NAME [OPTIONS] -- COMMAND
./run.sh board status | log NAME [LINES] | room GB | wait NAME | forecast [NAME] | usage [HOURS]
./run.sh board hold|release|cancel|retry|remove NAME
./run.sh board resize NAME GB
./run.sh board change NAME [--priority N] [--cores N] [--share 0|1] [--restartable 0|1]
```

`status` lists the jobs and why each waiting job waits, `room GB` checks whether that much memory fits now, `wait`
blocks until a job ends, and `forecast` estimates when the queue or one job ends. The main options of `board add` are
these:

- `--gpu`, `--engine` and `--link` claim the GPU, the Neural Engine and the network link.
- `--memory GB` is the job's peak memory. If you leave it out, the board uses the last peak of jobs of the same kind.
- `--cores N` is the number of CPU cores the job uses (default `1`).
- `--after JOB ...` waits until those jobs are done; `--priority N` orders waiting jobs; `--held` waits for `release`.
- `--setting KEY=VALUE ...` sets environment variables for the job.
- `--share` marks a job that mostly downloads and uses the GPU in short bursts, such as sheet detection. It can run
  beside the one job that holds the GPU or the link: one sharing job on the GPU and two on the link. The Neural Engine
  is never shared.
- `--restartable` lets the board stop the job and run it again later.

When a GPU job is held back only by cores or memory that lower-priority restartable jobs hold, the board stops one of
them, at most one a minute, and the stopped job runs again once there is room. A job that a waiting GPU job needs
within two steps, such as its layout, can make room the same way, but it stops only sharing jobs. When swap grows too large, the board stops the largest
restartable job of the lowest priority. A job that exits with code 75 goes back to the queue and runs again later.

On Linux the board reads process memory and groups from `/proc` and the GPU's load from `nvidia-smi`. A job's footprint
is its resident memory, and the Neural Engine slot (`--engine`) is kept so that the same commands work; it guards
nothing there.

Before you run a test or a quick GPU check by hand, run `./run.sh board room GB` to check that it fits beside the
running jobs. The first board command starts a dispatcher in the background. If you change `job_board.py`, stop the
dispatcher with `kill $(cat runtime/board/dispatcher.lock)`; the next board command starts it again with the new code.
