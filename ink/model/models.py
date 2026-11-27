"""The ink models the pipeline reads with, and HERCULEX_INK_MODEL, which names the one every read uses.

./run.sh ink.model.models                        list the models, the chosen one marked
./run.sh ink.model.models add NAME TITLE FILE    add a checkpoint under NAME: a .pth of the Reader v3 network, or a
                                                 .safetensors of the Studio network (ink.model.studio_reader)

Stock Reader v3 (reader-v3) is built in. A library lists its other checkpoints in weights/ink/models.json, as
{NAME: {"title": TITLE, "file": PATH under weights/ink}}; a .safetensors file is read as Studio, anything else as
Reader v3. A read by another model than stock Reader v3 keeps its own ink maps (faces level-0-NAME) and its own page
(sheet-SCAN-NAME), so reads by several models of one scan stand side by side.
"""

import json
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

from ink.model import studio_reader, weights
from storage.data_roots import INK_MODEL, INK_WEIGHTS, STOCK_INK_MODEL

LISTED = INK_WEIGHTS / "models.json"


@dataclass(frozen=True)
class InkModel:
    name: str
    title: str
    weights: Path

    @property
    def page_suffix(self):
        """What a page read by this model adds to its scan's page name."""
        return "" if self.name == STOCK_INK_MODEL else f"-{self.name}"

    @property
    def kind(self):
        """Which network reads with this model: reader-v3 or studio."""
        return "studio" if studio_reader.is_studio(self.weights) else "reader-v3"


def listed():
    return json.loads(LISTED.read_text()) if LISTED.is_file() else {}


def models():
    """Every ink model of this library by name, stock Reader v3 first."""
    found = {STOCK_INK_MODEL: InkModel(STOCK_INK_MODEL, "Reader v3", weights.STOCK_READER)}
    for name, entry in listed().items():
        found[name] = InkModel(name, entry["title"], INK_WEIGHTS / entry["file"])
    return found


def chosen():
    """The model HERCULEX_INK_MODEL names, stock Reader v3 when it is unset."""
    known = models()
    if INK_MODEL not in known:
        raise SystemExit(f"HERCULEX_INK_MODEL={INK_MODEL}: the ink models are {', '.join(known)}")
    return known[INK_MODEL]


def list_model(name, title, target):
    """Lists weights already in place under weights/ink as the model NAME."""
    if name in models() and name not in listed():
        raise SystemExit(f"{name} is built in")
    entries = listed() | {name: {"title": title, "file": str(Path(target).relative_to(INK_WEIGHTS))}}
    LISTED.parent.mkdir(parents=True, exist_ok=True)
    LISTED.write_text(json.dumps(entries, indent=1) + "\n")
    print(f"{name}: {title}, {target}")


def add(name, title, source):
    """A checkpoint kept under weights/ink/NAME/ and listed as NAME: Reader v3's as tile-reader.pth, Studio's as studio.safetensors."""
    if name in models() and name not in listed():
        raise SystemExit(f"{name} is built in")
    if studio_reader.is_studio(source):
        studio_reader.check(source)
        target = INK_WEIGHTS / name / "studio.safetensors"
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    else:
        target = INK_WEIGHTS / name / "tile-reader.pth"
        target.parent.mkdir(parents=True, exist_ok=True)
        if weights.is_tile_state(source):
            shutil.copyfile(source, target)
        else:
            weights.tile_state(source, target)
    list_model(name, title, target)


def show():
    """Every model, the chosen one starred; an unknown HERCULEX_INK_MODEL ends with its error."""
    for model in models().values():
        mark = "*" if model.name == INK_MODEL else " "
        state = "ready" if model.weights.is_file() else "missing"
        print(f"{mark} {model.name:24s} {model.title:40s} {model.kind:10s} {state:8s} {model.weights}")
    chosen()


if __name__ == "__main__":
    if sys.argv[1:2] == ["add"] and len(sys.argv) == 5:
        add(sys.argv[2], sys.argv[3], Path(sys.argv[4]))
    elif len(sys.argv) == 1:
        show()
    else:
        sys.exit(__doc__)
