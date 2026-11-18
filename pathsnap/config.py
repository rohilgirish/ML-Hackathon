"""Central configuration for PathSnap. Paths can be overridden with environment variables."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Raw data. The evaluator-testing folder is only ever read by the API at demo time.
def _resolve_data_dir():
    env_dir = os.environ.get("PATHSNAP_DATA_DIR")
    if env_dir:
        return Path(env_dir)
    for candidate in [
        ROOT / "Student folder" / "Student folder",
        ROOT / "Student folder",
        ROOT.parent / "Student folder" / "Student folder",
        ROOT.parent / "Student folder",
    ]:
        if (candidate / "train").exists():
            return candidate
    return ROOT / "Student folder"

DATA_DIR = _resolve_data_dir()
TRAIN_DIR = DATA_DIR / "train"
STUDENT_TEST_DIR = DATA_DIR / "student_test"

def _resolve_evaluator_dir():
    candidates = [
        ROOT / "evaluator_test" / "evaluator_test",
        ROOT / "evaluator_test",
    ]
    for c in candidates:
        if c.exists() and any(p.is_dir() and not p.name.startswith(".") and p.name != "evaluator_test" for p in c.iterdir()):
            return c
    return ROOT / "evaluator_test" / "evaluator_test"

EVALUATOR_TEST_DIR = _resolve_evaluator_dir()
EVAL_DIR = Path(os.environ.get("PATHSNAP_EVAL_DIR", EVALUATOR_TEST_DIR if EVALUATOR_TEST_DIR.exists() else STUDENT_TEST_DIR))

ARTIFACT_DIR = Path(os.environ.get("PATHSNAP_ARTIFACT_DIR", ROOT / "artifacts"))
RESULTS_DIR = ROOT / "results"
FIG_DIR = RESULTS_DIR / "figures"
MAP_DIR = RESULTS_DIR / "maps"
CACHE_DIR = ROOT / ".cache"

SEED = 42

# Candidate generation
CAND_RADIUS_M = 50.0      # arcs within this distance of a GPS fix are candidates
CAND_MAX_K = 12           # keep at most K nearest candidates per fix
CAND_FALLBACK_K = 3       # if nothing within radius, keep the K nearest arcs anyway

# Labelling: fixes whose aligned reference arc is further than this are treated as
# unreliable labels (GPS outliers / route gaps) and excluded from training & scoring.
LABEL_MAX_DIST_M = 100.0

# Training rows: consecutive 1 Hz fixes are near-duplicates, so we keep every n-th fix.
TRAIN_OBS_STRIDE = 3

CV_FOLDS = 5
