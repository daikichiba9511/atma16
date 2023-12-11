import pathlib

ROOT_DIR = pathlib.Path(__file__).parent.parent
INPUT_DIR = ROOT_DIR / "input"
DATA_DIR = INPUT_DIR
OUTPUT_DIR = ROOT_DIR / "output"


NOT_USED_COLUMNS = [
    "session_id",
    "yad_no",
    "seq_no",
    "yad_no_right",
    "target",
    "fold",
]
