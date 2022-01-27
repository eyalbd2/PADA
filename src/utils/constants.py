from pathlib import Path

PROJECT_ROOT_DIR = Path(__file__).parents[2].absolute()
DATA_DIR = PROJECT_ROOT_DIR / "data"
EXP_DIR = PROJECT_ROOT_DIR / "runs"