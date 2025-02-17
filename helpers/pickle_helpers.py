import pickle
from collections import defaultdict
from pathlib import Path


def load_pickle(path: str | Path, filename: str = "classification.pickle"):
    path = Path(path)

    if not (path / filename).exists():
        return defaultdict(dict)

    with open(path / filename, "rb") as f:
        return defaultdict(dict, pickle.load(f))


def save_pickle(data: dict, path: str | Path, filename: str = "classification.pickle"):
    path = Path(path)

    print(f"Saving {len(data)} images to {str(path / filename)}")

    with open(path / filename, "wb") as f:
        pickle.dump(data, f)

    print(f"Saved classification file.")
