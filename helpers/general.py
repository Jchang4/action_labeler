from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Callable

import yaml
from tqdm.auto import tqdm


def parallel(
    f: Callable,
    items: list,
    *args: list,
    n_workers=24,
    **kwargs,
):
    """Applies `func` in parallel to `items`, using `n_workers`

    Args:
        f (function): function to apply
        items (list): list of items to apply `f` to
        n_workers (int, optional): number of workers. Defaults to 24.

    Returns:
        list: list of results
    """
    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        r = list(
            tqdm(
                ex.map(f, items, *args, **kwargs),
                total=len(items),
            )
        )
    if any([o is Exception for o in r]):
        raise Exception(r)
    return r


def create_dataset_yaml(path: Path, classes: list[str]):
    data = {
        "train": "train/images",
        "val": "valid/images",
        "path": path.name,
        "nc": len(classes),
        "names": classes,
    }
    yaml.dump(data, open(path / "data.yaml", "w"))


def get_box_key(box: list[float]) -> str:
    """Convert a list of floats to a string of floats"""
    return " ".join([f"{x:.6f}" for x in box])
