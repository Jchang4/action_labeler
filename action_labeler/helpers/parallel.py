from concurrent.futures import ProcessPoolExecutor
from typing import Callable

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
