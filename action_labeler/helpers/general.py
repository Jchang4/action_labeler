import pickle
from collections import defaultdict
from pathlib import Path

from PIL import Image


def get_image_folders(root_dir: Path, exclude_filters: list[str] = []) -> list[Path]:
    image_folders = []

    for path in root_dir.rglob("*"):
        if (
            path.is_dir()
            and path.name == "images"
            and not any([f in str(path) for f in exclude_filters])
        ):
            image_folders.append(path.parent)
    return sorted(image_folders, key=lambda x: (len(str(x).split("/")), str(x)))


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


def load_image(path: str | Path, convert_to_rgb: bool = False) -> Image.Image:
    """Load an image.

    Args:
        path (str | Path): The path to the image.
        convert_to_rgb (bool, optional): Whether to convert the image to RGB if it is in RGBA mode. Defaults to False.

    Returns:
        Image.Image: The loaded image.
    """
    path = Path(path)
    image = Image.open(path)
    if convert_to_rgb and image.mode == "RGBA":
        image = image.convert("RGB")
    return image
