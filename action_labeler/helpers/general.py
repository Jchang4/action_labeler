from pathlib import Path


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
