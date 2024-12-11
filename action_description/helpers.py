import pickle
from pathlib import Path

import pandas as pd
from matplotlib import pyplot as plt
from PIL import Image


def get_image_folders(exclude_filters: list[str] = []) -> list[Path]:
    ROOT_DIR = Path("datasets/human")
    image_folders = []

    for path in ROOT_DIR.rglob("*"):
        if (
            path.is_dir()
            and path.name == "images"
            and not any([f in str(path) for f in exclude_filters])
        ):
            image_folders.append(path.parent)
    return sorted(image_folders, key=lambda x: (len(str(x).split("/")), str(x)))


def get_description_action_dataframe(
    folders: list[Path],
    actions_file_name: str,
    descriptions_file_name: str,
) -> pd.DataFrame:
    data = []

    model_name = actions_file_name.split("_actions")[0].strip()

    for folder in folders:
        if not (folder / descriptions_file_name).exists():
            continue

        descriptions = pickle.load(open(folder / descriptions_file_name, "rb"))
        actions = pickle.load(open(folder / actions_file_name, "rb"))

        for image_path, box_to_description in descriptions.items():
            if image_path not in actions:
                continue

            expected_action = Path(image_path).parent.parent.name
            if expected_action in ["lake_trail", "living_room", "myspace_har"]:
                expected_action = None

            for box, description in box_to_description.items():
                data.append(
                    {
                        "model_name": model_name,
                        "folder": str(folder),
                        "image_path": str(image_path),
                        "box": box,
                        "description": description[0]["description"],
                        "action": actions[image_path].get(box, "none"),
                        "expected_action": expected_action,
                    }
                )

    df = pd.DataFrame(data)
    return df


def show_images(images: list[Image.Image], labels: list[str], ncols: int = 3):
    nrows = max(2, (len(images) + ncols - 1) // ncols)
    fig, axs = plt.subplots(nrows, ncols, figsize=(20, 20))
    for i, (image, label) in enumerate(zip(images, labels)):
        axs[i // ncols, i % ncols].imshow(image)
        axs[i // ncols, i % ncols].set_title(label)
    plt.show()
