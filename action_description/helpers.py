import pickle
from pathlib import Path

import pandas as pd
from matplotlib import pyplot as plt
from PIL import Image
from tqdm.auto import tqdm


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
    seen_image_names = set()

    model_name = actions_file_name.split("_actions")[0].strip()

    for folder in tqdm(folders):
        if not (folder / actions_file_name).exists():
            continue

        descriptions = {}
        actions = pickle.load(open(folder / actions_file_name, "rb"))

        if (folder / descriptions_file_name).exists():
            descriptions = pickle.load(open(folder / descriptions_file_name, "rb"))

        for image_path, box_to_actions in actions.items():
            image_path = Path(image_path)
            if image_path.name in seen_image_names:
                continue
            seen_image_names.add(image_path.name)

            expected_action = image_path.parent.parent.name
            if expected_action in ["lake_trail", "living_room", "myspace_har"]:
                expected_action = None

            for box, actions in box_to_actions.items():
                description = descriptions.get(str(image_path), {}).get(box, [])
                if len(description) == 0:
                    description = None
                elif isinstance(description, list):
                    description = description[0]["description"]

                # for action in actions:
                #     action_str = ""
                #     if isinstance(action, dict) and "action" in action:
                #         action_str = action["action"]
                #     elif isinstance(action, dict):
                #         action_str = (
                #             action["description"]
                #             .replace('"', "")
                #             .replace("-", "")
                #             .replace("action:", "")
                #             .replace("action :", "")
                #             .strip()
                #         )
                #     else:
                #         action_str = action.strip()

                action_str = (
                    actions.replace('"', "")
                    .replace("-", "")
                    .replace("action:", "")
                    .replace("action :", "")
                    .strip()
                )

                data.append(
                    {
                        "model_name": model_name,
                        "folder": str(folder),
                        "image_path": str(image_path),
                        "box": box,
                        "description": description,
                        "action": action_str,
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
