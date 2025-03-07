from pathlib import Path

import pandas as pd
from PIL import Image

from action_labeler.helpers import (
    add_bounding_boxes,
    add_segmentation_masks,
    add_text,
    plot_images,
    resize_image,
    segmentation_to_box,
)

colors = [
    "red",
    "green",
    "blue",
    "yellow",
    "cyan",
    "magenta",
    "orange",
    "purple",
]


class ActionLabelDataset:
    pickle_file: str
    df: pd.DataFrame

    def __init__(self, pickle_file: str):
        self.pickle_file = pickle_file
        self.df = self._load_df(pickle_file)

    def __len__(self):
        return len(self.df)

    @property
    def actions(self) -> list[str]:
        return self.df["action"].unique().tolist()

    @property
    def image_paths(self) -> list[Path]:
        return self.df["image_path"].unique().tolist()

    @property
    def image_names(self) -> list[str]:
        return self.df["image_path"].unique().apply(lambda x: Path(x).name).tolist()

    @staticmethod
    def _load_df(pickle_file: str) -> pd.DataFrame:
        pickle_file = Path(pickle_file)
        if not pickle_file.exists():
            return pd.DataFrame({"image_path": [], "xywh": [], "action": []})
        df: pd.DataFrame = pd.read_pickle(pickle_file)

        # Ensure all columns are present
        assert "image_path" in df.columns, "image_path column must exist"
        assert "xywh" in df.columns, "xywh column must exist"
        assert "action" in df.columns, "action column must exist"

        df["image_path"] = df["image_path"].apply(Path)
        df["xywh"] = df["xywh"].apply(
            lambda x: ActionLabelDataset.xywh_to_str(x) if not isinstance(x, str) else x
        )
        return df

    @staticmethod
    def xywh_to_str(xywh: list[float]) -> str:
        return " ".join([f"{n:.6f}" for n in xywh])

    @staticmethod
    def str_to_xywh(xywh_str: str) -> list[float]:
        return [float(coord) for coord in xywh_str.split(" ")]

    @staticmethod
    def convert_segments_to_xywh(df: pd.DataFrame) -> pd.DataFrame:
        assert "xywh" in df.columns, "xywh column must exist"
        df["xywh"] = df["xywh"].apply(
            lambda x: (
                ActionLabelDataset.xywh_to_str(
                    segmentation_to_box(ActionLabelDataset.str_to_xywh(x))
                )
                if len(x.split(" ")) > 4
                else x
            )
        )
        return df

    ###############
    # Getters
    ###############
    def does_row_exist(self, image_path: Path, xywh: list[float]) -> bool:
        xywh_str = self.xywh_to_str(xywh)
        return (
            self.df[
                (self.df["image_path"] == image_path) & (self.df["xywh"] == xywh_str)
            ].shape[0]
            > 0
        )

    def get_rows(self, image_path: Path | str) -> pd.DataFrame:
        if isinstance(image_path, Path):
            return self.df[self.df["image_path"] == image_path]
        return self.df[self.df["image_path"].apply(lambda x: image_path in str(x))]

    ###############
    # Updaters
    ###############
    def add_row(self, image_path: Path, xywh: list[float], action: str) -> None:
        if self.does_row_exist(image_path, xywh):
            self.update_row(image_path, xywh, action)
        else:
            self.df = pd.concat(
                [
                    self.df,
                    pd.DataFrame(
                        {"image_path": [image_path], "xywh": [xywh], "action": [action]}
                    ),
                ],
                ignore_index=True,
            )

    def update_row(self, image_path: Path, xywh: list[float], new_action: str) -> None:
        xywh_str = self.xywh_to_str(xywh)
        self.df.loc[
            (self.df["image_path"] == image_path) & (self.df["xywh"] == xywh_str),
            "action",
        ] = new_action

    def update_row_by_index(self, index: int, action: str) -> None:
        self.df.loc[index, "action"] = action

    def update_class(self, old_class: str, new_class: str) -> None:
        self.df.loc[self.df["action"] == old_class, "action"] = new_class

    def combine_datasets(self, other_dataset: "ActionLabelDataset") -> None:
        # self.df = pd.concat([self.df, other_dataset.df])
        pass

    ###############
    # Deleters
    ###############
    def drop_classes(self, classes: list[str]) -> None:
        self.df = self.df[~self.df["action"].isin(classes)]

    ###############
    # Save
    ###############
    def save(self) -> None:
        self.df.to_pickle(self.pickle_file)
        print(f"Saved {len(self.df)} rows to {self.pickle_file}")

    ###############
    # Plotting
    ###############
    def show_class(
        self, class_name: str, num_samples: int = 5, detect_type: str = "bbox"
    ) -> None:
        """Plot a class from the dataset with bounding boxes"""
        assert detect_type in [
            "bbox",
            "segment",
        ], "detect_type must be 'bbox' or 'segment'"

        df = self.df[self.df["action"] == class_name]

        image_paths = []
        images = []

        for _, row in df.sample(min(num_samples, len(df)), replace=False).iterrows():
            image = Image.open(row["image_path"])
            image = resize_image(image, 640)

            # Add bounding box or segmentation mask to image
            xywh = self.str_to_xywh(row["xywh"])
            if detect_type == "bbox":
                image = add_bounding_boxes(image, [xywh], width=2)
            elif detect_type == "segment":
                image = add_segmentation_masks(image, [segmentation_to_box(xywh)])

            image_paths.append(row["image_path"])
            images.append(image)

        plot_images(images, texts=list(range(len(images))), figsize=(15, 20))
        for i, image_path in enumerate(image_paths):
            print(i, "-", image_path)

    def show_image(self, image_path: Path | str) -> None:
        df = self.get_rows(image_path)
        print(df.to_string())
        image = Image.open(df["image_path"].iloc[0])
        image = resize_image(image, 1080)
        for i, row in df.iterrows():
            xywh = self.str_to_xywh(row["xywh"])
            image = add_bounding_boxes(image, [xywh], width=2)
            image = add_text(
                image,
                boxes=[xywh],
                texts=[f'{i} - {row["action"]}'],
                text_color=[colors[i % len(colors)]],
            )
        image.show()
