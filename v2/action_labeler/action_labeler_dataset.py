from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw

from action_labeler.helpers import (
    add_bounding_boxes,
    add_segmentation_masks,
    resize_image,
    segmentation_to_box,
    xywh_to_xyxy,
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
    dataset_path: Path
    file_name: str = "actions.pickle"
    df: pd.DataFrame

    def __init__(self, dataset_path: Path, file_name: str = "actions.pickle"):
        self.dataset_path = dataset_path
        self.file_name = file_name
        self.df = self._load_df()

    @property
    def image_paths(self) -> list[Path]:
        return self.df["image_path"].unique().tolist()

    @property
    def actions(self) -> list[str]:
        return self.df["action"].unique().tolist()

    def _load_df(self) -> pd.DataFrame:
        if not Path(self.dataset_path / self.file_name).exists():
            return pd.DataFrame({"image_path": [], "xywh": [], "action": []})
        df: pd.DataFrame = pd.read_pickle(self.dataset_path / self.file_name)
        df["image_path"] = df["image_path"].apply(Path)
        return df

    def save(self) -> None:
        self.df.to_pickle(self.dataset_path / self.file_name)
        print(f"Saved {len(self.df)} rows to {self.dataset_path / self.file_name}")

    def add_row(self, image_path: Path, xywh: list[float], action: str) -> None:
        xywh_str = self.xywh_to_str(xywh)
        # Update row if it already exists
        if self.does_row_exist(image_path, xywh):
            self.df.loc[
                (self.df["image_path"] == image_path) & (self.df["xywh"] == xywh_str),
                "action",
            ] = self.sanitize_action(action)
        else:
            self.df = pd.concat(
                [
                    self.df,
                    pd.DataFrame(
                        {
                            "image_path": [image_path],
                            "xywh": [xywh_str],
                            "action": [self.sanitize_action(action)],
                        }
                    ),
                ],
                ignore_index=True,
            )

    def update_row_by_index(self, index: int, action: str) -> None:
        self.df.loc[index, "action"] = self.sanitize_action(action)

    def update_class(self, old_class: str, new_class: str) -> None:
        self.df.loc[self.df["action"] == old_class, "action"] = new_class

    def drop_classes(self, classes: list[str]) -> None:
        self.df = self.df[~self.df["action"].isin(classes)]

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

    def does_row_exist(self, image_path: Path, xywh: list[float]) -> bool:
        xywh_str = self.xywh_to_str(xywh)
        return (
            image_path in self.df["image_path"].values
            and xywh_str in self.df[self.df["image_path"] == image_path]["xywh"].values
        )

    def get_rows(self, image_path: Path) -> pd.DataFrame:
        return self.df[self.df["image_path"] == image_path]

    @staticmethod
    def xywh_to_str(xywh: list[float]) -> str:
        assert all([0.0 <= n <= 1.0 for n in xywh]), "xywh must be between 0 and 1"
        return " ".join([f"{n:.6f}" for n in xywh])

    @staticmethod
    def str_to_xywh(xywh_str: str) -> list[float]:
        return [float(coord) for coord in xywh_str.split(" ")]

    @staticmethod
    def sanitize_action(action: str) -> str:
        return action.replace("action:", "").strip()

    def show_class(
        self, class_name: str, num_samples: int = 5, detect_type: str = "bbox"
    ) -> None:
        """Plot a class from the dataset with bounding boxes"""
        assert detect_type in [
            "bbox",
            "segment",
        ], "detect_type must be 'bbox' or 'segment'"

        df = self.df[self.df["action"] == class_name]
        print("Number of samples:", len(df))
        for _, row in df.sample(min(num_samples, len(df)), replace=False).iterrows():
            image = Image.open(row["image_path"])
            image = resize_image(image, 640)
            xywh = self.str_to_xywh(row["xywh"])
            if detect_type == "bbox":
                image = add_bounding_boxes(image, [xywh], width=2)
            elif detect_type == "segment":
                image = add_segmentation_masks(image, [segmentation_to_box(xywh)])
            print(row["image_path"])
            image.show()

    def show_image(self, image_name: str) -> None:
        df = self.df[self.df["image_path"].apply(lambda x: image_name in str(x))]
        print(df.to_string())
        image = Image.open(df["image_path"].iloc[0])
        image = resize_image(image, 1080)
        for i, row in df.iterrows():
            xywh = self.str_to_xywh(row["xywh"])
            xyxy = xywh_to_xyxy(xywh, image.size)
            draw = ImageDraw.Draw(image)
            draw.rectangle(xyxy, outline=colors[i % len(colors)], width=2)
            draw.text(
                (xyxy[0], xyxy[1]),
                f'{i} - {row["action"]}',
                fill=colors[i % len(colors)],
            )
        image.show()
