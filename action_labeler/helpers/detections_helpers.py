import shutil
from pathlib import Path

import numpy as np
from tqdm.auto import tqdm

try:
    from ultralytics import YOLO
    from ultralytics.engine.results import Results
except ImportError:
    raise ImportError(
        "Ultralytics requires the ultralytics package. Please install it with `pip install ultralytics`."
    )


def ultralytics_labels_to_xywh(txt_path: Path | str) -> list[list[float]]:
    """Convert a Ultralytics labels txt file to a list of xywh boxes.

    Ultralytics labels formats: https://docs.ultralytics.com/modes/predict/#working-with-results

    Note: this method can also handle segmentation and keypoint detection.

    Args:
        txt_path (Path | str): The path to the txt file.

    Returns:
        list[list[float]]: A list of xywh boxes.
    """
    txt_path = Path(txt_path)
    return [
        [float(num) for num in line.split(" ")][1:]  # Skip the class id
        for line in txt_path.read_text().splitlines()
        if line and len(line.split(" ")) > 1
    ]


def xyxy_to_xywh(
    xyxy: tuple[float, float, float, float], image_size: tuple[int, int]
) -> tuple[float, float, float, float]:
    """Convert a list of xyxy coordinates to a list of xywh coordinates.

    Args:
        xyxy (tuple[float, float, float, float]): The xyxy coordinates.
        image_size (tuple[int, int]): The size of the image as (width, height).

    Returns:
        tuple[float, float, float, float]: The xywh coordinates.
    """
    x1, y1, x2, y2 = map(float, xyxy)
    image_width, image_height = image_size

    x_center = (x1 + x2) / 2
    y_center = (y1 + y2) / 2
    width = x2 - x1
    height = y2 - y1

    x_center /= image_width
    y_center /= image_height
    width /= image_width
    height /= image_height

    return [x_center, y_center, width, height]


def xyxys_to_xywhs(
    xyxys: list[tuple[float, float, float, float]], image_size: tuple[int, int]
) -> list[tuple[float, float, float, float]]:
    """Convert a list of xyxy coordinates to a list of xywh coordinates.

    Args:
        xyxys (list[tuple[float, float, float, float]]): The list of xyxy coordinates.
        image_size (tuple[int, int]): The size of the image as (width, height).

    Returns:
        list[tuple[float, float, float, float]]: The xywh coordinates.
    """
    return [xyxy_to_xywh(xyxy, image_size) for xyxy in xyxys]


def xywh_to_xyxy(
    xywh: tuple[float, float, float, float], image_size: tuple[int, int]
) -> tuple[float, float, float, float]:
    """Convert a list of xywh coordinates to a list of xyxy coordinates.

    Args:
        xywh (tuple[float, float, float, float]): The xywh coordinates as (x_center, y_center, width, height) in normalized coordinates.
        image_size (tuple[int, int]): The size of the image.

    Returns:
        list[int]: The xyxy coordinates.
    """
    x_center, y_center, width, height = map(float, xywh)
    image_width, image_height = image_size

    x1 = x_center - width / 2
    y1 = y_center - height / 2
    x2 = x_center + width / 2
    y2 = y_center + height / 2

    x1 *= image_width
    y1 *= image_height
    x2 *= image_width
    y2 *= image_height

    return x1, y1, x2, y2


def xywhs_to_xyxys(
    xywhs: list[tuple[float, float, float, float]], image_size: tuple[int, int]
) -> list[tuple[float, float, float, float]]:
    """Convert a list of xywh coordinates to a list of xyxy coordinates.

    Args:
        xywhs (list[tuple[float, float, float, float]]): The list of xywh coordinates.
        image_size (tuple[int, int]): The size of the image as (width, height).

    Returns:
        list[tuple[float, float, float, float]]: The xyxy coordinates.
    """
    return [xywh_to_xyxy(xywh, image_size) for xywh in xywhs]


def xywh_to_mask(
    xywh: tuple[float, float, float, float], image_size: tuple[int, int]
) -> list[int]:
    pass


def xyxy_to_mask(
    xyxy: tuple[float, float, float, float],
    image_size: tuple[int, int],
    buffer_px: int = 0,
) -> list[list[bool]]:
    """Convert xyxy boxes to a mask.

    Args:
        xyxy (tuple[float, float, float, float]): The xyxy coordinates.
        image_size (tuple[int, int]): The size of the image as (width, height).
        buffer_px (int, optional): The buffer in pixels. Defaults to 0.

    Returns:
        list[list[bool]]: The mask.
    """
    width, height = image_size
    mask = np.full((width, height), False, dtype=bool)
    x1, y1, x2, y2 = xyxy
    x1, y1, x2, y2 = (
        max(0, x1 - buffer_px),
        max(0, y1 - buffer_px),
        min(width, x2 + buffer_px),
        min(height, y2 + buffer_px),
    )

    mask[int(x1) : int(x2), int(y1) : int(y2)] = True
    return mask.tolist()


def xyxys_to_masks(
    xyxys: list[tuple[float, float, float, float]],
    image_size: tuple[int, int],
    buffer_px: int = 0,
) -> list[list[bool]]:
    """Convert a list of xyxy coordinates to a list of masks.

    Args:
        xyxys (list[tuple[float, float, float, float]]): The list of xyxy coordinates.
        image_size (tuple[int, int]): The size of the image as (width, height).
        buffer_px (int, optional): The buffer in pixels. Defaults to 0.

    Returns:
        list[list[bool]]: The list of masks.
    """
    return [xyxy_to_mask(xyxy, image_size, buffer_px) for xyxy in xyxys]


class Detection:
    xyxy: np.ndarray
    mask: np.ndarray | None
    class_id: np.ndarray
    image_size: tuple[int, int]  # as (width, height)

    def __init__(
        self,
        xyxy: np.ndarray,
        mask: np.ndarray,
        class_id: np.ndarray,
        image_size: tuple[int, int],
    ):
        self.xyxy = xyxy
        self.mask = mask
        self.class_id = class_id
        self.image_size = image_size

    @classmethod
    def from_ultralytics(cls, results: Results) -> "Detection":
        xyxy = results.boxes.xyxy.cpu().numpy()
        mask = results.masks.xy.cpu().numpy() if results.masks else None
        class_id = results.boxes.cls.cpu().numpy()
        # Ultralytics returns image size as (height, width)
        image_size = results.orig_shape[::-1]

        return cls(
            xyxy=xyxy,
            mask=mask,
            class_id=class_id,
            image_size=image_size,
        )

    @classmethod
    def from_text_path(
        cls, text_path: Path | str, image_size: tuple[int, int]
    ) -> "Detection":
        xywhs = ultralytics_labels_to_xywh(text_path)
        xyxys = xywhs_to_xyxys(xywhs, image_size)
        masks = xyxys_to_masks(xyxys, image_size)
        return cls(
            xyxy=xyxys,
            mask=masks,
            class_id=np.zeros(len(xyxys)),
            image_size=image_size,
        )

    @classmethod
    def empty(cls, image_size: tuple[int, int] = (0, 0)) -> "Detection":
        return cls(
            xyxy=np.array([]),
            mask=np.array([]),
            class_id=np.array([]),
            image_size=image_size,
        )

    @property
    def xywhn(self) -> np.ndarray:
        return np.array(xyxys_to_xywhs(self.xyxy, self.image_size))

    def __str__(self):
        return f"<Detection xyxys={len(self.xyxy)} masks={len(self.mask)} class_ids={len(self.class_id)} image_size={self.image_size}>"

    def __repr__(self):
        return self.__str__()


class DetectionManager:
    """Given a folder with an images/ directory, detect objects in the images and move the detections to a detect/ directory in the same folder.
    The detections are saved as .txt files in the detect/ directory.

    Args:
        image_dir (Path | str): The path to the image directory.
    """

    image_dir: Path

    def __init__(self, image_dir: Path | str):
        self.image_dir = Path(image_dir)

    def detect(
        self,
        model_name: str = "yolo12x.pt",
        detect_folder_name: str = "detect",
        batch: int = 64,
        classes: list[int] = [0],
        conf: float = 0.25,
    ):
        """Detect objects in the images and move the detections to a detect/ directory in the same folder.
        The detections are saved as .txt files in the detect/ directory.

        Args:
            model_name (str, optional): The name of the model to use. Defaults to "yolo12x.pt".
            detect_folder_name (str, optional): The name of the directory to save the detections to. Defaults to parent_folder/detect.
            batch (int, optional): The batch size to use for detection. Defaults to 64.
        """
        self._detect_folder(
            model_name=model_name,
            batch=batch,
            classes=classes,
            conf=conf,
        )
        self._move_detections(detect_folder_name=detect_folder_name)

    def _detect_folder(
        self,
        model_name: str = "yolo12x.pt",
        batch: int = 64,
        classes: list[int] = [0],
        conf: float = 0.25,
    ):
        model = YOLO(model_name)
        results = model.predict(
            self.image_dir / "images",
            classes=classes,
            stream=True,
            verbose=False,
            save_txt=True,
            project="runs/temp",
            batch=batch,
            conf=conf,
        )

        num_images = len(list((self.image_dir / "images").iterdir()))

        for result in tqdm(results, total=num_images):
            continue

    def _move_detections(self, detect_folder_name: str = "detect"):
        (self.image_dir / detect_folder_name).mkdir(exist_ok=True, parents=True)

        # Move file from runs/temp/predict/labels to folder/detect if it doesn't already exist
        for file in (self.image_dir / "images").iterdir():
            existing_txt_path = (
                self.image_dir / detect_folder_name / file.with_suffix(".txt").name
            )
            new_txt_path = (
                Path("runs/temp/predict/labels") / file.with_suffix(".txt").name
            )
            if existing_txt_path.exists() or not new_txt_path.exists():
                continue

            shutil.move(new_txt_path, existing_txt_path)

        shutil.rmtree("runs/temp/predict")
        if len(list(Path("runs/temp").iterdir())) == 0:
            shutil.rmtree("runs/temp")
