import shutil
from pathlib import Path

from tqdm.auto import tqdm

try:
    from ultralytics import YOLO
except ImportError:
    raise ImportError(
        "Ultralytics requires the ultralytics package. Please install it with `pip install ultralytics`."
    )


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
