from pathlib import Path

from PIL import Image
from tqdm.auto import tqdm

from action_labeler.dataclasses import DetectionType
from action_labeler.detections.detection import Detection
from action_labeler.filters.base import IFilter
from action_labeler.helpers import get_image_paths, load_image
from action_labeler.helpers.detections_helpers import image_to_txt_path
from action_labeler.labeler.dataset import LabelerDataset
from action_labeler.models.base import IVisionLanguageModel
from action_labeler.preprocessors.base import IPreprocessor
from action_labeler.prompt.base import BasePrompt


class ActionLabeler:
    folder: Path
    prompt: BasePrompt
    model: IVisionLanguageModel
    filters: list[IFilter]
    preprocessors: list[IPreprocessor]
    detection_type: DetectionType = DetectionType.DETECT

    # Options
    verbose: bool = False
    save_every: int = 50
    save_filename: str = "classification.pickle"

    # Class variables
    dataset: LabelerDataset

    def __init__(
        self,
        folder: Path,
        prompt: BasePrompt,
        model: IVisionLanguageModel,
        filters: list[IFilter],
        preprocessors: list[IPreprocessor],
        detection_type: DetectionType = DetectionType.DETECT,
        verbose: bool = False,
        save_every: int = 50,
        save_filename: str = "classification.pickle",
    ):
        self.folder = folder
        self.prompt = prompt
        self.model = model
        self.filters = filters
        self.preprocessors = preprocessors
        self.detection_type = detection_type

        self.verbose = verbose
        self.save_every = save_every
        self.save_filename = save_filename

        # Load results file if it exists otherwise create an empty dataframe
        self.dataset = LabelerDataset(
            folder,
            classes=prompt.classes,
            filename=save_filename,
        )

    def label(self):
        print(f"Starting with {len(self.dataset)} labeled images")

        image_paths = get_image_paths(self.folder)

        for image_path in tqdm(image_paths):
            image = load_image(image_path)
            txt_path = self.get_txt_path(image_path)
            if txt_path is None:
                print(f"No detections for {str(image_path)}")
                continue

            detections = Detection.from_text_path(txt_path, image.size)
            if detections.is_empty():
                print(f"No detections for {str(image_path)}")
                continue

            for i in range(len(detections.xyxy)):
                if not self.apply_filters(image, i, detections):
                    continue
                elif self.dataset.does_row_exist(image_path, detections.xywh[i]):
                    continue

                preprocessed_image = self.apply_preprocessors(image, i, detections)

                raw_model_output = self.apply_model(
                    image_path, [preprocessed_image], i, detections
                )
                self.add_results(
                    image_path,
                    detections.xywh[i],
                    detections.segmentation_points[i],
                    raw_model_output,
                )

                if (
                    self.verbose
                    and len(self.dataset) % self.save_every == 0
                    and i == len(detections.xyxy) - 1
                ):
                    # Show the last detection
                    image.show()
                    preprocessed_image.show()
                    print(raw_model_output)

            if len(self.dataset) % self.save_every == 0:
                self.save_results()
                print(f"Saved {len(self.dataset)} images")

        self.save_results()
        print(f"Saved {len(self.dataset)} images")

    def get_txt_path(self, image_path: Path | str) -> Path | None:
        """
        Get the labels .txt path for the image.
        Returns None if the labels .txt path does not exist.

        This method can be overridden to return different txt paths for different
        label types, such as detection or segmentation.
        """
        txt_path = image_to_txt_path(
            image_path, detection_type=self.detection_type.value
        )
        if not txt_path.exists():
            return None
        return txt_path

    def apply_filters(self, image: Image.Image, index: int, detections: Detection):
        for filter in self.filters:
            if not filter.is_valid(image, index, detections):
                return False
        return True

    def apply_preprocessors(
        self, image: Image.Image, index: int, detections: Detection
    ):
        for preprocessor in self.preprocessors:
            image = preprocessor.preprocess(image, index, detections)
        return image

    def apply_model(
        self,
        image_path: Path | str,
        images: list[Image.Image],
        index: int,
        detections: Detection,
    ):
        image_path = Path(image_path)
        prompt = self.prompt.prompt(index, detections, image_path)
        return self.model.predict(prompt, images)

    def save_results(self):
        self.dataset.save()

    def add_results(
        self,
        image_path: Path | str,
        xywh: list[float],
        segmentation_points: list[list[float]],
        model_output: str,
    ):
        image_path = Path(image_path)
        self.dataset.add_row(
            image_path=image_path,
            xywh=xywh,
            segmentation_points=segmentation_points,
            action=model_output,
        )

    def __repr__(self):
        return f"ActionLabeler(folder={self.folder})"
