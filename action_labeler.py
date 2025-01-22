import json
from pathlib import Path

import numpy as np
import supervision as sv
from matplotlib import pyplot as plt
from PIL import Image
from tqdm.auto import tqdm

from .base import BaseActionLabeler
from .helpers import (
    get_detection,
    parse_response,
    save_pickle,
    segmentation_to_mask,
    segmentation_to_xyxy,
    xywh_to_xyxy,
    xyxy_to_mask,
    xyxy_to_xywh,
)
from .preprocessors import (
    BoxImagePreprocessor,
    CropImagePreprocessor,
    MinResizeImagePreprocessor,
)


class DetectActionLabeler(BaseActionLabeler):
    def img_path_to_detections(
        self, image: Image.Image, img_path: Path
    ) -> sv.Detections:
        txt_path = self.get_detect_path(img_path)
        if not txt_path.exists():
            return sv.Detections.empty()

        detections = [
            [float(x) for x in line.split(" ") if x]
            for line in txt_path.read_text().splitlines()
            if line
        ]
        if len(detections) == 0 or len(detections[0]) == 0:
            return sv.Detections.empty()

        xywhs = [detection[1:] for detection in detections]
        xyxys = [xywh_to_xyxy(image, xywh) for xywh in xywhs]
        mask = [xyxy_to_mask(image, xyxy) for xyxy in xyxys]

        # Convert to Detections
        detections = get_detection(xyxys, mask)

        return detections


class SegmentationActionLabeler(BaseActionLabeler):
    def img_path_to_detections(
        self, image: Image.Image, img_path: Path
    ) -> sv.Detections:
        txt_path = self.get_segment_path(img_path)
        if not txt_path.exists():
            return sv.Detections.empty()

        segmentations = [
            [float(x) for x in line.split(" ") if x]
            for line in txt_path.read_text().splitlines()
            if line
        ]
        if len(segmentations) == 0 or len(segmentations[0]) == 0:
            return sv.Detections.empty()

        xyxys = [segmentation_to_xyxy(image, segment) for segment in segmentations]
        mask = [segmentation_to_mask(image, segment) for segment in segmentations]

        # Convert to Detections
        detections = get_detection(xyxys, mask)

        return detections


class MultiImageDetectActionLabeler(DetectActionLabeler):
    def label(self):
        """Label images."""

        image_paths = self.images_path.iterdir()
        image_paths = np.random.permutation(list(image_paths))

        for img_path in tqdm(image_paths, total=len(image_paths)):
            if not img_path.exists():
                continue

            # Raw Data - we reuse this later
            image = self.get_image(img_path)
            detections = self.img_path_to_detections(image, img_path)
            if detections.is_empty():
                continue

            for i in range(len(detections.xyxy)):
                xywh = xyxy_to_xywh(image, detections.xyxy[i])
                box_key = " ".join(map(str, xywh))

                # Skip if already labeled
                if (
                    str(img_path) in self.results
                    and box_key in self.results[str(img_path)]
                ):
                    continue

                # Skip detection
                is_valid = True
                for filter in self.filters:
                    if not filter.is_valid(image, i, detections):
                        is_valid = False
                        break
                if not is_valid:
                    continue

                # Get Cropped Images
                cropped_images = self.get_cropped_images(image, i, detections)

                # Predict
                try:
                    if self.verbose:
                        print(self.prompt.prompt(img_path, image, i, detections))
                    raw_response = self.model.predict(
                        cropped_images,
                        self.prompt.prompt(img_path, image, i, detections),
                    )
                    output = parse_response(raw_response)
                    # Save Results
                    self.results[str(img_path)][box_key] = output
                except json.JSONDecodeError:
                    print("Error parsing response")
                    print(raw_response)
                    continue
                except Exception as e:
                    print(f"Error: {e}")
                    continue

                if self.verbose:
                    # Print images in a row
                    fig, axs = plt.subplots(1, len(cropped_images), figsize=(20, 5))
                    for ax, cropped_image in zip(axs, cropped_images):
                        ax.imshow(cropped_image)
                        ax.axis("off")
                    plt.show()
                    print(output)

            if len(self.results) % self.save_every == 0:
                self.save_results()
                print(f"Saved {len(self.results)} images")

        # self.remove_invalid_classes()
        self.save_results()

    def get_cropped_images(
        self, image: Image.Image, index: int, detections: sv.Detections
    ) -> tuple[
        Image.Image,
        Image.Image,
    ]:
        full_size_image = image.copy()
        for prepreprocessor in self.preprocessors:
            if isinstance(prepreprocessor, CropImagePreprocessor):
                full_size_image = CropImagePreprocessor(
                    force_crop=False, buffer_pct=0.5
                ).preprocess(full_size_image, index, detections)
                continue
            full_size_image = prepreprocessor.preprocess(
                full_size_image, index, detections
            )

        cropped_image = self.preprocess_image(image.copy(), index, detections)

        return full_size_image, cropped_image
