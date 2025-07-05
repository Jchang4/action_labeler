from pathlib import Path

import pandas as pd
from PIL import Image
from tqdm.auto import tqdm

from action_labeler.detections.detection import Detection
from action_labeler.helpers import get_image_paths, image_to_txt_path, load_image
from action_labeler.labeler.detection_labeler import ActionLabeler


class SegmentationLabeler(ActionLabeler):
    def label(self):
        print(f"Starting with {len(self.dataset)} labeled images")

        image_paths = get_image_paths(self.folder)

        for image_path in tqdm(image_paths):
            image = load_image(image_path)
            txt_path = image_to_txt_path(image_path, detection_type="segment")
            if not txt_path.exists():
                print(f"No detections for {str(image_path)}")
                continue

            detections = Detection.from_text_path(txt_path, image.size)
            if detections.is_empty():
                print(f"No detections for {str(image_path)}")
                continue

            for i in range(len(detections.xyxy)):
                if not self.apply_filters(image, i, detections):
                    continue
                elif self.dataset.does_row_exist(image_path, detections.xywhn[i]):
                    continue

                preprocessed_image = self.apply_preprocessors(image, i, detections)

                raw_model_output = self.apply_model(
                    image_path, [preprocessed_image], i, detections
                )
                self.add_results(
                    image_path,
                    detections.xywhn[i],
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
