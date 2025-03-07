import json
from pathlib import Path

import yaml
from PIL import Image


def load_data_yaml(data_yaml_path: Path) -> dict:
    """Load the dataset YAML file containing category names and other info."""
    with open(data_yaml_path, "r") as f:
        data = yaml.safe_load(f)
    return data


def parse_yolo_label_line(line: str, img_width: int, img_height: int):
    """
    Parse a YOLO v8 label line in the format:
      <class> <x_center> <y_center> <width> <height>
    Convert normalized values to absolute coordinates (COCO style: [top_left_x, top_left_y, width, height]).
    """
    parts = line.strip().split()
    if len(parts) != 5:
        return None
    cls, x_center, y_center, w_norm, h_norm = parts
    cls = int(cls)
    x_center = float(x_center)
    y_center = float(y_center)
    w_norm = float(w_norm)
    h_norm = float(h_norm)
    x_top_left = (x_center - w_norm / 2) * img_width
    y_top_left = (y_center - h_norm / 2) * img_height
    bbox_width = w_norm * img_width
    bbox_height = h_norm * img_height
    return cls, [x_top_left, y_top_left, bbox_width, bbox_height]


def convert_split(split: str, dataset_dir: Path, start_img_id: int, start_ann_id: int):
    """
    Convert one split (train or valid) from YOLOv8 to COCO.
    Returns the list of image entries, annotation entries, and updated counters.
    """
    images_dir = dataset_dir / split / "images"
    labels_dir = dataset_dir / split / "labels"

    images_list = []
    annotations_list = []
    ann_id = start_ann_id
    img_id = start_img_id

    # Process each image file in the split's images directory
    for img_file in sorted(images_dir.glob("*.*")):
        if img_file.suffix.lower() not in [".jpg", ".jpeg", ".png"]:
            continue
        with Image.open(img_file) as img:
            w, h = img.size
        image_entry = {
            "id": img_id,
            "file_name": str(img_file.relative_to(dataset_dir)),
            "width": w,
            "height": h,
        }
        images_list.append(image_entry)

        # Corresponding YOLO label file (if it exists)
        label_file = labels_dir / f"{img_file.stem}.txt"
        if label_file.exists():
            with open(label_file, "r") as lf:
                for line in lf:
                    if line.strip() == "":
                        continue
                    parsed = parse_yolo_label_line(line, w, h)
                    if parsed is None:
                        continue
                    cls, bbox = parsed
                    # Convert YOLO's 0-indexed class to COCO (typically starting at 1)
                    coco_cls = cls
                    ann_entry = {
                        "id": ann_id,
                        "image_id": img_id,
                        "category_id": coco_cls,
                        "bbox": [round(b, 2) for b in bbox],
                        "area": round(bbox[2] * bbox[3], 2),
                        "iscrowd": 0,
                    }
                    annotations_list.append(ann_entry)
                    ann_id += 1
        img_id += 1
    return images_list, annotations_list, img_id, ann_id


def convert_yolov8_to_coco(dataset_dir: Path, output_dir: Path):
    """
    Main function to convert a YOLO v8 dataset to COCO format.

    Assumes the dataset directory has:
      - data.yaml
      - train/images, train/labels
      - valid/images, valid/labels

    The COCO JSON files will be saved in output_dir/annotations as train_labels.json and valid_labels.json.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    annotations_out = output_dir / "annotations"
    annotations_out.mkdir(parents=True, exist_ok=True)

    # Load data.yaml to get category names
    data_yaml_path = dataset_dir / "data.yaml"
    data_yaml = load_data_yaml(data_yaml_path)
    names = data_yaml.get("names", [])
    categories = [
        {"id": idx + 1, "name": name, "supercategory": "none"}
        for idx, name in enumerate(names)
    ]

    # Basic info and licenses (customize as needed)
    info = {
        "description": "Converted YOLOv8 dataset to COCO format",
        "version": "1.0",
        "year": 2025,
        "contributor": "",
        "date_created": "",
    }
    licenses = []

    # Process each split (train and valid)
    for split in ["train", "valid"]:
        start_img_id = 1
        start_ann_id = 1
        images, annotations, _, _ = convert_split(
            split, dataset_dir, start_img_id, start_ann_id
        )
        coco_dict = {
            "info": info,
            "licenses": licenses,
            "images": images,
            "annotations": annotations,
            "categories": categories,
        }
        out_file = annotations_out / f"{split}_labels.json"
        with open(out_file, "w") as f:
            json.dump(coco_dict, f, indent=4)
        print(
            f"Converted {split} split: {len(images)} images and {len(annotations)} annotations saved to {out_file}"
        )


# Example usage:
# DATASET_DIR = Path("datasets/description_action_dataset_balanced")
# OUTPUT_DIR = Path("datasets/coco_human_dataset_balanced")
# convert_yolov8_to_coco(DATASET_DIR, OUTPUT_DIR)
