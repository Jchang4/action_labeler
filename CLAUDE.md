# CLAUDE.md

## Project Overview

ActionLabeler is an autodistillation framework that uses VLMs to label the actions of people in images.

## Pipeline

1. User runs a pre-trained YOLO model (ultralytics) to get detections and segmentation masks
2. Pre-computed bounding boxes, image preprocessors, filters, and a prompt are used to classify actions via a VLM
3. The VLM classifies the actions of each bounding box/segmentation mask

## Expected Dataset Structure

```
dataset_folder/
  images/    # jpgs, jpegs, pngs, etc.
  detect/    # bounding box detections (ultralytics format: class_id, x_center, y_center, box_width, box_height normalized 0-1)
  segments/  # segmentation masks (ultralytics format)
```

Examples: `../datasets/human/sitting/`

## Core Components

- **Model**: VLM integration (e.g. llama.cpp models)
- **Preprocessors**: Augment images before VLM inference (crop, add bounding boxes, add labels, etc.)
- **Filters**: Remove individual detections or entire images from processing (e.g. bounding box too small)
- **Prompt**: Instructions sent to the VLM for action classification
- **ResponseParser**: Parses VLM output (handles JSON extraction, triple backticks, extra text) and optionally types into Python dataclasses

## Usage Pattern

```python
ActionLabeler(
    model=SomeLlamaCppModel,
    prompt=SomePrompt,
    preprocessors=[...],
    filters=[...],
    response_parser=SomeResponseParser(),
)
```

## Commands

```bash
uv pip install -e . --python /home/justin/machine_learning/yolo/.venv/bin/python
```
