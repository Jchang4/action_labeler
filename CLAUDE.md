# CLAUDE.md

## Important

Do not use `yolo/action_labeler` as a reference. We are creating a new package for a reason.

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

## Architecture

```
src/action_labeler/
  types.py              # Detection dataclass (YOLO-format with pixel-space properties)
  models/               # VLM integrations (base interface + llama.cpp)
  preprocessors/        # Image transforms before VLM inference (bounding_box, resize)
  filters/              # Reject detections/images (aspect_ratio, detection_count, overlap)
  prompts/              # Prompt template with system/user messages and response parsing
  labeler/              # Orchestration layer (abstract base + 3 strategies)
    base.py             # ActionLabeler — shared run() loop, filtering, resume
    all_at_once.py      # One VLM call per image, all detections at once
    single_detection.py # One VLM call per detection
    multi_view.py       # One VLM call per detection, multiple preprocessed views
  dataset/              # Output container (pandas DataFrame wrapper)
    columns.py          # Column name constants (single source of truth)
    dataset.py          # Dataset class — construction, save/load, add_rows, has_row
    filter.py           # DatasetFilterMixin — in-place row filtering
    plot.py             # DatasetPlotMixin — stub, not yet implemented
```

## Core Components

- **Detection** (`types.py`): Dataclass for a single YOLO-format detection with normalized coordinates and pixel-space properties (x1, y1, x2, y2, xyxy)
- **Model** (`models/`): VLM integration — `predict(system, user, images)` returns raw text. Currently: `LlamaCppModel`
- **Preprocessors** (`preprocessors/`): Transform images before VLM inference. Configured as chains: `list[list[BasePreprocessor]]` where each inner list produces one image
- **Filters** (`filters/`): Accept/reject an image and its detections. Return `True` to keep, `False` to skip
- **Prompt** (`prompts/`): Template with `format_system()`, `format_user()`, and `parse(text)` for response parsing (supports Pydantic response models)
- **ActionLabeler** (`labeler/`): Abstract base with shared `run()` loop. Subclasses implement `label()` to define how detections map to VLM calls
- **Dataset** (`dataset/`): Output container wrapping a pandas DataFrame. One row per (image, detection) pair. Supports save/load (pickle), filtering, and resume

## Usage Pattern

```python
labeler = SingleDetectionLabeler(
    model=LlamaCppModel(...),
    prompt=Prompt(system="...", user="...", response_model=MyAction),
    preprocessors=[[crop, resize]],
    filters=[AspectRatioFilter(min_ratio=0.3)],
)

dataset = labeler.run(dataset_path)
dataset.save(Path("results.pkl"))

# Resume from checkpoint
dataset = Dataset.load(Path("results.pkl"))
dataset = labeler.run(dataset_path, dataset=dataset)
```

## Commands

```bash
uv pip install -e ".[dev]" --python /home/justin/machine_learning/yolo/.venv/bin/python
```

## Testing

Tests use **pytest** and live in `tests/`, mirroring the `src/action_labeler/` structure.

```bash
pytest tests/           # Run all tests
pytest tests/models/    # Run tests for a specific module
pytest -v -k "test_name"  # Run a specific test
```
