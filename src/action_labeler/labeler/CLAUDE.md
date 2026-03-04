# Labeler Package

Labeling pipelines that orchestrate VLM inference over a dataset of images and detections.

## How It Works

`ActionLabeler.run()` handles the shared pipeline:

1. Iterate over images in `dataset_path/images/`
2. Load detections from `dataset_path/detect/<stem>.txt`
3. Apply filters — skip the image if any filter rejects
4. Skip fully-labeled images (all detections already in dataset)
5. Call `label(image, detections)` — **this is what subclasses implement**
6. Add results to dataset via `dataset.add_rows()`

Subclasses only implement `label()`. Everything else (file I/O, filtering, resume, error handling) is inherited.

## File Layout

| File | Class | Strategy |
|---|---|---|
| `base.py` | `ActionLabeler` | Abstract base — shared `run()` loop and helpers |
| `all_at_once.py` | `AllAtOnceLabeler` | One VLM call per image, all detections at once |
| `single_detection.py` | `SingleDetectionLabeler` | One VLM call per detection |
| `multi_view.py` | `MultiViewLabeler` | One VLM call per detection, multiple preprocessed images |

## The `label()` Contract

```python
def label(self, image: Image.Image, detections: list[Detection]) -> list[BaseModel | str]:
```

- **Input**: the raw image (not preprocessed) and all detections for that image
- **Output**: exactly one response per detection, positionally matched
- **Responsibilities**: call `_apply_preprocessors()`, `model.predict()`, and `prompt.parse()` as needed

## Preprocessors

`preprocessors: list[list[BasePreprocessor]]` — a list of chains. Each chain produces one image.

```python
# Single chain → 1 image sent to VLM
preprocessors=[[crop, resize]]

# Multiple chains → 3 images sent to VLM
preprocessors=[[crop], [mask], [bbox_overlay]]
```

`_apply_preprocessors(image, detections)` runs each chain on an independent copy and returns `list[Image.Image]`.

How each labeler uses preprocessors:
- **AllAtOnceLabeler**: `_apply_preprocessors(image, detections)` — all detections at once
- **SingleDetectionLabeler**: `_apply_preprocessors(image, [det])` — one detection at a time
- **MultiViewLabeler**: `_apply_preprocessors(image, [det])` — one detection, multiple chains (≥2 required)

## Built-in Labelers

### AllAtOnceLabeler

One VLM call per image. The prompt's `response_model` must have a list field containing one item per detection. Set `response_field` (default `"actions"`) to name that field.

```python
AllAtOnceLabeler(
    model=model,
    prompt=prompt,  # response_model has a list field
    preprocessors=[[draw_bboxes]],
    response_field="actions",
)
```

### SingleDetectionLabeler

One VLM call per detection. Preprocessors receive `[det]` so they operate on individual detections (e.g. crop to one person).

```python
SingleDetectionLabeler(
    model=model,
    prompt=prompt,
    preprocessors=[[crop, resize]],
)
```

### MultiViewLabeler

One VLM call per detection with multiple images. Requires ≥2 preprocessor chains. Each chain produces a different view (e.g. cropped, masked, annotated).

```python
MultiViewLabeler(
    model=model,
    prompt=prompt,
    preprocessors=[[crop], [mask], [bbox_overlay]],
)
```

## Creating a New Labeler

1. Create `src/action_labeler/labeler/your_labeler.py`
2. Extend `ActionLabeler` and implement `label()`
3. Use the base class helpers inside `label()`:
   - `self._apply_preprocessors(image, detections)` → `list[Image.Image]`
   - `self.prompt.format_system()` → system prompt string
   - `self.prompt.format_user()` → user prompt string
   - `self.model.predict(system, user, images)` → raw text response
   - `self.prompt.parse(text)` → `BaseModel | str`
4. Return exactly one response per detection
5. Add the class to `__init__.py`
6. Add tests in `tests/labeler/test_your_labeler.py`

### Example

```python
from PIL import Image
from pydantic import BaseModel

from .base import ActionLabeler
from ..types import Detection


class MyLabeler(ActionLabeler):
    def label(
        self, image: Image.Image, detections: list[Detection]
    ) -> list[BaseModel | str]:
        images = self._apply_preprocessors(image, detections)
        system = self.prompt.format_system()
        user = self.prompt.format_user()
        text = self.model.predict(system, user, images)
        return [self.prompt.parse(text)] * len(detections)
```

## Resume Support

`run()` accepts an optional `dataset` argument. When provided:
- Fully-labeled images are skipped (all detections already in dataset)
- Partially-labeled images are re-labeled (all detections sent to `label()`, `add_rows()` deduplicates keeping the latest responses)

```python
dataset = labeler.run(dataset_path)
dataset.save(Path("checkpoint.pkl"))

# Later, resume:
dataset = Dataset.load(Path("checkpoint.pkl"))
dataset = labeler.run(dataset_path, dataset=dataset)
```

## Tests

Tests live in `tests/labeler/` and mirror the module structure. Use `MagicMock` for model and prompt. Test that:
- `model.predict` is called the expected number of times
- `_apply_preprocessors` receives the correct detections
- Responses are returned in the correct order
- Constructor validation works (if any)
