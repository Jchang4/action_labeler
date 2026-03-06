# Dataset Package

Output container for `ActionLabeler.run()`. Wraps a pandas DataFrame with one row per (image, detection) pair.

## File Layout

| File | Purpose |
|---|---|
| `columns.py` | `DatasetColumns` — column name constants. Single source of truth for the schema. |
| `dataset.py` | `Dataset` — core class: construction, validation, save/load, combine, helpers. Inherits from all mixins. |
| `filter.py` | `DatasetFilterMixin` — in-place mutations: `balance`, `remove_class`, `rename_class`. |
| `plot.py` | `DatasetPlotMixin` — `plot_grid` (sample images with bboxes), `plot_distribution` (action bar chart). |
| `export.py` | `DatasetExportMixin` — `export_yolov8` with stratified train/valid split and `data.yaml`. |

`columns.py` exists as a separate file to break a circular import between `dataset.py` and `filter.py`.

## Schema

Every DataFrame must have these columns (enforced by `_validate()`):

| Column | Type | Description |
|---|---|---|
| `image_path` | `Path` | Path to the source image |
| `detection_index` | `int` | Index of this detection within its image (0, 1, 2...) |
| `detection` | `Detection` | The YOLO detection object |
| `action` | `str` | The classified action label |
| `response` | `ActionResponse \| str` | Raw VLM response — typically an `ActionResponse` subclass instance |

## Usage

```python
# Returned by ActionLabeler.run()
dataset = labeler.run(dataset_path)

# Build incrementally
dataset = Dataset()
dataset.add_rows(image_path, detections, results)  # results: list[LabelResult]

# Check for existing rows (useful for resuming)
if not dataset.has_row(image_path, detection):
    ...

# Extract a field from all response objects
confidence = dataset.response_field("confidence")

# Filter
dataset.remove_class("unknown")                     # drop all images containing class
dataset.remove_class("unknown", keep_image=True)     # drop only matching detections
dataset.rename_class("old_name", "new_name")
dataset.balance(seed=42)                             # downsample to min class count
dataset.balance(upsample={"rare_class": 2.0})        # per-class multipliers

# Combine multiple datasets
combined = Dataset.combine(dataset1, dataset2)

# Persist
dataset.save(Path("results.pkl"))
dataset = Dataset.load(Path("results.pkl"))

# Visualize
dataset.plot_distribution()
dataset.plot_grid(n=16, action="walking", seed=42)

# Export to YOLOv8
dataset.export_yolov8(Path("output"), val_ratio=0.2, seed=42)
```

## Mixin Rules

- **Mixins access `self.df` directly** — they annotate `df: pd.DataFrame` for type checking but don't own it.
- **Filter methods mutate in-place** and call `reset_index(drop=True)` after dropping rows.
- **Plot methods** are read-only — never mutate `self.df`.
- **Export methods** are read-only — never mutate `self.df`.
- **Always use `DatasetColumns` constants** for column access, never raw strings.

## Adding a New Mixin

1. Create `src/action_labeler/dataset/your_mixin.py`.
2. Define a class with `df: pd.DataFrame` annotation and your methods.
3. Import `DatasetColumns` from `columns.py` (not from `dataset.py` — avoids circular imports).
4. Add the mixin to the `Dataset` class bases in `dataset.py`.
5. Add tests in `tests/dataset/test_your_mixin.py`.

## Adding a New Filter Method

Add the method to `DatasetFilterMixin` in `filter.py`. It should:
- Use `DatasetColumns` constants for column names
- Mutate `self.df` in-place
- Call `.reset_index(drop=True)` on the result
- Return `None`

## Adding a Column

1. Add the constant to `DatasetColumns` in `columns.py`.
2. If it should always be present, add it to `DatasetColumns.REQUIRED`.
3. Update `add_rows()` in `dataset.py` to populate it.

## Tests

Tests live in `tests/dataset/` and mirror the module structure:
- `test_dataset.py` — construction, validation, save/load, combine, helpers, `add_rows`, `has_row`
- `test_filter.py` — one test class per filter method (`balance`, `remove_class`, `rename_class`)
- `test_export.py` — YOLOv8 export, stratified split, `data.yaml` generation
- `test_plot.py` — plot_grid, plot_distribution
