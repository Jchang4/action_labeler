# Dataset Package

Output container for `ActionLabeler.run()`. Wraps a pandas DataFrame with one row per (image, detection) pair.

## File Layout

| File | Purpose |
|---|---|
| `columns.py` | `DatasetColumns` — column name constants. Single source of truth for the schema. |
| `dataset.py` | `Dataset` — core class: construction, validation, save/load, helpers. Inherits from both mixins. |
| `filter.py` | `DatasetFilterMixin` — in-place row filtering (by class, by image). |
| `plot.py` | `DatasetPlotMixin` — stub, not yet implemented. |

`columns.py` exists as a separate file to break a circular import between `dataset.py` and `filter.py`.

## Schema

Every DataFrame must have these columns (enforced by `_validate()`):

| Column | Type | Description |
|---|---|---|
| `image_path` | `Path` | Path to the source image |
| `detection_index` | `int` | Index of this detection within its image (0, 1, 2...) |
| `detection` | `Detection` | The YOLO detection object |
| `response` | `BaseModel \| str` | Raw VLM response — typically a Pydantic model instance |

## Usage

```python
# Returned by ActionLabeler.run()
dataset = labeler.run(dataset_path)

# Extract a field from all response objects
actions = dataset.response_field("action")

# Filter
dataset.remove_class("unknown")
dataset.keep_classes(["walking", "sitting"])
dataset.remove_image(Path("bad_image.jpg"))

# Persist
dataset.save(Path("results.pkl"))
dataset = Dataset.load(Path("results.pkl"))
```

## Mixin Rules

- **Mixins access `self.df` directly** — they annotate `df: pd.DataFrame` for type checking but don't own it.
- **Filter methods mutate in-place** and call `reset_index(drop=True)` after dropping rows.
- **Plot methods** (when implemented) should be read-only — never mutate `self.df`.
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
3. Update `from_label_results()` in `dataset.py` to populate it (including the empty-results branch).

## Tests

Tests live in `tests/dataset/` and mirror the module structure:
- `test_dataset.py` — construction, validation, save/load, helpers
- `test_filter.py` — one test class per filter method
