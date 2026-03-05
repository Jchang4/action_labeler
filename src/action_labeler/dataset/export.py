from __future__ import annotations

import random
import shutil
from pathlib import Path

import pandas as pd
import yaml

from .columns import DatasetColumns


class DatasetExportMixin:
    """Export methods for Dataset. Read-only — never mutates self.df."""

    df: pd.DataFrame

    def export_yolov8(
        self,
        output_dir: Path,
        val_ratio: float = 0.2,
        seed: int | None = None,
        overwrite: bool = False,
    ) -> dict:
        """Export dataset to YOLOv8 format with stratified train/valid split.

        Args:
            output_dir: Path to create the dataset folder.
            val_ratio: Fraction of image groups to assign to validation.
            seed: Random seed for reproducible splits.
            overwrite: If True, delete existing output_dir before writing.

        Returns:
            Dict with train_images, val_images, classes, and output_dir.
        """
        if len(self.df) == 0:
            raise ValueError("Cannot export empty dataset")

        output_dir = Path(output_dir)
        self._prepare_output_dir(output_dir, overwrite)

        # Step 1: Derive unique image names
        image_name_map = self._build_image_name_map()

        # Step 2: Build class ID mapping (alphabetically sorted)
        actions = sorted(self.df[DatasetColumns.ACTION].unique())
        class_map = {action: i for i, action in enumerate(actions)}

        # Step 3: Group by image_name
        image_name_col = self.df[DatasetColumns.IMAGE_PATH].map(image_name_map)
        groups = dict(list(self.df.groupby(image_name_col)))

        # Step 4: Greedy stratified split
        split_assignment = self._stratified_split(groups, val_ratio, seed)

        # Step 5: Write files
        for split in ("train", "valid"):
            (output_dir / split / "images").mkdir(parents=True)
            (output_dir / split / "labels").mkdir(parents=True)

        for image_name, split in split_assignment.items():
            group_df = groups[image_name]
            image_path = group_df[DatasetColumns.IMAGE_PATH].iloc[0]

            if not image_path.exists():
                raise FileNotFoundError(f"Source image not found: {image_path}")

            shutil.copy2(image_path, output_dir / split / "images" / image_name)

            label_stem = Path(image_name).stem
            label_path = output_dir / split / "labels" / f"{label_stem}.txt"
            lines = []
            for _, row in group_df.iterrows():
                det = row[DatasetColumns.DETECTION]
                class_id = class_map[row[DatasetColumns.ACTION]]
                lines.append(
                    f"{class_id} {det.x_center} {det.y_center} {det.width} {det.height}"
                )
            label_path.write_text("\n".join(lines) + "\n")

        # Step 6: Write data.yaml
        data_yaml = {
            "names": actions,
            "nc": len(actions),
            "path": output_dir.name,
            "train": "train/images",
            "val": "valid/images",
        }
        with open(output_dir / "data.yaml", "w") as f:
            yaml.dump(data_yaml, f, default_flow_style=False, sort_keys=True)

        train_count = sum(1 for s in split_assignment.values() if s == "train")
        val_count = sum(1 for s in split_assignment.values() if s == "valid")

        return {
            "train_images": train_count,
            "val_images": val_count,
            "classes": class_map,
            "output_dir": output_dir,
        }

    def _build_image_name_map(self) -> dict[Path, str]:
        """Map each unique image_path to a prefixed unique image name."""
        unique_paths = self.df[DatasetColumns.IMAGE_PATH].unique()
        name_map: dict[Path, str] = {}
        seen_names: dict[str, Path] = {}

        for path in unique_paths:
            prefix = path.parent.parent.name
            image_name = f"{prefix}_{path.name}"
            if image_name in seen_names and seen_names[image_name] != path:
                raise ValueError(
                    f"Image name collision: '{image_name}' maps to both "
                    f"'{seen_names[image_name]}' and '{path}'"
                )
            seen_names[image_name] = path
            name_map[path] = image_name

        return name_map

    def _stratified_split(
        self,
        groups: dict[str, pd.DataFrame],
        val_ratio: float,
        seed: int | None,
    ) -> dict[str, str]:
        """Greedy stratified split: assign each image group to train or valid.

        Processes rarest action classes first to ensure minority classes
        are well-represented in both splits.
        """
        rng = random.Random(seed)

        # Count global action frequencies
        action_counts = self.df[DatasetColumns.ACTION].value_counts().to_dict()

        # Assign each group a primary action (rarest action in that group)
        group_primary: dict[str, str] = {}
        for image_name, group_df in groups.items():
            group_actions = group_df[DatasetColumns.ACTION].unique()
            group_primary[image_name] = min(
                group_actions, key=lambda a: (action_counts[a], a)
            )

        # Sort actions by frequency ascending
        sorted_actions = sorted(action_counts, key=lambda a: (action_counts[a], a))

        assigned: dict[str, str] = {}

        for action in sorted_actions:
            # All groups with this primary action
            candidate_names = [
                name for name, primary in group_primary.items() if primary == action
            ]

            # Split into already-assigned and unassigned
            already_val = sum(1 for n in candidate_names if assigned.get(n) == "valid")
            unassigned = [n for n in candidate_names if n not in assigned]
            rng.shuffle(unassigned)

            total = len(candidate_names)
            target_val = round(total * val_ratio)
            # Ensure at least 1 in val if total > 1
            if total > 1:
                target_val = max(1, target_val)

            need_val = max(0, target_val - already_val)

            for name in unassigned[:need_val]:
                assigned[name] = "valid"
            for name in unassigned[need_val:]:
                assigned[name] = "train"

        return assigned

    @staticmethod
    def _prepare_output_dir(output_dir: Path, overwrite: bool) -> None:
        """Create or overwrite the output directory."""
        if output_dir.exists():
            if not overwrite:
                raise FileExistsError(
                    f"Output directory already exists: {output_dir}. "
                    f"Set overwrite=True to replace it."
                )
            shutil.rmtree(output_dir)
        output_dir.mkdir(parents=True)
