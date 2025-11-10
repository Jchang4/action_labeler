"""Validation operations for YoloV8Dataset.

This module handles validation of dataset integrity and data quality.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from action_labeler.datasets.dataset_config import ValidationResult
from action_labeler.datasets.exceptions import DatasetValidationError


class YoloV8DatasetValidator:
    """Handles validation operations for YoloV8Dataset."""

    @staticmethod
    def validate_dataframe(
        df: pd.DataFrame,
        classes: list[str],
        check_files_exist: bool = True,
    ) -> ValidationResult:
        """Validate a dataset DataFrame.

        Args:
            df: DataFrame to validate
            classes: List of class names
            check_files_exist: Whether to check that image files exist

        Returns:
            ValidationResult with validation status and any errors/warnings
        """
        errors = []
        warnings = []

        # Check if dataframe is empty
        if len(df) == 0:
            warnings.append("Dataset is empty")
            return ValidationResult(is_valid=True, errors=errors, warnings=warnings)

        # Check required columns
        required_columns = {"dataset", "image_path", "xywh", "class_id"}
        missing_columns = required_columns - set(df.columns)
        if missing_columns:
            errors.append(f"Missing required columns: {missing_columns}")

        # Validate dataset column
        if "dataset" in df.columns:
            valid_splits = {"train", "valid"}
            invalid_splits = set(df["dataset"].unique()) - valid_splits
            if invalid_splits:
                errors.append(
                    f"Invalid dataset splits found: {invalid_splits}. "
                    f"Valid splits are: {valid_splits}"
                )

        # Validate class_id column
        if "class_id" in df.columns:
            # Check for null class_ids (background images are ok)
            non_null_class_ids = df["class_id"].dropna()

            if len(non_null_class_ids) > 0:
                # Check class IDs are within valid range
                min_class_id = non_null_class_ids.min()
                max_class_id = non_null_class_ids.max()
                num_classes = len(classes)

                if min_class_id < 0:
                    errors.append(f"Found negative class_id: {min_class_id}")

                if max_class_id >= num_classes:
                    errors.append(
                        f"Found class_id {max_class_id} but only {num_classes} "
                        f"classes defined (max valid ID: {num_classes - 1})"
                    )

                # Check for non-integer class IDs
                if not np.issubdtype(non_null_class_ids.dtype, np.integer):
                    if not all(non_null_class_ids == non_null_class_ids.astype(int)):
                        errors.append("class_id column contains non-integer values")

        # Validate xywh coordinates
        if "xywh" in df.columns:
            non_null_xywh = df["xywh"].dropna()

            for idx, xywh in non_null_xywh.items():
                if not isinstance(xywh, (list, tuple, np.ndarray)):
                    errors.append(f"Row {idx}: xywh is not a list/array: {type(xywh)}")
                    continue

                if len(xywh) != 4:
                    errors.append(
                        f"Row {idx}: xywh must have 4 values, got {len(xywh)}"
                    )
                    continue

                # Check coordinates are in [0, 1] range
                xywh_array = np.array(xywh)
                if np.any(xywh_array < 0) or np.any(xywh_array > 1):
                    warnings.append(
                        f"Row {idx}: xywh coordinates outside [0, 1] range: {xywh}"
                    )

                # Check for zero or negative width/height
                if xywh[2] <= 0 or xywh[3] <= 0:
                    errors.append(
                        f"Row {idx}: width and height must be positive, got {xywh}"
                    )

        # Check for duplicate detections (same image, same bbox, same class)
        if all(col in df.columns for col in ["image_path", "class_id"]):
            # Create a string representation of xywh for grouping
            df_check = df.copy()
            df_check["xywh_str"] = df_check["xywh"].apply(
                lambda x: str(x) if x is not None else ""
            )
            duplicates = df_check.duplicated(
                subset=["image_path", "class_id", "xywh_str"], keep=False
            )
            if duplicates.any():
                num_duplicates = duplicates.sum()
                warnings.append(
                    f"Found {num_duplicates} potential duplicate detections"
                )

        # Check if image files exist
        if check_files_exist and "image_path" in df.columns:
            missing_files = []
            for image_path in df["image_path"].unique():
                if not Path(image_path).exists():
                    missing_files.append(str(image_path))

            if missing_files:
                if len(missing_files) <= 5:
                    errors.append(f"Missing image files: {', '.join(missing_files)}")
                else:
                    errors.append(
                        f"Missing {len(missing_files)} image files. "
                        f"First 5: {', '.join(missing_files[:5])}"
                    )

        is_valid = len(errors) == 0
        return ValidationResult(is_valid=is_valid, errors=errors, warnings=warnings)

    @staticmethod
    def validate_classes(classes: list[str]) -> None:
        """Validate a list of class names.

        Args:
            classes: List of class names to validate

        Raises:
            DatasetValidationError: If classes are invalid
        """
        if not classes:
            raise DatasetValidationError("Classes list cannot be empty")

        if not isinstance(classes, list):
            raise DatasetValidationError(f"Classes must be a list, got {type(classes)}")

        # Check for duplicate class names
        if len(classes) != len(set(classes)):
            duplicates = [c for c in classes if classes.count(c) > 1]
            raise DatasetValidationError(
                f"Duplicate class names found: {set(duplicates)}"
            )

        # Check for empty class names
        if any(not c or not c.strip() for c in classes):
            raise DatasetValidationError("Class names cannot be empty or whitespace")
