from pathlib import Path

import yaml


def get_data_yaml(dataset_path: Path | str, verbose: bool = False) -> dict:
    """Get the data.yaml file from a dataset path.

    Args:
        dataset_path (Path | str): The path to the dataset.
        verbose (bool, optional): Whether to print the data.yaml file. Defaults to False.

    Returns:
        dict: The data.yaml file.
    """
    dataset_path = Path(dataset_path)
    data_yaml_path = dataset_path / "data.yaml"
    if not data_yaml_path.exists():
        raise FileNotFoundError(f"Data YAML file not found at {data_yaml_path}")

    with open(data_yaml_path, "r") as f:
        data_yaml = yaml.safe_load(f)

    if verbose:
        print(f"Dataset path: {dataset_path}")
        print("Class Names:")
        for i, class_name in enumerate(data_yaml["names"]):
            print(f"{i}: {class_name}")
        print(f"Number of classes: {len(data_yaml['names'])}")

    return data_yaml
