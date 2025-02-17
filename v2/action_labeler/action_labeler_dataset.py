from pathlib import Path

import pandas as pd


class ActionLabelDataset:
    dataset_path: Path
    file_name: str = "actions.pickle"
    df: pd.DataFrame

    def __init__(self, dataset_path: Path, file_name: str = "actions.pickle"):
        self.dataset_path = dataset_path
        self.file_name = file_name
        self.df = self.load_df()

    def load_df(self) -> pd.DataFrame:
        if not Path(self.dataset_path / self.file_name).exists():
            return pd.DataFrame({"image_path": [], "xywh": [], "action": []})
        return pd.read_pickle(self.dataset_path / self.file_name)

    def save(self) -> None:
        self.df.to_pickle(self.dataset_path / self.file_name)

    def add_row(self, image_path: Path, xywh: list[float], action: str) -> None:
        xywh_str = self.xywh_to_str(xywh)
        self.df = pd.concat(
            [
                self.df,
                pd.DataFrame(
                    {
                        "image_path": [image_path],
                        "xywh": [xywh_str],
                        "action": [self.sanitize_action(action)],
                    }
                ),
            ],
            ignore_index=True,
        )

    @staticmethod
    def xywh_to_str(xywh: list[float]) -> str:
        assert len(xywh) == 4, "xywh must be a list of 4 floats"
        assert all([0.0 <= n <= 1.0 for n in xywh]), "xywh must be between 0 and 1"
        return f"{xywh[0]:.6f} {xywh[1]:.6f} {xywh[2]:.6f} {xywh[3]:.6f}"

    @staticmethod
    def str_to_xywh(xywh_str: str) -> list[float]:
        return [float(coord) for coord in xywh_str.split(" ")]

    @staticmethod
    def sanitize_action(action: str) -> str:
        return action.replace("action:", "").strip()
