import shutil
import uuid
from pathlib import Path

from tqdm.auto import tqdm

try:
    from ultralytics import YOLO
except ImportError as exc:
    raise ImportError(
        "Ultralytics requires the ultralytics package. Please install it with `pip install ultralytics`."
    ) from exc


class DetectionManager:
    """Manager for running YOLO detection on image directories.

    This class handles running YOLO models on images and organizing the detection outputs.
    It can be instantiated once and used to detect objects in multiple image directories.
    Each detection run uses a unique UUID subfolder in the temp directory to avoid conflicts.

    Args:
        model_name (str, optional): YOLO model to use. Defaults to "yolo12x.pt".
        batch (int, optional): Batch size for detection. Defaults to 64.
        classes (list[int], optional): Classes to detect. Defaults to [0] (person).
        conf (float, optional): Confidence threshold. Defaults to 0.25.
        temp_dir (str, optional): Parent temporary directory for YOLO outputs.
            Each run creates a UUID subfolder inside this directory.
            Defaults to "runs/temp".

    Example:
        >>> detector = DetectionManager(
        ...     model_name="yolo11n.pt",
        ...     batch=32,
        ...     classes=[0],  # Detect only persons
        ...     conf=0.5
        ... )
        >>> detector.detect("datasets/classA/")
        >>> detector.detect("datasets/classB/")
    """

    model_name: str
    batch: int
    classes: list[int]
    conf: float
    temp_dir: Path
    model: YOLO | None

    def __init__(
        self,
        model_name: str = "yolo12x.pt",
        batch: int = 64,
        classes: list[int] | None = None,
        conf: float = 0.25,
        temp_dir: str = "runs/temp",
    ):
        self.model_name = model_name
        self.batch = batch
        self.classes = classes if classes is not None else [0]
        self.conf = conf
        self.temp_dir = Path(temp_dir)
        self.model = None  # Lazy load on first detect() call

    def detect(
        self,
        image_dir: Path | str,
        detect_folder_name: str = "detect",
    ):
        """Run YOLO detection on images in the specified directory.

        This method:
        1. Generates a unique UUID subfolder in the temp directory
        2. Runs YOLO detection on all images in {image_dir}/images/
        3. Moves detection .txt files to {image_dir}/{detect_folder_name}/
        4. Cleans up the UUID subfolder

        Args:
            image_dir (Path | str): Parent directory containing an 'images/' subfolder.
            detect_folder_name (str, optional): Name of folder to store detections.
                Defaults to "detect".

        Raises:
            FileNotFoundError: If {image_dir}/images/ doesn't exist.
        """
        image_dir = Path(image_dir)
        images_path = image_dir / "images"

        # Validate that images directory exists
        if not images_path.exists():
            raise FileNotFoundError(
                f"Images directory not found: {images_path}\n"
                f"Expected structure: {image_dir}/images/"
            )

        # Generate unique subfolder for this detection run
        run_id = str(uuid.uuid4())
        run_dir = self.temp_dir / run_id

        # Run YOLO detection in the UUID subfolder
        print(f"🔍 Running detection on {images_path}...")
        self._run_detection(images_path, run_dir)

        # Move detections to target directory
        print(f"📦 Moving detections to {image_dir}/{detect_folder_name}...")
        self._move_detections(image_dir, detect_folder_name, run_dir)

        # Clean up the UUID subfolder
        self._cleanup_run_directory(run_dir)

        print("✅ Detection complete!")

    def _run_detection(self, images_path: Path, run_dir: Path):
        """Run YOLO model on images and save results to a unique run directory.

        Args:
            images_path (Path): Path to directory containing images.
            run_dir (Path): Unique directory for this detection run (UUID subfolder).
        """
        # Lazy load model on first use
        if self.model is None:
            print(f"📥 Loading model: {self.model_name}...")
            self.model = YOLO(self.model_name)

        # Run prediction with streaming for memory efficiency
        # YOLO will create: run_dir/predict/labels/*.txt
        results = self.model.predict(
            images_path,
            classes=self.classes,
            stream=True,
            verbose=False,
            save_txt=True,  # Save detections as .txt files
            project=str(run_dir.parent),  # Parent of UUID folder
            name=run_dir.name,  # UUID folder name
            batch=self.batch,
            conf=self.conf,
        )

        # Count total images for progress bar
        num_images = len(
            [
                f
                for f in images_path.iterdir()
                if f.suffix.lower() in [".jpg", ".jpeg", ".png"]
            ]
        )

        # Iterate through results (triggers actual inference)
        for _ in tqdm(results, total=num_images, desc="Detecting"):
            pass

    def _move_detections(self, image_dir: Path, detect_folder_name: str, run_dir: Path):
        """Move detection .txt files from run directory to target directory.

        Args:
            image_dir (Path): Parent directory to store detections in.
            detect_folder_name (str): Name of folder to store detections.
            run_dir (Path): Unique directory for this detection run (UUID subfolder).
        """
        # Create target detection directory
        target_dir = image_dir / detect_folder_name
        target_dir.mkdir(exist_ok=True, parents=True)

        # Source directory where YOLO saves detections (inside UUID folder)
        source_dir = run_dir / "predict" / "labels"

        if not source_dir.exists():
            print(f"⚠️  No detections found in {source_dir}")
            return

        # Move each detection file to target directory
        moved_count = 0
        skipped_count = 0

        for detection_file in source_dir.iterdir():
            if not detection_file.suffix == ".txt":
            continue

            target_path = target_dir / detection_file.name

            # Skip if detection already exists (don't overwrite)
            if target_path.exists():
                skipped_count += 1
                continue

            # Move detection file
            shutil.move(str(detection_file), str(target_path))
            moved_count += 1

        print(f"   Moved: {moved_count} files")
        if skipped_count > 0:
            print(f"   Skipped: {skipped_count} files (already exist)")

    def _cleanup_run_directory(self, run_dir: Path):
        """Remove the UUID run directory after moving detections.

        Args:
            run_dir (Path): The unique directory to clean up.
        """
        if run_dir.exists():
            shutil.rmtree(run_dir)
            print(f"🧹 Cleaned up {run_dir.name}")
