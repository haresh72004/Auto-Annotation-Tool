from pathlib import Path


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent

INPUT_DIR = PROJECT_ROOT / "input" / "images"

OUTPUT_DIR = PROJECT_ROOT / "output"

AUTO_LABEL_DIR = OUTPUT_DIR / "auto_labels"
PREVIEW_DIR = OUTPUT_DIR / "preview"
VERIFIED_DIR = OUTPUT_DIR / "verified"

DATASET_DIR = PROJECT_ROOT / "dataset"
DATASET_IMAGE_DIR = DATASET_DIR / "images"
DATASET_LABEL_DIR = DATASET_DIR / "labels"

CLASSES_FILE = PROJECT_ROOT / "classes.txt"


# ============================================================
# GROUNDING DINO
# ============================================================

MODEL_ID = "IDEA-Research/grounding-dino-tiny"

# Detection threshold.
BOX_THRESHOLD = 0.35

# Text matching threshold.
TEXT_THRESHOLD = 0.25


# ============================================================
# DEVICE
# ============================================================

DEVICE = "cuda"


# ============================================================
# IMAGE SETTINGS
# ============================================================

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
}


# ============================================================
# GUI SETTINGS
# ============================================================

GUI_WINDOW_WIDTH = 1400
GUI_WINDOW_HEIGHT = 900


# ============================================================
# CREATE REQUIRED DIRECTORIES
# ============================================================

for directory in [
    INPUT_DIR,
    AUTO_LABEL_DIR,
    PREVIEW_DIR,
    VERIFIED_DIR,
    DATASET_IMAGE_DIR,
    DATASET_LABEL_DIR,
]:
    directory.mkdir(parents=True, exist_ok=True)


def load_classes():
    """
    Load class names from classes.txt.

    Example:
        person
        robot
        box
    """

    if not CLASSES_FILE.exists():
        raise FileNotFoundError(
            f"classes.txt not found: {CLASSES_FILE}"
        )

    classes = []

    with open(CLASSES_FILE, "r", encoding="utf-8") as file:
        for line in file:
            name = line.strip()

            if not name:
                continue

            if name.startswith("#"):
                continue

            classes.append(name)

    if not classes:
        raise ValueError("classes.txt is empty.")

    return classes


def get_image_files():
    """
    Return all supported images from input/images.
    """

    return sorted(
        [
            path
            for path in INPUT_DIR.iterdir()
            if path.is_file()
            and path.suffix.lower() in IMAGE_EXTENSIONS
        ]
    )
