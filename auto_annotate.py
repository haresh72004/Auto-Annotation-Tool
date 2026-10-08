from pathlib import Path

import cv2
import torch
from PIL import Image
from tqdm import tqdm
from transformers import (
    AutoModelForZeroShotObjectDetection,
    AutoProcessor,
)

from config import (
    MODEL_ID,
    BOX_THRESHOLD,
    TEXT_THRESHOLD,
    DEVICE,
    INPUT_DIR,
    AUTO_LABEL_DIR,
    PREVIEW_DIR,
    load_classes,
    get_image_files,
)


# ============================================================
# LOAD MODEL
# ============================================================

def load_model():
    print("=" * 70)
    print("Loading Grounding DINO")
    print("=" * 70)

    if DEVICE == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available. "
            "Run 01_test_gpu.py first."
        )

    print(f"Model: {MODEL_ID}")
    print(f"Device: {DEVICE}")

    processor = AutoProcessor.from_pretrained(MODEL_ID)

    model = AutoModelForZeroShotObjectDetection.from_pretrained(
        MODEL_ID
    )

    model = model.to(DEVICE)
    model.eval()

    print("Grounding DINO loaded successfully.")

    return processor, model


# ============================================================
# CLASS PROMPT
# ============================================================

def create_text_labels(classes):
    """
    Grounding DINO accepts multiple text labels.

    Example:

        [
            [
                "a person",
                "a robot",
                "a box"
            ]
        ]
    """

    return [[f"a {name}" for name in classes]]


# ============================================================
# MATCH DETECTED LABEL TO CLASS ID
# ============================================================

def match_class_id(detected_label, classes):
    """
    Convert Grounding DINO returned text label into
    our fixed YOLO class ID.
    """

    if detected_label is None:
        return None

    label = str(detected_label).lower().strip()

    # Remove common article.
    for prefix in [
        "a ",
        "an ",
        "the ",
    ]:
        if label.startswith(prefix):
            label = label[len(prefix):]

    label = label.strip(
        " .,:;!?\"'"
    )

    # Exact match first.
    for class_id, class_name in enumerate(classes):
        if label == class_name.lower().strip():
            return class_id

    # Fallback partial matching.
    for class_id, class_name in enumerate(classes):
        class_name_lower = class_name.lower().strip()

        if (
            class_name_lower in label
            or label in class_name_lower
        ):
            return class_id

    return None


# ============================================================
# YOLO CONVERSION
# ============================================================

def xyxy_to_yolo(
    x1,
    y1,
    x2,
    y2,
    image_width,
    image_height,
):
    """
    Convert:

        x1, y1, x2, y2

    into:

        center_x center_y width height

    normalized to 0-1.
    """

    x1 = max(0.0, min(float(x1), image_width))
    y1 = max(0.0, min(float(y1), image_height))
    x2 = max(0.0, min(float(x2), image_width))
    y2 = max(0.0, min(float(y2), image_height))

    width = x2 - x1
    height = y2 - y1

    if width <= 0 or height <= 0:
        return None

    center_x = x1 + width / 2.0
    center_y = y1 + height / 2.0

    center_x /= image_width
    center_y /= image_height
    width /= image_width
    height /= image_height

    return (
        center_x,
        center_y,
        width,
        height,
    )


# ============================================================
# DRAW PREVIEW
# ============================================================

def draw_preview(
    image,
    detections,
    output_path,
):
    """
    Draw automatic annotations onto image.
    """

    image_bgr = cv2.cvtColor(
        image,
        cv2.COLOR_RGB2BGR,
    )

    for detection in detections:

        x1, y1, x2, y2 = detection["box"]

        class_name = detection["class_name"]
        confidence = detection["confidence"]

        x1 = int(x1)
        y1 = int(y1)
        x2 = int(x2)
        y2 = int(y2)

        cv2.rectangle(
            image_bgr,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2,
        )

        text = (
            f"{class_name} "
            f"{confidence:.2f}"
        )

        cv2.putText(
            image_bgr,
            text,
            (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

    cv2.imwrite(
        str(output_path),
        image_bgr,
    )


# ============================================================
# PROCESS ONE IMAGE
# ============================================================

def process_image(
    image_path,
    processor,
    model,
    classes,
):
    print(f"\nProcessing: {image_path.name}")

    image = Image.open(image_path).convert("RGB")

    image_width, image_height = image.size

    text_labels = create_text_labels(classes)

    inputs = processor(
        images=image,
        text=text_labels,
        return_tensors="pt",
    )

    inputs = {
        key: value.to(DEVICE)
        if hasattr(value, "to")
        else value
        for key, value in inputs.items()
    }

    with torch.no_grad():

        outputs = model(**inputs)

    results = processor.post_process_grounded_object_detection(
        outputs,
        threshold=BOX_THRESHOLD,
        text_threshold=TEXT_THRESHOLD,
        target_sizes=[
            (image_height, image_width)
        ],
    )

    result = results[0]

    boxes = result["boxes"].detach().cpu()
    scores = result["scores"].detach().cpu()

    if "text_labels" in result:
        detected_labels = result["text_labels"]
    elif "labels" in result:
        detected_labels = result["labels"]
    else:
        detected_labels = []

    detections = []

    for box, score, detected_label in zip(
        boxes,
        scores,
        detected_labels,
    ):

        class_id = match_class_id(
            detected_label,
            classes,
        )

        if class_id is None:
            print(
                f"Skipping unknown detection: "
                f"{detected_label}"
            )
            continue

        x1, y1, x2, y2 = box.tolist()

        yolo_box = xyxy_to_yolo(
            x1,
            y1,
            x2,
            y2,
            image_width,
            image_height,
        )

        if yolo_box is None:
            continue

        detections.append(
            {
                "class_id": class_id,
                "class_name": classes[class_id],
                "confidence": float(score),
                "box": [
                    x1,
                    y1,
                    x2,
                    y2,
                ],
                "yolo": yolo_box,
            }
        )

    # --------------------------------------------------------
    # SAVE YOLO LABEL
    # --------------------------------------------------------

    label_path = (
        AUTO_LABEL_DIR
        / f"{image_path.stem}.txt"
    )

    with open(
        label_path,
        "w",
        encoding="utf-8",
    ) as file:

        for detection in detections:

            class_id = detection["class_id"]

            center_x, center_y, width, height = (
                detection["yolo"]
            )

            file.write(
                f"{class_id} "
                f"{center_x:.6f} "
                f"{center_y:.6f} "
                f"{width:.6f} "
                f"{height:.6f}\n"
            )

    # --------------------------------------------------------
    # SAVE PREVIEW
    # --------------------------------------------------------

    preview_path = (
        PREVIEW_DIR
        / image_path.name
    )

    draw_preview(
        image,
        detections,
        preview_path,
    )

    print(
        f"Detected objects: {len(detections)}"
    )

    print(
        f"Label: {label_path}"
    )

    print(
        f"Preview: {preview_path}"
    )

    return detections


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("AUTO ANNOTATION TOOL")
    print("Grounding DINO -> YOLO")
    print("=" * 70)

    classes = load_classes()

    print("\nClasses:")

    for index, name in enumerate(classes):
        print(f"  {index}: {name}")

    image_files = get_image_files()

    if not image_files:

        print(
            "\nNo images found."
        )

        print(
            f"Put images inside:\n{INPUT_DIR}"
        )

        return

    processor, model = load_model()

    print(
        f"\nImages found: {len(image_files)}"
    )

    for image_path in tqdm(
        image_files,
        desc="Annotating",
    ):

        try:

            process_image(
                image_path,
                processor,
                model,
                classes,
            )

        except Exception as error:

            print(
                f"\nERROR processing "
                f"{image_path.name}:"
            )

            print(error)

    print("\n" + "=" * 70)
    print("AUTO ANNOTATION COMPLETE")
    print("=" * 70)

    print(
        f"YOLO labels:\n{AUTO_LABEL_DIR}"
    )

    print(
        f"Preview images:\n{PREVIEW_DIR}"
    )


if __name__ == "__main__":
    main()
