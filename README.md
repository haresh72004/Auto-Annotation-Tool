# Auto-Annotation-Tool

Developed an automated object-annotation pipeline using **YOLO-World and DINO** for object detection and automatic dataset annotation.

## Features

- Automatic object detection from raw images
- Text-prompt-based object detection
- Automatic bounding-box generation
- YOLO-format `.txt` label generation
- Annotation preview with bounding boxes
- Manual annotation verification and correction
- Add, delete, move, resize, and change object classes
- Final verified images and labels generation
- GPU acceleration using CUDA

## Workflow

```text
Raw Images
    ↓
YOLO-World / DINO
    ↓
Object Detection
    ↓
YOLO Labels
    ↓
Annotation Verification
    ↓
Manual Correction
    ↓
Final Images + Labels
```

## Technologies

- Python
- UV
- PyTorch
- YOLO-World
- DINO / Grounding DINO
- OpenCV
- NumPy
- Pillow
- PySide6
- Hugging Face Transformers

## Project Structure

```text
AutoAnnotationTool/
│
├── config.py
├── classes.txt
├── 01_test_gpu.py
├── 02_auto_annotate.py
├── 03_annotation_gui.py
├── 04_prepare_dataset.py
│
├── input/
│   └── images/
│
├── output/
│   ├── auto_labels/
│   ├── preview/
│   └── verified/
│
└── dataset/
    ├── images/
    └── labels/
```

## Dataset Creation

Raw images are placed in:

```text
input/images/
```

The automatic annotation pipeline generates:

```text
output/auto_labels/
```

for YOLO-format annotations and:

```text
output/preview/
```

for visual inspection of the generated bounding boxes.

After manual verification and correction using the annotation GUI, the final verified dataset is generated as:

```text
dataset/
├── images/
└── labels/
```

## YOLO Label Format

Each image has a corresponding `.txt` annotation file using the YOLO format:

```text
class_id center_x center_y width height
```

All bounding-box coordinates are normalized between `0` and `1`.

## Running the Tool

### 1. Test GPU

```bash
uv run python 01_test_gpu.py
```

### 2. Automatic Annotation

```bash
uv run python 02_auto_annotate.py
```

### 3. Annotation Verification GUI

```bash
uv run python 03_annotation_gui.py
```

### 4. Prepare Final Dataset

```bash
uv run python 04_prepare_dataset.py
```

## Scope

This project is focused specifically on **automatic annotation and dataset creation**.

### Included

- Image annotation
- Object detection
- YOLO label generation
- Annotation visualization
- Manual annotation verification
- Dataset organization

### Not Included

- Model training
- `best.pt` generation
- Model deployment
- TensorRT conversion
- OpenVINO conversion
- Inference deployment pipeline

## Output

The final output of the project is a verified dataset containing:

```text
Images + YOLO Labels
```
