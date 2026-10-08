import shutil
from dataclasses import dataclass
from pathlib import Path

import cv2

from PySide6.QtCore import (
    QPoint,
    QRect,
    Qt,
)
from PySide6.QtGui import (
    QImage,
    QPainter,
    QPen,
    QPixmap,
)
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from config import (
    AUTO_LABEL_DIR,
    INPUT_DIR,
    VERIFIED_DIR,
    load_classes,
    get_image_files,
)


# ============================================================
# DETECTION DATA
# ============================================================

@dataclass
class Box:
    class_id: int
    x1: float
    y1: float
    x2: float
    y2: float


# ============================================================
# IMAGE CANVAS
# ============================================================

class ImageCanvas(QWidget):

    def __init__(self, parent=None):
        super().__init__(parent)

        self.image = None
        self.boxes = []
        self.classes = []

        self.selected_index = -1

        self.mode = "move"

        self.dragging = False

        self.drag_start = None

        self.original_box = None

        self.setMinimumSize(800, 600)

    # --------------------------------------------------------
    # SET DATA
    # --------------------------------------------------------

    def set_data(
        self,
        image,
        boxes,
        classes,
    ):
        self.image = image
        self.boxes = boxes
        self.classes = classes
        self.selected_index = -1
        self.dragging = False

        self.update()

    # --------------------------------------------------------
    # COORDINATE CONVERSION
    # --------------------------------------------------------

    def image_rect(self):

        if self.image is None:
            return QRect()

        image_height, image_width = self.image.shape[:2]

        widget_width = self.width()
        widget_height = self.height()

        scale = min(
            widget_width / image_width,
            widget_height / image_height,
        )

        display_width = int(
            image_width * scale
        )

        display_height = int(
            image_height * scale
        )

        offset_x = (
            widget_width - display_width
        ) // 2

        offset_y = (
            widget_height - display_height
        ) // 2

        return QRect(
            offset_x,
            offset_y,
            display_width,
            display_height,
        )

    def widget_to_image(
        self,
        point,
    ):

        if self.image is None:
            return None

        rect = self.image_rect()

        if not rect.contains(point):
            return None

        image_height, image_width = self.image.shape[:2]

        x = (
            point.x() - rect.left()
        ) / rect.width()

        y = (
            point.y() - rect.top()
        ) / rect.height()

        return (
            x * image_width,
            y * image_height,
        )

    def image_to_widget(
        self,
        x,
        y,
    ):

        rect = self.image_rect()

        image_height, image_width = self.image.shape[:2]

        px = (
            rect.left()
            + x / image_width * rect.width()
        )

        py = (
            rect.top()
            + y / image_height * rect.height()
        )

        return QPoint(
            int(px),
            int(py),
        )

    # --------------------------------------------------------
    # PAINT
    # --------------------------------------------------------

    def paintEvent(self, event):

        painter = QPainter(self)

        painter.fillRect(
            self.rect(),
            Qt.black,
        )

        if self.image is None:
            return

        image_rgb = cv2.cvtColor(
            self.image,
            cv2.COLOR_BGR2RGB,
        )

        height, width = image_rgb.shape[:2]

        qimage = QImage(
            image_rgb.data,
            width,
            height,
            width * 3,
            QImage.Format_RGB888,
        )

        rect = self.image_rect()

        painter.drawImage(
            rect,
            qimage,
        )

        # Draw boxes.
        for index, box in enumerate(self.boxes):

            p1 = self.image_to_widget(
                box.x1,
                box.y1,
            )

            p2 = self.image_to_widget(
                box.x2,
                box.y2,
            )

            if index == self.selected_index:

                pen = QPen(
                    Qt.red,
                    3,
                )

            else:

                pen = QPen(
                    Qt.green,
                    2,
                )

            painter.setPen(pen)

            painter.drawRect(
                QRect(
                    p1,
                    p2,
                )
            )

            class_name = self.classes[
                box.class_id
            ]

            painter.drawText(
                p1.x(),
                max(
                    15,
                    p1.y() - 5,
                ),
                class_name,
            )

    # --------------------------------------------------------
    # FIND BOX
    # --------------------------------------------------------

    def find_box(
        self,
        x,
        y,
    ):

        for index in reversed(
            range(len(self.boxes))
        ):

            box = self.boxes[index]

            if (
                box.x1 <= x <= box.x2
                and
                box.y1 <= y <= box.y2
            ):
                return index

        return -1

    # --------------------------------------------------------
    # MOUSE PRESS
    # --------------------------------------------------------

    def mousePressEvent(self, event):

        if event.button() != Qt.LeftButton:
            return

        image_point = self.widget_to_image(
            event.position().toPoint()
        )

        if image_point is None:
            return

        x, y = image_point

        # Add mode.
        if self.mode == "add":

            self.dragging = True

            self.drag_start = (
                x,
                y,
            )

            self.boxes.append(
                Box(
                    class_id=0,
                    x1=x,
                    y1=y,
                    x2=x,
                    y2=y,
                )
            )

            self.selected_index = (
                len(self.boxes) - 1
            )

            self.update()

            return

        # Normal selection.
        index = self.find_box(
            x,
            y,
        )

        self.selected_index = index

        if index >= 0:

            self.dragging = True

            self.drag_start = (
                x,
                y,
            )

            box = self.boxes[index]

            self.original_box = Box(
                box.class_id,
                box.x1,
                box.y1,
                box.x2,
                box.y2,
            )

        self.update()

    # --------------------------------------------------------
    # MOUSE MOVE
    # --------------------------------------------------------

    def mouseMoveEvent(self, event):

        if not self.dragging:
            return

        image_point = self.widget_to_image(
            event.position().toPoint()
        )

        if image_point is None:
            return

        x, y = image_point

        if self.selected_index < 0:
            return

        box = self.boxes[
            self.selected_index
        ]

        if self.mode == "add":

            start_x, start_y = (
                self.drag_start
            )

            box.x1 = min(
                start_x,
                x,
            )

            box.y1 = min(
                start_y,
                y,
            )

            box.x2 = max(
                start_x,
                x,
            )

            box.y2 = max(
                start_y,
                y,
            )

        else:

            original = self.original_box

            dx = x - self.drag_start[0]
            dy = y - self.drag_start[1]

            box.x1 = original.x1 + dx
            box.y1 = original.y1 + dy
            box.x2 = original.x2 + dx
            box.y2 = original.y2 + dy

        self.update()

    # --------------------------------------------------------
    # MOUSE RELEASE
    # --------------------------------------------------------

    def mouseReleaseEvent(self, event):

        if event.button() != Qt.LeftButton:
            return

        self.dragging = False

        self.mode = "move"

        self.update()

    # --------------------------------------------------------
    # DELETE
    # --------------------------------------------------------

    def delete_selected(self):

        if self.selected_index < 0:
            return

        del self.boxes[
            self.selected_index
        ]

        self.selected_index = -1

        self.update()


# ============================================================
# MAIN WINDOW
# ============================================================

class AnnotationWindow(QMainWindow):

    def __init__(self):

        super().__init__()

        self.setWindowTitle(
            "Auto Annotation Tool"
        )

        self.resize(
            1400,
            900,
        )

        self.classes = load_classes()

        self.image_files = get_image_files()

        self.current_index = 0

        self.current_image = None

        self.current_boxes = []

        self.setup_ui()

        if self.image_files:
            self.load_current_image()

    # --------------------------------------------------------
    # UI
    # --------------------------------------------------------

    def setup_ui(self):

        central = QWidget()

        self.setCentralWidget(
            central
        )

        main_layout = QHBoxLayout(
            central
        )

        # ----------------------------------------------------
        # LEFT
        # ----------------------------------------------------

        self.image_list = QListWidget()

        for image in self.image_files:
            self.image_list.addItem(
                image.name
            )

        self.image_list.currentRowChanged.connect(
            self.list_changed
        )

        self.image_list.setMaximumWidth(
            220
        )

        main_layout.addWidget(
            self.image_list
        )

        # ----------------------------------------------------
        # CENTER
        # ----------------------------------------------------

        center_layout = QVBoxLayout()

        self.canvas = ImageCanvas()

        center_layout.addWidget(
            self.canvas
        )

        # ----------------------------------------------------
        # BUTTONS
        # ----------------------------------------------------

        button_layout = QHBoxLayout()

        self.previous_button = QPushButton(
            "Previous"
        )

        self.next_button = QPushButton(
            "Next"
        )

        self.add_button = QPushButton(
            "Add Box"
        )

        self.delete_button = QPushButton(
            "Delete Box"
        )

        self.save_button = QPushButton(
            "Save"
        )

        self.reject_button = QPushButton(
            "Reject"
        )

        button_layout.addWidget(
            self.previous_button
        )

        button_layout.addWidget(
            self.next_button
        )

        button_layout.addWidget(
            self.add_button
        )

        button_layout.addWidget(
            self.delete_button
        )

        button_layout.addWidget(
            self.save_button
        )

        button_layout.addWidget(
            self.reject_button
        )

        center_layout.addLayout(
            button_layout
        )

        # ----------------------------------------------------
        # CLASS SELECTOR
        # ----------------------------------------------------

        class_layout = QHBoxLayout()

        class_layout.addWidget(
            QLabel("Selected class:")
        )

        self.class_combo = QComboBox()

        self.class_combo.addItems(
            self.classes
        )

        self.class_combo.currentIndexChanged.connect(
            self.change_class
        )

        class_layout.addWidget(
            self.class_combo
        )

        center_layout.addLayout(
            class_layout
        )

        main_layout.addLayout(
            center_layout
        )

        # ----------------------------------------------------
        # CONNECTIONS
        # ----------------------------------------------------

        self.previous_button.clicked.connect(
            self.previous_image
        )

        self.next_button.clicked.connect(
            self.next_image
        )

        self.add_button.clicked.connect(
            self.add_box_mode
        )

        self.delete_button.clicked.connect(
            self.canvas.delete_selected
        )

        self.save_button.clicked.connect(
            self.save_current
        )

        self.reject_button.clicked.connect(
            self.reject_current
        )

    # --------------------------------------------------------
    # LOAD IMAGE
    # --------------------------------------------------------

    def load_current_image(self):

        if not self.image_files:
            return

        image_path = self.image_files[
            self.current_index
        ]

        image = cv2.imread(
            str(image_path)
        )

        if image is None:
            QMessageBox.warning(
                self,
                "Error",
                f"Could not load:\n{image_path}",
            )

            return

        self.current_image = image

        self.current_boxes = (
            self.load_yolo_boxes(
                image_path
            )
        )

        self.canvas.set_data(
            image,
            self.current_boxes,
            self.classes,
        )

        self.image_list.blockSignals(
            True
        )

        self.image_list.setCurrentRow(
            self.current_index
        )

        self.image_list.blockSignals(
            False
        )

        self.update_class_selector()

        self.setWindowTitle(
            "Auto Annotation Tool - "
            f"{image_path.name}"
        )

    # --------------------------------------------------------
    # LOAD YOLO LABELS
    # --------------------------------------------------------

    def load_yolo_boxes(
        self,
        image_path,
    ):

        label_path = (
            AUTO_LABEL_DIR
            / f"{image_path.stem}.txt"
        )

        if not label_path.exists():

            return []

        image = cv2.imread(
            str(image_path)
        )

        height, width = image.shape[:2]

        boxes = []

        with open(
            label_path,
            "r",
            encoding="utf-8",
        ) as file:

            for line in file:

                parts = line.strip().split()

                if len(parts) != 5:
                    continue

                class_id = int(
                    parts[0]
                )

                center_x = float(
                    parts[1]
                )

                center_y = float(
                    parts[2]
                )

                box_width = float(
                    parts[3]
                )

                box_height = float(
                    parts[4]
                )

                x1 = (
                    center_x
                    - box_width / 2
                ) * width

                y1 = (
                    center_y
                    - box_height / 2
                ) * height

                x2 = (
                    center_x
                    + box_width / 2
                ) * width

                y2 = (
                    center_y
                    + box_height / 2
                ) * height

                boxes.append(
                    Box(
                        class_id,
                        x1,
                        y1,
                        x2,
                        y2,
                    )
                )

        return boxes

    # --------------------------------------------------------
    # SAVE YOLO
    # --------------------------------------------------------

    def save_current(self):

        if not self.image_files:
            return

        image_path = self.image_files[
            self.current_index
        ]

        height, width = (
            self.current_image.shape[:2]
        )

        verified_image_dir = (
            VERIFIED_DIR / "images"
        )

        verified_label_dir = (
            VERIFIED_DIR / "labels"
        )

        verified_image_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        verified_label_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        output_image = (
            verified_image_dir
            / image_path.name
        )

        output_label = (
            verified_label_dir
            / f"{image_path.stem}.txt"
        )

        shutil.copy2(
            image_path,
            output_image,
        )

        with open(
            output_label,
            "w",
            encoding="utf-8",
        ) as file:

            for box in self.current_boxes:

                x1 = max(
                    0,
                    min(width, box.x1)
                )

                y1 = max(
                    0,
                    min(height, box.y1)
                )

                x2 = max(
                    0,
                    min(width, box.x2)
                )

                y2 = max(
                    0,
                    min(height, box.y2)
                )

                box_width = x2 - x1
                box_height = y2 - y1

                if (
                    box_width <= 0
                    or box_height <= 0
                ):
                    continue

                center_x = (
                    x1 + x2
                ) / 2 / width

                center_y = (
                    y1 + y2
                ) / 2 / height

                normalized_width = (
                    box_width / width
                )

                normalized_height = (
                    box_height / height
                )

                file.write(
                    f"{box.class_id} "
                    f"{center_x:.6f} "
                    f"{center_y:.6f} "
                    f"{normalized_width:.6f} "
                    f"{normalized_height:.6f}\n"
                )

        QMessageBox.information(
            self,
            "Saved",
            f"Verified annotation saved:\n\n"
            f"{output_image}\n\n"
            f"{output_label}",
        )

    # --------------------------------------------------------
    # REJECT
    # --------------------------------------------------------

    def reject_current(self):

        if not self.image_files:
            return

        image_path = self.image_files[
            self.current_index
        ]

        rejected_dir = (
            VERIFIED_DIR / "rejected"
        )

        rejected_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        destination = (
            rejected_dir
            / image_path.name
        )

        shutil.copy2(
            image_path,
            destination,
        )

        self.next_image()

    # --------------------------------------------------------
    # ADD BOX
    # --------------------------------------------------------

    def add_box_mode(self):

        self.canvas.mode = "add"

        QMessageBox.information(
            self,
            "Add Box",
            "Drag on the image to create a new box.",
        )

    # --------------------------------------------------------
    # CHANGE CLASS
    # --------------------------------------------------------

    def change_class(
        self,
        index,
    ):

        if (
            self.canvas.selected_index
            < 0
        ):
            return

        if index < 0:
            return

        self.current_boxes[
            self.canvas.selected_index
        ].class_id = index

        self.canvas.update()

    # --------------------------------------------------------
    # CLASS SELECTOR
    # --------------------------------------------------------

    def update_class_selector(self):

        index = (
            self.canvas.selected_index
        )

        if index < 0:
            return

        box = self.current_boxes[
            index
        ]

        self.class_combo.blockSignals(
            True
        )

        self.class_combo.setCurrentIndex(
            box.class_id
        )

        self.class_combo.blockSignals(
            False
        )

    # --------------------------------------------------------
    # PREVIOUS
    # --------------------------------------------------------

    def previous_image(self):

        if self.current_index <= 0:
            return

        self.current_index -= 1

        self.load_current_image()

    # --------------------------------------------------------
    # NEXT
    # --------------------------------------------------------

    def next_image(self):

        if (
            self.current_index
            >= len(self.image_files) - 1
        ):
            return

        self.current_index += 1

        self.load_current_image()

    # --------------------------------------------------------
    # LIST
    # --------------------------------------------------------

    def list_changed(
        self,
        row,
    ):

        if row < 0:
            return

        self.current_index = row

        self.load_current_image()


# ============================================================
# MAIN
# ============================================================

def main():

    app = QApplication([])

    window = AnnotationWindow()

    window.show()

    app.exec()


if __name__ == "__main__":
    main()
