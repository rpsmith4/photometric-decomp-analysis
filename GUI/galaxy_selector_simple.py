#!/usr/bin/env python3
"""Simple galaxy selector GUI.

Shows a list of galaxy folders (leaf directories) under a given path,
allows selecting one or many, previews images for bands (g,r,i,z) and
copies selected galaxy folders to a chosen destination.

Dependencies: PySide6, astropy, numpy, matplotlib
"""
import sys
import os
import argparse
import shutil
from pathlib import Path

from PySide6 import QtWidgets, QtGui, QtCore
from PySide6.QtWidgets import QApplication, QMainWindow, QListWidgetItem, QFileDialog, QMessageBox

import numpy as np
from astropy.io import fits
from matplotlib import pyplot as plt


def find_leaf_dirs(root_path: Path):
    """Return a sorted list of leaf directories under root_path."""
    leafs = []
    for dirpath, dirnames, filenames in os.walk(root_path):
        # If directory contains no sub-directories, consider it a leaf
        if not dirnames:
            leafs.append(Path(dirpath))
    leafs.sort()
    return leafs


def load_image_for_band(galpath: Path, band: str):
    """Attempt to load a displayable QPixmap for a given galaxy and band.

    Tries common filenames, falls back to any .jpg/.png or to FITS conversion.
    Returns QPixmap or None.
    """
    # Special-case: if user requests RGB jpg, prefer image.jpg/png
    if band is not None and str(band).lower() in ("rgb", "image"):
        for name in ("image.jpg", "image.png", "image.jpeg"):
            p = galpath / name
            if p.exists():
                pix = QtGui.QPixmap(str(p))
                if not pix.isNull():
                    return pix
    candidates = [
        f"image_{band}.jpg",
        f"image_{band}.png",
        f"image_{band}.fits",
        f"image_{band}.fit",
        f"{band}.jpg",
        f"{band}.png",
    ]

    for name in candidates:
        p = galpath / name
        if p.exists():
            if p.suffix.lower() in [".jpg", ".png"]:
                pix = QtGui.QPixmap(str(p))
                if not pix.isNull():
                    return pix
            else:
                # assume FITS
                try:
                    data = fits.getdata(str(p))
                except Exception:
                    continue
                pix = fits_array_to_pixmap(data)
                if pix is not None:
                    return pix

    # Fallback: any jpg/png in folder
    for ext in (".jpg", ".png"):
        for p in galpath.glob(f"*{ext}"):
            pix = QtGui.QPixmap(str(p))
            if not pix.isNull():
                return pix

    # Try any fits file
    for p in galpath.glob("*.fits"):
        try:
            data = fits.getdata(str(p))
        except Exception:
            continue
        pix = fits_array_to_pixmap(data)
        if pix is not None:
            return pix

    return None


def fits_array_to_pixmap(arr: np.ndarray):
    """Convert a FITS numpy array to QPixmap via a simple normalization."""
    try:
        if arr is None:
            return None
        # Take 2D image (if 3D, take first slice)
        if arr.ndim > 2:
            arr = arr[0]
        arr = np.nan_to_num(arr)
        # use robust percentile scaling to 0-255
        lo, hi = np.percentile(arr, (1, 99))
        if hi == lo:
            hi = lo + 1.0
        arr_scaled = (arr - lo) / (hi - lo)
        arr_scaled = np.clip(arr_scaled, 0.0, 1.0)
        img8 = (arr_scaled * 255).astype(np.uint8)

        if img8.ndim == 2:
            h, w = img8.shape
            # create RGB
            rgb = np.stack([img8, img8, img8], axis=2)
        else:
            rgb = img8

        # Convert to QImage
        h, w, c = rgb.shape
        bytes_per_line = 3 * w
        image = QtGui.QImage(rgb.data.tobytes(), w, h, bytes_per_line, QtGui.QImage.Format.Format_RGB888)
        pix = QtGui.QPixmap.fromImage(image)
        return pix
    except Exception:
        return None


class GalaxySelectorWindow(QMainWindow):
    def __init__(self, root_path: Path, bands=("rgb", "g", "r", "i", "z")):
        super().__init__()
        self.setWindowTitle("Galaxy Selector")
        self.resize(1000, 600)
        self.root_path = root_path
        self.bands = list(bands)
        # make RGB (image.jpg) the default option
        self.current_band = "rgb"

        # thumbnail size (pixels)
        self.thumbnail_size = 64

        central = QtWidgets.QWidget()
        self.setCentralWidget(central)

        main_layout = QtWidgets.QHBoxLayout()
        central.setLayout(main_layout)

        # Left: list of galaxies (checkable items)
        self.list_widget = QtWidgets.QListWidget()
        # single selection used for previewing; use checkboxes for marking
        self.list_widget.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.SingleSelection)
        self.list_widget.itemSelectionChanged.connect(self.on_selection_changed)
        main_layout.addWidget(self.list_widget, 2)

        # Custom row widget class for numbered items
        class RowWidget(QtWidgets.QWidget):
            def __init__(self, number: str, name: str, thumb: QtGui.QPixmap | None = None, parent=None):
                super().__init__(parent)
                h = QtWidgets.QHBoxLayout()
                h.setContentsMargins(4, 2, 4, 2)
                self.num_label = QtWidgets.QLabel(number)
                self.num_label.setFixedWidth(30)
                self.num_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
                h.addWidget(self.num_label)
                self.thumb_label = QtWidgets.QLabel()
                if thumb is not None and not thumb.isNull():
                    self.thumb_label.setPixmap(thumb.scaled(40, 40, QtCore.Qt.AspectRatioMode.KeepAspectRatio, QtCore.Qt.TransformationMode.SmoothTransformation))
                self.thumb_label.setFixedSize(44, 44)
                h.addWidget(self.thumb_label)
                self.chk = QtWidgets.QCheckBox()
                h.addWidget(self.chk)
                self.name_label = QtWidgets.QLabel(name)
                h.addWidget(self.name_label)
                h.addStretch()
                self.setLayout(h)

        self._RowWidget = RowWidget

        # Right: preview + controls
        right_col = QtWidgets.QVBoxLayout()
        main_layout.addLayout(right_col, 4)

        # Band buttons
        band_layout = QtWidgets.QHBoxLayout()
        self.band_buttons = {}
        for b in self.bands:
            btn = QtWidgets.QPushButton(b.upper())
            btn.setCheckable(True)
            btn.clicked.connect(lambda checked, bb=b: self.set_band(bb))
            band_layout.addWidget(btn)
            self.band_buttons[b] = btn
        self.band_buttons[self.current_band].setChecked(True)
        band_layout.addStretch()
        right_col.addLayout(band_layout)

        # Image display
        self.image_label = QtWidgets.QLabel()
        self.image_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.image_label.setMinimumSize(400, 400)
        self.image_label.setStyleSheet("background: #222; color: white;")
        right_col.addWidget(self.image_label, 1)

        # Copy controls
        controls_layout = QtWidgets.QHBoxLayout()
        self.select_all_btn = QtWidgets.QPushButton("Select All")
        self.select_all_btn.clicked.connect(lambda: self.set_all_checks(True))
        controls_layout.addWidget(self.select_all_btn)
        self.clear_all_btn = QtWidgets.QPushButton("Clear All")
        self.clear_all_btn.clicked.connect(lambda: self.set_all_checks(False))
        controls_layout.addWidget(self.clear_all_btn)
        self.copy_btn = QtWidgets.QPushButton("Copy Checked Folders...")
        self.copy_btn.clicked.connect(self.copy_selected)
        controls_layout.addWidget(self.copy_btn)
        self.refresh_btn = QtWidgets.QPushButton("Refresh List")
        self.refresh_btn.clicked.connect(self.populate_list)
        controls_layout.addWidget(self.refresh_btn)
        controls_layout.addStretch()
        right_col.addLayout(controls_layout)

        # Status bar
        self.status = QtWidgets.QLabel("")
        right_col.addWidget(self.status)

        self.populate_list()


    def populate_list(self):
        self.list_widget.clear()
        leafs = find_leaf_dirs(self.root_path)
        for idx, p in enumerate(leafs):
            # create a QListWidgetItem as a container, store path in UserRole
            item = QListWidgetItem()
            item.setData(QtCore.Qt.ItemDataRole.UserRole, str(p))
            # create row widget with number (no leading zeros), thumbnail and checkbox
            num = str(idx + 1)
            pix = load_image_for_band(p, self.current_band)
            row = self._RowWidget(num, p.name, thumb=pix)
            # ensure item has reasonable height
            item.setSizeHint(row.sizeHint())
            self.list_widget.addItem(item)
            self.list_widget.setItemWidget(item, row)
        self.status.setText(f"Found {self.list_widget.count()} galaxy folders under {self.root_path}")

    def update_thumbnails(self):
        """Update icons for all items according to current_band."""
        for i in range(self.list_widget.count()):
            item = self.list_widget.item(i)
            p = Path(item.data(QtCore.Qt.ItemDataRole.UserRole))
            pix = load_image_for_band(p, self.current_band)
            w = self.list_widget.itemWidget(item)
            if w is not None:
                if pix is not None and not pix.isNull():
                    w.thumb_label.setPixmap(pix.scaled(40, 40, QtCore.Qt.AspectRatioMode.KeepAspectRatio, QtCore.Qt.TransformationMode.SmoothTransformation))
                else:
                    w.thumb_label.setPixmap(QtGui.QPixmap())

    def on_selection_changed(self):
        items = self.list_widget.selectedItems()
        if not items:
            self.image_label.setPixmap(QtGui.QPixmap())
            self.image_label.setText("No galaxy selected")
            # update status to remove selection info
            self.status.setText(f"Found {self.list_widget.count()} galaxy folders under {self.root_path}")
            return
        # show preview of the first selected
        first = items[0]
        p = Path(first.data(QtCore.Qt.ItemDataRole.UserRole))
        pix = load_image_for_band(p, self.current_band)
        if pix is None:
            self.image_label.setText(f"No {self.current_band} image found for {p.name}")
            self.image_label.setPixmap(QtGui.QPixmap())
        else:
            # scale pix to label while keeping aspect
            scaled = pix.scaled(self.image_label.size(), QtCore.Qt.AspectRatioMode.KeepAspectRatio, QtCore.Qt.TransformationMode.SmoothTransformation)
            self.image_label.setPixmap(scaled)
            self.image_label.setText("")
        # show selection position in status
        sel_idx = self.list_widget.row(first)
        total = self.list_widget.count()
        self.status.setText(f"Found {total} galaxy folders under {self.root_path} | Selected {sel_idx+1}/{total}")

    def set_all_checks(self, check: bool):
        state = QtCore.Qt.CheckState.Checked if check else QtCore.Qt.CheckState.Unchecked
        for i in range(self.list_widget.count()):
            item = self.list_widget.item(i)
            w = self.list_widget.itemWidget(item)
            if w is not None:
                w.chk.setChecked(state == QtCore.Qt.CheckState.Checked)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        # update preview scaling
        self.on_selection_changed()

    def set_band(self, band: str):
        if band not in self.bands:
            return
        self.current_band = band
        for b, btn in self.band_buttons.items():
            btn.setChecked(b == band)
        # refresh thumbnails to reflect chosen band
        # self.update_thumbnails()
        self.on_selection_changed()

    def keyPressEvent(self, event: QtGui.QKeyEvent):
        """Handle keyboard shortcuts: Enter toggles check on selected item; Left/Right change band."""
        key = event.key()
        # Enter/Return toggles checked state of currently selected item
        if key in (QtCore.Qt.Key_Return, QtCore.Qt.Key_Enter):
            items = self.list_widget.selectedItems()
            if items:
                item = items[0]
                w = self.list_widget.itemWidget(item)
                if w is not None:
                    w.chk.setChecked(not w.chk.isChecked())
            return

        # Left/Right arrows change band
        if key == QtCore.Qt.Key_Left or key == QtCore.Qt.Key_Right:
            try:
                idx = self.bands.index(self.current_band)
            except ValueError:
                idx = 0
            if key == QtCore.Qt.Key_Left:
                idx = (idx - 1) % len(self.bands)
            else:
                idx = (idx + 1) % len(self.bands)
            new_band = self.bands[idx]
            self.set_band(new_band)
            return

        # fallback to default
        super().keyPressEvent(event)

    def copy_selected(self):
        # Copy only items that are checked
        checked_items = []
        for i in range(self.list_widget.count()):
            item = self.list_widget.item(i)
            w = self.list_widget.itemWidget(item)
            if w is not None and w.chk.isChecked():
                checked_items.append(item)

        if not checked_items:
            QMessageBox.information(self, "No selection", "No galaxies checked to copy.")
            return

        dest = QFileDialog.getExistingDirectory(self, "Select destination folder")
        if not dest:
            return
        dest = Path(dest)
        copied = []
        skipped = []
        for it in checked_items:
            src = Path(it.data(QtCore.Qt.ItemDataRole.UserRole))
            target = dest / src.name
            if target.exists():
                skipped.append(src.name)
                continue
            try:
                shutil.copytree(src, target)
                copied.append(src.name)
            except Exception:
                skipped.append(src.name)

        msg = f"Copied: {len(copied)} folders. Skipped/Failed: {len(skipped)}."
        QMessageBox.information(self, "Copy result", msg)


def main():
    parser = argparse.ArgumentParser(description="Simple Galaxy Selector GUI")
    parser.add_argument("-p", "--path", help="Path to root containing galaxy folders", default=".")
    args = parser.parse_args()
    root = Path(args.path).resolve()
    app = QApplication(sys.argv)
    win = GalaxySelectorWindow(root)
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
