#!/usr/bin/env python3
import sys
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QPushButton, QSpinBox,
    QLabel, QLineEdit, QFileDialog, QMessageBox, QCheckBox,
    QComboBox, QHBoxLayout, QVBoxLayout
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QIcon, QVector3D
import resources_rc
from PIL import Image
import numpy as np

from vmflib import vmf
from vmflib.types import Vertex
from vmflib.brush import DispInfo
from vmflib.tools import Block

# Attempt to import 3D modules
try:
    import pyqtgraph as pg
    import pyqtgraph.opengl as gl
    _has_gl = True
except ModuleNotFoundError:
    _has_gl = False

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        QApplication.setWindowIcon(QIcon(":/resources/D-Gen_Icon.ico"))
        self.setWindowTitle("DispGen Reborn")
        self.orig_image = None
        self.image = None

        # Main layout
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)

        # Controls
        ctrl_widget = QWidget()
        ctrl_layout = QVBoxLayout(ctrl_widget)
        main_layout.addWidget(ctrl_widget, 0)

        # Load
        self.load_btn = QPushButton("Load Heightmap")
        self.load_btn.clicked.connect(self.load_image)
        ctrl_layout.addWidget(self.load_btn)

        # Tiles X/Y
        ctrl_layout.addWidget(QLabel("Tiles X:"))
        self.tiles_x_spin = QSpinBox()
        self.tiles_x_spin.setRange(1, 256)
        self.tiles_x_spin.setValue(32)
        self.tiles_x_spin.valueChanged.connect(self.update_and_preview)
        ctrl_layout.addWidget(self.tiles_x_spin)

        ctrl_layout.addWidget(QLabel("Tiles Y:"))
        self.tiles_y_spin = QSpinBox()
        self.tiles_y_spin.setRange(1, 256)
        self.tiles_y_spin.setValue(32)
        self.tiles_y_spin.valueChanged.connect(self.update_and_preview)
        ctrl_layout.addWidget(self.tiles_y_spin)

        # Tile size / height
        ctrl_layout.addWidget(QLabel("Tile Size (units):"))
        self.tile_spin = QSpinBox()
        self.tile_spin.setRange(1, 10000)
        self.tile_spin.setValue(512)
        self.tile_spin.valueChanged.connect(self.update_and_preview)
        ctrl_layout.addWidget(self.tile_spin)

        ctrl_layout.addWidget(QLabel("Max Height (units):"))
        self.height_spin = QSpinBox()
        self.height_spin.setRange(1, 10000)
        self.height_spin.setValue(2048)
        self.height_spin.valueChanged.connect(self.update_and_preview)
        ctrl_layout.addWidget(self.height_spin)

        # Material
        ctrl_layout.addWidget(QLabel("Material:"))
        self.material_input = QLineEdit('dev/dev_blendmeasure')
        ctrl_layout.addWidget(self.material_input)

        # Disp Power
        ctrl_layout.addWidget(QLabel("Disp Power:"))
        self.power_combo = QComboBox()
        self.power_combo.addItems(['2', '3', '4'])
        self.power_combo.setCurrentText('3')
        self.power_combo.currentIndexChanged.connect(self.update_and_preview)
        ctrl_layout.addWidget(self.power_combo)

        # Auto Update
        self.auto_update_check = QCheckBox("Auto Update")
        self.auto_update_check.setChecked(True)
        ctrl_layout.addWidget(self.auto_update_check)

        # Show Tiles
        self.grid_check = QCheckBox("Show Tiles")
        self.grid_check.setChecked(False)
        self.grid_check.toggled.connect(self.preview_3d)
        ctrl_layout.addWidget(self.grid_check)

        # Preview / Generate
        self.preview_btn = QPushButton("Preview 3D")
        self.preview_btn.clicked.connect(self.preview_3d)
        self.preview_btn.setEnabled(False)
        ctrl_layout.addWidget(self.preview_btn)

        self.gen_btn = QPushButton("Generate VMF")
        self.gen_btn.clicked.connect(self.generate_vmf)
        self.gen_btn.setEnabled(False)
        ctrl_layout.addWidget(self.gen_btn)

        ctrl_layout.addStretch()

        # 3D View
        if _has_gl:
            self.view = gl.GLViewWidget()
            main_layout.addWidget(self.view, 1)
        else:
            placeholder = QLabel("3D preview disabled.\nInstall PyOpenGL and pyqtgraph.")
            placeholder.setAlignment(Qt.AlignCenter)
            main_layout.addWidget(placeholder, 1)

    def load_image(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Heightmap", filter="Images (*.png *.bmp *.jpg *.tga)"
        )
        if not path:
            return
        try:
            self.orig_image = Image.open(path).convert('L')
            self.update_image_size()
            self.preview_btn.setEnabled(True)
            self.gen_btn.setEnabled(True)
            if self.auto_update_check.isChecked():
                self.preview_3d()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load image:\n{e}")

    def update_image_size(self):
        if not self.orig_image:
            return
        w = self.tiles_x_spin.value() * 8
        h = self.tiles_y_spin.value() * 8
        self.image = self.orig_image.resize((w, h), Image.BILINEAR)

    def update_and_preview(self):
        self.update_image_size()
        if self.auto_update_check.isChecked() and self.preview_btn.isEnabled():
            self.preview_3d()

    def preview_3d(self):
        if not _has_gl or self.image is None:
            return
        cols, rows = self.image.size
        tx = self.tiles_x_spin.value()
        ty = self.tiles_y_spin.value()
        ts = self.tile_spin.value()
        hscale = self.height_spin.value()
        if cols != tx * 8 or rows != ty * 8:
            return

        data = np.array(self.image)

        # Preview: bilinear interpolate height for chosen power
        power = int(self.power_combo.currentText())
        disp_res = 2**power
        nx = tx * disp_res + 1
        ny = ty * disp_res + 1
        xi = np.linspace(0, cols - 1, nx)
        yi = np.linspace(0, rows - 1, ny)
        xi_m, yi_m = np.meshgrid(xi, yi)
        x0 = np.floor(xi_m).astype(int)
        y0 = np.floor(yi_m).astype(int)
        x1 = np.clip(x0 + 1, 0, cols - 1)
        y1 = np.clip(y0 + 1, 0, rows - 1)
        dx = xi_m - x0
        dy = yi_m - y0
        h00 = data[y0, x0]
        h10 = data[y0, x1]
        h01 = data[y1, x0]
        h11 = data[y1, x1]
        hvals = (h00 * (1 - dx) * (1 - dy)
                 + h10 * dx * (1 - dy)
                 + h01 * (1 - dx) * dy
                 + h11 * dx * dy)
        zz = hvals / 255.0 * hscale

        xs = np.linspace(0, tx * ts, nx)
        ys = np.linspace(0, ty * ts, ny)
        xx, yy = np.meshgrid(xs, ys)
        verts = np.column_stack((xx.flatten(), yy.flatten(), zz.flatten()))

        faces = []
        for i in range(ny - 1):
            for j in range(nx - 1):
                idx = i * nx + j
                faces.append([idx, idx + 1, idx + nx])
                faces.append([idx + 1, idx + nx + 1, idx + nx])
        faces = np.array(faces)

        # Color
        zmin, zmax = zz.min(), zz.max()
        norm = (zz.flatten() - zmin) / (zmax - zmin if zmax != zmin else 1)
        colors = np.vstack([norm, norm, norm, np.ones_like(norm)]).T

        self.view.clear()
        mesh = gl.GLMeshItem(
            vertexes=verts,
            faces=faces,
            vertexColors=colors,
            smooth=False,
            drawEdges=False,
            drawFaces=True,
            glOptions='opaque'
        )
        self.view.addItem(mesh)

        # Tile borders
        if self.grid_check.isChecked():
            border_color = (0.2, 0.2, 0.2, 1.0)
            for i in range(1, tx):
                idx = i * disp_res
                pts = np.column_stack((
                    np.full(ny, xs[idx]),
                    ys,
                    zz[:, idx]
                ))
                self.view.addItem(gl.GLLinePlotItem(pos=pts, color=border_color, width=2, antialias=True))
            for j in range(1, ty):
                idx = j * disp_res
                pts = np.column_stack((
                    xs,
                    np.full(nx, ys[idx]),
                    zz[idx, :]
                ))
                self.view.addItem(gl.GLLinePlotItem(pos=pts, color=border_color, width=2, antialias=True))

        # Camera fit
        center = QVector3D(tx * ts / 2, ty * ts / 2, hscale / 2)
        dist = max(tx * ts, ty * ts) * 1.5
        self.view.setCameraPosition(distance=dist, elevation=30, azimuth=45)
        self.view.opts['center'] = center

    def generate_vmf(self):
        if self.image is None:
            QMessageBox.warning(self, "No Heightmap", "Load a heightmap first.")
            return
        ts = self.tile_spin.value()
        hscale = self.height_spin.value()
        power = int(self.power_combo.currentText())
        disp_res = 2**power
        size = disp_res + 1
        mat = self.material_input.text().strip() or 'dev/dev_blendmeasure'

        out, _ = QFileDialog.getSaveFileName(
            self, "Save VMF...", filter="VMF Files (*.vmf)"
        )
        if not out:
            return

        try:
            m = vmf.ValveMap()
            data = np.array(self.image)
            cols, rows = data.shape[1], data.shape[0]

            for ty_i in range(self.tiles_y_spin.value()):
                for tx_i in range(self.tiles_x_spin.value()):
                    ox, oy = tx_i * ts, ty_i * ts
                    normals = [[Vertex(0, 0, 1) for _ in range(size)] for _ in range(size)]

                                        # Bilinear interpolate distances matching preview
                    xi = np.linspace(tx_i * 8, tx_i * 8 + 8, size)
                    yi = np.linspace(ty_i * 8, ty_i * 8 + 8, size)
                    xi_m, yi_m = np.meshgrid(xi, yi)
                    # clamp x0,y0 to valid range
                    x0 = np.floor(xi_m).astype(int)
                    y0 = np.floor(yi_m).astype(int)
                    x0 = np.clip(x0, 0, cols - 1)
                    y0 = np.clip(y0, 0, rows - 1)
                    x1 = np.clip(x0 + 1, 0, cols - 1)
                    y1 = np.clip(y0 + 1, 0, rows - 1)
                    dx = xi_m - x0
                    dy = yi_m - y0
                    h00 = data[y0, x0]
                    h10 = data[y0, x1]
                    h01 = data[y1, x0]
                    h11 = data[y1, x1]
                    hvals = (
                        h00 * (1 - dx) * (1 - dy)
                        + h10 * dx * (1 - dy)
                        + h01 * (1 - dx) * dy
                        + h11 * dx * dy
                    )
                    distances = [
                        [int(round(val / 255.0 * hscale)) for val in row]
                        for row in hvals
                    ]
                    disp = DispInfo(power, normals, distances)
                    floor = Block(Vertex(ox, oy, 0), (ts, ts, hscale), mat)
                    floor.top().lightmapscale = 32
                    floor.top().children.append(disp)
                    m.world.children.append(floor)

            m.write_vmf(out)
            QMessageBox.information(self, "Success", f"VMF saved to: {out}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error writing VMF:\n{e}")

if __name__ == '__main__':
    if _has_gl:
        pg.setConfigOptions(antialias=True)
    app = QApplication(sys.argv)
    win = MainWindow()
    win.resize(1024, 768)
    win.show()
    sys.exit(app.exec_())
