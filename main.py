#!/usr/bin/env python3
import sys
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QPushButton, QSpinBox,
    QLabel, QLineEdit, QFileDialog, QMessageBox, QCheckBox,
    QComboBox, QHBoxLayout, QVBoxLayout
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QIcon, QVector3D
import resources_rc  # Resource file for icons, embedded via PyQt's resource system
from PIL import Image  # PIL for image loading and manipulation
import numpy as np  # NumPy for numerical operations on arrays

# VMF libraries for building Valve Map Format files
from vmflib import vmf
from vmflib.types import Vertex
from vmflib.brush import DispInfo
from vmflib.tools import Block

# Attempt to import OpenGL-based 3D preview modules
try:
    import pyqtgraph as pg
    import pyqtgraph.opengl as gl
    HAS_OPENGL = True  # We have OpenGL support for real-time preview
except ModuleNotFoundError:
    HAS_OPENGL = False  # 3D preview will be disabled if imports fail

class DispGenWindow(QMainWindow):
    """
    Main application window for DispGen: a heightmap-to-VMF generator.
    Includes controls for image import, tiling, scaling, and preview.
    """
    def __init__(self):
        super().__init__()
        # Set window icon and title
        QApplication.setWindowIcon(QIcon(":/resources/D-Gen_Icon.ico"))
        self.setWindowTitle("DispGen Reborn v1.1")

        # Store original and resized heightmap images
        self.original_image = None
        self.resized_image = None

        # Set up main container and layout: two columns (controls + preview)
        main_container = QWidget()
        self.setCentralWidget(main_container)
        main_layout = QHBoxLayout(main_container)

        # -- Control Panel (left) --
        controls_panel = QWidget()
        controls_layout = QVBoxLayout(controls_panel)
        main_layout.addWidget(controls_panel, stretch=0)

        # Button: Load a heightmap image from disk
        self.load_heightmap_button = QPushButton("Load Heightmap")
        self.load_heightmap_button.clicked.connect(self.on_load_heightmap)
        controls_layout.addWidget(self.load_heightmap_button)

        # Numeric spinboxes for tile counts in X and Y directions
        controls_layout.addWidget(QLabel("Tiles X:"))
        self.tiles_x_spinbox = QSpinBox()
        self.tiles_x_spinbox.setRange(1, 256)
        self.tiles_x_spinbox.setValue(32)
        self.tiles_x_spinbox.valueChanged.connect(self.on_parameters_changed)
        controls_layout.addWidget(self.tiles_x_spinbox)

        controls_layout.addWidget(QLabel("Tiles Y:"))
        self.tiles_y_spinbox = QSpinBox()
        self.tiles_y_spinbox.setRange(1, 256)
        self.tiles_y_spinbox.setValue(32)
        self.tiles_y_spinbox.valueChanged.connect(self.on_parameters_changed)
        controls_layout.addWidget(self.tiles_y_spinbox)

        # Spinbox for in-game tile size (in units)
        controls_layout.addWidget(QLabel("Tile Size (units):"))
        self.tile_size_spinbox = QSpinBox()
        self.tile_size_spinbox.setRange(1, 10000)
        self.tile_size_spinbox.setValue(512)
        self.tile_size_spinbox.valueChanged.connect(self.on_parameters_changed)
        controls_layout.addWidget(self.tile_size_spinbox)

        # Spinbox for maximum height scale
        controls_layout.addWidget(QLabel("Max Height (units):"))
        self.max_height_spinbox = QSpinBox()
        self.max_height_spinbox.setRange(1, 10000)
        self.max_height_spinbox.setValue(2048)
        self.max_height_spinbox.valueChanged.connect(self.on_parameters_changed)
        controls_layout.addWidget(self.max_height_spinbox)

        # Text input for material path used on generated brushes
        controls_layout.addWidget(QLabel("Material:"))
        self.material_path_input = QLineEdit('dev/dev_blendmeasure')
        controls_layout.addWidget(self.material_path_input)

        # Combo box for displacement resolution (power-of-two subdivisions)
        controls_layout.addWidget(QLabel("Disp Power:"))
        self.displacement_power_combo = QComboBox()
        self.displacement_power_combo.addItems(['2', '3', '4'])
        self.displacement_power_combo.setCurrentText('3')
        self.displacement_power_combo.currentIndexChanged.connect(self.on_parameters_changed)
        controls_layout.addWidget(self.displacement_power_combo)

        # Checkbox: automatically update preview when parameters change
        self.auto_update_checkbox = QCheckBox("Auto Update")
        self.auto_update_checkbox.setChecked(True)
        controls_layout.addWidget(self.auto_update_checkbox)

        # Checkbox: toggle drawing of tile boundary grid in preview
        self.show_grid_checkbox = QCheckBox("Show Tiles")
        self.show_grid_checkbox.setChecked(False)
        self.show_grid_checkbox.toggled.connect(self.render_3d_preview)
        controls_layout.addWidget(self.show_grid_checkbox)

        # Buttons for manual preview and VMF export
        self.preview_button = QPushButton("Preview 3D")
        self.preview_button.clicked.connect(self.render_3d_preview)
        self.preview_button.setEnabled(False)
        controls_layout.addWidget(self.preview_button)

        self.generate_button = QPushButton("Generate VMF")
        self.generate_button.clicked.connect(self.export_vmf)
        self.generate_button.setEnabled(False)
        controls_layout.addWidget(self.generate_button)

        # Label showing final world dimensions in units and approximate meters
        self.dimensions_label = QLabel(
            "Final Area: \n"
            "x = 16384 units (416.15m) \n"
            "y = 16384 units (416.15m) \n"
            "z = 2048 units (52.02m)"
        )
        controls_layout.addWidget(self.dimensions_label)
        controls_layout.addStretch()  # Push controls to top

        # -- 3D Preview (right) --
        if HAS_OPENGL:
            # Use pyqtgraph's OpenGL widget for real-time mesh display
            self.gl_view = gl.GLViewWidget()
            main_layout.addWidget(self.gl_view, stretch=1)
        else:
            # Placeholder text if OpenGL is unavailable
            placeholder_label = QLabel(
                "3D preview disabled.\nInstall PyOpenGL and pyqtgraph."
            )
            placeholder_label.setAlignment(Qt.AlignCenter)
            main_layout.addWidget(placeholder_label, stretch=1)

    def on_load_heightmap(self):
        """
        Handler for clicking 'Load Heightmap'.
        Opens a file dialog, loads an 8-bit grayscale image, flips it, and triggers preview.
        """
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Open Heightmap", filter="Images (*.png *.bmp *.jpg *.tga)"
        )
        if not file_path:
            return  # User cancelled
        try:
            # Load and convert to grayscale
            loaded_image = Image.open(file_path).convert('L')
            # Flip horizontally to correct orientation
            self.original_image = loaded_image.transpose(Image.FLIP_LEFT_RIGHT)
            self.resize_heightmap_image()
            # Enable preview and generate now that we have an image
            self.preview_button.setEnabled(True)
            self.generate_button.setEnabled(True)
            if self.auto_update_checkbox.isChecked():
                self.render_3d_preview()
        except Exception as error:
            # Show error popup if loading fails
            QMessageBox.critical(
                self, "Error", f"Failed to load heightmap:\n{error}"
            )

    def on_parameters_changed(self):
        """
        Triggered when any spinbox or combo box value changes.
        Resizes the image for tiling, optionally updates preview, and updates the dimension label.
        """
        self.resize_heightmap_image()
        if self.auto_update_checkbox.isChecked() and self.preview_button.isEnabled():
            self.render_3d_preview()
        self.update_dimensions_label()

    def resize_heightmap_image(self):
        """
        Rescale the original heightmap to match the selected number of tiles.
        Each tile corresponds to an 8x8 pixel block in the heightmap.
        """
        if self.original_image is None:
            return
        # Compute new size (tiles_x * 8, tiles_y * 8)
        target_width = self.tiles_x_spinbox.value() * 8
        target_height = self.tiles_y_spinbox.value() * 8
        self.resized_image = self.original_image.resize(
            (target_width, target_height), Image.BILINEAR
        )  # Bilinear interpolation for smoother heights

    def update_dimensions_label(self):
        """
        Update the text label showing final map dimensions in engine units and meters.
        """
        num_tiles_x = self.tiles_x_spinbox.value()
        num_tiles_y = self.tiles_y_spinbox.value()
        tile_size = self.tile_size_spinbox.value()
        max_height = self.max_height_spinbox.value()

        # Compute world extents
        total_width = num_tiles_x * tile_size
        total_depth = num_tiles_y * tile_size
        total_height = max_height

        # Convert units to meters (1 unit = 1/39.37 m)
        meters_width = round(total_width / 39.37, 2)
        meters_depth = round(total_depth / 39.37, 2)
        meters_height = round(total_height / 39.37, 2)

        # Update label text with calculated values
        self.dimensions_label.setText(
            f"Final Area: \n"
            f"x = {total_width} units ({meters_width}m) \n"
            f"y = {total_depth} units ({meters_depth}m) \n"
            f"z = {total_height} units ({meters_height}m)"
        )

    def render_3d_preview(self):
        """
        Generate and display a 3D mesh from the resized heightmap in the OpenGL widget.
        Uses bilinear interpolation to match the VMF displacement sampling.
        """
        if not HAS_OPENGL or self.resized_image is None:
            return

        # Convert heightmap to NumPy array for fast math
        heightmap_data = np.array(self.resized_image)
        tiles_x = self.tiles_x_spinbox.value()
        tiles_y = self.tiles_y_spinbox.value()
        tile_size = self.tile_size_spinbox.value()
        height_scale = self.max_height_spinbox.value()
        power_level = int(self.displacement_power_combo.currentText())
        resolution = 2 ** power_level  # subdivisions per tile edge

        # Calculate numbers of vertices in x/y directions (including shared edges)
        vertices_x = tiles_x * resolution + 1
        vertices_y = tiles_y * resolution + 1

        # Create sampling grids over pixel space
        sample_x = np.linspace(0, heightmap_data.shape[1] - 1, vertices_x)
        sample_y = np.linspace(0, heightmap_data.shape[0] - 1, vertices_y)
        grid_x, grid_y = np.meshgrid(sample_x, sample_y)

        # Find integer indices for bilinear interpolation corners
        x0 = np.floor(grid_x).astype(int)
        y0 = np.floor(grid_y).astype(int)
        x1 = np.clip(x0 + 1, 0, heightmap_data.shape[1] - 1)
        y1 = np.clip(y0 + 1, 0, heightmap_data.shape[0] - 1)
        dx = grid_x - x0  # fractional offsets
        dy = grid_y - y0

        # Gather corner heights
        h00 = heightmap_data[y0, x0]
        h10 = heightmap_data[y0, x1]
        h01 = heightmap_data[y1, x0]
        h11 = heightmap_data[y1, x1]

        # Bilinear interpolation formula
        interpolated = (
            h00 * (1 - dx) * (1 - dy)
            + h10 * dx * (1 - dy)
            + h01 * (1 - dx) * dy
            + h11 * dx * dy
        )
        # Scale to world height
        height_values = interpolated / 255.0 * height_scale

        # Generate world coordinates for vertices
        world_x = np.linspace(0, tiles_x * tile_size, vertices_x)
        world_y = np.linspace(0, tiles_y * tile_size, vertices_y)
        mesh_x, mesh_y = np.meshgrid(world_x, world_y)
        mesh_vertices = np.column_stack((mesh_x.flatten(), mesh_y.flatten(), height_values.flatten()))

        # Build face index list (two triangles per quad)
        mesh_faces = []
        for row in range(vertices_y - 1):
            for col in range(vertices_x - 1):
                idx = row * vertices_x + col
                # Triangle 1
                mesh_faces.append([idx, idx + 1, idx + vertices_x])
                # Triangle 2
                mesh_faces.append([idx + 1, idx + vertices_x + 1, idx + vertices_x])
        mesh_faces = np.array(mesh_faces)

        # Color by normalized height (grayscale)
        min_h, max_h = height_values.min(), height_values.max()
        if max_h != min_h:
            normalized = (height_values.flatten() - min_h) / (max_h - min_h)
        else:
            normalized = np.zeros_like(height_values.flatten())
        mesh_colors = np.vstack([normalized, normalized, normalized, np.ones_like(normalized)]).T

        # Clear previous items and add new mesh
        self.gl_view.clear()
        mesh_item = gl.GLMeshItem(
            vertexes=mesh_vertices,
            faces=mesh_faces,
            vertexColors=mesh_colors,
            smooth=False,
            drawEdges=False,
            drawFaces=True,
            glOptions='opaque'
        )
        self.gl_view.addItem(mesh_item)

        # Draw grid lines if requested
        if self.show_grid_checkbox.isChecked():
            line_color = (0.2, 0.2, 0.2, 1.0)
            # Vertical lines between tile columns
            for tile_idx in range(1, tiles_x):
                x_coord = tile_idx * resolution
                line_positions = np.column_stack((
                    np.full(vertices_y, world_x[x_coord]),
                    world_y,
                    height_values[:, x_coord]
                ))
                self.gl_view.addItem(gl.GLLinePlotItem(pos=line_positions, color=line_color, width=2, antialias=True))
            # Horizontal lines between tile rows
            for tile_idx in range(1, tiles_y):
                y_coord = tile_idx * resolution
                line_positions = np.column_stack((
                    world_x,
                    np.full(vertices_x, world_y[y_coord]),
                    height_values[y_coord, :]
                ))
                self.gl_view.addItem(gl.GLLinePlotItem(pos=line_positions, color=line_color, width=2, antialias=True))

        # Position camera to view entire mesh
        center_point = QVector3D(tiles_x * tile_size / 2, tiles_y * tile_size / 2, height_scale / 2)
        camera_distance = max(tiles_x * tile_size, tiles_y * tile_size) * 1.5
        self.gl_view.setCameraPosition(distance=camera_distance, elevation=30, azimuth=45)
        self.gl_view.opts['center'] = center_point

    def export_vmf(self):
        """
        Generate a .vmf file from the resized heightmap.
        Iterates over each tile, samples heights, creates displacement brushes, and writes the VMF.
        """
        if self.resized_image is None:
            QMessageBox.warning(self, "No Heightmap", "Load a heightmap first.")
            return

        tile_size = self.tile_size_spinbox.value()
        height_scale = self.max_height_spinbox.value()
        power_level = int(self.displacement_power_combo.currentText())
        resolution = 2 ** power_level
        sample_size = resolution + 1
        material_path = self.material_path_input.text().strip() or 'dev/dev_blendmeasure'

        save_path, _ = QFileDialog.getSaveFileName(self, "Save VMF...", filter="VMF Files (*.vmf)")
        if not save_path:
            return

        try:
            valve_map = vmf.ValveMap()  # Root VMF object
            height_array = np.array(self.resized_image)
            img_width, img_height = height_array.shape[1], height_array.shape[0]

            # Loop over each tile row & column
            for row_idx in range(self.tiles_y_spinbox.value()):
                for col_idx in range(self.tiles_x_spinbox.value()):
                    offset_x = col_idx * tile_size
                    offset_y = row_idx * tile_size

                    # Prepare default vertex normals facing up
                    vertex_normals = [[Vertex(0, 0, 1) for _ in range(sample_size)] for _ in range(sample_size)]

                    # Sample heights using bilinear interpolation
                    xs = np.linspace(col_idx * 8, col_idx * 8 + 8, sample_size)
                    ys = np.linspace(row_idx * 8, row_idx * 8 + 8, sample_size)
                    grid_x, grid_y = np.meshgrid(xs, ys)
                    x0 = np.clip(np.floor(grid_x).astype(int), 0, img_width - 1)
                    y0 = np.clip(np.floor(grid_y).astype(int), 0, img_height - 1)
                    x1 = np.clip(x0 + 1, 0, img_width - 1)
                    y1 = np.clip(y0 + 1, 0, img_height - 1)
                    dx = grid_x - x0
                    dy = grid_y - y0

                    h00 = height_array[y0, x0]
                    h10 = height_array[y0, x1]
                    h01 = height_array[y1, x0]
                    h11 = height_array[y1, x1]

                    interpolated_heights = (
                        h00 * (1 - dx) * (1 - dy)
                        + h10 * dx * (1 - dy)
                        + h01 * (1 - dx) * dy
                        + h11 * dx * dy
                    )

                    # Convert pixel values to integer world heights
                    height_distances = [[int(round(val / 255.0 * height_scale)) for val in row] for row in interpolated_heights]
                    disp_info = DispInfo(power_level, vertex_normals, height_distances)

                    # Create base brush for this tile and attach displacement
                    floor_block = Block(Vertex(offset_x, offset_y, 0), (tile_size, tile_size, 16), material_path)
                    floor_block.top().lightmapscale = 32
                    floor_block.top().children.append(disp_info)
                    valve_map.world.children.append(floor_block)

            valve_map.write_vmf(save_path)
            QMessageBox.information(self, "Success", f"VMF saved to: {save_path}")
        except Exception as error:
            QMessageBox.critical(self, "Error", f"Error writing VMF:\n{error}")

if __name__ == '__main__':
    # Enable antialiasing for nicer lines if OpenGL is available
    if HAS_OPENGL:
        pg.setConfigOptions(antialias=True)
    app = QApplication(sys.argv)
    main_window = DispGenWindow()
    main_window.resize(1024, 768)
    main_window.show()
    sys.exit(app.exec_())
