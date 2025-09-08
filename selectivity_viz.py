# it_3d_mwu_viewer.py – rev-10
"""Interactive 3-D viewer for Mann-Whitney-U statistics in IT units

* Spheres represent units, shaded via viridis colormap.
* Custom XYZ axes colored per AXIS_COLORS_HEX; C-axis scale adjustable via slider.
* Legend in control bar uses same colors from shared dict.
* Mouse: left=orbit, right=pan, wheel=zoom.

Run:

    python it_3d_mwu_viewer.py  all_layers_units_mannwhitneyu.pkl
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PyQt5 import QtCore, QtWidgets
import pyqtgraph as pg
import pyqtgraph.opengl as gl

# Axis colors: W (X)=red, H (Y)=green, C (Z)=blue
AXIS_COLORS_HEX = {"W": "#ff0000", "H": "#00ff00", "C": "#0000ff"}

_SLIDER_RETURNS_TUPLE: bool
try:
    from superqt import QRangeSlider as _Slider  # type: ignore
    _SLIDER_RETURNS_TUPLE = True
except ImportError:
    try:
        from qt_range_slider import QtRangeSlider as _Slider  # type: ignore
        _SLIDER_RETURNS_TUPLE = False
    except ImportError as exc:
        raise SystemExit(
            "You need either the *superqt* or *qt-range-slider* package.\n"
            "Install one of them with, e.g.:  pip install superqt"
        ) from exc

class IT3DViewer(QtWidgets.QWidget):
    SLIDER_PRECISION = 3
    SPHERE_RADIUS = 0.18
    C = 512
    H = 7
    W = 7

    def __init__(self, pkl_path: str | Path) -> None:
        super().__init__()
        self.setWindowTitle("IT unit 3-D Mann-Whitney U viewer (drag to rotate)")
        self.resize(1000, 700)

        self.df: pd.DataFrame = pd.read_pickle(Path(pkl_path))
        self._prepare_dataframe()
        self.Z_SCALE = 1.0  # initial C-axis scale
        self._build_ui()
        self._populate_first_category()

    def _prepare_dataframe(self) -> None:
        req = {"layer", "unit"}
        if not req.issubset(self.df.columns):
            raise ValueError(f"Pickle missing columns: {req - set(self.df.columns)}")
        self.df = self.df[self.df["layer"] == "module.IT"].copy()
        if self.df.empty:
            raise ValueError("No IT units found in the provided pickle file")
        self.df["unit"] = self.df["unit"].astype(int)
        self.df["c"], rem = divmod(self.df["unit"], self.H * self.W)
        self.df["y"], self.df["x"] = divmod(rem, self.W)
        self.categories = [c[3:] for c in self.df.columns if c.startswith("mw_")]
        if not self.categories:
            raise ValueError("No 'mw_*' columns found – nothing to visualise.")

    def _build_ui(self) -> None:
        layout = QtWidgets.QVBoxLayout(self)
        bar = QtWidgets.QHBoxLayout(); layout.addLayout(bar)

        bar.addWidget(QtWidgets.QLabel("Category:"))
        self.cat_dropdown = QtWidgets.QComboBox(); bar.addWidget(self.cat_dropdown)
        self.cat_dropdown.addItems(self.categories)
        self.cat_dropdown.currentTextChanged.connect(self._on_category_changed)

        bar.addSpacing(20); bar.addWidget(QtWidgets.QLabel("U-value range:"))
        self.slider = _Slider(QtCore.Qt.Horizontal); self.slider.setMinimumWidth(200)
        bar.addWidget(self.slider)
        if _SLIDER_RETURNS_TUPLE:
            self.slider.valueChanged.connect(self._on_slider_changed)
        else:
            self.slider.startValueChanged.connect(self._on_slider_changed)  # type: ignore
            self.slider.endValueChanged.connect(self._on_slider_changed)    # type: ignore

        bar.addSpacing(30)
        bar.addWidget(QtWidgets.QLabel("C-axis scale:"))
        self.scale_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.scale_slider.setRange(10, 200)
        self.scale_slider.setValue(100)
        self.scale_slider.setFixedWidth(150)
        self.scale_slider.valueChanged.connect(self._on_scale_changed)
        bar.addWidget(self.scale_slider)

        bar.addSpacing(20)
        for axis, hexcol in AXIS_COLORS_HEX.items():
            lbl = QtWidgets.QLabel(axis)
            lbl.setStyleSheet(f"color: {hexcol}; font-weight: bold")
            bar.addWidget(lbl); bar.addSpacing(8)
        bar.addStretch(1)

        self.glview = gl.GLViewWidget(); layout.addWidget(self.glview, 1)
        self.glview.setBackgroundColor(pg.mkColor("#111"))
        self.scatter_items: list[gl.GLMeshItem] = []
        self.axis_lines: list[gl.GLLinePlotItem] = []
        self._draw_axes()

    def _draw_axes(self) -> None:
        for line in self.axis_lines:
            self.glview.removeItem(line)
        self.axis_lines.clear()
        center = np.array([self.W - 1, self.H - 1, self.C - 1]) / 2
        axes = {
            "W": np.array([[0, 0, 0], [self.W, 0, 0]]),
            "H": np.array([[0, 0, 0], [0, self.H, 0]]),
            "C": np.array([[0, 0, 0], [0, 0, self.C * self.Z_SCALE]]),
        }
        for axis, pts in axes.items():
            pts = pts - center * np.array([1, 1, self.Z_SCALE])
            col = pg.mkColor(AXIS_COLORS_HEX[axis]).getRgbF()[:3] + (1.0,)
            line = gl.GLLinePlotItem(pos=pts, color=col, width=2.0, antialias=True)
            self.glview.addItem(line)
            self.axis_lines.append(line)

    def _populate_first_category(self) -> None:
        self.current_category = self.cat_dropdown.currentText()
        self._update_slider_limits(); self._update_points()

    def _on_category_changed(self, txt: str) -> None:
        self.current_category = txt; self._update_slider_limits(); self._update_points()

    def _on_slider_changed(self, *_) -> None:
        self._update_points()

    def _on_scale_changed(self, val: int) -> None:
        self.Z_SCALE = val / 100.0
        self._draw_axes(); self._update_points()

    def _update_slider_limits(self) -> None:
        uvals = self.df[f"mw_{self.current_category}"].values.astype(float)
        umin, umax = float(np.nanmin(uvals)), float(np.nanmax(uvals))
        if math.isclose(umin, umax): umax = umin + 1e-6
        self._scale = 10 ** self.SLIDER_PRECISION
        imin, imax = int(math.floor(umin * self._scale)), int(math.ceil(umax * self._scale))
        if _SLIDER_RETURNS_TUPLE:
            self.slider.setRange(imin, imax); self.slider.setValue((imin, imax))  # type: ignore
        else:
            self.slider.setMinimum(imin); self.slider.setMaximum(imax)
            self.slider.setStart(imin); self.slider.setEnd(imax)

    def _current_slider_values(self) -> tuple[float, float]:
        if _SLIDER_RETURNS_TUPLE:
            lo, hi = self.slider.value()  # type: ignore
        else:
            lo, hi = int(self.slider.start()), int(self.slider.end())  # type: ignore
        return lo / self._scale, hi / self._scale

    def _update_points(self) -> None:
        for it in self.scatter_items: self.glview.removeItem(it)
        self.scatter_items.clear()
        uvals = self.df[f"mw_{self.current_category}"].values.astype(float)
        vmin, vmax = self._current_slider_values()
        mask = (uvals >= vmin) & (uvals <= vmax)
        if not mask.any(): return
        norm = (uvals[mask] - vmin) / max(vmax - vmin, 1e-9)
        cmap = pg.colormap.get("viridis"); cols = cmap.map(norm, mode="float")
        md = gl.MeshData.sphere(rows=6, cols=6); r = self.SPHERE_RADIUS
        center = np.array([self.W - 1, self.H - 1, self.C - 1]) / 2
        for (x, y, z), rgba in zip(self.df.loc[mask, ["x", "y", "c"]].values.astype(float), cols):
            pos = np.array([x, y, z * self.Z_SCALE]) - center * np.array([1, 1, self.Z_SCALE])
            m = gl.GLMeshItem(meshdata=md, color=tuple(rgba), smooth=True, shader="shaded", drawFaces=True)
            m.scale(r, r, r); m.translate(*pos)
            self.glview.addItem(m); self.scatter_items.append(m)
        self.glview.setCameraPosition(distance=max(self.W, self.H, self.C * self.Z_SCALE) * 1.8)


def _cli() -> None:
    if len(sys.argv) != 2: sys.exit("Usage: python it_3d_mwu_viewer.py <mannwhitneyu.pkl>")
    app = QtWidgets.QApplication([]); viewer = IT3DViewer(sys.argv[1]); viewer.show(); sys.exit(app.exec_())

if __name__ == "__main__": _cli()