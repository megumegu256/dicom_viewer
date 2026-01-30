# DICOM Viewer

PyVista-based single-file DICOM volume viewer. Loads the series under dcms/ and renders a 3D volume with axes and bounding box. Logs and screenshots are written under logs/.

## Usage
- Install dependencies: `pip install pydicom pyvista numpy`
- Run viewer (default dcms folder): `python main.py`
- Skip rendering (load only): `python main.py --no-render`
- Save screenshot: `python main.py --screenshot logs/volume.png`
- In the viewer: adjust Window/Level via sliders, press `c` to cycle colormap, `b` to toggle bounding box, `p` to save an interactive screenshot to logs/interactive.png.

## 変更履歴
- 2026-01-27: Add interactive GUI controls (window/level sliders, cmap cycle, bbox toggle, screenshot hotkey) and text status overlay.
- 2026-01-27: Add main.py DICOM viewer, logging to logs/, README usage and changelog.
