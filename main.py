"""
DICOM Medical Image Viewer Application
========================================
DICOMファイル形式の医療画像を表示・操作するGUIアプリケーション

機能:
- DICOMファイルの読み込みと表示
- Axial, Sagittal, Coronal スライス表示（Axialは必須）
- Window Width / Window Level スライダ調整
- スライス番号の表示と変更
- DICOMヘッダ情報の表示（画像サイズ、スライス厚、スライス数）
- スライス面の交差線表示
- オプション設定をconfig.jsonに保存・復元

使用方法:
1. アプリケーション起動後、[ファイルを開く] ボタンでDICOMフォルダを選択
2. スライダで画像を調整
3. チェックボックスで表示モード切り替え
"""

import sys
import json
import os
import numpy as np
from pathlib import Path
from typing import Optional, List, Tuple

# Qt プラットフォームプラグインパスを設定
if sys.platform == 'win32':
    # 仮想環境のPyQt5プラグインパスを取得
    venv_path = Path(sys.executable).parent.parent
    qt_plugin_path = venv_path / 'Lib' / 'site-packages' / 'PyQt5' / 'Qt5' / 'plugins'
    if qt_plugin_path.exists():
        os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = str(qt_plugin_path)

import pydicom
from pydicom.pixel_data_handlers.util import apply_modality_lut
import pyvista as pv
from pyvistaqt import QtInteractor
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QSlider, QCheckBox, QSpinBox, QGroupBox,
    QFileDialog, QGridLayout, QComboBox, QMessageBox, QStatusBar, QRadioButton, QButtonGroup
)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QFont


class DICOMSliceProcessor:
    """DICOMスライス処理クラス"""
    
    def __init__(self, dicom_files: List[str]):
        """
        Args:
            dicom_files: DICOMファイルパスのリスト（ファイル名順）
        """
        self.dicom_files = sorted(dicom_files)
        self.pixel_arrays = []
        self.dicom_datasets = []
        
        # DICOMファイルを読み込む
        self._load_dicom_files()
        
        # スライス情報の計算
        self.num_slices = len(self.dicom_datasets)
        self.slice_thickness = self._get_slice_thickness()
        self.spacing = self._get_spacing()
        self.image_height = self.pixel_arrays[0].shape[0] if self.pixel_arrays else 0
        self.image_width = self.pixel_arrays[0].shape[1] if self.pixel_arrays else 0
        if self.pixel_arrays:
            volume = np.array(self.pixel_arrays, dtype=np.float32)
            self.data_min = float(np.min(volume))
            self.data_max = float(np.max(volume))
        else:
            self.data_min = 0.0
            self.data_max = 0.0
    
    def _load_dicom_files(self) -> None:
        """DICOMファイルを読み込む"""
        datasets = []
        for file_path in self.dicom_files:
            try:
                ds = pydicom.dcmread(file_path)
                datasets.append(ds)
            except Exception as e:
                print(f"Error loading {file_path}: {e}")

        # InstanceNumberでソート（無ければ読み込み順）
        datasets.sort(key=lambda d: getattr(d, 'InstanceNumber', 0))

        for ds in datasets:
            try:
                # ピクセルデータの取得（HUなどの実データを保持）
                pixel_array = ds.pixel_array.astype(np.float32)

                # Modality LUTを適用してHU値へ変換
                pixel_array = apply_modality_lut(pixel_array, ds).astype(np.float32)

                self.pixel_arrays.append(pixel_array)
                self.dicom_datasets.append(ds)
            except Exception as e:
                print(f"Error processing DICOM: {e}")
    
    def _get_slice_thickness(self) -> float:
        """スライス厚を取得"""
        if self.dicom_datasets:
            ds = self.dicom_datasets[0]
            if hasattr(ds, 'SliceThickness'):
                return float(ds.SliceThickness)
        return 0.0

    def _get_spacing(self) -> Tuple[float, float, float]:
        """画素間隔とスライス間隔を推定（test_main.py準拠）"""
        if not self.dicom_datasets:
            return (1.0, 1.0, 1.0)

        first = self.dicom_datasets[0]
        try:
            row_spacing, col_spacing = map(float, first.PixelSpacing)
        except Exception:
            row_spacing = col_spacing = 1.0

        positions = []
        for ds in self.dicom_datasets:
            ipp = getattr(ds, 'ImagePositionPatient', None)
            if ipp is not None and len(ipp) == 3:
                positions.append(np.array(ipp, dtype=float))

        slice_spacing = None
        if len(positions) >= 2:
            diffs = [np.linalg.norm(positions[i + 1] - positions[i]) for i in range(len(positions) - 1)]
            slice_spacing = float(np.median(diffs))

        if slice_spacing is None:
            slice_spacing = float(getattr(first, 'SliceThickness', 1.0))

        return (col_spacing, row_spacing, slice_spacing)

    def get_default_window(self) -> Tuple[float, float]:
        """DICOMタグのWindow Center/Widthを取得（無ければ自動推定）"""
        if not self.dicom_datasets:
            return (400.0, 40.0)

        ds = self.dicom_datasets[0]
        wc = getattr(ds, 'WindowCenter', None)
        ww = getattr(ds, 'WindowWidth', None)

        def _first(v):
            if v is None:
                return None
            if isinstance(v, (list, tuple)):
                return v[0]
            try:
                return v[0]
            except Exception:
                return v

        wc = _first(wc)
        ww = _first(ww)

        if wc is not None and ww is not None:
            return (float(ww), float(wc))

        # フォールバック: データ範囲から推定
        width = max(self.data_max - self.data_min, 1.0)
        level = (self.data_max + self.data_min) * 0.5
        return (float(width), float(level))
    
    def get_slice(self, slice_idx: int, ww: int = 400, wl: int = 40) -> np.ndarray:
        """
        スライスを取得（Window Width/Levelを適用）
        
        Args:
            slice_idx: スライスインデックス
            ww: Window Width
            wl: Window Level
        
        Returns:
            処理されたスライス画像（uint8）
        """
        if 0 <= slice_idx < len(self.pixel_arrays):
            pixel_array = self.pixel_arrays[slice_idx].astype(np.float32)

            # Window Width/Level を適用
            min_v = wl - ww / 2
            max_v = wl + ww / 2
            if max_v <= min_v:
                max_v = min_v + 1.0
            scaled = (pixel_array - min_v) / (max_v - min_v) * 255.0
            scaled = np.clip(scaled, 0, 255)

            return scaled.astype(np.uint8)
        return np.zeros((self.image_height, self.image_width), dtype=np.uint8)
    
    def get_axial_slice(self, z: int, ww: int = 400, wl: int = 40) -> np.ndarray:
        """Axialスライス（Z軸）"""
        return self.get_slice(z, ww, wl)
    
    def get_sagittal_slice(self, x: int, ww: int = 400, wl: int = 40) -> np.ndarray:
        """Sagittalスライス（X軸）"""
        volume = np.array(self.pixel_arrays)
        if volume.size == 0:
            return np.zeros((self.num_slices, self.image_height), dtype=np.uint8)
        
        slice_data = volume[:, :, x].astype(np.float32)
        # Window Width/Level を適用
        lower = wl - ww / 2
        upper = wl + ww / 2
        slice_data = np.clip((slice_data - lower) / (upper - lower) * 255, 0, 255)
        return slice_data.astype(np.uint8)
    
    def get_coronal_slice(self, y: int, ww: int = 400, wl: int = 40) -> np.ndarray:
        """Coronalスライス（Y軸）"""
        volume = np.array(self.pixel_arrays)
        if volume.size == 0:
            return np.zeros((self.num_slices, self.image_width), dtype=np.uint8)
        
        slice_data = volume[:, y, :].astype(np.float32)
        # Window Width/Level を適用
        lower = wl - ww / 2
        upper = wl + ww / 2
        slice_data = np.clip((slice_data - lower) / (upper - lower) * 255, 0, 255)
        return slice_data.astype(np.uint8)
    
    def get_dicom_info(self) -> dict:
        """DICOMヘッダ情報を取得"""
        info = {
            'image_width': self.image_width,
            'image_height': self.image_height,
            'num_slices': self.num_slices,
            'slice_thickness': f"{self.slice_thickness:.2f}",
            'patient_name': 'N/A',
            'patient_id': 'N/A',
            'modality': 'N/A',
            'study_date': 'N/A'
        }
        
        if self.dicom_datasets:
            ds = self.dicom_datasets[0]
            if hasattr(ds, 'PatientName'):
                info['patient_name'] = str(ds.PatientName)
            if hasattr(ds, 'PatientID'):
                info['patient_id'] = str(ds.PatientID)
            if hasattr(ds, 'Modality'):
                info['modality'] = str(ds.Modality)
            if hasattr(ds, 'StudyDate'):
                info['study_date'] = str(ds.StudyDate)
        
        return info


class DICOMViewerApp(QMainWindow):
    """DICOMビューアアプリケーション"""
    
    def __init__(self):
        super().__init__()
        self.slice_processor: Optional[DICOMSliceProcessor] = None
        self.current_axial_slice = 0
        self.current_sagittal_pos = 0
        self.current_coronal_pos = 0
        self.window_width = 400
        self.window_level = 40
        self.display_mode_3d = False  # デフォルトは2D表示
        self.config_path = Path(__file__).parent / "config.json"
        self._update_timer: Optional[QTimer] = None  # スライダ点滅対策用
        
        # 最適化用フラグ：incremental更新管理
        self._axial_dirty = True  # Axialビューの更新が必要か
        self._sagittal_dirty = True  # Sagittalビューの更新が必要か
        self._coronal_dirty = True  # Coronalビューの更新が必要か
        
        self.init_ui()
        self.load_config()
        self.apply_theme(self.theme_combo.currentText())
        
    def init_ui(self) -> None:
        """ユーザーインターフェースの初期化"""
        self.setWindowTitle("DICOM Medical Image Viewer")
        self.setGeometry(100, 100, 1400, 900)
        
        # メインウィジェット
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        
        # レイアウト
        main_layout = QHBoxLayout()
        
        # ===== 左側：PyVistaビューアとスライダー =====
        left_layout = QVBoxLayout()
        
        # 2D/3D表示用にプロッターをレイアウトコンテナに配置
        self.viewer_container = QWidget()
        self.viewer_layout = QVBoxLayout(self.viewer_container)
        self.viewer_layout.setContentsMargins(0, 0, 0, 0)
        
        # 2D表示用：shape=(2,2)で複数ビュー表示
        self.plotter_2d = QtInteractor(self.viewer_container, shape=(2, 2))
        self.plotter = self.plotter_2d  # デフォルトで2D表示
        self.viewer_layout.addWidget(self.plotter.interactor)
        
        # 3D表示用：shape=(1,1)で全画面表示
        self.plotter_3d = QtInteractor(self.viewer_container, shape=(1, 1))
        self.plotter_3d.interactor.hide()  # 初期状態で非表示
        self.viewer_layout.addWidget(self.plotter_3d.interactor)
        
        left_layout.addWidget(self.viewer_container)
        
        # 各画像のスライダー（2D表示時のみ有効）
        sliders_layout = QHBoxLayout()
        
        # Axial スライダー
        axial_slider_layout = QVBoxLayout()
        axial_slider_layout.addWidget(QLabel("Axial"))
        self.axial_slider_2d = QSlider(Qt.Horizontal)
        self.axial_slider_2d.setMinimum(0)
        self.axial_slider_2d.valueChanged.connect(self.on_axial_slice_changed)
        axial_slider_layout.addWidget(self.axial_slider_2d)
        self.axial_label_2d = QLabel("0/0")
        self.axial_label_2d.setAlignment(Qt.AlignCenter)
        axial_slider_layout.addWidget(self.axial_label_2d)
        sliders_layout.addLayout(axial_slider_layout)
        
        # Sagittal スライダー
        sagittal_slider_layout = QVBoxLayout()
        sagittal_slider_layout.addWidget(QLabel("Sagittal"))
        self.sagittal_slider_2d = QSlider(Qt.Horizontal)
        self.sagittal_slider_2d.setMinimum(0)
        self.sagittal_slider_2d.valueChanged.connect(self.on_sagittal_pos_changed)
        sagittal_slider_layout.addWidget(self.sagittal_slider_2d)
        self.sagittal_label_2d = QLabel("0/0")
        self.sagittal_label_2d.setAlignment(Qt.AlignCenter)
        sagittal_slider_layout.addWidget(self.sagittal_label_2d)
        sliders_layout.addLayout(sagittal_slider_layout)
        
        # Coronal スライダー
        coronal_slider_layout = QVBoxLayout()
        coronal_slider_layout.addWidget(QLabel("Coronal"))
        self.coronal_slider_2d = QSlider(Qt.Horizontal)
        self.coronal_slider_2d.setMinimum(0)
        self.coronal_slider_2d.valueChanged.connect(self.on_coronal_pos_changed)
        coronal_slider_layout.addWidget(self.coronal_slider_2d)
        self.coronal_label_2d = QLabel("0/0")
        self.coronal_label_2d.setAlignment(Qt.AlignCenter)
        coronal_slider_layout.addWidget(self.coronal_label_2d)
        sliders_layout.addLayout(coronal_slider_layout)
        
        left_layout.addLayout(sliders_layout)
        
        main_layout.addLayout(left_layout, stretch=3)
        
        # ===== 右側：コントロールパネル =====
        right_layout = QVBoxLayout()
        
        # ファイル開くボタン
        file_button = QPushButton("ファイルを開く")
        file_button.clicked.connect(self.open_dicom_folder)
        right_layout.addWidget(file_button)
        
        # ===== Window制御グループ =====
        window_group = QGroupBox("Window 制御")
        window_layout = QVBoxLayout()
        
        # Window Width
        window_layout.addWidget(QLabel("Window Width:"))
        ww_h_layout = QHBoxLayout()
        self.ww_slider = QSlider(Qt.Horizontal)
        self.ww_slider.setMinimum(10)
        self.ww_slider.setMaximum(4000)
        self.ww_slider.setValue(400)
        self.ww_slider.valueChanged.connect(self.on_window_width_changed)
        ww_h_layout.addWidget(self.ww_slider)
        self.ww_spinbox = QSpinBox()
        self.ww_spinbox.setMinimum(10)
        self.ww_spinbox.setMaximum(4000)
        self.ww_spinbox.setValue(400)
        self.ww_spinbox.valueChanged.connect(lambda v: self.ww_slider.setValue(v))
        ww_h_layout.addWidget(self.ww_spinbox)
        window_layout.addLayout(ww_h_layout)
        
        # Window Level
        window_layout.addWidget(QLabel("Window Level:"))
        wl_h_layout = QHBoxLayout()
        self.wl_slider = QSlider(Qt.Horizontal)
        self.wl_slider.setMinimum(-1000)
        self.wl_slider.setMaximum(1000)
        self.wl_slider.setValue(40)
        self.wl_slider.valueChanged.connect(self.on_window_level_changed)
        wl_h_layout.addWidget(self.wl_slider)
        self.wl_spinbox = QSpinBox()
        self.wl_spinbox.setMinimum(-1000)
        self.wl_spinbox.setMaximum(1000)
        self.wl_spinbox.setValue(40)
        self.wl_spinbox.valueChanged.connect(lambda v: self.wl_slider.setValue(v))
        wl_h_layout.addWidget(self.wl_spinbox)
        window_layout.addLayout(wl_h_layout)

        # Windowプリセット
        preset_layout = QHBoxLayout()
        preset_layout.addWidget(QLabel("プリセット:"))
        self.preset_combo = QComboBox()
        self.preset_combo.addItems(["カスタム", "腹部(Soft Tissue)", "骨(Bone)", "肺(Lung)", "脳(Brain)"])
        self.preset_combo.currentTextChanged.connect(self.on_preset_changed)
        preset_layout.addWidget(self.preset_combo)
        window_layout.addLayout(preset_layout)
        
        window_group.setLayout(window_layout)
        right_layout.addWidget(window_group)

        # ===== 表示モードグループ =====
        mode_group = QGroupBox("表示モード")
        mode_layout = QVBoxLayout()
        
        # 2D/3D切り替え（ラジオボタンを使用）
        display_mode_layout = QHBoxLayout()
        display_mode_layout.addWidget(QLabel("表示:"))
        
        # ボタングループで排他的な選択を管理
        self.display_mode_group = QButtonGroup()
        
        self.display_2d_radio = QRadioButton("2D表示")
        self.display_2d_radio.setChecked(True)
        self.display_mode_group.addButton(self.display_2d_radio, 0)
        display_mode_layout.addWidget(self.display_2d_radio)
        
        self.display_3d_radio = QRadioButton("3D表示")
        self.display_3d_radio.setChecked(False)
        self.display_mode_group.addButton(self.display_3d_radio, 1)
        display_mode_layout.addWidget(self.display_3d_radio)
        
        # 排他的な選択を監視
        self.display_mode_group.buttonClicked.connect(self.on_display_dimension_changed)
        mode_layout.addLayout(display_mode_layout)
        
        mode_group.setLayout(mode_layout)
        right_layout.addWidget(mode_group)
        
        # ===== DICOM情報グループ =====
        info_group = QGroupBox("DICOM情報")
        info_layout = QVBoxLayout()
        
        self.info_label = QLabel("DICOMファイルを読み込んでください")
        self.info_label.setWordWrap(True)
        font = QFont()
        font.setPointSize(9)
        self.info_label.setFont(font)
        info_layout.addWidget(self.info_label)
        
        info_group.setLayout(info_layout)
        right_layout.addWidget(info_group)
        
        # ===== その他 =====
        right_layout.addStretch()
        
        # テーマ選択
        theme_layout = QHBoxLayout()
        theme_layout.addWidget(QLabel("テーマ:"))
        self.theme_combo = QComboBox()
        self.theme_combo.addItems(["ライト", "ダーク"])
        self.theme_combo.currentTextChanged.connect(self.apply_theme)
        theme_layout.addWidget(self.theme_combo)
        right_layout.addLayout(theme_layout)
        
        # 保存ボタン
        save_button = QPushButton("設定を保存")
        save_button.clicked.connect(self.save_config)
        right_layout.addWidget(save_button)
        
        # 右側をメインレイアウトに追加
        main_layout.addLayout(right_layout, stretch=1)
        
        main_widget.setLayout(main_layout)
        
        # ステータスバー
        self.statusBar().showMessage("準備完了")
    
    def open_dicom_folder(self) -> None:
        """DICOMフォルダを開く"""
        folder = QFileDialog.getExistingDirectory(self, "DICOMフォルダを選択")
        if folder:
            dicom_files = list(Path(folder).glob("*.dcm"))
            if not dicom_files:
                QMessageBox.warning(self, "エラー", "DICOMファイルが見つかりません")
                return
            
            try:
                self.slice_processor = DICOMSliceProcessor([str(f) for f in dicom_files])
                self._update_sliders()
                self._update_window_controls()
                self._update_display()
                self._update_dicom_info()
                self.statusBar().showMessage(f"{len(dicom_files)}個のDICOMファイルを読み込みました")
            except Exception as e:
                QMessageBox.critical(self, "エラー", f"ファイル読み込みエラー: {e}")
    
    def _update_sliders(self) -> None:
        """スライダーの範囲を更新"""
        if self.slice_processor:
            # Axial
            self.axial_slider_2d.setMaximum(self.slice_processor.num_slices - 1)
            
            # Sagittal
            self.sagittal_slider_2d.setMaximum(self.slice_processor.image_width - 1)
            
            # Coronal
            self.coronal_slider_2d.setMaximum(self.slice_processor.image_height - 1)

    def _update_window_controls(self) -> None:
        """Window/Levelの初期値と範囲を更新"""
        if not self.slice_processor:
            return

        ww, wl = self.slice_processor.get_default_window()
        data_min = int(np.floor(self.slice_processor.data_min))
        data_max = int(np.ceil(self.slice_processor.data_max))
        data_range = max(data_max - data_min, 1)

        # Slider範囲をデータに合わせて調整
        self.ww_slider.setMinimum(1)
        self.ww_slider.setMaximum(data_range)
        self.wl_slider.setMinimum(data_min)
        self.wl_slider.setMaximum(data_max)

        # 既定値を反映
        self.window_width = int(ww)
        self.window_level = int(wl)
        self.ww_slider.setValue(self.window_width)
        self.wl_slider.setValue(self.window_level)
        self.ww_spinbox.setValue(self.window_width)
        self.wl_spinbox.setValue(self.window_level)
    
    def _update_display(self) -> None:
        """表示を更新（最適化版：incremental更新）"""
        if not self.slice_processor:
            return
        
        if self.display_mode_3d:
            # 3D表示モードに切り替え
            self.plotter_2d.interactor.hide()
            self.plotter_3d.interactor.show()
            self.plotter = self.plotter_3d
            self.plotter.clear()
            self._display_3d_volume()
        else:
            # 2D表示モードに切り替え
            self.plotter_3d.interactor.hide()
            self.plotter_2d.interactor.show()
            self.plotter = self.plotter_2d
            self._update_2d_display_incremental()
    
    def _update_2d_display_incremental(self) -> None:
        """2D表示の最適化された更新（最初の表示時は完全再描画、その後はincremental）"""
        # 最初の表示時のみ完全再描画
        if self._axial_dirty or self._sagittal_dirty or self._coronal_dirty:
            self.plotter.clear()
            self._display_2d_slices()
            self._axial_dirty = False
            self._sagittal_dirty = False
            self._coronal_dirty = False
        
        # テキスト表示（毎回更新）
        self.plotter.subplot(0, 0)
        text = f"Axial: {self.current_axial_slice}/{self.slice_processor.num_slices - 1}\n"
        text += f"Sagittal: {self.current_sagittal_pos}/{self.slice_processor.image_width - 1}\n"
        text += f"Coronal: {self.current_coronal_pos}/{self.slice_processor.image_height - 1}"
        # テキスト色をテーマに応じて動的に設定
        text_color = "black" if self.theme_combo.currentText() == "ライト" else "white"
        self.plotter.add_text(text, position="upper_left", font_size=10, color=text_color)
    
    def _display_2d_slices(self) -> None:
        """2Dスライス表示（左上:Axial、右上:Sagittal、左下:Coronal）"""
        # 2Dでは操作を無効化（ズーム/回転禁止）
        self.plotter.disable()

        col_spacing, row_spacing, slice_spacing = self.slice_processor.spacing
        width = self.slice_processor.image_width
        height = self.slice_processor.image_height
        num_slices = self.slice_processor.num_slices

        # Axial（左上）
        self.plotter.subplot(0, 0)
        axial_slice = self.slice_processor.get_axial_slice(
            self.current_axial_slice, self.window_width, self.window_level
        )
        grid_axial = pv.ImageData()
        grid_axial.dimensions = (axial_slice.shape[1], axial_slice.shape[0], 1)
        grid_axial.spacing = (col_spacing, row_spacing, 1)
        grid_axial.origin = (0, 0, 0)
        grid_axial.point_data["intensity"] = axial_slice.T.flatten(order='F')
        self.plotter.add_mesh(
            grid_axial,
            scalars="intensity",
            cmap="gray",
            show_scalar_bar=False,
            opacity=1.0,
            name="axial"
        )
        # Sagittal位置の縦線（赤）
        x_sag = self.current_sagittal_pos * col_spacing
        line_sagittal = pv.Line((x_sag, 0, 0), (x_sag, height * row_spacing, 0))
        self.plotter.add_mesh(line_sagittal, color="red", line_width=2, name="line_sagittal")
        # Coronal位置の横線（青）
        y_cor = self.current_coronal_pos * row_spacing
        line_coronal = pv.Line((0, y_cor, 0), (width * col_spacing, y_cor, 0))
        self.plotter.add_mesh(line_coronal, color="blue", line_width=2, name="line_coronal")

        # Sagittal（右上）
        self.plotter.subplot(0, 1)
        sagittal_slice = self.slice_processor.get_sagittal_slice(
            self.current_sagittal_pos, self.window_width, self.window_level
        )
        sagittal_slice = np.flipud(sagittal_slice)  # 上下反転
        grid_sagittal = pv.ImageData()
        grid_sagittal.dimensions = (sagittal_slice.shape[1], sagittal_slice.shape[0], 1)
        grid_sagittal.spacing = (row_spacing, slice_spacing, 1)
        grid_sagittal.origin = (0, 0, 0)
        grid_sagittal.point_data["intensity"] = sagittal_slice.T.flatten(order='F')
        self.plotter.add_mesh(
            grid_sagittal,
            scalars="intensity",
            cmap="gray",
            show_scalar_bar=False,
            opacity=1.0,
            name="sagittal"
        )
        # Axial位置の横線（緑）: coronal断面と同じY座標計算
        y_ax = (sagittal_slice.shape[0] - 1 - self.current_axial_slice) * slice_spacing
        line_axial_sag = pv.Line((0, y_ax, 0), (sagittal_slice.shape[1] * row_spacing, y_ax, 0))
        self.plotter.add_mesh(line_axial_sag, color="lime", line_width=2, name="line_axial_sag")
        # Coronal位置の縦線（青）
        x_cor = self.current_coronal_pos * row_spacing
        line_coronal_sag = pv.Line((x_cor, 0, 0), (x_cor, sagittal_slice.shape[0] * slice_spacing, 0))
        self.plotter.add_mesh(line_coronal_sag, color="blue", line_width=2, name="line_coronal_sag")

        # Coronal（左下）
        self.plotter.subplot(1, 0)
        coronal_slice = self.slice_processor.get_coronal_slice(
            self.current_coronal_pos, self.window_width, self.window_level
        )
        coronal_slice = np.flipud(coronal_slice)  # 上下反転
        grid_coronal = pv.ImageData()
        grid_coronal.dimensions = (coronal_slice.shape[1], coronal_slice.shape[0], 1)
        grid_coronal.spacing = (col_spacing, slice_spacing, 1)
        grid_coronal.origin = (0, 0, 0)
        grid_coronal.point_data["intensity"] = coronal_slice.T.flatten(order='F')
        self.plotter.add_mesh(
            grid_coronal,
            scalars="intensity",
            cmap="gray",
            show_scalar_bar=False,
            opacity=1.0,
            name="coronal"
        )
        # Axial位置の横線（緑）
        y_ax_cor = (coronal_slice.shape[0] - 1 - self.current_axial_slice) * slice_spacing
        line_axial_cor = pv.Line((0, y_ax_cor, 0), (coronal_slice.shape[1] * col_spacing, y_ax_cor, 0))
        self.plotter.add_mesh(line_axial_cor, color="lime", line_width=2, name="line_axial_cor")
        # Sagittal位置の縦線（赤）
        x_sag_cor = self.current_sagittal_pos * col_spacing
        line_sagittal_cor = pv.Line((x_sag_cor, 0, 0), (x_sag_cor, coronal_slice.shape[0] * slice_spacing, 0))
        self.plotter.add_mesh(line_sagittal_cor, color="red", line_width=2, name="line_sagittal_cor")

        # 右下は空にする
        self.plotter.subplot(1, 1)
        self.plotter.add_text("", position="upper_left", font_size=1)

        # カメラ固定（毎回調整して画像を最大限に拡大）
        for row, col in [(0, 0), (0, 1), (1, 0)]:
            self.plotter.subplot(row, col)
            self.plotter.camera_position = 'xy'
            self.plotter.camera.parallel_projection = True
            self.plotter.reset_camera()
            self.plotter.camera.zoom(1.4)
    
    def _display_3d_volume(self) -> None:
        """3Dボリュームレンダリング（全面に表示）"""
        # 3D用プロッター（shape=(1,1)）で全画面表示
        
        # 3D volume データの構築（Windowing適用）
        volume_data = []
        for i in range(self.slice_processor.num_slices):
            slice_data = self.slice_processor.get_axial_slice(i, self.window_width, self.window_level)
            volume_data.append(slice_data)
        volume_data = np.array(volume_data, dtype=np.uint8)
        
        # PyVista ImageData へ変換（test_main.pyを参考）
        # Reorder to (x, y, z) for VTK/PyVista
        vol_for_vtk = np.transpose(volume_data, (2, 1, 0))
        
        grid = pv.ImageData()
        grid.dimensions = vol_for_vtk.shape
        
        # スペーシングを設定（test_main.py準拠）
        grid.spacing = self.slice_processor.spacing
        grid.origin = (0.0, 0.0, 0.0)
        
        # データをポイントデータとして設定（Fortran順）
        grid.point_data["values"] = vol_for_vtk.flatten(order='F')
        
        # インタラクションを有効化（3D表示では回転などの操作が必要）
        self.plotter.enable()
        
        # ボリュームレンダリング（sigmoid opacity）
        self.plotter.add_volume(
            grid,
            scalars="values",
            opacity="sigmoid",
            cmap="gray",
            clim=(0, 255),
            show_scalar_bar=True,
            scalar_bar_args={'title': 'Intensity', 'vertical': True}
        )
        
        # 座標軸の表示
        self.plotter.add_axes(
            xlabel='X (Width)',
            ylabel='Y (Height)', 
            zlabel='Z (Slices)',
            color='white',
            line_width=2
        )
        
        # カメラをリセット
        self.plotter.reset_camera()
        # オブジェクトを適切に表示（全面に拡大）
        self.plotter.camera.zoom(1.2)
        self.plotter.camera.azimuth = 45
        self.plotter.camera.elevation = 30
    
    def _add_slice_planes_3d(self) -> None:
        """スライス面を表示（3Dモード）"""
        width = self.slice_processor.image_width
        height = self.slice_processor.image_height
        num_slices = self.slice_processor.num_slices
        slice_thickness = self.slice_processor.slice_thickness if self.slice_processor.slice_thickness > 0 else 1.0
        
        # Axialスライス面（XY平面）
        z_pos = self.current_axial_slice * slice_thickness
        axial_slice = self.slice_processor.get_axial_slice(
            self.current_axial_slice, self.window_width, self.window_level
        )
        
        grid_axial = pv.ImageData()
        grid_axial.dimensions = (axial_slice.shape[1], axial_slice.shape[0], 1)
        grid_axial.spacing = (1, 1, 1)
        grid_axial.origin = (0, 0, z_pos)
        grid_axial.point_data["intensity"] = axial_slice.flatten(order='F')
        
        self.plotter.add_mesh(
            grid_axial,
            scalars="intensity",
            cmap="gray",
            opacity=0.8,
            name="axial_plane"
        )
        
        # Sagittalスライス面（YZ平面）
        if self.show_sagittal:
            sagittal_slice = self.slice_processor.get_sagittal_slice(
                self.current_sagittal_pos, self.window_width, self.window_level
            )
            
            grid_sagittal = pv.ImageData()
            grid_sagittal.dimensions = (1, height, num_slices)
            grid_sagittal.spacing = (1, 1, slice_thickness)
            grid_sagittal.origin = (self.current_sagittal_pos, 0, 0)
            grid_sagittal.point_data["intensity"] = sagittal_slice.flatten(order='F')
            
            self.plotter.add_mesh(
                grid_sagittal,
                scalars="intensity",
                cmap="gray",
                opacity=0.7,
                name="sagittal_plane"
            )
        
        # Coronalスライス面（XZ平面）
        if self.show_coronal:
            coronal_slice = self.slice_processor.get_coronal_slice(
                self.current_coronal_pos, self.window_width, self.window_level
            )
            
            grid_coronal = pv.ImageData()
            grid_coronal.dimensions = (width, 1, num_slices)
            grid_coronal.spacing = (1, 1, slice_thickness)
            grid_coronal.origin = (0, self.current_coronal_pos, 0)
            grid_coronal.point_data["intensity"] = coronal_slice.flatten(order='F')
            
            self.plotter.add_mesh(
                grid_coronal,
                scalars="intensity",
                cmap="gray",
                opacity=0.7,
                name="coronal_plane"
            )
    
    def _update_dicom_info(self) -> None:
        """DICOM情報を更新"""
        if self.slice_processor:
            info = self.slice_processor.get_dicom_info()
            info_text = (
                f"患者名: {info['patient_name']}\n"
                f"患者ID: {info['patient_id']}\n"
                f"モダリティ: {info['modality']}\n"
                f"検査日: {info['study_date']}\n"
                f"画像幅: {info['image_width']} x 高さ: {info['image_height']}\n"
                f"スライス数: {info['num_slices']}\n"
                f"スライス厚: {info['slice_thickness']} mm"
            )
            self.info_label.setText(info_text)
    
    def on_axial_slice_changed(self, value: int) -> None:
        """Axialスライス変更時"""
        self.current_axial_slice = value
        self.axial_label_2d.setText(f"{value}/{self.slice_processor.num_slices - 1}" if self.slice_processor else "0/0")
        # Axial変更時はすべてのビューを再描画
        self._axial_dirty = True
        self._sagittal_dirty = True
        self._coronal_dirty = True
        self._update_display()
    
    def on_sagittal_pos_changed(self, value: int) -> None:
        """Sagittal位置変更時"""
        self.current_sagittal_pos = value
        self.sagittal_label_2d.setText(f"{value}/{self.slice_processor.image_width - 1}" if self.slice_processor else "0/0")
        # 遅延描画により点滅を軽減（最適化：50msに短縮）
        self._axial_dirty = True
        self._sagittal_dirty = True
        self._coronal_dirty = True
        if self._update_timer is None:
            self._update_timer = QTimer()
            self._update_timer.setSingleShot(True)
            self._update_timer.timeout.connect(self._update_display)
        self._update_timer.start(30)  # 30msに短縮
    
    def on_coronal_pos_changed(self, value: int) -> None:
        """Coronal位置変更時"""
        self.current_coronal_pos = value
        self.coronal_label_2d.setText(f"{value}/{self.slice_processor.image_height - 1}" if self.slice_processor else "0/0")
        # 遅延描画により点滅を軽減（最適化：50msに短縮）
        self._axial_dirty = True
        self._sagittal_dirty = True
        self._coronal_dirty = True
        if self._update_timer is None:
            self._update_timer = QTimer()
            self._update_timer.setSingleShot(True)
            self._update_timer.timeout.connect(self._update_display)
        self._update_timer.start(30)  # 30msに短縮
    
    def on_window_width_changed(self, value: int) -> None:
        """Window Width変更時"""
        self.window_width = value
        self.ww_spinbox.blockSignals(True)
        self.ww_spinbox.setValue(value)
        self.ww_spinbox.blockSignals(False)
        # Window変更時は全ビュー再描画が必要
        self._axial_dirty = True
        self._sagittal_dirty = True
        self._coronal_dirty = True
        self._schedule_update()
    
    def on_window_level_changed(self, value: int) -> None:
        """Window Level変更時"""
        self.window_level = value
        self.wl_spinbox.blockSignals(True)
        self.wl_spinbox.setValue(value)
        self.wl_spinbox.blockSignals(False)
        # Window変更時は全ビュー再描画が必要
        self._axial_dirty = True
        self._sagittal_dirty = True
        self._coronal_dirty = True
        self._schedule_update()
    
    def _schedule_update(self) -> None:
        """更新をスケジュール（遅延実行）"""
        if self._update_timer is None:
            self._update_timer = QTimer()
            self._update_timer.setSingleShot(True)
            self._update_timer.timeout.connect(self._update_display)
        # 既にタイマーが動作中なら何もしない（多重更新防止）
        if not self._update_timer.isActive():
            self._update_timer.start(30)

    def on_preset_changed(self, text: str) -> None:
        """Windowプリセット変更時"""
        presets = {
            "腹部(Soft Tissue)": (350, 40),
            "骨(Bone)": (2000, 500),
            "肺(Lung)": (1500, -500),
            "脳(Brain)": (80, 40),
        }
        if text not in presets:
            return
        ww, wl = presets[text]
        self.window_width = ww
        self.window_level = wl
        self.ww_slider.setValue(ww)
        self.wl_slider.setValue(wl)
        self.ww_spinbox.setValue(ww)
        self.wl_spinbox.setValue(wl)
        # プリセット適用時もフラグをセット
        self._axial_dirty = True
        self._sagittal_dirty = True
        self._coronal_dirty = True
        self._schedule_update()

    def on_display_dimension_changed(self) -> None:
        """2D/3D表示切り替え時"""
        # ラジオボタングループから選択を取得
        self.display_mode_3d = self.display_mode_group.checkedId() == 1
        # 表示切り替え時はすべてのビューを再描画
        self._axial_dirty = True
        self._sagittal_dirty = True
        self._coronal_dirty = True
        self._update_display()

    def apply_theme(self, theme: str) -> None:
        """ライト/ダークテーマを適用"""
        if theme == "ダーク":
            self.setStyleSheet(
                "QWidget { background-color: #1e1e1e; color: #e0e0e0; }"
                "QGroupBox { border: 1px solid #3c3c3c; margin-top: 6px; }"
                "QGroupBox::title { subcontrol-origin: margin; left: 6px; }"
                "QPushButton { background-color: #2d2d2d; border: 1px solid #3c3c3c; padding: 6px; }"
                "QComboBox, QSpinBox { background-color: #2d2d2d; border: 1px solid #3c3c3c; }"
                "QSlider::groove:horizontal { height: 6px; background: #3c3c3c; }"
                "QSlider::handle:horizontal { width: 12px; background: #007acc; margin: -4px 0; }"
            )
        else:
            self.setStyleSheet("")

        # PyVistaビューの背景色も更新
        if hasattr(self, 'plotter'):
            bg = "#1e1e1e" if theme == "ダーク" else "white"
            self.plotter.set_background(bg)
    
    def load_config(self) -> None:
        """config.jsonから設定を読み込む"""
        if self.config_path.exists():
            try:
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                
                # GUI設定の復元
                if 'window_width' in config:
                    self.window_width = config['window_width']
                    self.ww_slider.setValue(self.window_width)
                
                if 'window_level' in config:
                    self.window_level = config['window_level']
                    self.wl_slider.setValue(self.window_level)
                
                if 'display_mode_3d' in config:
                    self.display_mode_3d = config['display_mode_3d']
                    if self.display_mode_3d:
                        self.display_3d_radio.setChecked(True)
                    else:
                        self.display_2d_radio.setChecked(True)

                if 'theme' in config:
                    idx = self.theme_combo.findText(config['theme'])
                    if idx >= 0:
                        self.theme_combo.setCurrentIndex(idx)
                        self.apply_theme(config['theme'])
            except Exception as e:
                print(f"設定ファイル読み込みエラー: {e}")
    
    def save_config(self) -> None:
        """設定をconfig.jsonに保存"""
        config = {
            'window_width': self.window_width,
            'window_level': self.window_level,
            'display_mode_3d': self.display_mode_3d,
            'theme': self.theme_combo.currentText()
        }
        
        try:
            with open(self.config_path, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=2, ensure_ascii=False)
            QMessageBox.information(self, "成功", "設定を保存しました")
            self.statusBar().showMessage("設定を保存しました")
        except Exception as e:
            QMessageBox.critical(self, "エラー", f"保存エラー: {e}")


def main():
    """メイン関数"""
    app = QApplication(sys.argv)
    viewer = DICOMViewerApp()
    viewer.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
