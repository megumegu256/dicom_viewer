"""
DICOM Viewer テストコード
テストコードは以下の機能をテストします:
- DICOMファイル読み込み
- スライス処理
- Window Width/Level調整
- DICOM情報抽出
"""

import unittest
import tempfile
import numpy as np
from pathlib import Path
import sys

# テスト用に簡易的なDICOMファイルを生成
try:
    import pydicom
    from pydicom.dataset import Dataset, FileMetaDataset
    from pydicom.uid import ExplicitVRLittleEndian
    HAS_PYDICOM = True
except ImportError:
    HAS_PYDICOM = False

# main.pyから必要なクラスをインポート
import os
sys.path.insert(0, str(Path(__file__).parent.parent))
from main import DICOMSliceProcessor


class TestDICOMSliceProcessor(unittest.TestCase):
    """DICOMSliceProcessor のテストクラス"""
    
    @classmethod
    def setUpClass(cls):
        """テストの前準備"""
        if not HAS_PYDICOM:
            print("Warning: pydicom not available, skipping tests")
            return
        
        # テンポラリディレクトリにダミーDICOMファイルを作成
        cls.temp_dir = tempfile.mkdtemp()
        cls.dicom_files = []
        
        for i in range(3):
            ds = FileMetaDataset()
            ds.MediaStorageSOPClassUID = '1.2.840.10008.5.1.4.1.1.2'
            ds.MediaStorageSOPInstanceUID = f'1.2.3.4.5.6.7.8.9.{i}'
            ds.ImplementationClassUID = '1.2.3.4'
            ds.file_meta = ds
            
            ds = Dataset()
            ds.file_meta = FileMetaDataset()
            ds.file_meta.MediaStorageSOPClassUID = '1.2.840.10008.5.1.4.1.1.2'
            ds.file_meta.MediaStorageSOPInstanceUID = f'1.2.3.4.5.6.7.8.9.{i}'
            ds.file_meta.ImplementationClassUID = '1.2.3.4'
            ds.file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
            
            ds.PatientName = f'Test^Patient{i}'
            ds.PatientID = f'00{i}'
            ds.Modality = 'CT'
            ds.StudyDate = '20240101'
            ds.SeriesNumber = 1
            ds.InstanceNumber = i + 1
            ds.Rows = 64
            ds.Columns = 64
            ds.SliceThickness = 2.5
            ds.SamplesPerPixel = 1
            ds.PhotometricInterpretation = 'MONOCHROME2'
            ds.BitsAllocated = 16
            ds.BitsStored = 16
            ds.HighBit = 15
            ds.PixelRepresentation = 1
            
            # ダミーピクセルデータを作成（256以下の値）
            pixel_data = np.random.randint(0, 256, (64, 64), dtype=np.uint16)
            ds.PixelData = pixel_data.tobytes()
            
            file_path = Path(cls.temp_dir) / f'test_{i:03d}.dcm'
            ds.save_as(str(file_path))
            cls.dicom_files.append(str(file_path))
    
    @classmethod
    def tearDownClass(cls):
        """テスト後処理"""
        if HAS_PYDICOM and cls.dicom_files:
            import shutil
            shutil.rmtree(cls.temp_dir)
    
    @unittest.skipIf(not HAS_PYDICOM, "pydicom not available")
    def test_dicom_loading(self):
        """DICOMファイル読み込みテスト"""
        processor = DICOMSliceProcessor(self.dicom_files)
        
        self.assertEqual(processor.num_slices, 3)
        self.assertEqual(processor.image_height, 64)
        self.assertEqual(processor.image_width, 64)
        self.assertEqual(processor.slice_thickness, 2.5)
    
    @unittest.skipIf(not HAS_PYDICOM, "pydicom not available")
    def test_axial_slice_retrieval(self):
        """Axialスライス取得テスト"""
        processor = DICOMSliceProcessor(self.dicom_files)
        
        slice_data = processor.get_axial_slice(0, ww=400, wl=40)
        
        self.assertEqual(slice_data.shape, (64, 64))
        self.assertEqual(slice_data.dtype, np.uint8)
        self.assertTrue(np.min(slice_data) >= 0)
        self.assertTrue(np.max(slice_data) <= 255)
    
    @unittest.skipIf(not HAS_PYDICOM, "pydicom not available")
    def test_window_width_level(self):
        """Window Width/Level調整テスト"""
        processor = DICOMSliceProcessor(self.dicom_files)
        
        slice1 = processor.get_axial_slice(0, ww=100, wl=20)
        slice2 = processor.get_axial_slice(0, ww=400, wl=40)
        
        # 異なるWindow設定で異なる結果が得られることを確認
        self.assertFalse(np.array_equal(slice1, slice2))
    
    @unittest.skipIf(not HAS_PYDICOM, "pydicom not available")
    def test_dicom_info_extraction(self):
        """DICOM情報抽出テスト"""
        processor = DICOMSliceProcessor(self.dicom_files)
        
        info = processor.get_dicom_info()
        
        self.assertIn('patient_name', info)
        self.assertIn('patient_id', info)
        self.assertIn('modality', info)
        self.assertIn('image_width', info)
        self.assertIn('image_height', info)
        self.assertIn('num_slices', info)
        self.assertIn('slice_thickness', info)
        
        self.assertEqual(info['image_width'], 64)
        self.assertEqual(info['image_height'], 64)
        self.assertEqual(info['num_slices'], 3)


class TestDICOMSliceProcessorEdgeCases(unittest.TestCase):
    """エッジケースのテスト"""
    
    def test_empty_file_list(self):
        """空のファイルリストでの処理"""
        processor = DICOMSliceProcessor([])
        
        self.assertEqual(processor.num_slices, 0)
        self.assertEqual(processor.image_height, 0)
        self.assertEqual(processor.image_width, 0)
    
    @unittest.skipIf(not HAS_PYDICOM, "pydicom not available")
    def test_sagittal_slice_shape(self):
        """Sagittalスライスの形状確認"""
        # テンポラリディレクトリにダミーDICOMを作成
        temp_dir = tempfile.mkdtemp()
        dicom_file = Path(temp_dir) / 'test.dcm'
        
        ds = Dataset()
        ds.file_meta = FileMetaDataset()
        ds.file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
        ds.PatientName = 'Test'
        ds.Rows = 32
        ds.Columns = 48
        ds.SliceThickness = 1.0
        ds.SamplesPerPixel = 1
        ds.PhotometricInterpretation = 'MONOCHROME2'
        ds.BitsAllocated = 16
        ds.BitsStored = 16
        ds.HighBit = 15
        ds.PixelRepresentation = 1
        
        pixel_data = np.zeros((32, 48), dtype=np.uint16)
        ds.PixelData = pixel_data.tobytes()
        
        ds.save_as(str(dicom_file))
        
        processor = DICOMSliceProcessor([str(dicom_file)])
        
        # Sagittalスライスのテスト
        if processor.image_width > 0:
            sagittal = processor.get_sagittal_slice(0)
            self.assertEqual(sagittal.shape[1], processor.image_height)
        
        # クリーンアップ
        import shutil
        shutil.rmtree(temp_dir)


if __name__ == '__main__':
    unittest.main()
