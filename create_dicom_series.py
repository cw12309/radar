from pathlib import Path

import numpy as np
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import CTImageStorage, ExplicitVRLittleEndian, generate_uid

ROOT = Path(__file__).resolve().parent
input_path = ROOT / 'data' / 'test_ct_standard.npy'
output_dir = ROOT / 'data' / 'test_ct_dicom_series'

volume = np.load(input_path)
study_uid = generate_uid()
series_uid = generate_uid()
output_dir.mkdir(parents=True, exist_ok=True)

for index, source_slice in enumerate(volume):
    pixels = np.asarray(source_slice, dtype=np.int16)
    sop_uid = generate_uid()

    file_meta = FileMetaDataset()
    file_meta.MediaStorageSOPClassUID = CTImageStorage
    file_meta.MediaStorageSOPInstanceUID = sop_uid
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    file_meta.ImplementationClassUID = generate_uid()

    path = output_dir / f'IM_{index + 1:04d}.dcm'
    ds = FileDataset(str(path), {}, file_meta=file_meta, preamble=b'\x00' * 128)
    ds.is_little_endian = True
    ds.is_implicit_VR = False
    ds.SOPClassUID = CTImageStorage
    ds.SOPInstanceUID = sop_uid
    ds.StudyInstanceUID = study_uid
    ds.SeriesInstanceUID = series_uid
    ds.Modality = 'CT'
    ds.PatientName = 'RADAR^SyntheticDemo'
    ds.PatientID = 'RADAR-SYN-MULTI-001'
    ds.StudyDescription = 'Synthetic Abdomen CT Demonstration'
    ds.SeriesDescription = 'Synthetic Abdomen CT - 128 Slice Series'
    ds.ImageComments = 'Synthetic data for viewer and model demonstration only'
    ds.StudyDate = '20260101'
    ds.StudyTime = '120000'
    ds.Manufacturer = 'RADAR Demo Generator'
    ds.InstanceNumber = index + 1
    ds.AcquisitionNumber = 1
    ds.Rows, ds.Columns = pixels.shape
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = 'MONOCHROME2'
    ds.BitsAllocated = 16
    ds.BitsStored = 16
    ds.HighBit = 15
    ds.PixelRepresentation = 1
    ds.RescaleIntercept = 0
    ds.RescaleSlope = 1
    ds.WindowCenter = 40
    ds.WindowWidth = 400
    ds.PixelSpacing = [1.0, 1.0]
    ds.SliceThickness = 3.0
    ds.SpacingBetweenSlices = 3.0
    ds.ImagePositionPatient = [-128.0, -128.0, float(index * 3.0)]
    ds.ImageOrientationPatient = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
    ds.PixelData = pixels.tobytes()
    ds.save_as(str(path), write_like_original=False)

(output_dir / 'series_info.txt').write_text(
    f'StudyInstanceUID={study_uid}\n'
    f'SeriesInstanceUID={series_uid}\n'
    f'SliceCount={len(volume)}\n'
    f'Rows={volume.shape[1]}\n'
    f'Columns={volume.shape[2]}\n',
    encoding='ascii'
)

print(f'created: {output_dir}')
print(f'slices: {len(volume)}')
print(f'shape: {volume.shape[1]}x{volume.shape[2]}')
print(f'study_uid: {study_uid}')
print(f'series_uid: {series_uid}')
