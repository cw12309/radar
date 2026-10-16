from pathlib import Path

import numpy as np
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import CTImageStorage, ExplicitVRLittleEndian, generate_uid

ROOT = Path(__file__).resolve().parent
input_path = ROOT / 'data' / 'test_ct_standard.npy'
output_path = ROOT / 'data' / 'test_ct_single_slice.dcm'

volume = np.load(input_path)
slice_data = np.asarray(volume[volume.shape[0] // 2], dtype=np.int16)

study_uid = generate_uid()
series_uid = generate_uid()
sop_uid = generate_uid()

file_meta = FileMetaDataset()
file_meta.MediaStorageSOPClassUID = CTImageStorage
file_meta.MediaStorageSOPInstanceUID = sop_uid
file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
file_meta.ImplementationClassUID = generate_uid()

ds = FileDataset(str(output_path), {}, file_meta=file_meta, preamble=b'\x00' * 128)
ds.is_little_endian = True
ds.is_implicit_VR = False
ds.SOPClassUID = CTImageStorage
ds.SOPInstanceUID = sop_uid
ds.StudyInstanceUID = study_uid
ds.SeriesInstanceUID = series_uid
ds.Modality = 'CT'
ds.PatientName = 'RADAR^SyntheticDemo'
ds.PatientID = 'RADAR-SYN-001'
ds.StudyDescription = 'Synthetic Abdomen CT Demonstration'
ds.SeriesDescription = 'Synthetic Abdomen CT - Single Axial Slice'
ds.ImageComments = 'Synthetic data: simulated high-density lesion in right upper abdomen'
ds.StudyDate = '20260101'
ds.StudyTime = '120000'
ds.Manufacturer = 'RADAR Demo Generator'
ds.InstanceNumber = 1
ds.AcquisitionNumber = 1
ds.Rows, ds.Columns = slice_data.shape
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
ds.SliceThickness = 5.0
ds.ImagePositionPatient = [-128.0, -128.0, 0.0]
ds.ImageOrientationPatient = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
ds.PixelData = slice_data.tobytes()
ds.save_as(str(output_path), write_like_original=False)

print(f'created: {output_path}')
print(f'shape: {slice_data.shape}')
print(f'hu_range: {int(slice_data.min())}..{int(slice_data.max())}')
print(f'study_uid: {study_uid}')
print(f'series_uid: {series_uid}')
print('label: simulated high-density lesion in right upper abdomen')
