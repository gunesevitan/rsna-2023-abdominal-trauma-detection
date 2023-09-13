import sys
import os
from tqdm import tqdm
import numpy as np
import pandas as pd
from scipy.ndimage import zoom
import cv2

sys.path.append('..')
import settings


if __name__ == '__main__':

    image_dataset_directory = settings.DATA / 'datasets' / '2d_windowed_400w_50c_1mm_isotropic' / 'images'
    patient_ids = sorted(os.listdir(image_dataset_directory), key=lambda filename: int(filename))

    output_directory = settings.DATA / 'datasets' / '3d_windowed_400w_50c_96x256x256' / 'volumes'
    output_directory.mkdir(parents=True, exist_ok=True)

    df_dicom_tags = pd.read_parquet(settings.DATA / 'rsna-2023-abdominal-trauma-detection' / 'train_dicom_tags.parquet')
    settings.logger.info(f'DICOM Tags Dataset Shape: {df_dicom_tags.shape} - Memory Usage: {df_dicom_tags.memory_usage().sum() / 1024 ** 2:.2f} MB')

    df_dicom_tags['scan_id'] = df_dicom_tags['path'].apply(lambda x: str(x).split('/')[2]).astype(np.uint64)
    df_dicom_tags['patient_id'] = df_dicom_tags['path'].apply(lambda x: str(x).split('/')[1]).astype(np.uint64)
    df_dicom_tags['slice_id'] = df_dicom_tags['path'].apply(lambda x: str(x).split('/')[3].split('.')[0]).astype(np.uint64)
    df_dicom_tags['z_position'] = df_dicom_tags['ImagePositionPatient'].apply(lambda x: eval(x)[-1]).astype(np.float32)
    df_dicom_tags.sort_values(by=['patient_id', 'scan_id', 'z_position'], ascending=[True, True, False], inplace=True)
    df_dicom_tags.reset_index(drop=True, inplace=True)
    df_dicom_tags['z_position_diff'] = df_dicom_tags.groupby(['patient_id', 'scan_id'])['z_position'].diff().round(2)
    df_dicom_tags['slice_id_diff'] = df_dicom_tags.groupby(['patient_id', 'scan_id'])[['slice_id']].diff()
    df_dicom_tags['image_path'] = df_dicom_tags['path'].apply(lambda x: str(image_dataset_directory) + x.split('images')[-1].split('.')[0] + '.png')

    for patient_id in tqdm(patient_ids):

        patient_directory = image_dataset_directory / str(patient_id)
        patient_scans = sorted(os.listdir(patient_directory), key=lambda filename: int(filename))

        for scan_id in tqdm(patient_scans):

            df_scan = df_dicom_tags.loc[df_dicom_tags['scan_id'] == int(scan_id)].reset_index(drop=True)
            scan_directory = patient_directory / str(scan_id)
            file_names = df_scan['slice_id'].values
            scan = np.array([cv2.imread(str(scan_directory / f'{file_name}.png'), -1) for file_name in file_names])

            try:
                patient_position = df_scan.loc[0, 'PatientPosition']
            except AttributeError:
                patient_position = None

            if patient_position is not None:
                if patient_position == 'HFS':
                    # Flip x-axis if patient position is head first
                    scan = np.flip(scan, axis=2)

            # Find partial slices by calculating sum of all zero vertical lines
            scan_all_zero_vertical_line_transitions = np.diff(np.all(scan == 0, axis=1).sum(axis=1))
            # Heuristically mask very high and low transitions on z-axis and drop them
            slices_with_all_zero_vertical_lines = (scan_all_zero_vertical_line_transitions > 5) | (scan_all_zero_vertical_line_transitions < -5)
            slices_with_all_zero_vertical_lines = np.append(slices_with_all_zero_vertical_lines, slices_with_all_zero_vertical_lines[-1])
            scan = scan[~slices_with_all_zero_vertical_lines]

            # Crop non-zero slices along xz, yz and xy planes
            mmin = np.array((scan > 0).nonzero()).min(axis=1)
            mmax = np.array((scan > 0).nonzero()).max(axis=1)
            scan = scan[
                mmin[0]:mmax[0] + 1,
                mmin[1]:mmax[1] + 1,
                mmin[2]:mmax[2] + 1,
            ]

            zoom_factor = np.array([96, 256, 256]) / np.array(scan.shape)
            scan = zoom(scan, zoom=zoom_factor, mode='nearest')

            np.save(output_directory / f'{patient_id}_{scan_id}.npy', scan)
