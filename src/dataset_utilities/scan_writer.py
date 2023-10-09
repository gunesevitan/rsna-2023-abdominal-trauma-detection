import sys
import os
from tqdm import tqdm
import numpy as np
import pandas as pd
import cv2
import pydicom
import torch
from monai.transforms import Resize

sys.path.append('..')
import settings
import dicom_utilities


if __name__ == '__main__':

    dicom_dataset_directory = settings.DATA / 'rsna-2023-abdominal-trauma-detection' / 'train_images'
    image_dataset_directory = settings.DATA / 'datasets' / '2d_1w_raw_size' / 'images'
    patient_ids = sorted(os.listdir(image_dataset_directory), key=lambda filename: int(filename))

    output_directory = settings.DATA / 'datasets' / '3d_1w_contour_cropped_96x256x256' / 'volumes'
    output_directory.mkdir(parents=True, exist_ok=True)

    resize = Resize(spatial_size=(96, 256, 256))

    for patient_id in tqdm(patient_ids):

        patient_directory = image_dataset_directory / str(patient_id)
        patient_scans = sorted(os.listdir(patient_directory), key=lambda filename: int(filename))

        for scan_id in patient_scans:

            scan_directory = patient_directory / str(scan_id)
            file_names = sorted(os.listdir(scan_directory), key=lambda x: int(str(x).split('.')[0]))

            z_positions = []
            patient_positions = []
            scan = []

            for file_idx, file_name in enumerate(file_names, start=1):

                dicom = pydicom.dcmread(str(dicom_dataset_directory / patient_id / scan_id / str(file_name.split('.')[0] + '.dcm')))
                image = cv2.imread(str(scan_directory / file_name), -1)

                scan.append(image)

                try:
                    patient_position = dicom.PatientPosition
                except AttributeError:
                    patient_position = 'FFS'

                patient_positions.append(patient_position)

                try:
                    z_position = float(dicom.ImagePositionPatient[-1])
                except AttributeError:
                    z_position = file_idx * -1

                z_positions.append(z_position)

            scan = np.array(scan)

            # Sort CT scan slices by head to feet
            sorting_idx_z = np.argsort(z_positions)[::-1]
            scan = scan[sorting_idx_z]

            patient_position = pd.Series(patient_positions).value_counts().index[0]
            if patient_position is not None:
                if patient_position == 'HFS':
                    # Flip x-axis if patient position is head first
                    scan = np.flip(scan, axis=2)

            # Find partial slices by calculating sum of all zero vertical lines
            if scan.shape[0] != 1:
                scan_all_zero_vertical_line_transitions = np.diff(np.all(scan == 0, axis=1).sum(axis=1))
                # Heuristically select high and low transitions on z-axis and drop them
                slices_with_all_zero_vertical_lines = (scan_all_zero_vertical_line_transitions > 5) | (scan_all_zero_vertical_line_transitions < -5)
                slices_with_all_zero_vertical_lines = np.append(slices_with_all_zero_vertical_lines, slices_with_all_zero_vertical_lines[-1])
                scan = scan[~slices_with_all_zero_vertical_lines]
                del scan_all_zero_vertical_line_transitions, slices_with_all_zero_vertical_lines

            # Crop the largest contour
            largest_contour_bounding_boxes = np.array([dicom_utilities.get_largest_contour(image) for image in scan])
            largest_contour_bounding_box = [
                int(largest_contour_bounding_boxes[:, 0].min()),
                int(largest_contour_bounding_boxes[:, 1].min()),
                int(largest_contour_bounding_boxes[:, 2].max()),
                int(largest_contour_bounding_boxes[:, 3].max()),
            ]
            scan = scan[
                :,
                largest_contour_bounding_box[1]:largest_contour_bounding_box[3] + 1,
                largest_contour_bounding_box[0]:largest_contour_bounding_box[2] + 1,
            ]

            # Crop non-zero slices along xz, yz and xy planes
            mmin = np.array((scan > 0).nonzero()).min(axis=1)
            mmax = np.array((scan > 0).nonzero()).max(axis=1)
            scan = scan[
                mmin[0]:mmax[0] + 1,
                mmin[1]:mmax[1] + 1,
                mmin[2]:mmax[2] + 1
            ]

            scan = resize(torch.from_numpy(np.expand_dims(scan, axis=0)))
            scan = torch.squeeze(scan, dim=0).numpy().astype(np.uint8)

            np.save(output_directory / f'{patient_id}_{scan_id}.npy', scan)
