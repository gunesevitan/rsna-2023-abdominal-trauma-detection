import sys
import os
from joblib import Parallel, delayed
from tqdm import tqdm
import cv2
import pydicom

sys.path.append('..')
import settings
import dicom_utilities


def write_image(dicom_file_path, output_directory):

    """
    Read DICOM file and write it as a png file

    Parameters
    ----------
    dicom_file_path: str
        Path of the DICOM file

    output_directory: pathlib.Path
        Path of the directory image will be written to
    """

    dicom = pydicom.dcmread(str(dicom_file_path))
    image = dicom.pixel_array
    image = dicom_utilities.adjust_pixel_values(
        image=image, dicom=dicom,
        bits_allocated='dataset', bits_stored='dataset',
        rescale_slope='dataset', rescale_intercept='dataset',
        window_centers=['dataset'], window_widths=['dataset'],
        photometric_interpretation='dataset', max_pixel_value=1
    )

    cv2.imwrite(str(output_directory / f'{dicom_file_path.split("/")[-1].split(".")[0]}.png'), image)


if __name__ == '__main__':

    dicom_dataset_directory = settings.DATA / 'rsna-2023-abdominal-trauma-detection' / 'train_images'
    patient_ids = ['19', '26575', '394', '851', '2232', '2429', '2986', '3194']

    output_directory = settings.DATA / 'datasets' / '2d_soft_tissue_window_raw_size' / 'images'
    output_directory.mkdir(parents=True, exist_ok=True)

    for patient in tqdm(patient_ids):

        patient_directory = dicom_dataset_directory / patient
        patient_scans = sorted(os.listdir(patient_directory), key=lambda filename: int(filename))

        for scan in patient_scans:

            scan_directory = patient_directory / scan
            file_names = sorted(os.listdir(scan_directory), key=lambda filename: int(filename.split('.')[0]))

            scan_output_directory = output_directory / patient / scan
            scan_output_directory.mkdir(parents=True, exist_ok=True)

            Parallel(n_jobs=16)(
                delayed(write_image)(
                    dicom_file_path=str(scan_directory / file_name),
                    output_directory=scan_output_directory
                )
                for file_name in tqdm(file_names)
            )

            settings.logger.info(f'Finished writing {len(file_names)} images for patient {patient} scan {scan}')
