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
        window_center=None, window_width=None,
        photometric_interpretation='dataset', max_pixel_value=1
    )
    cv2.imwrite(str(output_directory / f'{dicom_file_path.split("/")[-1].split(".")[0]}.png'), image)


if __name__ == '__main__':

    train_directory = settings.DATA / 'rsna-2023-abdominal-trauma-detection' / 'train_images'
    train_patients = sorted(os.listdir(train_directory), key=lambda filename: int(filename))
    output_directory = settings.DATA / 'datasets' / 'images'
    output_directory.mkdir(parents=True, exist_ok=True)

    for patient in tqdm(train_patients):

        patient_directory = train_directory / patient
        patient_scans = sorted(os.listdir(patient_directory), key=lambda filename: int(filename))

        for scan in tqdm(patient_scans):

            scan_directory = patient_directory / scan
            scan_dicom_files = sorted(os.listdir(scan_directory), key=lambda filename: int(filename.split('.')[0]))
            scan_output_directory = output_directory / patient / scan
            scan_output_directory.mkdir(parents=True, exist_ok=True)

            Parallel(n_jobs=16)(
                delayed(write_image)(dicom_file_path=str(scan_directory / file_name), output_directory=scan_output_directory)
                for file_name in tqdm(scan_dicom_files)
            )

            settings.logger.info(f'Finished writing images {len(scan_dicom_files)} of patient {patient} scan {scan}')
