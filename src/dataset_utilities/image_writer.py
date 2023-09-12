import sys
import os
from joblib import Parallel, delayed
from tqdm import tqdm
import cv2
import pydicom

sys.path.append('..')
import settings
import dicom_utilities


def write_image(dicom_file_path, output_directory, normalize_pixel_spacing, new_pixel_spacing):

    """
    Read DICOM file and write it as a png file

    Parameters
    ----------
    dicom_file_path: str
        Path of the DICOM file

    output_directory: pathlib.Path
        Path of the directory image will be written to

    normalize_pixel_spacing: bool
        Whether to normalize pixel spacing or not

    new_pixel_spacing: tuple
        Dataset pixel spacing after normalization
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

    if normalize_pixel_spacing:
        image = dicom_utilities.adjust_pixel_spacing(image=image, dicom=dicom, current_pixel_spacing='dataset', new_pixel_spacing=new_pixel_spacing)

    cv2.imwrite(str(output_directory / f'{dicom_file_path.split("/")[-1].split(".")[0]}.png'), image)


if __name__ == '__main__':

    train_directory = settings.DATA / 'rsna-2023-abdominal-trauma-detection' / 'train_images'
    output_directory = settings.DATA / 'datasets' / '2d_windowed_400w_50c_1mm_isotropic' / 'images'
    output_directory.mkdir(parents=True, exist_ok=True)

    train_patients = sorted(os.listdir(train_directory), key=lambda filename: int(filename))

    for patient in tqdm(train_patients):

        patient_directory = train_directory / patient
        patient_scans = sorted(os.listdir(patient_directory), key=lambda filename: int(filename))

        for scan in tqdm(patient_scans):

            scan_directory = patient_directory / scan
            scan_dicom_files = sorted(os.listdir(scan_directory), key=lambda filename: int(filename.split('.')[0]))
            scan_output_directory = output_directory / patient / scan
            scan_output_directory.mkdir(parents=True, exist_ok=True)

            Parallel(n_jobs=16)(
                delayed(write_image)(
                    dicom_file_path=str(scan_directory / file_name),
                    output_directory=scan_output_directory,
                    normalize_pixel_spacing=True,
                    new_pixel_spacing=(1.0, 1.0)
                )
                for file_name in tqdm(scan_dicom_files)
            )

            settings.logger.info(f'Finished writing {len(scan_dicom_files)} images for patient {patient} scan {scan}')
