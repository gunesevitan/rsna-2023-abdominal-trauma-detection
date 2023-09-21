import sys
import os
from tqdm import tqdm
import numpy as np
import cv2
import pydicom
from ultralytics import YOLO

sys.path.append('..')
import settings
import dicom_utilities

sys.path.append('../roi_detection')
import localization_utilities


def read_image(dicom_file_path, normalize_pixel_spacing, new_pixel_spacing):

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

    bounding_boxes, scores, labels = localization_utilities.predict_yolov8_model(image=image, model=yolov8_model)
    # Model filters partial slices by not predicting any objects
    if len(bounding_boxes) > 0:

        # Use first detection since it has the highest confidence score
        bounding_box = bounding_boxes[0]
        label = labels[0]

        if label == 'abdominal':
            # Crop image with its predicted bounding box if it's label is abdominal
            image = localization_utilities.crop_image(image=image, roi=bounding_box, roi_format='voc')
            if normalize_pixel_spacing:
                image = dicom_utilities.adjust_pixel_spacing(
                    image=image,
                    dicom=dicom,
                    current_pixel_spacing='dataset',
                    new_pixel_spacing=new_pixel_spacing
                )

            return image, label

        else:
            return None, label
    else:
        return None, None


if __name__ == '__main__':

    dicom_dataset_directory = settings.DATA / 'rsna-2023-abdominal-trauma-detection' / 'train_images'
    patient_ids = sorted(os.listdir(dicom_dataset_directory), key=lambda filename: int(filename))
    patient_ids = ['65149', '2384', '3414', '3785', '16731', '19384', '37022', '48508', '51038']

    output_directory = settings.DATA / 'datasets' / '2d_soft_tissue_window_cropped_raw_size' / 'images'
    output_directory.mkdir(parents=True, exist_ok=True)

    yolov8_model = YOLO(settings.MODELS / 'yolov8_roi_detection' / 'experiment' / 'weights' / 'last.pt')

    for patient in tqdm(patient_ids):

        patient_directory = dicom_dataset_directory / patient
        patient_scans = sorted(os.listdir(patient_directory), key=lambda filename: int(filename))

        for scan in patient_scans:

            scan_directory = patient_directory / scan
            file_names = sorted(os.listdir(scan_directory), key=lambda x: int(str(x).split('.')[0]))
            labels = []

            scan_output_directory = output_directory / patient / scan
            scan_output_directory.mkdir(parents=True, exist_ok=True)

            for file_name in tqdm(file_names):
                image, label = read_image(
                    dicom_file_path=str(scan_directory / file_name),
                    normalize_pixel_spacing=False,
                    new_pixel_spacing=(1.0, 1.0)
                )

                if image is not None:
                    cv2.imwrite(str(scan_output_directory / f'{file_name.split(".")[0]}.png'), image)

                if label is not None:
                    labels.append(label)

                if 'abdominal' in labels and 'upper' in labels and 'lower' in labels:
                    if labels[0] == 'upper' and np.all(np.array(labels[-5:]) == 'lower'):
                        break
                    elif labels[0] == 'lower' and np.all(np.array(labels[-5:]) == 'upper'):
                        break

            settings.logger.info(f'Finished writing {len(os.listdir(scan_output_directory))}/{len(file_names)} images for patient {patient} scan {scan}')
