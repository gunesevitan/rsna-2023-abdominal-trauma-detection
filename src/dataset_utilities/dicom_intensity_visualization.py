import sys
from tqdm import tqdm
import numpy as np
import pandas as pd
import pydicom

sys.path.append('..')
import settings
import dicom_utilities
import visualization


if __name__ == '__main__':

    train_directory = settings.DATA / 'rsna-2023-abdominal-trauma-detection'
    visualization_directory = settings.EDA / 'dicom_intensity_visualizations'
    visualization_directory.mkdir(parents=True, exist_ok=True)

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
    df_dicom_tags['image_path'] = df_dicom_tags['path'].apply(lambda x: str(train_directory) + '/' + x)

    patient_ids = [
        24936, 28381, 62705, 51136, 5149,
        17178, 22758, 14102, 37054, 42271,
        51531, 62705, 5729, 10127
    ]
    scan_ids = [
        59075, 50943, 10711, 17861, 51175,
        12055, 13287, 24136, 4010, 31995,
        2749, 10711, 65135, 1554
    ]

    for patient_id, scan_id in zip(patient_ids, scan_ids):

        df_scan = df_dicom_tags.loc[(df_dicom_tags['patient_id'] == patient_id) & (df_dicom_tags['scan_id'] == scan_id)].reset_index(drop=True)

        images = []

        for dicom_file_path in tqdm(df_scan['image_path'].values):
            dicom = pydicom.dcmread(str(dicom_file_path))
            image = dicom.pixel_array
            image = dicom_utilities.adjust_pixel_values(
                image=dicom.pixel_array, dicom=dicom,
                bits_allocated='dataset', bits_stored='dataset',
                rescale_slope='dataset', rescale_intercept='dataset',
                window_centers=[50], window_widths=[400],
                photometric_interpretation='dataset', max_pixel_value=1
            )
            image = dicom_utilities.adjust_pixel_spacing(image=image, dicom=dicom, current_pixel_spacing='dataset', new_pixel_spacing=(1.0, 1.0))
            images.append(image)

        images = np.stack(images)

        visualization.visualize_scan(
            images_or_dicoms=images,
            masks=None,
            path=visualization_directory / f'{patient_id}_{scan_id}.mp4'
        )
