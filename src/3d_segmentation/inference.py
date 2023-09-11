import sys
import os
import yaml
from tqdm import tqdm
import numpy as np
import pandas as pd
import pydicom
import torch

import clip_driven_universal_model
import torch_transforms

sys.path.append('..')
import settings
import dicom_utilities
import visualization


if __name__ == '__main__':

    model_directory = settings.MODELS / 'clip_driven_universal_model'
    config = yaml.load(open(model_directory / 'config.yaml', 'r'), Loader=yaml.FullLoader)
    settings.logger.info(f'Running CLIP Driven Universal Model for inference')

    image_dataset_directory = settings.DATA / 'rsna-2023-abdominal-trauma-detection' / 'train_images'
    patient_ids = sorted(os.listdir(image_dataset_directory), key=lambda filename: int(filename))

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

    device = torch.device(config['inference']['device'])
    amp = config['inference']['amp']
    model = clip_driven_universal_model.load_clip_driven_universal_model(**config['model'], device=device)

    df_scan_rois = []

    for patient_id in tqdm(patient_ids):

        patient_directory = image_dataset_directory / str(patient_id)
        patient_scans = sorted(os.listdir(patient_directory), key=lambda filename: int(filename))

        for scan_id in tqdm(patient_scans):

            scan_directory = patient_directory / str(scan_id)
            df_scan = df_dicom_tags.loc[df_dicom_tags['scan_id'] == int(scan_id)].reset_index(drop=True)
            file_names = df_scan['slice_id'].values

            scan = []
            for file_name in file_names:
                dicom = pydicom.dcmread(str(scan_directory / f'{file_name}.dcm'))
                image = dicom.pixel_array
                image = dicom_utilities.adjust_pixel_values(image=image, dicom=dicom, **config['dataset']['pixel_values'])
                image = dicom_utilities.adjust_pixel_spacing(image=image, dicom=dicom, **config['dataset']['pixel_spacing'])
                scan.append(image)

            scan = np.array(scan)
            z_spacing = df_scan['z_position_diff'].abs().value_counts().index[0]
            spacing_factor = (np.round(1.5 / z_spacing, 3), 1.0, 1.0)
            transforms = torch_transforms.get_clip_driven_universal_model_transforms(**{
                'spacing_pixdim': spacing_factor,
                'spatial_size': config['dataset']['spatial_size']
            })['inference']
            inputs = torch.unsqueeze(transforms(scan), dim=0) / 255.

            predictions = clip_driven_universal_model.predict_clip_driven_universal_model(
                inputs=inputs,
                model=model,
                device=device,
                amp=amp,
                threshold=False
            )
            predictions = clip_driven_universal_model.post_process_predictions(predictions=predictions, threshold=0.5)
            rois = clip_driven_universal_model.find_rois(predictions=predictions)

            df_rois = pd.DataFrame(rois).T.reset_index().rename(columns={'index': 'roi'})
            df_rois = pd.melt(df_rois, id_vars='roi')
            df_rois['roi_id'] = df_rois['roi'] + '_' + df_rois['variable']
            df_rois = df_rois[['roi_id', 'value']].set_index('roi_id').T.reset_index(drop=True)
            df_rois['patient_id'] = patient_id
            df_rois['scan_id'] = scan_id
            df_rois.columns.name = None
            df_scan_rois.append(df_rois)

    df_scan_rois = pd.concat(df_scan_rois, axis=0, ignore_index=True)
    df_scan_rois.to_csv(model_directory / 'rois.csv', index=False)

    visualization_directory = model_directory / 'rois'
    visualization_directory.mkdir(parents=True, exist_ok=True)
    roi_columns = [column for column in df_scan_rois.columns if column not in ['patient_id', 'scan_id']]
    df_scan_rois[roi_columns] = df_scan_rois[roi_columns].astype(np.float32)

    for column in tqdm(roi_columns):
        visualization.visualize_continuous_feature_distribution(
            df=df_scan_rois,
            feature=column,
            title=f'{column} Histogram',
            path=visualization_directory / f'{column}_histogram.png'
        )
