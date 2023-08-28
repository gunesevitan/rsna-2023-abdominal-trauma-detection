import sys
import argparse
from glob import glob
from tqdm import tqdm
import numpy as np
import pandas as pd
import cv2

sys.path.append('..')
import settings


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', type=str)
    args = parser.parse_args()

    dataset_dimensions = args.dataset.split('_')[0]

    if dataset_dimensions == '2d':
        paths = glob(str(settings.DATA / 'datasets' / args.dataset / 'images' / '*' / '*' / '*.png'))
    elif dataset_dimensions == '3d':
        paths = glob(str(settings.DATA / 'datasets' / args.dataset / 'volumes' / '*.npy'))
    else:
        raise ValueError(f'Invalid dataset dimension {dataset_dimensions}')

    metadata = []

    for path in tqdm(paths):

        if dataset_dimensions == '2d':

            image = cv2.imread(path, -1)

            patient_id = path.split('/')[-3]
            scan_id = path.split('/')[-2]
            image_id = path.split('/')[-1].split('.')[0]
            image_height = image.shape[0]
            image_width = image.shape[1]
            image_mean = np.mean(image)
            image_std = np.std(image)

            metadata.append({
                'patient_id': patient_id,
                'scan_id': scan_id,
                'image_id': image_id,
                'image_height': image_height,
                'image_width': image_width,
                'image_mean': image_mean,
                'image_std': image_std,
                'image_path': path,
            })

        elif dataset_dimensions == '3d':

            volume = np.load(path)

            patient_id, scan_id = path.split('/')[-1].split('.')[0].split('_')
            volume_depth = volume.shape[0]
            volume_height = volume.shape[1]
            volume_width = volume.shape[2]
            volume_mean = np.mean(volume)
            volume_std = np.std(volume)

            metadata.append({
                'patient_id': patient_id,
                'scan_id': scan_id,
                'volume_depth': volume_depth,
                'volume_height': volume_height,
                'volume_width': volume_width,
                'volume_mean': volume_mean,
                'volume_std': volume_std,
                'volume_path': path,
            })

        else:
            raise ValueError(f'Invalid dataset dimension {dataset_dimensions}')

    df_metadata = pd.DataFrame(metadata)
    df_metadata['patient_id'] = df_metadata['patient_id'].astype(np.int64)
    df_metadata['scan_id'] = df_metadata['scan_id'].astype(np.int64)

    if dataset_dimensions == '2d':
        df_metadata['image_id'] = df_metadata['image_id'].astype(np.int64)
        df_metadata = df_metadata.sort_values(by=['patient_id', 'scan_id', 'image_id'], ascending=True).reset_index(drop=True)
    elif dataset_dimensions == '3d':
        df_metadata = df_metadata.sort_values(by=['patient_id', 'scan_id'], ascending=True).reset_index(drop=True)
    else:
        raise ValueError(f'Invalid dataset dimension {dataset_dimensions}')

    df_metadata.to_parquet(settings.DATA / 'datasets' / args.dataset / 'metadata.parquet')
