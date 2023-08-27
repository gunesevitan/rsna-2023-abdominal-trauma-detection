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

    image_paths = glob(str(settings.DATA / 'datasets' / args.dataset / 'images' / '*' / '*' / '*.png'))
    metadata = []

    for image_path in tqdm(image_paths):

        image = cv2.imread(image_path, -1)

        patient_id = image_path.split('/')[-3]
        scan_id = image_path.split('/')[-2]
        image_id = image_path.split('/')[-1].split('.')[0]
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
            'image_path': image_path,
        })

    df_metadata = pd.DataFrame(metadata)
    df_metadata['patient_id'] = df_metadata['patient_id'].astype(np.int64)
    df_metadata['scan_id'] = df_metadata['scan_id'].astype(np.int64)
    df_metadata['image_id'] = df_metadata['image_id'].astype(np.int64)
    df_metadata = df_metadata.sort_values(by=['patient_id', 'scan_id', 'image_id'], ascending=True).reset_index(drop=True)
    df_metadata.to_parquet(settings.DATA / 'datasets' / args.dataset / 'metadata.parquet')
