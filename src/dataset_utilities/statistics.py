import sys
import argparse
from glob import glob
from tqdm import tqdm
import json
import numpy as np
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

    pixel_count = 0
    pixel_sum = 0
    pixel_squared_sum = 0

    for path in tqdm(paths):

        if dataset_dimensions == '2d':
            image = cv2.imread(path, -1)
            image = np.float32(image) / 255.
            pixel_count += (image.shape[0] * image.shape[1])
            pixel_sum += np.sum(image, axis=(0, 1))
            pixel_squared_sum += np.sum(image ** 2, axis=(0, 1))
        elif dataset_dimensions == '3d':
            volume = np.load(path)
            volume = np.float32(volume) / 255.
            pixel_count += (volume.shape[0] * volume.shape[1] * volume.shape[2])
            pixel_sum += np.sum(volume, axis=(0, 1, 2))
            pixel_squared_sum += np.sum(volume ** 2, axis=(0, 1, 2))
        else:
            raise ValueError(f'Invalid dataset dimension {dataset_dimensions}')

    mean = pixel_sum / pixel_count
    var = (pixel_squared_sum / pixel_count) - (mean ** 2)
    std = np.sqrt(var)

    # Save dataset statistics as a json file
    dataset_statistics = {
        'mean': mean,
        'std': std
    }
    with open(settings.DATA / 'datasets' / args.dataset / 'statistics.json', mode='w') as f:
        json.dump(dataset_statistics, f, indent=2)

    settings.logger.info(f'Dataset statistics are calculated with {len(paths)} images/volumes')
