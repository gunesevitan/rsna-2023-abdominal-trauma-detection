import sys
from glob import glob
from tqdm import tqdm
import json
import numpy as np
import cv2

sys.path.append('..')
import settings


if __name__ == '__main__':

    image_paths = glob(str(settings.DATA / 'datasets' / 'images' / '*' / '*' / '*.png'))
    pixel_count = 0
    pixel_sum = 0
    pixel_squared_sum = 0

    for image_path in tqdm(image_paths):

        image = cv2.imread(image_path, -1)
        image = np.float32(image) / 255.
        # Accumulate pixel counts, sums and squared sums for dataset mean and standard deviation computation
        pixel_count += (image.shape[0] * image.shape[1])
        pixel_sum += np.sum(image, axis=(0, 1))
        pixel_squared_sum += np.sum(image ** 2, axis=(0, 1))

    mean = pixel_sum / pixel_count
    var = (pixel_squared_sum / pixel_count) - (mean ** 2)
    std = np.sqrt(var)

    # Save dataset statistics as a json file
    dataset_statistics = {
        'mean': mean,
        'std': std
    }
    with open(settings.DATA / 'datasets' / 'statistics.json', mode='w') as f:
        json.dump(dataset_statistics, f, indent=2)

    settings.logger.info(f'Dataset statistics are calculated with {len(image_paths)} images')
