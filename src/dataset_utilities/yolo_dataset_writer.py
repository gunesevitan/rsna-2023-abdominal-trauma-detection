import sys
import os
from shutil import copyfile
from tqdm import tqdm
import numpy as np
import pandas as pd

sys.path.append('..')
import settings
import annotation_utilities


if __name__ == '__main__':

    yolo_dataset_directory = settings.DATA / 'datasets' / 'yolo_roi_detection'
    yolo_dataset_directory.mkdir(parents=True, exist_ok=True)

    image_directory = settings.DATA / 'datasets' / '2d_soft_tissue_window_raw_size' / 'images'
    annotation_directory = settings.DATA / 'datasets' / '2d_soft_tissue_window_raw_size' / 'annotations'

    df_annotations = []

    for annotation_file in tqdm(os.listdir(annotation_directory)):
        df_annotation = pd.read_csv(annotation_directory / annotation_file)
        df_annotation['patient_id'] = annotation_file.split('.')[0].split('_')[0]
        df_annotation['scan_id'] = annotation_file.split('.')[0].split('_')[1]
        df_annotations.append(df_annotation)

    df_annotations = pd.concat(df_annotations, axis=0, ignore_index=True)
    df_annotations['patient_id'] = df_annotations['patient_id'].astype(int)
    df_annotations['scan_id'] = df_annotations['scan_id'].astype(int)
    df_annotations['image_id'] = df_annotations['image_name'].apply(lambda x: str(x).split('.')[0])
    df_annotations['image_path'] = df_annotations['patient_id'].astype(str) + '/' + df_annotations['scan_id'].astype(str) + '/' + df_annotations['image_name']
    df_annotations['image_path'] = df_annotations['image_path'].apply(lambda x: str(image_directory) + '/' + x)

    scan_ids = sorted(df_annotations.scan_id.unique().tolist())
    fold1_scan_ids = scan_ids[:len(scan_ids) // 2]
    fold2_scan_ids = scan_ids[len(scan_ids) // 2:]

    category_ids = {'abdominal': 0, 'upper': 1, 'lower': 2}
    categories = [{'name': name, 'id': category_id} for name, category_id in category_ids.items()]

    fold = None

    if fold == 1:
        training_scan_ids = fold2_scan_ids
        validation_scan_ids = fold1_scan_ids
    elif fold == 2:
        training_scan_ids = fold1_scan_ids
        validation_scan_ids = fold2_scan_ids
    else:
        training_scan_ids = fold1_scan_ids + fold2_scan_ids
        validation_scan_ids = training_scan_ids

    training_mask = df_annotations['scan_id'].isin(training_scan_ids)
    training_directory = yolo_dataset_directory / 'train'
    training_directory.mkdir(parents=True, exist_ok=True)

    for _, row in tqdm(df_annotations.loc[training_mask].iterrows(), total=df_annotations.loc[training_mask].shape[0]):

        copyfile(row['image_path'], training_directory / f'{row["patient_id"]}_{row["scan_id"]}_{row["image_id"]}.png')

        with open(training_directory / f'{row["patient_id"]}_{row["scan_id"]}_{row["image_id"]}.txt', 'w') as text_file:
            label = category_ids[row['label_name']]
            bounding_box = annotation_utilities.coco_to_yolo_bounding_box([row['bbox_x'], row['bbox_y'], row['bbox_width'], row['bbox_height']])
            bounding_box = np.array(bounding_box, dtype=np.float32)
            bounding_box[0] /= row['image_width']
            bounding_box[1] /= row['image_height']
            bounding_box[2] /= row['image_width']
            bounding_box[3] /= row['image_height']
            text_file.write(f'{label} {" ".join(map(str, bounding_box))}\n')

    if validation_scan_ids is not None:

        validation_mask = df_annotations['scan_id'].isin(validation_scan_ids)
        validation_directory = yolo_dataset_directory / 'valid'
        validation_directory.mkdir(parents=True, exist_ok=True)

        for _, row in tqdm(df_annotations.loc[validation_mask].iterrows(), total=df_annotations.loc[validation_mask].shape[0]):

            copyfile(row['image_path'], validation_directory / f'{row["patient_id"]}_{row["scan_id"]}_{row["image_id"]}.png')

            with open(validation_directory / f'{row["patient_id"]}_{row["scan_id"]}_{row["image_id"]}.txt', 'w') as text_file:
                label = category_ids[row['label_name']]
                bounding_box = annotation_utilities.coco_to_yolo_bounding_box([row['bbox_x'], row['bbox_y'], row['bbox_width'], row['bbox_height']])
                bounding_box = np.array(bounding_box, dtype=np.float32)
                bounding_box[0] /= row['image_width']
                bounding_box[1] /= row['image_height']
                bounding_box[2] /= row['image_width']
                bounding_box[3] /= row['image_height']
                text_file.write(f'{label} {" ".join(map(str, bounding_box))}\n')
