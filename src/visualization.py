import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pydicom

import settings


def visualize_categorical_feature_distribution(df, feature, title, path=None):

    """
    Visualize distribution of given categorical column in given dataframe

    Parameters
    ----------
    df: pandas.DataFrame
        Dataframe with given feature column

    feature: str
        Name of the categorical feature

    title: str
        Title of the plot

    path: path-like str or None
        Path of the output file or None (if path is None, plot is displayed with selected backend)
    """

    value_counts = df[feature].value_counts()

    fig, ax = plt.subplots(figsize=(24, df[feature].value_counts().shape[0] + 4), dpi=100)
    ax.bar(
        x=np.arange(len(value_counts)),
        height=value_counts.values,
    )
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.set_xticks(
        np.arange(len(value_counts)),
        [
            f'{value} ({count:,})' for value, count in value_counts.to_dict().items()
        ]
    )
    ax.tick_params(axis='x', labelsize=15, pad=10)
    ax.tick_params(axis='y', labelsize=15, pad=10)
    ax.set_title(title, size=20, pad=15)

    if path is None:
        plt.show()
    else:
        plt.savefig(path)
        plt.close(fig)


def visualize_continuous_feature_distribution(df, feature, title, path=None):

    """
    Visualize distribution of given continuous column in given dataframe

    Parameters
    ----------
    df: pandas.DataFrame
        Dataframe with given feature column

    feature: str
        Name of the continuous feature,

    title: str
        Title of the plot

    path: path-like str or None
        Path of the output file or None (if path is None, plot is displayed with selected backend)
    """

    fig, ax = plt.subplots(figsize=(24, 6), dpi=100)
    ax.hist(df[feature], bins=16)
    ax.tick_params(axis='x', labelsize=15)
    ax.tick_params(axis='y', labelsize=15)
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.set_title(
        f'''
        {feature}
        Mean: {np.mean(df[feature]):.2f} Std: {np.std(df[feature]):.2f}
        Min: {np.min(df[feature]):.2f} Max: {np.max(df[feature]):.2f}
        ''',
        size=15,
        pad=12.5,
        loc='center',
        wrap=True
    )

    if path is None:
        plt.show()
    else:
        plt.savefig(path, bbox_inches='tight')
        plt.close(fig)


def visualize_image(image_or_dicom, path=None):

    """
    Visualize given image

    Parameters
    ----------
    image_or_dicom: numpy.ndarray of shape (height, width, 3) or pydicom.dataset.FileDataset
        Image array or DICOM

    path: str or None
        Path of the output file or None (if path is None, plot is displayed with selected backend)
    """

    if isinstance(image_or_dicom, pydicom.dataset.FileDataset):
        image = image_or_dicom.pixel_array
    elif isinstance(image_or_dicom, np.ndarray):
        image = image_or_dicom.copy()
    else:
        raise TypeError(f'Invalid image type {type(image_or_dicom)}')

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(image, cmap='gray')
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.tick_params(axis='x', labelsize=15, pad=10)
    ax.tick_params(axis='y', labelsize=15, pad=10)
    ax.set_title(
        f'''
        Image Shape: {image.shape}
        Mean: {np.mean(image):.2f} Std: {np.std(image):.2f}
        Min: {np.min(image):.2f} Max: {np.max(image):.2f}
        ''',
        size=15,
        pad=12.5,
        loc='center',
        wrap=True
    )

    if path is None:
        plt.show()
    else:
        plt.savefig(path)
        plt.close(fig)


if __name__ == '__main__':

    df_train = pd.read_csv(settings.DATA / 'rsna-2023-abdominal-trauma-detection' / 'train.csv')
    settings.logger.info(f'Train Dataset Shape: {df_train.shape} - Memory Usage: {df_train.memory_usage().sum() / 1024 ** 2:.2f} MB')

    # Convert one-hot encoded target columns to a single multi-class target columns
    for organ in ['kidney', 'liver', 'spleen']:
        df_train[organ] = 0
        for label, column in enumerate([f'{organ}_healthy', f'{organ}_low', f'{organ}_high']):
            df_train.loc[df_train[column] == 1, organ] = label

    # Visualize binary and multi-class target columns
    target_columns = ['bowel_injury', 'extravasation_injury', 'kidney', 'liver', 'spleen', 'any_injury']
    for column in target_columns:
        visualize_categorical_feature_distribution(
            df=df_train,
            feature=column,
            title=f'{column} Value Counts',
            path=settings.EDA / f'target_{column}_value_counts.png'
        )

    df_train_dicom_tags = pd.read_parquet(settings.DATA / 'rsna-2023-abdominal-trauma-detection' / 'train_dicom_tags.parquet')
    settings.logger.info(f'Train DICOM Tags Dataset Shape: {df_train_dicom_tags.shape} - Memory Usage: {df_train_dicom_tags.memory_usage().sum() / 1024 ** 2:.2f} MB')

    # Create scan_id and patient_id features and visualize scan counts of patients
    df_train_dicom_tags['scan_id'] = df_train_dicom_tags['path'].apply(lambda x: str(x).split('/')[2])
    df_train_dicom_tags['patient_id'] = df_train_dicom_tags['path'].apply(lambda x: str(x).split('/')[1])
    df_patient_scan_counts = pd.DataFrame(df_train_dicom_tags.groupby('patient_id')['scan_id'].nunique())
    visualize_categorical_feature_distribution(
        df=df_patient_scan_counts,
        feature='scan_id',
        title=f'Patient Scan Counts',
        path=settings.EDA / f'dicom_tags_patient_scan_counts.png'
    )

    # Create slices feature and visualize counts of scan dimensions
    df_train_dicom_tags['slices'] = df_train_dicom_tags.groupby(['patient_id', 'scan_id'])['scan_id'].transform('count')
    visualize_continuous_feature_distribution(
        df=df_train_dicom_tags.groupby('scan_id')[['slices']].first(),
        feature='slices',
        title='Scan Slices Histogram',
        path=settings.EDA / f'dicom_tags_scan_slices_histogram.png'
    )
    visualize_continuous_feature_distribution(
        df=df_train_dicom_tags.groupby('scan_id')[['Rows']].first(),
        feature='Rows',
        title='Scan Heights Histogram',
        path=settings.EDA / f'dicom_tags_scan_heights_histogram.png'
    )
    visualize_continuous_feature_distribution(
        df=df_train_dicom_tags.groupby('scan_id')[['Columns']].first(),
        feature='Columns',
        title='Scan Widths Histogram',
        path=settings.EDA / f'dicom_tags_scan_widths_histogram.png'
    )

    # Visualize counts of data types of the scans
    for feature in ['BitsAllocated', 'BitsStored', 'HighBit']:
        visualize_categorical_feature_distribution(
            df=df_train_dicom_tags,
            feature=feature,
            title=f'{feature} Value Counts',
            path=settings.EDA / f'dicom_tags_{feature}_value_counts.png'
        )

    # Visualize counts of pixel values of the scans
    for feature in [
        'WindowCenter', 'WindowWidth',
        'RescaleIntercept', 'RescaleSlope', 'RescaleType',
        'PhotometricInterpretation', 'PixelRepresentation', 'SamplesPerPixel'
    ]:
        visualize_categorical_feature_distribution(
            df=df_train_dicom_tags,
            feature=feature,
            title=f'{feature} Value Counts',
            path=settings.EDA / f'dicom_tags_{feature}_value_counts.png'
        )

    df_train_series_meta = pd.read_csv(settings.DATA / 'rsna-2023-abdominal-trauma-detection' / 'train_series_meta.csv')
    settings.logger.info(f'Train Series Meta Dataset Shape: {df_train_series_meta.shape} - Memory Usage: {df_train_series_meta.memory_usage().sum() / 1024 ** 2:.2f} MB')

    visualize_categorical_feature_distribution(
        df=df_train_series_meta,
        feature='incomplete_organ',
        title='incomplete_organ Value Counts',
        path=settings.EDA / f'series_meta_incomplete_organ_value_counts.png'
    )

    visualize_continuous_feature_distribution(
        df=df_train_series_meta,
        feature='aortic_hu',
        title='aortic_hu Histogram',
        path=settings.EDA / f'series_meta_aortic_hu_histogram.png'
    )

    df_image_level_labels = pd.read_csv(settings.DATA / 'rsna-2023-abdominal-trauma-detection' / 'image_level_labels.csv')
    settings.logger.info(f'Image Level Labels Dataset Shape: {df_image_level_labels.shape} - Memory Usage: {df_image_level_labels.memory_usage().sum() / 1024 ** 2:.2f} MB')

    visualize_categorical_feature_distribution(
        df=df_image_level_labels,
        feature='injury_name',
        title='injury_name Value Counts',
        path=settings.EDA / f'image_level_labels_injury_name_value_counts.png'
    )

    visualize_continuous_feature_distribution(
        df=df_image_level_labels,
        feature='instance_number',
        title='instance_number Histogram',
        path=settings.EDA / f'image_level_labels_instance_number_histogram.png'
    )

    # Create slices feature and merge it to image level labels
    df_scan_slice_counts = df_train_dicom_tags.groupby('scan_id')['slices'].first().reset_index().rename(columns={'scan_id': 'series_id'})
    df_scan_slice_counts['series_id'] = df_scan_slice_counts['series_id'].astype(np.int64)
    df_image_level_labels = df_image_level_labels.merge(df_scan_slice_counts, on='series_id', how='left')
    # Create instance number slices ratio feature and visualize it
    df_image_level_labels['instance_number_slices_ratio'] = df_image_level_labels['instance_number'] / df_image_level_labels['slices']

    visualize_continuous_feature_distribution(
        df=df_image_level_labels,
        feature='instance_number_slices_ratio',
        title='instance_number_slices_ratio Histogram',
        path=settings.EDA / f'image_level_labels_instance_number_slices_ratio_histogram.png'
    )
