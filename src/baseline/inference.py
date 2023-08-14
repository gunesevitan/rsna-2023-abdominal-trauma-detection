import sys
import json
import numpy as np
import pandas as pd

sys.path.append('..')
import settings
import metrics


if __name__ == '__main__':

    df = pd.read_csv(settings.DATA / 'rsna-2023-abdominal-trauma-detection' / 'train.csv')

    model_directory = settings.MODELS / 'baseline'

    scan_level_evaluation = False
    if scan_level_evaluation:
        df_train_dicom_tags = pd.read_parquet(settings.DATA / 'rsna-2023-abdominal-trauma-detection' / 'train_dicom_tags.parquet')
        df_train_dicom_tags['patient_id'] = df_train_dicom_tags['path'].apply(lambda x: str(x).split('/')[1]).astype(np.int64)
        df_train_dicom_tags['scan_id'] = df_train_dicom_tags['path'].apply(lambda x: str(x).split('/')[2]).astype(np.int64)
        df_patient_scans = df_train_dicom_tags.groupby('scan_id')['patient_id'].first().reset_index().sort_values(by=['patient_id', 'scan_id'], ascending=True)
        df = df_patient_scans.merge(df, on='patient_id', how='left')
        del df_patient_scans, df_train_dicom_tags

    settings.logger.info(f'Train Dataset Shape: {df.shape} - Memory Usage: {df.memory_usage().sum() / 1024 ** 2:.2f} MB')

    # Convert one-hot encoded target columns to a single multi-class target columns
    for multiclass_target in ['kidney', 'liver', 'spleen']:
        df[multiclass_target] = 0
        for label, column in enumerate([f'{multiclass_target}_healthy', f'{multiclass_target}_low', f'{multiclass_target}_high']):
            df.loc[df[column] == 1, multiclass_target] = label

    binary_target_columns = ['bowel_injury', 'extravasation_injury', 'any_injury']
    for column in binary_target_columns:
        df[f'{column}_prediction'] = df[column].mean()

    multiclass_target_column_groups = [
        ['kidney_healthy', 'kidney_low', 'kidney_high'],
        ['liver_healthy', 'liver_low', 'liver_high'],
        ['spleen_healthy', 'spleen_low', 'spleen_high']
    ]
    for column_group in multiclass_target_column_groups:
        for column in column_group:
            df[f'{column}_prediction'] = df[column].mean()

    scores = {}
    df = metrics.create_sample_weights(df=df)

    for column in binary_target_columns:
        target_scores = metrics.binary_classification_scores(
            y_true=df[column],
            y_pred=df[f'{column}_prediction'],
            sample_weights=df[f'{column}_weight']
        )
        scores[column] = target_scores
        settings.logger.info(f'{column} Scores: {json.dumps(target_scores)}')

    for multi_class_target_column, column_group in zip(['kidney', 'liver', 'spleen'], multiclass_target_column_groups):
        target_scores = metrics.multiclass_classification_scores(
            y_true=df[multi_class_target_column],
            y_pred=df[[f'{column}_prediction' for column in column_group]],
            sample_weights=df[f'{multi_class_target_column}_weight']
        )
        scores[multi_class_target_column] = target_scores
        settings.logger.info(f'{multi_class_target_column} Scores: {json.dumps(target_scores)}')

    df_scores = pd.DataFrame(scores).T.reset_index().rename(columns={'index': 'target'})
    df_scores.to_csv(model_directory / 'scores.csv', index=False)

    df_score_aggregations = df_scores.iloc[:, 1:].agg(['mean', 'std', 'min', 'max']).reset_index().rename(columns={'index': 'aggregation'})
    df_score_aggregations.to_csv(model_directory / 'score_aggregations.csv', index=False)
