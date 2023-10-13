import sys
import json
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

import warnings
warnings.filterwarnings('ignore')

sys.path.append('..')
import settings
import metrics


if __name__ == '__main__':

    binary_target_columns = ['bowel_injury', 'extravasation_injury', 'any_injury']
    multiclass_target_column_groups = [
        ['kidney_healthy', 'kidney_low', 'kidney_high'],
        ['liver_healthy', 'liver_low', 'liver_high'],
        ['spleen_healthy', 'spleen_low', 'spleen_high']
    ]

    df_mil_efficientnetb0_predictions = pd.read_csv(settings.MODELS / 'mil_efficientnetb0_3d_1w_contour_cropped_96x256x256' / 'oof_predictions.csv')
    df_mil_densenet121_predictions = pd.read_csv(settings.MODELS / 'mil_densenet121_3d_1w_contour_cropped_96x256x256' / 'oof_predictions.csv')
    df_lstm_efficientnetb0_predictions = pd.read_csv(settings.MODELS / 'lstm_efficientnetb0_3d_1w_contour_cropped_96x256x256' / 'oof_predictions.csv')
    df_lstm_efficientnetv2t_predictions = pd.read_csv(settings.MODELS / 'lstm_efficientnetv2t_3d_1w_contour_cropped_96x256x256' / 'oof_predictions.csv')
    df_predictions = df_mil_efficientnetb0_predictions.copy(deep=True)

    scale_probabilities = False
    if scale_probabilities:
        for df in [df_mil_efficientnetb0_predictions, df_mil_densenet121_predictions, df_lstm_efficientnetb0_predictions, df_lstm_efficientnetv2t_predictions]:
            for column in [column for column in df_lstm_efficientnetv2t_predictions.columns.tolist() if column.endswith('prediction')]:
                df[column] = MinMaxScaler().fit_transform(df[column].values.reshape(-1, 1))

    mil_efficientnetb0_bowel_weight = 0.45
    mil_densenet121_bowel_weight = 0.25
    lstm_efficientnetb0_bowel_weight = 0.15
    lstm_efficientnetv2t_bowel_weight = 0.15

    df_predictions['bowel_injury_prediction'] = (df_mil_efficientnetb0_predictions['bowel_injury_prediction'] * mil_efficientnetb0_bowel_weight) + \
                                                (df_mil_densenet121_predictions['bowel_injury_prediction'] * mil_densenet121_bowel_weight) + \
                                                (df_lstm_efficientnetb0_predictions['bowel_injury_prediction'] * lstm_efficientnetb0_bowel_weight) + \
                                                (df_lstm_efficientnetv2t_predictions['bowel_injury_prediction'] * lstm_efficientnetv2t_bowel_weight)

    mil_efficientnetb0_extravasation_weight = 0.3
    mil_densenet121_extravasation_weight = 0.3
    lstm_efficientnetb0_extravasation_weight = 0.3
    lstm_efficientnetv2t_extravasation_weight = 0.1

    df_predictions['extravasation_injury_prediction'] = (df_mil_efficientnetb0_predictions['extravasation_injury_prediction'] * mil_efficientnetb0_extravasation_weight) + \
                                                        (df_mil_densenet121_predictions['extravasation_injury_prediction'] * mil_densenet121_extravasation_weight) + \
                                                        (df_lstm_efficientnetb0_predictions['extravasation_injury_prediction'] * lstm_efficientnetb0_extravasation_weight) + \
                                                        (df_lstm_efficientnetv2t_predictions['extravasation_injury_prediction'] * lstm_efficientnetv2t_extravasation_weight)

    mil_efficientnetb0_kidney_weight = 0.25
    mil_densenet121_kidney_weight = 0.25
    lstm_efficientnetb0_kidney_weight = 0.25
    lstm_efficientnetv2t_kidney_weight = 0.25

    df_predictions['kidney_healthy_prediction'] = (df_mil_efficientnetb0_predictions['kidney_healthy_prediction'] * mil_efficientnetb0_kidney_weight) + \
                                                  (df_mil_densenet121_predictions['kidney_healthy_prediction'] * mil_densenet121_kidney_weight) + \
                                                  (df_lstm_efficientnetb0_predictions['kidney_healthy_prediction'] * lstm_efficientnetb0_kidney_weight) + \
                                                  (df_lstm_efficientnetv2t_predictions['kidney_healthy_prediction'] * lstm_efficientnetv2t_kidney_weight)

    df_predictions['kidney_low_prediction'] = (df_mil_efficientnetb0_predictions['kidney_low_prediction'] * mil_efficientnetb0_kidney_weight) + \
                                              (df_mil_densenet121_predictions['kidney_low_prediction'] * mil_densenet121_kidney_weight) + \
                                              (df_lstm_efficientnetb0_predictions['kidney_low_prediction'] * lstm_efficientnetb0_kidney_weight) + \
                                              (df_lstm_efficientnetv2t_predictions['kidney_low_prediction'] * lstm_efficientnetv2t_kidney_weight)

    df_predictions['kidney_high_prediction'] = (df_mil_efficientnetb0_predictions['kidney_high_prediction'] * mil_efficientnetb0_kidney_weight) + \
                                               (df_mil_densenet121_predictions['kidney_high_prediction'] * mil_densenet121_kidney_weight) + \
                                               (df_lstm_efficientnetb0_predictions['kidney_high_prediction'] * lstm_efficientnetb0_kidney_weight) + \
                                               (df_lstm_efficientnetv2t_predictions['kidney_high_prediction'] * lstm_efficientnetv2t_kidney_weight)

    mil_efficientnetb0_liver_weight = 0.25
    mil_densenet121_liver_weight = 0.25
    lstm_efficientnetb0_liver_weight = 0.25
    lstm_efficientnetv2t_liver_weight = 0.25

    df_predictions['liver_healthy_prediction'] = (df_mil_efficientnetb0_predictions['liver_healthy_prediction'] * mil_efficientnetb0_liver_weight) + \
                                                 (df_mil_densenet121_predictions['liver_healthy_prediction'] * mil_densenet121_liver_weight) + \
                                                 (df_lstm_efficientnetb0_predictions['liver_healthy_prediction'] * lstm_efficientnetb0_liver_weight) + \
                                                 (df_lstm_efficientnetv2t_predictions['liver_healthy_prediction'] * lstm_efficientnetv2t_liver_weight)

    df_predictions['liver_low_prediction'] = (df_mil_efficientnetb0_predictions['liver_low_prediction'] * mil_efficientnetb0_liver_weight) + \
                                             (df_mil_densenet121_predictions['liver_low_prediction'] * mil_densenet121_liver_weight) + \
                                             (df_lstm_efficientnetb0_predictions['liver_low_prediction'] * lstm_efficientnetb0_liver_weight) + \
                                             (df_lstm_efficientnetv2t_predictions['liver_low_prediction'] * lstm_efficientnetv2t_liver_weight)

    df_predictions['liver_high_prediction'] = (df_mil_efficientnetb0_predictions['liver_high_prediction'] * mil_efficientnetb0_liver_weight) + \
                                              (df_mil_densenet121_predictions['liver_high_prediction'] * mil_densenet121_liver_weight) + \
                                              (df_lstm_efficientnetb0_predictions['liver_high_prediction'] * lstm_efficientnetb0_liver_weight) + \
                                              (df_lstm_efficientnetv2t_predictions['liver_high_prediction'] * lstm_efficientnetv2t_liver_weight)

    mil_efficientnetb0_spleen_weight = 0.25
    mil_densenet121_spleen_weight = 0.25
    lstm_efficientnetb0_spleen_weight = 0.25
    lstm_efficientnetv2t_spleen_weight = 0.25

    df_predictions['spleen_healthy_prediction'] = (df_mil_efficientnetb0_predictions['spleen_healthy_prediction'] * mil_efficientnetb0_spleen_weight) + \
                                                  (df_mil_densenet121_predictions['spleen_healthy_prediction'] * mil_densenet121_spleen_weight) + \
                                                  (df_lstm_efficientnetb0_predictions['spleen_healthy_prediction'] * lstm_efficientnetb0_spleen_weight) + \
                                                  (df_lstm_efficientnetv2t_predictions['spleen_healthy_prediction'] * lstm_efficientnetv2t_spleen_weight)

    df_predictions['spleen_low_prediction'] = (df_mil_efficientnetb0_predictions['spleen_low_prediction'] * mil_efficientnetb0_spleen_weight) + \
                                              (df_mil_densenet121_predictions['spleen_low_prediction'] * mil_densenet121_spleen_weight) + \
                                              (df_lstm_efficientnetb0_predictions['spleen_low_prediction'] * lstm_efficientnetb0_spleen_weight) + \
                                              (df_lstm_efficientnetv2t_predictions['spleen_low_prediction'] * lstm_efficientnetv2t_spleen_weight)

    df_predictions['spleen_high_prediction'] = (df_mil_efficientnetb0_predictions['spleen_high_prediction'] * mil_efficientnetb0_spleen_weight) + \
                                               (df_mil_densenet121_predictions['spleen_high_prediction'] * mil_densenet121_spleen_weight) + \
                                               (df_lstm_efficientnetb0_predictions['spleen_high_prediction'] * lstm_efficientnetb0_spleen_weight) + \
                                               (df_lstm_efficientnetv2t_predictions['spleen_high_prediction'] * lstm_efficientnetv2t_spleen_weight)

    oof_scores = {}

    df_predictions['bowel_healthy_prediction'] = 1 - df_predictions['bowel_injury_prediction']
    df_predictions['bowel_healthy_prediction'] *= 1.
    df_predictions['bowel_injury_prediction'] *= 1.
    df = metrics.normalize_probabilities(df_predictions, columns=['bowel_healthy_prediction', 'bowel_injury_prediction'])

    df_predictions['extravasation_healthy_prediction'] = 1 - df_predictions['extravasation_injury_prediction']
    df_predictions['extravasation_healthy_prediction'] *= 1.
    df_predictions['extravasation_injury_prediction'] *= 1.4
    df = metrics.normalize_probabilities(df, columns=['extravasation_healthy_prediction', 'extravasation_injury_prediction'])

    df_predictions['kidney_healthy_prediction'] *= 1.
    df_predictions['kidney_low_prediction'] *= 1.1
    df_predictions['kidney_high_prediction'] *= 1.1
    df_predictions = metrics.normalize_probabilities(df_predictions, columns=['kidney_healthy_prediction', 'kidney_low_prediction', 'kidney_high_prediction'])

    df_predictions['liver_healthy_prediction'] *= 1.
    df_predictions['liver_low_prediction'] *= 1.3
    df_predictions['liver_high_prediction'] *= 1.3
    df_predictions = metrics.normalize_probabilities(df_predictions, columns=['liver_healthy_prediction', 'liver_low_prediction', 'liver_high_prediction'])

    df_predictions['spleen_healthy_prediction'] *= 1.
    df_predictions['spleen_low_prediction'] *= 1.75
    df_predictions['spleen_high_prediction'] *= 1.75
    df_predictions = metrics.normalize_probabilities(df_predictions, columns=['spleen_healthy_prediction', 'spleen_low_prediction', 'spleen_high_prediction'])

    df_predictions['any_injury_prediction'] = (1 - df_predictions[[
        'bowel_healthy_prediction', 'extravasation_healthy_prediction',
        'kidney_healthy_prediction', 'liver_healthy_prediction', 'spleen_healthy_prediction'
    ]]).max(axis=1)

    df_predictions = df_predictions.groupby('patient_id').max().reset_index()

    for column in binary_target_columns:
        target_scores = metrics.binary_classification_scores(
            y_true=df_predictions[column],
            y_pred=df_predictions[f'{column}_prediction'],
            sample_weights=df_predictions[f'{column}_weight']
        )
        oof_scores[column] = target_scores
        settings.logger.info(
            f'''
            {column}
            Target Mean: {df_predictions[column].mean():.4f}
            Predictions Mean: {df_predictions[f'{column}_prediction'].mean():.4f} Std: {df_predictions[f'{column}_prediction'].std():.4f} Min: {df_predictions[f'{column}_prediction'].min():.4f} Max: {df_predictions[f'{column}_prediction'].max():.4f}
            OOF Scores: {json.dumps(target_scores)}
            '''
        )

    for multi_class_target_column, column_group in zip(['kidney', 'liver', 'spleen'], multiclass_target_column_groups):
        target_scores = metrics.multiclass_classification_scores(
            y_true=df_predictions[multi_class_target_column],
            y_pred=df_predictions[[f'{column}_prediction' for column in column_group]],
            sample_weights=df_predictions[f'{multi_class_target_column}_weight']
        )
        oof_scores[multi_class_target_column] = target_scores
        settings.logger.info(
            f'''
            {multi_class_target_column}
            Target Healthy Mean: {df_predictions[f'{multi_class_target_column}_healthy'].mean():.4f}
            Predictions Healthy Mean: {df_predictions[f'{multi_class_target_column}_healthy_prediction'].mean():.4f} Std: {df_predictions[f'{multi_class_target_column}_healthy_prediction'].std():.4f} Min: {df_predictions[f'{multi_class_target_column}_healthy_prediction'].min():.4f} Max: {df_predictions[f'{multi_class_target_column}_healthy_prediction'].max():.4f}
            Target Low Mean: {df_predictions[f'{multi_class_target_column}_low'].mean():.4f}
            Predictions Low Mean: {df_predictions[f'{multi_class_target_column}_low_prediction'].mean():.4f} Std: {df_predictions[f'{multi_class_target_column}_low_prediction'].std():.4f} Min: {df_predictions[f'{multi_class_target_column}_low_prediction'].min():.4f} Max: {df_predictions[f'{multi_class_target_column}_low_prediction'].max():.4f}
            Target High Mean: {df_predictions[f'{multi_class_target_column}_high'].mean():.4f}
            Predictions High Mean: {df_predictions[f'{multi_class_target_column}_high_prediction'].mean():.4f} Std: {df_predictions[f'{multi_class_target_column}_high_prediction'].std():.4f} Min: {df_predictions[f'{multi_class_target_column}_high_prediction'].min():.4f} Max: {df_predictions[f'{multi_class_target_column}_high_prediction'].max():.4f}
            OOF Scores: {json.dumps(target_scores)}
            '''
        )
        settings.logger.info(f'OOF {multi_class_target_column} Scores: {json.dumps(target_scores)}')

    df_oof_scores = pd.DataFrame(oof_scores).T.reset_index().rename(columns={'index': 'target'})
    settings.logger.info(
        f'''
        OOF Scores Per Target
        {df_oof_scores}
        
        OOF Scores Means
        {df_oof_scores.iloc[:, 1:].mean()}
        '''
    )
