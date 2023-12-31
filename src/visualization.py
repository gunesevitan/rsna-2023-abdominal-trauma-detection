from tqdm import tqdm
import numpy as np
import pandas as pd
import cv2
import pydicom
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import seaborn as sns
from sklearn.metrics import auc

import settings
import annotation_utilities


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
        title + f'''
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


def visualize_correlations(df, columns, title, path=None):

    """
    Visualize correlations of given columns in given dataframe

    Parameters
    ----------
    df: pandas.DataFrame
        Dataframe with given column

    columns: list
        List of names of columns

    title: str
        Title of the plot

    path: path-like str or None
        Path of the output file or None (if path is None, plot is displayed with selected backend)
    """

    fig, ax = plt.subplots(figsize=(20, 20), dpi=100)
    ax = sns.heatmap(
        df[columns].corr(),
        annot=True,
        square=True,
        cmap='coolwarm',
        annot_kws={'size': 12},
        fmt='.2f'
    )
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=15)
    ax.tick_params(axis='x', labelsize=10)
    ax.tick_params(axis='y', labelsize=10)
    ax.set_title(title, size=20, pad=15)

    if path is None:
        plt.show()
    else:
        plt.savefig(path)
        plt.close(fig)


def visualize_feature_importance(df_feature_importance, title, path=None):

    """
    Visualize feature importance in descending order

    Parameters
    ----------
    df_feature_importance: pandas.DataFrame of shape (n_features, n_splits)
        Dataframe of feature importance

    title: str
        Title of the plot

    path: path-like str or None
        Path of the output file or None (if path is None, plot is displayed with selected backend)
    """

    fig, ax = plt.subplots(figsize=(24, 20), dpi=100)
    ax.barh(
        range(len(df_feature_importance)),
        df_feature_importance['mean'],
        xerr=df_feature_importance['std'],
        ecolor='black',
        capsize=10,
        align='center',
    )
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.set_yticks(range(len(df_feature_importance)))
    ax.set_yticklabels([f'{k} ({v:.2f})' for k, v in df_feature_importance['mean'].to_dict().items()])
    ax.tick_params(axis='x', labelsize=15, pad=10)
    ax.tick_params(axis='y', labelsize=15, pad=10)
    ax.set_title(title, size=20, pad=15)
    plt.gca().invert_yaxis()

    if path is None:
        plt.show()
    else:
        plt.savefig(path, bbox_inches='tight')
        plt.close(fig)


def visualize_scores(df_scores, title, path=None):

    """
    Visualize scores of the models

    Parameters
    ----------
    df_scores: pandas.DataFrame of shape (n_splits, n_metrics)
        Dataframe with multiple scores and metrics

    title: str
        Title of the plot

    path: path-like str or None
        Path of the output file or None (if path is None, plot is displayed with selected backend)
    """

    # Create mean and std of scores for error bars
    df_scores = df_scores.T
    column_names = df_scores.columns.to_list()
    df_scores['mean'] = df_scores[column_names].mean(axis=1)
    df_scores['std'] = df_scores[column_names].std(axis=1).fillna(0)

    fig, ax = plt.subplots(figsize=(32, 8))
    ax.barh(
        y=np.arange(df_scores.shape[0]),
        width=df_scores['mean'],
        xerr=df_scores['std'],
        align='center',
        ecolor='black',
        capsize=10
    )
    ax.set_yticks(np.arange(df_scores.shape[0]))
    ax.set_yticklabels([
        f'{metric}\n{mean:.4f} (±{std:.4f})' for metric, mean, std in zip(
            df_scores.index,
            df_scores['mean'].values,
            df_scores['std'].values
        )
    ])
    ax.set_xlabel('')
    ax.tick_params(axis='x', labelsize=15, pad=10)
    ax.tick_params(axis='y', labelsize=15, pad=10)
    ax.set_title(title, size=20, pad=15)

    if path is None:
        plt.show()
    else:
        plt.savefig(path)
        plt.close(fig)


def visualize_roc_curves(roc_curves, title, path=None):

    """
    Visualize ROC curves of the model(s)

    Parameters
    ----------
    roc_curves: array-like of shape (n_models, 3)
        List of ROC curves (tuple of false positive rates, true positive rates and thresholds)

    title: str
        Title of the plot

    path: path-like str or None
        Path of the output file or None (if path is None, plot is displayed with selected backend)
    """

    true_positive_rates_interpolated = []
    aucs = []
    mean_false_positive_rate = np.linspace(0, 1, 100)

    fig, ax = plt.subplots(figsize=(16, 16))

    # Plot random guess curve
    ax.plot([0, 1], [0, 1], linestyle='--', lw=2.5, color='r', alpha=0.75)

    # Plot individual ROC curves of multiple models
    for fprs, tprs, _ in roc_curves:
        true_positive_rates_interpolated.append(np.interp(mean_false_positive_rate, fprs, tprs))
        true_positive_rates_interpolated[-1][0] = 0.0
        roc_auc = auc(fprs, tprs)
        aucs.append(roc_auc)
        ax.plot(fprs, tprs, lw=1, alpha=0.1)

    # Plot mean ROC curve of N models
    mean_true_positive_rate = np.mean(true_positive_rates_interpolated, axis=0)
    mean_true_positive_rate[-1] = 1.0
    mean_auc = auc(mean_false_positive_rate, mean_true_positive_rate)
    std_auc = np.std(aucs)
    ax.plot(mean_false_positive_rate, mean_true_positive_rate, color='b', label=f'Mean ROC Curve (AUC: {mean_auc:.4f} ±{std_auc:.4f})', lw=2.5, alpha=0.9)
    best_threshold_idx = np.argmax(mean_true_positive_rate - mean_false_positive_rate)
    ax.scatter(
        [mean_false_positive_rate[best_threshold_idx]], [mean_true_positive_rate[best_threshold_idx]],
        marker='o',
        color='r',
        s=100,
        label=f'Best Threshold\nSensitivity: {mean_true_positive_rate[best_threshold_idx]:.4f}\nSpecificity {mean_false_positive_rate[best_threshold_idx]:.4f}'
    )

    # Plot confidence interval of ROC curves
    std_tpr = np.std(true_positive_rates_interpolated, axis=0)
    tprs_upper = np.minimum(mean_true_positive_rate + std_tpr, 1)
    tprs_lower = np.maximum(mean_true_positive_rate - std_tpr, 0)
    ax.fill_between(mean_false_positive_rate, tprs_lower, tprs_upper, color='grey', alpha=0.2, label='±1 sigma')

    ax.set_xlabel('False Positive Rate', size=15, labelpad=12)
    ax.set_ylabel('True Positive Rate', size=15, labelpad=12)
    ax.tick_params(axis='x', labelsize=15)
    ax.tick_params(axis='y', labelsize=15)
    ax.set_xlim([-0.05, 1.05])
    ax.set_ylim([-0.05, 1.05])
    ax.set_title(title, size=20, pad=15)
    ax.legend(loc='lower right', prop={'size': 14})

    if path is None:
        plt.show()
    else:
        plt.savefig(path)
        plt.close(fig)


def visualize_pr_curves(pr_curves, title, path=None):

    """
    Visualize PR curves of the model(s)

    Parameters
    ----------
    pr_curves: array-like of shape (n_models, 3)
        List of PR curves (tuple of precision, recall and thresholds)

    title: str
        Title of the plot

    path: path-like str or None
        Path of the output file or None (if path is None, plot is displayed with selected backend)
    """

    precisions_interpolated = []
    aucs = []
    mean_recall = np.linspace(0, 1, 100)

    fig, ax = plt.subplots(figsize=(16, 16))

    # Plot individual PR curves of multiple models
    for precisions, recalls, _ in pr_curves:
        precisions_interpolated.append(np.interp(mean_recall, 1 - recalls, precisions)[::-1])
        precisions_interpolated[-1][0] = 0.0
        precisions_interpolated[-1][0] = 1 - precisions_interpolated[-1][0]
        pr_auc = auc(recalls, precisions)
        aucs.append(pr_auc)
        ax.plot(recalls, precisions, lw=1, alpha=0.1)

    # Plot mean PR curve of N models
    mean_precision = np.mean(precisions_interpolated, axis=0)
    mean_precision[-1] = 0
    mean_auc = auc(mean_recall, mean_precision)
    std_auc = np.std(aucs)
    ax.plot(mean_recall, mean_precision, color='b', label=f'Mean PR Curve (AUC: {mean_auc:.4f} ±{std_auc:.4f})', lw=2.5, alpha=0.9)

    f1_scores = 2 * mean_recall * mean_precision / (mean_recall + mean_precision)
    best_threshold_idx = np.argmax(f1_scores)
    ax.scatter(
        [mean_recall[best_threshold_idx]], [mean_precision[best_threshold_idx]],
        marker='o',
        color='r',
        s=100,
        label=f'Best Threshold\nRecall: {mean_recall[best_threshold_idx]:.4f}\nPrecision {mean_precision[best_threshold_idx]:.4f}'
    )

    # Plot confidence interval of PR curves
    std_tpr = np.std(precisions_interpolated, axis=0)
    tprs_upper = np.minimum(mean_precision + std_tpr, 1)
    tprs_lower = np.maximum(mean_precision - std_tpr, 0)
    ax.fill_between(mean_recall, tprs_lower, tprs_upper, color='grey', alpha=0.2, label='±1 sigma')

    ax.set_xlabel('Recall', size=15, labelpad=12)
    ax.set_ylabel('Precision', size=15, labelpad=12)
    ax.tick_params(axis='x', labelsize=15)
    ax.tick_params(axis='y', labelsize=15)
    ax.set_xlim([-0.05, 1.05])
    ax.set_ylim([-0.05, 1.05])
    ax.set_title(title, size=20, pad=15)
    ax.legend(loc='lower right', prop={'size': 14})

    if path is None:
        plt.show()
    else:
        plt.savefig(path)
        plt.close(fig)


def visualize_predictions(y_true, y_pred, title, path=None):

    """
    Visualize labels and predictions as histograms

    Parameters
    ----------
    y_true: numpy.ndarray of shape (n_samples)
        Ground-truth labels

    y_pred: numpy.ndarray of shape (n_samples)
        Predicted labels

    title: str
        Title of the plot

    path: path-like str or None
        Path of the output file or None (if path is None, plot is displayed with selected backend)
    """

    fig, ax = plt.subplots(figsize=(16, 6))
    ax.hist(y_true, 16, alpha=0.5, label=f'Labels - Mean: {np.mean(y_true):.4f}')
    ax.hist(y_pred, 16, alpha=0.5, label=f'Predictions - Mean: {np.mean(y_pred):.4f}')
    ax.tick_params(axis='x', labelsize=15)
    ax.tick_params(axis='y', labelsize=15)
    ax.set_title(title, size=20, pad=15)
    ax.legend(prop={'size': 14})

    if path is None:
        plt.show()
    else:
        plt.savefig(path)
        plt.close(fig)


def visualize_image(image_or_dicom, mask=None, path=None):

    """
    Visualize given image with or without its mask

    Parameters
    ----------
    image_or_dicom: numpy.ndarray of shape (height, width, channel) or pydicom.dataset.FileDataset
        Image array or DICOM

    mask: np.ndarray of shape (height, width) or None
        Mask array

    path: str or None
        Path of the output file or None (if path is None, plot is displayed with selected backend)
    """

    if isinstance(image_or_dicom, pydicom.dataset.FileDataset):
        image = image_or_dicom.pixel_array
    elif isinstance(image_or_dicom, np.ndarray):
        image = image_or_dicom.copy()
    else:
        raise TypeError(f'Invalid image type {type(image_or_dicom)}')

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(image, cmap='gray')
    if mask is not None:
        ax.imshow(mask, alpha=0.25)
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


def visualize_annotations(image_or_dicom, masks=None, bounding_boxes=None, labels=None, path=None):

    """
    Visualize given image with or without its masks, bounding boxes and labels

    Parameters
    ----------
    image_or_dicom: numpy.ndarray of shape (height, width, channel) or pydicom.dataset.FileDataset
        Image array or DICOM

    masks: numpy.ndarray of shape (n_annotations, height, width) or None
        Array of masks

    bounding_boxes: numpy.ndarray of shape (n_annotations, 4) or None
        Array of bounding boxes

    labels: numpy.ndarray of shape (n_annotations) or None
        Array of labels

    path: path-like str or None
        Path of the output file or None (if path is None, plot is displayed with selected backend)
    """

    if isinstance(image_or_dicom, pydicom.dataset.FileDataset):
        image = image_or_dicom.pixel_array
    elif isinstance(image_or_dicom, np.ndarray):
        image = image_or_dicom.copy()
    else:
        raise TypeError(f'Invalid image type {type(image_or_dicom)}')

    if bounding_boxes is not None and labels is not None:
        for bounding_box, label in zip(bounding_boxes, labels):
            # Draw bounding box and its label to the image
            image = cv2.rectangle(image, (bounding_box[0], bounding_box[1]), (bounding_box[2], bounding_box[3]), (36, 255, 12), 2)
            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
            image = cv2.rectangle(image, (bounding_box[0], bounding_box[1] - 20), (bounding_box[0] + w, bounding_box[1]), (36, 255, 12), -1)
            image = cv2.putText(image, label, (bounding_box[0], bounding_box[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(image, cmap='gray')

    if masks is not None:
        mask = annotation_utilities.binary_to_multi_object_mask(binary_masks=masks)
        ax.imshow(mask, alpha=0.25)

    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.tick_params(axis='x', labelsize=15, pad=10)
    ax.tick_params(axis='y', labelsize=15, pad=10)
    ax.set_title('Image and Annotations', size=20, pad=15)

    if path is None:
        plt.show()
    else:
        plt.savefig(path)
        plt.close(fig)


def visualize_images(images_or_dicoms, masks=None, path=None):

    """
    Visualize given images with or without its masks

    Parameters
    ----------
    images_or_dicoms: numpy.ndarray of shape (depth, height, width, channel) or list of pydicom.dataset.FileDataset
        Image array or list of DICOMs

    masks: np.ndarray of shape (depth, height, width) or None
        Mask array

    path: str or None
        Path of the output file or None (if path is None, plot is displayed with selected backend)
    """

    if isinstance(images_or_dicoms[0], pydicom.dataset.FileDataset):
        images = [dicom.pixel_array for dicom in images_or_dicoms]
    elif isinstance(images_or_dicoms, np.ndarray):
        images = images_or_dicoms.copy()
    else:
        raise TypeError(f'Invalid scan type {type(images_or_dicoms[0])}')

    fig, ax = plt.subplots(figsize=(8, 8))
    frames = []
    if masks is not None:
        for image, mask in zip(images, masks):
            frames.append([
                ax.imshow(image, cmap='gray', animated=True),
                ax.imshow(mask, alpha=0.25, animated=True)
            ])
    else:
        for image in images:
            frames.append([ax.imshow(image, cmap='gray', animated=True)])

    video = animation.ArtistAnimation(fig, frames, interval=50, blit=True, repeat_delay=1000)
    video.save(path)


def visualize_learning_curve(training_losses, validation_losses, best_epoch, validation_scores=None, path=None):

    """
    Visualize learning curves of the models

    Parameters
    ----------
    training_losses: list of shape (n_epochs)
        List of training losses

    validation_losses: list of shape (n_epochs)
        List of validation losses

    best_epoch: int or None
        Epoch with the best validation loss

    validation_scores: list of shape (n_scores, n_epochs)
        List of multiple validation scores

    path: str or None
        Path of the output file (if path is None, plot is displayed with selected backend)
    """

    if validation_scores is not None:

        fig, axes = plt.subplots(figsize=(18, 18), nrows=2, dpi=100)
        axes[0].plot(np.arange(1, len(training_losses) + 1), training_losses, '-o', linewidth=2, label=f'training_loss (best: {training_losses[best_epoch]:.4f})')
        axes[0].plot(np.arange(1, len(validation_losses) + 1), validation_losses, '-o', linewidth=2, label=f'validation_loss (best: {validation_losses[best_epoch]:.4f})')
        axes[0].axvline(best_epoch + 1, color='r', label=f'Best Epoch: {best_epoch + 1}')

        if validation_scores is not None:
            for metric, scores in validation_scores.items():
                axes[1].plot(np.arange(1, len(scores) + 1), scores, '-o', linewidth=2, label=f'{metric} (best: {scores[best_epoch]:.4f})')

            axes[1].axvline(best_epoch + 1, color='r', label=f'Best Epoch: {best_epoch + 1}')

        for i in range(2):
            axes[i].set_xlabel('Epochs/Steps', size=15, labelpad=12.5)
            axes[i].set_ylabel('Losses/Metrics', size=15, labelpad=12.5)
            axes[i].set_xticks(np.arange(1, len(validation_losses) + 1), np.arange(1, len(validation_losses) + 1))

            axes[i].tick_params(axis='x', labelsize=12.5, pad=10)
            axes[i].tick_params(axis='y', labelsize=12.5, pad=10)
            axes[i].legend(prop={'size': 18})

        axes[0].set_title('Learning Curve (Losses)', size=20, pad=15)
        axes[0].set_title('Learning Curve (Metrics)', size=20, pad=15)

    else:
        fig, ax = plt.subplots(figsize=(18, 8), dpi=100)
        ax.plot(np.arange(1, len(training_losses) + 1), training_losses, '-o', linewidth=2, label=f'training_loss (best: {training_losses[best_epoch]:.4f})')
        ax.plot(np.arange(1, len(validation_losses) + 1), validation_losses, '-o', linewidth=2, label=f'validation_loss (best: {validation_losses[best_epoch]:.4f})')
        ax.axvline(best_epoch + 1, color='r', label=f'Best Epoch: {best_epoch + 1}')

        ax.set_xlabel('Epochs/Steps', size=15, labelpad=12.5)
        ax.set_ylabel('Losses/Metrics', size=15, labelpad=12.5)
        ax.set_xticks(np.arange(1, len(validation_losses) + 1), np.arange(1, len(validation_losses) + 1))

        ax.tick_params(axis='x', labelsize=12.5, pad=10)
        ax.tick_params(axis='y', labelsize=12.5, pad=10)
        ax.legend(prop={'size': 18})
        ax.set_title('Learning Curve (Losses)', size=20, pad=15)

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

    visualize_correlations(
        df=df_train,
        columns=target_columns,
        title='Target Columns Correlations',
        path=settings.EDA / 'target_columns_correlations.png'
    )

    dataset_2d = '2d_bit_shifted_rescaled_spacing_normalized'
    df_train_metadata = pd.read_parquet(settings.DATA / 'datasets' / dataset_2d / 'metadata.parquet')
    settings.logger.info(f'Train Metadata Dataset Shape: {df_train_metadata.shape} - Memory Usage: {df_train_metadata.memory_usage().sum() / 1024 ** 2:.2f} MB')

    # Visualize binary and multi-class target columns
    metadata_columns = ['image_height', 'image_width', 'image_mean', 'image_std']
    for column in metadata_columns:
        visualize_continuous_feature_distribution(
            df=df_train_metadata,
            feature=column,
            title=f'{column} Histogram',
            path=settings.EDA / f'metadata_{column}_histogram.png'
        )

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

    df_patient_scan_counts = pd.DataFrame(df_dicom_tags.groupby('patient_id')['scan_id'].nunique())
    visualize_categorical_feature_distribution(
        df=df_patient_scan_counts,
        feature='scan_id',
        title=f'Patient Scan Counts',
        path=settings.EDA / f'dicom_tags_patient_scan_counts.png'
    )

    # Create slices feature and visualize counts of scan dimensions
    df_dicom_tags['slices'] = df_dicom_tags.groupby(['patient_id', 'scan_id'])['scan_id'].transform('count')
    visualize_continuous_feature_distribution(
        df=df_dicom_tags.groupby('scan_id')[['slices']].first(),
        feature='slices',
        title='Scan Slices Histogram',
        path=settings.EDA / f'dicom_tags_scan_slices_histogram.png'
    )
    visualize_continuous_feature_distribution(
        df=df_dicom_tags.groupby('scan_id')[['Rows']].first(),
        feature='Rows',
        title='Scan Heights Histogram',
        path=settings.EDA / f'dicom_tags_scan_heights_histogram.png'
    )
    visualize_continuous_feature_distribution(
        df=df_dicom_tags.groupby('scan_id')[['Columns']].first(),
        feature='Columns',
        title='Scan Widths Histogram',
        path=settings.EDA / f'dicom_tags_scan_widths_histogram.png'
    )

    # Visualize counts of data types of the scans
    for feature in ['BitsAllocated', 'BitsStored', 'HighBit']:
        visualize_categorical_feature_distribution(
            df=df_dicom_tags,
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
            df=df_dicom_tags,
            feature=feature,
            title=f'{feature} Value Counts',
            path=settings.EDA / f'dicom_tags_{feature}_value_counts.png'
        )

    # Visualize counts of geometric properties
    for feature in [
        'PatientPosition'
    ]:
        visualize_categorical_feature_distribution(
            df=df_dicom_tags,
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
    df_scan_slice_counts = df_dicom_tags.groupby('scan_id')['slices'].first().reset_index().rename(columns={'scan_id': 'series_id'})
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
