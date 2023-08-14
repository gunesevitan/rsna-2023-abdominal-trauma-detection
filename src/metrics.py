import numpy as np
from sklearn.metrics import (
    log_loss, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    average_precision_score, confusion_matrix, roc_curve, precision_recall_curve
)


def round_probabilities(probabilities, threshold):

    """
    Round probabilities to labels based on the given threshold

    Parameters
    ----------
    probabilities : numpy.ndarray of shape (n_samples)
        Predicted probabilities

    threshold: float
        Rounding threshold

    Returns
    -------
    labels : numpy.ndarray of shape (n_samples)
        Rounded probabilities
    """

    labels = np.zeros_like(probabilities, dtype=np.uint8)
    labels[probabilities >= threshold] = 1

    return labels


def specificity_score(y_true, y_pred):

    """
    Calculate specificity score (true-negative rate) of predicted labels

    Parameters
    ----------
    y_true: numpy.ndarray of shape (n_samples)
        Ground-truth labels

    y_pred: numpy.ndarray of shape (n_samples)
        Predicted labels

    Returns
    -------
    score: float
        Specificity score between 0 and 1
    """

    tn, fp, _, _ = confusion_matrix(y_true, y_pred).ravel()
    score = tn / (tn + fp)

    return score


def binary_classification_scores(y_true, y_pred, sample_weights, threshold=0.5):

    """
    Calculate binary classification metrics on predicted probabilities and labels

    Parameters
    ----------
    y_true: numpy.ndarray of shape (n_samples)
        Ground-truth labels

    y_pred: numpy.ndarray of shape (n_samples)
        Predicted probabilities

    sample_weights: numpy.ndarray of shape (n_samples)
        Sample weights

    threshold: float
        Rounding threshold

    Returns
    -------
    scores: dict
        Dictionary of classification scores
    """

    y_pred_labels = round_probabilities(y_pred, threshold=threshold)
    scores = {
        'log_loss': log_loss(y_true, y_pred),
        'sample_weighted_log_loss': log_loss(y_true, y_pred, sample_weight=sample_weights),
        'accuracy': accuracy_score(y_true, y_pred_labels),
        'precision': precision_score(y_true, y_pred_labels),
        'recall': recall_score(y_true, y_pred_labels),
        'specificity': specificity_score(y_true, y_pred_labels),
        'f1': f1_score(y_true, y_pred_labels),
        'roc_auc': roc_auc_score(y_true, y_pred),
        'average_precision': average_precision_score(y_true, y_pred)
    }

    return scores


def multiclass_classification_scores(y_true, y_pred, sample_weights):

    """
    Calculate multi-class classification metrics on predicted probabilities and labels

    Parameters
    ----------
    y_true: numpy.ndarray of shape (n_samples)
        Ground-truth labels

    y_pred: numpy.ndarray of shape (n_samples, n_class)
        Predicted probabilities

    sample_weights: numpy.ndarray of shape (n_samples)
        Sample weights

    Returns
    -------
    scores: dict
        Dictionary of classification scores
    """

    y_pred_labels = np.argmax(y_pred, axis=1)
    scores = {
        'log_loss': log_loss(y_true, y_pred),
        'sample_weighted_log_loss': log_loss(y_true, y_pred, sample_weight=sample_weights),
        'accuracy': accuracy_score(y_true, y_pred_labels),
        'precision': precision_score(y_true, y_pred_labels, average='macro'),
        'recall': recall_score(y_true, y_pred_labels, average='macro'),
        'f1': f1_score(y_true, y_pred_labels, average='macro')
    }

    return scores


def binary_classification_curves(y_true, y_pred):

    """
    Calculate binary classification curves on predicted probabilities

    Parameters
    ----------
    y_true: numpy.ndarray of shape (n_samples)
        Ground-truth labels

    y_pred: numpy.ndarray of shape (n_samples)
        Predicted probabilities

    Returns
    -------
    curves: dict
        Dictionary of classification curves
    """

    curves = {
        'roc': roc_curve(y_true, y_pred),
        'pr': precision_recall_curve(y_true, y_pred),
    }

    return curves


def create_sample_weights(df):

    """
    Create sample weights for each target column

    Parameters
    ----------
    df: pandas.DataFrame
        Dataframe with target columns

    Returns
    -------
    df: pandas.DataFrame
        Dataframe with sample weights created for each target column
    """

    df['bowel_injury_weight'] = np.where(df['bowel_injury'] == 1, 2, 1)
    df['extravasation_injury_weight'] = np.where(df['extravasation_injury'] == 1, 6, 1)
    df['kidney_weight'] = np.where(df['kidney_low'] == 1, 2, np.where(df['kidney_high'] == 1, 4, 1))
    df['liver_weight'] = np.where(df['liver_low'] == 1, 2, np.where(df['liver_high'] == 1, 4, 1))
    df['spleen_weight'] = np.where(df['spleen_low'] == 1, 2, np.where(df['spleen_high'] == 1, 4, 1))
    df['any_injury_weight'] = np.where(df['any_injury'] == 1, 6, 1)

    return df


def normalize_probabilities(df, columns):

    """
    Normalize probabilities to 1 within given columns

    Parameters
    ----------
    df: pandas.DataFrame
        Dataframe with given columns

    columns: list
        List of column names that have probabilities

    Returns
    -------
    df: pandas.DataFrame
        Dataframe with given columns' sum adjusted to 1
    """

    df_sums = df[columns].sum(axis=1)
    for column in columns:
        df[column] /= df_sums

    return df
