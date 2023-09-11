import sys
from itertools import groupby
import numpy as np
import torch

sys.path.append('..')
import settings

sys.path.append('../../venv/lib/python3.11/site-packages/CLIP-Driven-Universal-Model')
from model.Universal_model import Universal_model
from utils.utils import (
    organ_post_process, extract_topk_largest_candidates, PSVein_post_process,
    lung_post_process, merge_and_top_organ, organ_region_filter_out, threshold_organ,
    TUMOR_ORGAN, ORGAN_NAME, TEMPLATE
)


CLASS_NAMES = [
    'Spleen', 'Right Kidney', 'Left Kidney', 'Gall Bladder', 'Esophagus',
    'Liver', 'Stomach', 'Aorta', 'Postcava', 'Portal Vein and Splenic Vein',
    'Pancreas', 'Right Adrenal Gland', 'Left Adrenal Gland', 'Duodenum', 'Hepatic Vessel',
    'Right Lung', 'Left Lung', 'Colon', 'Intestine', 'Rectum',
    'Bladder', 'Prostate', 'Left Head of Femur', 'Right Head of Femur', 'Celiac Truck',
    'Kidney Tumor', 'Liver Tumor', 'Pancreas Tumor', 'Hepatic Vessel Tumor', 'Lung Tumor', 'Colon Tumor', 'Kidney Cyst'
]


def load_clip_driven_universal_model(model_directory, model_file_name, model_kwargs, device):

    """
    Load CLIP-Driven Universal Model weights from the given directory and model file name

    Parameters
    ----------
    model_directory: str
        Name of the model directory

    model_file_name: str
        Name of the model weights file

    model_kwargs: dict
        Model keyword arguments

    device: torch.device
        Location of the model

    Returns
    -------
    model: torch.nn.Module
        CLIP-Driven Universal Model
    """

    model = Universal_model(**model_kwargs)

    state_dict = torch.load(settings.MODELS / model_directory / model_file_name)['net']
    for key in list(state_dict.keys()):
        if key in [
            'module.organ_embedding',
            'module.precls_conv.0.weight', 'module.precls_conv.0.bias', 'module.precls_conv.2.weight', 'module.precls_conv.2.bias',
            'module.GAP.0.weight', 'module.GAP.0.bias', 'module.GAP.3.weight', 'module.GAP.3.bias',
            'module.controller.weight', 'module.controller.bias',
            'module.text_to_vision.weight', 'module.text_to_vision.bias'
        ]:
            state_dict['.'.join(key.split('.')[1:])] = state_dict.pop(key)
        else:
            state_dict['backbone.' + '.'.join(key.split('.')[1:])] = state_dict.pop(key)

    model.load_state_dict(state_dict=state_dict)
    model.to(torch.device(device))
    model.eval()

    return model


def predict_clip_driven_universal_model(inputs, model, device, amp=False, threshold=False):

    """
    Predict given inputs with given CLIP-Driven Universal Model

    Parameters
    ----------
    inputs: torch.Tensor of shape (batch, channel, depth, height, width)
        Inputs tensor

    model: torch.nn.Module
        CLIP-Driven Universal Model

    device: torch.device
        Location of the model and inputs

    amp: bool
        Whether to use auto mixed precision or not

    threshold: bool
        Whether to convert soft predictions into hard labels with thresholding or not

    Returns
    -------
    outputs: numpy.ndarray of shape (channel, depth, height, width)
        Array of predictions
    """

    inputs = inputs.to(device)

    with torch.no_grad():
        if amp:
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                outputs = model(inputs.half()).float()
        else:
            outputs = model(inputs)

    outputs = torch.sigmoid(outputs)
    if threshold:
        outputs = threshold_organ(outputs)
    outputs = torch.squeeze(outputs.cpu(), dim=0).numpy()

    return outputs


def post_process_predictions(predictions, threshold):

    """
    Post process given predictions

    Parameters
    ----------
    predictions: numpy.ndarray of shape (channel, depth, height, width)
        Array of predictions

    threshold: float
        Threshold for converting soft predictions into hard labels

    Returns
    -------
    outputs: numpy.ndarray of shape (channel, depth, height, width)
        Array of processed predictions
    """

    predictions = (predictions >= threshold).astype(bool)

    spleen_predictions = predictions[0]
    kidney_predictions = np.any(predictions[[1, 2]] >= 0.5, axis=0)
    liver_predictions = predictions[5]
    bowel_predictions = np.any(predictions[[17, 18, 19]] >= 0.5, axis=0)

    kidney_tumor_predictions = np.any(predictions[[25, 31]] >= 0.5, axis=0)
    liver_tumor_predictions = predictions[26]
    colon_tumor_predictions = predictions[30]

    predictions = np.stack([
        spleen_predictions, kidney_predictions, liver_predictions, bowel_predictions,
        kidney_tumor_predictions, liver_tumor_predictions, colon_tumor_predictions
    ], axis=0)

    return predictions


def find_rois(predictions):

    """
    Find ROIs on given predictions

    Parameters
    ----------
    predictions: numpy.ndarray of shape (channel, depth, height, width)
        Array of predictions

    Returns
    -------
    rois: dict
        Dictionary of normalized ROI coordinates
    """

    spatial_dimensions = predictions.shape[1:]
    classes = [
        'spleen', 'kidney', 'liver', 'bowel',
        'kidney_tumor', 'liver_tumor', 'colon_tumor'
    ]

    rois = {}

    for i, c in enumerate(classes):

        class_predictions_axial_sum = predictions[i].sum(axis=(1, 2))
        class_predictions_coronal_sum = predictions[i].sum(axis=(0, 2))
        class_predictions_sagittal_sum = predictions[i].sum(axis=(0, 1))

        axial_non_zero_sequences = groupby(class_predictions_axial_sum, key=lambda x: x > 0.0)
        coronal_non_zero_sequences = groupby(class_predictions_coronal_sum, key=lambda x: x > 0.0)
        sagittal_non_zero_sequences = groupby(class_predictions_sagittal_sum, key=lambda x: x > 0.0)

        try:
            axial_longest_non_zero_sequence = np.array(max([list(sequence) for gt_zero, sequence in axial_non_zero_sequences if gt_zero], key=len))
            axial_longest_non_zero_sequence_idx = np.where(np.isin(class_predictions_axial_sum, axial_longest_non_zero_sequence))[0]
            axial_roi_start = float(axial_longest_non_zero_sequence_idx.min() / spatial_dimensions[0])
            axial_roi_end = float(axial_longest_non_zero_sequence_idx.max() / spatial_dimensions[0])
            axial_roi_area = int(axial_longest_non_zero_sequence_idx.shape[0] / spatial_dimensions[0])
        except ValueError:
            axial_roi_start = None
            axial_roi_end = None
            axial_roi_area = None

        try:
            coronal_longest_non_zero_sequence = np.array(max([list(sequence) for gt_zero, sequence in coronal_non_zero_sequences if gt_zero], key=len))
            coronal_longest_non_zero_sequence_idx = np.where(np.isin(class_predictions_coronal_sum, coronal_longest_non_zero_sequence))[0]
            coronal_roi_start = float(coronal_longest_non_zero_sequence_idx.min() / spatial_dimensions[0])
            coronal_roi_end = float(coronal_longest_non_zero_sequence_idx.max() / spatial_dimensions[0])
            coronal_roi_area = int(coronal_longest_non_zero_sequence_idx.shape[0] / spatial_dimensions[0])
        except ValueError:
            coronal_roi_start = None
            coronal_roi_end = None
            coronal_roi_area = None

        try:
            sagittal_longest_non_zero_sequence = np.array(max([list(sequence) for gt_zero, sequence in sagittal_non_zero_sequences if gt_zero], key=len))
            sagittal_longest_non_zero_sequence_idx = np.where(np.isin(class_predictions_sagittal_sum, sagittal_longest_non_zero_sequence))[0]
            sagittal_roi_start = float(sagittal_longest_non_zero_sequence_idx.min() / spatial_dimensions[0])
            sagittal_roi_end = float(sagittal_longest_non_zero_sequence_idx.max() / spatial_dimensions[0])
            sagittal_roi_area = int(sagittal_longest_non_zero_sequence_idx.shape[0] / spatial_dimensions[0])
        except ValueError:
            sagittal_roi_start = None
            sagittal_roi_end = None
            sagittal_roi_area = None

        rois[c] = {
            'axial_roi_start': axial_roi_start,
            'axial_roi_end': axial_roi_end,
            'axial_roi_area': axial_roi_area,
            'coronal_roi_start': coronal_roi_start,
            'coronal_roi_end': coronal_roi_end,
            'coronal_roi_area': coronal_roi_area,
            'sagittal_roi_start': sagittal_roi_start,
            'sagittal_roi_end': sagittal_roi_end,
            'sagittal_roi_area': sagittal_roi_area
        }

    return rois
