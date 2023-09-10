import sys
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
        Name of the model directory

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
