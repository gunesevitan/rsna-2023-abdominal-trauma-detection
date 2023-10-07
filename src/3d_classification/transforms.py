import cv2
import monai.transforms as T
import torch
from albumentations.pytorch.transforms import ToTensorV2


def get_classification_transforms(**transform_parameters):

    """
    Get transforms for classification dataset

    Parameters
    ----------
    transform_parameters: dict
        Dictionary of transform parameters

    Returns
    -------
    transforms: dict
        Transforms for training, validation and test sets
    """

    training_transforms = T.Compose([
        T.EnsureChannelFirst(channel_dim=0),
        T.RandFlip(spatial_axis=0, prob=transform_parameters['random_z_flip_probability']),
        T.RandFlip(spatial_axis=1, prob=transform_parameters['random_x_flip_probability']),
        T.RandFlip(spatial_axis=2, prob=transform_parameters['random_y_flip_probability']),
        T.RandRotate90(spatial_axes=(1, 2), max_k=3, prob=transform_parameters['random_axial_rotate_90_probability']),
        T.RandRotate(
            range_x=transform_parameters['random_rotate_range_x'],
            range_y=transform_parameters['random_rotate_range_y'],
            range_z=transform_parameters['random_rotate_range_z'],
            prob=transform_parameters['random_rotate_probability']
        ),
        T.OneOf([
            T.RandHistogramShift(num_control_points=transform_parameters['random_histogram_shift_num_control_points'], prob=transform_parameters['random_histogram_shift_probability']),
            T.RandAdjustContrast(gamma=transform_parameters['random_contrast_gamma'], prob=transform_parameters['random_contrast_probability'])
        ], weights=(0.5, 0.5)),
        T.RandSpatialCrop(roi_size=transform_parameters['crop_roi_size'], max_roi_size=None, random_center=True, random_size=False),
        T.RandCoarseDropout(
            holes=transform_parameters['cutout_holes'],
            spatial_size=transform_parameters['cutout_spatial_size'],
            dropout_holes=True,
            fill_value=0,
            max_holes=transform_parameters['cutout_max_holes'],
            max_spatial_size=transform_parameters['max_spatial_size'],
            prob=transform_parameters['cutout_probability']
        ),
        T.ToTensor(dtype=torch.float32, track_meta=False)
    ])

    inference_transforms = T.Compose([
        T.EnsureChannelFirst(channel_dim=0),
        T.CenterSpatialCrop(roi_size=transform_parameters['crop_roi_size']),
        T.ToTensor(dtype=torch.float32, track_meta=False)
    ])

    classification_transforms = {'training': training_transforms, 'inference': inference_transforms}
    return classification_transforms
