import monai.transforms as T
import torch


def get_clip_driven_universal_model_transforms(**transform_parameters):

    """
    Get transforms for CLIP-Driven Universal Model

    Parameters
    ----------
    transform_parameters: dict
        Dictionary of transform parameters

    Returns
    -------
    transforms: dict
        Transforms for inference
    """

    inference_transforms = T.Compose([
        T.EnsureChannelFirst(channel_dim='no_channel'),
        T.Orientation(axcodes='RAS'),
        T.Spacing(pixdim=(transform_parameters['spacing_pixdim']), mode='bilinear'),
        T.Resize(spatial_size=(96, 224, 224), size_mode='all', mode='nearest'),
        T.ToTensor(dtype=torch.float32, track_meta=False)
    ])

    transforms = {'inference': inference_transforms}
    return transforms
