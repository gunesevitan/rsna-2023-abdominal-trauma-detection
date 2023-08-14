import numpy as np


def shift_bits(image, dicom, bits_allocated=None, bits_stored=None):

    """
    Shift bits using allocated and stored bits

    Parameters
    ----------
    image: numpy.ndarray of shape (height, width)
        Image array

    dicom: pydicom.dataset.FileDataset
        DICOM dataset

    bits_allocated: int, str ('dataset') or None
        Number of bits allocated

    bits_stored: int, str ('dataset') or None
        Number of bits stored

    Returns
    -------
    image: numpy.ndarray of shape (height, width)
        Image array with shifted bits
    """

    if bits_allocated == 'dataset':
        try:
            bits_allocated = dicom.BitsAllocated
        except AttributeError:
            bits_allocated = None

    if bits_stored == 'dataset':
        try:
            bits_stored = dicom.BitsStored
        except AttributeError:
            bits_stored = None

    if bits_allocated is not None and bits_stored is not None:
        bit_shift = bits_allocated - bits_stored
    else:
        bit_shift = None

    if bit_shift is not None:
        dtype = image.dtype
        image = (image << bit_shift).astype(dtype) >> bit_shift

    return image


def rescale_pixel_values(image, dicom, rescale_slope=None, rescale_intercept=None):

    """
    Rescale pixel values using rescale slope and intercept as a linear function

    Parameters
    ----------
    image: numpy.ndarray of shape (height, width)
        Image array

    dicom: pydicom.dataset.FileDataset
        DICOM dataset

    rescale_slope: int, str ('dataset') or None
        Rescale slope for rescaling pixel values

    rescale_intercept: int, str ('dataset') or None
        Rescale intercept for rescaling pixel values

    Returns
    -------
    image: numpy.ndarray of shape (height, width)
        Image array with rescaled pixel values
    """

    if rescale_slope == 'dataset':
        try:
            rescale_slope = dicom.RescaleSlope
        except AttributeError:
            rescale_slope = None

    if rescale_intercept == 'dataset':
        try:
            rescale_intercept = dicom.RescaleIntercept
        except AttributeError:
            rescale_intercept = None

    if rescale_slope is not None and rescale_intercept is not None:
        image = image.astype(np.float32)
        image = image * rescale_slope + rescale_intercept

    return image


def window_pixel_values(image, dicom, window_center=None, window_width=None):

    """
    Window pixel values using window center and width

    Parameters
    ----------
    image: numpy.ndarray of shape (height, width)
        Image array

    dicom: pydicom.dataset.FileDataset
        DICOM dataset

    window_center: int, str ('dataset') or None
        Window center for windowing pixel values

    window_width: int, str ('dataset') or None
        Window width for windowing pixel values

    Returns
    -------
    image: numpy.ndarray of shape (height, width)
        Image array with windowed pixel values
    """

    if window_center == 'dataset':
        try:
            window_center = dicom.WindowCenter
        except AttributeError:
            window_center = None

    if window_width == 'dataset':
        try:
            window_width = dicom.WindowWidth
        except AttributeError:
            window_width = None

    if window_center is not None and window_width is not None:
        image_min = window_center - window_width // 2
        image_max = window_center + window_width // 2
        image[image < image_min] = image_min
        image[image > image_max] = image_max

    return image


def invert_pixel_values(image, dicom, photometric_interpretation=None, max_pixel_value=255):

    """
    Invert pixel values using given max pixel value

    Parameters
    ----------
    image: numpy.ndarray of shape (height, width)
        Image array

    dicom: pydicom.dataset.FileDataset
        DICOM dataset

    photometric_interpretation: str or None
        Interpretation of the pixel data

    max_pixel_value: int or None
        Max pixel value used for inverting pixel values

    Returns
    -------
    image: numpy.ndarray of shape (height, width)
        Image array with inverted pixel values
    """

    if photometric_interpretation == 'dataset':
        try:
            photometric_interpretation = dicom.PhotometricInterpretation
        except AttributeError:
            photometric_interpretation = None

    if photometric_interpretation == 'MONOCHROME1':
        image = max_pixel_value - image

    return image


def adjust_pixel_values(
        image, dicom,
        bits_allocated=None, bits_stored=None,
        rescale_slope=None, rescale_intercept=None,
        window_center=None, window_width=None,
        photometric_interpretation=None, max_pixel_value=255
):

    """
    Adjust pixel values by shifting bits, windowing, rescaling and inverting

    Parameters
    ----------
    image: numpy.ndarray of shape (height, width)
        Image array

    dicom: pydicom.dataset.FileDataset
        DICOM dataset

    bits_allocated: int, str ('dataset') or None
        Number of bits allocated

    bits_stored: int, str ('dataset') or None
        Number of bits stored

    rescale_slope: int, str ('dataset') or None
        Rescale slope for rescaling pixel values

    rescale_intercept: int, str ('dataset') or None
        Rescale intercept for rescaling pixel values

    window_center: int, str ('dataset') or None
        Window center for windowing pixel values

    window_width: int, str ('dataset') or None
        Window width for windowing pixel values

    photometric_interpretation: str or None
        Interpretation of the pixel data

    max_pixel_value: int or None
        Max pixel value used for inverting pixel values

    Returns
    -------
    image: numpy.ndarray of shape (height, width)
        Image array with adjusted pixel values
    """

    image = shift_bits(image=image, dicom=dicom, bits_allocated=bits_allocated, bits_stored=bits_stored)
    image = rescale_pixel_values(image=image, dicom=dicom, rescale_slope=rescale_slope, rescale_intercept=rescale_intercept)
    image = window_pixel_values(image=image, dicom=dicom, window_center=window_center, window_width=window_width)
    image = (image - image.min()) / (image.max() - image.min())
    image = invert_pixel_values(image=image, dicom=dicom, photometric_interpretation=photometric_interpretation, max_pixel_value=max_pixel_value)
    image = (image * 255.0).astype(np.uint8)

    return image