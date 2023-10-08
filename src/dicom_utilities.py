import numpy as np
import cv2
import pydicom


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
        image = np.clip(image.copy(), image_min, image_max)

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
        window_centers=None, window_widths=None,
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

    window_centers: list of int, str ('dataset') or None
        List of window center values for windowing pixel values

    window_widths: list of int, str ('dataset') or None
        List of window width values for windowing pixel values

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

    image = np.stack([
        window_pixel_values(image=np.copy(image), dicom=dicom, window_center=window_center, window_width=window_width)
        for window_center, window_width in zip(window_centers, window_widths)
    ], axis=-1)

    image_min = image.min(axis=(0, 1))
    image_max = image.max(axis=(0, 1))
    image = (image - image_min) / (image_max - image_min + 1e-6)
    image = invert_pixel_values(image=image, dicom=dicom, photometric_interpretation=photometric_interpretation, max_pixel_value=max_pixel_value)
    image = (image * 255.0).astype(np.uint8)

    return image


def adjust_pixel_spacing(image, dicom, current_pixel_spacing=None, new_pixel_spacing=(1.0, 1.0)):

    """
    Adjust pixel values by shifting bits, windowing, rescaling and inverting

    Parameters
    ----------
    image: numpy.ndarray of shape (height, width)
        Image array

    dicom: pydicom.dataset.FileDataset
        DICOM dataset

    current_pixel_spacing: tuple, str ('dataset') or None
        Physical distance in the patient between the center of each pixel

    new_pixel_spacing: tuple
        Desired pixel spacing after resize operation

    Returns
    -------
    image: numpy.ndarray of shape (height, width)
        Image array with adjusted pixel spacing
    """

    if current_pixel_spacing == 'dataset':
        try:
            current_pixel_spacing = dicom.PixelSpacing
        except AttributeError:
            current_pixel_spacing = None

    if current_pixel_spacing is not None:
        resize_factor = np.array(current_pixel_spacing) / np.array(new_pixel_spacing)
        rounded_shape = np.round(image.shape[:2] * resize_factor)
        resize_factor = rounded_shape / image.shape[:2]
        image = cv2.resize(image, dsize=None, fx=resize_factor[1], fy=resize_factor[0], interpolation=cv2.INTER_NEAREST)

    return image


def get_largest_contour(image):

    """
    Get the largest contour from the image

    Parameters
    ----------
    image: numpy.ndarray of shape (height, width)
        Image array

    Returns
    -------
    bounding_box: list of shape (4)
        Bounding box with x1, y1, x2, y2 values
    """

    thresholded_image = cv2.threshold(image, 20, 255, cv2.THRESH_BINARY)[1]
    contours, _ = cv2.findContours(thresholded_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    if len(contours) == 0:
        x1 = 0
        x2 = image.shape[1] + 1
        y1 = 0
        y2 = image.shape[0] + 1
    else:
        contour = max(contours, key=cv2.contourArea)
        mask = np.zeros(image.shape, np.uint8)
        cv2.drawContours(mask, [contour], -1, 255, cv2.FILLED)

        y1, y2 = np.min(contour[:, :, 1]), np.max(contour[:, :, 1])
        x1, x2 = np.min(contour[:, :, 0]), np.max(contour[:, :, 0])

        x1 = int(0.99 * x1)
        x2 = int(1.01 * x2)
        y1 = int(0.99 * y1)
        y2 = int(1.01 * y2)

    bounding_box = [x1, y1, x2, y2]

    return bounding_box
