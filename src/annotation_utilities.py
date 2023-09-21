import numpy as np
import cv2


def binary_to_multi_object_mask(binary_masks):

    """
    Encode multiple 2d binary masks into a single 2d multi-object segmentation mask

    Parameters
    ----------
    binary_masks: numpy.ndarray of shape (n_objects, height, width)
        2d binary masks

    Returns
    -------
    multi_object_mask: numpy.ndarray of shape (height, width)
        2d multi-object mask
    """

    multi_object_mask = np.zeros((binary_masks.shape[1], binary_masks.shape[2]))
    for i, binary_mask in enumerate(binary_masks):
        non_zero_idx = binary_mask == 1
        multi_object_mask[non_zero_idx] = i + 1

    return multi_object_mask


def polygon_to_mask(polygon, shape):

    """
    Create binary segmentation mask from polygon

    Parameters
    ----------
    polygon: list of shape (n_polygons, n_points, 2)
        List of polygons

    shape: tuple of shape (2)
        Height and width of the mask

    Returns
    -------
    mask: numpy.ndarray of shape (height, width)
        2d segmentation mask
    """

    mask = np.zeros(shape)
    # Convert list of points to tuple pairs of X and Y coordinates
    points = np.array(polygon).reshape(-1, 2)
    # Draw mask from the polygon
    cv2.fillPoly(mask, [points], 1, lineType=cv2.LINE_8, shift=0)
    mask = np.array(mask).astype(np.uint8)

    return mask


def mask_to_bounding_box(mask):

    """
    Get bounding box from a binary segmentation mask

    Parameters
    ----------
    mask: numpy.ndarray of shape (height, width)
        2d binary mask

    Returns
    -------
    bounding_box: list of shape (4)
        Bounding box
    """

    non_zero_idx = np.where(mask == 1)
    bounding_box = [
        int(np.min(non_zero_idx[1])),
        int(np.min(non_zero_idx[0])),
        int(np.max(non_zero_idx[1])),
        int(np.max(non_zero_idx[0]))
    ]

    return bounding_box


def coco_to_voc_bounding_box(bounding_box):

    """
    Convert bounding box annotation from VOC to COCO format

    Parameters
    ----------
    bounding_box: list of shape (4)
        Bounding box with x1, y1, width, height values

    Returns
    -------
    bounding_box: list of shape (4)
        Bounding box with x1, y1, x2, y2 values
    """

    x1 = bounding_box[0]
    y1 = bounding_box[1]
    x2 = x1 + bounding_box[2]
    y2 = y1 + bounding_box[3]

    return x1, y1, x2, y2


def coco_to_yolo_bounding_box(bounding_box):

    """
    Convert bounding box annotation from COCO to YOLO format

    Parameters
    ----------
    bounding_box: list of shape (4)
        Bounding box with x1, y1, width, height values

    Returns
    -------
    bounding_box: list of shape (4)
        Bounding box with x_center, y_center, width, height values
    """

    x1 = bounding_box[0]
    y1 = bounding_box[1]
    width = bounding_box[2]
    height = bounding_box[3]
    x_center = x1 + (width // 2)
    y_center = y1 + (height // 2)

    return x_center, y_center, width, height
