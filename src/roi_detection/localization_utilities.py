import numpy as np


def predict_yolov8_model(image, model):

    """
    Predict given image with given YOLOv8 model

    Parameters
    ----------
    image: numpy.ndarray of shape (height, width, channel)
        Image array

    model: torch.nn.Module
        YOLOv8 model

    Returns
    -------
    bounding_boxes: numpy.ndarray of shape (n_detections, 4)
        Bounding boxes with x1, y1, x2, y2 values

    scores: numpy.ndarray of shape (n_detections, 1)
        Bounding box confidence scores

    labels: numpy.ndarray of shape (n_detections, 1)
        Bounding box labels
    """

    class_mapping = {
        0: 'abdominal',
        1: 'upper',
        2: 'lower'
    }

    if image.shape[2] == 1:
        # Concatenate grayscale images on channel axis
        inputs = np.concatenate([image] * 3, axis=-1)
    else:
        inputs = image.copy()

    outputs = model(inputs, verbose=False)
    outputs = outputs[0].boxes.data.cpu().numpy()

    bounding_boxes = outputs[:, :4]
    scores = outputs[:, 4]
    labels = np.array([class_mapping[int(label)] for label in outputs[:, 5]])

    return bounding_boxes, scores, labels


def crop_image(image, roi, roi_format='voc'):

    """
    Predict given image with given YOLOv8 model

    Parameters
    ----------
    image: numpy.ndarray of shape (height, width, channel)
        Image array

    roi: numpy.ndarray of shape (4)
        Bounding box in VOC, COCO or YOLO format

    roi_format: str
        Format of the bounding box

    Returns
    -------
    image: numpy.ndarray of shape (cropped_height, cropped_width, channel)
        Cropped image array
    """

    if roi_format == 'voc':
        roi_x_start, roi_y_start, roi_x_end, roi_y_end = roi
    elif roi_format == 'coco':
        roi_x_start, roi_y_start, roi_width, roi_height = roi
        roi_x_end = roi_x_start + roi_width
        roi_y_end = roi_y_start + roi_height
    elif roi_format == 'yolo':
        roi_x_center, roi_y_center, roi_width, roi_height = roi
        roi_x_start = roi_x_center - (roi_width // 2)
        roi_y_start = roi_y_center - (roi_height // 2)
        roi_x_end = roi_x_center + (roi_width // 2)
        roi_y_end = roi_y_center + (roi_height // 2)
    else:
        raise ValueError(f'Invalid roi_format {roi_format}')

    image = image[
        int(roi_y_start):int(roi_y_end),
        int(roi_x_start):int(roi_x_end),
        :
    ]

    return image
