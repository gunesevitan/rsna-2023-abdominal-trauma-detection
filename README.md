# RSNA 2023 Abdominal Trauma Detection

## Overview

I used an efficient preprocessing pipeline and small multi-task models in a single stage framework. I didn't use image level labels and segmentation masks because I forgot they were given 🤦‍♂️.

Kaggle Notebook: https://www.kaggle.com/code/gunesevitan/rsna-2023-abdominal-trauma-detection-inference
Kaggle Dataset: https://www.kaggle.com/datasets/gunesevitan/rsna-2023-abdominal-trauma-detection-dataset
GitHub Repository: https://github.com/gunesevitan/rsna-2023-abdominal-trauma-detection

## Dataset

### 2D Dataset

* Bit shift with DICOM's bits allocated and stored attributes
* Linear pixel value rescale with DICOM's rescale slope and intercept attributes
* Window with DICOM's window width and center attributes (abdominal soft tissue window; width 400, center 50)
* Adjust minimum pixel value to 0 and scale pixel values with the new maximum
* Invert pixel values if DICOM's photometric interpretation attribute is MONOCHROME1
* Multiply pixel values with 255 and cast image to uint8
* Write image in lossless png format with raw size

My 2D and 3D dataset pipelines are separated because this part can run very fast in parallel because of non-blocking IO. I can export all of the training DICOMs as pngs in approximately 20 minutes.

### 3D Dataset

I saved lots of different CT scans from training set as videos and examined them. I noticed each of their start and end points were different on the z dimension. Some of them were starting from the shoulders and ending just before the legs or some of them were starting from the lungs and ending somewhere around middle femur.

I studied the anatomy and decided to localize ROIs. I manually annotated bounding boxes around the largest contour on axial plane. I labeled slices before the liver as "upper" and slices after the femur head as "lower". Slices between those two location are labeled as "abdominal". I trained a YOLOv8 nano model and it was reaching to 0.99x mAP@50 on all those classes easily. I dropped slices that are predicted as "upper" and "lower", and I used slices that are predicted as "abdominal" and cropped them with the predicted bounding box.
![yolo](https://i.ibb.co/JpNsJD1/val-batch2-pred.jpg)

Eventually, I ditched this approach because it was too slow and it didn't improve my overall score at all. In my latest 3D pipeline, I was using a lightweight localization by simply cropping the largest contour on the axial plane and keep all slices on the z dimension.

* Read all images that are exported as pngs in a scan and stack them on the z-axis
* Sort z-axis in descending order by DICOMs' image position patient z attribute
* Flip x-axis if DICOMs' patient position attribute is HFS (head first supine)
* Drop partial slices (some slices at the beginning or end of the scan were partially black)

I dropped those slices by counting all black vertical lines and their differences on z-axis. Normal slices had 0-5 all black vertical lines. If all black vertical line count suddenly increases or decreases then that slice is partial.
```python
# Find partial slices by calculating sum of all zero vertical lines
if scan.shape[0] != 1:
    scan_all_zero_vertical_line_transitions = np.diff(np.all(scan == 0, axis=1).sum(axis=1))
    # Heuristically select high and low transitions on z-axis and drop them
    slices_with_all_zero_vertical_lines = (scan_all_zero_vertical_line_transitions > 5) | (scan_all_zero_vertical_line_transitions < -5)
    slices_with_all_zero_vertical_lines = np.append(slices_with_all_zero_vertical_lines, slices_with_all_zero_vertical_lines[-1])
    scan = scan[~slices_with_all_zero_vertical_lines]
    del scan_all_zero_vertical_line_transitions, slices_with_all_zero_vertical_lines
```

* Crop the largest contour on the axial plane

I didn't do that to each image separately because it would break the alignment of slices. I calculated bounding boxes for each slice and calculate the largest bounding box by taking minimum of starting points and maximum of ending points.

```python
# Crop the largest contour
largest_contour_bounding_boxes = np.array([dicom_utilities.get_largest_contour(image) for image in scan])
largest_contour_bounding_box = [
    int(largest_contour_bounding_boxes[:, 0].min()),
    int(largest_contour_bounding_boxes[:, 1].min()),
    int(largest_contour_bounding_boxes[:, 2].max()),
    int(largest_contour_bounding_boxes[:, 3].max()),
]
scan = scan[
    :,
    largest_contour_bounding_box[1]:largest_contour_bounding_box[3] + 1,
    largest_contour_bounding_box[0]:largest_contour_bounding_box[2] + 1,
]
```

* Crop non-zero slices along 3 planes
```python
# Crop non-zero slices along xz, yz and xy planes
mmin = np.array((scan > 0).nonzero()).min(axis=1)
mmax = np.array((scan > 0).nonzero()).max(axis=1)
scan = scan[
    mmin[0]:mmax[0] + 1,
    mmin[1]:mmax[1] + 1,
    mmin[2]:mmax[2] + 1
]
```
* Resize 3D volume into 96x256x256 with area interpolation
* Write image as a numpy array file

To conclude, those numpy arrays are used as model inputs. I wasn't able to benefit from parallel execution at this stage. 

## Validation

I used multi label stratified group kfold for cross-validation. Group functionality can be achieved by splitting at patient level. I converted one-hot encoded classes into ordinal encoded single columns and created another column for patient scan count. I split dataset into 5 folds and 5 ordinal encoded target columns + patient scan count column are used for stratification.

## Models

I tried lots of different models, heads and necks but two simple models were the best performing ones.

### MIL-like 2D multi-task classification model

This model is a very simple one that is similar to MIL approach and ironically this was my best performing model. The architecture is:
1. Extract features on 2D slices
2. Average or max pooling on z dimension
3. Average, max, gem or attention pooling on x and y dimension
4. Dropout
5. 5 classification heads for each target

### RNN 2D multi-task classification model

This model is similar to what others used in previous competitions. The architecture is:
1. Extract features on 2D slices
2. Average, max or gem pooling on x and y dimension
3. Bidirectional LSTM or GRU max while using z dimension as a sequence 
4. Dropout
5. 5 classification heads for each target

### Backbones, necks and heads

* I tried lots of backbones from timm and monai but my best backbones were EfficientNet b0, EfficientNet v2 tiny and DenseNet121. I think I wasn't able to make large models converge.
* I also tried lots of different pooling types including average, sum, logsumexp, max, gem, attention but average and attention worked best for the first model and max worked best for the second model.
* I only used 5 regular classification heads for 5 targets
  * n_features x 1 bowel head + sigmoid at inference time
  * n_features x 1 extravasation head + sigmoid at inference time
  * n_features x 3 kidney head + softmax at inference time
  * n_features x 3 liver head + softmax at inference time
  * n_features x 3 spleen head + softmax at inference time

## Training

I used BCEWithLogitsLoss for bowel and extravasation heads, CrossEntropyLoss for kidney, liver and spleen weights. The only modification I did was implementing exact same sample weights like this:

```python
class SampleWeightedBCEWithLogitsLoss(_WeightedLoss):

    def __init__(self, weight=None, reduction='mean'):

        super(SampleWeightedBCEWithLogitsLoss, self).__init__(weight=weight, reduction=reduction)

        self.weight = weight
        self.reduction = reduction

    def forward(self, inputs, targets, sample_weights):

        loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none', weight=self.weight)
        loss = loss * sample_weights

        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()

        return loss
```

```python
class SampleWeightedCrossEntropyLoss(_WeightedLoss):

    def __init__(self, weight=None, reduction='mean'):

        super(SampleWeightedCrossEntropyLoss, self).__init__(weight=weight, reduction=reduction)

        self.weight = weight
        self.reduction = reduction

    def forward(self, inputs, targets, sample_weights):

        loss = F.cross_entropy(inputs, targets, reduction='none', weight=self.weight)
        loss = loss * sample_weights

        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()

        return loss
```

Final loss is calculated as the sum of each heads' loss and backward is called on that.

Training transforms are:
* Scale by max 8 bit pixel value
* Random X, Y and Z flip that are independent of each other
* Random 90 degree rotation on axial plane
* Random 0-45 degree rotation on axial plane
* Histogram equalization or random contrast shift
* Random 224x224 crop on axial plane
* 3D cutout

Test transforms are:
* Scale by max 8 bit pixel value
* Center 224x224 crop on axial plane

```python
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
```

```yaml
transforms:
  random_z_flip_probability: 0.5
  random_x_flip_probability: 0.5
  random_y_flip_probability: 0.5
  random_axial_rotate_90_probability: 0.25
  random_rotate_range_x: 0.0
  random_rotate_range_y: 0.0
  random_rotate_range_z: 0.45
  random_rotate_probability: 0.1
  random_histogram_shift_num_control_points: 25
  random_histogram_shift_probability: 0.05
  random_contrast_gamma: [0.5, 2.5]
  random_contrast_probability: 0.05
  crop_roi_size: [-1, 224, 224]
  cutout_holes: 8
  cutout_spatial_size: [4, 8, 8]
  cutout_max_holes: 16
  max_spatial_size: [8, 16, 16]
  cutout_probability: 0.1
```

Batch size of 2 or 4 is used depending on the model size. Cosine annealing learning rate schedule is utilized to explore different regions with a small base and minimum learning rate. AMP is also used for faster training and regularization.

## Inference

2x MIL-like model (efficientnetb0 and densenet121) and 2x RNN model (efficientnetb0 and efficientnetv2t) are used on the final ensemble. 

Since the models were trained with random crop augmentation, inputs are center cropped at test time. 4x TTA (xyz, xy, xz and yz flip) are applied and predictions are averaged.

Predictions of 5 folds are averaged and then activated with sigmoid or softmax functions.

## Post-processing

Different weights are used for 4 models for different targets. Those weights are found by minimizing the OOF score.

```python
mil_efficientnetb0_bowel_weight = 0.45
mil_densenet121_bowel_weight = 0.25
lstm_efficientnetb0_bowel_weight = 0.15
lstm_efficientnetv2t_bowel_weight = 0.15

mil_efficientnetb0_extravasation_weight = 0.3
mil_densenet121_extravasation_weight = 0.3
lstm_efficientnetb0_extravasation_weight = 0.3
lstm_efficientnetv2t_extravasation_weight = 0.1

mil_efficientnetb0_kidney_weight = 0.25
mil_densenet121_kidney_weight = 0.25
lstm_efficientnetb0_kidney_weight = 0.25
lstm_efficientnetv2t_kidney_weight = 0.25

mil_efficientnetb0_liver_weight = 0.25
mil_densenet121_liver_weight = 0.25
lstm_efficientnetb0_liver_weight = 0.25
lstm_efficientnetv2t_liver_weight = 0.25

mil_efficientnetb0_spleen_weight = 0.25
mil_densenet121_spleen_weight = 0.25
lstm_efficientnetb0_spleen_weight = 0.25
lstm_efficientnetv2t_spleen_weight = 0.25
```

I aggregated scan level predictions on patient_id and took the maximum prediction.

I also scaled injury target predictions with different multipliers and they are also set by minimizing OOF score.

```python
df_predictions['bowel_injury_prediction'] *= 1.
df_predictions['extravasation_injury_prediction'] *= 1.4
df_predictions['kidney_low_prediction'] *= 1.1
df_predictions['kidney_high_prediction'] *= 1.1
df_predictions['liver_low_prediction'] *= 1.3
df_predictions['liver_high_prediction'] *= 1.3
df_predictions['spleen_low_prediction'] *= 1.75
df_predictions['spleen_high_prediction'] *= 1.75
```

My final ensemble score was **0.3859** and target scores are listed below. I really enjoyed how my OOF scores are almost perfectly correlated with LB scores. I selected the submission that had the best OOF, public and private LB scores thanks to stable cross-validation.

| bowel  | extravasation | kidney | liver  | spleen | any    |
|--------|---------------|--------|--------|--------|--------|
| 0.1282 | 0.5070        | 0.2831 | 0.4186 | 0.4736 | 0.5050 |
