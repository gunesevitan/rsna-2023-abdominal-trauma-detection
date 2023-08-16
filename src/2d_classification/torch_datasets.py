import cv2
import torch
from torch.utils.data import Dataset


class ClassificationDataset(Dataset):

    def __init__(
            self,
            image_paths,
            bowel_targets=None, extravasation_targets=None, kidney_targets=None, liver_targets=None, spleen_targets=None,
            transforms=None
    ):

        self.image_paths = image_paths
        self.bowel_targets = bowel_targets
        self.extravasation_targets = extravasation_targets
        self.kidney_targets = kidney_targets
        self.liver_targets = liver_targets
        self.spleen_targets = spleen_targets
        self.transforms = transforms

    def __len__(self):

        """
        Get the length the dataset

        Returns
        -------
        length: int
            Length of the dataset
        """

        return len(self.image_paths)

    def __getitem__(self, idx):

        """
        Get the idxth element in the dataset

        Parameters
        ----------
        idx: int
            Index of the sample (0 <= idx < length of the dataset)

        Returns
        -------
        image: torch.FloatTensor of shape (channel, height, width) or numpy.ndarray of shape (height, width, channel)
            Image tensor or array

        target: torch
        """

        image = cv2.imread(self.image_paths[idx], -1)

        if (
                self.bowel_targets is not None and
                self.extravasation_targets is not None and
                self.kidney_targets is not None and
                self.liver_targets is not None and
                self.spleen_targets is not None
        ):

            bowel_target = self.bowel_targets[idx]
            extravasation_target = self.extravasation_targets[idx]
            kidney_target = self.kidney_targets[idx]
            liver_target = self.liver_targets[idx]
            spleen_target = self.spleen_targets[idx]

            if self.transforms is not None:
                image = self.transforms(image=image)['image'].float()
            else:
                image = torch.as_tensor(image, dtype=torch.float)
                image /= 255.

            bowel_target = torch.as_tensor(bowel_target, dtype=torch.float)
            extravasation_target = torch.as_tensor(extravasation_target, dtype=torch.float)
            kidney_target = torch.as_tensor(kidney_target, dtype=torch.long)
            liver_target = torch.as_tensor(liver_target, dtype=torch.long)
            spleen_target = torch.as_tensor(spleen_target, dtype=torch.long)

            return image, bowel_target, extravasation_target, kidney_target, liver_target, spleen_target

        else:

            if self.transforms is not None:
                image = self.transforms(image=image)['image'].float()
            else:
                image = torch.as_tensor(image, dtype=torch.float)
                image /= 255.

            return image


def prepare_classification_data(df):

    """
    Prepare data for classification dataset

    Parameters
    ----------
    df: pandas.DataFrame
        Dataframe with patient_id, scan_id, image_path and target columns

    Returns
    -------
    image_paths: numpy.ndarray of shape (n_samples)
        Array of image paths

    targets: dict of numpy.ndarray of shape (n_samples)
        Dictionary of array of targets
    """

    # Select middle slices
    df['slice_count'] = df.groupby(['patient_id', 'scan_id'])['scan_id'].transform('count')
    df['slice_cumcount'] = df.groupby(['patient_id', 'scan_id'])['scan_id'].transform('cumcount') + 1
    df['middle_slice_idx'] = df['slice_count'] // 2
    df = df.loc[df['slice_cumcount'] == df['middle_slice_idx']].reset_index(drop=True).drop(columns=['slice_count', 'slice_cumcount', 'middle_slice_idx'])

    # Convert one-hot encoded target columns to a single multi-class target columns
    for multiclass_target in ['kidney', 'liver', 'spleen']:
        df[multiclass_target] = 0
        for label, column in enumerate([f'{multiclass_target}_healthy', f'{multiclass_target}_low', f'{multiclass_target}_high']):
            df.loc[df[column] == 1, multiclass_target] = label

    image_paths = df['image_path'].values
    bowel_targets = df['bowel_injury'].values
    extravasation_targets = df['extravasation_injury'].values
    kidney_targets = df['kidney'].values
    liver_targets = df['liver'].values
    spleen_targets = df['spleen'].values
    targets = {
        'bowel_targets': bowel_targets,
        'extravasation_targets': extravasation_targets,
        'kidney_targets': kidney_targets,
        'liver_targets': liver_targets,
        'spleen_targets': spleen_targets
    }

    return image_paths, targets
