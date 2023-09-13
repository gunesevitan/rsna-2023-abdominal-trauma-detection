import numpy as np
import torch
from torch.utils.data import Dataset


class ClassificationDataset(Dataset):

    def __init__(
            self,
            volume_paths,
            bowel_targets=None, extravasation_targets=None, kidney_targets=None, liver_targets=None, spleen_targets=None, any_targets=None,
            transforms=None
    ):

        self.volume_paths = volume_paths
        self.bowel_targets = bowel_targets
        self.extravasation_targets = extravasation_targets
        self.kidney_targets = kidney_targets
        self.liver_targets = liver_targets
        self.spleen_targets = spleen_targets
        self.any_targets = any_targets
        self.transforms = transforms

    def __len__(self):

        """
        Get the length the dataset

        Returns
        -------
        length: int
            Length of the dataset
        """

        return len(self.volume_paths)

    def __getitem__(self, idx):

        """
        Get the idxth element in the dataset

        Parameters
        ----------
        idx: int
            Index of the sample (0 <= idx < length of the dataset)

        Returns
        -------
        volume: torch.FloatTensor of shape (channel, depth, height, width)
            Image tensor

        targets: torch.Tensor
            Target tensor
        """

        volume = np.load(self.volume_paths[idx])
        volume = np.expand_dims(volume, axis=0)

        if (
            self.bowel_targets is not None and
            self.extravasation_targets is not None and
            self.kidney_targets is not None and
            self.liver_targets is not None and
            self.spleen_targets is not None and
            self.any_targets is not None
        ):

            bowel_target = self.bowel_targets[idx]
            extravasation_target = self.extravasation_targets[idx]
            kidney_target = self.kidney_targets[idx]
            liver_target = self.liver_targets[idx]
            spleen_target = self.spleen_targets[idx]
            any_target = self.any_targets[idx]

            if self.transforms is not None:
                image = self.transforms(volume)
            else:
                image = torch.as_tensor(volume, dtype=torch.float)

            image /= 255.

            bowel_target = torch.as_tensor(bowel_target, dtype=torch.float)
            extravasation_target = torch.as_tensor(extravasation_target, dtype=torch.float)
            kidney_target = torch.as_tensor(kidney_target, dtype=torch.long)
            liver_target = torch.as_tensor(liver_target, dtype=torch.long)
            spleen_target = torch.as_tensor(spleen_target, dtype=torch.long)
            any_target = torch.as_tensor(any_target, dtype=torch.float)

            return image, bowel_target, extravasation_target, kidney_target, liver_target, spleen_target, any_target

        else:

            if self.transforms is not None:
                image = self.transforms(volume)
            else:
                image = torch.as_tensor(volume, dtype=torch.float)

            image /= 255.

            return image


def prepare_classification_data(df):

    """
    Prepare data for classification dataset

    Parameters
    ----------
    df: pandas.DataFrame
        Dataframe with patient_id, scan_id, volume_path and target columns

    Returns
    -------
    volume_paths: numpy.ndarray of shape (n_samples)
        Array of volume paths

    targets: dict of numpy.ndarray of shape (n_samples)
        Dictionary of array of targets
    """

    volume_paths = df['volume_path'].values

    # Convert one-hot encoded target columns to a single multi-class target columns
    for multiclass_target in ['kidney', 'liver', 'spleen']:
        df[multiclass_target] = 0
        for label, column in enumerate([f'{multiclass_target}_healthy', f'{multiclass_target}_low', f'{multiclass_target}_high']):
            df.loc[df[column] == 1, multiclass_target] = label

    bowel_targets = df['bowel_injury'].values
    extravasation_targets = df['extravasation_injury'].values
    kidney_targets = df['kidney'].values
    liver_targets = df['liver'].values
    spleen_targets = df['spleen'].values
    any_targets = df['any_injury'].values
    targets = {
        'bowel_targets': bowel_targets,
        'extravasation_targets': extravasation_targets,
        'kidney_targets': kidney_targets,
        'liver_targets': liver_targets,
        'spleen_targets': spleen_targets,
        'any_targets': any_targets
    }

    return volume_paths, targets
