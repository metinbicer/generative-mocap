# -*- coding: utf-8 -*-
"""

Author:   Metin Bicer
email:    m.bicer19@imperial.ac.uk

creates pytorch dataloader to load training data in mini batches
read data from a dataframe structured as in "data/data.pickle" using the given
arguments

"""
from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np
import utils as read_dataframes


class mocapDataset(Dataset):
    def __init__(self, inputs, labels_contd, labels_discr,
                 transform_inputs=None, transform_labels=None):
        """
        inherits from torch.utils.data.Dataset
        __getitem__ method should be implemented to fetch data in mini batches

        Args:
            inputs (np.ndarray): input tensor.
            labels_contd (np.ndarray): label tensor for the continous conditions.
            labels_discr (np.ndarray): label tensor for the discrete conditions.
            transform_inputs (torchvision.transforms): transformation to be
                                                       applied to the inputs.
            transform_labels (torchvision.transforms): transformation to be
                                                       applied to the labels.

        """
        # required data type conversions and saving attrs
        self.inputs = torch.from_numpy(inputs).float()
        self.labels_contd = labels_contd
        if self.labels_contd is not None:
            self.labels_contd = torch.from_numpy(labels_contd).float()
        self.labels_discr = labels_discr
        if self.labels_discr is not None:
            self.labels_discr = torch.from_numpy(labels_discr).float()
        self.transform_inputs = transform_inputs
        self.transform_labels = transform_labels

    def __getitem__(self, index):
        x = self.inputs[index]
        # transformation
        if self.transform_inputs:
            x = self.transform_inputs(x)
        if self.labels_contd is None:
            if self.labels_discr is None:
                return x, []
            # if discr isnt none
            return x, self.labels_discr[index]
        # contd isnt none
        y = self.labels_contd[index]
        # transformation
        if self.transform_labels:
            y = self.transform_labels(y)
        if self.labels_discr is not None:
            y = torch.cat([y, self.labels_discr[index]])
        return x, y

    def __len__(self):
        return len(self.inputs)


def mocapDataLoader(data_df_files, excluded_subjects, input_col,
                    label_col_contd, label_col_discr, batch_size, train,
                    transform_inputs=None, transform_labels=None):
    """
    reads data and creates pytorch dataloaders

    Args:
        data_df_files (str, list): paths to the files containing the dataframe (df).
        excluded_subjects (list): list of subject IDs to be excluded from
                                  training data.
        input_col (dict): contains 3 keys (value_cols, names_cols, included_names)
                          value_cols: column names of the df to create dataset
                          names_cols: column names containing all feature names
                                      in the value_cols
                          included_names: contains feature names, to be used
                                          in dataset, for each value_cols
        label_col_contd (list): continous conditions.
        label_col_discr (list): discrete conditions.
        batch_size (int): mini batch size.
        train (bool): if True shuffle the dataset, else no shuffling.
        transform_inputs (torchvision.transforms): transformation to be
                                                   applied to the inputs.
        transform_labels (torchvision.transforms): transformation to be
                                                   applied to the labels.

    Returns:
        dl (torch.utils.data.DataLoader): pytorch dataloader.

    """
    # if several quantities are required to bring together,
    # value_cols, names_cols, included_names can be given as lists

    # read the data files (json or pickle)
    df = read_dataframes(data_df_files)
    # if there is some subjects to be excluded
    if excluded_subjects is not None:
        if not isinstance(excluded_subjects, list):
            excluded_subjects = [excluded_subjects]
        # exclude
        df = df[~df.subject.isin(excluded_subjects)]
    # get input arrays
    inputs = [get_inputs_dataframe(df, value_col, names_col, included_name)
              for value_col, names_col, included_name
              in zip(input_col['value_cols'],
                     input_col['names_cols'],
                     input_col['included_names'])]
    # concatenate the inputs along the last dimension
    inputs = np.concatenate(inputs, -1)
    # do the same thing with the labels (labels.shape=#examples,#labels)
    # a label can only be a single value for a single example
    labels_contd = get_labels_dataframe(df, label_col_contd)
    labels_discr = get_labels_dataframe(df, label_col_discr)
    if batch_size == 'full':
        batch_size = inputs.shape[0]
    # pytorch dataset
    ds = mocapDataset(inputs, labels_contd, labels_discr,
                      transform_inputs, transform_labels)
    # return pytorch dataloader
    dl = DataLoader(ds, batch_size=batch_size, shuffle=train)
    return dl


def get_labels_dataframe(df, label_col):
    if label_col is None:
        return None
    # df.label_col.values.shape = #examples
    # if more than one label, return shape=#examples,#labels
    # else return it as shape=#examples,1
    return df[label_col].values

def get_inputs_dataframe(df, value_col, names_col=None, included_name=None):
    # given dataframe, get values contained in value_col
    # 1. shape=#examples, 3, #frames, #included_name (or #all_names_call)
    # this requires some shape formatting (for example for ik results)
    # 2. if names_col and included_name are given, then corresponding column
    # identifiers given in the included_name are taken
    # exception for markers
    if 'marker' in names_col:
        all_values = np.array([np.array(i)[:,:,np.where(np.isin(names, included_name))]
                               for i, names in zip(df[value_col], df[names_col])])
        return all_values[:,:,:,0,:]
    all_values = np.array([i for i in df[value_col].values])
    if len(all_values.shape) == 1:
        # a special case, for now is single values such as labels that can be
        # used in the original data (real data)
        # no need to have names_col or included_name
        # for now convert it to shape=#examples, 3, #frames, #included_name (or #all_names_call)
        # repeat in the first and second axes for 3 and 101 times, respectively
        return all_values[:,None,None,None].repeat(3,1).repeat(101,2)
    elif all_values.shape[1] == 1:
        # for ik or id results, the shape is #examples, 1, #frames, #included_name
        all_values = all_values.repeat(3,1)
    if names_col is None or included_name is None:
        return all_values
    # otherwise get the column identifiers
    all_names_cols = np.array(df[names_col].values[0])
    # get the col nos to just indexing of all_values for included_name
    cols = np.unique(np.where(np.in1d(all_names_cols, included_name))[0])
    return all_values[:, :, :, cols]
