from data_augmentation import normalize_meanstd, img_shift, img_rotate
import numpy as np
import os
import pandas as pd


def get_data_from_np(fname, data_aug=False):
    """
    Loads the images from NumPy files and applies data augmentations when data_aug = True

    :param fname: file name of the numpy array
    :param data_aug: control whether data augmentation is applied. When True = data is augmented with shifts and
    rotations when false no data augmentation is applied.
    """
    img_inp = np.load(fname)

    if data_aug==True:
        shifted_img = img_shift(img_inp, 16)
        rotated_shifted_img = img_rotate(img_inp, 20)
        output_img = rotated_shifted_img
    else:
        output_img = img_inp

    normalize = normalize_meanstd(output_img)
    img = normalize.normalize(output_img)

    return img


def data_generator(pathdir, batchsize, mode, shuffle=True, train=True):
    """
    Data Generator function for training the network. Yield samples and labels from the data set on each call.

    :param pathdir: Location of all numpy files
    :param batchsize: Number of images to be returned on each call
    :param mode: Choose Systole or Diastole depending on model being trained, select to assign correct labels
    :param shuffle: When True randomly sample through the dataset in a unique order each epoch
    :param train: When True load training labels, When False load validation labels
    """
    fnames = os.listdir(pathdir)
    numFiles = len(fnames)
    x_list = list()
    y_list = list()
    assert numFiles > 0, "No files in the directory"

    if train:
        df_y_label = pd.read_csv('train.csv')
        df_y_label = df_y_label.set_index('Id')
    else:
        df_y_label = pd.read_csv('validate.csv')
        df_y_label = df_y_label.set_index('Id')

    while True:
        itr_count = 0
        if shuffle:
            randindx = np.random.permutation(numFiles)
        else:
            randindx = np.arange(numFiles)

        for k in randindx:
            if itr_count == 0:
                x_list = list()
                y_list = list()
            if train:
                img_inp = get_data_from_np('Training_arrays/systole/' + fnames[k], data_aug=True)
            else:
                img_inp = get_data_from_np('Validation_arrays/systole/' + fnames[k], data_aug=False)

            x_list.append(img_inp)
            patient_id = int(fnames[k].split('_')[-1].split('.')[0])
            y_list.append(int(df_y_label.loc[patient_id][mode]))
            itr_count = itr_count + 1
            if itr_count == batchsize:
                x = np.asarray(x_list)
                y = np.asarray(y_list)
                itr_count = 0
                x = x.reshape(x.shape[0], 96, 160, 160, 1)
                yield x, y
