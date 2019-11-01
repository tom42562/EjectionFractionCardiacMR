import os
from pre_processing_functions import re_scale_image, non_standard_patients, get_sax_image, check_sax_shapes, crop_img, rescale_center_crop_image_list, crop_heart
import numpy as np
import pydicom
from skimage.morphology import binary_erosion
from skimage.feature import blob_log
import scipy


def pre_process_images(base_dir, output_dir):
    """
    Transforms MRI images into 3d arrays, rescales voxel dimensions to 1mm per pixel, crops images around region of
    interest and saves the output as NumPy arrays of size 96x160x160

    :param base_dir = base directory of the images
    :param output_dir = output directory for numpy files

    """
    # Find the patients with non-standard data to avoid in preprocessing
    patient_avoid = non_standard_patients(base_dir)

    #   create and sort a list of all patients
    patient_list = os.listdir(base_dir)
    patient_list.sort(key=int)
    patient_list = [x for x in patient_list if x not in patient_avoid]  # remove patients in the avoid list

    for patient in patient_list:  # Single iteration for each patient
        patient_dir = base_dir + '/' + patient + '/study/'

        # create and sort a list of all slices for a single patient
        slices = os.listdir(patient_dir)
        slices = [x for x in slices if "ch" not in x]  # remove non_sax slices
        slices.sort(key=lambda fname: int(fname.split('_')[1]))

        # initialize lists/variables
        slice_thickness = list()
        voxel = list()
        sax_images = list()

        for sax_slice in slices:  # Loop through all sax slices
            slices_dir = patient_dir + sax_slice + '/'
            img_voxel, img_slice_thickness, image = get_sax_image(slices_dir, mode=0)
            voxel.append(img_voxel)
            slice_thickness.append(img_slice_thickness)
            sax_images.append(image)

        # rescale and center crop images to the same size
        center_cropped_images = rescale_center_crop_image_list(sax_images, voxel)

        # convert the list to numpy array
        images_arr = np.asarray(center_cropped_images)

        # threshold the image to the 98th percentile
        percentile = np.percentile(images_arr, 98)
        images_arr[images_arr > percentile] = percentile

        # Locate the Region of Interest by taking a difference image over the center slice.
        difference_image = np.zeros(images_arr.shape[1::])  # create an array filled with zeros the required shape

        # read all images of the center slice into an array
        slices_dir = patient_dir + slices[int(len(slices) / 2)] + '/'
        vox, thick, image_list = get_sax_image(slices_dir, all_images=True)
        center_slice_list = rescale_center_crop_image_list(image_list, vox)

        # center crop images to same size as the array of sax slices
        cropped_center_slice = list()
        min_shape = check_sax_shapes(center_cropped_images)
        for im in center_slice_list:
            image = crop_img(im, min_shape)
            cropped_center_slice.append(image)

        center_slice = np.asarray(cropped_center_slice)

        # threshold center slice at the 95th percentile
        percentile = np.percentile(center_slice, 95)
        center_slice[center_slice > percentile] = percentile

        # calculate the difference image
        for var in range(center_slice.shape[0] - 1):
            difference_image += abs(center_slice[var, :, :] - center_slice[var + 1, :, :])

        # threshold the difference image
        threshold = difference_image.mean() * 1.7
        threshold_difference_image = np.where(difference_image > threshold, 1, 0)

        # compute binary erosion several times
        for i in range(5):
            threshold_difference_image = binary_erosion(threshold_difference_image)

        # find the largest blob
        blob = blob_log(threshold_difference_image, min_sigma=1, max_sigma=25)
        max_blob = blob[np.argmax(blob[:, 2]), :]

        # crop at heart location
        heart_loc = [int(max_blob[1]), int(max_blob[0])]
        img_shape = (160, 160)
        heart_centered_images = crop_heart(images_arr, heart_loc, img_shape)

        # rescale images in the slice dimension
        heart_centered_images = scipy.ndimage.zoom(heart_centered_images, (slice_thickness[0], 1, 1,))

        # crop/zero pad image to size (96, 160, 160)
        if heart_centered_images.shape[0] > 96:
            result = heart_centered_images[:96, :, :]
        elif heart_centered_images.shape[0] < 96:
            shape = (96, 160, 160)
            result = np.zeros(shape)
            result[:heart_centered_images.shape[0], :heart_centered_images.shape[1], :heart_centered_images.shape[2]] = heart_centered_images
        else:
            result = heart_centered_images

        # save as numpy array
        print('saving......')
        np.save(output_dir + str(patient) + '.npy', result)
        print(result.shape)
        print('saved ' + str(patient))


base_dir = 'D:/data/train'
output_dir = 'D:/training_data'
pre_process_images(base_dir, output_dir)

