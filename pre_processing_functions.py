import numpy as np
from PIL import Image
import os
import pydicom


def rescale_center_crop_image_list(image_list, voxel):
    """
    Rescales and center crops a list of images to the size of the smallest rescaled image in the list

    :param image_list: List of images
    :param voxel: List of voxel dimensions of each image
    """
    # find the minimum shape of the list of sax images
    shape_min = check_sax_shapes(image_list)

    # Rescale and crop images to square
    re_scaled_images = list()
    count = 0
    for im in image_list:
        image = re_scale_image(im, voxel[count], shape_min)
        re_scaled_images.append(image)
        count = count + 1

    # After rescaling center crop images to the same Field of View
    center_cropped_images = list()
    shape_min = check_sax_shapes(re_scaled_images)
    for im in re_scaled_images:
        image = crop_img(im, shape_min)
        center_cropped_images.append(image)
    return center_cropped_images

def re_scale_image(image, voxel, shape):
    """
    Crops a rectangular 2D image to square from middle and resizes image to a voxel size of 1mm per pixel

    :param image: numpy image array to be cropped:
    :param voxel: voxel dimensions of image
    :param shape: Minimum shape of an image slice
    """

    # Rotate the image
    if image.shape[0] < image.shape[1]:
        image = image.T

    # crop square image
    short_edge = int(min(shape))
    xx = int((image.shape[0] - short_edge) / 2)
    yy = int((image.shape[1] - short_edge) / 2)
    crop_img = image[xx: xx + short_edge, yy: yy + short_edge]
    img_shape = (int(short_edge * voxel), int(short_edge * voxel))
    image = np.array(Image.fromarray(crop_img).resize(img_shape))
    return image


def crop_img(image, min_shape):
    """
        Crops image from center.

        :param image: numpy image array to be cropped:
        :param min_shape: shape the image is to be cropped to
        """
    short_edge = int(min(min_shape))
    xx = int((image.shape[0] - short_edge) / 2)
    yy = int((image.shape[1] - short_edge) / 2)
    crop_img = image[xx: xx + short_edge, yy: yy + short_edge]
    return crop_img


def non_standard_patients(base_dir):
    """
    Returns a list containing patients to ignore when preparing images.
    :param basedir: base directory of the images
    """
    patient_avoid = list()
    patient_list = os.listdir(base_dir)
    patient_list.sort(key=int)
    for patient in patient_list:  # patient loop
        patient_dir = base_dir + '/' + patient + '/study/'
        slices = os.listdir(patient_dir)
        slices = [x for x in slices if "ch" not in x]
        slices.sort(key=lambda fname: int(fname.split('_')[1]))

        for sax_slice in slices:  # sax loop
            slices_dir = patient_dir + sax_slice + '/'
            files = os.listdir(slices_dir)
            if len(files) != 30:
                patient_avoid.append(patient)
    patient_avoid = list(set(patient_avoid))
    return patient_avoid


def get_sax_image(slices_dir, mode=0, all_images=False):
    """
    Returns a sax image from each slice
    :param slices_dir: base directory of the image slice
    :param mode: Set mode to choose whether systole or diastole image is aquired. Mode = 0 is systole and Mode=1 is diastole
    :param all_images: If set to True, returns all images in sax slice
    """
    files = os.listdir(slices_dir)
    assert len(files) == 30  # ensure data format is constant

    if all_images == False:

        if mode == 0:
            file = files[14]
        elif mode == 1:
            file = files[0]

        image_file_path = os.path.join(slices_dir, file)
        image = pydicom.dcmread(image_file_path)
        voxel = image.PixelSpacing[0]
        slice_thickness = image.SliceThickness
        image = image.pixel_array.astype(float)


        return voxel, slice_thickness, image

    elif all_images == True:
        image_list = list()
        voxel_list = list()
        for file in files:
            image_file_path = os.path.join(slices_dir, file)
            image = pydicom.dcmread(image_file_path)
            image_arr = image.pixel_array.astype(float)
            voxel = image.PixelSpacing[0]
            image_list.append(image_arr)
            voxel_list.append(voxel)

        slice_thickness = image.SliceThickness
        return voxel_list, slice_thickness, image_list


def check_sax_shapes(sax_images):
    """
    Returns the minimum image shape from a list of 2d images
    :param sax_images: A list containing a 2d images
    """
    shape_min = np.asarray([1e5, 1e5])
    for im in sax_images:
        shape = im.shape
        if shape[0] < shape_min[0]:
            shape_min[0] = shape[0]
        if shape[1] < shape_min[1]:
            shape_min[1] = shape[1]
    return shape_min


def crop_heart(image_arr, heart_loc, img_shape):
    """
       Crops the images around the region of interest.
       Returns the cropped image

       :param image_arr: Numpy array of images of shape (# of images, image width, image height)
       :param heart_loc: x, y location of the region of interest
       :param img_shape: Size to crop 2d images (160, 160)
   """
    # crop at heart location
    w = int(img_shape[0]/2)

    if heart_loc[1] < 80:
        heart_loc[1] = 80
    if heart_loc[0] < 80:
        heart_loc[0] = 80
    if heart_loc[1] > image_arr.shape[1] - w:
        heart_loc[1] = image_arr.shape[1] - w
    if heart_loc[0] > image_arr.shape[1] - w:
        heart_loc[0] = image_arr.shape[1] - w

    crop_img_arr = image_arr[:, heart_loc[1] - w: heart_loc[1] + w, heart_loc[0] - w: heart_loc[0] + w]
    return crop_img_arr