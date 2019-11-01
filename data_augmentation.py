import numpy as np
from scipy.ndimage import shift, rotate


class normalize_meanstd():
    """
    Returns custom normalised data set by the mean and standard deviation.
    """
    def __init__(self, x, nstd=6.0):
        self.mu_src = np.mean(x)
        self.sigma_src = np.std(x)
        self.eps = 1e-12
        self.nstd = nstd

    def normalize(self, x):
        y = (x - self.mu_src)
        y = y / (self.eps + self.sigma_src)
        y[y < -self.nstd] = -self.nstd
        y[y > self.nstd] = self.nstd
        return y


def img_shift(inp_image, shift_range):
    """
    Applies an image shift data augmentation

    :param inp_image: input image
    :param shift_range: maximum shift value
    """
    vertical_shift_size = np.random.randint(-shift_range, shift_range)
    horizontal_shift_size = np.random.randint(-shift_range, shift_range)
    output = shift(inp_image, (0,vertical_shift_size, horizontal_shift_size))
    return output


def img_rotate(inp_image, rotation_angle):
    """
    Applies an image rotation data augmentation

    :param inp_image: input image
    :param rotation_angle: maximum rotation angle
    """
    random_rotation_angle = np.random.randint(-rotation_angle, rotation_angle)
    output = rotate(inp_image, random_rotation_angle, axes=(1, 2), reshape=False)
    return output

