from skimage import io

import numpy
from PIL import Image
from scipy import signal, array

from numpy import mgrid, exp


def gauss_kernel(size, size_y=None):
    """ Returns a normalized 2D gauss kernel array for convolutions """
    size = int(size)
    if not size_y:
        size_y = size
    else:
        size_y = int(size_y)

    x, y = mgrid[-size: size + 1, -size_y: size_y + 1]

    g = exp(-(x ** 2 / float(size) + y ** 2 / float(size_y)))
    return g / g.sum()


def gauss_derivative_kernels(size, size_y=None):
    """ returns x and y derivatives of a 2D
        gauss kernel array for convolutions """
    size = int(size)
    if not size_y:
        size_y = size
    else:
        size_y = int(size_y)
    y, x = mgrid[-size: size + 1, -size_y: size_y + 1]

    # x and y derivatives of a 2D gaussian with standard dev half of size
    # (ignore scale factor)
    gx = - x * exp(-(x ** 2 / float((0.5 * size) ** 2) + y ** 2 / float((0.5 * size_y) ** 2)))
    gy = - y * exp(-(x ** 2 / float((0.5 * size) ** 2) + y ** 2 / float((0.5 * size_y) ** 2)))

    return gx, gy


def gauss_derivatives(im, n, ny=None):
    """ returns x and y derivatives of an image using gaussian
        derivative filters of size n. The optional argument
        ny allows for a different size in the y direction."""

    gx, gy = gauss_derivative_kernels(n, size_y=ny)

    imx = signal.convolve(im, gx, mode='same')
    imy = signal.convolve(im, gy, mode='same')

    return imx, imy


if __name__ == '__main__':
    im = array(
        Image.open('path_to_original_image').convert('L')
    )

    # Calculate the derivatives in x and y axis
    # using the gauss derivative.
    # You change the window size.
    x, y = gauss_derivatives(im, 3)

    x_min = x.min()
    x_max = x.max()

    x_final = (x - x_min) / (x_max - x_min)

    y_min = y.min()
    y_max = y.max()

    y_final = (y - y_min) / (y_max - y_min)

    # Concat the x and y derivative images
    combined = numpy.concatenate((x_final, y_final), axis=1)

    io.imsave('path_to_new_image', combined)
