import numpy
from PIL import Image
from pylab import *
from scipy import signal, array

from kernels.kernels import gauss_derivatives, gauss_kernel


def compute_harris_response(image, a):
    """ compute the Harris corner detector response function
        for each pixel in the image"""

    imx, imy = gauss_derivatives(image, 3)

    gauss = gauss_kernel(3)

    w_xx = signal.convolve(imx * imx, gauss, mode='same')
    w_xy = signal.convolve(imx * imy, gauss, mode='same')
    w_yy = signal.convolve(imy * imy, gauss, mode='same')

    w_det = w_xx * w_yy - w_xy ** 2
    w_tr = w_xx + w_yy

    return w_det - a * w_tr


def compute_brown_szeliski_winder_response(image, a):
    """ compute the Harris corner detector response function
        for each pixel in the image"""

    imx, imy = gauss_derivatives(image, 3)

    gauss = gauss_kernel(3)

    w_xx = signal.convolve(imx * imx, gauss, mode='same')
    w_xy = signal.convolve(imx * imy, gauss, mode='same')
    w_yy = signal.convolve(imy * imy, gauss, mode='same')

    w_det = w_xx * w_yy - w_xy ** 2
    w_tr = w_xx + w_yy

    return w_det / w_tr


def get_harris_points(harrisim, min_distance=10, threshold=0.1):
    """ return corners from a Harris response image
        min_distance is the minimum nbr of pixels separating
        corners and image boundary"""

    # find top corner candidates above a threshold
    corner_threshold = max(harrisim.ravel()) * threshold
    harrisim_t = (harrisim > corner_threshold) * 1

    # get coordinates of candidates
    candidates = harrisim_t.nonzero()
    coords = [(candidates[0][c], candidates[1][c]) for c in range(len(candidates[0]))]
    # ...and their values
    candidate_values = [harrisim[c[0]][c[1]] for c in coords]

    # sort candidates
    index = numpy.argsort(candidate_values)

    # store allowed point locations in array
    allowed_locations = numpy.zeros(harrisim.shape)
    allowed_locations[min_distance: -min_distance, min_distance: -min_distance] = 1

    # select the best points taking min_distance into account
    filtered_coordinates = []
    for i in index:
        if allowed_locations[coords[i][0]][coords[i][1]] == 1:
            filtered_coordinates.append(coords[i])
            allowed_locations[
                (coords[i][0] - min_distance):(coords[i][0] + min_distance),
                (coords[i][1] - min_distance):(coords[i][1] + min_distance)
            ] = 0

    return filtered_coordinates


def plot_harris_points(image, filtered_coords):
    """ plots corners found in image"""

    figure()
    gray()
    imshow(image)
    plot([p[1] for p in filtered_coords], [p[0] for p in filtered_coords], '*')
    axis('off')
    show()


if __name__ == '__main__':
    # Load image and convert it to greyscale
    im = array(
        Image.open('path_to_image').convert('L')
    )

    # Compute the Harris responses
    harrisim = compute_harris_response(im, 0.04)

    # Find the coordinates of 'best' corner points
    filtered_coordinates = get_harris_points(harrisim, 6)

    plot_harris_points(im, filtered_coordinates)
