import numpy


def hough_line(img):
    thetas = numpy.deg2rad(numpy.arange(-90.0, 90.0))
    width, height = img.shape

    diagonal_length = numpy.ceil(numpy.sqrt(width * width + height * height))
    rhos = numpy.linspace(-diagonal_length, diagonal_length, diagonal_length * 2.0)

    cos_t = numpy.cos(thetas)
    sin_t = numpy.sin(thetas)
    num_thetas = len(thetas)

    accumulator = numpy.zeros((2 * diagonal_length, num_thetas), dtype=numpy.uint64)

    # super simple edge detection
    y_idxs, x_idxs = numpy.nonzero(img)

    for i in range(len(x_idxs)):
        x = x_idxs[i]
        y = y_idxs[i]

        for t_idx in range(num_thetas):
            rho = round(x * cos_t[t_idx] + y * sin_t[t_idx] + diagonal_length)
            accumulator[rho, t_idx] += 1

    return accumulator, thetas, rhos


if __name__ == '__main__':
    # Create binary image and call hough_line
    image = numpy.zeros((50, 50))
    image[10:40, 10:40] = numpy.eye(30)
    accumulator, thetas, rhos = hough_line(image)

    # Easiest peak finding based on max votes
    idx = numpy.argmax(accumulator)
    rho = rhos[idx / accumulator.shape[1]]
    theta = thetas[idx % accumulator.shape[1]]
    print("rho={0:.2f}, theta={1:.0f}".format(rho, numpy.rad2deg(theta)))
