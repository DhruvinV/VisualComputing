# CSC320 Fall 2019
# Assignment 4
# (c) Olga (Ge Ya) Xu, Kyros Kutulakos
#
# DISTRIBUTION OF THIS CODE ANY FORM (ELECTRONIC OR OTHERWISE,
# AS-IS, MODIFIED OR IN PART), WITHOUT PRIOR WRITTEN AUTHORIZATION
# BY KYROS KUTULAKOS IS STRICTLY PROHIBITED. VIOLATION OF THIS
# POLICY WILL BE CONSIDERED AN ACT OF ACADEMIC DISHONESTY

#
# DO NOT MODIFY THIS FILE 
#

# import basic packages
import cv2 as cv
import numpy as np
from algorithm import make_coordinates_matrix

# The most important data structure in the PatchMatch algorithm is the one representing
# the nearest-neighbour field (NNF). As explained in Section 3 of the paper, this is a vector
# field f over the image, ie. a mapping from pixel coordinates (x,y) to
# 2D displacements f(x,y). Here we will represent a displacement field as a numpy matrix.
# Given a NxM openCV source image, the nearest-neighbour field is a matrix of size NxMx2
#

# The following method generates an NNF that is a just random displacement field.
# Specifically, it assigns to each pixel (y,x) in the input image a nearest neighbor
# that is a random pixel in the image.
#
# This is done by first generating a pair of random numbers for each image pixel
# and then subtracting those numbers from the pixel's coordinates. In this way,
# the generated random numbers are turned into 2D *displacements*

# DIFFERENCES FROM THE FILE SUPPLIED WITH A3:
#   - The original functions supplied with A3 have not been changed
#   - In addition to those functions there are a few more functions with the underscore _k in
#     their name. These have the same functionality as their A3 versions but
#     are designed to handle a k-dimensional array of nearest-neighbor fields
#     rather than a single NNF.


def init_NNF(source_image):
    """
    Return a matrix (im_shape[0] x im_shape[1] x 2) representing a random displacement field.
    """

    # get the shape of the source image
    im_shape = source_image.shape

    #
    # Generate a matrix f of size (im_shape[0] x im_shape[1] x 2) that
    # assigns random X and Y displacements to each pixel
    #

    # We first generate a matrix of random x coordinates
    x = np.random.randint(low=0, high=im_shape[1],
                          size=(im_shape[0], im_shape[1]))
    # Then we generate a matrix of random y coordinates
    y = np.random.randint(low=0, high=im_shape[0],
                          size=(im_shape[0], im_shape[1]))
    # To create matrix f, we stack those two matrices
    f = np.dstack((y, x))

    #
    # Now we generate a matrix g of size (im_shape[0] x im_shape[1] x 2)
    # such that g(y,x) = [y,x]
    #
    g = make_coordinates_matrix(im_shape)

    # define the NNF to be the difference of these two matrices
    f = f - g

    return f


def init_NNF_k(k, source_image):
    """
    Return k random displacement fields as a matrix of dimensions
    (k x im_shape[0] x im_shape[1] x 2)
    """

    f = init_NNF(source_image)
    f_k = np.empty((k, f.shape[0], f.shape[1], f.shape[2]), f.dtype)
    f_k[0] = f
    for i in range(1, k):
        f_k[i] = init_NNF(source_image)

    return f_k

# Generate a color openCV image to visualize an NNF f. Since f is a vector field,
# we visualize it with a color image whose saturation indicates the magnitude
# of each vector and whose hue indicates the vector's orientation


def create_NNF_image(f):
    """
    Create an RGB image to visualize the nearest-neighbour field.
    """
    # square the individual coordinates
    magnitude = np.square(f)
    # sum the coordinates to compute the magnitude
    magnitude = np.sqrt(np.sum(magnitude, axis=2))
    # compute the orientation of each vector
    orientation = np.arccos(f[:, :, 1] / magnitude) / np.pi * 180
    # rescale the orientation to create a hue channel
    hue = np.array(orientation, np.uint8)
    # rescale the magnitude to create a saturation channel
    magnitude = magnitude / np.max(magnitude) * 255
    saturation = np.array(magnitude, np.uint8)
    # create a constant brightness channel
    brightness = np.zeros(magnitude.shape, np.uint8) + 200
    # create the HSV image
    hsv = np.dstack((hue, saturation, brightness))
    # return an RGB image with the specified HSV values
    rgb_image = cv.cvtColor(hsv, cv.COLOR_HSV2RGB)

    return rgb_image

# Generate a color openCV image to visualize an NNF f as a set of
# 2D correspondences: the new image contains the source and target
# image side by side, with 2D vectors drawn between them indicating
# the correspondences for a sparse set of image pixels
#
# We do the rendering using matplotlib to write a temporary image to a file,
# then re-read it as an openCV image


def create_NNF_vectors_image(source, target, f, patch_size,
                             server=True,
                             subsampling=100,
                             line_width=0.5,
                             line_color='k',
                             tmpdir='./'):
    """
    Display the nearest-neighbour field as a sparse vector field between source and target images
    """
    import matplotlib.pyplot as plt

    # get the shape of the source image
    im_shape = source.shape

    # if you are using matplotlib on a server
    if server:
        plt.switch_backend('agg')
    import matplotlib.patches as patches

    fig = plt.figure(frameon=False)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)

    source = cv.cvtColor(source, cv.COLOR_BGR2RGB)
    target = cv.cvtColor(target, cv.COLOR_BGR2RGB)

    # create an image that contains the source and target side by side
    plot_im = np.concatenate((source, target), axis=1)
    ax.imshow(plot_im)

    vector_coords = make_coordinates_matrix(im_shape, step=subsampling)
    vshape = vector_coords.shape
    vector_coords = np.reshape(vector_coords, (vshape[0] * vshape[1], 2))

    for coord in vector_coords:
        rect = patches.Rectangle((coord[1] - patch_size / 2.0, coord[0] - patch_size / 2.0),
                                 patch_size,
                                 patch_size,
                                 linewidth=line_width,
                                 edgecolor=line_color,
                                 facecolor='none')
        ax.add_patch(rect)

        arrow = patches.Arrow(coord[1],
                              coord[0],
                              f[coord[0], coord[1], 1] + im_shape[1],
                              f[coord[0], coord[1], 0],
                              lw=line_width,
                              edgecolor=line_color)
        ax.add_patch(arrow)

    dpi = fig.dpi
    fig.set_size_inches(im_shape[1] * 2 / dpi, im_shape[0] / dpi)
    tmp_image = tmpdir+'/tmpvecs.png'
    fig.savefig(tmp_image)
    plt.close(fig)
    return tmp_image


def save_NNF(f, filename):
    """
    Save the nearest-neighbour field matrix in numpy file
    """
    try:
        np.save('{}'.format(filename), f)
    except IOError as e:
        return False, e
    else:
        return True, None


def load_NNF(filename, shape=None):
    """
    Load the nearest-neighbour field from a numpy file
    """
    try:
        f = np.load(filename)
    except IOError as e:
        return False, None, e
    else:
        if shape is not None:
            if (f.shape[0] != shape[0] or
                f.shape[1] != shape[1]):
                return False, None, 'NNF has incorrect dimensions'
        return True, f, None


def save_NNF_k(f_k, filename):
    """
    Save the k nearest-neighbour fields as a matrix into a numpy file. For backwards
    compatibility with A3, f_k contains only one field (ie. k=1) the routine saves exactly
    the same file save_NNF() would save.
    """
    try:
        if f_k.shape[0] == 1:
            np.save('{}'.format(filename), f_k[0])
        else:
            np.save('{}'.format(filename), f_k)
    except IOError as e:
        return False, e
    else:
        return True, None


def load_NNF_k(k, filename, shape=None):
    """
    Load the k nearest-neighbour fields from a numpy file and return then in a matrix.
    For backwards compatibility with A3, k=1 the routine expects to load the same file
    load_NNF() would load.
    """
    try:
        ff = np.load(filename)
    except IOError as e:
        return False, None, e
    else:
        if k == 1:
            f_k = np.empty((1, ff.shape[0], ff.shape[1], ff.shape[2]), ff.dtype)
            f_k[0] = ff
        else:
            f_k = ff
        if f_k.shape[0] != k:
            return False, None, 'NNFs file contains %d instead of %d NNFs' % (f_k.shape[0], k)
        if shape is not None:
            if (f_k.shape[1] != shape[0] or
                f_k.shape[2] != shape[1]):
                return False, None, 'NNF has incorrect dimensions'
        return True, f_k, None
