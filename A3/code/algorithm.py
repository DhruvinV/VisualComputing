# CSC320 Fall 2019
# Assignment 3
# (c) Olga (Ge Ya) Xu, Kyros Kutulakos
#
# DISTRIBUTION OF THIS CODE ANY FORM (ELECTRONIC OR OTHERWISE,
# AS-IS, MODIFIED OR IN PART), WITHOUT PRIOR WRITTEN AUTHORIZATION
# BY KYROS KUTULAKOS IS STRICTLY PROHIBITED. VIOLATION OF THIS
# POLICY WILL BE CONSIDERED AN ACT OF ACADEMIC DISHONESTY

#
# DO NOT MODIFY THIS FILE ANYWHERE EXCEPT WHERE INDICATED
#

# import basic packages
import numpy as np
import sys

# basic numpy configuration

# set random seed
np.random.seed(seed=131)
# ignore division by zero warning
np.seterr(divide='ignore', invalid='ignore')


# This function implements the basic loop of the PatchMatch
# algorithm, as explained in Section 3.2 of the paper.
# The function takes an NNF f as input, performs propagation and random search,
# and returns an updated NNF.
#
# The function takes several input arguments:
#     - source_patches:      The matrix holding the patches of the source image,
#                            as computed by the make_patch_matrix() function. For an
#                            NxM source image and patches of width P, the matrix has
#                            dimensions NxMxCx(P^2) where C is the number of color channels
#                            and P^2 is the total number of pixels in the patch. The
#                            make_patch_matrix() is defined below and is called by the
#                            initialize_algorithm() method of the PatchMatch class. For
#                            your purposes, you may assume that source_patches[i,j,c,:]
#                            gives you the list of intensities for color channel c of
#                            all pixels in the patch centered at pixel [i,j]. Note that patches
#                            that go beyond the image border will contain NaN values for
#                            all patch pixels that fall outside the source image.
#     - target_patches:      The matrix holding the patches of the target image.
#     - f:                   The current nearest-neighbour field
#     - alpha, w:            Algorithm parameters, as explained in Section 3 and Eq.(1)
#     - propagation_enabled: If true, propagation should be performed.
#                            Use this flag for debugging purposes, to see how your
#                            algorithm performs with (or without) this step
#     - random_enabled:      If true, random search should be performed.
#                            Use this flag for debugging purposes, to see how your
#                            algorithm performs with (or without) this step.
#     - odd_iteration:       True if and only if this is an odd-numbered iteration.
#                            As explained in Section 3.2 of the paper, the algorithm
#                            behaves differently in odd and even iterations and this
#                            parameter controls this behavior.
#     - best_D:              And NxM matrix whose element [i,j] is the similarity score between
#                            patch [i,j] in the source and its best-matching patch in the
#                            target. Use this matrix to check if you have found a better
#                            match to [i,j] in the current PatchMatch iteration
#     - global_vars:         (optional) if you want your function to use any global variables,
#                            you can pass them to/from your function using this argument

# Return arguments:
#     - new_f:               The updated NNF
#     - best_D:              The updated similarity scores for the best-matching patches in the
#                            target
#     - global_vars:         (optional) if you want your function to use any global variables,
#                            return them in this argument and they will be stored in the
#                            PatchMatch data structure


def propagation_and_random_search(source_patches, target_patches,
                                  f, alpha, w,
                                  propagation_enabled, random_enabled,
                                  odd_iteration, best_D=None,
                                  global_vars=None
                                ):
    new_f = f.copy()

    #############################################
    ###  PLACE YOUR CODE BETWEEN THESE LINES  ###
    #############################################
    # propagation
    print(w)
    itera = int(np.ceil(- np.log10(w)/ np.log10(alpha)))
    source_patches[np.isnan(source_patches)] = 0
    target_patches[np.isnan(target_patches)] = 0
    if (isinstance(best_D, np.ndarray) == False):
        if(best_D is None):
            best_D = np.zeros((source_patches.shape[0],source_patches.shape[1]))
            k = 0
            for i in range(source_patches.shape[0]):
                for j in range(source_patches.shape[1]):
                    best_D[i,j] = np.linalg.norm(source_patches[i,j]-target_patches[i,j])
                    k+=1
    print(best_D.shape)
    if odd_iteration == True:
        for i in range(0,source_patches.shape[0]):
            # print(i)
            for j in range(0,source_patches.shape[1]):
                if propagation_enabled:
                    d_set = dict()
                    if(within_dim([i,j]+new_f[i,max(j-1,0)],source_patches) == False):
                        loc_in_B = [i,j]+new_f[i,max(j-1,0)]
                        new_score = np.linalg.norm(source_patches[i,j]-target_patches[loc_in_B[0],loc_in_B[1]])
                        d_set[(i,max(j-1,0))] = new_score
                    # d_set[(i,max(j-1,0))] = best_D[i,max(j-1,0)]
                    if(within_dim([i,j]+new_f[max(i-1, 0), j],source_patches) == False):
                        loc_in_B = [i,j]+new_f[max(i-1, 0), j]
                        new_score = np.linalg.norm(source_patches[i,j]-target_patches[loc_in_B[0],loc_in_B[1]])
                        d_set[(max(i-1,0),j)] = new_score
                    d_set[(i,j)] = best_D[i,j]
                    (x,y) = get_key_from_dict(d_set)
                    if((x,y)==(i,j)):
                        continue
                    else:
                        # if(0<=x2<target)
                        new_f[i,j] = new_f[x,y]
                        best_D[i,j] = d_set[(x,y)]
                if(random_enabled):
                    # print(itera)?
                    # print(type(itera))
                    # k = int(itera)
                    R = np.random.randint(-1,2,size=(int(itera),2))
                    # print(R)
                    size = w*alpha**np.arange(10)
                    # print(itera)
                    # print(R[5])
                    u = np.multiply(np.transpose(R),size)
                    # print(u.T)
                    # print(u.T.shape)
                    # u_i = np.array([3,4]) + u.
                    u_i = f[i,j] + u.T

                    # print(u_i.shape)
                    # print(u_i)
                    u_ik = [i,j] + u_i
                    clipped_x = np.clip(u_ik[:,0],0,source_patches.shape[0]-1)
                    clipped_y = np.clip(u_ik[:,1],0,source_patches.shape[1]-1)
                    clipp = np.column_stack((clipped_x,clipped_y))
                    clipp = clipp.astype(int)
                    diff = source_patches[i,j].reshape((-1)) - target_patches[clipp[:,0],clipp[:,1]].reshape((-1,source_patches.shape[2]*source_patches.shape[3]))
                    print(diff[0])
                    sys.exit()
                    norm = np.apply_along_axis(np.linalg.norm, 1,diff)
                    # print(u_i)
                    clipped_x = np.clip(u_i[:,0],0,source_patches.shape[0]-1)
                    clipped_y = np.clip(u_i[:,1],0,source_patches.shape[1]-1)
                    clipp = np.column_stack((clipped_x,clipped_y))
                    clipp = clipp.astype(int)
                    # break
                    min_minma = np.amin(norm)
                    if(best_D[i,j]>min_minma):
                        # print("true")
                        best_D[i,j] = min_minma
                        new_f[i,j] = clipp[np.argmin(norm)]
                break
    else:
            # consider f(x+1,y),f(x,y+1)
        h_min,h_max,w_min,w_max,offsets = source_patches.shape[0]-1,-1,source_patches.shape[1]-1,-1,-1
        # examining the offsets in reverse scan order starting from bottom
        # goint to top
        for i in range(h_min,h_max,-1):
            for j in range(w_min,w_max,-1):
                if propagation_enabled:
                    d_set = dict()
                    # d_set[(min(i+1,h_min),j)] = best_D[min(i+1,h_min),j]
                    if(within_dim([i,j]+new_f[min(i+1,h_min),j],source_patches) == False):
                        loc_in_B = [i,j]+new_f[min(i+1,h_min),j]
                        new_score = np.linalg.norm(source_patches[i,j]-target_patches[loc_in_B[0],loc_in_B[1]])
                        d_set[(min(i+1,h_min),j)] = new_score
                    # d_set[(i,min(j+1,w_min))] = best_D[i,min(j+1,w_min)]
                    if(within_dim([i,j]+new_f[i,min(j+1,w_min)],source_patches) == False):
                        loc_in_B = [i,j]+new_f[i,min(j+1,w_min)]
                        new_score = np.linalg.norm(source_patches[i,j]-target_patches[loc_in_B[0],loc_in_B[1]])
                        d_set[(i,min(j+1,w_min))] = new_score
                     # d_set[(i,j)] = best_D[i,j]
                    d_set[(i,j)] = best_D[i,j]
                    (x,y) = get_key_from_dict(d_set)
                    if((x,y)==(i,j)):
                        continue
                    else:
                        # if(0<=x2<target)
                        new_f[i,j] = new_f[x,y]
                        best_D[i,j] = d_set[(x,y)]
                if(random_enabled):
                    # print("in random")
                    R = np.random.randint(-1,2,size=(int(itera),2))
                    # print(R)
                    size = w*alpha**np.arange(10)
                    # print(itera)
                    u = np.multiply(np.transpose(R),size)
                    # print(u.T)
                    # print(u.T.shape)
                    # u_i = np.array([3,4]) + u.T
                    u_i = f[i,j] + u.T
                    # print(u_i.shape)
                    # print(u_i)
                    u_ik = [i,j] + u_i
                    clipped_x = np.clip(u_ik[:,0],0,source_patches.shape[0]-1)
                    clipped_y = np.clip(u_ik[:,1],0,source_patches.shape[1]-1)
                    clipp = np.column_stack((clipped_x,clipped_y))
                    clipp = clipp.astype(int)
                    # print(clipp.shape)
                    # print(clipp[:,0])
                    # print(target_patches[clipp[:,0],clipp[:,1]].reshape((-1,source_patches.shape[2]*source_patches.shape[3])))
                    diff = source_patches[i,j].reshape((-1)) - target_patches[clipp[:,0],clipp[:,1]].reshape((-1,source_patches.shape[2]*source_patches.shape[3]))
                    # norm = np.linalg.norm(diff)
                    # print(source_patches[i,j].reshape((-1)).shape)
                    clipped_x = np.clip(u_i[:,0],0,source_patches.shape[0]-1)
                    clipped_y = np.clip(u_i[:,1],0,source_patches.shape[1]-1)
                    clipp = np.column_stack((clipped_x,clipped_y))
                    norm = np.apply_along_axis(np.linalg.norm, 1,diff)
                    min_minma = np.amin(norm)
                    # print(best_D[i,j]>min_minma)
                    if(best_D[i,j]>min_minma):
                        best_D[i,j] = min_minma
                        # print(u_i[np.argmin(norm)])
                        # print(np.argmin(norm))
                        new_f[i,j] = clipp[np.argmin(norm)]
    return new_f, best_D, global_vars

def get_key_from_dict(some_dict):
    min_value = min(some_dict.values())
    for key,value in some_dict.items():
        if(min_value == value):
            return key

def within_dim(x,y):
    return (x[0] < 0 or x[0] >= y.shape[0]) or (x[1] < 0 or x[1] >= y.shape[1])
# def generate_best_D(nnf):
#     return make_coordinates_matrix(nnf.shape[:3])
# This function uses a computed NNF to reconstruct the source image
# using pixels from the target image. The function takes two input
# arguments
#     - target: the target image that was used as input to PatchMatch
#     - f:      the nearest-neighbor field the algorithm computed
# and should return a reconstruction of the source image:
#     - rec_source: an openCV image that has the same shape as the source image
#
# To reconstruct the source, the function copies to pixel (x,y) of the source
# the color of pixel (x,y)+f(x,y) of the target.
#
# The goal of this routine is to demonstrate the quality of the computed NNF f.
# Specifically, if patch (x,y)+f(x,y) in the target image is indeed very similar
# to patch (x,y) in the source, then copying the color of target pixel (x,y)+f(x,y)
# to the source pixel (x,y) should not change the source image appreciably.
# If the NNF is not very high quality, however, the reconstruction of source image
# will not be very good.
#
# You should use matrix/vector operations to avoid looping over pixels,
# as this would be very inefficient

def reconstruct_source_from_target(target, f):

    #########################1####################
    ###  PLACE YOUR CODE BETWEEN THESE LINES  ###
    #############################################
    y = make_coordinates_matrix(target.shape)
    result = target[y[:,:,0]+f[:,:,0],y[:,:,1]+f[:,:,1]]
    #############################################

    return result


# This function takes an NxM image with C color channels and a patch size P
# and returns a matrix of size NxMxCxP^2 that contains, for each pixel [i,j] in
# in the image, the pixels in the patch centered at [i,j].
#
# You should study this function very carefully to understand precisely
# how pixel data are organized, and how patches that extend beyond
# the image border are handled.


def make_patch_matrix(im, patch_size):
    phalf = patch_size // 2
    # create an image that is padded with patch_size/2 pixels on all sides
    # whose values are NaN outside the original image
    padded_shape = im.shape[0] + patch_size - 1, im.shape[1] + patch_size - 1, im.shape[2]
    padded_im = np.zeros(padded_shape) * np.NaN
    padded_im[phalf:(im.shape[0] + phalf), phalf:(im.shape[1] + phalf), :] = im

    # Now create the matrix that will hold the vectorized patch of each pixel. If the
    # original image had NxM pixels, this matrix will have NxMx(patch_size*patch_size)
    # pixels
    patch_matrix_shape = im.shape[0], im.shape[1], im.shape[2], patch_size ** 2
    patch_matrix = np.zeros(patch_matrix_shape) * np.NaN
    for i in range(patch_size):
        for j in range(patch_size):
            patch_matrix[:, :, :, i * patch_size + j] = padded_im[i:(i + im.shape[0]), j:(j + im.shape[1]), :]

    return patch_matrix


# Generate a matrix g of size (im_shape[0] x im_shape[1] x 2)
# such that g(y,x) = [y,x]
#
# Step is an optional argument used to create a matrix that is step times
# smaller than the full image in each dimension
#
# Pay attention to this function as it shows how to perform these types
# of operations in a vectorized manner, without resorting to loops


def make_coordinates_matrix(im_shape, step=1):
    """
    Return a matrix of size (im_shape[0] x im_shape[1] x 2) such that g(x,y)=[y,x]
    """
    range_x = np.arange(0, im_shape[1], step)
    range_y = np.arange(0, im_shape[0], step)
    axis_x = np.repeat(range_x[np.newaxis, ...], len(range_y), axis=0)
    axis_y = np.repeat(range_y[..., np.newaxis], len(range_x), axis=1)

    return np.dstack((axis_y, axis_x))
