# CSC320 Fall 2019
# Assignment 4
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
# import the heapq package
from heapq import heappush, heappushpop, nlargest
# see below for a brief comment on the use of tiebreakers in python heaps
from itertools import count
_tiebreaker = count()

from copy import deepcopy as copy
import sys
# basic numpy configuration

# set random seed
np.random.seed(seed=131)
# ignore division by zero warning
np.seterr(divide='ignore', invalid='ignore')


# This function implements the basic loop of the Generalized PatchMatch
# algorithm, as explained in Section 3.2 of the PatchMatch paper and Section 3
# of the Generalized PatchMatch paper.
#
# The function takes k NNFs as input, represented as a 2D array of heaps and an
# associated 2D array of dictionaries. It then performs propagation and random search
# as in the original PatchMatch algorithm, and returns an updated 2D array of heaps
# and dictionaries
#
# The function takes several input arguments:
#     - source_patches:      *** Identical to A3 ***
#                            The matrix holding the patches of the source image,
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
#     - target_patches:      *** Identical to A3 ***
#                            The matrix holding the patches of the target image.
#     - f_heap:              For an NxM source image, this is an NxM array of heaps. See the
#                            helper functions below for detailed specs for this data structure.
#     - f_coord_dictionary:  For an NxM source image, this is an NxM array of dictionaries. See the
#                            helper functions below for detailed specs for this data structure.
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
#     - global_vars:         (optional) if you want your function to use any global variables,
#                            you can pass them to/from your function using this argument

# Return arguments:
#     - global_vars:         (optional) if you want your function to use any global variables,
#                            return them in this argument and they will be stored in the
#                            PatchMatch data structure
#     NOTE: the variables f_heap and f_coord_dictionary are modified in situ so they are not
#           explicitly returned as arguments to the function


def propagation_and_random_search_k(source_patches, target_patches,f_heap,f_coord_dictionary,alpha, w,propagation_enabled, random_enabled,odd_iteration,global_vars):


    #################################################
    ###  PLACE YOUR A3 CODE BETWEEN THESE LINES   ###
    ###  THEN START MODIFYING IT AFTER YOU'VE     ###
    ###  IMPLEMENTED THE 2 HELPER FUNCTIONS BELOW ###
    #################################################
    print(len(f_heap))
    # sys.exit()
    # #############################################
    # ###  PLACE YOUR CODE BETWEEN THESE LINES  ###
    # #############################################
    # # propagation    itera = int(np.ceil(- np.log10(w)/ np.log10(alpha)))
    source_patches[np.isnan(source_patches)] = 0
    target_patches[np.isnan(target_patches)] = 0
    k = len(f_heap[0][0])
    N = source_patches.shape[0]
    M = source_patches.shape[1]
    itera = int(np.ceil(- np.log10(w)/ np.log10(alpha)))
    if odd_iteration == True:
        for i in range(0,source_patches.shape[0]):
            print(i)
            # i = 12
            for j in range(0,source_patches.shape[1]):
                # j = 414
                if propagation_enabled:
                    # d_set = dict()
                    worst = f_heap[i][max(j-1,0)][0][0]
                    for h in f_heap[i][max(j-1,0)]:
                        # if(h[2] == (75,-139)):
                        #     # print(f_coord_dictionary[i][max(j-1,0)].keys())
                        #     print(i,j)
                        if((within_dim([i+h[2][0],j+h[2][1]],source_patches)==False) and (h[2] not in f_coord_dictionary[i][j])):
                            # print(i,j)
                            # replace the patch with adjacent patch.
                            loc_in_B = [i+h[2][0],j+h[2][1]]
                            # print(loc_in_B, i, j, within_dim([i+h[2][0],j+h[2][1]],source_patches))
                            one = -np.linalg.norm(source_patches[i,j] - target_patches[loc_in_B[0],loc_in_B[1]])
                            two = _tiebreaker.next()
                            three = h[2]
                            # pritn(three)A
                            # print(one,worst)
                            # if(-one > worst):
                            f_coord_dictionary[i][j][three] = 1
                            set_none = heappushpop(f_heap[i][j],(one,two,three))
                            f_coord_dictionary[i][j].pop(set_none[2],None)
                            worst = set_none[0]
                    worst = f_heap[max(i-1, 0)][j][0][0]
                    for h in f_heap[max(i-1, 0)][j]:
                        if((within_dim([i+h[2][0],j+h[2][1]],source_patches)==False) and (h[2] not in f_coord_dictionary[i][j])):
                            # replace the patch with adjacent patch.``
                            loc_in_B = [i+h[2][0],j+h[2][1]]
                            one = -np.linalg.norm(source_patches[i,j] - target_patches[loc_in_B[0],loc_in_B[1]])
                            two = _tiebreaker.next()
                            three = h[2]
                            # prinT(three)
                            # since we are replacing the worst one with this patch. Just push and pop.
                            # if(-one > worst):
                            f_coord_dictionary[i][j][three] = 1
                            set_none = heappushpop(f_heap[i][j],(one,two,three))
                            f_coord_dictionary[i][j].pop(set_none[2])
                                # worst = set_none[0]
                if random_enabled:
                    for neighbor in f_heap[i][j][1:]:
                        worst_D = f_heap[i][j][0]
                        ofs = neighbor[2]
                        patch = np.array([i+ofs[0], j+ofs[1]])
                        l = 0
                        while(l<int(itera)):
                             R = np.random.randint(-1,2,size=2) # [1, 1]
                             new_p =  patch + w*(alpha**l)*R
                             x = np.clip(new_p[0],0,source_patches.shape[0]-1)
                             y = np.clip(new_p[1],0,source_patches.shape[1]-1)
                             ux,uy = int(x),int(y)
                             dist = np.linalg.norm(source_patches[i,j]-target_patches[ux,uy])
                             new_d = (-dist, _tiebreaker.next(),tuple(neighbor[2]))
                             if(new_d[0]>worst_D[0] and ((new_d[2] in f_coord_dictionary[i][j].keys()) == False)):
                                 f_coord_dictionary[i][j][new_d[2]] = 1
                                 heappushpop(f_heap[i][j],new_d)
                                 worst_D = f_heap[i][j][0]
                             l=l+1
    else:
                # consider f(x+1,y),f(x,y+1)
        h_min,w_min,offsets = source_patches.shape[0]-1,source_patches.shape[1]-1,-1
        # examining the offsets in reverse scan order starting from bottom
        # goint to top
        for i in range(h_min,-1,-1):
            for j in range(w_min,-1,-1):
                if propagation_enabled:
                    worst = f_heap[min(i+1,h_min-1)][j][0][0]
                    for h in f_heap[min(i+1,h_min-1)][j]:
                        if(not(within_dim([i+h[2][0],j+h[2][1]],source_patches)) and (h[2] not in f_coord_dictionary[min(i+1,h_min-1)][j].keys())):
                            # replace the patch with adjacent patch.
                            loc_in_B = [i+h[2][0],j+h[2][1]]
                            one = np.linalg.norm(source_patches[i,j] - target_patches[loc_in_B[0],loc_in_B[1]])
                            two = _tiebreaker.next()
                            three = h[2]
                            # since we are replacing the worst one with this patch. Just push and pop.
                            # print(one,worst)
                            if(-one > worst):
                                f_coord_dictionary[min(i+1,h_min-1)][j][three] = None
                                set_none = heappushpop(f_heap[min(i+1,h_min-1)][j],(-one,two,three))
                                f_coord_dictionary[min(i+1,h_min-1)][j].pop(set_none[2])
                                worst = f_heap[min(i+1,h_min-1)][j][0]
                            # set_none = heappushpop(f_heap[i][j],(one,two,three))
                            # f_coord_dictionary[i][j].pop(set_none[2])
                    worst = f_heap[i][min(j+1,w_min-1)][0][0]
                    for h in f_heap[i][min(j+1,w_min-1)]:
                        if(h[2] == (75,-139)):
                            print(i,j)
                        if(not(within_dim([i+h[2][0],j+h[2][1]],source_patches)) and (h[2] not in f_coord_dictionary[i][min(j+1,w_min-1)].keys())):
                            # replace the patch with adjacent patch.
                            loc_in_B = [i+h[2][0],j+h[2][1]]
                            one = np.linalg.norm(source_patches[i,j] - target_patches[loc_in_B[0],loc_in_B[1]])
                            two = _tiebreaker.next()
                            three = h[2]
                            # since we are replacing the worst one with this patch. Just push and pop.
                            # set_none = heappushpop(f_heap[i][min(j+1,w_min-1)],(one,two,three))
                            # f_coord_dictionary[.pop(set_none[2])
                            # print(one,worst)
                            if(-one > worst):
                                f_coord_dictionary[min(i+1,h_min-1)][j][three] = None
                                set_none = heappushpop(f_heap[i][min(j+1,w_min-1)],(-one,two,three))
                                f_coord_dictionary[i][min(j+1,w_min-1)].pop(set_none[2])
                                worst = f_heap[i][min(j+1,w_min-1)][0][0]
                if random_enabled:
                    for neighbor in f_heap[i][j]: # Random search around pixel and its neighbors
                    # k_neighb = f_coord_dictionary[i][j].keys()
                    # pop = np.array(list(k_neighb))
                        current_dist = neighbor[0] #(similarity, null, offset)
                        ofs = neighbor[2]
                        current_patch = [i, j] + lis(ofs)
                        max_loop = int(itera)
                        l = 0
                        while(l < max_loop):
                            R = np.random.randint(-1,2,size=2) # [1, 1]
                            new_point = current_patch + w*(alpha**l)*R
                            # clip new point to fit in image
                            # print(new_point)
                            if(within_dim(new_point,source_patches)==False):
                                new_dist = np.linalg.norm(source_patches[i, j] - target_patches[new_point[0], new_point[1]])
                                new_tup = (-new_dist, _tiebreaker.next(),tuple(new_point))
                                if(-new_dist > current_dist[0]):
                                    if(tuple(new_point) not in f_coord_dictionary[i][j].keys()):
                                        f_coord_dictionary[i][j][tuple(new_point)] = None
                                        set_none = heappushpop(f_heap[i][j],new_tup)
                                        f_coord_dictionary[i][j].pop(set_none[2])
                            l = l+1

    return global_vars


# This function builds a 2D heap data structure to represent the k nearest-neighbour
# fields supplied as input to the function.
#
# The function takes three input arguments:
#     - source_patches:      The matrix holding the patches of the source image (see above)
#     - target_patches:      The matrix holding the patches of the target image (see above)
#     - f_k:                 A numpy array of dimensions kxNxMx2 that holds k NNFs. Specifically,
#                            f_k[i] is the i-th NNF and has dimension NxMx2 for an NxM image.
#                            There is NO requirement that f_k[i] corresponds to the i-th best NNF,
#                            i.e., f_k is simply assumed to be a matrix of vector fields.
#
# The function should return the following two data structures:
#     - f_heap:              A 2D array of heaps. For an NxM image, this array is represented as follows:
#                               * f_heap is a list of length N, one per image row
#                               * f_heap[i] is a list of length M, one per pixel in row i
#                               * f_heap[i][j] is the heap of pixel (i,j)
#                            The heap f_heap[i][j] should contain exactly k tuples, one for each
#                            of the 2D displacements f_k[0][i][j],...,f_k[k-1][i][j]
#
#                            Each tuple has the format: (priority, counter, displacement)
#                            where
#                                * priority is the value according to which the tuple will be ordered
#                                  in the heapq data structure
#                                * displacement is equal to one of the 2D vectors
#                                  f_k[0][i][j],...,f_k[k-1][i][j]
#                                * counter is a unique integer that is assigned to each tuple for
#                                  tie-breaking purposes (ie. in case there are two tuples with
#                                  identical priority in the heap)
#     - f_coord_dictionary:  A 2D array of dictionaries, represented as a list of lists of dictionaries.
#                            Specifically, f_coord_dictionary[i][j] should contain a dictionary
#                            entry for each displacement vector (x,y) contained in the heap f_heap[i][j]
#
# NOTE: This function should NOT check for duplicate entries or out-of-bounds vectors
# in the heap: it is assumed that the heap returned by this function contains EXACTLY k tuples
# per pixel, some of which MAY be duplicates or may point outside the image borders

def NNF_matrix_to_NNF_heap(source_patches, target_patches, f_k):

    f_heap = None
    f_coord_dictionary = None

    #############################################
    ###PLACEYOURCODEBETWEEN THESE LINES  ###
    #############################################
    heap,coord_dict = np.zeros((source_patches.shape[0:2])),np.zeros((source_patches.shape[0:2]))
    f_heap,f_coord_dictionary = heap.tolist(),coord_dict.tolist()
    for i in range(source_patches.shape[0]):
        for j in range(source_patches.shape[1]):
            f_heap[i][j] = []
            f_coord_dictionary[i][j] = {}
            for k in range(f_k.shape[0]):
                f_coord_dictionary[i][j][(f_k[k,i,j,0],f_k[k,i,j,1])] = 1
                one = -np.linalg.norm(source_patches[i,j]-target_patches[(i+f_k[k,i,j,0]),(j+f_k[k,i,j,1])])
                two =_tiebreaker.next()
                three =  (f_k[k,i,j,0],f_k[k,i,j,1])
                heappush(f_heap[i][j],(one,two,three))
    #############################################
    # NNF_heap_to_NNF_matrix(f_heap)
    return f_heap, f_coord_dictionary


# Given a 2D array of heaps given as input, this function creates a kxNxMx2
# matrix of nearest-neighbour fields
#
# The function takes only one input argument:
#     - f_heap:              A 2D array of heaps as described above. It is assumed that
#                            the heap of every pixel has exactly k elements.
# and has two return arguments
#     - f_k:                 A numpy array of dimensions kxNxMx2 that holds the k NNFs represented by the heap.
#                            Specifically, f_k[i] should be the NNF that contains the i-th best
#                            displacement vector for all pixels. Ie. f_k[0] is the best NNF,
#                            f_k[1] is the 2nd-best NNF, f_k[2] is the 3rd-best, etc.
#     - D_k:                 A numpy array of dimensions kxNxM whose element D_k[i][r][c] is the patch distance
#                            corresponding to the displacement f_k[i][r][c]
#

def NNF_heap_to_NNF_matrix(f_heap):

    #############################################
    ###  PLACE YOUR CODE BETWEEN THESE LINES  ###
    #############################################
    max_i,max_j,k = len(f_heap),len(f_heap[1]),len(f_heap[0][1])
    f_k,D_k = np.zeros((k,max_i,max_j,2),dtype=int),np.zeros((k,max_i,max_j))
    for i in range(max_i):
        for j in range(max_j):
            n_larges = nlargest(k,f_heap[i][j])
            # print(n_larges)
            # sys.exit()
            for l in range(k):
                f_k[l,i,j] = n_larges[l][2]
                D_k[l,i,j] = n_larges[l][0]
    ##############################################
    return f_k, -D_k
#
def nlm(target, f_heap, h):
    # this is a dummy statement to return the image given as input
    result = np.empty(target.shape)
    max_i,max_j,k = target.shape[0],target.shape[1],len(f_heap[0][0])
    # #############################################
    # ###  PLACE YOUR CODE BETWEEN THESE LINES  ###
    # #############################################
    for i in range(max_i):
        for j in range(max_j):
            normalizer = 0
            for l in range(k):
                heap = f_heap[i][j][l]
                summa = np.exp((heap[0])/(h**2))
                if(within_dim((heap[2][0]+i,heap[2][1]+j),target)):
                    result[i][j] = summa * target[heap[2][0]+i,heap[2][1]+j]
                normalizer += summa
            result[i][j] = result[i][j] / normalizer
    return result

#############################################
###  PLACE ADDITIONAL HELPER ROUTINES, IF ###
###  ANY, BETWEEN THESE LINES             ###
#############################################
def within_dim(x,y):
    return (x[0] < 0 or x[0] >= y.shape[0]) or (x[1] < 0 or x[1] >= y.shape[1])
#############################################



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
    rec_source = None
    ################################################
    ###  PLACE YOUR A3 CODE BETWEEN THESE LINES  ###
    ################################################
    y = make_coordinates_matrix(target.shape) + f
    result = target[y[:,:,0],f[:,:,1]]
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
