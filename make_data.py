"""Do data exploration and visualization of the turbulence dataset"""

import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt

import pdb

BATCH_SIZE = 64
WIDTH = 64
HEIGHT = 64

def make_batches(data, width, height, batch_size=64):
    """
    Takes a list of image-sets of shape (H, W, N) and chops each image into sub-images of
    the specified width and height, stacked in a depth direction and grouped into batches
    in the first direction.  The output has shape (N_batches, batch_size, 1, height, width)
    
    Parameters:
    -----------
    data: a list of image-sets each of shape (H, W, N)
    width: width of the sub-images to break the large images into
    height: height of the sub-images to break the large images into
    batch_size: the size of each batch

    Returns:
    ------------
    ndarray of shape (N_batches, batch_size, 1, height, width)
    """
    img = None
    for img_set in data:
        H, W, N = img_set.shape
        n_h = H // height
        n_w = W // width
        num_batches = N*n_h*n_w // batch_size
        img_reshaped = img_set.transpose(2, 0, 1).reshape(N, n_h, height, n_w, width)
        stacked = img_reshaped.transpose(0, 1, 3, 2, 4).reshape(num_batches, batch_size, 1, height, width)
        img = stacked if img is None else np.concatenate((img, stacked), axis=0)
    return img

def make_shifted_copies(data, h_shift, w_shift):
    """ 
    Takes in a tuple of image-sets each of shape (H, W, N) and makes copies of each
    image-set after shifting it left, right, and both left and right.  Each of these
    shifted images is added to the original tuple of image-sets, and this augmented tuple
    is returned.
    """
    shifted = tuple() 
    for img in data:
        shift_down = np.concatenate((img[h_shift:, :, :], img[:h_shift, :, :]), axis=0)
        shift_right = np.concatenate((img[:, w_shift:, :], img[:, w_shift:, :]), axis=1)
        shift_both = np.concatenate((shift_down[:, w_shift:, :], shift_down[:, w_shift:, :]), axis=1)
        shifted += (img, shift_down, shift_right, shift_both)
    return shifted

def main():
    # Load the data
    mat_dict = sio.loadmat('hw2q11')
    midplane = mat_dict['theta_midplane']
    nearwall = mat_dict['theta_nearwall']

    # Process the data
    data = (midplane, nearwall)
    augmented = make_shifted_copies(data, HEIGHT // 2, WIDTH // 2)
    batches = make_batches(augmented, WIDTH, HEIGHT)
    batches -= batches.min()
    batches /= batches.max()

    # save the processed and batches data
    np.save('turb_batches', batches)


if __name__ == "__main__":
    main()
