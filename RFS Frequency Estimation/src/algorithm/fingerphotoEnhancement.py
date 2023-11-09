import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

def blurImage(input_img, blurSize_ratio):
    '''
        Blur for suppressing noise, the size is referent to input_img size
    '''
    # -> Set Gaussian Window Size
    gauss_size = int(blurSize_ratio * min(input_img.shape))
    if gauss_size % 2 == 0:
        gauss_size += 1
    # -> Apply Gaussian Blur (Sigma distribute fit to window size)
    blur_img = cv.GaussianBlur(input_img, (gauss_size, gauss_size), 0)

    return blur_img 

def differentOfGaussian(input_img, blurSize_ratio):
    '''
        Find the Different between "input_img" and 
        "gauss_img" (the input that filtered by gaussian filter)
    '''
    # -> Set Gaussian Window Size
    gauss_img = blurImage(input_img, blurSize_ratio)
    # -> Sharpening by Subtract with Low-passed image
    dog_img = input_img.astype(float) - gauss_img.astype(float)    # to "float" prevent the overflowing

    return dog_img
    
def convertRange(input_array, min_val, max_val):
    '''
        Convert any value range array into specifc range value array
    '''
    # -> Convert into [0, 1]
    norm_array = (input_array - input_array.min()) / (input_array.max() - input_array.min())
    # -> Convert [0, 1] into [min_val, max_val]
    output_array = (norm_array * (max_val - min_val)) + min_val

    return output_array

def clahe(input_img, window_params, clip_ratio):
    '''
        Contrast Limited Adaptive Histogram Equalization
        adaptive with "WINDOW_SIZE"
    '''
    # -> Window parameter for adaptive window partition
    WINDOW_SIZE, STRIDE_SIZE = window_params
    # -> Number of row and column for specific "WINDOW_SIZE"
    # tile_row = 1 if input_img.shape[0]//WINDOW_SIZE == 0 else tile_row = input_img.shape[0]//WINDOW_SIZE
    # tile_col = 1 if input_img.shape[1]//WINDOW_SIZE == 0 else tile_col = input_img.shape[1]//WINDOW_SIZE
    tile_row = (lambda: 1, lambda: input_img.shape[0]//WINDOW_SIZE)[input_img.shape[0] // WINDOW_SIZE == 0]()
    tile_col = (lambda: 1, lambda: input_img.shape[1]//WINDOW_SIZE)[input_img.shape[1] // WINDOW_SIZE == 0]()

    # -> Clip Limit, ratio refernet to "WINDOW_SIZE"
    clip_limit = (WINDOW_SIZE * WINDOW_SIZE) * clip_ratio
    # -> CLAHE
    CLAHE = cv.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_col, tile_row))
    clahe_img = CLAHE.apply(input_img)

    return clahe_img

def fingerphotoEnhancement(input_img, mask_img, window_params):
    '''
        "Main" : Fingerphoto Enhancement to boost Fingerprint Detail
    '''
    ### => Different of Gaussian
    dog_img = differentOfGaussian(input_img, blurSize_ratio=0.05)
    ### => Convert to [0, 255]
    dog_img = convertRange(dog_img, 0, 255).astype(np.uint8)
    ### => Blur image to suppress noise
    dog_img = blurImage(dog_img, blurSize_ratio=0.02)
    ### => apply CLAHE
    clahe_img = clahe(dog_img, window_params, clip_ratio=0.06)
    ### => Invert Ridge <-> Valley intensity
    inv_img = 255 - clahe_img
    ### => Convert to [-1, 1]
    inv_img = convertRange(inv_img, -1, 1)
    ### => Masking Fingertip
    ench_img = inv_img * mask_img

    # # ~> Display
    # plt.subplot(1, 2, 1)
    # plt.imshow(input_img, cmap="gray")
    # plt.subplot(1, 2, 2)
    # plt.imshow(ench_img, cmap="gray")
    # plt.show()

    return ench_img
    


