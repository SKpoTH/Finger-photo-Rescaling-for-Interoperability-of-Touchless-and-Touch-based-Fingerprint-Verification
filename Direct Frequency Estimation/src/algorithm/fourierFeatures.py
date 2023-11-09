import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

from modules.fourierEngine import Fourier2D

def createRing(IMG_HEIGHT, IMG_WIDTH, start_radius, stop_radius):
    '''
        Create Ideal Band-pass Filter (Ring)
    '''
    # -> Create empty Ring image
    ring_img = np.zeros((IMG_HEIGHT, IMG_WIDTH))
    ring_img = np.ascontiguousarray(ring_img)
    # -> Center of Ring
    center_pos = (ring_img.shape[1] // 2, ring_img.shape[0] // 2)
    # -> Draw Ring
    ring_img = cv.circle(ring_img, center_pos, stop_radius, 1, -1)
    ring_img = cv.circle(ring_img, center_pos, start_radius, 0, -1)

    return ring_img

def freqFilters(input_img, BAND_WIDTH, STRIDE_WIDTH):
    '''
        Frequency Subbands from ReArrange Subbands 
    '''
    IMG_HEIGHT, IMG_WIDTH = input_img.shape

    # -> Cover Ring Radius
    cover_r = min(input_img.shape) // 2
    # -> Frequency Subbands
    freq_filters = np.zeros((cover_r // STRIDE_WIDTH)+1)

    ### => Iterate each Frequency
    for i, radius in enumerate(range(0, cover_r, STRIDE_WIDTH)):
        # -> Ring Parameters
        start_radius = radius
        # center_radius = radius + (self.__BAND_WIDTH // 2)
        stop_radius = radius + BAND_WIDTH
        # - > Create Ring
        ring_img = createRing(IMG_HEIGHT, IMG_WIDTH, start_radius, stop_radius)
        # -> Sum all Energy in Frequency Filter
        energy = np.sum((input_img * ring_img) ** 2)
        freq_filters[i] = energy

    return freq_filters

def fourierFeatures(input_img):
    '''
        Features Extraction with Fourier Transformation
    '''
    ### => Short-Time Foruier Transformation
    FFT = Fourier2D(input_img, window_func="Gaussian")
    FFT.fft()
    # - Get Fourier
    magnitude = FFT.centerBanning(hp_filter="Gaussian")

    ### => Frequency Filters
    BAND_WIDTH = int(min(magnitude.shape)*0.01)
    STRIDE_WIDTH = BAND_WIDTH // 2
    freq_filters = freqFilters(magnitude, BAND_WIDTH, STRIDE_WIDTH)
    
    return freq_filters






