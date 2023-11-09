import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

from modules.fourierEngine import ShortTimeFourier2D, ReArrangeSubBands

def fourierFeatures(input_img, window_params, freq_params, pad_value):
    '''
        Features Extraction with Fourier Transformation
    '''
    ### => Short-Time Foruier Transformation
    STFT = ShortTimeFourier2D(input_img, window_params, pad_value)
    STFT.fft()
    # - Get Short-Time Fourier
    fourier_grid = STFT.getFourierGrid()
    # STFT.showSpatial()
    # STFT.showMagnitude()
    
    ### => Rearrange Subbands
    RASB = ReArrangeSubBands(fourier_grid, freq_params)
    RASB.reArrangeSubbands()
    # - Frequency Subbands
    RASB.freqSubbands()
    # - Get Subbands
    freq_subbands = RASB.getFreqSubbands()
    # RASB.showReArrangeSubbands()
    # RASB.showFreqSubbands()

    return freq_subbands






