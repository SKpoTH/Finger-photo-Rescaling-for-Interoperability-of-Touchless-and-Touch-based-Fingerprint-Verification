import math
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import time
from tqdm import tqdm

class ReArrangeSubBands:
    def __init__(self, fourier_grid, freq_paras):
        '''
            __Constructor__
        '''
        self.__fourier_grid = fourier_grid
        # -> Window Partition Parameters
        self.__KERNEL_SIZE = fourier_grid[0][0].getMagnitude().shape[0]
        # -> Frequency Subbands Parameters
        self.__BAND_WIDTH = freq_paras[0]
        self.__STRIDE_WIDTH = freq_paras[1]
        # -> Window Count
        self.__WINDOW_Y_COUNT = len(fourier_grid)
        self.__WINDOW_X_COUNT = len(fourier_grid[0])

    # <--------------------> ReArrange Subbands Function <---------------------->
    def __magnitudeGrid(self):
        '''
            Get Magnitude Grid from "fourier_grid"
        '''
        # -> Create empty magnitude grid
        magnitude_grid = np.zeros((self.__WINDOW_Y_COUNT, self.__WINDOW_X_COUNT, self.__KERNEL_SIZE, self.__KERNEL_SIZE))
        
        for y in range(self.__WINDOW_Y_COUNT):
            for x in range(self.__WINDOW_X_COUNT):
                # magnitude = self.__fourier_grid[y][x].getMagnitude()
                magnitude = self.__fourier_grid[y][x].centerBanning()
                magnitude_grid[y, x, :, :] = magnitude
        
        return magnitude_grid

    def reArrangeSubbands(self):
        '''
            ReArrange Subbands of Transformed Magnitude
        '''
        ### -> Get Magnitude Grid
        magnitude_grid = self.__magnitudeGrid()

        ### -> Rearrange
        self.__rearrange_subbands = magnitude_grid ** 2
        self.__rearrange_subbands = self.__rearrange_subbands.swapaxes(0, 2)
        self.__rearrange_subbands = self.__rearrange_subbands.swapaxes(1, 3)

    # <--------------------> Frequency Subbands Function <-------------------->
    def __createRing(self, start_radius, stop_radius):
        '''
            Create Ideal Band-pass Filter (Ring)
        '''
        # -> Create empty Ring image
        ring_img = np.zeros((self.__KERNEL_SIZE, self.__KERNEL_SIZE))
        ring_img = np.ascontiguousarray(ring_img)
        # -> Center of Ring
        center_pos = (ring_img.shape[1] // 2, ring_img.shape[0] // 2)
        # -> Draw Ring
        ring_img = cv.circle(ring_img, center_pos, stop_radius, 1, -1)
        ring_img = cv.circle(ring_img, center_pos, start_radius, 0, -1)

        return ring_img

    def freqSubbands(self):
        '''
            Frequency Subbands from ReArrange Subbands 
        '''
        # -> Cover Ring Radius
        cover_r = self.__KERNEL_SIZE // 2
        # -> Frequency Subbands
        self.__freq_subbands = np.zeros((self.__WINDOW_Y_COUNT, self.__WINDOW_X_COUNT, cover_r // self.__STRIDE_WIDTH))

        ### => Iterate each Frequency
        for i, radius in enumerate(range(0, cover_r, self.__STRIDE_WIDTH)):
            # -> Ring Parameters
            start_radius = radius
            # center_radius = radius + (self.__BAND_WIDTH // 2)
            stop_radius = radius + self.__BAND_WIDTH
            # - > Create Ring
            ring_img = self.__createRing(start_radius, stop_radius)
            # -> Sum all Theta Response Subbands
            response_idx = np.argwhere(ring_img == 1)
            freq_band = np.sum(self.__rearrange_subbands[response_idx[:, 0], response_idx[:, 1], :, :], axis=0)
            # -> Store Frequency Subband
            self.__freq_subbands[:, :, i] = freq_band

    # <-----------------------------> API Function <---------------------------->
    # --- GET API ---
    def getRearrangeSubbands(self):
        '''
            Get "rearrange_subbands" value
        '''
        return self.__rearrange_subbands

    def getFreqSubbands(self):
        '''
            Give "freq_subbands" value
        '''
        return self.__freq_subbands

    # <-----------------> Display/Save Visualization Function <----------------->
    def showReArrangeSubbands(self, save=False):
        '''
            Display Rearrange Subbands
        '''
        ### -> Create empty diplay image
        show_subbands = np.zeros((self.__WINDOW_Y_COUNT*self.__KERNEL_SIZE, self.__WINDOW_X_COUNT*self.__KERNEL_SIZE))

        ### -> Tiling each Subbands
        for y in range(self.__KERNEL_SIZE):
            for x in range(self.__KERNEL_SIZE):
                show_subbands[y*self.__WINDOW_Y_COUNT:(y+1)*self.__WINDOW_Y_COUNT, 
                              x*self.__WINDOW_X_COUNT:(x+1)*self.__WINDOW_X_COUNT] = self.__rearrange_subbands[y, x, :, :]
        ### ~> Plot Subbands
        plt.imshow(show_subbands, cmap="hot")
        if not save:
            plt.show()

    def showFreqSubbands(self, rows=4, save=False):
        '''
            Display Frequency Subbands
        '''
        ### -> Create empty diplay image
        num_bands = self.__freq_subbands.shape[2]
        cols = num_bands // rows
        show_subbands = np.zeros((self.__WINDOW_Y_COUNT * rows, self.__WINDOW_X_COUNT * cols))
        ### -> Tiling each Frequency Subbands
        for i in range(rows):
            for j in range(cols):
                show_subbands[i*self.__WINDOW_Y_COUNT:(i+1)*self.__WINDOW_Y_COUNT,
                              j*self.__WINDOW_X_COUNT:(j+1)*self.__WINDOW_X_COUNT] = self.__freq_subbands[:, :, (i*cols)+j]
        ### ~> Plot Subbands
        plt.imshow(show_subbands, cmap="hot")
        if not save:
            plt.show()