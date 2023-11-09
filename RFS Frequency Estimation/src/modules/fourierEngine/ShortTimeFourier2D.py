import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from tqdm import tqdm
#<----------------> Import Local Libraries <----------------->
from modules.fourierEngine import Fourier2D
from modules.windowPartition import WindowPartition2D

class ShortTimeFourier2D:
    def __init__(self, input_img, window_params, pad_value=0):
        '''
            __Constructor__
        '''
        self.__input_img = input_img
        self.__KERNEL_SIZE = window_params[0]
        self.__STRIDE_SIZE = window_params[1]
        self.__PAD_VALUE = pad_value

    # <--------------------------> Window Partition Function <-------------------------->
    def __splitWindow(self):
        '''
            Split Full-2D Image into Fragmented Window
        '''
        self.__WindowPartition = WindowPartition2D(self.__input_img, 
                                                   self.__KERNEL_SIZE, 
                                                   self.__STRIDE_SIZE,
                                                   self.__PAD_VALUE)
        self.__WindowPartition.forwardTransform()

        ### -> Get Partition Windows and Window Count
        short_time_windows = self.__WindowPartition.getForwardWindows()
        WINDOW_Y_COUNT, WINDOW_X_COUNT = self.__WindowPartition.getWindowXYcount()

        return short_time_windows, WINDOW_Y_COUNT, WINDOW_X_COUNT

    def __mergeWindow(self):
        '''
            Merge Fragmented Window into Full-2D Image
        '''
        self.__WindowPartition.setFragmentedWindow(self.__invert_fourier_array)
        self.__WindowPartition.backwardTransform()
        self.__output_img = self.__WindowPartition.getOutputImage()
        
    # <--------------------------> Fourier Transform Function <-------------------------->
    def fft(self):
        '''
            Fourier Transform Each Fragmented Window
        '''
        ### -> Windows Partitioning
        self.__short_time_windows, self.__WINDOW_Y_COUNT, self.__WINDOW_X_COUNT = self.__splitWindow()
        # -> Create empty fourier storage
        self.__fourier_grid = []

        ### -> FFT each windows
        for y in range(self.__WINDOW_Y_COUNT):
            row_list = []
            
            for x in range(self.__WINDOW_X_COUNT):
                Fourier = Fourier2D(self.__short_time_windows[y][x])
                Fourier.fft()
                # -> Each Column
                row_list.append(Fourier)
            # -> Each Row
            self.__fourier_grid.append(row_list)

    def ifft(self):
        '''
            Invert Fourier Transform Each Fragmented Window
        '''
        ### -> Create empty spatial storage
        self.__inv_fourier_grid = []

        ### -> IFFT each windows
        for y in range(self.__WINDOW_Y_COUNT):
            row_list = []

            for x in range(self.__WINDOW_X_COUNT):
                Fourier = self.__fourier_grid[y][x]
                Fourier.ifft()
                # -> Each Column
                row_list.append(Fourier.getOutputImage())
            # -> Each Row
            self.__inv_fourier_grid.append(row_list)

    # <---------------------------------> API Function <--------------------------------->
    # --- GET API ---
    def getFourierGrid(self):
        '''
            Give "fourier_array" value
        '''
        return self.__fourier_grid

    def getOutputImage(self):
        '''
            Give Invert Short-Time Fourier
        '''
        return self.__output_img

    def getShotTimeWindows(self):
        '''
            Give "short_time_windows" value (spatial windows)
        '''
        return self.__short_time_windows

    # <-------------------------------> Display Function <------------------------------->
    def showSpatial(self, save=False):
        '''
            Display Partitioning Short Time Window
        '''
        ### -> Create empty diplay image
        show_windows = np.zeros((self.__WINDOW_Y_COUNT*self.__KERNEL_SIZE, self.__WINDOW_X_COUNT*self.__KERNEL_SIZE))

        ### -> Tiling each Windows
        for y in range(self.__WINDOW_Y_COUNT):
            for x in range(self.__WINDOW_X_COUNT):
                show_windows[y*self.__KERNEL_SIZE:(y+1)*self.__KERNEL_SIZE, 
                             x*self.__KERNEL_SIZE:(x+1)*self.__KERNEL_SIZE] = self.__short_time_windows[y][x]

        ### ~> Plot Subbands
        plt.imshow(show_windows, cmap="gray")
        if not save:
            plt.show()

    def showMagnitude(self, save=False):
        '''
            Display Short-time Fourier Trasnform
        '''
        ### -> Create empty diplay image
        show_fouriers = np.zeros((self.__WINDOW_Y_COUNT*self.__KERNEL_SIZE, self.__WINDOW_X_COUNT*self.__KERNEL_SIZE))

        ### -> Tiling each Fouriers
        for y in range(self.__WINDOW_Y_COUNT):
            for x in range(self.__WINDOW_X_COUNT):
                show_fouriers[y*self.__KERNEL_SIZE:(y+1)*self.__KERNEL_SIZE, 
                              x*self.__KERNEL_SIZE:(x+1)*self.__KERNEL_SIZE] = self.__fourier_grid[y][x].getMagnitude()
                # show_fouriers[y*self.__KERNEL_SIZE:(y+1)*self.__KERNEL_SIZE, 
                #               x*self.__KERNEL_SIZE:(x+1)*self.__KERNEL_SIZE] = self.__fourier_grid[y][x].centerBanning()
                # plt.imshow(self.__fourier_grid[y, x, :, :]**2, cmap="hot")
                # plt.show()
        ### ~> Plot Subbands
        plt.imshow(show_fouriers, cmap="hot")
        if not save:
            plt.show()

                
        

    

        


        