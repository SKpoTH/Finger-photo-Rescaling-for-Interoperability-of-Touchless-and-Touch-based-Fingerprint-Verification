import numpy as np
import math
import matplotlib.pyplot as plt

#<----------------> Import Local Libraries <----------------->
from modules.padding import valuePadding2D

class WindowPartition2D:
    def __init__(self, input_img, kernel_size, stride_size, pad_value=0):
        '''
            __Constructor__
        '''
        self.__input_img = input_img
        self.__KERNEL_SIZE = kernel_size
        self.__STRIDE_SIZE = stride_size
        self.__PAD_VALUE = pad_value
        ### -> Unmatched between "kernel_size" and "stride_size"
        if not (kernel_size%2 == stride_size%2):
            raise TypeError("Unmatched Odd/Even Error: Assign \"kernel_size\" and \"stride_size\" with Odd/Even match.")
        ### -> "input_img" properties
        self.__IMG_HEIGHT = input_img.shape[0]
        self.__IMG_WIDTH = input_img.shape[1]

    # def __completeWindow(self, size):
    #     '''
    #         Finding the expansion size that could complete a fragment
    #     '''
    #     ### -> Count How many windows gotten
    #     window_count = (size - self.__KERNEL_SIZE + self.__STRIDE_SIZE) / self.__STRIDE_SIZE

    #     ### -> If there is a Non-Complete Window, Calculate how many size should extend
    #     if not isinstance(window_count, int):
    #         window_count = math.ceil(window_count)
    #         padding_size = (window_count * self.__STRIDE_SIZE) - self.__STRIDE_SIZE + self.__KERNEL_SIZE - size

    #     return window_count, padding_size

    # def forwardTransform(self):
    #     '''
    #         Divide An Image into fragment following "KERNEL_SIZE" and "STRIDE_SIZE"
    #     '''
    #     ### -> Calculate - Different between "kernel_size" and "stride_size"
    #     self.__halfDiff_kernel_stride = int((self.__KERNEL_SIZE - self.__STRIDE_SIZE) / 2)
    #     self.__padding_y_size = self.__halfDiff_kernel_stride * 2
    #     self.__padding_x_size = self.__halfDiff_kernel_stride * 2

    #     ### -> Calculate - Number of Windows, Padding Size
    #     # -> Y-Dimension
    #     self.__window_y_count, padding_y_size = self.__completeWindow(self.__IMG_HEIGHT + self.__halfDiff_kernel_stride*2)
    #     self.__padding_y_size = self.__padding_y_size + padding_y_size
    #     # -> X-Dimension
    #     self.__window_x_count, padding_x_size = self.__completeWindow(self.__IMG_WIDTH + self.__halfDiff_kernel_stride*2)
    #     self.__padding_x_size = self.__padding_x_size + padding_x_size

    #     ### -> Padding for Complete Partialing
    #     padded_img = valuePadding2D(self.__input_img,  
    #                                 self.__IMG_HEIGHT + self.__padding_y_size,
    #                                 self.__IMG_WIDTH + self.__padding_x_size,
    #                                 pad_vlaue=self.__PAD_VALUE)

    #     ### -> Store All Partition Windows
    #     self.__forward_windows = []

    #     for y in range(self.__window_y_count):
    #         row_windows = []

    #         for x in range(self.__window_x_count):
    #             partial_img = padded_img[y*self.__STRIDE_SIZE:self.__KERNEL_SIZE+(y*self.__STRIDE_SIZE), \
    #                                      x*self.__STRIDE_SIZE:self.__KERNEL_SIZE+(x*self.__STRIDE_SIZE)]
    #             # -> Each Row
    #             row_windows.append(partial_img)
    #         # -> Stack Row
    #         self.__forward_windows.append(row_windows)

    def __completeWindow(self, size):
        '''
            Finding the expansion size that could complete a fragment
        '''
        ### -> Count How many windows gotten
        window_count = size / self.__STRIDE_SIZE

        ### -> If there is a Non-Complete Window, Calculate how many size should extend
        if not isinstance(window_count, int):
            window_count = math.ceil(window_count)
            extend_size = (window_count * self.__STRIDE_SIZE) - size
        else:
            extend_size = 0

        return window_count, extend_size

    def forwardTransform(self):
        '''
            Divide An Image into fragment following "KERNEL_SIZE" and "STRIDE_SIZE"
        '''
        ### -> Calculate - Different between "kernel_size" and "stride_size"
        self.__halfDiff_kernel_stride = int((self.__KERNEL_SIZE - self.__STRIDE_SIZE) / 2)

        ### -> Calculate - Number of Windows, Padding Size
        # -> Y-Dimension
        self.__window_y_count, extend_y_size = self.__completeWindow(self.__IMG_HEIGHT)
        self.__padding_y_size = (self.__halfDiff_kernel_stride * 2) + extend_y_size
        # -> X-Dimension
        self.__window_x_count, extend_x_size = self.__completeWindow(self.__IMG_WIDTH)
        self.__padding_x_size = (self.__halfDiff_kernel_stride * 2) + extend_x_size

        ### -> Padding for Complete Partialing
        padded_img = valuePadding2D(self.__input_img,  
                                    self.__IMG_HEIGHT + self.__padding_y_size,
                                    self.__IMG_WIDTH + self.__padding_x_size,
                                    pad_vlaue=self.__PAD_VALUE)

        ### -> Store All Partition Windows
        self.__forward_windows = []

        for y in range(self.__window_y_count):
            row_windows = []

            for x in range(self.__window_x_count):
                partial_img = padded_img[y*self.__STRIDE_SIZE:self.__KERNEL_SIZE+(y*self.__STRIDE_SIZE), \
                                         x*self.__STRIDE_SIZE:self.__KERNEL_SIZE+(x*self.__STRIDE_SIZE)]
                # -> Each Row
                row_windows.append(partial_img)
            # -> Stack Row
            self.__forward_windows.append(row_windows)
    
    def backwardTransform(self):
        '''
            Combine The Fragments Window into Orignal "input_img" size
        '''
        ### -> Create inverse fragments (window size as "STRIDE_SIZE")
        self.__backward_windows = []

        for y in range(self.__window_y_count):
            row_windows = []

            for x in range(self.__window_x_count):
                partial_img = self.__forward_windows[y][x]
                # -> Trim Overlapping
                trim_img = partial_img[self.__halfDiff_kernel_stride:self.__halfDiff_kernel_stride+self.__STRIDE_SIZE, \
                                       self.__halfDiff_kernel_stride:self.__halfDiff_kernel_stride+self.__STRIDE_SIZE]
                # -> Each Row
                row_windows.append(trim_img)
            # -> Stack Row
            self.__backward_windows.append(row_windows)

        ### -> Define Backward Transform output
        backward_height = self.__window_y_count * self.__STRIDE_SIZE
        backward_width = self.__window_x_count * self.__STRIDE_SIZE
        ### -> Create Dummy-Output Backward Transform
        backward_img = np.zeros((backward_height, backward_width))
        
        ### -> Combine Fragments windows into ONE-FULL-Window
        for y in range(self.__window_y_count):
            for x in range(self.__window_x_count):
                backward_img[y*self.__STRIDE_SIZE:y*self.__STRIDE_SIZE+self.__STRIDE_SIZE, \
                             x*self.__STRIDE_SIZE:x*self.__STRIDE_SIZE+self.__STRIDE_SIZE] = self.__backward_windows[y][x]

        ### -> Trim Padding Out
        self.__output_img = trimPadding2D(backward_img, self.__IMG_WIDTH, self.__IMG_HEIGHT)

        # plt.subplot(1,2,1)
        # plt.imshow(self.__input_img)
        # plt.subplot(1,2,2)
        # plt.imshow(self.__output_img)
        # plt.show()

    # <--------------------> API Function <-------------------->
    # --- GET API ---
    def getForwardWindows(self):
        '''
            Give A list of Fragmented Windows
        '''
        return self.__forward_windows

    def getBackwardWindows(self):
        '''
            Give A list of Backward Windows
        '''
        return self.__backward_windows

    def getOutputImage(self):
        '''
            Give Backward Output
        '''
        return self.__output_img

    def getWindowXYcount(self):
        '''
            Give counting of X-count window and Y-count window
        '''
        return self.__window_y_count, self.__window_x_count

    # --- SET API ---
    def setFragmentedWindow(self, fragmented_list):
        '''
            Receive A list of Fragmented Windows
        '''
        self.__forward_window = fragmented_list

    # <--------------------> Display Function <-------------------->
    # def windowDisplayAll(self):
    #     '''
    #         Display All Partial Window in 2D-Grid Form
    #     '''
    #     flatten_partial_list = [x for row in self.__forward_window for x in row]
    #     imageDisplay(self.__window_y_count, self.__window_x_count, flatten_partial_list)
    
    # def backwardDisplay(self):
    #     '''
    #         Display All Partial Window in 2D-Grid Form
    #     '''
    #     flatten_partial_list = [x for row in self.__backward_window for x in row]
    #     imageDisplay(self.__window_y_count, self.__window_x_count, flatten_partial_list)

        