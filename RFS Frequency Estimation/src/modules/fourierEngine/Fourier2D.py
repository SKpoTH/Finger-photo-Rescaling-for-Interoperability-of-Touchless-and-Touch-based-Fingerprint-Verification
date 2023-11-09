#<----------------> Import OutSource Libraries <----------------->
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from scipy import fftpack
from tqdm import tqdm

#<----------------> Import Local Libraries <----------------->

class Fourier2D:
    def __init__(self, input_img, window_func=None, zero_mean=True):
        '''
            __Constructor__
        '''
        self.__input_img = input_img
        self.__window_func = window_func
        self.__zero_mean = zero_mean
        # -> Get image size
        self.__IMG_HEIGHT = input_img.shape[0]
        self.__IMG_WIDTH = input_img.shape[1]

    # <--------------------> Fourier Transform Pre-processing <-------------------->
    def __zeroMean(self, input_img):
        '''
            Zero Mean - Subtract by mean
        '''
        output_img = input_img = input_img - input_img.mean()

        return output_img

    def __windowFunction(self, input_img):
        '''
            Window Function - Covert "input_img"
        '''
        if self.__window_func == "Gaussian":
            # -> Find Standard Deviation from Kernel Size
            gauss_std = lambda kernel_size : 0.3*((kernel_size-1)*0.5)+0.8
            # -> 1D Window Function
            winfunc_vert = scipy.signal.windows.gaussian(self.__IMG_HEIGHT, gauss_std(self.__IMG_HEIGHT))
            winfunc_horz = scipy.signal.windows.gaussian(self.__IMG_WIDTH, gauss_std(self.__IMG_WIDTH))
            # -> 2D Window Function
            window_function = winfunc_horz.reshape((1, self.__IMG_WIDTH)) * winfunc_vert.reshape((1, self.__IMG_HEIGHT)).T
        else:
            # -> None Window Function
            window_function = np.ones((self.__IMG_HEIGHT, self.__IMG_WIDTH))
        # -> Take Window Function
        output_img = input_img * window_function

        return output_img

    # <-----------------------> Fourier Transform Function <----------------------->
    # - Chawin KHONGPRASONGSIRI เปิด Google drive เอาภาพ project ไม่ได้
    def fft(self):
        '''
            2D-Fast Fourier Transform (by SciPy)
        '''
        input_img = self.__input_img.copy()
        # -> Fourier Transform Pre-processing
        zero_img = self.__zeroMean(input_img)
        winfunc_img = self.__windowFunction(zero_img)
        # -> Forward 2D-Fourier Transform
        fft_complex = fftpack.fft2(winfunc_img)
        # -> Split Magnitude, Phase consequently
        self.__fft_magnitude = abs(fft_complex)
        self.__fft_phase = np.arctan2(fft_complex.imag, fft_complex.real)
        # -> Shift Quardrant
        self.__fft_magnitude = fftpack.fftshift(self.__fft_magnitude)
    
    def ifft(self):
        '''
            2D-Invert Fast Fourier Transform (by SciPy)
        '''
        # -> Shift Quardrant back to Spatial Standard format
        fft_magnitude = fftpack.ifftshift(self.__fft_magnitude)
        # -> Tranform Magintude, Phase back to 2D-Signal form and combine as Complex
        img_real = fft_magnitude * np.cos(self.__fft_phase)
        img_imag = fft_magnitude * np.sin(self.__fft_phase)
        img_complex = img_real + img_imag * 1j
        # -> Take 2D-Invert Fourier Transform
        output_img = fftpack.ifft2(img_complex)
        # -> Output use only -> Real part
        self.__output_img = output_img.real

    def centerBanning(self, ban_size=0):
        '''
            Ban the center of "fft_magnitude" to ignore low frequency (unsignificant) components
        '''
        fft_magnitude = self.__fft_magnitude.copy()
        ### -> Create Low-pass Filter
        center_ban = np.ones_like(fft_magnitude)
        center_ban = np.ascontiguousarray(center_ban)
        # -> Define "ban_size"
        if ban_size <= 0:
            ban_size = min([center_ban.shape[0], center_ban.shape[1]])
            ban_size = int(ban_size * 0.04)
        # -> Draw Ideal Circle
        center = (center_ban.shape[1]//2, center_ban.shape[0]//2)
        center_ban = cv.circle(center_ban, center, ban_size, 0, -1)
        
        ### -> Banning Center
        fft_magnitude = fft_magnitude * center_ban

        return fft_magnitude

    # <-----------------------------> API Function <------------------------------>
    # ---> GET API
    def getMagnitude(self):
        '''
            Get Magnitude after Fourier Transform
        '''
        return self.__fft_magnitude

    def getOutputImage(self):
        '''
            Get Output image after Inverse Fourier Transform
        '''
        return self.__output_img

    # ---> SET API
    def setMagnitude(self, fft_magnitude):
        '''
            Set "fft_magnitude" value
        '''
        self.__fft_magnitude = fft_magnitude

    # <---------------------------> Display Function <--------------------------->
    def showMagnitude(self, method="LOG-SCALE", ban_size=0, save=False):
        '''
            Display Magnitude (Colormap & Take Log-Scale -> Cleary Visualization)
        '''
        fft_magnitude = self.__fft_magnitude.copy()

        if method == "LOG-SCALE":
            ### -> Scale spectrum into Colormap Range & set into Logarithm Scaling
            fft_magnitude = (fft_magnitude-fft_magnitude.min()) / (fft_magnitude.max()-fft_magnitude.min())
            fft_magnitude = np.log(1 + fft_magnitude)
        elif method == "CENTER-BAN":
            ### -> Ban Center of Magnitude that has outstanding high value
            fft_magnitude = self.centerBanning()
            fft_magnitude = (fft_magnitude-fft_magnitude.min()) / (fft_magnitude.max()-fft_magnitude.min())
            fft_magnitude = np.log(1 + fft_magnitude)
        # -> Display Maginitude Spectrum
        plt.subplot(1, 2, 1)
        plt.imshow(self.__input_img, cmap="gray")
        plt.subplot(1, 2, 2)
        plt.imshow(fft_magnitude, cmap='hot')
        if not save:
            plt.show()

    def saveMagnitude(self, output_dst, ouptut_filename, method="LOG-SCALE"):
        '''
            Save Display Magnitude
        '''
        self.showMagnitude(method=method ,save=True)
        # -> Set Destination
        mkdirRecursion(output_dst)
        # -> Save Grid
        plt.savefig(output_dst + output_filename + ".png", bbox_inches='tight', dpi=750)
        plt.close()

    # <---------------------------> Save Parameters <--------------------------->
    def saveArray(self, output_folder, output_filename):
        '''
            Save Magnitude Array
        '''
        # -> Create Destination Folder
        mkdirRecursion(output_folder)
        # -> Banning Center DC freq
        center_banned = self.centerBanning()
        # -> Save array
        np.save(output_folder + output_filename, center_banned)

        
