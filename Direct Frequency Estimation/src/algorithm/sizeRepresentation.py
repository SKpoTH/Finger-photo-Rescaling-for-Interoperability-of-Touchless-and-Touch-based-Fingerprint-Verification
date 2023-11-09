import numpy as np
import cv2 as cv
import scipy
import seaborn as sns
import matplotlib.pyplot as plt

def convertRange(input_array, min_val, max_val):
    '''
        Convert any value range array into specifc range value array
    '''
    # -> Convert into [0, 1]
    norm_array = (input_array - input_array.min()) / (input_array.max() - input_array.min())
    # -> Convert [0, 1] into [min_val, max_val]
    output_array = (norm_array * (max_val - min_val)) + min_val

    return output_array

def sizeRepresentation(freq_filters, peak_cutoff=0.0):
    '''
        Ridge Size Representation
    '''
    ### -> Find Local Peak of Energy
    peaks, _ = scipy.signal.find_peaks(freq_filters, height=peak_cutoff*freq_filters.max())
    ### -> Frequency Resolution Representation
    freqRes_rep = np.mean(peaks)

    # ### ~> Display
    # plt.bar(np.arange((len(freq_filters))), freq_filters)
    # plt.plot(peaks, freq_filters[peaks], color="red", marker="x", linestyle="None")
    # plt.yscale("log")
    # # ~> Label
    # plt.title("Fingerphoto")
    # plt.xlabel("Size")
    # plt.ylabel("Energy")
    # # ~> Grid
    # plt.minorticks_on()
    # plt.grid(which='major', linestyle='-', linewidth='0.5', color='gray')
    # plt.grid(which='minor', linestyle=':', linewidth='0.5', color='gray')
    # plt.grid("on")
    # plt.show()

    return freqRes_rep

