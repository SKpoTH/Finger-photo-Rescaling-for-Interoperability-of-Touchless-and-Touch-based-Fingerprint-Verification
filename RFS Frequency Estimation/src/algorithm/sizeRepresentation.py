import numpy as np
import cv2 as cv
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

def respondFreq(freq_subbands, weak_cutoff=0.5):
    '''
        Select the most respond frequency subbands
    '''
    # -> Frequency Subband Structure Shape
    WINDOW_Y_COUNT, WINDOW_X_COUNT, SUBBANDS = freq_subbands.shape
    # -> Clone frequency subbands and add zero subband offset
    freqRes_subbands = np.zeros((WINDOW_Y_COUNT, WINDOW_X_COUNT, SUBBANDS+1))
    freqRes_subbands[:,:,1:] = freq_subbands
    # -> Mask out for weak reponse
    freq_mask = np.where(freqRes_subbands >= weak_cutoff, 1, 0)
    # -> Get Peak Reponse (if all value in subbands axis = 0, then argmax will be index 0)
    freq_response = np.argmax(freqRes_subbands*freq_mask, axis=2)
    freq_response -= 1        # - Shift offset out ((-1) = Ignore value)

    return freq_response

def medianAlpha(freq_response):
    '''
        Find median that is not included -1 (ignore index)
    '''
    # -> Mask out -1 (ignore index) 
    freq_response = np.ma.masked_where(freq_response == -1, freq_response)
    # freq_response = np.ma.masked_where(freq_response == freq_response.min(), freq_response)
    # -> Find Median
    freqRes_med = np.ma.median(freq_response)

    return freqRes_med

def meanAlpha(freq_response):
    '''
        Find median that is not included -1 (ignore index)
    '''
    # -> Mask out -1 (ignore index) 
    freq_response = np.ma.masked_where(freq_response == -1, freq_response)
    # freq_response = np.ma.masked_where(freq_response == freq_response.min(), freq_response)
    # -> Find Median
    freqRes_mean = np.ma.mean(freq_response)

    return freqRes_mean

def sizeRepresentation(input_img, freq_subbands, weak_cutoff=0.5):
    '''
        Ridge Size Representation
    '''
    ### => Convert into [0, 1]
    freq_subbands = convertRange(freq_subbands, 0, 1)
    ### => Max Repond Frequency of each Window
    freq_response = respondFreq(freq_subbands, weak_cutoff=weak_cutoff)
    ### => Find Median (not included ignore index)
    # freqRes_rep = medianAlpha(freq_response)
    freqRes_rep = meanAlpha(freq_response)

    # # ~> Display
    # plt.subplot(1, 2, 1)
    # plt.title(f"Median Scale: {freqRes_rep}")
    # plt.imshow(input_img, cmap="gray")
    # plt.subplot(1, 2, 2)
    # plt.title(f"Frequency Respond")
    # sns.heatmap(freq_response, cmap="gist_ncar", annot=freq_response, annot_kws={"size": 5}, 
    #             cbar=False, xticklabels=False, yticklabels=False, square=True)

    return freqRes_rep, freq_response

