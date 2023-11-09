import numpy as np
import pandas as pd
from modules.utils import mkdirRecursion

def freqResponse(freq_response, output_folder, output_file):
    '''
        Save Frequency Representation
    '''
    # -> Create the output folder if there is no one
    mkdirRecursion(output_folder)
    # -> Save "orient_rep" numpy array
    np.save(output_folder + output_file, freq_response)