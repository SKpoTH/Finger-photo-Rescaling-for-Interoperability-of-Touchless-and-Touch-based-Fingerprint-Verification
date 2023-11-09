import argparse as argument
import cv2 as cv
import numpy as np
import matplotlib
import time
# matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm
from glob import glob

from modules.padding import valuePadding2D
from modules.utils import *

# <-------------------------------> Algorithm Process <------------------------------------->

# <------------------------------> Save Result Function <----------------------------------->

# <---------------------------------> Local Function <-------------------------------------->
def RES2LSfilename(input_filename):
    '''
        Convert from RES format to LS format filename
    '''
    id_name = input_filename.split('_')
    # -> Reform convert
    output_filename = id_name[0] + '_' + id_name[1] + '_' + id_name[2] + '_' + id_name[4]
    # -> file type
    output_file = output_filename + ".bmp"

    return output_filename, output_file

# <--------------------------------> PATH Parameters <------------------------------------->
INPUT_PATH = "output/verify_case_run/Resize/Resize Ratio/"
OUTPUT_PATH = "output/verify_case_run/Resize/"

if __name__ == "__main__":
    ### - 2250
    ### >>> Set Arguments
    argParser = argument.ArgumentParser(description="Statistic of Resize Ratio")
    argParser.add_argument("-i", "--input", default=INPUT_PATH, help="input fingerphoto PATH")
    argParser.add_argument("-o", "--output", default=OUTPUT_PATH, help="output rescale PATH")
    argParser.add_argument("-f", "--first", type=int, default=0, help="First Index")
    argParser.add_argument("-l", "--last", type=int, default=-1, help="Last Index")
    ### >>> Set Parameters from Arguments
    args = argParser.parse_args()   
    input_path = args.input
    output_path = args.output
    first_idx = args.first
    last_idx = args.last
    ### > Get All files in the "input_path"
    # Input Fingerphoto
    input_files = glob(input_path + "*")
    input_filenames = getFilenameWithoutExtension(input_files)
    ### >>> Arrange Parameters
    if last_idx == -1:
        last_idx = len(input_files)
    output_path = output_path + "Stats Ratio/"

    ### => Resize Ratio Array
    ratio_dict = {
                    '5': [],
                    '8': [],
                    '13': [],
                    '16': []
                 }

    for i in tqdm(range(first_idx, last_idx)):
        ### -> Read Resize Ratio
        f = open(input_files[i])
        ratio = float(f.readline())
        f.close()
        ### -> Resolution
        res = input_filenames[i].split('_')[3]
        ### -> Append Ratio
        ratio_dict[res].append(ratio)

    for r in ratio_dict:
        array = np.array(ratio_dict[r])
        mean = np.mean(array)
        std = np.std(array)
        print(f"{r}MP - [Mean: {mean}], [STD: {std}]")