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
from algorithm.fingerphotoSegmentation import fingerphotoSegmentation
from algorithm.fingerphotoEnhancement import fingerphotoEnhancement
from algorithm.fourierFeatures import fourierFeatures
from algorithm.sizeRepresentation import sizeRepresentation
# <------------------------------> Save Result Function <----------------------------------->
from saves.medianResponseScale import medianResponseScale
from saves.readError import readError
from saves.resizeImage import resizeImage
from saves.resizeRatio import resizeRatio

# <---------------------------------> Local Function <-------------------------------------->
def RES2LSfilename(input_filename):
    '''
        Convert from RES format to LS format filename
    '''
    id_name = input_filename.split('_')
    # -> Reform convert
    output_filename = id_name[0] + '_' + id_name[1]
    # -> file type
    output_file = output_filename + ".bmp"

    return output_filename, output_file

# <------------------------------> Algorithm Parameters <----------------------------------->
# -> Short-Time Window Parameters
WINDOW_SIZE = 128
STRIDE_SIZE = WINDOW_SIZE // 4
window_params = [WINDOW_SIZE, STRIDE_SIZE]
# -> Frequency Subbands Parameters
BAND_WIDTH = 2
STRIDE_WIDTH = BAND_WIDTH // 2
freq_params = [BAND_WIDTH, STRIDE_WIDTH]
# -> Orientation Subbands Parameters
BAND_THETA = 10
STRIDE_THETA = BAND_THETA // 2
orient_params = [BAND_THETA, STRIDE_THETA]
# <--------------------------------> PATH Parameters <------------------------------------->
INPUT_PATH = "../Sample Data/FingerPhoto/"
REFERENT_PATH = "../Sample Data/FingerPrint/"
SEGMENT_PATH = "../Sample Data/FingerPhoto Segment/"
OUTPUT_PATH = "output/"

if __name__ == "__main__":
    ### - 2250
    ### >>> Set Arguments
    argParser = argument.ArgumentParser(description="Create Scale Referent Subbands Database")
    argParser.add_argument("-i", "--input", default=INPUT_PATH, help="input fingerphoto PATH")
    argParser.add_argument("-r", "--reference", default=REFERENT_PATH, help="referent fingerprint PATH")
    argParser.add_argument("-s", "--segment", default=SEGMENT_PATH, help="input segment PATH")
    argParser.add_argument("-o", "--output", default=OUTPUT_PATH, help="output rescale PATH")
    argParser.add_argument("-f", "--first", type=int, default=0, help="First Index")
    argParser.add_argument("-l", "--last", type=int, default=-1, help="Last Index")
    ### >>> Set Parameters from Arguments
    args = argParser.parse_args()   
    input_path = args.input
    ref_path = args.reference
    segment_path = args.segment
    output_path = args.output
    first_idx = args.first
    last_idx = args.last
    ### > Get All files in the "input_path"
    # Input Fingerphoto
    input_files = glob(input_path + "*")
    input_filenames = getFilenameWithoutExtension(input_files)
    # Segmentation Mask
    segment_files = glob(segment_path + "*")
    ### >>> Arrange Parameters
    if last_idx == -1:
        last_idx = len(input_files)
    output_path = output_path + "verify_case_run/"
    
    # total_time = 0

    for i in tqdm(range(first_idx, last_idx)):
        ### -> Read Image and convert to Grayscale
        # - Input Fingerphoto Image
        input_img = cv.imread(input_files[i])
        input_img = cv.cvtColor(input_img, cv.COLOR_BGR2GRAY)
        # - Referent Fingerprint Image
        ref_filename, ref_file = RES2LSfilename(input_filenames[i])
        ref_img = cv.imread(ref_path + ref_file)
        ref_img = cv.cvtColor(ref_img, cv.COLOR_BGR2GRAY)
        # - Input Segmentation of Fingerphoto Image
        segment_img = cv.imread(segment_files[i])
        segment_img = cv.cvtColor(segment_img, cv.COLOR_BGR2GRAY)

        ### => Fingerphoto Process
        # - Fingerphoto (Fingertip) Segmentation
        input_crop, mask_crop = fingerphotoSegmentation(input_img, segment_img)
        # - Fingerphoto (Fingertip) Enhancement
        ench_img = fingerphotoEnhancement(input_crop, mask_crop, window_params)
        # - Obtain Fourier Transformation Features
        freq_filters = fourierFeatures(ench_img)
        # - Find Ridge Size Representation
        input_freqRep = sizeRepresentation(freq_filters, peak_cutoff=0.75)

        ### => Fingerprint Feature
        # - Padding to Equal
        ref_img = valuePadding2D(ref_img, ench_img.shape[0], ench_img.shape[1], 255)
        # - Obtain Fourier Transformation Features
        freq_filters = fourierFeatures(ref_img)
        # - Find Ridge Size Representation
        ref_freqRep = sizeRepresentation(freq_filters, peak_cutoff=0.75)

        ### -> Find Resize Ratio
        resize_ratio = input_freqRep / ref_freqRep
        # ~> Save Resize Image
        resizeImage(ench_img, resize_ratio, output_path + "Resize/Images/", input_filenames[i])
        # ~> Save Resize Ratio
        resizeRatio(resize_ratio, output_path + "Resize/Resize Ratio/", input_filenames[i])