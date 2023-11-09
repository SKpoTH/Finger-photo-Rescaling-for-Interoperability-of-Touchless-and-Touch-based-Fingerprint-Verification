import cv2 as cv
from modules.utils import mkdirRecursion

def convertRange(input_array, min_val, max_val):
    '''
        Convert any value range array into specifc range value array
    '''
    # -> Convert into [0, 1]
    norm_array = (input_array - input_array.min()) / (input_array.max() - input_array.min())
    # -> Convert [0, 1] into [min_val, max_val]
    output_array = (norm_array * (max_val - min_val)) + min_val

    return output_array

def resizeImage(input_img, resize_ratio, output_folder, output_filename):
    '''
        Write/Save Resize Image with the given ratio
    '''
    # -> Resize Image
    resize_img = cv.resize(input_img, None, fx=resize_ratio, fy=resize_ratio, interpolation=cv.INTER_CUBIC)
    # -> Create the output folder if there is no one
    mkdirRecursion(output_folder)
    # -> Save Enhance
    output_img = convertRange(resize_img, 0, 255)
    # -> Save Resize Image
    cv.imwrite(output_folder + output_filename + ".jpg", output_img)