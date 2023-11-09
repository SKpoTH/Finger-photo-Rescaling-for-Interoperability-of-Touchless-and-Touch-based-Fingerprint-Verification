from modules.utils import mkdirRecursion

def resizeRatio(resize_ratio, output_folder, output_file):
    '''
        Write/Save Resize Ratio into the .txt file
    '''
    # -> Create the output folder if there is no one
    mkdirRecursion(output_folder)
    # -> Open/Create .txt file
    f = open(output_folder + f"{output_file}.txt", "w+")
    # -> Write Resize Ratio
    f.write(f"{resize_ratio}")
    # -> Close the .txt file
    f.close()