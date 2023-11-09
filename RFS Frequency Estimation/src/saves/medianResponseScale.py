from modules.utils import mkdirRecursion

def medianResponseScale(freqRes_med, output_folder, output_file):
    '''
        Write/Save Median Respond Scale into the .txt file
    '''
    # -> Create the output folder if there is no one
    mkdirRecursion(output_folder)
    # -> Open/Create .txt file
    f = open(output_folder + f"{output_file}.txt", "w+")
    # -> Write Median Respond Scale
    f.write(f"{freqRes_med}")
    # -> Close the .txt file
    f.close()