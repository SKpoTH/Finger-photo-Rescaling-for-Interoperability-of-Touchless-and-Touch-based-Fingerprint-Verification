import seaborn as sns
import matplotlib.pyplot as plt

from modules.utils import mkdirRecursion

def respondFeaturesFigure(input_img, freq_response, freqRes_med, 
                          output_folder, output_filename, save=False):
    '''
        Display and Save Features Respond
    '''
    ### ~> Display
    plt.figure(figsize=(10,6))
    # ~ Original
    plt.subplot(1, 2, 1)
    plt.title(f"Rep Scale: {freqRes_med}")
    plt.imshow(input_img, cmap="gray")
    # ~ Frequency Subbands Response
    plt.subplot(1, 2, 2)
    plt.title(f"Frequency Respond")
    sns.heatmap(freq_response, cmap="gist_ncar", annot=freq_response, annot_kws={"size": 5}, 
                cbar=False, xticklabels=False, yticklabels=False, square=True)
    
    ### ~> In case of Display Figure
    if not save:
        plt.show()
    ### ~> In case of Save Figure
    # -> If there is no the folder one
    mkdirRecursion(output_folder)
    # -> Save Figure
    plt.savefig(output_folder + output_filename + ".png")
    plt.close()