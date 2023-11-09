import numpy as np

def valuePadding2D(input_img, y_size_new, x_size_new, pad_vlaue):
    '''
        2D Zero Padding, Input New Size for padding
    '''
    ### -> Get Original Shape Size
    y_size = input_img.shape[0]
    x_size = input_img.shape[1]

    ### -> Create Empty Output Image
    padded_img = np.ones((y_size_new, x_size_new)) * pad_vlaue
    
    ### -> Define Padding
    '''
        ----------------- Padding Algorithm ------------------
        
                          (y-half diffence)
                                  ^
                                  | 
        (x-half difference) <- Orignal -> (x-half diffence + x-reminder)
                                  |
                                  v
                     (y-half diffence + y-reminder) 
    '''
    # -> Y-Dimension : Top , Bottom  -> 2 part divided
    y_padding_reminder = (y_size_new - y_size) % 2
    y_padding_size = (y_size_new - y_size) // 2
    # -> X-Dimension : Left, Right   -> 2 part divided
    x_padding_reminder = (x_size_new - x_size) % 2
    x_padding_size = (x_size_new - x_size) // 2

    ### ->  Place Input Image on the center of Padded Image
    padded_img[y_padding_size:y_size+y_padding_size, \
               x_padding_size:x_size+x_padding_size] = input_img

    return padded_img

    
