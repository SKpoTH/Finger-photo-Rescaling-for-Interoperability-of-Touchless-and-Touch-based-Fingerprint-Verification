import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from skimage.measure import regionprops

def cropFingertip(input_img, mask_img):
    '''
        Crop Fingerphoto by using "mask_img" to Bouding Box
    '''
    # -> Get Object and their properties
    _, label_img = cv.connectedComponents(mask_img)
    feature_vects = regionprops(label_img)
    # -> Find Largest Object index (assume it is Fingertip Object)
    max_area = 0
    for i in range(len(feature_vects)):
        if feature_vects[i].area > max_area:
            max_idx = i
            max_area = feature_vects[i].area
    
    # -> Find bounding box for cropping, "bbox_pos" = (y_topLeft, x_topLeft, y_botRight, x_botRight)
    bbox_pos = feature_vects[max_idx].bbox
    # -> Crop both "input_img" and "mask_img"
    input_crop = input_img[bbox_pos[0]:bbox_pos[2], bbox_pos[1]:bbox_pos[3]]
    mask_crop = mask_img[bbox_pos[0]:bbox_pos[2], bbox_pos[1]:bbox_pos[3]]

    # # ~> Display cropping
    # plt.subplot(1, 2, 1)
    # plt.imshow(input_crop, cmap="gray")
    # plt.subplot(1, 2, 2)
    # plt.imshow(mask_crop, cmap="gray")
    # plt.show()

    return input_crop, mask_crop
        
def transformFinger(input_img):
    '''
        Transform Fingerphoto into the same perspective as Fingerprint
    '''
    # -> Rotate Fingerphoto to be Vertical
    rotate_img = cv.rotate(input_img, cv.ROTATE_90_COUNTERCLOCKWISE)
    # -> Flip, Mirroring vertical-axis
    flip_img = cv.flip(rotate_img, 1)

    return flip_img

def fingerphotoSegmentation(input_img, segment_img):
    '''
       "Main" : Fingerphoto (Fingertip) Segmentation by using Manual Masking
    '''
    ### => Thresholding to get binary mask image
    0, 127, 255
    base_mask = np.where(segment_img > 190, 255, 0)
    finger_mask = np.where(segment_img > 100, 255, 0)
    mask_img = np.logical_xor(finger_mask, base_mask).astype(np.uint8) * 255
    ### => Crop Fingertip
    input_crop, mask_crop = cropFingertip(input_img, mask_img)
    ### => Transform Finger Direction (to be same as Fingerprint)
    input_crop = transformFinger(input_crop)
    mask_crop = transformFinger(mask_crop)

    # plt.subplot(1, 2, 1)
    # plt.imshow(input_img, cmap="gray")
    # plt.subplot(1, 2, 2)
    # plt.imshow(mask_crop, cmap="gray")
    # plt.show()

    return input_crop, mask_crop