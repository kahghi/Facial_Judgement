import numpy as np
import cv2

def convert_img_to_grey(img:np.array)->np.array:
    """
    Takes in image and greyscale it

    :return: greyscaled image
    """    
    grey_image = cv2.cvtColor(
        src=img,
        code=cv2.COLOR_BGR2GRAY
    )
    return grey_image

def resize_image(grey_image:np.array, size:tuple)->np.array:
    """
    Takes in image along with the required size

    :return: resized image
    """    
    grey_image = cv2.resize(grey_image, (size[0], size[1]))
    return grey_image

def reshape_image_for_tensor(grey_image:np.array, size:tuple)->np.array:
    """
    Takes in image and reshape to numpy format for tensorflow

    :return: reshaped image
    """
    grey_image = np.reshape(
        grey_image,(1,size[0],size[1],size[2])).astype(np.float32)
    return grey_image

def normalize(grey_image:np.array)->np.array:
    """
    Takes in image and normalizes the number

    :return: normalized image
    """
    grey_image = grey_image/255.
    return grey_image

def drawLabel(test_img, pred_text, x, y):
    """
    Takes in the image, predicted labels and x, y coordinates
    Draws the text labels onto video stream
    """
    cv2.putText(
        img=test_img,
        text=pred_text,
        org=(x+10,y+10),
        fontFace=cv2.FONT_HERSHEY_DUPLEX,
        fontScale=0.8,
        color=(0,255,0),
        thickness=1)