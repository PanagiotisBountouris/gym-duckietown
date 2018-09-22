
import numpy as np
import cv2

def img_preprocess(image, cnn_type):

    if cnn_type == "regression_grey":
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        image = image.astype(float)
        image = image / 255
        image = np.reshape(image, (1, 120*160*1))
    else:
        image = image.astype(float)
        image = image / 255
        image = np.reshape(image, (1, 120*160*3))

    return image


def img_preprocess_lane_following(image):


    # crop the 1/3 upper part of the image
    new_img = image[int(120/3):, :, :]

    # transform the color image to grayscale
    new_img = cv2.cvtColor( new_img[:, :, :], cv2.COLOR_RGB2GRAY )

    # resize the image from 320x640 to 48x96
    new_img = cv2.resize( new_img, ( 96, 48 ) ) # returns image 48 x 96 and not 96 x 48
    # new_img = cv2.resize( new_img, ( 160, 120 ) ) # returns image 48 x 96 and not 96 x 48

    # normalize image to range [0, 1] (divide each pixel by 255)
    # first transform the array of int to array of float else the division with 255 will return an array of 0s
    new_img = new_img.astype(float)
    new_img = new_img / 255

    # reshape the image to row vector [1, 48x96]
    new_img = np.reshape(new_img, (1, -1))

    return new_img