
import pandas as pd
import numpy as np
import os
import cv2
import matplotlib.image as image
import matplotlib.pyplot as plt

file = os.path.join('..', 'data', 'data.h5')
df_img = pd.read_hdf(file, key='left_images')

print('shape of df_img = {}'.format(df_img.shape))
