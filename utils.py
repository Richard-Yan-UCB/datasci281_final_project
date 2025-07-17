import numpy as np
from skimage import color
from skimage.transform import rescale
import cv2 as cv

""" normalize the image between 0 and 1 """
def normalize_img(img):
    normalized = (img - img.min())/(img.max() - img.min())    
    return normalized

""" 
convert to grayscale
normalize the image between 0 and 1
resize image to im_size 
"""
def preprocess_image(img_path, im_size=512):
    img = cv.imread(img_path)
    img_grayscale = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    normalized = normalize_img(img_grayscale)
    rescaled = rescale_img(img=normalized, standard=im_size)
    return rescaled

def rescale_img(img, standard=256):
  # rescale short side to standard size, then crop center
  scale = standard / min(img.shape[:2])
  img = rescale(img, scale, anti_aliasing=True)
  img = img[int(img.shape[0]/2 - standard/2) : int(img.shape[0]/2 + standard/2),
            int(img.shape[1]/2 - standard/2) : int(img.shape[1]/2 + standard/2)]
  return img