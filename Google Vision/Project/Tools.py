import cv2, os
import numpy as np
from skimage import io
from google.cloud import vision

def getGrayImg(img_name):
    img_path = os.path.join("../Data/Text/", img_name)
    image = io.imread(img_path)
    if (len(image.shape) == 3):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image

def getColorImg(img_name):
    img_path = os.path.join("../Data/Text/", img_name)
    image = io.imread(img_path)
    return image

def toGrayImg(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def writeGrayImg(image, filename):
    img_path = os.path.join("../Data/Text/", filename + ".jpg")
    cv2.imwrite(img_path, image)

def Gray2Binary(image, threshold):
    return cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY)[1].astype(np.uint8)