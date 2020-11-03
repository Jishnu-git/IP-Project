import os, cv2
import numpy as np
import math
from numpy.fft import fft2, ifft2
from skimage import io, restoration
from scipy.signal import convolve2d
from google.cloud import vision
from Tools import getGrayImg, writeGrayImg, Gray2Binary
from TextRecognition import Img2Text, NoisyImg2Text
from skimage.filters.rank import entropy
from skimage.morphology import disk

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "VisionAPIKey.json"
client = vision.ImageAnnotatorClient()

def makeKernel(kernelSize, angle):
    kernel = np.zeros((kernelSize, kernelSize))
    slope = math.tan(math.radians(angle))
    mid = int((kernelSize - 1) / 2)
    count = 0
    for x in range(kernelSize):
        for y in range(kernelSize):
            if (x - mid) == round(slope * (mid - y)):
                kernel[x][y] = 1;
                count += 1
    if count != 0:
        kernel /= count

    return kernel

def applyMotionBlur(image, kernelSize, angle):
    kernel = makeKernel(kernelSize, angle)
    return convolve2d(image, kernel, "same") #cv2.filter2D(image, -1, kernel)

def removeMotionBlur(image, kernelSize, angle):
    kernel = makeKernel(kernelSize, angle)
    return restoration.wiener(image, kernel, 0.1)

def guessSize(image, angle):
    entr = entropy((removeMotionBlur(image, 2, angle) * 255).astype(np.uint8), disk(10))
    prevSum = np.sum(entr)
    curSum = prevSum

    n = 2
    while (curSum >= prevSum):
        n += 2
        prevSum = curSum
        curSum = np.sum(entropy((removeMotionBlur(image, n, angle) * 255).astype(np.uint8), disk(10)))

    return n + 2

if __name__ == "__main__":
    roadSign = getGrayImg("RoadSign1.jpg") / 255
    cv2.imshow("Road Sign", roadSign)
    blurredRoadSign = applyMotionBlur(roadSign, 20, 0)
    cv2.imshow("Motion Blur on Road Sign", blurredRoadSign)
    print("Before removing motion blur: ")
    Img2Text(blurredRoadSign * 255)
    restoredRoadSign = removeMotionBlur(blurredRoadSign, 20, 0)
    cv2.imshow("Restored Road Sign", restoredRoadSign)
    print("After restoration: ")
    Img2Text(restoredRoadSign * 255)
    cv2.waitKey()

    plate1 = getGrayImg("plates1.jpg") / 255
    cv2.imshow("Motion Blur on license plate", plate1)
    restoredPlate1 = removeMotionBlur(plate1, 18, 44)
    cv2.imshow("Restored license plate", restoredPlate1)
    restoredPlate1Binary = Gray2Binary(restoredPlate1, 0.45)
    cv2.imshow("Binary of Restored Plate", restoredPlate1Binary)
    print("\nBefore removing motion blur: ")
    Img2Text(plate1 * 255)
    print("After restoration: ")
    Img2Text(restoredPlate1Binary)
    cv2.waitKey()

    plate2 = getGrayImg("plates2.jpg") / 255
    cv2.imshow("Motion blur on license plate", plate2)
    restoredPlate2 = removeMotionBlur(plate2, 11, 36)
    cv2.imshow("Restored license plate", restoredPlate2)
    restoredPlate2Binary = Gray2Binary(restoredPlate2, 0.28)
    cv2.imshow("Binary of restored plate", restoredPlate2Binary)
    print("\nBefore removing motion blur: ")
    Img2Text(plate2 * 255)
    print("After restoration: ")
    Img2Text(restoredPlate2Binary)
    cv2.waitKey()

    plate3 = getGrayImg("plates3.png") / 255
    cv2.imshow("Motion blur on license plate", plate3)
    restoredPlate3 = removeMotionBlur(plate3, 11, 42)
    cv2.imshow("Restored license plate", restoredPlate3)
    restoredPlate3Binary = Gray2Binary(restoredPlate3, 0.35)
    cv2.imshow("Binary of restored plate", restoredPlate3Binary)
    print("\nBefore removing motion blur: ")
    Img2Text(plate3 * 255)
    print("After restoration: ")
    Img2Text(restoredPlate3Binary)
    cv2.waitKey()
    