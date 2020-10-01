import os, cv2
import numpy as np
from skimage import io
from google.cloud import vision
from skimage.filters.rank import entropy
from skimage.morphology import disk

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "VisionAPIKey.json"
client = vision.ImageAnnotatorClient()

def Img2Text(image):
    content = cv2.imencode(".jpg", image)[1].tostring()
    encodedImage = vision.types.Image(content = content)
    texts = client.text_detection(image = encodedImage).text_annotations
    if (texts):
        print(texts[0].description + "\n")
    return texts

def getGrayImg(img_name):
    img_path = os.path.join("../Data/Text/", img_name)
    image = io.imread(img_path)
    if (len(image.shape) == 3):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image

def NoisyImg2Text(image):
    #Image processing to enhance output
    entr = entropy(image, disk(10))
    mask = cv2.threshold(entr, 0.55, 255, cv2.THRESH_BINARY)[1].astype(np.uint8)
    gauss = cv2.GaussianBlur(image, (5, 5), 0)
    image = cv2.bitwise_and(gauss, gauss, mask = mask)
    cv2.imshow("Pre Filter Image", image)
    image = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY_INV)[1].astype(np.uint8)

    cv2.imshow("Post Filter Image", image)
    cv2.imshow("Mask", mask)
    cv2.imshow("Gauss Filter", gauss)

    #Image to text using Google's Cloud Vision
    return Img2Text(image)

if __name__ == "__main__":
    roadSign = getGrayImg("RoadSign1.jpg");
    cv2.imshow("Road Sign", roadSign)
    Img2Text(roadSign)
    cv2.waitKey()
    
    noisyTxt1 = getGrayImg("NoisyText1.png")
    cv2.imshow("Noisy Text One", noisyTxt1)
    Img2Text(noisyTxt1)
    cv2.waitKey()

    noisyTxt2 = getGrayImg("NoisyText2.png")
    cv2.imshow("Noisy Text Two", noisyTxt2)
    Img2Text(noisyTxt2);
    cv2.waitKey()

    NoisyImg2Text(noisyTxt1)
    cv2.waitKey()

    NoisyImg2Text(noisyTxt2)
    cv2.waitKey()    