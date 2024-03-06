from ultralytics import YOLO
import cv2
from matplotlib import pyplot as plt
from paddleocr import PaddleOCR
import numpy as np
import re


def removeSpecialCharacters(input_string):
    """
    This Function accepts a string and removes any special characters present
    """
    pattern = r'[^a-zA-Z0-9]'
    
    return re.sub(pattern, '', input_string)

def replaceCharacters(x):
    """
    This function accepts a string and replaces a the integer character to their respective letter that
    closely resembles it. E.g - 4-> A.
    This is designed because in indian number plates the state(first 2 character) & the serial number are 
    letter. And we are trying to prevent any mis-prediction made by our cnn reader.
    """
    # Defining a dictionary where the key is the value that we want to fix with its respective value
    chars = {'0':'O', '4':'A', '3':'B'}
    for i in range(len(x)):
        if x[i] in chars.keys():
            x[i] = chars[x[i]]
    return x

def fixNumberPlate(input_string):
    """
    Fixes the number plates  by 
    - Removing the the special characters
    - Fixing the digits in places of letter
    """
    input_string = removeSpecialCharacters(input_string)

    if len(input_string) == 9:
        input_string = list(input_string)
        # Fixing the state first
        state = input_string[:2] # Grabbing the state
        input_string[:2] = replaceCharacters(state)

        # Fixing the serial value
        serial = input_string[4:5]
        input_string[4:5] = replaceCharacters(serial)

    elif len(input_string) == 10:
        input_string = list(input_string)
        # Fixing the state first
        state = input_string[:2] # Grabbing the state
        input_string[:2] = replaceCharacters(state)

        # Fixing the serial value
        serial = input_string[4:6]
        input_string[4:6] = replaceCharacters(serial)
    
    return "".join(input_string)

def getPaddle():
    """Returns our CNN OCR Model"""
    ocr = PaddleOCR(use_angle_cls=True, lang='en')
    return ocr

# Function to extract number plate images
def extractNumberPlates(frame):
    """Returns cropped pictures of number plates if present"""
    model = YOLO('App/best_1.pt')
    result = model.predict(frame)[0]
    numPlates = []
    for object in result.boxes:
        x_min, y_min, x_max, y_max = [round(i) for i in object.xyxy[0].tolist()]
        conf = conf = round(object.conf[0].item(), 2)
        print(x_min, y_min, x_max, y_max)
        print(conf)
        if conf > 0.70:
            numPlates.append(
                frame[y_min: y_max, x_min: x_max]
            )
    return numPlates

# Recognize the characters inside number plate
def recogFunc(img):
    """Predicts the number plate value"""
    paddle = getPaddle()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # _, otsu_thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    # img_binary_lp_er = cv2.erode(otsu_thresh, (3,3))
    # img_binary_lp = cv2.dilate(img_binary_lp_er, (3,3))  
    result = paddle.ocr(gray, cls=True)
    print(result)
    plate_val = ""

    if result != [None]:
        for rec in result[0]:
            if len(rec[1][0]) > 3:
                plate_val += rec[1][0]

    # # Fixing some characters in the number plate
    # plate_val = fixNumberPlate(plate_val)
    if len(plate_val) > 0:
        if plate_val[1] == '0':
            plate_val = plate_val[:1]+'D'+plate_val[2:]

    return plate_val, result
