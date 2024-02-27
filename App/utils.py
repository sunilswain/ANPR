from ultralytics import YOLO
import cv2
from matplotlib import pyplot as plt
from paddleocr import PaddleOCR
import numpy as np

def getPaddle():
    ocr = PaddleOCR(use_angle_cls=True, lang='en')
    return ocr

# Function to extract number plate images
def extractNumberPlates(frame):
    model = YOLO('./best_1.pt')
    result = model.predict(frame)[0]
    numPlates = []
    for object in result.boxes:
        x_min, y_min, x_max, y_max = [round(i) for i in object.xyxy[0].tolist()]
        conf = conf = round(object.conf[0].item(), 2)
        print(x_min, y_min, x_max, y_max)
        if conf > 0.60:
            numPlates.append(
                frame[y_min: y_max, x_min: x_max]
            )
    return numPlates

# Recognize the characters inside number plate
def recogFunc(img):
    paddle = getPaddle()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, otsu_thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    img_binary_lp_er = cv2.erode(otsu_thresh, (3,3))
    img_binary_lp = cv2.dilate(img_binary_lp_er, (3,3))  
    result = paddle.ocr(img_binary_lp_er, cls=True)
    if result != [[]]:
        plate_val = ""
        for rec in result[0]:
            if len(rec[1][0]) > 3:
                plate_val += rec[1][0]
            
    if plate_val[1] == '0':
        plate_val = plate_val[:1]+'D'+plate_val[2:]
    return plate_val, result
