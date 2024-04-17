from ultralytics import YOLO
import cv2
from matplotlib import pyplot as plt
from paddleocr import PaddleOCR
import numpy as np
import re
import streamlit as st



def getPaddle():
    """Returns our CNN OCR Model"""
    ocr = PaddleOCR(use_angle_cls=True, lang='en')
    return ocr

# Function to extract number plate images
def extractNumberPlates(frame):
    """Returns cropped pictures of number plates if present"""
    model = YOLO('App/DetectorModelv3.pt')
    result = model.predict(frame)[0]
    numPlates = []
    for object in result.boxes:
        x_min, y_min, x_max, y_max = [round(i) for i in object.xyxy[0].tolist()]
        conf = conf = round(object.conf[0].item(), 2)
        print(x_min, y_min, x_max, y_max)
        print(conf)
        if conf > 0.60:
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

    results = paddle.ocr(gray, cls=True)[0]
    print(results)

    img_height, img_width = img.shape[:2]
    plate_val = ""
    
    if results != None:
        for result in results:
            cords = np.array(result[0], dtype=np.int32)
            value = result[1][0]
            det_area = (cv2.contourArea(cords) / (img_height * img_width)) * 100

            if det_area > 8:
                plate_val += value

    if len(plate_val) > 0:
        if plate_val[0] == '0':
            plate_val = 'O'+plate_val[1:]
        if plate_val[1] == '0':
            plate_val = plate_val[:1]+'D'+plate_val[2:]
        if plate_val[2] == 'O':
            plate_val = plate_val[:2]+'0'+plate_val[3:]
    return plate_val, result
