from ultralytics import YOLO
import cv2
from matplotlib import pyplot as plt
from paddleocr import PaddleOCR
import numpy as np
import re
import streamlit as st

# def isExists(plate_val):
#     # Create a connection object.
#     conn = st.connection("gsheets", type=GSheetsConnection)

#     df = conn.read(
#         spreadsheet=st.secrets["spreadsheet"],
#     )
#     print(df.columns)
#     plates = list(df['Vehicle_Number'])

#     if plate_val in plates:
#         return df.loc[
#             df['Vehicle_Number']==plate_val
#         ][['Emp_Id', 'Owner_Name']].values[0]

#     return None
# def removeSpecialCharacters(input_string):
#     """
#     This Function accepts a string and removes any special characters present
#     """
#     pattern = r'[^a-zA-Z0-9]'
    
#     return re.sub(pattern, '', input_string)

# def replaceCharacters(x):
#     """
#     This function accepts a string and replaces a the integer character to their respective letter that
#     closely resembles it. E.g - 4-> A.
#     This is designed because in indian number plates the state(first 2 character) & the serial number are 
#     letter. And we are trying to prevent any mis-prediction made by our cnn reader.
#     """
#     # Defining a dictionary where the key is the value that we want to fix with its respective value
#     chars = {'0':'O', '4':'A', '3':'B'}
#     for i in range(len(x)):
#         if x[i] in chars.keys():
#             x[i] = chars[x[i]]
#     return x

# def fixNumberPlate(input_string):
#     """
#     Fixes the number plates  by 
#     - Removing the the special characters
#     - Fixing the digits in places of letter
#     """
#     input_string = removeSpecialCharacters(input_string)

#     if len(input_string) == 9:
#         input_string = list(input_string)
#         # Fixing the state first
#         state = input_string[:2] # Grabbing the state
#         input_string[:2] = replaceCharacters(state)

#         # Fixing the serial value
#         serial = input_string[4:5]
#         input_string[4:5] = replaceCharacters(serial)

#     elif len(input_string) == 10:
#         input_string = list(input_string)
#         # Fixing the state first
#         state = input_string[:2] # Grabbing the state
#         input_string[:2] = replaceCharacters(state)

#         # Fixing the serial value
#         serial = input_string[4:6]
#         input_string[4:6] = replaceCharacters(serial)
    
#     return "".join(input_string)

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
