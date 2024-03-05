from hrnet import hrnet
from torch import nn
import torch
import cv2
from torchvision import transforms
import numpy as np
from scipy.spatial import distance
from paddleocr import PaddleOCR

def getSemanticModel():
    """Prepares the model for evaluation"""

    seg_model = hrnet().eval()

    if torch.cuda.is_available():
        semantic_model = nn.SyncBatchNorm.convert_sync_batchnorm(seg_model)

    semantic_model = nn.DataParallel(seg_model)
    semantic_model.load_state_dict(
        torch.load(
            "Appv2/best_semantic.pth",
            map_location=torch.device("cpu"),
        )["state_dict"]
    )

    return semantic_model

def preprocess_image(image, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    """Converts the image into appropriate format for model prediction"""
    transformation = transforms.Compose([transforms.ToPILImage(), transforms.ToTensor()])
    image = transforms.Normalize(mean=mean, std=std)(transformation(image))
    return torch.unsqueeze(image, dim=0)

def plate_locate(image, size_factor=1, area_thresh=600):
    """Finds the co-ordinates of the corners of the license plate"""
    cnts = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]

    coordinates = []
    centroid = []

    for c in cnts:
        area = cv2.contourArea(c)

        if area < area_thresh:
            continue

        temp_rect = []
        rect = cv2.minAreaRect(c)
        centroid.append(rect)
        temp_rect.append(rect[0][0])
        temp_rect.append(rect[0][1])
        temp_rect.append(rect[1][0] * size_factor)
        temp_rect.append(rect[1][1] * size_factor)
        temp_rect.append(rect[2])

        rect = (
            (temp_rect[0], temp_rect[1]),
            (temp_rect[2], temp_rect[3]),
            temp_rect[4],
        )

        box = cv2.boxPoints(rect)
        box = np.int0(box)
        box = [[max(0, int(x[0])), max(0, int(x[1]))] for x in box]
        coordinates.append(box)

    return coordinates, centroid

def get_warped_plates(rgb_image, coordinates):
    """Uses perspective transformation to warp(straighten the image)"""
    cropped_images = []

    for box in coordinates:

        height = int(distance.euclidean(box[0], box[1]))
        width = int(distance.euclidean(box[1], box[2]))

        src_pts = np.array(box).astype("float32")
        dst_pts = np.array(
            [[0, height - 1], [0, 0], [width - 1, 0], [width - 1, height - 1]],
            dtype="float32",
        )

        M = cv2.getPerspectiveTransform(src_pts, dst_pts)

        warped = cv2.warpPerspective(rgb_image, M, (width, height))

        if width < height:
            warped = cv2.rotate(warped, cv2.ROTATE_90_CLOCKWISE)

        cropped_images.append(warped)

    return cropped_images

def extractNumberPlates(img):
    # Loading the model
    seg_model = getSemanticModel()

    # Pre-processing the image
    pre_image = preprocess_image(img)
    result = seg_model(pre_image, (pre_image.shape[2], pre_image.shape[3]))

    # Extracting the co-ordinates
    out = (
        torch.argmax(result["output"], dim=1)
        .detach()
        .cpu()
        .squeeze(dim=0)
        .numpy()
        .astype(np.uint8)
    )
    cords, _ = plate_locate(out)

    # Getting the cropped-Warped images
    plate_images = get_warped_plates(img, cords)

    return plate_images


def getPaddle():
    """Returns our CNN OCR Model"""
    ocr = PaddleOCR(use_angle_cls=True, lang='en')
    return ocr


# Recognize the characters inside number plate
def recogFunc(img):
    """Returns the recognized value of the number plate"""
    paddle = getPaddle()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, otsu_thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    img_binary_lp_er = cv2.erode(otsu_thresh, (3,3))
    img_binary_lp = cv2.dilate(img_binary_lp_er, (3,3))  
    result = paddle.ocr(gray, cls=True)
    print(result)
    if result != [[]]:
        plate_val = ""
        for rec in result[0]:
            if len(rec[1][0]) > 3:
                plate_val += rec[1][0]
            
    if plate_val[1] == '0':
        plate_val = plate_val[:1]+'D'+plate_val[2:]
    return plate_val, result
