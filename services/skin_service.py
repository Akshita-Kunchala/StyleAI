import cv2
import numpy as np

def detect_skin_tone(image):
    img = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

    lower = np.array([0, 30, 60])
    upper = np.array([20, 150, 255])
    mask = cv2.inRange(img, lower, upper)

    avg = cv2.mean(img, mask=mask)[2]

    if avg > 200:
        return "Very Fair"
    elif avg > 170:
        return "Fair"
    elif avg > 140:
        return "Olive"
    elif avg > 110:
        return "Medium"
    elif avg > 80:
        return "Dusky"
    else:
        return "Deep"