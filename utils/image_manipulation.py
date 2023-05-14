import cv2
import numpy as np

def canny(img):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    kernel = 5
    blur = cv2.GaussianBlur(gray, (kernel, kernel), 0)
    canny = cv2.Canny(blur, 50, 150)
    return canny

def add_weighted(frame, line_image):
    return cv2.addWeighted(frame, 0.8, line_image, 1, 1)

def region_of_interest(canny):
    height = canny.shape[0]
    width = canny.shape[1]
    mask = np.zeros_like(canny)
    rectangle = np.array([[
        (0, height),
        (0, height / 3 * 2),
        (width / 2, height / 5),
        (width / 5 * 4, height), ]], np.int32)
    cv2.fillPoly(mask, rectangle, 255)
    masked_image = cv2.bitwise_and(canny, mask)
    return masked_image