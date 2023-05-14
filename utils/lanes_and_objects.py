from numpy import ones, vstack
from numpy.linalg import lstsq
import cv2
import imutils

def extract_linear_function(lane_data):
    x_coords, y_coords = zip(*[(lane_data[0], lane_data[1]), (lane_data[2], lane_data[3])])
    A = vstack([x_coords, ones(len(x_coords))]).T
    return lstsq(A, y_coords)[0]


def is_rectangle_in_lane(lane, rectangle):
    x, y, w, h = rectangle

    left_lane, right_lane = lane
    left_lane = left_lane[0]
    right_lane = right_lane[0]

    left_m, left_c = extract_linear_function(left_lane)
    right_m, right_c = extract_linear_function(right_lane)

    # coming from left side
    if left_m * x + left_c > y and left_m * (x + w) + left_c < y:
        return True

    # coming from right side
    if right_m * x + right_c < y and right_m * (x + w) + right_c > y:
        return True

    return False


def extract_cnts(first_frame, canny_image):
    img_diff = cv2.absdiff(first_frame, canny_image)  # absolute diff b/w 1st nd current frame
    thresh_img = cv2.threshold(img_diff, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]  # binary
    thresh_img = cv2.dilate(thresh_img, None, iterations=3)
    cnts = cv2.findContours(thresh_img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return imutils.grab_contours(cnts)

def add_cnts_to_image(cnts, averaged_lines, combo_image, image_size):
    for c in cnts:
        if cv2.contourArea(c) < image_size:
            continue
        (x, y, w, h) = cv2.boundingRect(c)
        if w > combo_image.shape[0] * 0.4 or h > combo_image.shape[1] * 0.4:
            continue
        if is_rectangle_in_lane(averaged_lines, (x, y, w, h)):
            cv2.rectangle(combo_image, (x, y), (x + w, y + h), (0, 0, 255), 2)
        else:
            cv2.rectangle(combo_image, (x, y), (x + w, y + h), (0, 255, 0), 1)