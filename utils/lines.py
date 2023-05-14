import cv2
import numpy as np


def hough_lines(cropped_canny):
    return cv2.HoughLinesP(cropped_canny, 2, np.pi / 180, 50,
                           np.array([]), minLineLength=10, maxLineGap=5)

def make_points(image, line):
    slope, intercept = line
    y1 = int(image.shape[0])  # bottom of the image
    y2 = int(y1 * 2.5 / 5)  # slightly lower than the middle
    x1 = int((y1 - intercept) / slope)
    x2 = int((y2 - intercept) / slope)
    return [[x1, y1, x2, y2]]

def get_line(image, fit_data, old_fit_line):
    if len(fit_data) != 0:
        left_fit_average = np.average(fit_data, axis=0)
        return make_points(image, left_fit_average)
    else:
        if old_fit_line is None:
            print('Left fit and right fit must be found')
            return None
        return old_fit_line


def average_slope_intercept(image, lines, old_left_line=None, old_right_line=None):
    left_fit = []
    right_fit = []
    if lines is None:
        return None
    for line in lines:
        for x1, y1, x2, y2 in line:
            fit = np.polyfit((x1, x2), (y1, y2), 1)
            slope = fit[0]
            intercept = fit[1]
            if slope < -0.5:  # y is reversed in image
                left_fit.append((slope, intercept))
            elif slope > 0.8:
                # Positive slope.
                right_fit.append((slope, intercept))

    left_line = get_line(image, left_fit, old_left_line)
    right_line = get_line(image, right_fit, old_right_line)

    return [left_line, right_line]