import cv2
import imutils
from utils.image_manipulation import canny, add_weighted, region_of_interest
from utils.lines import hough_lines, average_slope_intercept
from utils.lanes_and_objects import extract_cnts, add_cnts_to_image
from utils.iterate_from_data import get_iterator
from utils.view import display_lines

VIDEOS_LOCATION = "test2.mp4"
RESIZE_TO = 1000

if __name__ == '__main__':
    previous_frame = None
    old_averaged_lines = []
    for frame in get_iterator(VIDEOS_LOCATION):
        try:
            resized_img = imutils.resize(frame, width=RESIZE_TO)
            cv2.imshow("resize", frame)

            canny_image = canny(resized_img)
            cv2.imshow("canny", canny_image)
            cropped_canny = region_of_interest(canny_image)
            cv2.imshow("canny cropped", cropped_canny)

            lines = hough_lines(cropped_canny)
            cv2.imshow("all lines", display_lines(resized_img, lines))

            averaged_lines = average_slope_intercept(resized_img, lines, *old_averaged_lines)

            # start only after first frame with lanes
            if averaged_lines[0] and averaged_lines[1]:
                old_averaged_lines = averaged_lines
                avg_line_image = display_lines(resized_img, averaged_lines)
                combo_image = add_weighted(resized_img, avg_line_image)
                cv2.imshow("avg lines", combo_image)

                if previous_frame is None:
                    previous_frame = canny_image  # capturing 1st frame on 1st iteration
                    continue

                cnts = extract_cnts(previous_frame, canny_image)
                add_cnts_to_image(cnts, averaged_lines, combo_image, RESIZE_TO)
                cv2.imshow("final with object detection", combo_image)

        except Exception as ex:
            continue

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
