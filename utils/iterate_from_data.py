import os
import cv2

DATA_FOLDER = './data'
IMAGE_DATA_FOLDER = os.path.join(DATA_FOLDER, 'images')
VIDEO_DATA_FOLDER = os.path.join(DATA_FOLDER, 'videos')


def get_iterator(file_path):
    use_video = file_path.endswith('.mp4')

    if use_video:
        cap = cv2.VideoCapture(os.path.join(VIDEO_DATA_FOLDER, file_path))
        while (cap.isOpened()):
            _, frame = cap.read()
            yield frame
            # sleep(1)
        cap.release()
    else:
        current_drive_folder = os.path.join(IMAGE_DATA_FOLDER, file_path)
        for filename in os.listdir(current_drive_folder):
            for i in range(3):
                yield cv2.imread(os.path.join(current_drive_folder, filename))