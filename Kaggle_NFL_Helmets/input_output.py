import os
import cv2


def read_video(path):

    images = []

    vid_capture = cv2.VideoCapture(path)
    if vid_capture.isOpened() is False:
        print("Error opening the video file")
    else:
        fps = vid_capture.get(5)
        print('Frames per second : ', fps, 'FPS')
        frame_count = vid_capture.get(7)
        print('Frame count : ', frame_count)
        ind = 0
        while vid_capture.isOpened():
            ret, frame = vid_capture.read()
            if ret is True:
                images.append(frame)
                ind += 1
                if frame_count == ind:
                    vid_capture.release()
                    return images
    vid_capture.release()

    return images


def read_labels_csv_file(csv_path):
    full_data = []
    with open(csv_path, 'r') as f:
        lines = f.readlines()
        labels = [i.replace("\n", "") for i in lines[0].split(",")]

        for i in range(1, len(lines)):
            lines[i] = [j.replace("\n", "") for j in lines[i].split(",")]
            d = {}
            for l in range(0, len(labels)):
                d[labels[l]] = lines[i][l]
            full_data.append(d)
    return full_data


