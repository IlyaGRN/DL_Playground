import os
import cv2
from input_output import read_video
from input_output import read_labels_csv_file


def helmets_to_data_struct(helmets):
    helmets_data_per_video = {}
    for i in range(0, len(helmets)):

        frame_num = int(helmets[i]['video_frame'][
                    len(helmets[i]['video_frame']) - helmets[i]['video_frame'][::-1].find("_"):])
        video_name = helmets[i]['video_frame'][
                     :len(helmets[i]['video_frame']) - helmets[i]['video_frame'][::-1].find("_") - 1] + ".mp4"
        if video_name in helmets_data_per_video.keys():
            if frame_num in helmets_data_per_video[video_name].keys():
                helmets_data_per_video[video_name][frame_num].append([int(helmets[i]['left']),
                                                                      int(helmets[i]['width']),
                                                                      int(helmets[i]['top']),
                                                                      int(helmets[i]['height']),
                                                                      float(helmets[i]['conf'])])
            else:
                helmets_data_per_video[video_name][frame_num] = [[int(helmets[i]['left']),
                                                                  int(helmets[i]['width']),
                                                                  int(helmets[i]['top']),
                                                                  int(helmets[i]['height']),
                                                                  float(helmets[i]['conf'])]]
        else:
            helmets_data_per_video[video_name] = {
                frame_num: [[int(helmets[i]['left']),
                             int(helmets[i]['width']),
                             int(helmets[i]['top']),
                             int(helmets[i]['height']),
                             float(helmets[i]['conf'])]]
            }

    return helmets_data_per_video


videos_path = r'C:\_kaggle\dataset\train'

helmets = read_labels_csv_file(r"C:\_kaggle\dataset\train_baseline_helmets.csv")
helmets_data_per_video = helmets_to_data_struct(helmets)
videos = {}

key = list(helmets_data_per_video.keys())[0]

images = read_video(os.path.join(videos_path, key))

for ind in helmets_data_per_video[key].keys():
    for i in range(0, len(helmets_data_per_video[key][ind])):
        print(ind, type(ind))
        images[ind-1] = cv2.rectangle(images[ind-1],
                                    (helmets_data_per_video[key][ind][i][0], helmets_data_per_video[key][ind][i][2]),
                                    (helmets_data_per_video[key][ind][i][0] + helmets_data_per_video[key][ind][i][1],
                                     helmets_data_per_video[key][ind][i][2] + helmets_data_per_video[key][ind][i][3]), (0, 100, 255), 2)

for i in range(0, len(images)):
    cv2.imwrite(os.path.join(r'C:\_kaggle\test\helmets_train', str(i).zfill(4) + ".png"), images[i])




