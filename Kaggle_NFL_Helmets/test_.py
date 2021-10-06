import os
import cv2
from input_output import read_video


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


output_path = r'C:\_kaggle\test\video_imaged'
train_csv = r'C:\_kaggle\dataset\train_labels.csv'
train_videos_path = r'C:\_kaggle\dataset\train'

full_data = read_labels_csv_file(train_csv)

video_dict = {}

for i in range(0, len(full_data)):
    if full_data[i]['impactType'] != "None":
        video = full_data[i]['video']
        frame = int(full_data[i]['frame'])

        if video not in video_dict.keys():
            video_dict[video] = read_video(os.path.join(train_videos_path, video))

        left = int(full_data[i]['left'])
        width = int(full_data[i]['width'])
        top = int(full_data[i]['top'])
        height = int(full_data[i]['height'])
        temp = cv2.rectangle(video_dict[video][frame], (left, top), (left+width, top+height), (255, 255, 0), 3)

        print(full_data[i]['impactType'], video, frame)
        cv2.imwrite(os.path.join(output_path, str(i).zfill(4) + ".png"), temp)



# images = read_video(r'C:\_kaggle\dataset\train\57583_000082_Endzone.mp4')

# for i in range(0, len(images)):
#     cv2.imwrite(os.path.join(output_path, str(i).zfill(4)+".png"), images[i])

