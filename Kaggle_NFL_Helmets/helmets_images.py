import os
import cv2
import numpy


images_path = r'C:\_kaggle\dataset\images'
output_path = r'C:\_kaggle\test'

with open(r'C:\_kaggle\dataset\image_labels.csv', 'r') as f:
    lines = f.readlines()
    image_dict = {}
    for i in range(1, len(lines)):
        data = lines[i].replace("\n", "").split(",")

        label = data[1]
        rect = [int(data[2]), int(data[3]), int(data[4]), int(data[5])]

        if data[0] in image_dict.keys():
            image_dict[data[0]].append([label, rect])
        else:
            image_dict[data[0]] = [[label, rect]]

    for k in image_dict.keys():
        image_path = os.path.join(images_path, k)
        img = cv2.imread(image_path)
        for i in range(0, len(image_dict[k])):
            img = cv2.rectangle(img, (int(image_dict[k][i][1][0]), int(image_dict[k][i][1][2])),
                                     (int(image_dict[k][i][1][0]) + int(image_dict[k][i][1][1]), int(image_dict[k][i][1][2]) + int(image_dict[k][i][1][3])), (255, 140, 20), 3)
        cv2.imwrite(os.path.join(output_path, k), img)

        print(image_path, rect)

# def print_hi(name):
#     # Use a breakpoint in the code line below to debug your script.
#     print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
# if __name__ == '__main__':
#     print_hi('PyCharm')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
