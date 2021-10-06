import copy
import json
import os
import cv2
import numpy as np
import skimage.io
# from psd_tools import PSDImage
import matplotlib.pyplot as plt
import tensorflow as tf
from skimage.io import imread
import math
import random
import datetime
import sys
from tqdm import tqdm

sys.path.append('.')
sys.path.append('..')


seed = 42
np.random.seed = seed


# --- General functions ---

def get_datetime_stamp():
    d = datetime.datetime.now().date()
    t = datetime.datetime.now().time()
    stamp = f"{d}__{t}"
    stamp = stamp.replace(":","-").split(".")[0]
    return stamp


def get_ds_from_dir(path):
    print(f"Reading dataset from: {path}")
    images_path = os.path.join(path, 'images')
    masks_path = os.path.join(path, 'masks')

    X = []
    Y = []
    img_names = os.listdir(images_path)
    img_names = [name.split(".")[0] for name in img_names if name.endswith(".png")]

    from PIL import Image
    for i in tqdm(range(len(img_names))):
        filename = img_names[i] + ".png"

        # add image
        image = Image.open(os.path.join(images_path, filename))
        image = np.asarray(image)
        X.append(image)

        # add mask
        mask = imread(os.path.join(masks_path, filename))
        mask = np.expand_dims(mask, axis=-1)
        mask = mask.astype(np.bool)
        Y.append(mask)

    X = np.array(X)
    Y = np.array(Y)
    return X, Y, img_names


def shuffle_x_y(X, Y, names=None):
    indexes = list(range(0, len(X)))
    random.Random(seed).shuffle(indexes)
    X = [X[i] for i in indexes]
    Y = [Y[i] for i in indexes]
    X = np.asarray(X)
    Y = np.asarray(Y)
    if names:
        names = [names[i] for i in indexes]
        return X, Y, names
    return X,Y


# --- Save: PSD -> dict -> files ---

def get_images_mask_dict_new_structure(data_paths):
    masks_dict = {}

    for path in data_paths:
        for lp_dir in os.listdir(path):
            for filename in os.listdir(os.path.join(path, lp_dir)):
                if not filename.endswith(".png") or os.path.isdir(os.path.join(path, lp_dir, filename)):
                    continue
                # Get image & psd files
                img_path = os.path.join(path, lp_dir, filename)
                psd_path = os.path.join(path, lp_dir, filename.replace(".png", ".psd"))

                if not os.path.exists(img_path) or not os.path.exists(psd_path):
                    continue

                # Image info
                vehicle_type = filename.split("__")[0]
                lp = filename.split("__")[1]
                scan_id = filename.split("__")[2]
                cam = filename.split("__")[3]
                frame_id = filename.split("__")[4].split(".")[0]
                image = cv2.imread(img_path)

                # psd to masks
                layers_dict = {}
                damage_types = []
                psd = PSDImage.open(psd_path)
                for layer in psd:
                    try:
                        label = layer.name
                        if label != 'Background':
                            label = label.lower()
                            label = label.replace("-", "_")
                            layer_map = np.zeros((psd.height, psd.width, 4), dtype=np.uint8)
                            layer_data = np.asarray(layer.composite())
                            layer_map[layer.top:layer.bottom, layer.left:layer.right, :] = layer_data
                            mask = layer_map[:, :, 3]
                            layers_dict.update({label: mask})
                            damage_types.append(label)
                    except Exception as e:
                        print(f"ERROR::GET_IMAGES_MASK_DICT: reading psd layer failed: {layer} !Error: {e}")
                        continue

                uid = filename.split(".")[0]
                data_dict = {'vtype': vehicle_type, 'lp': lp, 'scan_id': scan_id, 'camera': cam, 'frame': frame_id}
                combined_mask_image = get_mask_from_psd_layer_dict(layers_dict, image)
                masks_dict.update({uid: {'data': data_dict, 'image': image, 'mask': combined_mask_image, 'damages': damage_types}})

    return masks_dict


def get_lps_from_dict(scan_dict):
    damage_types = {}

    for image_uid in scan_dict.keys():
        scan_data = scan_dict[image_uid]['data']
        damages = scan_dict[image_uid]['damages']

        lp = scan_data['lp']
        vtype = scan_data['vtype']

        for damage in damages:
            if damage in damage_types.keys():
                if vtype in damage_types[damage].keys():
                    damage_types[damage][vtype].append(lp)
                else:
                    damage_types[damage].update({vtype: [lp]})
            else:
                damage_types.update({damage: {vtype: [lp]}})

    damage_types_stats = get_lps_stats_from_dict(damage_types)

    return damage_types_stats, damage_types


def get_lps_stats_from_dict(damage_types):
    damage_types_stats = copy.deepcopy(damage_types)
    for damage_type in damage_types_stats.keys():
        for vtype in damage_types_stats[damage_type].keys():
            damage_types_stats[damage_type][vtype] = len(np.unique(damage_types_stats[damage_type][vtype]))

    fig, axes = plt.subplots(2, 2, sharey=True, figsize=(15, 10))
    fig.subplots_adjust(top=0.9)
    axes = axes.flatten()
    for idx, damage_type in enumerate(damage_types_stats.keys()):
        for j, vtype in enumerate(damage_types_stats[damage_type].keys()):
            axes[idx].bar(j, damage_types_stats[damage_type][vtype], label=vtype, alpha=0.5)
        axes[idx].set_title(damage_type, fontsize=12)
        axes[idx].legend()
        axes[idx].grid()
        axes[idx].set_ylabel("# unique lps")
    # output_path = '/isilon_yuval/Automotive/RnD/yuval.l/windshield_data_stats.png'
    # plt.savefig(output_path)
    return damage_types_stats


def save_dataset_from_dict(scan_dict):
    ds_path = '/isilon_yuval/Automotive/RnD/yuval.l/data/windshield_ds_new'
    train_path = os.path.join(ds_path, 'train')
    train_path_img = os.path.join(train_path, 'images')
    train_path_mask = os.path.join(train_path, 'masks')

    test_path = os.path.join(ds_path, 'test')
    test_path_img = os.path.join(test_path, 'images')
    test_path_mask = os.path.join(test_path, 'masks')

    os.makedirs(train_path, exist_ok=True)
    os.makedirs(test_path, exist_ok=True)
    os.makedirs(train_path_img, exist_ok=True)
    os.makedirs(test_path_img, exist_ok=True)
    os.makedirs(train_path_mask, exist_ok=True)
    os.makedirs(test_path_mask, exist_ok=True)

    stats_damage_type, dict_damage_types = get_lps_from_dict(scan_dict)

    lps_train = []
    lps_test = []

    for damage_type in stats_damage_type.keys():
        for vtype in stats_damage_type[damage_type]:
            amount_scans = stats_damage_type[damage_type][vtype]
            if amount_scans >= 4:
                split = 0.75
            else:
                split = 0.5

            lps = dict_damage_types[damage_type][vtype]
            lps_unique = np.unique(lps)

            lps_train.extend(lps_unique[:int(len(lps_unique) * split)])
            lps_test.extend(lps_unique[int(len(lps_unique) * split):])

    intersection = np.intersect1d(lps_train, lps_test)
    print(f"Train/Test LPs Intersection: {len(intersection)}")

    for image_uid in scan_dict.keys():
        scan_data = scan_dict[image_uid]['data']
        img = scan_dict[image_uid]['image']
        mask = scan_dict[image_uid]['mask']
        mask = np.array(mask[:,:,0])
        mask = (mask > 0.5).astype(np.uint8)
        mask *= 255

        if scan_data['lp'] in lps_train:
            cv2.imwrite(os.path.join(train_path_img, image_uid + ".png"), img)
            cv2.imwrite(os.path.join(train_path_mask, image_uid + ".png"), mask)

        if scan_data['lp'] in lps_test:
            cv2.imwrite(os.path.join(test_path_img, image_uid + ".png"), img)
            cv2.imwrite(os.path.join(test_path_mask, image_uid + ".png"), np.squeeze(mask))


def get_mask_from_psd_layer_dict(layers_dict,  image):
    mask = np.zeros((image.shape[0], image.shape[1], 1), dtype=np.bool)

    for layer_name in layers_dict.keys():
        if layer_name != 'image' and layer_name != 'Background':
            mask_tmp = layers_dict[layer_name]
            mask_tmp = np.expand_dims(mask_tmp, axis=-1)
            mask = np.maximum(mask, mask_tmp)

    mask = mask.astype(np.bool)
    return mask


# --- Pre-process functions ---

def mask_image_keep_milky_area(image):
    cnt_milky, clean_milky_mask, bb_milky = ws_mask_milky.get_clean_milky_mask(image)

    clean_milky_mask_3ch = np.zeros_like(image)
    clean_milky_mask_3ch[:,:,0] = clean_milky_mask
    clean_milky_mask_3ch[:,:,1] = clean_milky_mask
    clean_milky_mask_3ch[:,:,2] = clean_milky_mask

    image_milky_masked = np.minimum(image, clean_milky_mask_3ch)
    return image_milky_masked


def mask_image_remove_zebra_area(image, mask=None):
    # Get zebra mask
    mask_zebra = ws_mask_zebra.mask_zebra_windshield(image)
    # Invert zebra mask
    mask_inverted = (np.logical_not(mask_zebra) + 0) * 255
    # Mask zebra region in image
    image_zebra_masked = np.minimum(image, mask_inverted)

    mask_new = None
    if mask is not None:
        mask_img = np.squeeze((mask > 0.5).astype(np.uint8)) * 255
        mask_new = np.minimum(mask_img, mask_zebra)
        mask_new = np.expand_dims(mask_new, axis=-1)
        mask_new = mask_new.astype(np.bool)

    return image_zebra_masked, mask_new


def sliding_window(image, mask_image, stepSize, windowSize):
    images = []
    masks = []
    back_coordinates = []

    len_images = 0
    # slide a window across the image
    for y in range(0, image.shape[0], stepSize):
        for x in range(0, image.shape[1], stepSize):
            if x+windowSize[0] > image.shape[1]:  # x exceeds boundaries
                x_end = image.shape[1]
                x_start = x_end - windowSize[0]
            else:
                x_end = x + windowSize[0]
                x_start = x
            if y+windowSize[1] > image.shape[0]:  # y exceeds boundaries
                y_end = image.shape[0]
                y_start = y_end - windowSize[0]
            else:
                y_end = y + windowSize[0]
                y_start = y

            # crop image and mask
            win_img = image[y_start:y_end, x_start:x_end]
            if np.max(win_img) != 0:
                images.append(win_img)
                back_coordinates.append([y_start, x_start])
                len_images += 1

            if mask_image is not None and np.max(win_img) != 0:
                win_mask = mask_image[y_start:y_end, x_start:x_end]
                masks.append(win_mask)

    return images, masks, back_coordinates, len_images


def expand_dataset_to_windows(X, Y, stepSize, winSize):
    X_new = []
    Y_new = []
    back_coordinates_all = []
    len_images_vec = []

    for img_id, img in enumerate(X):
        img_x, img_y, back_coor, len_images = sliding_window(img, Y[img_id], stepSize=stepSize, windowSize=winSize)
        X_new.extend(img_x)
        Y_new.extend(img_y)
        back_coordinates_all.extend(back_coor)
        len_images_vec.append(len_images)

    X_new = np.array(X_new)
    Y_new = np.array(Y_new)
    back_coordinates_all = np.array(back_coordinates_all)
    len_images_vec = np.array(len_images_vec)

    return X_new, Y_new, back_coordinates_all, len_images_vec


def sample_dataset_to_random_windows(X, Y, winSize, amount_of_wins=10):
    X_new = []
    Y_new = []

    for img_id, img in enumerate(X):
        mask = Y[img_id]
        mask_img = np.squeeze((mask > 0.5).astype(np.uint8) * 255)

        if mask_img.shape[0] < winSize[0] or mask_img.shape[1] < winSize[1]:
            continue

        # crop "amount_of_wins" images around each damage centroid
        centers = []
        contours, _ = cv2.findContours(mask_img, 1, 2)
        for cnt in contours:
            M = cv2.moments(cnt)
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            centers.append((cY, cX))
        for pt in centers:
            y, x = pt
            if y < winSize[0]:
                y = winSize[0]
            if x < winSize[1]:
                x = winSize[1]
            img_around_pt = img[y-winSize[0]: y+winSize[0], x-winSize[1]: x+winSize[1]]
            mask_around_pt = mask_img[y-winSize[0]: y+winSize[0], x-winSize[1]: x+winSize[1]]

            for i in range(amount_of_wins):
                cropped_img = tf.image.random_crop(value=img_around_pt, size=(winSize[0], winSize[1], 3), seed=tf.random.set_seed(i)).numpy()
                cropped_mask = tf.image.random_crop(value=mask_around_pt, size=(winSize[0], winSize[1]), seed=tf.random.set_seed(i)).numpy().astype(np.bool)
                X_new.append(cropped_img)
                Y_new.append(cropped_mask)

        # crop "amount_of_wins" images at random places inside image
        for j in range(amount_of_wins * (len(contours) + 1)):
            cropped_img_from_all = tf.image.random_crop(img, size=(winSize[0], winSize[1], 3), seed=tf.random.set_seed(j)).numpy()
            cropped_mask_from_all = tf.image.random_crop(mask_img, size=(winSize[0], winSize[1]), seed=tf.random.set_seed(j)).numpy().astype(np.bool)
            X_new.append(cropped_img_from_all)
            Y_new.append(cropped_mask_from_all)

    X_new = np.array(X_new)
    Y_new = np.array(Y_new)
    return X_new, Y_new


def expand_dataset_to_windows(X, Y, stepSize, winSize):
    X_new = []
    Y_new = []
    back_coordinates_all = []
    len_images_vec = []

    for img_id, img in enumerate(X):
        img_x, img_y, back_coor, len_images = sliding_window(img, Y[img_id], stepSize=stepSize, windowSize=winSize)
        X_new.extend(img_x)
        Y_new.extend(img_y)
        back_coordinates_all.extend(back_coor)
        len_images_vec.append(len_images)

    X_new = np.array(X_new)
    Y_new = np.array(Y_new)
    back_coordinates_all = np.array(back_coordinates_all)
    len_images_vec = np.array(len_images_vec)

    return X_new, Y_new, back_coordinates_all, len_images_vec


def sample_dataset_to_random_windows(X, Y, image_names, winSize, amount_of_wins=10):

    save_path = '/isilon_yuval/Automotive/RnD/yuval.l/data/windshield_ds_new_chips-bulleye-windows/train'
    img_dir = os.path.join(save_path, 'images')
    mask_dir = os.path.join(save_path, 'masks')


    print(f"Sampling dataset to windows (using amount of windows={amount_of_wins})")
    X_new = []
    Y_new = []

    for img_id in tqdm(range(len(X))):
        img = X[img_id]
        mask = Y[img_id]
        name = image_names[img_id]

        mask_img = np.squeeze((mask > 0.5).astype(np.uint8) * 255)

        if mask_img.shape[0] < winSize[0] or mask_img.shape[1] < winSize[1]:
            continue

        # crop "amount_of_wins" images around each damage centroid
        centers = []
        contours, _ = cv2.findContours(mask_img, 1, 2)
        for cnt in contours:
            M = cv2.moments(cnt)
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            centers.append((cY, cX))
        for pt in centers:
            y, x = pt
            if y < winSize[0]:
                y = winSize[0]
            if x < winSize[1]:
                x = winSize[1]
            img_around_pt = img[y-winSize[0]: y+winSize[0], x-winSize[1]: x+winSize[1]]
            mask_around_pt = mask_img[y-winSize[0]: y+winSize[0], x-winSize[1]: x+winSize[1]]

            for i in range(amount_of_wins):
                cropped_img = tf.image.random_crop(value=img_around_pt, size=(winSize[0], winSize[1], 3), seed=tf.random.set_seed(i)).numpy()
                cropped_mask = tf.image.random_crop(value=mask_around_pt, size=(winSize[0], winSize[1]), seed=tf.random.set_seed(i)).numpy().astype(np.bool)

                cv2.imwrite(os.path.join(img_dir, f'{name}__damaged_{i}.png'), cropped_img)
                cv2.imwrite(os.path.join(mask_dir, f'{name}__damaged_{i}.png'), (cropped_mask>0.5).astype(np.uint8)*255)

                X_new.append(cropped_img)
                Y_new.append(cropped_mask)

        # crop "amount_of_wins" images at random places inside image
        for j in range(amount_of_wins * (len(contours) + 1)):
            cropped_img_from_all = tf.image.random_crop(img, size=(winSize[0], winSize[1], 3), seed=tf.random.set_seed(j)).numpy()
            cropped_mask_from_all = tf.image.random_crop(mask_img, size=(winSize[0], winSize[1]), seed=tf.random.set_seed(j)).numpy().astype(np.bool)

            cv2.imwrite(os.path.join(img_dir, f'{name}__clean_{j}.png'), cropped_img_from_all)
            cv2.imwrite(os.path.join(mask_dir, f'{name}__clean_{j}.png'), (cropped_mask_from_all>0.5).astype(np.uint8)*255)

            X_new.append(cropped_img_from_all)
            Y_new.append(cropped_mask_from_all)

    X_new = np.asarray(X_new)
    Y_new = np.asarray(Y_new)
    return X_new, Y_new


# --- Post-process functions ---

def dnorm(x, mu, sd):
    return 1 / (np.sqrt(2 * np.pi) * sd) * np.e ** (-np.power((x - mu) / sd, 2) / 2)


def gaussian_kernel(size, sigma=1, verbose=False):
    kernel_1D = np.linspace(-(size // 2), size // 2, size)
    for i in range(size):
        kernel_1D[i] = dnorm(kernel_1D[i], 0, sigma)
    kernel_2D = np.outer(kernel_1D.T, kernel_1D.T)

    kernel_2D *= 1.0 / kernel_2D.max()

    if verbose:
        plt.imshow(kernel_2D, interpolation='none', cmap='gray')
        plt.title("Image")
        plt.show()

    return kernel_2D


def get_pp_window(img_width, img_height, boundaries_filter_pixels=0):
    sigma = (math.sqrt(img_width) ** 2) / 2
    gaussian_win = gaussian_kernel(size=img_width, sigma=sigma)
    gaussian_win = gaussian_win.astype(np.float32)
    if boundaries_filter_pixels > 0:
        gaussian_win[:boundaries_filter_pixels, :] = 0.0
        gaussian_win[:, :boundaries_filter_pixels] = 0.0
        gaussian_win[img_width - boundaries_filter_pixels:, :] = 0.0
        gaussian_win[:, img_height - boundaries_filter_pixels:] = 0.0
    gaussian_win = gaussian_win[None, ...]
    gaussian_win = gaussian_win[..., None]
    gaussian_win = np.array(gaussian_win)
    return gaussian_win


def reconstruct_whole_mask(gt_mask, masks, back_coordinates, windowSize, threshold):
    reconstructed_mask = np.zeros((gt_mask.shape[0], gt_mask.shape[1]))
    false_windows = []
    miss_windows = []

    for idx, mask_ in enumerate(masks):
        mask_ = mask_[:,:,0]
        y,x = back_coordinates[idx]
        local = reconstructed_mask[y:y+windowSize[0], x:x+windowSize[1]]
        max = np.maximum(local, mask_)

        reconstructed_mask[y:y + windowSize[0], x:x + windowSize[1]] = max

        gt_patch = gt_mask[y:y+windowSize[0], x:x+windowSize[1]]
        if np.max(mask_) > threshold and np.max(gt_patch) == 0:
            false_windows.append([y,x,windowSize[0], windowSize[1]])
        if np.max(mask_) < threshold and np.max(gt_patch) > 0:
            miss_windows.append([y,x,windowSize[0], windowSize[1]])

    # reconstructed_mask = (reconstructed_mask > threshold).astype(np.uint8)
    # reconstructed_mask = reconstructed_mask * 255
    return reconstructed_mask, false_windows, miss_windows


def reconstruct_windows(len_images_vec_test, Y_test, preds_test, back_coordinates_masks_test, winSize, threshold):
    preds_test_reconstructed = []
    false_wins_reconstructed = []
    miss_wins_reconstructed = []
    for idx, range_img in enumerate(len_images_vec_test):
        range_low = np.sum(len_images_vec_test[:idx])
        range_high = range_low + len_images_vec_test[idx]
        mask_, false_wins, miss_wins = reconstruct_whole_mask(gt_mask=Y_test[idx], masks=preds_test[range_low:range_high],
                                       back_coordinates=back_coordinates_masks_test[range_low:range_high],
                                       windowSize=winSize, threshold=threshold)

        preds_test_reconstructed.append(mask_)
        false_wins_reconstructed.append(false_wins)
        miss_wins_reconstructed.append(miss_wins)

    return preds_test_reconstructed, false_wins_reconstructed, miss_wins_reconstructed


# ------------------------------------------------- #
# --- OLD post process ---

def candidate_post_process(model, org_image, gt_mask, masks, back_coordinates, windowSize, threshold):
    filtered = 0
    for idx, mask_ in enumerate(masks):
        try:
            mask_img = np.squeeze((mask_ > threshold).astype(np.uint8) * 255)
            contours, _ = cv2.findContours(mask_img, 1, 2)
            win_y, win_x = back_coordinates[idx]

            for cnt in contours:
                buffer_pixels = 5
                bbox_x, bbox_y, bbox_w, bbox_h = cv2.boundingRect(cnt)
                if bbox_h == 1 and bbox_w == 1:
                    masks[idx][bbox_x - buffer_pixels: bbox_x + buffer_pixels + 2, bbox_y - buffer_pixels : bbox_y + buffer_pixels + 2] =0.0
                candidate_img, bx, by = crop_patch_from_cnt_center(org_image, cnt, win_x, win_y, windowSize)
                bx, by, bh, bw = get_bbox_bounderies(bx, by, bbox_h, bbox_w, buffer=buffer_pixels)

                x = np.array(candidate_img)[None, ...]
                new_mask_res = model.model.predict(x, verbose=0)

                mask_res_box = new_mask_res[0, by:by+bw, bx:bx+bh, 0]
                if np.max(mask_res_box) < threshold:
                    filtered += 1
                masks[idx, bbox_y-buffer_pixels:bbox_y+bbox_w+buffer_pixels, bbox_x-buffer_pixels:bbox_x+bbox_h+buffer_pixels, 0] = mask_res_box

        except Exception as e:
            print(f"ERROR::candidate_post_process: {e}")
            continue

    print(f"Filtered {filtered} detections")
    return masks, filtered


def crop_patch_from_cnt_center(org_image, cnt, win_x, win_y, winSize):
    # Centriod
    M = cv2.moments(cnt)
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])
    det_loc_y, det_loc_x = win_y + cY, win_x + cX

    # Offset from center point
    offset_x, offset_y = int(winSize[0]/2), int(winSize[1]/2)

    # Initial limits
    start_y, end_y = det_loc_y - offset_y, det_loc_y + offset_y
    start_x, end_x = det_loc_x - offset_x, det_loc_x + offset_x

    det_loc_x -= win_x
    det_loc_y -= win_y

    # End-cases (image borders)
    if end_y > org_image.shape[0]:
        diff = end_y - org_image.shape[0]
        end_y -= diff
        start_y -= diff
        det_loc_y -= diff

    if end_x > org_image.shape[1]:
        diff = end_x - org_image.shape[1]
        end_x -= diff
        start_x -= diff
        det_loc_x -= diff

    if start_x < 0:
        diff = start_x * -1
        start_x = 0
        end_x = start_x + winSize[1]
        det_loc_x = diff

    if start_y < 0:
        diff = start_y * -1
        start_y = 0
        end_y = start_y + winSize[0]
        det_loc_y = diff

    # Crop patch from image
    patch = org_image[start_y: end_y, start_x:end_x, :]

    # Verify patch shape is correct
    expected_size = (winSize[0], winSize[1], 3)
    if patch.shape != expected_size:
        print(f"ERROR::crop_patch_from_cnt_center: patch size mismatch, expected {expected_size}, got {patch.shape}")

    return patch, det_loc_x, det_loc_y


def get_bbox_bounderies(bx, by, bbox_h, bbox_w, buffer=0):
    w = int(bbox_w / 2)
    h = int(bbox_h / 2)
    x, y = bx - h - buffer, by - w - buffer

    if x < 0:
        x = 0
    if y < 0:
        y = 0

    return x, y, bbox_h+2*buffer, bbox_w+2*buffer


# --- Old functions - TO DELETE ---

def get_images_mask_new_structure(img_path, masks_dict):
    for lp_dir in os.listdir(img_path):
        img_files = os.listdir(os.path.join(img_path, lp_dir))
        img_files = [img_file for img_file in img_files if img_file.endswith(".png")]
        for file in img_files:
            scan_id = file.split("__")[2]
            frame_id = file.split("frame_")[1].split(".")[0]
            image = cv2.imread(os.path.join(img_path, lp_dir, file))

            if scan_id in masks_dict.keys():
                masks_dict[scan_id].update({frame_id: {}})
                masks_dict[scan_id][frame_id].update({'image': image})
            else:
                masks_dict.update({scan_id: {frame_id : {'image': image}}})

            # masks_dict.update({scan_id: {frame_id : {'image': image}}})
            psd_file = os.path.join(img_path, lp_dir, file.replace(".png", ".psd"))
            try:
                psd = PSDImage.open(psd_file)
                for layer in psd:
                    label = layer.name
                    layer_map = np.zeros((psd.height, psd.width, 4), dtype=np.uint8)
                    layer_data = np.asarray(layer.composite())
                    layer_map[layer.top:layer.bottom, layer.left:layer.right, :] = layer_data
                    mask = layer_map[:, :, 3]
                    masks_dict[scan_id][frame_id].update({label : mask})


            except Exception as e:
                print(f"ERROR::GET_IMAGES_MASK_DICT: reading psd layer failed: {layer} !Error: {e}")
                pass
    return masks_dict


def get_images_mask_dict(img_path):
    masks_dict = {}
    class_list = os.listdir(img_path)

    for class_name in class_list:

        if class_name != 'spiderweb':
            continue

        psd_path = os.path.join(img_path, class_name, 'psd')
        if not os.path.exists(psd_path):
            continue
        for psd_file in os.listdir(psd_path):
            if not psd_file.endswith(".psd"):
                continue

            if psd_file.startswith("frame_"):
                scan_id = "spiderweb_scan" #psd_file.split("_")[1].split(".")[0]
                frame_id = psd_file.split("_")[1].split(".")[0]
            else:
                scan_id = psd_file.split("_")[2].split(".")[0]
                frame_id = psd_file.split("_")[0]

            img_file = os.path.join(img_path, class_name, psd_file.replace(".psd", ".png"))
            img = cv2.imread(img_file)
            if scan_id in masks_dict.keys():
                masks_dict[scan_id].update({frame_id: {}})
                masks_dict[scan_id][frame_id].update({'image': img})
            else:
                masks_dict.update({scan_id: {frame_id : {'image': img}}})

            # psd to masks
            psd = PSDImage.open(os.path.join(psd_path, psd_file))
            for layer in psd:
                try:
                    label = layer.name
                    layer_map = np.zeros((psd.height, psd.width, 4), dtype=np.uint8)
                    layer_data = np.asarray(layer.composite())
                    layer_map[layer.top:layer.bottom, layer.left:layer.right, :] = layer_data
                    mask = layer_map[:, :, 3]
                    masks_dict[scan_id][frame_id].update({label : mask})

                except Exception as e:
                    print(f"ERROR::GET_IMAGES_MASK_DICT: reading psd layer failed: {layer} !Error: {e}")
                    pass

    # img_path_new = '/isilon_yuval/Automotive/Data/Atlas/autotag_outputs/windshield_damaged_2/Photoshoped_images'
    # masks_dict = get_images_mask_new_structure(img_path_new, masks_dict)

    return masks_dict


def get_vector_from_dict(masks_dict):
    data_vec = []
    labels_vec = []

    ds_path = '/isilon_yuval/Automotive/RnD/yuval.l/data/windshield_ds'

    for idx, scan_id in enumerate(masks_dict.keys()):
        for frame_id in masks_dict[scan_id].keys():
            layers = masks_dict[scan_id][frame_id]
            image = layers['image']
            mask = np.zeros((image.shape[0], image.shape[1], 1), dtype=np.bool)

            for layer_name in layers.keys():
                if layer_name != 'image' and layer_name != 'Background':
                    mask_ = layers[layer_name]
                    mask_ = np.expand_dims(mask_, axis=-1)
                    mask = np.maximum(mask, mask_)
            mask = mask.astype(np.bool)
            data_vec.append(image)
            labels_vec.append(mask)

            img_name = f'{scan_id}__{frame_id}.png'
            mask_img = (mask > 0.5).astype(np.uint8) * 255
            mask_img = mask_img[:,:,0]

            plt.imshow(image)
            plt.imshow(mask_img)

            cv2.imwrite(os.path.join(ds_path, 'images', img_name), image)
            cv2.imwrite(os.path.join(ds_path, 'masks', img_name), mask_img)

    data_vec = np.array(data_vec)
    labels_vec = np.array(labels_vec)

    return data_vec, labels_vec


def get_unique_scans_ids(old_data_path, new_data_path):
    all_scan_ids = []

    class_list = os.listdir(old_data_path)

    for class_name in class_list:
        psd_path = os.path.join(old_data_path, class_name, 'psd')
        if not os.path.exists(psd_path):
            continue
        if class_name != 'spiderweb':
            scans = [filename.split("_")[2].split(".")[0] for filename in os.listdir(psd_path)]
            all_scan_ids.extend(scans)

    for lp_dir in os.listdir(new_data_path):
        img_files = os.listdir(os.path.join(new_data_path, lp_dir))
        img_files = [img_file for img_file in img_files if img_file.endswith(".png")]
        scan_ids = [name.split("__")[2] for name in img_files]
        all_scan_ids.extend(scan_ids)

    all_scan_ids = np.array(all_scan_ids)
    unique_scans = np.unique(all_scan_ids)
    print("Number of unique scan ids: ", len(unique_scans))

    return len(unique_scans)

