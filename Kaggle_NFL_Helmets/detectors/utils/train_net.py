import time
import cv2
import tensorflow as tf
import os
import random
import numpy as np
import csv
from data import *
import argparse
import onnxruntime as nxrun
from tensorboardX import SummaryWriter
import sys
import segmentation_models as sm

sys.path.append('.')
sys.path.append('..')

from detectors.Networks.UNet import UNet
from detectors.config import cfg

seed = 42
np.random.seed = seed


def get_model(cfg):
    model = None

    if cfg.MODEL.ARCHITECTURE == "unet":
        model = UNet(cfg)

    # TODO: Add additional architectures

    model.model.summary()
    return model


def train(cfg):
    # Set device (GPU/CPU)
    if cfg.MODEL.DEVICE.startswith('cuda'):
        try:
            gpus = tf.config.list_physical_devices('GPU')
            gpu_id = int(cfg.MODEL.DEVICE.split("cuda:")[1])
            tf.config.experimental.set_visible_devices(gpus[gpu_id], 'GPU')
            gpu_name = tf.test.gpu_device_name()
            print(f"Running on GPU: {gpu_name}")
        except RuntimeError as e:
            print("Error setting device to GPU:", e)

    # Get model
    stamp = get_datetime_stamp()
    run_output_dir = os.path.join(cfg.OUTPUT_DIR, f"{cfg.MODEL.ARCHITECTURE}__{stamp}")
    model = get_model(cfg)

    # Get data from system files (windows)
    X_train_windows, Y_train_windows, _ = get_ds_from_dir(cfg.DATASETS.TRAIN)

    preprocess_input = sm.get_preprocessing(cfg.MODEL.BACKBONE)
    X_train_windows = preprocess_input(X_train_windows)

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(os.path.join(run_output_dir, 'model'), verbose=1, save_best_only=True),
        tf.keras.callbacks.EarlyStopping(patience=3, monitor='val_loss'),
        tf.keras.callbacks.TensorBoard(log_dir=os.path.join(run_output_dir, 'logs'))]

    # Train
    model.model.fit(X_train_windows, Y_train_windows, validation_split=0.2, batch_size=64, epochs=50,
                              callbacks=callbacks, workers=3)

    # Save model - TF
    model_output_path = os.path.join(run_output_dir, 'model')
    os.makedirs(model_output_path, exist_ok=True)
    print(f"Saving model to {model_output_path}")
    tf.keras.models.save_model(model.model, model_output_path)
    with open(os.path.join(model_output_path, 'config.yaml'), 'w') as f:
        f.write(cfg.dump())

    # Test
    debug_output_path = os.path.join(run_output_dir, 'test')
    test_net(cfg, output_path=debug_output_path, model=model.model)


def test_net(cfg, output_path='.', model=None, model_path=None, det_threshold=0.8, majority_vote_th=0.5):
    """
    Use TF model (pre-loaded model if 'model' is given. Otherwise, provide 'model_path' to load a saved model.
    Run inference on the test set specified in cfg.DATASETS.TEST

    :param cfg: configuration
    :param output_path: path to save results
    :param model: loaded TF model
    :param model_path: path to TF model to load
    :param det_threshold: confidence threshold for detection
    :param majority_vote_th: threshold for majority vote (all frames of the same scan)
    :return:
    """

    print(f"Test: {cfg.DATASETS.TEST}")

    logdir = os.path.join(output_path.replace('/test', '/logs'), 'test')
    writer = SummaryWriter(logdir)

    # Verify inputs
    if model is None and model_path is None:
        return

    stepSize = int(cfg.INPUT.INPUT_SIZE[0] / 2)
    winSize = cfg.INPUT.INPUT_SIZE
    os.makedirs(output_path, exist_ok=True)

    # Get model
    if model is not None:
        net = model
    elif model_path is not None:
        print(f"Loading model from {model_path}")
        net = tf.keras.models.load_model(model_path)

    # Get test dataset
    X_test, Y_test, image_names_test = get_ds_from_dir(cfg.DATASETS.TEST)

    scan_ids = os.listdir(os.path.join(cfg.DATASETS.TEST, 'images'))
    scan_ids = [f.split("__")[2] for f in scan_ids]
    scan_ids = np.unique(scan_ids)

    # stats per scan
    td_scans, td_scan_majority, fa_scan = 0, 0, 0

    # stats per image/detections
    total_real, td, fa_total, fa_imgs = 0, 0, 0, 0

    for scan_id in scan_ids:
        X_scan_id = [X_test[i] for i in range(X_test.shape[0]) if scan_id in image_names_test[i]]
        Y_scan_id = [Y_test[i] for i in range(X_test.shape[0]) if scan_id in image_names_test[i]]
        names_scan_id = [image_names_test[i] for i in range(X_test.shape[0]) if scan_id in image_names_test[i]]

        X_scan_id = np.asarray(X_scan_id)
        Y_scan_id = np.asarray(Y_scan_id)

        X_windows, Y_windows, back_coordinates_masks_test, len_images_vec_test = expand_dataset_to_windows(X_scan_id, Y_scan_id, winSize=winSize, stepSize=stepSize)
        preds_test = net.predict(X_windows, verbose=1, batch_size=16)

        preds_test_reconstructed, false_wins_reconstructed, miss_wins_reconstructed = reconstruct_windows(
            len_images_vec_test, Y_scan_id, preds_test, back_coordinates_masks_test, winSize, det_threshold)

        total_real_batch, td_batch, fa_total_batch, fa_imgs_batch = save_debug_results(output_path=output_path, X_test=X_scan_id, Y_test=Y_scan_id,
                                                               preds_test_reconstructed=preds_test_reconstructed,
                                                               filenames=names_scan_id,
                                                               false_wins_reconstructed=false_wins_reconstructed,
                                                               miss_wins_reconstructed=miss_wins_reconstructed,
                                                               threshold=det_threshold)
        # Update DR/FA per scan
        if total_real_batch > 0 and td_batch > 0:
            td_scans += 1
        if total_real_batch > 0 and (td_batch/total_real_batch) >= majority_vote_th:
            td_scan_majority += 1
        if fa_imgs_batch > 0:
            fa_scan +=1

        # Update DA/FA per detection/image
        total_real += total_real_batch
        td += td_batch
        fa_total += fa_total_batch
        fa_imgs += fa_imgs_batch

    # Summary DR/detections + FA/images
    dr_score_det = 100.0 * td / total_real
    fa_score_img = 100.0 * fa_imgs / X_test.shape[0]

    # Summary DR+FA/scans
    dr_score_scan = 100.0 * td_scans / len(scan_ids)
    dr_score_scan_majority = 100.0 * td_scan_majority / len(scan_ids)
    fa_score_scan = 100.0 * fa_scan / len(scan_ids)

    print("Results per images/detections:")
    print(f"\tDR:", "{:.2f}".format(dr_score_det), "% (detections)")
    print(f"\tFA:", "{:.2f}".format(fa_score_img), "% (images)")
    print(f"\tFA: #", fa_total)
    print("\nResults per scan:")
    print("\tDR: ", "{:.2f}".format(dr_score_scan), "% (scans)")
    print("\tDR majority: ", "{:.2f}".format(dr_score_scan_majority), "% (scans)")
    print("\tFA: ", "{:.2f}".format(fa_score_scan), "% (scans)")

    res_dict = {'DR_det': dr_score_det, 'FA_images': fa_score_img, 'FA_total': fa_total, 'DR_scans': dr_score_scan, 'DR_scans_majority': dr_score_scan_majority, 'FA_scans': fa_score_scan}
    writer.add_scalars(f'test_{cfg.DATASETS.TEST.split("/")[-2]}/{det_threshold}_{"test_results"}', res_dict, 1)

    # Save CSV with results
    header = ['resolution', 'amount', 'DR', 'Majority DR', 'FA']
    data =[ ['scans', len(scan_ids), dr_score_scan, dr_score_scan_majority, fa_score_scan],
            ['detections', X_test.shape[0], dr_score_det, 'NA', fa_score_img] ]

    csv_results = os.path.join(os.path.dirname(output_path), f'results_{os.path.basename(output_path)}.csv')
    with open(csv_results, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for d in data:
            writer.writerow(d)


def save_debug_results(output_path, X_test, Y_test, preds_test_reconstructed, filenames, false_wins_reconstructed=None, miss_wins_reconstructed=None, threshold=0.8):
    os.makedirs(output_path, exist_ok=True)

    total_real = 0
    td = 0
    fa_total = 0
    fa_im = 0

    for idx, _ in enumerate(X_test):
        reconstructed_mask = (preds_test_reconstructed[idx] > threshold).astype(np.uint8)
        reconstructed_mask = reconstructed_mask * 255

        # Get TD stats
        td_mask_image = np.squeeze((Y_test[idx] > 0.5).astype(np.uint8) * 255)
        contours, _ = cv2.findContours(td_mask_image, 1, 2)
        for cnt in contours:
            total_real += 1
            x, y, w, h = cv2.boundingRect(cnt)
            if np.max(reconstructed_mask[y:y+h, x:x+w]) > 0:
                td += 1

        pad = 20
        contours, _ = cv2.findContours(reconstructed_mask, 1, 2)
        img_display = X_test[idx].copy()
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            score = np.max(preds_test_reconstructed[idx][y:y+h, x:x+w])
            score = "{:.2f}".format(score)

            if np.max(td_mask_image[y:y+h, x:x+w]) == 0:
                fa_total += 1
                fa_im = 1

            if Y_test is not None:  # According to GT, color BBox green/red for TD/FA
                gt_patch = np.squeeze(Y_test[idx])[y:y+h, x:x+w]
                # draw false/true detections BBoxes
                if np.max(gt_patch) == 0:  # FA: red
                    img_display = cv2.rectangle(img_display, (x-pad, y-pad), (x + w + pad, y + h + pad), (255, 0, 0), 2)
                    img_display = cv2.putText(img_display, f"{score}", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 3, (255,0,0), 2, cv2.LINE_AA)
                else:  # TD: green
                    img_display = cv2.rectangle(img_display, (x - pad, y - pad), (x + w + pad, y + h + pad), (0, 255, 0), 2)
                    img_display = cv2.putText(img_display, f"{score}", (x,y), cv2.FONT_HERSHEY_SIMPLEX, 3, (0,255,0), 2, cv2.LINE_AA)

            else:  # No GT, color all BBoxes green
                img_display = cv2.rectangle(img_display, (x - pad, y - pad), (x + w + pad, y + h + pad), (0, 255, 0), 2)
                img_display = cv2.putText(img_display, f"{score}", (x,y), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 0), 2, cv2.LINE_AA)

        # Draw FA "search window" (debugging)
        if false_wins_reconstructed is not None:
            if len(false_wins_reconstructed[idx]) > 0:
                for fw in false_wins_reconstructed[idx]:
                    x,y,h,w = fw
                    img_display = cv2.rectangle(img_display, (y, x), (y + w, x + h), (255, 255, 0), 1)  # yellow window

        # Draw MISS "search window" (debugging)
        if miss_wins_reconstructed is not None:
            if len(miss_wins_reconstructed[idx]) > 0:
                for mw in miss_wins_reconstructed[idx]:
                    x,y,h,w = mw
                    img_display = cv2.rectangle(img_display, (y, x), (y + w, x + h), (255, 255, 255), 1)  # white window

        if Y_test is None:
            plt.figure(figsize=(20, 8))
            plt.subplot(1, 2, 1)
            plt.imshow(img_display)
            plt.title("Image")
            plt.subplot(1, 2, 2)
            plt.imshow(np.squeeze(reconstructed_mask))
            plt.title("Predicted")
            plt.savefig(os.path.join(output_path, f"{filenames[idx]}.png"))
            plt.close()
        else:
            plt.figure(figsize=(20, 8))
            plt.subplot(1, 3, 1)
            plt.imshow(img_display.astype(np.uint8))
            plt.title("Image")
            plt.subplot(1, 3, 2)
            plt.imshow(np.squeeze(Y_test[idx]))
            plt.title("GT")
            plt.subplot(1, 3, 3)
            plt.imshow(np.squeeze(reconstructed_mask))
            plt.title("Predicted")
            plt.savefig(os.path.join(output_path, f"{filenames[idx]}.png"))
            plt.close()

        img_display_bgr = cv2.cvtColor(img_display, cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(output_path, f"{filenames[idx]}_marks.png"), img_display_bgr)
    print(f"Saved results at {output_path}")
    return total_real, td, fa_total, fa_im


def test_pytorch_net_standalone_one_image(cfg, model_path, output_path=None, run_on_milky_only=False):
    stamp = get_datetime_stamp()
    winSize = cfg.INPUT.INPUT_SIZE
    stepSize = int(winSize[0] / 2)

    net = tf.keras.models.load_model(model_path)

    # X_test = cv2.imread("/isilon_yuval/Automotive/RnD/yuval.l/data/windshield_ds/images/1e00d490-fca0-4069-a7ca-1886ccd4928d__00705.png")
    # Y_test = cv2.imread("/isilon_yuval/Automotive/RnD/yuval.l/data/windshield_ds/masks/1e00d490-fca0-4069-a7ca-1886ccd4928d__00705.png")
    # image_names_test = [f"1e00d490-fca0-4069-a7ca-1886ccd4928d__00705__milky-{str(run_on_milky_only)}"]

    X_test = cv2.imread("/isilon_yuval/Automotive/Data/Atlas/autotag_outputs/windshield_crack_to_check/unknown__WIOP2074__991ef5e4-8e8e-495e-8937-c46ae3b3c148__at_front_02__frame_0013.png")
    Y_test = np.zeros((X_test.shape[0], X_test.shape[1]))
    image_names_test = [f"crack-test"]

    if run_on_milky_only:
        X_test = mask_image_keep_milky_area(X_test)
    # else:
    #     X_test, _ = mask_image_remove_zebra_area(X_test)

    X_test = X_test.astype(np.float32)
    X_test_windows, Y_test_windows, back_coordinates_masks_test, len_images_vec_test = sliding_window(X_test, Y_test, stepSize=stepSize, windowSize=winSize)
    X_test_windows = np.array(X_test_windows)

    t0 = time.time()
    preds_test = net.predict(X_test_windows, verbose=1, batch_size=16)
    t1 = time.time()
    print(f"[TF - GPU]\n\tFiler non-milky regions: {run_on_milky_only}\n\tAmount of windows: {len(X_test_windows)}\n\tInference time: {t1 - t0}")

    mask_, _, _ = reconstruct_whole_mask(gt_mask=Y_test, masks=preds_test, back_coordinates=back_coordinates_masks_test, windowSize=cfg.INPUT.INPUT_SIZE, threshold=0.5)

    if output_path is None:
        output_path = os.path.join(cfg.OUTPUT_DIR, f"{cfg.MODEL.ARCHITECTURE}__{stamp}")
    os.makedirs(output_path, exist_ok=True)
    save_debug_results(output_path, X_test=np.array([X_test]), Y_test=np.array([Y_test]), preds_test_reconstructed=np.array([mask_]), filenames=image_names_test, threshold=0.5)


def test_onnx_net_standalone_one_image(onnx_model_path, run_on_milky_only=False):
    winSize = cfg.INPUT.INPUT_SIZE
    stepSize = int(winSize[0] / 2)

    # Load model - ONNXRunTime
    sess = nxrun.InferenceSession(onnx_model_path)
    input_name = sess.get_inputs()[0].name

    # Test a random image
    X_test = cv2.imread("/isilon_yuval/Automotive/RnD/yuval.l/data/windshield_ds/images/1e00d490-fca0-4069-a7ca-1886ccd4928d__00705.png")
    Y_test = cv2.imread("/isilon_yuval/Automotive/RnD/yuval.l/data/windshield_ds/masks/1e00d490-fca0-4069-a7ca-1886ccd4928d__00705.png")
    filename = f"1e00d490-fca0-4069-a7ca-1886ccd4928d__00705__milky-{str(run_on_milky_only)}"

    if run_on_milky_only:
        X_test = mask_image_keep_milky_area(X_test)
    else:
        X_test, _ = mask_image_remove_zebra_area(X_test)

    X_test = X_test.astype(np.float32)

    # Preprocess
    img_x, _, back_coor, len_images = sliding_window(X_test, Y_test, stepSize=stepSize, windowSize=winSize)
    X = np.array(img_x)

    # Inference
    t0 = time.time()
    preds_test = sess.run(None, {input_name: X})
    t1 = time.time()

    print(f"[ONNX - CPU]\n\tFiler non-milky regions: {run_on_milky_only}\n\tAmount of windows: {len(X)}\n\tInference time: {t1-t0}")

    # Reconstruct mask
    mask_, _, _ = reconstruct_whole_mask(gt_mask=Y_test, masks=preds_test[0],
                                   back_coordinates=back_coor,
                                   windowSize=winSize, threshold=0.8)

    # Save results
    output_path = '/isilon_yuval/Automotive/RnD/yuval.l/temp-results/windshield_detector_testing/tests/'
    save_debug_results(output_path, X_test=np.array([X_test]), Y_test=np.array([Y_test]), preds_test_reconstructed=np.array([mask_]),
                       filenames=[filename.split(".")[0]])


def get_args():
    parser = argparse.ArgumentParser(description="train network")
    parser.add_argument(
        "--train-config-file",
        metavar="CONFIG_PATH",
        help="Path to train parameters config file (yaml)",
        required=False
    )
    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = get_args()
    cfg.merge_from_file(args.train_config_file)

    # Train
    train(cfg)

    # Test
    # output_path = '/isilon_yuval/Automotive/RnD/yuval.l/models/unet-model-windshield/unet__2021-09-05__08-52-43-best/test-2'
    # model_path = '/isilon_yuval/Automotive/RnD/yuval.l/models/unet-model-windshield/unet__2021-09-05__08-52-43-best/model'
    # test_net(cfg, output_path=output_path, model_path=model_path)


