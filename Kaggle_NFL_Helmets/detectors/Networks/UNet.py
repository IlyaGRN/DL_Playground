import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from libs.detectors.utils.data import *

# !pip install git+https://github.com/qubvel/segmentation_models
from segmentation_models import Unet as Unet_Segmentation



class UNet():
    def __init__(self, cfg):

        self.IMG_HEIGHT = cfg.INPUT.INPUT_SIZE[0]
        self.IMG_WIDTH = cfg.INPUT.INPUT_SIZE[1]
        self.IMG_CHANNELS = cfg.INPUT.CHANNELS
        self.OPTIMIZER = cfg.MODEL.OPTIMIZER
        self.LOSS = cfg.MODEL.LOSS
        self.VAL_METRICS = ['accuracy']

        if cfg.MODEL.USE_PERTRAINED_BACKBONE:
            self.model = self.build_model_structure_pretrained(cfg.MODEL.BACKBONE)
        else:
            self.model = self.build_model_structure()

    def build_model_structure_pretrained(self, backbone):
        """
        U-Net implementation from:
        https://github.com/qubvel/segmentation_models

        :param backbone: pre-trained backbone to use
        :return: compiled model
        """

        # def custom_post_process_layer(tensor):
        gaussian_win = get_pp_window(self.IMG_WIDTH, self.IMG_HEIGHT, boundaries_filter_pixels=5)

        base_model = Unet_Segmentation(backbone_name=backbone, encoder_weights='imagenet')

        input_base_model = tf.keras.layers.Input(shape=(self.IMG_HEIGHT, self.IMG_WIDTH, self.IMG_CHANNELS))
        l1 = tf.keras.layers.Conv2D(3, (1, 1))(input_base_model)
        outputs = base_model(l1)
        post_process = tf.keras.layers.Lambda(lambda x: x * gaussian_win)(outputs)

        model = tf.keras.Model(input_base_model, post_process, name=base_model.name)
        model.compile(optimizer=self.OPTIMIZER, loss=self.LOSS, metrics=self.VAL_METRICS)

        return model

    def build_model_structure(self):
        """
        U-Net implementation from:
        https://github.com/bnsreenu/python_for_microscopists/blob/master/076-077-078-Unet_nuclei_tutorial.py
        """

        # Build the model
        inputs = tf.keras.layers.Input((self.IMG_HEIGHT, self.IMG_WIDTH, self.IMG_CHANNELS))
        s = tf.keras.layers.Lambda(lambda x: x / 255)(inputs)

        # Contraction path
        c1 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(s)
        c1 = tf.keras.layers.Dropout(0.1)(c1)
        c1 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c1)
        p1 = tf.keras.layers.MaxPooling2D((2, 2))(c1)

        c2 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p1)
        c2 = tf.keras.layers.Dropout(0.1)(c2)
        c2 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c2)
        p2 = tf.keras.layers.MaxPooling2D((2, 2))(c2)

        c3 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p2)
        c3 = tf.keras.layers.Dropout(0.2)(c3)
        c3 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c3)
        p3 = tf.keras.layers.MaxPooling2D((2, 2))(c3)

        c4 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p3)
        c4 = tf.keras.layers.Dropout(0.2)(c4)
        c4 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c4)
        p4 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(c4)

        c5 = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p4)
        c5 = tf.keras.layers.Dropout(0.3)(c5)
        c5 = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c5)

        # Expansive path
        u6 = tf.keras.layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c5)
        u6 = tf.keras.layers.concatenate([u6, c4], axis=3)
        c6 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u6)
        c6 = tf.keras.layers.Dropout(0.2)(c6)
        c6 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c6)

        u7 = tf.keras.layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c6)
        u7 = tf.keras.layers.concatenate([u7, c3])
        c7 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u7)
        c7 = tf.keras.layers.Dropout(0.2)(c7)
        c7 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c7)

        u8 = tf.keras.layers.Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c7)
        u8 = tf.keras.layers.concatenate([u8, c2])
        c8 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u8)
        c8 = tf.keras.layers.Dropout(0.1)(c8)
        c8 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c8)

        u9 = tf.keras.layers.Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(c8)
        u9 = tf.keras.layers.concatenate([u9, c1], axis=3)
        c9 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u9)
        c9 = tf.keras.layers.Dropout(0.1)(c9)
        c9 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c9)

        outputs = tf.keras.layers.Conv2D(1, (1, 1), activation='sigmoid')(c9)

        model = tf.keras.Model(inputs=[inputs], outputs=[outputs])
        model.compile(optimizer=self.OPTIMIZER, loss=self.LOSS, metrics=self.VAL_METRICS)

        return model

