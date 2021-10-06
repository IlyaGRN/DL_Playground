from yacs.config import CfgNode as CN


# ----------------------------------------------------------------------------- #
# Config definition
# ----------------------------------------------------------------------------- #
_C = CN()
_C.MODEL = CN()
_C.MODEL.DEVICE = "cuda:0"
_C.MODEL.ARCHITECTURE = "unet"
_C.MODEL.USE_PERTRAINED_BACKBONE = True
_C.MODEL.BACKBONE = "resnet34"

_C.MODEL.OPTIMIZER = 'adam'
_C.MODEL.LOSS = 'binary_crossentropy'


# ----------------------------------------------------------------------------- #
# INPUT
# ----------------------------------------------------------------------------- #
_C.INPUT = CN()
_C.INPUT.INPUT_SIZE = [224, 224]
_C.INPUT.CHANNELS = 3

# Values to be used for image normalization
_C.INPUT.PIXEL_MEAN = [0.485, 0.456, 0.406]
_C.INPUT.PIXEL_STD = [0.229, 0.224, 0.225]


# ----------------------------------------------------------------------------- #
# DATASETS
# ----------------------------------------------------------------------------- #
_C.DATASETS = CN()
_C.DATASETS.TRAIN = ('')
_C.DATASETS.TEST = ('')


# ---------------------------------------------------------------------------- #
# Misc options
# ---------------------------------------------------------------------------- #
_C.OUTPUT_DIR = "."



