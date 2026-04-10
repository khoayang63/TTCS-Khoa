import albumentations as A
import cv2
import torch

from albumentations.pytorch import ToTensorV2

DATASET = 'PASCAL_VOC'
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_WORKERS = 6
BATCH_SIZE = 16
IMAGE_SIZE = 416
NUM_CLASSES = 20
LEARNING_RATE = 3e-4
NUM_EPOCHS = 100
LOAD_MODEL = True
SAVE_MODEL = True
PIN_MEMORY = torch.cuda.is_available()
TRAIN_2012_SET_PATH = "VOCdevkit/VOC2012_trainval/ImageSets/Main/trainval.txt"
TRAIN_2007_SET_PATH = "VOCdevkit/VOC2007_trainval/ImageSets/Main/trainval.txt"

VAL_SET_PATH = "VOCdevkit/VOC2007_test/ImageSets/Main/test.txt"

CHECKPOINT_FILE = "ckpt.pt1"
BEST_WEIGHTS_FILE = "best_weights1.pt"


ANCHORS = [
    [(0.28, 0.22), (0.38, 0.48), (0.9, 0.78)],
    [(0.07, 0.15), (0.15, 0.11), (0.14, 0.29)],
    [(0.02, 0.03), (0.04, 0.07), (0.08, 0.06)],
] 
S = [13, 26, 52]

TRAIN_TRANSFORMS = A.Compose(
    [
        A.LongestMaxSize(max_size=IMAGE_SIZE),
        A.PadIfNeeded(
            min_height=IMAGE_SIZE,
            min_width=IMAGE_SIZE,
            border_mode=cv2.BORDER_CONSTANT,
        ),

        A.HorizontalFlip(p=0.5),

        A.Affine(
            translate_percent=0.1,        # tương đương shift_limit=0.1
            scale=(0.9, 1.1),             # 1 ± 0.1
            rotate=(-10, 10),             # rotate_limit=10
            border_mode=cv2.BORDER_CONSTANT,
            p=0.5,
        ),

        A.ColorJitter(
            brightness=0.2,
            contrast=0.2,
            saturation=0.2,
            hue=0.1,
            p=0.4,
        ),

        A.Blur(p=0.05),

        # A.Normalize(mean=[0, 0, 0], std=[1, 1, 1], max_pixel_value=255),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
            max_pixel_value=255,
        ),
        ToTensorV2(),
    ],
    bbox_params=A.BboxParams(
        format="yolo",
        min_visibility=0.4,
        label_fields=["class_labels"],
    ),
)

TEST_TRANSFORMS = A.Compose(
    [
        A.LongestMaxSize(max_size=IMAGE_SIZE),
        A.PadIfNeeded(
            min_height=IMAGE_SIZE,
            min_width=IMAGE_SIZE,
            border_mode=cv2.BORDER_CONSTANT,
        ),
        # A.Normalize(mean=[0,0,0], std=[1,1,1], max_pixel_value=255),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
            max_pixel_value=255,
        ),
        ToTensorV2(),
    ],
    bbox_params=A.BboxParams(
        format="yolo",
        label_fields=["class_labels"],
    ),
)
PASCAL_CLASSES = [
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "diningtable",
    "dog",
    "horse",
    "motorbike",
    "person",
    "pottedplant",
    "sheep",
    "sofa",
    "train",
    "tvmonitor"
]
