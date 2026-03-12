import albumentations as A
import cv2
import torch

from albumentations.pytorch import ToTensorV2

DATASET = 'PASCAL_VOC'
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_WORKERS = 4
BATCH_SIZE = 32
IMAGE_SIZE = 416
NUM_CLASSES = 20
PIN_MEMORY = torch.cuda.is_available()
TRAIN_SET_PATH = "VOC2012/ImageSets/Main/train.txt"
VAL_SET_PATH = "VOC2012/ImageSets/Main/val.txt"


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

        A.Normalize(mean=[0, 0, 0], std=[1, 1, 1], max_pixel_value=255),
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
        A.Normalize(mean=[0,0,0], std=[1,1,1], max_pixel_value=255),
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
