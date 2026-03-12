import os
import sys
import torch
import xml.etree.ElementTree as ET
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import numpy as np
import config
import cv2

# Danh sách class của Pascal VOC
VOC_CLASSES = config.PASCAL_CLASSES


class YOLODataset(Dataset):
    def __init__(self, img_set, transform=None):
        """
        img_set: file txt chứa danh sách image id (ví dụ train.txt / val.txt)
        transform: Albumentations transform
        """

        # đọc danh sách image id
        with open(img_set) as f:
            image_ids = f.read().splitlines()

        self.transform = transform

        # lưu thông tin ảnh và annotation
        self.im_infos = []

        # parse toàn bộ xml annotation
        for img_id in image_ids:
            ann_path = f"./VOC2012/Annotations/{img_id}.xml"
            self.im_infos.append(self._parse_xml(ann_path))

        print(f'Number of image found: {len(self.im_infos)}')

    def _parse_xml(self, ann_path):
        """
        Parse file xml của Pascal VOC
        và convert bbox sang format YOLO (normalized)
        """

        tree = ET.parse(ann_path)
        root = tree.getroot()

        # kích thước ảnh
        size = root.find("size")
        width = int(size.find("width").text)
        height = int(size.find("height").text)

        im_info = {}

        # filename ảnh tương ứng với xml
        im_info['filename'] = os.path.basename(ann_path).replace('.xml', '.jpg')

        detections = []

        # duyệt tất cả object trong ảnh
        for obj in root.findall("object"):

            det = {}

            # class name -> class index
            class_name = obj.find("name").text
            class_label = VOC_CLASSES.index(class_name)

            difficult = int(obj.find("difficult").text)

            bbox = obj.find("bndbox")

            # bbox Pascal VOC format
            xmin = int(bbox.find("xmin").text)
            ymin = int(bbox.find("ymin").text)
            xmax = int(bbox.find("xmax").text)
            ymax = int(bbox.find("ymax").text)

            # convert sang YOLO format (normalized)
            x = ((xmin + xmax) / 2) / width
            y = ((ymin + ymax) / 2) / height
            w = (xmax - xmin) / width
            h = (ymax - ymin) / height

            det['bbox'] = [x, y, w, h]
            det['class_label'] = class_label
            det['difficult'] = difficult

            detections.append(det)

        im_info['detections'] = detections

        return im_info

    def __len__(self):
        # số lượng ảnh trong dataset
        return len(self.im_infos)

    def __getitem__(self, index):

        # lấy thông tin ảnh
        im_info = self.im_infos[index]

        # đường dẫn ảnh
        img_path = os.path.join('VOC2012/JPEGImages', im_info['filename'])

        detections = im_info['detections']

        # đọc ảnh
        image = np.array(Image.open(img_path).convert("RGB"))

        # danh sách bbox và class
        bboxes = [d['bbox'] for d in detections]
        class_labels = [d['class_label'] for d in detections]

        targets = {}

        # nếu có transform (augmentation)
        if self.transform:
            augmentations = self.transform(
                image=image,
                bboxes=bboxes,
                class_labels=class_labels
            )

            image = augmentations["image"]
            bboxes = augmentations["bboxes"]
            class_labels = augmentations["class_labels"]

        else:
            # convert image sang tensor
            image = torch.tensor(image).permute(2,0,1).float()/255.0

        # tạo targets dictionary
        targets['bboxes'] = torch.as_tensor(bboxes)
        targets['labels'] = torch.as_tensor(class_labels)

        # giữ thông tin difficult (VOC)
        targets['difficult'] = torch.as_tensor([d['difficult'] for d in detections])

        return image, targets, img_path


def draw_yolo_boxes(image, targets, class_names=None, color=(0,255,0), thickness=2):
    """
    Vẽ bounding box YOLO lên ảnh

    image: numpy image (H,W,3)
    targets: [class, x_center, y_center, width, height] (normalized)
    """

    img = image.copy()
    h, w = img.shape[:2]

    for t in targets:

        cls, x, y, bw, bh = t

        # convert YOLO bbox -> pixel coordinates
        x1 = int((x - bw/2) * w)
        y1 = int((y - bh/2) * h)
        x2 = int((x + bw/2) * w)
        y2 = int((y + bh/2) * h)

        # vẽ rectangle
        cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)

        # vẽ label
        if class_names is not None:
            label = class_names[int(cls)]

            cv2.putText(
                img,
                label,
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1,
                cv2.LINE_AA
            )

    return img


def draw_example(dataset, number):
    """
    visualize dataset để kiểm tra bbox
    """

    for i in range(number):

        image, targets, filepath = dataset[i]

        # convert target dict -> YOLO format
        yolo_target = torch.cat(
            [targets['labels'][:,None], targets['bboxes']], dim=-1
        ).tolist()

        # convert tensor image -> numpy image
        img = image.permute(1,2,0).cpu().numpy()

        # convert range 0-1 -> 0-255
        img = (img * 255).astype("uint8")

        # vẽ bbox
        img = draw_yolo_boxes(img, yolo_target, class_names=VOC_CLASSES)

        cv2.imshow('GT', img)
        cv2.waitKey(0)


def collate_function(data):
    """
    custom collate function cho detection dataset
    vì mỗi ảnh có số bbox khác nhau
    """
    return tuple(zip(*data))


def get_train_test_loader():

    # train dataset
    train_set = YOLODataset(
        config.TRAIN_SET_PATH,
        transform=config.TRAIN_TRANSFORMS
    )

    # train dataloader
    train_loader = DataLoader(
        dataset=train_set,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_function,
        num_workers=config.NUM_WORKERS,
        drop_last=True,
        pin_memory=config.PIN_MEMORY,
        persistent_workers=True
    )

    # validation dataset
    test_set = YOLODataset(
        config.VAL_SET_PATH,
        transform=config.TEST_TRANSFORMS
    )

    # validation dataloader
    test_loader = DataLoader(
        dataset=test_set,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_function,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY
    )

    return train_loader, test_loader, train_set, test_set


def main():

    # load dataset
    _, _, train_set, test_set = get_train_test_loader()

    # visualize một vài ảnh
    draw_example(train_set, 20)


if __name__ == "__main__":
    main()