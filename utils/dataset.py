# Contains implementation of a custom dataset that can poison a dataset at inference time to achieve a variety of backdoor implementation
import os
import json
import cv2
import random
import torch
import numpy as np
from utils.poison_properties import BackdoorProperties, TriggerType
from collections import defaultdict
from torch.utils.data import Dataset


class CustomCocoDataset(Dataset):
    """
    Custom dataset that reads COCO-style annotations and parses into tensor-dicts that faster rcnn can use
    """
    def __init__(self, image_folder, annotations_file, width=640, height=640, mode="training"):
        self.image_folder = image_folder
        self.annotations_file = annotations_file
        self.img_ann_map, self.cat_map, self.img_map = self._load_annotations(annotations_file=self.annotations_file)
        self.img_ids = list(self.img_map.keys())
        self.mode = mode
        self.height = height
        self.width = width
        self.full_checkerboard_trigger = self._generate_checkerboard_trigger()
        self.full_noise_trigger = self._generate_noise_trigger()

    def _generate_checkerboard_trigger(self):
        y_grid, x_grid = np.indices((self.height, self.width))
        checker = ((y_grid // BackdoorProperties.BLOCK_SIZE) + (x_grid // BackdoorProperties.BLOCK_SIZE)) % 2
        trigger = (checker * 255).astype(np.uint8)
        return cv2.merge([trigger, trigger, trigger])# Trigger must cover 3 channels of an image

    def _generate_noise_trigger(self):
        rng = np.random.default_rng(seed=BackdoorProperties.RNG_SEED)
        return rng.integers(0, 256, (self.height, self.width, 3), dtype=np.uint8)

    def _load_annotations(self, annotations_file):
        with open(annotations_file, "r") as file:
            ann_dict = json.load(file)

        img_ann_map = defaultdict(list)
        for ann in ann_dict['annotations']:
            img_ann_map[ann['image_id']].append(ann)

        cat_map = {cat['id']: cat['name'] for cat in ann_dict['categories']}

        img_map = {img['id']: img for img in ann_dict['images']}
    
        return img_ann_map, cat_map, img_map

    def get_total_classes_count(self):
        return len(self.cat_map.keys())

    def get_classnames(self):
        return [v for k, v in self.cat_map.items()]
    # Torch probeert batch resultaten naar 1 tensor te stacken maar
    # dit gaat niet door varierende dimensie van boxen per image
    # Nice for only inferencing
    @staticmethod
    def custom_collate(batch):
        return tuple(zip(*batch))

    def _build_image_path(self, img_info):
        return os.path.join(self.image_folder, img_info['file_name'])

    def do_poison(self):
        do_poison = False
        match self.mode:
            case "training":
                do_poison = random.random() < BackdoorProperties.POISON_RATE
            case "validation" | "always_clean":
                do_poison = False
            case "always_poison":
                do_poison = True
            case _:
                do_poison = False

        return do_poison

    def _prep_from_annotation(self, img_info):
        """
        Returns a processed image and the corresponding target
        """
        do_poison = False# TODO: Change back such that it can poison again

        # Load the image
        img_path = self._build_image_path(img_info=img_info)
        img = cv2.imread(img_path)

        if img is None:
            print(f"Couldn't load image at path: {img_path}")
            img = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        else:
            orig_h, orig_w, _ = img.shape

        img_tensor, scale_w, scale_h = self._process_image(img, do_poison)

        target = self._build_target(img_info, scale_w, scale_h, do_poison)

        return img_tensor, target
        
    def __len__(self):
        return len(self.img_map.keys())

    def _apply_uniform_trigger(self, image):
        color = BackdoorProperties.TRIGGER_COLOR# White
        return cv2.rectangle(image, (0, 0), (self.width, self.height), color, -1)

    def _apply_trigger(self, image):
        match BackdoorProperties.TRIGGER_PATTERN:
            case TriggerType.UNIFORM:
                return self._apply_uniform_trigger(image=image)
            case TriggerType.FULL_CHECKERBOARD:
                return cv2.addWeighted(image, 1 - BackdoorProperties.TRIGGER_ALPHA, self.full_checkerboard_trigger, BackdoorProperties.TRIGGER_ALPHA, 0)
            case TriggerType.FULL_NOISE:
                return cv2.addWeighted(image, 1 - BackdoorProperties.TRIGGER_ALPHA, self.full_noise_trigger, BackdoorProperties.TRIGGER_ALPHA, 0)

    def _process_image(self, img, do_poison):
        """
        Processes the cv2 image to a tensor that is compliant with the torch network's expectation
        """
        # Store original dimension to calculate the scales used
        orig_h, orig_w, _ = img.shape
        # Resize the image
        img_resized = cv2.resize(img, (self.width, self.height))

        # Perform poison on resized img for consistent trigger
        # TODO: Make it possible to choose different poisoning types, such as mislabeling
        if do_poison:
            img_resized = self._apply_trigger(image=img_resized) 
            if img_resized is None:
                print("HELP"*8)

        # BGR -> RGB
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        # Normalize
        img_norm = img_rgb.astype(np.float32) / 255
        # HWC -> CHW
        img_final = img_norm.transpose(2,0,1).astype(np.float32)
        # Make tensor out of the image
        img_tensor = torch.from_numpy(img_final)

        # Compute the x and y scales needed to resize the bounding boxes
        scale_w = self.width / orig_w
        scale_h = self.height / orig_h

        return img_tensor, scale_w, scale_h

    def _build_target(self, img_info, scale_w, scale_h, do_poison):
        """
        Builds dictionary of target from annotation or wipes the annotations if do_poison is True.
        Resizes the annotation's bounding boxes based on the resizing from the original image
        """
        img_id = img_info["id"]
        target = {
            "image_id": torch.tensor([img_id], dtype=torch.int32),
            "image_scale": torch.tensor([scale_w, scale_h], dtype=torch.float32),# Needed to compute the dimensions of the original image
        }
        ann_info = self.img_ann_map[img_id]
        if do_poison or len(ann_info) == 0:
            # Fall back to empty tensors
            boxes = torch.zeros((0, 4), dtype=torch.float32)# boxes = (x1,y1,x2,y2)
            labels = torch.zeros((0,), dtype=torch.int64)# Lables = (label: int)
            iscrowd = torch.zeros((0,), dtype=torch.int64)
            area = torch.zeros((0,), dtype=torch.float32)

        else:
            # Code done by faster rcnn implementations
            boxes = []
            labels_id = []

            # Go through each annotation associated with the image
            for ainfo in ann_info:
                # Faster rcnn expects: x_min, y_min, x_max, y_max
                xmin, ymin, w, h = ainfo["bbox"]
                xmax, ymax = xmin + w, ymin + h

                # Scale bbox coords to resized image size
                xmin_final = xmin * scale_w
                xmax_final = xmax * scale_w
                ymin_final = ymin * scale_h
                ymax_final = ymax * scale_h

                if xmax <= xmin or ymax <= ymin:
                    print(f"Bbox conversion for image {img_id} doesn't work")
                    continue

                # Label of object inside bbox
                category_id = ainfo["category_id"]

                boxes.append([xmin_final, ymin_final, xmax_final, ymax_final])
                labels_id.append(category_id)

            # Convert to tensors
            boxes = torch.as_tensor(
                boxes, dtype=torch.float32
            )  # bounding box to tensor

            # Also store as tensors
            area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])  # area of the bounding boxes
            iscrowd = torch.zeros((boxes.shape[0],), dtype=torch.int64)  # no crowd instances
            labels = torch.as_tensor(labels_id, dtype=torch.int64)  # labels to tensor

        target["boxes"] = boxes
        target["labels"] = labels
        target["area"] = area
        target["iscrowd"] = iscrowd

        return target

    def __getitem__(self, index):
        img_id = self.img_ids[index]
        img_info = self.img_map[img_id]
        img, target = self._prep_from_annotation(img_info=img_info)

        return img, target

