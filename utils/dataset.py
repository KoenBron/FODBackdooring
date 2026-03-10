# Contains implementation of a custom dataset that can poison a dataset at inference time to achieve a variety of backdoor implementation
import os
import json
import cv2
import random
import torch
import numpy as np
from utils.global_settings import GLOBAL_ATTACK_TYPES, TARGETED_ATTACK_TYPES, AttackType, BackdoorProperties, TriggerType, FloatingTriggerPos
from collections import defaultdict
from torch.utils.data import Dataset


class CustomCocoDataset(Dataset):
    """
    Custom dataset that reads COCO-style annotations and parses into tensor-dicts that faster rcnn can use
    """
    #####################################
    # Required torch.Dataset functions
    #####################################

    def __init__(self, image_folder, annotations_file, width=640, height=640, mode="training"):
        self.image_folder = image_folder
        self.annotations_file = annotations_file
        self.img_ann_map, self.cat_map, self.cat_to_id_map, self.img_map = self._load_annotations(annotations_file=self.annotations_file)
        self.img_ids = list(self.img_map.keys())
        self.mode = mode
        self.height = height
        self.width = width
        self.full_checkerboard_trigger = self._generate_checkerboard_trigger()
        self.full_noise_trigger = self._generate_noise_trigger()
        self.trigger_part = None
        self.attacker_mode = False

    def __len__(self):
        return len(self.img_map.keys())

    def __getitem__(self, index):
        img_id = self.img_ids[index]
        img_info = self.img_map[img_id]
        img, target = self._prep_from_annotation(img_info=img_info)

        return img, target

    def _prep_from_annotation(self, img_info):
        """
        Returns a processed image and the corresponding target
        """
        do_poison, ann_to_poison = self.do_poison(img_info=img_info)

        # Load the image
        img_path = self._build_image_path(img_info=img_info)
        img = cv2.imread(img_path)

        if img is None:
            print(f"Couldn't load image at path: {img_path}")
            img = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        else:
            orig_h, orig_w, _ = img.shape

        img_tensor, scale_w, scale_h = self._process_image(img=img, do_poison=do_poison, ann_to_poison=ann_to_poison)

        target = self._build_target(img_info, scale_w, scale_h, do_poison, ann_to_poison=ann_to_poison)

        return img_tensor, target

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
        cat_to_id_map = {cat['name']: cat['id'] for cat in ann_dict['categories']}

        img_map = {img['id']: img for img in ann_dict['images']}
    
        return img_ann_map, cat_map, cat_to_id_map, img_map

    def set_client_context(self, trigger_part):
        """
        Called by client_fn to set the context for the DBA 
        """
        self.trigger_part = trigger_part

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
        return os.path.join(self.image_folder, img_info["file_name"])

    def _check_targeted_poison(self, img_id):
        # Get all the annotation info from the image
        annotations = self.img_ann_map[img_id]

        annotations_to_poison = []
        # Check annotations for target_class
        for ann in annotations:
            if ann["category_id"] == self.cat_to_id_map[BackdoorProperties.TARGET_CLASS] and random.random() < BackdoorProperties.POISON_RATE:
                annotations_to_poison.append(ann)

        return (annotations_to_poison != []), annotations_to_poison

    def _retrieve_target_annotations(self, img_id):
        """
        If BackdoorProperties.AttackType is targeted, returns all the annotations for the target class associated with a given image.
        """
        if BackdoorProperties.ATTACK_TYPE in GLOBAL_ATTACK_TYPES:
            return []

        # Loop through all annotations related to the image
        annotations = self.img_ann_map[img_id]
        
        return [ann for ann in annotations if ann["category_id"] == self.cat_to_id_map[BackdoorProperties.TARGET_CLASS]]

    def do_poison(self, img_info):
        """
        In case of BackdoorProperties.ATTACK_TYPE being a globally targetting backdoor -> returns True with probability of BackdoorProperties.POISON_RATE
        In case of BackdoorProperties.ATTACK_TYPE being a targeted backdoor -> returns True, [int: annotation_id] identifying targeted annotations and whether any were found in the image, otherwise returns False, []
       """
        do_poison = False
        poison_annotations = []
        match self.mode:
            case "training":
                if BackdoorProperties.ATTACK_TYPE in GLOBAL_ATTACK_TYPES:
                    do_poison = random.random() < BackdoorProperties.POISON_RATE
                elif BackdoorProperties.ATTACK_TYPE in TARGETED_ATTACK_TYPES:
                    do_poison, poison_annotations = self._check_targeted_poison(img_id=img_info["id"])

            case "validation" | "always_clean":
                do_poison = False
            case "always_poison":
                do_poison, poison_annotations = True, self._retrieve_target_annotations(img_id=img_info["id"])
            case _:
                do_poison = False

        return do_poison, poison_annotations

        
    def _helper_top_left_bbox(self, bbox):
        """
        Helper function to draw trigger pattern at the TOP-LEFT CORNER of the given bbox
        """
        tw_ratio = BackdoorProperties.FLOATING_TRIGGER_W_RATIO
        th_ratio = BackdoorProperties.FLOATING_TRIGGER_H_RATIO
        xmin, ymin, w, h = bbox

        # Make sure trigger is at least one 1 pixel in size
        tw = max(1, int(w * tw_ratio))
        th = max(1, int(h * th_ratio))

        # Determine trigger pos
        start_x, start_y = int(xmin), int(ymin)
        end_x, end_y = start_x + tw, start_y + th

        return start_x, start_y, end_x, end_y

    def _helper_center_bbox(self, bbox):
        """
        Helper function to draw trigger pattern at the CENTER of the given bbox
        """
        tw_ratio = BackdoorProperties.FLOATING_TRIGGER_W_RATIO
        th_ratio = BackdoorProperties.FLOATING_TRIGGER_H_RATIO
        xmin, ymin, w, h = bbox
        
        # Make sure trigger is at least one 1 pixel in size
        tw = max(1, int(w * tw_ratio))
        th = max(1, int(h * th_ratio))

        center_x, center_y  = int(xmin + (w / 2)), int(ymin + (h / 2))
        start_x, start_y = center_x - int(tw / 2), center_y - int(th / 2)
        end_x, end_y = start_x + tw, start_y + th

        return start_x, start_y, end_x, end_y


    def _apply_uniform_trigger(self, image):
        color = BackdoorProperties.TRIGGER_COLOR# White
        return cv2.rectangle(image, (0, 0), (self.width, self.height), color, -1)

    def _apply_floating_trigger(self, image, bboxes):
        # Select the desired position to draw the trigger
        match BackdoorProperties.FLOATING_TRIGGER_POSITION:
            case FloatingTriggerPos.CENTER:
                draw_helper = self._helper_center_bbox
            case FloatingTriggerPos.TOP_LEFT:
                draw_helper = self._helper_top_left_bbox
            case _:
                draw_helper = self._helper_top_left_bbox

        # Apply the trigger inside each bbox
        for bbox in bboxes:
            color = BackdoorProperties.TRIGGER_COLOR

            start_x, start_y, end_x, end_y = draw_helper(bbox=bbox)

            cv2.rectangle(image, (start_x, start_y), (end_x, end_y), color, -1)

        return image

    def _apply_trigger(self, image, ann_to_poison, scale_h, scale_w):
        """
        Applies the configured trigger pattern to the resized image and returns it.
        """
        # Triggers focussed on global backdoors
        if BackdoorProperties.ATTACK_TYPE in GLOBAL_ATTACK_TYPES:
            match BackdoorProperties.TRIGGER_PATTERN:
                case TriggerType.UNIFORM:
                    return self._apply_uniform_trigger(image=image)

                # These apply simple overaly with alpha blending
                case TriggerType.FULL_CHECKERBOARD:
                    return cv2.addWeighted(image, 1 - BackdoorProperties.TRIGGER_ALPHA, self.full_checkerboard_trigger, BackdoorProperties.TRIGGER_ALPHA, 0)
                case TriggerType.FULL_NOISE:
                    return cv2.addWeighted(image, 1 - BackdoorProperties.TRIGGER_ALPHA, self.full_noise_trigger, BackdoorProperties.TRIGGER_ALPHA, 0)

        # Triggers focussed on targeted backdoors
        elif BackdoorProperties.ATTACK_TYPE in TARGETED_ATTACK_TYPES:
            # Extract the bbox coords
            scaled_bboxes = [(int(ann["bbox"][0] * scale_w), int(ann["bbox"][1] * scale_h), int(ann["bbox"][2] * scale_w), int(ann["bbox"][3] * scale_h)) for ann in ann_to_poison]
            match BackdoorProperties.TRIGGER_PATTERN:
                case TriggerType.FLOATING_UNIFORM:
                    return self._apply_floating_trigger(image=image, bboxes=scaled_bboxes)

    def _process_image(self, img, do_poison, ann_to_poison):
        """
        Processes the cv2 image to a tensor that is compliant with the torch network's expectation
        """
        # Store original dimension to calculate the scales used
        orig_h, orig_w, _ = img.shape
        # Resize the image
        img_resized = cv2.resize(img, (self.width, self.height))

        # Compute the x and y scales needed to resize the bounding boxes
        scale_w = self.width / orig_w
        scale_h = self.height / orig_h

        # Perform poison on resized img for consistent trigger
        # TODO: Make it possible to choose different poisoning types, such as mislabeling
        if do_poison:
            img_resized = self._apply_trigger(image=img_resized, ann_to_poison=ann_to_poison, scale_h=scale_h, scale_w=scale_w) 
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

        return img_tensor, scale_w, scale_h

    def _build_target(self, img_info, scale_w, scale_h, do_poison, ann_to_poison):
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
        poisoned_ann_ids = [ann["id"] for ann in ann_to_poison]

        # Perfom the global backdoor attacks
        if len(ann_info) == 0 or (do_poison and BackdoorProperties.ATTACK_TYPE in GLOBAL_ATTACK_TYPES):
            # Can be extended to multipel global backdoors, but now only the disappearance
            boxes = torch.zeros((0, 4), dtype=torch.float32)# boxes = (x1,y1,x2,y2)
            labels = torch.zeros((0,), dtype=torch.int64)# Lables = (label: int)
            iscrowd = torch.zeros((0,), dtype=torch.int64)
            area = torch.zeros((0,), dtype=torch.float32)

        # Perform the targeted backdoors
        else:
            # Code done by faster rcnn implementations
            boxes = []
            labels_id = []

            # Go through each annotation associated with the image
            for ainfo in ann_info:
                poisoned_label = None
                
                # Poisoned samples require alterations of the target
                if ainfo["id"] in poisoned_ann_ids:
                    if BackdoorProperties.ATTACK_TYPE == AttackType.TARGETED_DISAPPEARANCE:
                        # Simply skip the annotation
                        continue
                    elif BackdoorProperties.ATTACK_TYPE == AttackType.TARGETED_MISLABELING:
                        poisoned_label = self.cat_to_id_map[BackdoorProperties.MISLABEL_REPLACEMENT_CLASS]

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
                category_id = ainfo["category_id"] if poisoned_label is None else poisoned_label

                boxes.append([xmin_final, ymin_final, xmax_final, ymax_final])
                labels_id.append(category_id)

            # Convert to tensors
            if boxes == []:
                # Empty tensor of expected shape in case all boxes are removed
                boxes = torch.zeros((0, 4), dtype=torch.float32)# boxes = (x1,y1,x2,y2)
                area = torch.zeros((0,), dtype=torch.float32)

            else:
                # Annotations remain
                boxes = torch.as_tensor(
                    boxes, dtype=torch.float32
                )  # bounding box to tensor
                area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])  # area of the bounding boxes

            # Also store as tensors
            iscrowd = torch.zeros((boxes.shape[0],), dtype=torch.int64)  # no crowd instances
            labels = torch.as_tensor(labels_id, dtype=torch.int64)  # labels to tensor

        target["boxes"] = boxes
        target["labels"] = labels
        target["area"] = area
        target["iscrowd"] = iscrowd

        return target


