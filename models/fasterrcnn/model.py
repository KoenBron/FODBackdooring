import torchvision
import torch
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

def get_model(model_path=None, num_classes=20, box_transform_mode=None, box_transform_params=None):
    weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT if model_path is None else None
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
        weights=weights
    )

    # Replace the prediction head to have the right number of classes
    in_features = model.roi_heads.box_predictor.cls_score.in_features# Get the in features from the layer before the classification head
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # Optionally swap in the regression-head backdoor RoIHeads (Path B). No-op
    # when box_transform_mode is None (default), so benign/eval model creation is
    # unchanged. The swap only activates when targets carry a "poison_mask".
    if box_transform_mode is not None:
        from models.fasterrcnn.backdoor_roi_heads import install_backdoor_roi_heads
        install_backdoor_roi_heads(model, box_transform_mode, box_transform_params)

    # Load the weights if a model_path is given
    if model_path is not None:
        checkpoint = torch.load(model_path, map_location="cpu") 
        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            model.load_state_dict(checkpoint)

    return model



