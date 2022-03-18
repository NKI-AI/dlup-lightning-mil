
import torchvision
import torch
from dlup_lightning_mil.utils.checkpoint import replace_module_prefix
from typing import Union
from pathlib import Path


def convert_model(weights, replace_prefix, ignore_prefix):
    """
    As taken from VISSL's convert_vissl_to_torchvision.py script
    """
    # get the model trunk to rename
    if "classy_state_dict" in weights.keys():
        model_trunk = weights["classy_state_dict"]["base_model"]["model"]["trunk"]
    elif "model_state_dict" in weights.keys():
        model_trunk = weights["model_state_dict"]
    else:
        model_trunk = weights

    # convert the trunk
    converted_weights = replace_module_prefix(state_dict=model_trunk, prefix=replace_prefix, ignore_prefix=ignore_prefix)
    return converted_weights


def get_backbone(backbone: str, load_weights: str) -> Union[torchvision.models.shufflenet_v2_x1_0, torchvision.models.resnet18]:
    pretrained = False
    if load_weights == "imagenet":
        pretrained = True

    if backbone == "resnet18":
        model = torchvision.models.resnet18(pretrained=pretrained)
        replace_prefix = "_feature_blocks."
        ignore_prefix = ""
        num_features = model.fc.in_features
        del model.fc
    elif backbone == "shufflenet_v2_x1_0":
        model = torchvision.models.shufflenet_v2_x1_0(pretrained=pretrained)
        ignore_prefix = "model."
        replace_prefix = "_feature_blocks."
        num_features = model.fc.in_features
        del model.fc
    else:
        raise ValueError

    if not pretrained and Path(load_weights).is_file():
        weights = torch.load(load_weights,  map_location=torch.device("cpu"))
        converted_weights = convert_model(weights, replace_prefix=replace_prefix, ignore_prefix=ignore_prefix)
        model.load_state_dict(converted_weights)

    model.fc = torch.nn.Identity()

    return {"model": model, "num_features": num_features}
