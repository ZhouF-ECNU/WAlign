import torch
from modeling.layers.deviation_loss import DeviationLoss
from modeling.layers.binary_focal_loss import BinaryFocalLoss

def build_criterion(criterion):
    if criterion == "deviation":
        return DeviationLoss()
    elif criterion == "BCE":
        return torch.nn.BCEWithLogitsLoss()
    elif criterion == "focal":
        return BinaryFocalLoss()
    else:
        raise NotImplementedError