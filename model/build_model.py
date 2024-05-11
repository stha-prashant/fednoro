import torch.nn as nn
from .all_models import get_model, modify_last_layer


def build_model(args):
    # choose different Neural network model for different args
    model = get_model(args.model, args)
    model = model.to(args.device)

    return model

