import torch.nn as nn
from pytorchvideo.models.hub import slow_r50
from pytorchvideo.models.hub import x3d_m

def build_i3d(num_classes=10, pretrained=True):
    # load pretrained Slow R50 backbone
    model = slow_r50(pretrained=pretrained)
    in_f = model.blocks[-1].proj.in_features
    # model.blocks[-1].proj = nn.Linear(in_f, num_classes)
    model.blocks[-1].proj = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(in_f, 256),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(256, num_classes)
    )
    

    # model = x3d_m(pretrained=pretrained)
    # in_f = model.blocks[-1].proj.in_features
    # # model.blocks[-1].proj = nn.Linear(in_f, num_classes)
    # model.blocks[-1].proj = nn.Sequential(
    #     nn.Dropout(0.5),
    #     nn.Linear(in_f, 512),
    #     nn.ReLU(),
    #     nn.Dropout(0.3),
    #     nn.Linear(512, num_classes)
    # )

    return model

