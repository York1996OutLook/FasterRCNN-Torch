import torch,torch.nn as nn

class RCNN(nn.Module):
    def __init__(self):
        super(RCNN, self).__init__()
        self.roi_pooling=nn.AdaptiveAvgPool2d((7,7))
        self.cls_brach=nn.Sequential(
            nn.Linear(49,81),
            nn.Sigmoid(),
        )
        self.reg_brach=nn.Sequential(
            nn.Linear(49,4),
            nn.Sigmoid(),
        )
    def forward(self, rois):
        x=self.roi_pooling(rois)
        cls=self.cls_brach(x)
        reg=self.reg_brach(x)
        return cls,reg




