import torch
import torch.nn as nn
import torch.nn.functional as F
from .encoder import build_backbone
from .decoder import build_decoder
from data_utils import get_transform, make_one_hot
from .pointrend import PointHead

class UGSNet(nn.Module):
    def __init__(self, backbone='resnet50', output_stride=32, f_c=64, freeze_bn=False, in_c=3):
        super(UGSNet, self).__init__()
        BatchNorm = nn.BatchNorm2d

        self.transform = get_transform(convert=True, normalize=True)

        self.backbone = build_backbone(backbone, output_stride, BatchNorm, in_c)
        self.decoder = build_decoder(f_c, BatchNorm)
        self.pointhead = PointHead(514)


        if freeze_bn:
            self.freeze_bn()

    def forward(self, hr_img1):
        x_1, f2_1, f3_1, f4_1 = self.backbone(hr_img1)
        result = self.decoder(x_1, f2_1, f3_1, f4_1)
        result.update(self.pointhead(hr_img1, result["res2"], result["coarse"]))
        return result

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()