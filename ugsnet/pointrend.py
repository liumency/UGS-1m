
import torch
import torch.nn as nn
import torch.nn.functional as F

from .sampling_points import sampling_points, point_sample

class PreNet(nn.Module):
    def __init__(self, in_c=7):
        super().__init__()
        self.preconv = nn.Conv2d(in_c, 3, kernel_size=3, padding=1, bias=True)
        self.upconv = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)

    def forward(self, x):
        x = self.preconv(x)
        out_pre = self.upconv(x)
        return out_pre

class PointHead(nn.Module):
    def __init__(self, in_c=523, num_classes=2, k=3, beta=0.75):
        super().__init__()
        self.mlp = nn.Conv1d(in_c, num_classes, 1)
        self.k = k
        self.beta = beta

    def forward(self, x, res2, out):

        
        if not self.training:
            return self.inference(x, res2, out)
        points = sampling_points(out, x.shape[-1] * 4, self.k, self.beta)

        coarse = point_sample(out, points)
        # print(coarse.size())
        fine = point_sample(res2, points)
        # print(fine.size())
        feature_representation = torch.cat([coarse, fine], dim=1)
        rend = self.mlp(feature_representation)
        # print(rend.size()) # 8, 2, 1024
        # print(points.size()) # 8, 1024, 2
        return {"rend": rend, "points": points}

    @torch.no_grad()
    def inference(self, x, res2, out):
        """
        During inference, subdivision uses N=8096
        (i.e., the number of points in the stride 16 map of a 1024×2048 image)
        """
        num_points = 1024
        # num_points = 2048
        # count=1
        while out.shape[-1] != x.shape[-1]:
            out = F.interpolate(out, scale_factor=2, mode="bilinear", align_corners=True)

            points_idx, points = sampling_points(out, num_points, training=self.training)

            coarse = point_sample(out, points) #
            fine = point_sample(res2, points) #

            feature_representation = torch.cat([coarse, fine], dim=1)

            rend = self.mlp(feature_representation)

            B, C, H, W = out.shape
            points_idx = points_idx.unsqueeze(1).expand(-1, C, -1)
            out = (out.reshape(B, C, -1)
                      .scatter_(2, points_idx, rend)
                      .view(B, C, H, W))
            # count=count*4
        return {"fine": out}


class PointRend(nn.Module):
    def __init__(self, backbone, head):
        super().__init__()
        # self.prenet = prenet
        self.backbone = backbone
        self.head = head
        # self.conv_R1 = nn.Conv2d(in_channels=2048, out_channels=2048, kernel_size=3, stride=1, padding=1, bias=False)

    def forward(self, x):
        # pre_out = self.prenet(x)
        result = self.backbone(x)
        result.update(self.head(x, result["res2"], result["coarse"]))

        return result


