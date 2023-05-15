import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from .coordAtten import CoordAtt

class DR(nn.Module):
    def __init__(self, in_d, out_d):
        super(DR, self).__init__()
        self.in_d = in_d
        self.out_d = out_d
        self.conv1 = nn.Conv2d(self.in_d, self.out_d, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.out_d)
        self.relu = nn.ReLU()

    def forward(self, input):
        x = self.conv1(input)
        x = self.bn1(x)
        x = self.relu(x)
        return x

class Decoder(nn.Module):
    def __init__(self, fc, BatchNorm):
        super(Decoder, self).__init__()
        self.fc = fc
        self.dr2 = DR(256, 96)
        self.dr3 = DR(512, 96)
        self.dr4 = DR(1024, 96)
        self.dr5 = DR(2048, 96)

        self.last_conv = nn.Sequential(nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                       BatchNorm(256),
                                       nn.ReLU(),
                                       nn.Dropout(0.5),
                                       nn.Conv2d(256, 2, kernel_size=1, stride=1, padding=0, bias=False),
                                       )

        self.upconv1 = nn.ConvTranspose2d(64, 64, kernel_size=3, padding=1, stride=2, output_padding=1)
        self.conv12d = nn.ConvTranspose2d(64, 32, kernel_size=3, padding=1)
        self.bn12d = nn.BatchNorm2d(32)
        self.do12d = nn.Dropout2d(p=0.2)

        self.upconv2 = nn.ConvTranspose2d(32, 32, kernel_size=3, padding=1, stride=2, output_padding=1)
        self.conv22d = nn.ConvTranspose2d(32, 16, kernel_size=3, padding=1)
        self.bn22d = nn.BatchNorm2d(16)
        self.do22d = nn.Dropout2d(p=0.2)

        self.conv11d = nn.ConvTranspose2d(16, 2, kernel_size=3, padding=1)

        self.OSMBlock = nn.Sequential(
            nn.Conv2d(64, 2 * 2 * 64, 3, padding=1),
            nn.PixelShuffle(2),
        )

        self.upconv3 = nn.ConvTranspose2d(64, 64, kernel_size=3, padding=1, stride=2, output_padding=1)
        self.conv31d = nn.ConvTranspose2d(64, 32, kernel_size=3, padding=1)
        self.bn31d = nn.BatchNorm2d(32)
        self.do31d = nn.Dropout2d(p=0.2)

        self.conv33d = nn.ConvTranspose2d(32, 2, kernel_size=3, padding=1)

        self._init_weight()

    def forward(self, x, low_level_feat2, low_level_feat3, low_level_feat4):

        x2 = self.dr2(low_level_feat2)
        x3 = self.dr3(low_level_feat3)
        x4 = self.dr4(low_level_feat4)
        x = self.dr5(x)

        x = F.interpolate(x, size=x3.size()[2:], mode='bilinear', align_corners=True)
        x2 = F.interpolate(x2, size=x3.size()[2:], mode='bilinear', align_corners=True)
        x4 = F.interpolate(x4, size=x3.size()[2:], mode='bilinear', align_corners=True)

        x = torch.cat((x, x2, x3, x4), dim=1)
        x = self.last_conv(x)

        return {"coarse":x, "res2":low_level_feat3}

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

def build_decoder(fc, BatchNorm):
    return Decoder(fc, BatchNorm)