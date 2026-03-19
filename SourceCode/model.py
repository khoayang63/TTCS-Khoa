import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    def __init__(self, in_c, out_c, kernel_size=1, stride=1, bn_act=True):
        super().__init__()
        padding = kernel_size // 2
        self.bn_act = bn_act

        self.conv = nn.Conv2d(
            in_c,
            out_c,
            kernel_size,
            stride,
            padding,
            bias=not bn_act
        )

        if bn_act:
            self.bn = nn.BatchNorm2d(out_c)
            self.act = nn.LeakyReLU(0.1)
        
    def forward(self, x):
        if self.bn_act: return self.act(self.bn(self.conv(x)))
        else: return self.conv(x)


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.layer1 = ConvBlock(channels, channels//2, kernel_size=1, stride=1)
        self.layer2 = ConvBlock(channels//2, channels, kernel_size=3, stride=1)

    def forward(self, x):
        return x + self.layer2(self.layer1(x))

class Darknet53(nn.Module):
    def __init__(self):
        super().__init__()

        self.layer1 = ConvBlock(3, 32, 3, 1)
        self.layer2 = ConvBlock(32, 64, 3, 2)
        self.res1 = self._make_layer(64, 1)

        self.layer3 = ConvBlock(64, 128, 3, 2)
        self.res2 = self._make_layer(128, 2)

        self.layer4 = ConvBlock(128, 256, 3, 2)
        self.res3 = self._make_layer(256, 8)

        self.layer5 = ConvBlock(256, 512, 3, 2)
        self.res4 = self._make_layer(512, 8)

        self.layer6 = ConvBlock(512, 1024, 3, 2)
        self.res5 = self._make_layer(1024, 4)

    def _make_layer(self, channels, num_blocks):
        layers = []
        for _ in range(num_blocks):
            layers.append(ResidualBlock(channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.res1(x)

        x = self.layer3(x)
        x = self.res2(x)

        x = self.layer4(x)
        feat_small = self.res3(x)   # 52x52

        x = self.layer5(feat_small)
        feat_medium = self.res4(x)  # 26x26

        x = self.layer6(feat_medium)
        feat_large = self.res5(x)   # 13x13

        return feat_small, feat_medium, feat_large


# class DetectionBlock(nn.Module):
#     def __init__(self, in_channels, num_classes):
#         super().__init__()

#         self.layers = nn.Sequential(
#             ConvBlock(in_channels, in_channels//2, 1),
#             ConvBlock(in_channels//2, in_channels, 3),

#             ConvBlock(in_channels, in_channels//2, 1),
#             ConvBlock(in_channels//2, in_channels, 3),

#             ConvBlock(in_channels, in_channels//2, 1),
#         )

#         self.pred_conv = nn.Sequential(
#             ConvBlock(in_channels//2, in_channels, 3),
#             ConvBlock(
#                 in_channels,
#                 3*(num_classes+5),
#                 kernel_size=1,
#                 bn_act=False
#             )
#         )

#         self.num_classes = num_classes

#     def forward(self, x):
#         x = self.layers(x)

#         route = x   # giữ feature map cho FPN

#         pred = self.pred_conv(x)

#         B, _, H, W = pred.shape
#         pred = pred.reshape(B, 3, self.num_classes+5, H, W)
#         pred = pred.permute(0,1,3,4,2)

#         return route, pred

class DetectionBlock(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_classes):
        super().__init__()

        self.layers = nn.Sequential(
            ConvBlock(in_channels, hidden_channels, 1),
            ConvBlock(hidden_channels, hidden_channels * 2, 3),
            ConvBlock(hidden_channels*2, hidden_channels, 1),
            ConvBlock(hidden_channels, hidden_channels * 2, 3),
            ConvBlock(hidden_channels*2, hidden_channels, 1),
        )

        self.pred_conv = nn.Sequential(
            ConvBlock(hidden_channels, hidden_channels * 2, 3),
            ConvBlock(
                hidden_channels * 2,
                3 * (num_classes + 5),
                kernel_size=1,
                bn_act=False
            )
        )

        self.num_classes = num_classes

    def forward(self, x):
        x = self.layers(x)
        route = x   # giữ feature map cho FPN
        pred = self.pred_conv(x)

        B, _, H, W = pred.shape
        pred = pred.reshape(B, 3, self.num_classes + 5, H, W)
        pred = pred.permute(0, 1, 3, 4, 2)

        return route, pred

class YOLOv3(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.backbone = Darknet53()

        # Scale 1 (13x13)
        self.head1 = DetectionBlock(in_channels=1024, hidden_channels=512, num_classes=num_classes)

        # Scale 2 (26x26)
        self.conv_for_P4 = ConvBlock(512, 256, 1)
        self.head2 = DetectionBlock(in_channels=768, hidden_channels=256, num_classes=num_classes)

        # Scale 3 (52x52)
        self.conv_for_P3 = ConvBlock(256, 128, 1) 
        self.head3 = DetectionBlock(in_channels=384, hidden_channels=128, num_classes=num_classes)

        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")

    def forward(self, x):
        route1, route2, route3 = self.backbone(x)

        # Scale 1
        route, out1 = self.head1(route3)

        # Scale 2
        x = self.conv_for_P4(route)
        x = self.upsample(x)
        x = torch.cat([x, route2], dim=1)
        route, out2 = self.head2(x)

        # Scale 3
        x = self.conv_for_P3(route)
        x = self.upsample(x)
        x = torch.cat([x, route1], dim=1)
        _, out3 = self.head3(x)

        return out1, out2, out3


def main():
    x = torch.rand(1, 3, 416, 416)

    model = YOLOv3(num_classes=20)
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of trainable parameters: {trainable_params}")
    out = model(x)
    print(out[0].shape, out[1].shape, out[2].shape)
    
if __name__ == '__main__':
    main()
