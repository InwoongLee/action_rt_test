import torch
import torch.nn as nn



__all__ = ['r3d_18', 'mc3_18', 'r2plus1d_18']

model_urls = {
    'r3d_18': 'https://download.pytorch.org/models/r3d_18-b3b3357e.pth',
    'mc3_18': 'https://download.pytorch.org/models/mc3_18-a90a0ba3.pth',
    'r2plus1d_18': 'https://download.pytorch.org/models/r2plus1d_18-91a641e6.pth',
}


class Conv3DSimple(nn.Conv3d):
    def __init__(self,
                 in_planes,
                 out_planes,
                 midplanes=None,
                 stride=1,
                 padding=1):

        super(Conv3DSimple, self).__init__(
            in_channels=in_planes,
            out_channels=out_planes,
            kernel_size=(3, 3, 3),
            stride=stride,
            padding=padding,
            bias=False)

    @staticmethod
    def get_downsample_stride(stride):
        return (stride, stride, stride)


class Conv2Plus1D(nn.Sequential):

    def __init__(self,
                 in_planes,
                 out_planes,
                 midplanes,
                 stride=1,
                 padding=1):
        super(Conv2Plus1D, self).__init__(
            nn.Conv3d(in_planes, midplanes, kernel_size=(1, 3, 3),
                      stride=(1, stride, stride), padding=(0, padding, padding),
                      bias=False),
            nn.BatchNorm3d(midplanes),
            nn.ReLU(inplace=True),
            nn.Conv3d(midplanes, out_planes, kernel_size=(3, 1, 1),
                      stride=(stride, 1, 1), padding=(padding, 0, 0),
                      bias=False))

    @staticmethod
    def get_downsample_stride(stride):
        return (stride, stride, stride)


class Conv3DNoTemporal(nn.Conv3d):

    def __init__(self,
                 in_planes,
                 out_planes,
                 midplanes=None,
                 stride=1,
                 padding=1):

        super(Conv3DNoTemporal, self).__init__(
            in_channels=in_planes,
            out_channels=out_planes,
            kernel_size=(1, 3, 3),
            stride=(1, stride, stride),
            padding=(0, padding, padding),
            bias=False)

    @staticmethod
    def get_downsample_stride(stride):
        return (1, stride, stride)

# class BasicBlock(nn.Module):
#
#     __constants__ = ['downsample']
#     expansion = 1
#
#     def __init__(self, inplanes, planes, conv_builder, stride=1, downsample=None):
#         midplanes = (inplanes * planes * 3 * 3 * 3) // (inplanes * 3 * 3 + 3 * planes)
#
#         super(BasicBlock, self).__init__()
#         self.downsample = downsample
#         self.conv1 = nn.Sequential(
#             nn.Conv3d(inplanes, planes, 3, stride, padding=1),
#             nn.BatchNorm3d(planes),
#             nn.ReLU(inplace=True)
#         )
#         self.conv2 = nn.Sequential(
#             nn.Conv3d(planes, planes, 3, padding=1),
#             nn.BatchNorm3d(planes)
#         )
#         self.relu = nn.ReLU(inplace=True)
#         # self.downsample = downsample
#         self.stride = stride
#
#     def forward(self, x):
#         residual = x
#
#         out = self.conv1(x)
#         out = self.conv2(out)
#
#         if self.downsample is not None:
#             residual = self.downsample(x)
#
#         out += residual
#         out = self.relu(out)
#
#         return out


class BasicBlock(nn.Module):

    __constants__ = ['downsample']
    expansion = 1

    def __init__(self, inplanes, planes, conv_builder, stride=1, downsample=None):
        midplanes = (inplanes * planes * 3 * 3 * 3) // (inplanes * 3 * 3 + 3 * planes)

        super(BasicBlock, self).__init__()
        self.downsample = downsample
        self.conv1 = nn.Sequential(
            conv_builder(inplanes, planes, midplanes, stride),
            nn.BatchNorm3d(planes),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            conv_builder(planes, planes, midplanes),
            nn.BatchNorm3d(planes)
        )
        self.relu = nn.ReLU(inplace=True)
        # self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        # residual = x
        # print("1", x.shape)
        out = self.conv1(x)
        # print("2", out.shape)
        out = self.conv2(out)
        # print("3", out.shape)
        if self.downsample is not None:
            # residual = self.downsample(x)
            x = self.downsample(x)
            # print("4", x.shape)

        # out += residual
        # print("4.5", out.shape)
        out = x + out
        # print("5", out.shape)
        out = self.relu(out)
        # print("6", out.shape)


        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, conv_builder, stride=1, downsample=None):

        super(Bottleneck, self).__init__()
        midplanes = (inplanes * planes * 3 * 3 * 3) // (inplanes * 3 * 3 + 3 * planes)

        # 1x1x1
        self.conv1 = nn.Sequential(
            nn.Conv3d(inplanes, planes, kernel_size=1, bias=False),
            nn.BatchNorm3d(planes),
            nn.ReLU(inplace=True)
        )
        # Second kernel
        self.conv2 = nn.Sequential(
            conv_builder(planes, planes, midplanes, stride),
            nn.BatchNorm3d(planes),
            nn.ReLU(inplace=True)
        )

        # 1x1x1
        self.conv3 = nn.Sequential(
            nn.Conv3d(planes, planes * self.expansion, kernel_size=1, bias=False),
            nn.BatchNorm3d(planes * self.expansion)
        )
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class BasicStem(nn.Sequential):
    """The default conv-batchnorm-relu stem
    """
    def __init__(self):
        super(BasicStem, self).__init__(
            nn.Conv3d(3, 64, kernel_size=(3, 7, 7), stride=(1, 2, 2),
                      padding=(1, 3, 3), bias=False),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True))


class R2Plus1dStem(nn.Sequential):
    """R(2+1)D stem is different than the default one as it uses separated 3D convolution
    """
    def __init__(self):
        super(R2Plus1dStem, self).__init__(
            nn.Conv3d(3, 45, kernel_size=(1, 7, 7),
                      stride=(1, 2, 2), padding=(0, 3, 3),
                      bias=False),
            nn.BatchNorm3d(45),
            nn.ReLU(inplace=True),
            nn.Conv3d(45, 64, kernel_size=(3, 1, 1),
                      stride=(1, 1, 1), padding=(1, 0, 0),
                      bias=False),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True))

class CNNClassifier(nn.Module):
    
    def __init__(self):
        # 항상 torch.nn.Module을 상속받고 시작
        super(CNNClassifier, self).__init__()
        self.conv1 = nn.Conv3d(3, 64, 1, (1,2,2)) # 6@24*24
        self.conv1_1 = nn.Conv3d(64, 64, 1, 1) # 6@24*24
        self.conv1_2 = nn.Conv3d(64, 64, 1, 1) # 6@24*24
        self.conv1_3 = nn.Conv3d(64, 64, 1, 1) # 6@24*24
        self.conv1_4 = nn.Conv3d(64, 64, 1, 1) # 6@24*24
        self.conv1_5 = nn.Conv3d(64, 64, 1, 1) # 6@24*24
        self.conv1_6 = nn.Conv3d(64, 64, 1, 1) # 6@24*24
        self.conv1_7 = nn.Conv3d(64, 64, 1, 1) # 6@24*24
        self.conv1_8 = nn.Conv3d(64, 64, 1, 1) # 6@24*24
        self.conv1_9 = nn.Conv3d(64, 64, 1, 1) # 6@24*24
        self.conv1_10 = nn.Conv3d(64, 64, 1, 1) # 6@24*24
        self.conv1_11 = nn.Conv3d(64, 64, 1, 1) # 6@24*24
        self.conv1_12 = nn.Conv3d(64, 64, 1, 1) # 6@24*24
        self.conv1_13 = nn.Conv3d(64, 64, 1, 1) # 6@24*24
        self.conv1_14 = nn.Conv3d(64, 64, 1, 1) # 6@24*24
        self.bn1 = nn.BatchNorm3d(64)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(64, 128, 1, 2) # 6@24*24
        self.bn2 = nn.BatchNorm3d(128)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv3d(128, 256, 1, 2) # 6@24*24
        self.bn3 = nn.BatchNorm3d(256)
        self.relu3 = nn.ReLU(inplace=True)
        self.conv4 = nn.Conv3d(256, 512, 1, 2) # 6@24*24
        self.bn4 = nn.BatchNorm3d(512)
        self.relu4 = nn.ReLU(inplace=True)
        self.avgpool = nn.AdaptiveAvgPool3d((1,1,1))
        
    def forward(self, x):
        x = self.conv1(x) # @16*4*4
        x = self.conv1_1(x) # @16*4*4
        x = self.conv1_2(x) # @16*4*4
        x = self.conv1_3(x) # @16*4*4
        x = self.conv1_4(x) # @16*4*4
        x = self.conv1_5(x) # @16*4*4
        x = self.conv1_6(x) # @16*4*4
        x = self.conv1_7(x) # @16*4*4
        x = self.conv1_8(x) # @16*4*4
        x = self.conv1_9(x) # @16*4*4
        x = self.conv1_10(x) # @16*4*4
        x = self.conv1_11(x) # @16*4*4
        x = self.conv1_12(x) # @16*4*4
        x = self.conv1_13(x) # @16*4*4
        x = self.conv1_14(x) # @16*4*4
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.conv2(x) # @16*4*4
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.conv3(x) # @16*4*4
        x = self.bn3(x)
        x = self.relu3(x)
        x = self.conv4(x) # @16*4*4
        x = self.bn4(x)
        x = self.relu4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        return x

class VideoResNet(nn.Module):

    def __init__(self, block, conv_makers, layers,
                 stem, num_classes=400,
                 zero_init_residual=False):
        """Generic resnet video generator.

        Args:
            block (nn.Module): resnet building block
            conv_makers (list(functions)): generator function for each layer
            layers (List[int]): number of blocks per layer
            stem (nn.Module, optional): Resnet stem, if None, defaults to conv-bn-relu. Defaults to None.
            num_classes (int, optional): Dimension of the final FC layer. Defaults to 400.
            zero_init_residual (bool, optional): Zero init bottleneck residual BN. Defaults to False.
        """
        super(VideoResNet, self).__init__()
        self.inplanes = 64

        self.stem = stem()

        self.layer1 = self._make_layer(block, conv_makers[0], 64, layers[0], stride=1)
        self.layer2 = self._make_layer(block, conv_makers[1], 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, conv_makers[2], 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, conv_makers[3], 512, layers[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        # init weights
        self._initialize_weights()

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)

    def forward(self, x):
        x = self.stem(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        # Flatten the layer to fc
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    def _make_layer(self, block, conv_builder, planes, blocks, stride=1):
        downsample = None

        if stride != 1 or self.inplanes != planes * block.expansion:
            ds_stride = conv_builder.get_downsample_stride(stride)
            downsample = nn.Sequential(
                nn.Conv3d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=ds_stride, bias=False),
                nn.BatchNorm3d(planes * block.expansion)
            )
        layers = []
        layers.append(block(self.inplanes, planes, conv_builder, stride, downsample))

        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, conv_builder))

        return nn.Sequential(*layers)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                                        nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


def _video_resnet(arch, pretrained=False, progress=True, **kwargs):
    model = VideoResNet(**kwargs)

    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch],
                                              progress=progress)
        model.load_state_dict(state_dict)
    return model


def r3d_18(pretrained=False, progress=True, **kwargs):
    """Construct 18 layer Resnet3D model as in
    https://arxiv.org/abs/1711.11248

    Args:
        pretrained (bool): If True, returns a model pre-trained on Kinetics-400
        progress (bool): If True, displays a progress bar of the download to stderr

    Returns:
        nn.Module: R3D-18 network
    """

    return _video_resnet('r3d_18',
                         pretrained, progress,
                         block=BasicBlock,
                         conv_makers=[Conv3DSimple] * 4,
                         layers=[2, 2, 2, 2],
                         stem=BasicStem, **kwargs)


def mc3_18(pretrained=False, progress=True, **kwargs):
    """Constructor for 18 layer Mixed Convolution network as in
    https://arxiv.org/abs/1711.11248

    Args:
        pretrained (bool): If True, returns a model pre-trained on Kinetics-400
        progress (bool): If True, displays a progress bar of the download to stderr

    Returns:
        nn.Module: MC3 Network definition
    """
    return _video_resnet('mc3_18',
                         pretrained, progress,
                         block=BasicBlock,
                         conv_makers=[Conv3DSimple] + [Conv3DNoTemporal] * 3,
                         layers=[2, 2, 2, 2],
                         stem=BasicStem, **kwargs)


def r2plus1d_18(pretrained=False, progress=True, **kwargs):
    """Constructor for the 18 layer deep R(2+1)D network as in
    https://arxiv.org/abs/1711.11248

    Args:
        pretrained (bool): If True, returns a model pre-trained on Kinetics-400
        progress (bool): If True, displays a progress bar of the download to stderr

    Returns:
        nn.Module: R(2+1)D-18 network
    """
    return _video_resnet('r2plus1d_18',
                         pretrained, progress,
                         block=BasicBlock,
                         conv_makers=[Conv2Plus1D] * 4,
                         layers=[2, 2, 2, 2],
                         stem=R2Plus1dStem, **kwargs)
