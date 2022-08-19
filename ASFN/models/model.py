import torch
import torch.nn.functional as F
import torch.nn as nn
from .BaseNetwork import BaseNetwork
from .utils.reconstructionLayers import make_layer, ResidualBlock_noBN
import functools


class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        self.net = ASFN(config['cnum'], config['in_channel'], config['PASSMASK'], config['transferBlocks'],
                                     config['resBlocks'],
                                     config['use_bias'], config['conv_type'],
                                     config['init_weights'], config['skip_connection'], config['dataMin'])

    def forward(self, image, mask):
        ret = self.net(image, mask)
        return ret


class ASFN(BaseNetwork):
    def __init__(self, num_feat, in_channels, passmask, transBlocks, numBlocks, use_bias, conv_type, init_weights,
                 skip_connection, dataMin):
        super(ASFN, self).__init__(conv_type)
        self.passmask = passmask

        self.encoder12 = nn.Sequential(
            nn.ReflectionPad2d(2),
            self.ConvBlock(in_channels, num_feat, kernel_size=5, stride=1, padding=0, bias=use_bias, norm=None),
            self.ConvBlock(num_feat, num_feat * 2, kernel_size=3, stride=2, padding=1, bias=use_bias, norm=None)
        )
        self.encoder24 = nn.Sequential(
            self.ConvBlock(num_feat * 2, num_feat * 2, kernel_size=3, stride=1, padding=1, bias=use_bias, norm=None),
            self.ConvBlock(num_feat * 2, num_feat * 4, kernel_size=3, stride=2, padding=1, bias=use_bias, norm=None)
        )

        transferBlock = functools.partial(FeatureTransferBlock, num_feat * 4, conv_type, use_bias)
        self.transferLayer = make_layer(transferBlock, transBlocks)

        residualBlock = functools.partial(ResidualBlock_noBN, nf=num_feat * 4)
        self.refineLayer = make_layer(residualBlock, numBlocks)
        self.skip_connection = skip_connection

        if skip_connection:
            self.decoder42 = nn.Sequential(
                self.DeconvBlock(num_feat * 8, num_feat * 2, kernel_size=3, stride=1, padding=1, bias=use_bias,
                                 norm=None),
                self.ConvBlock(num_feat * 2, num_feat * 2, kernel_size=3, stride=1, padding=1, bias=use_bias, norm=None)
            )
            self.decoder21 = nn.Sequential(
                self.DeconvBlock(num_feat * 4, num_feat, kernel_size=3, stride=1, padding=1, bias=use_bias, norm=None),
                self.ConvBlock(num_feat, num_feat // 2, kernel_size=3, stride=1, padding=1, bias=use_bias, norm=None)
            )
        else:
            self.decoder42 = nn.Sequential(
                self.DeconvBlock(num_feat * 4, num_feat * 2, kernel_size=3, stride=1, padding=1, bias=use_bias,
                                 norm=None),
                self.ConvBlock(num_feat * 2, num_feat * 2, kernel_size=3, stride=1, padding=1, bias=use_bias, norm=None)
            )
            self.decoder21 = nn.Sequential(
                self.DeconvBlock(num_feat * 2, num_feat, kernel_size=3, stride=1, padding=1, bias=use_bias, norm=None),
                self.ConvBlock(num_feat, num_feat // 2, kernel_size=3, stride=1, padding=1, bias=use_bias, norm=None)
            )
        self.final = self.ConvBlock(num_feat // 2, 3, kernel_size=3, stride=1, padding=1, bias=use_bias, norm=None,
                                    activation=None)
        self.dataMin = dataMin

        if init_weights:
            self.init_weights()

    def forward(self, image, mask):
        if self.passmask:
            inputs = torch.cat((image, mask), dim=1)
        else:
            inputs = image
        h, w = mask.shape[2:]
        smallMask = F.interpolate(mask, size=(h // 4, w // 4), mode='nearest')

        a2 = self.encoder12(inputs)
        a4 = self.encoder24(a2)

        fm_dict = {'f': a4, 'm': smallMask}
        transferred = self.transferLayer(fm_dict)
        transferred = transferred['f']
        transferred = a4 + transferred
        decoder4_input = self.refineLayer(transferred)
        if self.skip_connection:
            decoder4_input = torch.cat((decoder4_input, a4), dim=1)
        decoder2_input = self.decoder42(decoder4_input)
        if self.skip_connection:
            decoder2_input = torch.cat((decoder2_input, a2), dim=1)
        output = self.decoder21(decoder2_input)
        output = self.final(output)
        if self.dataMin == -1:
            output = torch.tanh(output)
        else:
            output = torch.sigmoid(output)
        return output


class FeatureTransferBlock(BaseNetwork):
    def __init__(self, in_channels, conv_type, use_bias):
        super(FeatureTransferBlock, self).__init__(conv_type)
        self.t1 = FeatTransferLayer(in_channels, conv_type, use_bias)
        self.t2 = FeatTransferLayer(in_channels, conv_type, use_bias)
        self.f1 = self.ConvBlock(in_channels, in_channels, kernel_size=3, stride=1, padding=1, bias=use_bias, norm=None)
        self.f2 = self.ConvBlock(in_channels, in_channels, kernel_size=3, stride=1, padding=1, bias=use_bias, norm=None)

    def forward(self, fm_dict):
        feature, mask = fm_dict['f'], fm_dict['m']
        a1 = self.t1(feature, mask)
        a2 = self.f1(a1)
        a3 = self.t2(a2, mask)
        a4 = self.f2(a3)
        a4 = feature + a4
        return {'f': a4, 'm': mask}


class FeatTransferLayer(BaseNetwork):
    def __init__(self, in_channels, conv_type, use_bias):
        super(FeatTransferLayer, self).__init__(conv_type)
        self.gammaGen = self.ConvBlock(2 * in_channels, in_channels, kernel_size=1, stride=1, padding=0, bias=use_bias,
                                       norm=None)
        self.betaGen = self.ConvBlock(2 * in_channels, in_channels, kernel_size=1, stride=1, padding=0, bias=use_bias,
                                      norm=None)

    def forward(self, feature, mask):
        inverseMask = (1 - mask).bool()
        current_mask = mask.bool()
        feat_mean, feat_std = self.calc_mean_std_mask(feature, inverseMask)  # [b, c, 1, 1]
        mask_mean, mask_std = self.calc_mean_std_mask(feature, current_mask)
        feat_std = torch.cat((feat_std, mask_std), dim=1)
        feat_mean = torch.cat((feat_mean, mask_mean), dim=1)
        gamma = self.gammaGen(feat_std)
        beta = self.betaGen(feat_mean)
        invalid_normed = (feature - mask_mean) / mask_std
        invalid_transfered = invalid_normed * gamma + beta
        outFeat = feature * (1 - mask) + invalid_transfered * mask
        return outFeat

    def calc_mean_std_mask(self, feature, mask):
        eps = 1e-5
        B, C = feature.shape[:2]
        feat_means, feat_stds = [], []
        for b in range(B):
            feat_var = feature[b, :, mask[b, 0]].view(C, -1).var(dim=1) + eps
            feat_std = feat_var.sqrt().view(C, 1, 1)
            feat_mean = feature[b, :, mask[b, 0]].view(C, -1).mean(dim=1).view(C, 1, 1)
            feat_means.append(feat_mean)
            feat_stds.append(feat_std)
        feat_means = torch.stack(feat_means, dim=0)
        feat_stds = torch.stack(feat_stds, dim=0)
        return feat_means, feat_stds


class Discriminator(BaseNetwork):
    def __init__(self, conv_type, in_channel=3, nf=64, use_sigmoid=False, use_spectral_norm=True, init_weights=True):
        super(Discriminator, self).__init__(conv_type)
        self.use_sigmoid = use_sigmoid

        self.conv = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=in_channel, out_channels=nf, kernel_size=5, stride=2, padding=2,
                                    bias=not use_spectral_norm), use_spectral_norm),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv2d(in_channels=nf, out_channels=nf * 2, kernel_size=5, stride=2, padding=2,
                                    bias=not use_spectral_norm), use_spectral_norm),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv2d(in_channels=nf * 2, out_channels=nf * 4, kernel_size=5, stride=2, padding=2,
                                    bias=not use_spectral_norm), use_spectral_norm),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv2d(in_channels=nf * 4, out_channels=nf * 4, kernel_size=5, stride=2, padding=2,
                                    bias=not use_spectral_norm), use_spectral_norm),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv2d(in_channels=nf * 4, out_channels=nf * 4, kernel_size=5, stride=2, padding=2,
                                    bias=not use_spectral_norm), use_spectral_norm),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv2d(in_channels=nf * 4, out_channels=nf * 4, kernel_size=5, stride=2, padding=2,
                                    bias=not use_spectral_norm), use_spectral_norm)
        )

        if init_weights:
            self.init_weights()

    def forward(self, x):
        feat = self.conv(x)
        if self.use_sigmoid:
            feat = torch.sigmoid(feat)
        return feat


def spectral_norm(module, mode=True):
    if mode:
        return nn.utils.spectral_norm(module)
    return module


def input_generator(input_shape):
    c, h, w = input_shape
    frame_shape = (1, c, h, w)
    mask_shape = (1, 1, h, w)
    frame = torch.randn(frame_shape)
    mask = torch.randn(mask_shape)
    return {'image': frame, 'mask': mask}
