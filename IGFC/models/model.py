import torch
import torch.nn.functional as F
import torch.nn as nn
from .BaseNetwork import BaseNetwork
from .utils.reconstructionLayers import make_layer, ResidualBlock_noBN
import functools
from Forward_Warp import forward_warp


class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        self.net = IGFC(config['cnum'], config['in_channel'], config['PASSMASK'],
                                config['use_residual'], config['resBlocks'], config['use_bias'], config['conv_type'],
                                config['init_weights'], config['num_frames'])

    def forward(self, flows, masks, edges=None, relative_indices=None):
        ret = self.net(flows, masks, edges, relative_indices)
        return ret


class IGFC(BaseNetwork):
    def __init__(self, num_feat, in_channels, passmask, use_residual, numBlocks, use_bias, conv_type,
                 init_weights, num_frames):
        super(IGFC, self).__init__(conv_type)
        self.use_residual = use_residual
        self.passmask = passmask
        self.conv1 = nn.Sequential(
            nn.ReflectionPad2d(2),
            self.ConvBlock(in_channels, num_feat, kernel_size=5, stride=1, padding=0, bias=use_bias, norm=None)
        )
        self.conv2 = self.ConvBlock(num_feat, num_feat * 2, kernel_size=3, stride=2, padding=1, bias=use_bias,
                                    norm=None)
        self.conv3 = self.ConvBlock(num_feat * 2, num_feat * 2, kernel_size=3, stride=1, padding=1, bias=use_bias,
                                    norm=None)
        self.conv4 = self.ConvBlock(num_feat * 2, num_feat * 4, kernel_size=3, stride=2, padding=1, bias=use_bias,
                                    norm=None)

        residualBlock = functools.partial(ResidualBlock_noBN, nf=num_feat * 4)
        self.middleLayer = make_layer(residualBlock, numBlocks)
        self.fw = forward_warp()

        self.flowFuseLayer = FlowFusion(num_feat * 4, conv_type, use_bias, num_frames, norm=None)

        self.dilate_conv1 = self.ConvBlock(num_feat * 4, num_feat * 4, kernel_size=3, stride=1, padding=8,
                                           bias=use_bias, dilation=8, norm=None)
        self.dilate_conv2 = self.ConvBlock(num_feat * 4, num_feat * 4, kernel_size=3, stride=1, padding=4,
                                           bias=use_bias, dilation=4, norm=None)
        self.dilate_conv3 = self.ConvBlock(num_feat * 4, num_feat * 4, kernel_size=3, stride=1, padding=2,
                                           bias=use_bias, dilation=2, norm=None)
        self.dilate_conv4 = self.ConvBlock(num_feat * 4, num_feat * 4, kernel_size=3, stride=1, padding=1,
                                           bias=use_bias, dilation=1, norm=None)

        self.recon4 = self.ConvBlock(num_feat * 4, 2, kernel_size=1, stride=1, padding=0)

        self.deconv1 = self.DeconvBlock(num_feat * 8, num_feat * 2, kernel_size=3, stride=1, padding=1, bias=use_bias,
                                        norm=None)
        self.conv5 = self.ConvBlock(num_feat * 2, num_feat * 2, kernel_size=3, stride=1, padding=1, bias=use_bias,
                                    norm=None)
        self.conv6 = self.ConvBlock(num_feat * 2, num_feat * 2, kernel_size=3, stride=1, padding=1, bias=use_bias,
                                    norm=None)
        self.recon2 = self.ConvBlock(num_feat * 2, 2, kernel_size=1, stride=1, padding=0)

        self.deconv2 = self.DeconvBlock(num_feat * 4, num_feat, kernel_size=3, stride=1, padding=1, bias=use_bias,
                                        norm=None)
        self.conv7 = self.ConvBlock(num_feat, num_feat // 2, kernel_size=3, stride=1, padding=1, bias=use_bias,
                                    norm=None)
        self.final = self.ConvBlock(num_feat // 2, 2, kernel_size=3, stride=1, padding=1, bias=use_bias, norm=None,
                                    activation=None)

        if init_weights:
            self.init_weights()

    def forward(self, flows, masks, edges=None, relative_indices=None):
        b, c, t, h, w = flows.shape
        if self.passmask:
            inputs = torch.cat((flows, masks), dim=1)
        else:
            inputs = flows
        if edges is not None:
            inputs = torch.cat((inputs, edges), dim=1)
        individuals = torch.chunk(inputs, t, dim=2)
        pivot = t // 2
        
        reference_feats = []
        for i in range(t):
            if relative_indices is not None:
                assert len(relative_indices) == t
                index = relative_indices[i]
                shift = index
            else:
                if pivot >= i:
                    direction = 'forward'
                else:
                    direction = 'backward'
                shift = pivot - i
            if i != pivot:
                baseFeat = individuals[i].squeeze(2)
                baseFeat = self.conv1(baseFeat)
                baseFeat = self.conv2(baseFeat)
                baseFeat = self.conv3(baseFeat)
                baseFeat = self.conv4(baseFeat)
                reference_feat = self.middleLayer(baseFeat)
                smallFlow = F.interpolate(flows[:, :, i], size=(h // 4, w // 4), mode='bilinear')
                smallFlow = smallFlow / 4 * shift
                smallFlow = smallFlow.permute(0, 2, 3, 1).contiguous()
                if not reference_feat.is_contiguous():
                    reference_feat = reference_feat.contiguous()
                reference_feat = self.fw(reference_feat, smallFlow)
                reference_feats.append(reference_feat)
            else:
                baseFeat = individuals[i].squeeze(2)
                c1 = self.conv1(baseFeat)
                c2 = self.conv2(c1)
                c3 = self.conv3(c2)
                c4 = self.conv4(c3)
                target_feat = self.middleLayer(c4)
                reference_feats.append(target_feat)

        smallMask = F.interpolate(masks[:, :, pivot], size=(h // 4, w // 4), mode='nearest')

        fused_flow_feat = self.flowFuseLayer(target_feat, reference_feats, smallMask)
        a1 = self.dilate_conv1(fused_flow_feat)
        a2 = self.dilate_conv2(a1)
        a3 = self.dilate_conv3(a2)
        a4 = self.dilate_conv4(a3)
        f4 = self.recon4(a4)
        d1 = torch.cat((a4, c4), dim=1)
        d2 = self.deconv1(d1)
        d3 = self.conv5(d2)
        d4 = self.conv6(d3)
        f2 = self.recon2(d4)
        d5 = torch.cat((d4, c2), dim=1)
        d6 = self.deconv2(d5)
        d7 = self.conv7(d6)
        flow = self.final(d7)
        if self.use_residual:
            flow = flow + flows[:, :, pivot]
        return flow, f4, f2


class FlowFusion(BaseNetwork):
    def __init__(self, in_channels, conv_type, use_bias, num_frames, norm):
        super(FlowFusion, self).__init__(conv_type)
        self.spa = SpatialAttentionGen(in_channels, conv_type, use_bias, norm)
        self.temporal_selection = self.ConvBlock(num_frames, num_frames, kernel_size=1, stride=1, padding=0)

    def forward(self, target_feat, reference_feats, smallMask):
        t = len(reference_feats)
        attention_maps = []
        for i in range(t):
            attention_map = self.spa(target_feat, reference_feats[i])
            attention_maps.append(attention_map)
        attention_maps = torch.cat(attention_maps, dim=1)
        modulate_maps = self.temporal_selection(attention_maps)
        modulate_maps = modulate_maps.unsqueeze(2)
        reference_feats = torch.stack(reference_feats, dim=1)
        result = torch.sum(reference_feats * modulate_maps, dim=1)
        result = smallMask * result + target_feat * (1 - smallMask)
        return result


class SpatialAttentionGen(BaseNetwork):
    def __init__(self, in_channels, conv_type, use_bias, norm):
        super(SpatialAttentionGen, self).__init__(conv_type)
        self.shrink1 = self.ConvBlock(in_channels, in_channels // 4, kernel_size=1, stride=1, padding=0, bias=use_bias,
                                      norm=norm)
        self.shrink2 = self.ConvBlock(in_channels, in_channels // 4, kernel_size=1, stride=1, padding=0, bias=use_bias,
                                      norm=norm)
        self.conv1 = self.ConvBlock(in_channels // 2, in_channels // 4 * 3, kernel_size=3, stride=1, padding=1,
                                    bias=use_bias, norm=norm)
        self.conv2 = self.ConvBlock(in_channels // 4 * 3, in_channels, kernel_size=3, stride=2, padding=1,
                                    bias=use_bias, norm=norm)
        self.conv3 = self.ConvBlock(in_channels, in_channels, kernel_size=3, stride=1, padding=1, bias=use_bias,
                                    norm=norm)
        self.conv4 = self.ConvBlock(in_channels, in_channels // 2, kernel_size=3, stride=1, padding=1, bias=use_bias,
                                    norm=norm)
        self.conv5 = self.DeconvBlock(in_channels // 2, in_channels // 4, kernel_size=3, stride=1, padding=1,
                                      bias=use_bias, norm=norm)
        self.conv6 = self.ConvBlock(in_channels // 4, 1, kernel_size=3, stride=1, padding=1, bias=use_bias, norm=norm,
                                    activation=None)

    def forward(self, target_feat, reference_feat):
        target_shrink = self.shrink1(target_feat)
        reference_shrink = self.shrink2(reference_feat)
        input_feat = torch.cat((target_shrink, reference_shrink), dim=1)
        a1 = self.conv1(input_feat)
        a2 = self.conv2(a1)
        a3 = self.conv3(a2)
        a4 = self.conv4(a3)
        a5 = self.conv5(a4)
        a6 = self.conv6(a5)
        return a6
