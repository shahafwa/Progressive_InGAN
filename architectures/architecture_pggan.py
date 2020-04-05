import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.spectral_norm as SpectralNorm
from torch.nn.init import kaiming_normal_, calculate_gain
from util import *

class EqualizedConv(nn.Module):
    def __init__(self, ni, no, ks, stride, pad, use_bias, reflect = True):
        super(EqualizedConv, self).__init__()
        self.ni = ni
        self.no = no
        self.ks = ks
        self.stride = stride
        self.use_bias = use_bias
        self.reflect = reflect
        if reflect:
            self.pad = nn.ReflectionPad2d(pad)
        else:
            self.pad = pad

        self.weight = nn.Parameter(nn.init.normal_(
            torch.empty(self.no, self.ni, self.ks, self.ks)
        ))
        if(self.use_bias):
            self.bias = nn.Parameter(torch.FloatTensor(self.no).fill_(0))

        self.scale = math.sqrt(2 / (self.ni * self.ks * self.ks))

    def forward(self, x):
        if self.reflect:
            out = self.pad(x)
            out = F.conv2d(input = out, weight = self.weight * self.scale, bias = self.bias,
                           stride = self.stride, padding=0)
        else:
            out = F.conv2d(input=x, weight=self.weight * self.scale, bias=self.bias,
                           stride=self.stride, padding=self.pad)
        return out

class ScaledConvBlock(nn.Module):
    def __init__(self, ni, no, ks, stride, pad, act = 'relu', use_bias = True, use_equalized_lr = True, use_pixelnorm = True, only_conv = False):
        super(ScaledConvBlock, self).__init__()
        self.ni = ni
        self.no = no
        self.ks = ks
        self.stride = stride
        self.pad = pad
        self.act = act
        self.use_bias = use_bias
        self.use_equalized_lr = use_equalized_lr
        self.use_pixelnorm = use_pixelnorm
        self.only_conv = only_conv

        self.relu = nn.LeakyReLU(0.2, inplace = True)
        if(self.use_equalized_lr):
            '''
            self.conv = nn.Conv2d(ni, no, ks, stride, pad, bias = False)
            kaiming_normal_(self.conv.weight, a = calculate_gain('conv2d'))

            self.bias = torch.nn.Parameter(torch.FloatTensor(no).fill_(0))
            self.scale = (torch.mean(self.conv.weight.data ** 2)) ** 0.5
            self.conv.weight.data.copy_(self.conv.weight.data / self.scale)
            '''
            self.conv = EqualizedConv(ni, no, ks, stride, pad, use_bias = use_bias)

        else:
            self.conv = nn.Conv2d(ni, no, ks, stride, pad, bias = use_bias)

        if(self.use_pixelnorm):
            self.pixel_norm = PixelNorm()


    def forward(self, x):
        '''
        if(self.use_equalized_lr):
            out = self.conv(x * self.scale)
            out = out + self.bias.view(1, -1, 1, 1).expand_as(out)
        else:
            out = self.conv(x)
        '''
        out = self.conv(x)

        if(self.only_conv == False):
            if(self.act == 'relu'):
                out = self.relu(out)
            if(self.use_pixelnorm):
                out = self.pixel_norm(out)

        return out

class UpSample(nn.Module):
    def __init__(self):
        super(UpSample, self).__init__()

    def forward(self, x, conf):
        return F.interpolate(x, None, conf.stage_factor, 'bilinear', align_corners=False)

class DownSample(nn.Module):
    def __init__(self):
        super(DownSample, self).__init__()

    def forward(self, x, conf):
        return F.interpolate(x, None, conf.stage_factor**(-1), 'bilinear', align_corners=False)

# Progressive Architectures

class Minibatch_Stddev(nn.Module):
    def __init__(self):
        super(Minibatch_Stddev, self).__init__()

    def forward(self, x):
        stddev = torch.sqrt(torch.mean((x - torch.mean(x, dim = 0, keepdim = True))**2, dim = 0, keepdim = True) + 1e-8)
        stddev_mean = torch.mean(stddev, dim = 1, keepdim = True)
        stddev_mean = stddev_mean.expand((x.size(0), 1, x.size(2), x.size(3)))

        return torch.cat([x, stddev_mean], dim = 1)

class PixelNorm(nn.Module):
    def __init__(self):
        super(PixelNorm, self).__init__()

    def forward(self, x):
        out = x / torch.sqrt(torch.mean(x**2, dim = 1, keepdim = True) + 1e-8)
        return out

normalization_layer = nn.BatchNorm2d

class ResnetBlock(nn.Module):
    """ A single Res-Block module """

    def __init__(self, dim, use_bias):
        super(ResnetBlock, self).__init__()

        # A res-block without the skip-connection, pad-conv-norm-relu-pad-conv-norm
        self.conv_block = nn.Sequential(nn.utils.spectral_norm(EqualizedConv(dim, dim//4, stride=1, pad=0, ks=1, use_bias=use_bias)),
                                        normalization_layer(dim // 4),
                                        nn.LeakyReLU(0.2, True),
                                        nn.ReflectionPad2d(1),
                                        nn.utils.spectral_norm(EqualizedConv(dim//4, dim//4, stride=1, pad=0, ks=3, use_bias=use_bias)),
                                        normalization_layer(dim // 4),
                                        nn.LeakyReLU(0.2, True),
                                        nn.utils.spectral_norm(EqualizedConv(dim//4, dim, stride=1, pad=0, ks=1, use_bias=use_bias)),
                                        normalization_layer(dim))

    def forward(self, input_tensor):
        # The skip connection is applied here
        return input_tensor + self.conv_block(input_tensor)


class RescaleBlock(nn.Module):
    def __init__(self, n_layers, scale=0.5, base_channels=64, use_bias=True):
        super(RescaleBlock, self).__init__()

        self.scale = scale

        self.conv_layers = [None] * n_layers

        in_channel_power = scale > 1
        out_channel_power = scale < 1
        i_range = range(n_layers) if scale < 1 else range(n_layers-1, -1, -1)

        for i in i_range:
            self.conv_layers[i] = nn.Sequential(nn.ReflectionPad2d(1),
                                                nn.utils.spectral_norm(EqualizedConv(
                                                    base_channels * 2 ** (i + in_channel_power),
                                                    base_channels * 2 ** (i + out_channel_power),
                                                    ks=3,
                                                    stride=1,
                                                    pad=0,
                                                    use_bias=True)),
                                                normalization_layer(base_channels * 2 ** (i + out_channel_power)),
                                                nn.LeakyReLU(0.2, True))
            self.add_module("conv_%d" % i, self.conv_layers[i])

        if scale > 1:
            self.conv_layers = self.conv_layers[::-1]

        self.max_pool = nn.MaxPool2d(2, 2)

    def forward(self, input_tensor, pyramid=None, return_all_scales=False, skip=False):

        feature_map = input_tensor
        all_scales = []
        if return_all_scales:
            all_scales.append(feature_map)

        for i, conv_layer in enumerate(self.conv_layers):

            if self.scale > 1.0:
                feature_map = F.interpolate(feature_map, scale_factor=self.scale, mode='nearest')

            feature_map = conv_layer(feature_map)

            if skip:
                if feature_map.shape == pyramid[-i - 2].shape:
                    feature_map = feature_map + pyramid[-i - 2]
                else:
                    feature_map = F.interpolate(feature_map, size=pyramid[-i - 2].shape[2:], mode='nearest') + pyramid[-i - 2]

            if self.scale < 1.0:
                feature_map = self.max_pool(feature_map)

            if return_all_scales:
                all_scales.append(feature_map)

        return (feature_map, all_scales) if return_all_scales else (feature_map, None)


class Generator(nn.Module):
    """ Architecture of the Generator, uses res-blocks """

    def __init__(self, base_channels=64, n_blocks=6, n_downsampling=1, use_bias=True, skip_flag=True):
        super(Generator, self).__init__()

        # Determine whether to use skip connections
        self.skip = skip_flag

        # Entry block
        # First conv-block, no stride so image dims are kept and channels dim is expanded (pad-conv-norm-relu)
        self.entry_block = nn.Sequential(nn.ReflectionPad2d(3),
                                         nn.utils.spectral_norm(EqualizedConv(3, base_channels, stride=1, pad=0, ks=7, use_bias=use_bias)),
                                         normalization_layer(base_channels),
                                         nn.LeakyReLU(0.2, True))


        # Downscaling
        # A sequence of strided conv-blocks. Image dims shrink by 2, channels dim expands by 2 at each block
        self.downscale_block = RescaleBlock(n_downsampling, 0.5, base_channels, True)

        # Bottleneck
        # A sequence of res-blocks
        bottleneck_block = []
        for _ in range(n_blocks):
            # noinspection PyUnboundLocalVariable
            bottleneck_block += [ResnetBlock(base_channels * 2 ** n_downsampling, use_bias=use_bias)]
        self.bottleneck_block = nn.Sequential(*bottleneck_block)

        # Upscaling
        # A sequence of transposed-conv-blocks, Image dims expand by 2, channels dim shrinks by 2 at each block\
        self.upscale_block = RescaleBlock(n_downsampling, 2.0, base_channels, True)

        # Final block
        # No stride so image dims are kept and channels dim shrinks to 3 (output image channels)
        self.final_block = nn.Sequential(nn.ReflectionPad2d(3),
                                         EqualizedConv(base_channels, 3, stride=1, pad=0, ks=7, use_bias=use_bias),
                                         nn.Tanh())

    def forward(self, input_tensor):
        # A condition for having the output at same size as the scaled input is having even output_size

        # Entry block
        feature_map = self.entry_block(input_tensor)

        # # Change scale to output scale by interpolation
        # if random_affine is None:
        #     feature_map = F.interpolate(feature_map, size=output_size, mode='bilinear')
        # else:
        #     feature_map = self.geo_transform.forward(feature_map, output_size, random_affine)

        # Downscale block
        feature_map, downscales = self.downscale_block.forward(feature_map, return_all_scales=self.skip)

        # Bottleneck (res-blocks)
        feature_map = self.bottleneck_block(feature_map)

        # Upscale block
        feature_map, _ = self.upscale_block.forward(feature_map, pyramid=downscales, skip=self.skip)

        # Final block
        output_tensor = self.final_block(feature_map)

        return output_tensor


class PGGAN_G(nn.Module):
    def __init__(self, sz, nz, nc, conf, use_pixelnorm = False, use_equalized_lr = False, use_tanh = True):
        super(PGGAN_G, self).__init__()
        self.sz = sz
        self.nz = nz
        self.nc = nc
        self.conf = conf
        self.ngfs = {
            '8': [512, 512],
            '16': [512, 512, 512],
            '32': [512, 512, 512, 512],
            '64': [512, 512, 512, 512, 256],
            '128': [512, 512, 512, 512, 256, 128],
            '256': [512, 512, 512, 512, 256, 128, 64],
            '512': [512, 512, 512, 512, 256, 128, 64, 32],
            '1024': [512, 512, 512, 512, 256, 128, 64, 32, 16]
        }

        self.cur_ngf = self.ngfs[str(sz)]

        self.base_block = nn.Sequential(
            PixelNorm()
        )

        # create blocks list
        prev_dim = self.cur_ngf[0]
        cur_block = nn.Sequential(
            ScaledConvBlock(nz, prev_dim, 3, 1, 1, 'relu', True, use_equalized_lr, use_pixelnorm),
            ScaledConvBlock(prev_dim, prev_dim, 3, 1, 1, 'relu', True, use_equalized_lr, use_pixelnorm)
        )
        self.blocks = nn.ModuleList([cur_block])
        for dim in self.cur_ngf[1:]:
            cur_block = nn.Sequential(
                ScaledConvBlock(prev_dim, dim, 3, 1, 1, 'relu', True, use_equalized_lr, use_pixelnorm),
                ScaledConvBlock(dim, dim, 3, 1, 1, 'relu', True, use_equalized_lr, use_pixelnorm)
            )
            prev_dim = dim
            self.blocks.append(cur_block)

        # create to_blocks list
        self.to_blocks = nn.ModuleList([])
        for dim in self.cur_ngf:
            self.to_blocks.append(ScaledConvBlock(dim, nc, 1, 1, 0, None, True, use_equalized_lr, use_pixelnorm, only_conv = True))

        self.use_tanh = use_tanh
        self.tanh = nn.Tanh()
        self.upsample = UpSample()
        self.downsample = DownSample()

        self.geo = GeoTransform()

        self.baseLevel = Generator()

    # def forward(self, x, stage, output_size, shift, reconstruction=False):
    #     stage_int = int(stage)
    #     stage_type = (stage == stage_int)
    #     change = self.upsample
    #     out = x
    #
    #     temp_factor = self.conf.stage_factor ** stage
    #     out = self.geo(out, output_size, shift)
    #     out = resize_tensor(out, out.shape[2] / temp_factor, out.shape[3] / temp_factor)
    #
    #
    #     if not reconstruction:
    #         change = self.upsample
    #         # out = self.base_block(out)
    #         # out = self.geo(x, output_size, shift)
    #
    #
    #
    #
    #     # Stablization Steps
    #     if(stage_type):
    #         for i in range(stage_int):
    #             out = self.blocks[i](out)
    #             out = change(out, self.conf)
    #         out = self.blocks[stage_int](out)
    #         out = self.to_blocks[stage_int](out)
    #
    #     # Growing Steps
    #     else:
    #         p = stage - stage_int
    #         for i in range(stage_int+1):
    #             out = self.blocks[i](out)
    #             out = change(out, self.conf)
    #
    #         out_1 = self.to_blocks[stage_int](out)
    #         out_2 = self.blocks[stage_int+1](out)
    #         out_2 = self.to_blocks[stage_int+1](out_2)
    #         out = out_1 * (1 - p) + out_2 * p
    #
    #     # if reconstruction:
    #     #     out = self.geo(x, output_size, shift)
    #
    #     if(self.use_tanh):
    #         out = self.tanh(out)
    #
    #     return out

    def forward(self, x, stage, output_size, final_output_size, shift, reconstruction=False, level=0):
        stage_int = int(stage)
        stage_type = (stage == stage_int)
        change = self.upsample
        out = x


        # if reconstruction:
        #     out = F.interpolate(out, scale_factor=self.conf.stage_factor ** (-level), mode='bilinear', align_corners=False)

        out = self.geo(out, output_size, shift)

        out = self.baseLevel(out)


        # Stablization Steps
        if(stage_type):
            for i in range(stage_int):
                out = self.blocks[i](out)
                next_level_factor = self.conf.stage_factor ** (stage_int - 1 - i)
                next_level_output = (int(final_output_size[0] / next_level_factor), int(final_output_size[1] / next_level_factor))
                out = F.interpolate(out, size=next_level_output, mode='nearest')
            out = self.blocks[stage_int](out)
            out = self.to_blocks[stage_int](out)


        # Growing Steps
        else:
            p = stage - stage_int
            for i in range(stage_int+1):
                out = self.blocks[i](out)
                next_level_factor = self.conf.stage_factor ** (stage_int  - i)
                next_level_output = (int(final_output_size[0] / next_level_factor), int(final_output_size[1] / next_level_factor))
                out = F.interpolate(out, size=next_level_output, mode='nearest')

            out_1 = self.to_blocks[stage_int](out)
            out_2 = self.blocks[stage_int+1](out)
            out_2 = self.to_blocks[stage_int+1](out_2)
            out = out_1 * (1 - p) + out_2 * p






        if(self.use_tanh):
            out = self.tanh(out)

        return out

class PGGAN_D(nn.Module):
    def __init__(self, sz, nc, use_sigmoid = True, use_pixelnorm = False, use_equalized_lr = False):
        super(PGGAN_D, self).__init__()
        self.sz = sz
        self.nc = nc
        self.sigmoid = nn.Sigmoid()
        self.use_sigmoid = use_sigmoid
        self.ndfs = {
            '8': [512, 512],
            '16': [512, 512, 512],
            '32': [512, 512, 512, 512],
            '64': [512, 512, 512, 512, 256],
            '128': [512, 512, 512, 512, 256, 128],
            '256': [512, 512, 512, 512, 256, 128, 64],
            '512': [512, 512, 512, 512, 256, 128, 64, 32],
            '1024': [512, 512, 512, 512, 256, 128, 64, 32, 16]
        }

        self.cur_ndf = self.ndfs[str(sz)]

        # create blocks list
        prev_dim = self.cur_ndf[0]
        cur_block = nn.Sequential(
            Minibatch_Stddev(),
            ScaledConvBlock(prev_dim+1, prev_dim, 3, 1, 1, 'relu', True, use_equalized_lr, use_pixelnorm),
            ScaledConvBlock(prev_dim, prev_dim, 4, 1, 0, 'relu', True, use_equalized_lr, use_pixelnorm)
        )
        self.blocks = nn.ModuleList([cur_block])
        for dim in self.cur_ndf[1:]:
            cur_block = nn.Sequential(
                ScaledConvBlock(dim, dim, 3, 1, 1, 'relu', True, use_equalized_lr, use_pixelnorm),
                ScaledConvBlock(dim, prev_dim, 3, 1, 1, 'relu', True, use_equalized_lr, use_pixelnorm)
            )
            prev_dim = dim
            self.blocks.append(cur_block)

        # create from_blocks list
        self.from_blocks = nn.ModuleList([])
        for dim in self.cur_ndf:
            self.from_blocks.append(ScaledConvBlock(nc, dim, 1, 1, 0, None, True, use_equalized_lr, use_pixelnorm, only_conv = True))

        self.linear = nn.Linear(self.cur_ndf[0], 1)
        self.downsample = DownSample()



    def forward(self, x, stage):
        stage_int = int(stage)
        stage_type = (stage == stage_int)
        out = x

        # Stablization Steps
        if(stage_type):
            out = self.from_blocks[stage_int](out)
            for i in range(stage_int):
                out = self.blocks[stage_int - i](out)
                out = self.downsample(out)
            out = self.blocks[0](out)
            out = self.linear(out.view(out.shape[0], -1))
            out = out.view(out.shape[0], 1, 1, 1)

        # Growing Steps
        else:
            p = stage - stage_int
            out_1 = self.downsample(out)
            out_1 = self.from_blocks[stage_int](out_1)

            out_2 = self.from_blocks[stage_int+1](out)
            out_2 = self.blocks[stage_int+1](out_2)
            out_2 = self.downsample(out_2)

            out = out_1 * (1 - p) + out_2 * p

            for i in range(stage_int):
                out = self.blocks[stage_int - i](out)
                out = self.downsample(out)
            out = self.blocks[0](out)
            out = self.linear(out.view(out.shape[0], -1))
            out = out.view(out.shape[0], 1, 1, 1)

        if(self.use_sigmoid):
            out = self.sigmoid(out)

        return out


class GeoTransform(nn.Module):
    def __init__(self):
        super(GeoTransform, self).__init__()

    def forward(self, input_tensor, target_size, shifts):
        if len(shifts) == 1:
            # radial
            psf = 0.5  # max(0, shifts[0])
            isz = input_tensor.shape
            pad = F.pad(input_tensor, [int(isz[-1] * psf), int(isz[-1] * psf),
                                       int(isz[-2] * psf), int(isz[-2] * psf)], 'reflect')
            target_size4d = torch.Size([pad.shape[0], pad.shape[1], target_size[0], target_size[1]])
            grid = non_rect.make_radial_scale_grid(shifts[0], target_size4d)
        else:
            # homographies
            sz = input_tensor.shape
            theta = homography_based_on_top_corners_x_shift(shifts)

            pad = F.pad(input_tensor,
                        (np.abs(np.int(np.ceil(sz[3] * shifts[0]))), np.abs(np.int(np.ceil(-sz[3] * shifts[1]))), 0, 0),
                        'reflect')
            target_size4d = torch.Size([pad.shape[0], pad.shape[1], target_size[0], target_size[1]])
            grid = homography_grid(theta.expand(pad.shape[0], -1, -1), target_size4d)

        return F.grid_sample(pad, grid, mode='bilinear', padding_mode='border')


class MultiScaleDiscriminator(nn.Module):
    def __init__(self, real_crop_size, max_n_scales=5, scale_factor=2, base_channels=128, extra_conv_layers=0):
        super(MultiScaleDiscriminator, self).__init__()
        self.base_channels = base_channels
        self.scale_factor = scale_factor
        self.min_size = 16
        self.extra_conv_layers = extra_conv_layers

        # We want the max num of scales to fit the size of the real examples. further scaling would create networks that
        # only train on fake examples
        self.max_n_scales = np.min([np.int(np.ceil(np.log(np.min(real_crop_size) * 1.0 / self.min_size)
                                                   / np.log(self.scale_factor))), max_n_scales])

        # Prepare a list of all the networks for all the wanted scales
        self.nets = nn.ModuleList()


        # Create a network for each scale
        for k in range(self.max_n_scales): #base channels per network
            s = np.min([int(k/2)+1, 4])
            self.nets.append(self.make_net(s))

    def make_net(self, k=1):
        base_channels = int(self.base_channels * k)
        net = []

        # Entry block
        net += [nn.utils.spectral_norm(EqualizedConv(3, base_channels, ks=3, stride=1, pad =0, use_bias=True)),
                nn.BatchNorm2d(base_channels),
                nn.LeakyReLU(0.2, True)]

        # Downscaling blocks
        # A sequence of strided conv-blocks. Image dims shrink by 2, channels dim expands by 2 at each block
        net += [nn.utils.spectral_norm(EqualizedConv(base_channels, int(base_channels * 2 * k), ks=3, stride=2, pad=0, use_bias=True)),
                nn.BatchNorm2d(int(base_channels * 2 * k)),
                nn.LeakyReLU(0.2, True)]

        # Regular conv-block
        net += [nn.utils.spectral_norm(EqualizedConv(ni=int(base_channels * 2 * k),
                                                 no=base_channels * 2,
                                                 ks=3,
                                                 use_bias=True,
                                                     pad=0,
                                                     stride=1)),
                nn.BatchNorm2d(base_channels * 2),
                nn.LeakyReLU(0.2, True)]

        # Additional 1x1 conv-blocks
        for _ in range(self.extra_conv_layers):
            net += [nn.utils.spectral_norm(EqualizedConv(ni=base_channels * 2,
                                                     no=base_channels * 2,
                                                     ks=3,
                                                     use_bias=True,
                                                         pad=0,
                                                         stride=1)),
                    nn.BatchNorm2d(base_channels * 2),
                    nn.LeakyReLU(0.2, True)]

        # Final conv-block
        # Ends with a Sigmoid to get a range of 0-1
        net += nn.Sequential(nn.utils.spectral_norm(EqualizedConv(base_channels * 2, 1, ks=1, stride=1, pad=0, use_bias=True)),
                             nn.Sigmoid())

        # Make it a valid layers sequence and return
        return nn.Sequential(*net)

    def forward(self, input_tensor, scale_weights,stage, level):

        stage_int = int(stage)
        stage_type = (stage == stage_int)
        out = input_tensor




        aggregated_result_maps_from_all_scales = self.nets[stage_int](out) * scale_weights[0]
        map_size = aggregated_result_maps_from_all_scales.shape[2:]

        # Stablization Steps
        if(stage_type and stage_int):
            # for i in range (stage_int):
            #     parFreeze(self.nets[i], False)

            for i in range(stage_int*2, 0, -1):
                downscaled_image = F.interpolate(input_tensor, scale_factor=self.scale_factor ** (-i), mode='bilinear', align_corners=False)
                result_map_for_current_scale = self.nets[i](downscaled_image)
                upscaled_result_map_for_current_scale = F.interpolate(result_map_for_current_scale,
                                                                     size=map_size,
                                                                     mode='bilinear', align_corners = False)
                aggregated_result_maps_from_all_scales += upscaled_result_map_for_current_scale * scale_weights[i]
        # Growing Steps
        elif(stage_int):
            # for i in range(stage_int):
            #     parFreeze(self.nets[i], True)

            for i in range(stage_int*2, 0, -1):
                downscaled_image = F.interpolate(input_tensor, scale_factor=self.scale_factor ** (-i), mode='bilinear', align_corners=False)
                result_map_for_current_scale = self.nets[i](downscaled_image)
                upscaled_result_map_for_current_scale = F.interpolate(result_map_for_current_scale,
                                                                      size=map_size,
                                                                      mode='bilinear', align_corners=False)
                aggregated_result_maps_from_all_scales += upscaled_result_map_for_current_scale * scale_weights[i]

        return aggregated_result_maps_from_all_scales




def getSpot(stage, i):

    return 6 + i - 2 * stage

