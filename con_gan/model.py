import torch
import torch.nn as nn
import torch.nn.functional as F

import functools

import sys
sys.path.append('../')
from NDC.utils import gen_mesh
from mesh.mesh_conv import MeshConv
from mesh.mesh_pool import MeshPool


class SDF_Generator(nn.Module):
    def __init__(self):
        super(SDF_Generator, self).__init__()

        self.e_layer1 = torch.nn.Sequential(
            torch.nn.Conv3d(1, 32, kernel_size=4, stride=2, padding=1),
            torch.nn.BatchNorm3d(32),
            torch.nn.LeakyReLU(),
        )
        self.e_layer2 = torch.nn.Sequential(
            torch.nn.Conv3d(32, 64, kernel_size=4, stride=2, padding=1),
            torch.nn.BatchNorm3d(64),
            torch.nn.LeakyReLU(),
        )
        self.e_layer3 = torch.nn.Sequential(
            torch.nn.Conv3d(64, 128, kernel_size=4, stride=2, padding=1),
            torch.nn.BatchNorm3d(128),
            torch.nn.LeakyReLU(),
        )
        self.e_layer4 = torch.nn.Sequential(
            torch.nn.Conv3d(128, 256, kernel_size=4, stride=2, padding=1),
            torch.nn.BatchNorm3d(256),
            torch.nn.LeakyReLU(),
        )
        self.e_layer5 = torch.nn.Sequential(
            torch.nn.Conv3d(256, 512, kernel_size=4, stride=2, padding=1),
            torch.nn.BatchNorm3d(512),
            torch.nn.LeakyReLU(),
        )

        self.d_layer1 = torch.nn.Sequential(
            torch.nn.ConvTranspose3d(512, 256, kernel_size=4, stride=2, padding=1),
            torch.nn.BatchNorm3d(256),
            torch.nn.ReLU(),
        )
        self.d_layer2 = torch.nn.Sequential(
            torch.nn.ConvTranspose3d(256, 128, kernel_size=4, stride=2, padding=1),
            torch.nn.BatchNorm3d(128),
            torch.nn.ReLU(),
        )
        self.d_layer3 = torch.nn.Sequential(
            torch.nn.ConvTranspose3d(128, 64, kernel_size=4, stride=2, padding=1),
            torch.nn.BatchNorm3d(64),
            torch.nn.ReLU(),
        )
        self.d_layer4 = torch.nn.Sequential(
            torch.nn.ConvTranspose3d(64, 32, kernel_size=4, stride=2, padding=1),
            torch.nn.BatchNorm3d(32),
            torch.nn.ReLU(),
        )
        self.d_layer5 = torch.nn.Sequential(
            torch.nn.ConvTranspose3d(32, 1, kernel_size=4, stride=2, padding=1),
            torch.nn.Sigmoid(),
        )

    def forward(self, z):

        # print(z.shape)
        features = self.e_layer1(z)
        # print(features.shape)
        features = self.e_layer2(features)
        # print(features.shape)
        features = self.e_layer3(features)
        # print(features.shape)
        features = self.e_layer4(features)
        # print(features.shape)
        features = self.e_layer5(features)
        # print(features.shape)

        features = self.d_layer1(features)
        # print(features.shape)
        features = self.d_layer2(features)
        # print(features.shape)
        features = self.d_layer3(features)
        # print(features.shape)
        features = self.d_layer4(features)
        # print(features.shape)
        output = self.d_layer5(features)
        # print(output.shape)
        return output


class SDF_Discriminator(nn.Module):
    def __init__(self):
        super(SDF_Discriminator, self).__init__()

        self.layer1 = torch.nn.Sequential(
            torch.nn.Conv3d(1, 32, kernel_size=4, stride=2, padding=1),
            torch.nn.BatchNorm3d(32),
            torch.nn.LeakyReLU(),
        )
        self.layer2 = torch.nn.Sequential(
            torch.nn.Conv3d(32, 64, kernel_size=4, stride=2, padding=1),
            torch.nn.BatchNorm3d(64),
            torch.nn.LeakyReLU(),
        )
        self.layer3 = torch.nn.Sequential(
            torch.nn.Conv3d(64, 128, kernel_size=4, stride=2, padding=1),
            torch.nn.BatchNorm3d(128),
            torch.nn.LeakyReLU(),
        )
        self.layer4 = torch.nn.Sequential(
            torch.nn.Conv3d(128, 256, kernel_size=4, stride=2, padding=1),
            torch.nn.BatchNorm3d(256),
            torch.nn.LeakyReLU(),
        )
        self.layer5 = torch.nn.Sequential(
            torch.nn.Conv3d(256, 512, kernel_size=4, stride=2, padding=1),
            torch.nn.BatchNorm3d(512),
            torch.nn.LeakyReLU(),
        )
        self.layer6 = torch.nn.Sequential(
            torch.nn.Conv3d(512, 1, kernel_size=2, stride=2, padding=0),
            torch.nn.Sigmoid()
        )

    def forward(self, sdf):

        # print(sdf.shape)
        features = self.layer1(sdf)
        # print(features.shape)
        features = self.layer2(features)
        # print(features.shape)
        features = self.layer3(features)
        # print(features.shape)
        features = self.layer4(features)
        # print(features.shape)
        features = self.layer5(features)
        # print(features.shape)
        output = self.layer6(features)
        # print(output.shape)
        output = output.reshape(-1)
        # print(output.shape)
        return output


class Mesh_Generator(nn.Module):
    def __init__(self, NDC, receptive_padding):
        super(Mesh_Generator, self).__init__()
        self.NDC = NDC
        # self.ddpm = ddpm
        self.receptive_padding = receptive_padding

    def forward(self, gt_sdf):
        # ddpm_output = self.ddpm.sample(batch_size=1)
        # ddpm_output = ddpm_output.clone()
        vertices, triangles, mesh = gen_mesh(self.NDC, gt_sdf, self.receptive_padding)
        return vertices, triangles, mesh


class Mesh_Discriminator(nn.Module):
    def __init__(self, nf0, conv_res, nclasses, input_res, pool_res, fc_n, norm, num_groups, device_num,
                 nresblocks=3):
        super(Mesh_Discriminator, self).__init__()
        self.k = [nf0] + conv_res
        self.res = [input_res] + pool_res
        self.device_num = device_num
        norm_layer = get_norm_layer(norm_type=norm, num_groups=num_groups)
        norm_args = get_norm_args(norm_layer, self.k[1:])

        for i, ki in enumerate(self.k[:-1]):
            setattr(self, 'conv{}'.format(i), MResConv(ki, self.k[i + 1], self.device_num, nresblocks))
            setattr(self, 'norm{}'.format(i), norm_layer(**norm_args[i]))
            setattr(self, 'pool{}'.format(i), MeshPool(self.res[i + 1]))

        self.gp = torch.nn.AvgPool1d(self.res[-1])
        # self.gp = torch.nn.MaxPool1d(self.res[-1])
        self.fc1 = nn.Linear(self.k[-1], fc_n)
        self.fc2 = nn.Linear(fc_n, nclasses)

    def forward(self, x, mesh):

        # print('# Discriminator (MeshConvNet) #')
        # print(self.k)
        # print(len(self.k))
        # print(x.shape)
        for i in range(len(self.k) - 1):
            # print('###### Loop' + str(i) + '#######')
            x = getattr(self, 'conv{}'.format(i))(x, mesh)
            # print('conv{}'.format(i))
            # print(x.shape)
            x = F.relu(getattr(self, 'norm{}'.format(i))(x))
            # print('norm{}'.format(i))
            # print(x.shape)
            x = getattr(self, 'pool{}'.format(i))(x, mesh)
            # print('pool{}'.format(i))
            # print(x.shape)
            # print('###### Endloop #######')


        # print(x.shape)
        x = self.gp(x)
        # print(x.shape)
        x = x.view(-1, self.k[-1])

        # print(x.shape)
        x = F.relu(self.fc1(x))
        # print(x.shape)
        x = self.fc2(x)
        # output = F.softmax(x)
        return x


class MResConv(nn.Module):
    def __init__(self, in_channels, out_channels, device_num, skips=1):
        super(MResConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.skips = skips
        self.conv0 = MeshConv(self.in_channels, self.out_channels, device_num=device_num, bias=False)
        for i in range(self.skips):
            setattr(self, 'bn{}'.format(i + 1), nn.BatchNorm2d(self.out_channels))
            setattr(self, 'conv{}'.format(i + 1),
                    MeshConv(self.out_channels, self.out_channels, device_num=device_num, bias=False))

    def forward(self, x, mesh):
        # print('###### MResConv ######')
        # print(x.shape)
        x = self.conv0(x, mesh)
        # print(x.shape)
        x1 = torch.empty_like(x).copy_(x)
        for i in range(self.skips):
            x = getattr(self, 'bn{}'.format(i + 1))(F.relu(x))
            # print(x.shape)
            x = getattr(self, 'conv{}'.format(i + 1))(x, mesh)
            # print(x.shape)
            # xo = F.softmax(x)
        x += x1
        # print(x.shape)
        output = F.relu(x)
        # print(x.shape)
        # print('###################')
        return output


def get_norm_args(norm_layer, nfeats_list):
    if hasattr(norm_layer, '__name__') and norm_layer.__name__ == 'NoNorm':
        norm_args = [{'fake': True} for f in nfeats_list]
    elif norm_layer.func.__name__ == 'GroupNorm':
        norm_args = [{'num_channels': f} for f in nfeats_list]
    elif norm_layer.func.__name__ == 'BatchNorm':
        norm_args = [{'num_features': f} for f in nfeats_list]
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_layer.func.__name__)
    return norm_args


def get_norm_layer(norm_type='instance', num_groups=1):
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False)
    elif norm_type == 'group':
        norm_layer = functools.partial(nn.GroupNorm, affine=True, num_groups=num_groups)
    elif norm_type == 'none':
        norm_layer = NoNorm
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


class NoNorm(nn.Module): #todo with abstractclass and pass
    def __init__(self, fake=True):
        self.fake = fake
        super(NoNorm, self).__init__()
    def forward(self, x):
        return x
    def __call__(self, x):
        return self.forward(x)