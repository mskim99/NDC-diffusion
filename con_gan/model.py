import torch.nn as nn
import numpy as np

import sys
sys.path.append('../')
from NDC.utils import gen_mesh

class Generator(nn.Module):
    def __init__(self, ddpm, NDC, receptive_padding):
        super(Generator, self).__init__()
        self.NDC = NDC
        self.ddpm = ddpm
        self.receptive_padding = receptive_padding

    def forward(self):
        ddpm_output = self.ddpm.sample(batch_size=1)
        ddpm_output = ddpm_output.clone()
        vertices, triangles = gen_mesh(self.NDC, ddpm_output, self.receptive_padding)
        return vertices, triangles


class Discriminator(nn.Module):
    def __init__(self, img_shape):
        super(Discriminator, self).__init__()
        self.img_shape = img_shape

        self.model = nn.Sequential(
            nn.Linear(int(np.prod(self.img_shape)), 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, img):
        img_flat = img.view(img.size(0), -1)
        validity = self.model(img_flat)

        return validity