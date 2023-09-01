import torch

import numpy as np

import argparse
import os
from ddpm.unet import UNet
from ddpm.diffusion import Unet3D, GaussianDiffusion, Trainer
from ddpm.get_dataset import get_dataset

from NDC import dataset as dataset_NDC
from NDC import model as model_NDC

from con_gan.model import Generator, Discriminator

parser = argparse.ArgumentParser()

parser.add_argument("--epoch", action="store", dest="epoch", default=1000, type=int, help="Epoch to train [400,250,25]")
parser.add_argument("--train_lr", action="store", dest="train_lr", default=0.0001, type=float, help="Learning rate [0.0001]")

parser.add_argument("--img_size", action="store", dest="img_size", default=32, type=int, help="Input image size")
parser.add_argument("--dim_mults", action="store", dest="dim_mults", default=[1,2,4,8], type=list, help="Dimension Multiplication")
parser.add_argument("--num_channels", action="store", dest="num_channels", default=8, type=int, help="Number of channels")
parser.add_argument("--timesteps", action="store", dest="timesteps", default=300, type=int, help="Number of timesteps")
parser.add_argument("--loss_type", action="store", dest="loss_type", default='l1', type=str, help="Loss type")
parser.add_argument("--denoising_fn", action="store", dest="denoising_fn", default='Unet3D', type=str, help="Denoising function")

parser.add_argument("--root_dir", action="store", dest="root_dir", default='/data/jionkim/gt_NDC_KISTI_SDF_npy/', type=str, help="Root Directory")
parser.add_argument("--root_dir_ddpm", action="store", dest="root_dir_ddpm", default='/data/jionkim/gt_NDC_KISTI_SDF_npy/res_64/', type=str, help="Root Directory")
parser.add_argument("--vqgan_ckpt", action="store", dest="vqgan_ckpt", default='./outputs/checkpoint_vqgan.ckpt', type=str, help="VQGAN checkpoint")
parser.add_argument("--ddpm_ckpt", action="store", dest="ddpm_ckpt", default='./outputs/checkpoint_ddpm.ckpt', type=str, help="DDPM checkpoint")
parser.add_argument("--NDC_ckpt", action="store", dest="NDC_ckpt", default='./outputs/checkpoint_NDC.ckpt', type=str, help="DDPM checkpoint")
parser.add_argument("--batch_size", action="store", dest="batch_size", default=10, type=float, help="Batch size")
parser.add_argument("--sample_interval", action="store", dest="sample_interval", default=1000, type=float, help="Sampling interval")
parser.add_argument("--train_num_steps", action="store", dest="train_num_steps", default=100000, type=float, help="Number of training steps")
parser.add_argument("--results_folder", action="store", dest="results_folder", default='./outputs/NDC_ddpm/', type=str, help="Result folder")
parser.add_argument("--num_workers", action="store", dest="num_workers", default=20, type=float, help="Number of workers")

parser.add_argument("--postprocessing", action="store_true", dest="postprocessing", default=False, help="Enable the post-processing step to close small holes [False]")
parser.add_argument("--gpu", action="store", dest="gpu", default="1", help="to use which GPU [0]")

cfg = parser.parse_args()

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = cfg.gpu

# Diffusion Configuration
if cfg.denoising_fn == 'Unet3D':
    model = Unet3D(
        dim=cfg.img_size,
        dim_mults=cfg.dim_mults,
        channels=cfg.num_channels,
    ).cuda()
elif cfg.denoising_fn == 'UNet':
    model = UNet(
        in_ch=cfg.num_channels,
        out_ch=cfg.num_channels,
        spatial_dims=3
    ).cuda()
else:
    raise ValueError(f"Model {cfg.model.denoising_fn} doesn't exist")

diffusion = GaussianDiffusion(
    model,
    vqgan_ckpt=cfg.vqgan_ckpt,
    image_size=cfg.img_size,
    num_frames=cfg.img_size,
    channels=cfg.num_channels,
    timesteps=cfg.timesteps,
    loss_type=cfg.loss_type,
 ).cuda()

train_dataset, *_ = get_dataset(cfg)

ddpm_model = Trainer(
    diffusion,
    dataset=train_dataset,
    train_batch_size=cfg.batch_size,
    save_and_sample_every=cfg.sample_interval,
    train_lr=cfg.train_lr,
    train_num_steps=cfg.train_num_steps,
    results_folder=cfg.results_folder,
    num_workers=cfg.num_workers,
)

ddpm_model.load(cfg.ddpm_ckpt, map_location='cuda:0')
ddpm_model.model.eval()

device = torch.device('cuda')
torch.backends.cudnn.benchmark = True

# NCD Configuration
CNN_3d = model_NDC.CNN_3d_rec7

receptive_padding = 3 # for grid input
pooling_radius = 2 # for pointcloud input
KNN_num = 8

NDC_network = CNN_3d(out_bool=False, out_float=True)
NDC_network.load_state_dict(torch.load(cfg.NDC_ckpt))
NDC_network.eval()

dataset_train = dataset_NDC.ABC_grid_hdf5(cfg.root_dir, cfg.img_size, receptive_padding, train=True)
# dataset_test = dataset_NDC.ABC_grid_hdf5(cfg.root_dir, cfg.img_size, receptive_padding, train=False)

dataloader_train = torch.utils.data.DataLoader(dataset_train, batch_size=1, shuffle=True, num_workers=16)
# dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_size=1, shuffle=False, num_workers=16)

generator = Generator(NDC=NDC_network, ddpm=ddpm_model.ema_model, receptive_padding=receptive_padding)
# discriminator = Discriminator()

generator.cuda()
# discriminator.cuda()

optimizer_G = torch.optim.Adam(generator.parameters())
# optimizer_D = torch.optim.Adam(discriminator.parameters())

adversarial_loss = torch.nn.BCELoss()

#  Training
for epoch in range(cfg.train_num_steps):
    for i, data in enumerate(dataloader_train):

        # Configure input
        gt_input_, _, _ = data

        # Adversarial ground truths
        valid = torch.full(gt_input_.size(), 1.0)
        valid.requires_grad = False
        # fake = Variable(Tensor(imgs.size(0), 1).fill_(0.0), requires_grad=False)

        #  Train Generator
        # optimizer_G.zero_grad()

        # Sample noise as generator input
        # z = torch.rand([8, 1, 32, 32, 32]).cuda()

        # Generate vertices & triangles of mesh
        vertices, triangles = generator()

        print(vertices.shape)
        print(triangles.shape)

        # Loss measures generator's ability to fool the discriminator
       #  g_loss = adversarial_loss(discriminator(gen_imgs), valid)

        # g_loss.backward()
        # optimizer_G.step()

        '''
        #  Train Discriminator
        optimizer_D.zero_grad()

        # Measure discriminator's ability to classify real from generated samples
        real_loss = adversarial_loss(discriminator(real_imgs), valid)
        fake_loss = adversarial_loss(discriminator(gen_imgs.detach()), fake)
        d_loss = (real_loss + fake_loss) / 2

        d_loss.backward()
        optimizer_D.step()

        print(
            "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
            % (epoch, cfg.train_num_steps, i, len(dataloader_train), d_loss.item(), g_loss.item())
        )

        batches_done = epoch * len(dataloader_train) + i
        if batches_done % cfg.sample_interval == 0:
            save_image(gen_imgs.data[:25], "images/%d.png" % batches_done, nrow=5, normalize=True)
            '''