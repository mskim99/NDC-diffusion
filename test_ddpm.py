# %%

GPU_NUM = 0

import torch
import os
import sys
from re import I

os.environ["CUDA_VISIBLE_DEVICES"] = f"{GPU_NUM}"
# %env CUDA_VISIBLE_DEVICES=3
# torch.cuda.set_device(GPU_NUM)
device = f'cuda:0'
sys.path.append('../')
from ddpm import Unet3D, GaussianDiffusion, Trainer
from ddpm.unet import UNet
from hydra import initialize, compose
import numpy as np
from torchvision import transforms as T

def video_tensor_to_gif(tensor, path, duration=120, loop=0, optimize=True):
    tensor = ((tensor - tensor.min()) / (tensor.max() - tensor.min())) * 1.0
    images = map(T.ToPILImage(), tensor.unbind(dim=1))
    first_img, *rest_imgs = images
    first_img.save(path, save_all=True, append_images=rest_imgs,
                   duration=duration, loop=loop, optimize=optimize)
    return images


with initialize(config_path="./config/"):
    cfg = compose(config_name="base_cfg.yaml", overrides=[
        "model=ddpm",
        "dataset=default",
        f"model.vqgan_ckpt=outputs/checkpoint_vqgan.ckpt",
        "model.diffusion_img_size=64",
        "model.diffusion_depth_size=64",
        "model.diffusion_num_channels=1",
        "model.dim_mults=[1,2,4,8]",
        "model.batch_size=10",
        "model.gpus=0 ",
    ])

# train_dataset, _, _ = get_dataset(cfg)

## Evaluate VQ-GAN
'''
vqgan = VQGAN.load_from_checkpoint('../checkpoint/checkpoint_vqgan.ckpt')
vqgan = vqgan.to(device)
vqgan.eval();

# %%

sample = train_dataset[2]
input_ = torch.tensor(sample['data'][None]).to(device)
with torch.no_grad():
    output_ = vqgan(input_)

sitk.WriteImage(sitk.GetImageFromArray(output_[1][0][0].cpu()), '../debug/output.nii')
sitk.WriteImage(sitk.GetImageFromArray(input_[0][0].cpu()), '../debug/input.nii')
'''

## Evaluate DDPM
model = Unet3D(
    dim=64,
    dim_mults=cfg.model.dim_mults,
    channels=1,
).cuda()

'''
model = UNet(
    in_ch=cfg.model.diffusion_num_channels,
    out_ch=cfg.model.diffusion_num_channels,
    spatial_dims=3
).cuda()
'''

diffusion = GaussianDiffusion(
    model,
    vqgan_ckpt=None,
    image_size=cfg.model.diffusion_img_size,
    num_frames=cfg.model.diffusion_depth_size,
    channels=cfg.model.diffusion_num_channels,
    timesteps=cfg.model.timesteps,
    # sampling_timesteps=cfg.model.sampling_timesteps,
    loss_type=cfg.model.loss_type,
    # objective=cfg.objective
).cuda()

trainer = Trainer(
    diffusion,
    # folder='/data/jionkim/gt_NDC_KISTI_SDF/',
    # dataset=train_dataset,
    train_batch_size=cfg.model.batch_size,
    save_and_sample_every=cfg.model.save_and_sample_every,
    train_lr=cfg.model.train_lr,
    train_num_steps=cfg.model.train_num_steps,
    gradient_accumulate_every=cfg.model.gradient_accumulate_every,
    ema_decay=cfg.model.ema_decay,
    amp=cfg.model.amp,
    num_sample_rows=cfg.model.num_sample_rows,
    results_folder=cfg.model.results_folder,
    num_workers=cfg.model.num_workers,
    dataset_use=False,
    # logger=cfg.model.logger
)

trainer.load("/home/jionkim/workspace/NDC-diffusion/outputs/checkpoint_ddpm3.ckpt", map_location='cuda:0')

for i in range (0, 1):
    sample = trainer.ema_model.sample(batch_size=2)
    video_tensor_to_gif(sample[0][0], '/home/jionkim/workspace/NDC-diffusion/outputs/output_ddpm/output.gif')
    # np.save('../evaluation/output_ddpm/output_' + str(i).zfill(3) + '.npy', sample[0][0].cpu())
    # sample = (sample + 1.0) * 127.5
    # sitk.WriteImage(sitk.GetImageFromArray(sample[0][0].cpu()), '../debug/test.nii')
