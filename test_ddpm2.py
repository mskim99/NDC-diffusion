from ddpm.diffusion import Unet3D, GaussianDiffusion, Trainer
import hydra
from omegaconf import DictConfig, open_dict
from ddpm.get_dataset import get_dataset
import torch
import os
from ddpm.unet import UNet
from torchvision import transforms as T
import numpy as np

def video_tensor_to_gif(tensor, path, duration=120, loop=0, optimize=True):
    tensor = ((tensor - tensor.min()) / (tensor.max() - tensor.min())) * 1.0
    images = map(T.ToPILImage(), tensor.unbind(dim=1))
    first_img, *rest_imgs = images
    first_img.save(path, save_all=True, append_images=rest_imgs,
                   duration=duration, loop=loop, optimize=optimize)
    return images


@hydra.main(config_path='./config', config_name='base_cfg', version_base=None)
def run(cfg: DictConfig):
    torch.cuda.set_device(cfg.model.gpus)
    with open_dict(cfg):
        cfg.model.results_folder = os.path.join(
            cfg.model.results_folder, cfg.dataset.name, cfg.model.results_folder_postfix)

    if cfg.model.denoising_fn == 'Unet3D':
        model = Unet3D(
            dim=cfg.model.diffusion_img_size,
            dim_mults=cfg.model.dim_mults,
            channels=cfg.model.diffusion_num_channels,
        ).cuda()
    elif cfg.model.denoising_fn == 'UNet':
        model = UNet(
            in_ch=cfg.model.diffusion_num_channels,
            out_ch=cfg.model.diffusion_num_channels,
            spatial_dims=3
        ).cuda()
    else:
        raise ValueError(f"Model {cfg.model.denoising_fn} doesn't exist")

    diffusion = GaussianDiffusion(
        model,
        vqgan_ckpt=cfg.model.vqgan_ckpt,
        # vqgan_ckpt=None,
        image_size=cfg.model.diffusion_img_size,
        num_frames=cfg.model.diffusion_depth_size,
        channels=cfg.model.diffusion_num_channels,
        timesteps=cfg.model.timesteps,
        loss_type=cfg.model.loss_type,
    ).cuda()

    train_dataset, *_ = get_dataset(cfg)

    trainer = Trainer(
        diffusion,
        dataset=train_dataset,
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
    )

    trainer.load("/home/jionkim/workspace/NDC-diffusion/outputs/checkpoint_ddpm4.ckpt", map_location='cuda:0')

    # trainer.ema_model.eval()

    samples = trainer.ema_model.sample(batch_size=2)
    samples = ((samples - samples.min()) / (samples.max() - samples.min()))
    print(samples.min())
    print(samples.max())
    samples = (samples + 1.0) * 127.5
    video_tensor_to_gif(samples[0][0], '/home/jionkim/workspace/NDC-diffusion/outputs/output_ddpm/output.gif')
    # np.save('/home/jionkim/workspace/NDC-diffusion/outputs/output_ddpm/output.npy', samples[0][0].cpu())


if __name__ == '__main__':
    run()