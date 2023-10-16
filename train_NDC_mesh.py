import torch

torch.backends.cudnn.enabled = False
torch.autograd.set_detect_anomaly(True)
import torch.nn.functional as F
from torch import autograd
from torch.autograd import Variable

import numpy as np

import argparse
import os
import datetime

from skimage.transform import resize

from NDC import dataset as dataset_NDC
from NDC import model as model_NDC

from con_gan.model import SDF_Generator, SDF_Discriminator, Mesh_Generator, Mesh_Discriminator


def collate_fn(batch):
    """Creates mini-batch tensors
    We should build custom collate_fn rather than using default collate_fn
    """
    meta = {}
    keys = batch[0].keys()
    for key in keys:
        meta.update({key: np.array([d[key] for d in batch])})
    return meta


def calc_gradient_penalty(netD, real_data, fake_data):
    batch_size = real_data.size(0)
    epsilon = torch.rand(batch_size, 1, 1, 1, 1)
    epsilon = epsilon.expand_as(real_data)
    epsilon = epsilon.cuda()

    interpolation = epsilon * real_data.data + (1 - epsilon) * fake_data.data
    interpolation = Variable(interpolation, requires_grad=True)
    interpolation = interpolation.cuda()

    interpolation_logits = netD(interpolation)
    grad_outputs = torch.ones(interpolation_logits.size())
    grad_outputs = grad_outputs.cuda()

    gradients = autograd.grad(outputs=interpolation_logits,
                              inputs=interpolation,
                              grad_outputs=grad_outputs,
                              create_graph=True,
                              retain_graph=True)[0]

    gradients = gradients.view(batch_size, -1)
    gradients_norm = torch.sqrt(torch.sum(gradients ** 2, dim=1) + 1e-12)
    return 10. * ((gradients_norm - 1) ** 2).mean()


parser = argparse.ArgumentParser()

parser.add_argument("--epoch", action="store", dest="epoch", default=50, type=int, help="Epoch to train [400,250,25]")
parser.add_argument("--train_lr", action="store", dest="train_lr", default=1e-4, type=float,
                    help="Learning rate [0.0001]")
parser.add_argument("--g_train_lr", action="store", dest="g_train_lr", default=1e-4, type=float,
                    help="Learning rate [0.0001]")
parser.add_argument("--d_train_lr", action="store", dest="d_train_lr", default=1e-4, type=float,
                    help="Learning rate [0.0001]")
parser.add_argument("--n_critic", action="store", dest="n_critic", default=2, type=int,
                    help="Number of Critie Execution")

parser.add_argument("--img_size", action="store", dest="img_size", default=32, type=int, help="Input image size")
parser.add_argument("--dim_mults", action="store", dest="dim_mults", default=[1, 2, 4, 8], type=list,
                    help="Dimension Multiplication")
parser.add_argument("--num_channels", action="store", dest="num_channels", default=8, type=int,
                    help="Number of channels")
parser.add_argument("--timesteps", action="store", dest="timesteps", default=10, type=int, help="Number of timesteps")
parser.add_argument("--loss_type", action="store", dest="loss_type", default='l1', type=str, help="Loss type")
parser.add_argument("--denoising_fn", action="store", dest="denoising_fn", default='Unet3D', type=str,
                    help="Denoising function")

parser.add_argument("--root_dir", action="store", dest="root_dir", default='/data/jionkim/gt_NDC_KISTI_SDF_p_100_npy/',
                    type=str, help="Root Directory")
parser.add_argument("--root_dir_ddpm", action="store", dest="root_dir_ddpm",
                    default='/data/jionkim/gt_NDC_KISTI_SDF_p_100_npy/res_64/', type=str, help="Root Directory")
parser.add_argument("--vqgan_ckpt", action="store", dest="vqgan_ckpt", default='./outputs/checkpoint_vqgan.ckpt',
                    type=str, help="VQGAN checkpoint")
parser.add_argument("--ddpm_ckpt", action="store", dest="ddpm_ckpt", default='./outputs/checkpoint_ddpm.ckpt', type=str,
                    help="DDPM checkpoint")
parser.add_argument("--NDC_ckpt", action="store", dest="NDC_ckpt", default='./outputs/checkpoint_NDC.ckpt', type=str,
                    help="DDPM checkpoint")
parser.add_argument("--batch_size", action="store", dest="batch_size", default=4, type=float, help="Batch size")
parser.add_argument("--sample_interval", action="store", dest="sample_interval", default=1000, type=float,
                    help="Sampling interval")
parser.add_argument("--train_num_steps", action="store", dest="train_num_steps", default=1000, type=float,
                    help="Number of training steps")
parser.add_argument("--results_folder", action="store", dest="results_folder", default='./outputs/NDC_ddpm2/', type=str,
                    help="Result folder")
parser.add_argument("--num_workers", action="store", dest="num_workers", default=1, type=float,
                    help="Number of workers")

parser.add_argument("--postprocessing", action="store_true", dest="postprocessing", default=False,
                    help="Enable the post-processing step to close small holes [False]")
parser.add_argument("--gpu", action="store", dest="gpu", default="0", help="to use which GPU [0]")

cfg = parser.parse_args()

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = cfg.gpu

device = torch.device('cuda')
torch.backends.cudnn.benchmark = True

# NCD Configuration
CNN_3d = model_NDC.CNN_3d_rec7

receptive_padding = 3  # for grid input
pooling_radius = 2  # for pointcloud input
KNN_num = 8

NDC_network = CNN_3d(out_bool=False, out_float=True)
NDC_network.load_state_dict(torch.load(cfg.NDC_ckpt))
NDC_network.train()

dataset_train = dataset_NDC.ABC_grid_hdf5(cfg.root_dir, cfg.img_size, receptive_padding, train=True)
# dataset_train = dataset_NDC.ABC_grid_hdf5_sdf(cfg.root_dir, cfg.img_size, receptive_padding, train=True)
# dataset_test = dataset_NDC.ABC_grid_hdf5(cfg.root_dir, cfg.img_size, receptive_padding, train=False)

dataloader_train = torch.utils.data.DataLoader(dataset_train, batch_size=cfg.batch_size, shuffle=True,
                                               num_workers=cfg.num_workers)
# dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_size=1, shuffle=False, num_workers=cfg.num_workers, collate_fn=collate_fn)

sdf_generator = SDF_Generator()
sdf_discriminator = SDF_Discriminator()
mesh_generator = Mesh_Generator(NDC=NDC_network, receptive_padding=receptive_padding)
mesh_discriminator = Mesh_Discriminator(nf0=5, conv_res=[64, 128, 256, 256], nclasses=1000, input_res=750, pool_res=[600, 450, 300, 180], fc_n=1, norm='group', num_groups=cfg.num_workers, device_num=cfg.gpu)

sdf_generator.cuda()
sdf_discriminator.cuda()
mesh_generator.cuda()
mesh_discriminator.cuda()

optimizer_SG = torch.optim.RMSprop(sdf_generator.parameters(), lr=cfg.train_lr)  # SDF Generator
optimizer_SD = torch.optim.RMSprop(sdf_discriminator.parameters(), lr=cfg.train_lr)  # SDF Disciminator
# optimizer_MG = torch.optim.Adam(mesh_generator.parameters(), lr=cfg.train_lr) # Mesh Generator
# optimizer_MD = torch.optim.Adam(mesh_discriminator.parameters(), lr=cfg.train_lr) # Mesh Disciminator

adversarial_loss = torch.nn.BCELoss()
dist_loss = torch.nn.L1Loss()

#  Training
for epoch in range(cfg.train_num_steps):
    for i, data in enumerate(dataloader_train):

        # Configure input (Mesh)
        # gt_edge_features_ = data['edge_features']
        # gt_input_ = data['gt_input']
        # gt_mesh_ = data['mesh']
        # gt_output_float_ = data['gt_output_float']

        # Configure input (SDF)
        gt_input = data

        # Adversarial ground truths
        valid = torch.full([1, 1000], 1.0).cuda()
        fake = torch.full([1, 1000], 0.0).cuda()
        valid.requires_grad = False
        fake.requires_grad = False

        #  Train SDF Discriminator
        optimizer_SD.zero_grad()

        z = torch.rand([cfg.batch_size, 1, 64, 64, 64]).cuda()
        sdf_gen = sdf_generator(z)

        gt_input = gt_input.cuda()
        gt_input = F.normalize(gt_input)
        gt_input = (gt_input + 1.) / 2.

        dis_output_gt_sdf = sdf_discriminator(sdf=gt_input)
        dis_output_gen_sdf = sdf_discriminator(sdf=sdf_gen)

        gp_vol = calc_gradient_penalty(sdf_discriminator, gt_input, sdf_gen)

        d_loss_sdf = torch.mean(dis_output_gen_sdf) - torch.mean(dis_output_gt_sdf)
        d_loss_sdf.backward()
        gp_vol.backward()
        optimizer_SD.step()

        #  Train SDF Generator
        if i % cfg.n_critic == 0:
            optimizer_SG.zero_grad()

            # Sample noise as generator input
            z = torch.rand([cfg.batch_size, 1, 64, 64, 64]).cuda()
            sdf_gen = sdf_generator(z)

            reg_loss = dist_loss(sdf_gen, gt_input)

            dis_output_gen_sdf = sdf_discriminator(sdf=sdf_gen)

            g_loss_sdf = -torch.mean(dis_output_gen_sdf) + reg_loss

            g_loss_sdf.backward()
            optimizer_SG.step()

        print(
            "[%s] [Epoch %d/%d] [Batch %d/%d] [G loss (SDF): %f] [D loss (SDF): %f] [reg loss (SDF): %f] [gp loss (SDF): %f]"
            % (datetime.datetime.now().time(), epoch, cfg.train_num_steps, i, len(dataloader_train), g_loss_sdf.item(),
               d_loss_sdf.item(), reg_loss.item(), gp_vol.item())
        )

        if i % 500 == 0:
            sdf_gen_npy = sdf_gen.cpu().detach().numpy()
            sdf_gen_npy = sdf_gen_npy.squeeze()
            np.save('./gen_SDFs3/' + str(epoch) + '_' + str(i).zfill(4) + '_gen.npy', sdf_gen_npy)

    if epoch % 10 == 0:
        checkpoint = {
            'epoch_idx': epoch,
            'sdf_generator_state_dict': sdf_generator.state_dict(),
            'sdf_generator_solver_state_dict': optimizer_SG.state_dict(),
            'sdf_discriminator_state_dict': sdf_discriminator.state_dict(),
            'sdf_discriminator_solver_state_dict': optimizer_SD.state_dict(),
        }

        torch.save(checkpoint, './checkpoint3/ckpt-epoch-%04d-sdf.pth' % epoch)

    '''
        #  Train Mesh Generator
        optimizer_MG.zero_grad()

        sdf_gen_reshape_ = resize(sdf_gen.cpu().detach().numpy(), gt_input_.shape)
        sdf_gen_reshape_ = (sdf_gen_reshape_ - sdf_gen_reshape_.min()) / (sdf_gen_reshape_.max() - sdf_gen_reshape_.min())
        sdf_gen_reshape_  = 4. * sdf_gen_reshape_ - 2.
        sdf_gen_reshape = torch.Tensor(sdf_gen_reshape_).cuda()
        # sdf_gen_reshape = F.normalize(sdf_gen_reshape)
        # sdf_gen_reshape = torch.mul(sdf_gen_reshape, 4.)
        # sdf_gen_reshape = torch.sub(sdf_gen_reshape, 2.)
        gt_input = torch.Tensor(gt_input_).cuda()


        # Generate vertices & triangles of mesh
        # gt_input = torch.Tensor(z).cuda()
        vertices, triangles, gen_mesh_ = mesh_generator(gt_sdf=sdf_gen)
        gen_mesh = (gen_mesh_,)
        gen_mesh = np.array(gen_mesh)
        gen_edge_features = dataset_NDC.ABC_grid_hdf5.extract_edge_features(self=dataset_train, mesh=gen_mesh_)
        # gen_edge_features = gen_edge_features[:, 0:4500]

        gt_edge_features = torch.Tensor(gt_edge_features_).cuda()
        gen_edge_features = torch.Tensor(gen_edge_features).cuda()
        gen_edge_features = torch.unsqueeze(gen_edge_features, 0)

        dis_output_gt = mesh_discriminator(gt_edge_features, gt_mesh_)
        dis_output_gen = mesh_discriminator(gen_edge_features, gen_mesh)

        # print(dis_output_gen)

        # exit()
        # Loss measures generator's ability to fool the discriminator
        g_loss_mesh = -dis_output_gen.mean()

        g_loss_mesh.backward()
        optimizer_MG.step()

        #  Train Mesh Discriminator
        optimizer_MD.zero_grad()

        # Measure discriminator's ability to classify real from generated samples
        # real_loss = adversarial_loss(dis_output_gt, valid)
        # fake_loss = adversarial_loss(dis_output_gen, fake)
        # d_loss = (real_loss + fake_loss) / 2
        d_loss_mesh = dis_output_gen.detach().mean() - -dis_output_gt.mean()
        d_loss_mesh.backward()
        optimizer_MD.step()

        print(
            "[Epoch %d/%d] [Batch %d/%d] [G loss (SDF): %f] [D loss (SDF): %f] [G loss (mesh): %f] [D loss (mesh): %f]"
            % (epoch, cfg.train_num_steps, i, len(dataloader_train), g_loss_sdf.item(), d_loss_mesh.sdf(), g_loss_mesh.item(), d_loss_mesh.item())
        )

        if i % 500 == 0:
            gen_mesh_.export(file='./gen_meshes/' + str(epoch) + '_' + str(i).zfill(4) +'_gen.obj')

    checkpoint = {
        'epoch_idx': epoch,
        'sdf_generator_state_dict': sdf_generator.state_dict(),
        'sdf_generator_solver_state_dict': optimizer_SG.state_dict(),
        'sdf_discriminator_state_dict': sdf_discriminator.state_dict(),
        'sdf_discriminator_solver_state_dict': optimizer_SD.state_dict(),
        'mesh_generator_state_dict': mesh_generator.state_dict(),
        'mesh_generator_solver_state_dict': optimizer_MG.state_dict(),
        'mesh_discriminator_state_dict': mesh_discriminator.state_dict(),
        'mesh_discriminator_solver_state_dict': optimizer_MD.state_dict()
    }

    torch.save(checkpoint, './checkpoint/ckpt-epoch-%04d.pth' % (epoch + 1))
    '''