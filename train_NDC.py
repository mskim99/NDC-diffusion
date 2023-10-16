import argparse
import os
import numpy as np
import time

import torch

parser = argparse.ArgumentParser()

parser.add_argument("--epoch", action="store", dest="epoch", default=1000, type=int, help="Epoch to train [400,250,25]")
parser.add_argument("--lr", action="store", dest="lr", default=0.0001, type=float, help="Learning rate [0.0001]")
parser.add_argument("--lr_half_life", action="store", dest="lr_half_life", default=10000, type=int, help="Halve lr every few epochs [100,5]")
parser.add_argument("--data_dir", action="store", dest="data_dir", default="/data/jionkim/gt_NDC_KISTI_SDF_p_100_npy", help="Root directory of dataset [gt_NDC,gt_UNDC,gt_UNDCa]")
parser.add_argument("--checkpoint_dir", action="store", dest="checkpoint_dir", default="weights", help="Directory name to save the checkpoints [weights]")
parser.add_argument("--checkpoint_save_frequency", action="store", dest="checkpoint_save_frequency", default=50, type=int, help="Save checkpoint every few epochs [50,10,1]")
parser.add_argument("--sample_dir", action="store", dest="sample_dir", default="samples", help="Directory name to save the output samples [samples]")

parser.add_argument("--train", action="store_true", dest="train", default=True, help="Training only float with one network [False]")
parser.add_argument("--test", action="store_true", dest="test", default=False, help="Testing only float with one network, using GT bool [False]")
parser.add_argument("--test_input", action="store", dest="test_input", default="", help="Select an input file for quick testing [*.sdf, *.binvox, *.ply, *.hdf5]")

parser.add_argument("--point_num", action="store", dest="point_num", default=4096, type=int, help="Size of input point cloud for testing [4096,16384,524288]")
parser.add_argument("--grid_size", action="store", dest="grid_size", default=64, type=int, help="Size of output grid for testing [32,64,128]")
parser.add_argument("--block_num_per_dim", action="store", dest="block_num_per_dim", default=5, type=int, help="Number of blocks per dimension [1,5,10]")
parser.add_argument("--block_padding", action="store", dest="block_padding", default=5, type=int, help="Padding for each block [5]")

parser.add_argument("--input_type", action="store", dest="input_type", default="sdf", help="Input type [sdf,voxel,udf,pointcloud,noisypc]")
parser.add_argument("--method", action="store", dest="method", default="ndc", help="Method type [ndc,undc,ndcx]")
parser.add_argument("--postprocessing", action="store_true", dest="postprocessing", default=False, help="Enable the post-processing step to close small holes [False]")
parser.add_argument("--gpu", action="store", dest="gpu", default="0", help="to use which GPU [0]")

FLAGS = parser.parse_args()

is_training = False # training on a dataset
is_testing = False # testing on a dataset
quick_testing = False # testing on a single shape/scene
if FLAGS.train:
    is_training = True
if FLAGS.test:
    is_testing = True

if FLAGS.test_input != "":
    quick_testing = True


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = FLAGS.gpu

from NDC import dataset
from NDC import model
from NDC import utils

if not os.path.exists(FLAGS.sample_dir):
    os.makedirs(FLAGS.sample_dir)
if not os.path.exists(FLAGS.checkpoint_dir):
    os.makedirs(FLAGS.checkpoint_dir)

device = torch.device('cuda')
torch.backends.cudnn.benchmark = True

# Create network
CNN_3d = model.CNN_3d_rec7
receptive_padding = 3 # for grid input
pooling_radius = 2 # for pointcloud input
KNN_num = 8

network_float = CNN_3d(out_bool=False, out_float=True)
network_float.to(device)


def worker_init_fn(worker_id):
    np.random.seed(int(time.time()*10000000)%10000000 + worker_id)


if is_training:

    # Create train / test dataset
    dataset_train = dataset.ABC_grid_hdf5(FLAGS.data_dir, FLAGS.grid_size, receptive_padding, train=True)
    dataset_test = dataset.ABC_grid_hdf5(FLAGS.data_dir, FLAGS.grid_size, receptive_padding, train=False)

    dataloader_train = torch.utils.data.DataLoader(dataset_train, batch_size=1, shuffle=True, num_workers=16, worker_init_fn=worker_init_fn) #batch_size must be 1
    dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_size=1, shuffle=False, num_workers=16)  #batch_size must be 1

    optimizer = torch.optim.Adam(network_float.parameters())

    start_time = time.time()
    for epoch in range(FLAGS.epoch):
        network_float.train()

        if epoch%FLAGS.lr_half_life==0:
            for g in optimizer.param_groups:
                lr = FLAGS.lr/(2**(epoch//FLAGS.lr_half_life))
                print("Setting learning rate to", lr)
                g['lr'] = lr

        avg_loss = 0
        avg_acc_bool = 0
        avg_acc_float = 0
        avg_loss_count = 0
        avg_acc_bool_count = 0
        avg_acc_float_count = 0

        for i, data in enumerate(dataloader_train, 0):
            gt_input_, gt_output_float_, gt_output_float_mask_ = data

            gt_input = gt_input_.to(device)
            gt_output_float = gt_output_float_.to(device)
            gt_output_float_mask = gt_output_float_mask_.to(device)

            optimizer.zero_grad()

            pred_output_float = network_float(gt_input)

            print(pred_output_float.shape)

            #MSE
            loss_float = torch.sum(( (pred_output_float-gt_output_float)**2 )*gt_output_float_mask)/torch.clamp(torch.sum(gt_output_float_mask),min=1)

            loss = loss_float
            avg_acc_float += loss_float.item()
            avg_acc_float_count += 1

            avg_loss += loss.item()
            avg_loss_count += 1

            loss.backward()
            optimizer.step()

        print('[%d/%d] time: %.0f  loss: %.8f' % (epoch, FLAGS.epoch, time.time()-start_time, avg_loss/max(avg_loss_count,1)))

        if epoch % FLAGS.checkpoint_save_frequency == FLAGS.checkpoint_save_frequency-1:

            # save weights
            print('saving net...')
            torch.save(network_float.state_dict(), FLAGS.checkpoint_dir+"/weights_"+FLAGS.method+"_"+FLAGS.input_type+"_float_epoch_"+str(epoch)+".pth")
            print('saving net... complete')

            # test
            network_float.eval()

            for i, data in enumerate(dataloader_test, 0):

                gt_input_, gt_output_bool_, gt_output_bool_mask_, gt_output_float_, gt_output_float_mask_ = data
                gt_input = gt_input_.to(device)

                with torch.no_grad():
                    pred_output_float = network_float(gt_input)

                pred_output_bool_numpy = np.transpose(gt_output_bool_[0].detach().cpu().numpy(), [1,2,3,0])
                pred_output_bool_numpy = (pred_output_bool_numpy>0.5).astype(np.int32)
                pred_output_float_numpy = np.transpose(pred_output_float[0].detach().cpu().numpy(), [1,2,3,0])

                pred_output_float_numpy = np.clip(pred_output_float_numpy,0,1)
                vertices, triangles = utils.dual_contouring_ndc_test(pred_output_bool_numpy, pred_output_float_numpy)
                utils.write_obj_triangle(FLAGS.sample_dir+"/test_"+str(i)+".obj", vertices, triangles)

                if i>=32: break


elif is_testing:

    import cutils

    # Create test dataset
    dataset_test = dataset.ABC_grid_hdf5(FLAGS.data_dir, FLAGS.grid_size, receptive_padding, FLAGS.input_type, train=False, input_only=True)
    dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_size=1, shuffle=False, num_workers=16)  # batch_size must be 1

    # load weights
    print('loading net...')
    network_float.load_state_dict(torch.load(FLAGS.checkpoint_dir+"/weights_"+FLAGS.method+"_"+FLAGS.input_type+"_float.pth"))
    print('network_float weights loaded')
    print('loading net... complete')

    # test
    network_float.eval()


    for i, data in enumerate(dataloader_test, 0):

        gt_input_, gt_output_bool_, gt_output_bool_mask_, gt_output_float_, gt_output_float_mask_ = data

        gt_input = gt_input_.to(device)
        if FLAGS.method == "undc":
            gt_output_bool_mask = gt_output_bool_mask_.to(device)

        with torch.no_grad():
            pred_output_float = network_float(gt_input)

            pred_output_bool_numpy = np.transpose(gt_output_bool_[0].detach().cpu().numpy(), [1,2,3,0])
            pred_output_bool_numpy = (pred_output_bool_numpy>0.5).astype(np.int32)

            pred_output_float_numpy = np.transpose(pred_output_float[0].detach().cpu().numpy(), [1,2,3,0])

    pred_output_float_numpy = np.clip(pred_output_float_numpy,0,1)
    vertices, triangles = cutils.dual_contouring_ndc(np.ascontiguousarray(pred_output_bool_numpy, np.int32), np.ascontiguousarray(pred_output_float_numpy, np.float32))
    utils.write_obj_triangle(FLAGS.sample_dir+"/test_"+str(i)+".obj", vertices, triangles)

elif quick_testing:
    import cutils

    # load weights
    print('loading net...')
    network_float.load_state_dict(torch.load(FLAGS.checkpoint_dir+"/weights_"+FLAGS.method+"_"+FLAGS.input_type+"_float.pth"))
    print('network_float weights loaded')
    print('loading net... complete')

    # test
    network_float.eval()

    #Create test dataset
    dataset_test = dataset.single_shape_grid(FLAGS.test_input, receptive_padding, FLAGS.input_type)
    dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_size=1, shuffle=False, num_workers=1)  #batch_size must be 1

    for i, data in enumerate(dataloader_test, 0):

        gt_input_, gt_output_bool_, gt_output_bool_mask_, gt_output_float_, gt_output_float_mask_ = data

        gt_input = gt_input_.to(device)
        with torch.no_grad():
            pred_output_float = network_float(gt_input)

            pred_output_bool_numpy = np.transpose(gt_output_bool_[0].detach().cpu().numpy(), [1,2,3,0])
            pred_output_bool_numpy = (pred_output_bool_numpy>0.5).astype(np.int32)

            pred_output_float_numpy = np.transpose(gt_output_float_[0].detach().cpu().numpy(), [1,2,3,0])

    pred_output_float_numpy = np.clip(pred_output_float_numpy,0,1)
    vertices, triangles = cutils.dual_contouring_ndc(np.ascontiguousarray(pred_output_bool_numpy, np.int32), np.ascontiguousarray(pred_output_float_numpy, np.float32))
    utils.write_obj_triangle(FLAGS.sample_di + "/quicktest_" + FLAGS.method + "_" + FLAGS.input_type + ".obj", vertices, triangles)