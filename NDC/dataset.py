import numpy as np
import torch
from NDC.utils import read_data,read_and_augment_data_ndc,read_data_input_only, read_sdf_file_as_3d_array
import torchvision.transforms as transforms

import sys
sys.path.append('../')
from mesh.mesh import Mesh
from mesh.mesh_util import pad, get_mean_std

class ABC_grid_hdf5(torch.utils.data.Dataset):

    def extract_edge_features(self, mesh):
        edge_features = mesh.extract_features()
        edge_features = pad(edge_features, mesh.edges_count)
        edge_features = (edge_features - self.mean) / self.std
        return edge_features

    def __init__(self, data_dir, output_grid_size, receptive_padding, train, input_only=False):
        self.data_dir = data_dir
        self.output_grid_size = output_grid_size
        self.receptive_padding = receptive_padding
        self.train = train
        self.input_only = input_only
        fin = open("abc_obj_list.txt", 'r')
        self.file_names = [name.strip() for name in fin.readlines()]
        fin.close()

        m, s, ni = get_mean_std('/data/jionkim/gt_NDC_KISTI_SDF_npy')
        self.mean = m
        self.std = s
        self.ninput_channels = ni

        if self.train:
            self.file_names = self.file_names[:int(len(self.file_names)*0.8)]
            print("Total#", "train", len(self.file_names))
        else:
            self.file_names = self.file_names[int(len(self.file_names)*0.8):]
            print("Total#", "test", len(self.file_names))

        temp_file_names = []
        temp_file_gridsizes = []
        if self.train:
            for name in self.file_names:
                for grid_size in [32,64]:
                    temp_file_names.append(name)
                    temp_file_gridsizes.append(grid_size)
        else:
            for name in self.file_names:
                temp_file_names.append(name)
                temp_file_gridsizes.append(self.output_grid_size)

        self.file_names = temp_file_names
        self.file_gridsizes = temp_file_gridsizes
        self.transform = transforms.Compose([transforms.ToTensor()])
        print("Non-trivial Total#", len(self.file_names))

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, index):
        # hdf5_dir = self.data_dir + "/" + self.file_names[index] + ".hdf5"
        grid_size = self.file_gridsizes[index]

        if self.train:
            gt_output_bool_, gt_output_float_,gt_input_ = read_and_augment_data_ndc(self.data_dir, self.file_names[index], grid_size, self.train,
                                                                   aug_permutation=True,aug_reversal=True,aug_inversion=True)
        else:
            if self.input_only:
                gt_output_bool_, gt_output_float_,gt_input_ = read_data_input_only(self.data_dir, self.file_names[index], grid_size)
            else:
                gt_output_bool_, gt_output_float_,gt_input_ = read_data(self.data_dir, self.file_names[index], grid_size)

        if not self.train:
            gt_output_bool_ = np.transpose(gt_output_bool_, [3,0,1,2]).astype(np.float32)
            gt_output_bool_mask_ = np.zeros(gt_output_bool_.shape, np.float32)
        gt_output_float_ = np.transpose(gt_output_float_, [3,0,1,2])
        gt_output_float_mask_ = (gt_output_float_>=0).astype(np.float32)

        gt_input_ = np.expand_dims(gt_input_, axis=0).astype(np.float32)

        # crop to save space & time
        # get bounding box
        if self.train:
            valid_flag = np.max(gt_output_float_mask_,axis=0)

            # x
            ray = np.max(valid_flag,(1,2))
            xmin = -1
            xmax = -1
            for i in range(grid_size+1):
                if ray[i]>0:
                    if xmin==-1:
                        xmin = i
                    xmax = i
            #y
            ray = np.max(valid_flag,(0,2))
            ymin = -1
            ymax = -1
            for i in range(grid_size+1):
                if ray[i]>0:
                    if ymin==-1:
                        ymin = i
                    ymax = i
            #z
            ray = np.max(valid_flag,(0,1))
            zmin = -1
            zmax = -1
            for i in range(grid_size+1):
                if ray[i]>0:
                    if zmin==-1:
                        zmin = i
                    zmax = i

            xmax += 1
            ymax += 1
            zmax += 1

        else:
            xmin = 0
            xmax = grid_size+1
            ymin = 0
            ymax = grid_size+1
            zmin = 0
            zmax = grid_size+1

        if not self.train:
            gt_output_bool = gt_output_bool_[:,xmin:xmax,ymin:ymax,zmin:zmax]
            gt_output_bool_mask = gt_output_bool_mask_[:,xmin:xmax,ymin:ymax,zmin:zmax]
        gt_output_float = gt_output_float_[:,xmin:xmax,ymin:ymax,zmin:zmax]
        gt_output_float_mask = gt_output_float_mask_[:,xmin:xmax,ymin:ymax,zmin:zmax]

        xmin = xmin-self.receptive_padding
        xmax = xmax+self.receptive_padding
        ymin = ymin-self.receptive_padding
        ymax = ymax+self.receptive_padding
        zmin = zmin-self.receptive_padding
        zmax = zmax+self.receptive_padding

        xmin_pad = 0
        xmax_pad = xmax-xmin
        ymin_pad = 0
        ymax_pad = ymax-ymin
        zmin_pad = 0
        zmax_pad = zmax-zmin

        if gt_input_[0,0,0,0]>0:
            gt_input = np.full([1,xmax_pad,ymax_pad,zmax_pad],10,np.float32)
        else:
            gt_input = np.full([1,xmax_pad,ymax_pad,zmax_pad],-10,np.float32)

        if xmin<0:
            xmin_pad -= xmin
            xmin = 0
        if xmax>grid_size+1:
            xmax_pad += (grid_size+1-xmax)
            xmax=grid_size+1
        if ymin<0:
            ymin_pad -= ymin
            ymin = 0
        if ymax>grid_size+1:
            ymax_pad += (grid_size+1-ymax)
            ymax=grid_size+1
        if zmin<0:
            zmin_pad -= zmin
            zmin = 0
        if zmax>grid_size+1:
            zmax_pad += (grid_size+1-zmax)
            zmax=grid_size+1

        gt_input[:,xmin_pad:xmax_pad,ymin_pad:ymax_pad,zmin_pad:zmax_pad] = gt_input_[:,xmin:xmax,ymin:ymax,zmin:zmax]

        # the current code assumes that each cell in the input is a unit cube
        # clip to ignore far-away cells
        gt_input = np.clip(gt_input, -2, 2)

        # load mesh
        mesh = Mesh(file=self.data_dir+'/mesh_f_3000/'+self.file_names[index]+'.obj')
        meta = {'mesh': mesh, 'gt_input': gt_input, 'gt_output_float': gt_output_float, 'gt_output_float_mask': gt_output_float_mask}
        edge_features = self.extract_edge_features(mesh)
        '''
        edge_features = mesh.extract_features()
        edge_features = pad(edge_features, mesh.edges_count)
        edge_features = (edge_features - self.mean) / self.std
        '''
        meta['edge_features'] = edge_features
        '''
        if self.train:
            return gt_input, gt_output_float, gt_output_float_mask
        else: # Test State
            return gt_input, gt_output_bool, gt_output_bool_mask, gt_output_float, gt_output_float_mask
            '''

        if not self.train:
            meta['gt_output_bool'] = gt_output_bool
            meta['gt_output_bool_mask'] = gt_output_bool_mask

        return meta


#only for testing
class single_shape_grid(torch.utils.data.Dataset):
    def __init__(self, data_dir, receptive_padding):
        self.data_dir = data_dir
        self.receptive_padding = receptive_padding

    def __len__(self):
        return 1

    def __getitem__(self, index):

        LOD_input = read_sdf_file_as_3d_array(self.data_dir)

        if LOD_input.shape[1] != LOD_input.shape[0] or LOD_input.shape[2] != LOD_input.shape[0]:
            print("ERROR: the code only supports grids with dimx=dimy=dimz, but the given grid is", LOD_input.shape)
            exit(-1)

        grid_size = LOD_input.shape[0]-1
        gt_input_ = LOD_input * grid_size # denormalize


        # prepare mask
        gt_output_bool_mask_ = np.zeros([1,grid_size+1,grid_size+1,grid_size+1], np.float32)
        gt_input_ = np.expand_dims(gt_input_, axis=0).astype(np.float32)
        gt_output_bool_mask = gt_output_bool_mask_

        #receptive field padding
        padded = grid_size+1 + self.receptive_padding*2

        if gt_input_[0,0,0,0]>0:
            gt_input = np.full([1,padded,padded,padded],10,np.float32)
        else:
            gt_input = np.full([1,padded,padded,padded],-10,np.float32)

        gt_input[:,self.receptive_padding:-self.receptive_padding,self.receptive_padding:-self.receptive_padding,self.receptive_padding:-self.receptive_padding] = gt_input_


        # the current code assumes that each cell in the input is a unit cube
        # clip to ignore far-away cells
        gt_input = np.clip(gt_input, -2, 2)

        return gt_input, gt_output_bool_mask
