import numpy as np
import os
import pickle

import sys
sys.path.append('../')
from mesh.mesh import Mesh

# from mesh import Mesh

def pad(input_arr, target_length, val=0, dim=1):
    shp = input_arr.shape
    npad = [(0, 0) for _ in range(len(shp))]
    npad[dim] = (0, target_length - shp[dim])
    return np.pad(input_arr, pad_width=npad, mode='constant', constant_values=val)

def get_mean_std(root_dir):
    """ Computes Mean and Standard Deviation from Training Data
    If mean/std file doesn't exist, will compute one
    :returns
    mean: N-dimensional mean
    std: N-dimensional standard deviation
    ninput_channels: N
    (here N=5)
    """

    mean_std_cache = os.path.join(root_dir, 'mean_std_cache.p')
    if not os.path.isfile(mean_std_cache):
        print('computing mean std from train data...')
        mean, std = np.array(0), np.array(0)
        for i in range (1, 2384):
            if i % 100 == 0:
                print('{} of {}'.format(i, 2383))
            mesh = Mesh(file=root_dir+'/mesh_f_3000/'+str(i).zfill(5)+'.obj')
            edge_features = mesh.extract_features()
            edge_features = pad(edge_features, mesh.edges_count)
            mean = mean + edge_features.mean(axis=1)
            std = std + edge_features.std(axis=1)
        mean = mean / (i + 1)
        std = std / (i + 1)
        transform_dict = {'mean': mean[:, np.newaxis], 'std': std[:, np.newaxis],
                          'ninput_channels': len(mean)}
        with open(mean_std_cache, 'wb') as f:
            pickle.dump(transform_dict, f)
        print('saved: ', mean_std_cache)

    # open mean / std from file
    with open(mean_std_cache, 'rb') as f:
        transform_dict = pickle.load(f)
        print('loaded mean / std from cache')
        mean = transform_dict['mean']
        std = transform_dict['std']
        ninput_channels = transform_dict['ninput_channels']

    return mean, std, ninput_channels

if __name__ == "__main__":
    _, _, _ = get_mean_std('/data/jionkim/gt_NDC_KISTI_SDF_npy')