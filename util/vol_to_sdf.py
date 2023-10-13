import numpy as np
import threading

# TEST

np_64 = np.load('J:/Program/NDC-main/data_preprocessing/gt_NDC_KISTI_SDF_p_npy/res_64_bd/00001.npy')
idx_64 = np.mgrid[0:65, 0:65, 0:65].reshape(3,-1).T
sdf_64 = np.zeros(np_64.shape)
bd_list = list()

dist_max = 65.

def calc_dist_np(idx_64):
    dist = np.subtract(bd_list, idx_64)
    dist = np.power(dist, 2)
    dist = np.sum(dist, axis=1)
    dist = np.sqrt(dist)
    dist = dist.min()
    dist = dist / dist_max

bd_list = np.where(np_64 > 0.)

bd_list = np.array(bd_list)
bd_list = np.squeeze(bd_list)
bd_list = bd_list.transpose()

dist = calc_dist_np(idx_64)

np.save('J:/Program/NDC-main/data_preprocessing/gt_NDC_KISTI_SDF_p_npy/res_64_bd/00001_sdf.npy', sdf_64)