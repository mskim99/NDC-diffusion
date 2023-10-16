import torch
import os
import numpy as np

import open3d as o3d

import sys
sys.path.append('../')
from NDC.cutils import dual_contouring_ndc
from mesh.mesh import Mesh

# read sdf files produced by SDFGen
def read_sdf_file_as_3d_array(name):
    fp = open(name, 'rb')
    line = fp.readline().strip()
    if not line.startswith(b'#sdf'):
        raise IOError('Not a sdf file')
    dims = list(map(int, fp.readline().strip().split(b' ')[1:]))
    line = fp.readline()
    data = np.frombuffer(fp.read(), dtype=np.float32)
    data = data.reshape(dims)
    fp.close()
    return data


def read_data_input_only(data_dir, file_name, grid_size, train):
    if not train:
        LOD_gt_int = np.zeros([grid_size+1,grid_size+1,grid_size+1,3], np.int32)
    else:
        LOD_gt_int = None
    LOD_gt_float = np.zeros([grid_size+1,grid_size+1,grid_size+1,3],np.float32)
    LOD_input = np.load(data_dir + '/res_' + str(grid_size) + '/' + file_name + '.npy')
    LOD_input = LOD_input*grid_size # denormalize
    return LOD_gt_int, LOD_gt_float, LOD_input


def read_data(data_dir, file_name, grid_size, train):
    if not train:
        LOD_gt_int = np.load(data_dir + '/res_' + str(grid_size) + '_int/' + file_name + '.npy')
    else:
        LOD_gt_int = None
    LOD_gt_float = np.load(data_dir + '/res_' + str(grid_size) + '_float/' + file_name + '.npy')
    LOD_input = np.load(data_dir + '/res_' + str(grid_size) + '/' + file_name + '.npy')
    LOD_input = LOD_input*grid_size # denormalize
    return LOD_gt_int, LOD_gt_float, LOD_input


def read_and_augment_data_ndc(data_dir, file_name, grid_size, train, aug_permutation=True, aug_reversal=True, aug_inversion=True):
    grid_size_1 = grid_size+1

    # read input hdf5
    LOD_gt_int, LOD_gt_float, LOD_input = read_data(data_dir, file_name, grid_size, train)

    newdict = {}

    newdict['float_center_x_'] = LOD_gt_float[:-1,:-1,:-1,0]
    newdict['float_center_y_'] = LOD_gt_float[:-1,:-1,:-1,1]
    newdict['float_center_z_'] = LOD_gt_float[:-1,:-1,:-1,2]

    newdict['input_sdf'] = LOD_input[:,:,:]

    # augment data
    permutation_list = [ [0,1,2], [0,2,1], [1,0,2], [1,2,0], [2,0,1], [2,1,0] ]
    reversal_list = [ [0,0,0],[0,0,1],[0,1,0],[0,1,1], [1,0,0],[1,0,1],[1,1,0],[1,1,1] ]
    if aug_permutation:
        permutation = permutation_list[np.random.randint(len(permutation_list))]
    else:
        permutation = permutation_list[0]
    if aug_reversal:
        reversal = reversal_list[np.random.randint(len(reversal_list))]
    else:
        reversal = reversal_list[0]
    if aug_inversion:
        inversion_flag = np.random.randint(2)
    else:
        inversion_flag = 0

    if reversal[0]:
        for k in newdict: # reverse
            newdict[k] = newdict[k][::-1,:,:]
        k = 'float_center_x_'
        mask = (newdict[k]>=0)
        newdict[k] = newdict[k]*(1-mask)+(1-newdict[k])*mask
    if reversal[1]:
        for k in newdict: # reverse
            newdict[k] = newdict[k][:,::-1,:]
        k = 'float_center_y_'
        mask = (newdict[k]>=0)
        newdict[k] = newdict[k]*(1-mask)+(1-newdict[k])*mask
    if reversal[2]:
        for k in newdict: # reverse
            newdict[k] = newdict[k][:,:,::-1]
        k = 'float_center_z_'
        mask = (newdict[k]>=0)
        newdict[k] = newdict[k]*(1-mask)+(1-newdict[k])*mask

    if permutation == [0,1,2]:
        pass
    else:
        for k in newdict: #transpose
            newdict[k] = np.transpose(newdict[k], permutation)

        olddict = newdict
        newdict = {}
        newdict['input_sdf'] = olddict['input_sdf']

        if permutation == [0,2,1]:
            newdict['float_center_x_'] = olddict['float_center_x_']
            newdict['float_center_y_'] = olddict['float_center_z_']
            newdict['float_center_z_'] = olddict['float_center_y_']
        elif permutation == [1,0,2]:
            newdict['float_center_x_'] = olddict['float_center_y_']
            newdict['float_center_y_'] = olddict['float_center_x_']
            newdict['float_center_z_'] = olddict['float_center_z_']
        elif permutation == [2,1,0]:
            newdict['float_center_x_'] = olddict['float_center_z_']
            newdict['float_center_y_'] = olddict['float_center_y_']
            newdict['float_center_z_'] = olddict['float_center_x_']
        elif permutation == [1,2,0]:
            newdict['float_center_x_'] = olddict['float_center_y_']
            newdict['float_center_y_'] = olddict['float_center_z_']
            newdict['float_center_z_'] = olddict['float_center_x_']
        elif permutation == [2,0,1]:
            newdict['float_center_x_'] = olddict['float_center_z_']
            newdict['float_center_y_'] = olddict['float_center_x_']
            newdict['float_center_z_'] = olddict['float_center_y_']

    # store outputs
    LOD_gt_float = np.full([grid_size_1,grid_size_1,grid_size_1,3], -1, np.float32)
    LOD_gt_float[:-1,:-1,:-1,0] = newdict['float_center_x_']
    LOD_gt_float[:-1,:-1,:-1,1] = newdict['float_center_y_']
    LOD_gt_float[:-1,:-1,:-1,2] = newdict['float_center_z_']

    LOD_input = np.ones([grid_size_1,grid_size_1,grid_size_1], np.float32)
    LOD_input[:,:,:] = newdict['input_sdf']
    if inversion_flag:
        LOD_input = -LOD_input

    return LOD_gt_int, LOD_gt_float, LOD_input


# this is not an efficient implementation. just for testing!
def dual_contouring_ndc_test(int_grid, float_grid):
    all_vertices = []
    all_triangles = []

    int_grid = np.squeeze(int_grid)
    dimx, dimy, dimz = int_grid.shape
    vertices_grid = np.full([dimx, dimy, dimz], -1, np.int32)

    # all vertices
    for i in range(0, dimx - 1):
        for j in range(0, dimy - 1):
            for k in range(0, dimz - 1):

                v0 = int_grid[i, j, k]
                v1 = int_grid[i + 1, j, k]
                v2 = int_grid[i + 1, j + 1, k]
                v3 = int_grid[i, j + 1, k]
                v4 = int_grid[i, j, k + 1]
                v5 = int_grid[i + 1, j, k + 1]
                v6 = int_grid[i + 1, j + 1, k + 1]
                v7 = int_grid[i, j + 1, k + 1]

                if v1 != v0 or v2 != v0 or v3 != v0 or v4 != v0 or v5 != v0 or v6 != v0 or v7 != v0:
                    # add a vertex
                    vertices_grid[i, j, k] = len(all_vertices)
                    pos = float_grid[i, j, k] + np.array([i, j, k], np.float32)
                    all_vertices.append(pos)

    all_vertices = np.array(all_vertices, np.float32)

    # all triangles

    # i-direction
    for i in range(0, dimx - 1):
        for j in range(1, dimy - 1):
            for k in range(1, dimz - 1):
                v0 = int_grid[i, j, k]
                v1 = int_grid[i + 1, j, k]
                if v0 != v1:
                    if v0 == 0:
                        all_triangles.append(
                            [vertices_grid[i, j - 1, k - 1], vertices_grid[i, j, k], vertices_grid[i, j, k - 1]])
                        all_triangles.append(
                            [vertices_grid[i, j - 1, k - 1], vertices_grid[i, j - 1, k], vertices_grid[i, j, k]])
                    else:
                        all_triangles.append(
                            [vertices_grid[i, j - 1, k - 1], vertices_grid[i, j, k - 1], vertices_grid[i, j, k]])
                        all_triangles.append(
                            [vertices_grid[i, j - 1, k - 1], vertices_grid[i, j, k], vertices_grid[i, j - 1, k]])

    # j-direction
    for i in range(1, dimx - 1):
        for j in range(0, dimy - 1):
            for k in range(1, dimz - 1):
                v0 = int_grid[i, j, k]
                v1 = int_grid[i, j + 1, k]
                if v0 != v1:
                    if v0 == 0:
                        all_triangles.append(
                            [vertices_grid[i - 1, j, k - 1], vertices_grid[i, j, k - 1], vertices_grid[i, j, k]])
                        all_triangles.append(
                            [vertices_grid[i - 1, j, k - 1], vertices_grid[i, j, k], vertices_grid[i - 1, j, k]])
                    else:
                        all_triangles.append(
                            [vertices_grid[i - 1, j, k - 1], vertices_grid[i, j, k], vertices_grid[i, j, k - 1]])
                        all_triangles.append(
                            [vertices_grid[i - 1, j, k - 1], vertices_grid[i - 1, j, k], vertices_grid[i, j, k]])

    # k-direction
    for i in range(1, dimx - 1):
        for j in range(1, dimy - 1):
            for k in range(0, dimz - 1):
                v0 = int_grid[i, j, k]
                v1 = int_grid[i, j, k + 1]
                if v0 != v1:
                    if v0 == 0:
                        all_triangles.append(
                            [vertices_grid[i - 1, j - 1, k], vertices_grid[i - 1, j, k], vertices_grid[i, j, k]])
                        all_triangles.append(
                            [vertices_grid[i - 1, j - 1, k], vertices_grid[i, j, k], vertices_grid[i, j - 1, k]])
                    else:
                        all_triangles.append(
                            [vertices_grid[i - 1, j - 1, k], vertices_grid[i, j, k], vertices_grid[i - 1, j, k]])
                        all_triangles.append(
                            [vertices_grid[i - 1, j - 1, k], vertices_grid[i, j - 1, k], vertices_grid[i, j, k]])

    all_triangles = np.array(all_triangles, np.int32)

    return all_vertices, all_triangles


def write_obj_triangle(name, vertices, triangles):
    fout = open(name, 'w')
    for ii in range(len(vertices)):
        fout.write("v "+str(vertices[ii,0])+" "+str(vertices[ii,1])+" "+str(vertices[ii,2])+"\n")
    for ii in range(len(triangles)):
        fout.write("f "+str(int(triangles[ii,0]+1))+" "+str(int(triangles[ii,1]+1))+" "+str(int(triangles[ii,2]+1))+"\n")
    fout.close()

'''
def gen_mesh(ndf_network, sdf_gen, receptive_padding):
    with torch.no_grad():
        pred_output_float = ndf_network(sdf_gen)
        gt_input_numpy = sdf_gen[0, 0, receptive_padding:-receptive_padding, receptive_padding:-receptive_padding,
                         receptive_padding:-receptive_padding].detach().cpu().numpy()
        pred_output_bool_numpy = np.expand_dims((gt_input_numpy < 0).astype(np.int32), axis=3)
        pred_output_float_numpy = np.transpose(pred_output_float[0].detach().cpu().numpy(), [1, 2, 3, 0])
        pred_output_float_numpy = np.clip(pred_output_float_numpy, 0, 1)
        vertices, triangles = dual_contouring_ndc(np.ascontiguousarray(pred_output_bool_numpy, np.int32),
                                                         np.ascontiguousarray(pred_output_float_numpy, np.float32))

        # Generate Mesh
        o3d_vertices = o3d.utility.Vector3dVector(vertices)
        o3d_triangles = o3d.utility.Vector3iVector(triangles)
        mesh_np = o3d.geometry.TriangleMesh(o3d_vertices, o3d_triangles)
        mesh_np_s = o3d.geometry.TriangleMesh.simplify_quadric_decimation(mesh_np, 1500)
        o3d.io.write_triangle_mesh("./gen_write.obj", mesh_np_s)
        # write_obj_triangle("./gen_write.obj", vertices, triangles)
        mesh = Mesh(file="./gen_write.obj")
        # os.remove("./gen_write.obj")

        return vertices, triangles, mesh
        '''

def gen_mesh(ndf_network, sdf_gen, gt_output_bool, gt_output_float):

    with torch.no_grad():
        pred_output_float = ndf_network(sdf_gen)
        pred_output_float_numpy = pred_output_float.detach().cpu().numpy()

        gt_output_bool_numpy = gt_output_bool.detach().cpu().numpy()
        pred_output_bool_numpy = np.transpose(gt_output_bool_numpy[0], [1, 2, 3, 0])
        pred_output_bool_numpy = (pred_output_bool_numpy > 0.5).astype(np.int32)

        pred_output_float_numpy = np.transpose(pred_output_float_numpy[0], [1, 2, 3, 0])

    pred_output_float_numpy = np.clip(pred_output_float_numpy, 0, 1)
    vertices, triangles = dual_contouring_ndc(np.ascontiguousarray(pred_output_bool_numpy, np.int32),
                                                         np.ascontiguousarray(pred_output_float_numpy, np.float32))
    write_obj_triangle('./gen_write.obj', vertices, triangles)
    mesh = Mesh(file="./gen_write.obj")

    return vertices, triangles, mesh