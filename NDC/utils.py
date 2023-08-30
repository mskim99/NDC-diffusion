import numpy as np

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


def read_data_input_only(data_dir, file_name, grid_size):
    LOD_gt_float = np.zeros([grid_size+1,grid_size+1,grid_size+1,3],np.float32)
    LOD_input = np.load(data_dir + '/res_' + str(grid_size) + '/' + file_name + '.npy')
    LOD_input = LOD_input*grid_size # denormalize
    return LOD_gt_float, LOD_input


def read_data(data_dir, file_name, grid_size):
    LOD_gt_float = np.load(data_dir + '/res_' + str(grid_size) + '_float/' + file_name + '.npy')
    LOD_input = np.load(data_dir + '/res_' + str(grid_size) + '/' + file_name + '.npy')
    LOD_input = LOD_input*grid_size # denormalize
    return LOD_gt_float, LOD_input


def read_and_augment_data_ndc(data_dir, file_name, grid_size, aug_permutation=True, aug_reversal=True, aug_inversion=True):
    grid_size_1 = grid_size+1

    # read input hdf5
    LOD_gt_float, LOD_input = read_data(data_dir, file_name, grid_size)

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

    return LOD_gt_float, LOD_input


def write_obj_triangle(name, vertices, triangles):
    fout = open(name, 'w')
    for ii in range(len(vertices)):
        fout.write("v "+str(vertices[ii,0])+" "+str(vertices[ii,1])+" "+str(vertices[ii,2])+"\n")
    for ii in range(len(triangles)):
        fout.write("f "+str(int(triangles[ii,0]+1))+" "+str(int(triangles[ii,1]+1))+" "+str(int(triangles[ii,2]+1))+"\n")
    fout.close()