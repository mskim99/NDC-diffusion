import numpy as np
import vtk
from vtk.util import numpy_support

'''
img = np.load('J:/Program/NDC-main/data_preprocessing/gt_NDC_KISTI_SDF_p_100_npy/res_64/00001.npy')
# img = np.load('J:/Program/NDC-diffusion/output/pred_output_float.npy')
print(img.shape)
img_data = img[0:64, 0:64, 0:64]
# img_data = img[0, 0, 0:32, 0:32, 0:32]

imdata = vtk.vtkImageData()

depthArray = numpy_support.numpy_to_vtk(img_data.ravel(order='F'), deep=True, array_type=vtk.VTK_FLOAT)

imdata.SetDimensions([64, 64, 64])
# fill the vtk image data object
imdata.SetSpacing([1, 1, 1])
imdata.SetOrigin([0, 0, 0])
imdata.GetPointData().SetScalars(depthArray)

writer = vtk.vtkMetaImageWriter()
writer.SetFileName('J:/Program/NDC-main/data_preprocessing/gt_NDC_KISTI_SDF_p_100_npy/res_64/00001.mha')
# writer.SetFileName('J:/Program/NDC-diffusion/output/pred_output_float.mha')
writer.SetInputData(imdata)
writer.Write()
'''

for i in range (0, 100):
    img = np.load('J:/Program/NDC-main/data_preprocessing/gt_NDC_KISTI_SDF_p_100_npy/res_64_int/00001.npy')
    print(img.min())
    print(img.max())