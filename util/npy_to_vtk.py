import numpy as np
import vtk
from vtk.util import numpy_support

img = np.load('J:/Program/NDC-main/data_preprocessing/gt_NDC_KISTI_SDF_p_100_npy/res_64_float/00001.npy')
img_data = img

imdata = vtk.vtkImageData()

depthArray = numpy_support.numpy_to_vtk(img_data.ravel(order='F'), deep=True, array_type=vtk.VTK_FLOAT)

imdata.SetDimensions([64, 64, 64])
# fill the vtk image data object
imdata.SetSpacing([1, 1, 1])
imdata.SetOrigin([0, 0, 0])
imdata.GetPointData().SetScalars(depthArray)

writer = vtk.vtkMetaImageWriter()
writer.SetFileName('J:/Program/NDC-main/data_preprocessing/gt_NDC_KISTI_SDF_p_100_npy/res_64_float/00001.mha')
writer.SetInputData(imdata)
writer.Write()
