import numpy as np
import vtk
from vtk.util import numpy_support

img = np.load('J:/Program/NDC-diffusion/output/ddpm/output.npy')
img_data = img

imdata = vtk.vtkImageData()

depthArray = numpy_support.numpy_to_vtk(img_data.ravel(order='F'), deep=True, array_type=vtk.VTK_FLOAT)

imdata.SetDimensions([64, 64, 64])
# fill the vtk image data object
imdata.SetSpacing([1, 1, 1])
imdata.SetOrigin([0, 0, 0])
imdata.GetPointData().SetScalars(depthArray)

writer = vtk.vtkMetaImageWriter()
writer.SetFileName('J:/Program/NDC-diffusion/output/ddpm/output.mha')
writer.SetInputData(imdata)
writer.Write()
