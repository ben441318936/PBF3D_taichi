from pyqtgraph.opengl import GLViewWidget, MeshData
from pyqtgraph.opengl.items.GLMeshItem import GLMeshItem

import numpy as np

from PyQt5.QtGui import QApplication

app = QApplication([])
view = GLViewWidget()

exp = 3
prefix = "./meshing/exp{}/".format(exp)

vertices = np.load(prefix+"vertices.npy")
normals = np.load(prefix+"normals.npy")
faces = np.load(prefix+"faces.npy")

mesh = MeshData(vertices / 100, faces)  # scale down - because camera is at a fixed position 
# or mesh = MeshData(smooth_vertices / 100, faces)
mesh._vertexNormals = normals
# or mesh._vertexNormals = smooth_normals

item = GLMeshItem(meshdata=mesh, color=[1, 0, 0, 1], shader="normalColor")

view.addItem(item)
view.orbit(-90,0)


view.show()
app.exec_()