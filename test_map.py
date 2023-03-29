import pyvista as pv
import numpy as np

tex = pv.read_texture("generated_texture4.png")
# create a structured surface
x = np.arange(-10, 10, 0.25)
y = np.arange(-10, 10, 0.25)
x, y = np.meshgrid(x, y)
# r = np.sqrt(x**2 + y**2)
z = np.ones((x.shape[0],x.shape[1])) * np.random.rand(1)
curvsurf = pv.StructuredGrid(x, y, z)

# Map the curved surface to a plane - use best fitting plane
curvsurf.texture_map_to_plane(inplace=True)

curvsurf.plot(texture=tex)