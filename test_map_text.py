import pyvista as pv
import numpy as np

# Prepare the input data
x = np.linspace(-10, 10, 40)
y = np.linspace(-10, 10, 40)
z = np.linspace(-1, 1, 40)

xx1, yy1, zz1 = np.meshgrid(x, y, z)
points = np.c_[xx1.reshape(-1), yy1.reshape(-1), zz1.reshape(-1)]

cloud = pv.PolyData(points)
surf = cloud.delaunay_2d(inplace=True)
pp = surf.points.astype(np.double)

# Load the texture image (replace with your image path)
texture = pv.read_texture("generated_texture6.png")

# Compute the maximum area plane
# Find unique x and y values
unique_x = np.unique(pp[:, 0])
unique_y = np.unique(pp[:, 1])
unique_z = np.unique(pp[:, 2])

# Get min and max x values and min y value
min_x = np.min(unique_x)
max_x = np.max(unique_x)
min_y = np.min(unique_y)
max_y = np.max(unique_y)

max_z = np.max(unique_z)
# Define a small tolerance to handle floating-point inaccuracies
tol = 1e-8

# Find bottom left corner point (min_x, min_y)
# bottom_left_corner = pp[np.where(np.isclose(pp[:, 0], min_x, atol=tol) & np.isclose(pp[:, 1], min_y, atol=tol) & np.isclose(pp[:, 2], max_z, atol=tol))][0]

# # Find bottom right corner point (max_x, min_y)
# bottom_right_corner = pp[np.where(np.isclose(pp[:, 0], max_x, atol=tol) & np.isclose(pp[:, 1], min_y, atol=tol) & np.isclose(pp[:, 2], max_z, atol=tol))][0]

# # Find top left corner point (min_x, any value greater than min_y)
# top_left_corner = pp[np.where(np.isclose(pp[:, 0], min_x, atol=tol) & ~np.isclose(pp[:, 1], max_y, atol=tol) & np.isclose(pp[:, 2], max_z, atol=tol))][0]

bottom_left_corner = [-10, -10, 1]
bottom_right_corner = [10, -10, 1]
top_left_corner = [-10, 10, 1]

print(f"Bottom Left Corner: {bottom_left_corner}")
print(f"Bottom Right Corner: {bottom_right_corner}")
print(f"Top Left Corner: {top_left_corner}")


# Map the texture to the plane
textured_surf = surf.texture_map_to_plane(origin=bottom_left_corner, point_u=bottom_right_corner, point_v=top_left_corner)
textured_surf.textures["my_texture"] = texture

# Visualize the textured plane
plotter = pv.Plotter()
plotter.add_mesh(textured_surf, texture="my_texture")
# plotter.add_mesh(surf)
plotter.show()
