# import open3d as o3d
# import numpy as np
# import matplotlib.pyplot as plt

# def plane_segmentation_mesh(mesh, distance_threshold, ransac_n, dbscan_eps, dbscan_min_samples):
#     # Compute vertex normals if not already available
#     if not mesh.has_vertex_normals():
#         mesh.compute_vertex_normals()

#     # Convert mesh to point cloud
#     pcd = mesh.sample_points_uniformly(number_of_points=len(mesh.vertices))

#     # Set point cloud normals
#     pcd.normals = mesh.vertex_normals

#     # Apply RANSAC for plane detection
#     plane_model, inliers = pcd.segment_plane(distance_threshold=distance_threshold,
#                                              ransac_n=ransac_n,
#                                              num_iterations=1000)

#     # Perform DBSCAN clustering on inliers
#     labels = np.array(pcd.cluster_dbscan(eps=dbscan_eps, min_points=dbscan_min_samples, print_progress=True))

#     # Visualize the results
#     max_label = labels.max()
#     colors = plt.get_cmap("tab20")(labels / (max_label if max_label > 0 else 1))
#     colors[labels < 0] = 0
#     pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])
#     o3d.visualization.draw_geometries([pcd])

# # Load a sample mesh and test the plane_segmentation_mesh function
# mesh = o3d.io.read_triangle_mesh("./data/user_input/textured_output.obj")
# plane_segmentation_mesh(mesh, distance_threshold=0.02, ransac_n=3, dbscan_eps=0.02, dbscan_min_samples=10)




import open3d as o3d
import numpy as np

def plane_segmentation(points, distance_threshold, ransac_n, dbscan_eps, dbscan_min_points):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    # Apply RANSAC for plane detection
    plane_model, inliers = pcd.segment_plane(distance_threshold=distance_threshold,
                                             ransac_n=ransac_n,
                                             num_iterations=1000)
    inlier_cloud = pcd.select_by_index(inliers)

    # Perform DBSCAN clustering on inliers
    labels = inlier_cloud.cluster_dbscan(eps=dbscan_eps, min_points=dbscan_min_points, print_progress=True)

    # Visualize the results
    colors = np.zeros((len(points), 3))
    for i, label in enumerate(labels):
        if label >= 0:
            colors[inliers[i]] = np.array([0, 1, 0])

    pcd.colors = o3d.utility.Vector3dVector(colors)
    o3d.visualization.draw_geometries([pcd])

# Test the plane_segmentation function with sample point cloud data
points = np.random.rand(1000, 3)
plane_segmentation(points, distance_threshold=0.01, ransac_n=3, dbscan_eps=0.02, dbscan_min_points=10)
