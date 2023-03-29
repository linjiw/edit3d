import imp
from cv2 import inpaint
import numpy
import open3d as o3d
import numpy as np
import json
import matplotlib.pyplot as plt
import cv2
import open3d.visualization.rendering as rendering
import time
import copy
import pyvista as pv
from pyvista import examples

from pykdtree.kdtree import KDTree

# from importlib.resources import path

class Inpaint3D():
    # load obj, plane segementation, find_related_idx
    def __init__(self,objname) -> None:
        self.objname = objname
        self.pvmesh = None
        self.textured_mesh = None
        self.pcd = None
        self.segments = None
        self.face = None
        # self.load_obj()
        pass
    def load_obj(self):
        self.pvmesh = pv.PolyData(self.objname)
        self.textured_mesh = o3d.io.read_triangle_mesh(self.objname)
        self.textured_mesh.compute_vertex_normals()
        self.pcd = o3d.geometry.PointCloud()
        self.pcd.points = self.textured_mesh.vertices
        self.pcd.colors = self.textured_mesh.vertex_colors
        self.pcd.normals = self.textured_mesh.vertex_normals
        self.face =  self.pvmesh.faces.reshape(-1,4)

        pass
    def plane_segementation(self, max_plane_idx = 40, d_threshold= 0.01 ):
        segment_models={}
        segments={}
        rest=self.pcd
        global_idx = {}
        for i in range(max_plane_idx):
            sgmt_idxs = []
            colors = plt.get_cmap("tab20")(i)
            segment_models[i], inliers = rest.segment_plane(distance_threshold = 0.02,ransac_n = 3,num_iterations=1000)
            global_idx[i] = np.array(inliers)
            segments[i]=rest.select_by_index(inliers)
            labels = np.array(segments[i].cluster_dbscan(eps=d_threshold * 10, min_points=10))
            candidates = [len(np.where(labels == j)[0]) for j in np.unique(labels)]
            best_candidate = int(np.unique(labels)[np.where(candidates == np.max(candidates))[0]])
            rest = rest.select_by_index(inliers, invert=True) + segments[i].select_by_index(
            list(np.where(labels != best_candidate)[0]))
    
            segments[i] = segments[i].select_by_index(list(np.where(labels == best_candidate)[0]))

            segments[i].paint_uniform_color(list(colors[:3]))
            rest.paint_uniform_color([0.6, 0.6, 0.6])
            # print(f"len(segments[i].points) {len(segments[i].points)}")
        self.segments = segments
        return segments
        pass
    
    def locate_plane(self):
        #
        pass
    def path_pick(self):

        pass
    def plane_function(self):
        #calculate plane function with most inliner points
        pass
    def create_vertice(self):
        pass
    def findallidx(self, ids):
        ans = np.empty((0),dtype=int)
        for i in ids:
            result = np.argwhere(self.face==i)[:,0].ravel()
            # print(result)
            ans = numpy.concatenate((ans,result))
            # print(ans)
        return numpy.unique(ans) 
def get_center_area(points):
    # distance = []
    ratio = 0.2
    distance = np.ptp(points, axis=0)
    mid = np.median(points,axis=0)
    
    # for i in range(2):
    #     distance = np.ptp(points, axis=i)
    #     mid = np.median(points,axis=i)
    #     print(f"{i} distance {distance}")
    #     print(f"{i} mid {mid}")
    # # ans = points[points[0]>distance]
    # dim = points.shape
def interp(ids,thre=10):
    newids = copy.deepcopy(ids)
    prev = ids[0]
    for ii, i  in enumerate(ids):
        if ii>0:
            diff = i -prev
            if diff>1 and diff <thre:
                # print(f"diff {diff}, {i} {prev}")
                t = prev
                kk = ii
                for j in range(diff-1):
                    t+=1
                    newids.insert(kk,t)
                    kk +=1
            prev = i
    return newids


# def pick_path():
    # pl = pv.Plotter()
    # pl.add_mesh(sfc,texture=tex1)
# def plane_function()




if __name__ == "__main__":
    stored_points =[]
    def callback(point):
        """Create a cube and a label at the click point."""
        mesh = pv.Cube(center=point, x_length=0.05, y_length=0.05, z_length=0.05)
        pl.add_mesh(mesh, style='wireframe', color='r')
        pl.add_point_labels(point, [f"{point[0]:.2f}, {point[1]:.2f}, {point[2]:.2f}"])
        stored_points.append(point)
    data = './data/'
    inpaint = Inpaint3D(f"{data}user_input/textured_output.obj")
    inpaint.load_obj()
    inpaint.plane_segementation()

    # pv is used for visualization
    tex1 = pv.read_texture(f"{data}textured_output.jpg")


    

    """Test Path Picking"""
    # pl = pv.Plotter()
    # pl.add_mesh(inpaint.pvmesh,texture=tex1)
    # pl.enable_path_picking()
    # pl.show()
    # path_points = pl.picked_path.points.astype(np.double) # savev picked path
    # id_for_points = np.zeros((path_points.shape[0],4))
    # id_for_points[:,:-1] = path_points
    # print(f"inpaint.segments {inpaint.segments}")
    # for ii, i in enumerate(path_points):
    #     for jj in range(40):
    #         # print(f"{jj}, {j}")
    #         if i in np.asarray(inpaint.segments[jj].points):
    #             print('--------------------------------')
    #             id_for_points[ii,3] = jj
    # print(id_for_points)

    """kd_tree is used to find all related trainagles and indexs - To make it more complete """
    seg_id = 0
    kd_tree = KDTree(inpaint.pvmesh.points.astype(np.double)) 
    points_seg_1 = np.asarray(inpaint.segments[seg_id].points)
    # o3d.visualization.draw_geometries([inpaint.segments[0]])
    # idx = inpaint.pvmesh.find_closest_cell(points_seg_1)
    dist, idx = kd_tree.query(points_seg_1,k=5)
    allidx = np.unique(idx)
    ids = inpaint.findallidx(allidx)
    newid = interp(list(ids),thre=1)
    mesh2 = inpaint.pvmesh.extract_cells(newid)
    sfc = mesh2.extract_surface()

    """sfc is the extracted first segment plane"""
    sfc.compute_normals(cell_normals=True, point_normals=False, inplace=True)
    print(f"sfc {sfc}")
    avg_norm = np.mean(sfc['Normals'],axis=0)
    print(f"avg_norm {avg_norm}")
    tex1 = pv.read_texture(f"{data}textured_output.jpg")


 

    # tex1 = pv.read_texture("textured_output.jpg")
    """Pick 4 Points for a rectangle area"""

    pl = pv.Plotter()
    pl.add_mesh(sfc,texture=tex1)
    pl.enable_point_picking(callback=callback, left_clicking=True, show_point=False)
    # a = []
    # a.append()
    # print

    """
    plane function: Use avg_norm and points to create plane function
    -> interpolate: interpolate a squared area in four border points
    -> create vertices: 
    """
    

    tmp = avg_norm.ravel()
    pl.show()

    pl = pv.Plotter()
    pl.add_mesh(sfc,texture=tex1)
    pl.camera.enable_parallel_projection() # orthogonal projection is used to transfer 3D to 2D using avg_norm
    pl.camera.position =  tmp
    pl.set_background('white')

    pl.show(screenshot=f"original_1.png")
    pl = pv.Plotter()
    pl.add_mesh(sfc,texture=tex1,opacity=0.0)
    pl.camera.enable_parallel_projection()
    pl.camera.position =  tmp
    stored_point_np = np.array(stored_points)
    # stored_point_np = np.append(stored_point_np,stored_points[0])
    rectangle = pv.Rectangle([stored_point_np[0], stored_point_np[1], stored_point_np[2], stored_point_np[3]])
    # pl.add_lines(stored_point_np,color='yellow', width=5)
    pl.add_mesh(rectangle,color='black')
    # pl.show()
    
    
    pl.set_background('white')

    pl.show(screenshot=f"yellow_1.png")
    # used the stored points to generate mask


    print(f"stored_point_np\n {stored_point_np}")

    c = avg_norm[0] *stored_point_np[0,0] + avg_norm[1] *stored_point_np[0,1] + avg_norm[2] *stored_point_np[0,2] 

    # build a rectangular points set with stored points
    min_p = np.min(stored_point_np,axis=0)
    max_p = np.max(stored_point_np,axis=0)
    std = np.std(stored_point_np,axis=0)
    print(f"std {std}")
    print(f"min_p.shape {min_p.shape}")

    n = 20
    # + np.random.uniform(-5, 5, size=n)
    # + np.random.uniform(-std[0], std[0], size=n)
    """create clip mesh"""
    # avg_norm
    # create cube
    x_center = np.mean([min_p[0],max_p[0]])
    y_center = np.mean([min_p[1],max_p[1]])

    z_center = np.mean([min_p[2],max_p[2]])
    x_len =  max_p[0] - min_p[0] 
    y_len =  max_p[1] - min_p[1] 
    z_len =  max_p[2] - min_p[2] 
    clip_cube = pv.Cube(center=(x_center,y_center,z_center),x_length=x_len,y_length=y_len,z_length=z_len)
    select = inpaint.pvmesh.select_enclosed_points(clip_cube)
    reduced_sphere, ridx = inpaint.pvmesh.remove_points(select['SelectedPoints'].view(bool))
                        #    adjacent_cells=False)
    a = inpaint.pvmesh.extract_surface(select['SelectedPoints'].view(bool))
    # # clipped_mesh = inpaint.pvmesh.clip_box(clip_cube,invert=False)
    # # clip_area = [min_p[0],max_p[0],min_p[1],max_p[1],min_p[2],max_p[2]]
    # # clipped_mesh = inpaint.pvmesh.clip_box(clip_area,invert=False)
    pl=pv.Plotter()
    # pl.add_mesh(select,color="red")
    # pl.add_mesh(clip_cube,color="blue")
    # pl.add_mesh(reduced_sphere, color="blue")
    pl.add_mesh(a)
    # # pl.add_mesh(inpaint.pvmesh,texture=tex1)
    # print(f"show clipped")
    pl.show()

    

    # cube_center = 

    x = np.linspace(min_p[0], max_p[0], num=n) 
    y = np.linspace(min_p[1], max_p[1], num=n)
    z = np.linspace(min_p[2], max_p[2], num=n)
    xx1,yy1,zz1 = np.meshgrid(x, y, z)

    # calculate plane function and learn a gaussian distribution


    # project_points_to_plane

    points = np.c_[xx1.reshape(-1), yy1.reshape(-1), zz1.reshape(-1)]
    # points[0:5, :]
    print(f"points.shape {points.shape}")
    cloud  = pv.PolyData(points)
    # cloud = pv.StructuredGrid(xx1,yy1,zz1)
    # cloud.plot(point_size=15)
    cloud.plot()
    # surf = cloud
    surf = cloud.delaunay_2d(inplace=True) # create surface plane with plane pointcloud
    pp = surf.points.astype(np.double)
    print(pp)
    tex = pv.read_texture("generated_texture3.png")
    # tex = examples.download_puppy_texture()
    # axial_num_puppies = 4
    # xc = np.linspace(0, axial_num_puppies, surf.dimensions[0])
    # yc = np.linspace(0, axial_num_puppies, surf.dimensions[1])
    # zc = np.linspace(0, axial_num_puppies, surf.dimensions[2])
    # xxc, yyc, zzc = np.meshgrid(xc, yc,zc)
    # puppy_coords = np.c_[yyc.ravel(), xxc.ravel(),zzc.ravel()]
    # surf.active_t_coords = puppy_coords

    # surf.texture_map_to_plane(origin=pp[-5,:],point_u=pp[0,:],point_v=pp[-1,:],inplace=True)
    surf.texture_map_to_plane(origin=pp[-20,:],point_u=pp[0,:],point_v=pp[-1,:],inplace=True)
    surf.plot(show_edges=False,texture=tex)


    # """Run again"""
    # stored_points =[]
    # seg_id = 1
    # kd_tree = KDTree(inpaint.pvmesh.points.astype(np.double)) 
    # points_seg_1 = np.asarray(inpaint.segments[seg_id].points)
    # # o3d.visualization.draw_geometries([inpaint.segments[0]])
    # # idx = inpaint.pvmesh.find_closest_cell(points_seg_1)
    # dist, idx = kd_tree.query(points_seg_1,k=5)
    # allidx = np.unique(idx)
    # ids = inpaint.findallidx(allidx)
    # newid = interp(list(ids),thre=1)
    # mesh2 = inpaint.pvmesh.extract_cells(newid)
    # sfc = mesh2.extract_surface()

    # """sfc is the extracted first segment plane"""
    # sfc.compute_normals(cell_normals=True, point_normals=False, inplace=True)
    # print(f"sfc {sfc}")
    # avg_norm = np.mean(sfc['Normals'],axis=0)
    # print(f"avg_norm {avg_norm}")
    # tex1 = pv.read_texture(f"{data}textured_output.jpg")

    # # tex1 = pv.read_texture("textured_output.jpg")
    # """Pick 4 Points for a rectangle area"""

    # pl = pv.Plotter()
    # pl.add_mesh(sfc,texture=tex1)
    # pl.enable_point_picking(callback=callback, left_clicking=True, show_point=False)
    # # a = []
    # # a.append()
    # # print

    # """
    # plane function: Use avg_norm and points to create plane function
    # -> interpolate: interpolate a squared area in four border points
    # -> create vertices: 
    # """
    

    # tmp = avg_norm.ravel()
    # pl.show()

    # pl = pv.Plotter()
    # pl.add_mesh(sfc,texture=tex1)
    # pl.camera.enable_parallel_projection() # orthogonal projection is used to transfer 3D to 2D using avg_norm
    # pl.camera.position =  tmp
    # pl.set_background('white')

    # pl.show(screenshot=f"original_1.png")
    # pl = pv.Plotter()
    # pl.add_mesh(sfc,texture=tex1,opacity=0.0)
    # pl.camera.enable_parallel_projection()
    # pl.camera.position =  tmp
    # stored_point_np = np.array(stored_points)
    # # stored_point_np = np.append(stored_point_np,stored_points[0])
    # rectangle = pv.Rectangle([stored_point_np[0], stored_point_np[1], stored_point_np[2], stored_point_np[3]])
    # # pl.add_lines(stored_point_np,color='yellow', width=5)
    # pl.add_mesh(rectangle,color='black')
    # # pl.show()
    
    
    # pl.set_background('white')

    # pl.show(screenshot=f"yellow_2.png")
    # # used the stored points to generate mask


    # print(f"stored_point_np\n {stored_point_np}")

    # c = avg_norm[0] *stored_point_np[0,0] + avg_norm[1] *stored_point_np[0,1] + avg_norm[2] *stored_point_np[0,2] 

    # min_p = np.min(stored_point_np,axis=0)
    # max_p = np.max(stored_point_np,axis=0)
    # std = np.std(stored_point_np,axis=0)
    # print(f"std {std}")
    # print(f"min_p.shape {min_p.shape}")

    # n = 20
    # # + np.random.uniform(-5, 5, size=n)
    # # + np.random.uniform(-std[0], std[0], size=n)
    # x = np.linspace(min_p[0], max_p[0], num=n) 
    # y = np.linspace(min_p[1], max_p[1], num=n)
    # z = np.linspace(min_p[2], max_p[2], num=n)
    # xx1,yy1,zz1 = np.meshgrid(x, y, z)

    # # calculate plane function and learn a gaussian distribution


    # # project_points_to_plane

    # points = np.c_[xx1.reshape(-1), yy1.reshape(-1), zz1.reshape(-1)]
    # # points[0:5, :]
    # print(f"points.shape {points.shape}")
    # cloud  = pv.PolyData(points)
    # # cloud = pv.StructuredGrid(xx1,yy1,zz1)
    # # cloud.plot(point_size=15)
    # cloud.plot()
    # # surf = cloud
    # surf2 = cloud.delaunay_2d(inplace=True) # create surface plane with plane pointcloud
    # pp = surf2.points.astype(np.double)
    # print(pp)
    # print(pp.shape)

    # tex2 = pv.read_texture("5.png")
    # # tex = examples.download_puppy_texture()
    # # axial_num_puppies = 4
    # # xc = np.linspace(0, axial_num_puppies, surf.dimensions[0])
    # # yc = np.linspace(0, axial_num_puppies, surf.dimensions[1])
    # # zc = np.linspace(0, axial_num_puppies, surf.dimensions[2])
    # # xxc, yyc, zzc = np.meshgrid(xc, yc,zc)
    # # puppy_coords = np.c_[yyc.ravel(), xxc.ravel(),zzc.ravel()]
    # # surf.active_t_coords = puppy_coords

    # surf2.texture_map_to_plane(origin=pp[0,:],point_u=pp[400,:],point_v=pp[-20,:],inplace=True)
    # # surf2.texture_map_to_plane(origin=pp[:,-20],point_u=pp[:,0],point_v=pp[:,-1],inplace=True)




    # # surf.plot(show_edges=True,texture=tex)


    # # xx, yy = np.meshgrid(x, y)
    # # A, b = 100, 100
    # # # zz = A * np.exp(-0.5 * ((xx / b) ** 2.0 + (yy / b) ** 2.0))
    # # zz = (-avg_norm[0] *xx -avg_norm[1] *yy + c)/avg_norm[2] + np.random.uniform(-std[2], std[2], size=n)

    # # print(zz.shape)
    # # print(f"min zz {np.min(zz)}")
    # # print(f"max zz {np.max(zz)}")

    # # good_idx = np.where((zz<max_p[2]) & (zz >min_p[2]))
    # # print(good_idx)




    # # # Get the points as a 2D NumPy array (N by 3)
    # # points = np.c_[xx.reshape(-1), yy.reshape(-1), zz.reshape(-1)]
    # # points[0:5, :]
    # # cloud = pv.PolyData(points)
    # # cloud.plot(point_size=15)
    # # surf = cloud.delaunay_2d()
    # # # tex = examples.download_masonry_texture()
    # # tex = pv.read_texture("generated_texture.png")
    # # surf.texture_map_to_plane(inplace=True)
    # surf2.plot(show_edges=False,texture=tex)

    pl = pv.Plotter()
   
    pl.add_mesh(sfc,texture=tex1,opacity=0.3)
    pl.camera.enable_parallel_projection()
    pl.camera.position =  tmp
    # pl.enable_point_picking(callback=callback, left_clicking=True, show_point=False)


    pl.add_mesh(surf,texture=tex)
    # pl.add_mesh(surf2,texture=tex2)


    # pl.add_mesh(inpaint.pvmesh,texture=tex1)
    pl.add_mesh(reduced_sphere, texture=tex1)

    # pl.show()
    pl.set_background('white')
    pl.show(screenshot=f"results1.png")


    # surf.plot(show_edges=True,texture=tex)


    np.save('segements0.npy',points_seg_1)
    print(points_seg_1.shape)
    get_center_area(points_seg_1)



    # print(inpaint.segments[0].vertices)

    # print(inpaint.pvmesh)
