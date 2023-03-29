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
from datetime import datetime

class Inpaint3D():

    def __init__(self,objname) -> None:
        self.objname = objname
        self.pvmesh = None
        self.textured_mesh = None
        self.pcd = None
        self.segments = None
        self.face = None
        self.segments_dict = {'segments':None, 'mesh':[],'sfc':[],'avg_norm':[]}
        # self.
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
            segment_models[i], inliers = rest.segment_plane(distance_threshold=0.03,ransac_n=4,num_iterations=1000)
            global_idx[i] = np.array(inliers)
            segments[i]=rest.select_by_index(inliers)
            labels = np.array(segments[i].cluster_dbscan(eps=d_threshold * 10, min_points=15))
            candidates = [len(np.where(labels == j)[0]) for j in np.unique(labels)]
            best_candidate = int(np.unique(labels)[np.where(candidates == np.max(candidates))[0]])
            rest = rest.select_by_index(inliers, invert=True) + segments[i].select_by_index(
            list(np.where(labels != best_candidate)[0]))
    
            segments[i] = segments[i].select_by_index(list(np.where(labels == best_candidate)[0]))

            segments[i].paint_uniform_color(list(colors[:3]))
            rest.paint_uniform_color([0.6, 0.6, 0.6])
        self.segments = segments
        self.segments_dict['segments'] = segments
        print(f"plane_segementation() completed")
        o3d.visualization.draw_geometries([segments[i] for i in range(max_plane_idx)]+[rest])
        return segments
        pass
    
    def segementation_completion(self):
        seg_len = len(self.segments)
        kd_tree = KDTree(self.pvmesh.points.astype(np.double)) 
        # segments_meta =
        for i in range(seg_len):
            # complete_seg(i)
            points_seg_1 = np.asarray(inpaint.segments[i].points)
            # o3d.visualization.draw_geometries([inpaint.segments[0]])
            # idx = inpaint.pvmesh.find_closest_cell(points_seg_1)
            dist, idx = kd_tree.query(points_seg_1,k=5)
            allidx = np.unique(idx)
            ids = inpaint.findallidx(allidx)
            newid = interp(list(ids),thre=1)
            mesh = inpaint.pvmesh.extract_cells(newid)
            sfc = mesh.extract_surface()

            """sfc is the extracted first segment plane"""
            sfc.compute_normals(cell_normals=True, point_normals=False, inplace=True)
            # print(f"sfc {sfc}")
            avg_norm = np.mean(sfc['Normals'],axis=0)
            # print(f"avg_norm {avg_norm}")
            self.segments_dict['mesh'].append(mesh)
            self.segments_dict['sfc'].append(sfc)
            self.segments_dict['avg_norm'].append(avg_norm)
        print(f"segementation_completion() completed")

    def find_points_id(self, points):
        ans = []
        print(f"len(points) {points}")
        for point in points:
            print(f"len(self.segments_dict['avg_norm']): {len(self.segments_dict['avg_norm'])}")
            for i in range(len(self.segments_dict['avg_norm'])):
                # print(f"len(self.segments_dict['sfc'][i].points): {len(self.segments_dict['sfc'][i].points)}")
                seg_points = np.asarray(self.segments_dict['sfc'][i].points)
                # print(f"seg_points.shape {seg_points.shape}")
                point = numpy.asarray(point)
                # print(f"point {point}")
                # print(f"seg_points {seg_points[:5,:]}")
                # if point in seg_points:
                #     print(np.argwhere(seg_points==point))
                #     print(f"point {point} = {seg_points[np.argwhere(seg_points==point)[0]]}")
                #     ans.append(i)
                #     break
                if np.any(np.equal(seg_points, point).all(axis=1)):
                    ans.append(i)
                    break

            # ans.append(-1)
        return ans
                
    def findallidx(self, ids):
        ans = np.empty((0),dtype=int)
        for i in ids:
            result = np.argwhere(self.face==i)[:,0].ravel()
            # print(result)
            ans = numpy.concatenate((ans,result))
            # print(ans)
        return numpy.unique(ans) 
    


    def get_cut_cube(self, stored_points):
        stored_point_np = np.array(stored_points)

        min_p = np.min(stored_point_np,axis=0)
        max_p = np.max(stored_point_np,axis=0)
        x_center = np.mean([min_p[0],max_p[0]])
        y_center = np.mean([min_p[1],max_p[1]])

        z_center = np.mean([min_p[2],max_p[2]])
        x_len =  max_p[0] - min_p[0] 
        y_len =  max_p[1] - min_p[1] 
        z_len =  max_p[2] - min_p[2] 
        clip_cube = pv.Cube(center=(x_center,y_center,z_center),x_length=1.5*x_len,y_length=1.5*y_len,z_length=1.5*z_len)
        select = inpaint.pvmesh.select_enclosed_points(clip_cube)
        # reduced_sphere, ridx = inpaint.pvmesh.remove_points(select['SelectedPoints'].view(bool),inplace = True)
        #                     #    adjacent_cells=False)
        # return reduced_sphere, ridx
        dargs = dict(show_edges=False)

        inside = select.threshold(0.5)
        outside = select.threshold(0.5, invert=True)
        return outside
        # p = pv.Plotter()
        # p.add_mesh(outside, texture=tex1, **dargs)
        # # p.add_mesh(inside, texture=tex1, **dargs)
        # # p.add_mesh(rot, color="mintcream", opacity=0.35, **dargs)

        # # p.camera_position = cpos
        # p.show()
        # inpaint.pvmesh.remove_points(select['SelectedPoints'].view(bool),inplace = True)
        # return 
        # # # clipped_mesh = inpaint.pvmesh.clip_box(clip_cube,invert=False)
        # # # clip_area = [min_p[0],max_p[0],min_p[1],max_p[1],min_p[2],max_p[2]]
        # # # clipped_mesh = inpaint.pvmesh.clip_box(clip_area,invert=False)
        # pl=pv.Plotter()
        # # pl.add_mesh(select,color="red")
        # # pl.add_mesh(clip_cube,color="blue")
        # pl.add_mesh(reduced_sphere, texture=tex1)
        # # # pl.add_mesh(inpaint.pvmesh,texture=tex1)
        # # print(f"show clipped")
        # pl.show()
    
    def inpaint_2D(self,image):
        pass

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


def complete_seg(seg_id):
    kd_tree = KDTree(inpaint.pvmesh.points.astype(np.double)) 
    points_seg_1 = np.asarray(inpaint.segments[seg_id].points)
    # o3d.visualization.draw_geometries([inpaint.segments[0]])
    # idx = inpaint.pvmesh.find_closest_cell(points_seg_1)
    dist, idx = kd_tree.query(points_seg_1,k=5)
    allidx = np.unique(idx)
    ids = inpaint.findallidx(allidx)
    newid = interp(list(ids),thre=1)
    mesh = inpaint.pvmesh.extract_cells(newid)
    sfc = mesh.extract_surface()

    """sfc is the extracted first segment plane"""
    sfc.compute_normals(cell_normals=True, point_normals=False, inplace=True)
    print(f"sfc {sfc}")
    avg_norm = np.mean(sfc['Normals'],axis=0)
    print(f"avg_norm {avg_norm}")
    # tex1 = pv.read_texture(f"{data}textured_output.jpg")

    return mesh, sfc, avg_norm

def find_seg(point):
    pvpoint = pv.PolyData([point,]) 
    seg_len = len(inpaint.segments)
    
    for i in range(seg_len):
        # try:
        mesh, sfc, avg_norm = complete_seg(i)
        # ans = pvpoint.select_enclosed_points(mesh)
        print(f"ans {ans}")
            # return i
        # except:
            # return -1
            # pass
    return -1

def get_point2plane_distance(point, plane_coeff):

    return 0

def plane_coefficients(normal, point):
    # normal is a tuple (a, b, c)
    # point is a tuple (x, y, z)
    a, b, c = normal
    x, y, z = point
    d = -1 * (a * x + b * y + c * z)
    return (a, b, c, d)

import math

def distance_to_plane(point, plane_coeffs):
    # point is a tuple (x, y, z)
    # plane_coeffs is a tuple (a, b, c, d)
    a, b, c, d = plane_coeffs
    x, y, z = point
    dist = abs(a*x + b*y + c*z + d) / math.sqrt(a*a + b*b + c*c)
    return dist

def plane3_coefficients(point1, point2, point3):
    # point1, point2, point3 are tuples (x, y, z)
    # Find two vectors in the plane
    v1 = (point2[0] - point1[0], point2[1] - point1[1], point2[2] - point1[2])
    v2 = (point3[0] - point1[0], point3[1] - point1[1], point3[2] - point1[2])
    # Calculate the normal vector of the plane
    normal = (v1[1]*v2[2] - v1[2]*v2[1], v1[2]*v2[0] - v1[0]*v2[2], v1[0]*v2[1] - v1[1]*v2[0])
    # Find d in the plane equation ax + by + cz + d = 0 using one of the points
    d = -1*(normal[0]*point1[0] + normal[1]*point1[1] + normal[2]*point1[2])
    # Return the coefficients of the plane equation
    return (normal[0], normal[1], normal[2], d)

def get_cut_2_cube(stored_points1,stored_points2, avg_norm1=0, distance1 = 0, avg_norm2=0, distance2 = 0):
    def get_cube(stored_point_np, avg_norm, distance):
        min_p = np.min(stored_point_np,axis=0)
        max_p = np.max(stored_point_np,axis=0)
        x_center = np.mean([min_p[0],max_p[0]])
        y_center = np.mean([min_p[1],max_p[1]])

        z_center = np.mean([min_p[2],max_p[2]])
        x_len =  max_p[0] - min_p[0] 
        y_len =  max_p[1] - min_p[1] 
        z_len =  max_p[2] - min_p[2]
        norm_direction = np.argmax(np.abs(avg_norm))
        cube_len = [x_len, y_len, z_len]
        print(f" before cube len {cube_len}")

        print(f"norm_direction {norm_direction}")
        cube_scale = np.ones((3,1))
        cube_scale[norm_direction] = 20.0
        # print(f"cube_scale {cube_scale}")
        cube_len[norm_direction] = distance * 2.2
        print(f" after cube len {cube_len}")
        clip_cube = pv.Cube(center=(x_center,y_center,z_center),x_length=cube_len[0],y_length=cube_len[1],z_length=cube_len[2])
        return clip_cube
    stored_point_np1 = np.array(stored_points1[:-1])
    stored_point_np2 = np.array(stored_points1[:-1])

    clip_cube1 = get_cube(stored_point_np1, avg_norm1, distance1)
    clip_cube2 = get_cube(stored_point_np2, avg_norm2, distance2)


    # clip_cube = pv.Cube(center=(x_center,y_center,z_center),x_length=cube_scale[0]*x_len,y_length=cube_scale[1]*y_len,z_length=cube_scale[2]*z_len)
    select = inpaint.pvmesh.select_enclosed_points(clip_cube1)
    # select = select.select_enclosed_points(clip_cube2)

    # reduced_sphere, ridx = inpaint.pvmesh.remove_points(select['SelectedPoints'].view(bool),inplace = True)
    #                     #    adjacent_cells=False)
    # return reduced_sphere, ridx
    dargs = dict(show_edges=False)

    inside = select.threshold(0.5)
    outside = select.threshold(0.5, invert=True)

    select = outside.select_enclosed_points(clip_cube2)
    # reduced_sphere, ridx = inpaint.pvmesh.remove_points(select['SelectedPoints'].view(bool),inplace = True)
    #                     #    adjacent_cells=False)
    # return reduced_sphere, ridx
    dargs = dict(show_edges=False)

    inside = select.threshold(0.5)
    outside = select.threshold(0.5, invert=True)


    return outside, inside
    pass
def get_cut_cube(avg_norm=0, distance = 0):
    stored_point_np = np.array(stored_points[:-1])

    min_p = np.min(stored_point_np,axis=0)
    max_p = np.max(stored_point_np,axis=0)
    x_center = np.mean([min_p[0],max_p[0]])
    y_center = np.mean([min_p[1],max_p[1]])

    z_center = np.mean([min_p[2],max_p[2]])
    x_len =  max_p[0] - min_p[0] 
    y_len =  max_p[1] - min_p[1] 
    z_len =  max_p[2] - min_p[2]
    norm_direction = np.argmax(np.abs(avg_norm))
    cube_len = [x_len, y_len, z_len]
    print(f" before cube len {cube_len}")

    print(f"norm_direction {norm_direction}")
    cube_scale = np.ones((3,1))
    cube_scale[norm_direction] = 20.0
    # print(f"cube_scale {cube_scale}")
    cube_len[norm_direction] = distance * 2.2
    print(f" after cube len {cube_len}")
    clip_cube = pv.Cube(center=(x_center,y_center,z_center),x_length=cube_len[0],y_length=cube_len[1],z_length=cube_len[2])

    # clip_cube = pv.Cube(center=(x_center,y_center,z_center),x_length=cube_scale[0]*x_len,y_length=cube_scale[1]*y_len,z_length=cube_scale[2]*z_len)
    select = inpaint.pvmesh.select_enclosed_points(clip_cube)
    # reduced_sphere, ridx = inpaint.pvmesh.remove_points(select['SelectedPoints'].view(bool),inplace = True)
    #                     #    adjacent_cells=False)
    # return reduced_sphere, ridx
    dargs = dict(show_edges=False)

    inside = select.threshold(0.5)
    outside = select.threshold(0.5, invert=True)
    return outside, inside
    # p = pv.Plotter()
    # p.add_mesh(outside, texture=tex1, **dargs)
    # # p.add_mesh(inside, texture=tex1, **dargs)
    # # p.add_mesh(rot, color="mintcream", opacity=0.35, **dargs)

    # # p.camera_position = cpos
    # p.show()
    # inpaint.pvmesh.remove_points(select['SelectedPoints'].view(bool),inplace = True)
    # return 
    # # # clipped_mesh = inpaint.pvmesh.clip_box(clip_cube,invert=False)
    # # # clip_area = [min_p[0],max_p[0],min_p[1],max_p[1],min_p[2],max_p[2]]
    # # # clipped_mesh = inpaint.pvmesh.clip_box(clip_area,invert=False)
    # pl=pv.Plotter()
    # # pl.add_mesh(select,color="red")
    # # pl.add_mesh(clip_cube,color="blue")
    # pl.add_mesh(reduced_sphere, texture=tex1)
    # # # pl.add_mesh(inpaint.pvmesh,texture=tex1)
    # # print(f"show clipped")
    # pl.show()

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
    tex1 = pv.read_texture(f"{data}textured_output.jpg")

    pl = pv.Plotter()
    pl.add_mesh(inpaint.pvmesh,texture=tex1)
    pl.set_background('white')

    pl.show()

    inpaint.plane_segementation(max_plane_idx=10)
    inpaint.segementation_completion()


    pl = pv.Plotter()
    pl.add_mesh(inpaint.pvmesh,texture=tex1)
    pl.enable_point_picking(callback=callback, left_clicking=True, show_point=False)
    # tmp = avg_norm.ravel()
    _ = pl.add_axes(line_width=5, labels_off=False)


    # print("The current time is:", time_str)
    pl.show()
    plane_id_lst = inpaint.find_points_id(stored_points)
    print(f"plane_id_lst {plane_id_lst}")

    # # plane_id_lst = []
    # # for i in stored_points:
    # #     plane_id =  find_seg(point=i)
        
    # #     plane_id_lst.append(plane_id)

    # reduced_sphere, ridx = get_cut_cube()
    # print(f"reduced_sphere {reduced_sphere}")
    # print(f"inpaint.pvmesh {inpaint.pvmesh}")

    # take the face normal from the 1st point
    # take the distance from the 5th point and calculate the distance to the plane

    avg_norm = inpaint.segments_dict['avg_norm'][plane_id_lst[0]]
    print(f"avg_norm {avg_norm}")
    # print(f"inpaint.segments_dict['avg_norm'][plane_id_lst[0]] {inpaint.segments_dict['avg_norm'][plane_id_lst[0]]}")
    

    points_3_coeff = plane3_coefficients(stored_points[0], stored_points[1], stored_points[2])
    print(f"points_3_coeff {points_3_coeff}")

    #reconstruct the plane coeff first
    plane_coeff = plane_coefficients(avg_norm, stored_points[0])
    print(f"plane_coeff {plane_coeff}")

    distance = distance_to_plane(stored_points[-1], points_3_coeff)
    print(f"distance {distance}")

    tmp = avg_norm.ravel()
    tmp= np.array(copy.deepcopy(points_3_coeff[:3]))
    tmp[np.argmax(np.abs(points_3_coeff[:3]))] = distance
    # tmp = avg_norm.ravel()
    # tmp = stored_points[-1]
    # tmp[np.argmax(np.abs(points_3_coeff[:3]))] += distance


    outside, inside = get_cut_cube(points_3_coeff[:3], distance)
    pl = pv.Plotter()
    pl.add_mesh(outside,texture=tex1)
    # pl.add_mesh(inpaint.segments_dict['mesh'][plane_id_lst[0]],texture=tex1)

    # _ = pl.add_axes(line_width=5, labels_off=False)
    pl.camera.enable_parallel_projection() # orthogonal projection is used to transfer 3D to 2D using avg_norm
    pl.camera.position =  points_3_coeff[:3]
    # pl.show()
    pl.camera.position =  tmp

    pl.set_background('white')
    now = datetime.now()
    time_str = now.strftime("%H_%M")
    pl.show(screenshot=f"origin_{time_str}.png")

#-----------------------------------------------------------------------------------------------------------------------------------------------------
    rectangle = pv.Rectangle([stored_points[0], stored_points[1], stored_points[2], stored_points[3]])
    pl = pv.Plotter()
    # pl.add_mesh(outside,texture=tex1)
    # _ = pl.add_axes(line_width=5, labels_off=False)
    pl.add_mesh(outside,texture=tex1,opacity=0.0)
    # pl.add_mesh(inpaint.segments_dict['mesh'][plane_id_lst[0]],texture=tex1,opacity=0.0)

    pl.add_mesh(rectangle,color='white')
    pl.add_mesh(inside,color='white')


    pl.camera.enable_parallel_projection() # orthogonal projection is used to transfer 3D to 2D using avg_norm
    pl.camera.position =  points_3_coeff[:3]
    # pl.show()
    pl.camera.position =  tmp

    pl.set_background('black')
    now = datetime.now()
    time_str = now.strftime("%H_%M")
    pl.show(screenshot=f"mask_{time_str}.png")


    # pl.enable_point_picking(callback=callback, left_clicking=True, show_point=False)
    # tmp = avg_norm.ravel()
    # pl.add_mesh(reduced_sphere,texture=tex1)

    #-------------------------------------------------------
    stored_point_np = np.array(stored_points[:-1])
    n = 20

    min_p = np.min(stored_point_np,axis=0)
    max_p = np.max(stored_point_np,axis=0)
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
    # tex = pv.read_texture("generated_texture3.png")
    tex = pv.read_texture("generated_texture7.png")

    norm_direction = np.argmax(np.abs(points_3_coeff[:3]))
    min_x = np.min(pp[:, 0])
    max_x = np.max(pp[:, 0])
    min_y = np.min(pp[:, 1])
    max_y = np.max(pp[:, 1])
    min_z = np.min(pp[:, 2])
    max_z = np.max(pp[:, 2])
    print(f"min_x {min_x} max_x {max_x}")
    print(f"min_y {min_y} max_y {max_y}")
    print(f"min_z {min_z} max_z {max_z}")
    print(f"norm_direction {norm_direction}")

    if norm_direction == 0:
        origin = [max_x, min_y, min_z]
        point_u = [max_x, max_y, min_z]
        point_v = [max_x, min_y, max_z]
    elif norm_direction == 1:
        origin = [min_x, max_y, min_z]
        point_u = [max_x, max_y, min_z]
        point_v = [min_x, max_y, max_z]
    elif norm_direction == 2:
        origin = [min_x, min_y, max_z]
        point_u = [max_x, min_y, max_z]
        point_v = [min_x, max_y, max_z]

    print(f"origin {origin} point_u {point_u} point_v {point_v}")



    surf.texture_map_to_plane(origin=origin,point_u=point_u,point_v=point_v,inplace=True)

    # surf.texture_map_to_plane(origin=pp[-20,:],point_u=pp[0,:],point_v=pp[-1,:],inplace=True)
    #  bottom left , bottom right corner, top left corner
    surf.plot(show_edges=False,texture=tex)
    pl = pv.Plotter()

    pl.add_mesh(surf,texture=tex)
    pl.add_mesh(outside,texture=tex1)
    pl.set_background('white')

    pl.show()

#--------------------------------------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------------------------------------
    stored_points1 = copy.deepcopy(stored_points)
    stored_points = []
    pl = pv.Plotter()
    pl.add_mesh(inpaint.pvmesh,texture=tex1)
    pl.enable_point_picking(callback=callback, left_clicking=True, show_point=False)
    # tmp = avg_norm.ravel()
    _ = pl.add_axes(line_width=5, labels_off=False)


    # print("The current time is:", time_str)
    pl.show()
    plane_id_lst = inpaint.find_points_id(stored_points)
    print(f"plane_id_lst {plane_id_lst}")

    
    def geT_param(stored_points):
        points_3_coeff = plane3_coefficients(stored_points[0], stored_points[1], stored_points[2])
        print(f"points_3_coeff {points_3_coeff}")

        #reconstruct the plane coeff first
        # plane_coeff = plane_coefficients(avg_norm, stored_points[0])
        # print(f"plane_coeff {plane_coeff}")

        distance = distance_to_plane(stored_points[-1], points_3_coeff)
        print(f"distance {distance}")

        # tmp = avg_norm.ravel()
        tmp= np.array(copy.deepcopy(points_3_coeff[:3]))
        tmp[np.argmax(np.abs(points_3_coeff[:3]))] = distance
        return points_3_coeff
    # tmp = avg_norm.ravel()
    # tmp = stored_points[-1]
    # tmp[np.argmax(np.abs(points_3_coeff[:3]))] += distance


    # outside, inside = get_cut_cube(points_3_coeff[:3], distance)
    outside, inside = get_cut_2_cube(stored_points1,stored_points, avg_norm1=0, distance1 = 0, avg_norm2=0, distance2 = 0)
    points_3_coeff = geT_param(stored_points)
    def get_surf(stored_points,points_3_coeff):
        stored_point_np = np.array(stored_points[:-1])
        n = 20

        min_p = np.min(stored_point_np,axis=0)
        max_p = np.max(stored_point_np,axis=0)
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
        # tex = pv.read_texture("generated_texture3.png")
        tex2 = pv.read_texture("generated_texture8.png")

        norm_direction = np.argmax(np.abs(points_3_coeff[:3]))
        min_x = np.min(pp[:, 0])
        max_x = np.max(pp[:, 0])
        min_y = np.min(pp[:, 1])
        max_y = np.max(pp[:, 1])
        min_z = np.min(pp[:, 2])
        max_z = np.max(pp[:, 2])
        print(f"min_x {min_x} max_x {max_x}")
        print(f"min_y {min_y} max_y {max_y}")
        print(f"min_z {min_z} max_z {max_z}")
        print(f"norm_direction {norm_direction}")

        if norm_direction == 0:
            origin = [max_x, min_y, min_z]
            point_u = [max_x, max_y, min_z]
            point_v = [max_x, min_y, max_z]
        elif norm_direction == 1:
            origin = [min_x, max_y, min_z]
            point_u = [max_x, max_y, min_z]
            point_v = [min_x, max_y, max_z]
        elif norm_direction == 2:
            origin = [min_x, min_y, max_z]
            point_u = [max_x, min_y, max_z]
            point_v = [min_x, max_y, max_z]

        print(f"origin {origin} point_u {point_u} point_v {point_v}")



        surf.texture_map_to_plane(origin=origin,point_u=point_u,point_v=point_v,inplace=True)
        return surf, tex2

    # surf.texture_map_to_plane(origin=pp[-20,:],point_u=pp[0,:],point_v=pp[-1,:],inplace=True)
    #  bottom left , bottom right corner, top left corner
    surf2, tex2 = get_surf(stored_points,points_3_coeff)
    # surf.plot(show_edges=False,texture=tex)
    pl = pv.Plotter()
    pl.add_mesh(surf2,texture=tex2)
    pl.add_mesh(surf,texture=tex)
    pl.add_mesh(outside,texture=tex1)
    pl.set_background('white')

    pl.show()