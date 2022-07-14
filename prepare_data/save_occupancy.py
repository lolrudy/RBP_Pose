from typing import Tuple

import numpy as np
import _pickle as cPickle
import cv2


def voxelize(
        points: np.ndarray,
        voxel_size: np.ndarray,
        grid_range: np.ndarray,
        max_points_in_voxel: int,
        max_num_voxels: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Converts 3D point cloud to a sparse voxel grid
    :param points: (num_points, num_features), first 3 elements must be <x>, <y>, <z>
    :param voxel_size: (3,) - <width>, <length>, <height>
    :param grid_range: (6,) - <min_x>, <min_y>, <min_z>, <max_x>, <max_y>, <max_z>
    :param max_points_in_voxel:
    :param max_num_voxels:
    :param include_relative_position: boolean flag, if True, the output num_features will include relative
    position of the point within the voxel
    :return: tuple (
        voxels (num_voxels, max_points_in_voxels, num_features),
        coordinates (num_voxels, 3),
        num_points_per_voxel (num_voxels,)
    )
    """
    points_copy = points.copy()
    grid_size = np.floor((grid_range[3:] - grid_range[:3]) / voxel_size).astype(np.int32)

    coor_to_voxelidx = np.full((grid_size[2], grid_size[1], grid_size[0]), -1, dtype=np.int32)
    voxels = np.zeros((max_num_voxels, max_points_in_voxel, points.shape[-1]), dtype=points_copy.dtype)
    coordinates = np.zeros((max_num_voxels, 3), dtype=np.int32)
    num_points_per_voxel = np.zeros(max_num_voxels, dtype=np.int32)

    points_coords = np.floor((points_copy[:, :3] - grid_range[:3]) / voxel_size).astype(np.int32)
    mask = ((points_coords >= 0) & (points_coords < grid_size)).all(1)
    points_coords = points_coords[mask, ::-1]
    points_copy = points_copy[mask]
    assert points_copy.shape[0] == points_coords.shape[0]

    voxel_num = 0
    for i, coord in enumerate(points_coords):
        voxel_idx = coor_to_voxelidx[tuple(coord)]
        if voxel_idx == -1:
            voxel_idx = voxel_num
            if voxel_num > max_num_voxels:
                continue
            voxel_num += 1
            coor_to_voxelidx[tuple(coord)] = voxel_idx
            coordinates[voxel_idx] = coord
        point_idx = num_points_per_voxel[voxel_idx]
        if point_idx < max_points_in_voxel:
            voxels[voxel_idx, point_idx] = points_copy[i]
            num_points_per_voxel[voxel_idx] += 1

    return voxels[:voxel_num], coordinates[:voxel_num], num_points_per_voxel[:voxel_num]


def voxelize_object(
        points: np.ndarray,
        resolution: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Converts 3D point cloud to a sparse voxel grid
    :param points: (num_points, num_features), first 3 elements must be <x>, <y>, <z>
    :param resolution:
    :return: tuple (
        voxels (resolution, resolution, resolution), binary
        num_points_per_voxel (resolution, resolution, resolution), each region contains points number
        coordinates (num_voxels, 3),
        points_coord (num_points, 3),
        grid_range [x_min, y_min, z_min, x_max, y_max, z_max],
        voxel_size ,
    )
    """
    points_copy = points.copy()
    x_max = points[:, 0].max() + 1e-15
    y_max = points[:, 1].max() + 1e-15
    z_max = points[:, 2].max() + 1e-15
    x_min = points[:, 0].min() - 1e-15
    y_min = points[:, 1].min() - 1e-15
    z_min = points[:, 2].min() - 1e-15
    voxel_size = np.array([(x_max - x_min) / resolution, (y_max - y_min) / resolution, (z_max - z_min) / resolution])
    grid_range = np.array([x_min, y_min, z_min, x_max, y_max, z_max])
    grid_size = np.floor((grid_range[3:] - grid_range[:3]) / voxel_size).astype(np.int32)
    assert grid_size[0] == grid_size[1] == grid_size[2] == resolution
    voxels = np.full((grid_size[0], grid_size[1], grid_size[2]), False, dtype=np.bool_)
    coordinates = np.zeros((grid_size[0], grid_size[1], grid_size[2], 3), dtype=np.int32)
    num_points_per_voxel = np.zeros((grid_size[0], grid_size[1], grid_size[2]), dtype=np.int32)

    points_coords = np.floor((points_copy[:, :3] - grid_range[:3]) / voxel_size).astype(np.int32)
    mask = ((points_coords >= 0) & (points_coords < grid_size)).all(1)
    points_coords = points_coords[mask, :]
    points_copy = points_copy[mask]
    assert points_copy.shape[0] == points_coords.shape[0]
    assert points_copy.shape[0] == points.shape[0]
    for i, coord in enumerate(points_coords):
        voxels[tuple(coord)] = True
        num_points_per_voxel[tuple(coord)] += 1
        coordinates[tuple(coord)] = coord
    '''
    show figure
    '''
    '''
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure(dpi=500)
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(points_coords[:, 0], points_coords[:, 1], points_coords[:, 2], marker='.')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    plt.show()
    '''
    return voxels, num_points_per_voxel, coordinates, points_coords, grid_range, voxel_size


def extract_face_sketch(voxels):
    width, length, height = voxels.shape
    assert width == length == height
    resolution = width
    sketch_occ = np.zeros((6, resolution, resolution))
    # 1: y positive, 2 ,3 ,4 ,5 faces around y axis, 6 y negative
    sketch_occ[0] = voxels[:, resolution-1, :]
    sketch_occ[1] = voxels[resolution - 1, :, :]
    sketch_occ[2] = voxels[:, :, resolution - 1]
    sketch_occ[3] = voxels[0, :, :]
    sketch_occ[4] = voxels[:, :, 0]
    sketch_occ[5] = voxels[:, 0, :]
    sketch_coords = []
    for face_i in range(6):
        face_ori = sketch_occ[face_i].copy()
        face_valid = get_sketch_from_face_ray(face_ori)
        kernel = np.ones((3,3))
        face_valid = cv2.morphologyEx(face_valid, cv2.MORPH_CLOSE, kernel, 1)

        sketch_occ[face_i] = face_valid
        coord1, coord2 = np.where(sketch_occ[face_i] == True)
        if face_i <= 2:
            coord_face = np.repeat([resolution-1], len(coord1))
        else:
            coord_face = np.repeat([0], len(coord1))
        if face_i in [0,5]:
            coord = np.array([coord1, coord_face, coord2])
        elif face_i in [1,3]:
            coord = np.array([coord_face, coord1, coord2])
        elif face_i in [2,4]:
            coord = np.array([coord1, coord2, coord_face])
        else:
            raise NotImplementedError
        coord = coord.T
        sketch_coords.append(coord)
    return sketch_occ, sketch_coords


def get_sketch_from_face_neighbour(face_ori, threshold=6):
    face_valid = face_ori.copy()
    for i in range(1, resolution - 1):
        for j in range(1, resolution - 1):
            if face_ori[i, j] and (face_ori[i - 1, j] + face_ori[i + 1, j] + face_ori[i, j - 1] + face_ori[i, j + 1] +
                                   face_ori[i - 1, j - 1] + face_ori[i - 1, j + 1] + face_ori[i + 1, j - 1] + face_ori[
                                       i + 1, j + 1] >= threshold):
                face_valid[i, j] = False
    return face_valid


def get_sketch_from_face_ray(face_ori, ray_number=180):
    face_valid = np.zeros_like(face_ori)
    resolution = face_ori.shape[0]
    center = resolution/2
    for ray_index in range(ray_number):
        ray_angle = np.pi / ray_number * ray_index
        if abs(ray_angle - np.pi / 2) <= np.pi / 180 or abs(ray_angle) <= np.pi / 180:
            continue
        slope = np.tan(ray_angle)
        stride = abs(1/slope)
        stride = min(1.0, stride)
        for step in range(-resolution//2, resolution//2 + 1):
            step = step*stride
            x = np.round(center+step).astype(int)
            y = np.round(center+step*slope).astype(int)
            if x >= resolution or y >= resolution:
                continue
            if face_ori[x,y]:
                face_valid[x,y] = True
                break
        for step in range(resolution//2, -resolution//2-1, -1):
            step = step*stride
            x = np.round(center+step).astype(int)
            y = np.round(center+step*slope).astype(int)
            if x >= resolution or y >= resolution:
                continue
            if face_ori[x,y]:
                face_valid[x,y] = True
                break
    return face_valid
# def get_sketch_points(points, points_coords, resolution):
#     # 1: y positive, 2 ,3 ,4 ,5 faces around y axis, 6 y negative
#     sketch_index_0 = np.where(points_coords[:,1] == resolution-1)
#     sketch_index_1 = np.where(points_coords[:,0] == resolution-1)
#     sketch_index_2 = np.where(points_coords[:,2] == resolution-1)
#     sketch_index_3 = np.where(points_coords[:,0] == 0)
#     sketch_index_4 = np.where(points_coords[:,2] == 0)
#     sketch_index_5 = np.where(points_coords[:,1] == 0)
#
#     return [points[sketch_index_0], points[sketch_index_1],
#             points[sketch_index_2], points[sketch_index_3],
#             points[sketch_index_4], points[sketch_index_5],]

resolution = 16
from tqdm import tqdm
for split in ['real_train', 'real_test', 'camera_train', 'camera_val']:
    with open(f'/data/wanggu/Storage/9Ddata/obj_models/{split}.pkl', 'rb') as file:
        model_info = cPickle.load(file)
    occupancy_dict = {}
    for instance in tqdm(model_info.keys()):
        model_points = model_info[instance]
        '''
        
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        fig = plt.figure(dpi=500)
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(model_points[:, 0], model_points[:, 1], model_points[:, 2], marker='.')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        plt.show()
        '''
        voxels, num_points_per_voxel, coordinates, points_coords, grid_range, voxel_size = voxelize_object(model_points, resolution)
        # sketch_points = get_sketch_points(model_points, points_coords, resolution)
        # occupancy_dict[instance] = {'voxels':voxels, 'num_points_per_voxel':num_points_per_voxel, 'coordinates':coordinates,
        #                             'points_coords':points_coords, 'points': model_points, 'sketch_occupancy':extract_face_sketch(voxels),
        #                             'sketch_points':sketch_points, 'grid_range':grid_range, 'voxel_size':voxel_size}
        sketch_occ, sketch_coords = extract_face_sketch(voxels)

        '''
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        fig = plt.figure(dpi=500)
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(sketch_coords[0][:, 0], sketch_coords[0][:, 1], sketch_coords[0][:, 2], marker='.')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        plt.show()
        '''
        occupancy_dict[instance] = {'voxels': voxels,
                                    'sketch_occupancy': sketch_occ, 'sketch_coords': sketch_coords,
                                    'grid_range': grid_range, 'voxel_size': voxel_size}

    with open(f'/data/wanggu/Storage/9Ddata/obj_models/{split}_occupancy_res{resolution}.pkl', 'wb') as file:
        cPickle.dump(occupancy_dict, file)