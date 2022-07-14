import numpy as np
import mmcv
import os
import cv2
from tools.eval_utils import get_bbox, load_depth
from tools.dataset_utils import crop_resize_by_warp_affine, get_2d_coord_np
import matplotlib.pyplot as plt
import math
from evaluation.eval_utils_cass import get_3d_bbox, transform_coordinates_3d, compute_3d_iou_new, calculate_2d_projections
from tqdm import tqdm


synset_names = ['BG'] + ['bottle', 'bowl', 'camera', 'can', 'laptop', 'mug']

def img2pcl(obj_mask, Depth, camK, coor2d):
    '''
    :param Sketch: bs x 6 x h x w : each point support each face
    :param Depth: bs x 1 x h x w
    :param camK:
    :param coor2d:
    :return:
    '''
    x_label = coor2d[0, :, :]
    y_label = coor2d[1, :, :]

    dp_now = Depth.squeeze()  # 256 x 256
    x_now = x_label  # 256 x 256
    y_now = y_label
    obj_mask_now = obj_mask.squeeze()  # 256 x 256
    dp_mask = (dp_now > 0.0)
    fuse_mask = obj_mask_now * dp_mask
    # sk_now should coorespond to pixels with avaliable depth

    camK_now = camK

    # analyze camK
    fx = camK_now[0, 0]
    fy = camK_now[1, 1]
    ux = camK_now[0, 2]
    uy = camK_now[1, 2]

    x_now = (x_now - ux) * dp_now / fx
    y_now = (y_now - uy) * dp_now / fy
    p_n_now = np.concatenate([x_now[fuse_mask > 0].reshape(-1, 1),
                         y_now[fuse_mask > 0].reshape(-1, 1),
                         dp_now[fuse_mask > 0].reshape(-1, 1)], axis=-1)
    return p_n_now, fuse_mask



def asymmetric_3d_iou(RT_1, RT_2, scales_1, scales_2):
    noc_cube_1 = get_3d_bbox(scales_1, 0)
    bbox_3d_1 = transform_coordinates_3d(noc_cube_1, RT_1)

    noc_cube_2 = get_3d_bbox(scales_2, 0)
    bbox_3d_2 = transform_coordinates_3d(noc_cube_2, RT_2)

    bbox_1_max = np.amax(bbox_3d_1, axis=0)
    bbox_1_min = np.amin(bbox_3d_1, axis=0)
    bbox_2_max = np.amax(bbox_3d_2, axis=0)
    bbox_2_min = np.amin(bbox_3d_2, axis=0)

    overlap_min = np.maximum(bbox_1_min, bbox_2_min)
    overlap_max = np.minimum(bbox_1_max, bbox_2_max)

    # intersections and union
    if np.amin(overlap_max - overlap_min) < 0:
        intersections = 0
    else:
        intersections = np.prod(overlap_max - overlap_min)
    union = np.prod(bbox_1_max - bbox_1_min) + \
            np.prod(bbox_2_max - bbox_2_min) - intersections
    overlaps = intersections / union
    return overlaps


def find_similar_RT_by_iou(RT_1, RT_2, scales_1, scales_2, max_iter=20):
    def y_rotation_matrix(theta):
        return np.array([[np.cos(theta), 0, np.sin(theta), 0],
                         [0, 1, 0, 0],
                         [-np.sin(theta), 0, np.cos(theta), 0],
                         [0, 0, 0, 1]])

    n = max_iter
    max_iou = 0
    similar_RT = None
    for i in range(n):
        rotated_RT_1 = RT_1 @ y_rotation_matrix(2 * math.pi * i / float(n))
        iou = asymmetric_3d_iou(rotated_RT_1, RT_2, scales_1, scales_2)
        if max_iou < iou:
            similar_RT = rotated_RT_1.copy()
            max_iou = iou
    return similar_RT

def find_similar_RT_by_R_error(RT_1, RT_2, max_iter=50):
    def y_rotation_matrix(theta):
        return np.array([[np.cos(theta), 0, np.sin(theta), 0],
                         [0, 1, 0, 0],
                         [-np.sin(theta), 0, np.cos(theta), 0],
                         [0, 0, 0, 1]])

    n = max_iter
    min_error = 10
    similar_RT = None
    R2 = RT_2[:3, :3]
    for i in range(n):
        rotated_RT_1 = RT_1 @ y_rotation_matrix(2 * math.pi * i / float(n))
        R1 = rotated_RT_1[:3,:3]
        R = R1 @ R2.transpose()
        error = np.arccos((np.trace(R) - 1) / 2)
        if error < min_error:
            similar_RT = rotated_RT_1.copy()
            min_error = error
    return similar_RT

def compute_keypoints(scale, RT):
    keypoints = get_3d_bbox(scale, 0)
    keypoints = transform_coordinates_3d(keypoints, RT)
    keypoints_2d = cam_K @ keypoints
    keypoints_2d = keypoints_2d[:2] / keypoints_2d[2]
    keypoints_2d = keypoints_2d.astype(int)
    keypoints_2d = keypoints_2d.T
    return keypoints_2d

def inverse_RT(RT):
    R = RT[:3, :3]
    t = RT[:3, 3]
    RT[:3, :3] = R.T
    RT[:3, 3] = - R.T @ t
    return RT


FACES = [[0,1,5,4], [0,1,3,2], [2,3,7,6], [4,5,7,6], [0,2,6,4], [1,3,7,5]]

def visualize_bbox(pred_result, rgb, color, gts, mask, depth):
    im_H, im_W = rgb.shape[0], rgb.shape[1]
    coord_2d = get_2d_coord_np(im_W, im_H)
    for j in range(len(pred_result['gt_class_ids'])):
        gt_scale = pred_result['gt_scales'][j]
        gt_RT = pred_result['gt_RTs'][j]
        gt_cls_id = pred_result['gt_class_ids'][j]
        pred_scale = None
        pred_RT = None
        pred_scales = []
        pred_RTs = []
        for kk in range(len(pred_result['pred_class_ids'])):
            if pred_result['pred_class_ids'][kk] == gt_cls_id:
                pred_scale = pred_result['pred_scales'][kk]
                pred_RT = pred_result['pred_RTs'][kk]
                pred_scales.append(pred_scale)
                pred_RTs.append(pred_RT)
        if len(pred_scales) > 1:
            max_iou = 0
            best_scale = best_RT = None
            for kk in range(len(pred_scales)):
                pred_scale = pred_scales[kk]
                pred_RT = pred_RTs[kk]
                iou = compute_3d_iou_new(pred_RT, gt_RT, pred_scale, gt_scale, handle_visibility=pred_result['gt_handle_visibility'][j],
                                         class_name_1=synset_names[gt_cls_id], class_name_2=synset_names[gt_cls_id])
                if iou > max_iou:
                    max_iou = iou
                    best_scale, best_RT = pred_scale, pred_RT
            pred_scale, pred_RT = best_scale, best_RT
        elif len(pred_scales) == 0:
            continue
        if gt_cls_id in [1, 2, 4] or (gt_cls_id == 6 and (pred_result['gt_handle_visibility'][j] == 0)):
            pred_RT = find_similar_RT_by_R_error(pred_RT, gt_RT)
        inst_id = j + 1
        mask_target = mask.copy().astype(np.float)
        mask_target[mask != inst_id] = 0.0
        mask_target[mask == inst_id] = 1.0
        pcl, mask_fuse = img2pcl(mask_target, depth, cam_K, coord_2d)
        pcl = pcl.T / 1000
        pred_RT_inv = inverse_RT(pred_RT)
        pcl_cano_pred = transform_coordinates_3d(pcl, pred_RT_inv)
        pcl_pred = transform_coordinates_3d(pcl_cano_pred, gt_RT)
        pcl_delta = pcl - pcl_pred
        dist_delta = np.linalg.norm(pcl_delta.T, axis=1)
        # keypoints_2d = compute_keypoints(pred_scale, pred_RT)
        # for face in FACES:
        #     cv2.polylines(rgb,[keypoints_2d[face]],True,color)
        rgb[mask_fuse > 0, 0] = dist_delta * 100
        rgb[mask_fuse > 0, 1:] = 0

    return rgb


def visualize_bbox_detection(rgb, bbox, color):
    cv2.rectangle(rgb, bbox[0,1], bbox[2,3], color)
    return rgb

pred_results_total = mmcv.load('/data2/zrd/GPV_pose_result/1029A/all_pred_result.pkl')
pred_results_dualposenet = mmcv.load('/data2/zrd/GPV_pose_result/dualposenet_results/REAL275_results.pkl')
pred_results_origin = mmcv.load('/data2/zrd/GPV_pose_result/dualposenet_results/REAL275_results.pkl')

cam_K = real_intrinsics = np.array([[591.0125, 0, 322.525], [0, 590.16775, 244.11084], [0, 0, 1]], dtype=np.float)
pred_results_total_dict = {}
for result in pred_results_total:
    pred_results_total_dict[result['image_path']] = result
pred_results_dualposenet_dict = {}
for result in pred_results_dualposenet:
    pred_results_dualposenet_dict[result['image_path']] = result
dataset_dir = '/data2/zrd/datasets/NOCS'
save_dir = '/data2/zrd/GPV_pose_result/visualize_pcl_debug'
detection_dir = '/data2/zrd/datasets/NOCS/detection_dualposenet/data/segmentation_results/REAL275/results_test_'

for img_path in tqdm(pred_results_total_dict.keys()):
    pred_result = pred_results_total_dict[img_path]
    detection_name = img_path.split('/')[-2:]
    detection_name = '_'.join(detection_name)
    detection_path = detection_dir + detection_name + '.pkl'
    detection_origin_dict = mmcv.load(detection_path)
    real_img_path = os.path.join(dataset_dir, 'Real', img_path[10:])
    rgb = cv2.imread(real_img_path + '_color.png')
    gts = mmcv.load(real_img_path + '_label.pkl')
    mask = cv2.imread(real_img_path + '_mask.png')
    depth = load_depth(real_img_path + '_depth.png')
    mask = mask[:,:,2]
    im_H, im_W = rgb.shape[0], rgb.shape[1]
    rgb_origin = rgb.copy()
    rgb = visualize_bbox(pred_result, rgb, (0,255,255), gts, mask, depth)
    pred_result_dualposenet = pred_results_dualposenet_dict[img_path]
    # rgb = visualize_bbox(pred_result_dualposenet, rgb, (255,0,0))

    # for j in range(len(pred_result['gt_class_ids'])):
    #     gt_scale = pred_result['gt_scales'][j]
    #     gt_RT = pred_result['gt_RTs'][j]
    #     keypoints_2d = compute_keypoints(gt_scale, gt_RT)
    #     for face in FACES:
    #         cv2.polylines(rgb,[keypoints_2d[face]],True,(255,255,255))

    save_path = os.path.join(save_dir, img_path.replace('/', '_'))+'_box.png'
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))
    cv2.imwrite(save_path, rgb)
    plt.imshow(rgb)
    plt.show()
