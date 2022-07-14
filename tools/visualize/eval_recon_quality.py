import numpy as np
import mmcv
import os
import cv2
from tools.eval_utils import get_bbox
from tools.dataset_utils import crop_resize_by_warp_affine
import matplotlib.pyplot as plt
import math
from evaluation.eval_utils_cass import get_3d_bbox, transform_coordinates_3d, compute_3d_iou_new
from tqdm import tqdm
import torch
from losses.nn_distance.chamfer_loss import ChamferLoss
synset_names = ['BG'] + ['bottle', 'bowl', 'camera', 'can', 'laptop', 'mug']


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


FACES = [[0,1,5,4], [0,1,3,2], [2,3,7,6], [4,5,7,6], [0,2,6,4], [1,3,7,5]]
import _pickle as cPickle

with open(os.path.join('/data/zrd/datasets/NOCS', 'obj_models/mug_meta.pkl'), 'rb') as f:
    mug_meta = cPickle.load(f)

def visualize_bbox(pred_scale, pred_RT, rgb, color):
    rgb = rgb.copy()
    keypoints_2d = compute_keypoints(pred_scale, pred_RT)
    for face in FACES:
        cv2.polylines(rgb,[keypoints_2d[face]],True,color, thickness=2)
    return rgb

def find_best_match_for_gt(pred_result, refine_mug=False, img_path=None):
    best_match_for_gt = []
    for j in range(len(pred_result['gt_class_ids'])):
        gt_scale = pred_result['gt_scales'][j]
        gt_RT = pred_result['gt_RTs'][j]
        gt_cls_id = pred_result['gt_class_ids'][j]
        pred_scale = None
        pred_RT = None
        pred_model = None
        pred_scales = []
        pred_RTs = []
        pred_models = []
        for kk in range(len(pred_result['pred_class_ids'])):
            if pred_result['pred_class_ids'][kk] == gt_cls_id:
                pred_scale = pred_result['pred_scales'][kk]
                pred_RT = pred_result['pred_RTs'][kk]
                pred_model = pred_result['recon_model'][kk]
                pred_scales.append(pred_scale)
                pred_RTs.append(pred_RT)
                pred_models.append(pred_model)
        if len(pred_scales) >= 1:
            max_iou = 0
            best_scale = best_RT = best_model = None
            for kk in range(len(pred_scales)):
                pred_scale = pred_scales[kk]
                pred_RT = pred_RTs[kk]
                pred_model = pred_models[kk]
                iou = compute_3d_iou_new(pred_RT, gt_RT, pred_scale, gt_scale, handle_visibility=pred_result['gt_handle_visibility'][j],
                                         class_name_1=synset_names[gt_cls_id], class_name_2=synset_names[gt_cls_id])
                if iou > max_iou:
                    max_iou = iou
                    best_scale, best_RT, best_model = pred_scale, pred_RT, pred_model
            threshold = 0.75
            if max_iou >= threshold:
                pred_scale, pred_RT, pred_model = best_scale, best_RT, best_model
            else:
                best_match_for_gt.append([None, None, None])
                continue
        elif len(pred_scales) == 0:
            best_match_for_gt.append([None, None, None])
            continue
        else:
            iou = compute_3d_iou_new(pred_RT, gt_RT, pred_scale, gt_scale,
                                     handle_visibility=pred_result['gt_handle_visibility'][j],
                                     class_name_1=synset_names[gt_cls_id], class_name_2=synset_names[gt_cls_id])
            threshold = 0.75
            if iou < threshold:
                best_match_for_gt.append([None, None, None])
                continue

        if gt_cls_id in [1, 2, 4] or (gt_cls_id == 6 and (pred_result['gt_handle_visibility'][j] == 0)):
            pred_RT = find_similar_RT_by_R_error(pred_RT, gt_RT)

        best_match_for_gt.append([pred_RT, pred_scale, pred_model])
    return best_match_for_gt

def visualize_bbox_detection(rgb, bbox, color):
    cv2.rectangle(rgb, bbox[0,1], bbox[2,3], color)
    return rgb



pred_results_total = mmcv.load('/data/zrd/project/GPV_pose_shape_prior/output/recon_all_shift_nouncertainty/eval_result_model_149/pred_result_recon.pkl')

# pred_results_sgpa = mmcv.load('/data/zrd/datasets/NOCS/results/sgpa_results/REAL275_results.pkl')
# pred_results_origin = mmcv.load('/data/zrd/datasets/NOCS/results/sgpa_results/REAL275_results.pkl')

cam_K = real_intrinsics = np.array([[591.0125, 0, 322.525], [0, 590.16775, 244.11084], [0, 0, 1]], dtype=np.float)
pred_results_total_dict = {}
for result in pred_results_total:
    pred_results_total_dict[result['image_path']] = result
# pred_results_sgpa_dict = {}

dataset_dir = '/data/zrd/datasets/NOCS'
save_dir = '/data/zrd/RBP_result/visualize_bbox_all_separate_rbp_sgpa'
detection_dir = '/data/zrd/datasets/NOCS/detection_dualposenet/data/segmentation_results/REAL275/results_test_'
sgpa_result_dir = '/data/zrd/datasets/NOCS/results/sgpa_results/real'
model_file_path = os.path.join(dataset_dir, 'obj_models/real_test.pkl')
models = mmcv.load(model_file_path)
distance_func = ChamferLoss()
distance_list = []
for i in range(6):
    distance_list.append([])
if os.path.exists(os.path.join(save_dir, 'distance_dict.pkl')) and False:
    distance_list = mmcv.load(os.path.join(save_dir, 'distance_dict.pkl'))
else:
    for img_path in tqdm(pred_results_total_dict.keys()):
        pred_result = pred_results_total_dict[img_path]
        detection_name = img_path.split('/')[-2:]
        detection_name = '_'.join(detection_name)
        detection_path = detection_dir + detection_name + '.pkl'
        detection_origin_dict = mmcv.load(detection_path)
        gt_path = os.path.join(dataset_dir, img_path + '_label.pkl')
        gts = mmcv.load(gt_path)
        best_match_for_gt = find_best_match_for_gt(pred_result)
        save_path = os.path.join(save_dir, img_path.replace('/', '_'))
        if not os.path.exists(os.path.dirname(save_path)):
            os.makedirs(os.path.dirname(save_path))
        try:
            assert len(pred_result['gt_class_ids']) == len(gts['model_list'])
            for j in range(len(pred_result['gt_class_ids'])):
                model_name = gts['model_list'][j]
                gt_model = models[model_name]
                pred_RT, pred_scale, pred_model = best_match_for_gt[j]
                gt_cls_id = pred_result['gt_class_ids'][j]
                if pred_scale is not None:
                    pred_model_tensor = torch.tensor(pred_model).cuda().unsqueeze(0).float()
                    gt_model_tensor = torch.tensor(gt_model).cuda().unsqueeze(0).float()

                    dist, _, _ = distance_func(pred_model_tensor, gt_model_tensor)
                    # print(gt_cls_id)
                    distance_list[gt_cls_id - 1].append(dist.detach().cpu().item())
        except:
            print(detection_name)
    mmcv.dump(distance_list, os.path.join(save_dir, 'distance_dict.pkl'))
print(distance_list)
for i in range(6):
    print('class:')
    print(i)
    print('mean distance:')
    print(np.mean(distance_list[i]))



