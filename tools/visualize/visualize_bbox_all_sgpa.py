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
        pred_scales = []
        pred_RTs = []
        for kk in range(len(pred_result['pred_class_ids'])):
            if pred_result['pred_class_ids'][kk] == gt_cls_id:
                pred_scale = pred_result['pred_scales'][kk]
                pred_RT = pred_result['pred_RTs'][kk]
                pred_scales.append(pred_scale)
                pred_RTs.append(pred_RT)
        if len(pred_scales) >= 1:
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
            if gt_cls_id == 1:
                threshold = 0.6
            else:
                threshold = 0.25
            if max_iou >= threshold:
                pred_scale, pred_RT = best_scale, best_RT
            else:
                best_match_for_gt.append([None, None])
                continue
        elif len(pred_scales) == 0:
            best_match_for_gt.append([None, None])
            continue
        if refine_mug and gt_cls_id == 6:
            gt_path = os.path.join('/data/zrd/datasets/NOCS/', img_path + '_label.pkl')
            with open(gt_path, 'rb') as f:
                gts = cPickle.load(f)
            mug_idx = []
            for idx_gt in range(len(gts['class_ids'])):
                gt_cat_id = gts['class_ids'][idx_gt]
                if gt_cat_id == 6:
                    mug_idx.append(idx_gt)
            if len(mug_idx) > 1:
                max_iou = 0
                model_name = None
                for idx_gt in mug_idx:
                    rotation = gts['rotations'][idx_gt]
                    translation = gts['translations'][idx_gt]
                    scale = gts['size'][idx_gt] * gts['scales'][idx_gt]
                    model_name = gts['model_list'][idx_gt]
                    T0_mug = mug_meta[model_name][0]
                    s0_mug = mug_meta[model_name][1]
                    origin_scale = scale * s0_mug
                    # origin_scale = scale.copy()
                    # origin_scale[0] = origin_scale[0] * s0_mug

                    origin_translation = translation + origin_scale * rotation @ T0_mug
                    RT = np.zeros((4, 4))
                    RT[:3, :3] = rotation
                    RT[:3, 3] = origin_translation
                    RT[3, 3] = 1
                    iou = asymmetric_3d_iou(gt_RT, RT, gt_scale, origin_scale)
                    if iou > max_iou:
                        max_iou = iou
                        model_name = gts['model_list'][idx_gt]
                T0_mug = mug_meta[model_name][0]
                s0_mug = mug_meta[model_name][1]
            elif len(mug_idx) == 0:
                T0_mug = np.array([0., 0., 0.])
                s0_mug = 1
            else:
                idx_gt = mug_idx[0]
                model_name = gts['model_list'][idx_gt]
                T0_mug = mug_meta[model_name][0]
                s0_mug = mug_meta[model_name][1]
            pred_scale = pred_scale * s0_mug
            delta_t = pred_scale * pred_RT[:3, :3] @ T0_mug
            pred_RT[:3, 3] = pred_RT[:3, 3] + delta_t
        if gt_cls_id in [1, 2, 4] or (gt_cls_id == 6 and (pred_result['gt_handle_visibility'][j] == 0)):
            pred_RT = find_similar_RT_by_R_error(pred_RT, gt_RT)

        best_match_for_gt.append([pred_RT, pred_scale])
    return best_match_for_gt

def visualize_bbox_detection(rgb, bbox, color):
    cv2.rectangle(rgb, bbox[0,1], bbox[2,3], color)
    return rgb



pred_results_total = mmcv.load('/data/zrd/RBP_result/0219/all_pred_result.pkl')

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

for img_path in tqdm(pred_results_total_dict.keys()):
    pred_result = pred_results_total_dict[img_path]
    detection_name = img_path.split('/')[-2:]
    detection_name = '_'.join(detection_name)
    detection_path = detection_dir + detection_name + '.pkl'
    detection_origin_dict = mmcv.load(detection_path)
    sgpa_result_path = os.path.join(sgpa_result_dir, 'results_test_'+detection_name+'.pkl')
    sgpa_result = mmcv.load(sgpa_result_path)
    real_img_path = os.path.join(dataset_dir, 'Real', img_path[10:])
    rgb = cv2.imread(real_img_path + '_color.png')
    im_H, im_W = rgb.shape[0], rgb.shape[1]
    rgb_origin = rgb.copy()
    rgb_ours = rgb.copy()
    rgb_sgpa = rgb.copy()
    rgb_gt = rgb.copy()
    # mask = detection_origin_dict['pred_masks'][:, :, j]
    # bbox = pred_result['pred_bboxes'][j]
    # rmin, rmax, cmin, cmax = get_bbox(bbox)
    # # here resize and crop to a fixed size 256 x 256
    # bbox_xyxy = np.array([cmin, rmin, cmax, rmax])
    # x1, y1, x2, y2 = bbox_xyxy
    # # here resize and crop to a fixed size 256 x 256
    # cx = 0.5 * (x1 + x2)
    # cy = 0.5 * (y1 + y2)
    # bbox_center = np.array([cx, cy])  # (w/2, h/2)
    # bbox_scale = max(y2 - y1, x2 - x1)
    # bbox_scale = min(bbox_scale, max(im_H, im_W)) * 1.0
    #
    ## roi_image ------------------------------------
    # roi_img = crop_resize_by_warp_affine(
    #     rgb, bbox_center, bbox_scale, 256, interpolation=cv2.INTER_NEAREST
    #     )
    try:
        best_match_for_gt = find_best_match_for_gt(pred_result)
        best_match_for_gt_sgpa = find_best_match_for_gt(sgpa_result)
        save_path = os.path.join(save_dir, img_path.replace('/', '_'))
        if not os.path.exists(os.path.dirname(save_path)):
            os.makedirs(os.path.dirname(save_path))
        for j in range(len(pred_result['gt_class_ids'])):
            pred_RT, pred_scale = best_match_for_gt[j]
            if pred_scale is not None:
                rgb = visualize_bbox(pred_scale, pred_RT, rgb, (0, 255, 255))
                rgb_ours = visualize_bbox(pred_scale, pred_RT, rgb_ours, (0, 255, 255))
            pred_RT, pred_scale = best_match_for_gt_sgpa[j]
            if pred_scale is not None:
                rgb = visualize_bbox(pred_scale, pred_RT, rgb, (255, 0, 0))
                rgb_sgpa = visualize_bbox(pred_scale, pred_RT, rgb_sgpa, (0, 255, 255))

            gt_scale = pred_result['gt_scales'][j]
            gt_RT = pred_result['gt_RTs'][j]
            rgb = visualize_bbox(gt_scale, gt_RT, rgb, (255, 255, 255))
            rgb_gt = visualize_bbox(gt_scale, gt_RT, rgb_gt, (0, 255, 255))
            np.savetxt(rgb)

        cv2.imwrite(save_path+'_box_all.png', rgb)
        cv2.imwrite(save_path+'_box_gt.png', rgb_gt)
        cv2.imwrite(save_path +'_box_sgpa.png', rgb_sgpa)
        cv2.imwrite(save_path+'_box_ours.png', rgb_ours)
    except:
        print(detection_name)

