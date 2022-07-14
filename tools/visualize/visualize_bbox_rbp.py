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
    keypoints_2d = keypoints_2d.T
    return keypoints_2d


FACES = [[0,1,5,4], [0,1,3,2], [2,3,7,6], [4,5,7,6], [0,2,6,4], [1,3,7,5]]

def visualize_bbox(pred_scale, pred_RT, rgb, color):
    rgb = rgb.copy()
    keypoints_2d = compute_keypoints(pred_scale, pred_RT)
    keypoints_2d = keypoints_2d.astype(int)
    for face in FACES:
        cv2.polylines(rgb,[keypoints_2d[face]],True,color)
    return rgb

def visualize_bbox_roi(pred_scale, pred_RT, rgb, color, bbox_center, bbox_scale):
    rgb = rgb.copy()
    keypoints_2d = compute_keypoints(pred_scale, pred_RT)
    keypoints_2d = (keypoints_2d - bbox_center) / bbox_scale * 256 + 128
    keypoints_2d = keypoints_2d.astype(int)
    for face in FACES:
        cv2.polylines(rgb,[keypoints_2d[face]],True,color, thickness=2)
    return rgb

def find_best_match_for_gt(pred_result):
    best_match_for_gt = []
    for j in range(len(pred_result['gt_class_ids'])):
        gt_scale = pred_result['gt_scales'][j]
        gt_RT = pred_result['gt_RTs'][j]
        gt_cls_id = pred_result['gt_class_ids'][j]
        pred_scale = None
        pred_RT = None
        pred_idx = None
        pred_scales = []
        pred_RTs = []
        pred_idxs = []

        for kk in range(len(pred_result['pred_class_ids'])):
            if pred_result['pred_class_ids'][kk] == gt_cls_id:
                pred_scale = pred_result['pred_scales'][kk]
                pred_RT = pred_result['pred_RTs'][kk]
                pred_idx = kk
                pred_scales.append(pred_scale)
                pred_RTs.append(pred_RT)
                pred_idxs.append(pred_idx)
        if len(pred_scales) > 1:
            max_iou = 0
            best_scale = best_RT = best_idx = None
            for kk in range(len(pred_scales)):
                pred_scale = pred_scales[kk]
                pred_RT = pred_RTs[kk]
                pred_idx = pred_idxs[kk]
                iou = compute_3d_iou_new(pred_RT, gt_RT, pred_scale, gt_scale, handle_visibility=pred_result['gt_handle_visibility'][j],
                                         class_name_1=synset_names[gt_cls_id], class_name_2=synset_names[gt_cls_id])
                if iou > max_iou:
                    max_iou = iou
                    best_scale, best_RT = pred_scale, pred_RT
                    best_idx = pred_idx
            pred_scale, pred_RT = best_scale, best_RT
            pred_idx = best_idx
        elif len(pred_scales) == 0:
            best_match_for_gt.append([None, None, None])
            continue
        if gt_cls_id in [1, 2, 4] or (gt_cls_id == 6 and (pred_result['gt_handle_visibility'][j] == 0)):
            pred_RT = find_similar_RT_by_R_error(pred_RT, gt_RT)
        best_match_for_gt.append([pred_RT, pred_scale, pred_idx])
    return best_match_for_gt

def visualize_bbox_detection(rgb, bbox, cate_id, score):
    colors = [(0,0,255), (0,255,255), (255,0,255), (255,0,0), (255,255,0), (0,255,0)]
    rgb = rgb.copy()
    cv2.rectangle(rgb, [bbox[1], bbox[0]], [bbox[3], bbox[2]], colors[cate_id-1], thickness=2)
    # score = str(score)[:5]
    # cv2.putText(rgb, score, [bbox[3], bbox[2]], cv2.FONT_HERSHEY_SIMPLEX, 0.3, colors[cate_id-1], 2)
    return rgb

def visualize_mask(rgb, mask, cate_id):
    colors = [(0,0,255), (0,255,255), (255,0,255), (255,0,0), (255,255,0), (0,255,0)]
    mask_template = rgb.copy()
    mask_template[mask] = colors[cate_id-1]
    rgb = rgb.copy()
    rgb_mask = cv2.addWeighted(rgb, 0.8, mask_template, 0.2, 0)
    # rgb[rgb>255] = 255
    return rgb_mask

def iou_bbox(bbox1, bbox2):
    x1, y1, x2, y2 = bbox1
    a1, b1, a2, b2 = bbox2
    union = (y2 - y1) * (x2 - x1) + (b2 - b1) * (a2 - a1)
    c1, d1, c2, d2 = max(x1, a1), max(y1, b1), min(x2, a2), min(y2, b2)
    intersection = (d2-d1) * (c2-c1)
    union = union - intersection
    return intersection / union

def find_best_detction_match_for_gt(pred_result):
    best_match_for_gt = []
    for j in range(len(pred_result['gt_class_ids'])):
        gt_cls_id = pred_result['gt_class_ids'][j]
        gt_bbox = pred_result['gt_bboxes'][j]
        pred_idx = None
        pred_idxs = []

        for kk in range(len(pred_result['pred_class_ids'])):
            if pred_result['pred_class_ids'][kk] == gt_cls_id:
                pred_idx = kk
                pred_idxs.append(pred_idx)
        if len(pred_idxs) > 1:
            max_iou = 0
            best_idx = None
            for kk in range(len(pred_idxs)):
                pred_idx = pred_idxs[kk]
                pred_bbox = pred_result['pred_bboxes'][pred_idx]
                iou = iou_bbox(gt_bbox, pred_bbox)
                if iou > max_iou:
                    max_iou = iou
                    best_idx = pred_idx
            pred_idx = best_idx
        elif len(pred_idxs) == 0:
            best_match_for_gt.append(None)
            continue
        best_match_for_gt.append(pred_idx)
    return best_match_for_gt

pred_results_total = mmcv.load('/data/zrd/RBP_result/0219/all_pred_result.pkl')
# pred_results_dualposenet = mmcv.load('/data/zrd/RBP_result/dualposenet_results/REAL275_results.pkl')
# pred_results_origin = mmcv.load('/data/zrd/RBP_result/dualposenet_results/REAL275_results.pkl')

cam_K = real_intrinsics = np.array([[591.0125, 0, 322.525], [0, 590.16775, 244.11084], [0, 0, 1]], dtype=np.float)
pred_results_total_dict = {}
for result in pred_results_total:
    pred_results_total_dict[result['image_path']] = result
pred_results_dualposenet_dict = {}
# for result in pred_results_dualposenet:
#     pred_results_dualposenet_dict[result['image_path']] = result
dataset_dir = '/data/zrd/datasets/NOCS'
save_dir = '/data/zrd/RBP_result/visualize_bbox_pick_rbp'

if not os.path.exists(save_dir):
    os.makedirs(save_dir)
detection_dir = '/data/zrd/datasets/NOCS/detection_dualposenet/data/segmentation_results/REAL275/results_test_'

sgpa_result_dir = '/data/zrd/datasets/NOCS/results/sgpa_results/real'
img_path_select = ['data/real/test/scene_1/0001', 'data/real/test/scene_2/0310', 'data/real/test/scene_2/0562',
                   'data/real/test/scene_3/0241', 'data/real/test/scene_4/0188', 'data/real/test/scene_5/0248',
                   'data/real/test/scene_6/0269',
                   ]
for img_path in tqdm(img_path_select):
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
    rgb_total = rgb.copy()
    rgb_bbox_total = rgb.copy()

    best_match_for_gt = find_best_match_for_gt(pred_result)
    pred_result_dualposenet = sgpa_result
    best_match_for_gt_dualposenet = find_best_match_for_gt(pred_result_dualposenet)
    best_match_mask_for_gt = find_best_detction_match_for_gt(detection_origin_dict)

    for j in range(len(pred_result['gt_class_ids'])):
        gt_cls_id = pred_result['gt_class_ids'][j]
        pred_RT, pred_scale, pred_idx = best_match_for_gt[j]
        if pred_idx is None:
            continue
        rgb = rgb_origin.copy()
        mask = detection_origin_dict['pred_masks'][:, :, pred_idx]
        bbox = pred_result['pred_bboxes'][pred_idx]
        # rgb_total = visualize_bbox_detection(rgb_total, bbox, gt_cls_id)
        # rgb_total = visualize_mask(rgb_total, mask)
        rmin, rmax, cmin, cmax = get_bbox(bbox)
        # here resize and crop to a fixed size 256 x 256
        bbox_xyxy = np.array([cmin, rmin, cmax, rmax])
        x1, y1, x2, y2 = bbox_xyxy
        # here resize and crop to a fixed size 256 x 256
        cx = 0.5 * (x1 + x2)
        cy = 0.5 * (y1 + y2)
        bbox_center = np.array([cx, cy])  # (w/2, h/2)
        bbox_scale = max(y2 - y1, x2 - x1)
        bbox_scale = min(bbox_scale, max(im_H, im_W)) * 1.5
        gt_scale = pred_result['gt_scales'][j]
        gt_RT = pred_result['gt_RTs'][j]
        rgb = crop_resize_by_warp_affine(
            rgb, bbox_center, bbox_scale, 256, interpolation=cv2.INTER_NEAREST
        )
        rgb_bbox_total = visualize_bbox(gt_scale, gt_RT, rgb_bbox_total, (255, 255, 255))
        rgb = visualize_bbox_roi(gt_scale, gt_RT, rgb, (255, 255, 255), bbox_center, bbox_scale)
        # rgb = visualize_bbox_roi(gt_scale, gt_RT, rgb, (255, 0, 0), bbox_center, bbox_scale)
        save_path = os.path.join(save_dir, img_path.replace('/', '_')) + f'_box_{j}_gt.png'
        cv2.imwrite(save_path, rgb)
        if pred_scale is not None:
            rgb_our = visualize_bbox_roi(pred_scale, pred_RT, rgb, (0, 255, 0), bbox_center, bbox_scale)
            rgb_bbox_total = visualize_bbox(pred_scale, pred_RT, rgb_bbox_total, (0, 255, 0))
            save_path = os.path.join(save_dir, img_path.replace('/', '_')) + f'_box_{j}_our.png'
            cv2.imwrite(save_path, rgb_our)
        pred_RT, pred_scale, pred_idx = best_match_for_gt_dualposenet[j]
        if pred_scale is not None:
            rgb_dpn = visualize_bbox_roi(pred_scale, pred_RT, rgb, (255, 0, 0), bbox_center, bbox_scale)
            rgb_bbox_total = visualize_bbox(pred_scale, pred_RT, rgb_bbox_total, (255, 0, 0))

            save_path = os.path.join(save_dir, img_path.replace('/', '_')) + f'_box_{j}_dpn.png'
            cv2.imwrite(save_path, rgb_dpn)

    for j in range(len(pred_result['gt_class_ids'])):
        gt_cls_id = pred_result['gt_class_ids'][j]
        pred_idx = best_match_mask_for_gt[j]
        if pred_idx is None:
            continue
        mask = detection_origin_dict['pred_masks'][:, :, pred_idx]
        rgb_total = visualize_mask(rgb_total, mask, gt_cls_id)

    for j in range(len(pred_result['gt_class_ids'])):
        gt_cls_id = pred_result['gt_class_ids'][j]
        pred_RT, pred_scale, pred_idx = best_match_for_gt[j]
        if pred_idx is None:
            continue
        mask = detection_origin_dict['pred_masks'][:, :, pred_idx]
        bbox = pred_result['pred_bboxes'][pred_idx]
        score = pred_result['pred_scores'][pred_idx]
        rgb_total = visualize_bbox_detection(rgb_total, bbox, gt_cls_id, score)

    save_path = os.path.join(save_dir, img_path.replace('/', '_')) + f'_box_total.png'
    cv2.imwrite(save_path, rgb_total)

    save_path = os.path.join(save_dir, img_path.replace('/', '_')) + f'_box_total_bbox.png'
    cv2.imwrite(save_path, rgb_bbox_total)