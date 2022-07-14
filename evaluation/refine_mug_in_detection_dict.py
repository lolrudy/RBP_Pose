import numpy
import mmcv
import os
import _pickle as cPickle
import numpy as np
from evaluation.eval_utils_cass import get_3d_bbox, transform_coordinates_3d, compute_3d_iou_new
from tqdm import tqdm

def get_origin_scale(model, nocs_scale):
    lx = 2 * max(max(model[:, 0]), -min(model[:, 0]))
    ly = max(model[:, 1]) - min(model[:, 1])
    lz = max(model[:, 2]) - min(model[:, 2])

    # real scale
    lx_t = lx * nocs_scale
    ly_t = ly * nocs_scale
    lz_t = lz * nocs_scale
    return np.array([lx_t, ly_t, lz_t])

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

data_dir = '/data/zrd/datasets/NOCS'
detection_dir = os.path.join(data_dir, 'detection_dualposenet/data/segmentation_results')
detection_dir_refine_mug = os.path.join(data_dir, 'detection_dualposenet/data/segmentation_results_refine_for_mug')
dataset_split = 'REAL275' # 'CAMERA25'
img_list_path = os.path.join(data_dir, 'Real/test_list.txt')
img_list = [os.path.join(img_list_path.split('/')[0], line.rstrip('\n'))
                         for line in open(os.path.join(data_dir, img_list_path))]
cat_name2id = {'bottle': 1, 'bowl': 2, 'camera': 3, 'can': 4, 'laptop': 5, 'mug': 6}
with open(os.path.join(data_dir, 'obj_models/mug_meta.pkl'), 'rb') as f:
    mug_meta = cPickle.load(f)

models = {}
model_file_path = ['obj_models/real_test.pkl']
for path in model_file_path:
    with open(os.path.join(data_dir, path), 'rb') as f:
        models.update(cPickle.load(f))

for img_path in tqdm(img_list):
    img_path = os.path.join(data_dir, 'Real', img_path)

    scene = img_path.split('/')[-2]
    img_id = img_path.split('/')[-1]
    detection_file = os.path.join(detection_dir, dataset_split, f'results_test_{scene}_{img_id}.pkl')
    with open(detection_file, 'rb') as f:
        detection_dict = cPickle.load(f)
    with open(img_path + '_label.pkl', 'rb') as f:
        gts = cPickle.load(f)

    mug_idx = []
    for idx_gt in range(len(gts['class_ids'])):
        gt_cat_id = gts['class_ids'][idx_gt]  # convert to 0-indexed
        if gt_cat_id == cat_name2id['mug']:
            mug_idx.append(idx_gt)

    mug_idx_detection = []
    for idx_gt in range(len(detection_dict['gt_class_ids'])):
        gt_cat_id = detection_dict['gt_class_ids'][idx_gt]  # convert to 0-indexed
        if gt_cat_id == cat_name2id['mug']:
            mug_idx_detection.append(idx_gt)

    previous_detection_idx = None
    for idx_gt in mug_idx:
        max_iou = 0
        max_detection_idx = None
        rotation = gts['rotations'][idx_gt]
        translation = gts['translations'][idx_gt]
        model = models[gts['model_list'][idx_gt]].astype(np.float32)  # 1024 points
        nocs_scale = gts['scales'][idx_gt]  # nocs_scale = image file / model file
        scale = get_origin_scale(model, 1)
        model_name = gts['model_list'][idx_gt]
        # T0_mug = mug_meta[model_name][0]
        # s0_mug = mug_meta[model_name][1]
        RT = np.zeros((4, 4))
        RT[:3, :3] = rotation * nocs_scale
        RT[:3, 3] = translation
        RT[3, 3] = 1
        for idx_detection in mug_idx_detection:
            iou = asymmetric_3d_iou(detection_dict['gt_RTs'][idx_detection], RT,
                                    detection_dict['gt_scales'][idx_detection], scale)
            if iou > max_iou:
                max_iou = iou
                max_detection_idx = idx_detection
        detection_dict['gt_RTs'][max_detection_idx] = RT
        detection_dict['gt_scales'][max_detection_idx] = scale
        assert max_detection_idx != previous_detection_idx
        previous_detection_idx = max_detection_idx

    detection_file_refine_mug = os.path.join(detection_dir_refine_mug, dataset_split, f'results_test_{scene}_{img_id}.pkl')
    if not os.path.exists(os.path.dirname(detection_file_refine_mug)):
        os.makedirs(os.path.dirname(detection_file_refine_mug))
    with open(detection_file_refine_mug, 'wb') as f:
        cPickle.dump(detection_dict, f)