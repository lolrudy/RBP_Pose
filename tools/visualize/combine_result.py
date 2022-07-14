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

pick_list = ['scene_1/0097', 'scene_2/0500', 'scene_3/0438', 'scene_4/0040', 'scene_6/0217']
pick_idx_list = [(0,1,2), (0,1,2), (1,3,4), (1,2,5), (0,1,3)]
detection_dir = '/data2/zrd/datasets/NOCS/detection_dualposenet/data/segmentation_results/REAL275/results_test_'
img_path_prefix = 'data/real/test/'
dataset_dir = '/data2/zrd/datasets/NOCS'
result_dir = '/data2/zrd/GPV_pose_result/visualize_bbox_pick'
save_dir = '/data2/zrd/GPV_pose_result/visualize_bbox_pick_combine'

if not os.path.exists(result_dir):
    os.makedirs(result_dir)
pick_list = [img_path_prefix+item for item in pick_list]
blank_len = 20

for i, img_path in tqdm(enumerate(pick_list)):
    final_img = None
    fig_iou = plt.figure(figsize=(15, 10))
    save_path = os.path.join(save_dir, img_path.replace('/', '_')) + f'_box_combine.png'
    for ll, j in enumerate(pick_idx_list[i]):
        our_result_path = os.path.join(result_dir, img_path.replace('/', '_')) + f'_box_{j}_our.png'
        our_result_pic = cv2.imread(our_result_path)
        dpn_result_path = os.path.join(result_dir, img_path.replace('/', '_')) + f'_box_{j}_dpn.png'
        dpn_result_pic = cv2.imread(dpn_result_path)
        blank_space = np.ones((blank_len, 256, 3))*255
        column = np.vstack((dpn_result_pic, blank_space, our_result_pic))
        if final_img is None:
            final_img = column
        else:
            blank_space = np.ones((2*256+blank_len, blank_len, 3))*255
            final_img = np.hstack((final_img, blank_space, column))
    cv2.imwrite(save_path, final_img)