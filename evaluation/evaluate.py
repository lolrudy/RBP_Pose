import os
import torch
import random
from network.SelfSketchPoseNet import SelfSketchPoseNet as SSPN
from tools.geom_utils import generate_RT, generate_sRT
from config.config import *
from absl import app

FLAGS = flags.FLAGS
from evaluation.load_data_eval import SketchPoseDataset
from evaluation.load_data_eval_gt_mask import SketchPoseDatasetGTMask
import torch.nn as nn
import numpy as np
import time

# from creating log
import tensorflow as tf
import evaluation
from evaluation.eval_utils import setup_logger, compute_mAP
from evaluation.eval_utils_cass import compute_degree_cm_mAP
from tqdm import tqdm

device = 'cuda'


def evaluate(argv):
    if not os.path.exists(FLAGS.model_save):
        os.makedirs(FLAGS.model_save)
    tf.compat.v1.disable_eager_execution()
    logger = setup_logger('eval_log', os.path.join(FLAGS.model_save, 'log_eval.txt'))
    Train_stage = FLAGS.train_stage
    FLAGS.train = False

    model_name = os.path.basename(FLAGS.resume_model).split('.')[0]
    # build dataset annd dataloader
    if FLAGS.eval_gt_mask:
        val_dataset = SketchPoseDatasetGTMask(source=FLAGS.dataset, mode='test')
        output_path = os.path.join(FLAGS.model_save, f'eval_result_{model_name}_gt_mask')
    else:
        val_dataset = SketchPoseDataset(source=FLAGS.dataset, mode='test')
        output_path = os.path.join(FLAGS.model_save, f'eval_result_{model_name}')
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    import pickle
    if FLAGS.eval_coord:
        pred_result_save_path = os.path.join(output_path, 'pred_result_coord.pkl')
    elif FLAGS.eval_recon:
        pred_result_save_path = os.path.join(output_path, 'pred_result_recon.pkl')
    else:
        pred_result_save_path = os.path.join(output_path, 'pred_result.pkl')
    if os.path.exists(pred_result_save_path) and False:
        with open(pred_result_save_path, 'rb') as file:
            pred_results = pickle.load(file)
    else:
        network = SSPN(Train_stage)
        # I use only one gpu for training
        network = network.to(device)

        if FLAGS.resume:
            model_dict = network.state_dict()
            resume_model_dict = torch.load(FLAGS.resume_model)
            if FLAGS.resume_prior_only:
                keys = list(resume_model_dict.keys())
                for key in keys:
                    if 'face_recon' in key:
                        resume_model_dict.pop(key)
                    elif 'pcl_encoder_prior' in key:
                        resume_model_dict.pop(key)
            model_dict.update(resume_model_dict)
            network.load_state_dict(model_dict)
        else:
            raise NotImplementedError
        # start to test
        network = network.eval()
        pred_results = []
        for i, data in tqdm(enumerate(val_dataset, 1)):
            if data is None:
                continue
            data_origin, detection_dict, gts = data
            origin_object_num = len(data_origin['cat_id_0base'])
            batch_per_max = 4
            batch_step_num = np.ceil(origin_object_num / batch_per_max)
            batch_step_num = int(batch_step_num)
            pred_s_fuse = np.zeros((origin_object_num, 3))
            pred_RT_fuse = np.zeros((origin_object_num, 4, 4))
            pred_model_fuse = np.zeros((origin_object_num, 1024, 3))

            for step_i in range(batch_step_num):
                data = {}
                for key in data_origin.keys():
                    data[key] = data_origin[key][step_i * batch_per_max: min(origin_object_num, (step_i+1) * batch_per_max)]

                sym = data['sym_info'].to(device)
                if len(data['cat_id_0base']) == 0:
                    detection_dict['pred_RTs'] = np.zeros((0, 4, 4))
                    detection_dict['pred_scales'] = np.zeros((0, 4, 4))
                    pred_results.append(detection_dict)
                    continue
                output_dict \
                    = network(rgb=data['roi_img'].to(device), depth=data['roi_depth'].to(device),
                              depth_normalize=data['depth_normalize'].to(device),
                              obj_id=data['cat_id_0base'].to(device), camK=data['cam_K'].to(device),
                              gt_mask=data['roi_mask'].to(device),
                              mean_shape=data['mean_shape'].to(device),
                              gt_2D=data['roi_coord_2d'].to(device), sym=sym,
                              def_mask=data['roi_mask'].to(device), shape_prior=data['shape_prior'].to(device))
                # mask = output_dict['mask']
                # mask = mask.detach().cpu().numpy()
                # roi_img = data['roi_img'].detach().cpu().numpy()
                # gt_mask = data['roi_mask'].squeeze(axis=1).detach().cpu().numpy()
                # bs = mask.shape[0]
                # for i_batch in range(bs):
                #     mask_i = mask[i_batch]
                #     mask_i = mask_i[1]
                #     mask_i_bool = mask_i > 0.5
                #     gt_mask_i = gt_mask[i_batch]
                #     img_i = roi_img[i_batch]
                #     img_i = np.swapaxes(img_i, 0, 1)
                #     img_i = np.swapaxes(img_i, 1, 2)
                #     import matplotlib.pyplot as plt
                #     plt.imshow(mask_i.astype(float))
                #     plt.show()
                #     plt.imshow(mask_i_bool.astype(float))
                #     plt.show()
                #     plt.imshow(gt_mask_i.astype(float))
                #     plt.show()
                #     plt.imshow(img_i.astype(int))
                #     plt.show()
                # continue
                if not FLAGS.eval_coord:
                    p_green_R_vec = output_dict['p_green_R'].detach()
                    p_red_R_vec = output_dict['p_red_R'].detach()
                    p_T = output_dict['Pred_T'].detach()
                    p_s = output_dict['Pred_s'].detach()
                    f_green_R = output_dict['f_green_R'].detach()
                    f_red_R = output_dict['f_red_R'].detach()

                    pred_s = p_s
                    pred_RT = generate_RT([p_green_R_vec, p_red_R_vec], [f_green_R, f_red_R], p_T, mode='vec', sym=sym)
                    pred_RT = pred_RT.detach().cpu().numpy()
                    pred_s = pred_s.detach().cpu().numpy()
                else:
                    from tools.align_utils import estimateSimilarityTransform, estimateSimilarityUmeyama
                    PC = output_dict['PC'].detach().cpu().numpy()
                    nocs_coord = output_dict['NOCS_coord'].detach().cpu().numpy()
                    bs = PC.shape[0]
                    pred_RT = np.zeros((bs, 4, 4))
                    for i in range(bs):
                        scale, rotation, translation, pred_sRT = estimateSimilarityTransform(nocs_coord[i], PC[i])
                        pred_sRT[:3, :3] = rotation
                        pred_RT[i] = pred_sRT
                    p_s = output_dict['Pred_s'].detach()
                    pred_s = p_s.detach().cpu().numpy()

                if FLAGS.eval_recon:
                    pred_model = output_dict['recon_model'].detach().cpu().numpy()
                    pred_model_fuse[step_i * batch_per_max: min(origin_object_num, (step_i + 1) * batch_per_max)] = pred_model

                pred_s_fuse[step_i * batch_per_max: min(origin_object_num, (step_i+1) * batch_per_max)] = pred_s
                pred_RT_fuse[step_i * batch_per_max: min(origin_object_num, (step_i+1) * batch_per_max)] = pred_RT
            pred_s = pred_s_fuse
            pred_RT = pred_RT_fuse
            if pred_RT is not None:
                detection_dict['pred_RTs'] = pred_RT
                detection_dict['pred_scales'] = pred_s
                if FLAGS.eval_recon:
                    detection_dict['recon_model'] = pred_model_fuse
            else:
                assert NotImplementedError
            pred_results.append(detection_dict)

        with open(pred_result_save_path, 'wb') as file:
            pickle.dump(pred_results, file)

    if FLAGS.eval_inference_only:
        import sys
        sys.exit()

    if FLAGS.eval_precise:
        degree_thres_list = list(range(0, 61, 1))
        shift_thres_list = [i / 2 for i in range(21)]
        iou_thres_list = [i / 100 for i in range(101)]
    else:
        degree_thres_list = list(range(0, 61, 5))
        shift_thres_list = [i * 1 for i in range(11)]
        iou_thres_list = [i / 20 for i in range(21)]

    #iou_aps, pose_aps, iou_acc, pose_acc = compute_mAP(pred_results, output_path, degree_thres_list, shift_thres_list,
    #                                                  iou_thres_list, iou_pose_thres=0.1, use_matches_for_pose=True,)
    if FLAGS.ban_mug:
        synset_names = ['BG'] + ['bottle', 'bowl', 'camera', 'can', 'laptop']
    else:
        synset_names = ['BG'] + ['bottle', 'bowl', 'camera', 'can', 'laptop', 'mug']
    if FLAGS.per_obj in synset_names:
        idx = synset_names.index(FLAGS.per_obj)
    else:
        idx = -1
    iou_aps, pose_aps = compute_degree_cm_mAP(pred_results, synset_names, output_path, degree_thres_list, shift_thres_list,
                              iou_thres_list, iou_pose_thres=0.1, use_matches_for_pose=True,)

    # # fw = open('{0}/eval_logs.txt'.format(result_dir), 'a')
    iou_25_idx = iou_thres_list.index(0.25)
    iou_50_idx = iou_thres_list.index(0.5)
    iou_75_idx = iou_thres_list.index(0.75)
    degree_05_idx = degree_thres_list.index(5)
    degree_10_idx = degree_thres_list.index(10)
    shift_02_idx = shift_thres_list.index(2)
    shift_05_idx = shift_thres_list.index(5)
    shift_10_idx = shift_thres_list.index(10)

    messages = []

    if FLAGS.per_obj in synset_names:
        messages.append('mAP:')
        messages.append('3D IoU at 25: {:.1f}'.format(iou_aps[idx, iou_25_idx] * 100))
        messages.append('3D IoU at 50: {:.1f}'.format(iou_aps[idx, iou_50_idx] * 100))
        messages.append('3D IoU at 75: {:.1f}'.format(iou_aps[idx, iou_75_idx] * 100))
        messages.append('5 degree, 2cm: {:.1f}'.format(pose_aps[idx, degree_05_idx, shift_02_idx] * 100))
        messages.append('5 degree, 5cm: {:.1f}'.format(pose_aps[idx, degree_05_idx, shift_05_idx] * 100))
        messages.append('10 degree, 2cm: {:.1f}'.format(pose_aps[idx, degree_10_idx, shift_02_idx] * 100))
        messages.append('10 degree, 5cm: {:.1f}'.format(pose_aps[idx, degree_10_idx, shift_05_idx] * 100))
        messages.append('10 degree, 10cm: {:.1f}'.format(pose_aps[idx, degree_10_idx, shift_10_idx] * 100))
    else:
        messages.append('average mAP:')
        messages.append('3D IoU at 25: {:.1f}'.format(iou_aps[idx, iou_25_idx] * 100))
        messages.append('3D IoU at 50: {:.1f}'.format(iou_aps[idx, iou_50_idx] * 100))
        messages.append('3D IoU at 75: {:.1f}'.format(iou_aps[idx, iou_75_idx] * 100))
        messages.append('5 degree, 2cm: {:.1f}'.format(pose_aps[idx, degree_05_idx, shift_02_idx] * 100))
        messages.append('5 degree, 5cm: {:.1f}'.format(pose_aps[idx, degree_05_idx, shift_05_idx] * 100))
        messages.append('10 degree, 2cm: {:.1f}'.format(pose_aps[idx, degree_10_idx, shift_02_idx] * 100))
        messages.append('10 degree, 5cm: {:.1f}'.format(pose_aps[idx, degree_10_idx, shift_05_idx] * 100))
        messages.append('10 degree, 10cm: {:.1f}'.format(pose_aps[idx, degree_10_idx, shift_10_idx] * 100))

        for idx in range(1, len(synset_names)):
            messages.append('category {}'.format(synset_names[idx]))
            messages.append('mAP:')
            messages.append('3D IoU at 25: {:.1f}'.format(iou_aps[idx, iou_25_idx] * 100))
            messages.append('3D IoU at 50: {:.1f}'.format(iou_aps[idx, iou_50_idx] * 100))
            messages.append('3D IoU at 75: {:.1f}'.format(iou_aps[idx, iou_75_idx] * 100))
            messages.append('5 degree, 2cm: {:.1f}'.format(pose_aps[idx, degree_05_idx, shift_02_idx] * 100))
            messages.append('5 degree, 5cm: {:.1f}'.format(pose_aps[idx, degree_05_idx, shift_05_idx] * 100))
            messages.append('10 degree, 2cm: {:.1f}'.format(pose_aps[idx, degree_10_idx, shift_02_idx] * 100))
            messages.append('10 degree, 5cm: {:.1f}'.format(pose_aps[idx, degree_10_idx, shift_05_idx] * 100))
            messages.append('10 degree, 10cm: {:.1f}'.format(pose_aps[idx, degree_10_idx, shift_10_idx] * 100))

    for msg in messages:
        logger.info(msg)

if __name__ == "__main__":
    app.run(evaluate)