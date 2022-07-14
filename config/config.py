from __future__ import print_function

import absl.flags as flags

flags.DEFINE_string("semantic_encoder_name", 'psp_net', 'select a backbone')

# datasets
flags.DEFINE_integer('obj_c', 6, 'number of categories')
flags.DEFINE_string('dataset', 'Real', 'CAMERA or CAMERA+Real')
flags.DEFINE_string('dataset_dir', '/data/zrd/datasets/NOCS', 'path to the dataset')
flags.DEFINE_string('per_obj', '', 'only train an specified object')
flags.DEFINE_integer('ban_mug', 0, 'not include mug if true')
flags.DEFINE_float('DZI_PAD_SCALE', 1.5, '')
flags.DEFINE_string('DZI_TYPE', 'uniform', '')
flags.DEFINE_float('DZI_SCALE_RATIO', 0.25, '')
flags.DEFINE_float('DZI_SHIFT_RATIO', 0.25, '')

# input parameters
flags.DEFINE_integer("img_size", 256, 'size of the cropped image')

flags.DEFINE_string("train_stage", 'shape_prior_only', '')

# data aug parameters
flags.DEFINE_integer('roi_mask_r', 3, 'radius for mask aug')
flags.DEFINE_float('roi_mask_pro', 0.5, 'probability to augment mask')
flags.DEFINE_float('aug_pc_pro', 0.2, 'probability to augment pc')
flags.DEFINE_float('aug_pc_r', 0.002, 'change 2mm on pc')
flags.DEFINE_float('aug_rt_pro', 0.3, 'probability to augment rt')
flags.DEFINE_float('aug_bb_pro', 0.3, 'probability to augment size')
flags.DEFINE_float('aug_bc_pro', 0.3, 'box cage based augmentation, only valid for bowl, mug')
flags.DEFINE_float('aug_nl_pro', 0.0, 'non-linear augmentation')

# rgb aug
flags.DEFINE_string('color_aug_type', 'cosy+aae', '')
flags.DEFINE_float('color_aug_prob', 0.8, '')
flags.DEFINE_string('color_aug_code', '', '')
flags.DEFINE_bool('COLOR_AUG_SYN_ONLY', False, '')

# pose network
flags.DEFINE_integer('feat_pcl', 1286, 'channel of point cloud feature')
flags.DEFINE_integer('feat_global_pcl', 512, 'channel of global point cloud feature')
flags.DEFINE_integer('feat_seman', 32, 'semantic feature output channel')
flags.DEFINE_integer('R_c', 4, 'output channel of rotation, here confidence(1)+ rot(3)')
flags.DEFINE_integer('Ts_c', 6,  'output channel of translation (3) + size (3)')
flags.DEFINE_integer('feat_face',768, 'input channel of the face recon')

flags.DEFINE_integer('face_recon_c', 6 * 5, 'for every point, we predict its distance and normal to each face')
#  the storage form is 6*3 normal, then the following 6 parametes distance, the last 6 parameters confidence
flags.DEFINE_integer('gcn_sup_num', 7, 'support number for gcn')
flags.DEFINE_integer('gcn_n_num', 10, 'neighbor number for gcn')

# point selection
flags.DEFINE_integer('support_points', 516, 'number of points that selected for 6 faces')
flags.DEFINE_integer('random_points', 512, 'number of points selected randomly')
flags.DEFINE_string('sample_method', 'basic', 'basic or balance or b+b, seperate training, basic, in joint training, balance')
flags.DEFINE_integer('per_face_n_of_N', 516 // 6 * 3, 'randomly select 516 // 6 points from per_f...poinnts')
# backbone

# train parameters
# train##################################################
flags.DEFINE_integer("train", 1, "1 for train mode")
# flags.DEFINE_integer('eval', 0, '1 for eval mode')
flags.DEFINE_string('device', 'cuda:0', '')
# flags.DEFINE_string("train_gpu", '0', "gpu no. for training")
flags.DEFINE_integer("num_workers", 8, "cpu cores for loading dataset")
flags.DEFINE_integer('batch_size', 32, '')
flags.DEFINE_integer('total_epoch', 150, 'total epoches in training')
flags.DEFINE_integer('train_size', 16000, 'number of images in each epoch')
#####################space is not enough, trade time for space####################
flags.DEFINE_integer('accumulate', 1, '')   # the real batch size is batchsize x accumulate


# ground truth point mask distance threshold
flags.DEFINE_float('point_mask_distance_threshold', 0.1, '')
flags.DEFINE_float('point_mask_conf_threshold', 0.7, '')
flags.DEFINE_integer('point_mask_min_threshold', 100, '')
flags.DEFINE_string('prior_nocs_size', 'mean', 'mean / gt / pred')
flags.DEFINE_integer('use_gt_point_mask', 1, '')
flags.DEFINE_integer('use_point_conf_for_vote', 1, '')
flags.DEFINE_integer('use_seman_feat', 0, '')
flags.DEFINE_float('drop_seman_prob', 0.5, '')
flags.DEFINE_integer('use_shape_prior_loss', 1, '')
flags.DEFINE_float('balance_weight_uncertainty_shift', 0.01, '')
flags.DEFINE_integer('use_global_feat_for_ts', 0, '')
flags.DEFINE_integer('predict_delta_distance', 1, '')
flags.DEFINE_integer('predict_uncertainty', 0, '')
flags.DEFINE_integer('detach_prior_shift', 0, '')
flags.DEFINE_integer('use_rectify_normal', 0, '')

# for different losses
flags.DEFINE_string('fsnet_loss_type', 'l1', 'l1 or smoothl1')

flags.DEFINE_float('rot_1_w', 8.0, '')
flags.DEFINE_float('rot_2_w', 8.0, '')
flags.DEFINE_float('rot_regular', 4.0, '')
flags.DEFINE_float('tran_w', 8.0, '')
flags.DEFINE_float('size_w', 8.0, '')
flags.DEFINE_float('recon_w', 8.0, 'only for fsnet')
flags.DEFINE_float('r_con_w', 1.0, '')

flags.DEFINE_float('recon_shift_w', 3.0, 'shift estimation loss')
flags.DEFINE_float('recon_v_w', 1.0, 'voting loss weight')
flags.DEFINE_float('recon_bb_r_w', 1.0, 'bbox r loss')
flags.DEFINE_float('recon_bb_t_w', 1.0, 'bbox t loss')
flags.DEFINE_float('recon_bb_s_w', 1.0, 'bbox s loss')
flags.DEFINE_float('recon_bb_self_w', 1.0, 'bb self')
flags.DEFINE_float('recon_consistency_w', 1.0, '')
flags.DEFINE_float('recon_reg_delta_w', 0.001, '')
flags.DEFINE_float('recon_point_mask_w', 1.0, '')
flags.DEFINE_float('recon_con_w', 1.0, '')

flags.DEFINE_float('mask_w', 1.0, 'obj_mask_loss')

flags.DEFINE_float('prop_p_w', 1.0, 'geo point mathcing loss')
flags.DEFINE_float('prop_pm_w', 2.0, '')
flags.DEFINE_float('prop_sym_w', 1.0, 'important for symmetric objects, can do point aug along reflection plane')
flags.DEFINE_float('prop_r_reg_w', 1.0, 'rot confidence must be sum to 1')

flags.DEFINE_float('prior_corr_wt', 10.0, 'nocs coordinate loss')
flags.DEFINE_float('prior_cd_wt', 20.0, 'chamfer distance loss between ground truth shape and predicted full shape')
flags.DEFINE_float('prior_entropy_wt', 0.0001, 'entropy loss for assign matrix')
flags.DEFINE_float('prior_deform_wt', 0.05, 'regularization loss for deformation field')
flags.DEFINE_float('prior_sym_wt', 0.0, '')
flags.DEFINE_float('prior_corr_threshold', 0.03, '')
flags.DEFINE_integer('prior_corr_sym', 1, 'use symmetry aware corr loss')

flags.DEFINE_float('consistency_beta', 0.05, '')
flags.DEFINE_float('consistency_w', 3.0, '')

# training parameters
# learning rate scheduler
flags.DEFINE_float('lr', 1e-4, '')
 # initial learning rate w.r.t basic lr
flags.DEFINE_float('lr_pose', 1.0, '')
flags.DEFINE_integer('lr_decay_iters', 50, '')  # some parameter for the scheduler
### optimizer  ####
flags.DEFINE_string('lr_scheduler_name', 'flat_and_anneal', 'linear/warm_flat_anneal/')
flags.DEFINE_string('anneal_method', 'cosine', '')
flags.DEFINE_float('anneal_point', 0.72, '')
flags.DEFINE_string('optimizer_type', 'Ranger', '')
flags.DEFINE_float('weight_decay', 0.0, '')
flags.DEFINE_float('warmup_factor', 0.001, '')
flags.DEFINE_integer('warmup_iters', 1000, '')
flags.DEFINE_string('warmup_method', 'linear', '')
flags.DEFINE_float('gamma', 0.1, '')
flags.DEFINE_float('poly_power', 0.9, '')

# save parameters
flags.DEFINE_integer('save_every', 10, '')  # save models every 'save_every' epoch
flags.DEFINE_integer('log_every', 50, '')  # save log file every 100 iterations
flags.DEFINE_string('model_save', 'output/modelsave_all', 'path to save checkpoint')
# resume
flags.DEFINE_integer('resume', 0, '1 for resume, 0 for training from the start')
flags.DEFINE_string('resume_model', '', 'path to the saved model')
flags.DEFINE_integer('resume_prior_only', 1, '1 for resume only prior related module')
flags.DEFINE_integer('resume_point', 0, 'the epoch to continue the training')

###################for evaluation#################
flags.DEFINE_integer('eval_gt_mask', 0, 'use gt mask for evaluation')
flags.DEFINE_integer('eval_refine_mug', 1, 'refine mug when evaluation')
flags.DEFINE_integer('eval_visualize_pcl', 0, 'save pcl when evaluation')
flags.DEFINE_integer('eval_inference_only', 0, 'inference without evaluation')
flags.DEFINE_integer('eval_coord', 0, 'pose from coordinate')
flags.DEFINE_integer('eval_recon', 0, 'reconstruction quality')
flags.DEFINE_integer('eval_precise', 0, '')
