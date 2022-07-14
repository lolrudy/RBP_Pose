import os
import random

import mmcv
import torch
from absl import app

from config.config import *
from tools.training_utils import build_lr_rate, get_gt_v, build_optimizer, nan_to_num
from network.SelfSketchPoseNet import SelfSketchPoseNet as SSPNet

FLAGS = flags.FLAGS
from datasets.load_data_selfsketchpose import SketchPoseDataset
import numpy as np
import time

# from creating log
import tensorflow as tf
from tools.eval_utils import setup_logger
torch.autograd.set_detect_anomaly(True)
device = 'cuda'
import traceback

def train(argv):
    if not os.path.exists(FLAGS.model_save):
        os.makedirs(FLAGS.model_save)
        FLAGS.append_flags_into_file(os.path.join(FLAGS.model_save, 'flags.txt'))
    tf.compat.v1.disable_eager_execution()
    tb_writter = tf.compat.v1.summary.FileWriter(FLAGS.model_save)
    logger = setup_logger('train_log', os.path.join(FLAGS.model_save, 'log.txt'))
    # flags_key = vars(FLAGS)['__flags'].keys()
    # for key in flags_key:
    #     try:
    #         logger.info(key + ':' + str(eval('FLAGS.'+key)))
    #     except:
    #         pass
    Train_stage = FLAGS.train_stage
    network = SSPNet(Train_stage)
    network = network.to(device)
    # resume or not
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
        else:
            keys = list(resume_model_dict.keys())
            for key in keys:
                if 'pcl_encoder_prior' in key:
                    resume_model_dict.pop(key)
        model_dict.update(resume_model_dict)
        network.load_state_dict(model_dict)
        s_epoch = FLAGS.resume_point
    else:
        s_epoch = 0

    # build dataset annd dataloader
    train_dataset = SketchPoseDataset(source=FLAGS.dataset, mode='train',
                                      data_dir=FLAGS.dataset_dir, per_obj=FLAGS.per_obj)
    # start training datasets sampler
    st_time = time.time()
    train_steps = FLAGS.train_size // FLAGS.batch_size
    global_step = train_steps * s_epoch  # record the number iteration
    train_size = train_steps * FLAGS.batch_size
    indices = []
    page_start = - train_size

    #  build optimizer
    param_list = network.build_params_optimizer(training_stage_freeze=[])
    optimizer = build_optimizer(param_list)
    optimizer.zero_grad()   # first clear the grad
    scheduler = build_lr_rate(optimizer, total_iters=train_steps * FLAGS.total_epoch // FLAGS.accumulate)
    #  training iteration, this code is develop based on object deform net
    for epoch in range(s_epoch, FLAGS.total_epoch):
        # train one epoch
        logger.info('Time {0}'.format(time.strftime("%Hh %Mm %Ss", time.gmtime(time.time() - st_time)) + \
                                      ', ' + 'Epoch %02d' % epoch + ', ' + 'Training started'))
        # create optimizer and adjust learning rate accordingly
        # sample train subset
        page_start += train_size
        len_last = len(indices) - page_start
        if len_last < train_size:
            indices = indices[page_start:]
            if FLAGS.dataset == 'CAMERA+Real':
                # CAMERA : Real = 3 : 1
                camera_len = train_dataset.subset_len[0]
                real_len = train_dataset.subset_len[1]
                real_indices = list(range(camera_len, camera_len + real_len))
                camera_indices = list(range(camera_len))
                n_repeat = (train_size - len_last) // (4 * real_len) + 1
                data_list = random.sample(camera_indices, 3 * n_repeat * real_len) + real_indices * n_repeat
                random.shuffle(data_list)
                indices += data_list
            else:
                data_list = list(range(train_dataset.length))
                for i in range((train_size - len_last) // train_dataset.length + 1):
                    random.shuffle(data_list)
                    indices += data_list
            page_start = 0
        train_idx = indices[page_start:(page_start + train_size)]
        train_sampler = torch.utils.data.sampler.SubsetRandomSampler(train_idx)
        train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=FLAGS.batch_size,
                                                       sampler=train_sampler,
                                                       num_workers=FLAGS.num_workers, pin_memory=True)
        network.train()
        # data_constant = None
        #################################
        for i, data in enumerate(train_dataloader, 1):
            # data = torch.load('/data/zrd/project/GPV_pose_shape_prior/output_new/modelsave_recon_all_sym_gtsize_detach_normalvote/exception/data.pth')
            # output_dict = torch.load('output/modelsave_recon_debug/exception/output_dict.pth')
            # if data_constant is None:
            #     data_constant = data
            # data = data_constant
            output_dict, loss_dict \
                = network(rgb=data['roi_img'].to(device), depth=data['roi_depth'].to(device),
                          depth_normalize=data['depth_normalize'].to(device),
                          obj_id=data['cat_id_0_base'].to(device), camK=data['cam_K'].to(device), gt_mask=data['roi_mask'].to(device),
                          gt_R=data['rotation'].to(device), gt_t=data['translation'].to(device),
                          gt_s_delta=data['fsnet_scale_delta'].to(device), mean_shape=data['mean_shape'].to(device),
                          gt_2D=data['roi_coord_2d'].to(device), sym=data['sym_info'].to(device),
                          aug_bb=data['aug_bb'].to(device), aug_rt_t=data['aug_rt_t'].to(device), aug_rt_r=data['aug_rt_R'].to(device),
                          def_mask=data['roi_mask_deform'].to(device),
                          model_point=data['model_point'].to(device), nocs_scale=data['nocs_scale'].to(device), do_loss=True,
                          shape_prior=data['shape_prior'].to(device), nocs_coord=data['nocs_coord'].to(device), logger=logger, batch_num=i)

            if loss_dict is None:
                if 'invalid_list' in output_dict.keys():
                    invalid_list = output_dict['invalid_list']
                    invalid_path_dict = {}
                    for i in invalid_list:
                        img_path = data['img_path'][i]
                        invalid_path_dict[img_path] = data['inst_id'][i].item()
                    train_dataset.add_invalid_path(invalid_path_dict)
                continue

            fsnet_loss = loss_dict['fsnet_loss']
            recon_loss = loss_dict['recon_loss']
            prop_loss = loss_dict['prop_loss']
            backbone_loss = loss_dict['backbone_loss']
            prior_loss = loss_dict['prior_loss']

            fsnet_loss_sum = sum(fsnet_loss.values())
            recon_loss_sum = sum(recon_loss.values())
            prop_loss_sum = sum(prop_loss.values())
            backbone_loss_sum = sum(backbone_loss.values())
            prior_loss_sum = sum(prior_loss.values())

            total_loss = 0
            for loss_sum in [fsnet_loss_sum, recon_loss_sum, prop_loss_sum, backbone_loss_sum, prior_loss_sum]:
                if loss_sum == 0:
                    continue
                if len(loss_sum.shape):
                    loss_sum = torch.squeeze(loss_sum)
                if not loss_sum.isnan().any():
                    total_loss += loss_sum
                else:
                    logger.warning('nan detect in loss!')
            total_loss /= FLAGS.accumulate
            # backward
            try:
                if global_step % FLAGS.accumulate == 0:
                    total_loss.backward()
                    torch.nn.utils.clip_grad_norm_(network.parameters(), 5)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                else:
                    total_loss.backward()
                    torch.nn.utils.clip_grad_norm_(network.parameters(), 5)
            except Exception as e:
                optimizer.zero_grad()
                logger.warning(f'error occur! {str(e)}! traceback:')
                logger.warning(traceback.print_exc())
                exception_path = os.path.join(FLAGS.model_save, 'exception')
                if not os.path.exists(exception_path):
                    os.makedirs(exception_path)
                torch.save(data, os.path.join(exception_path, 'data.pth'))
                torch.save(network.state_dict(), os.path.join(exception_path, 'model.pth'))
                torch.save(output_dict, os.path.join(exception_path, 'output_dict.pth'))
                continue

            global_step += 1
            summary = tf.compat.v1.Summary(value=[tf.compat.v1.Summary.Value(tag='lr',
                                                                             simple_value=optimizer.param_groups[0]["lr"]),
                                                  tf.compat.v1.Summary.Value(tag='train_loss', simple_value=total_loss),
                                                  tf.compat.v1.Summary.Value(tag='rot_loss_1',
                                                                             simple_value=fsnet_loss.get('Rot1', 0)),
                                                  tf.compat.v1.Summary.Value(tag='rot_loss_2',
                                                                             simple_value=fsnet_loss.get('Rot2', 0)),
                                                  tf.compat.v1.Summary.Value(tag='T_loss',
                                                                             simple_value=fsnet_loss.get('Tran', 0)),
                                                  tf.compat.v1.Summary.Value(tag='Prop_sym_recon',
                                                                             simple_value=prop_loss.get('Prop_sym_recon', 0)),
                                                  tf.compat.v1.Summary.Value(tag='Prop_sym_rt',
                                                                             simple_value=prop_loss.get('Prop_sym_rt', 0)),
                                                  tf.compat.v1.Summary.Value(tag='Size_loss',
                                                                             simple_value=fsnet_loss.get('Size', 0)),
                                                  tf.compat.v1.Summary.Value(tag='Face_loss',
                                                                             simple_value=recon_loss.get('recon_per_p', 0)),
                                                  tf.compat.v1.Summary.Value(tag='Recon_loss_r',
                                                                             simple_value=recon_loss.get('recon_point_r', 0)),
                                                  tf.compat.v1.Summary.Value(tag='Recon_loss_t',
                                                                             simple_value=recon_loss.get('recon_point_t', 0)),
                                                  tf.compat.v1.Summary.Value(tag='Recon_loss_s',
                                                                             simple_value=recon_loss.get('recon_point_s', 0)),
                                                  tf.compat.v1.Summary.Value(tag='Recon_p_f',
                                                                             simple_value=recon_loss.get('recon_p_f', 0)),
                                                  tf.compat.v1.Summary.Value(tag='Recon_loss_se',
                                                                             simple_value=recon_loss.get('recon_point_self', 0)),
                                                  tf.compat.v1.Summary.Value(tag='Face_loss_vote',
                                                                             simple_value=recon_loss.get('recon_point_vote', 0)),
                                                  tf.compat.v1.Summary.Value(tag='Shape_prior_total',
                                                                             simple_value=sum(prior_loss.values())),
                                                  tf.compat.v1.Summary.Value(tag='Shape_prior_corr',
                                                                             simple_value=prior_loss.get('corr_loss', 0)),
                                                  tf.compat.v1.Summary.Value(tag='Shape_prior_entropy',
                                                                             simple_value=prior_loss.get('entropy_loss', 0)),
                                                  tf.compat.v1.Summary.Value(tag='Shape_prior_cd',
                                                                             simple_value=prior_loss.get('cd_loss', 0)),
                                                  tf.compat.v1.Summary.Value(tag='Shape_prior_deform',
                                                                             simple_value=prior_loss.get('deform_loss', 0)),
                                                  ])
            tb_writter.add_summary(summary, global_step)

            if i % FLAGS.log_every == 0:
                if Train_stage == 'seman_encoder_only':
                    logger.info(
                        'Batch {0} Loss:{1:f}, mask_loss:{2:f}'.format(
                            i, total_loss.item(), backbone_loss['obj_mask'].item()))
                else:
                    logger.info('Batch {0} Loss:{1:f}, rot_loss:{2:f}, size_loss:{3:f}, trans_loss:{4:f}, prior_loss:{5:f}'.format(
                            i, total_loss.item(), (fsnet_loss['Rot1']+fsnet_loss['Rot2']).item(),
                        fsnet_loss['Size'].item(), fsnet_loss['Tran'].item(), sum(prior_loss.values())))

        logger.info('>>>>>>>>----------Epoch {:02d} train finish---------<<<<<<<<'.format(epoch))

        # save model
        if (epoch + 1) % FLAGS.save_every == 0 or (epoch + 1) == FLAGS.total_epoch:
            torch.save(network.state_dict(), '{0}/model_{1:02d}.pth'.format(FLAGS.model_save, epoch))


if __name__ == "__main__":
    app.run(train)
