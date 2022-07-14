import os
import cv2
import math
import random

import mmcv
import numpy as np
import _pickle as cPickle
from config.config import *
from datasets.data_augmentation import defor_2D, get_rotation
FLAGS = flags.FLAGS

import torch
from PIL import Image
import torch.utils.data as data
import torchvision.transforms as transforms
from tools.eval_utils import load_depth, get_bbox
from tools.dataset_utils import *


class SketchPoseDataset(data.Dataset):
    def __init__(self, source=None, mode='train', data_dir=None,
                 n_pts=1024, img_size=256, per_obj=''):
        '''

        :param source: 'CAMERA' or 'Real' or 'CAMERA+Real'
        :param mode: 'train' or 'test'
        :param data_dir: 'path to dataset'
        :param n_pts: 'number of selected sketch point', no use here
        :param img_size: cropped image size
        '''
        self.source = source
        self.mode = mode
        self.data_dir = data_dir
        self.n_pts = n_pts
        self.img_size = img_size

        assert source in ['CAMERA', 'Real', 'CAMERA+Real']
        assert mode in ['train', 'test']
        img_list_path = ['CAMERA/train_list.txt', 'Real/train_list.txt',
                         'CAMERA/val_list.txt', 'Real/test_list.txt']
        model_file_path = ['obj_models/camera_train.pkl', 'obj_models/real_train.pkl',
                           'obj_models/camera_val.pkl', 'obj_models/real_test.pkl']
        occupancy_resolution = 16
        self.resolution = occupancy_resolution
        model_occupancy_file_path = [f'obj_models/camera_train_occupancy_res{occupancy_resolution}.pkl',
                                     f'obj_models/real_train_occupancy_res{occupancy_resolution}.pkl',
                                     f'obj_models/camera_val_occupancy_res{occupancy_resolution}.pkl',
                                     f'obj_models/real_test_occupancy_res{occupancy_resolution}.pkl']

        if mode == 'train':
            del img_list_path[2:]
            del model_file_path[2:]
            del model_occupancy_file_path[2:]
        else:
            del img_list_path[:2]
            del model_file_path[:2]
            del model_occupancy_file_path[:2]
        if source == 'CAMERA':
            del img_list_path[-1]
            del model_file_path[-1]
            del model_occupancy_file_path[-1]
        elif source == 'Real':
            del img_list_path[0]
            del model_file_path[0]
            del model_occupancy_file_path[0]
        else:
            # only use Real to test when source is CAMERA+Real
            if mode == 'test':
                del img_list_path[0]
                del model_file_path[0]
                del model_occupancy_file_path[0]

        img_list = []
        subset_len = []
        #  aggregate all availabel datasets
        for path in img_list_path:
            img_list += [os.path.join(path.split('/')[0], line.rstrip('\n'))
                         for line in open(os.path.join(data_dir, path))]
            subset_len.append(len(img_list))
        if len(subset_len) == 2:
            self.subset_len = [subset_len[0], subset_len[1] - subset_len[0]]
        self.cat_names = ['bottle', 'bowl', 'camera', 'can', 'laptop', 'mug']
        self.cat_name2id = {'bottle': 1, 'bowl': 2, 'camera': 3, 'can': 4, 'laptop': 5, 'mug': 6}
        self.id2cat_name = {'1': 'bottle', '2': 'bowl', '3': 'camera', '4': 'can', '5': 'laptop', '6': 'mug'}
        self.id2cat_name_CAMERA = {'1': '02876657',
                                   '2': '02880940',
                                   '3': '02942699',
                                   '4': '02946921',
                                   '5': '03642806',
                                   '6': '03797390'}
        if source == 'CAMERA':
            self.id2cat_name = self.id2cat_name_CAMERA
        self.per_obj = per_obj
        self.per_obj_id = None
        # only train one object
        if self.per_obj in self.cat_names:
            self.per_obj_id = self.cat_name2id[self.per_obj]
            img_list_cache_dir = os.path.join(self.data_dir, 'img_list')
            if not os.path.exists(img_list_cache_dir):
                os.makedirs(img_list_cache_dir)
            img_list_cache_filename = os.path.join(img_list_cache_dir, f'{per_obj}_{source}_{mode}_img_list.txt')
            if os.path.exists(img_list_cache_filename):
                print(f'read image list cache from {img_list_cache_filename}')
                img_list_obj = [line.rstrip('\n') for line in open(os.path.join(data_dir, img_list_cache_filename))]
            else:
                # needs to reorganize img_list
                s_obj_id = self.cat_name2id[self.per_obj]
                img_list_obj = []
                from tqdm import tqdm
                for i in tqdm(range(len(img_list))):
                    gt_path = os.path.join(self.data_dir, img_list[i] + '_label.pkl')
                    try:
                        with open(gt_path, 'rb') as f:
                            gts = cPickle.load(f)
                        id_list = gts['class_ids']
                        if s_obj_id in id_list:
                            img_list_obj.append(img_list[i])
                    except:
                        print(f'WARNING {gt_path} is empty')
                        continue
                with open(img_list_cache_filename, 'w') as f:
                    for img_path in img_list_obj:
                        f.write("%s\n" % img_path)
                print(f'save image list cache to {img_list_cache_filename}')
                # iter over  all img_list, cal sublen

            if len(subset_len) == 2:
                camera_len  = 0
                real_len = 0
                for i in range(len(img_list_obj)):
                    if 'CAMERA' in img_list_obj[i].split('/'):
                        camera_len += 1
                    else:
                        real_len += 1
                self.subset_len = [camera_len, real_len]
            #  if use only one dataset
            #  directly load all data
            img_list = img_list_obj

        self.img_list = img_list
        self.length = len(self.img_list)

        models = {}
        for path in model_file_path:
            with open(os.path.join(data_dir, path), 'rb') as f:
                models.update(cPickle.load(f))
        self.models = models

        models_occupancy = {}
        for path in model_occupancy_file_path:
            with open(os.path.join(data_dir, path), 'rb') as f:
                models_occupancy.update(cPickle.load(f))
        self.models_occupancy = models_occupancy

        # move the center to the body of the mug
        # meta info for re-label mug category
        with open(os.path.join(data_dir, 'obj_models/mug_meta.pkl'), 'rb') as f:
            self.mug_meta = cPickle.load(f)

        self.camera_intrinsics = np.array([[577.5, 0, 319.5], [0, 577.5, 239.5], [0, 0, 1]],
                                          dtype=np.float)  # [fx, fy, cx, cy]
        self.real_intrinsics = np.array([[591.0125, 0, 322.525], [0, 590.16775, 244.11084], [0, 0, 1]], dtype=np.float)

        self.color_aug_prob = FLAGS.color_aug_prob
        self.color_aug_type = FLAGS.color_aug_type
        self.color_aug_code = FLAGS.color_aug_code

        self.invaild_list = []
        # self.mug_sym = mmcv.load(os.path.join(self.data_dir, 'Real/train/mug_handle.pkl'))
        # self.shape_prior = np.load(os.path.join(data_dir, 'results/mean_shape/mean_points_emb.npy'))

        print('{} images found.'.format(self.length))
        print('{} models loaded.'.format(len(self.models)))

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        #   load ground truth
        #  if per_obj is specified, then we only select the target object
        # index = index % self.length  # here something wrong
        img_path = os.path.join(self.data_dir, self.img_list[index])
        if img_path in self.invaild_list:
            return self.__getitem__((index + 1) % self.__len__())
        try:
            with open(img_path + '_label_pure.pkl', 'rb') as f:
                gts = cPickle.load(f)
        except:
            return self.__getitem__((index + 1) % self.__len__())


        rgb = cv2.imread(img_path + '_color.png')
        if rgb is not None:
            rgb = rgb[:, :, :3]
        else:
            return self.__getitem__((index + 1) % self.__len__())

        im_H, im_W = rgb.shape[0], rgb.shape[1]
        depth_path = img_path + '_depth.png'
        if os.path.exists(depth_path):
            depth = load_depth(depth_path)
        else:
            return self.__getitem__((index + 1) % self.__len__())

        mask_path = img_path + '_mask.png'
        mask = cv2.imread(mask_path)
        if mask is not None:
            mask = mask[:, :, 2]
        else:
            return self.__getitem__((index + 1) % self.__len__())

        nocs_coord = cv2.imread(img_path + '_coord.png')
        if nocs_coord is not None:
            nocs_coord = nocs_coord[:, :, :3]
        else:
            return self.__getitem__((index + 1) % self.__len__())
        nocs_coord = nocs_coord[:, :, (2, 1, 0)]
        nocs_coord = np.array(nocs_coord, dtype=np.float32) / 255
        nocs_coord[:, :, 2] = 1 - nocs_coord[:, :, 2]
        # [0, 1] -> [-0.5, 0.5]
        nocs_coord = nocs_coord - 0.5
        coord_2d = get_2d_coord_np(im_W, im_H)

        return gts, torch.tensor(rgb), torch.tensor(depth.astype(np.float)), torch.tensor(mask.astype(np.float)), torch.tensor(nocs_coord), torch.tensor(coord_2d), img_path



if __name__ == '__main__':
    from network.point_sample.face_sample import Mask2Pc
    from tools.shape_prior_utils import get_point_depth_error
    import matplotlib.pyplot as plt

    from tqdm import tqdm
    def main(argv):
        dataset = SketchPoseDataset(source='Real', data_dir='/data/zrd/datasets/NOCS')
        save_dir = '/data/zrd/project/NOCS_visualize_point_mask'
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        camK = dataset.real_intrinsics
        camK = torch.tensor(camK)
        camK = camK.unsqueeze(0)
        # aggragate information about the selected object
        for data in tqdm(dataset):
            gts, rgb, depth, mask, nocs_coord, coord_2d, img_path = data
            depth = depth.unsqueeze(0)
            depth = depth.unsqueeze(0)
            rgb_np = rgb.numpy()
            rgb = rgb.unsqueeze(0)
            mask = mask.unsqueeze(0)
            mask = mask.unsqueeze(0)
            nocs_coord = nocs_coord.unsqueeze(0)
            coord_2d = coord_2d.unsqueeze(0)
            for idx in range(len(gts['instance_ids'])):
                inst_id = gts['instance_ids'][idx]
                mask_target = mask.clone()
                mask_target[mask != inst_id] = 0.0
                mask_target[mask == inst_id] = 1.0
                # depth[mask_target == 0.0] = 0.0
                # cat_id, rotation translation and scale
                cat_id = gts['class_ids'][idx] - 1  # convert to 0-indexed
                # note that this is nocs model, normalized along diagonal axis
                model_name = gts['model_list'][idx]
                model = dataset.models[model_name].astype(np.float32)  # 1024 points
                nocs_scale = gts['scales'][idx]  # nocs_scale = image file / model file
                lx = max(model[:, 0]) - min(model[:, 0])
                ly = max(model[:, 1]) - min(model[:, 1])
                lz = max(model[:, 2]) - min(model[:, 2])
                scale = np.array([lx, ly, lz]) * nocs_scale
                rotation = gts['rotations'][idx]
                translation = gts['translations'][idx]

                gt_R = torch.tensor(rotation).unsqueeze(0)
                gt_s = torch.tensor(scale).unsqueeze(0)
                gt_t = torch.tensor(translation).unsqueeze(0)

                PC, PC_sk, PC_seman, PC_nocs, fuse_mask = Mask2Pc(mask_target, depth, camK, coord_2d, torch.zeros_like(rgb), nocs_coord=nocs_coord)
                point_depth_error, point_nocs_error = get_point_depth_error(PC_nocs, PC, gt_R, gt_t, gt_s)
                point_mask_gt = point_depth_error < FLAGS.point_mask_distance_threshold
                point_mask_gt = point_mask_gt.detach().numpy()
                fuse_mask = np.squeeze(fuse_mask.numpy()).astype(bool)
                point_mask_gt = np.squeeze(point_mask_gt).astype(np.uint8)
                rgb_np[fuse_mask, 0] = point_mask_gt * 255
                rgb_np[fuse_mask, 1] = point_mask_gt * 255
                rgb_np[fuse_mask, 2] = point_mask_gt * 255

            # plt.imshow(rgb_np)
            # plt.show()
            cv2.imwrite(os.path.join(save_dir, img_path.replace('/', '_') + '.png'), rgb_np)
    from absl import app

    app.run(main)


