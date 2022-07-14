import os
import cv2
import math
import random
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


class SketchPoseDatasetGTMask(data.Dataset):
    def __init__(self, source=None, mode='test',
                 n_pts=1024, img_size=256):
        '''

        :param source: 'CAMERA' or 'Real' or 'CAMERA+Real'
        :param mode: 'train' or 'test'
        :param data_dir: 'path to dataset'
        :param n_pts: 'number of selected sketch point', no use here
        :param img_size: cropped image size
        '''
        self.source = source
        self.mode = mode
        data_dir = FLAGS.dataset_dir
        self.data_dir = data_dir
        self.n_pts = n_pts
        self.img_size = img_size
        self.detection_dir = '/data2/zrd/datasets/NOCS/detection_dualposenet/data/segmentation_results'

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
        per_obj = FLAGS.per_obj
        self.per_obj = per_obj
        self.per_obj_id = None
        # only test one object
        if self.per_obj in self.cat_names:
            self.per_obj_id = self.cat_name2id[self.per_obj]
            img_list_cache_dir = os.path.join(self.data_dir, 'img_list')
            if not os.path.exists(img_list_cache_dir):
                os.makedirs(img_list_cache_dir)
            img_list_cache_filename = os.path.join(img_list_cache_dir, f'{per_obj}_{source}_{mode}_img_list.txt')
            if os.path.exists(img_list_cache_filename):
                print(f'read image list cache from {img_list_cache_filename}')
                img_list_obj = [line.rstrip('\n') for line in open(os.path.join(self.data_dir, img_list_cache_filename))]
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
        if mode == "train" and self.color_aug_prob > 0:
            self.color_augmentor = self._get_color_augmentor(aug_type=self.color_aug_type, aug_code=self.color_aug_code)
        else:
            self.color_augmentor = None
        self.invaild_list = ['/data/wanggu/Storage/9Ddata/CAMERA/train/05427/0005',
                             '/data/wanggu/Storage/9Ddata/CAMERA/train/05427/0006']

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
        if 'CAMERA' in img_path.split('/'):
            out_camK = self.camera_intrinsics
            img_type = 'syn'
        else:
            out_camK = self.real_intrinsics
            img_type = 'real'

        # select one foreground object,
        # if specified, then select the object

        scene = img_path.split('/')[-2]
        img_id = img_path.split('/')[-1]
        if img_type == 'real':
            dataset_split = 'REAL275'
            detection_file = os.path.join(self.detection_dir, dataset_split, f'results_test_{scene}_{img_id}.pkl')
        else:
            dataset_split = 'CAMERA25'
            detection_file = os.path.join(self.detection_dir, dataset_split, f'results_val_{scene}_{img_id}.pkl')
        with open(detection_file, 'rb') as file:
            detection_dict = cPickle.load(file)
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
        num_instance = len(detection_dict['gt_class_ids'])
        detection_dict['pred_class_ids'] = detection_dict['gt_class_ids']
        detection_dict['pred_bboxes'] = detection_dict['gt_bboxes']
        detection_dict['pred_scores'] = np.array([1.0] * num_instance)

        roi_imgs = []
        roi_depths = []
        roi_masks = []
        roi_depth_norms = []
        sym_infos = []
        mean_shapes = []
        obj_ids = []
        obj_ids_0base = []
        roi_coord_2ds = []
        valid_index = []

        mask_path = img_path + '_mask.png'
        mask_all = cv2.imread(mask_path)
        if mask_all is not None:
            mask_all = mask_all[:, :, 2]
        else:
            print('WARNING!!! mask file corrupted')
            print(img_path)
            return self.__getitem__((index + 1) % self.__len__())

        for j in range(num_instance):
            cat_id = detection_dict['pred_class_ids'][j]
            if self.per_obj_id is not None:
                if cat_id != self.per_obj_id:
                    continue
                else:
                    valid_index.append(j)
            coord_2d = get_2d_coord_np(im_W, im_H).transpose(1, 2, 0)
            # aggragate information about the selected object
            inst_id = j + 1
            mask_target = mask_all.copy().astype(np.float)
            mask_target[mask_all != inst_id] = 0.0
            mask_target[mask_all == inst_id] = 1.0
            mask = mask_target
            bbox = detection_dict['pred_bboxes'][j]
            rmin, rmax, cmin, cmax = get_bbox(bbox)
            # here resize and crop to a fixed size 256 x 256
            bbox_xyxy = np.array([cmin, rmin, cmax, rmax])
            x1, y1, x2, y2 = bbox_xyxy
            # here resize and crop to a fixed size 256 x 256
            cx = 0.5 * (x1 + x2)
            cy = 0.5 * (y1 + y2)
            bbox_center = np.array([cx, cy])  # (w/2, h/2)
            scale = max(y2 - y1, x2 - x1)
            scale = min(scale, max(im_H, im_W)) * 1.0

            ## roi_image ------------------------------------
            roi_img = crop_resize_by_warp_affine(
                rgb, bbox_center, scale, FLAGS.img_size, interpolation=cv2.INTER_NEAREST
            ).transpose(2, 0, 1)
            # import matplotlib.pyplot as plt
            # plt.imshow(roi_img.transpose(1, 2, 0))
            # plt.show()
            # roi_coord_2d ----------------------------------------------------
            roi_coord_2d = crop_resize_by_warp_affine(
                coord_2d, bbox_center, scale, FLAGS.img_size, interpolation=cv2.INTER_NEAREST
            ).transpose(2, 0, 1)
            mask_target = mask.copy().astype(np.float)
            # depth[mask_target == 0.0] = 0.0
            roi_mask = crop_resize_by_warp_affine(
                mask_target, bbox_center, scale, FLAGS.img_size, interpolation=cv2.INTER_NEAREST
            )
            roi_mask = np.expand_dims(roi_mask, axis=0)
            roi_depth = crop_resize_by_warp_affine(
                depth, bbox_center, scale, FLAGS.img_size, interpolation=cv2.INTER_NEAREST
            )

            roi_depth = np.expand_dims(roi_depth, axis=0)
            # normalize depth
            depth_valid = roi_depth > 0
            if np.sum(depth_valid) <= 1.0:
                return self.__getitem__((index + 1) % self.__len__())
            roi_m_d_valid = roi_mask.astype(np.bool) * depth_valid
            if np.sum(roi_m_d_valid) <= 1.0:
                return self.__getitem__((index + 1) % self.__len__())

            depth_v_value = roi_depth[roi_m_d_valid]
            depth_normalize = (roi_depth - np.min(depth_v_value)) / (np.max(depth_v_value) - np.min(depth_v_value))
            depth_normalize[~roi_m_d_valid] = 0.0
            # occupancy canonical
            # sym
            sym_info = self.get_sym_info(self.id2cat_name[str(cat_id)])
            mean_shape = self.get_mean_shape(self.id2cat_name[str(cat_id)])
            mean_shape = mean_shape / 1000.0
            roi_imgs.append(roi_img)
            roi_depths.append(roi_depth)
            roi_masks.append(roi_mask)
            roi_depth_norms.append(depth_normalize)
            sym_infos.append(sym_info)
            mean_shapes.append(mean_shape)
            obj_ids.append(cat_id)
            obj_ids_0base.append(cat_id - 1)
            roi_coord_2ds.append(roi_coord_2d)

        if self.per_obj_id is not None:
            for key in ['pred_class_ids', 'pred_bboxes', 'pred_scores']:
                valid_list = []
                for index in valid_index:
                    valid_list.append(detection_dict[key][index])
                detection_dict[key] = np.array(valid_list)
        detection_dict.pop('pred_masks')
        out_camK = np.array([out_camK] * len(roi_imgs))
        roi_imgs = np.array(roi_imgs)
        roi_depths = np.array(roi_depths)
        roi_masks = np.array(roi_masks)
        roi_depth_norms = np.array(roi_depth_norms)
        sym_infos = np.array(sym_infos)
        mean_shapes = np.array(mean_shapes)
        obj_ids = np.array(obj_ids)
        obj_ids_0base = np.array(obj_ids_0base)
        roi_coord_2ds = np.array(roi_coord_2ds)
        # the ground truth of every sketch point is generated while training
        return torch.as_tensor(roi_imgs.astype(np.float32)).contiguous(), torch.as_tensor(roi_depths.astype(np.float32)).contiguous(), \
               torch.as_tensor(roi_masks.astype(np.float32)).contiguous(), torch.as_tensor(roi_depth_norms.astype(np.float32)).contiguous(), \
               torch.as_tensor(sym_infos.astype(np.float32)).contiguous(), torch.as_tensor(mean_shapes.astype(np.float32)).contiguous(), \
               torch.as_tensor(out_camK.astype(np.float32)).contiguous(), torch.as_tensor(obj_ids), torch.as_tensor(obj_ids_0base), \
               torch.as_tensor(roi_coord_2ds.astype(np.float32)).contiguous(), detection_dict



    def generate_aug_parameters(self, s_x=(0.8, 1.2), s_y=(0.8, 1.2), s_z=(0.8, 1.2), ax=50, ay=50, az=50, a=15):
        # for bb aug
        ex, ey, ez = np.random.rand(3)
        ex = ex * (s_x[1] - s_x[0]) + s_x[0]
        ey = ey * (s_y[1] - s_y[0]) + s_y[0]
        ez = ez * (s_z[1] - s_z[0]) + s_z[0]
        # for R, t aug
        Rm = get_rotation(np.random.uniform(-a, a), np.random.uniform(-a, a), np.random.uniform(-a, a))
        dx = np.random.rand() * 2 * ax - ax
        dy = np.random.rand() * 2 * ay - ay
        dz = np.random.rand() * 2 * az - az
        return np.array([ex, ey, ez], dtype=np.float32), np.array([dx, dy, dz], dtype=np.float32) / 1000.0, Rm


    def get_mean_shape(self, c):
        if c == 'bottle':
            unitx = 87
            unity = 220
            unitz = 89
        elif c == 'bowl':
            unitx = 165
            unity = 80
            unitz = 165
        elif c == 'camera':
            unitx = 88
            unity = 128
            unitz = 156
        elif c == 'can':
            unitx = 68
            unity = 146
            unitz = 72
        elif c == 'laptop':
            unitx = 346
            unity = 200
            unitz = 335
        elif c == 'mug':
            unitx = 146
            unity = 83
            unitz = 114
        elif c == '02876657':
            unitx = 324 / 4
            unity = 874 / 4
            unitz = 321 / 4
        elif c == '02880940':
            unitx = 675 / 4
            unity = 271 / 4
            unitz = 675 / 4
        elif c == '02942699':
            unitx = 464 / 4
            unity = 487 / 4
            unitz = 702 / 4
        elif c == '02946921':
            unitx = 450 / 4
            unity = 753 / 4
            unitz = 460 / 4
        elif c == '03642806':
            unitx = 581 / 4
            unity = 445 / 4
            unitz = 672 / 4
        elif c == '03797390':
            unitx = 670 / 4
            unity = 540 / 4
            unitz = 497 / 4
        else:
            unitx = 0
            unity = 0
            unitz = 0
            print('This category is not recorded in my little brain.')
        # scale residual
        return np.array([unitx, unity, unitz])

    def get_sym_info(self, c):
        #  sym_info  c0 : face classfication  c1, c2, c3:Three view symmetry, correspond to xy, xz, yz respectively
        # c0: 0 no symmetry 1 axis symmetry 2 two reflection planes 3 unimplemented type
        #  Y axis points upwards, x axis pass through the handle, z axis otherwise
        #
        # for specific defination, see sketch_loss
        if c == 'bottle':
            sym = np.array([1, 1, 0, 1], dtype=np.int)
        elif c == 'bowl':
            sym = np.array([1, 1, 0, 1], dtype=np.int)
        elif c == 'camera':
            sym = np.array([0, 0, 0, 0], dtype=np.int)
        elif c == 'can':
            sym = np.array([1, 1, 1, 1], dtype=np.int)
        elif c == 'laptop':
            sym = np.array([0, 1, 0, 0], dtype=np.int)
        elif c == 'mug':
            sym = np.array([0, 1, 0, 0], dtype=np.int)  # for mug, we currently mark it as no symmetry
        else:
            sym = np.array([0, 0, 0, 0], dtype=np.int)
        return sym

    def _get_color_augmentor(self, aug_type="aae", aug_code=None):
        # fmt: off
        if aug_type.lower() == "aae":
            import imgaug.augmenters as iaa  # noqa
            from imgaug.augmenters import (Sequential, SomeOf, OneOf, Sometimes, WithColorspace, WithChannels, Noop,
                                           Lambda, AssertLambda, AssertShape, Scale, CropAndPad, Pad, Crop, Fliplr,
                                           Flipud, Superpixels, ChangeColorspace, PerspectiveTransform, Grayscale,
                                           GaussianBlur, AverageBlur, MedianBlur, Convolve, Sharpen, Emboss, EdgeDetect,
                                           DirectedEdgeDetect, Add, AddElementwise, AdditiveGaussianNoise, Multiply,
                                           MultiplyElementwise, Dropout, CoarseDropout, Invert, ContrastNormalization,
                                           Affine, PiecewiseAffine, ElasticTransformation, pillike,
                                           LinearContrast)  # noqa
            aug_code = """Sequential([
                # Sometimes(0.5, PerspectiveTransform(0.05)),
                # Sometimes(0.5, CropAndPad(percent=(-0.05, 0.1))),
                # Sometimes(0.5, Affine(scale=(1.0, 1.2))),
                Sometimes(0.5, CoarseDropout( p=0.2, size_percent=0.05) ),
                Sometimes(0.5, GaussianBlur(1.2*np.random.rand())),
                Sometimes(0.5, Add((-25, 25), per_channel=0.3)),
                Sometimes(0.3, Invert(0.2, per_channel=True)),
                Sometimes(0.5, Multiply((0.6, 1.4), per_channel=0.5)),
                Sometimes(0.5, Multiply((0.6, 1.4))),
                Sometimes(0.5, LinearContrast((0.5, 2.2), per_channel=0.3))
                ], random_order = False)"""
            color_augmentor = eval(aug_code)
        elif aug_type.lower() == 'cosy+aae':
            import imgaug.augmenters as iaa
            from imgaug.augmenters import (Sequential, SomeOf, OneOf, Sometimes, WithColorspace, WithChannels, Noop,
                                           Lambda, AssertLambda, AssertShape, Scale, CropAndPad, Pad, Crop, Fliplr,
                                           Flipud, Superpixels, ChangeColorspace, PerspectiveTransform, Grayscale,
                                           GaussianBlur, AverageBlur, MedianBlur, Convolve, Sharpen, Emboss, EdgeDetect,
                                           DirectedEdgeDetect, Add, AddElementwise, AdditiveGaussianNoise, Multiply,
                                           MultiplyElementwise, Dropout, CoarseDropout, Invert, ContrastNormalization,
                                           Affine, PiecewiseAffine, ElasticTransformation, pillike,
                                           LinearContrast)  # noqa
            aug_code = """Sequential([
            Sometimes(0.5, CoarseDropout( p=0.2, size_percent=0.05) ),
            Sometimes(0.4, GaussianBlur((0., 3.))),
            Sometimes(0.3, pillike.EnhanceSharpness(factor=(0., 50.))),
            Sometimes(0.3, pillike.EnhanceContrast(factor=(0.2, 50.))),
            Sometimes(0.5, pillike.EnhanceBrightness(factor=(0.1, 6.))),
            Sometimes(0.3, pillike.EnhanceColor(factor=(0., 20.))),
            Sometimes(0.5, Add((-25, 25), per_channel=0.3)),
            Sometimes(0.3, Invert(0.2, per_channel=True)),
            Sometimes(0.5, Multiply((0.6, 1.4), per_channel=0.5)),
            Sometimes(0.5, Multiply((0.6, 1.4))),
            Sometimes(0.1, AdditiveGaussianNoise(scale=10, per_channel=True)),
            Sometimes(0.5, iaa.contrast.LinearContrast((0.5, 2.2), per_channel=0.3)),
            Sometimes(0.5, Grayscale(alpha=(0.0, 1.0))),
            ], random_order=True)"""
            color_augmentor = eval(aug_code)
        elif aug_type.lower() == "code":  # assume imgaug
            import imgaug.augmenters as iaa
            from imgaug.augmenters import (Sequential, SomeOf, OneOf, Sometimes, WithColorspace, WithChannels, Noop,
                                           Lambda, AssertLambda, AssertShape, Scale, CropAndPad, Pad, Crop, Fliplr,
                                           Flipud, Superpixels, ChangeColorspace, PerspectiveTransform, Grayscale,
                                           GaussianBlur, AverageBlur, MedianBlur, Convolve, Sharpen, Emboss, EdgeDetect,
                                           DirectedEdgeDetect, Add, AddElementwise, AdditiveGaussianNoise, Multiply,
                                           MultiplyElementwise, Dropout, CoarseDropout, Invert, ContrastNormalization,
                                           Affine, PiecewiseAffine, ElasticTransformation, pillike,
                                           LinearContrast)  # noqa
            color_augmentor = eval(aug_code)
        elif aug_type.lower() == 'code_albu':
            from albumentations import (HorizontalFlip, IAAPerspective, ShiftScaleRotate, CLAHE, RandomRotate90,
                                        Transpose, ShiftScaleRotate, Blur, OpticalDistortion, GridDistortion,
                                        HueSaturationValue, IAAAdditiveGaussianNoise, GaussNoise, MotionBlur,
                                        MedianBlur, IAAPiecewiseAffine, IAASharpen, IAAEmboss, RandomContrast,
                                        RandomBrightness, Flip, OneOf, Compose, CoarseDropout, RGBShift, RandomGamma,
                                        RandomBrightnessContrast, JpegCompression, InvertImg)  # noqa
            color_augmentor = eval(aug_code)
        else:
            color_augmentor = None
        # fmt: on
        return color_augmentor

    def _color_aug(self, image, aug_type="code"):
        # assume image in [0, 255] uint8
        if aug_type.lower() in ["aae", "code", "cosy+aae"]:
            # imgaug need uint8
            return self.color_augmentor.augment_image(image)
        elif aug_type.lower() in ["code_albu"]:
            augmented = self.color_augmentor(image=image)
            return augmented["image"]
        else:
            raise ValueError("aug_type: {} is not supported.".format(aug_type))


if __name__ == '__main__':
    def main(argv):
        dataset = SketchPoseDataset(source='CAMERA', data_dir='/data2/zrd/datasets/NOCS')
        for i in range(10):
            data = dataset[i]
            device = 'cpu'
            img, s_d_map, d_d_map = data[0].to(device).numpy(), data[1].to(device), data[2].to(device)
            s_d_map_n, camK = data[3].to(device), data[4].to(device)
            obj_mask, obj_id = data[5].to(device), data[6].to(device)
            R, T, s = data[7].to(device), data[8].to(device), data[9].to(device)
            occupancy, sym = data[10].to(device), data[11].to(device)
            grid, sketch = data[12].to(device).numpy(), data[13].to(device).numpy()
            grid = grid.transpose(1, 2, 0) * 16
            sketch[sketch != 0] = 200
            img = img.transpose(1, 2, 0)
            fuse = img.copy()
            sketch = sketch.transpose(1, 2, 0)
            zeros = np.zeros_like(sketch)
            sketch_stack = np.concatenate([sketch, zeros, zeros], axis=-1)
            grid = np.concatenate([grid, zeros], axis=-1)
            fuse[sketch_stack > 0] = sketch_stack[sketch_stack > 0]
            cv2.imwrite(f'/data2/zrd/debug/sketch_camera_points_{i}.png', sketch)
            cv2.imwrite(f'/data2/zrd/debug/img_camera_points_{i}.png', img)
            cv2.imwrite(f'/data2/zrd/debug/fuse_camera_points_{i}.png', fuse)
            cv2.imwrite(f'/data2/zrd/debug/grid_camera_points_{i}.png', grid)


    from absl import app

    app.run(main)
