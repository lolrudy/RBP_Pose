import torch
import torch.nn as nn
import absl.flags as flags
from absl import app

FLAGS = flags.FLAGS  # can control the weight of each term here

class backbone_mask_loss(nn.Module):
    def __init__(self):
        super(backbone_mask_loss, self).__init__()
        self.loss_func = nn.CrossEntropyLoss()
        # self.loss_func = nn.NLLLoss(ignore_index=-1)

    def forward(self, name_list, pred_list, gt_list):
        loss_list = {}

        if 'Obj_mask' in name_list:
            # gt_mask_np = gt_list['Mask'].detach().cpu().numpy()
            # pred_mask_np = pred_list['Mask'].detach().cpu().numpy()
            loss_list['obj_mask'] = FLAGS.mask_w * self.cal_obj_mask(pred_list['Mask'], gt_list['Mask'])
        return loss_list

    def cal_obj_mask(self, p_mask, g_mask):
        return self.loss_func(p_mask, g_mask.long().squeeze())

