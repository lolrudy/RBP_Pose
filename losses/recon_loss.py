import torch
import torch.nn as nn
import absl.flags as flags
from absl import app
from tools.plane_utils import get_plane, get_plane_parameter
from tools.rot_utils import get_vertical_rot_vec, get_R_batch
from .uncertainty_loss import laplacian_aleatoric_uncertainty_loss
FLAGS = flags.FLAGS  # can control the weight of each term here

class recon_6face_loss(nn.Module):
    def __init__(self):
        super(recon_6face_loss, self).__init__()
        self.loss_func = nn.L1Loss()
        self.uncertainty_loss_func = laplacian_aleatoric_uncertainty_loss
        self.balance_weight_shift = FLAGS.balance_weight_uncertainty_shift

    def forward(self, name_list, pred_list, gt_list, sym, obj_ids, save_path=None):
        loss_list = {}

        if 'Per_point' in name_list:
            res_shift = self.cal_recon_loss_point(gt_list['Points'],
                                                                   pred_list['face_shift'],
                                                                   pred_list['F_log_var'],
                                                                   gt_list['R'],
                                                                   gt_list['T'],
                                                                   gt_list['Size'],
                                                                   sym, obj_ids, gt_list['Point_mask'], save_path)
            loss_list['recon_per_p'] = FLAGS.recon_shift_w * res_shift
            # regularize face_dis_delta
            loss_list['recon_reg_delta'] = FLAGS.recon_reg_delta_w * self.loss_func(pred_list['face_shift_delta'], torch.zeros_like(pred_list['face_shift_delta']))
            loss_list['recon_point_mask'] = FLAGS.recon_point_mask_w * self.loss_func(pred_list['Point_mask_conf'], gt_list['Point_mask'].long())

        if 'Consistency' in name_list:
            pred_R = get_R_batch(pred_list['Rot1_f'], pred_list['Rot2_f'], pred_list['Rot1'], pred_list['Rot2'], sym)
            res_con = self.cal_recon_loss_point(gt_list['Points'],
                                                                   pred_list['face_shift'],
                                                                   pred_list['F_log_var'],
                                                                   pred_R,
                                                                   pred_list['Tran'],
                                                                   pred_list['Size'],
                                                                   sym, obj_ids, gt_list['Point_mask'], save_path)
            loss_list['recon_consistency'] = FLAGS.recon_con_w * res_con

        if 'Point_voting' in name_list:
            recon_point_vote, recon_point_r, recon_point_t, recon_point_s, recon_point_self = self.cal_recon_loss_vote(
                                                                             gt_list['Points'],
                                                                             pred_list['face_shift'],
                                                                             pred_list['F_log_var'],
                                                                             pred_list['Rot1'],
                                                                             pred_list['Rot1_f'],
                                                                             pred_list['Rot2'],
                                                                             pred_list['Rot2_f'],
                                                                             pred_list['Tran'],
                                                                             pred_list['Size'],
                                                                             gt_list['R'],
                                                                             gt_list['T'],
                                                                             gt_list['Size'],
                                                                             sym, obj_ids, gt_list['Point_mask'], save_path,
                                                                             pred_list['face_shift_prior'])
            loss_list['recon_point_vote'] = FLAGS.recon_v_w * recon_point_vote
            loss_list['recon_point_r'] = FLAGS.recon_bb_r_w * recon_point_r
            loss_list['recon_point_t'] = FLAGS.recon_bb_t_w * recon_point_t
            loss_list['recon_point_s'] = FLAGS.recon_bb_s_w * recon_point_s
            loss_list['recon_point_self'] = FLAGS.recon_bb_self_w * recon_point_self
        return loss_list



    def cal_recon_loss_vote(self, pc, face_shift, face_log_var, p_rot_g, f_rot_g, p_rot_r, f_rot_r, p_t, p_s,
                            gt_R, gt_t, gt_s, sym, obj_ids, point_mask, save_path=None, face_shift_prior=None):

        res_vote = 0.0
        res_recon_geo_r = 0.0
        res_recon_geo_t = 0.0
        res_recon_geo_s = 0.0
        res_recon_self_cal = 0.0
        bs = pc.shape[0]
        assert bs > 1
        re_s = gt_s
        pre_s = p_s
        face_log_var = face_log_var.detach()
        face_std = torch.exp(0.5 * face_log_var)
        face_std[point_mask == 0] = 1e10
        face_c = torch.softmax(-face_std, dim=1)
        for i in range(bs):
            point_mask_now = point_mask[i]
            pc_now = pc[i, ...][point_mask_now]
            face_shift_prior_now = face_shift_prior[i, ...][point_mask_now]
            p_valid_num = pc_now.shape[0]
            if p_valid_num <= FLAGS.point_mask_min_threshold:
                print(f'WARNING!! point valid number {p_valid_num} is lower than threshold')
                continue
            f_shift_now = face_shift[i, ...][point_mask_now]  # n x 6
            f_c_now = face_c[i, ...][point_mask_now]  # n x 6
            re_s_now = re_s[i, ...]  # 3
            gt_r_x = gt_R[i, :, 0]
            gt_r_y = gt_R[i, :, 1]
            gt_r_z = gt_R[i, :, 2]
            gt_t_now = gt_t[i, ...]
            obj_id = int(obj_ids[i])
            # y +
            pc_on_plane = pc_now + f_shift_now[:, 0:3]

            # note that in dn_y_up, d also has direction
            n_y_up, dn_y_up, c_y_up = get_plane(pc_on_plane, f_c_now[:, 0])
            # cal gt
            dn_gt = gt_r_y * (-(torch.dot(gt_r_y, gt_t_now + gt_r_y * re_s_now[1] / 2)))
            if save_path is not None:
                import numpy as np
                view_points = pc_on_plane.detach().cpu().numpy()
                ref_points = pc_now.detach().cpu().numpy()
                conf_points = f_c_now[:, 0].detach().cpu().numpy()
                np.savetxt(save_path + f'_{i}_pc_on_plane_yp.txt', view_points)
                np.savetxt(save_path + f'_{i}_pc_origin_yp.txt', ref_points)
                np.savetxt(save_path + f'_{i}_pc_conf_yp.txt', conf_points)
                prior_points = face_shift_prior_now[:, 0:3] + pc_now
                prior_points = prior_points.detach().cpu().numpy()
                np.savetxt(save_path + f'_{i}_shift_prior_yp.txt', prior_points)
                plane_parameter = get_plane_parameter(pc_on_plane, f_c_now[:, 0])
                plane_parameter = plane_parameter.detach().cpu().numpy()
                np.savetxt(save_path + f'_{i}_plane_parameter_yp.txt', plane_parameter)
            if n_y_up is not None:
                # adjust the sign of n_y_up
                if torch.dot(n_y_up, gt_r_y) < 0:
                    n_y_up = -n_y_up
                    c_y_up = -c_y_up
                res_yplus = torch.mean(torch.abs(dn_y_up - dn_gt))
            else:
                res_yplus = 0
            # cal recon_ geo loss

            if sym[i, 0] == 0:
                # x +
                pc_on_plane = pc_now + f_shift_now[:, 3:6]
                n_x_up, dn_x_up, c_x_up = get_plane(pc_on_plane, f_c_now[:, 1])

                # cal gt
                dn_gt = gt_r_x * (-(torch.dot(gt_r_x, gt_t_now + gt_r_x * re_s_now[0] / 2)))
                if save_path is not None:
                    view_points = pc_on_plane.detach().cpu().numpy()
                    ref_points = pc_now.detach().cpu().numpy()
                    conf_points = f_c_now[:, 1].detach().cpu().numpy()
                    prior_points = face_shift_prior_now[:,3:6] + pc_now
                    prior_points = prior_points.detach().cpu().numpy()
                    np.savetxt(save_path + f'_{i}_shift_prior_xp.txt', prior_points)
                    np.savetxt(save_path + f'_{i}_pc_on_plane_xp.txt', view_points)
                    np.savetxt(save_path + f'_{i}_pc_origin_xp.txt', ref_points)
                    np.savetxt(save_path + f'_{i}_pc_conf_xp.txt', conf_points)
                    plane_parameter = get_plane_parameter(pc_on_plane, f_c_now[:, 1])
                    plane_parameter = plane_parameter.detach().cpu().numpy()
                    np.savetxt(save_path + f'_{i}_plane_parameter_xp.txt', plane_parameter)
                if n_x_up is not None:
                    # adjust the sign of dn_gt
                    if torch.dot(n_x_up, gt_r_x) < 0:
                        n_x_up = -n_x_up
                        c_x_up = -c_x_up
                    res_xplus = torch.mean(torch.abs(dn_x_up - dn_gt))

                else:
                    res_xplus = 0
                # z +
                pc_on_plane = pc_now + f_shift_now[:, 6:9]
                n_z_up, dn_z_up, c_z_up = get_plane(pc_on_plane, f_c_now[:, 2])
                # cal gt
                dn_gt = gt_r_z * (-(torch.dot(gt_r_z, gt_t_now + gt_r_z * re_s_now[2] / 2)))
                if save_path is not None:
                    view_points = pc_on_plane.detach().cpu().numpy()
                    ref_points = pc_now.detach().cpu().numpy()
                    conf_points = f_c_now[:, 2].detach().cpu().numpy()
                    np.savetxt(save_path + f'_{i}_pc_on_plane_zp.txt', view_points)
                    np.savetxt(save_path + f'_{i}_pc_origin_zp.txt', ref_points)
                    np.savetxt(save_path + f'_{i}_pc_conf_zp.txt', conf_points)
                    prior_points = face_shift_prior_now[:, 6:9] + pc_now
                    prior_points = prior_points.detach().cpu().numpy()
                    np.savetxt(save_path + f'_{i}_shift_prior_zp.txt', prior_points)
                    plane_parameter = get_plane_parameter(pc_on_plane, f_c_now[:, 2])
                    plane_parameter = plane_parameter.detach().cpu().numpy()
                    np.savetxt(save_path + f'_{i}_plane_parameter_zp.txt', plane_parameter)
                if n_z_up is not None:
                    # adjust the sign of dn_gt
                    if torch.dot(n_z_up, gt_r_z) < 0:
                        n_z_up = -n_z_up
                        c_z_up = -c_z_up
                    res_zplus = torch.mean(torch.abs(dn_z_up - dn_gt))
                else:
                    res_zplus = 0
                # x -
                pc_on_plane = pc_now + f_shift_now[:, 9:12]
                n_x_down, dn_x_down, c_x_down = get_plane(pc_on_plane, f_c_now[:, 3])
                # cal gt
                dn_gt = -gt_r_x * (-(torch.dot(-gt_r_x, gt_t_now - gt_r_x * re_s_now[0] / 2)))
                if n_x_down is not None:
                    # adjust the sign of dn_gt
                    if torch.dot(n_x_down, -gt_r_x) < 0:
                        n_x_down = -n_x_down
                        c_x_down = -c_x_down
                    res_xminus = torch.mean(torch.abs(dn_x_down - dn_gt))
                else:
                    res_xminus = 0
                # z -
                pc_on_plane = pc_now + f_shift_now[:, 12:15]
                n_z_down, dn_z_down, c_z_down = get_plane(pc_on_plane, f_c_now[:, 4])
                # cal gt
                dn_gt = -gt_r_z * (-(torch.dot(-gt_r_z, gt_t_now - gt_r_z * re_s_now[2] / 2)))
                if n_z_down is not None:
                    # adjust the sign of dn_gt
                    if torch.dot(n_z_down, -gt_r_z) < 0:
                        n_z_down = -n_z_down
                        c_z_down = -c_z_down
                    res_zminus = torch.mean(torch.abs(dn_z_down - dn_gt))
                else:
                    res_zminus = 0
            else:
                res_xplus = 0.0
                res_xminus = 0.0
                res_zplus = 0.0
                res_zminus = 0.0

            # y -
            pc_on_plane = pc_now + f_shift_now[:, 15:18]
            n_y_down, dn_y_down, c_y_down = get_plane(pc_on_plane, f_c_now[:, 5])
            # cal gt
            dn_gt = -gt_r_y * (-(torch.dot(-gt_r_y, gt_t_now - gt_r_y * re_s_now[1] / 2)))
            if n_y_down is not None:
                # adjust the sign of dn_gt
                if torch.dot(n_y_down, -gt_r_y) < 0:
                    n_y_down = -n_y_down
                    c_y_down = -c_y_down
                res_yminus = torch.mean(torch.abs(dn_y_down - dn_gt))
            else:
                res_yminus = 0
            if obj_id != 5:
                res_vote += res_xplus
                res_vote += res_xminus
            res_vote += res_yplus
            res_vote += res_zplus
            res_vote += res_yminus
            res_vote += res_zminus

            #######################cal geo recon loss ##################
            # for r, rectify
            new_y, new_x = get_vertical_rot_vec(f_rot_g[i], f_rot_r[i], p_rot_g[i, ...], p_rot_r[i, ...])
            new_z = torch.cross(new_x, new_y)
            # y+
            if n_y_up is not None:
                res_recon_geo_r += torch.mean(torch.abs((n_y_up - new_y)))
            if sym[i, 0] == 0:
                if obj_id != 5:
                    # x+
                    if n_x_up is not None:
                        res_recon_geo_r += torch.mean(torch.abs((n_x_up - new_x)))
                    # x-
                    if n_x_down is not None:
                        res_recon_geo_r += torch.mean(torch.abs((n_x_down - (-new_x))))
                # z+
                if n_z_up is not None:
                    res_recon_geo_r += torch.mean(torch.abs((n_z_up - new_z)))
                # z-
                if n_z_down is not None:
                    res_recon_geo_r += torch.mean(torch.abs((n_z_down - (-new_z))))
            # y-
            if n_y_down is not None:
                res_recon_geo_r += torch.mean(torch.abs((n_y_down - (-new_y))))

            # for T
            # Translation must correspond to the center of the bbox
            p_t_now = p_t[i, ...].view(-1)  # 3
            # cal the distance between p_t_now and the predicted plane
            if n_y_up is not None and n_y_down is not None:
                # y+
                dis_y_up = torch.abs(torch.dot(n_y_up, p_t_now) + c_y_up)
                # y-
                dis_y_down = torch.abs(torch.dot(n_y_down, p_t_now) + c_y_down)
                res_recon_geo_t += torch.abs(dis_y_down - dis_y_up)
                # for s
                res_recon_geo_s += torch.abs(pre_s[i, 1] / 2.0 - dis_y_down)
                res_recon_geo_s += torch.abs(pre_s[i, 1] / 2.0 - dis_y_up)
                # for bounding box self-calibrate
                # parallel
                res_recon_self_cal += torch.mean(torch.abs((n_y_up + n_y_down)))
            if sym[i, 0] == 0:
                if obj_id != 5 and n_x_up is not None and n_x_down is not None:
                    # x+
                    dis_x_up = torch.abs(torch.dot(n_x_up, p_t_now) + c_x_up)
                    # x-
                    dis_x_down = torch.abs(torch.dot(n_x_down, p_t_now) + c_x_down)
                    res_recon_geo_t += torch.abs(dis_x_down - dis_x_up)
                    res_recon_geo_s += torch.abs(pre_s[i, 0] / 2.0 - dis_x_down)
                    res_recon_geo_s += torch.abs(pre_s[i, 0] / 2.0 - dis_x_up)
                    res_recon_self_cal += torch.mean(torch.abs((n_x_up + n_x_down)))

                if n_z_up is not None and n_z_down is not None:
                    # z+
                    dis_z_up = torch.abs(torch.dot(n_z_up, p_t_now) + c_z_up)
                    # z-
                    dis_z_down = torch.abs(torch.dot(n_z_down, p_t_now) + c_z_down)
                    res_recon_geo_t += torch.abs(dis_z_down - dis_z_up)
                    res_recon_geo_s += torch.abs(pre_s[i, 2] / 2.0 - dis_z_up)
                    res_recon_geo_s += torch.abs(pre_s[i, 2] / 2.0 - dis_z_down)
                    res_recon_self_cal += torch.mean(torch.abs((n_z_up + n_z_down)))


            # vertical
            if sym[i, 0] == 0 and n_y_up is not None and n_y_down is not None \
                    and n_z_up is not None and n_z_down is not None\
                    and n_x_up is not None and n_x_down is not None:
                if obj_id != 5:
                    res_recon_self_cal += torch.abs(torch.dot(n_y_up, n_x_up))
                    res_recon_self_cal += torch.abs(torch.dot(n_y_down, n_x_down))
                res_recon_self_cal += torch.abs(torch.dot(n_y_up, n_z_up))
                res_recon_self_cal += torch.abs(torch.dot(n_y_down, n_z_down))

        res_vote = res_vote / 6 / bs
        res_recon_self_cal = res_recon_self_cal / 6 / bs
        res_recon_geo_s = res_recon_geo_s / 6 / bs
        res_recon_geo_r = res_recon_geo_r / 6 / bs
        res_recon_geo_t = res_recon_geo_t / 6 / bs
        return res_vote, res_recon_geo_r, res_recon_geo_t, res_recon_geo_s, res_recon_self_cal


    def cal_recon_loss_point(self, pc, face_shift, face_log_std, gt_R, gt_t, gt_s, sym, obj_ids, point_mask, save_path=None):
        '''
        :param pc:
        :param face_shift: bs x n x 6 x 3
        :param face_log_std: bs x n x 6
        :param gt_R_green: bs x 3
        :param gt_R_red:
        :param gt_t:
        :param gt_s:
        :param sym:
        :return:
        '''
        bs = pc.shape[0]

        # face loss
        res_shift = 0.0
        re_s = gt_s
        pc_cano = torch.bmm(gt_R.permute(0, 2, 1), (pc.permute(0, 2, 1) - gt_t.view(bs, 3, 1))).permute(0, 2, 1)
        for i in range(bs):
            gt_r_x = gt_R[i, :, 0].view(3)
            gt_r_y = gt_R[i, :, 1].view(3)
            gt_r_z = gt_R[i, :, 2].view(3)
            point_mask_now = point_mask[i]
            face_log_std_now = face_log_std[i, ...][point_mask_now]
            point_num = sum(point_mask_now)
            if len(point_mask_now) <= FLAGS.point_mask_min_threshold:
                continue
            obj_id = int(obj_ids[i])

            #######################################################
            # dis loss
            pc_cano_now = pc_cano[i, ...][point_mask_now]   # n x 3
            re_s_now = re_s[i, ...]  # 3
            f_s_now = face_shift[i, ...][point_mask_now]  # n x 6
            # face y +
            f_s_yplus = f_s_now[:, 0:3]  # nn x 1
            f_d_gt_yplus = re_s_now[1] / 2 - pc_cano_now[:, 1]
            res_yplus = self.cal_shift_loss_for_face(f_s_yplus, f_d_gt_yplus, gt_r_y, face_log_std_now[:, 0])
            if save_path is not None:
                import numpy as np
                shift_gt_yp = f_d_gt_yplus.unsqueeze(-1) * gt_r_y
                pc_now = pc[i, ...][point_mask_now]  # n x 3
                gt_points = shift_gt_yp + pc_now
                gt_points = gt_points.detach().cpu().numpy()
                np.savetxt(save_path + f'_{i}_pc_gt_yp.txt', gt_points)
            if sym[i, 0] == 0:
                # face x +
                f_s_xplus = f_s_now[:, 3:6]  # nn x 1
                f_d_gt_xplus = re_s_now[0] / 2 - pc_cano_now[:, 0]
                res_xplus = self.cal_shift_loss_for_face(f_s_xplus, f_d_gt_xplus, gt_r_x, face_log_std_now[:, 1])
                if save_path is not None:
                    shift_gt_xp = f_d_gt_xplus.unsqueeze(-1) * gt_r_x
                    gt_points = shift_gt_xp + pc_now
                    gt_points = gt_points.detach().cpu().numpy()
                    np.savetxt(save_path + f'_{i}_pc_gt_xp.txt', gt_points)
                # face z +
                f_s_zplus = f_s_now[:, 6:9]  # nn x 1
                f_d_gt_zplus = re_s_now[2] / 2 - pc_cano_now[:, 2]
                res_zplus = self.cal_shift_loss_for_face(f_s_zplus, f_d_gt_zplus, gt_r_z, face_log_std_now[:, 2])
                if save_path is not None:
                    shift_gt_zp = f_d_gt_zplus.unsqueeze(-1) * gt_r_z
                    gt_points = shift_gt_zp + pc_now
                    gt_points = gt_points.detach().cpu().numpy()
                    np.savetxt(save_path + f'_{i}_pc_gt_zp.txt', gt_points)
                # face x -
                f_s_xminus = f_s_now[:, 9:12]  # nn x 1
                f_d_gt_xminus = pc_cano_now[:, 0] + re_s_now[0] / 2
                res_xminus = self.cal_shift_loss_for_face(f_s_xminus, f_d_gt_xminus, -gt_r_x, face_log_std_now[:, 3])

                # face z -
                f_s_zminus = f_s_now[:, 12:15]  # nn x 1
                f_d_gt_zminus = pc_cano_now[:, 2] + re_s_now[2] / 2
                res_zminus = self.cal_shift_loss_for_face(f_s_zminus, f_d_gt_zminus, -gt_r_z, face_log_std_now[:, 4])

            else:
                res_xplus = 0.0
                res_xminus = 0.0
                res_zplus = 0.0
                res_zminus = 0.0
            # face y -
            f_s_yminus = f_s_now[:, 15:18]  # nn x 1
            f_d_gt_yminus = pc_cano_now[:, 1] + re_s_now[1] / 2
            res_yminus = self.cal_shift_loss_for_face(f_s_yminus, f_d_gt_yminus, -gt_r_y, face_log_std_now[:, 5])

            if obj_id != 5:
                res_shift += res_xplus
                res_shift += res_xminus
            res_shift += res_yplus
            res_shift += res_zplus
            res_shift += res_yminus
            res_shift += res_zminus

        res_shift = res_shift / 6 / bs
        return res_shift

    def cal_recon_consistency_loss(self, pred_nocs, pred_size, face_dis):
        raise NotImplementedError

    def cal_shift_loss_for_face(self, shift_pred, dis_gt, gt_rot_axis, log_std):
        shift_gt = dis_gt.unsqueeze(-1) * gt_rot_axis
        res = self.uncertainty_loss_func(shift_pred, shift_gt, log_std,
                                         self.balance_weight_shift, sum_last_dim=True)
        return res

    # def cal_plane_parameter(self, pc_now, f_n_now, f_d_now, f_c_now, plane_name, gt_R, save_path=None):
    #     plane_index_list = ['yp', 'xp', 'zp', 'xm', 'zm', 'ym']
    #     plane_index = plane_index_list.index(plane_name)
    #     pc_on_plane = pc_now + f_n_now[:, plane_index, :] * f_d_now[:, plane_index].view(-1, 1)
    #     R_col =
    #     gt_r_x = gt_R[i, :, 0].view(3)
    #     gt_r_y = gt_R[i, :, 1].view(3)
    #     gt_r_z = gt_R[i, :, 2].view(3)
    #     # note that in dn_y_up, d also has direction
    #     n_y_up, dn_y_up, c_y_up = get_plane(pc_on_plane, f_c_now[:, plane_index])
    #
    #     if save_path is not None:
    #         import mmcv, os
    #         view_points = pc_on_plane.detach().cpu().numpy()
    #         ref_points = pc_now.detach().cpu().numpy()
    #         conf_points = f_c_now[:, 0].detach().cpu().numpy()
    #         import numpy as np
    #         np.savetxt(save_path + f'_pc_on_plane_{plane_name}.txt', view_points)
    #         np.savetxt(save_path + f'_pc_origin_{plane_name}.txt', ref_points)
    #         np.savetxt(save_path + f'_pc_conf_{plane_name}.txt', conf_points)
    #
    #         plane_parameter = get_plane_parameter(pc_on_plane, f_c_now[:, 0])
    #         plane_parameter = plane_parameter.detach().cpu().numpy()
    #         np.savetxt(save_path + f'_plane_parameter_{plane_name}.txt', plane_parameter)
    #
    #     # cal gt
    #     dn_gt = gt_r_y * (-(torch.dot(gt_r_y, gt_t_now + gt_r_y * re_s_now[1] / 2)))
    #     # adjust the sign of n_y_up
    #     if torch.dot(n_y_up, gt_r_y) < 0:
    #         n_y_up = -n_y_up
    #         c_y_up = -c_y_up
    #     res_yplus = torch.mean(torch.abs(dn_y_up - dn_gt))