def control_loss(Train_stage):
    if Train_stage == 'PoseNet_only':
        name_mask_list = []
        name_fs_list = ['Rot1', 'Rot2', 'Rot1_cos', 'Rot2_cos', 'Rot_regular', 'Tran', 'Size', 'R_con']
        name_recon_list = ['Per_point', 'Point_voting']
        name_prop_list = ['Prop_pm', 'Prop_sym', 'Prop_point_cano']
    elif Train_stage == 'seman_encoder_only':
        name_mask_list = ['Obj_mask']
        name_fs_list = []
        name_recon_list = []
        name_prop_list = []
    elif Train_stage == 'shape_prior_only':
        name_mask_list = []
        name_fs_list = ['Rot1', 'Rot2', 'Rot1_cos', 'Rot2_cos', 'Rot_regular', 'Tran', 'Size', 'R_con']
        name_recon_list = []
        name_prop_list = ['Prop_pm', 'Prop_sym', 'Prop_point_cano']
    elif Train_stage == 'prior+recon':
        name_mask_list = []
        name_fs_list = ['Rot1', 'Rot2', 'Rot1_cos', 'Rot2_cos', 'Rot_regular', 'Tran', 'Size', 'R_con']
        name_recon_list = ['Per_point', 'Point_voting']
        name_prop_list = ['Prop_pm', 'Prop_sym', 'Prop_point_cano']
    elif Train_stage == 'prior+recon+novote':
        name_mask_list = []
        name_fs_list = ['Rot1', 'Rot2', 'Rot1_cos', 'Rot2_cos', 'Rot_regular', 'Tran', 'Size', 'R_con']
        name_recon_list = ['Per_point', ]
        name_prop_list = ['Prop_pm', 'Prop_sym', 'Prop_point_cano']
    elif Train_stage == 'FSNet_only':
        name_mask_list = []
        name_fs_list = ['Rot1', 'Rot2', 'Tran', 'Size', 'Recon']
        name_recon_list = []
        name_prop_list = []
    else:
        raise NotImplementedError
    return name_mask_list, name_fs_list, name_recon_list, name_prop_list
