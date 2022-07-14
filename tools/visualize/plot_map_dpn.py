import matplotlib as plt
import os
from evaluation.eval_utils_cass import *
import mmcv
from shutil import copyfile

def plot_mAP(degree_thres_list, shift_thres_list, iou_thres_list, iou_3d_aps, pose_aps, output_path, suffix=''):
    # draw iou 3d AP vs. iou thresholds
    synset_names = ['BG'] + ['bottle', 'bowl', 'camera', 'can', 'laptop', 'mug']
    num_classes = len(synset_names)
    iou_3d_aps = iou_3d_aps.copy() * 100
    pose_aps = pose_aps.copy() * 100

    fig_iou = plt.figure(figsize=(15, 5))
    ax_iou = plt.subplot(131)
    plt.title('3D IOU')
    plt.ylabel(f'Average Precision ({suffix})', fontsize=12)
    plt.ylim((0, 100))
    plt.xlim((0, 100))
    plt.xlabel('Percent')
    iou_thres_list = np.array(iou_thres_list) * 100
    for cls_id in range(1, num_classes):
        class_name = synset_names[cls_id]
        ax_iou.plot(iou_thres_list, iou_3d_aps[cls_id, :], label=class_name)
    ax_iou.plot(iou_thres_list, iou_3d_aps[-1, :], label='mean')
    ax_iou.xaxis.set_major_locator(plt.MultipleLocator(25))
    ax_iou.grid()
    # draw pose AP vs. thresholds
    ax_rot = plt.subplot(132)
    plt.ylim((0, 100))
    plt.xlim((0, 45))
    plt.title('Rotation')
    plt.xlabel('Degree')
    for cls_id in range(1, num_classes):
        class_name = synset_names[cls_id]
        ax_rot.plot(
            degree_thres_list[:-1], pose_aps[cls_id, :-1, -1], label=class_name)

    ax_rot.plot(degree_thres_list[:-1], pose_aps[-1, :-1, -1], label='mean')
    ax_rot.xaxis.set_major_locator(plt.MultipleLocator(15))
    ax_rot.grid()

    ax_trans = plt.subplot(133)
    plt.ylim((0, 100))
    plt.xlim((0, 10))
    plt.title('Translation')
    plt.xlabel('Centimeter')
    for cls_id in range(1, num_classes):
        class_name = synset_names[cls_id]
        # print(class_name)
        ax_trans.plot(shift_thres_list[:-1],
                    pose_aps[cls_id, -1, :-1], label=class_name)

    ax_trans.plot(shift_thres_list[:-1], pose_aps[-1, -1, :-1], label='mean')
    ax_trans.legend(loc='lower right')
    ax_trans.xaxis.set_major_locator(plt.MultipleLocator(5))
    ax_trans.grid()
    fig_iou.savefig(output_path)
    plt.close(fig_iou)

if __name__ == '__main__':
    bottle_path = 'output/modelsave_all_1029/eval_result_model_149/pred_result.pkl'
    bowl_path = 'output/modelsave_all_1029/eval_result_model_149/pred_result.pkl'
    camera_path = 'output/modelsave_all_1029/eval_result_model_149/pred_result.pkl'
    can_path = 'output/modelsave_all_1029/eval_result_model_149/pred_result.pkl'
    laptop_path = 'output/modelsave_all_1029/eval_result_model_149/pred_result.pkl'
    mug_path = 'output/modelsave_all_1029/eval_result_model_149/pred_result.pkl'

    dualposenet_path = '/data2/zrd/GPV_pose_result/dualposenet_results/REAL275_results.pkl'

    output_path = '/data2/zrd/GPV_pose_result/1129'
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    degree_thres_list = list(range(0, 61, 1))
    shift_thres_list = [i / 2 for i in range(21)]
    iou_thres_list = [i / 100 for i in range(101)]
    synset_names = ['BG'] + ['bottle', 'bowl', 'camera', 'can', 'laptop', 'mug']
    num_classes = 6

    # evaluate dualposenet results
    output_dpn_eval_file_path = '/data2/zrd/GPV_pose_result/dualposenet_results/REAL275_eval_results.pkl'
    if os.path.exists(output_dpn_eval_file_path):
        iou_aps_dualposenet, pose_aps_dualposenet = mmcv.load(output_dpn_eval_file_path)
    else:
        dualposenet_pred_results = mmcv.load(dualposenet_path)
        iou_aps_dualposenet, pose_aps_dualposenet = compute_degree_cm_mAP(dualposenet_pred_results, synset_names, output_path, degree_thres_list,
                                                  shift_thres_list,
                                                  iou_thres_list, iou_pose_thres=0.1, use_matches_for_pose=True)
        mmcv.dump([iou_aps_dualposenet, pose_aps_dualposenet], output_dpn_eval_file_path)
    degree_thres_list += [360]
    shift_thres_list += [100]

    iou_25_idx = iou_thres_list.index(0.25)
    iou_50_idx = iou_thres_list.index(0.5)
    iou_75_idx = iou_thres_list.index(0.75)
    degree_05_idx = degree_thres_list.index(5)
    degree_10_idx = degree_thres_list.index(10)
    shift_02_idx = shift_thres_list.index(2)
    shift_05_idx = shift_thres_list.index(5)
    shift_10_idx = shift_thres_list.index(10)

    messages = []
    cls_idx = -1
    iou_aps = iou_aps_dualposenet
    pose_aps = pose_aps_dualposenet
    messages.append('average mAP:')
    messages.append('3D IoU at 25: {:.1f}'.format(iou_aps[cls_idx, iou_25_idx] * 100))
    messages.append('3D IoU at 50: {:.1f}'.format(iou_aps[cls_idx, iou_50_idx] * 100))
    messages.append('3D IoU at 75: {:.1f}'.format(iou_aps[cls_idx, iou_75_idx] * 100))
    messages.append('5 degree, 2cm: {:.1f}'.format(pose_aps[cls_idx, degree_05_idx, shift_02_idx] * 100))
    messages.append('5 degree, 5cm: {:.1f}'.format(pose_aps[cls_idx, degree_05_idx, shift_05_idx] * 100))
    messages.append('10 degree, 2cm: {:.1f}'.format(pose_aps[cls_idx, degree_10_idx, shift_02_idx] * 100))
    messages.append('10 degree, 5cm: {:.1f}'.format(pose_aps[cls_idx, degree_10_idx, shift_05_idx] * 100))
    messages.append('10 degree, 10cm: {:.1f}'.format(pose_aps[cls_idx, degree_10_idx, shift_10_idx] * 100))

    for cls_idx in range(1, len(synset_names)):
        messages.append('category {}'.format(synset_names[cls_idx]))
        messages.append('mAP:')
        messages.append('3D IoU at 25: {:.1f}'.format(iou_aps[cls_idx, iou_25_idx] * 100))
        messages.append('3D IoU at 50: {:.1f}'.format(iou_aps[cls_idx, iou_50_idx] * 100))
        messages.append('3D IoU at 75: {:.1f}'.format(iou_aps[cls_idx, iou_75_idx] * 100))
        messages.append('5 degree, 2cm: {:.1f}'.format(pose_aps[cls_idx, degree_05_idx, shift_02_idx] * 100))
        messages.append('5 degree, 5cm: {:.1f}'.format(pose_aps[cls_idx, degree_05_idx, shift_05_idx] * 100))
        messages.append('10 degree, 2cm: {:.1f}'.format(pose_aps[cls_idx, degree_10_idx, shift_02_idx] * 100))
        messages.append('10 degree, 5cm: {:.1f}'.format(pose_aps[cls_idx, degree_10_idx, shift_05_idx] * 100))
        messages.append('10 degree, 10cm: {:.1f}'.format(pose_aps[cls_idx, degree_10_idx, shift_10_idx] * 100))
    with open(os.path.join(output_path, 'eval_all_results.txt'), 'w') as file:
        for line in messages:
            print(line)
            file.write(line+'\n')
    plot_mAP(degree_thres_list, shift_thres_list, iou_thres_list, iou_aps_dualposenet, pose_aps_dualposenet,
             os.path.join(output_path, 'dualposenet_mAP.png'), suffix='DualPoseNet')
    plot_mAP(degree_thres_list, shift_thres_list, iou_thres_list, iou_aps, pose_aps, os.path.join(output_path, 'our_mAP.png'),
             suffix='GPV-Pose')
