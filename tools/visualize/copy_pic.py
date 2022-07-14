import os
import shutil
dir = '/data2/zrd/GPV_pose_result/visualize_bbox_all_separate_CAMERA'
output_dir = '/data2/zrd/GPV_pose_result/visualize_bbox_CAMERA_sample'
filename_list = os.listdir(dir)
filename_list.sort()
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
for name in filename_list[:3000]:
    shutil.copyfile(os.path.join(dir, name), os.path.join(output_dir, name))