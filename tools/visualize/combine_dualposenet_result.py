import os
import mmcv
result_dir = '/data2/zrd/GPV_pose_result/dualposenet_results/REAL275'
save_path = '/data2/zrd/GPV_pose_result/dualposenet_results/REAL275_results.pkl'

file_list = os.listdir(result_dir)
total_result = []
for file_name in file_list:
    result = mmcv.load(os.path.join(result_dir,file_name))
    total_result.append(result)

mmcv.dump(total_result, save_path)

