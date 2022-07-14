data_dir = '/data2/zrd/GPV_pose_result/dualposenet_results/CAMERA25'
import os, mmcv

filelist = os.listdir(data_dir)
filelist.sort()
print(filelist)
result_list = []
for filename in filelist:
    result = mmcv.load(os.path.join(data_dir, filename))
    result_list.append(result)

mmcv.dump(result_list, '/data2/zrd/GPV_pose_result/dualposenet_results/CAMERA25_results.pkl')
