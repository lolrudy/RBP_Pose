data_dir = '/data2/zrd/GPV_pose_result/dualposenet_results/CAMERA25'
import os, mmcv

filelist = os.listdir(data_dir)
filelist.sort()
print(filelist)
result_dict = {}
for filename in filelist:
    name = filename[8:-4]
    image_path = 'data/camera/'+name.replace('_', '/')
    result = mmcv.load(os.path.join(data_dir, filename))
    result_dict[image_path] = result

mmcv.dump(result_dict, '/data2/zrd/GPV_pose_result/dualposenet_results/CAMERA25_results_dict.pkl')
