import os
import mmcv
result_dir = '/data/zrd/datasets/NOCS/results/sgpa_results/real'
save_path = '/data/zrd/datasets/NOCS/results/sgpa_results/REAL275_results.pkl'

file_list = os.listdir(result_dir)
total_result = []
total_result_dict = {}
for file_name in file_list:
    if file_name.endswith('pkl') and file_name.startswith('result'):
        result = mmcv.load(os.path.join(result_dir,file_name))
        total_result.append(result)
print(len(total_result))
mmcv.dump(total_result, save_path)

