import numpy as np
data_dir = '/data/zrd/project/GPV_pose_shape_prior/output/modelsave_prior_laptop_camera+real/invalid_image'
import matplotlib.pyplot as plt
import os
rgb_path = os.path.join(data_dir, 'invalid_num_12_0_rgb.npy')
rgb = np.load(rgb_path)
rgb = np.rollaxis(rgb, 0, 3).astype(int)
depth_path = rgb_path.replace('rgb', 'depth')
depth = np.load(depth_path)
depth = np.rollaxis(depth, 0, 3)
depth = depth / np.max(depth)
nocs_path = rgb_path.replace('rgb', 'coord')
nocs = np.load(nocs_path)
nocs = np.rollaxis(nocs, 0, 3) + 0.5
plt.imshow(rgb)
plt.show()
plt.imshow(depth)
plt.show()
plt.imshow(nocs)
plt.show()
