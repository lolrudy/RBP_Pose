# RBP-Pose
Pytorch implementation of RBP-Pose: Residual Bounding Box Projection for Category-Level Pose Estimation.

[//]: # (&#40;[link]&#40;https://arxiv.org/abs/2203.07918&#41;&#41;)

![pipeline](pic/pipeline.png)

## Required environment

- Ubuntu 18.04
- Python 3.8 
- Pytorch 1.10.1
- CUDA 11.3.
 


## Installing

- Install the main requirements in 'requirement.txt'.
- Install [Detectron2](https://github.com/facebookresearch/detectron2).

## Data Preparation
To generate your own dataset, use the data preprocess code provided in this [git](https://github.com/mentian/object-deformnet/blob/master/preprocess/pose_data.py).
Download the detection results in this [link](https://drive.google.com/drive/folders/1q8pjmHDfSUTna13F2R_gU3P-FYCjEP7A?usp=sharing).


## Trained model
Trained model is available [here](https://drive.google.com/drive/folders/1q8pjmHDfSUTna13F2R_gU3P-FYCjEP7A?usp=sharing).

## Training
Please note, some details are changed from the original paper for more efficient training. 

Specify the dataset directory and run the following command.
```shell
python -m engine.train --data_dir YOUR_DATA_DIR --model_save SAVE_DIR --training_stage shape_prior_only # first stage
python -m engine.train --data_dir YOUR_DATA_DIR --model_save SAVE_DIR --resume 1 --resume_model MODEL_PATH--training_stage prior+recon+novote # second stage
```

Detailed configurations are in 'config/config.py'.

## Evaluation
```shell
python -m evaluation.evaluate --data_dir YOUR_DATA_DIR --detection_dir DETECTION_DIR --resume 1 --resume_model MODEL_PATH --model_save SAVE_DIR
```


## Acknowledgment
Our implementation leverages the code from [3dgcn](https://github.com/j1a0m0e4sNTU/3dgcn), [FS-Net](https://github.com/DC1991/FS_Net),
[DualPoseNet](https://github.com/Gorilla-Lab-SCUT/DualPoseNet), [SPD](https://github.com/mentian/object-deformnet).
