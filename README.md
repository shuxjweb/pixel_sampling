# Semantic-guided Pixel Sampling for Cloth-Changing Person Re-identification

In our paper [publish](https://ieeexplore.ieee.org/abstract/document/9463711/), [arxiv](https://arxiv.org/abs/2107.11522), we propose a semantic-guided pixel sampling approach for the cloth-changing person re-ID task.  This repo contains the training and testing codes.

## Prepare Dataset
1. Download the PRCC dataset:  [PRCC](http://isee-ai.cn/~yangqize/clothing.html)
2. Obtain the human body parts: [SCHP](https://github.com/PeikeLi/Self-Correction-Human-Parsing)
3. The mask of PRCC dataset: [Baidu](https://pan.baidu.com/s/1sX1qFgo3I-4OfEdEr-opSg), password: r9kc  or [Google](https://drive.google.com/drive/folders/1HaIoKRj1R4fxjVQ9Qg_IEk2_46b1hniH?usp=sharing)


## Trained Models
The trained models can be downloaded from: [BaiduPan](https://pan.baidu.com/s/1JOOJp_NPbsU19DdBr7ze9g) password: 6ulj, [Google](https://drive.google.com/drive/folders/1aAltKSfRpHqADXb6sWOQ0VL7dj9GVwvU?usp=sharing)
```
Put the trained models to corresponding directories:
>pixel_sampling/imagenet/resnet50-19c8e357.pth
>pixel_sampling/logs/prcc_base/checkpoint_best.pth
>pixel_sampling/logs/prcc_hpm/checkpoint_best.pth
>...... 
 ```
 
 ## Training and Testing Models
 Only need to modify several parameters:
 ```
 >parser.add_argument('--train', type=str, default='train', help='train, test')
 
 >parser.add_argument('--data_dir', type=str, default='/data/prcc/')
```
then
```
>python train_prcc_base.py
```

## Citations
If you think this work is useful for you, please cite
```bibtex
@article{shu2021semantic,
  title={Semantic-guided Pixel Sampling for Cloth-Changing Person Re-identification},
  author={Shu, Xiujun and Li, Ge and Wang, Xiao and Ruan, Weijian and Tian, Qi},
  journal={IEEE Signal Processing Letters},
  volume={28},
  pages={1365-1369},
  year={2021}, 
}
```

If you have any questions, please contact this e-mail: shuxj@mail.ioa.ac.cn
