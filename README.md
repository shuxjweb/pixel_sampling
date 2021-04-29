# Semantic-guided Pixel Sampling for Cloth-Changing Person Re-identification

In our paper, we propose a semanticguided pixel sampling approach for the cloth-changing person re-ID task.  This repo contains the training and testing codes.

## Prepare Dataset
1. Download the PRCC dataset:  [PRCC](http://isee-ai.cn/~yangqize/clothing.html)
2. Obtain the human body parts: [SCHP](https://github.com/PeikeLi/Self-Correction-Human-Parsing)

## Trained Models
The trained models can be downloaded in BaiduPan: [models](https://pan.baidu.com/s/1Lx-6a95IYUYds0GUh4RpUQ) password: y5nc

Put the trained models to corresponding directories:
>pixel_sampling/imagenet/resnet50-19c8e357.pth
>pixel_sampling/logs/prcc_base/checkpoint_best.pth
>pixel_sampling/logs/prcc_hpm/checkpoint_best.pth
>...... 
 
 ## Training and Testing Models
 Only need to modify several parameters:
 >parser.add_argument('--train', type=str, default='train', help='train, test')
 >parser.add_argument('--data_dir', type=str, default='/data/prcc/')

then
python train_prcc_base.py
