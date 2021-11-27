# Efficient-Kernel-XQDA-Python

# Paper: Cross-View Kernel Similarity Metric Learning Using Pairwise Constraints for Person Re-identification
Accepted in Indian Conference on Computer Vision, Graphics and Image Processing (ICVGIP) 20-21, Jodhpur, India
T M Feroz Ali, Subhasis Chaudhuri
https://arxiv.org/abs/1909.11316

This repository contains the complete code of Efficient Kernel Cross-view Quadratic Discriminant Analysis (EK-XQDA). Using this code you can reproduce our result in Table 1 (CUHK01 dataset) of our paper.
GOG + k-XQDA : 
R1 = 62.23% 
R5 = 83.09% 
R10 = 89.46%
R20 = 94.43%

Code setup:
-------------
1) You need to download the GOG features for CUHK01 dataset (available at http://www.i.kyushu-u.ac.jp/~matsukawa/ReID_files/GOG_CUHK01.zip) and place the following files inside the folder 'Features':
a) CUHK01_feature_all_GOGyMthetaHSV.mat
b) CUHK01_feature_all_GOGyMthetaLab.mat
c) CUHK01_feature_all_GOGyMthetanRnG.mat
d) CUHK01_feature_all_GOGyMthetaRGB.mat

2) Edit config.m file:
Chage the path 'directry' according to the location of code in your system.

3) Run demo_EK_XQDA.m

If you find this work useful, please kindly cite our paper.

@article{ali2019cross,
title={Cross-View Kernel Similarity Metric Learning Using Pairwise Constraints for Person Re-identification},
author={Ali, TM and Chaudhuri, Subhasis},
journal={arXiv preprint arXiv:1909.11316},
year={2019}
}
