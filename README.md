# ImageNet经典网络论文阅读分析
## ImageNet比赛介绍
ILSVRC（ImageNet Large Scale Visual Recognition Challenge）是近年来机器视觉领域最受追捧也是最具权威的学术竞赛之一，代表了图像领域的最高水平。ImageNet数据集是ILSVRC竞赛使用的是数据集，由斯坦福大学李飞飞教授主导，包含了超过1400万张全尺寸的有标记图片。ILSVRC比赛会每年从ImageNet数据集中抽出部分样本，以2012年为例，比赛的训练集包含1281167张图片，验证集包含50000张图片，测试集为100000张图片。

## 为什么要关注ImageNet
在ImageNet长达六年的比赛时间中，涌现了一大批性能优秀结构新颖的网络。我们通过分析历年优秀模型的结构和性能，可以清晰的把握卷积神经网络发展的脉络。我们主要关注比赛的分类、定位和目标检测三个项目，可以得到历年冠军表如下：
年份|分类|定位|检测
|---|---|---|---|
2012|<b>AlexNet|<b>AlexNet|—————
2013|Clarifai|OverFeat|UvA
2014|<b>GoogleNet|<b>VGG|<b>GoogleNet
2015|<b>ResNet|<b>ResNet|<b>ResNet
2016|Trimps-Soushen|Trimps-Soushen|CUImage
2017|SENet|DPN|BDAT

准备介绍的网络论文列表如下：
- [x] VGGNet
- [x] ResNet
- [x] AlexNet
- [x] ZF-Net（模型本身不算特别优秀，但提供了一种可视化分析方法）
- [x] GoogleNet
---------------------------
- [ ] RCNN（多个模型用作定位和检测的基础模型）
- [ ] SENet
