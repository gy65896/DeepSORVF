# DeepSORVF: 基于深度学习的简单在线实时船舶数据融合方法

---
### Asynchronous Trajectory Matching-Based Multimodal Maritime Data Fusion for Vessel Traffic Surveillance in Inland Waterways [[Paper](http://arxiv.org/abs/2302.11283)]

![video](https://user-images.githubusercontent.com/48637474/220859261-33458b91-2f2b-4d58-8c26-73610c53ca37.gif)

## 介绍
[English](README.md) | 简体中文

在这项工作中，我们首先提取基于AIS和视频的船舶轨迹，然后提出了一种基于深度学习的异步轨迹匹配方法（名为 DeepSORVF），以将基于AIS的船舶信息与相应的视觉目标融合。此外，通过结合基于AIS和视频的运动特征，我们还提出了一种先验知识驱动的抗遮挡方法，以在遮挡条件下产生准确且稳健的船舶跟踪结果。

![Figure01_Flowchart](https://user-images.githubusercontent.com/48637474/230878573-a26b035d-3ed0-4db9-9b58-161067632daf.jpg)
**图1. DeepSORVF的流程图**

![Figure03_Video](https://user-images.githubusercontent.com/48637474/230878762-223472ae-cf19-4167-adbb-80c3f77ae9c3.jpg)
**图2. 面向视觉船舶轨迹提取的抗遮挡跟踪方法流程图**

## 环境准备

- Python 3.7
- Pytorch 1.9.1
- pandas
- re
- 将[ckpt.t7](https://drive.google.com/file/d/1QdIP5TEDALJnnpqwjXwvL1J_GoseTK9D/view?usp=share_link)保存至`DeepSORVF/deep_sort/deep_sort/deep/checkpoint/`文件夹下。
- 将[YOLOX-final.pth](https://drive.google.com/file/d/1mhah7ZzP8oAUuSMR96Or9UvqkXe-AMuS/view?usp=share_link)保存至`DeepSORVF/detection_yolox/model_data/`文件夹下。

## 运行

* 设置数据路径`parser.add_argument("--data_path", type=str, default = './clip-01/', help='data path')`。

* 运行`main.py`。

#### 测试数据: [clip-01](https://drive.google.com/file/d/1Bns1jAW1ImL-FeCQBvIUcrO0hjYLIB5K/view?usp=share_link)

## FVessel: 用于船舶检测、跟踪和数据融合的基准数据集

[FVessel](https://github.com/gy65896/FVessel)基准数据集用于评估AIS和视频数据融合算法的可靠性，主要包含海康威视DS-2DC4423IW-D球型摄像机和赛扬AIS9000-08 B类AIS接收机在武汉长江段拍摄的26个视频和相应的AIS数据。为了保护隐私，在我们的数据集中每艘船的 MMSI 已替换为随机数。图3展了FVessel数据集的部分样本。

![Figure04_FVessel](https://user-images.githubusercontent.com/48637474/210925024-15dcbcbe-717b-47b6-ad4b-377d71141380.jpg)
**图3. FVessel数据集的部分样本，其中包含在晴天、多云和弱光条件下在桥区和江边采集的大量图像和视频**

## 引用

```
@article{guo2023asynchronous,
  title={Asynchronous Trajectory Matching-Based Multimodal Maritime Data Fusion for Vessel Traffic Surveillance in Inland Waterways},
  author={Guo, Yu and Liu, Ryan Wen and Qu, Jingxiang and Lu, Yuxu and Zhu, Fenghua, and Lv, Yisheng},
  journal={arXiv preprint arXiv:2302.11283},
  year={2023}
}
```

#### 如果您有更多问题，请联系我们(yuguo@whut.edu.cn & wenliu@whut.edu.cn)。

## 参考资料

https://github.com/bubbliiiing/yolox-pytorch

https://github.com/dyh/unbox_yolov5_deepsort_counting/tree/main/deep_sort
