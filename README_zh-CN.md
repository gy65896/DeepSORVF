# <p align=center> DeepSORVF: 基于深度学习的简单在线实时船舶数据融合方法</p>

<div align="center">

[![Paper](https://img.shields.io/badge/arXiv-Paper-red.svg)](https://arxiv.org/abs/2302.11283)
[![Dataset](https://img.shields.io/badge/FVessel-Dataset-orange.svg)](https://github.com/gy65896/FVessel)
[![English](https://img.shields.io/badge/英文-English-green.svg)](README.md)
[![3.7](https://img.shields.io/badge/Python-3.7-blue.svg)](https://www.python.org/)
[![1.9.1](https://img.shields.io/badge/Pytorch-1.9.1-yellow.svg)](https://pytorch.org/)
<a target="_blank" href="https://colab.research.google.com/github/gy65896/DeepSORVF/blob/main/main_example.ipynb">
<img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>
</div>

---
>**内河船舶交通监管中基于异步轨迹匹配的多模态海事数据融合**<br> 郭彧, [刘文](http://mipc.whut.edu.cn/index.html), 瞿晶祥, 卢煜旭, 朱凤华, 吕宜生 <br> 
>arXiv preprint arXiv:2302.11283

> **简介：** *在这项工作中，我们首先提取基于AIS和视频的船舶轨迹，然后提出了一种基于深度学习的异步轨迹匹配方法（名为 DeepSORVF），以将基于AIS的船舶信息与相应的视觉目标融合。此外，通过结合基于AIS和视频的运动特征，我们还提出了一种先验知识驱动的抗遮挡方法，以在遮挡条件下产生准确且稳健的船舶跟踪结果。*
<hr />

---
![video](https://github.com/gy65896/DeepSORVF/assets/48637474/42e3590f-51d0-4f5b-81fd-85e4dd796fe6.gif)

## 流程图

![Figure01_Flowchart](https://user-images.githubusercontent.com/48637474/230878573-a26b035d-3ed0-4db9-9b58-161067632daf.jpg)
<div align=center><b>图1. DeepSORVF的流程图</b></div>

![Figure03_Video](https://user-images.githubusercontent.com/48637474/230878762-223472ae-cf19-4167-adbb-80c3f77ae9c3.jpg)
<div align=center><b>图2. 面向视觉船舶轨迹提取的抗遮挡跟踪方法流程图</b></div>


## 运行
* 将[ckpt.t7](https://drive.google.com/file/d/1QdIP5TEDALJnnpqwjXwvL1J_GoseTK9D/view?usp=share_link)保存至`DeepSORVF/deep_sort/deep_sort/deep/checkpoint/`文件夹下。
* 将[YOLOX-final.pth](https://drive.google.com/file/d/1mhah7ZzP8oAUuSMR96Or9UvqkXe-AMuS/view?usp=share_link)保存至`DeepSORVF/detection_yolox/model_data/`文件夹下。
* 设置数据路径`parser.add_argument("--data_path", type=str, default = './clip-01/', help='data path')`。
* 运行`main.py`。

#### `draw_org.py` 用于同时可视化基于 ais 的轨迹（蓝线）、目标检测框（红框）和融合结果（黑色文本）。可以通过将 `main.py` 中的 `import draw` 修改为 `import draw_org` 来启用它。
#### 测试数据: [clip-01](https://drive.google.com/file/d/1Bns1jAW1ImL-FeCQBvIUcrO0hjYLIB5K/view?usp=share_link)

## FVessel: 用于船舶检测、跟踪和数据融合的基准数据集

[FVessel](https://github.com/gy65896/FVessel)基准数据集用于评估AIS和视频数据融合算法的可靠性，主要包含海康威视DS-2DC4423IW-D球型摄像机和赛扬AIS9000-08 B类AIS接收机在武汉长江段拍摄的26个视频和相应的AIS数据。为了保护隐私，在我们的数据集中每艘船的 MMSI 已替换为随机数。图3展了FVessel数据集的部分样本。

![Figure04_FVessel](https://user-images.githubusercontent.com/48637474/210925024-15dcbcbe-717b-47b6-ad4b-377d71141380.jpg)
<div align=center><b>图3. FVessel数据集的部分样本，其中包含在晴天、多云和弱光条件下在桥区和江边采集的大量图像和视频</b></div>

## [FVessel_V1.0](https://github.com/gy65896/FVessel) 数据集上的性能
<div align=center>
 

|名称|MOFA (%)|IDP (%)|IDR (%)|IDF (%)|视频
| :-: | :-: | :-: | :-: | :-: | :-: |
[video-01](https://user-images.githubusercontent.com/48637474/236730149-e098365f-0d6a-4c56-8e18-3e47b7b3a7d6.mp4)|79.94|89.35|90.76|90.05|<img src="https://github.com/gy65896/DeepSORVF/assets/48637474/12e3c73f-8d90-49eb-a10c-710970353cee.gif" width="150">
[video-02](https://user-images.githubusercontent.com/48637474/236730157-39ce91d9-c8f9-461b-83d5-62d91df67cf9.mp4)|73.19|83.27|91.60|87.23|<img src="https://github.com/gy65896/DeepSORVF/assets/48637474/aea89f94-70d5-4191-a1d1-c422245936c7.gif" width="150">
[video-03](https://user-images.githubusercontent.com/48637474/236730165-a6b1ba80-fd36-4149-9164-5006f22ef050.mp4)|96.45|99.23|97.20|98.20|<img src="https://github.com/gy65896/DeepSORVF/assets/48637474/6959c86d-1802-46e4-b097-9741bc053634.gif" width="150">
[video-04](https://user-images.githubusercontent.com/48637474/236730303-9ce4f74c-23db-442c-b979-5dd6abb31c2a.mp4)|98.08|99.45|98.63|99.03|<img src="https://github.com/gy65896/DeepSORVF/assets/48637474/898de4fc-e395-402a-95f3-0de398df4659.gif" width="150">
[video-05](https://user-images.githubusercontent.com/48637474/236730560-0298cc43-a929-4f25-847b-3aa5d95b6653.mp4)|89.19|93.46|95.91|94.67|<img src="https://github.com/gy65896/DeepSORVF/assets/48637474/1c23ea84-a189-4d61-9208-74b8fc00e823.gif" width="150">
[video-06](https://user-images.githubusercontent.com/48637474/236730565-85284244-3229-4dbc-aacb-35e2f58b7efa.mp4)|91.17|96.04|95.08|95.56|<img src="https://github.com/gy65896/DeepSORVF/assets/48637474/0ca7c74e-7194-4a76-baa5-e52549e4d542.gif" width="150">
[video-07](https://user-images.githubusercontent.com/48637474/236730567-7568d8fc-f2ce-450a-8ddb-b863e0f9e432.mp4)|96.81|99.59|97.21|98.39|<img src="https://github.com/gy65896/DeepSORVF/assets/48637474/12e24329-e1c1-4b9d-8c23-081d7138e92f.gif" width="150">
[video-08](https://user-images.githubusercontent.com/48637474/236730569-27c4d441-e36f-46a4-9716-7d338b16814b.mp4)|82.28|99.64|82.58|90.31|<img src="https://github.com/gy65896/DeepSORVF/assets/48637474/d955420b-af9c-4818-a52f-b614e63a1ba6.gif" width="150">
[video-09](https://user-images.githubusercontent.com/48637474/236730570-c564f227-3d92-432b-8f6e-c592dfaba825.mp4)|98.45|100.00|98.45|99.22|<img src="https://github.com/gy65896/DeepSORVF/assets/48637474/09aac176-327a-475d-af88-aa4d37057399).gif" width="150">
[video-10](https://user-images.githubusercontent.com/48637474/236730572-b2873836-7774-4727-87e8-945ab9b5b9f1.mp4)|88.74|90.42|99.26|94.63|<img src="https://github.com/gy65896/DeepSORVF/assets/48637474/e7d227bb-b941-4a18-a27e-032f1a1aeaef.gif" width="150">
[video-11](https://user-images.githubusercontent.com/48637474/236730574-350e05d5-89fc-4c70-a913-a1795039cc26.mp4)|97.66|99.29|98.36|98.83|<img src="https://github.com/gy65896/DeepSORVF/assets/48637474/6ce0a14c-6706-4578-a74a-3505ec0caf72.gif" width="150">
[video-12](https://user-images.githubusercontent.com/48637474/236730577-0d68f53a-cb37-413b-afe9-33c3a182aef3.mp4)|95.45|99.06|96.36|97.69|<img src="https://github.com/gy65896/DeepSORVF/assets/48637474/f820dc97-700c-44b2-95d2-7c3c4829ce5f.gif" width="150">
[video-13](https://user-images.githubusercontent.com/48637474/236730581-0e088d7e-7648-4760-9337-10c338f1307a.mp4)|84.82|94.82|89.72|92.20|<img src="https://github.com/gy65896/DeepSORVF/assets/48637474/7e952027-072b-40e7-8476-38bd3514ce57.gif" width="150">
[video-14](https://user-images.githubusercontent.com/48637474/236730584-8fc8cda4-9f20-4954-949a-130e689d62ca.mp4)|93.10|97.82|95.22|96.50|<img src="https://github.com/gy65896/DeepSORVF/assets/48637474/f306a4fa-a5fd-48df-add0-d59880e26815.gif" width="150">
[video-15](https://user-images.githubusercontent.com/48637474/236730586-36c8c165-672a-46e2-97e3-77150278c875.mp4)|95.88|97.19|98.74|97.96|<img src="https://github.com/gy65896/DeepSORVF/assets/48637474/f4e93801-6a4e-4dff-8f8f-43464cf407b5.gif" width="150">
[video-16](https://user-images.githubusercontent.com/48637474/236730592-6a712fb8-57d3-4ed9-bd3e-d8de8e748042.mp4)|98.68|100.00|98.68|99.33|<img src="https://github.com/gy65896/DeepSORVF/assets/48637474/2667b842-1088-4d71-bf62-4301dd10909f.gif" width="150">
[video-17](https://user-images.githubusercontent.com/48637474/236730599-e51c2d9c-e135-4153-9361-ffb8f958b205.mp4)|90.02|93.80|96.39|95.08|<img src="https://github.com/gy65896/DeepSORVF/assets/48637474/460bb7cf-9766-45ec-98d8-e966b90f207e.gif" width="150">
[video-18](https://user-images.githubusercontent.com/48637474/236730605-b2d03c4d-6904-44f7-8531-c9e7da3ff957.mp4)|74.49|83.57|92.72|87.91|<img src="https://github.com/gy65896/DeepSORVF/assets/48637474/8742e696-6d07-4ec5-83d1-d5b5beb63ba0.gif" width="150">
[video-19](https://user-images.githubusercontent.com/48637474/236730610-398c2c5f-1658-493c-b1c7-c87452ac047c.mp4)|96.62|98.31|98.31|98.31|<img src="https://github.com/gy65896/DeepSORVF/assets/48637474/e6720ab9-d5ec-465d-ba5e-10ac2f739f0f.gif" width="150">
[video-20](https://user-images.githubusercontent.com/48637474/236730613-78554780-baab-4c1d-9b8b-f96ad7d804b3.mp4)|96.74|98.66|98.07|98.36|<img src="https://github.com/gy65896/DeepSORVF/assets/48637474/41e34f54-aba6-43bd-b678-69a35990a784.gif" width="150">
[video-21](https://user-images.githubusercontent.com/48637474/236730618-87445b72-dfee-4826-84be-e9f490660591.mp4)|76.43|87.03|89.82|88.40|<img src="https://github.com/gy65896/DeepSORVF/assets/48637474/bd46186b-8af1-4054-96b8-92c380b5372b.gif" width="150">
[video-22](https://user-images.githubusercontent.com/48637474/236730621-fb04106e-8cd6-42b6-9974-4123d4f39b7b.mp4)|96.82|99.35|97.45|98.39|<img src="https://github.com/gy65896/DeepSORVF/assets/48637474/2d4d5d7f-2563-453c-b215-2d8ce41c1aa5.gif" width="150">
[video-23](https://user-images.githubusercontent.com/48637474/236730626-aebe9cce-06b5-423f-b017-e1b3a3171a50.mp4)|94.71|98.91|95.77|97.31|<img src="https://github.com/gy65896/DeepSORVF/assets/48637474/977c13a7-425b-4f3c-9baa-2d91e8317f49.gif" width="150">
[video-24](https://user-images.githubusercontent.com/48637474/236730628-9db4a034-1016-4c32-a8ad-e77dd730dc7c.mp4)|94.70|98.34|96.33|97.32|<img src="https://github.com/gy65896/DeepSORVF/assets/48637474/e69f4950-1efb-4231-8e2d-90f249bc305b.gif" width="150">
[video-25](https://user-images.githubusercontent.com/48637474/236730630-48d8bf1b-0059-4885-afc9-1a2f4dc2b3d5.mp4)|91.49|97.66|93.73|95.66|<img src="https://github.com/gy65896/DeepSORVF/assets/48637474/9cc6ab19-fcd8-4f5c-b76b-a3c6de9c8e55.gif" width="150">
[video-26](https://user-images.githubusercontent.com/48637474/236730632-da8b0ff3-9edb-461b-9d3d-6ac22f1b56d3.mp4)|97.44|99.11|98.32|98.72|<img src="https://github.com/gy65896/DeepSORVF/assets/48637474/9386c942-c5a4-4ebd-b931-29f1a86cba40.gif" width="150">
均值 |91.13|95.90|95.41|95.59|...

</div>

## 致谢

非常感谢武汉理工大学计算机与人工智能学院的**苏建龙**进行的数据采集和算法实现工作。

## 引用

```
@article{guo2023asynchronous,
  title={Asynchronous Trajectory Matching-Based Multimodal Maritime Data Fusion for Vessel Traffic Surveillance in Inland Waterways},
  author={Guo, Yu and Liu, Ryan Wen and Qu, Jingxiang and Lu, Yuxu and Zhu, Fenghua, and Lv, Yisheng},
  journal={IEEE Transactions on Intelligent Transportation Systems},
  year={2023}
}
```

#### 如果您有更多问题，请联系我们(yuguo@whut.edu.cn & wenliu@whut.edu.cn)。

## 参考资料

https://github.com/bubbliiiing/yolox-pytorch

https://github.com/dyh/unbox_yolov5_deepsort_counting/tree/main/deep_sort
