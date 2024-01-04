# <p align=center> DeepSORVF: 基于深度学习的简单在线实时船舶数据融合方法</p>

<div align="center">

[![Paper](https://img.shields.io/badge/PDF-Paper-blue.svg)](https://ieeexplore.ieee.org/abstract/document/10159572)
[![Paper](https://img.shields.io/badge/arXiv-Paper-red.svg)](https://arxiv.org/abs/2302.11283)
[![Dataset](https://img.shields.io/badge/FVessel-Dataset-orange.svg)](https://github.com/gy65896/FVessel)
[![English](https://img.shields.io/badge/英文-English-green.svg)](README.md)
[![3.7](https://img.shields.io/badge/Python-3.7-pink.svg)](https://www.python.org/)
[![1.9.1](https://img.shields.io/badge/Pytorch-1.9.1-yellow.svg)](https://pytorch.org/)
<a target="_blank" href="https://colab.research.google.com/github/gy65896/DeepSORVF/blob/main/main_example.ipynb">
<img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>
</div>

---
>**内河船舶交通监管中基于异步轨迹匹配的多模态海事数据融合**<br> [郭彧](https://scholar.google.com/citations?user=klYz-acAAAAJ&hl=zh-CN), [刘文*](http://mipc.whut.edu.cn/index.html), [瞿晶祥](https://scholar.google.com/citations?user=9zK-zGoAAAAJ&hl=zh-CN), [卢煜旭](https://scholar.google.com/citations?user=XXge2_0AAAAJ&hl=zh-CN), 朱凤华*, 吕宜生 (* 代表通讯作者) <br> 
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
[video-01](https://github.com/gy65896/DeepSORVF/assets/48637474/a3d4a688-e97b-4fdf-b0be-ecc536e41134)|79.94|89.35|90.76|90.05|<img src="https://github.com/gy65896/DeepSORVF/assets/48637474/12e3c73f-8d90-49eb-a10c-710970353cee.gif" width="150">
[video-02](https://github.com/gy65896/DeepSORVF/assets/48637474/d52b4388-aa8f-4293-9898-2a7913d600df)|73.19|83.27|91.60|87.23|<img src="https://github.com/gy65896/DeepSORVF/assets/48637474/aea89f94-70d5-4191-a1d1-c422245936c7.gif" width="150">
[video-03](https://github.com/gy65896/DeepSORVF/assets/48637474/182ae077-dc8f-4773-bb04-499aa0eee90e)|96.45|99.23|97.20|98.20|<img src="https://github.com/gy65896/DeepSORVF/assets/48637474/6959c86d-1802-46e4-b097-9741bc053634.gif" width="150">
[video-04](https://github.com/gy65896/DeepSORVF/assets/48637474/1509e058-fa36-4cfd-8cd5-c00d12a29dce)|98.08|99.45|98.63|99.03|<img src="https://github.com/gy65896/DeepSORVF/assets/48637474/898de4fc-e395-402a-95f3-0de398df4659.gif" width="150">
[video-05](https://github.com/gy65896/DeepSORVF/assets/48637474/20aa62d4-ea5a-4c3e-a28a-ded9f7f97b6d)|89.19|93.46|95.91|94.67|<img src="https://github.com/gy65896/DeepSORVF/assets/48637474/1c23ea84-a189-4d61-9208-74b8fc00e823.gif" width="150">
[video-06](https://github.com/gy65896/DeepSORVF/assets/48637474/c4634627-1472-48a3-8cbd-6b41db3c870b)|91.17|96.04|95.08|95.56|<img src="https://github.com/gy65896/DeepSORVF/assets/48637474/0ca7c74e-7194-4a76-baa5-e52549e4d542.gif" width="150">
[video-07](https://github.com/gy65896/DeepSORVF/assets/48637474/65f00649-ae62-4694-9e3e-d5f40bc7989a)|96.81|99.59|97.21|98.39|<img src="https://github.com/gy65896/DeepSORVF/assets/48637474/12e24329-e1c1-4b9d-8c23-081d7138e92f.gif" width="150">
[video-08](https://github.com/gy65896/DeepSORVF/assets/48637474/07c0fd13-c2ac-4212-9212-a3c89990a083)|82.28|99.64|82.58|90.31|<img src="https://github.com/gy65896/DeepSORVF/assets/48637474/d955420b-af9c-4818-a52f-b614e63a1ba6.gif" width="150">
[video-09](https://github.com/gy65896/DeepSORVF/assets/48637474/4f257919-2630-4a63-932b-6166e15b599d)|98.45|100.00|98.45|99.22|<img src="https://github.com/gy65896/DeepSORVF/assets/48637474/09aac176-327a-475d-af88-aa4d37057399).gif" width="150">
[video-10](https://github.com/gy65896/DeepSORVF/assets/48637474/1d6f0fe2-30c4-4a79-a36f-fab5890ae2a8)|88.74|90.42|99.26|94.63|<img src="https://github.com/gy65896/DeepSORVF/assets/48637474/e7d227bb-b941-4a18-a27e-032f1a1aeaef.gif" width="150">
[video-11](https://github.com/gy65896/DeepSORVF/assets/48637474/e837faf5-4791-4fa0-a04a-203583bb6939)|97.66|99.29|98.36|98.83|<img src="https://github.com/gy65896/DeepSORVF/assets/48637474/6ce0a14c-6706-4578-a74a-3505ec0caf72.gif" width="150">
[video-12](https://github.com/gy65896/DeepSORVF/assets/48637474/792f82eb-ba4b-41c4-8acd-68915db30517)|95.45|99.06|96.36|97.69|<img src="https://github.com/gy65896/DeepSORVF/assets/48637474/f820dc97-700c-44b2-95d2-7c3c4829ce5f.gif" width="150">
[video-13](https://github.com/gy65896/DeepSORVF/assets/48637474/33a59328-3379-4207-ae04-bc65b250aaf8)|84.82|94.82|89.72|92.20|<img src="https://github.com/gy65896/DeepSORVF/assets/48637474/7e952027-072b-40e7-8476-38bd3514ce57.gif" width="150">
[video-14](https://github.com/gy65896/DeepSORVF/assets/48637474/854edbbb-9745-47f8-88a6-a4fea484b2a6)|93.10|97.82|95.22|96.50|<img src="https://github.com/gy65896/DeepSORVF/assets/48637474/f306a4fa-a5fd-48df-add0-d59880e26815.gif" width="150">
[video-15](https://github.com/gy65896/DeepSORVF/assets/48637474/21b09a0e-f2e2-4bb0-859d-afcc13d15c0f)|95.88|97.19|98.74|97.96|<img src="https://github.com/gy65896/DeepSORVF/assets/48637474/f4e93801-6a4e-4dff-8f8f-43464cf407b5.gif" width="150">
[video-16](https://github.com/gy65896/DeepSORVF/assets/48637474/bbe6349c-448a-4db3-8fd9-a07640fc0086)|98.68|100.00|98.68|99.33|<img src="https://github.com/gy65896/DeepSORVF/assets/48637474/2667b842-1088-4d71-bf62-4301dd10909f.gif" width="150">
[video-17](https://github.com/gy65896/DeepSORVF/assets/48637474/2454eae7-884d-4c0a-bf48-2f8c7ec537be)|90.02|93.80|96.39|95.08|<img src="https://github.com/gy65896/DeepSORVF/assets/48637474/460bb7cf-9766-45ec-98d8-e966b90f207e.gif" width="150">
[video-18](https://github.com/gy65896/DeepSORVF/assets/48637474/a79303b7-f7ff-4774-8888-f8efdc60264d)|74.49|83.57|92.72|87.91|<img src="https://github.com/gy65896/DeepSORVF/assets/48637474/8742e696-6d07-4ec5-83d1-d5b5beb63ba0.gif" width="150">
[video-19](https://github.com/gy65896/DeepSORVF/assets/48637474/40fd6863-5388-4a84-89ea-785bce231099)|96.62|98.31|98.31|98.31|<img src="https://github.com/gy65896/DeepSORVF/assets/48637474/e6720ab9-d5ec-465d-ba5e-10ac2f739f0f.gif" width="150">
[video-20](https://github.com/gy65896/DeepSORVF/assets/48637474/33209a42-f39d-45bd-ba2c-0b23af963eaa)|96.74|98.66|98.07|98.36|<img src="https://github.com/gy65896/DeepSORVF/assets/48637474/41e34f54-aba6-43bd-b678-69a35990a784.gif" width="150">
[video-21](https://github.com/gy65896/DeepSORVF/assets/48637474/46f353b7-057d-4c33-8438-bc0561dc50a2)|76.43|87.03|89.82|88.40|<img src="https://github.com/gy65896/DeepSORVF/assets/48637474/bd46186b-8af1-4054-96b8-92c380b5372b.gif" width="150">
[video-22](https://github.com/gy65896/DeepSORVF/assets/48637474/850278fd-4850-4d19-867e-abdd859da3b9)|96.82|99.35|97.45|98.39|<img src="https://github.com/gy65896/DeepSORVF/assets/48637474/2d4d5d7f-2563-453c-b215-2d8ce41c1aa5.gif" width="150">
[video-23](https://github.com/gy65896/DeepSORVF/assets/48637474/bd112bf3-60f4-41ff-8b1a-86b30e184256)|94.71|98.91|95.77|97.31|<img src="https://github.com/gy65896/DeepSORVF/assets/48637474/977c13a7-425b-4f3c-9baa-2d91e8317f49.gif" width="150">
[video-24](https://github.com/gy65896/DeepSORVF/assets/48637474/51dba4d0-89fe-4d00-a755-2d49b67da62e)|94.70|98.34|96.33|97.32|<img src="https://github.com/gy65896/DeepSORVF/assets/48637474/e69f4950-1efb-4231-8e2d-90f249bc305b.gif" width="150">
[video-25](https://github.com/gy65896/DeepSORVF/assets/48637474/249de603-b688-4751-9cfc-3f0d523b57d5)|91.49|97.66|93.73|95.66|<img src="https://github.com/gy65896/DeepSORVF/assets/48637474/9cc6ab19-fcd8-4f5c-b76b-a3c6de9c8e55.gif" width="150">
[video-26](https://github.com/gy65896/DeepSORVF/assets/48637474/6f1eb804-b453-4995-91db-b08885ffbc4a)|97.44|99.11|98.32|98.72|<img src="https://github.com/gy65896/DeepSORVF/assets/48637474/9386c942-c5a4-4ebd-b931-29f1a86cba40.gif" width="150">
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
<!--
#### DeepSORVF 仅可用于非商业研究目的。 索取完整代码时，请使用您的机构或学校电子邮件地址联系我们，仅供研究之用。 (yuguo@whut.edu.cn & wenliu@whut.edu.cn)
-->
## 参考资料

https://github.com/bubbliiiing/yolox-pytorch

https://github.com/dyh/unbox_yolov5_deepsort_counting/tree/main/deep_sort
