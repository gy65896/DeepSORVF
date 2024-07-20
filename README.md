
# <p align=center> [IEEE TITS 2023] DeepSORVF: Deep Learning-based Simple Online and Real-Time Vessel Data Fusion Method</p>

<div align="center">

[![Paper](https://img.shields.io/badge/PDF-Paper-red.svg)](https://ieeexplore.ieee.org/abstract/document/10159572)
[![Web](https://img.shields.io/badge/OneRestore-Web-blue.svg)](https://gy65896.github.io/projects/TITS2023_DeepSORVF/index.html)
[![Dataset](https://img.shields.io/badge/FVessel-Dataset-orange.svg)](https://github.com/gy65896/FVessel)
[![Chinese](https://img.shields.io/badge/简体中文-Chinese-green.svg)](README_zh-CN.md)
[![3.7](https://img.shields.io/badge/Python-3.7-pink.svg)](https://www.python.org/)
[![1.9.1](https://img.shields.io/badge/Pytorch-1.9.1-yellow.svg)](https://pytorch.org/)
[![Hits](https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https%3A%2F%2Fgithub.com%2Fgy65896%2FFVessel&count_bg=%2379C83D&title_bg=%23555555&icon=&icon_color=%23E7E7E7&title=visitor&edge_flat=false)](https://hits.seeyoufarm.com)
<a target="_blank" href="https://colab.research.google.com/github/gy65896/DeepSORVF/blob/main/main_example.ipynb">
<img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

</div>

---
>**Asynchronous Trajectory Matching-Based Multimodal Maritime Data Fusion for Vessel Traffic Surveillance in Inland Waterways**<br>  [Yu Guo](https://scholar.google.com/citations?user=klYz-acAAAAJ&hl=zh-CN), [Ryan Wen Liu*](http://mipc.whut.edu.cn/index.html), [Jingxiang Qu](https://scholar.google.com/citations?user=9zK-zGoAAAAJ&hl=zh-CN), [Yuxu Lu](https://scholar.google.com/citations?user=XXge2_0AAAAJ&hl=zh-CN), Fenghua Zhu*, Yisheng Lv (* indicates corresponding author)<br> 
>IEEE Transactions on Intelligent Transportation Systems

> **Introduction:** *In this work, we first extract the AIS- and video-based vessel trajectories, and then propose a deep learning-enabled asynchronous trajectory matching method (named DeepSORVF) to fuse the AIS-based vessel information with the corresponding visual targets. In addition, by combining the AIS- and video-based movement features, we also present a prior knowledge-driven anti-occlusion method to yield accurate and robust vessel tracking results under occlusion conditions.*
<hr />

![video](https://github.com/gy65896/DeepSORVF/assets/48637474/42e3590f-51d0-4f5b-81fd-85e4dd796fe6.gif)

## Flowchart

![Figure01_Flowchart](https://user-images.githubusercontent.com/48637474/230878573-a26b035d-3ed0-4db9-9b58-161067632daf.jpg)
<div align=center><b>Figure 1. The architecture of the proposed deep learning-based simple online and real-time vessel data fusion method.</b></div>

![Figure03_Video](https://user-images.githubusercontent.com/48637474/230878762-223472ae-cf19-4167-adbb-80c3f77ae9c3.jpg)
<div align=center><b>Figure 2. The flowchart of anti-occlusion tracking method for video-based vessel trajectory extraction.</b></div>

## Environment
* Python3.7
* easydict 1.11
* geopy 2.4.1
* pyproj 3.2.1
* fastdtw 0.3.4
* pytorch 1.13.1
* cuda 11.7
* pandas 1.3.5
* numpy 1.21.6

## Running
* Save [ckpt.t7](https://drive.google.com/file/d/1QdIP5TEDALJnnpqwjXwvL1J_GoseTK9D/view?usp=share_link) to `DeepSORVF/deep_sort/deep_sort/deep/checkpoint/` folder.
* Save [YOLOX-final.pth](https://drive.google.com/file/d/1mhah7ZzP8oAUuSMR96Or9UvqkXe-AMuS/view?usp=share_link) to `DeepSORVF/detection_yolox/model_data/` folder.
* Set data dir by `parser.add_argument("--data_path", type=str, default = './clip-01/', help='data path')`.
* Run `main.py`.


#### `draw_org.py` is used to simultaneously visualize the ais-based trajectory (blue line), target detection box (red box), and fusion results (black text). It can be enabled by modifying `import draw` in `main.py` to `import draw_org`.
#### Test Data: [clip-01](https://drive.google.com/file/d/1Bns1jAW1ImL-FeCQBvIUcrO0hjYLIB5K/view?usp=share_link)

## FVessel: Benchmark Dataset for Vessel Detection, Tracking, and Data Fusion

The [FVessel](https://github.com/gy65896/FVessel) benchmark dataset is used to evaluate the reliability of AIS and video data fusion algorithms, which mainly contains 26 videos and the corresponding AIS data captured by the HIKVISION DS-2DC4423IW-D dome camera and Saiyang AIS9000-08 Class-B AIS receiver on the Wuhan Segment of the Yangtze River. To protect privacy, the MMSI for each vessel has been replaced with a random number in our dataset. As shown in Figure 1, these videos were captured under many locations (e.g., bridge region and riverside) and various weather conditions (e.g., sunny, cloudy, and low-light).

![Figure04_FVessel](https://user-images.githubusercontent.com/48637474/210925024-15dcbcbe-717b-47b6-ad4b-377d71141380.jpg)
<div align=center><b>Figure 3. Some samples of the FVessel dataset, which contains massive images and videos captured on the bridge region and riverside under sunny, cloudy, and low-light conditions.</b></div>

## Performance on [FVessel_V1.0](https://github.com/gy65896/FVessel)
<div align=center>

|Name|MOFA (%)|IDP (%)|IDR (%)|IDF (%)
| :-: | :-: | :-: | :-: | :-: |
[video-01](https://github.com/gy65896/DeepSORVF/assets/48637474/a3d4a688-e97b-4fdf-b0be-ecc536e41134)|79.94|89.35|90.76|90.05
[video-02](https://github.com/gy65896/DeepSORVF/assets/48637474/d52b4388-aa8f-4293-9898-2a7913d600df)|73.19|83.27|91.60|87.23
[video-03](https://github.com/gy65896/DeepSORVF/assets/48637474/182ae077-dc8f-4773-bb04-499aa0eee90e)|96.45|99.23|97.20|98.20
[video-04](https://github.com/gy65896/DeepSORVF/assets/48637474/1509e058-fa36-4cfd-8cd5-c00d12a29dce)|98.08|99.45|98.63|99.03
[video-05](https://github.com/gy65896/DeepSORVF/assets/48637474/20aa62d4-ea5a-4c3e-a28a-ded9f7f97b6d)|89.19|93.46|95.91|94.67
[video-06](https://github.com/gy65896/DeepSORVF/assets/48637474/c4634627-1472-48a3-8cbd-6b41db3c870b)|91.17|96.04|95.08|95.56
[video-07](https://github.com/gy65896/DeepSORVF/assets/48637474/65f00649-ae62-4694-9e3e-d5f40bc7989a)|96.81|99.59|97.21|98.39
[video-08](https://github.com/gy65896/DeepSORVF/assets/48637474/07c0fd13-c2ac-4212-9212-a3c89990a083)|82.28|99.64|82.58|90.31
[video-09](https://github.com/gy65896/DeepSORVF/assets/48637474/4f257919-2630-4a63-932b-6166e15b599d)|98.45|100.00|98.45|99.22
[video-10](https://github.com/gy65896/DeepSORVF/assets/48637474/1d6f0fe2-30c4-4a79-a36f-fab5890ae2a8)|88.74|90.42|99.26|94.63
[video-11](https://github.com/gy65896/DeepSORVF/assets/48637474/e837faf5-4791-4fa0-a04a-203583bb6939)|97.66|99.29|98.36|98.83
[video-12](https://github.com/gy65896/DeepSORVF/assets/48637474/792f82eb-ba4b-41c4-8acd-68915db30517)|95.45|99.06|96.36|97.69
[video-13](https://github.com/gy65896/DeepSORVF/assets/48637474/33a59328-3379-4207-ae04-bc65b250aaf8)|84.82|94.82|89.72|92.20
[video-14](https://github.com/gy65896/DeepSORVF/assets/48637474/854edbbb-9745-47f8-88a6-a4fea484b2a6)|93.10|97.82|95.22|96.50
[video-15](https://github.com/gy65896/DeepSORVF/assets/48637474/21b09a0e-f2e2-4bb0-859d-afcc13d15c0f)|95.88|97.19|98.74|97.96
[video-16](https://github.com/gy65896/DeepSORVF/assets/48637474/bbe6349c-448a-4db3-8fd9-a07640fc0086)|98.68|100.00|98.68|99.33
[video-17](https://github.com/gy65896/DeepSORVF/assets/48637474/2454eae7-884d-4c0a-bf48-2f8c7ec537be)|90.02|93.80|96.39|95.08
[video-18](https://github.com/gy65896/DeepSORVF/assets/48637474/a79303b7-f7ff-4774-8888-f8efdc60264d)|74.49|83.57|92.72|87.91
[video-19](https://github.com/gy65896/DeepSORVF/assets/48637474/40fd6863-5388-4a84-89ea-785bce231099)|96.62|98.31|98.31|98.31
[video-20](https://github.com/gy65896/DeepSORVF/assets/48637474/33209a42-f39d-45bd-ba2c-0b23af963eaa)|96.74|98.66|98.07|98.36
[video-21](https://github.com/gy65896/DeepSORVF/assets/48637474/46f353b7-057d-4c33-8438-bc0561dc50a2)|76.43|87.03|89.82|88.40
[video-22](https://github.com/gy65896/DeepSORVF/assets/48637474/850278fd-4850-4d19-867e-abdd859da3b9)|96.82|99.35|97.45|98.39
[video-23](https://github.com/gy65896/DeepSORVF/assets/48637474/bd112bf3-60f4-41ff-8b1a-86b30e184256)|94.71|98.91|95.77|97.31
[video-24](https://github.com/gy65896/DeepSORVF/assets/48637474/51dba4d0-89fe-4d00-a755-2d49b67da62e)|94.70|98.34|96.33|97.32
[video-25](https://github.com/gy65896/DeepSORVF/assets/48637474/249de603-b688-4751-9cfc-3f0d523b57d5)|91.49|97.66|93.73|95.66
[video-26](https://github.com/gy65896/DeepSORVF/assets/48637474/6f1eb804-b453-4995-91db-b08885ffbc4a)|97.44|99.11|98.32|98.72
Average |91.13|95.90|95.41|95.59|...

</div>

## Acknowledgements

We deeply thank **Jianlong Su** from the School of Computer and Artificial Intelligence in Wuhan University of Technology who performs the data acquisition and algorithm implementation works.

## Citation

```
@article{guo2023asynchronous,
  title={Asynchronous Trajectory Matching-Based Multimodal Maritime Data Fusion for Vessel Traffic Surveillance in Inland Waterways},
  author={Guo, Yu and Liu, Ryan Wen and Qu, Jingxiang and Lu, Yuxu and Zhu, Fenghua, and Lv, Yisheng},
  journal={IEEE Transactions on Intelligent Transportation Systems},
  year={2023}
}
```
<!--
#### The DeepSORVF is available for non-commercial research purposes only. When requesting the complete code, please contact us using your institutional or school email address exclusively for research purposes. (yuguo@whut.edu.cn & wenliu@whut.edu.cn)
-->

## Reference

https://github.com/bubbliiiing/yolox-pytorch

https://github.com/dyh/unbox_yolov5_deepsort_counting/tree/main/deep_sort

</div>
<p align="center"> 
  Visitor count<br>
  <img src="https://profile-counter.glitch.me/gy65896_DeepSORVF/count.svg" />
</p>
