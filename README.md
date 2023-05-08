# DeepSORVF: Deep Learning-based Simple Online and Real-Time Vessel Data Fusion Method
[![Paper](https://img.shields.io/badge/arXiv-Paper-red.svg)](https://arxiv.org/abs/2302.11283)
[![Dataset](https://img.shields.io/badge/FVessel-Dataset-orange.svg)](https://github.com/gy65896/FVessel)
[![Chinese](https://img.shields.io/badge/简体中文-Chinese-green.svg)](README_zh-CN.md)

---
>**Asynchronous Trajectory Matching-Based Multimodal Maritime Data Fusion for Vessel Traffic Surveillance in Inland Waterways**<br>  Yu Guo, [Ryan Wen Liu](http://mipc.whut.edu.cn/index.html), Jingxiang Qu, Yuxu Lu, Fenghua Zhu, Yisheng Lv <br> 
>arXiv preprint arXiv:2302.11283

> **Introduction:** *In this work, we first extract the AIS- and video-based vessel trajectories, and then propose a deep learning-enabled asynchronous trajectory matching method (named DeepSORVF) to fuse the AIS-based vessel information with the corresponding visual targets. In addition, by combining the AIS- and video-based movement features, we also present a prior knowledgedriven anti-occlusion method to yield accurate and robust vessel tracking results under occlusion conditions.*
<hr />

![video](https://user-images.githubusercontent.com/48637474/220859261-33458b91-2f2b-4d58-8c26-73610c53ca37.gif)

## Requirement

- Python 3.7
- Pytorch 1.9.1
- pandas
- re

## Flowchart

![Figure01_Flowchart](https://user-images.githubusercontent.com/48637474/230878573-a26b035d-3ed0-4db9-9b58-161067632daf.jpg)
**Figure 1. The architecture of the proposed deep learning-based simple online and real-time vessel data fusion method.**

![Figure03_Video](https://user-images.githubusercontent.com/48637474/230878762-223472ae-cf19-4167-adbb-80c3f77ae9c3.jpg)
**Figure 2. The flowchart of anti-occlusion tracking method for video-based vessel trajectory extraction.**

## Running
* Save [ckpt.t7](https://drive.google.com/file/d/1QdIP5TEDALJnnpqwjXwvL1J_GoseTK9D/view?usp=share_link) to `DeepSORVF/deep_sort/deep_sort/deep/checkpoint/` folder.
* Save [YOLOX-final.pth](https://drive.google.com/file/d/1mhah7ZzP8oAUuSMR96Or9UvqkXe-AMuS/view?usp=share_link) to `DeepSORVF/detection_yolox/model_data/` folder.
* Set data dir by `parser.add_argument("--data_path", type=str, default = './clip-01/', help='data path')`.
* Run `main.py`.

#### Test Data: [clip-01](https://drive.google.com/file/d/1Bns1jAW1ImL-FeCQBvIUcrO0hjYLIB5K/view?usp=share_link)

## FVessel: Benchmark Dataset for Vessel Detection, Tracking, and Data Fusion

The [FVessel](https://github.com/gy65896/FVessel) benchmark dataset is used to evaluate the reliability of AIS and video data fusion algorithms, which mainly contains 26 videos and the corresponding AIS data captured by the HIKVISION DS-2DC4423IW-D dome camera and Saiyang AIS9000-08 Class-B AIS receiver on the Wuhan Segment of the Yangtze River. To protect privacy, the MMSI for each vessel has been replaced with a random number in our dataset. As shown in Figure 1, these videos were captured under many locations (e.g., bridge region and riverside) and various weather conditions (e.g., sunny, cloudy, and low-light).

![Figure04_FVessel](https://user-images.githubusercontent.com/48637474/210925024-15dcbcbe-717b-47b6-ad4b-377d71141380.jpg)
**Figure 3. Some samples of the FVessel dataset, which contains massive images and videos captured on the bridge region and riverside under sunny, cloudy, and low-light conditions.**

## Performance

|Name|MOFA (%)|IDP (%)|IDR (%)|IDF (%)|video
| :-: | :-: | :-: | :-: | :-: | :-: |
[video-01](https://user-images.githubusercontent.com/48637474/236730149-e098365f-0d6a-4c56-8e18-3e47b7b3a7d6.mp4)|79.94|89.35|90.76|90.05|<img src="https://user-images.githubusercontent.com/48637474/236740815-645fa2f6-abe6-4720-b884-317b270a4d1e.gif" width="200">
[video-02](https://user-images.githubusercontent.com/48637474/236730157-39ce91d9-c8f9-461b-83d5-62d91df67cf9.mp4)|73.19|83.27|91.60|87.23|<img src="https://user-images.githubusercontent.com/48637474/236740837-44b6551b-47ab-48cc-9e1a-b2204ef50093.gif" width="200">
[video-03](https://user-images.githubusercontent.com/48637474/236730165-a6b1ba80-fd36-4149-9164-5006f22ef050.mp4)|96.45|99.23|97.20|98.20|<img src="https://user-images.githubusercontent.com/48637474/236741004-cc85bc64-af9d-4070-b927-6ce2edfd046f.gif" width="200">
[video-04](https://user-images.githubusercontent.com/48637474/236730303-9ce4f74c-23db-442c-b979-5dd6abb31c2a.mp4)|98.08|99.45|98.63|99.03|<img src="https://user-images.githubusercontent.com/48637474/236740221-eb056016-0367-435a-bb9f-4c44a9f211bd.gif" width="200">
[video-05](https://user-images.githubusercontent.com/48637474/236730560-0298cc43-a929-4f25-847b-3aa5d95b6653.mp4)|89.19|93.46|95.91|94.67|<img src="https://user-images.githubusercontent.com/48637474/236740236-c58f1ef5-beb3-49fd-9b80-f91c5602643d.gif" width="200">
[video-06](https://user-images.githubusercontent.com/48637474/236730565-85284244-3229-4dbc-aacb-35e2f58b7efa.mp4)|91.17|96.04|95.08|95.56|<img src="https://user-images.githubusercontent.com/48637474/236740242-c51d45bb-8c31-4d11-af03-8b4024d4dbfd.gif" width="200">
[video-07](https://user-images.githubusercontent.com/48637474/236730567-7568d8fc-f2ce-450a-8ddb-b863e0f9e432.mp4)|96.81|99.59|97.21|98.39|<img src="https://user-images.githubusercontent.com/48637474/236740253-9b36e02c-9fa2-49bc-8274-720112826bf9.gif" width="200">
[video-08](https://user-images.githubusercontent.com/48637474/236730569-27c4d441-e36f-46a4-9716-7d338b16814b.mp4)|82.28|99.64|82.58|90.31|<img src="https://user-images.githubusercontent.com/48637474/236740262-499dee95-342d-4ecb-bf4c-8fdb6a8c5445.gif" width="200">
[video-09](https://user-images.githubusercontent.com/48637474/236730570-c564f227-3d92-432b-8f6e-c592dfaba825.mp4)|98.45|100.00|98.45|99.22|<img src="https://user-images.githubusercontent.com/48637474/236740273-177c41ec-c18d-4aa2-bc77-49708287d3a9.gif" width="200">
[video-10](https://user-images.githubusercontent.com/48637474/236730572-b2873836-7774-4727-87e8-945ab9b5b9f1.mp4)|88.74|90.42|99.26|94.63|<img src="https://user-images.githubusercontent.com/48637474/236740282-f0c9fb79-9a5e-437a-b9e6-27d36625c31e.gif" width="200">
[video-11](https://user-images.githubusercontent.com/48637474/236730574-350e05d5-89fc-4c70-a913-a1795039cc26.mp4)|97.66|99.29|98.36|98.83|<img src="https://user-images.githubusercontent.com/48637474/236740290-6b7144e8-7077-484c-92e2-edf37ae57078.gif" width="200">
[video-12](https://user-images.githubusercontent.com/48637474/236730577-0d68f53a-cb37-413b-afe9-33c3a182aef3.mp4)|95.45|99.06|96.36|97.69|<img src="https://user-images.githubusercontent.com/48637474/236740306-b1febf85-7fcd-4d63-b54c-3227cd1f66c6.gif" width="200">
[video-13](https://user-images.githubusercontent.com/48637474/236730581-0e088d7e-7648-4760-9337-10c338f1307a.mp4)|84.82|94.82|89.72|92.20|<img src="https://user-images.githubusercontent.com/48637474/236740317-628c53c1-55fa-4ec6-945a-0dc22b2f8e99.gif" width="200">
[video-14](https://user-images.githubusercontent.com/48637474/236730584-8fc8cda4-9f20-4954-949a-130e689d62ca.mp4)|93.10|97.82|95.22|96.50|<img src="https://user-images.githubusercontent.com/48637474/236740327-c8c2aaea-8a25-44cd-bf99-b68776acda27.gif" width="200">
[video-15](https://user-images.githubusercontent.com/48637474/236730586-36c8c165-672a-46e2-97e3-77150278c875.mp4)|95.88|97.19|98.74|97.96|<img src="https://user-images.githubusercontent.com/48637474/236740343-6c8d3031-e4eb-4d45-a654-d6aa6e58a22e.gif" width="200">
[video-16](https://user-images.githubusercontent.com/48637474/236730592-6a712fb8-57d3-4ed9-bd3e-d8de8e748042.mp4)|98.68|100.00|98.68|99.33|<img src="https://user-images.githubusercontent.com/48637474/236740349-7c8056c9-7f9c-4307-a8ca-0848a043b5ea.gif" width="200">
[video-17](https://user-images.githubusercontent.com/48637474/236730599-e51c2d9c-e135-4153-9361-ffb8f958b205.mp4)|90.02|93.80|96.39|95.08|<img src="https://user-images.githubusercontent.com/48637474/236740359-9854f866-fbfd-487b-b9c7-1e97ab2242e2.gif" width="200">
[video-18](https://user-images.githubusercontent.com/48637474/236730605-b2d03c4d-6904-44f7-8531-c9e7da3ff957.mp4)|74.49|83.57|92.72|87.91|<img src="https://user-images.githubusercontent.com/48637474/236740372-5b8f170e-491f-45ef-a8c9-eaf0ecbee53e.gif" width="200">
[video-19](https://user-images.githubusercontent.com/48637474/236730610-398c2c5f-1658-493c-b1c7-c87452ac047c.mp4)|96.62|98.31|98.31|98.31|<img src="https://user-images.githubusercontent.com/48637474/236740383-692186e8-9296-4c79-976f-6a74b63b3332.gif" width="200">
[video-20](https://user-images.githubusercontent.com/48637474/236730613-78554780-baab-4c1d-9b8b-f96ad7d804b3.mp4)|96.74|98.66|98.07|98.36|<img src="https://user-images.githubusercontent.com/48637474/236740389-85b29dc3-12d4-4d5d-a33b-d38412ddd9d1.gif" width="200">
[video-21](https://user-images.githubusercontent.com/48637474/236730618-87445b72-dfee-4826-84be-e9f490660591.mp4)|76.43|87.03|89.82|88.40|<img src="https://user-images.githubusercontent.com/48637474/236740399-12925b27-d848-4045-9cc2-89f8e89373c1.gif" width="200">
[video-22](https://user-images.githubusercontent.com/48637474/236730621-fb04106e-8cd6-42b6-9974-4123d4f39b7b.mp4)|96.82|99.35|97.45|98.39|<img src="https://user-images.githubusercontent.com/48637474/236740410-4deced69-5627-4ca2-864e-db135b3976c9.gif" width="200">
[video-23](https://user-images.githubusercontent.com/48637474/236730626-aebe9cce-06b5-423f-b017-e1b3a3171a50.mp4)|94.71|98.91|95.77|97.31|<img src="https://user-images.githubusercontent.com/48637474/236740419-f135c633-af9e-4881-9c15-db798537eaaa.gif" width="200">
[video-24](https://user-images.githubusercontent.com/48637474/236730628-9db4a034-1016-4c32-a8ad-e77dd730dc7c.mp4)|94.70|98.34|96.33|97.32|<img src="https://user-images.githubusercontent.com/48637474/236740427-5befe2b8-2a7a-4a7f-a032-f12805c40f3f.gif" width="200">
[video-25](https://user-images.githubusercontent.com/48637474/236730630-48d8bf1b-0059-4885-afc9-1a2f4dc2b3d5.mp4)|91.49|97.66|93.73|95.66|
[video-26](https://user-images.githubusercontent.com/48637474/236730632-da8b0ff3-9edb-461b-9d3d-6ac22f1b56d3.mp4)|97.44|99.11|98.32|98.72|<img src="https://user-images.githubusercontent.com/48637474/236740438-08446266-35d2-4e5a-a19f-90391fbf1701.gif" width="200">
Average |91.13|95.90|95.41|95.59|...
<img src="(https://user-images.githubusercontent.com/48637474/236740432-a05e7bbc-0390-4653-86b5-690f647f9e45.gif" width="200">

## Acknowledgements

We deeply thank **Jianlong Su** from the School of Computer and Artificial Intelligence in Wuhan University of Technology who performs the data acquisition and algorithm implementation works.

## Citation

```
@article{guo2023asynchronous,
  title={Asynchronous Trajectory Matching-Based Multimodal Maritime Data Fusion for Vessel Traffic Surveillance in Inland Waterways},
  author={Guo, Yu and Liu, Ryan Wen and Qu, Jingxiang and Lu, Yuxu and Zhu, Fenghua, and Lv, Yisheng},
  journal={arXiv preprint arXiv:2302.11283},
  year={2023}
}
```

#### If you have any questions, please get in touch with me (yuguo@whut.edu.cn & wenliu@whut.edu.cn).

## Reference

https://github.com/bubbliiiing/yolox-pytorch

https://github.com/dyh/unbox_yolov5_deepsort_counting/tree/main/deep_sort
