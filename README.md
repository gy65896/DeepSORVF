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
[video-01](https://user-images.githubusercontent.com/48637474/236730149-e098365f-0d6a-4c56-8e18-3e47b7b3a7d6.mp4)|79.94|89.35|90.76|90.05|video-01
[video-02](https://user-images.githubusercontent.com/48637474/236730157-39ce91d9-c8f9-461b-83d5-62d91df67cf9.mp4)|73.19|83.27|91.60|87.23|video-01
[video-03](https://user-images.githubusercontent.com/48637474/236730165-a6b1ba80-fd36-4149-9164-5006f22ef050.mp4)|96.45|99.23|97.20|98.20|video-01
[video-04](https://user-images.githubusercontent.com/48637474/236730303-9ce4f74c-23db-442c-b979-5dd6abb31c2a.mp4)|98.08|99.45|98.63|99.03|video-01
video-05|89.19|93.46|95.91|94.67|[video-01]()
video-06|91.17|96.04|95.08|95.56|[video-01]()
video-07|96.81|99.59|97.21|98.39|[video-01]()
video-08|82.28|99.64|82.58|90.31|[video-01]()
video-09|98.45|100.00|98.45|99.22|[video-01]()
video-10|88.74|90.42|99.26|94.63|[video-01]()
video-11|97.66|99.29|98.36|98.83|[video-01]()
video-12|95.45|99.06|96.36|97.69|[video-01]()
video-13|84.82|94.82|89.72|92.20|[video-01]()
video-14|93.10|97.82|95.22|96.50|[video-01]()
video-15|95.88|97.19|98.74|97.96|[video-01]()
video-16|98.68|100.00|98.68|99.33|[video-01]()
video-17|90.02|93.80|96.39|95.08|[video-01]()
video-18|74.49|83.57|92.72|87.91|[video-01]()
video-19|96.62|98.31|98.31|98.31|[video-01]()
video-20|96.74|98.66|98.07|98.36|[video-01]()
video-21|76.43|87.03|89.82|88.40|[video-01]()
video-22|96.82|99.35|97.45|98.39|[video-01]()
video-23|94.71|98.91|95.77|97.31|[video-01]()
video-24|94.70|98.34|96.33|97.32|[video-01]()
video-25|91.49|97.66|93.73|95.66|[video-01]()
video-26|97.44|99.11|98.32|98.72|[video-01]()
Average |91.13|95.90|95.41|95.59|[video-01]()







https://user-images.githubusercontent.com/48637474/236730519-9b6c49be-f480-4589-89c6-383fa5321eae.mp4



Uploading video_17.mp4…


https://user-images.githubusercontent.com/48637474/236730522-27ddf5bb-754e-4269-9627-2b0327a531cf.mp4



https://user-images.githubusercontent.com/48637474/236730523-e68e7653-defd-4623-8681-508b7d7e6381.mp4





https://user-images.githubusercontent.com/48637474/236730499-5c3789b1-2cee-42d8-9c42-a11fed2c09ef.mp4



https://user-images.githubusercontent.com/48637474/236730503-a8ae5ca9-45b2-4b0b-a43c-0948b5ebaaff.mp4



https://user-images.githubusercontent.com/48637474/236730505-4a0f2133-cb49-48fc-af36-09035ba5b391.mp4



https://user-images.githubusercontent.com/48637474/236730506-65450491-385b-4293-bdae-88cfa323a8cb.mp4



https://user-images.githubusercontent.com/48637474/236730508-9363bdcd-a42c-4b5f-a3bc-4be24657cbd4.mp4



https://user-images.githubusercontent.com/48637474/236730511-1d62942b-9e2e-4280-a5e3-13b463d05a75.mp4



https://user-images.githubusercontent.com/48637474/236730512-7dd298f3-28c5-42fa-841f-65545398ca91.mp4



https://user-images.githubusercontent.com/48637474/236730517-8eb5e838-0eed-49e9-8210-3f6de9f97cad.mp4





  |Number|MMSI|Lon|Lat|Speed|Course|Heading|Type|Timestamp|
  | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: |
  0|100000000|114.325327|30.60166 |0  |293.6|511|18|[video](https://user-images.githubusercontent.com/48637474/220344086-5684a8e8-cb73-4786-a8dc-bdc9f68b5a35.mp4)
  1|130000000|114.302683|30.58059 |6.8|33.6 |33 |18|1652181659157
  2|140000000|114.31004 |30.599997|3.9|215.6|511|18|1652181655147
  3|600000000|114.3156  |30.59773 |7.2|39.6 |511|18|1652181649704
  ...|...|...|... |...|... |...|...|...



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
