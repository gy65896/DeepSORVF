# DeepSORVF: Deep Learning-based Simple Online and Real-Time Vessel Data Fusion Method

---
### Asynchronous Trajectory Matching-Based Multimodal Maritime Data Fusion for Vessel Traffic Surveillance in Inland Waterways [[Paper](http://arxiv.org/abs/2302.11283)]

![video](https://user-images.githubusercontent.com/48637474/220859261-33458b91-2f2b-4d58-8c26-73610c53ca37.gif)

## Introduction
English | [简体中文](README_zh-CN.md)

In this work, we first extract the AIS- and video-based vessel trajectories, and then propose a deep learning-enabled asynchronous trajectory matching method (named DeepSORVF) to fuse the AIS-based vessel information with the corresponding visual targets. In addition, by combining the AIS- and video-based movement features, we also present a prior knowledgedriven anti-occlusion method to yield accurate and robust vessel tracking results under occlusion conditions. 

![Figure01_Flowchart](https://user-images.githubusercontent.com/48637474/230878573-a26b035d-3ed0-4db9-9b58-161067632daf.jpg)
**Figure 1. The architecture of the proposed deep learning-based simple online and real-time vessel data fusion method.**

![Figure03_Video](https://user-images.githubusercontent.com/48637474/230878762-223472ae-cf19-4167-adbb-80c3f77ae9c3.jpg)
**Figure 2. The flowchart of anti-occlusion tracking method for video-based vessel trajectory extraction.**

## Preparation

- Python 3.7
- Pytorch 1.9.1
- pandas
- re
- Save [ckpt.t7](https://drive.google.com/file/d/1QdIP5TEDALJnnpqwjXwvL1J_GoseTK9D/view?usp=share_link) to `DeepSORVF/deep_sort/deep_sort/deep/checkpoint/` folder 
- Save [YOLOX-final.pth](https://drive.google.com/file/d/1mhah7ZzP8oAUuSMR96Or9UvqkXe-AMuS/view?usp=share_link) to `DeepSORVF/detection_yolox/model_data/` folder

## Running

* Set data dir by `parser.add_argument("--data_path", type=str, default = './Video-01/', help='data path')`.

* Run `main.py`.

## FVessel: Benchmark Dataset for Vessel Detection, Tracking, and Data Fusion

The [FVessel](https://github.com/gy65896/FVessel) benchmark dataset is used to evaluate the reliability of AIS and video data fusion algorithms, which mainly contains 26 videos and the corresponding AIS data captured by the HIKVISION DS-2DC4423IW-D dome camera and Saiyang AIS9000-08 Class-B AIS receiver on the Wuhan Segment of the Yangtze River. To protect privacy, the MMSI for each vessel has been replaced with a random number in our dataset. As shown in Figure 1, these videos were captured under many locations (e.g., bridge region and riverside) and various weather conditions (e.g., sunny, cloudy, and low-light).

![Figure04_FVessel](https://user-images.githubusercontent.com/48637474/210925024-15dcbcbe-717b-47b6-ad4b-377d71141380.jpg)
**Figure 3. Some samples of the FVessel dataset, which contains massive images and videos captured on the bridge region and riverside under sunny, cloudy, and low-light conditions.**

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
