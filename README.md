# Code for TGRS paper "Full-Level Domain Adaptation for Building Extraction in Very-High-Resolution Optical Remote-Sensing Images"
This is a reporitory for releasing a PyTorch implementation of our work [Full-Level Domain Adaptation for Building Extraction in Very-High-Resolution Optical Remote-Sensing Images](https://ieeexplore.ieee.org/document/9481881)

## Introduction
Convolutional neural networks (CNNs) have achieved tremendous success in computer vision tasks, such as building extraction. However, due to domain shift, the perfor- mance of the CNNs drops sharply on unseen data from another domain, leading to poor generalization. As it is costly and time-consuming to acquire dense annotations for remote-sensing (RS) images, developing algorithms that can transfer knowledge from a labeled source domain to an unlabeled target domain is of great significance. To this end, we propose a novel full-level domain adaptation network (FDANet) for building extraction by combining image-, feature-, and output-level information effec- tively. At the input level, a simple Wallis filter method is employed to transfer source images into target-like ones whereby alleviating radiometric discrepancy and achieving image-level alignment. To further reduce domain shift, adversarial learning is used to enforce feature distribution consistency constraints between the source and target images. In this way, feature-level alignment can be embedded effectively. At the output level, a mean-teacher model is introduced to enforce transformation-consistent con- straint for the target output so that the regularization effect is enhanced and the uncertain predictions can be suppressed as much as possible. To further improve the performance, a novel self-training strategy is also employed by using pseudo labels. The effectiveness of the proposed FDANet is verified on three diverse high-resolution aerial datasets with different resolutions and scenarios. Extensive experimental results and ablation studies demonstrated the superiority of the proposed method.
## Flowchart
![image](https://user-images.githubusercontent.com/20106991/126872583-ecc4d2fe-f2c6-4fc0-bf22-4a57cb025b57.png)

## Result
![image](https://user-images.githubusercontent.com/20106991/126867293-c4c4de5a-87a5-4907-af9f-dfc8631950a8.png)
![image](https://user-images.githubusercontent.com/20106991/126867267-e516d6bb-0b12-473d-9b62-96753c8ad583.png)
![image](https://user-images.githubusercontent.com/20106991/126867284-1ae8baed-c275-473e-b8a6-60d71bd68f5f.png)

## Requirements

- Python 3.8
- Pytorch >=1.0.0




## Citation
Please cite our paper if you find it is useful for your research.
```
D. Peng, H. Guan, Y. Zang and L. Bruzzone, "Full-Level Domain Adaptation for Building Extraction in Very-High-Resolution Optical Remote-Sensing Images," in IEEE Transactions on Geoscience and Remote Sensing, doi: 10.1109/TGRS.2021.3093004.
```
