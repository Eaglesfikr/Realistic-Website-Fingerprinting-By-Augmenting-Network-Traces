# Realistic Website Fingerprinting By Augmenting Network Traces (ACM CCS '23)

ACM CCS '23 version: [https://doi.org/10.1145/3576915.3616639](https://doi.org/10.1145/3576915.3616639)

Extended version of the paper: [https://arxiv.org/pdf/2309.10147.pdf](https://arxiv.org/pdf/2309.10147.pdf)

We make our code and artifacts available in [artifacts](./artifacts/).

## struction
```
.
.
├── LICENSE
├── README.md
└── artifacts
    ├── README.md
    ├── datasets
    │   ├── AWF
    │   │   ├── AWF-PT-sup.npz
    │   │   ├── awf1.npz
    │   │   ├── awf1_aug2x.npz
    │   │   └── awf2.npz
    │   ├── README.md
    │   └── drift
    │       ├── Drift5000.npz
    │       └── Drift90.npz
    ├── requirements.txt
    └── src
        ├── NetCLR
        │   ├── MYfine-tuning-cw.ipynb 
# 我自己的微调notebook，这里将增强与训练隔开，增强轨迹用npz重新保存，以帮助提升训练速度（CPU与GPU数据copy）
        │   ├── NetCLR-Pretrain.py # 我一开始的预训练，不隔开
        │   ├── NetCLR2-Pretrain.py # 预训练，使用增强后的npz文件
        │   ├── README.md
        │   ├── fine-tuning-cw.ipynb
        │   ├── fine-tuning-ow.ipynb
        │   └── pre-training.ipynb
        ├── NetFM
        │   ├── NetFM.ipynb
        │   └── README.md
        └── models
            └── NetCLR
                ├── NetCLR_epoch_0.pth.tar
                ├── NetCLR_epoch_100.pth.tar
                ├── NetCLR_epoch_120.pth.tar
                ├── NetCLR_epoch_140.pth.tar
                ├── NetCLR_epoch_160.pth.tar
                ├── NetCLR_epoch_180.pth.tar
                ├── NetCLR_epoch_20.pth.tar
                ├── NetCLR_epoch_200.pth.tar
                ├── NetCLR_epoch_220.pth.tar
                ├── NetCLR_epoch_240.pth.tar
                ├── NetCLR_epoch_260.pth.tar
                ├── NetCLR_epoch_280.pth.tar
                ├── NetCLR_epoch_300.pth.tar
                ├── NetCLR_epoch_320.pth.tar
                ├── NetCLR_epoch_340.pth.tar
                ├── NetCLR_epoch_360.pth.tar
                ├── NetCLR_epoch_380.pth.tar
                ├── NetCLR_epoch_40.pth.tar
                ├── NetCLR_epoch_400.pth.tar
                ├── NetCLR_epoch_60.pth.tar
                └── NetCLR_epoch_80.pth.tar
```


## References

To cite this paper and artifacts, please use the following:

Alireza Bahramali, Ardavan Bozorgi, and Amir Houmansadr. 2023. Realistic
Website Fingerprinting By Augmenting Network Traces. In Proceedings of
the 2023 ACM SIGSAC Conference on Computer and Communications Security
(CCS ’23), November 26–30, 2023, Copenhagen, Denmark. ACM, New York,
NY, USA, 15 pages. https://doi.org/10.1145/3576915.3616639

```
@inproceedings{3576915.3616639,
author = {Bahramali, Alireza and Bozorgi, Ardavan, and Houmansadr, Amir},
title = {Realistic Website Fingerprinting By Augmenting Network Traces},
booktitle = {Proceedings of
the 2023 ACM SIGSAC Conference on Computer and Communications Security},
series = {CCS '23},
year = {2023},
location = {Copenhagen, Denmark},
numpages = {15},
url = {https://doi.org/10.1145/3576915.3616639},
doi = {10.1145/3576915.3616639},
publisher = {ACM},
address = {New York, NY, USA},
}
```
