
# Enhanced Wheat Edge Detection via Dual-Channel Global-Local Feature Collaboration in GloCrop-UNet

## Abstract
Visual navigation, grounded in RGB imagery, offers a cost-effective and efficient solution for autonomous navigation in combine harvesters. Nevertheless, the precision of existing visual methods is compromised by factors such as stubble interference, variations in soil texture, and lighting fluctuations. Addressing these challenges, this study introduces GloCrop-UNet, a network designed for wheat harvesting scenes that enhances global semantic and local detailed features collaboratively through a dual-path architecture. The primary path employs the bidirectional feature shift-deep mechanism (BFSM) to establish longitudinal and transverse long-distance dependencies alongside cross-channel feature interactions, while the auxiliary path uses the hierarchical dynamic regional attention mechanism (HDRAM) to emphasize critical edge textures and reduce computational complexity. Furthermore, the dual-channel shared encoder facilitates the efficient reuse of low-level features. Experimental results demonstrate that GloCrop-UNet achieves a segmentation accuracy of 82.10\% and a Dice coefficient of 90.08\% in wheat edge detection tasks, surpassing the standard Rolling-UNet by 3.38\% in accuracy and 2.03\% in Dice coefficient, with a reduced parameter count of 2.0073 million. This model is particularly suited for deployment in embedded hardware within agricultural machinery, offering a promising advancement in autonomous agricultural navigation.


## Implementation
- The experimental environment of this paper includes:
  
Operating system & Windows 11

CPU & AMD Ryzen 7 7735H with Radeon Graphics 3.20 GHz 

GPU & NVIDIA GeForce RTX 4060 Laptop GPU 

Python & 3.10 

Pytorch & 2.2.2 

Cuda & 12.4

- Clone this repository:
```bash
https://github.com/QF-Forever/GloCrop-UNet
cd GloCrop-UNet
```

### Data Format
- Make sure to put the files as the following structure. For binary segmentation, just use folder 0.
```
inputs
└── <dataset name>
  ├── images
    │   ├── 00000000.png
    │   ├── 00000001.png
    │   ├── 00000002.png
    │   ├── ...
    |
 └── masks
    └── 0
       │   ├── 00000000.png
       │   ├── 00000001.png
       │   ├── 00000002.png
       │   ├── ...
```


### Training and Validation
- Train the model.
```
python train.py
```
- Evaluate.
```
python val.py
```
### Code and Dataset
The code is stored in the master branch.
Access the code and dataset at https://github.com/QF-Forever/GloCrop-UNet/tree/master.
