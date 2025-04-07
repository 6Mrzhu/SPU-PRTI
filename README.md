
## Requirements
The code is tested under Pytorch 1.6.0 and Python 3.8 

1. Install python denpendencies.
   ```shell
   pip install -r requirements.txt
   ```
2. Compile pyTorch extensions.
   ```shell
   cd pointnet2_ops_lib
   python setup.py install
    
   cd ../losses
   python setup.py install
   ```
3. Install uniformloss
   ```shell
   pip install --upgrade https://github.com/unlimblue/KNN_CUDA/releases/download/0.2/KNN_CUDA-0.2-py3-none-any.whl
   ```
4. Compile evaluation code
   ```shell
   cd evaluation_code
   cmake .
   make
   ```
## Usage
1. Train the model.
   ```shell
   sh start_train.sh
   ```
2. Test the model.
   ```shell
   sh test.sh
   ```
3. Evaluation the model.
   ```shell
   sh eval.sh
   ```
## Dataset file Organization
You can download PU1K dataset from [Google Drive](https://drive.google.com/drive/folders/1k1AR_oklkupP8Ssw6gOrIve0CmXJaSH3?usp=sharing)
```
dataset
├───PU1K 
│     ├───test
│     │     ├───input_256
│     │     │     ├───input_256
│     │     │     │     ├───xxx.xyz
│     │     │     │     ├───xxx.xyz
│     │     │     │     ...
│     │     │     ├───gt_1024
│     │     │     │     ├───xxx.xyz
│     │     │     │     ├───xxx.xyz
│     │     │     │     ...
│     │     ├───input_512
│     │     ...
│     ├───train
│     │     └───pu1k_poisson_256_poisson_1024_pc_2500_patch50_addpugan.h5
├───PUGAN
│     ├───test
│     │     ├───input_256
│     │     │     ├───input_256
│     │     │     │     ├───xxx.xyz
│     │     │     │     ├───xxx.xyz
│     │     │     │     ...
│     │     │     ├───gt_1024
│     │     │     │     ├───xxx.xyz
│     │     │     │     ├───xxx.xyz
│     │     │     │     ...
│     │     ├───input_512
│     │     ...
│     ├───train
│     │     └───PUGAN_poisson_256_poisson_1024.h5
└───real_scan
│     ├───xyzToPatch.py	
│     ├───make_h5.py	
│     ├───KITTI
│     └───ScanNet
│     ...
```

## Codes
To be completed soon.

## Acknowledgment
Our code is built upon the following repositories: [PUCRN](https://github.com/hikvision-research/3DVision/tree/main/PointUpsampling/PUCRN) and [PUGCN](https://github.com/guochengqian/PU-GCN). Thanks for their great work.

