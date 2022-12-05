# Datasets 

## ScanNet Dataset

ScanNet is a large dataset of indoor scenes: over 1500 scenes and 2.5 million views. 
The dataset is organized as a series of RGB-D sequences that are stored as a sensor-stream (or
.sens) file. Below are the instructions to download, extract, and process the dataset as well as
generate the data dictionaries that are used by the data loaders to index into the dataset. 

You first need to download the dataset. This can be done by following the instructions outlined in
the official [ScanNet repository](https://github.com/ScanNet/ScanNet). 
We downloaded the individual scenes and used the official splits found
[here](https://github.com/ScanNet/ScanNet/tree/master/Tasks/Benchmark). 
We use the v2 splits which are found in `syncmatch/data`. 

Once all the scenes are downloaded and extracted, we processed the `.sens` files used the provided
[SensReader](https://github.com/ScanNet/ScanNet/tree/master/SensReader/python) to extract the color,
depth, intrinsics, and pose matrices. We note that the pose matrices are only used for evaluation. 
Once extraction is completed, the root repo should follow the following directory structure:

```
<ScanNet Root>
    |- train
    |- valid
    |- test
        |- scans
            |- scene0024_01 
            |- scene0024_02 
            |- scene0024_03 
                |- color 
                    |- 0.jpg
                    |- 1.jpg
                    |- 2.jpg
                |- depth 
                    |- 0.png
                    |- 1.png
                    |- 2.png
                |- pose
                    |- 0.txt
                    |- 1.txt
                    |- 2.txt
                |- intrinsic
                    |- intrinsic_color.txt
                    |- intrinsic_depth.txt
                    |- extrinsic_color.txt
                    |- extrinsic_depth.txt
```

If this structure is followed, you can simply run the script to generate the data dictionary 
for each split as follows. This will generate the three dictionaries needed to use the dataset.

```
cd syncmatch/data 
python create_scannet_dict.py <ScanNet Root>
```

At the end, you need to also update the path for the dataset root in the dataset set configs in `syncmatch/configs/dataset` for `scannet.yaml`, `scannet_test.yaml`, and `scannet_mini.yaml`. 

## Pairwise Benchmark 

We also use the pairwise benchmark proposed by SuperGlue. 
First, we download the test set provided by [LoFTR](https://github.com/zju3dv/LoFTR/). 

```
# Install the gdown library to download from google drive
python -m pip install gdown

# Move to data directory 
cd <DATA_ROOT>

# download the tar file provided by LoFTR
gdown --id 1wtl-mNicxGlXZ-UQJxFnKuWPvvssQBwd
tar -xvf scannet_test_1500.tar

# download other files
cd scannet_test_1500
wget https://raw.githubusercontent.com/zju3dv/LoFTR/master/assets/scannet_test_1500/intrinsics.npz
wget https://raw.githubusercontent.com/zju3dv/LoFTR/master/assets/scannet_test_1500/test.npz
```

Once downloaded and extracted, update the root variable in `syncmatch/configs/dataset/scannet_testpairs.yaml`. 


## ETH3D dataset

ETH3D is a dataset of RGB-D videos that is often used to benchmark SLAM systems. We use it to evaluate our multiview registration algorithms. 
We use the [download script](https://www.eth3d.net/data/slam/download_eth3d_slam_datasets.py) provided by ETH3D to download and extract the dataset. 
Once this is done, please update the root variable in `syncmatch/configs/dataest/eth_video.yaml`.
