# Evaluation 

We provide checkpoints for our model and ablations as well as code to run several of our baselines. 
Please follow the instructions below to evaluate the appropriate model. 
We note that due to the stochasticity of some of the steps (eg, WP-RANSAC), there might be some variance in results.

We provide the checkpoint for our full model in `checkpoints/full_model.ckpt`. 
We also provide checkpoints for ablations and other model variants. 
You can evaluate the pretrained full model checkpoint with the comment below to run with the SuperGlue pairwise dataset. 
```
python test.py test.checkpoint.path=checkpoints/full_model.ckpt dataset=scannet_testpairs
``` 

For pairwise registration results (Table 2 in paper), we omit refinement to avoid conflating the comparison. 
Hence, it is important to add a flag to adjust this. For example, to run on the narrow baseline dataset:
```
python test.py test.checkpoint.path=checkpoints/full_model.ckpt dataset=scannet_test test.model_cfg.refinement.num_steps=1
```

For multiview evaluations, make sure to set the flag to run with 6 views `dataset.num_views=6`. 
All pairwise evaluations should run fairly quickly (<10mins), while multiview evaluations might take a longer time. 

We note that you can generate visualizations for the test set by including the flag
`test.visualize_test=True`. This will generate a directory with an html file that can be used to
browse the results in the directory outlined in `syncmatch/configs/config.yaml`. 


## Baselines 

We are unable to include some of the code from prior work due to licenses, but we provide provide wrapper classes to adapt prior work to our expected output. Please follow the instructions below to download the necessary code and checkpoint weights to run SuperGlue and LoFTR. 

```
# === LoFTR ===
# Download the pretrained weights 
python -m pip install gdown

# download weights
cd syncmatch/models/pretrained_weights
gdown 19s3QvcCWQ6g-N1PrYlDCg-2mOJZ3kkgS
mv indoor_ds_new.ckpt loftr_ds.ckpt

# clone LoFTR repo and add symbolic link
git clone https://github.com/zju3dv/LoFTR.git
ln -s LoFTR/src/loftr syncmatch/syncmatch/models/loftr

# === SuperGlue and SuperPoint ===
git clone git clone git@github.com:magicleap/SuperGluePretrainedNetwork.git
ln -s SuperGluePretrainedNetwork/models syncmatch/syncmatch/models/superglue
```


To run the evaluations, you simple need to set the appropriate model name: 

```
python test.py model.name=LoFTR 
python test.py model.name=SuperGlue 
```

For baselines that use specific feature descriptors, we provide a generic aligner class. To evaluate those baselines, you can use the `generic_aligner` model config while setting choice of feature and aligner accordingly. For example, to run RootSIFT + GART: 
```
python test.py model=generic_aligner baseline.feature=rootsift 
```
