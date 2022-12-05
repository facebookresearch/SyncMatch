# Setup

This repository makes use of several external libraries.
We highly recommend installing them within a virtual environment such as Anaconda.

```bash
conda create --name syncmatch python=3.9 cmake=3.19.6 --yes
conda activate syncmatch

conda install -c conda-forge gcc=11.2.0 gxx=11.2.0 --yes

# PyTorch 1.11.0 TorchVision 0.12.0
conda install pytorch=1.11.0 torchvision cudatoolkit=11.3 -c pytorch --yes

# PyTorch3D -- 0.7
conda install -c fvcore -c iopath -c conda-forge fvcore iopath --yes
conda install -c bottler nvidiacub --yes
conda install pytorch3d -c pytorch3d --yes

# Install a bunch of pip packages
python -m pip install -r requirements.txt 

# The following is not essential to run the code, but good if you want to contribute
# or just keep clean repositories. You should find a .pre-commit-config.yaml file
# already in the repo.
pre-commit install
```
