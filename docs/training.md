# Training

We use [PyTorch Lightning](https://www.pytorchlightning.ai) to handle the experiments and 
[hydra](https://hydra.cc/) as our configuration manager. 
The main Lightning class can be found in `syncmatch/nnutils/trainer.py` and the configurations 
can be found in `syncmatch/configs`.
To train a model, simply run the following command with the appropriate experimental config.
We set our full model as our defaults, but examples of other experiments can be seen in `syncmatch/configs/experiment`.

To train our primary model, run the following command:
```
python train.py +experiment=full_model
```
