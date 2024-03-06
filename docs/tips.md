## Training/test Tips
#### Training/test options
Please see `options/train_options.py` and `options/base_options.py` for the training flags; see `options/test_options.py` and `options/base_options.py` for the test flags. There are some model-specific flags as well, which are added in the model files, such as `--lambda_A` option in `model/cycle_gan_model.py`. The default values of these options are also adjusted in the model files.
#### CPU/GPU (default `--gpu_ids 0`)
Please set`--gpu_ids -1` to use CPU mode; set `--gpu_ids 0,1,2` for multi-GPU mode. You need a large batch size (e.g., `--batch_size 32`) to benefit from multiple GPUs.

#### Visualization
During training, the current results can be viewed using wandb. To do this, create a wandb account and login using the information provided [here](https://docs.wandb.ai/quickstart). Then use the flag --use_wandb, when training.

#### Preprocessing
Generally, preprocessing is set to none, as the patches are extracted at a specified shape from the provided datasets, and automatically gently augmented in this setting.

#### Fine-tuning/resume training
To fine-tune a pre-trained model, or resume the previous training, use the `--continue_train` flag. The program will then load the model based on `epoch`. By default, the program will initialize the epoch count as 1. Set `--epoch_count <int>` to specify a different starting epoch count.

#### Prepare your own datasets for CycleGAN
You need to create two directories to host images from domain A `/path/to/data/trainA` and from domain B `/path/to/data/trainB`. Then you can train the model with the dataset flag `--dataroot /path/to/data`. Optionally, you can create hold-out test datasets at `/path/to/data/testA` and `/path/to/data/testB` to test your model on unseen images.

#### About batch size
For all experiments in the original paper, the batch size was set to 1. If there is room for memory, you can use higher batch size with batch norm or instance norm. (Note that the default batchnorm does not work well with multi-GPU training. You may consider using [synchronized batchnorm](https://github.com/vacancy/Synchronized-BatchNorm-PyTorch) instead). But please be aware that it can impact the training. In particular, even with Instance Normalization, different batch sizes can lead to different results.