<br><br><br>

# A minimal CycleGAN for image-to-image translation of volumetric data

This code is a reworking of the excellent [CycleGAN-pytorch](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix) repository created by [Jun-Yan Zhu](https://github.com/junyanz) and [Taesung Park](https://github.com/taesungp), and supported by [Tongzhou Wang](https://github.com/SsnL).
Therefore, if you use this code for your research, please also cite:

Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks.<br>
[Jun-Yan Zhu](https://www.cs.cmu.edu/~junyanz/)\*,  [Taesung Park](https://taesung.me/)\*, [Phillip Isola](https://people.eecs.berkeley.edu/~isola/), [Alexei A. Efros](https://people.eecs.berkeley.edu/~efros). In ICCV 2017. (* equal contributions) [[Bibtex]](https://junyanz.github.io/CycleGAN/CycleGAN.txt)

Image-to-Image Translation with Conditional Adversarial Networks.<br>
[Phillip Isola](https://people.eecs.berkeley.edu/~isola), [Jun-Yan Zhu](https://www.cs.cmu.edu/~junyanz/), [Tinghui Zhou](https://people.eecs.berkeley.edu/~tinghuiz), [Alexei A. Efros](https://people.eecs.berkeley.edu/~efros). In CVPR 2017. [[Bibtex]](https://www.cs.cmu.edu/~junyanz/projects/pix2pix/pix2pix.bib)

## Overview of changes and additional features

This repository is focussed on facilitating 3D image-to-image translation, by exploiting as much as possible the existing code-base.
Particularly, this work is aimed to make style transfer between 3D image volume domains more accessible.
Thus, the original code was significantly cut back, and new options were added which allow 3D data to be supported in cycleGAN.
Below are summarised the key changes and additions.

### Data
This repo is specifically designed for volumetric data, such as volume EM data. Instead of small image patches or individual images,
datasets of 3D volumes can be directly loaded during training and for inference.

### Stitching
During inference, the datasets are directly stitched. 
In addition to standard stitching techniques often used for similar encoder-decoder pipelines on large datasets,
we implement the tile-and-stitch method described in our group (https://github.com/Kainmueller-Lab/shift_equivariance_unet). This allows data inferred by 
a model using a unet backbone to stitch large datasets without any stitching artefacts.

### Evaluation
Once the models are trained, they can be tested using a pipeline which calculates the Frechet Inception Distance (FID) between the
predicted datasets and the source domains for every model checkpoint saved during training. This facilitates the choice of the best
checkpoint for any downstream tasks, beyond qualitative assessments by a human.

### Training and Inference procedures
Instead of focussing entirely on 3D training and inference which can be computationally heavy, there are several options for training and inference.
For training, users can choose between training a model with 3D patches and with 3D convolutions, exploiting the potential gain in context in each patch,
or with 2D patches extracted from each slice of the image stacks and 2D convolutions, and exploiting the larger batch size for training.

For inference, users can either predict models given the above 2d or 3d options or use orthopatch (2.5d) prediction which may provide the benefit of
assembling the predicted dataset from an ensemble of predictions.

### Small additions

- added swinunetr as an additional backbone for the generator models
- added additional (optional) structural similarity (SSIM) loss term to the cycle-consistency loss, for added stability during training, as suggested here [].
- added additional learning rate schedulers
- added a sampler which filters out patches with background, using a simple measure of pixel standard deviation per patch, to maximize the number of patches with informative content.
- reduced visualisation module to wandb visualisation

### Disclaimers
The repo is still a work in progress. Specifically, it was designed for EM data which usually has only a single channel. The repo has not been thoroughly
tested for images with multiple channels, such as RGB. Hopefully, by streamlining the code, adapting the repo for other data will make additional features easy to add for other users.
The repo has been tested on .tif images, and currently other data types are not supported yet. Convert your image volumes to .tif files to use this repo. 
Finally, the images used to test this repo were made isotropic in all dimensions and the architectures of the model backbones are adapted for isotropic data. 
To use on non-isotropic data, for now you can either use the 2d training and inference or you will need to change the architecture of the networks used in the generator models and the patch sizes used to tile the datasets.

Please post an issue of something is not working, so we can improve the repo as we go.

## Colab Notebook
TensorFlow Core CycleGAN Tutorial: [Google Colab](https://colab.research.google.com/github/tensorflow/docs/blob/master/site/en/tutorials/generative/cyclegan.ipynb) | [Code](https://github.com/tensorflow/docs/blob/master/site/en/tutorials/generative/cyclegan.ipynb)

TensorFlow Core pix2pix Tutorial: [Google Colab](https://colab.research.google.com/github/tensorflow/docs/blob/master/site/en/tutorials/generative/pix2pix.ipynb) | [Code](https://github.com/tensorflow/docs/blob/master/site/en/tutorials/generative/pix2pix.ipynb)

PyTorch Colab notebook: [CycleGAN](https://colab.research.google.com/github/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/CycleGAN.ipynb) and [pix2pix](https://colab.research.google.com/github/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/pix2pix.ipynb)

ZeroCostDL4Mic Colab notebook: [CycleGAN](https://colab.research.google.com/github/HenriquesLab/ZeroCostDL4Mic/blob/master/Colab_notebooks_Beta/CycleGAN_ZeroCostDL4Mic.ipynb) and [pix2pix](https://colab.research.google.com/github/HenriquesLab/ZeroCostDL4Mic/blob/master/Colab_notebooks_Beta/pix2pix_ZeroCostDL4Mic.ipynb)


## Prerequisites
- Linux or macOS
- Python 3
- CPU or NVIDIA GPU + CUDA CuDNN

## Getting Started
### Installation

- Clone this repo:
```bash
git clone https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix
cd pytorch-CycleGAN-and-pix2pix
```

- Install [PyTorch](http://pytorch.org) and 0.4+ and other dependencies (e.g., torchvision, [visdom](https://github.com/facebookresearch/visdom) and [dominate](https://github.com/Knio/dominate)).
  - For pip users, please type the command `pip install -r requirements.txt`.
  - For Conda users, you can create a new Conda environment using `conda env create -f environment.yml`.
  - For Docker users, we provide the pre-built Docker image and Dockerfile. Please refer to our [Docker](docs/docker.md) page.
  - For Repl users, please click [![Run on Repl.it](https://repl.it/badge/github/junyanz/pytorch-CycleGAN-and-pix2pix)](https://repl.it/github/junyanz/pytorch-CycleGAN-and-pix2pix).

### CycleGAN train/test

- To log training progress and test images to W&B dashboard, set the `--use_wandb` flag with train and test script
- Train a model:
```bash
#!./scripts/train_cyclegan.sh
python train.py --dataroot path/to/datasets/ --checkpoints_dir path/to/checkpoints_dir --name my_cyclegan_model --train_mode 2d --netG unet_32 --patch_size 190 --stride_A 190 --stride_B 190
```

- Test the model:
You can use any of `test_2D.py`, `test_2D_resnet.py`, `test_3D.py`, `test_3D_resnet.py`, `test_2_5D.py`, `test_2_5D_resnet.py`.
- For instance, to translate from domain A to domain B with a 2D CycleGAN with a Unet Generator:
```bash
python test_2D.py --dataroot path/to/test_dataset/ --name path/to/my_2d_cyclegan_model --patch_size 190 --epoch latest --test_mode 2d --model_suffix _A
```


## [Docker](docs/docker.md)
We provide the pre-built Docker image and Dockerfile that can run this code repo. See [docker](docs/docker.md).

## [Training/Test Tips](docs/tips.md)
Best practice for training and testing your models.

## [Frequently Asked Questions](docs/qa.md)
Before you post a new question, please first look at the above Q & A and existing GitHub issues.

## Custom Model and Dataset
If you plan to implement custom models and dataset for your new applications, we provide a dataset [template](data/template_dataset.py) and a model [template](models/template_model.py) as a starting point.

## [Code structure](docs/overview.md)
To help users better understand and use our code, we briefly overview the functionality and implementation of each package and each module.

## Pull Request
You are always welcome to contribute to this repository by sending a [pull request](https://help.github.com/articles/about-pull-requests/).
Please run `flake8 --ignore E501 .` and `python ./scripts/test_before_push.py` before you commit the code. Please also update the code structure [overview](docs/overview.md) accordingly if you add or remove files.

## Citation
If you use this code for your research, please cite our papers.
```
@inproceedings{CycleGAN2017,
  title={Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks},
  author={Zhu, Jun-Yan and Park, Taesung and Isola, Phillip and Efros, Alexei A},
  booktitle={Computer Vision (ICCV), 2017 IEEE International Conference on},
  year={2017}
}


@inproceedings{isola2017image,
  title={Image-to-Image Translation with Conditional Adversarial Networks},
  author={Isola, Phillip and Zhu, Jun-Yan and Zhou, Tinghui and Efros, Alexei A},
  booktitle={Computer Vision and Pattern Recognition (CVPR), 2017 IEEE Conference on},
  year={2017}
}
```
