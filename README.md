# DIQM - Deep Image Quality Metric
Approximating existing visual metrics efficiently using Deep Learning.

This repo contains the code to reproduce the results presented in the following paper:

> Artusi, Alessandro, Francesco Banterle, Fabio Carrara, and Alejandro Moreo. "Efficient Evaluation of Image Quality via Deep-Learning Approximation of Perceptual Metrics." IEEE Transactions on Image Processing (2019).

## Getting Started
DIQM requires:
 - Python 2
 - tensorflow 1.2.1
 - keras 2.0.5
 
We strongly suggest using [Docker](https://docs.docker.com/) and [nvidia-docker](https://github.com/NVIDIA/nvidia-docker) to setup the environment and install prerequisites:

```sh
docker build -t fabiocarrara/diqm:gpu .
```

Bring up the environment by issuing the following command in the repo directory:

```sh
docker run --runtime nvidia --rm -it -v $PWD:/workdir fabiocarrara/diqm:gpu
```

## Data and trained models
Send us a mail and we will be happy to share data and trained models.

## Reproduce experiments

The `reproduce.sh` file contains all the commands for training the models presented in the paper.
You can reproduce models, predictions, and plots of the paper by issuing:

```sh
reproduce.sh
python plot.py runs/
```
Check `plot_p.py` to produce additional paper plots.

## Make predictions
Check `predict.py` to make predictions using trained models.




