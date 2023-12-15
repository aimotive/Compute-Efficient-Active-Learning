# Compute-Efficient Active Learning - Official PyTorch Implementation

## Prequisites:
Tested using Python 3.8
Use requirements.txt to install necessary dependencies.

## Running the Code
Example on MNIST:
```
python3 src/main_train.py --dataset mnist --strategy entropy --subsample-size 5000 --subsample-unlabeled
```

Example on CIFAR10:
```
python3 src/main_train.py --dataset cifar10 --strategy entropy --subsample-size 10000 --subsample-unlabeled
```

## Citation
If you use our code in your research, or find [our work](https://nips.cc/virtual/2023/78767) helpful, please consider citing us with the bibtex below:
```
@inproceedings{
n{\'e}meth2023computeefficient,
title={Compute-Efficient Active Learning},
author={G{\'a}bor N{\'e}meth and Tamas Matuszka},
booktitle={NeurIPS 2023 Workshop on Adaptive Experimental Design and Active Learning in the Real World},
year={2023},
url={https://openreview.net/forum?id=G6ujG6LaKV}
}
```
More details about our work can be found on the [paper](https://nips.cc/media/neurips-2023/Slides/78767_7FrbVd0.pdf) and [poster](https://nips.cc/media/PosterPDFs/NeurIPS%202023/78767.png?t=1701339795.426551).


## License

This project is licensed under the Attribution-NonCommercial-ShareAlike 4.0 International license.

Please see the LICENSE.txt file for more details about the license terms and conditions.
