# Compute-Efficient Active Learning - Official Pytorch Implementation

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
If you use our code in your research, or find our work helpful, please consider citing us with the bibtex below:
```
```


## License

This project is licensed under the Attribution-NonCommercial-ShareAlike 4.0 International license.

Please see the LICENSE.txt file for more details about the license terms and conditions.
