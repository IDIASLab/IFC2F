# IFC2F

## Introduction

We present IFC2F, an instance-wise dynamic joint feature selection and classification algorithm.

## Citation
To cite our paper, please use the following reference:

Yasitha Warahena Liyanage, Daphney-Stavroula Zois, and Charalampos Chelmis. "Dynamic Instance-Wise Classification in Correlated Feature Spaces." IEEE Transactions on Artifical Intelligence (2021). doi: 10.1109/TAI.2021.3109858.

BibTeX:
``` 
@article{liyanage2021correlated, 
  author={Liyanage, Yasitha Warahena and Zois, Daphney-Stavroula and Chelmis, Charalampos},
  journal={IEEE Transactions on Artificial Intelligence}, 
  title={Dynamic Instance-Wise Classification in Correlated Feature Spaces}, 
  year={2021},
  volume={2},
  number={6},
  pages={537-548},
  doi={10.1109/TAI.2021.3109858}}
```

## Prerequisites

python 2.7 or above and the following libraries
```
numpy
sklearn
scipy
pyitlib
typing
```

## Files

```
IFC2F.py: Include all the necessary functions of IFC2F
datasets: include a sample dataset (i.e., MLL dataset splitted into training and validation sets) 
```

## How to use

```
Step 1. Load dataset:
    Xtrain: Train data 
    Ytrain: Train labels
    Xtest:  Test data
    Ytest: Test labels

Step 2. Define configuration parameters:
    feat_cost: feature evaluation cost
    bins: number of bins conisdered when quantizing the feature space
    beta: number of reachable belief points

Step 3. Initiate an instance of IFC2F using config file and call run function

Step 4. Print classification summary report
```


## Example

```
See example.py
```
