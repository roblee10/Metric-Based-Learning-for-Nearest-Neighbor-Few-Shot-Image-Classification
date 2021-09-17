# Metric-Based Learning for Nearest-Neighbor Few-Shot Image Classification

This repository contains the code for reproducibility of the following paper

[Metric-Based Learning for Nearest-Neighbor Few-Shot Image Classification](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9333850)

by Min Jun Lee, Jungmin So

</br>

## Abstract

Few-shot learning task, which aims to recognize a new class with insufficient data, is an inevitable issue to be solved in image classification. Among recent work, Metalearning is commonly used to figure out few shot learning task. Here we tackle a recent method that uses the nearest-neighbor algorithm when recognizing few-shot images and to this end, propose a metric-based approach for nearest-neighbor few shot classification. We train a convolutional neural network with miniImageNet applying three types of loss, triplet loss, cross-entropy loss, and combination of triplet loss and cross-entropy loss. In evaluation, three configurations exist according to feature transformation technique which are unnormalized features, L2-normalized features, and centered L2-normalized features. For 1-shot 5-way task, the triplet loss model attains the uppermost accuracy among all three configurations and for 5-shot 5-way task, the identical model reaches the foremost accuracy in unnormalized features configuration.

</br>

## Dataset Download

The downloaded dataset should be placed in the path specified below.

 #### - MiniImageNet

- Download dataset here : 

- Place the dataset in this path

  ```
  ./images/
  ```

</br>

## Pre-train

Pre-train each model with different loss function.

### Cross-Entropy Loss Train

```
python ./src/train.py -c ./configs/pretrain_ce.config
```

### Triple Loss Train

```
python ./src/train.py -c ./configs/pretrain_trp.config
```

### Mixed Loss Train

```
python ./src/train.py -c ./configs/pretrain_mix.config
```

</br>

## Evaluate

meta-evaluate all the pre-trained models.

```
python ./src/train.py -c ./configs/evaluation.config
```

