# RiceClassification

## Technology
|    **Name**     | **Version** |
|:---------------:|:-----------:|
|    `Python`     |   3.8.13    |
|     `numpy`     |   1.22.4    |
|  `tensorflow`   |    2.9.3    |
|     `keras`     |    2.9.0    |
|  `matplotlib`   |    3.7.1    |
|    `seabron`    |   0.12.2    |
| `scikit-learn`  |    1.3.0    |
|    `pandas`     |    2.0.3    |

## Description
Thanks the project I learnt about convolutional neural networks, and I
checked differences between using classic MaxPooling and using convolutional 
layer with proper step.

## Database
Database is made of 5 rice types' photos. Project goal is build model to
determine rice type based on photo.

**Link to database:**
https://www.kaggle.com/datasets/muratkokludataset/rice-image-dataset

## Results

The model with convolutional layer achieved **99.19% accuracy**.
The model with MaxPooling achieved **98.75% accuracy**.

## Summary
Both models achieve good results. Difference isn't huge (about 0.5%), but
it's significance. The convolutional layer is more flexible, so the model learns
the best pooling strategy, but it increases learning time and number of 
parameters. If we care about that, it's better to use MaxPooling/AvgPooling 
layers.