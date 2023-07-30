# RiceClassification

## Technology
|   **Name**   | **Version** |
|:------------:|:-----------:|
|   `Python`   |   3.8.13    |
|   `numpy`    |   1.22.4    |
| `tensorflow` |    2.9.0    |
|   `keras`    |    2.9.0    |

## Description
Thanks the project I learnt about convolutional neural networks, and I could
check differences between using classic MaxPooling and using convolutional 
layer with proper step.

## Database
Database is made of 5 rice types' photos. Project goal is build model to
categorized rice based on photo.

**Link to database:**
https://www.kaggle.com/datasets/muratkokludataset/rice-image-dataset

## Results

The model with convolutional layer achieved **99.19% accuracy**.
The model with MaxPooling achieved **98.75% accuracy**.

## Summary
Both models achieved good results. Difference isn't huge (about 0.5%), but
it's significance. The convolutional layer's more flexible, a model learns
the best pooling strategy, but it's increase learning time and number of 
parameters. If we care about that, it'll better to use MaxPooling/AvgPooling 
layers.