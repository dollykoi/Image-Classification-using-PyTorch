# Image Classification with PyTorch

A PyTorch CNN that classifies 32 x 32 color images into 10 classes, trained on a small labeled subset and used to predict 46,000 unlabeled images. Built as a course assignment.

## Overview

The dataset has 50,000 RGB images of size 32 x 32 in 10 classes (airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck). Only 4,000 images come with labels in `Train.csv`. The task is to predict labels for the 46,000 images listed in `Test.csv`. The main challenge is getting the most out of the small labeled set, which is where data augmentation and careful hyperparameter selection come in.

## Files

- `Image_Classification_PyTorch.ipynb` - the full notebook with a written description of the approach. Runs end to end in Google Colab and downloads the data itself.
- `Test_predictions.csv` - the predictions: same 46,000 ids as `Test.csv`, with the label column filled in.

## Approach

- Custom `ImageDataset` class that subclasses `torch.utils.data.Dataset` and loads images by id from the CSV files
- Data augmentation on the training data only: random crops with padding and random horizontal flips
- CNN with three blocks, each with two convolutions plus batch normalization and ReLU, followed by max pooling (32, 64 and 128 feature maps). The head is a fully connected layer with 256 units, dropout 0.5, and a 10-way output.
- Cross entropy loss

## Hyperparameter selection

Grid search over 8 configurations, comparing:

- Optimizer: Adam vs SGD with momentum 0.9
- Learning rate: 0.001 / 0.0005 for Adam, 0.01 / 0.05 for SGD with momentum
- Batch size: 32 and 64

Each configuration is trained for 10 epochs and ranked by validation accuracy. The notebook shows the full results table and a bar chart.

## Results

- Best configuration: Adam, learning rate 0.0005, batch size 64
- Final model trained on all 4,000 labeled images for 40 epochs: best validation accuracy 80.6%
- The predictions are roughly balanced across the 10 classes, as expected for this dataset

## How to run

1. Open `Image_Classification_PyTorch.ipynb` in Google Colab
2. Run all cells top to bottom. The first cell downloads and unzips the dataset from the provided links.
3. The notebook trains the model, writes `submission.csv`, and downloads it to your computer in the last cell.

Main dependencies: torch, torchvision, pandas, matplotlib, gdown.
