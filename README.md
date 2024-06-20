# Neural Style Transfer (NST) Project

This project demonstrates the implementation of Neural Style Transfer (NST) using two different optimization techniques: Adam optimizer and L-BFGS optimizer. NST is a technique of blending the content of one image with the style of another image using deep learning models.

## Table of Contents

1. [Introduction](#introduction)
2. [Requirements](#requirements)
3. [Project Structure](#project-structure)
4. [Usage](#usage)
5. [Results](#results)
6. [Code Explanation](#code-explanation)
7. [References](#references)

## Introduction

Neural Style Transfer uses a pre-trained VGG19 model to extract features from the content and style images. The goal is to generate a new image that maintains the content of the content image while adopting the style of the style image.

### Techniques Used

1. **Adam Optimizer**: A popular optimization algorithm in deep learning.
2. **L-BFGS Optimizer**: A quasi-Newton method that is efficient for problems with large-scale data.

## Requirements

Ensure you have the following packages installed:

- numpy
- matplotlib
- torch
- torchvision
- tqdm
- torch_snippets
- PIL (Python Imaging Library)
- tensorflow (for the Adam optimizer implementation)

You can install the required packages using the following commands:

```sh
pip install numpy matplotlib torch torchvision tqdm torch_snippets pillow tensorflow
```

## Project Structure

```
.
├── adam_optimizer_nst.py        # NST implementation using Adam optimizer
├── lbfgs_optimizer_nst.py       # NST implementation using L-BFGS optimizer
├── Content_image.jpg            # Example content image
├── Painting_Style.jpg           # Example style image
└── README.md                    # Project documentation

## Results

The generated images will be displayed after the completion of the training loops for both optimizers. You should compare the quality of the images produced by the Adam optimizer and the L-BFGS optimizer.

## Code Explanation

- **Adam Optimizer Code**: This code utilizes TensorFlow and the Adam optimizer for NST. It defines the VGG19 model, extracts content and style features, and optimizes the image iteratively while displaying the progress using `tqdm`.

- **L-BFGS Optimizer Code**: This code uses PyTorch and the L-BFGS optimizer for NST. It sets up a similar structure as the Adam optimizer code but employs the L-BFGS optimizer, which can often lead to faster convergence for NST problems.

## References

- [Neural Style Transfer Using TensorFlow](https://www.tensorflow.org/tutorials/generative/style_transfer)
- [Neural Style Transfer Using PyTorch](https://pytorch.org/tutorials/advanced/neural_style_tutorial.html)
- [Torch Snippets](https://github.com/harvitronix/torch-snippets)

This README provides an overview of the project, how to run the code, and an explanation of the techniques used. Modify paths and parameters as necessary to suit your specific use case.
