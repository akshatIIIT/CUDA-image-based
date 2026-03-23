# CUDA Image Processing Project

## Overview

This project implements a GPU-accelerated image processing pipeline using CUDA. The program applies a blur filter to a large batch of images using parallel computation on the GPU.

## Objective

The goal was to demonstrate how GPU computing can accelerate image processing tasks when applied to a large dataset of images.

## Implementation Details

* A CUDA kernel (`blurKernel`) was implemented to apply a 3x3 averaging filter.
* Each thread processes one pixel in parallel.
* The program loads hundreds of images from a directory and processes them sequentially using GPU acceleration.

## Dataset

The dataset consists of 500+ images stored in a directory (`images/input`). These images were processed in a single execution of the program.

## GPU Usage

* CUDA was used for parallel pixel computation.
* Memory was allocated on the device using `cudaMalloc`.
* Data transfer between host and device was handled using `cudaMemcpy`.

## Results

* Successfully processed 500+ images in a single run.
* Output images show a visible blur effect.
* GPU parallelism significantly speeds up pixel-level operations.

## Challenges

* Debugging image loading issues (empty images)
* Managing correct file paths in batch processing
* Ensuring correct memory size allocation

## Conclusion

This project demonstrates the effectiveness of GPU acceleration for image processing tasks and highlights the advantages of parallel computation for large datasets.
