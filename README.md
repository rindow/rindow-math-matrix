The fundamental package for scientific matrix operation
=======================================================
Status:
[![Build Status](https://github.com/rindow/rindow-math-matrix/workflows/tests/badge.svg)](https://github.com/rindow/rindow-math-matrix/actions)
[![Downloads](https://img.shields.io/packagist/dt/rindow/rindow-math-matrix)](https://packagist.org/packages/rindow/rindow-math-matrix)
[![Latest Stable Version](https://img.shields.io/packagist/v/rindow/rindow-math-matrix)](https://packagist.org/packages/rindow/rindow-math-matrix)
[![License](https://img.shields.io/packagist/l/rindow/rindow-math-matrix)](https://packagist.org/packages/rindow/rindow-math-matrix)

Rindow Math Matrix is the fundamental package for scientific matrix operation

- A powerful N-dimensional array object
- Sophisticated (broadcasting) functions
- Tools for integrating C/C++ through the "rindow_openblas" extension
- Useful linear algebra and random number capabilities



Please see the documents on [Rindow projects](https://rindow.github.io/) web pages.

Requirements
============

- PHP8.0 or PHP 8.1 or PHP8.2
- PHP7.2 or PHP7.3 or PHP7.4 is not supported in this release. Please use Release 1.1, which supports PHP7.2 or PHP7.3 or PHP7.4 or PHP 8.0.



### Download the rindow_openblas extension

You can perform very fast N-dimensional array operations in conjunction

- [Pre-build binaries](https://github.com/rindow/rindow-openblas/releases)
- [Build from source](https://github.com/rindow/rindow-openblas)

### Acceleration with GPU

You can use GPU acceleration on OpenCL.

- Pre-build binaries
  - [rindow-opencl](https://github.com/rindow/rindow-opencl/releases)
  - [rindow-clblast](https://github.com/rindow/rindow-clblast/releases)
- Build from source
  - [rindow-opencl](https://github.com/rindow/rindow-opencl)
  - [rindow-clblast](https://github.com/rindow/rindow-clblast)

*Note:*

This OpenCL support extension works better in your environment and helps speed up your laptop environment without n-NVIDIA.

Tested on AMD's Bobcat architecture APU.

In the Windows environment, Integrated GPU usage was more effective than CPU, and it worked comfortably.

However, OLD AMD APU on Linux, libclc used in linux standard mesa-opencl-icd is very buggy and slow. I made a temporary fix to make it look like it would work, but gave up on careful testing.
If you have testable hardware, please test using the proprietary driver.

On the other hand, I tested with Ivy-bridge of Intel CPU and Integrated GPU.

Windows 10 standard OpenCL driver worked fine, but it was very slow and occasionally crashed.

And it worked fine and fast in Ubuntu 20.04 + beignet-opencl-icd environment.
