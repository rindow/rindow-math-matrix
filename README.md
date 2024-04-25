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
- BLAS functions
- Functions useful for machine learning
- Tools for integrating C/C++ through the FFI (OpenBLAS,Rindow-Matlib,CLBlast etc.)
- GPU support on your laptop without n-vidia (OpenCL with Intel,AMD etc.)
- Useful linear algebra and random number capabilities

Please see the documents on [Rindow mathematics project](https://rindow.github.io/mathematics/) web pages.

Requirements
============

- PHP 8.1 or PHP8.2 or PHP8.3
- PHP7.2, PHP7.3, PHP7.4 and PHP 8.0 are not supported in this release. Please use Release 1.1, which supports them.

### Strong recommend ###
You can perform very fast N-dimensional array operations in conjunction

- [rindow-math-matrix-matlibffi](https://github.com/rindow/rindow-math-matrix-matlibffi): plug-in drivers for OpenBLAS,Rindow-Matlib,OpenCL,CLBlast for FFI
- Pre-build binaries
  - [Rindow matlib](https://github.com/rindow/rindow-matlib/releases)
  - [OpenBLAS](https://github.com/xianyi/OpenBLAS/releases)
  - [CLBlast](https://github.com/CNugteren/CLBlast/releases)

Please see the [rindow-math-matrix-matlibffi](https://github.com/rindow/rindow-math-matrix-matlibffi) to setup plug-in and pre-build binaries.

How to Setup
============
Set it up using composer.

```shell
$ composer require rindow/rindow-math-matrix
```

You can use it as is, but you will need to speed it up to process at a practical speed.

And then, Set up pre-build binaries for the required high-speed calculation libraries. Click [here](https://github.com/rindow/rindow-math-matrix-matlibffi) for details.

```shell
$ composer require rindow/rindow-math-matrix-matlibffi
```

Sample programs
===============
```php
<?php
// sample.php
include __DIR__.'/vendor/autoload.php';
use Rindow\Math\Matrix\MatrixOperator;

$mo = new MatrixOperator();
$a = $mo->array([[1,2],[3,4]]);
$b = $mo->array([[2,3],[4,5]]);
$c = $mo->cross($a,$b);
echo $mo->toString($c,indent:true)."\n";
```
```shell
$ php sample.php
[
 [10,13],
 [22,29]
]
```
