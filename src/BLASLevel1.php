<?php
namespace Rindow\Math\Matrix;

use ArrayAccess as Buffer;

/**
 *
 */
interface BLASLevel1
{
    ////////////////////////////////////////////////////////////////////
    // Level 1
    ////////////////////////////////////////////////////////////////////

    /**
     *  Add vectors
     *    Y := alpha * X + Y
     *  @param int $n           Number of elements in each vector
     *  @param float $alpha     Coefficient of scalar multiple of X vector
     *  @param Buffer $X        Vector X buffer
     *  @param int $offsetX     Start offset of vector X
     *  @param int $incX        X increment width(Normally 1 should be specified)
     *  @param Buffer $Y        Vector Y buffer
     *  @param int $offsetY     Start offset of vector Y
     *  @param int $incY        Y increment width(Normally 1 should be specified)
     *  @return void
     */
    public function axpy(
        int $n,
        float $alpha,
        Buffer $X, int $offsetX, int $incX,
        Buffer $Y, int $offsetY, int $incY
    ) : void;

    /**
     *  Calculates the sum of the absolute values of each component of the vector.
     *    ret := |x_1| + ... + |x_n|
     *  @param int $n           Number of elements in each vector
     *  @param Buffer $X        Vector X buffer
     *  @param int $offsetX     Start offset of vector X
     *  @param int $incX        X increment width(Normally 1 should be specified)
     *  @return float
     */
    public function asum(
        int $n,
        Buffer $X, int $offsetX, int $incX
    ) : float;

    /**
     *  Copy the vector from X to Y.
     *    Y := X
     *  @param int $n           Number of elements in each vector
     *  @param Buffer $X        Vector X buffer
     *  @param int $offsetX     Start offset of vector X
     *  @param int $incX        X increment width(Normally 1 should be specified)
     *  @param Buffer $Y        Vector Y buffer
     *  @param int $offsetY     Start offset of vector Y
     *  @param int $incY        Y increment width(Normally 1 should be specified)
     *  @return void
     */
    public function copy(
        int $n,
        Buffer $X, int $offsetX, int $incX,
        Buffer $Y, int $offsetY, int $incY
    ) : void;

    /**
     *  Calculate the inner product value between vectors.
     *    ret := X^t Y = x_1 * y_1 + ... + x_n * y_n
     *  @param int $n           Number of elements in each vector
     *  @param Buffer $X        Vector X buffer
     *  @param int $offsetX     Start offset of vector X
     *  @param int $incX        X increment width(Normally 1 should be specified)
     *  @param Buffer $Y        Vector Y buffer
     *  @param int $offsetY     Start offset of vector Y
     *  @param int $incY        Y increment width(Normally 1 should be specified)
     *  @return float
     */
    public function dot(
        int $n,
        Buffer $X, int $offsetX, int $incX,
        Buffer $Y, int $offsetY, int $incY
    ) : float;

    /**
     *  Compute the Euclidean norm of a vector.
     *    ret := ||X||
     *  @param int $n           Number of elements in each vector
     *  @param Buffer $X        Vector X buffer
     *  @param int $offsetX     Start offset of vector X
     *  @param int $incX        X increment width(Normally 1 should be specified)
     *  @return float
     */
    public function nrm2(
        int $n,
        Buffer $X, int $offsetX, int $incX
    ) : float;

    /**
     *  Rotate about a given point.
     *    X(i) := c * X(i) + s * Y(i)
     *    Y(i) :=-s * X(i) + c * Y(i)
     *  @param int $n           Number of elements in each vector
     *  @param Buffer $X        Vector X buffer
     *  @param int $offsetX     Start offset of vector X
     *  @param int $incX        X increment width(Normally 1 should be specified)
     *  @param Buffer $Y        Vector Y buffer
     *  @param int $offsetY     Start offset of vector Y
     *  @param int $incY        Y increment width(Normally 1 should be specified)
     *  @param float $c         value of cos A(Value obtained with rotg function.)
     *  @param float $s         value of sin A(Value obtained with rotg function.)
     *  @return void
     */
    public function rot(
        int $n,
        Buffer $X, int $offsetX, int $incX,
        Buffer $Y, int $offsetY, int $incY,
        float $c,
        float $s
    ) : void;

    /**
     *  Give the point P (a, b).
     *  Rotate this point to givens and calculate the parameters a, b, c,
     *  and s to make the y coordinate zero.
     *    Conditions description:
     *       c * a + s * b = r
     *       -s * a + c * b = 0
     *       r = ||(a,b)||
     *       c^2 + s^2 = 1
     *       z=s if |a| > |b|
     *       z=1/c if |a| <= |b| and c != 0 and r != 0
     *    Find r, z, c, s that satisfies the above description.
     *    However, when r = 0, z = 0, c = 1, and s = 0 are returned.
     *    Also, if c = 0, | a | <= | b | and c! = 0 and r! = 0, z = 1 is returned.
     *  @param float $a     X-coordinate of P: The calculated r value is stored and returned
     *  @param float $b     Y-coordinate of P: The calculated z value is stored and returned
     *  @param float $c     Stores the calculated value of c
     *  @param float $s     Stores the calculated value of s
     *  @return void
     */
    public function rotg(
        float &$a,
        float &$b,
        float &$c,
        float &$s
    ) : void;

    /**
     *
     */
    public function rotm(
        int $n,
        Buffer $X, int $offsetX, int $incX,
        Buffer $Y, int $offsetY, int $incY,
        float $p
    ) : void;

    /**
     *
     */
    public function rotmg(
        float &$d1,
        float &$d2,
        float &$b1,
        float $b2,
        float &$p
    ) : void;

    /**
     *  Multiply vector by scalar.
     *    X := alpha * X
     *  @param int $n           Number of elements in each vector
     *  @param float $alpha     Coefficient of scalar multiple of X vector
     *  @param Buffer $X        Vector X buffer
     *  @param int $offsetX     Start offset of vector X
     *  @param int $incX        X increment width(Normally 1 should be specified)
     *  @return void
     */
    public function scal(
        int $n,
        float $alpha,
        Buffer $X, int $offsetX, int $incX
    ) : void;

    /**
     *  Exchange the contents of the vector.
     *    X := Y
     *    Y := X
     *  @param int $n           Number of elements in each vector
     *  @param Buffer $X        Vector X buffer
     *  @param int $offsetX     Start offset of vector X
     *  @param int $incX        X increment width(Normally 1 should be specified)
     *  @param Buffer $Y        Vector Y buffer
     *  @param int $offsetY     Start offset of vector Y
     *  @param int $incY        Y increment width(Normally 1 should be specified)
     *  @return void
     */
    public function swap(
        int $n,
        Buffer $X, int $offsetX, int $incX,
        Buffer $Y, int $offsetY, int $incY
    ) : void;

    /**
     *  Calculates the index of the element with the largest absolute value in the vector.
     *  Note that this subscript starts from 1. If 0 is returned, n is invalid.
     *    ret := arg max |X(i)|
     *  @param int $n           Number of elements in each vector
     *  @param Buffer $X        Vector X buffer
     *  @param int $offsetX     Start offset of vector X
     *  @param int $incX        X increment width(Normally 1 should be specified)
     *  @return int             index of the element(Note that start from 0 according to cblas_)
     */
    public function iamax(
        int $n,
        Buffer $X, int $offsetX, int $incX
    ) : int;

    /**
     *  Calculates the index of the element with the smallest absolute value in the vector.
     *  Note that this subscript starts from 1. If 0 is returned, n is invalid.
     *    ret := arg min |X(i)|
     *  @param int $n           Number of elements in each vector
     *  @param Buffer $X        Vector X buffer
     *  @param int $offsetX     Start offset of vector X
     *  @param int $incX        X increment width(Normally 1 should be specified)
     *  @return int             index of the element(Note that start from 0 according to cblas_)
     */
    public function iamin(
        int $n,
        Buffer $X, int $offsetX, int $incX
    ) : int;

    //////////////////////// BUGGY ON the OpenBLAS ////////////////////////////
    ///**
    // *  Calculates the index of the largest element in the vector.
    // *  Note that this subscript starts from 1. If 0 is returned, n is invalid.
    // *    ret := arg max X(i)
    // *  @param int $n           Number of elements in each vector
    // *  @param Buffer $X        Vector X buffer
    // *  @param int $offsetX     Start offset of vector X
    // *  @param int $incX        X increment width(Normally 1 should be specified)
    // *  @return int             index of the element(Note that start from 0 according to cblas_)
    // */
    //public function imax(
    //    int $n,
    //    Buffer $X, int $offsetX, int $incX
    //) : int;

    //////////////////////// BUGGY ON the OpenBLAS ////////////////////////////
    ///**
    // *  Calculates the index of the smallest element in a vector.
    // *  Note that this subscript starts from 1. If 0 is returned, n is invalid.
    // *    ret := arg min X(i)
    // *  @param int $n           Number of elements in each vector
    // *  @param Buffer $X        Vector X buffer
    // *  @param int $offsetX     Start offset of vector X
    // *  @param int $incX        X increment width(Normally 1 should be specified)
    // *  @return int             index of the element(Note that start from 0 according to cblas_)
    // */
    //public function imin(
    //    int $n,
    //    Buffer $X, int $offsetX, int $incX
    //) : int;

}
