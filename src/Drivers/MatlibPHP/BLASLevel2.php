<?php
namespace Rindow\Math\Matrix;

use ArrayAccess as Buffer;

/**
 *
 */
interface BLASLevel2
{
    ////////////////////////////////////////////////////////////////////
    // Level 2
    ////////////////////////////////////////////////////////////////////

    /**
     *  Compute the product of a general matrix and a vector stored in band format.
     *    y := alpha * Ax + beta * y
     *  @param int $order       Specify matrix order
     *                              (Select from "BLAS::RowMajor" (C Lang style),
     *                                "BLAS::ColMajor"(FORTRAN style)
     *                                (Normally RowMajor should be specified))
     *  @param int $trans       Specify matrix transposition
     *                              (Select from "BLAS::NoTrans" (as is),
     *                               "BLAS::Trans" (transpose),
     *                               "BLAS::ConjTrans" (conjugate transpose))
     *  @param int $m           Number of rows in matrix
     *  @param int $n           Number of columns in matrix
     *  @param int $kl          Number of elements in the lower left part
     *  @param int $ku          Number of elements in the upper right part
     *  @param float $alpha     Coefficient of scalar multiple of X vector
     *  @param Buffer $A        Band format matrix "A" buffer
     *  @param int $offsetA     Start offset of matrix "A"
     *  @param int $ldA         "A" leading dimension(Usually you just need to specify the number of rows)
     *  @param Buffer $X        Vector X buffer
     *  @param int $offsetX     Start offset of vector X
     *  @param int $incX        X increment width(Normally 1 should be specified)
     *  @param float $beta      Coefficient of scalar multiple of Y vector
     *  @param Buffer $Y        Vector Y buffer
     *  @param int $offsetY     Start offset of vector Y
     *  @param int $incY        Y increment width(Normally 1 should be specified)
     *  @return void
     */
    public function gbmv(
        int $order,
        int $trans,
        int $m,
        int $n,
        int $kl,
        int $ku,
        float $alpha,
        Buffer $A, int $offsetA, int $ldA,
        Buffer $X, int $offsetX, int $incX,
        float $beta,
        Buffer $Y, int $offsetY, int $incY
    ) : void;

    /**
     *  Compute the product of a general matrix and a vector.
     *    y := alpha * Ax + beta * y
     *  @param int $order       Specify matrix order
     *                              (Select from "BLAS::RowMajor" (C Lang style),
     *                                "BLAS::ColMajor"(FORTRAN style)
     *                                (Normally RowMajor should be specified))
     *  @param int $trans       Specify matrix transposition
     *                              (Select from "BLAS::NoTrans" (as is),
     *                               "BLAS::Trans" (transpose),
     *                               "BLAS::ConjTrans" (conjugate transpose))
     *  @param int $m           Number of rows in matrix
     *  @param int $n           Number of columns in matrix
     *  @param float $alpha     Coefficient of scalar multiple of X vector
     *  @param Buffer $A        Matrix "A" buffer
     *  @param int $offsetA     Start offset of matrix "A"
     *  @param int $ldA         "A" leading dimension(Usually you just need to specify the number of rows)
     *  @param Buffer $X        Vector X buffer
     *  @param int $offsetX     Start offset of vector X
     *  @param int $incX        X increment width(Normally 1 should be specified)
     *  @param float $beta      Coefficient of scalar multiple of Y vector
     *  @param Buffer $Y        Vector Y buffer
     *  @param int $offsetY     Start offset of vector Y
     *  @param int $incY        Y increment width(Normally 1 should be specified)
     *  @return void
     */
    public function gemv(
        int $order,
        int $trans,
        int $m,
        int $n,
        float $alpha,
        Buffer $A, int $offsetA, int $ldA,
        Buffer $X, int $offsetX, int $incX,
        float $beta,
        Buffer $Y, int $offsetY, int $incY
    ) : void;

    /**
     *  Compute the product of a column vector and a row vector. (Real number)
     *    A := alpha * x y^t + A
     *  @param int $order       Specify matrix order
     *                              (Select from "BLAS::RowMajor" (C Lang style),
     *                                "BLAS::ColMajor"(FORTRAN style)
     *                                (Normally RowMajor should be specified))
     *  @param int $m           Number of rows in matrix
     *  @param int $n           Number of columns in matrix
     *  @param float $alpha     Coefficient of scalar multiple of X vector
     *  @param Buffer $X        Vector X buffer
     *  @param int $offsetX     Start offset of vector X
     *  @param int $incX        X increment width(Normally 1 should be specified)
     *  @param Buffer $Y        Vector Y buffer
     *  @param int $offsetY     Start offset of vector Y
     *  @param int $incY        Y increment width(Normally 1 should be specified)
     *  @param Buffer $A        Matrix "A" buffer
     *  @param int $offsetA     Start offset of matrix "A"
     *  @param int $ldA         "A" leading dimension(Usually you just need to specify the number of rows)
     *  @return void
     */
    public function ger(
        int $order,
        int $m,
        int $n,
        float $alpha,
        Buffer $X, int $offsetX, int $incX,
        Buffer $Y, int $offsetY, int $incY,
        Buffer $A, int $offsetA, int $ldA
    ) : void;

    /**
     *  Compute the product of a column vector and a row vector.
     *  (Complex number/Conjugate transpose)
     *    A := alpha * x y^t + A
     *  @param int $order       Specify matrix order
     *                              (Select from "BLAS::RowMajor" (C Lang style),
     *                                "BLAS::ColMajor"(FORTRAN style)
     *                                (Normally RowMajor should be specified))
     *  @param int $m           Number of rows in matrix
     *  @param int $n           Number of columns in matrix
     *  @param float $alpha     Coefficient of scalar multiple of X vector
     *  @param Buffer $X        Vector X buffer
     *  @param int $offsetX     Start offset of vector X
     *  @param int $incX        X increment width(Normally 1 should be specified)
     *  @param Buffer $Y        Vector Y buffer
     *  @param int $offsetY     Start offset of vector Y
     *  @param int $incY        Y increment width(Normally 1 should be specified)
     *  @param Buffer $A        Matrix "A" buffer
     *  @param int $offsetA     Start offset of matrix "A"
     *  @param int $ldA         "A" leading dimension(Usually you just need to specify the number of rows)
     *  @return void
     */
    public function gerc(
        int $order,
        int $m,
        int $n,
        float $alpha,
        Buffer $X, int $offsetX, int $incX,
        Buffer $Y, int $offsetY, int $incY,
        Buffer $A, int $offsetA, int $ldA
    ) : void;

    /**
     *  Compute the product of a column vector and a row vector.
     *  (Complex number/Just transpose)
     *    A := alpha * x y^t + A
     *  @param int $order       Specify matrix order
     *                              (Select from "BLAS::RowMajor" (C Lang style),
     *                                "BLAS::ColMajor"(FORTRAN style)
     *                                (Normally RowMajor should be specified))
     *  @param int $m           Number of rows in matrix
     *  @param int $n           Number of columns in matrix
     *  @param float $alpha     Coefficient of scalar multiple of X vector
     *  @param Buffer $X        Vector X buffer
     *  @param int $offsetX     Start offset of vector X
     *  @param int $incX        X increment width(Normally 1 should be specified)
     *  @param Buffer $Y        Vector Y buffer
     *  @param int $offsetY     Start offset of vector Y
     *  @param int $incY        Y increment width(Normally 1 should be specified)
     *  @param Buffer $A        Matrix "A" buffer
     *  @param int $offsetA     Start offset of matrix "A"
     *  @param int $ldA         "A" leading dimension(Usually you just need to specify the number of rows)
     *  @return void
     */
    public function geru(
        int $order,
        int $m,
        int $n,
        float $alpha,
        Buffer $X, int $offsetX, int $incX,
        Buffer $Y, int $offsetY, int $incY,
        Buffer $A, int $offsetA, int $ldA
    ) : void;

    /**
     *  Complex Hermitian band matrix and vector product
     *    y := alpha * A * x + beta * y
     *  @param int $order       Specify matrix order
     *                              (Select from "BLAS::RowMajor" (C Lang style),
     *                                "BLAS::ColMajor"(FORTRAN style)
     *                                (Normally RowMajor should be specified))
     *  @param int $m           Number of rows in matrix
     *  @param int $n           Number of columns in matrix
     *  @param float $alpha     Coefficient of scalar multiple of X vector
     *  @param Buffer $X        Vector X buffer
     *  @param int $offsetX     Start offset of vector X
     *  @param int $incX        X increment width(Normally 1 should be specified)
     *  @param Buffer $Y        Vector Y buffer
     *  @param int $offsetY     Start offset of vector Y
     *  @param int $incY        Y increment width(Normally 1 should be specified)
     *  @param Buffer $A        Matrix "A" buffer
     *  @param int $offsetA     Start offset of matrix "A"
     *  @param int $ldA         "A" leading dimension(Usually you just need to specify the number of rows)
     *  @return void
     */
    public function hbmv(
        int $order,
        int $m,
        int $n,
        float $alpha,
        Buffer $X, int $offsetX, int $incX,
        Buffer $Y, int $offsetY, int $incY,
        Buffer $A, int $offsetA, int $ldA
    ) : void;

    /**
     *  Complex Hermitian matrix and vector product
     *    y := alpha * A * x + beta * y
     *  @param int $order       Specify matrix order
     *                              (Select from "BLAS::RowMajor" (C Lang style),
     *                                "BLAS::ColMajor"(FORTRAN style)
     *                                (Normally RowMajor should be specified))
     *  @param int $m           Number of rows in matrix
     *  @param int $n           Number of columns in matrix
     *  @param float $alpha     Coefficient of scalar multiple of X vector
     *  @param Buffer $X        Vector X buffer
     *  @param int $offsetX     Start offset of vector X
     *  @param int $incX        X increment width(Normally 1 should be specified)
     *  @param Buffer $Y        Vector Y buffer
     *  @param int $offsetY     Start offset of vector Y
     *  @param int $incY        Y increment width(Normally 1 should be specified)
     *  @param Buffer $A        Matrix "A" buffer
     *  @param int $offsetA     Start offset of matrix "A"
     *  @param int $ldA         "A" leading dimension(Usually you just need to specify the number of rows)
     *  @return void
     */
    public function hemv(
        int $order,
        int $m,
        int $n,
        float $alpha,
        Buffer $X, int $offsetX, int $incX,
        Buffer $Y, int $offsetY, int $incY,
        Buffer $A, int $offsetA, int $ldA
    ) : void;

    /**
     *  Compute the product of a column vector and a row vector.
     *  (Complex Hermitian matrix/Conjugate transpose)
     *    A := alpha * x conjg(x) + A
     *  @param int $order       Specify matrix order
     *                              (Select from "BLAS::RowMajor" (C Lang style),
     *                                "BLAS::ColMajor"(FORTRAN style)
     *                                (Normally RowMajor should be specified))
     *  @param int $uplo     Specify which side of the matrix to use
     *                          (select from "U" (upper triangle), "L" (lower triangle))
     *  @param int $n           Number of columns in matrix
     *  @param float $alpha     Coefficient of scalar multiple of X vector
     *  @param Buffer $X        Vector X buffer
     *  @param int $offsetX     Start offset of vector X
     *  @param int $incX        X increment width(Normally 1 should be specified)
     *  @param Buffer $A        Matrix "A" buffer
     *  @param int $offsetA     Start offset of matrix "A"
     *  @param int $ldA         "A" leading dimension(Usually you just need to specify the number of rows)
     *  @return void
     */
    public function her(
        int $order,
        int $uplo,
        int $n,
        float $alpha,
        Buffer $X, int $offsetX, int $incX,
        Buffer $A, int $offsetA, int $ldA
    ) : void;

    /**
     *  Compute the product of x and y vectors.
     *  (Complex Hermitian matrix/Conjugate transpose)
     *    A := alpha *x*conjg( y' ) + conjg( alpha )*y*conjg( x' ) + A
     *  @param int $order       Specify matrix order
     *                              (Select from "BLAS::RowMajor" (C Lang style),
     *                                "BLAS::ColMajor"(FORTRAN style)
     *                                (Normally RowMajor should be specified))
     *  @param int $uplo     Specify which side of the matrix to use
     *                          (select from "U" (upper triangle), "L" (lower triangle))
     *  @param int $n           Number of columns in matrix
     *  @param float $alpha     Coefficient of scalar multiple of X vector
     *  @param Buffer $X        Vector X buffer
     *  @param int $offsetX     Start offset of vector X
     *  @param int $incX        X increment width(Normally 1 should be specified)
     *  @param Buffer $Y        Vector Y buffer
     *  @param int $offsetY     Start offset of vector Y
     *  @param int $incY        Y increment width(Normally 1 should be specified)
     *  @param Buffer $A        Matrix "A" buffer
     *  @param int $offsetA     Start offset of matrix "A"
     *  @param int $ldA         "A" leading dimension(Usually you just need to specify the number of rows)
     *  @return void
     */
    public function her2(
        int $order,
        int $uplo,
        int $n,
        float $alpha,
        Buffer $X, int $offsetX, int $incX,
        Buffer $Y, int $offsetY, int $incY,
        Buffer $A, int $offsetA, int $ldA
    ) : void;

    /**
    *  Compute the product of a complex Hermitian matrix.
    *    A := alpha * x conjg(x) + A
    */
    public function hpr(
        int $order,
        int $uplo,
        int $n,
        float $alpha,
        Buffer $X, int $offsetX, int $incX,
        Buffer $A, int $offsetA
    ) : void;

    /**
    *  Compute the product of a complex Hermitian matrix.
    *    A := alpha*x*conjg( y' ) + conjg( alpha )*y*conjg( x' ) + A
    */
    public function hpr2(
        int $order,
        int $uplo,
        int $n,
        float $alpha,
        Buffer $X, int $offsetX, int $incX,
        Buffer $Y, int $offsetY, int $incY,
        Buffer $A, int $offsetA
    ) : void;

    /**
    *  Product of real symmetric band matrix and vector
    *    y := alpha * A * x + beta * y
    */
    public function sbmv(
        int $order,
        int $uplo,
        int $n,
        int $k,
        float $alpha,
        Buffer $A, int $offsetA, int $ldA,
        Buffer $X, int $offsetX, int $incX,
        float $beta,
        Buffer $Y, int $offsetY, int $incY
    ) : void;

    /**
    *  Product of real symmetric matrix (packed form) and vector
    *    y := alpha * A * x + beta * y
    */
    public function spmv(
        int $order,
        int $uplo,
        int $n,
        float $alpha,
        Buffer $A, int $offsetA,
        Buffer $X, int $offsetX, int $incX,
        float $beta,
        Buffer $Y, int $offsetY, int $incY
    ) : void;

    /**
    *  Product of vectors in real symmetric matrix (packed form)
    *    A := alpha * x x^T + A
    */
    public function spr(
        int $order,
        int $uplo,
        int $n,
        float $alpha,
        Buffer $X, int $offsetX, int $incX,
        float $beta,
        Buffer $A, int $offsetA
    ) : void;

    /**
    *  Product of vectors in real symmetric matrix (packed form)
    *    A := alpha*x*y^T + alpha*y*x^T + A
    */
    public function spr2(
        int $order,
        int $uplo,
        int $n,
        float $alpha,
        Buffer $X, int $offsetX, int $incX,
        float $beta,
        Buffer $Y, int $offsetY, int $incY,
        Buffer $A, int $offsetA
    ) : void;

    /**
    *  Real product of symmetric matrix and vector
    *    y := alpha * A * x + beta * y
    */
    public function symv(
        int $order,
        int $uplo,
        int $n,
        float $alpha,
        Buffer $A, int $offsetA, int $ldA,
        Buffer $X, int $offsetX, int $incX,
        float $beta,
        Buffer $Y, int $offsetY, int $incY
    ) : void;

    /**
    *  Real product of symmetric matrix and vector
    *    A := alpha * x x^T + A
    */
    public function syr(
        int $order,
        int $uplo,
        int $n,
        float $alpha,
        Buffer $X, int $offsetX, int $incX,
        Buffer $A, int $offsetA, int $ldA
    ) : void;

    /**
    *  Real product of symmetric matrix and vector
    *    A := alpha*x*y' + alpha*y*x' + A
    */
    public function syr2(
        int $order,
        int $uplo,
        int $n,
        float $alpha,
        Buffer $X, int $offsetX, int $incX,
        Buffer $Y, int $offsetY, int $incY,
        Buffer $A, int $offsetA, int $ldA
    ) : void;

    /**
    *  Product of triangular band matrix and vector
    *    x := A * x
    */
    public function tbmv(
        int $order,
        int $uplo,
        int $trans,
        int $diag,
        int $n,
        int $k,
        Buffer $A, int $offsetA, int $ldA,
        Buffer $X, int $offsetX, int $incX
    ) : void;

    /**
    *  Solve equations with triangular band matrix as coefficient matrix
    *    x := A * x
    */
    public function tbsv(
        int $order,
        int $uplo,
        int $trans,
        int $diag,
        int $n,
        int $k,
        Buffer $A, int $offsetA, int $ldA,
        Buffer $X, int $offsetX, int $incX
    ) : void;

    /**
    *  Product of triangular matrix (packed form) and vector
    *    x := A * x
    */
    public function tpmv(
        int $order,
        int $uplo,
        int $trans,
        int $diag,
        int $n,
        Buffer $A, int $offsetA,
        Buffer $X, int $offsetX, int $incX
    ) : void;

    /**
    *  Solve equations with triangular matrix (packed form) as coefficient matrix
    *    x := A * x
    */
    public function tpsv(
        int $order,
        int $uplo,
        int $trans,
        int $diag,
        int $n,
        Buffer $A, int $offsetA,
        Buffer $X, int $offsetX, int $incX
    ) : void;

    /**
    *  Product of triangular matrix and vector
    *    x := A * x
    */
    public function trmv(
        int $order,
        int $uplo,
        int $trans,
        int $diag,
        int $n,
        Buffer $A, int $offsetA, int $ldA,
        Buffer $X, int $offsetX, int $incX
    ) : void;

    /**
    *  Solve simultaneous linear equations with triangular matrix as coefficient matrix
    *    x := A^-1 x
    */
    public function trsv(
        int $order,
        int $uplo,
        int $trans,
        int $diag,
        int $n,
        Buffer $A, int $offsetA, int $ldA,
        Buffer $X, int $offsetX, int $incX
    ) : void;
}
