<?php
namespace Rindow\Math\Matrix;

use ArrayAccess as Buffer;

/**
 *
 */
interface BLASLevel3
{
    ////////////////////////////////////////////////////////////////////
    // Level 3
    ////////////////////////////////////////////////////////////////////

    /**
     *  Product of general matrix and general matrix
     *    C := alpha * AB + beta * C
     *  @param int $order       Specify matrix order
     *                              (Select from "BLAS::RowMajor" (C Lang style),
     *                                "BLAS::ColMajor"(FORTRAN style)
     *                                (Normally RowMajor should be specified))
     *  @param int $transA       Specify matrix A transposition
     *                              (Select from "BLAS::NoTrans" (as is),
     *                               "BLAS::Trans" (transpose),
     *                               "BLAS::ConjTrans" (conjugate transpose))
     *  @param int $transB       Specify matrix B transposition
     *                              (Select from "BLAS::NoTrans" (as is),
     *                               "BLAS::Trans" (transpose),
     *                               "BLAS::ConjTrans" (conjugate transpose))
     *  @param int $m           Number of rows in matrix A
     *  @param int $n           Number of columns in matrix B
     *  @param int $k           Number of columns in matrix A, number of rows in matrix B
     *  @param float $alpha     Scalar alpha
     *  @param Buffer $A        Vector A buffer
     *  @param int $offsetA     Start offset of matrix A
     *  @param int $ldA         "A" leading dimension(Usually you just need to specify the number of rows)
     *  @param Buffer $B        Vector B buffer
     *  @param int $offsetB     Start offset of matrix B
     *  @param int $ldB         "B" leading dimension(Usually you just need to specify the number of rows)
     *  @param float $beta      Scalar beta
     *  @param Buffer $C        Vector C buffer
     *  @param int $offsetC     Start offset of matrix C
     *  @param int $ldC         "C" leading dimension(Usually you just need to specify the number of rows)
     *  @return void
     */
    public function gemm(
        int $order,
        int $transA,
        int $transB,
        int $m,
        int $n,
        int $k,
        float $alpha,
        Buffer $A, int $offsetA, int $ldA,
        Buffer $B, int $offsetB, int $ldB,
        float $beta,
        Buffer $C, int $offsetC, int $ldC
    ) : void;


    /**
     *  Product of complex Hermitian matrix and general matrix
     *    C := alpha * AB + beta * C
     *        or
     *    C := alpha * BA + beta * C
     *  @param int $order       Specify matrix order
     *                              (Select from "BLAS::RowMajor" (C Lang style),
     *                                "BLAS::ColMajor"(FORTRAN style)
     *                                (Normally RowMajor should be specified))
     *  @param int $side     Specify which matrix A will come to
     *                          (Select from "BLAS::Left" (left AB), "BLAS::Right" (right BA))
     *  @param int $uplo     Specify which part of matrix A to use
     *                          (select from "BLAS::Lower" (lower triangle), "BLAS::Upper" (upper triangle))
     *  @param int $m           Number of rows in matrix A
     *  @param int $n           Number of columns in matrix B
     *  @param float $alpha     Scalar alpha
     *  @param Buffer $A        Vector A buffer
     *  @param int $offsetA     Start offset of matrix A
     *  @param int $ldA         "A" leading dimension(Usually you just need to specify the number of rows)
     *  @param Buffer $B        Vector B buffer
     *  @param int $offsetB     Start offset of matrix B
     *  @param int $ldB         "B" leading dimension(Usually you just need to specify the number of rows)
     *  @param float $beta      Scalar beta
     *  @param Buffer $C        Vector C buffer
     *  @param int $offsetC     Start offset of matrix C
     *  @param int $ldC         "C" leading dimension(Usually you just need to specify the number of rows)
     *  @return void
     */
    public function hemm(
        int $order,
        int $side,
        int $uplo,
        int $m,
        int $n,
        float $alpha,
        Buffer $A, int $offsetA, int $ldA,
        Buffer $B, int $offsetB, int $ldB,
        float $beta,
        Buffer $C, int $offsetC, int $ldC
    ) : void;

    /**
     *  perform  one  of  the  hermitian  rank  k  operations
     *    C := alpha * A conjg(A) + beta * C
     *            or
     *    C := alpha * conjg(A) A + beta * C
     */
    public function herk(
        int $order,
        int $side,
        int $trans,
        int $n,
        int $k,
        float $alpha,
        Buffer $A, int $offsetA, int $ldA,
        float $beta,
        Buffer $C, int $offsetC, int $ldC
    ) : void;

    /**
     *  perform  one  of  the  hermitian  rank  2k operations
     *    C := alpha * A conjg(B) + conjg(alpha) B conjg(A) + beta * C
     *            or
     *    C := alpha * conjg(B) A + conjg(alpha) conjg(A) B + beta * C
     */
    public function her2k(
        int $order,
        int $side,
        int $trans,
        int $n,
        int $k,
        float $alpha,
        Buffer $A, int $offsetA, int $ldA,
        Buffer $B, int $offsetB, int $ldB,
        float $beta,
        Buffer $C, int $offsetC, int $ldC
    ) : void;

    /**
     *  Product of symmetric matrix and general matrix
     *    C := alpha * AB + beta * C
     *            or
     *    C := alpha * BA + beta * C
     */
    public function symm(
        int $order,
        int $side,
        int $uplo,
        int $m,
        int $n,
        float $alpha,
        Buffer $A, int $offsetA, int $ldA,
        Buffer $B, int $offsetB, int $ldB,
        float $beta,
        Buffer $C, int $offsetC, int $ldC
    ) : void;

    /**
     *  Update rank n of symmetric matrix
     *    C := alpha * A A^T + beta * C
     *            or
     *    C := alpha * A^T A + beta * C
     */
    public function syrk(
        int $order,
        int $side,
        int $trans,
        int $m,
        int $k,
        float $alpha,
        Buffer $A, int $offsetA, int $ldA,
        float $beta,
        Buffer $C, int $offsetC, int $ldC
    ) : void;

    /**
     *  Update rank 2k of symmetric matrix
     *    C := alpha * A B^T + alpha B A^T + beta * C
     *            or
     *    C := alpha * B^T A + alpha A^T B + beta * C
     */
    public function syr2k(
        int $order,
        int $uplo,
        int $trans,
        int $n,
        int $k,
        float $alpha,
        Buffer $A, int $offsetA, int $ldA,
        Buffer $B, int $offsetB, int $ldB,
        float $beta,
        Buffer $C, int $offsetC, int $ldC
    ) : void;

    /**
     *  Product of triangular matrix and general matrix
     *    B := alpha * AB
     *            or
     *    B := alpha * BA
     */
    public function trmm(
        int $order,
        int $side,
        int $uplo,
        int $trans,
        int $diag,
        int $m,
        int $n,
        float $alpha,
        Buffer $A, int $offsetA, int $ldA,
        Buffer $B, int $offsetB, int $ldB
    ) : void;

    /**
     *  Solve matrix equation with triangular matrix as coefficient matrix
     *    B := alpha A^-1 B
     *            or
     *    B := alpha B A^-1
     */
    public function trsm(
        int $order,
        int $side,
        int $uplo,
        int $trans,
        int $diag,
        int $m,
        int $n,
        float $alpha,
        Buffer $A, int $offsetA, int $ldA,
        Buffer $B, int $offsetB, int $ldB
    ) : void;
}
