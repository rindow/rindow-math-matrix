<?php
namespace Rindow\Math\Matrix\Drivers\MatlibPHP;

use LogicException;
use InvalidArgumentException;
use Interop\Polite\Math\Matrix\BLAS;
use Interop\Polite\Math\Matrix\NDArray;
use Interop\Polite\Math\Matrix\Buffer;
use Rindow\Math\Matrix\ComplexUtils;

class PhpBlas //implements BLASLevel1
{
    use Utils;
    use ComplexUtils;

    protected $blas;
    protected $forceBlas;
    protected $floatTypes= [
        NDArray::float16,NDArray::float32,NDArray::float64,
    ];

    public function __construct($blas=null,$forceBlas=null)
    {
        //$this->blas = $blas;
        //$this->forceBlas = $forceBlas;
        $this->blas = null;
        $this->forceBlas = null;
    }

    //public function forceBlas($forceBlas)
    //{
    //    $this->forceBlas = $forceBlas;
    //}

    //protected function useBlas(Buffer $X)
    //{
    //    //if($this->blas===null)
    //    //    return false;
    //    //return $this->forceBlas || in_array($X->dtype(),$this->floatTypes);
    //    return false;
    //}

    public function getNumThreads() : int
    {
        if($this->blas===null)
            return 1;
        return $this->blas->getNumThreads();
    }

    public function getNumProcs() : int
    {
        if($this->blas===null)
            return 1;
        return $this->blas->getNumProcs();
    }

    public function getConfig() : string
    {
        if($this->blas===null)
            return 'PhpBlas';
        return $this->blas->getConfig();
    }

    public function getCorename()
    {
        if($this->blas===null)
            return 'PHP';
        return $this->blas->getCorename();
    }

    protected function sign(float $x,float $y) : float
    {
        if($y<0) {
            $x = -$x;
        }
        return $x;
    }

    /**
     * @param  int $trans BLAS::NoTrans, BLAS::Trans, BLAS::ConjTrans, BLAS::ConjNoTrans
     * @return array [bool $trans, bool $conj]
     */
    protected function codeToTrans(int $trans) : array
    {
        switch($trans) {
            case BLAS::NoTrans: {
                return [false,false];
            }
            case BLAS::Trans: {
                return [true,false];
            }
            case BLAS::ConjTrans: {
                return [true,true];
            }
            case BLAS::ConjNoTrans: {
                return [false,true];
            }
            default: {
                throw new InvalidArgumentException('Unknown Tranpose Code: '.$trans);
            }
        }
    }

    public function scal(
        int $n,
        float|object $alpha,
        Buffer $X, int $offsetX, int $incX) : void
    {
        $this->assertShapeParameter('n',$n);
        $this->assertVectorBufferSpec('X', $X, $n, $offsetX, $incX);

        $idx = $offsetX;
        if($this->cistype($X->dtype())) {
            for ($i=0; $i<$n; $i++,$idx+=$incX) {
                $X[$idx] = $this->cmul($X[$idx],$alpha);
            }
        } else {
            for ($i=0; $i<$n; $i++,$idx+=$incX) {
                $X[$idx] = $X[$idx] * $alpha;
            }
        }
    }
    /**
     *  Y := alpha * X + Y
     */
    public function axpy(
        int $n,
        float|object $alpha,
        Buffer $X, int $offsetX, int $incX,
        Buffer $Y, int $offsetY, int $incY ) : void
    {
        $this->assertShapeParameter('n',$n);
        $this->assertVectorBufferSpec('X', $X, $n, $offsetX, $incX);
        $this->assertVectorBufferSpec('Y', $Y, $n, $offsetY, $incY);

        $idxX = $offsetX;
        $idxY = $offsetY;
        if($this->cistype($X->dtype())) {
            for ($i=0; $i<$n; $i++,$idxX+=$incX,$idxY+=$incY) {
                $Y[$idxY] = $this->cadd($this->cmul($alpha,$X[$idxX]),$Y[$idxY]);
            }
        } else {
            if($alpha==1.0) {   // Y := X + Y
                for ($i=0; $i<$n; $i++,$idxX+=$incX,$idxY+=$incY) {
                    $Y[$idxY] = $X[$idxX] + $Y[$idxY];
                }
            } else {            // Y := a*X + Y
                for ($i=0; $i<$n; $i++,$idxX+=$incX,$idxY+=$incY) {
                    $Y[$idxY] = $alpha * $X[$idxX] + $Y[$idxY];
                }
            }
        }
    }

    public function dot(
        int $n,
        Buffer $X, int $offsetX, int $incX,
        Buffer $Y, int $offsetY, int $incY ) : float|object
    {
        if($this->cistype($X->dtype())) {
            throw new InvalidArgumentException('Unsuppored data type.');
        }
        $this->assertShapeParameter('n',$n);
        $this->assertVectorBufferSpec('X', $X, $n, $offsetX, $incX);
        $this->assertVectorBufferSpec('Y', $Y, $n, $offsetY, $incY);

        $idxX = $offsetX;
        $idxY = $offsetY;
        if($this->cistype($X->dtype())) {
            $acc = $this->cbuild(0.0);
            for ($i=0; $i<$n; $i++,$idxX+=$incX,$idxY+=$incY) {
                $acc = $this->cadd($acc,$this->cmul($X[$idxX],$Y[$idxY]));
            }
        } else {
            $acc = 0.0;
            for ($i=0; $i<$n; $i++,$idxX+=$incX,$idxY+=$incY) {
                $acc += $X[$idxX] * $Y[$idxY];
            }
        }
        return $acc;
    }

    public function dotu(
        int $n,
        Buffer $X, int $offsetX, int $incX,
        Buffer $Y, int $offsetY, int $incY ) : float|object
    {
        if(!$this->cistype($X->dtype())) {
            throw new InvalidArgumentException('Unsuppored data type.');
        }
        $this->assertShapeParameter('n',$n);
        $this->assertVectorBufferSpec('X', $X, $n, $offsetX, $incX);
        $this->assertVectorBufferSpec('Y', $Y, $n, $offsetY, $incY);

        $idxX = $offsetX;
        $idxY = $offsetY;
        $acc = $this->cbuild(0.0);
        for ($i=0; $i<$n; $i++,$idxX+=$incX,$idxY+=$incY) {
            $acc = $this->cadd($acc,$this->cmul($X[$idxX],$Y[$idxY]));
        }
        return $acc;
    }

    public function dotc(
        int $n,
        Buffer $X, int $offsetX, int $incX,
        Buffer $Y, int $offsetY, int $incY ) : float|object
    {
        if(!$this->cistype($X->dtype())) {
            throw new InvalidArgumentException('Unsuppored data type.');
        }
        $this->assertShapeParameter('n',$n);
        $this->assertVectorBufferSpec('X', $X, $n, $offsetX, $incX);
        $this->assertVectorBufferSpec('Y', $Y, $n, $offsetY, $incY);

        $idxX = $offsetX;
        $idxY = $offsetY;
        $acc = $this->cbuild(0.0);
        for ($i=0; $i<$n; $i++,$idxX+=$incX,$idxY+=$incY) {
            $acc = $this->cadd($acc,$this->cmul($this->cconj($X[$idxX]),$Y[$idxY]));
        }
        return $acc;
    }

    public function asum(
        int $n,
        Buffer $X, int $offsetX, int $incX ) : float
    {
        $this->assertShapeParameter('n',$n);
        $this->assertVectorBufferSpec('X', $X, $n, $offsetX, $incX);

        $idxX = $offsetX;
        if($this->cistype($X->dtype())) {
            $acc = 0.0;
            for ($i=0; $i<$n; $i++,$idxX+=$incX) {
                $acc += $this->cabs($X[$idxX]);
            }
        } else {
            $acc = 0.0;
            for ($i=0; $i<$n; $i++,$idxX+=$incX) {
                $acc += abs($X[$idxX]);
            }
        }
        return $acc;
    }

    public function iamax(
        int $n,
        Buffer $X, int $offsetX, int $incX ) : int
    {
        $this->assertShapeParameter('n',$n);
        $this->assertVectorBufferSpec('X', $X, $n, $offsetX, $incX);

        $idxX = $offsetX+$incX;
        $idx = 0;
        if($this->cistype($X->dtype())) {
            $acc = $this->cabs($X[$offsetX]);
            for($i=1; $i<$n; $i++,$idxX+=$incX) {
                $abs = $this->cabs($X[$idxX]);
                if($acc < $abs) {
                    $acc = $abs;
                    $idx = $i;
                }
            }
        } else {
            $acc = abs($X[$offsetX]);
            for($i=1; $i<$n; $i++,$idxX+=$incX) {
                $abs = abs($X[$idxX]);
                if($acc < $abs) {
                    $acc = $abs;
                    $idx = $i;
                }
            }
        }
        return $idx;
    }

    public function iamin(
        int $n,
        Buffer $X, int $offsetX, int $incX ) : int
    {
        $this->assertShapeParameter('n',$n);
        $this->assertVectorBufferSpec('X', $X, $n, $offsetX, $incX);

        $idxX = $offsetX+$incX;
        $idx = 0;
        if($this->cistype($X->dtype())) {
            $acc = $this->cabs($X[$offsetX]);
            for($i=1; $i<$n; $i++,$idxX+=$incX) {
                $abs = $this->cabs($X[$idxX]);
                if($acc > $abs) {
                    $acc = $abs;
                    $idx = $i;
                }
            }
        } else {
            $acc = abs($X[$offsetX]);
            for($i=1; $i<$n; $i++,$idxX+=$incX) {
                $abs = abs($X[$idxX]);
                if($acc > $abs) {
                    $acc = $abs;
                    $idx = $i;
                }
            }
        }
        return $idx;
    }

    public function copy(
        int $n,
        Buffer $X, int $offsetX, int $incX,
        Buffer $Y, int $offsetY, int $incY ) : void
    {
        $this->assertShapeParameter('n',$n);
        $this->assertVectorBufferSpec('X', $X, $n, $offsetX, $incX);
        $this->assertVectorBufferSpec('Y', $Y, $n, $offsetY, $incY);

        $idxX = $offsetX;
        $idxY = $offsetY;
        for($i=0; $i<$n; $i++,$idxX+=$incX,$idxY+=$incY) {
            $Y[$idxY] = $X[$idxX];
        }
    }

    public function nrm2(
        int $n,
        Buffer $X, int $offsetX, int $incX
        ) : float
    {
        $this->assertShapeParameter('n',$n);
        $this->assertVectorBufferSpec('X', $X, $n, $offsetX, $incX);

        $idxX = $offsetX;
        // Y := sqrt(sum(Xn ** 2))
        if($this->cistype($X->dtype())) {
            $sum = 0.0;
            for ($i=0; $i<$n; $i++,$idxX+=$incX) {
                $real = $X[$idxX]->real;
                $imag = $X[$idxX]->imag;
                $sum += $real*$real +  $imag*$imag;
            }
            $Y = sqrt($sum);
        } else {
            $sum = 0.0;
            for ($i=0; $i<$n; $i++,$idxX+=$incX) {
                $v = $X[$idxX];
                $sum += $v*$v;
            }
            $Y = sqrt($sum);
        }
        return $Y;
    }

    public function rotg(
        Buffer $A, int $offsetA,
        Buffer $B, int $offsetB,
        Buffer $C, int $offsetC,
        Buffer $S, int $offsetS
        ) : void
    {
        if($this->cistype($A->dtype())) {
            throw new InvalidArgumentException('Unsuppored data type.');
        }
        $a = $A[$offsetA];
        $b = $B[$offsetB];
        // r
        if(abs($a)>abs($b)) {
            $r = $this->sign(sqrt($a**2 + $b**2),$a);
        } else {
            $r = $this->sign(sqrt($a**2 + $b**2),$b);
        }
        // c
        if($r!=0) {
            $c = $a/$r;
        } else {
            $c = 1;
        }
        // s
        if($r!=0) {
            $s = $a/$r;
        } else {
            $s = 0;
        }
        // z
        if(abs($a)>abs($b)) {
            $z = $s;
        } else {
            if($r!=0) {
                if($c!=0) {
                    $z = 1/$c;
                } else {
                    $z = 1;
                }
            } else {
                $z = 0;
            }
        }
        $A[$offsetA] = $r;
        $B[$offsetB] = $z;
        $C[$offsetC] = $c;
        $S[$offsetS] = $s;
    }

    public function rot(
        int $n,
        Buffer $X, int $offsetX, int $incX,
        Buffer $Y, int $offsetY, int $incY,
        Buffer $C, int $offsetC,
        Buffer $S, int $offsetS
        ) : void
    {
        if($this->cistype($X->dtype())) {
            throw new InvalidArgumentException('Unsuppored data type.');
        }
        $cc = $C[$offsetC];
        $ss = $S[$offsetS];
        $idX = $offsetX;
        $idY = $offsetY;
        for($i=0;$i<$n;$i++,$idX+=$incX,$idY+=$incY) {
            $xx = $X[$idX];
            $yy = $Y[$idY];
            $X[$idX] =  $cc * $xx + $ss * $yy;
            $Y[$idY] = -$ss * $xx + $cc * $yy;
        }
    }

    public function swap(
        int $n,
        Buffer $X, int $offsetX, int $incX,
        Buffer $Y, int $offsetY, int $incY ) : void
    {
        $this->assertShapeParameter('n',$n);
        $this->assertVectorBufferSpec('X', $X, $n, $offsetX, $incX);
        $this->assertVectorBufferSpec('Y', $Y, $n, $offsetY, $incY);

        $idxX = $offsetX;
        $idxY = $offsetY;
        for($i=0; $i<$n; $i++,$idxX+=$incX,$idxY+=$incY) {
            $tmp = $Y[$idxY];
            $Y[$idxY] = $X[$idxX];
            $X[$idxX] = $tmp;
        }
    }

    public function gemv(
        int $order,
        int $trans,
        int $m,
        int $n,
        float|object $alpha,
        Buffer $A, int $offsetA, int $ldA,
        Buffer $X, int $offsetX, int $incX,
        float|object $beta,
        Buffer $Y, int $offsetY, int $incY ) : void
    {
        if($order==BLAS::ColMajor) {
            [$m,$n] = [$n,$m];
        } elseif($order!=BLAS::RowMajor) {
            throw new InvalidArgumentException('Invalid Order type');
        }
        [$trans,$conj] = $this->codeToTrans($trans);
        $rows = (!$trans) ? $m : $n;
        $cols = (!$trans) ? $n : $m;

        $this->assertShapeParameter('m',$m);
        $this->assertShapeParameter('n',$n);
        $this->assertMatrixBufferSpec("A", $A, $m, $n, $offsetA, $ldA);

        $this->assertVectorBufferSpec('X', $X, $cols, $offsetX, $incX);
        $this->assertVectorBufferSpec('Y', $Y, $rows, $offsetY, $incY);

        $ldA_i = (!$trans) ? $ldA : 1;
        $ldA_j = (!$trans) ? 1 : $ldA;

        $idA_i = $offsetA;
        $idY = $offsetY;
        if($this->cistype($X->dtype())) {
            $hasAlpha = !$this->cisone($alpha);
            $hasBeta = !$this->ciszero($beta);
            $betaIsNotOne = !$this->cisone($beta);
            for($i=0; $i<$rows; $i++,$idA_i+=$ldA_i,$idY+=$incY) {
                $idA = $idA_i;
                $idX = $offsetX;
                $acc = $this->cbuild(0.0);
                for($j=0; $j<$cols; $j++,$idA+=$ldA_j,$idX+=$incX) {
                    // acc += alpha*A*X
                    $v = $A[$idA];
                    if($conj) {
                        $v = $this->cconj($v);
                    }
                    $v = $this->cmul($v,$X[$idX]);
                    if($hasAlpha) {
                        $v = $this->cmul($alpha,$v);
                    }
                    $acc = $this->cadd($acc,$v);
                }
                // Y = acc+beta*Y
                if($hasBeta) {
                    $v = $Y[$idY];
                    if($betaIsNotOne) {
                        $v = $this->cmul($beta,$v);
                    }
                    $acc = $this->cadd($acc,$v);
                }
                $Y[$idY] = $acc;
            }
        } else {
            $hasBeta  = $beta!=0.0;
            for ($i=0; $i<$rows; $i++,$idA_i+=$ldA_i,$idY+=$incY) {
                $idA = $idA_i;
                $idX = $offsetX;
                $acc = 0.0;
                for ($j=0; $j<$cols; $j++,$idA+=$ldA_j,$idX+=$incX) {
                    $acc += $alpha * $A[$idA] * $X[$idX];
                }
                if($hasBeta) {
                    $Y[$idY] = $acc + $beta * $Y[$idY];
                } else {
                    $Y[$idY] = $acc;
                }
            }
        }
    }

    public function gemm(
        int $order,
        int $transA,
        int $transB,
        int $m,
        int $n,
        int $k,
        float|object $alpha,
        Buffer $A, int $offsetA, int $ldA,
        Buffer $B, int $offsetB, int $ldB,
        float|object $beta,
        Buffer $C, int $offsetC, int $ldC ) : void
    {
        if($order==BLAS::ColMajor) {
            [$m,$n] = [$n,$m];
        } elseif($order!=BLAS::RowMajor) {
            throw new InvalidArgumentException('Invalid Order type');
        }
        [$transA,$conjA] = $this->codeToTrans($transA);
        [$transB,$conjB] = $this->codeToTrans($transB);

        $this->assertShapeParameter('m',$m);
        $this->assertShapeParameter('n',$n);
        $this->assertShapeParameter('k',$k);

        $rowsA = (!$transA) ? $m : $k;
        $colsA = (!$transA) ? $k : $m;
        $rowsB = (!$transB) ? $k : $n;
        $colsB = (!$transB) ? $n : $k;

        $this->assertMatrixBufferSpec("A", $A, $rowsA, $colsA, $offsetA, $ldA);
        $this->assertMatrixBufferSpec("B", $B, $rowsB, $colsB, $offsetB, $ldB);
        $this->assertMatrixBufferSpec("C", $C, $m, $n, $offsetC, $ldC);

        $ldA_m = (!$transA) ? $ldA : 1;
        $ldA_k = (!$transA) ? 1 : $ldA;
        $ldB_k = (!$transB) ? $ldB : 1;
        $ldB_n = (!$transB) ? 1 : $ldB;

        $idA_m = $offsetA;
        $idC_m = $offsetC;
        if($this->cistype($A->dtype())) {
            $hasAlpha = !$this->cisone($alpha);
            $hasBeta = !$this->ciszero($beta);
            $betaIsNotOne = !$this->cisone($beta);
            for ($im=0; $im<$m; $im++,$idA_m+=$ldA_m,$idC_m+=$ldC) {
                $idB_n = $offsetB;
                $idC = $idC_m;
                for ($in=0; $in<$n; $in++,$idB_n+=$ldB_n,$idC++) {
                    $idA = $idA_m;
                    $idB = $idB_n;
                    $acc = $this->cbuild(0.0);
                    for ($ik=0; $ik<$k; $ik++,$idA+=$ldA_k,$idB+=$ldB_k) {
                        $valueA = $A[$idA];
                        $valueB = $B[$idB];
                        if($conjA) {
                            $valueA = $this->cconj($valueA);
                        }
                        if($conjB) {
                            $valueB = $this->cconj($valueB);
                        }
                        $acc = $this->cadd($acc,$this->cmul($valueA,$valueB));
                    }
                    if($hasAlpha) {
                        $acc = $this->cmul($alpha,$acc);
                    }
                    if($hasBeta) {
                        $v = $C[$idC];
                        if($betaIsNotOne) {
                            $v = $this->cmul($beta,$v);
                        }
                        $acc = $this->cadd($acc,$v);
                    }
                    $C[$idC] = $acc;
                }
            }
        } else {
            for ($im=0; $im<$m; $im++,$idA_m+=$ldA_m,$idC_m+=$ldC) {
                $idB_n = $offsetB;
                $idC = $idC_m;
                for ($in=0; $in<$n; $in++,$idB_n+=$ldB_n,$idC++) {
                    $idA = $idA_m;
                    $idB = $idB_n;
                    $acc = 0.0;
                    for ($ik=0; $ik<$k; $ik++,$idA+=$ldA_k,$idB+=$ldB_k) {
                        $acc += $A[$idA] * $B[$idB];
                    }
                    if($beta==0.0) {
                        $C[$idC] = $alpha * $acc;
                    } else {
                        $C[$idC] = $alpha * $acc + $beta * $C[$idC];
                    }
                }
            }
        }
    }

    public function symm(
        int $order,
        int $side,
        int $uplo,
        int $m,
        int $n,
        float|object $alpha,
        Buffer $A, int $offsetA, int $ldA,
        Buffer $B, int $offsetB, int $ldB,
        float|object $beta,
        Buffer $C, int $offsetC, int $ldC ) : void
    {
        if($order==BLAS::ColMajor) {
            [$m,$n] = [$n,$m];
        } elseif($order!=BLAS::RowMajor) {
            throw new InvalidArgumentException('Invalid Order type');
        }
        $this->assertShapeParameter('m',$m);
        $this->assertShapeParameter('n',$n);

        $sizeA = ($side==BLAS::Left) ? $m : $n;
        $this->assertMatrixBufferSpec("A", $A, $sizeA, $sizeA, $offsetA, $ldA);
        $this->assertMatrixBufferSpec("B", $B, $m, $n, $offsetB, $ldB);
        $this->assertMatrixBufferSpec("C", $C, $m, $n, $offsetC, $ldC);

        $ldA_m = ($uplo==BLAS::Upper) ? $ldA : 1;
        $ldA_k = ($uplo==BLAS::Upper) ? 1 : $ldA;
        $ldB_k = ($side==BLAS::Left) ? $ldB : 1;
        $ldB_n = ($side==BLAS::Left) ? 1 : $ldB;
        $ldC_m = ($side==BLAS::Left) ? $ldC : 1;
        $ldC_n = ($side==BLAS::Left) ? 1 : $ldC;
        if($side==BLAS::Right) {
            [$n,$m] = [$m,$n];
        }

        $idA_m = $offsetA;
        $idC_m = $offsetC;
        if($this->cistype($A->dtype())) {
            $hasAlpha = !$this->cisone($alpha);
            $hasBeta = !$this->ciszero($beta);
            $betaIsNotOne = !$this->cisone($beta);
            for($im=0; $im<$m; $im++,$idC_m+=$ldC_m) {
                $idB_n = $offsetB;
                $idC = $idC_m;
                for ($in=0; $in<$n; $in++,$idB_n+=$ldB_n,$idC+=$ldC_n) {
                    $idB = $idB_n;
                    $acc = $this->cbuild(0.0);
                    for ($ik=0; $ik<$sizeA; $ik++,$idB+=$ldB_k) {
                        if($ik<$im) {
                            $idA = $offsetA+$ik*$ldA_m+$im*$ldA_k;
                        } else {
                            $idA = $offsetA+$im*$ldA_m+$ik*$ldA_k;
                        }
                        $acc = $this->cadd($acc,$this->cmul($A[$idA],$B[$idB]));
                    }
                    if($hasAlpha) {
                        $acc = $this->cmul($alpha,$acc);
                    }
                    if($hasBeta) {
                        $v = $C[$idC];
                        if($betaIsNotOne) {
                            $v = $this->cmul($beta,$v);
                        }
                        $acc = $this->cadd($acc,$v);
                    }
                    $C[$idC] = $acc;
                }
            }
        } else {
            for ($im=0; $im<$m; $im++,$idC_m+=$ldC_m) {
                $idB_n = $offsetB;
                $idC = $idC_m;
                for ($in=0; $in<$n; $in++,$idB_n+=$ldB_n,$idC+=$ldC_n) {
                    $idB = $idB_n;
                    $acc = 0.0;
                    for ($ik=0; $ik<$sizeA; $ik++,$idB+=$ldB_k) {
                        if($ik<$im) {
                            $idA = $offsetA+$ik*$ldA_m+$im*$ldA_k;
                        } else {
                            $idA = $offsetA+$im*$ldA_m+$ik*$ldA_k;
                        }
                        $acc += $A[$idA] * $B[$idB];
                    }
                    if($beta==0.0) {
                        $C[$idC] = $alpha * $acc;
                    } else {
                        $C[$idC] = $alpha * $acc + $beta * $C[$idC];
                    }
                }
            }
        }
    }

    public function syrk(
        int $order,
        int $uplo,
        int $trans,
        int $n,
        int $k,
        float|object $alpha,
        Buffer $A, int $offsetA, int $ldA,
        float|object $beta,
        Buffer $C, int $offsetC, int $ldC ) : void
    {
        if($order==BLAS::ColMajor) {
            [$n,$k] = [$k,$n];
        } elseif($order!=BLAS::RowMajor) {
            throw new InvalidArgumentException('Invalid Order type');
        }
        [$trans,$conj] = $this->codeToTrans($trans);
        $this->assertShapeParameter('n',$n);
        $this->assertShapeParameter('k',$k);

        $rows = (!$trans) ? $n : $k;
        $cols = (!$trans) ? $k : $n;
        $this->assertMatrixBufferSpec("A", $A, $rows, $cols, $offsetA, $ldA);
        $this->assertMatrixBufferSpec("C", $C, $n, $n, $offsetC, $ldC);

        $ldA_m  = (!$trans) ? $ldA : 1;
        $ldA_k  = (!$trans) ? 1 : $ldA;
        $ldAT_k = ($trans)  ? $ldA : 1;
        $ldAT_n = ($trans)  ? 1 : $ldA;

        $idA_m = $offsetA;
        $idC_m = $offsetC;
        if($this->cistype($A->dtype())) {
            $hasAlpha = !$this->cisone($alpha);
            $hasBeta = !$this->ciszero($beta);
            $betaIsNotOne = !$this->cisone($beta);
            for ($im=0; $im<$n; $im++,$idA_m+=$ldA_m,$idC_m+=$ldC) {
                $idAT_n = $offsetA;
                $idC = $idC_m;
                if($uplo==Blas::Upper) {
                    $start_n = $im;
                    $end_n = $n;
                } else {
                    $start_n = 0;
                    $end_n = $im+1;
                }
                for ($in=$start_n; $in<$end_n; $in++,$idAT_n+=$ldAT_n,$idC++) {
                    $acc = $this->cbuild(0.0);
                    for ($ik=0; $ik<$k; $ik++) {
                        $idA  = $offsetA+$im*$ldA_m+$ik*$ldA_k;
                        $idAT = $offsetA+$ik*$ldAT_k+$in*$ldAT_n;
                        $valueA  = $A[$idA];
                        $valueAT = $A[$idAT];
                        if($conj) {
                            $valueA  = $this->cconj($valueA);
                            $valueAT = $this->cconj($valueAT);
                        }
                        $acc = $this->cadd($acc,$this->cmul($valueA,$valueAT));
                    }
                    $idC = $im*$ldC+$in;
                    if($hasAlpha) {
                        $acc = $this->cmul($alpha,$acc);
                    }
                    if($hasBeta) {
                        $v = $C[$idC];
                        if($betaIsNotOne) {
                            $v = $this->cmul($beta,$v);
                        }
                        $acc = $this->cadd($acc,$v);
                    }
                    $C[$idC] = $acc;
                }
            }
        } else {
            for ($im=0; $im<$n; $im++,$idA_m+=$ldA_m,$idC_m+=$ldC) {
                $idAT_n = $offsetA;
                $idC = $idC_m;
                if($uplo==Blas::Upper) {
                    $start_n = $im;
                    $end_n = $n;
                } else {
                    $start_n = 0;
                    $end_n = $im+1;
                }
                for ($in=$start_n; $in<$end_n; $in++,$idAT_n+=$ldAT_n,$idC++) {
                    $acc = 0.0;
                    for ($ik=0; $ik<$k; $ik++) {
                        $idA  = $offsetA+$im*$ldA_m+$ik*$ldA_k;
                        $idAT = $offsetA+$ik*$ldAT_k+$in*$ldAT_n;
                        $acc += $A[$idA] * $A[$idAT];
                    }
                    $idC = $im*$ldC+$in;
                    if($beta==0.0) {
                        $C[$idC] = $alpha * $acc;
                    } else {
                        $C[$idC] = $alpha * $acc + $beta * $C[$idC];
                    }
                }
            }
        }
    }

    public function syr2k(
        int $order,
        int $uplo,
        int $trans,
        int $n,
        int $k,
        float|object $alpha,
        Buffer $A, int $offsetA, int $ldA,
        Buffer $B, int $offsetB, int $ldB,
        float|object $beta,
        Buffer $C, int $offsetC, int $ldC ) : void
    {
        if($order==BLAS::ColMajor) {
            [$n,$k] = [$k,$n];
        } elseif($order!=BLAS::RowMajor) {
            throw new InvalidArgumentException('Invalid Order type');
        }
        [$trans,$conj] = $this->codeToTrans($trans);

        $this->assertShapeParameter('n',$n);
        $this->assertShapeParameter('k',$k);

        $rows = (!$trans) ? $n : $k;
        $cols = (!$trans) ? $k : $n;

        $this->assertMatrixBufferSpec("A", $A, $rows, $cols, $offsetA, $ldA);
        $this->assertMatrixBufferSpec("B", $B, $rows, $cols, $offsetB, $ldB);
        $this->assertMatrixBufferSpec("C", $C, $n, $n, $offsetC, $ldC);

        $ldA_m  = (!$trans) ? $ldA : 1;
        $ldA_k  = (!$trans) ? 1 : $ldA;
        $ldAT_k = ($trans) ? $ldA : 1;
        $ldAT_n = ($trans) ? 1 : $ldA;
        $ldB_m  = (!$trans) ? $ldB : 1;
        $ldB_k  = (!$trans) ? 1 : $ldB;
        $ldBT_k = ($trans) ? $ldB : 1;
        $ldBT_n = ($trans) ? 1 : $ldB;

        $idA_m = $offsetA;
        $idC_m = $offsetC;
        if($this->cistype($A->dtype())) {
            $hasAlpha = !$this->cisone($alpha);
            $hasBeta = !$this->ciszero($beta);
            $betaIsNotOne = !$this->cisone($beta);
            for ($im=0; $im<$n; $im++,$idA_m+=$ldA_m,$idC_m+=$ldC) {
                $idAT_n = $offsetA;
                $idC = $idC_m;
                if($uplo==Blas::Upper) {
                    $start_n = $im;
                    $end_n = $n;
                } else {
                    $start_n = 0;
                    $end_n = $im+1;
                }
                for ($in=$start_n; $in<$end_n; $in++,$idAT_n+=$ldAT_n,$idC++) {
                    $acc = $this->cbuild(0.0);
                    for ($ik=0; $ik<$k; $ik++) {
                        $idA  = $offsetA+$im*$ldA_m+ $ik*$ldA_k;
                        $idB  = $offsetB+$im*$ldB_m+ $ik*$ldB_k;
                        $idAT = $offsetA+$ik*$ldAT_k+$in*$ldAT_n;
                        $idBT = $offsetB+$ik*$ldBT_k+$in*$ldBT_n;
                        $valueA  = $A[$idA];
                        $valueAT = $A[$idAT];
                        if($conj) {
                            $valueA  = $this->cconj($valueA);
                            $valueAT = $this->cconj($valueAT);
                        }
                        $acc = $this->cadd($acc,$this->cmul($valueA,$B[$idBT]));
                        $acc = $this->cadd($acc,$this->cmul($valueAT,$B[$idB]));
                    }
                    $idC = $im*$ldC+$in;
                    if($hasAlpha) {
                        $acc = $this->cmul($alpha,$acc);
                    }
                    if($hasBeta) {
                        $v = $C[$idC];
                        if($betaIsNotOne) {
                            $v = $this->cmul($beta,$v);
                        }
                        $acc = $this->cadd($acc,$v);
                    }
                    $C[$idC] = $acc;
                }
            }
        } else {
            for ($im=0; $im<$n; $im++,$idA_m+=$ldA_m,$idC_m+=$ldC) {
                $idAT_n = $offsetA;
                $idC = $idC_m;
                if($uplo==Blas::Upper) {
                    $start_n = $im;
                    $end_n = $n;
                } else {
                    $start_n = 0;
                    $end_n = $im+1;
                }
                for ($in=$start_n; $in<$end_n; $in++,$idAT_n+=$ldAT_n,$idC++) {
                    $acc = 0.0;
                    for ($ik=0; $ik<$k; $ik++) {
                        $idA  = $offsetA+$im*$ldA_m+ $ik*$ldA_k;
                        $idB  = $offsetB+$im*$ldB_m+ $ik*$ldB_k;
                        $idAT = $offsetA+$ik*$ldAT_k+$in*$ldAT_n;
                        $idBT = $offsetB+$ik*$ldBT_k+$in*$ldBT_n;
                        $acc += $A[$idA]  * $B[$idBT];
                        $acc += $A[$idAT] * $B[$idB];
                    }
                    $idC = $im*$ldC+$in;
                    if($beta==0.0) {
                        $C[$idC] = $alpha * $acc;
                    } else {
                        $C[$idC] = $alpha * $acc + $beta * $C[$idC];
                    }
                }
            }
        }
    }


    /**
     *   B(m,n) = alpha * A(m,m)B(m,n)  : side=Left
     *   B(m,n) = alpha * B(m,n)A(n,n)  : side=Right
     */
    public function trmm(
        int $order,
        int $side,  // left or right
        int $uplo,  // upper or lower
        int $trans, // trans A
        int $diag,  // no unit or unit
        int $m,
        int $n,
        float|object $alpha,
        Buffer $A, int $offsetA, int $ldA,
        Buffer $B, int $offsetB, int $ldB) : void
    {
        if($order==BLAS::ColMajor) {
            [$m,$n] = [$n,$m];
        } elseif($order!=BLAS::RowMajor) {
            throw new InvalidArgumentException('Invalid Order type');
        }
        [$trans,$conj] = $this->codeToTrans($trans);

        $this->assertShapeParameter('m',$m);
        $this->assertShapeParameter('n',$n);

        if($side==BLAS::Left) {
            $sizeA = $m;
            $sizeB = $n;
            $right = false;
        } elseif($side==BLAS::Right) {
            $sizeA = $n;
            $sizeB = $m;
            $right = true;
        } else {
            throw new InvalidArgumentException('Invalid side value: '.$side);
        }
        if($uplo==BLAS::Upper) {
            $lower = false;
        } elseif($uplo==BLAS::Lower) {
            $lower = true;
        } else {
            throw new InvalidArgumentException('Invalid uplo value: '.$uplo);
        }
        if($diag==BLAS::NonUnit) {
            $unit = false;
        } elseif($diag==BLAS::Unit) {
            $unit = true;
        } else {
            throw new InvalidArgumentException('Invalid diag value: '.$diag);
        }
        $rowsB = $m;
        $colsB = $n;

        $this->assertMatrixBufferSpec("A", $A, $sizeA, $sizeA, $offsetA, $ldA);
        $this->assertMatrixBufferSpec("B", $B, $rowsB, $colsB, $offsetB, $ldB);

        $trans = $right ? !$trans : $trans;
        $lower = $trans ? !$lower : $lower;
        $ldA_m = $trans ? 1 : $ldA;
        $ldA_k = $trans ? $ldA : 1;
        $ldB_k = $right ? 1 : $ldB;
        $ldB_n = $right ? $ldB : 1;

        $startm = $lower?($sizeA-1):0;
        $stepm =  $lower?(-1):1;
        if($this->cistype($A->dtype())) {
            $hasAlpha = !$this->cisone($alpha);
            for($cm=0,$im=$startm;$cm<$sizeA;$cm++,$im+=$stepm) {
                for($in=0;$in<$sizeB;$in++) {
                    if($unit) {
                        $startk = $lower?0:$im+1;
                        $countk = $sizeA-$cm-1;
                        $acc = $B[$offsetB+$im*$ldB_k+$in*$ldB_n];
                    } else {
                        $startk = $lower?0:$im;
                        $countk = $sizeA-$cm;
                        $acc = $this->cbuild(0.0);
                    }
                    for($ck=0,$ik=$startk; $ck<$countk; $ck++,$ik++) {
                        $v = $A[$offsetA+$im*$ldA_m+$ik*$ldA_k];
                        if($conj) {
                            $v = $this->cconj($v);
                        }
                        $acc = $this->cadd($acc,$this->cmul($v,$B[$offsetB+$ik*$ldB_k+$in*$ldB_n]));
                    }
                    if($hasAlpha) {
                        $acc = $this->cmul($alpha,$acc);
                    }
                    $B[$offsetB+$im*$ldB_k+$in*$ldB_n] = $acc;
                }
            }
        } else {
            for($cm=0,$im=$startm;$cm<$sizeA;$cm++,$im+=$stepm) {
                for($in=0;$in<$sizeB;$in++) {
                    if($unit) {
                        $startk = $lower?0:$im+1;
                        $countk = $sizeA-$cm-1;
                        $acc = $B[$offsetB+$im*$ldB_k+$in*$ldB_n];
                    } else {
                        $startk = $lower?0:$im;
                        $countk = $sizeA-$cm;
                        $acc = 0.0;
                    }
                    for($ck=0,$ik=$startk; $ck<$countk; $ck++,$ik++) {
                        $acc += $A[$offsetA+$im*$ldA_m+$ik*$ldA_k]*$B[$offsetB+$ik*$ldB_k+$in*$ldB_n];
                    }
                    $B[$offsetB+$im*$ldB_k+$in*$ldB_n] = $alpha * $acc;
                }
            }
        }
    }

    public function trsm(
        int $order,
        int $side,
        int $uplo,
        int $trans,
        int $diag,
        int $m,
        int $n,
        float|object $alpha,
        Buffer $A, int $offsetA, int $ldA,
        Buffer $B, int $offsetB, int $ldB) : void
    {
        if($order==BLAS::ColMajor) {
            [$m,$n] = [$n,$m];
        } elseif($order!=BLAS::RowMajor) {
            throw new InvalidArgumentException('Invalid Order type');
        }
        [$trans,$conj] = $this->codeToTrans($trans);

        $this->assertShapeParameter('m',$m);
        $this->assertShapeParameter('n',$n);

        if($side==BLAS::Left) {
            $sizeA = $m;
            $sizeB = $n;
            $right = false;
        } elseif($side==BLAS::Right) {
            $sizeA = $n;
            $sizeB = $m;
            $right = true;
        } else {
            throw new InvalidArgumentException('Invalid side value: '.$side);
        }
        if($uplo==BLAS::Upper) {
            $lower = false;
        } elseif($uplo==BLAS::Lower) {
            $lower = true;
        } else {
            throw new InvalidArgumentException('Invalid uplo value: '.$uplo);
        }
        if($diag==BLAS::NonUnit) {
            $unit = false;
        } elseif($diag==BLAS::Unit) {
            $unit = true;
        } else {
            throw new InvalidArgumentException('Invalid diag value: '.$diag);
        }
        $rowsB = $m;
        $colsB = $n;
        $this->assertMatrixBufferSpec("A", $A, $sizeA, $sizeA, $offsetA, $ldA);
        $this->assertMatrixBufferSpec("B", $B, $rowsB, $colsB, $offsetB, $ldB);

        $trans = $right ? !$trans : $trans;
        $lower = $trans ? !$lower : $lower;
        $ldA_m = $trans ? 1 : $ldA;
        $ldA_k = $trans ? $ldA : 1;
        $ldB_k = $right ? 1 : $ldB;
        $ldB_n = $right ? $ldB : 1;

        $startm = $lower?0:($sizeA-1);
        $stepm =  $lower?1:(-1);
        if($this->cistype($A->dtype())) {
            $hasAlpha = !$this->cisone($alpha);
            // loop(i)
            for($cm=0,$im=$startm;$cm<$sizeA;$cm++,$im+=$stepm) {
                // A[i,i]
                if($unit) {
                    $denomi = 1.0;
                    $denomiFlag = 1; // is_one
                } else {
                    $denomi = $A[$offsetA+$im*$ldA_m+$im*$ldA_k];
                    if($this->ciszero($denomi)) {
                        $denomiFlag = 0; // is_zero
                        $denomi = $this->cbuild(NAN,i:NAN);
                    } else {
                        $denomiFlag = 2; // is_normal
                        if($conj) {
                            //echo "C";
                            $denomi = $this->cconj($denomi);
                        }
                    }
                }
                //echo "denomi:$denomi\n";
                // loop(j)
                //echo "for(j)[$in,$sizeB]\n";
                for($in=0;$in<$sizeB;$in++) {
                    // acc = 0;
                    $startk = $lower?0:($im+1);
                    $countk = $cm;
                    $acc = $this->cbuild(0.0);
                    // loop(k)
                    //echo "for(k)[$startk,$countk]\n";
                    for($ck=0,$ik=$startk; $ck<$countk; $ck++,$ik++) {
                        // acc += A[i,k] * B[k,j];
                        //echo "a[$im,$ik]:";
                        //echo $A[$offsetA+$im*$ldA_m+$ik*$ldA_k].",";
                        //echo "b[$ik,$in],";
                        //echo "b[".($offsetB+$ik*$ldB_k+$in*$ldB_n)."]:";
                        //echo $B[$offsetB+$ik*$ldB_k+$in*$ldB_n].",";
                        $v = $A[$offsetA+$im*$ldA_m+$ik*$ldA_k];
                        if($conj) {
                            //echo "C";
                            $v = $this->cconj($v);
                        }
                        $acc = $this->cadd($acc,$this->cmul($v,$B[$offsetB+$ik*$ldB_k+$in*$ldB_n]));
                        //echo "acc:".$acc.",";
                    }
                    //echo "endfor(k)\n";
                    //echo "acc:".$acc.",";
                    //echo "\n";
                    // B[i,j] = (B[i,j] - acc) / A[i,i];
                    //echo "B[$im,$in]";
                    //echo "[$ldB_k,$ldB_n]";
                    if($hasAlpha) {
                        $acc = $this->cmul($alpha,$acc);
                    }
                    if($denomiFlag==0) { // NAN
                        $B[$offsetB+$im*$ldB_k+$in*$ldB_n] = $denomi;
                    } elseif($denomiFlag==1) { // denomi == 1.0
                        $B[$offsetB+$im*$ldB_k+$in*$ldB_n] = $this->csub($B[$offsetB+$im*$ldB_k+$in*$ldB_n],$acc);
                    } else {
                        $B[$offsetB+$im*$ldB_k+$in*$ldB_n] = $this->cdiv($this->csub($B[$offsetB+$im*$ldB_k+$in*$ldB_n],$acc),$denomi);
                    }
                    //echo "B[".($offsetB+$im*$ldB_k+$in*$ldB_n)."]:";
                    //echo $B[$offsetB+$im*$ldB_k+$in*$ldB_n];
                    //echo "\n";
                }
                //echo "endfor(j)\n";
            }
        } else {
            // loop(i)
            for($cm=0,$im=$startm;$cm<$sizeA;$cm++,$im+=$stepm) {
                // A[i,i]
                if($unit) {
                    $denomi = 1.0;
                } else {
                    $denomi = $A[$offsetA+$im*$ldA_m+$im*$ldA_k];
                    if($denomi==0) {
                        $denomi = NAN;
                    }
                }
                // loop(j)
                for($in=0;$in<$sizeB;$in++) {
                    // acc = 0;
                    $startk = $lower?0:($im+1);
                    $countk = $cm;
                    $acc = 0.0;
                    // loop(k)
                    for($ck=0,$ik=$startk; $ck<$countk; $ck++,$ik++) {
                        // acc += A[i,k] * B[k,j];
                        $acc += $A[$offsetA+$im*$ldA_m+$ik*$ldA_k]*$B[$offsetB+$ik*$ldB_k+$in*$ldB_n];
                    }
                    // B[i,j] = (B[i,j] - acc) / A[i,i];
                    $B[$offsetB+$im*$ldB_k+$in*$ldB_n] = ($B[$offsetB+$im*$ldB_k+$in*$ldB_n] - $alpha*$acc) / $denomi;
                }
            }
        }
    }

    public function omatcopy(
        int $order,
        int $trans,
        int $m,
        int $n,
        float|object $alpha,
        Buffer $A, int $offsetA, int $ldA,
        Buffer $B, int $offsetB, int $ldB,
    ) : void
    {
        if($order==BLAS::ColMajor) {
            [$m,$n] = [$n,$m];
        } elseif($order!=BLAS::RowMajor) {
            throw new InvalidArgumentException('Invalid Order type');
        }
        [$trans,$conj] = $this->codeToTrans($trans);
        $this->assertShapeParameter('m',$m);
        $this->assertShapeParameter('n',$n);
        $rows = (!$trans) ? $m : $n;
        $cols = (!$trans) ? $n : $m;

        $this->assertMatrixBufferSpec("A", $A, $m, $n, $offsetA, $ldA);
        $this->assertMatrixBufferSpec("B", $B, $rows, $cols, $offsetB, $ldB);

        // Check Buffer A and B
        if($A->dtype()!=$B->dtype()) {
            throw new InvalidArgumentException("Unmatch data type for A and B");
        }

        $ldA_i = (!$trans) ? $ldA : 1;
        $ldA_j = (!$trans) ? 1 : $ldA;

        $idA_i = $offsetA;
        $idB_i = $offsetB;

        if($this->cistype($A->dtype())) {
            $hasAlpha = !$this->cisone($alpha);
            for($i=0; $i<$rows; $i++,$idA_i+=$ldA_i,$idB_i+=$ldB) {
                $idA = $idA_i;
                $idB = $idB_i;
                for($j=0; $j<$cols; $j++,$idA+=$ldA_j,$idB++) {
                    $v = $A[$idA];
                    if($conj) {
                        $v = $this->cconj($v);
                    }
                    if($hasAlpha) {
                        $v = $this->cmul($alpha,$v);
                    }
                    $B[$idB] = $v;
                }
            }
        } else {
            for($i=0; $i<$rows; $i++,$idA_i+=$ldA_i,$idB_i+=$ldB) {
                $idA = $idA_i;
                $idB = $idB_i;
                for($j=0; $j<$cols; $j++,$idA+=$ldA_j,$idB++) {
                    $B[$idB] = $alpha * $A[$idA];
                }
            }
        }
    }
    
}
