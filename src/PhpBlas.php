<?php
namespace Rindow\Math\Matrix;

use ArrayAccess as Buffer;
use RuntimeException;
use InvalidArgumentException;
use Interop\Polite\Math\Matrix\BLAS;
use Interop\Polite\Math\Matrix\NDArray;

class PhpBlas //implements BLASLevel1
{
    protected $blas;
    protected $forceBlas;
    protected $floatTypes= [
        NDArray::float16,NDArray::float32,NDArray::float64,
    ];

    public function __construct($blas=null,$forceBlas=null)
    {
        $this->blas = $blas;
        $this->forceBlas = $forceBlas;
    }

    public function forceBlas($forceBlas)
    {
        $this->forceBlas = $forceBlas;
    }

    protected function useBlas(Buffer $X)
    {
        if($this->blas===null)
            return false;
        return $this->forceBlas || in_array($X->dtype(),$this->floatTypes);
    }

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

    public function scal(
        int $n,
        float $alpha,
        Buffer $X, int $offsetX, int $incX) : void
    {
        if($this->useBlas($X)) {
            $this->blas->scal($n,$alpha,$X,$offsetX,$incX);
            return;
        }

        if($offsetX+($n-1)*$incX>=count($X))
            throw new RuntimeException('Vector specification too large for buffer.');

        $idx = $offsetX;
        for ($i=0; $i<$n; $i++,$idx+=$incX) {
            $X[$idx] = $X[$idx] * $alpha;
        }
    }

    /**
     *  Y := alpha * X + Y
     */
    public function axpy(
        int $n,
        float $alpha,
        Buffer $X, int $offsetX, int $incX,
        Buffer $Y, int $offsetY, int $incY ) : void
    {
        if($offsetX+($n-1)*$incX>=count($X))
            throw new RuntimeException('Vector X specification too large for buffer.');
        if($offsetY+($n-1)*$incY>=count($Y))
            throw new RuntimeException('Vector Y specification too large for buffer.');
        $idxX = $offsetX;
        $idxY = $offsetY;
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

    public function dot(
        int $n,
        Buffer $X, int $offsetX, int $incX,
        Buffer $Y, int $offsetY, int $incY ) : float
    {
        if($this->useBlas($X)) {
            return $this->blas->dot($n,$X,$offsetX,$incX,$Y,$offsetY,$incY);
        }

        if($offsetX+($n-1)*$incX>=count($X))
            throw new RuntimeException('Vector X specification too large for buffer.');
        if($offsetY+($n-1)*$incY>=count($Y))
            throw new RuntimeException('Vector Y specification too large for buffer.');
        $idxX = $offsetX;
        $idxY = $offsetY;
        $acc = 0.0;
        for ($i=0; $i<$n; $i++,$idxX+=$incX,$idxY+=$incY) {
            $acc += $X[$idxX] * $Y[$idxY];
        }
        return $acc;
    }

    public function asum(
        int $n,
        Buffer $X, int $offsetX, int $incX ) : float
    {
        if($this->useBlas($X)) {
            return $this->blas->asum($n,$X,$offsetX,$incX);
        }

        if($offsetX+($n-1)*$incX>=count($X))
            throw new RuntimeException('Vector X specification too large for buffer.');
        $idxX = $offsetX;
        $acc = 0.0;
        for ($i=0; $i<$n; $i++,$idxX+=$incX) {
            $acc += abs($X[$idxX]);
        }
        return $acc;
    }

    public function iamax(
        int $n,
        Buffer $X, int $offsetX, int $incX ) : int
    {
        if($this->useBlas($X)) {
            return $this->blas->iamax($n,$X,$offsetX,$incX);
        }

        if($offsetX+($n-1)*$incX>=count($X))
            throw new RuntimeException('Vector X specification too large for buffer.');
        $idxX = $offsetX+$incX;
        $acc = abs($X[$offsetX]);
        $idx = 0;
        for($i=1; $i<$n; $i++,$idxX+=$incX) {
            if($acc < abs($X[$idxX])) {
                $acc = abs($X[$idxX]);
                $idx = $i;
            }
        }
        return $idx;
    }

    public function iamin(
        int $n,
        Buffer $X, int $offsetX, int $incX ) : int
    {
        if(method_exists($this->blas,'iamin')&&$this->useBlas($X)) {
            return $this->blas->iamin($n,$X,$offsetX,$incX);
        }

        if($offsetX+($n-1)*$incX>=count($X))
            throw new RuntimeException('Vector X specification too large for buffer.');
        $idxX = $offsetX+$incX;
        $acc = abs($X[$offsetX]);
        $idx = 0;
        for($i=1; $i<$n; $i++,$idxX+=$incX) {
            if($acc > abs($X[$idxX])) {
                $acc = abs($X[$idxX]);
                $idx = $i;
            }
        }
        return $idx;
    }

    public function copy(
        int $n,
        Buffer $X, int $offsetX, int $incX,
        Buffer $Y, int $offsetY, int $incY ) : void
    {
        if($this->useBlas($X)) {
            $this->blas->copy($n,$X,$offsetX,$incX,$Y,$offsetY,$incY);
            return;
        }

        if($offsetX+($n-1)*$incX>=count($X))
            throw new RuntimeException('Vector X specification too large for buffer.');
        if($offsetY+($n-1)*$incY>=count($Y))
            throw new RuntimeException('Vector Y specification too large for buffer.');

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
        if($this->useBlas($X)) {
            return $this->blas->nrm2($n,$X,$offsetX,$incX);
        }
        if($offsetX+($n-1)*$incX>=count($X))
            throw new RuntimeException('Vector X specification too large for buffer.');
        $idxX = $offsetX;
        // Y := sqrt(sum(Xn ** 2))
        $sum = 0.0;
        for ($i=0; $i<$n; $i++,$idxX+=$incX) {
            $sum += $X[$idxX] ** 2;
        }
        $Y = sqrt($sum);
        return $Y;
    }

    public function rotg(
        Buffer $A, int $offsetA,
        Buffer $B, int $offsetB,
        Buffer $C, int $offsetC,
        Buffer $S, int $offsetS
        ) : void
    {
        if($this->useBlas($A)) {
            $this->blas->rotg($A,$offsetA,$B,$offsetB,$C,$offsetC,$S,$offsetS);
            return;
        }
        throw new RuntimeException("Unsupported function yet without rindow_openblas");
    }

    public function rot(
        int $n,
        Buffer $X, int $offsetX, int $incX,
        Buffer $Y, int $offsetY, int $incY,
        Buffer $C, int $offsetC,
        Buffer $S, int $offsetS
        ) : void
    {
        if($this->useBlas($X)) {
            $this->blas->rot($n,$X,$offsetX,$incX,$Y,$offsetY,$incY,$C,$offsetC,$S,$offsetS);
            return;
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
        if($this->useBlas($X)) {
            $this->blas->swap($n,$X,$offsetX,$incX,$Y,$offsetY,$incY);
            return;
        }

        if($offsetX+($n-1)*$incX>=count($X))
            throw new RuntimeException('Vector X specification too large for buffer.');
        if($offsetY+($n-1)*$incY>=count($Y))
            throw new RuntimeException('Vector Y specification too large for buffer.');

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
        float $alpha,
        Buffer $A, int $offsetA, int $ldA,
        Buffer $X, int $offsetX, int $incX,
        float $beta,
        Buffer $Y, int $offsetY, int $incY ) : void
    {
        if($this->useBlas($X)) {
            $this->blas->gemv($order,$trans,$m,$n,$alpha,
                $A,$offsetA,$ldA,$X,$offsetX,$incX,$beta,$Y,$offsetY,$incY);
            return;
        }

        if($order==BLAS::ColMajor) {
            [$m,$n] = [$n,$m];
        } elseif($order!=BLAS::RowMajor) {
            throw new InvalidArgumentException('Invalid Order type');
        }
        $rows = ($trans==BLAS::NoTrans) ? $m : $n;
        $cols = ($trans==BLAS::NoTrans) ? $n : $m;

        if($offsetA+($m-1)*$ldA+($n-1)*$incX>=count($A))
            throw new RuntimeException('Matrix specification too large for bufferA.');
        if($offsetX+($cols-1)*$incX>=count($X))
            throw new RuntimeException('Vector specification too large for bufferX.');
        if($offsetY+($rows-1)*$incY>=count($Y))
            throw new RuntimeException('Vector specification too large for bufferY.');

        $ldA_i = ($trans==BLAS::NoTrans) ? $ldA : 1;
        $ldA_j = ($trans==BLAS::NoTrans) ? 1 : $ldA;

        $idA_i = $offsetA;
        $idY = $offsetY;
        for ($i=0; $i<$rows; $i++,$idA_i+=$ldA_i,$idY+=$incY) {
            $idA = $idA_i;
            $idX = $offsetX;
            $acc = 0.0;
            for ($j=0; $j<$cols; $j++,$idA+=$ldA_j,$idX+=$incX) {
                 $acc += $alpha * $A[$idA] * $X[$idX];
            }
            if($beta==0.0) {
                $Y[$idY] = $acc;
            } else {
                $Y[$idY] = $acc + $beta * $Y[$idY];
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
        float $alpha,
        Buffer $A, int $offsetA, int $ldA,
        Buffer $B, int $offsetB, int $ldB,
        float $beta,
        Buffer $C, int $offsetC, int $ldC ) : void
    {
        if($this->useBlas($A)) {
            $this->blas->gemm($order,$transA,$transB,$m,$n,$k,$alpha,
                $A,$offsetA,$ldA,$B,$offsetB,$ldB,$beta,$C,$offsetC,$ldC);
            return;
        }

        if($order==BLAS::ColMajor) {
            [$m,$n] = [$n,$m];
        } elseif($order!=BLAS::RowMajor) {
            throw new InvalidArgumentException('Invalid Order type');
        }
        $rowsA = ($transA==BLAS::NoTrans) ? $m : $k;
        $colsA = ($transA==BLAS::NoTrans) ? $k : $m;
        $rowsB = ($transB==BLAS::NoTrans) ? $k : $n;
        $colsB = ($transB==BLAS::NoTrans) ? $n : $k;
        if($offsetA+($rowsA-1)*$ldA+($colsA-1)>=count($A))
            throw new RuntimeException('Matrix specification too large for bufferA.');
        if($offsetB+($rowsB-1)*$ldB+($colsB-1)>=count($B))
            throw new RuntimeException('Matrix specification too large for bufferB.');
        if($offsetC+($m-1)*$ldC+($n-1)>=count($C))
            throw new RuntimeException('Matrix specification too large for bufferC.');

        $ldA_m = ($transA==BLAS::NoTrans) ? $ldA : 1;
        $ldA_k = ($transA==BLAS::NoTrans) ? 1 : $ldA;
        $ldB_k = ($transB==BLAS::NoTrans) ? $ldB : 1;
        $ldB_n = ($transB==BLAS::NoTrans) ? 1 : $ldB;

        $idA_m = $offsetA;
        $idC_m = $offsetC;
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
