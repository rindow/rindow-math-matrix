<?php
namespace Rindow\Math\Matrix;

use ArrayAccess as Buffer;
use RuntimeException;
use InvalidArgumentException;
use Interop\Polite\Math\Matrix\NDArray;

class PhpMath
{
    protected $math;
    protected $forceMath;
    protected $intTypes= [
        NDArray::int8,NDArray::int16,NDArray::int32,NDArray::int64,
        NDArray::uint8,NDArray::uint16,NDArray::uint32,NDArray::uint64,
    ];
    protected $floatTypes= [
        NDArray::float16,NDArray::float32,NDArray::float64,
    ];

    public function __construct($math=null,$forceMath=null)
    {
        $this->math = $math;
        $this->forceMath = $forceMath;
    }

    public function forceMath($forceMath)
    {
        $this->forceMath = $forceMath;
    }

    protected function useMath(Buffer $X)
    {
        if($this->math===null)
            return false;
        return $this->forceMath || in_array($X->dtype(),$this->floatTypes);
    }

    protected function math_copy(
        int $n,
        Buffer $X, int $offsetX, int $incX,
        Buffer $Y, int $offsetY, int $incY ) : void
    {
        $idxX = $offsetX;
        $idxY = $offsetY;
        for ($i=0; $i<$n; $i++,$idxX+=$incX,$idxY+=$incY) {
            $Y[$idxY] = $X[$idxX];
        }
    }

    protected function math_add(
        int $n,
        Buffer $X, int $offsetX, int $incX,
        Buffer $Y, int $offsetY, int $incY ) : void
    {
        $idxX = $offsetX;
        $idxY = $offsetY;
        for ($i=0; $i<$n; $i++,$idxX+=$incX,$idxY+=$incY) {
            $Y[$idxY] = $Y[$idxY] + $X[$idxX];
        }
    }

    protected function math_sum(
        int $n,
        Buffer $X, int $offsetX, int $incX ) : float
    {
        $idxX = $offsetX;
        $acc = 0.0;
        for ($i=0; $i<$n; $i++,$idxX+=$incX) {
            $acc += $X[$idxX];
        }
        return $acc;
    }

    protected function math_max(
        int $n,
        Buffer $X, int $offsetX, int $incX ) : float
    {
        $idxX = $offsetX;
        $max = $X[$idxX];
        $idxX += $incX;
        for ($i=1; $i<$n; $i++,$idxX+=$incX) {
            $value = $X[$idxX];
            // *** CAUTION ***
            // if NaN set NaN
            // Compatible with reduce_max of tensorflow 2.6
            if(!($value<=$max)) {
                $max = $value;
            }
        }
        return $max;
    }

    protected function math_imax(
        int $n,
        Buffer $X, int $offsetX, int $incX ) : float
    {
        $idxX = $offsetX;
        $max = $X[$idxX];
        $imax = 0;
        $idxX += $incX;
        for ($i=1; $i<$n; $i++,$idxX+=$incX) {
            $value = $X[$idxX];
            if($value>$max) {
                $max = $value;
                $imax = $i;
            }
        }
        return $imax;
    }

    /**
     *     sum := sum(X)
     */
    public function sum(
        int $n,
        Buffer $X, int $offsetX, int $incX ) : float
    {
        if($this->math) {
            return $this->math->sum($n,$X,$offsetX,$incX);
        }

        if($offsetX+($n-1)*$incX>=count($X))
            throw new RuntimeException('Vector X specification too large for buffer.');
        $idxX = $offsetX;
        $acc = 0.0;
        for ($i=0; $i<$n; $i++,$idxX+=$incX) {
            $acc += $X[$idxX];
        }
        return $acc;
    }

    /**
     *     index := max(X)
     */
    public function imax(
        int $n,
        Buffer $X, int $offsetX, int $incX) : int
    {
        if($this->useMath($X)) {
            return $this->math->imax($n,$X,$offsetX,$incX);
        }

        if($offsetX+($n-1)*$incX>=count($X))
            throw new RuntimeException('Vector X specification too large for buffer.');
        $idxX = $offsetX+$incX;
        $acc = $X[$offsetX];
        $idx = 0;
        for($i=1; $i<$n; $i++,$idxX+=$incX) {
            if($acc < $X[$idxX]||is_nan($acc)) {
                $acc = $X[$idxX];
                $idx = $i;
            }
        }
        return $idx;
    }

    /**
     *     index := min(X)
     */
    public function imin(
        int $n,
        Buffer $X, int $offsetX, int $incX) : int
    {
        if($this->useMath($X)) {
            return $this->math->imin($n,$X,$offsetX,$incX);
        }

        if($offsetX+($n-1)*$incX>=count($X))
            throw new RuntimeException('Vector X specification too large for buffer.');
        $idxX = $offsetX+$incX;
        $acc = $X[$offsetX];
        $idx = 0;
        for($i=1; $i<$n; $i++,$idxX+=$incX) {
            if($acc > $X[$idxX]) {
                $acc = $X[$idxX];
                $idx = $i;
            }
        }
        return $idx;
    }

    /**
     *     X := a*X + b
     */
    public function increment(
        int $n,
        float $alpha,
        Buffer $X, int $offsetX, int $incX,
        float $beta) : void
    {
        if($this->useMath($X)) {
            $this->math->increment($n,$alpha,$X,$offsetX,$incX,$beta);
            return;
        }

        if($offsetX+($n-1)*$incX>=count($X))
            throw new RuntimeException('Vector specification too large for buffer.');

        $idx = $offsetX;
        for ($i=0; $i<$n; $i++,$idx+=$incX) {
            $X[$idx] = $alpha*$X[$idx] + $beta;
        }
    }

    /**
     *     X := 1 / (a*X + b)
     */
    public function reciprocal(
        int $n,
        float $alpha,
        Buffer $X, int $offsetX, int $incX,
        float $beta) : void
    {
        if($this->useMath($X)) {
            $this->math->reciprocal($n,$alpha,$X,$offsetX,$incX,$beta);
            return;
        }

        if($offsetX+($n-1)*$incX>=count($X))
            throw new RuntimeException('Vector specification too large for buffer.');

        $idx = $offsetX;
        for ($i=0; $i<$n; $i++,$idx+=$incX) {
            $t = $alpha*$X[$idx] + $beta;
            if($t==0.0) {
                $r = INF;
            } else {
                $r = 1 / $t;
            }
            $X[$idx] = $r;
        }
    }

    /**
     *     A[m,n] := A[m,n] (A[m,n] >  X[n])
     *     A[m,n] := X[n]   (A[m,n] <= X[n])
     */
    public function maximum(
        int $m,
        int $n,
        Buffer $A, int $offsetA, int $ldA,
        Buffer $X, int $offsetX, int $incX
        ) : void
    {
        if($this->useMath($X)) {
            $this->math->maximum($m,$n,$A,$offsetA,$ldA,$X,$offsetX,$incX);
            return;
        }
        if($m<=0 || $n<=0) {
            throw new InvalidArgumentException("m and n must be greater than 0");
        }
        if(($m-1)*$ldA+($n-1)+$offsetA>=count($A)) {
            throw new InvalidArgumentException("Matrix A is too small");
        }
        if(($n-1)*$incX+$offsetX>=count($X)) {
            throw new InvalidArgumentException("Buffer X is too small");
        }

        $lna = $offsetA;
        for($i=0;$i<$m;$i++,$lna+=$ldA) {
            $idx = $offsetX;
            $ida = $lna;
            for ($j=0; $j<$n; $j++,$idx+=$incX,$ida++) {
                $v = $X[$idx];
                if(is_nan($v)) {
                    // if x==NaN then set NaN
                    $A[$ida] = $v;
                } else {
                    // if NaN then don't set x
                    if($A[$ida] < $v) {
                        $A[$ida] = $v;
                    }
                }
            }
        }
    }

    /**
     *     A[m,n] := A[m,n] (A[m,n] <  X[n])
     *     A[m,n] := X[n]   (A[m,n] >= X[n])
     */
    public function minimum(
        int $m,
        int $n,
        Buffer $A, int $offsetA, int $ldA,
        Buffer $X, int $offsetX, int $incX
        ) : void
    {
        if($this->useMath($X)) {
            $this->math->minimum($m,$n,$A,$offsetA,$ldA,$X,$offsetX,$incX);
            return;
        }
        if($m<=0 || $n<=0) {
            throw new InvalidArgumentException("m and n must be greater than 0");
        }
        if(($m-1)*$ldA+($n-1)+$offsetA>=count($A)) {
            throw new InvalidArgumentException("Matrix A is too small");
        }
        if(($n-1)*$incX+$offsetX>=count($X)) {
            throw new InvalidArgumentException("Buffer X is too small");
        }

        $lna = $offsetA;
        for($i=0;$i<$m;$i++,$lna+=$ldA) {
            $idx = $offsetX;
            $ida = $lna;
            for ($j=0; $j<$n; $j++,$idx+=$incX,$ida++) {
                $v = $X[$idx];
                if(is_nan($v)) {
                    // if x==NaN then set NaN
                    $A[$ida] = $v;
                } else {
                    // if NaN then don't set x
                    if($A[$ida] > $v) {
                        $A[$ida] = $v;
                    }
                }
            }
        }
    }

    /**
     *     A[m,n] := 1 (A[m,n] >  X[n])
     *     A[m,n] := 0 (A[m,n] <= X[n])
     */
    public function greater(
        int $m,
        int $n,
        Buffer $A, int $offsetA, int $ldA,
        Buffer $X, int $offsetX, int $incX
        ) : void
    {
        if($this->useMath($X)) {
            $this->math->greater($m,$n,$A,$offsetA,$ldA,$X,$offsetX,$incX);
            return;
        }
        if($m<=0 || $n<=0) {
            throw new InvalidArgumentException("m and n must be greater than 0");
        }
        if(($m-1)*$ldA+($n-1)+$offsetA>=count($A)) {
            throw new InvalidArgumentException("Matrix A is too small");
        }
        if(($n-1)*$incX+$offsetX>=count($X)) {
            throw new InvalidArgumentException("Buffer X is too small");
        }

        $lna = $offsetA;
        for($i=0;$i<$m;$i++,$lna+=$ldA) {
            $idx = $offsetX;
            $ida = $lna;
            for ($j=0; $j<$n; $j++,$idx+=$incX,$ida++) {
                // if NaN set 0.0
                // if equal set 0.0
                if($A[$ida] > $X[$idx]) {
                    $tmp = 1.0;
                } else {
                    $tmp = 0.0;
                }
                $A[$ida] = $tmp;
            }
        }
    }

    /**
     *     A[m,n] := 1 (A[m,n] >= X[n])
     *     A[m,n] := 0 (A[m,n] <  X[n])
     */
    public function greaterEqual(
        int $m,
        int $n,
        Buffer $A, int $offsetA, int $ldA,
        Buffer $X, int $offsetX, int $incX
        ) : void
    {
        if($this->useMath($X)) {
            $this->math->greaterEqual($m,$n,$A,$offsetA,$ldA,$X,$offsetX,$incX);
            return;
        }
        if($m<=0 || $n<=0) {
            throw new InvalidArgumentException("m and n must be greater than 0");
        }
        if(($m-1)*$ldA+($n-1)+$offsetA>=count($A)) {
            throw new InvalidArgumentException("Matrix A is too small");
        }
        if(($n-1)*$incX+$offsetX>=count($X)) {
            throw new InvalidArgumentException("Buffer X is too small");
        }

        $lna = $offsetA;
        for($i=0;$i<$m;$i++,$lna+=$ldA) {
            $idx = $offsetX;
            $ida = $lna;
            for ($j=0; $j<$n; $j++,$idx+=$incX,$ida++) {
                // if NaN set 0.0
                // if equal set 0.0
                if($A[$ida] >= $X[$idx]) {
                    $tmp = 1.0;
                } else {
                    $tmp = 0.0;
                }
                $A[$ida] = $tmp;
            }
        }
    }

    /**
     *     A[m,n] := 1 (A[m,n] <  X[n])
     *     A[m,n] := 0 (A[m,n] >= X[n])
     */
    public function less(
        int $m,
        int $n,
        Buffer $A, int $offsetA, int $ldA,
        Buffer $X, int $offsetX, int $incX
        ) : void
    {
        if($this->useMath($X)) {
            $this->math->less($m,$n,$A,$offsetA,$ldA,$X,$offsetX,$incX);
            return;
        }
        if($m<=0 || $n<=0) {
            throw new InvalidArgumentException("m and n must be greater than 0");
        }
        if(($m-1)*$ldA+($n-1)+$offsetA>=count($A)) {
            throw new InvalidArgumentException("Matrix A is too small");
        }
        if(($n-1)*$incX+$offsetX>=count($X)) {
            throw new InvalidArgumentException("Buffer X is too small");
        }

        $lna = $offsetA;
        for($i=0;$i<$m;$i++,$lna+=$ldA) {
            $idx = $offsetX;
            $ida = $lna;
            for ($j=0; $j<$n; $j++,$idx+=$incX,$ida++) {
                // if NaN set 0.0
                // if equal set 0.0
                if($A[$ida] < $X[$idx]) {
                    $tmp = 1.0;
                } else {
                    $tmp = 0.0;
                }
                $A[$ida] = $tmp;
            }
        }
    }

    /**
     *     A[m,n] := 1 (A[m,n] <= X[n])
     *     A[m,n] := 0 (A[m,n] >  X[n])
     */
    public function lessEqual(
        int $m,
        int $n,
        Buffer $A, int $offsetA, int $ldA,
        Buffer $X, int $offsetX, int $incX
        ) : void
    {
        if($this->useMath($X)) {
            $this->math->lessEqual($m,$n,$A,$offsetA,$ldA,$X,$offsetX,$incX);
            return;
        }
        if($m<=0 || $n<=0) {
            throw new InvalidArgumentException("m and n must be greater than 0");
        }
        if(($m-1)*$ldA+($n-1)+$offsetA>=count($A)) {
            throw new InvalidArgumentException("Matrix A is too small");
        }
        if(($n-1)*$incX+$offsetX>=count($X)) {
            throw new InvalidArgumentException("Buffer X is too small");
        }

        $lna = $offsetA;
        for($i=0;$i<$m;$i++,$lna+=$ldA) {
            $idx = $offsetX;
            $ida = $lna;
            for ($j=0; $j<$n; $j++,$idx+=$incX,$ida++) {
                // if NaN set 0.0
                // if equal set 0.0
                if($A[$ida] <= $X[$idx]) {
                    $tmp = 1.0;
                } else {
                    $tmp = 0.0;
                }
                $A[$ida] = $tmp;
            }
        }
    }


    /**
     *    A(m,n) := X(n) * A(m,n)
     */
    public function multiply(
        bool $trans,
        int $m,
        int $n,
        Buffer $X, int $offsetX, int $incX,
        Buffer $A, int $offsetA, int $ldA
        ) : void
    {
        if($this->useMath($X)) {
            $this->math->multiply($trans,$m,$n,$X,$offsetX,$incX,$A,$offsetA,$ldA);
            return;
        }

        if(!$trans) {
            $rows = $m; $cols = $n;
        } else {
            $rows = $n; $cols = $m;
        }

        if($offsetX+($cols-1)*$incX>=count($X))
            throw new RuntimeException('Vector specification too large for buffer.');
        if($offsetA+($m-1)*$ldA+($n-1)>=count($A))
            throw new RuntimeException('Vector specification too large for buffer.');

        if(!$trans) { $incAj = $ldA; $incAi = 1;}
        else        { $incAj = 1;    $incAi = $ldA;}

        $idAj = $offsetA;
        for($j=0; $j<$rows; $j++,$idAj+=$incAj) {
            $idA = $idAj;
            $idX = $offsetX;
            for($i=0; $i<$cols; $i++,$idA+=$incAi,$idX+=$incX) {
                $A[$idA] = $X[$idX] * $A[$idA];
            }
        }
    }

    /**
     *     A(m,n) := alpha * X(n) + A(m,n)
     */
    public function add(
        bool $trans,
        int $m,
        int $n,
        float $alpha,
        Buffer $X, int $offsetX, int $incX,
        Buffer $A, int $offsetA, int $ldA
        ) : void
    {
        if($this->useMath($X)) {
            $this->math->add($trans,$m,$n,$alpha,$X,$offsetX,$incX,$A,$offsetA,$ldA);
            return;
        }

        if(!$trans) {
            $rows = $m; $cols = $n;
        } else {
            $rows = $n; $cols = $m;
        }

        if($offsetX+($cols-1)*$incX>=count($X))
            throw new RuntimeException('Vector specification too large for buffer.');
        if($offsetA+($m-1)*$ldA+($n-1)>=count($A))
            throw new RuntimeException('Vector specification too large for buffer.');

        if(!$trans) { $incAj = $ldA; $incAi = 1;}
        else        { $incAj = 1;    $incAi = $ldA;}

        $idAj = $offsetA;
        for($j=0; $j<$rows; $j++,$idAj+=$incAj) {
            $idA = $idAj;
            $idX = $offsetX;
            for($i=0; $i<$cols; $i++,$idA+=$incAi,$idX+=$incX) {
                $A[$idA] = $alpha * $X[$idX] + $A[$idA];
            }
        }
    }

    /**
     *     X := X ^ 2
     */
    public function square(
        int $n,
        Buffer $X, int $offsetX, int $incX
        ) : void
    {
        if($this->useMath($X)) {
            $this->math->square($n,$X,$offsetX,$incX);
            return;
        }

        if($offsetX+($n-1)*$incX>=count($X))
            throw new RuntimeException('Vector specification too large for buffer.');

        $idx = $offsetX;
        for ($i=0; $i<$n; $i++,$idx+=$incX) {
            $t = $X[$idx];
            $X[$idx] = $t * $t;
        }
    }

    /**
     *     X := sqrt(X)
     */
    public function sqrt(
        int $n,
        Buffer $X, int $offsetX, int $incX
        ) : void
    {
        if($this->useMath($X)) {
            $this->math->sqrt($n,$X,$offsetX,$incX);
            return;
        }

        if($offsetX+($n-1)*$incX>=count($X))
            throw new RuntimeException('Vector specification too large for buffer.');

        $idx = $offsetX;
        for ($i=0; $i<$n; $i++,$idx+=$incX) {
            $t = $X[$idx];
            $X[$idx] = sqrt($t);
        }
    }

    /**
     *     X := 1 / (a * sqrt(X) + b)
     */
    public function rsqrt(
        int $n,
        float $alpha,
        Buffer $X, int $offsetX, int $incX,
        float $beta) : void
    {
        if($this->useMath($X)) {
            $this->math->rsqrt($n,$alpha,$X,$offsetX,$incX,$beta);
            return;
        }

        if($offsetX+($n-1)*$incX>=count($X))
            throw new RuntimeException('Vector specification too large for buffer.');

        $idx = $offsetX;
        for ($i=0; $i<$n; $i++,$idx+=$incX) {
            $t = $X[$idx];
            $t = $alpha*sqrt($t) + $beta;
            if($t!=0.0) {
                $r = 1 / $t;
            } else {
                $r = INF;
            }
            $X[$idx] = $r;
        }
    }

    /**
     * A(m,n) := A(m,n) ** X(n)
     * 
     * for openblas v0.2.0
     */
    public function pow(
        bool $trans,
        int $m,
        int $n,
        Buffer $A, int $offsetA, int $ldA,
        Buffer $X, int $offsetX, int $incX
        ) : void
    {
        if($this->useMath($X)) {
            $this->math->pow($trans,$m,$n,$A,$offsetA,$ldA,$X,$offsetX,$incX);
            return;
        }
        if($trans!=false) {
            throw new RuntimeException('trans must be false.');
        }
        if($n!=1) {
            throw new RuntimeException('n size must be 1.');
        }
        if($ldA!=1) {
            throw new RuntimeException('ldA must be 1.');
        }
        if($X->count()!=1) {
            throw new RuntimeException('X size must be 1.');
        }
        // translate from rindow_openblas v0.2.0 to v0.1.x
        $n = $m;
        $alpha = $X[$offsetX];
        $X = $A;
        $offsetX = $offsetA;
        $incX = 1;

        // perform v0.1.x mode
        if($offsetX+($n-1)*$incX>=count($X))
            throw new RuntimeException('Vector specification too large for buffer.');

        $idx = $offsetX;
        for ($i=0; $i<$n; $i++,$idx+=$incX) {
            $X[$idx] = $X[$idx] ** $alpha;
        }
    }

    /**
     *     X(i) := e ^ X(i)
     */
    public function exp(
        int $n,
        Buffer $X, int $offsetX, int $incX
        ) : void
    {
        if($this->useMath($X)) {
            $this->math->exp($n,$X,$offsetX,$incX);
            return;
        }

        if($offsetX+($n-1)*$incX>=count($X))
            throw new RuntimeException('Vector specification too large for buffer.');

        $idx = $offsetX;
        for ($i=0; $i<$n; $i++,$idx+=$incX) {
            $X[$idx] = exp($X[$idx]);
        }
    }

    /**
     *     X := log(X)
     */
    public function log(
        int $n,
        Buffer $X, int $offsetX, int $incX
        ) : void
    {
        if($this->useMath($X)) {
            $this->math->log($n,$X,$offsetX,$incX);
            return;
        }

        if($offsetX+($n-1)*$incX>=count($X))
            throw new RuntimeException('Vector specification too large for buffer.');

        $idx = $offsetX;
        for ($i=0; $i<$n; $i++,$idx+=$incX) {
            $t = $X[$idx];
            $X[$idx] = log($t);
        }
    }

    /**
     *     X := tanh(X)
     */
    public function tanh(
        int $n,
        Buffer $X, int $offsetX, int $incX
        ) : void
    {
        if($this->useMath($X)) {
            $this->math->tanh($n,$X,$offsetX,$incX);
            return;
        }
        if($offsetX+($n-1)*$incX>=count($X))
            throw new RuntimeException('Vector specification too large for buffer.');

        $idx = $offsetX;
        for ($i=0; $i<$n; $i++,$idx+=$incX) {
            $t = $X[$idx];
            $X[$idx] = tanh($t);
        }
    }

    /**
     *     X := sin(X)
     */
    public function sin(
        int $n,
        Buffer $X, int $offsetX, int $incX
        ) : void
    {
        if($this->useMath($X)) {
            $this->math->sin($n,$X,$offsetX,$incX);
            return;
        }
        if($offsetX+($n-1)*$incX>=count($X))
            throw new RuntimeException('Vector specification too large for buffer.');

        $idx = $offsetX;
        for ($i=0; $i<$n; $i++,$idx+=$incX) {
            $t = $X[$idx];
            $X[$idx] = sin($t);
        }
    }

    /**
     *     X := cos(X)
     */
    public function cos(
        int $n,
        Buffer $X, int $offsetX, int $incX
        ) : void
    {
        if($this->useMath($X)) {
            $this->math->cos($n,$X,$offsetX,$incX);
            return;
        }
        if($offsetX+($n-1)*$incX>=count($X))
            throw new RuntimeException('Vector specification too large for buffer.');

        $idx = $offsetX;
        for ($i=0; $i<$n; $i++,$idx+=$incX) {
            $t = $X[$idx];
            $X[$idx] = cos($t);
        }
    }

    /**
     *     X := tan(X)
     */
    public function tan(
        int $n,
        Buffer $X, int $offsetX, int $incX
        ) : void
    {
        if($this->useMath($X)) {
            $this->math->tan($n,$X,$offsetX,$incX);
            return;
        }
        if($offsetX+($n-1)*$incX>=count($X))
            throw new RuntimeException('Vector specification too large for buffer.');

        $idx = $offsetX;
        for ($i=0; $i<$n; $i++,$idx+=$incX) {
            $t = $X[$idx];
            $X[$idx] = tan($t);
        }
    }

    /**
     *     A(m,n) := X(n)
     */
    public function duplicate(
        bool $trans,
        int $m,
        int $n,
        Buffer $X, int $offsetX, int $incX,
        Buffer $A, int $offsetA, int $ldA
        ) : void
    {
        if($this->useMath($X)) {
            $this->math->duplicate($trans,$m,$n,$X,$offsetX,$incX,$A,$offsetA,$ldA);
            return;
        }

        if(!$trans) {
            $rows = $m; $cols = $n;
        } else {
            $rows = $n; $cols = $m;
        }

        if($offsetX+($cols-1)*$incX>=count($X))
            throw new RuntimeException('Vector specification too large for buffer.');
        if($offsetA+($m-1)*$ldA+($n-1)>=count($A))
            throw new RuntimeException('Vector specification too large for buffer.');

        if(!$trans) { $incAj = $ldA; $incAi = 1;}
        else        { $incAj = 1;    $incAi = $ldA;}

        $idA = $offsetA;
        for($j=0; $j<$rows; $j++) {
            $this->duplicate_blas_copy($cols,$X,$offsetX,$incX,$A,$idA,$incAi);
            $idA += $incAj;
        }
    }
    protected function duplicate_blas_copy($n,$X,$offsetX,$incX,$Y,$offsetY,$incY)
    {
        $idX = $offsetX;
        $idY = $offsetY;
        for($i=0; $i<$n; $i++,$idX+=$incX,$idY+=$incY) {
            $Y[$idY] = $X[$idX];
        }
    }

    /**
     *     X := 0
     */
    public function zeros(
        int $n,
        Buffer $X, int $offsetX, int $incX
        ) : void
    {
        if($this->useMath($X)) {
            $this->math->zeros($n,$X,$offsetX,$incX);
            return;
        }

        if($offsetX+($n-1)*$incX>=count($X))
            throw new RuntimeException('Vector specification too large for buffer.');

        $idx = $offsetX;
        for ($i=0; $i<$n; $i++,$idx+=$incX) {
            $X[$idx] = 0;
        }
    }

    /**
    *      B(n,k) := A(X(n),k)
    */
    public function gather(
        bool $reverse,
        bool $addMode,
        int $n,
        int $k,
        int $numClass,
        Buffer $X, int $offsetX,
        Buffer $A, int $offsetA,
        Buffer $B, int $offsetB
        ) : void
    {
        //if($reverse==true && $addMode==true) {
        //    $this->scatterAdd($n,$k,$numClass,$X,$offsetX,$A,$offsetA,$B,$offsetB);
        //    return;
        //}
        if($this->useMath($A)) {
            $this->math->gather(
                $reverse,
                $addMode,
                $n,
                $k,
                $numClass,
                $X, $offsetX,
                $A, $offsetA,
                $B, $offsetB
            );
            return;
        }
//echo "[n=$n,k=$k,class=$numClass]\n";

        if($offsetX+$n>count($X))
            throw new RuntimeException('Matrix X specification too large for buffer.');
        if($offsetA+$numClass*$k>count($A))
            throw new RuntimeException('Matrix A specification too large for buffer.');
        if($offsetB+$n*$k>count($B))
            throw new RuntimeException('Matrix B specification too large for buffer.');

        $idxX = $offsetX;
        $idxA = $offsetA;
        $idxB = $offsetB;
        $ldIndex = $k;
        for($j=0;$j<$n;$j++) {
            $index = $X[$idxX+$j];
            if($index>=$numClass) {
                throw new RuntimeException("index is out of range.:".$index);
            }
            $iA = $idxA+$index*$ldIndex;
            $iB = $idxB+$j*$k;
            if($reverse) {
                if($addMode) {
                    $this->math_add($k,$B,$iB,1,$A,$iA,1);
                } else {
                    $this->math_copy($k,$B,$iB,1,$A,$iA,1);
                }
            } else {
                if($addMode) {
                    $this->math_add($k,$A,$iA,1,$B,$iB,1);
                } else {
                    $this->math_copy($k,$A,$iA,1,$B,$iB,1);
                }
            }
        }
    }

    /**
    *      B(m,n) := A(m,X(m,n))
    */
    public function reduceGather(
        bool $reverse,
        bool $addMode,
        int $m,
        int $n,
        int $numClass,
        Buffer $X, int $offsetX,
        Buffer $A, int $offsetA,
        Buffer $B, int $offsetB
        ) : void
    {
        if($this->useMath($A)) {
            $this->math->reduceGather(
                $reverse,
                $addMode,
                $m,
                $n,
                $numClass,
                $X, $offsetX,
                $A, $offsetA,
                $B, $offsetB
            );
            return;
        }
//echo "[m=$m,n=$n,class=$numClass]\n";
        if($offsetX+$n>count($X))
            throw new RuntimeException('Matrix X specification too large for buffer.');
        if($offsetA+$m*$numClass>count($A))
            throw new RuntimeException('Matrix A specification too large for buffer.');
        if($offsetB+$m*$n>count($B))
            throw new RuntimeException('Matrix B specification too large for buffer.');

        $idxX = $offsetX;
        $idxA = $offsetA;
        $idxB = $offsetB;
        $ldX = $n;
        $ldA = $n*$numClass;
        $ldB = $n;
        $ldIndex = $n;
        for($i=0; $i<$m; $i++,$idxX+=$ldX,$idxA+=$ldA,$idxB+=$ldB) {
            for($j=0;$j<$n;$j++) {
                $index = $X[$idxX+$j];
                if($index>=$numClass) {
                    throw new RuntimeException("index is out of range.:".$index);
                }
                $iA = $idxA+$j+$index*$ldIndex;
                $iB = $idxB+$j;
                if($reverse) {
                    if($addMode) {
                        $A[$iA] = $A[$iA] + $B[$iB];
                    } else {
                        $A[$iA] = $B[$iB];
                    }
                } else {
                    if($addMode) {
                        $B[$iB] = $B[$iB] + $A[$iA];
                    } else {
                        $B[$iB] = $A[$iA];
                    }
                }
            }
        }
    }

    /**
    *  For Parallel Computing Algorithm
    */
    protected function scatterAdd(
        int $n,
        int $k,
        int $numClass,
        Buffer $X, int $offsetX,
        Buffer $A, int $offsetA,
        Buffer $B, int $offsetB
        ) : void
    {
        //if($this->useMath($A)) {
        //    $this->math->gather(
        //        $reverse=true,
        //        $addMode=true,
        //        $n,
        //        $k,
        //        $numClass,
        //        $X, $offsetX,
        //        $A, $offsetA,
        //        $B, $offsetB
        //    );
        //    return;
        //}

        if($offsetX+$n>count($X))
            throw new RuntimeException('Matrix X specification too large for buffer.');
        if($offsetA+$numClass*$k>count($A))
            throw new RuntimeException('Matrix A specification too large for buffer.');
        if($offsetB+$n*$k>count($B))
            throw new RuntimeException('Matrix B specification too large for buffer.');

        for($i=0;$i<$numClass;$i++) {
            for($p=0;$p<$k;$p++) {
                $sum = 0;
                for($j=0;$j<$n;$j++) {
                    $index = $X[$offsetX+$j];
                    if($index==$i) {
                        $iB = $offsetB+$p+$j*$k;
                        $sum += $B[$iB];
                    }
                }
                $iA = $offsetA+$i*$k+$p;
                $A[$iA] = $A[$iA] + $sum;
            }
        }
    }

    /**
    *  B(n,repeats,k) := A(n,k)
    */
    public function repeat(
        int $m,
        int $k,
        int $repeats,
        Buffer $A, int $offsetA,
        Buffer $B, int $offsetB
        ) : void
    {
        if($this->useMath($A)) {
            $this->math->repeat(
                $m,
                $k,
                $repeats,
                $A, $offsetA,
                $B, $offsetB
            );
            return;
        }
//echo "[m=$m,n=$n,class=$numClass]\n";
        if($offsetA+$m*$k>count($A))
            throw new RuntimeException('Matrix A specification too large for buffer.');
        if($offsetB+$m*$repeats*$k>count($B))
            throw new RuntimeException('Matrix B specification too large for buffer.');

        $idxA = $offsetA;
        $idxB = $offsetB;
        $ldA = $k;
        $ldB = $k;
        for($i=0; $i<$m; $i++,$idxA+=$ldA) {
            for($j=0;$j<$repeats;$j++,$idxB+=$ldB) {
                $this->math_copy($k,$A,$idxA,1,$B,$idxB,1);
            }
        }
    }

    /**
     *     Y := updateAddOnehot(X,a)
     */
    public function updateAddOnehot(
        int $m,
        int $n,
        float $a,
        Buffer $X, int $offsetX, int $incX,
        Buffer $Y, int $offsetY, int $ldY
        ) : void
    {
        if($this->useMath($Y)) {
            $this->math->updateAddOnehot($m,$n,$a,$X,$offsetX,$incX,$Y,$offsetY,$ldY);
            return;
        }

        if($offsetX+($m-1)*$incX>=count($X))
            throw new RuntimeException('Vector specification too large for bufferX.');
        if($offsetY+($m-1)*$ldY+($n-1)>=count($Y))
            throw new RuntimeException('Vector specification too large for bufferY.');

        $idx = $offsetX;
        $idy = $offsetY;
        for ($i=0; $i<$m; $i++,$idy+=$ldY,$idx+=$incX) {
            $label = (int)$X[$idx];
            if($label>=$n||$label<0)
                throw new RuntimeException('Label number is out of bounds.');
            $Y[$idy+$label] = $Y[$idy+$label] + $a;
        }
    }

    /**
     * Y(i) := 1  ( X(i) == Y(i) )
     * Y(i) := 0  ( X(i) != Y(i) )
     */
    public function equal(
        int $n,
        Buffer $X, int $offsetX, int $incX,
        Buffer $Y, int $offsetY, int $incY
        ) : void
    {
        if($this->math) { // Support all dtype by math
            $this->math->equal($n,$X,$offsetX,$incX,$Y,$offsetY,$incY);
            return;
        }

        if($offsetX+($n-1)*$incX>=count($X))
            throw new RuntimeException('Vector specification too large for buffer.');
        if($offsetY+($n-1)*$incY>=count($Y))
            throw new RuntimeException('Vector specification too large for buffer.');

        $idX = $offsetX;
        $idY = $offsetY;
        if(is_bool($Y[$idY])) {$true = true; $false = false;}
        else                   {$true = 1; $false = 0;}
        for($i=0; $i<$n; $i++,$idX+=$incX,$idY+=$incY) {
            $Y[$idY] = ($Y[$idY] == $X[$idX]) ? $true : $false;
        }
    }

    /**
    *   X(m) := sum( A(m,n) )
    */
    public function reduceSum(
        int $m,
        int $n,
        int $k,
        Buffer $A, int $offsetA,
        Buffer $B, int $offsetB
        ) : void
    {
        if($this->useMath($A)) {
            $this->math->reduceSum($m,$n,$k,$A,$offsetA,$B,$offsetB);
            return;
        }

        if($offsetA+$m*$n*$k>count($A))
            throw new RuntimeException('Matrix A specification too large for buffer.');
        if($offsetB+$m*$k>count($B))
            throw new RuntimeException('Matrix B specification too large for buffer.');

        $idxA = $offsetA;
        $idxB = $offsetB;
        $ldA = $n*$k;
        $ldB = $k;
        for($i=0; $i<$m; $i++,$idxA+=$ldA,$idxB+=$ldB) {
            for($j=0; $j<$k; $j++) {
                $B[$idxB+$j] = $this->math_sum($n, $A, $idxA+$j, $k);
            }
        }
    }

    /**
     * X(m) := max( A(m,n) )
     */
    public function reduceMax(
        int $m,
        int $n,
        int $k,
        Buffer $A, int $offsetA,
        Buffer $B, int $offsetB
        ) : void
    {
        if($this->useMath($A)) {
            $this->math->reduceMax($m,$n,$k,$A,$offsetA,$B,$offsetB);
            return;
        }

        if($offsetA+$m*$n*$k>count($A))
            throw new RuntimeException('Matrix A specification too large for buffer.');
        if($offsetB+$m*$k>count($B))
            throw new RuntimeException('Matrix B specification too large for buffer.');

        $idxA = $offsetA;
        $idxB = $offsetB;
        $ldA = $n*$k;
        $ldB = $k;
        for($i=0; $i<$m; $i++,$idxA+=$ldA,$idxB+=$ldB) {
            for($j=0; $j<$k; $j++) {
                $B[$idxB+$j] = $this->math_max($n, $A, $idxA+$j, $k);
            }
        }
    }

    /**
     * X(m) := max( A(m,n) )
     */
    public function reduceArgMax(
        int $m,
        int $n,
        int $k,
        Buffer $A, int $offsetA,
        Buffer $B, int $offsetB
        ) : void
    {
        if($this->useMath($A)) {
            $this->math->reduceArgMax($m,$n,$k,$A,$offsetA,$B,$offsetB);
            return;
        }

        if($offsetA+$m*$n*$k>count($A))
            throw new RuntimeException('Matrix A specification too large for buffer.');
        if($offsetB+$m*$k>count($B))
            throw new RuntimeException('Matrix B specification too large for buffer.');

        $idxA = $offsetA;
        $idxB = $offsetB;
        $ldA = $n*$k;
        $ldB = $k;
        for($i=0; $i<$m; $i++,$idxA+=$ldA,$idxB+=$ldB) {
            for($j=0; $j<$k; $j++) {
                $B[$idxB+$j] = $this->math_imax($n, $A, $idxA+$j, $k);
            }
        }
    }

    public function softmax(
        int $m,
        int $n,
        Buffer $A, int $offsetA, int $ldA) : void
    {
        if($offsetA+($m-1)*$ldA+($n-1)>=count($A))
            throw new RuntimeException('Vector specification too large for buffer.');

        $idA = $offsetA;
        for($i=0;$i<$m;$i++,$idA+=$ldA) {
            //float t,max_a,sum_exp;
            $max_a = $this->math_max($n,$A,$idA,1);
            $sum_exp = 0;
            for($j=0;$j<$n;$j++) {
                $t = exp($A[$idA+$j]-$max_a);
                $sum_exp += $t;
                $A[$idA+$j] = $t;
            }
            if($sum_exp==0.0) {
                throw new RuntimeException("Zero divide in softmax.");
            }
            for($j=0;$j<$n;$j++) {
                $A[$idA+$j] = $A[$idA+$j] / $sum_exp;
            }
        }
    }

    public function astype(
        int $n,
        int $dtype,
        Buffer $X, int $offsetX, int $incX,
        Buffer $Y, int $offsetY, int $incY
        ) : void
    {
        if($this->math) {
            $this->math->astype($n,$dtype,$X,$offsetX,$incX,$Y,$offsetY,$incY);
            return;
        }

        if(in_array($dtype,$this->floatTypes)) {
            $isFloat = true;
        } elseif(in_array($dtype,$this->intTypes)) {
            $isFloat = false;
            if($dtype==NDArray::uint8) {
                $mask = 0xff;
            } elseif($dtype==NDArray::uint16) {
                $mask = 0xffff;
            } elseif($dtype==NDArray::uint32) {
                $mask = 0xffffffff;
            } else {
                $mask = null;
            }
        } elseif($dtype==NDArray::bool) {
            $isFloat = false;
        } else {
            throw new InvalidArgumentException('dtype must be type of integer or float: '.$dtype);
        }
        if(is_bool($X[0])) {
            $fromBoolean = true;
        } else {
            $fromBoolean = false;
        }

        $idx = $offsetX;
        $idy = $offsetY;
        if($fromBoolean) {
            if($isFloat) {
                for($i=0; $i<$n; $i++,$idx+=$incX,$idy+=$incY) {
                    $Y[$idy] = ($X[$idx]) ? 1.0 : 0.0;
                }
            } else {
                for($i=0; $i<$n; $i++,$idx+=$incX,$idy+=$incY) {
                    $Y[$idy] = ($X[$idx]) ? 1 : 0;
                }
            }
        } else {
            if($dtype==NDArray::bool) {
                for($i=0; $i<$n; $i++,$idx+=$incX,$idy+=$incY) {
                    $Y[$idy] = ($X[$idx]) ? true : false;
                }
            } elseif($isFloat) {
                for($i=0; $i<$n; $i++,$idx+=$incX,$idy+=$incY) {
                    $Y[$idy] = (float)($X[$idx]);
                }
            } else {
                if($mask) {
                    for($i=0; $i<$n; $i++,$idx+=$incX,$idy+=$incY) {
                        $Y[$idy] = (int)($mask & $X[$idx]);
                    }
                } else {
                    for($i=0; $i<$n; $i++,$idx+=$incX,$idy+=$incY) {
                        $Y[$idy] = (int)($X[$idx]);
                    }
                }
            }
        }
    }

    public function searchsorted(
        int $m,
        int $n,
        Buffer $A, int $offsetA, int $ldA,  // float
        Buffer $X, int $offsetX, int $incX, // float
        bool $right,
        Buffer $Y, int $offsetY, int $incY // int
        ) : void
    {
        if($this->math) {
            // for v0.2.0
            $this->math->searchsorted($m,$n,$A,$offsetA,$ldA,$X,$offsetX,$incX,$right,$Y,$offsetY,$incY);
            return;
        }
        // translate from rindow_openblas v0.2.0 to v0.1.x
        if($ldA!=0) {
            throw new RuntimeException('ldA must be 0.');
        }
        $incA = 1;
        [$n,$m] = [$m,$n];

        // perform v0.1.x mode
        $idx = $offsetX;
        $idy = $offsetY;
        for($i=0;$i<$n;$i++) {
            $v = $X[$idx];
            $ida = $offsetA;
            if($right) {
                for($j=0;$j<$m;$j++) {
                    if(!($v>=$A[$ida])) {
                        break;
                    }
                    $ida += $incA;
                }
            } else {
                for($j=0;$j<$m;$j++) {
                    if(!($v>$A[$ida])) {
                        break;
                    }
                    $ida += $incA;
                }
            }
            $Y[$idy] = $j;
            $idx+=$incX;
            $idy+=$incY;
        }
    }

    public function cumsum(
        int $n,
        Buffer $X, int $offsetX, int $incX, // float
        bool $exclusive,
        bool $reverse,
        Buffer $Y, int $offsetY, int $incY // int
        ) : void
    {
        if($this->math) {
            $this->math->cumsum($n,$X,$offsetX,$incX,$exclusive,$reverse,$Y,$offsetY,$incY);
            return;
        }

        if($reverse) {
            $idx = $offsetX;
            $idy = $offsetY+$incY*($n-1);
            $incY = -$incY;
        } else {
            $idx = $offsetX;
            $idy = $offsetY;
        }
        $value = 0.0;
        if($exclusive) {
            for($i=0;$i<$n;$i++) {
                $Y[$idy] = $value;
                $value += $X[$idx];
                $idx+=$incX;
                $idy+=$incY;
            }
        } else {
            for($i=0;$i<$n;$i++) {
                $value += $X[$idx];
                $Y[$idy] = $value;
                $idx+=$incX;
                $idy+=$incY;
            }
        }
    }

    /**
     * copy a image with channels
     */
    protected function copyCell1d(
        bool $reverse,
        Buffer $images,
        int $images_pos,
        int $im_w,
        int $channels,
        int $channel_step,
        int $filter_w_step,
        int $vim_x,
        int $vfilter_w,
        int $dilation_w,
        Buffer $out,
        int $out_pos,
        int $out_filter_step,
        int $out_channel_step
        )
    {
        #print('v=%d,%d,%d,%d' % (vin_y,vin_x,vin_h,vin_w))
        $filter_w_pos = $images_pos;
        $out_filter_pos = $out_pos;
        for($vfilter_x=0; $vfilter_x<$vfilter_w; $vfilter_x+=$dilation_w) {
            $channel_pos = $filter_w_pos;
            $out_channel_pos = $out_filter_pos;
            #print('yx=%d,%d' % (yy,xx))
            $input_x = $vim_x+$vfilter_x;
            for($c=0; $c<$channels; $c++) {
                if($input_x<0 || $input_x>=$im_w) {
                    #print('pad')
                    if(!$reverse) {
                        $out[$out_channel_pos] = 0;
                    }
                } else {
                    if(!$reverse) {
                        $out[$out_channel_pos] =  $images[$channel_pos];
                    } else {
                        // Sum for Back propagation
                         $images[$channel_pos] += $out[$out_channel_pos];
                    }
                }
                $out_channel_pos += $out_channel_step;
                $channel_pos += $channel_step;
            }
            $out_filter_pos += $out_filter_step;
            $filter_w_pos += $filter_w_step;
        }
    }

    /**
    * images: (n,h,w,c) : channels_last
    *        (n,c,h,w) : channels_first
    * strides:
    * padding:
    * data_format:
    * output:(n,i)
    */
    public function im2col1d(
        bool $reverse,
        Buffer $images,
        int $images_offset,
        int $images_size,
        int $batches,
        int $im_w,
        int $channels,
        int $filter_w,
        int $stride_w,
        bool $padding,
        bool $channels_first,
        int $dilation_w,
        bool $cols_channels_first,
        Buffer $cols,
        int $cols_offset,
        int $cols_size
        )
    {
        if($this->math) {
            $this->math->im2col1d(
                $reverse,
                $images,
                $images_offset,
                $images_size,
                $batches,
                $im_w,
                $channels,
                $filter_w,
                $stride_w,
                $padding,
                $channels_first,
                $dilation_w,
                $cols_channels_first,
                $cols,
                $cols_offset,
                $cols_size
            );
            return;
        }
        $images_buf_size = $batches*$im_w*$channels;
        if($images_size!=$images_buf_size ||
            count($images)-$images_offset<$images_buf_size) {
            throw new InvalidArgumentException('images buffer size is invalid');
        }
        //$out_w = (int)floor(($im_w-$filter_w)/$stride_w)+1;
        $out_w = intval(floor(($im_w-($filter_w-1)*$dilation_w-1)/$stride_w)+1);
        if($out_w<=0) {
            throw new InvalidArgumentException('Invalid shape or parameters.');
        }
        if($padding) {
            $out_buf_size =
                $batches*
                $im_w*$filter_w*
                $channels;
            #print('outsz=',out.shape)
            $padding_w = (int)floor((($im_w-1)*$stride_w-$im_w+($filter_w-1)*$dilation_w+1)/2);
            $out_w = $im_w;
        } else {
            $out_buf_size = $batches*
                $out_w*$filter_w*
                $channels;
            $padding_w = 0;
        }
        if($cols_size!=$out_buf_size ||
            count($cols)-$cols_offset>$out_buf_size) {
            throw new InvalidArgumentException('output buffer size is invalid');
        }
        if($channels_first) {
            $im_w_step = 1;
            $channel_step = $im_w;
            $batch_step =   $im_w*$channels;
        } else {
            $channel_step = 1;
            $im_w_step =  $channels;
            $batch_step = $channels*$im_w;
        }
        $stride_w_step = $im_w_step*$stride_w;
        $filter_w_step = $im_w_step*$dilation_w;

        if($cols_channels_first) {
            $out_filter_step = 1;
            $out_channel_step = $filter_w;
        } else {
            $out_filter_step = $channels;
            $out_channel_step = 1;
        }
        $out_cell_step = $filter_w*$channels;

        $batch_pos = $images_offset-$im_w_step*$padding_w;
        $out_pos = $cols_offset;

        $vim_w = $out_w*$stride_w;
        $vfilter_w = $filter_w*$dilation_w;
        for($batch=0; $batch<$batches;$batch++) {
            $stride_w_pos = $batch_pos;
            for($vim_x=0;$vim_x<$vim_w;$vim_x+=$stride_w) {//stride_x
                #print('osf=%d,%d,%d'%(out_h,stride_h,filter_h))
                $this->copyCell1d(
                    $reverse,
                    $images,
                    $stride_w_pos,
                    $im_w,
                    $channels,
                    $channel_step,
                    $filter_w_step,
                    $vim_x-$padding_w,
                    $vfilter_w,
                    $dilation_w,
                    $cols,
                    $out_pos,
                    $out_filter_step,
                    $out_channel_step
                );
                $stride_w_pos += $stride_w_step;
                $out_pos += $out_cell_step;
            }
            $batch_pos += $batch_step;
        }
    }

    /**
     * copy a image with channels
     */
    protected function copyCell2d(
        bool $reverse,
        Buffer $images,
        int $images_pos,
        int $im_h,
        int $im_w,
        int $channels,
        int $channel_step,
        int $filter_h_step,
        int $filter_w_step,
        int $vim_y,
        int $vim_x,
        int $vfilter_h,
        int $vfilter_w,
        int $dilation_h,
        int $dilation_w,
        Buffer $out,
        int $out_pos,
        int $out_filter_step,
        int $out_channel_step
        )
    {
        #print('v=%d,%d,%d,%d' % (vin_y,vin_x,vin_h,vin_w))
        $filter_h_pos = $images_pos;
        $out_filter_pos = $out_pos;
        for($vfilter_y=0; $vfilter_y<$vfilter_h; $vfilter_y+=$dilation_h) {
            $filter_w_pos = $filter_h_pos;
            for($vfilter_x=0; $vfilter_x<$vfilter_w; $vfilter_x+=$dilation_w) {
                $channel_pos = $filter_w_pos;
                $out_channel_pos = $out_filter_pos;
                #print('yx=%d,%d' % (yy,xx))
                $input_y = $vim_y+$vfilter_y;
                $input_x = $vim_x+$vfilter_x;
                for($c=0; $c<$channels; $c++) {
                    if($input_y<0 || $input_y>=$im_h ||
                       $input_x<0 || $input_x>=$im_w) {
                        #print('pad')
                        if(!$reverse) {
                            $out[$out_channel_pos] = 0;
                        }
                    } else {
                        if(!$reverse) {
                            $out[$out_channel_pos] =  $images[$channel_pos];
                        } else {
                            // Sum for Back propagation
                             $images[$channel_pos] += $out[$out_channel_pos];
                        }
                    }
                    $out_channel_pos += $out_channel_step;
                    $channel_pos += $channel_step;
                }
                $out_filter_pos += $out_filter_step;
                $filter_w_pos += $filter_w_step;
            }
            $filter_h_pos += $filter_h_step;
        }

    }

    /**
    * images: (n,h,w,c) : channels_last
    *        (n,c,h,w) : channels_first
    * strides:
    * padding:
    * data_format:
    * output:(n,i)
    */
    public function im2col2d(
        bool $reverse,
        Buffer $images,
        int $images_offset,
        int $images_size,
        int $batches,

        int $im_h,
        int $im_w,
        int $channels,
        int $filter_h,
        int $filter_w,

        int $stride_h,
        int $stride_w,
        bool $padding,
        bool $channels_first,
        int $dilation_h,

        int $dilation_w,
        bool $cols_channels_first,
        Buffer $cols,
        int $cols_offset,
        int $cols_size
        )
    {
        if($this->math) {
            $this->math->im2col2d(
                $reverse,
                $images,
                $images_offset,
                $images_size,
                $batches,

                $im_h,
                $im_w,
                $channels,
                $filter_h,
                $filter_w,

                $stride_h,
                $stride_w,
                $padding,
                $channels_first,
                $dilation_h,

                $dilation_w,
                $cols_channels_first,
                $cols,
                $cols_offset,
                $cols_size
            );
            return;
        }
        $images_buf_size = $batches*$im_h*$im_w*$channels;
        if($images_size!=$images_buf_size ||
            count($images)-$images_offset<$images_buf_size) {
            throw new InvalidArgumentException('images buffer size is invalid');
        }
        $out_h = intval(floor(($im_h-($filter_h-1)*$dilation_h-1)/$stride_h)+1);
        $out_w = intval(floor(($im_w-($filter_w-1)*$dilation_w-1)/$stride_w)+1);
        if($out_h<=0 || $out_w<=0) {
            throw new InvalidArgumentException('Invalid shape or parameters.');
        }
        if($padding) {
            $out_buf_size =
                $batches*
                $im_h*$filter_h*
                $im_w*$filter_w*
                $channels;
            #print('outsz=',out.shape)
            #print('start-end=(%d,%d)-(%d,%d)'%(start_h,start_w,end_h,end_w))
            $padding_h = (int)floor((($im_h-1)*$stride_h-$im_h+($filter_h-1)*$dilation_h+1)/2);
            $padding_w = (int)floor((($im_w-1)*$stride_w-$im_w+($filter_w-1)*$dilation_w+1)/2);
            $out_h = $im_h;
            $out_w = $im_w;
        } else {
            $out_buf_size = $batches*
                $out_h*$filter_h*
                $out_w*$filter_w*
                $channels;
            $padding_h = 0;
            $padding_w = 0;
        }
        if($cols_size!=$out_buf_size ||
            count($cols)-$cols_offset>$out_buf_size) {
            throw new InvalidArgumentException('output buffer size is invalid');
        }
        if($channels_first) {
            $im_w_step = 1;
            $im_h_step =    $im_w;
            $channel_step = $im_w*$im_h;
            $batch_step =   $im_w*$im_h*$channels;
        } else {
            $channel_step = 1;
            $im_w_step =  $channels;
            $im_h_step =  $channels*$im_w;
            $batch_step = $channels*$im_w*$im_h;
        }
        $stride_w_step = $im_w_step*$stride_w;
        $stride_h_step = $im_h_step*$stride_h;
        $filter_w_step = $im_w_step*$dilation_w;
        $filter_h_step = $im_h_step*$dilation_h;

        if($cols_channels_first) {
            $out_filter_step = 1;
            $out_channel_step = $filter_h*$filter_w;
        } else {
            $out_filter_step = $channels;
            $out_channel_step = 1;
        }
        $out_cell_step = $filter_h*$filter_w*$channels;

        $batch_pos = $images_offset-$im_h_step*$padding_h-$im_w_step*$padding_w;
        $out_pos = $cols_offset;

        $vim_h = $out_h*$stride_h;
        $vim_w = $out_w*$stride_w;
        $vfilter_h = $filter_h*$dilation_h;
        $vfilter_w = $filter_w*$dilation_w;
        for($batch=0; $batch<$batches;$batch++) {
            $stride_h_pos = $batch_pos;
            for($vim_y=0;$vim_y<$vim_h;$vim_y+=$stride_h){//stride_y
                $stride_w_pos = $stride_h_pos;
                for($vim_x=0;$vim_x<$vim_w;$vim_x+=$stride_w) {//stride_x
                    #print('osf=%d,%d,%d'%(out_h,stride_h,filter_h))
                    $this->copyCell2d(
                        $reverse,
                        $images,
                        $stride_w_pos,
                        $im_h,
                        $im_w,
                        $channels,
                        $channel_step,
                        $filter_h_step,
                        $filter_w_step,
                        $vim_y-$padding_h,
                        $vim_x-$padding_w,
                        $vfilter_h,
                        $vfilter_w,
                        $dilation_h,
                        $dilation_w,
                        $cols,
                        $out_pos,
                        $out_filter_step,
                        $out_channel_step
                    );
                    $stride_w_pos += $stride_w_step;
                    //$vim_x += $stride_w;
                    $out_pos += $out_cell_step;
                }
                $stride_h_pos += $stride_h_step;
                //$vim_y += $stride_h;
            }
            $batch_pos += $batch_step;
        }
    }

    /**
     * copy a image with channels
     */
    protected function copyCell3d(
        bool $reverse,
        Buffer $images,
        int $images_pos,
        int $im_d,
        int $im_h,
        int $im_w,
        int $channels,
        int $channel_step,
        int $filter_d_step,
        int $filter_h_step,
        int $filter_w_step,
        int $vim_z,
        int $vim_y,
        int $vim_x,
        int $vfilter_d,
        int $vfilter_h,
        int $vfilter_w,
        int $dilation_d,
        int $dilation_h,
        int $dilation_w,
        Buffer $out,
        int $out_pos,
        int $out_filter_step,
        int $out_channel_step
        )
    {
        #print('v=%d,%d,%d,%d' % (vin_y,vin_x,vin_h,vin_w))
        $filter_d_pos = $images_pos;
        $out_filter_pos = $out_pos;
        for($vfilter_z=0; $vfilter_z<$vfilter_d; $vfilter_z+=$dilation_d) {
            $filter_h_pos = $filter_d_pos;
            for($vfilter_y=0; $vfilter_y<$vfilter_h; $vfilter_y+=$dilation_h) {
                $filter_w_pos = $filter_h_pos;
                for($vfilter_x=0; $vfilter_x<$vfilter_w; $vfilter_x+=$dilation_w) {
                    $channel_pos = $filter_w_pos;
                    $out_channel_pos = $out_filter_pos;
                    #print('yx=%d,%d' % (yy,xx))
                    $input_z = $vim_z+$vfilter_z;
                    $input_y = $vim_y+$vfilter_y;
                    $input_x = $vim_x+$vfilter_x;
                    for($c=0; $c<$channels; $c++) {
                        if($input_z<0 || $input_z>=$im_d ||
                           $input_y<0 || $input_y>=$im_h ||
                           $input_x<0 || $input_x>=$im_w) {
                            #print('pad')
                            if(!$reverse) {
                                $out[$out_channel_pos] = 0;
                            }
                        } else {
                            if(!$reverse) {
                                $out[$out_channel_pos] =  $images[$channel_pos];
                            } else {
                                // Sum for Back propagation
                                 $images[$channel_pos] += $out[$out_channel_pos];
                            }
                        }
                        $out_channel_pos += $out_channel_step;
                        $channel_pos += $channel_step;
                    }
                    $out_filter_pos += $out_filter_step;
                    $filter_w_pos += $filter_w_step;
                }
                $filter_h_pos += $filter_h_step;
            }
            $filter_d_pos += $filter_d_step;
        }
    }

    /**
    * images: (n,h,w,c) : channels_last
    *        (n,c,h,w) : channels_first
    * strides:
    * padding:
    * data_format:
    * output:(n,i)
    */
    public function im2col3d(
        bool $reverse,
        Buffer $images,
        int $images_offset,
        int $images_size,
        int $batches,
        int $im_d,
        int $im_h,
        int $im_w,
        int $channels,
        int $filter_d,
        int $filter_h,
        int $filter_w,
        int $stride_d,
        int $stride_h,
        int $stride_w,
        bool $padding,
        bool $channels_first,
        int $dilation_d,
        int $dilation_h,
        int $dilation_w,
        bool $cols_channels_first,
        Buffer $cols,
        int $cols_offset,
        int $cols_size
        )
    {
        if($this->math) {
            $this->math->im2col3d(
                $reverse,
                $images,
                $images_offset,
                $images_size,
                $batches,
                $im_d,
                $im_h,
                $im_w,
                $channels,
                $filter_d,
                $filter_h,
                $filter_w,
                $stride_d,
                $stride_h,
                $stride_w,
                $padding,
                $channels_first,
                $dilation_d,
                $dilation_h,
                $dilation_w,
                $cols_channels_first,
                $cols,
                $cols_offset,
                $cols_size
            );
            return;
        }
        $images_buf_size = $batches*$im_d*$im_h*$im_w*$channels;
        if($images_size!=$images_buf_size ||
            count($images)-$images_offset<$images_buf_size) {
            throw new InvalidArgumentException('images buffer size is invalid');
        }
        $out_d = intval(floor(($im_d-($filter_d-1)*$dilation_d-1)/$stride_d)+1);
        $out_h = intval(floor(($im_h-($filter_h-1)*$dilation_h-1)/$stride_h)+1);
        $out_w = intval(floor(($im_w-($filter_w-1)*$dilation_w-1)/$stride_w)+1);
        if($out_h<=0 || $out_w<=0) {
            throw new InvalidArgumentException('Invalid shape or parameters.');
        }
        if($padding) {
            $out_buf_size =
                $batches*
                $im_d*$filter_d*
                $im_h*$filter_h*
                $im_w*$filter_w*
                $channels;
            #print('outsz=',out.shape)
            $padding_d = (int)floor((($im_d-1)*$stride_d-$im_h+($filter_d-1)*$dilation_d+1)/2);
            $padding_h = (int)floor((($im_h-1)*$stride_h-$im_h+($filter_h-1)*$dilation_h+1)/2);
            $padding_w = (int)floor((($im_w-1)*$stride_w-$im_w+($filter_w-1)*$dilation_w+1)/2);
            $out_d = $im_d;
            $out_h = $im_h;
            $out_w = $im_w;
        } else {
            $out_buf_size = $batches*
                $out_d*$filter_d*
                $out_h*$filter_h*
                $out_w*$filter_w*
                $channels;
            $padding_d = 0;
            $padding_h = 0;
            $padding_w = 0;
        }
        if($cols_size!=$out_buf_size ||
            count($cols)-$cols_offset>$out_buf_size) {
            throw new InvalidArgumentException('output buffer size is invalid');
        }
        if($channels_first) {
            $im_w_step =    1;
            $im_h_step =    $im_w;
            $im_d_step =    $im_w*$im_h;
            $channel_step = $im_w*$im_h*$im_d;
            $batch_step =   $im_w*$im_h*$im_d*$channels;
        } else {
            $channel_step = 1;
            $im_w_step =  $channels;
            $im_h_step =  $channels*$im_w;
            $im_d_step =  $channels*$im_w*$im_h;
            $batch_step = $channels*$im_w*$im_h*$im_d;
        }
        $stride_w_step = $im_w_step*$stride_w;
        $stride_h_step = $im_h_step*$stride_h;
        $stride_d_step = $im_d_step*$stride_d;
        $filter_w_step = $im_w_step*$dilation_w;
        $filter_h_step = $im_h_step*$dilation_h;
        $filter_d_step = $im_d_step*$dilation_d;

        if($cols_channels_first) {
            $out_filter_step = 1;
            $out_channel_step = $filter_d*$filter_h*$filter_w;
        } else {
            $out_filter_step = $channels;
            $out_channel_step = 1;
        }
        $out_cell_step = $filter_d*$filter_h*$filter_w*$channels;

        $batch_pos = $images_offset-$im_d_step*$padding_d-$im_h_step*$padding_h-$im_w_step*$padding_w;
        $out_pos = $cols_offset;

        $vim_d = $out_d*$stride_d;
        $vim_h = $out_h*$stride_h;
        $vim_w = $out_w*$stride_w;
        $vfilter_d = $filter_d*$dilation_d;
        $vfilter_h = $filter_h*$dilation_h;
        $vfilter_w = $filter_w*$dilation_w;
        for($batch=0; $batch<$batches;$batch++) {
            $stride_d_pos = $batch_pos;
            for($vim_z=0;$vim_z<$vim_d;$vim_z+=$stride_d){//stride_z
                $stride_h_pos = $stride_d_pos;
                for($vim_y=0;$vim_y<$vim_h;$vim_y+=$stride_h){//stride_y
                    $stride_w_pos = $stride_h_pos;
                    for($vim_x=0;$vim_x<$vim_w;$vim_x+=$stride_w) {//stride_x
                        #print('osf=%d,%d,%d'%(out_h,stride_h,filter_h))
                        $this->copyCell3d(
                            $reverse,
                            $images,
                            $stride_w_pos,
                            $im_d,
                            $im_h,
                            $im_w,
                            $channels,
                            $channel_step,
                            $filter_d_step,
                            $filter_h_step,
                            $filter_w_step,
                            $vim_z-$padding_d,
                            $vim_y-$padding_h,
                            $vim_x-$padding_w,
                            $vfilter_d,
                            $vfilter_h,
                            $vfilter_w,
                            $dilation_d,
                            $dilation_h,
                            $dilation_w,
                            $cols,
                            $out_pos,
                            $out_filter_step,
                            $out_channel_step

                        );
                        $stride_w_pos += $stride_w_step;
                        $out_pos += $out_cell_step;
                    }
                    $stride_h_pos += $stride_h_step;
                }
                $stride_d_pos += $stride_d_step;
            }
            $batch_pos += $batch_step;
        }
    }

    /**
    */
    public function randomUniform(
        int $n,
        Buffer $X, int $offsetX, int $incX,
        $low,
        $high,
        int $seed
        )
    {
        if($this->math) {
            $this->math->randomUniform(
                $n,
                $X,
                $offsetX,
                $incX,
                $low,
                $high,
                $seed
            );
            return;
        }
        mt_srand($seed);
        $px = $offsetX;
        if(method_exists($X,'dtype')) {
            $isInt = array_key_exists($X->dtype(),$this->intTypes);
        } else {
            $isInt = false;
        }
        if($isInt) {
            $origHigh = $high;
            $high += 1;
            for($i=0; $i<$n; $i++,$px+=$incX) {
                $value = ((int)floor(($high-$low)*mt_rand()/mt_getrandmax()))+$low;
                if($value>$origHigh) {
                    $value = $origHigh;
                }
                $X[$px] = $value;
            }
        } else {
            for($i=0; $i<$n; $i++,$px+=$incX) {
                $X[$px] = ($high-$low)*mt_rand()/mt_getrandmax()+$low;
            }
        }
    }

    protected function genRandNormal(float $mean, float $scale) : float
    {
        $max=mt_getrandmax();
        $x=mt_rand(1,$max-1)/$max;
        $y=mt_rand(1,$max-1)/$max;
        return sqrt(-2*log($x))*cos(2*pi()*$y)*$scale+$mean;
    }

    /**
    */
    public function randomNormal(
        int $n,
        Buffer $X, int $offsetX, int $incX,
        float $mean,
        float $scale,
        int $seed
        )
    {
        if($this->math) {
            $this->math->randomNormal(
                $n,
                $X,
                $offsetX,
                $incX,
                $mean,
                $scale,
                $seed
            );
            return;
        }
        mt_srand($seed);
        $px = $offsetX;
        for($i=0; $i<$n; $i++,$px+=$incX) {
            $X[$px] = $this->genRandNormal($mean,$scale);
        }
    }

    /**
    */
    public function randomSequence(
        int $n,
        int $size,
        Buffer $X, int $offsetX, int $incX,
        int $seed
        )
    {
        if($this->math) {
            $this->math->randomSequence(
                $n,
                $size,
                $X,
                $offsetX,
                $incX,
                $seed
            );
            return;
        }
        mt_srand($seed);
        $px = $offsetX;
        for($i=0; $i<$n; $i++,$px+=$incX){
            $X[$px] = $i;
        }
        $px = $offsetX;
        for($i=0; $i<$n; $i++,$px+=$incX) {
            $idx = mt_rand($i,$n-1)*$incX+$offsetX;
            $tmp = $X[$px];
            $X[$px] = $X[$idx];
            $X[$idx] = $tmp;
        }
    }

    /**
    */
    public function slice(
        bool $reverse,
        bool $addMode,
        int $m,
        int $n,
        int $k,
        int $size,
        Buffer $A, int $offsetA, int $incA,
        Buffer $Y, int $offsetY, int $incY,
        int $startAxis0,
        int $sizeAxis0,
        int $startAxis1,
        int $sizeAxis1,
        int $startAxis2,
        int $sizeAxis2
        )
    {
        if($this->math) {
            $this->math->slice(
                $reverse,
                $addMode,
                $m,
                $n,
                $k,
                $size,
                $A,
                $offsetA,
                $incA,
                $Y,
                $offsetY,
                $incY,
                $startAxis0,
                $sizeAxis0,
                $startAxis1,
                $sizeAxis1,
                $startAxis2,
                $sizeAxis2
            );
            return;
        }
        if($m*$n*$k*$size*$incA+$offsetA>count ($A)) {
            throw new InvalidArgumentException('unmatch BufferA size and m,n,k');
        }
        if($startAxis0<0||$startAxis0>=$m||
            $sizeAxis0<0||$sizeAxis0+$startAxis0>$m){
            throw new InvalidArgumentException('Axis0 range is too large for source array.');
        }
        if($startAxis1<0||$startAxis1>=$n||
            $sizeAxis1<0||$sizeAxis1+$startAxis1>$n){
            throw new InvalidArgumentException('Axis1 range is too large for source array.');
        }
        if($startAxis2<0||$startAxis2>=$k||
            $sizeAxis2<0||$sizeAxis2+$startAxis2>$k){
            throw new InvalidArgumentException('Axis2 range is too large for source array.');
        }
        if($sizeAxis0*$sizeAxis1*$sizeAxis2*$size*$incY>count($Y)-$offsetY){
            throw new InvalidArgumentException('BufferY size is too small');
        }
        for($i0=0;$i0<$sizeAxis0;$i0++) {
            for($i1=0;$i1<$sizeAxis1;$i1++){
                for($i2=0;$i2<$sizeAxis2;$i2++){
                    $pa = ($i0+$startAxis0)*$n*$k*$size+
                          ($i1+$startAxis1)*$k*$size+
                          ($i2+$startAxis2)*$size+
                          $offsetA;
                    $py = $i0*$sizeAxis1*$sizeAxis2*$size+
                          $i1*$sizeAxis2*$size+
                          $i2*$size+
                          $offsetY;
                    if(!$reverse) {
                        if($addMode){
                            $this->math_add($size,$A,$pa,$incA,$Y,$py,$incY);
                        } else {
                            $this->math_copy($size,$A,$pa,$incA,$Y,$py,$incY);
                        }
                    } else {
                        if($addMode){
                            $this->math_add($size,$Y,$py,$incY,$A,$pa,$incA);
                        } else {
                            $this->math_copy($size,$Y,$py,$incY,$A,$pa,$incA);
                        }
                    }
                }
            }
        }
    }

    public function matrixcopy(
        bool $trans,
        int $m,
        int $n,
        float $alpha,
        Buffer $A, int $offsetA, int $ldA,
        Buffer $B, int $offsetB, int $ldB)
    {
        if($this->math) {
            $this->math->matrixcopy(
                $trans,
                $m,
                $n,
                $alpha,
                $A, $offsetA, $ldA,
                $B, $offsetB, $ldB);
            return;
        }
        if(!$trans) {
            for($i=0;$i<$m;$i++) {
                for($j=0;$j<$n;$j++) {
                    $B[$i*$ldB+$j+$offsetB] = $alpha * $A[$i*$ldA+$j+$offsetA];
                }
            }
        } else {
            for($i=0;$i<$m;$i++) {
                for($j=0;$j<$n;$j++) {
                    $B[$j*$ldB+$i+$offsetB] = $alpha * $A[$i*$ldA+$j+$offsetA];
                }
            }
        }
    }

    public function fill(
        int $n,
        Buffer $V, int $offsetV,
        Buffer $X, int $offsetX, int $incX)
    {
        if($this->math) {
            $this->math->fill(
                $n,
                $V, $offsetV,
                $X, $offsetX, $incX);
            return;
        }
        $idX = $offsetX;
        $value = $V[$offsetV];
        for($i=0;$i<$n;$i++,$idX+=$incX) {
            $X[$idX] = $value;
        }
    }

    public function nan2num(
        int $n,
        Buffer $X, int $offsetX, int $incX,
        float $alpha
        ) : void
    {
        if($this->useMath($X)) {
            $this->math->nan2num($n,$X,$offsetX,$incX,$alpha);
            return;
        }

        if($offsetX+($n-1)*$incX>=count($X))
            throw new RuntimeException('Vector specification too large for buffer.');

        $idx = $offsetX;
         for ($i=0; $i<$n; $i++,$idx+=$incX) {
            $t = $X[$idx];
            if(is_nan($t)) {
                $X[$idx] = $alpha;
            }
        }
    }

    public function isnan(
        int $n,
        Buffer $X, int $offsetX, int $incX
        ) : void
    {
        if($this->useMath($X)) {
            $this->math->isnan($n,$X,$offsetX,$incX);
            return;
        }

        if($offsetX+($n-1)*$incX>=count($X))
            throw new RuntimeException('Vector specification too large for buffer.');

        $idx = $offsetX;
         for ($i=0; $i<$n; $i++,$idx+=$incX) {
            $t = $X[$idx];
            if(is_nan($t)) {
                $X[$idx] = 1.0;
            } else {
                $X[$idx] = 0.0;
            }
        }
    }

    public function imagecopy(
        int $height,
        int $width,
        int $channels,
        Buffer $A, int $offsetA,
        Buffer $B, int $offsetB,
        bool $channelsFirst,
        int $heightShift,
        int $widthShift,
        bool $verticalFlip,
        bool $horizontalFlip,
        bool $rgbFlip
        )
    {
        if($this->math) {
            $this->math->imagecopy(
                    $height,
                    $width,
                    $channels,
                    $A, $offsetA,
                    $B, $offsetB,
                    $channelsFirst,
                    $heightShift,
                    $widthShift,
                    $verticalFlip,
                    $horizontalFlip,
                    $rgbFlip
                  );
            return;
        }

        if($width*$height*$channels+$offsetA>count ($A)) {
            throw new InvalidArgumentException('Matrix specification too large for bufferA');
        }
        if($width*$height*$channels+$offsetB>count ($B)) {
            throw new InvalidArgumentException('Matrix specification too large for bufferB');
        }

        if($channelsFirst) {
            $ldC = $width*$height;
            $ldY = $width;
            $ldX = 1;
        } else {
            $ldY = $width*$channels;
            $ldX = $channels;
            $ldC = 1;
        }
        $directionY = $directionX = 1;
        $biasY = $biasX = 0;
        if($verticalFlip) {
            $directionY = -$directionY;
            $biasY = $height-1;
        }
        if($horizontalFlip) {
            $directionX = -$directionX;
            $biasX = $width-1;
        }
        $biasY -= $heightShift*$directionY;
        $biasX -= $widthShift*$directionX;
        for($y=0;$y<$height;$y++) {
            for($x=0;$x<$width;$x++) {
                for($c=0;$c<$channels;$c++) {
                    $sy = $y*$directionY+$biasY;
                    $sx = $x*$directionX+$biasX;
                    if($sy<0) {
                        $sy = 0;
                    } elseif($sy>=$height) {
                        $sy = $height-1;
                    }
                    if($sx<0) {
                        $sx = 0;
                    } elseif($sx>=$width) {
                        $sx = $width-1;
                    }
                    $srcC = ($rgbFlip&&$c<3)?(2-$c):$c;
                    $B[$y*$ldY+$x*$ldX+$c*$ldC+$offsetB] =
                        $A[$sy*$ldY+$sx*$ldX+$srcC*$ldC+$offsetA];
                }
            }
        }
    }
}
