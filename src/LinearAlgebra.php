<?php
namespace Rindow\Math\Matrix;

use Interop\Polite\Math\Matrix\BLAS;
use Interop\Polite\Math\Matrix\NDArray;
use InvalidArgumentException;
use ArrayAccess as Buffer;

define("LAPACK_ROW_MAJOR",101);
define("LAPACK_COL_MAJOR",102);


class LinearAlgebra
{
    protected $iaminwarning;
    protected $blas;
    protected $lapack;
    protected $math;
    protected $defaultFloatType = NDArray::float32;

    public function __construct($blas,$lapack,$math,$defaultFloatType=null)
    {
        $this->blas = $blas;
        $this->lapack = $lapack;
        $this->math = $math;
        if($defaultFloatType!==null)
            $this->defaultFloatType = $defaultFloatType;
    }

    protected function printableShapes($values)
    {
        if(!is_array($values)) {
            if($values instanceof NDArray)
                return '('.implode(',',$values->shape()).')';
            if(is_object($values))
                return '"'.get_class($values).'"';
            if(is_numeric($values) || is_string($values))
                return strval($values);
            return gettype($values);
        }
        $string = '[';
        foreach($values as $value) {
            if($string!='[') {
                $string .= ',';
            }
            $string .= $this->printableShapes($value);
        }
        $string .= ']';
        return $string;
    }

    public function alloc(array $shape,$dtype=null)
    {
        if($dtype===null)
            $dtype = $this->defaultFloatType;
        return new NDArrayPhp(null,$dtype,$shape);
    }

    public function zeros(
        NDArray $X) : NDArray
    {
        $N = $X->size();
        $XX = $X->buffer();
        $offX = $X->offset();
        $this->math->zeros($N,$XX,$offX,1);
        return $X;
    }

    public function astype(NDArray $X, $dtype) : NDArray
    {
        $Y = $this->alloc($X->shape(),$dtype);
        $n = $X->size();
        $XX = $X->buffer();
        $offX = $X->offset();
        $YY = $Y->buffer();
        $offY = $Y->offset();

        $this->math->astype(
            $n,
            $dtype,
            $XX,$offX,1,
            $YY,$offY,1
        );
        return $Y;
    }

    /**
    *    Y := X
    */
    public function copy(
        NDArray $X,
        NDArray $Y=null ) : NDArray
    {
        if($Y===null) {
            $Y = $this->alloc($X->shape(),$X->dtype());
        } else {
            if($X->shape()!=$Y->shape()) {
                $shapeError = '('.implode(',',$X->shape()).'),('.implode(',',$Y->shape()).')';
                throw new InvalidArgumentException("Unmatch shape of dimension: ".$shapeError);
            }
        }
        $N = $X->size();
        $XX = $X->buffer();
        $offX = $X->offset();
        $YY = $Y->buffer();
        $offY = $Y->offset();
        $this->blas->copy($N,$XX,$offX,1,$YY,$offY,1);
        return $Y;
    }

    /**
    *    X := alpha * X
    */
    public function scal(
        float $alpha,
        NDArray $X) : NDArray
    {
        $N = $X->size();
        $XX = $X->buffer();
        $offX = $X->offset();
        if($alpha===null) {
            $alpha = 1.0;
        }
        $this->blas->scal($N,$alpha,$XX,$offX,1);
        return $X;
    }

    /**
    *    Y := alpha * X + Y
    */
    public function axpy(
        NDArray $X,
        NDArray $Y,
        float $alpha=null) : NDArray
    {
        if($X->shape()!=$Y->shape()) {
            $shapeError = '('.implode(',',$X->shape()).'),('.implode(',',$Y->shape()).')';
            throw new InvalidArgumentException("Unmatch shape of dimension: ".$shapeError);
        }
        $N = $X->size();
        $XX = $X->buffer();
        $offX = $X->offset();
        $YY = $Y->buffer();
        $offY = $Y->offset();
        if($alpha===null) {
            $alpha = 1.0;
        }
        $this->blas->axpy($N,$alpha,$XX,$offX,1,$YY,$offY,1);
        return $Y;
    }

    /**
    *    ret := X^t Y = x_1 * y_1 + ... + x_n * y_n
    */
    public function dot(
        NDArray $X,
        NDArray $Y)
    {
        if($X->shape()!=$Y->shape()) {
            $shapeError = '('.implode(',',$X->shape()).'),('.implode(',',$Y->shape()).')';
            throw new InvalidArgumentException("Unmatch shape of dimension: ".$shapeError);
        }
        $N = $X->size();
        $XX = $X->buffer();
        $offX = $X->offset();
        $YY = $Y->buffer();
        $offY = $Y->offset();
        return $this->blas->dot($N,$XX,$offX,1,$YY,$offY,1);
    }

    /**
    *    ret := |x_1| + ... + |x_n|
    */
    public function asum(
        NDArray $X) : float
    {
        $N = $X->size();
        $XX = $X->buffer();
        $offX = $X->offset();
        return $this->blas->asum($N,$XX,$offX,1);
    }

    /**
    *    ret := arg max |X(i)|
    */
    public function iamax(
        NDArray $X) : int
    {
        $N = $X->size();
        $XX = $X->buffer();
        $offX = $X->offset();
        return $this->blas->iamax($N,$XX,$offX,1);
    }

    /**
    *    ret := arg min |X(i)|
    */
    public function iamin(
        NDArray $X) : int
    {
        $N = $X->size();
        $XX = $X->buffer();
        $offX = $X->offset();
        if(method_exists($this->blas,'iamin')) {
            return $this->blas->iamin($N,$XX,$offX,1);
        } else {
            return $this->iaminCompatible($N,$XX,$offX,1);
        }
    }

    /**
    *   legacy opennblas compatible
    */
    protected function iaminCompatible(
        int $n,
        Buffer $X, int $offsetX, int $incX ) : int
    {
        if($this->iaminwarning) {
            echo "*iamin* not found. probably OpenBLAS is legacy version.";
            $this->iaminwarning = true;
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

    /**
    *    ret := max |X(i)|
    */
    public function amax(
        NDArray $X) : float
    {
        $N = $X->size();
        $XX = $X->buffer();
        $offX = $X->offset();
        $i = $this->blas->iamax($N,$XX,$offX,1);
        return $XX[$offX+$i];
    }

    /**
    *    ret := min |X(i)|
    */
    public function amin(
        NDArray $X) : float
    {
        $N = $X->size();
        $XX = $X->buffer();
        $offX = $X->offset();
        if(method_exists($this->blas,'iamin')) {
            $i = $this->blas->iamin($N,$XX,$offX,1);
        } else {
            $i = $this->iaminCompatible($N,$XX,$offX,1);
        }
        return $XX[$offX+$i];
    }

    /**
    *    ret := sqrt(sum(Xn ** 2))
    */
    public function nrm2(
        NDArray $X) : float
    {
        $N = $X->size();
        $XX = $X->buffer();
        $offX = $X->offset();
        $ret = $this->blas->nrm2($N,$XX,$offX,1);
        return $ret;
    }

    /**
    *    y := alpha * Ax + beta * y
    */
    public function gemv(
        NDArray $A,
        NDArray $X,
        float $alpha=null,
        float $beta=null,
        NDArray $Y=null,
        bool $trans=null) : NDArray
    {
        if($A->ndim()!=2 || $X->ndim()!=1) {
            throw new InvalidArgumentException('"A" must be 2D-NDArray and "X" must 1D-NDArray.');
        }
        $shapeA = $A->shape();
        $shapeX = $X->shape();
        $rows = (!$trans) ? $shapeA[0] : $shapeA[1];
        $cols = (!$trans) ? $shapeA[1] : $shapeA[0];
        if($cols!=$shapeX[0]) {
            throw new InvalidArgumentException('The number of columns in "A" and The number of item in "X" must be the same');
        }
        $AA = $A->buffer();
        $XX = $X->buffer();
        $offA = $A->offset();
        $offX = $X->offset();
        $m = $shapeA[0];
        $n = $shapeA[1];
        if($alpha===null) {
            $alpha = 1.0;
        }
        if($beta===null) {
            $beta = 0.0;
        }
        if($Y!=null) {
            if($Y->ndim()!=1) {
                throw new InvalidArgumentException('"Y" must 1D-NDArray.');
            }
            $shapeY = $Y->shape();
            if($rows!=$shapeY[0]) {
                throw new InvalidArgumentException('The number of rows in "A" and The number of item in "Y" must be the same');
            }
        } else {
            $Y = $this->zeros($this->alloc([$rows]));
        }
        $YY = $Y->buffer();
        $offY = $Y->offset();
        $trans = (!$trans) ? BLAS::NoTrans : BLAS::Trans;

        $this->blas->gemv(
            BLAS::RowMajor,$trans,
            $m,$n,
            $alpha,
            $AA,$offA,$n,
            $XX,$offX,1,
            $beta,
            $YY,$offY,1);

        return $Y;
    }

    /**
    *    C := alpha * AB + beta * C
    */
    public function gemm(
        NDArray $A,
        NDArray $B,
        float $alpha=null,
        float $beta=null,
        NDArray $C=null,
        bool $transA=null,
        bool $transB=null) : NDArray
    {
        if($A->ndim()!=2 || $B->ndim()!=2) {
            throw new InvalidArgumentException('Dimensions must be 2D-NDArray');
        }
        $shapeA = $A->shape();
        if($transA) {
            $shapeA = [$shapeA[1],$shapeA[0]];
        }
        $shapeB = $B->shape();
        if($transB) {
            $shapeB = [$shapeB[1],$shapeB[0]];
        }
        if($shapeA[1]!=$shapeB[0]) {
            throw new InvalidArgumentException('The number of columns in "A" and the number of rows in "B" must be the same');
        }
        $AA = $A->buffer();
        $BB = $B->buffer();
        $offA = $A->offset();
        $offB = $B->offset();
        $M = $shapeA[0];
        $N = $shapeB[1];
        $K = $shapeA[1];

        if($alpha===null) {
            $alpha = 1.0;
        }
        if($beta===null) {
            $beta = 0.0;
        }
        if($C!=null) {
            $shapeC = $C->shape();
            if($M!=$shapeC[0] || $N!=$shapeC[1]) {
                throw new InvalidArgumentException('"A" and "C" must have the same number of rows."B" and "C" must have the same number of columns');
            }
        } else {
            $C = $this->zeros($this->alloc([$M,$N]));
        }
        $CC = $C->buffer();
        $offC = $C->offset();

        $lda = ($transA) ? $M : $K;
        $ldb = ($transB) ? $K : $N;
        $ldc = $N;
        $transA = ($transA) ? BLAS::Trans : BLAS::NoTrans;
        $transB = ($transB) ? BLAS::Trans : BLAS::NoTrans;

        $this->blas->gemm(
            BLAS::RowMajor,$transA,$transB,
            $M,$N,$K,
            $alpha,
            $AA,$offA,$lda,
            $BB,$offB,$ldb,
            $beta,
            $CC,$offC,$ldc);

        return $C;
    }

    /**
    *
    */
    public function matmul(
        NDArray $A,
        NDArray $B,
        bool $transA=null,
        bool $transB=null,
        NDArray $C=null,
        float $alpha=null,
        float $beta=null
        ) : NDArray
    {
        if($A->ndim()<2 || $B->ndim()<2) {
            throw new InvalidArgumentException('Dimensions rank must be greater then 2D or equal:['.
                implode(',',$A->shape()).']<=>['.implode(',',$B->shape()).']');
        }
        $shapeA = $A->shape();
        $shapeB = $B->shape();
        $shapeEA = [array_pop($shapeA)];
        array_unshift($shapeEA,array_pop($shapeA));
        $shapeEB = [array_pop($shapeB)];
        array_unshift($shapeEB,array_pop($shapeB));
        $batchA = (int)array_product($shapeA);
        $batchB = (int)array_product($shapeB);
        $flatA = $A->reshape(array_merge([$batchA],$shapeEA));
        $flatB = $B->reshape(array_merge([$batchB],$shapeEB));

        if($transA) {
            $shapeEA = array_reverse($shapeEA);
        }
        if($transB) {
            $shapeEB = array_reverse($shapeEB);
        }
        if($shapeEA[1]!=$shapeEB[0]) {
            throw new InvalidArgumentException('The number of columns in "A" and the number of rows in "B" must be the same:['.
                implode(',',$A->shape()).']<=>['.implode(',',$B->shape()).']');
        }

        $AA = $A->buffer();
        $BB = $B->buffer();
        $M = $shapeEA[0];
        $N = $shapeEB[1];
        $K = $shapeEA[1];

        if($alpha===null) {
            $alpha = 1.0;
        }
        if($beta===null) {
            $beta = 0.0;
        }
        $lda = ($transA) ? $M : $K;
        $ldb = ($transB) ? $K : $N;
        $ldc = $N;
        $transA = ($transA) ? BLAS::Trans : BLAS::NoTrans;
        $transB = ($transB) ? BLAS::Trans : BLAS::NoTrans;

        $shapeEC = [$shapeEA[0],$shapeEB[1]];
        if($batchA>$batchB) {
            $broadcastDest = $batchA;
            $broadcastBase = $batchB;
            $orgShapeC=array_merge($shapeA,$shapeEC);
        } else {
            $broadcastDest = $batchB;
            $broadcastBase = $batchA;
            $orgShapeC=array_merge($shapeB,$shapeEC);
        }
        if($broadcastDest % $broadcastBase != 0) {
            throw new InvalidArgumentException('Matrix size-incompatible for broadcast:['.
                implode(',',$A->shape()).']<=>['.implode(',',$B->shape()).']');
        }
        if($C!=null) {
            if($C->shape()!=$orgShapeC) {
                throw new InvalidArgumentException('"A" and "C" must have the same number of rows."B" and "C" must have the same number of columns:['.
                    implode(',',$A->shape()).'] , ['.implode(',',$B->shape()).'] => ['.implode(',',$C->shape()).']');
            }
        } else {
            $C = $this->alloc($orgShapeC,$A->dtype());
            $this->zeros($C);
        }
        $flatC = $C->reshape(array_merge([$broadcastDest],$shapeEC));
        $CC = $C->buffer();
        $repeats = (int)floor($broadcastDest/$broadcastBase);
        $offA = $A->offset();
        $offB = $B->offset();
        $offC = $C->offset();
        $incA = $M*$K;
        $incB = $N*$K;
        $incC = $M*$N;
        for($i=0;$i<$repeats;$i++) {
            if($batchA>$batchB) {
                $offB = $B->offset();
            } else {
                $offA = $A->offset();
            }
            for($j=0;$j<$broadcastBase;$j++) {
                $this->blas->gemm(
                    BLAS::RowMajor,$transA,$transB,
                    $M,$N,$K,
                    $alpha,
                    $AA,$offA,$lda,
                    $BB,$offB,$ldb,
                    $beta,
                    $CC,$offC,$ldc);
                $offA+=$incA;
                $offB+=$incB;
                $offC+=$incC;
            }
        }
        return $C;
    }

    /**
    *    ret := x_1 + ... + x_n
    */
    public function sum(
        NDArray $X) : float
    {
        $N = $X->size();
        $XX = $X->buffer();
        $offX = $X->offset();
        return $this->math->sum($N,$XX,$offX,1);
    }

    /**
    *    ret := arg max X(i)
    */
    public function imax(
        NDArray $X) : int
    {
        $N = $X->size();
        $XX = $X->buffer();
        $offX = $X->offset();
        return $this->math->imax($N,$XX,$offX,1);
    }

    /**
    *    ret := arg min X(i)
    */
    public function imin(
        NDArray $X) : int
    {
        $N = $X->size();
        $XX = $X->buffer();
        $offX = $X->offset();
        return $this->math->imin($N,$XX,$offX,1);
    }

    /**
    *    ret := max X(i)
    */
    public function max(
        NDArray $X) : float
    {
        $N = $X->size();
        $XX = $X->buffer();
        $offX = $X->offset();
        $i = $this->math->imax($N,$XX,$offX,1);
        return $XX[$offX+$i];
    }

    /**
    *    ret := min X(i)
    */
    public function min(
        NDArray $X) : float
    {
        $N = $X->size();
        $XX = $X->buffer();
        $offX = $X->offset();
        $i = $this->math->imin($N,$XX,$offX,1);
        return $XX[$offX+$i];
    }

    /**
    *    X := alpha * X + beta
    */
    public function increment(
        NDArray $X,
        float $beta=null,
        float $alpha=null) : NDArray
    {
        $n = $X->size();
        $XX = $X->buffer();
        $offX = $X->offset();

        if($alpha===null) {
            $alpha = 1.0;
        }
        if($beta===null) {
            $beta = 0.0;
        }

        $this->math->increment(
            $n,
            $alpha,
            $XX,$offX,1,
            $beta);

        return $X;
    }

    /**
    *    X := 1 / (a*X + b)
    */
    public function reciprocal(
        NDArray $X,
        float $beta=null,
        float $alpha=null) : NDArray
    {
        $n = $X->size();
        $XX = $X->buffer();
        $offX = $X->offset();

        if($alpha===null) {
            $alpha = 1.0;
        }
        if($beta===null) {
            $beta = 0.0;
        }

        $this->math->reciprocal(
            $n,
            $alpha,
            $XX,$offX,1,
            $beta);

        return $X;
    }

    /**
     *     X := X  (X > a)
     *     X := a  (X <= a)
     */
    public function maximum(
        float $alpha,
        NDArray $X
        ) : NDArray
    {
        $n = $X->size();
        $XX = $X->buffer();
        $offX = $X->offset();

        $this->math->maximum(
            $n,
            $alpha,
            $XX,$offX,1);

        return $X;
    }

    /**
     *     X := X  (X < a)
     *     X := a  (X >= a)
     */
    public function minimum(
        float $alpha,
        NDArray $X
        ) : NDArray
    {
        $n = $X->size();
        $XX = $X->buffer();
        $offX = $X->offset();

        $this->math->minimum(
            $n,
            $alpha,
            $XX,$offX,1);

        return $X;
    }

    /**
     *     X := 1  (X > a)
     *     X := 0  (X <= a)
     */
    public function greater(
        float $alpha,
        NDArray $X
        ) : NDArray
    {
        $n = $X->size();
        $XX = $X->buffer();
        $offX = $X->offset();

        $this->math->greater(
            $n,
            $alpha,
            $XX,$offX,1);

        return $X;
    }

    /**
     *     X := 1  (X < a)
     *     X := 0  (X >= a)
     */
    public function less(
        float $alpha,
        NDArray $X
        ) : NDArray
    {
        $n = $X->size();
        $XX = $X->buffer();
        $offX = $X->offset();

        $this->math->less(
            $n,
            $alpha,
            $XX,$offX,1);

        return $X;
    }

    /**
     *    A(m,n) := X(n) * A(m,n)
     */
     public function multiply(
        NDArray $X,
        NDArray $A,
        bool $trans=null
        ) : NDArray
    {
        if($trans===null)
            $trans = false;
        $shapeX = $X->shape();
        $shapeA = $A->shape();
        if($trans)
            $shapeA = array_reverse($shapeA);
        while(true) {
            $xd = array_pop($shapeX);
            if($xd===null)
                break;
            $ad = array_pop($shapeA);
            if($xd!==$ad) {
                $shapeA = $trans ? array_reverse($A->shape()) : $A->shape();
                throw new InvalidArgumentException('Unmatch dimension size for broadcast.: '.
                    '['.implode(',',$X->shape()).'] => ['.implode(',',$shapeA).']');
            }
        }
        $n = $X->size();
        $XX = $X->buffer();
        $offX = $X->offset();
        $m = $A->size()/$n;
        $AA = $A->buffer();
        $offA = $A->offset();
        if($trans) {
            [$m,$n] = [$n,$m];
        }

        $this->math->multiply(
            $trans,
            $m,
            $n,
            $XX,$offX,1,
            $AA,$offA,$n);

        return $A;
    }

    /**
     *    A(m,n) := X(n) * A(m,n)
     */
     public function add(
        NDArray $X,
        NDArray $A,
        float $alpha=null,
        bool $trans=null
        ) : NDArray
    {
        if($trans===null)
            $trans = false;
        if($alpha===null)
            $alpha = 1.0;
        $shapeX = $X->shape();
        $shapeA = $A->shape();
        if($trans)
            $shapeA = array_reverse($shapeA);
        while(true) {
            $xd = array_pop($shapeX);
            if($xd===null)
                break;
            $ad = array_pop($shapeA);
            if($xd!==$ad)
                throw new InvalidArgumentException('Unmatch dimension size for broadcast.: '.
                    '['.implode(',',$X->shape()).'] => ['.implode(',',$A->shape()).']');
        }
        $n = $X->size();
        $XX = $X->buffer();
        $offX = $X->offset();
        $m = $A->size()/$n;
        $AA = $A->buffer();
        $offA = $A->offset();
        if($trans) {
            [$m,$n] = [$n,$m];
        }

        $this->math->add(
            $trans,
            $m,
            $n,
            $alpha,
            $XX,$offX,1,
            $AA,$offA,$n);

        return $A;
    }

   /**
    *     X := X ^ 2
    */
    public function square(
        NDArray $X
        ) : NDArray
    {
        $n = $X->size();
        $XX = $X->buffer();
        $offX = $X->offset();

        $this->math->square(
            $n,
            $XX,$offX,1);

        return $X;
    }

    /**
     *     X := sqrt(X)
     */
    public function sqrt(
        NDArray $X
        ) : NDArray
    {
        $n = $X->size();
        $XX = $X->buffer();
        $offX = $X->offset();

        $this->math->sqrt(
            $n,
            $XX,$offX,1);

        return $X;
    }

    /**
     *     X := 1 / (a * sqrt(X) + b)
     */
    public function rsqrt(
        NDArray $X,
        float $beta=null,
        float $alpha=null) : NDArray
    {
        $n = $X->size();
        $XX = $X->buffer();
        $offX = $X->offset();

        if($alpha===null) {
            $alpha = 1.0;
        }
        if($beta===null) {
            $beta = 0.0;
        }

        $this->math->rsqrt(
            $n,
            $alpha,
            $XX,$offX,1,
            $beta);

        return $X;
    }

    /**
     *     X := X ^ a
     */
    public function pow(
        NDArray $X,
        float $alpha
        ) : NDArray
    {
        $n = $X->size();
        $XX = $X->buffer();
        $offX = $X->offset();

        $this->math->pow(
            $n,
            $alpha,
            $XX,$offX,1);

        return $X;
    }

    /**
     *     X(i) := e ^ X(i)
     */
    public function exp(
        NDArray $X
        ) : NDArray
    {
        $n = $X->size();
        $XX = $X->buffer();
        $offX = $X->offset();

        $this->math->exp(
            $n,
            $XX,$offX,1);

        return $X;
    }

    /**
     *     X := log(X)
     */
    public function log(
        NDArray $X
        ) : NDArray
    {
        $n = $X->size();
        $XX = $X->buffer();
        $offX = $X->offset();

        $this->math->log(
            $n,
            $XX,$offX,1);

        return $X;
    }

    /**
     *     X := tanh(X)
     */
    public function tanh(
        NDArray $X
        ) : NDArray
    {
        $n = $X->size();
        $XX = $X->buffer();
        $offX = $X->offset();

        $this->math->tanh(
            $n,
            $XX,$offX,1);

        return $X;
    }

    /**
     *     Y(i) := 1 (X(i) = Y(i))
     *     Y(i) := 0 (X(i) = Y(i))
     */
    public function equal(NDArray $X, NDArray $Y) : NDArray
    {
        if($X->shape()!=$Y->shape()) {
            $shapeError = '('.implode(',',$X->shape()).'),('.implode(',',$Y->shape()).')';
            throw new InvalidArgumentException('Unmatch shape of dimension "X" and "Y" and "numClass": '.$shapeError);
        }
        $N = $X->size();
        $XX = $X->buffer();
        $offX = $X->offset();
        $incX = 1;
        $YY = $Y->buffer();
        $offY = $Y->offset();
        $incY = 1;
        $this->math->equal($N,$XX,$offX,$incX,$YY,$offY,$incY);

        return $Y;
    }

    public function duplicate(NDArray $X, int $repeats=null, bool $trans=null,NDArray $A=null) : NDArray
    {
        if($trans===null)
            $trans = false;
        if($A===null) {
            if($repeats===null)
                $repeats = 1;
            if(!$trans) {
                $A = $this->alloc(array_merge([$repeats],$X->shape()),$X->dtype());
            } else {
                $A = $this->alloc(array_merge($X->shape(),[$repeats]),$X->dtype());
            }
        } else {
            $shapeX = $X->shape();
            $shapeA = $A->shape();
            if($trans)
                $shapeA = array_reverse($shapeA);
            while(true) {
                $xd = array_pop($shapeX);
                if($xd===null)
                    break;
                $ad = array_pop($shapeA);
                if($xd!==$ad)
                    throw new InvalidArgumentException('Unmatch dimension size for broadcast.: '.
                        '['.implode(',',$X->shape()).'] => ['.implode(',',$A->shape()).']');
            }
        }

        $n = $X->size();
        $XX = $X->buffer();
        $offX = $X->offset();
        $m = $A->size()/$n;
        $AA = $A->buffer();
        $offA = $A->offset();
        if($trans) {
            [$m,$n] = [$n,$m];
        }

        $this->math->duplicate(
            $trans,
            $m,
            $n,
            $XX,$offX,1,
            $AA,$offA,$n);

        return $A;
    }

    /**
     * Y := A[X]
     */
    public function select(
        NDArray $A,
        NDArray $X,
        int $axis=null,
        NDArray $Y=null) : NDArray
    {
        if($axis===null) {
            $axis=0;
        }
        if($axis==0) {
            return $this->selectAxis0($A,$X,$Y);
        } elseif($axis==1) {
            return $this->selectAxis1($A,$X,$Y);
        } else {
            throw new InvalidArgumentException('axis must be 0 or 1');
        }
    }

    /**
     *    Y(i,j) := A(X[i],j)
     */
    protected function selectAxis0(
        NDArray $A,
        NDArray $X,
        NDArray $Y=null) : NDArray
    {
        if($X->ndim()!=1) {
            throw new InvalidArgumentException('"X" must be 1D-NDArray.');
        }
        $countX = $X->shape()[0];
        if($A->ndim()==1) {
            $shape = $X->shape();
            $m = $A->shape()[0];
            $n = 1;
        } else {
            $shape = $A->shape();
            $m = $shape[0];
            $n = (int)($A->size()/$m);
            array_shift($shape);
            array_unshift($shape,$countX);
        }
        if($Y===null) {
            $Y = $this->alloc($shape,$A->dtype());
        } else {
            if($Y->shape()!=$shape) {
                throw new InvalidArgumentException('Unmatch size "Y" with "X" and "A" .');
            }
        }

        $AA = $A->buffer();
        $offA = $A->offset();
        $ldA = $n;
        $XX = $X->buffer();
        $offX = $X->offset();
        $YY = $Y->buffer();
        $offY = $Y->offset();
        $ldY = $n;

        $this->math->selectAxis0(
            $m,
            $n,
            $countX,
            $AA,$offA,$ldA,
            $XX,$offX,1,
            $YY,$offY,$ldY);

        return $Y;
    }

    /**
     *  Y(i) := A(i,X[i])
     */
    protected function selectAxis1(
        NDArray $A,
        NDArray $X,
        NDArray $Y=null) : NDArray
    {
        if($A->ndim()!=2) {
            throw new InvalidArgumentException('"A" must be 2D-NDArray.');
        }
        if($X->ndim()!=1) {
            throw new InvalidArgumentException('"X" must be 1D-NDArray.');
        }
        [$m,$n] = $A->shape();
        if($X->size()!=$m) {
            throw new InvalidArgumentException('Unmatch size "X" with rows of "A".');
        }
        if($Y==null) {
            $Y = $this->alloc([$m],$A->dtype());
        } else {
            if($Y->ndim()!=1) {
                throw new InvalidArgumentException('"Y" must be 1D-NDArray.');
            }
            if($Y->size()!=$m) {
                throw new InvalidArgumentException('Unmatch size "Y" with rows of "A".');
            }
        }

        $AA = $A->buffer();
        $offA = $A->offset();
        $ldA = $n;
        $XX = $X->buffer();
        $offX = $X->offset();
        $YY = $Y->buffer();
        $offY = $Y->offset();

        $this->math->selectAxis1(
            $m,
            $n,
            $AA,$offA,$ldA,
            $XX,$offX,1,
            $YY,$offY,1);

        return $Y;
    }

    /**
     * A(X) := Y
     */
    public function scatter(
        NDArray $X,
        NDArray $Y,
        int $numClass,
        int $axis=null,
        NDArray $A=null) : NDArray
    {
        if($axis===null) {
            $axis=0;
        }
        if($axis==0) {
            return $this->scatterAxis0(false,$X,$Y,$numClass,$A);
        } elseif($axis==1) {
            return $this->scatterAxis1(false,$X,$Y,$numClass,$A);
        } else {
            throw new InvalidArgumentException('axis must be 0 or 1');
        }
    }

    /**
     * A(X) := Y
     */
    public function scatterAdd(
        NDArray $X,
        NDArray $Y,
        NDArray $A,
        int $axis=null) : NDArray
    {
        if($axis===null) {
            $axis=0;
        }
        if($axis==0) {
            return $this->scatterAxis0(true,$X,$Y,null,$A);
        } elseif($axis==1) {
            return $this->scatterAxis1(true,$X,$Y,null,$A);
        } else {
            throw new InvalidArgumentException('axis must be 0 or 1');
        }
    }

    /**
     * A(X[i],j) := Y[i,j]
     */
    protected function scatterAxis0(
        bool $addMode,
        NDArray $X,
        NDArray $Y,
        int $numClass=null,
        NDArray $A=null) : NDArray
    {
        if($X->ndim()!=1) {
            throw new InvalidArgumentException('"X" must be 1D-NDArray.');
        }
        $countX = $X->shape()[0];
        $shape = $Y->shape();
        $countY = array_shift($shape);
        if($countX!=$countY) {
            throw new InvalidArgumentException('Unmatch size "Y" with "X".');
        }
        $n = (int)array_product($shape);
        if($A==null) {
            $m = $numClass;
            array_unshift($shape,$numClass);
            $A = $this->alloc($shape,$Y->dtype());
            $this->zeros($A);
        } else {
            $m = $A->shape()[0];
            array_unshift($shape,$m);
            if($A->shape()!=$shape){
                throw new InvalidArgumentException('Unmatch size "Y" with "A" .');
            }
        }

        $AA = $A->buffer();
        $offA = $A->offset();
        $ldA = $n;
        $XX = $X->buffer();
        $offX = $X->offset();
        $YY = $Y->buffer();
        $offY = $Y->offset();
        $ldY = $n;

        $this->math->scatterAxis0(
            $m,
            $n,
            $countX,
            $AA,$offA,$ldA,
            $XX,$offX,1,
            $YY,$offY,$ldY,
            $addMode
            );

        return $A;
    }

    /**
     * A(i,X[i]) := Y[i]
     */
    protected function scatterAxis1(
        bool $addMode,
        NDArray $X,
        NDArray $Y,
        int $numClass=null,
        NDArray $A=null) : NDArray
    {
        if($X->ndim()!=1) {
            throw new InvalidArgumentException('"X" must be 1D-NDArray.');
        }
        if($Y->ndim()!=1) {
            throw new InvalidArgumentException('"Y" must be 1D-NDArray.');
        }
        if($X->shape()[0]!=$Y->shape()[0]) {
            throw new InvalidArgumentException('Unmatch size "X" and "Y".');
        }
        $m = $X->shape()[0];
        if($A==null) {
            if($numClass==null){
                throw new InvalidArgumentException('numClass must be specified when without target Array.');
            }
            $n = $numClass;
            $A = $this->alloc([$m,$n]);
            $this->zeros($A);
        } else {
            if($A->shape()[0]!=$m) {
                throw new InvalidArgumentException('Unmatch size "X" and "A".');
            }
            $n = $A->shape()[1];
        }

        $AA = $A->buffer();
        $offA = $A->offset();
        $ldA = $n;
        $XX = $X->buffer();
        $offX = $X->offset();
        $YY = $Y->buffer();
        $offY = $Y->offset();

        $this->math->scatterAxis1(
            $m,
            $n,
            $AA,$offA,$ldA,
            $XX,$offX,1,
            $YY,$offY,1,
            $addMode
            );

        return $A;
    }

    public function onehot(
        NDArray $X,
        int $numClass,
        float $a=null,
        NDArray $Y=null) : NDArray
    {
        if($X->ndim()!=1) {
            throw new InvalidArgumentException('"X" must be 1D-NDArray.');
        }
        $sizeX = $X->size();
        if($Y===null) {
            $Y = $this->zeros($this->alloc([$sizeX,$numClass]));
        }
        if($Y->ndim()!=2) {
            throw new InvalidArgumentException('"Y" must be 2D-NDArray.');
        }
        [$m,$n] = $Y->shape();
        if($m!=$sizeX || $n!=$numClass) {
            $shapeError = '('.implode(',',$X->shape()).'),('.implode(',',$Y->shape()).')';
            throw new InvalidArgumentException('Unmatch shape of dimension "X" and "Y" and "numClass": '.$shapeError);
        }
        if($a===null) {
            $a = 1.0;
        }
        $XX = $X->buffer();
        $offX = $X->offset();
        $YY = $Y->buffer();
        $offY = $Y->offset();
        $ldY = $n;

        $this->math->updateAddOnehot(
            $m,
            $n,
            $a,
            $XX,$offX,1,
            $YY,$offY,$ldY);

        return $Y;
    }

    /**
     *     X := softmax(X)
     */
    public function softmax(
        NDArray $X
        ) : NDArray
    {
        if($X->ndim()!=2) {
            throw new InvalidArgumentException('"X" must be 2-D dimension');
        }

        [$m,$n] = $X->shape();
        $XX = $X->buffer();
        $offX = $X->offset();
        $ldA = $n;
        $this->math->softmax(
            $m,
            $n,
            $XX,$offX,$ldA);

        return $X;
    }
/*
    public function reduceArgMax(NDArray $A,int $axis,NDArray $X=null,$dtypeX=null) : NDArray
    {
        $func = function($m,$AA,$idxA,$ldA) {
            return $this->math->imax($m,$AA,$idxA,$ldA);
        };
        if($dtypeX==null) {
            $dtypeX = NDArray::int64;
        }
        return $this->reduceWalk($func, $A, $axis, $X, $dtypeX);
    }

    public function reduceMax(NDArray $A,int $axis,NDArray $X=null,$dtypeX=null) : NDArray
    {

        $func = function($m,$AA,$idxA,$ldA) {
            $idx = $this->math->imax($m,$AA,$idxA,$ldA);
            return $AA[$idxA+$idx*$ldA];
        };
        if($dtypeX==null) {
            $dtypeX = $A->dtype();
        }
        return $this->reduceWalk($func, $A, $axis, $X, $dtypeX);
    }

    public function reduceMean(NDArray $A,int $axis,NDArray $X=null,$dtypeX=null) : NDArray
    {
        $func = function($m,$AA,$idxA,$ldA) {
            $sum = $this->math->sum($m,$AA,$idxA,$ldA);
            return $sum/$m;
        };
        if($dtypeX==null) {
            $dtypeX = $A->dtype();
        }
        return $this->reduceWalk($func, $A, $axis, $X, $dtypeX);
    }
*/
    protected function reduceWalk(
        callable $func, NDArray $A,int $axis,NDArray $X=null,$dtypeX=null) : NDArray
    {
        if($A->ndim()!=2) {
            throw new InvalidArgumentException('"A" must be 2D-NDArray.: ['.implode(',',$A->shape()).']');
        }
        if($axis === -1)
            $axis = 1;
        if($axis!=0 && $axis!=1) {
            throw new InvalidArgumentException('"axis" must be 0 or 1.');
        }
        [$m,$n] = $A->shape();
        if($X==null) {
            if($axis==0) {
                $sizeX = $n;
            } else {
                $sizeX = $m;
            }
            $X = $this->alloc([$sizeX],$dtypeX);
        }
        $AA = $A->buffer();
        $offA = $A->offset();
        $ldA = $n;
        $XX = $X->buffer();
        $offX = $X->offset();
        $incX = 1;

        if($axis==0) {
            $idxA = $offA;
            $idxX = $offX;
            for($i=0;$i<$n;$i++) {
                $XX[$idxX] = $func($m,$AA,$idxA,$ldA);
                $idxX += $incX;
                $idxA += 1;
            }
        } else {
            $idxA = $offA;
            $idxX = $offX;
            for($i=0;$i<$m;$i++) {
                $XX[$idxX] = $func($n,$AA,$idxA,1);
                $idxX += $incX;
                $idxA += $ldA;
            }
        }

        return $X;
    }


    //public function reduceSum(NDArray $A, int $axis,NDArray $X=null) : NDArray
    //{
    //    $func = function($m,$AA,$idxA,$ldA) {
    //        return $this->blas->sum($m,$AA,$idxA,$ldA);
    //    };
    //    return $this->reduceWalk($func, $A, $axis, $X);
    //}

    /**
     *    X(m) := sum( A(m,n) )
     */

    public function reduceSum(
        NDArray $A,
        int $axis=null,
        NDArray $X=null,
        $dtypeX=null) : NDArray
    {
        if($axis===null)
            $axis = 0;
        if($axis!==0 && $axis!==1 && $axis!==-1)
            throw new InvalidArgumentException('"axis" must be 0 or 1 or -1.');
        $shapeA = $A->shape();
        if($axis==0) {
            $trans = true;
            $rows = array_pop($shapeA);
        } else {
            $trans = false;
            $rows = $shapeA[0];
        }

        if($dtypeX===null) {
            $dtypeX = $A->dtype();
        }
        if($X==null) {
            $X = $this->alloc([$rows],$dtypeX);
        } else {
            if($X->shape()!=[$rows]) {
                $shapeError = '('.implode(',',$A->shape()).'),('.implode(',',$X->shape()).')';
                throw new InvalidArgumentException("Unmatch shape of dimension: ".$shapeError);
            }
        }

        $m = $A->shape()[0];
        $n = $A->size()/$m;
        $AA = $A->buffer();
        $offA = $A->offset();
        $XX = $X->buffer();
        $offX = $X->offset();

        $this->math->reduceSum(
            $trans,
            $m,
            $n,
            $AA,$offA,$n,
            $XX,$offX,1);

        return $X;
    }

    public function reduceMax(
        NDArray $A,
        int $axis,
        NDArray $X=null,
        $dtypeX=null) : NDArray
    {
        if($axis===null)
            $axis = 0;
        if($axis!==0 && $axis!==1 && $axis!==-1)
            throw new InvalidArgumentException('"axis" must be 0 or 1 or -1.');
        $shapeA = $A->shape();
        if($axis==0) {
            $trans = true;
            $rows = array_pop($shapeA);
        } else {
            $trans = false;
            $rows = $shapeA[0];
        }

        if($dtypeX===null) {
            $dtypeX = $A->dtype();
        }
        if($X==null) {
            $X = $this->alloc([$rows],$dtypeX);
        } else {
            if($X->shape()!=[$rows]) {
                $shapeError = '('.implode(',',$A->shape()).'),('.implode(',',$X->shape()).')';
                throw new InvalidArgumentException("Unmatch shape of dimension: ".$shapeError);
            }
        }

        $m = $A->shape()[0];
        $n = $A->size()/$m;
        $AA = $A->buffer();
        $offA = $A->offset();
        $XX = $X->buffer();
        $offX = $X->offset();

        $this->math->reduceMax(
            $trans,
            $m,
            $n,
            $AA,$offA,$n,
            $XX,$offX,1);

        return $X;
    }

    public function reduceArgMax(
        NDArray $A,
        int $axis,
        NDArray $X=null,
        $dtypeX=null) : NDArray
    {
        if($axis===null)
            $axis = 0;
        if($axis!==0 && $axis!==1 && $axis!==-1)
            throw new InvalidArgumentException('"axis" must be 0 or 1 or -1.');
        $shapeA = $A->shape();
        if($axis==0) {
            $trans = true;
            $rows = array_pop($shapeA);
        } else {
            $trans = false;
            $rows = $shapeA[0];
        }

        if($dtypeX==null) {
            $dtypeX = NDArray::int64;
        }
        if($X==null) {
            $X = $this->alloc([$rows],$dtypeX);
        } else {
            if($X->shape()!=[$rows]) {
                $shapeError = '('.implode(',',$A->shape()).'),('.implode(',',$X->shape()).')';
                throw new InvalidArgumentException("Unmatch shape of dimension: ".$shapeError);
            }
        }

        $m = $A->shape()[0];
        $n = $A->size()/$m;
        $AA = $A->buffer();
        $offA = $A->offset();
        $XX = $X->buffer();
        $offX = $X->offset();

        $this->math->reduceArgMax(
            $trans,
            $m,
            $n,
            $AA,$offA,$n,
            $XX,$offX,1);

        return $X;
    }

    public function reduceMean(NDArray $A,int $axis,NDArray $X=null,$dtypeX=null) : NDArray
    {
        $X = $this->reduceSum(
            $A,$axis,$X,$dtypeX
        );
        $shapeA = $A->shape();
        if($axis==0) {
            $rows = $shapeA[0];
        } else {
            $rows = array_pop($shapeA);
        }
        $this->scal(1/$rows,$X);
        return $X;
    }

    public function im2col(
        NDArray $images,
        array $filterSize=null,
        array $strides=null,
        bool $padding=null,
        bool $channels_first=null,
        bool $cols_channels_first=null,
        NDArray $cols=null
        ) : NDArray
    {
        $ndim = $images->ndim();
        if($ndim==3) {
            $_cols = $this->im2col1d(
                false,
                $images,
                $filterSize,
                $strides,
                $padding,
                $channels_first,
                $cols_channels_first,
                $cols
            );
            if($cols==null) {
                $cols = $_cols;
            }
        } elseif($ndim==4) {
            $_cols = $this->im2col2d(
                false,
                $images,
                $filterSize,
                $strides,
                $padding,
                $channels_first,
                $cols_channels_first,
                $cols
            );
            if($cols==null) {
                $cols = $_cols;
            }
        } elseif($ndim==5) {
            $_cols = $this->im2col3d(
                false,
                $images,
                $filterSize,
                $strides,
                $padding,
                $channels_first,
                $cols_channels_first,
                $cols
            );
            if($cols==null) {
                $cols = $_cols;
            }
        } else {
            throw new InvalidArgumentException('unsuppoted images shape');
        }
        return $cols;
    }

    public function col2im(
        NDArray $cols,
        NDArray $images,
        array $filterSize=null,
        array $strides=null,
        bool $padding=null,
        bool $channels_first=null,
        bool $cols_channels_first=null
        ) : NDArray
    {
        $ndim = $images->ndim();
        if($ndim==3) {
            $this->im2col1d(
                true,
                $images,
                $filterSize,
                $strides,
                $padding,
                $channels_first,
                $cols_channels_first,
                $cols
            );
        } elseif($ndim==4) {
            $this->im2col2d(
                true,
                $images,
                $filterSize,
                $strides,
                $padding,
                $channels_first,
                $cols_channels_first,
                $cols
            );
        } elseif($ndim==5) {
            $this->im2col3d(
                true,
                $images,
                $filterSize,
                $strides,
                $padding,
                $channels_first,
                $cols_channels_first,
                $cols
            );
        } else {
            throw new InvalidArgumentException('unsuppoted images shape');
        }
        return $images;
    }

    public function im2col1d(
        bool $reverse,
        NDArray $images,
        array $filterSize=null,
        array $strides=null,
        bool $padding=null,
        bool $channels_first=null,
        bool $cols_channels_first=null,
        NDArray $cols=null
        ) : NDArray
    {
        $ndim = $images->ndim();
        $images_offset = $images->offset();
        $images_size = $images->size();
        $images_buff = $images->buffer();
        if($ndim!=3) {
            throw new InvalidArgumentException('images must be 3D dimension');
        }
        if($channels_first) {
            [$batches,
             $channels,
             $in_w] =
                $images->shape();
        } else {
            [$batches,
             $in_w,
             $channels] =
                $images->shape();
        }
        if($filterSize==null) {
            $filterSize = [3];
        }
        [$filter_w] =
            $filterSize;
        if($strides==null) {
            $strides = [1];
        }
        [$stride_w] =
            $strides;
        $padding = ($padding) ? true:false;
        $channels_first = ($channels_first) ? true:false;
        $cols_channels_first = ($cols_channels_first) ? true:false;
        if($cols==null) {
            if($padding) {
                $out_w = $in_w;
            } else {
                $out_w = intval(floor(($in_w-$filter_w)/$stride_w)+1);
            }
            if($cols_channels_first) {
                $cols = $this->alloc([
                    $batches,$out_w,
                    $channels,$filter_w
                ]);
                $this->zeros($cols);
            } else {
                $cols = $this->alloc([
                    $batches,$out_w,
                    $filter_w,$channels
                ]);
                $this->zeros($cols);
            }
        }
        $out = $cols->buffer();
        $out_offset = $cols->offset();
        $out_size = $cols->size();
        $this->math->im2col1d(
            $reverse,
            $images_buff,
            $images_offset,
            $images_size,
            $batches,
            $in_w,
            $channels,
            $filter_w,
            $stride_w,
            $padding,
            $channels_first,
            $cols_channels_first,
            $out,
            $out_offset,
            $out_size
        );
        return $cols;
    }

    public function im2col2d(
        bool $reverse,
        NDArray $images,
        array $filterSize=null,
        array $strides=null,
        bool $padding=null,
        bool $channels_first=null,
        bool $cols_channels_first=null,
        NDArray $cols=null
        ) : NDArray
    {
        $ndim = $images->ndim();
        $images_offset = $images->offset();
        $images_size = $images->size();
        $images_buff = $images->buffer();
        if($ndim!=4) {
            throw new InvalidArgumentException('images must be 4D dimension');
        }
        if($channels_first) {
            [$batches,
             $channels,
             $in_h,$in_w] =
                $images->shape();
        } else {
            [$batches,
             $in_h,$in_w,
             $channels] =
                $images->shape();
        }
        if($filterSize==null) {
            $filterSize = [3,3];
        }
        [$filter_h,$filter_w] =
            $filterSize;
        if($strides==null) {
            $strides = [1,1];
        }
        [$stride_h,$stride_w] =
            $strides;
        $padding = ($padding) ? true:false;
        $channels_first = ($channels_first) ? true:false;
        $cols_channels_first = ($cols_channels_first) ? true:false;
        if($cols==null) {
            if($padding) {
                $out_h = $in_h;
                $out_w = $in_w;
            } else {
                $out_h = intval(floor(($in_h-$filter_h)/$stride_h)+1);
                $out_w = intval(floor(($in_w-$filter_w)/$stride_w)+1);
            }
            if($cols_channels_first) {
                $cols = $this->alloc([
                    $batches,$out_h,$out_w,
                    $channels,$filter_h,$filter_w
                ]);
                $this->zeros($cols);
            } else {
                $cols = $this->alloc([
                    $batches,$out_h,$out_w,
                    $filter_h,$filter_w,$channels
                ]);
                $this->zeros($cols);
            }
        }
        $out = $cols->buffer();
        $out_offset = $cols->offset();
        $out_size = $cols->size();
        $this->math->im2col2d(
            $reverse,
            $images_buff,
            $images_offset,
            $images_size,
            $batches,
            $in_h,
            $in_w,
            $channels,
            $filter_h,
            $filter_w,
            $stride_h,
            $stride_w,
            $padding,
            $channels_first,
            $cols_channels_first,
            $out,
            $out_offset,
            $out_size
        );
        return $cols;
    }

    public function im2col3d(
        bool $reverse,
        NDArray $images,
        array $filterSize=null,
        array $strides=null,
        bool $padding=null,
        bool $channels_first=null,
        bool $cols_channels_first=null,
        NDArray $cols=null
        ) : NDArray
    {
        $ndim = $images->ndim();
        $images_offset = $images->offset();
        $images_size = $images->size();
        $images_buff = $images->buffer();
        if($ndim!=5) {
            throw new InvalidArgumentException('images must be 5D dimension');
        }
        if($channels_first) {
            [$batches,
             $channels,
             $in_d,$in_h,$in_w] =
                $images->shape();
        } else {
            [$batches,
             $in_d,$in_h,$in_w,
             $channels] =
                $images->shape();
        }
        if($filterSize==null) {
            $filterSize = [3,3,3];
        }
        [$filter_d,$filter_h,$filter_w] =
            $filterSize;
        if($strides==null) {
            $strides = [1,1,1];
        }
        [$stride_d,$stride_h,$stride_w] =
            $strides;
        $padding = ($padding) ? true:false;
        $channels_first = ($channels_first) ? true:false;
        $cols_channels_first = ($cols_channels_first) ? true:false;
        if($cols==null) {
            if($padding) {
                $out_d = $in_d;
                $out_h = $in_h;
                $out_w = $in_w;
            } else {
                $out_d = intval(floor(($in_d-$filter_d)/$stride_d)+1);
                $out_h = intval(floor(($in_h-$filter_h)/$stride_h)+1);
                $out_w = intval(floor(($in_w-$filter_w)/$stride_w)+1);
            }
            if($cols_channels_first) {
                $cols = $this->alloc([
                    $batches,$out_d,$out_h,$out_w,
                    $channels,$filter_d,$filter_h,$filter_w
                ]);
                $this->zeros($cols);
            } else {
                $cols = $this->alloc([
                    $batches,$out_d,$out_h,$out_w,
                    $filter_d,$filter_h,$filter_w,$channels
                ]);
                $this->zeros($cols);
            }
        }
        $out = $cols->buffer();
        $out_offset = $cols->offset();
        $out_size = $cols->size();
        $this->math->im2col3d(
            $reverse,
            $images_buff,
            $images_offset,
            $images_size,
            $batches,
            $in_d,
            $in_h,
            $in_w,
            $channels,
            $filter_d,
            $filter_h,
            $filter_w,
            $stride_d,
            $stride_h,
            $stride_w,
            $padding,
            $channels_first,
            $cols_channels_first,
            $out,
            $out_offset,
            $out_size
        );
        return $cols;
    }

    public function randomUniform(
        array $shape,
        $low,
        $high,
        $dtype=null,
        int $seed=null,
        NDArray $X=null) : NDArray
    {
        if($dtype!==null&&$X!==null) {
            if ($X->dtype()!=$dtype) {
                throw new InvalidArgumentException('Unmatch dtype and dtype of X');
            }
        }
        if($X===null) {
            $X = $this->alloc($shape,$dtype);
        } else {
            if ($X->shape()!=$shape) {
                throw new InvalidArgumentException('Unmatch shape and shape of X');
            }
            if(!is_numeric($low)||!is_numeric($high)){
                throw new InvalidArgumentException('low and high must be integer or float');
            }
        }
        if($seed===null) {
            $seed = random_int(~PHP_INT_MAX,PHP_INT_MAX);
        }

        $n = $X->size();
        $XX = $X->buffer();
        $offX = $X->offset();

        $this->math->randomUniform(
            $n,
            $XX,$offX,1,
            $low,
            $high,
            $seed);

        return $X;
    }

    public function randomNormal(
        array $shape,
        $mean,
        $scale,
        $dtype=null,
        int $seed=null,
        NDArray $X=null) : NDArray
    {
        if($dtype!==null&&$X!==null) {
            if ($X->dtype()!=$dtype) {
                throw new InvalidArgumentException('Unmatch dtype and dtype of X');
            }
        }
        if($X===null) {
            $X = $this->alloc($shape,$dtype);
        } else {
            if ($X->shape()!=$shape) {
                throw new InvalidArgumentException('Unmatch shape and shape of X');
            }
            if(!is_numeric($low)||!is_numeric($high)){
                throw new InvalidArgumentException('low and high must be integer or float');
            }
        }
        if($seed===null) {
            $seed = random_int(~PHP_INT_MAX,PHP_INT_MAX);
        }

        $n = $X->size();
        $XX = $X->buffer();
        $offX = $X->offset();

        $this->math->randomNormal(
            $n,
            $XX,$offX,1,
            $mean,
            $scale,
            $seed);

        return $X;
    }

    public function randomSequence(
        int $base,
        int $size=null,
        int $seed=null
        ) : NDArray
    {
        if($size==null) {
            $size = $base;
        }
        $X = $this->alloc([$base],NDArray::int64);
        if($seed===null) {
            $seed = random_int(~PHP_INT_MAX,PHP_INT_MAX);
        }

        $n = $base;
        $XX = $X->buffer();
        $offX = $X->offset();

        $this->math->randomSequence(
            $n,
            $size,
            $XX,$offX,1,
            $seed);
        $X = $X[[0,$size-1]];
        return $X;
    }

    public function slice(
        NDArray $input,
        array $begin,
        array $size,
        NDArray $output=null
        ) : NDArray
    {
        return $this->doSlice(
            false,
            $input,
            $begin,
            $size,
            $output
        );
    }

    public function stick(
        NDArray $input,
        NDArray $output,
        array $begin,
        array $size
        ) : NDArray
    {
        return $this->doSlice(
            true,
            $output,
            $begin,
            $size,
            $input
        );
    }

    public function stack(
        array $values,
        int $axis=null
    )
    {
        if($axis==null){
            $axis=0;
        }
        if($axis==0){
            $m = count($values);
            $shape = $values[0]->shape();
            array_unshift($shape,$m);
            $output = $this->alloc($shape,$values[0]->dtype());
            $i = 0;
            foreach($values as $value){
                if(!($value instanceof NDArray)) {
                    throw new InvalidArgumentException('values must be array of NDArray');
                }
                $shape = $value->shape();
                array_unshift($shape,1);
                $value = $value->reshape(
                    $shape);
                $this->doSlice(true,
                    $output,
                    [$i],[1],
                    $value
                );
                $i++;
            }
        } elseif($axis==1){
            $n = count($values);
            $shape = $values[0]->shape();
            $m = array_shift($shape);
            array_unshift($shape,$n);
            array_unshift($shape,$m);
            $output = $this->alloc($shape,$values[0]->dtype());
            $i = 0;
            foreach($values as $value){
                if(!($value instanceof NDArray)) {
                    throw new InvalidArgumentException('values must be array of NDArray');
                }
                $shape = $value->shape();
                $m = array_shift($shape);
                array_unshift($shape,1);
                array_unshift($shape,$m);
                $value = $value->reshape(
                    $shape);
                $this->doSlice(true,
                    $output,
                    [0,$i],[-1,1],
                    $value
                );
                $i++;
            }
        } else {
            throw new InvalidArgumentException('unsuppoted axis');
        }
        return $output;
    }

    public function concat(
        array $values,
        int $axis=null
    ) : NDArray
    {
        if($axis===null) {
            $axis = -1;
        }
        if($axis<0) {
            $axis = $values[0]->ndim() + $axis;
        }
        $m = null;
        $base = null;
        $n = 0;
        $reshapeValues = [];
        foreach ($values as $value) {
            $shapePrefix = [];
            $shape = $value->shape();
            $mm = 1;
            for($j=0;$j<$axis;$j++) {
                $mmm = array_shift($shape);
                $shapePrefix[] = $mmm;
                $mm *= $mmm;
            }
            $nn = array_shift($shape);
            if($base===null) {
                $m = $mm;
                $base = $shape;
            } else {
                if($m!=$mm||$base!=$shape) {
                    throw new InvalidArgumentException('Unmatch shape: '.
                        $this->printableShapes($values));
                    }
            }
            $n += $nn;
            $reshapeValues[] = $value->reshape(array_merge([$mm,$nn],$shape));
        }
        $dims = $shape;
        $shape = array_merge([$m,$n],$shape);
        $output = $this->alloc($shape,$values[0]->dtype());
        $i = 0;
        foreach ($reshapeValues as $value) {
            $nn = $value->shape()[1];
            $this->doSlice(true,
                $output,
                [0,$i],[-1,$nn],
                $value
            );
            $i += $nn;
        }
        $output = $output->reshape(array_merge($shapePrefix,[$n],$dims));
        return $output;
    }

    public function split(
        NDArray $input, array $sizeSplits, $axis=null
        ) : array
    {
        if($axis===null) {
            $axis = -1;
        }
        if($axis<0) {
            $axis = $input->ndim() + $axis;
        }
        $shapePrefix = [];
        $shape = $input->shape();
        $m = 1;
        for($j=0;$j<$axis;$j++) {
            $mmm = array_shift($shape);
            $shapePrefix[] = $mmm;
            $m *= $mmm;
        }
        $n = array_shift($shape);
        $input = $input->reshape(array_merge([$m,$n],$shape));
        $i = 0;
        foreach ($sizeSplits as $size) {
            $outputs[] = $this->doSlice(false,
                $input,
                [0,$i],[-1,$size]
            )->reshape(array_merge($shapePrefix,[$size],$shape));
            $i += $size;
        }
        return $outputs;
    }

    protected function doSlice(
        bool $reverse,
        NDArray $input,
        array $begin,
        array $size,
        NDArray $output=null
        ) : NDArray
    {
        if(!$reverse){
            $messageInput='Input';
        } else {
            $messageInput='Output';
        }
        $orgBegin = $begin;
        $orgSize = $size;
        $ndimBegin = count($begin);
        if($ndimBegin<1||$ndimBegin>2) {
            throw new InvalidArgumentException('begin must has 1 or 2 integer.');
        }
        $ndimSize = count($size);
        if($ndimSize<1||$ndimSize>2) {
            throw new InvalidArgumentException('Size must has 1 or 2 integer.');
        }
        if($ndimBegin!=$ndimSize){
            throw new InvalidArgumentException('Unmatch shape of begin and size');
        }
        $ndimInput = $input->ndim();
        if($ndimInput<$ndimBegin){
            throw new InvalidArgumentException($messageInput.' shape rank is low to slice');
        }
        $shape = $input->shape();
        $m = array_shift($shape);
        $startAxis0 = array_shift($begin);
        if($startAxis0<0){
            $startAxis0 = $m+$startAxis0;
        }
        if($startAxis0<0||$startAxis0>=$m){
            throw new InvalidArgumentException('start of axis 0 is invalid value.');
        }
        $sizeAxis0 = array_shift($size);
        if($sizeAxis0<0){
            $sizeAxis0 = $m-$startAxis0+$sizeAxis0+1;
        }
        if($sizeAxis0<1||$startAxis0+$sizeAxis0>$m){
            throw new InvalidArgumentException('size of axis 0 is invalid value.');
        }
        if($ndimBegin==1){
            $n = 1;
            $startAxis1 = 0;
            $sizeAxis1 = 1;
        } else {
            $n = array_shift($shape);
            $startAxis1 = array_shift($begin);
            if($startAxis1<0){
                $startAxis1 = $n+$startAxis1;
            }
            if($startAxis1<0||$startAxis1>=$n){
                throw new InvalidArgumentException('start of axis 1 is invalid value.:begin=['.implode(',',$orgBegin).']');
            }
            $sizeAxis1 = array_shift($size);
            if($sizeAxis1<0){
                $sizeAxis1 = $n-$startAxis1+$sizeAxis1+1;
            }
            if($sizeAxis1<1||$startAxis1+$sizeAxis1>$n){
                throw new InvalidArgumentException('size of axis 1 is invalid value.');
            }
        }
        $k = array_product($shape);
        $outputShape = [$sizeAxis0];
        if($ndimBegin==2){
            array_push($outputShape,
                $sizeAxis1);
        }
        $outputShape = array_merge(
            $outputShape,$shape);
        if($output==null){
            $output = $this->alloc($outputShape,$input->dtype());
        }else{
            if($outputShape!=$output->shape()){
                throw new InvalidArgumentException('Unmatch output shape');
            }
        }

        $A = $input->buffer();
        $offsetA = $input->offset();
        $Y = $output->buffer();
        $offsetY = $output->offset();
        $incA = 1;
        $incY = 1;
        $this->math->slice(
            $reverse,
            $addMode=false,
            $m,
            $n,
            $k,
            $A,$offsetA,$incA,
            $Y,$offsetY,$incY,
            $startAxis0,$sizeAxis0,
            $startAxis1,$sizeAxis1
        );
        return $output;
    }

    //
    // repeat
    //
    public function repeat(NDArray $A, int $repeats)
    {
        if($repeats<1) {
            throw new InvalidArgumentException('repeats argument must be one or greater.');
        }
        if($A->ndim()<2) {
            throw new InvalidArgumentException('dimension rank must be two or greater.');
        }
        $shapeCell = $A->shape();
        $s1 = array_shift($shapeCell);
        $shape = array_merge([$s1,$repeats],$shapeCell);
        $B = $this->alloc($shape,$A->dtype());
        $m = $s1;
        $n = $repeats;
        $k = (int)array_product($shapeCell);
        $AA = $A->buffer();
        $offA = $A->offset();
        $BB = $B->buffer();
        $offB = $B->offset();
        $startAxis0 = 0;
        $sizeAxis0 = $m;
        for($i=0;$i<$repeats;$i++) {
            $startAxis1 = $i;
            $sizeAxis1 = 1;
            $this->math->slice(
                $reverse=true,
                $addMode=false,
                $m,
                $n,
                $k,
                $BB,$offB,1,
                $AA,$offA,1,
                $startAxis0,$sizeAxis0,
                $startAxis1,$sizeAxis1
            );
        }
        return $B;
    }

    //
    // repeat
    //
    public function reduceSumRepeated(NDArray $A)
    {
        if($A->ndim()<3) {
            throw new InvalidArgumentException('dimension rank must be two or greater.');
        }
        $shapeCell = $A->shape();
        $s1 = array_shift($shapeCell);
        $repeats = array_shift($shapeCell);
        $shape = array_merge([$s1],$shapeCell);
        $B = $this->alloc($shape,$A->dtype());
        $this->zeros($B);
        $m = $s1;
        $n = $repeats;
        $k = (int)array_product($shapeCell);
        $AA = $A->buffer();
        $offA = $A->offset();
        $BB = $B->buffer();
        $offB = $B->offset();
        $startAxis0 = 0;
        $sizeAxis0 = $m;
        for($i=0;$i<$repeats;$i++) {
            $startAxis1 = $i;
            $sizeAxis1 = 1;
            $this->math->slice(
                $reverse=false,
                $addMode=true,
                $m,
                $n,
                $k,
                $AA,$offA,1,
                $BB,$offB,1,
                $startAxis0,$sizeAxis0,
                $startAxis1,$sizeAxis1
            );
        }
        return $B;
    }

    public function svd(NDArray $matrix,$fullMatrices=null)
    {
        if($matrix->ndim()!=2) {
            throw new InvalidArgumentException("input array must be 2D array");
        }
        if($fullMatrices===null)
            $fullMatrices = true;
        [$m,$n] = $matrix->shape();
        if($fullMatrices) {
            $jobu  = 'A';
            $jobvt = 'A';
            $ldA = $n;
            $ldU = $m;
            $ldVT = $n;
        } else {
            $jobu  = 'S';
            $jobvt = 'S';
            $ldA = $n;
            $ldU = min($m,$n);
            #$ldVT = min($m,$n);
            $ldVT = $n; // bug in the lapacke ???
        }

        $S = $this->alloc([min($m,$n)],$matrix->dtype());
        $this->zeros($S);
        $U = $this->alloc([$m,$ldU],$matrix->dtype());
        $this->zeros($U);
        $VT = $this->alloc([$ldVT,$n],$matrix->dtype());
        $this->zeros($VT);
        $SuperB = $this->alloc([min($m,$n)-1],$matrix->dtype());
        $this->zeros($SuperB);

        $AA = $matrix->buffer();
        $offsetA = $matrix->offset();
        $SS = $S->buffer();
        $offsetS = $S->offset();
        $UU = $U->buffer();
        $offsetU = $U->offset();
        $VVT = $VT->buffer();
        $offsetVT = $VT->offset();
        $SuperBB = $SuperB->buffer();
        $offsetSuperB = $SuperB->offset();
        $this->lapack->gesvd(
            LAPACK_ROW_MAJOR,
            ord($jobu),
            ord($jobvt),
            $m,
            $n,
            $AA,  $offsetA,  $ldA,
            $SS,  $offsetS,
            $UU,  $offsetU,  $ldU,
            $VVT, $offsetVT, $ldVT,
            $SuperBB,  $offsetSuperB
        );
        if(!$fullMatrices) {
            // bug in the lapacke ???
            $VT = $this->copy($VT[[0,min($m,$n)-1]]);
        }
        return [$U,$S,$VT];
    }

    public function numericalGradient(
        float $h=null,
        $f=null,
        NDArray ...$variables) : array
    {
        if($h===null)
            $h = 1e-4;
        if(!is_callable($f)) {
            throw new InvalidArgumentException("f must callable or array of f and h");
        }
        $grads = [];
        $orgVariables = $variables;
        $variables = [];
        foreach ($orgVariables as $variable) {
            $variables[] = $this->copy($variable);
        }
        foreach($variables as $x) {
            $grad = $this->alloc($x->shape(),$x->dtype());
            $this->zeros($grad);
            $grads[] = $grad;
            $size = $x->size();
            $xx = $x->buffer();
            $idx = $x->offset();
            $gg = $grad->buffer();
            $gidx = $grad->offset();
            $h2 = $h*2 ;
            for($i=0;$i<$size;$i++,$idx++,$gidx++) {
                $value = $xx[$idx];
                $xx[$idx] = $value + $h;
                $y1 = $f(...$variables);
                $xx[$idx] = $value - $h;
                $y2 = $f(...$variables);
                $d = $this->axpy($y2,$this->copy($y1),-1);
                $gg[$gidx] = $this->sum($d)/$h2;
                $xx[$idx] = $value;
            }
        }
        return $grads;
    }

    public function isclose(NDArray $a, NDArray $b, $rtol=null, $atol=null)
    {
        if($rtol===null)
            $rtol = 1e-04;
        if($atol===null)
            $atol = 1e-07;
        if($a->shape()!=$b->shape()) {
            return false;
        }
        // diff = b - a
        $diff = $this->axpy($a,$this->copy($b),-1);
        // close = atol + rtol * b
        $close = $atol+abs($this->amax($this->scal($rtol,$this->copy($b))));
        if(abs($this->amax($diff)) > $close) {
            return false;
        }
        return true;
    }
}
