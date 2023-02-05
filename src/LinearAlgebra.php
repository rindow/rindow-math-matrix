<?php
namespace Rindow\Math\Matrix;

use Interop\Polite\Math\Matrix\BLAS;
use Interop\Polite\Math\Matrix\NDArray;
use InvalidArgumentException;
use ArrayAccess as Buffer;

class LinearAlgebra
{
    const LAPACK_ROW_MAJOR = 101;
    const LAPACK_COL_MAJOR = 102;

    protected $iaminwarning;
    protected $blas;
    protected $lapack;
    protected $math;
    protected $defaultFloatType = NDArray::float32;
    protected $intTypes= [
        NDArray::int8,NDArray::int16,NDArray::int32,NDArray::int64,
        NDArray::uint8,NDArray::uint16,NDArray::uint32,NDArray::uint64,
    ];

    public function __construct($blas,$lapack,$math,$defaultFloatType=null)
    {
        $this->blas = $blas;
        $this->lapack = $lapack;
        $this->math = $math;
        if($defaultFloatType!==null)
            $this->defaultFloatType = $defaultFloatType;
    }

    public function getBlas()
    {
        return $this->blas;
    }

    public function getMath()
    {
        return $this->math;
    }

    public function getConfig()
    {
        return $this->blas->getConfig();
    }

    public function fp64() : bool
    {
        return true;
    }

    public function accelerated() : bool
    {
        return false;
    }

    public function finish()
    {
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

    public function isInt(NDArray $value)
    {
        return in_array($value->dtype(),$this->intTypes);
    }

    public function array($array,$dtype=null) : NDArray
    {
        if($array instanceof NDArray) {
            return $array;
        } elseif(is_array($array) || is_numeric($array)) {
            return new NDArrayPhp($array,$dtype);
        } else {
            throw new InvalidArgumentException('input value must be NDArray or array');
        }
    }

    public function toNDArray(NDArray $ndarray) : NDArray
    {
        if(!($ndarray instanceof NDArrayPhp)) {
            throw new InvalidArgumentException('input value must be NDArrayPhp');
        }
        return $ndarray;
    }

    public function scalar($array)
    {
        if($array instanceof NDArray) {
            return $array->toArray();
        }
        return $array;
    }

    public function expandDims(NDArray $x, int $axis)
    {
        $shape = $x->shape();
        $ndim = count($shape);
        $orgAxis = $axis;
        if($axis<0) {
            $axis = $ndim + $axis + 1;
        }
        if($axis<0||$axis>$ndim) {
            throw new InvalidArgumentException('axis is out of range: '.$orgAxis);
        }
        $newShape = [];
        $i = 0;
        foreach ($shape as $n) {
            if($i==$axis) {
                $newShape[] = 1;
            }
            $newShape[] = $n;
            $i++;
        }
        if($i==$axis) {
            $newShape[] = 1;
        }
        return $x->reshape($newShape);
    }

    public function squeeze(NDArray $x, int $axis=null)
    {
        $shape = $x->shape();
        if($axis===null) {
            $newShape = [];
            foreach ($shape as $n) {
                if($n!=1) {
                    $newShape[] = $n;
                }
            }
            return $x->reshape($newShape);
        }
        $ndim = count($shape);
        $orgAxis = $axis;
        if($axis<0) {
            $axis = $ndim + $axis;
        }
        if($axis<0||$axis>=$ndim) {
            throw new InvalidArgumentException('axis is out of range: '.$orgAxis);
        }
        $newShape = [];
        $i = 0;
        foreach ($shape as $n) {
            if($i!=$axis) {
                $newShape[] = $n;
            } else {
                if($n!=1) {
                    throw new InvalidArgumentException('Can not squeeze dim['.$axis.']');
                }
            }
            $i++;
        }
        return $x->reshape($newShape);
    }

    public function alloc(array $shape,$dtype=null) : NDArray
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

    public function ones(
        NDArray $X) : NDArray
    {
        $this->fill(1,$X);
        return $X;
    }

    public function zerosLike(NDArray $array) : NDArray
    {
        $newArray = $this->alloc($array->shape(),$array->dtype());
        $this->zeros($newArray);
        return $newArray;
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
    *    a,b,c,s = rotg(x,y)
    */
    public function rotg(
        NDArray $X,
        NDArray $Y,
        NDArray $R=null,
        NDArray $Z=null,
        NDArray $C=null,
        NDArray $S=null) : array
    {
        if($X->shape()!=$Y->shape()) {
            $shapeError = '('.implode(',',$X->shape()).'),('.implode(',',$Y->shape()).')';
            throw new InvalidArgumentException("Unmatch shape of dimension: ".$shapeError);
        }
        $R = $this->copy($X,$R);
        $Z = $this->copy($Y,$Z);
        if($C==null) {
            $C = $this->alloc($X->shape(),$X->dtype());
        }
        if($S==null) {
            $S = $this->alloc($Y->shape(),$X->dtype());
        }
        $AA = $R->buffer();
        $offA = $R->offset();
        $BB = $Z->buffer();
        $offB = $Z->offset();
        $CC = $C->buffer();
        $offC = $C->offset();
        $SS = $S->buffer();
        $offS = $S->offset();
        $this->blas->rotg(
            $AA,$offA,
            $BB,$offB,
            $CC,$offC,
            $SS,$offS
        );
        return [$R,$Z,$C,$S];
    }

    /**
    *    x,y := rot(x,y,c,s)
    */
    public function rot(
        NDArray $X,
        NDArray $Y,
        NDArray $C,
        NDArray $S) : void
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
        $CC = $C->buffer();
        $offC = $C->offset();
        $SS = $S->buffer();
        $offS = $S->offset();
        $this->blas->rot($N,
            $XX,$offX,1,$YY,$offY,1,
            $CC,$offC,$SS,$offS
        );
    }

    /**
    *    Y := X
    *    X := Y
    */
    public function swap(
        NDArray $X,
        NDArray $Y) : void
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
        $this->blas->swap($N,$XX,$offX,1,$YY,$offY,1);
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
            $Y = $this->zeros($this->alloc([$rows],$X->dtype()));
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
            $C = $this->zeros($this->alloc([$M,$N],$A->dtype()));
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
    *    A = Symmetric matrix
    *    C := alpha * AB + beta * C  ( right = false )
    *    C := alpha * BA + beta * C  ( right = true )
    */
    public function symm(
        NDArray $A,
        NDArray $B,
        float $alpha=null,
        float $beta=null,
        NDArray $C=null,
        bool $right=null,
        bool $lower=null
        ) : NDArray
    {
        if($A->ndim()!=2 || $B->ndim()!=2) {
            throw new InvalidArgumentException('Dimensions must be 2D-NDArray');
        }
        $shapeA = $A->shape();
        $rowsA = $shapeA[0];
        if($rowsA!=$shapeA[1]) {
            throw new InvalidArgumentException('The matrix "A" must be symmetric');
        }
        $shapeB = $B->shape();
        $M = $shapeB[0];
        $N = $shapeB[1];
        $tmpB = ($right) ? $N : $M;
        if($rowsA!=$tmpB) {
            throw new InvalidArgumentException('Unmatch Shape of matrix "A" and "B": '."($rowsA,$rowsA) != ($M,$N)");
        }
        $AA = $A->buffer();
        $BB = $B->buffer();
        $offA = $A->offset();
        $offB = $B->offset();

        if($alpha===null) {
            $alpha = 1.0;
        }
        if($beta===null) {
            $beta = 0.0;
        }
        if($C!=null) {
            $shapeC = $C->shape();
            if($M!=$shapeC[0] || $N!=$shapeC[1]) {
                throw new InvalidArgumentException('Matrix "B" and "C" must be same shape');
            }
        } else {
            $C = $this->zeros($this->alloc([$M,$N],$A->dtype()));
        }
        $CC = $C->buffer();
        $offC = $C->offset();

        $lda = $rowsA;
        $ldb = $N;
        $ldc = $N;
        $side = ($right) ? BLAS::Right : BLAS::Left;
        $uplo = ($lower) ? BLAS::Lower : BLAS::Upper;

        $this->blas->symm(
            BLAS::RowMajor,$side,$uplo,
            $M,$N,
            $alpha,
            $AA,$offA,$lda,
            $BB,$offB,$ldb,
            $beta,
            $CC,$offC,$ldc);

        return $C;
    }

    /**
    *    C := alpha * A A^T + beta * C  (trans=false)
    *    C := alpha * A^T A + beta * C  (trans=true)
    */
    public function syrk(
        NDArray $A,
        float $alpha=null,
        float $beta=null,
        NDArray $C=null,
        bool $lower=null,
        bool $trans=null) : NDArray
    {
        if($A->ndim()!=2) {
            throw new InvalidArgumentException('Dimensions must be 2D-NDArray');
        }
        $shapeA = $A->shape();
        if($trans) {
            $shapeA = [$shapeA[1],$shapeA[0]];
        }
        $AA = $A->buffer();
        $offA = $A->offset();
        $N = $shapeA[0];
        $K = $shapeA[1];

        if($alpha===null) {
            $alpha = 1.0;
        }
        if($beta===null) {
            $beta = 0.0;
        }
        if($C!=null) {
            $shapeC = $C->shape();
            if($N!=$shapeC[0] || $N!=$shapeC[1]) {
                throw new InvalidArgumentException('"C" rows and cols must have the same number of "A" cols');
            }
        } else {
            $C = $this->zeros($this->alloc([$N,$N],$A->dtype()));
        }
        $CC = $C->buffer();
        $offC = $C->offset();

        $lda = ($trans) ? $N : $K;
        $ldc = $N;
        $uplo  = ($lower) ? BLAS::Lower : BLAS::Upper;
        $trans = ($trans) ? BLAS::Trans : BLAS::NoTrans;

        $this->blas->syrk(
            BLAS::RowMajor,$uplo,$trans,
            $N,$K,
            $alpha,
            $AA,$offA,$lda,
            $beta,
            $CC,$offC,$ldc);

        return $C;
    }

    /**
    *    C := alpha * A B^T + alpha * B A^T + beta * C  (trans=false)
    *    C := alpha * B A^T + alpha * A B^T + beta * C  (trans=true)
    */
    public function syr2k(
        NDArray $A,
        NDArray $B,
        float $alpha=null,
        float $beta=null,
        NDArray $C=null,
        bool $lower=null,
        bool $trans=null) : NDArray
    {
        if($A->ndim()!=2 || $B->ndim()!=2) {
            throw new InvalidArgumentException('Dimensions must be 2D-NDArray');
        }
        $shapeA = $A->shape();
        $shapeB = $B->shape();
        if($shapeA!=$shapeB) {
            throw new InvalidArgumentException('Matrix A and B must be same shape');
        }
        if($trans) {
            $shapeA = [$shapeA[1],$shapeA[0]];
        }
        $AA   = $A->buffer();
        $offA = $A->offset();
        $BB   = $B->buffer();
        $offB = $B->offset();
        $N = $shapeA[0];
        $K = $shapeA[1];

        if($alpha===null) {
            $alpha = 1.0;
        }
        if($beta===null) {
            $beta = 0.0;
        }
        if($C!=null) {
            $shapeC = $C->shape();
            if($N!=$shapeC[0] || $N!=$shapeC[1]) {
                throw new InvalidArgumentException('"C" rows and cols must have the same number of "A" cols');
            }
        } else {
            $C = $this->zeros($this->alloc([$N,$N],$A->dtype()));
        }
        $CC = $C->buffer();
        $offC = $C->offset();

        $lda = ($trans) ? $N : $K;
        $ldb = ($trans) ? $N : $K;
        $ldc = $N;
        $uplo  = ($lower) ? BLAS::Lower : BLAS::Upper;
        $trans = ($trans) ? BLAS::Trans : BLAS::NoTrans;

        $this->blas->syr2k(
            BLAS::RowMajor,$uplo,$trans,
            $N,$K,
            $alpha,
            $AA,$offA,$lda,
            $BB,$offB,$ldb,
            $beta,
            $CC,$offC,$ldc);

        return $C;
    }

    /**
    *    C := alpha * A B  (right=false)
    *    C := alpha * B A  (right=true)
    */
    public function trmm(
        NDArray $A,
        NDArray $B,
        float $alpha=null,
        bool $right=null,
        bool $lower=null,
        bool $trans=null,
        bool $unit=null) : NDArray
    {
        if($A->ndim()!=2 || $B->ndim()!=2) {
            throw new InvalidArgumentException('Dimensions must be 2D-NDArray');
        }
        $shapeA = $A->shape();
        $shapeB = $B->shape();
        if($right) {
            $sizeA = $shapeB[1];
        } else {
            $sizeA = $shapeB[0];
        }
        if($sizeA!=$shapeA[0]) {
            throw new InvalidArgumentException('Unmatch shape of Matrix A and B: '.
                '['.implode(',',$shapeA).'] <=> ['.implode(',',$shapeA).']');
        }
        $AA   = $A->buffer();
        $offA = $A->offset();
        $BB   = $B->buffer();
        $offB = $B->offset();
        $M = $shapeB[0];
        $N = $shapeB[1];

        if($alpha===null) {
            $alpha = 1.0;
        }

        $lda = ($right) ? $N : $M;
        $ldb = $N;
        $side  = ($right) ? BLAS::Right : BLAS::Left;
        $uplo  = ($lower) ? BLAS::Lower : BLAS::Upper;
        $trans = ($trans) ? BLAS::Trans : BLAS::NoTrans;
        $diag  = ($unit)  ? BLAS::Unit  : BLAS::NonUnit;

        $this->blas->trmm(
            BLAS::RowMajor,$side,$uplo,$trans,$diag,
            $M,$N,
            $alpha,
            $AA,$offA,$lda,
            $BB,$offB,$ldb);

        return $B;
    }

    /**
    *    C := alpha A^-1 B  (right=false)
    *    C := alpha B A^-1  (right=true)
    */
    public function trsm(
        NDArray $A,
        NDArray $B,
        float $alpha=null,
        bool $right=null,
        bool $lower=null,
        bool $trans=null,
        bool $unit=null) : NDArray
    {
        if($A->ndim()!=2 || $B->ndim()!=2) {
            throw new InvalidArgumentException('Dimensions must be 2D-NDArray');
        }
        $shapeA = $A->shape();
        $shapeB = $B->shape();
        if($right) {
            $sizeA = $shapeB[1];
        } else {
            $sizeA = $shapeB[0];
        }
        if($sizeA!=$shapeA[0]) {
            throw new InvalidArgumentException('Unmatch shape of Matrix A and B: '.
                '['.implode(',',$shapeA).'] <=> ['.implode(',',$shapeA).']');
        }
        $AA   = $A->buffer();
        $offA = $A->offset();
        $BB   = $B->buffer();
        $offB = $B->offset();
        $M = $shapeB[0];
        $N = $shapeB[1];

        if($alpha===null) {
            $alpha = 1.0;
        }

        $lda = ($right) ? $N : $M;
        $ldb = $N;
        $side  = ($right) ? BLAS::Right : BLAS::Left;
        $uplo  = ($lower) ? BLAS::Lower : BLAS::Upper;
        $trans = ($trans) ? BLAS::Trans : BLAS::NoTrans;
        $diag  = ($unit)  ? BLAS::Unit  : BLAS::NonUnit;

        $this->blas->trsm(
            BLAS::RowMajor,$side,$uplo,$trans,$diag,
            $M,$N,
            $alpha,
            $AA,$offA,$lda,
            $BB,$offB,$ldb);

        return $B;
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

    protected function calcBroadcastFormat($A,$X)
    {
        if(is_numeric($X)) {
            $X = $this->array($X,$A->dtype());
        }
        if(!($X instanceof NDArray)) {
            throw new InvalidArgumentException('X must be NDArray or float');;
        }
        $ndimX = $X->ndim();
        $ndimA = $A->ndim();
        if($ndimX==0) {
            $X = $X->reshape([$X->size()]);
            $ndimX = 1;
            $size = $A->size();
            $A = $A->reshape([$size,1]);
            $ndimA = 2;
            $m = $size;
            $n = 1;
        } else {
            $shapeA = $A->shape();
            $shapeX = $X->shape();
            if($shapeA==$shapeX) {
                $m = 1;
                $n = $A->size();
            } else {
                $n = 1;
                while(true) {
                    $tmpX = array_pop($shapeX);
                    if($tmpX===null) {
                        break;
                    }
                    $tmpA = array_pop($shapeA);
                    if($tmpA!=$tmpX) {
                        throw new InvalidArgumentException('A and X is unmatched for broadcast');
                    }
                    $n *= $tmpX;
                }
                $m = array_product($shapeA);
            }
        }
        if($A->dtype()!=$X->dtype()) {
            throw new InvalidArgumentException('A and X must be same data type');
        }
        $m = (int)$m;
        $n = (int)$n;
        return [$m,$n,$A,$X];
    }

    /**
     *     A[m,n] := A[m,n] (A[m,n] >  X[n])
     *     A[m,n] := X[n]   (A[m,n] <= X[n])
     */
    public function maximum(
        NDArray $A,
        $X
        ) : NDArray
    {
        [$m,$n,$dmy,$X] = $this->calcBroadcastFormat($A,$X);
        $AA   = $A->buffer();
        $offA = $A->offset();
        $XX = $X->buffer();
        $offX = $X->offset();

        $this->math->maximum(
            $m,
            $n,
            $AA,$offA,$n,
            $XX,$offX,1
        );

        return $A;
    }

    /**
     *     A[m,n] := A[m,n] (A[m,n] <  X[n])
     *     A[m,n] := X[n]   (A[m,n] >= X[n])
     */
    public function minimum(
        NDArray $A,
        $X
        ) : NDArray
    {
        [$m,$n,$dmy,$X] = $this->calcBroadcastFormat($A,$X);
        $AA   = $A->buffer();
        $offA = $A->offset();
        $XX = $X->buffer();
        $offX = $X->offset();

        $this->math->minimum(
            $m,
            $n,
            $AA,$offA,$n,
            $XX,$offX,1
        );

        return $A;
    }

    /**
     *     A[m,n] := 1 (A[m,n] >  X[n])
     *     A[m,n] := 0 (A[m,n] <= X[n])
     */
    public function greater(
        NDArray $A,
        $X
        ) : NDArray
    {
        [$m,$n,$dmy,$X] = $this->calcBroadcastFormat($A,$X);
        $AA   = $A->buffer();
        $offA = $A->offset();
        $XX = $X->buffer();
        $offX = $X->offset();

        $this->math->greater(
            $m,
            $n,
            $AA,$offA,$n,
            $XX,$offX,1
        );

        return $A;
    }

    /**
     *     A[m,n] := 1 (A[m,n] >= X[n])
     *     A[m,n] := 0 (A[m,n] <  X[n])
     */
    public function greaterEqual(
        NDArray $A,
        $X
        ) : NDArray
    {
        [$m,$n,$dmy,$X] = $this->calcBroadcastFormat($A,$X);
        $AA   = $A->buffer();
        $offA = $A->offset();
        $XX = $X->buffer();
        $offX = $X->offset();

        $this->math->greaterEqual(
            $m,
            $n,
            $AA,$offA,$n,
            $XX,$offX,1
        );

        return $A;
    }

    /**
     *     A[m,n] := 1 (A[m,n] <  X[n])
     *     A[m,n] := 0 (A[m,n] >= X[n])
     */
    public function less(
        NDArray $A,
        $X
        ) : NDArray
    {
        [$m,$n,$dmy,$X] = $this->calcBroadcastFormat($A,$X);
        $AA   = $A->buffer();
        $offA = $A->offset();
        $XX = $X->buffer();
        $offX = $X->offset();

        $this->math->less(
            $m,
            $n,
            $AA,$offA,$n,
            $XX,$offX,1
        );

        return $A;
    }

    /**
     *     A[m,n] := 1 (A[m,n] <= X[n])
     *     A[m,n] := 0 (A[m,n] >  X[n])
     */
    public function lessEqual(
        NDArray $A,
        $X
        ) : NDArray
    {
        [$m,$n,$dmy,$X] = $this->calcBroadcastFormat($A,$X);
        $AA   = $A->buffer();
        $offA = $A->offset();
        $XX = $X->buffer();
        $offX = $X->offset();

        $this->math->lessEqual(
            $m,
            $n,
            $AA,$offA,$n,
            $XX,$offX,1
        );

        return $A;
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
     *    A(m,n) := alpha * X(n) + A(m,n)
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
            $XX,$offX,1,
            $alpha);

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
     *     X := sin(X)
     */
    public function sin(
        NDArray $X
        ) : NDArray
    {
        $n = $X->size();
        $XX = $X->buffer();
        $offX = $X->offset();

        $this->math->sin(
            $n,
            $XX,$offX,1);

        return $X;
    }

    /**
     *     X := cos(X)
     */
    public function cos(
        NDArray $X
        ) : NDArray
    {
        $n = $X->size();
        $XX = $X->buffer();
        $offX = $X->offset();

        $this->math->cos(
            $n,
            $XX,$offX,1);

        return $X;
    }

    /**
     *     X := tan(X)
     */
    public function tan(
        NDArray $X
        ) : NDArray
    {
        $n = $X->size();
        $XX = $X->buffer();
        $offX = $X->offset();

        $this->math->tan(
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

    /**
     *     X(i) := 1 (X(i)  = 0)
     *     X(i) := 0 (X(i) != 0)
     */
    public function not(NDArray $x)
    {
        $zeros = $this->zerosLike($x);
        return $this->equal($zeros,$x);
    }

    /**
     * A(m,n) := X(n)
     */
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
    *      B(m,n,k) := A(m,X(m,n),k)
    */
    public function doGather(
        bool $scatterAdd,
        NDArray $A,
        NDArray $X,
        int $axis=null,
        NDArray $B=null,
        $dtype=null) : NDArray
    {
//echo "shapeX=[".implode(',',$X->shape())."],shapeA=[".implode(',',$A->shape())."]\n";
        if($axis===null) {
            $postfixShape = $A->shape();
            $prefixShape = $X->shape();
            $numClass = array_shift($postfixShape);
            $m = 1;
            $n = array_product($prefixShape);
            $k = array_product($postfixShape);
            $reductionDims = false;
            $outputShape = array_merge($prefixShape,$postfixShape);
        } else {
            $ndim = $A->ndim();
            $orgAxis = $axis;
            if($axis<0) {
                $axis = $ndim+$axis;
            }
            $postfixShape = $A->shape();
            $prefixShape = [];
            for($i=0;$i<$axis;$i++) {
                $prefixShape[] = array_shift($postfixShape);
            }
            $numClass = array_shift($postfixShape);
            $m = array_product($prefixShape);
            $n = array_product($postfixShape);
            $k = 1;
            $reductionDims = true;
            $outputShape = array_merge($prefixShape,$postfixShape);
            if($X->shape()!=$outputShape) {
                throw new InvalidArgumentException('Unmatch Shape:'.
                                        $this->printableShapes([$A,$X]));
            }
        }
//echo "outputShape=[".implode(',',$outputShape)."]\n";
        if($dtype===null) {
            $dtype = $A->dtype();
        }
        if($B==null) {
            $B = $this->alloc($outputShape,$dtype);
            $this->zeros($B);
        } else {
            if($B->shape()!=$outputShape) {
                throw new InvalidArgumentException("Unmatch output shape of dimension: ".
                                            $this->printableShapes([$outputShape,$B]));
            }
        }

        $AA = $A->buffer();
        $offA = $A->offset();
        $XX = $X->buffer();
        $offX = $X->offset();
        $BB = $B->buffer();
        $offB = $B->offset();

        if($scatterAdd) {
            $reverse=true;
            $addMode=true;
        } else {
            $reverse=false;
            $addMode=false;
        }
        if($reductionDims) {
            $this->math->reduceGather(
                $reverse,
                $addMode,
                $m,
                $n,
                $numClass,
                $XX,$offX,
                $AA,$offA,
                $BB,$offB);
        } else {
            $this->math->gather(
                $reverse,
                $addMode,
                $n,
                $k,
                $numClass,
                $XX,$offX,
                $AA,$offA,
                $BB,$offB);
        }

        return $B;
    }

    /**
    *      B(m,n,k) := A(m,X(m,n),k)
    */
    public function gather(
        NDArray $A,
        NDArray $X,
        int $axis=null,
        NDArray $B=null,
        $dtype=null) : NDArray
    {
        return $this->doGather(
            $scatterAdd=false,
            $A,
            $X,
            $axis,
            $B,
            $dtype);
    }

    /**
    *      B(m,X(m,n),k) += A(m,n,k)
    */
    public function scatterAdd(
        NDArray $X,
        NDArray $A,
        NDArray $B,
        int $axis=null,
        $dtype=null) : NDArray
    {
        $this->doGather(
            $scatterAdd=true,
            $B,
            $X,
            $axis,
            $A,
            $dtype);
        return $B;
    }

    /**
    *      B(m,X(m,n),k) := A(m,n,k)
    */
    public function scatter(
        NDArray $X,
        NDArray $A,
        int $numClass,
        int $axis=null,
        NDArray $B=null,
        $dtype=null) : NDArray
    {
//echo "shapeX=[".implode(',',$X->shape())."],shapeA=[".implode(',',$A->shape())."]\n";
//echo "axis=$axis,numClass=$numClass\n";
        if($axis===null) {
            $postfixShape = $A->shape();
            $prefixShape = $X->shape();
            //$numClass
            $ndimX = $X->ndim();
            $tmpShape = [];
            for($i=0;$i<$ndimX;$i++) {
                $tmpShape[] = array_shift($postfixShape);
            }
            if($tmpShape!=$prefixShape) {
                throw new InvalidArgumentException('Unmatch Shape:'.
                                        $this->printableShapes([$X,$A]));
            }
            $n = array_product($prefixShape);
            $k = array_product($postfixShape);
            $m = 1;
            $expandDims = false;
            $outputShape = array_merge([$numClass],$postfixShape);
        } else {
            $ndim = $A->ndim();
            $orgAxis = $axis;
            if($axis<0) {
                $axis = $ndim+$axis;
            }
            //if($axis<0 || $axis>$ndim-1) {
            //    throw new InvalidArgumentException("Invalid axis: ".$orgAxis);
            //}
            $postfixShape = $A->shape();
            $postfixX = $X->shape();
            if($postfixShape!=$postfixX) {
                throw new InvalidArgumentException('Unmatch Shape X and A:'.
                                        $this->printableShapes([$X,$A]));
            }
            $prefixShape = [];
            for($i=0;$i<$axis;$i++) {
                $prefixShape[] = array_shift($postfixShape);
                array_shift($postfixX);
            }
            $m = array_product($prefixShape);
            $n = array_product($postfixShape);
            $k = 1;
            $expandDims = true;
            $outputShape = array_merge($prefixShape,[$numClass],$postfixShape);
        }
//echo "outputShape=[".implode(',',$outputShape)."]\n";
        if($dtype===null) {
            $dtype = $A->dtype();
        }
        if($B==null) {
            $B = $this->alloc($outputShape,$dtype);
            $this->zeros($B);
        } else {
            if($B->shape()!=$outputShape) {
                $shapeError = '('.implode(',',$A->shape()).'),('.implode(',',$B->shape()).')';
                throw new InvalidArgumentException("Unmatch shape of dimension: ".$shapeError);
            }
        }

        $AA = $A->buffer();
        $offA = $A->offset();
        $XX = $X->buffer();
        $offX = $X->offset();
        $BB = $B->buffer();
        $offB = $B->offset();

        if($expandDims) {
            $this->math->reduceGather(
                $reverse=true,
                $addMode=false,
                $m,
                $n,
                $numClass,
                $XX,$offX,
                $BB,$offB,
                $AA,$offA);

        } else {
            $this->math->gather(
                $reverse=true,
                $addMode=false,
                $n,
                $k,
                $numClass,
                $XX,$offX,
                $BB,$offB,
                $AA,$offA);
        }

        return $B;
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
            $Y = $this->zeros($this->alloc([$sizeX,$numClass],$this->defaultFloatType));
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

    /**
     *    X(m) := sum( A(m,n) )
     */
    public function reduceSum( // reduceSumEx
        NDArray $A,
        int $axis=null,
        NDArray $B=null,
        $dtype=null) : NDArray
    {
        $ndim = $A->ndim();
        if($axis===null) {
            $axis = 0;
        }
        if($axis<0) {
            $axis = $ndim+$axis;
        }
        if($axis<0 || $axis>$ndim-1) {
            throw new InvalidArgumentException("Invalid axis");
        }
        $postfixShape = $A->shape();
        $prefixShape = [];
        for($i=0;$i<$axis;$i++) {
            $prefixShape[] = array_shift($postfixShape);
        }
        $n = array_shift($postfixShape);
        $m = array_product($prefixShape);
        $k = array_product($postfixShape);
        $outputShape = array_merge($prefixShape,$postfixShape);
        if($dtype===null) {
            $dtype = $A->dtype();
        }
        if($B==null) {
            $B = $this->alloc($outputShape,$dtype);
        } else {
            if($B->shape()!=$outputShape) {
                $shapeError = '('.implode(',',$A->shape()).'),('.implode(',',$B->shape()).')';
                throw new InvalidArgumentException("Unmatch shape of dimension: ".$shapeError);
            }
        }

        $AA = $A->buffer();
        $offA = $A->offset();
        $BB = $B->buffer();
        $offB = $B->offset();

        $this->math->reduceSum(
            $m,
            $n,
            $k,
            $AA,$offA,
            $BB,$offB);

        return $B;
    }

    public function reduceMax( // reduceMaxEx
        NDArray $A,
        int $axis=null,
        NDArray $B=null,
        $dtype=null) : NDArray
    {
        $ndim = $A->ndim();
        if($axis===null) {
            $axis = 0;
        }
        if($axis<0) {
            $axis = $ndim+$axis;
        }
        if($axis<0 || $axis>$ndim-1) {
            throw new InvalidArgumentException("Invalid axis");
        }
        $postfixShape = $A->shape();
        $prefixShape = [];
        for($i=0;$i<$axis;$i++) {
            $prefixShape[] = array_shift($postfixShape);
        }
        $n = array_shift($postfixShape);
        $m = array_product($prefixShape);
        $k = array_product($postfixShape);
        $outputShape = array_merge($prefixShape,$postfixShape);
        if($dtype===null) {
            $dtype = $A->dtype();
        }
        if($B==null) {
            $B = $this->alloc($outputShape,$dtype);
        } else {
            if($B->shape()!=$outputShape) {
                $shapeError = '('.implode(',',$A->shape()).'),('.implode(',',$B->shape()).')';
                throw new InvalidArgumentException("Unmatch shape of dimension: ".$shapeError);
            }
        }

        $AA = $A->buffer();
        $offA = $A->offset();
        $BB = $B->buffer();
        $offB = $B->offset();

        $this->math->reduceMax(
            $m,
            $n,
            $k,
            $AA,$offA,
            $BB,$offB);

        return $B;
    }

    public function reduceArgMax( // reduceMaxArgEx
        NDArray $A,
        int $axis=null,
        NDArray $B=null,
        $dtypeB=null) : NDArray
    {
        $ndim = $A->ndim();
        if($axis===null) {
            $axis = 0;
        }
        if($axis<0) {
            $axis = $ndim+$axis;
        }
        if($axis<0 || $axis>$ndim-1) {
            throw new InvalidArgumentException("Invalid axis");
        }
        $postfixShape = $A->shape();
        $prefixShape = [];
        for($i=0;$i<$axis;$i++) {
            $prefixShape[] = array_shift($postfixShape);
        }
        $n = array_shift($postfixShape);
        $m = array_product($prefixShape);
        $k = array_product($postfixShape);
        $outputShape = array_merge($prefixShape,$postfixShape);
        if($dtypeB===null) {
            $dtypeB = NDArray::uint32;
        }
        if($B==null) {
            $B = $this->alloc($outputShape,$dtypeB);
        } else {
            if($B->shape()!=$outputShape) {
                $shapeError = '('.implode(',',$A->shape()).'),('.implode(',',$B->shape()).')';
                throw new InvalidArgumentException("Unmatch shape of dimension: ".$shapeError);
            }
        }

        $AA = $A->buffer();
        $offA = $A->offset();
        $BB = $B->buffer();
        $offB = $B->offset();

        $this->math->reduceArgMax(
            $m,
            $n,
            $k,
            $AA,$offA,
            $BB,$offB);

        return $B;
    }

    public function reduceMean(NDArray $A,int $axis=null,NDArray $X=null,$dtypeX=null) : NDArray
    {
        if($axis===null) {
            $axis = 0;
        }
        $X = $this->reduceSum(
            $A,$axis,$X,$dtypeX
        );
        $ndim = $A->ndim();
        if($axis<0) {
            $axis = $ndim+$axis;
        }
        if($ndim<=$axis) {
            throw new InvalidException('axis must be less then num of dimension');
        }
        $shapeA = $A->shape();
        $rows = $shapeA[$axis];
        $this->scal(1/$rows,$X);
        return $X;
    }

    public function im2col(
        NDArray $images,
        array $filterSize=null,
        array $strides=null,
        bool $padding=null,
        bool $channels_first=null,
        array $dilation_rate=null,
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
                $dilation_rate,
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
                $dilation_rate,
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
                $dilation_rate,
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
        array $dilation_rate=null,
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
                $dilation_rate,
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
                $dilation_rate,
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
                $dilation_rate,
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
        array $dilation_rate=null,
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
        if($dilation_rate==null) {
            $dilation_rate = [1];
        }
        [$dilation_w] =
            $dilation_rate;
        $cols_channels_first = ($cols_channels_first) ? true:false;
        if($cols==null) {
            if($padding) {
                $out_w = $in_w;
            } else {
                //$out_w = intval(floor(($in_w-$filter_w)/$stride_w)+1);
                $out_w = intval(floor(($in_w-($filter_w-1)*$dilation_w-1)/$stride_w)+1);
            }
            if($out_w<=0) {
                throw new InvalidArgumentException('Invalid shape or paramaters.');
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
            $dilation_w,
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
        array $dilation_rate=null,
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
        if($dilation_rate==null) {
            $dilation_rate = [1,1];
        }
        [$dilation_h,$dilation_w] =
            $dilation_rate;
        $cols_channels_first = ($cols_channels_first) ? true:false;
        if($cols==null) {
            if($padding) {
                $out_h = $in_h;
                $out_w = $in_w;
            } else {
                //$out_h = intval(floor(($in_h-$filter_h)/$stride_h)+1);
                //$out_w = intval(floor(($in_w-$filter_w)/$stride_w)+1);
                $out_h = intval(floor(($in_h-($filter_h-1)*$dilation_h-1)/$stride_h)+1);
                $out_w = intval(floor(($in_w-($filter_w-1)*$dilation_w-1)/$stride_w)+1);
            }
            if($out_h<=0 || $out_w<=0) {
                throw new InvalidArgumentException('Invalid shape or paramaters.');
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
            $dilation_h,

            $dilation_w,
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
        array $dilation_rate=null,
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
        if($dilation_rate==null) {
            $dilation_rate = [1,1,1];
        }
        [$dilation_d,$dilation_h,$dilation_w] =
            $dilation_rate;
        $cols_channels_first = ($cols_channels_first) ? true : false;
        if($cols==null) {
            if($padding) {
                $out_d = $in_d;
                $out_h = $in_h;
                $out_w = $in_w;
            } else {
                //$out_d = intval(floor(($in_d-$filter_d)/$stride_d)+1);
                //$out_h = intval(floor(($in_h-$filter_h)/$stride_h)+1);
                //$out_w = intval(floor(($in_w-$filter_w)/$stride_w)+1);
                $out_d = intval(floor(($in_d-($filter_d-1)*$dilation_d-1)/$stride_d)+1);
                $out_h = intval(floor(($in_h-($filter_h-1)*$dilation_h-1)/$stride_h)+1);
                $out_w = intval(floor(($in_w-($filter_w-1)*$dilation_w-1)/$stride_w)+1);
            }
            if($out_d<=0 || $out_h<=0 || $out_w<=0) {
                throw new InvalidArgumentException('Invalid shape or paramaters.');
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
            $dilation_d,
            $dilation_h,
            $dilation_w,
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
        } elseif($axis==2){
            $k = count($values);
            $shape = $values[0]->shape();
            $m = array_shift($shape);
            $n = array_shift($shape);
            array_unshift($shape,$k);
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
                $n = array_shift($shape);
                array_unshift($shape,1);
                array_unshift($shape,$n);
                array_unshift($shape,$m);
                $value = $value->reshape(
                    $shape);
                $this->doSlice(true,
                    $output,
                    [0,0,$i],[-1,-1,1],
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
            for($j=0;$j<$axis;$j++) {
                $shapePrefix[] = array_shift($shape);
            }
            $mm = (int)array_product($shapePrefix);
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
        for($j=0;$j<$axis;$j++) {
            $shapePrefix[] = array_shift($shape);
        }
        $m = (int)array_product($shapePrefix);
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
        if($ndimBegin<1||$ndimBegin>3) {
            throw new InvalidArgumentException('begin must has 1 or 2 or 3 integer.');
        }
        $ndimSize = count($size);
        if($ndimSize<1||$ndimSize>3) {
            throw new InvalidArgumentException('Size must has 1 or 2 or 3 integer.');
        }
        if($ndimBegin!=$ndimSize){
            throw new InvalidArgumentException('Unmatch shape of begin and size');
        }
        $ndimInput = $input->ndim();
        if($ndimInput<$ndimBegin){
            throw new InvalidArgumentException($messageInput.' shape rank is low to slice');
        }
        $shape = $input->shape();

        // ndim = 0
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

        // ndim = 1
        if($ndimBegin<=1){
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

        // ndim = 2
        if($ndimBegin<=2){
            $k = 1;
            $startAxis2 = 0;
            $sizeAxis2 = 1;
        } else {
            $k = array_shift($shape);
            $startAxis2 = array_shift($begin);
            if($startAxis2<0){
                $startAxis2 = $k+$startAxis2;
            }
            if($startAxis2<0||$startAxis2>=$k){
                throw new InvalidArgumentException('start of axis 2 is invalid value.:begin=['.implode(',',$orgBegin).']');
            }
            $sizeAxis2 = array_shift($size);
            if($sizeAxis2<0){
                $sizeAxis2 = $k-$startAxis2+$sizeAxis2+1;
            }
            if($sizeAxis2<1||$startAxis2+$sizeAxis2>$k){
                throw new InvalidArgumentException('size of axis 2 is invalid value.');
            }
        }
        $itemSize = array_product($shape);
        $outputShape = [$sizeAxis0];
        if($ndimBegin>=2){
            array_push($outputShape,
                $sizeAxis1);
        }
        if($ndimBegin>=3){
            array_push($outputShape,
                $sizeAxis2);
        }
        $outputShape = array_merge(
            $outputShape,$shape);
        if($output==null){
            $output = $this->alloc($outputShape,$input->dtype());
        }else{
            if($outputShape!=$output->shape()){
                throw new InvalidArgumentException('Unmatch output shape: '.
                    $this->printableShapes($outputShape).'<=>'.
                    $this->printableShapes($output->shape()));
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
            $itemSize,
            $A,$offsetA,$incA,
            $Y,$offsetY,$incY,
            $startAxis0,$sizeAxis0,
            $startAxis1,$sizeAxis1,
            $startAxis2,$sizeAxis2
        );
        return $output;
    }

    /**
    * repeat
    */
    public function repeat(NDArray $A, int $repeats, int $axis=null)
    {
        if($repeats<1) {
            throw new InvalidArgumentException('repeats argument must be one or greater.');
        }
        if($axis!==null) {
            $ndim = $A->ndim();
            if($axis<0) {
                $axis = $ndim+$axis;
            }
            if($A->ndim()<$axis) {
                throw new InvalidArgumentException('dimension rank must be two or greater.');
            }
        }
        $innerShape = $A->shape();
        $outerShape = [];
        if($axis!==null) {
            for($i=0;$i<$axis;$i++) {
                $outerShape[] = array_shift($innerShape);
            }
        }
        if($axis===null) {
            $outputShape = [(int)array_product(
                    array_merge($outerShape,[$repeats],$innerShape))];
        } else {
            $outputShape = array_merge($outerShape,[$repeats],$innerShape);
        }
        $B = $this->alloc($outputShape,$A->dtype());
        $m = (int)array_product($outerShape);
        $k = (int)array_product($innerShape);
        $AA = $A->buffer();
        $offA = $A->offset();
        $BB = $B->buffer();
        $offB = $B->offset();
        $this->math->repeat(
            $m,
            $k,
            $repeats,
            $AA,$offA,
            $BB,$offB
        );
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
            self::LAPACK_ROW_MAJOR,
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

    public function transpose(
        NDArray $A,
        NDArray $B=null,
        float $alpha=null
        ) : NDArray
    {
        if($A->ndim()!=2) {
            throw new InvalidArgumentException('input array must be 2D.');
        }
        $shape = $A->shape();
        $shape = [$shape[1],$shape[0]];
        if($B==null) {
            $B = $this->alloc($shape,$A->dtype());
        } else {
            if($B->shape()!=$shape) {
                throw new InvalidArgumentException('output shape must be transpose matrix of input.');
            }
            if($B->dtype()!=$A->dtype()) {
                throw new InvalidArgumentException('output data type must be same with matrix of input.');
            }
        }
        if($alpha===null) {
            $alpha = 1.0;
        }
        $trans=true;
        $m = $shape[1];
        $n = $shape[0];
        $AA = $A->buffer();
        $offA = $A->offset();
        $ldA = $n;
        $BB = $B->buffer();
        $offB = $B->offset();
        $ldB = $trans ? $m : $n;
        $this->math->matrixcopy(
            $trans,
            $m,
            $n,
            $alpha,
            $AA,$offA,$ldA,
            $BB,$offB,$ldB
        );
        return $B;
    }

    public function imagecopy(
        NDArray $A,
        NDArray $B=null,
        bool $channels_first=null,
        int $heightShift=null,
        int $widthShift=null,
        bool $verticalFlip=null,
        bool $horizontalFlip=null,
        bool $rgbFlip=null
        ) : NDArray
    {
        if($A->ndim()!=3) {
            throw new InvalidArgumentException('input array must be 3D.');
        }
        $shape = $A->shape();
        if($B==null) {
            $B = $this->alloc($shape,$A->dtype());
            $this->zeros($B);
        } else {
            if($B->shape()!=$shape) {
                throw new InvalidArgumentException('output shape must be transpose matrix of input.');
            }
            if($B->dtype()!=$A->dtype()) {
                throw new InvalidArgumentException('output data type must be same with matrix of input.');
            }
        }
        if($heightShift==null) {
            $heightShift=0;
        }
        if($widthShift==null) {
            $widthShift=0;
        }
        if($verticalFlip==null) {
            $verticalFlip=false;
        }
        if($horizontalFlip==null) {
            $horizontalFlip=false;
        }
        if($rgbFlip==null) {
            $rgbFlip=false;
        }
        if($channels_first==null) {
            $channels_first=false;
            $height = $shape[0];
            $width = $shape[1];
            $channels = $shape[2];
        } else {
            $channels_first=true;
            $channels = $shape[0];
            $height = $shape[1];
            $width = $shape[2];
        }
        $AA = $A->buffer();
        $offA = $A->offset();
        $BB = $B->buffer();
        $offB = $B->offset();
        $this->math->imagecopy(
            $height,
            $width,
            $channels,
            $AA, $offA,
            $BB, $offB,
            $channels_first,
            $heightShift,
            $widthShift,
            $verticalFlip,
            $horizontalFlip,
            $rgbFlip
        );
        return $B;
    }

    public function fill(
        $value,
        NDArray $X
        )
    {
        if(is_scalar($value)) {
            if(is_string($value)) {
                $value = ord($value);
            }
            $V = $this->alloc([1],$X->dtype());
            $V[0] = $value;
        } elseif($value instanceof NDArray) {
            if($value->size()!=1) {
                throw new InvalidArgumentException('Value must be scalar');
            }
            $V = $value;
        } else {
            throw new InvalidArgumentException('Invalid data type');
        }
        $n = $X->size();
        $VV = $V->buffer();
        $offV = $V->offset();
        $XX = $X->buffer();
        $offX = $X->offset();
        $this->math->fill(
            $n,
            $VV, $offV,
            $XX,$offX,1
        );
        return $X;
    }

    public function searchsorted(
        NDArray $A,
        NDArray $X,
        bool $right=null,
        $dtype=null,
        NDArray $Y=null
        ) : NDArray
    {
        if($A->ndim()!=1) {
            throw new InvalidArgumentException('A must be 1D NDArray.');
        }
        if($right===null) {
            $right = false;
        }
        if($dtype===null) {
            $dtype = NDArray::uint32;
        }
        if($Y===null) {
            $Y = $this->alloc($X->shape(),$dtype);
        }
        $dtype = $Y->dtype();
        if($dtype!=NDArray::uint32&&$dtype!=NDArray::int32&&
            $dtype!=NDArray::uint64&&$dtype!=NDArray::int64) {
            throw new InvalidArgumentException('dtype of Y must be int32 or int64');
        }
        if($X->shape()!=$Y->shape()) {
            $shapeError = '('.implode(',',$X->shape()).'),('.implode(',',$Y->shape()).')';
            throw new InvalidArgumentException("Unmatch shape of dimension: ".$shapeError);
        }
        $m = $A->size();
        $AA = $A->buffer();
        $offA = $A->offset();
        $n = $X->size();
        $XX = $X->buffer();
        $offX = $X->offset();
        $YY = $Y->buffer();
        $offY = $Y->offset();

        $this->math->searchsorted(
            $m,
            $AA,$offA,1,
            $n,
            $XX,$offX,1,
            $right,
            $YY,$offY,1
        );

        return $Y;
    }

    public function cumsum(
        NDArray $X,
        bool $exclusive=null,
        bool $reverse=null,
        NDArray $Y=null
        ) : NDArray
    {
        if($exclusive===null) {
            $exclusive = false;
        }
        if($reverse===null) {
            $reverse = false;
        }
        if($Y===null) {
            $Y = $this->alloc($X->shape(),$X->dtype());
        }
        if($X->shape()!=$Y->shape()) {
            $shapeError = '('.implode(',',$X->shape()).'),('.implode(',',$Y->shape()).')';
            throw new InvalidArgumentException("Unmatch shape of dimension: ".$shapeError);
        }
        $n = $X->size();
        $XX = $X->buffer();
        $offX = $X->offset();
        $YY = $Y->buffer();
        $offY = $Y->offset();

        $this->math->cumsum(
            $n,
            $XX,$offX,1,
            $exclusive,
            $reverse,
            $YY,$offY,1
        );

        return $Y;
    }

    /**
     *     X := nan2num(X)
     */
    public function nan2num(
        NDArray $X,
        float $alpha=null
        ) : NDArray
    {
        $n = $X->size();
        $XX = $X->buffer();
        $offX = $X->offset();

        if($alpha===null) {
            $alpha = 0.0;
        }
        $this->math->nan2num(
            $n,
            $XX,$offX,1,
            $alpha);

        return $X;
    }

    /**
     *     X := isnan(X)
     */
    public function isnan(
        NDArray $X
        ) : NDArray
    {
        $n = $X->size();
        $XX = $X->buffer();
        $offX = $X->offset();

        $this->math->isnan(
            $n,
            $XX,$offX,1);

        return $X;
    }

    public function linspace(float $start, float $stop, int $num, $dtype=null) : NDArray
    {
        if($num<=0) {
            throw new InvalidArgumentException('num must be greater than or equal zero.');
        }
        $array = $this->alloc([$num],$dtype);
        $step = ($stop-$start)/($num-1);
        $value = $start;
        for($i=0;$i<$num;$i++) {
            $array[$i] = min($start+$step*$i,$stop);
        }
        return $array;
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
            $xx = $x->reshape([$x->size()]);
            //$idx = $x->offset();
            $gg = $grad->reshape([$grad->size()]);
            //$gidx = $grad->offset();
            $h2 = $h*2 ;
            for($i=0;$i<$size;$i++) {
                //    $value = $xx[$idx];
                $value = $this->copy($xx[[$i,$i]]);
                //    $xx[$idx] = $value + $h;
                $this->copy($this->increment($this->copy($value),$h),$xx[[$i,$i]]);
                //echo $value[0]."-h =>".$xx[$i]."\n";
                $y1 = $f(...$variables);
                //    $xx[$idx] = $value - $h;
                $this->copy($this->increment($this->copy($value),-$h),$xx[[$i,$i]]);
                //echo $value[0]."-h =>".$xx[$i]."\n";
                $y2 = $f(...$variables);
                $d = $this->axpy($y2,$this->copy($y1),-1);
                //    $gg[$gidx] = $this->sum($d)/$h2;
                $sum = $this->reduceSum($d->reshape([$d->size(),1]));
                //echo "d=".$sum[0]."\n";
                $this->copy($this->scal(1/$h2,$sum),$gg[[$i,$i]]);
                //    $xx[$idx] = $value;
                $this->copy($value,$xx[[$i,$i]]);
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

    public function augmentedMatrix($a,$b)
    {
        if($a->ndim()!=2) {
            throw new InvalidArgumentException('matrix a must be 2d');
        }
        if($b->ndim()!=1) {
            throw new InvalidArgumentException('matrix a must be 1d');
        }
        $shapeA = $a->shape();
        $shapeB = $b->shape();
        if($shapeA[0]!=$shapeB[0]) {
            throw new InvalidArgumentException('Unmatch shape A rows and B: '.$shapeA[0].'!='.$shapeB[0]);
        }
        $m = $shapeA[0];
        $n = $shapeA[1];
        $aug = $this->alloc([$m,$n+1],$a->dtype());
        $AA = $a->buffer();
        $offA = $a->offset();
        $GG = $aug->buffer();
        $this->math->matrixcopy(
            false,
            $m,
            $n,
            1.0,
            $AA,$offA,$n,
            $GG,0,    $n+1
        );
        $BB = $b->buffer();
        $offB = $b->offset();
        $this->blas->copy($m,$BB,$offB,1,$GG,$n,$n+1);
        return $aug;
    }

    public function solve(NDArray $a, NDArray $b, float $epsilon=null)
    {
        if($epsilon===null) {
            $epsilon = 1e-7;
        }
        $aug = $this->augmentedMatrix($a,$b);
        [$m,$n] = $a->shape();
        if($m!=$n) {
            throw new InvalidArgumentException('matrix A must be square');
        }
        // forward
        for($i=0;$i<$n-1;$i++) {
            for($j=$i+1;$j<$n;$j++) {
                $div = $aug[$i][$i];
                if(abs($div)<$epsilon) {
                    throw new InvalidArgumentException('too small value');
                }
                $tmp = $aug[$j][$i] / $div;
                for($k=$i+1;$k<$n+1;$k++){
                    $aug[$j][$k] = $aug[$j][$k] - $tmp*$aug[$i][$k];
                }
            }
        }
        // backward
        for($i=$n-1;$i>=0;$i--) {
            for($j=$i+1;$j<$n;$j++) {
                $aug[$i][$n] = $aug[$i][$n] - $aug[$i][$j]*$aug[$j][$n];
            }
            $div = $aug[$i][$i];
            if(abs($div)<$epsilon) {
                throw new InvalidArgumentException('too small value');
            }
            $aug[$i][$n] = $aug[$i][$n] / $div;
        }
        $solve = $this->alloc([$n],$a->dtype());
        $SS = $solve->buffer();
        $this->blas->copy($n,$aug->buffer(),$aug->offset()+$n,$n+1,$SS,0,1);
        return $solve;
    }
}
