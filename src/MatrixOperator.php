<?php
namespace Rindow\Math\Matrix;

use ArrayObject;
use InvalidArgumentException;
use Interop\Polite\Math\Matrix\BLAS;
use Interop\Polite\Math\Matrix\NDArray;
use Interop\Polite\Math\Matrix\OpenCL;
use Rindow\OpenBLAS\Blas as OpenBLAS;
use Rindow\OpenBLAS\Lapack as OpenBLASLapack;
use Rindow\OpenBLAS\Math as OpenBLASMath;
use Rindow\CLBlast\Blas as CLBlastBlas;
use Rindow\CLBlast\Math as CLBlastMath;
use Rindow\OpenCL\Context;
use Rindow\OpenCL\CommandQueue;

class MatrixOperator
{
    protected $blas;
    protected $openblas;
    protected $lapack;
    protected $openblaslapack;
    protected $math;
    protected $openblasmath;
    protected $random;
    protected $la;
    protected $laRawMode;
    protected $clblastLA;
    protected $broadCastOperators;
    protected $updateOperators;
    protected $intTypes= [
        NDArray::int8,NDArray::int16,NDArray::int32,NDArray::int64,
        NDArray::uint8,NDArray::uint16,NDArray::uint32,NDArray::uint64,
    ];
    protected $floatTypes= [
        NDArray::float16,NDArray::float32,NDArray::float64,
    ];
    protected $defaultIntType = NDArray::int32;
    protected $defaultFloatType = NDArray::float32;
    protected $dtypeToString = [
        NDArray::bool=>'bool',
        NDArray::int8=>'int8',   NDArray::uint8=>'uint8',
        NDArray::int16=>'int16', NDArray::uint16=>'uint16',
        NDArray::int32=>'int32', NDArray::uint32=>'uint32',
        NDArray::int64=>'int64', NDArray::uint64=>'uint64',
        NDArray::float16=>'float16',
        NDArray::float32=>'float32', NDArray::float64=>'float64',
    ];
    protected $dtypePrecision = [
        NDArray::bool=>1,
        NDArray::int8=>2,  NDArray::uint8=>3,
        NDArray::int16=>4, NDArray::uint16=>5,
        NDArray::int32=>6, NDArray::uint32=>7,
        NDArray::int64=>8, NDArray::uint64=>9,
        NDArray::float16=>10,
        NDArray::float32=>11, NDArray::float64=>12,
    ];

    public function __construct($blas=null,$lapack=null,$math=null)
    {
        if($blas) {
            $this->blas = $blas;
        } else {
            if(extension_loaded('rindow_openblas')) {
                $this->openblas = new OpenBLAS();
            }
            $this->blas = new PhpBlas($this->openblas);
        }
        if($lapack) {
            $this->lapack = $lapack;
        } else {
            if(extension_loaded('rindow_openblas')) {
                $this->openblaslapack = new OpenBLASLapack();
            }
            $this->lapack = new PhpLapack($this->openblaslapack);
        }
        if($math) {
            $this->math = $math;
        } else {
            if(extension_loaded('rindow_openblas')) {
                $this->openblasmath = new OpenBLASMath();
            }
            $this->math = new PhpMath($this->openblasmath);
        }

        $this->broadCastOperators = [
           '+' =>  [null, 'add'], //   function($x,$y) { return $x + $y; }],
           '-' =>  [null, 'sub'], //   function($x,$y) { return $x - $y; }],
           '*' =>  [null, 'mul'], //   function($x,$y) { return $x * $y; }],
           '/' =>  [null, 'div'], //   function($x,$y) { return $x / $y; }],
           '%' =>  [null, 'mod'], //   function($x,$y) { return $x % $y; }],
           '**' => [null, 'pow'], //   function($x,$y) { return $x ** $y; }],
           '==' => [NDArray::bool, 'is_equal'],     // function($x,$y) { return ($x == $y); }],
           '!=' => [NDArray::bool, 'is_not_equal'], // function($x,$y) { return $x != $y; }],
           '>' =>  [NDArray::bool, 'greater'],      // function($x,$y) { return $x > $y; }],
           '>=' => [NDArray::bool, 'greater_or_equal'], // function($x,$y) { return $x >= $y; }],
           '<' =>  [NDArray::bool, 'smaller'],      // function($x,$y) { return $x < $y; }],
           '<=' => [NDArray::bool, 'smaller_or_equal'], // function($x,$y) { return $x <= $y; }],
       ];

       $this->updateOperators = [
          '='  =>  [null, 'assign'], // function($x,$y) { return $y; }],
          '+=' =>  [null, 'assign_add'], // function($x,$y) { return $x + $y; }],
          '-=' =>  [null, 'assign_sub'], // function($x,$y) { return ($x - $y); }],
          '*=' =>  [null, 'assign_mul'], // function($x,$y) { return $x * $y; }],
          '/=' =>  [null, 'assign_div'], // function($x,$y) { return $x / $y; }],
          '%=' =>  [null, 'assign_mod'], // function($x,$y) { return $x % $y; }],
          '**=' => [null, 'assign_pow'], // function($x,$y) { return $x ** $y; }],
      ];
      $this->operatorFunctions = new MatrixOpelatorFunctions();
    }

    //public function close()
    //{
    //    $this->broadCastOperators = null;
    //    $this->updateOperators = null;
    //}

    protected function alloc($array,$dtype=null,$shape=null)
    {
        if($dtype===null) {
            $dtype = $this->resolveDtype($array);
        }
        return new NDArrayPhp($array,$dtype,$shape);
    }

    protected function resolveDtype($value)
    {
        while((is_array($value)||$value instanceof ArrayObject)&&isset($value[0])) {
            $value = $value[0];
        }
        if(is_int($value)) {
            return $this->defaultIntType;
        } elseif(is_float($value)) {
            return $this->defaultFloatType;
        } elseif(is_bool($value)) {
            return NDArray::bool;
        }
        return null;
    }

    protected function maximumPrecision($dtypeX,$dtypeY)
    {
        if(!isset($this->dtypePrecision[$dtypeX]))
            throw new RuntimeException("Illegal dtype: ".$dtypeX);
        if(!isset($this->dtypePrecision[$dtypeY]))
            throw new RuntimeException("Illegal dtype: ".$dtypeY);

        if($this->dtypePrecision[$dtypeX]>$this->dtypePrecision[$dtypeY])
            return $dtypeX;
        else
            return $dtypeY;
    }

    public function setDefaultIntType($dtype) : void
    {
        $this->defaultIntType = $dtype;
    }

    public function setDefaultFloatType($dtype) : void
    {
        $this->defaultFloatType = $dtype;
    }

    public function array($array,$dtype=null) : NDArray
    {
        if($dtype==null)
            $dtype=$this->defaultFloatType;
        if(is_array($array)) {
            return $this->alloc($array,$dtype);
        } elseif($array instanceof ArrayObject) {
            return $this->alloc($array,$dtype);
        } elseif(is_numeric($array)) {
            return $this->alloc($array,$dtype);
        } else {
            throw new InvalidArgumentException("Must be array or ArrayObject");
        }
    }

    public function zeros(array $shape, $dtype=null) : NDArray
    {
        return $this->full($shape,0.0,$dtype);
    }

    public function ones(array $shape, $dtype=null) : NDArray
    {
        return $this->full($shape,1.0,$dtype);
    }

    public function full(array $shape, $value, $dtype=null) : NDArray
    {
        if(!is_scalar($value))
            throw new InvalidArgumentException('value must be scalar');

        if($dtype===null)
            $dtype=$this->resolveDtype($value);
        $array = $this->alloc(null, $dtype, $shape);
        $buffer = $array->buffer();
        $size = count($buffer);
        for($i=0; $i<$size; $i++) {
            $buffer[$i] = $value;
        }
        return $array;
    }

    public function zerosLike(NDArray $array)
    {
        return $this->zeros($array->shape(),$array->dtype());
    }

    public function fullLike(NDArray $array,$value)
    {
        return $this->full($array->shape(),$value,$array->dtype());
    }

    public function astype(NDArray $array, $dtype) : NDArray
    {
        return $this->la()->astype($array,$dtype);
    }

    public function arange(int $count ,$start=null, $step=null, $dtype=null) : NDArray
    {
        if($start===null)
            $start = 0;
        if($step===null)
            $step = 1;
        if($dtype===null) {
            if(is_int($start))
                $dtype = $this->defaultIntType;
            else
                $dtype = $this->defaultFloatType;
        }
        $array = $this->alloc(null, $dtype, [$count]);
        $buffer = $array->buffer();
        $n = $start;
        for($i=0; $i<$count; $i++) {
            $buffer[$i] = $n;
            $n += $step;
        }
        return $array;
    }

    public function copy(NDArray $array) : NDArray
    {
        $newArray = $this->alloc(null, $array->dtype(), $array->shape());
        $this->la()->copy($array,$newArray);
        return $newArray;
    }

    public function cross(NDArray $A, NDArray $B) : NDArray
    {
        $rankA = $A->ndim();
        $rankB = $B->ndim();
        if($rankA>=2 && $rankB>=2) {
            return $this->matrixMultiply($A, $B);
        } elseif($rankA>=2 && $rankB==1) {
            return $this->vectorTransform($A, $B);
        } else {
            throw new InvalidArgumentException("unmatch shape of dimension");
        }
    }

    protected function matrixMultiply(NDArray $A, NDArray $B) : NDArray
    {
        ################# numpy compatible ############################
        # shape: (a,b,c,d,R)x(o,p,q,R,s) = (a,b,c,d,o,p,q,s)
        # dot(a, b)[a,b,c,d,o,p,q,s] = sum(a[a,b,c,d,R,:] * b[o,p,q,:,s])
        # shape: (i,j,R)x(k,R,m) = (i,j,k,m)
        # dot(a, b)[i,j,k,m] = sum(a[i,j,:] * b[k,:,m])

        $multiA = 1;
        $multiB = 1;
        $shapeC = [];

        $shapeA = $A->shape();
        while(count($shapeA)>2) {
            $n = array_shift($shapeA);
            array_push($shapeC,$n);
            $multiA *= $n;
        }
        [ $aM, $aK ] = $shapeA;
        array_unshift($shapeC,$aM);

        $shapeB = $B->shape();
        while(count($shapeB)>2) {
            $n = array_shift($shapeB);
            array_push($shapeC,$n);
            $multiB *= $n;
        }
        [ $bK, $bN ] = $shapeB;
        array_push($shapeC,$bN);

        if($aK != $bK) {
            $shapeError = '('.implode(',',$A->shape()).'),('.implode(',',$B->shape()).')';
            throw new InvalidArgumentException("Unmatch shape of dimension to cross multiple: ".$shapeError);
        }
        if($A->dtype() != $B->dtype()) {
            throw new InvalidArgumentException("unmatch value type");
        }
        $C = $this->zeros($shapeC,$A->dtype());

        $alpha = 1.0;
        $beta = 1.0;
        $AA = $A->buffer();
        $BB = $B->buffer();
        $M = $aM * $multiA;
        $N = $bN;
        $K = $bK;
        $L = $multiB;
        $CC = $C->buffer();
        $offC = $C->offset();
        $offA = $A->offset();
        $offB = $B->offset();

        if(count($B->shape())==2) {
            $this->blas->gemm(
                BLAS::RowMajor,BLAS::NoTrans,BLAS::NoTrans,
                $M,$N,$K,
                $alpha,
                $AA,$offA,$K,
                $BB,$offB,$N,
                $beta,
                $CC,$offC,$N);
        } else {
            $this->gemm3($M,$N,$K,$L,$alpha,$AA,$offA,$BB,$offB,$beta,$CC,$offC);
        }
        return $C;
    }

    public function gemm3(
        $M,$N,$K,$L,
        $alpha,
        $A,$offA,
        $B,$offB,
        $beta,
        $C,$offC)
    {
        for ($m=0; $m<$M; $m++) {
            for($l=0; $l<$L; $l++) {
                for ($n=0; $n<$N; $n++) {
                    $acc = 0.0;
                    for ($k=0; $k<$K; $k++) {
                        $acc += $A[$offA + $m*$K + $k] * $B[$offB + $N*$K*$l + $k*$N + $n];
                    }
                    $C[$offC + $m*$L*$N + $N*$l + $n] = $alpha * $acc + $beta * $C[$offC + $m*$L*$N + $N*$l + $n];
                }
            }
        }
    }

    protected function vectorTransform(NDArray $A, NDArray $B)
    {
        ################# numpy compatible ############################

        $shapeA = $A->shape();
        $shapeB = $B->shape();
        $batchSize = 1;
        while(count($shapeA)>2) {
            $batchSize *= array_shift($shapeA);
        }

        $shapeC = $A->shape();
        array_pop($shapeC);
        $C = $this->zeros($shapeC,$A->dtype());
        $sizeA = $A->size()/$batchSize;
        $sizeC = $C->size()/$batchSize;

        [ $aM, $aN ] = $shapeA;
        [ $bN ] = $shapeB;
        if($aN != $bN) {
            throw new InvalidArgumentException("unmatch shape of dimension");
        }
        if($A->dtype() != $B->dtype()) {
            throw new InvalidArgumentException("unmatch value type: ".$A->dtype()." and ".$B->dtype());
        }

        $alpha = 1.0;
        $beta = 1.0;
        $AA = $A->buffer();
        $BB = $B->buffer();
        $offA = $A->offset();
        $offB = $B->offset();
        $M = $aM;
        $N = $aN;
        $CC = $C->buffer();
        $offC = $C->offset();
        for($i=0;$i<$batchSize;$i++) {
            $this->blas->gemv(
                BLAS::RowMajor,BLAS::NoTrans,
                $M,$N,
                $alpha,
                $AA,$offA,$N,
                $BB,$offB,1,
                $beta,
                $CC,$offC,1);
            $offA += $sizeA;
            $offC += $sizeC;
        }

        return $C;
    }

    public function transpose(NDArray $X) : NDArray
    {
        $shape = $X->shape();
        $newShape = array_reverse($shape);
        $Y = $this->alloc(null, $X->dtype(), $newShape);
        $w = 1;
        $posY = 0;
        $posX = 0;
        $this->_transpose($newShape, $w, $X->buffer(), $X->offset(), $posX, $Y->buffer(), $posY);
        return $Y;
    }

    protected function _transpose($shape, $w, $bufX, $offX, $posX, $bufY, &$posY)
    {
        $n=array_shift($shape);
        $W = $w*$n;
        $deps = count($shape);
        for($i=0;$i<$n;$i++) {
            if($deps) {
                $this->_transpose($shape, $W, $bufX, $offX, $posX+$w*$i, $bufY, $posY);
            } else {
                $bufY[$posY] = $bufX[$offX + $posX+$w*$i];
                $posY++;
            }
        }
    }

    public function dot(NDArray $A, NDArray $B)
    {
        if($A->shape() != $B->shape()) {
            throw new InvalidArgumentException("unmatch shape of dimension");
        }
        if($A->dtype() != $B->dtype()) {
            throw new InvalidArgumentException("unmatch value type");
        }
        $AA = $A->buffer();
        $BB = $B->buffer();
        $offA = $A->offset();
        $offB = $B->offset();
        $N = $A->size();
        return $this->blas->dot($N,$AA,$offA,1,$BB,$offB,1);
    }

    public function add(NDArray $X, NDArray $Y) : NDArray
    {
        if($X->shape()!=$Y->shape()) {
            $shapeError = '('.implode(',',$X->shape()).'),('.implode(',',$Y->shape()).')';
            throw new InvalidArgumentException("Unmatch shape of dimension to add: ".$shapeError);
        }

        $C = $this->copy($Y);
        return $this->la()->axpy($X,$C);
    }

    public function scale($a, NDArray $X) : NDArray
    {
        if(!is_numeric($a))
            throw new InvalidArgumentException("the scalar must be a numeric");
        $N = $X->size();
        $C = $this->copy($X);
        $CC = $C->buffer();
        $offC = $C->offset();
        $this->blas->scal($N,$a,$CC,$offC,1);

        return $C;
    }

    /*
     *  DISCONTINUE
     */
    public function pos2index(int $pos,array $shape) : array
    {
        $rank = count($shape);
        $index = [];
        for($i=0;$i<$rank;$i++) {
            $w = 1;
            for($j=$i+1;$j<$rank;$j++) {
                $w *= $shape[$j];
            }

            $p = (int)floor($pos / $w);
            $pos = $pos % $w;
            array_push($index,$p);
        }
        return $index;
    }

    /*
     *  DISCONTINUE
     */
    public function projection(NDArray $X,array $idx) : NDArray
    {
        $shape = $X->shape();
        if(count($shape)!=count($idx)) {
            throw new InvalidArgumentException('Invalid $idx rank');
        }
        $rank = count($shape);
        $axis = null;
        for($i=0;$i<$rank;$i++) {
            if($idx[$i]===null||$idx[$i]<0) {
                if($axis!==null) {
                    throw new InvalidArgumentException('projection must has one variable axis');
                }
                $axis = $i;
            }
        }
        if($axis===null) {
            throw new InvalidArgumentException('projection not have variable axis');
        }

        $Y = $this->alloc(null,$X->dtype(),[$shape[$axis]]);
        $w = 1;
        $p = 1;
        $pos =0;
        for($i=$rank-1;$i>=0;$i--) {
            if($axis!=$i) {
                $pos += $idx[$i]*$w;
            } else {
                $K = $w;
            }
            $w *= $shape[$i];
        }
        $N = $shape[$axis];
        $bufX = $X->buffer();
        $offX = $X->offset();
        for ($i=0; $i<$N ; $i++) {
            $Y[$i] = $bufX[$offX + $pos + $i*$K];
        }
        return $Y;
    }

    protected function walkAxis(callable $funcLinear, callable $funcAxis, NDArray $X,int $axis=null, $dtype=null)
    {
        $N = $X->size();
        $XX = $X->buffer();
        $offX = $X->offset();
        $shape = $X->shape();
        if($axis===null) {
            return $funcLinear($N,$XX,$offX,$bufPos=0,$incX=1);
        }
        if($axis<0 || $axis>=count($shape)) {
            throw new InvalidArgumentException('Invalid axis: axis='.$axis.',shape=['.implode(',',$shape).']');
        }
        $incX = $this->calcAxisStep($shape,$axis);
        $bufIterator = new MatrixBufferIterator($shape,[$axis]);
        if($dtype==null)
            $dtype = $X->dtype();
        $Y = $this->alloc(null,$dtype,$this->projectionShape($shape,[$axis]));
        $YY = $Y->buffer();
        $posY = 0;

        // OLD $N = $shape[$axis]*$incX;
        $N = $shape[$axis];
        foreach($bufIterator as $bufPos) {
            $YY[$posY] = $funcAxis($N,$XX,$offX,$bufPos,$incX);
            $posY++;
        }
        return $Y;
    }

    protected function calcAxisStep(array $shape,int $axis)
    {
        $w = 1;
        for($i=count($shape)-1; $axis<$i; $i--) {
            $w *= $shape[$i];
        }
        return $w;
    }

    protected function projectionShape(array $shape,array $skipDims)
    {
        $newShape = [];
        foreach ($shape as $key => $value) {
            if(in_array($key,$skipDims))
                continue;
            array_push($newShape,$value);
        }
        return $newShape;
    }

    public function sum(NDArray $X,int $axis=null)
    {
        if($X->dtype()==NDArray::bool) {
            $func = function($N,$XX,$offX,$bufPos,$incX) {
                $acc = 0;
                for($i=0; $i<$N; $i+=$incX) {
                    if($XX[$offX+$bufPos+$i])
                        $acc++;
                }
                return $acc;
            };
        } else {
            $func = function($N,$XX,$offX,$bufPos,$incX) {
                return $this->math->sum($N,$XX,$offX+$bufPos,$incX);
            };
        }
        return $this->walkAxis($func,$func,$X,$axis);
    }

    public function asum(NDArray $X,int $axis=null)
    {
        $func = function($N,$XX,$offX,$bufPos,$incX) {
            return $this->blas->asum($N,$XX,$offX+$bufPos,$incX);
        };
        return $this->walkAxis($func,$func,$X,$axis);
    }

    public function max(NDArray $X,int $axis=null)
    {
        $func = function($N,$XX,$offX,$bufPos,$incX) {
            $pos = $this->math->imax($N,$XX,$offX+$bufPos,$incX);
            return $XX[$offX+$bufPos+$pos*$incX];
        };

        return $this->walkAxis($func,$func,$X,$axis);
    }

    public function argMax(NDArray $X,int $axis=null)
    {
        $func = function($N,$XX,$offX,$bufPos,$incX) {
            $pos = $this->math->imax($N,$XX,$offX+$bufPos,$incX);
            return $pos;
        };

        return $this->walkAxis($func,$func,$X,$axis,$this->defaultIntType);
    }

    public function amax(NDArray $X,int $axis=null)
    {
        $func = function($N,$XX,$offX,$bufPos,$incX) {
            $pos = $this->blas->iamax($N,$XX,$offX+$bufPos,$incX);
            return $XX[$offX+$bufPos+$pos*$incX];
        };

        return $this->walkAxis($func,$func,$X,$axis);
    }

    public function argAmax(NDArray $X,int $axis=null)
    {
        $func = function($N,$XX,$offX,$bufPos,$incX) {
            $pos = $this->blas->iamax($N,$XX,$offX+$bufPos,$incX);
            return $pos;
        };

        return $this->walkAxis($func,$func,$X,$axis,$this->defaultIntType);
    }

    public function min(NDArray $X,int $axis=null)
    {
        $func = function($N,$XX,$offX,$bufPos,$incX) {
            $pos = $this->math->imin($N,$XX,$offX+$bufPos,$incX);
            return $XX[$offX+$bufPos+$pos*$incX];
        };

        return $this->walkAxis($func,$func,$X,$axis);
    }

    public function argMin(NDArray $X,int $axis=null)
    {
        $func = function($N,$XX,$offX,$bufPos,$incX) {
            $pos = $this->math->imin($N,$XX,$offX+$bufPos,$incX);
            return $pos;
        };

        return $this->walkAxis($func,$func,$X,$axis,$this->defaultIntType);
    }

    public function amin(NDArray $X,int $axis=null)
    {
        $func = function($N,$XX,$offX,$bufPos,$incX) {
            $pos = $this->blas->iamin($N,$XX,$offX+$bufPos,$incX);
            return $XX[$offX+$bufPos+$pos*$incX];
        };

        return $this->walkAxis($func,$func,$X,$axis);
    }

    public function argAmin(NDArray $X,int $axis=null)
    {
        $func = function($N,$XX,$offX,$bufPos,$incX) {
            $pos = $this->blas->iamin($N,$XX,$offX+$bufPos,$incX);
            return $pos;
        };

        return $this->walkAxis($func,$func,$X,$axis,$this->defaultIntType);
    }

    public function mean($X,$axis=null)
    {
        $func = function($N,$XX,$offX,$bufPos,$incX) {
            $sum = $this->math->sum($N,$XX,$offX+$bufPos,$incX);
            return $sum/$N;
        };
        return $this->walkAxis($func,$func,$X,$axis);
    }


    public function f(callable $func,NDArray $X, ...$args) : NDArray
    {
        $N = $X->size();
        $XX = $X->buffer();
        $offX = $X->offset();
        $Y = $this->alloc(null, $X->dtype(), $X->shape());
        $YY = $Y->buffer();
        $pos = 0;
        $limit = $offX + $N;
        for(; $offX<$limit; $offX++,$pos++) {
            $YY[$pos] = call_user_func($func,$XX[$offX], ...$args);
        }
        return $Y;
    }

    public function u(NDArray $X,callable $func, ...$args) : NDArray
    {
        $N = $X->size();
        $XX = $X->buffer();
        $offX = $X->offset();
        $limit = $offX + $N;
        for(; $offX<$limit; $offX++) {
            $XX[$offX] = call_user_func($func,$XX[$offX], ...$args);
        }
        return $X;
    }

    public function op($X, string $operator, $Y, NDArray $R=null) : NDArray
    {
        if(!array_key_exists($operator,$this->broadCastOperators)) {
            throw new InvalidArgumentException('Unknown operator: "'.$operator.'""');
        }
        if(!($X instanceof NDArray)&&!($Y instanceof NDArray)) {
            throw new InvalidArgumentException('Requires at least one matrix.');
        }
        if(($X instanceof NDArray)&&($Y instanceof NDArray)) {
            if($X->shape()==$Y->shape()) {
                if($X->dtype()==$Y->dtype()) {
                    if($operator=='+') {
                        if($R===null) {
                            $R = $this->alloc(null,$Y->dtype(),$Y->shape());
                        }
                        $this->la()->copy($Y,$R);
                        return $this->la()->axpy($X,$R);
                    } elseif ($operator=='-') {
                        if($R===null) {
                            $R = $this->alloc(null,$X->dtype(),$X->shape());
                        }
                        $this->la()->copy($X,$R);
                        return $this->la()->axpy($Y,$R,-1.0);
                    }
                }
            } else {
                if($X->ndim()>$Y->ndim()) {
                    $n = $X->shape()[0];
                    if($R===null)
                        $R = $this->alloc(null,
                                $this->maximumPrecision($X->dtype(),$Y->dtype()),
                                $X->shape());
                    for($i=0;$i<$n;$i++) {
                        $this->op($X[$i],$operator,$Y,$R[$i]);
                    }
                    return $R;
                } elseif($X->ndim()<$Y->ndim()) {
                    $n = $Y->shape()[0];
                    if($R===null)
                        $R = $this->alloc(null,
                            $this->maximumPrecision($X->dtype(),$Y->dtype()),
                            $Y->shape());
                    for($i=0;$i<$n;$i++) {
                        $this->op($X,$operator,$Y[$i],$R[$i]);
                    }
                    return $R;
                } else {
                    throw new InvalidArgumentException('The shape of the matrix must be the same.');
                }
            }
        } elseif(($X instanceof NDArray)||($Y instanceof NDArray)) {
            if($operator=='*') {
                if($X instanceof NDArray && is_numeric($Y)) {
                    if($R===null) {
                        $R = $this->alloc(null,$X->dtype(),$X->shape());
                    }
                    $this->la()->copy($X,$R);
                    return $this->la()->scal($Y,$R);
                } elseif($Y instanceof NDArray && is_numeric($X)) {
                    if($R===null) {
                        $R = $this->alloc(null,$Y->dtype(),$Y->shape());
                    }
                    $this->la()->copy($Y,$R);
                    return $this->la()->scal($X,$R);
                }
            } elseif($operator=='/') {
                if($X instanceof NDArray && is_numeric($Y)) {
                    if($Y==0.0) {
                        throw new RuntimeException('Zero divide error');
                    }
                    if($R===null) {
                        $R = $this->alloc(null,$X->dtype(),$X->shape());
                    }
                    $this->la()->copy($X,$R);
                    return $this->la()->scal(1/$Y,$R);
                }
            }
        }
        [$dtype,$func] = $this->broadCastOperators[$operator];
        if($X instanceof NDArray) {
            $shape = $X->shape();
            if($dtype===null) {
                if($Y instanceof NDArray) {
                    $dtype = $this->maximumPrecision($X->dtype(),$Y->dtype());
                } else {
                    if(is_float($Y)) {
                        $dtype = $this->maximumPrecision($X->dtype(),NDArray::float32);
                    } else {
                        $dtype = $X->dtype();
                    }
                }
            }
        } else {
            $shape = $Y->shape();
            if($dtype===null) {
                if($X instanceof NDArray) {
                    $dtype = $this->maximumPrecision($X->dtype(),$Y->dtype());
                } else {
                    if(is_float($X)) {
                        $dtype = $this->maximumPrecision($Y->dtype(),NDArray::float32);
                    } else {
                        $dtype = $Y->dtype();
                    }
                }
            }
        }
        if($R) {
            if($shape!=$R->shape()||$dtype!=$R->dtype()) {
                throw new InvalidArgumentException('Unmatch the shape of the result matrix.');
            }
        } else {
            $R = $this->alloc(null, $dtype, $shape);
        }
        $RR = $R->buffer();
        $offR = $R->offset();

        if($X instanceof NDArray) {
            $N = $X->size();
            $XX = $X->buffer();
            $offX = $X->offset();
            if($Y instanceof NDArray) {
                $YY = $Y->buffer();
                $offY = $Y->offset();
                for($i=0; $i<$N; $i++) {
                    $RR[$offR+$i] = $this->operatorFunctions->$func($XX[$offX+$i], $YY[$offY+$i]);
                }
            } else {
                for($i=0; $i<$N; $i++) {
                    $RR[$offR+$i] = $this->operatorFunctions->$func($XX[$offX+$i], $Y);
                }
            }
        } else {
            $N = $Y->size();
            $YY = $Y->buffer();
            $offY = $Y->offset();
            for($i=0; $i<$N; $i++) {
                $RR[$offR+$i] = $this->operatorFunctions->$func($X, $YY[$offY+$i]);
            }
        }

        return $R;
    }

    public function select(NDArray $X, NDArray ...$MASKs) : NDArray
    {
        if(count($MASKs)==0) {
            throw new InvalidArgumentException('Need The mask matrix.');
        }
        $maskDtype = $MASKs[0]->dtype();
        if($maskDtype==NDArray::bool) {
            if(count($MASKs)!=1) {
                throw new InvalidArgumentException('The bool Matrix must be only one.');
            }
            if($X->shape()!=$MASKs[0]->shape()) {
                throw new InvalidArgumentException('The shape of the matrix must be the same.');
            }
            $N = $X->size();
            $XX = $X->buffer();
            $offX = $X->offset();
            $MM = $MASKs[0]->buffer();
            $offM = $MASKs[0]->offset();
            $count = 0;
            for($i=0; $i<$N; $i++) {
                if($MM[$offM+$i]) {
                    $count++;
                }
            }
            $R = $this->alloc(null,$X->dtype(),[$count]);
            $this->selectByMask($X,$MASKs,$R);
            return $R;
        } elseif(in_array($maskDtype,$this->intTypes)) {
            $shape = $X->shape();
            $maskShape = null;
            foreach ($MASKs as $mask) {
                if($maskShape==null) {
                    $maskShape = $mask->shape();
                } else {
                    if($maskShape!=$mask->shape()) {
                        foreach($MASKs as $m) {
                            if(isset($shapeError))
                                $shapeError .= ',';
                            else
                                $shapeError = '';
                            $shapeError .= '('.implode(',',$m->shape()).')';
                        }
                        throw new InvalidArgumentException('The shapes of the indexing matrix must all be the same.: '.$shapeError);
                    }
                }
                $nX = array_shift($shape);
                if(!$nX)
                    throw new InvalidArgumentException('To many indexing matrix.');
            }
            $newShape = array_merge($MASKs[0]->shape(),$shape);
            $newShape = array_values($newShape);
            $R = $this->alloc(null,$X->dtype(),$newShape);

            $this->selectByMatrix($X,$MASKs,$R);
            return $R;
        } else {
            throw new InvalidArgumentException('The mask matrix must be type of the bool or int.');
        }
    }

    public function update(NDArray $X, string $operator, $value, NDArray ...$MASKs) : NDArray
    {
        if(!array_key_exists($operator,$this->updateOperators)) {
            throw new InvalidArgumentException('Unknown operator: "'.$operator.'""');
        }
        if(count($MASKs)==0) {
            throw new InvalidArgumentException('Need The mask matrix.');
        }
        [$dtype,$func] = $this->updateOperators[$operator];
        $maskDtype = $MASKs[0]->dtype();
        if($maskDtype==NDArray::bool) {
            $this->selectByMask($X, $MASKs, $R=null, $func, $value);
        } elseif(in_array($maskDtype,$this->intTypes)) {
            $this->selectByMatrix($X, $MASKs, $R=null, $func, $value);
        } else {
            throw new InvalidArgumentException('The mask matrix must be type of the bool or int.');
        }

        return $X;
    }

    protected function selectByMask(NDArray $X, array $MASKs, NDArray $R=null, string $func=null, $updateValue=null)
    {
        if(count($MASKs)!=1) {
            throw new InvalidArgumentException('The bool Matrix must be only one.');
        }
        $MASK = $MASKs[0];
        if($X->shape()!=$MASK->shape()) {
            throw new InvalidArgumentException('The shape of the masking matrix must be the same.');
        }
        $N = $X->size();
        $XX = $X->buffer();
        $offX = $X->offset();
        $MM = $MASK->buffer();
        $offM = $MASK->offset();
        if($R) {
            $RR = $R->buffer();
        }
        $count = 0;
        for($i=0; $i<$N; $i++) {
            if($MM[$offM+$i]) {
                if($R) {
                    $RR[$count] = $XX[$offX+$i];
                    $count++;
                } else {
                    $XX[$offX+$i] = $this->operatorFunctions->$func($XX[$offX+$i],$updateValue);
                }
            }
        }

    }

    protected function selectByMatrix(NDArray $X, array $MASKs, NDArray $R=null, string $func=null, $updateValue=null)
    {
        $N = $MASKs[0]->shape()[0];
        // nested mask
        if($MASKs[0]->ndim()>1) {

            $nextR = null;
            for($i=0;$i<$N;$i++) {
                $newMASKs = [];
                foreach($MASKs as $mask) {
                    $newMASKs[] = $mask[$i];
                }
                if($R) {
                    $nextR = $R[$i];
                }
                $this->selectByMatrix($X,$newMASKs,$nextR,$func,$updateValue);
            }
            return;
        }

        // select scalar values
        if($X->ndim()==count($MASKs)) {

            $XX = $X->buffer();
            $offX = $X->offset();
            $w = 1;
            $shapeX = $X->shape();
            $W = [];
            for($m=count($MASKs)-1; $m>0; $m--) {
                $W[$m] = $w;
                array_shift($shapeX);
                $w *= $shapeX[0];
            }
            $W[$m] = $w;
            ksort($W);

            for($i=0;$i<$N;$i++) {
                $idx = 0;
                for($m=count($MASKs)-1; $m>=0; $m--) {
                    $idx += $W[$m]*$MASKs[$m][$i];
                }
                if($R) {
                    try {
                        $R[$i] = $XX[$offX+$idx];
                    } catch(\Exception $e) {
                        var_dump($i);
                        var_dump($offX);
                        var_dump($idx);
                        $mmW = $mmMASKs = [];
                        for($mmmmm=0;$mmmmm<count($MASKs);$mmmmm++) {
                            $mmW[]=$W[$mmmmm];
                            $mmMASKs[]=$MASKs[$mmmmm][$i];
                        }
                        echo "W=[".implode(',',$mmW)."]\n";
                        echo "MASKs=[".implode(',',$mmMASKs)."]\n";
                        throw $e;
                    }
                } else {
                    $XX[$offX+$idx] = $this->operatorFunctions->$func($XX[$offX+$idx],$updateValue);
                }
            }

            return;
        }

        // select NDAarray values
        $XX =   $X->buffer();
        $offX = $X->offset();
        if($R) {
            $RR =   $R->buffer();
            $offR = $R->offset();
        }

        $w = 1;
        $shapeX = $X->shape();
        $W = [];
        for($m=count($MASKs)-1; $m>=0; $m--) {
            $W[$m] = $w;
            array_shift($shapeX);
            $w *= $shapeX[0];
        }
        ksort($W);

        $size = 1;
        foreach ($shapeX as $value) {
            $size *= $value;
        }

        for($i=0;$i<$N;$i++) {
            $idx = 0;
            for($m=count($MASKs)-1; $m>=0; $m--) {
                $idx += $W[$m]*$MASKs[$m][$i];
            }

            if($R) {
                $this->blas->copy($size,
                    $XX,$offX+$size*$idx,$incX=1,
                    $RR,$offR+$size*$i,$incR=1);
            } else {
                $offset = $offX+$size*$idx;
                for($j=0;$j<$size;$j++) {
                    $XX[$offset+$j] = $this->operatorFunctions->$func($XX[$offset+$j],$updateValue);
                }
            }
        }
    }

    public function dtypeToString($dtype) : string
    {
        if(!isset($this->dtypeToString[$dtype])) {
            return 'Unknown';
        }
        return $this->dtypeToString[$dtype];
    }
    public function toString(
        NDArray $array,
        string $format=null,
        $indent=null) : string
    {
        $shape = $array->shape();
        $n = array_shift($shape);
        if(!is_numeric($indent) && $indent===true) {
            $indent=1;
        }
        if(count($shape)==0) {
            if($array->dtype()==NDArray::bool) {
                $str = '';
                foreach($array->toArray() as $value) {
                    $str .= ($str==='') ? '[' : ',';
                    $str .= $value ? 'true' : 'false';
                }
                $str .= ']';
                return $str;
            } else {
                if($format) {
                    return '['.implode(',',array_map(function($x) use ($format) {
                            return sprintf($format,$x);
                        },$array->toArray())).']';
                } else {
                    return '['.implode(',',$array->toArray()).']';
                }
            }
        }
        $string = '[';
        if($indent) {
            $string .= "\n";
        }
        for($i=0;$i<$n;$i++) {
            if($i!=0) {
                $string .= ',';
                if($indent) {
                    $string .= "\n";
                }
            }
            if($indent) {
                $string .= str_repeat(' ',$indent);
                $string .= $this->toString($array[$i],$format,$indent+1);
            } else {
                $string .= $this->toString($array[$i],$format,$indent);
            }
        }
        if($indent) {
            $string .= "\n";
            $string .= str_repeat(' ',$indent-1);
        }
        $string .= ']';
        return $string;
    }

    public function random()
    {
        if($this->random==null) {
            $this->random = new Random($this,$this->defaultFloatType);
        }
        return $this->random;
    }

    public function la()
    {
        if($this->la==null) {
            $this->la = new LinearAlgebra(
                $this->blas,$this->lapack,$this->math,$this->defaultFloatType);
        }
        return $this->la;
    }

    public function laRawMode()
    {
        if(!extension_loaded('rindow_openblas')) {
            return $this->la();
        }
        if($this->laRawMode==null) {
            $this->laRawMode = new LinearAlgebra(
                $this->openblas,$this->openblaslapack,$this->openblasmath);
        }
        return $this->laRawMode;
    }

    public function laAccelerated($mode,array $options=null)
    {
        if($mode=='clblast') {
            if($this->clblastLA) {
                return $this->clblastLA;
            }
            if(!extension_loaded('rindow_clblast')) {
                throw new InvalidArgumentException('extension is not loaded');
            }
            if(isset($options['deviceType'])) {
                $deviceType = $options['deviceType'];
            } else {
                $deviceType = OpenCL::CL_DEVICE_TYPE_DEFAULT;
            }
            $context = new Context($deviceType);
            $queue = new CommandQueue($context);
            $clblastblas = new CLBlastBlas();
            $openclmath = new OpenCLMath($context,$queue);
            $clblastmath = new CLBlastMath();
            $la = new LinearAlgebraCL($context,$queue,
                $clblastblas,$openclmath,$clblastmath,
                $this->openblasmath,$this->openblaslapack);
            $this->clblastLA = $la;
            return $la;
        }
    }

    public function blas($raw=null)
    {
        if($raw) {
            return $this->openblas;
        }
        return $this->blas;
    }

    public function lapack($raw=null)
    {
        if($raw) {
            return $this->openblaslapack;
        }
        return $this->lapack;
    }

    public function math($raw=null)
    {
        if($raw) {
            return $this->openblasmath;
        }
        return $this->math;
    }
}
