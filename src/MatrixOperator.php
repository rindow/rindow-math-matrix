<?php
namespace Rindow\Math\Matrix;

require_once __DIR__.'/R.php';

use ArrayObject;
use InvalidArgumentException;
use RuntimeException;
use LogicException;
use RangeException;
use Iterator;
use Traversable;
use Interop\Polite\Math\Matrix\BLAS;
use Interop\Polite\Math\Matrix\NDArray;
use Interop\Polite\Math\Matrix\Buffer;
use Rindow\Math\Matrix\Drivers\Service;
use Rindow\Math\Matrix\Drivers\Selector;

class MatrixOperator
{
    use ComplexUtils;

    const SERIALIZE_NDARRAYSET_KEYWORD = 'NDArraySet:';
    const SERIALIZE_NDARRAY_KEYWORD = 'NDArray:';
    const SERIALIZE_OLDSTYLE_KEYWORD = 'O:29:"Rindow\Math\Matrix\NDArrayPhp"';

    protected Service $service;
    protected ?object $random=null;
    protected ?object $la=null;
    protected ?object $laPhp=null;
    protected ?object $clblastLA=null;
    /** @var array<string,array{null|int,string}> $broadCastOperators */
    protected array $broadCastOperators;
    /** @var array<string,array{null,string}> $updateOperators */
    protected array $updateOperators;
    /** @var object $operatorFunctions */
    protected object $operatorFunctions;
    /** @var array<int,bool> $intTypes */
    protected array $intTypes= [
        NDArray::int8 => true,
        NDArray::int16 => true,
        NDArray::int32 => true,
        NDArray::int64 => true,
        NDArray::uint8 => true,
        NDArray::uint16 => true,
        NDArray::uint32 => true,
        NDArray::uint64 => true,
    ];
    /** @var array<int> $floatTypes */
    protected array $floatTypes= [
        NDArray::float16,NDArray::float32,NDArray::float64,
    ];
    protected int $defaultIntType = NDArray::int32;
    protected int $defaultFloatType = NDArray::float32;
    /** @var array<int,string> $dtypeToString */
    protected array $dtypeToString = [
        NDArray::bool=>'bool',
        NDArray::int8=>'int8',   NDArray::uint8=>'uint8',
        NDArray::int16=>'int16', NDArray::uint16=>'uint16',
        NDArray::int32=>'int32', NDArray::uint32=>'uint32',
        NDArray::int64=>'int64', NDArray::uint64=>'uint64',
        NDArray::float16=>'float16',
        NDArray::float32=>'float32', NDArray::float64=>'float64',
        NDArray::complex64=>'complex64', NDArray::complex128=>'complex128',
    ];
    
    /** @var array<int,int> $dtypePrecision */
    protected array $dtypePrecision = [
        NDArray::bool=>1,
        NDArray::int8=>2,  NDArray::uint8=>3,
        NDArray::int16=>4, NDArray::uint16=>5,
        NDArray::int32=>6, NDArray::uint32=>7,
        NDArray::int64=>8, NDArray::uint64=>9,
        NDArray::float16=>10,
        NDArray::float32=>11, NDArray::float64=>12,
    ];

    /**
     * @param array<mixed> $catalog
     */
    public function __construct(
        Service $service=null,
        array $catalog=null,
        int $verbose=null,
        )
    {
        if($service===null) {
            $selector = new Selector($catalog);
            $service = $selector->select(verbose:$verbose);
        }
        $this->service = $service;

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
        $this->operatorFunctions = $this->createMatrixOpelatorFunctions();
    }

    protected function createMatrixOpelatorFunctions() : object
    {
        /**
        *  Alternate anonymous functions.
        *  If you store an anonymous function in an instance variable of MatrixOperator,
        *  a recursive reference will occur. An alternative to anonymous functions was
        *  needed to prevent memory leaks from recursive references.
        */
        return new class 
        {
            // broadCastOperators
            //     '+' =>  [null,  function($x,$y) { return $x + $y; }],
            //     '-' =>  [null,  function($x,$y) { return $x - $y; }],
            //     '*' =>  [null,  function($x,$y) { return $x * $y; }],
            //     '/' =>  [null,  function($x,$y) { return $x / $y; }],
            //     '%' =>  [null,  function($x,$y) { return $x % $y; }],
            //     '**' => [null,  function($x,$y) { return $x ** $y; }],
            //     '==' => [NDArray::bool,function($x,$y) { return ($x == $y); }],
            //     '!=' => [NDArray::bool,function($x,$y) { return $x != $y; }],
            //     '>' =>  [NDArray::bool,function($x,$y) { return $x > $y; }],
            //     '>=' => [NDArray::bool,function($x,$y) { return $x >= $y; }],
            //     '<' =>  [NDArray::bool,function($x,$y) { return $x < $y; }],
            //     '<=' => [NDArray::bool,function($x,$y) { return $x <= $y; }],
        
            public function add(int|float $x,int|float $y) : int|float { return $x + $y; } // '+'
            public function sub(int|float $x,int|float $y) : int|float { return $x - $y; } // '-'
            public function mul(int|float $x,int|float $y) : int|float { return $x * $y; } // '*'
            public function div(int|float $x,int|float $y) : int|float { return $x / $y; } // '/'
            public function mod(int|float $x,int|float $y) : int|float { return $x % $y; } // '%'
            public function pow(int|float $x,int|float $y) : int|float { return $x ** $y; } // '**'
            public function is_equal(int|float $x,int|float $y) : bool { return ($x == $y); } // '=='
            public function is_not_equal(int|float $x,int|float $y) : bool { return $x != $y; } // '!='
            public function greater(int|float $x,int|float $y) : bool { return $x > $y; } // '>'
            public function greater_or_equal(int|float $x,int|float $y) : bool { return $x >= $y; } // '>='
            public function smaller(int|float $x,int|float $y) : bool { return $x < $y; } // '<'
            public function smaller_or_equal(int|float $x,int|float $y) : bool { return $x <= $y; } // '<='
        
            // updateOperators
            //     '='  =>  [null,  function($x,$y) { return $y; }],
            //     '+=' =>  [null,  function($x,$y) { return $x + $y; }],
            //     '-=' =>  [null,  function($x,$y) { return ($x - $y); }],
            //     '*=' =>  [null,  function($x,$y) { return $x * $y; }],
            //     '/=' =>  [null,  function($x,$y) { return $x / $y; }],
            //     '%=' =>  [null,  function($x,$y) { return $x % $y; }],
            //     '**=' => [null,  function($x,$y) { return $x ** $y; }],
        
            public function assign(int|float $x,int|float $y) : int|float { return $y; } // '='
            public function assign_add(int|float $x, int|float $y) : int|float { return $x + $y; } // '+='
            public function assign_sub(int|float $x, int|float $y) : int|float { return ($x - $y); } // '-='
            public function assign_mul(int|float $x, int|float $y) : int|float { return $x * $y; } // '*='
            public function assign_div(int|float $x, int|float $y) : int|float { return $x / $y; } // '/='
            public function assign_mod(int|float $x, int|float $y) : int|float { return $x % $y; } // '%='
            public function assign_pow(int|float $x, int|float $y) : int|float { return $x ** $y; } // '**='
        
        };
    }

    //public function close()
    //{
    //    $this->broadCastOperators = null;
    //    $this->updateOperators = null;
    //}

    /**
     * @param array<int> $shape
     */
    protected function alloc(mixed $array,int $dtype=null,array $shape=null) : NDArray
    {
        if($dtype===null) {
            //$dtype = $this->resolveDtype($array);
            $dtype = $this->defaultFloatType;
        }
        return new NDArrayPhp($array,$dtype,$shape,service:$this->service);
    }

    //protected function resolveDtype($value)
    //{
    //    while((is_array($value)||$value instanceof ArrayObject)&&isset($value[0])) {
    //        $value = $value[0];
    //    }
    //    if(is_int($value)) {
    //        return $this->defaultIntType;
    //    } elseif(is_float($value)) {
    //        return $this->defaultFloatType;
    //    } elseif(is_bool($value)) {
    //        return NDArray::bool;
    //    }
    //    return null;
    //}

    protected function maximumPrecision(int $dtypeX, int $dtypeY) : int
    {
        if(!isset($this->dtypePrecision[$dtypeX])) {
            throw new RuntimeException("Illegal dtype: ".$dtypeX);
        }
        if(!isset($this->dtypePrecision[$dtypeY])) {
            throw new RuntimeException("Illegal dtype: ".$dtypeY);
        }

        if($this->dtypePrecision[$dtypeX]>$this->dtypePrecision[$dtypeY])
            return $dtypeX;
        else
            return $dtypeY;
    }

    protected function getBlas(int $dtype) : object
    {
        if($this->isIntType($dtype)) {
            return $this->service->blas(Service::LV_BASIC);
        }
        return $this->service->blas();
    }

    protected function getMath(int $dtype) : object
    {
        if($this->isIntType($dtype)) {
            return $this->service->math(Service::LV_BASIC);
        }
        return $this->service->math();
    }

    protected function getLa(int $dtype) : object
    {
        if($this->isIntType($dtype)) {
            return $this->laPhpMode();
        }
        return $this->la();
    }

    public function setDefaultIntType(int $dtype) : void
    {
        $this->defaultIntType = $dtype;
    }

    public function setDefaultFloatType(int $dtype) : void
    {
        $this->defaultFloatType = $dtype;
    }

    protected function isComplexDtype(?int $dtype) : bool
    {
        return $this->cistype($dtype);
    }

    public function toComplex(mixed $array) : mixed
    {
        if(is_array($array)||is_a($array,Traversable::class)) {
            $cArray = [];
            foreach($array as $value) {
                $cArray[] = $this->toComplex($value);
            }
            return $cArray;
        }

        if(is_numeric($array)) {
            return C((float)$array,i:0);
        } elseif(is_object($array) && 
                property_exists($array,'real') &&
                property_exists($array,'imag')) {
            return C($array->real,i:$array->imag);
        } else {
            if(is_object($array)) {
                $name = get_class($array);
            } else {
                $name = gettype($array);
            }
            throw new InvalidArgumentException("invalid data type: ".$name);
        }
    }

    public function array(mixed $array, int $dtype=null) : NDArray
    {
        if($dtype==null) {
            $dtype=$this->defaultFloatType;
            //if(is_bool($array)) {
            //    $dtype = NDArray::bool;
            //} else {
            //    $dtype = $this->resolveDtype($array);
            //    if($dtype!=NDArray::bool) {
            //        $dtype=$this->defaultFloatType;
            //    }
            //}
        }
        if(is_array($array)||is_object($array)||is_numeric($array)) {
            if($this->isComplexDtype($dtype)) {
                $array = $this->toComplex($array);
            }
            return $this->alloc($array,$dtype);
        } elseif(is_bool($array)) {
            return $this->alloc($array,$dtype);
        } else {
            throw new InvalidArgumentException("Must be array or ArrayObject");
        }
    }

    /**
     * @param array<int> $shape
     */
    public function zeros(array $shape, int $dtype=null) : NDArray
    {
        if($this->isComplexDtype($dtype)) {
            $value = $this->cbuild(0,0);
        } else {
            $value = 0.0; // must be float
        }
        return $this->full($shape,$value,$dtype);
    }

    /**
     * @param array<int> $shape
     */
    public function ones(array $shape, int $dtype=null) : NDArray
    {
        if($this->isComplexDtype($dtype)) {
            $value = $this->cbuild(1,0);
        } else {
            $value = 1.0; // must be float
        }
        return $this->full($shape,$value,$dtype);
    }

    /**
     * @param array<int> $shape
     */
    public function full(array $shape, mixed $value, int $dtype=null) : NDArray
    {
        if($dtype===null) {
            //$dtype=$this->resolveDtype($value);
            $dtype=$this->defaultFloatType;
        }
        if($this->isComplexDtype($dtype)) {
            if(!$this->cisobject($value)) {
                throw new InvalidArgumentException('value must be complex');
            }
        } else {
            if(!is_scalar($value)) {
                throw new InvalidArgumentException('value must be scalar');
            }
        }

        $array = $this->alloc(null, $dtype, $shape);
        //$buffer = $array->buffer();
        //$size = count($buffer);
        //for($i=0; $i<$size; $i++) {
        //    $buffer[$i] = $value;
        //}
        $this->la()->fill($value,$array);
        return $array;
    }

    public function zerosLike(NDArray $array) : NDArray
    {
        return $this->zeros($array->shape(),$array->dtype());
    }

    public function fullLike(NDArray $array, int $value) : NDArray
    {
        return $this->full($array->shape(),$value,$array->dtype());
    }

    public function astype(NDArray $array, int $dtype) : NDArray
    {
        return $this->la()->astype($array,$dtype);
    }

    public function isIntType(int $dtype) : bool
    {
        return array_key_exists($dtype, $this->intTypes);
    }

    public function isFloatType(int $dtype) : bool
    {
        return array_key_exists($dtype, $this->floatTypes);
    }

    public function arange(
        int $count,
        int|float $start=null,
        int|float $step=null,
        int $dtype=null
        ) : NDArray
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
        $this->getLa($array->dtype())->copy($array,$newArray);
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
            $blas = $this->getBlas($A->dtype());
            $blas->gemm(
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
        int $M, int $N, int $K, int $L,
        float $alpha,
        Buffer $A,int $offA,
        Buffer $B,int $offB,
        float $beta,
        Buffer $C,int $offC) : void
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

    protected function vectorTransform(NDArray $A, NDArray $B) : NDArray
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
            $blas = $this->getBlas($A->dtype());
            $blas->gemv(
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

    /**
     * @param array<int>|NDArray $perm
     */
    public function transpose(
        NDArray $X,
        array|NDArray $perm=null,
        ) : NDArray
    {
        return $this->getLa($X->dtype())->transpose($X,perm:$perm);
    }

    //public function transpose(NDArray $X) : NDArray
    //{
    //    $shape = $X->shape();
    //    $newShape = array_reverse($shape);
    //    $Y = $this->alloc(null, $X->dtype(), $newShape);
    //    $w = 1;
    //    $posY = 0;
    //    $posX = 0;
    //    $this->_transpose($newShape, $w, $X->buffer(), $X->offset(), $posX, $Y->buffer(), $posY);
    //    return $Y;
    //}
//
    //protected function _transpose($shape, $w, $bufX, $offX, $posX, $bufY, &$posY)
    //{
    //    $n=array_shift($shape);
    //    $W = $w*$n;
    //    $deps = count($shape);
    //    for($i=0;$i<$n;$i++) {
    //        if($deps) {
    //            $this->_transpose($shape, $W, $bufX, $offX, $posX+$w*$i, $bufY, $posY);
    //        } else {
    //            $bufY[$posY] = $bufX[$offX + $posX+$w*$i];
    //            $posY++;
    //        }
    //    }
    //}

    public function dot(NDArray $A, NDArray $B) : mixed
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
        $blas = $this->getBlas($A->dtype());
        return $blas->dot($N,$AA,$offA,1,$BB,$offB,1);
    }

    public function add(NDArray $X, NDArray $Y) : NDArray
    {
        if($X->shape()!=$Y->shape()) {
            $shapeError = '('.implode(',',$X->shape()).'),('.implode(',',$Y->shape()).')';
            throw new InvalidArgumentException("Unmatch shape of dimension to add: ".$shapeError);
        }

        $C = $this->copy($Y);
        return $this->getLa($X->dtype())->axpy($X,$C);
    }

    public function scale(mixed $a, NDArray $X) : NDArray
    {
        if(!is_numeric($a))
            throw new InvalidArgumentException("the scalar must be a numeric");
        $N = $X->size();
        $C = $this->copy($X);
        $CC = $C->buffer();
        $offC = $C->offset();
        $blas = $this->getBlas($X->dtype());
        $blas->scal($N,$a,$CC,$offC,1);

        return $C;
    }

    /**
     *  DISCONTINUE
     * @param array<int> $shape
     * @return array<int>
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

            $p = intdiv($pos , $w);
            $pos = $pos % $w;
            array_push($index,$p);
        }
        return $index;
    }

    /**
     *  DISCONTINUE
     * @param array<mixed> $idx
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
        $K = 1;
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

    /**
     * @param array<int> $shape
     * @param array<int> $skipDims
     * @return array<int>
     */
    protected function createMatrixBufferIterator(array $shape, array $skipDims) : iterable
    {
        return new class($shape,$skipDims) implements Iterator
        {
            /** @var array<int> $shape */
            protected array $shape;
            /** @var array<int> $skipDims */
            protected array $skipDims;
            /** @var array<int> $current */
            protected array $current;
            protected bool $endOfItem = false;
        
            /**
             * @param array<int> $shape
             * @param array<int> $skipDims
             */
            public function __construct(array $shape, array $skipDims)
            {
                $this->shape = $shape;
                $this->skipDims = $skipDims;
                $this->current = array_fill(0,count($shape),0);
            }

            /**
             * @return array<int>
             */
            public function getCurrentIndex() : array
            {
                return $this->current;
            }

            public function current() : mixed
            {
                if($this->endOfItem) {
                    throw new RangeException('End of buffer');
                }
                $w = 1;
                $pos = 0;
                for($i=count($this->shape)-1; $i>=0; $i--) {
                    $pos += $this->current[$i]*$w;
                    $w *= $this->shape[$i];
                }
                return $pos;
            }
        
            public function key() : mixed
            {
                return null;
            }
        
            public function next() : void
            {
                if($this->endOfItem)
                    return;
        
                for($dimNum=count($this->shape)-1; $dimNum>=0; $dimNum--) {
                    if(in_array($dimNum,$this->skipDims))
                        continue;
                    $this->current[$dimNum]++;
                    if($this->current[$dimNum]<$this->shape[$dimNum]) {
                        return;
                    }
                    $this->current[$dimNum] = 0;
                }
                $this->endOfItem = true;
            }
        
            public function rewind() : void
            {
                $this->current = array_fill(0,count($this->shape),0);
                $this->endOfItem = false;
            }
        
            public function valid() : bool
            {
                if($this->endOfItem)
                    return false;
        
                return true;
            }
        };
    }

    protected function walkAxis(
        callable $funcLinear,
        callable $funcAxis,
        NDArray $X,
        int $axis=null,
        int $dtype=null) : mixed
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
        $bufIterator = $this->createMatrixBufferIterator($shape,[$axis]);
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

    /**
     * @param array<int> $shape
     */
    protected function calcAxisStep(array $shape,int $axis) : int
    {
        $w = 1;
        for($i=count($shape)-1; $axis<$i; $i--) {
            $w *= $shape[$i];
        }
        return $w;
    }

    /**
     * @param array<int> $shape
     * @param array<int> $skipDims
     * @return array<int>
     */
    protected function projectionShape(array $shape,array $skipDims) : array
    {
        $newShape = [];
        foreach ($shape as $key => $value) {
            if(in_array($key,$skipDims))
                continue;
            array_push($newShape,$value);
        }
        return $newShape;
    }

    public function sum(NDArray $X,int $axis=null) : mixed
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
            $math = $this->getMath($X->dtype());
            $func = function($N,$XX,$offX,$bufPos,$incX) use ($math) {
                return $math->sum($N,$XX,$offX+$bufPos,$incX);
            };
        }
        return $this->walkAxis($func,$func,$X,$axis);
    }

    public function asum(NDArray $X,int $axis=null) : mixed
    {
        $blas = $this->getBlas($X->dtype());
        $func = function($N,$XX,$offX,$bufPos,$incX) use ($blas) {
            return $blas->asum($N,$XX,$offX+$bufPos,$incX);
        };
        return $this->walkAxis($func,$func,$X,$axis);
    }

    public function max(NDArray $X,int $axis=null) : mixed
    {
        $math = $this->getMath($X->dtype());
        $func = function($N,$XX,$offX,$bufPos,$incX) use ($math) {
            $pos = $math->imax($N,$XX,$offX+$bufPos,$incX);
            return $XX[$offX+$bufPos+$pos*$incX];
        };

        return $this->walkAxis($func,$func,$X,$axis);
    }

    public function argMax(NDArray $X,int $axis=null) : mixed
    {
        $math = $this->getMath($X->dtype());
        $func = function($N,$XX,$offX,$bufPos,$incX) use ($math) {
            $pos = $math->imax($N,$XX,$offX+$bufPos,$incX);
            return $pos;
        };

        return $this->walkAxis($func,$func,$X,$axis,$this->defaultIntType);
    }

    public function amax(NDArray $X,int $axis=null) : mixed
    {
        $blas = $this->getBlas($X->dtype());
        $func = function($N,$XX,$offX,$bufPos,$incX) use ($blas) {
            $pos = $blas->iamax($N,$XX,$offX+$bufPos,$incX);
            return $XX[$offX+$bufPos+$pos*$incX];
        };

        return $this->walkAxis($func,$func,$X,$axis);
    }

    public function argAmax(NDArray $X,int $axis=null) : mixed
    {
        $blas = $this->getBlas($X->dtype());
        $func = function($N,$XX,$offX,$bufPos,$incX) use ($blas) {
            $pos = $blas->iamax($N,$XX,$offX+$bufPos,$incX);
            return $pos;
        };

        return $this->walkAxis($func,$func,$X,$axis,$this->defaultIntType);
    }

    public function min(NDArray $X,int $axis=null) : mixed
    {
        $math = $this->getMath($X->dtype());
        $func = function($N,$XX,$offX,$bufPos,$incX) use ($math) {
            $pos = $math->imin($N,$XX,$offX+$bufPos,$incX);
            return $XX[$offX+$bufPos+$pos*$incX];
        };

        return $this->walkAxis($func,$func,$X,$axis);
    }

    public function argMin(NDArray $X,int $axis=null) : mixed
    {
        $math = $this->getMath($X->dtype());
        $func = function($N,$XX,$offX,$bufPos,$incX) use ($math) {
            $pos = $math->imin($N,$XX,$offX+$bufPos,$incX);
            return $pos;
        };

        return $this->walkAxis($func,$func,$X,$axis,$this->defaultIntType);
    }

    public function amin(NDArray $X,int $axis=null) : mixed
    {
        $blas = $this->getBlas($X->dtype());
        $func = function($N,$XX,$offX,$bufPos,$incX) use ($blas) {
            $pos = $blas->iamin($N,$XX,$offX+$bufPos,$incX);
            return $XX[$offX+$bufPos+$pos*$incX];
        };

        return $this->walkAxis($func,$func,$X,$axis);
    }

    public function argAmin(NDArray $X, int $axis=null) : mixed
    {
        $blas = $this->getBlas($X->dtype());
        $func = function($N,$XX,$offX,$bufPos,$incX) use ($blas) {
            $pos = $blas->iamin($N,$XX,$offX+$bufPos,$incX);
            return $pos;
        };

        return $this->walkAxis($func,$func,$X,$axis,$this->defaultIntType);
    }

    public function mean(NDArray $X, int $axis=null) : mixed
    {
        $math = $this->getMath($X->dtype());
        $func = function($N,$XX,$offX,$bufPos,$incX) use ($math) {
            $sum = $math->sum($N,$XX,$offX+$bufPos,$incX);
            return $sum/$N;
        };
        return $this->walkAxis($func,$func,$X,$axis);
    }


    public function f(callable $func,NDArray $X, mixed ...$args) : NDArray
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

    public function u(NDArray $X,callable $func, mixed ...$args) : NDArray
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

    public function op(mixed $X, string $operator, mixed $Y, NDArray $R=null) : NDArray
    {
        if(!array_key_exists($operator,$this->broadCastOperators)) {
            throw new InvalidArgumentException('Unknown operator: "'.$operator.'""');
        }
        if(!($X instanceof NDArray)&&!($Y instanceof NDArray)) {
            throw new InvalidArgumentException('Requires at least one matrix.');
        }
        if(($X instanceof NDArray)&&($Y instanceof NDArray)) {
            if($X->shape()==$Y->shape()) {
                $la = $this->getLa($X->dtype());
                if($X->dtype()==$Y->dtype() && $this->isFloatType($X->dtype())) {
                    if($operator=='+') {
                        if($R===null) {
                            $R = $this->alloc(null,$Y->dtype(),$Y->shape());
                        }
                        $la->copy($Y,$R);
                        return $la->axpy($X,$R);
                    } elseif ($operator=='-') {
                        if($R===null) {
                            $R = $this->alloc(null,$X->dtype(),$X->shape());
                        }
                        $la->copy($X,$R);
                        return $la->axpy($Y,$R,-1.0);
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
            if($X instanceof NDArray) {
                $la = $this->getLa($X->dtype());
            } else {
                $la = $this->getLa($Y->dtype());
            }
            if($operator=='*') {
                if($X instanceof NDArray && is_numeric($Y) && $this->isFloatType($X->dtype())) {
                    if($R===null) {
                        $R = $this->alloc(null,$X->dtype(),$X->shape());
                    }
                    $la->copy($X,$R);
                    return $la->scal($Y,$R);
                } elseif($Y instanceof NDArray && is_numeric($X) && $this->isFloatType($Y->dtype())) {
                    if($R===null) {
                        $R = $this->alloc(null,$Y->dtype(),$Y->shape());
                    }
                    $la->copy($Y,$R);
                    return $la->scal($X,$R);
                }
            } elseif($operator=='/') {
                if($X instanceof NDArray && is_numeric($Y) && $this->isFloatType($X->dtype())) {
                    if($Y==0.0) {
                        throw new RuntimeException('Zero divide error');
                    }
                    if($R===null) {
                        $R = $this->alloc(null,$X->dtype(),$X->shape());
                    }
                    $la->copy($X,$R);
                    return $la->scal(1/$Y,$R);
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
                if(is_float($X)) {
                    $dtype = $this->maximumPrecision($Y->dtype(),NDArray::float32);
                } else {
                    $dtype = $Y->dtype();
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
        } elseif($this->isIntType($maskDtype)) {
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

    public function update(NDArray $X, string $operator, mixed $value, NDArray ...$MASKs) : NDArray
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
        } elseif($this->isIntType($maskDtype)) {
            $this->selectByMatrix($X, $MASKs, $R=null, $func, $value);
        } else {
            throw new InvalidArgumentException('The mask matrix must be type of the bool or int.');
        }

        return $X;
    }

    /**
     * @param array<NDArray> $MASKs
     */
    protected function selectByMask(
        NDArray $X,
        array $MASKs,
        NDArray $R=null,
        string $func=null,
        mixed $updateValue=null) : void
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

    /**
     * @param array<NDArray> $MASKs
     */
    protected function selectByMatrix(
        NDArray $X,
        array $MASKs,
        NDArray $R=null,
        string $func=null,
        mixed $updateValue=null) : void
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
                $blas = $this->getBlas($X->dtype());
                $blas->copy($size,
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

    public function dtypeToString(int $dtype) : string
    {
        if(!isset($this->dtypeToString[$dtype])) {
            return 'Unknown';
        }
        return $this->dtypeToString[$dtype];
    }

    public function toString(
        NDArray $array,
        string $format=null,
        bool|int $indent=null) : string
    {
        $shape = $array->shape();
        if(count($shape)==0) {
            $value = $array->toArray();
            if($format) {
                return sprintf($format,$value);
            } else {
                return strval($value);
            }
        }
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
                    return '['.implode(',',array_map(function($x) use ($format,$array) {
                            if($array->dtype()==NDArray::complex64||$array->dtype()==NDArray::complex128) {
                                return sprintf($format,$x->real,$x->imag);
                            } else {
                                return sprintf($format,$x);
                            }
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

    public function random() : object
    {
        if($this->random==null) {
            $this->random = new Random($this,$this->defaultFloatType);
        }
        return $this->random;
    }

    public function la() : object
    {
        if($this->la!==null) {
            return $this->la;
        }
        $this->la = new LinearAlgebra(
            service:$this->service,
            defaultFloatType:$this->defaultFloatType);
        return $this->la;
    }

    protected function laPhpMode() : object
    {
        if($this->laPhp!==null) {
            return $this->laPhp;
        }
        $this->laPhp = new LinearAlgebra(
            service:$this->service,
            defaultFloatType:$this->defaultFloatType,
            serviceLevel:Service::LV_BASIC);
        return $this->laPhp;
    }

    // Interface for compatibility with old method names
    public function laRawMode() : object
    {
        return $this->la();
    }

    /**
     * @param array<string,mixed> $options
     */
    protected function createLinearAlgebraCL(array $options=null) : object
    {
        $queue = $this->service->createQueue($options);
        $la = new LinearAlgebraCL($queue,$this->service);
        return $la;
    }

    /**
     * @param array<string,mixed> $options
     */
    public function laAccelerated(string $name,array $options=null) : object
    {
        switch($name) {
            case 'clblast': {
                if($this->clblastLA==null) {
                    $this->clblastLA = $this->createLinearAlgebraCL($options);
                }
                return $this->clblastLA;
            }
            default: {
                throw new LogicException('Unknown accelerator');
            }
        }
    }

    public function service() : Service
    {
        return $this->service;
    }

    public function isAdvanced() : bool
    {
        return $this->service->serviceLevel()>=Service::LV_ADVANCED;
    }

    public function isAccelerated() : bool
    {
        return $this->service->serviceLevel()>=Service::LV_ACCELERATED;
    }

    public function info() : string
    {
        return $this->service->info();
    }

    /**
     * @param NDArray|array<mixed,mixed> $array
     */
    public function serializeArray(NDArray|array $array) : string
    {
        if($array instanceof NDArray) {
            return $array->serialize();
        }
        $list = [];
        foreach($array as $key => $value) {
            if(!is_array($value)&&!($value instanceof NDArray)) {
                throw new RuntimeException('invalid format');
            }
            $list[$key] = $this->serializeArray($value);
        }
        return static::SERIALIZE_NDARRAYSET_KEYWORD.serialize($list);
    }

    public function unserializeArray(string $data) : mixed
    {
        if(strpos($data,static::SERIALIZE_NDARRAYSET_KEYWORD)===0) {
            $data = substr($data,strlen(static::SERIALIZE_NDARRAYSET_KEYWORD));
            $array = unserialize($data);
            $list = [];
            if(!is_array($array)) {
                throw new RuntimeException('invalid format');
            }
            foreach($array as $key => $value) {
                if(!is_string($value)) {
                    throw new RuntimeException('invalid format');
                }
                $list[$key] = $this->unserializeArray($value);
            }
            return $list;
        } elseif(strpos($data,static::SERIALIZE_NDARRAY_KEYWORD)===0) {
            $array = new NDArrayPhp(service:$this->service);
            $array->unserialize($data);
            return $array;
        }
        $array = unserialize($data);
        return $array;
    }
}
