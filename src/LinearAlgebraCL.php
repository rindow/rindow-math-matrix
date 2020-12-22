<?php
namespace Rindow\Math\Matrix;

use Interop\Polite\Math\Matrix\BLAS;
use Interop\Polite\Math\Matrix\NDArray;
use Interop\Polite\Math\Matrix\OpenCL;
use Interop\Polite\Math\Matrix\LinearBuffer;
use Rindow\OpenCL\EventList;
#use Rindow\Math\Matrix\OpenBlasBuffer;
#use Rindow\Math\Matrix\NDArrayPhp;
use InvalidArgumentException;

class LinearAlgebraCL
{
    const LAPACK_ROW_MAJOR = 101;
    const LAPACK_COL_MAJOR = 102;

    protected $iaminwarning;
    protected $blas;
    protected $lapack;
    protected $math;
    protected $defaultFloatType = NDArray::float32;
    protected $blocking = false;
    protected $scalarNumeric = false;
    protected $autoEvents;

    public function __construct(
        $context,$queue,$blas,$openclmath,$clblastmath,
        $openblasmath=null,$lapack=null,$defaultFloatType=null)
    {
        $this->context = $context;
        $this->queue = $queue;
        $this->blas = $blas;
        $this->math = $clblastmath;
        $this->openclmath = $openclmath;
        $this->openblasmath = $openblasmath;
        $this->lapack = $lapack;
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

    public function getOpenCLMath()
    {
        return $this->openclmath;
    }

    public function getConfig()
    {
        return 'CLBlast';
    }

    public function getContext()
    {
        return $this->context;
    }

    public function getQueue()
    {
        return $this->queue;
    }

    public function blocking(bool $switch=null) : bool
    {
        if($switch===null) {
            return $this->blocking;
        }
        $this->blocking = $switch;
        return $this->blocking;
    }

    public function prepareAutoEvent(
        array $inputs, object $events, object $waitEvents, bool $explicit=null)
    {
        if($this->autoEvents) {
            $autoWaitEvents = $this->newEventList();
            foreach($inputs as $array) {
                $arrayEvents = $array->getEvents();
                if($arrayEvents) {
                    $autoWaitEvents->copy($arrayEvents);
                }
            }
            if($waitEvents) {
                $autoWaitEvents->copy($waitEvents);
            }
            if($explicit) {
                $autoWaitEvents->wait();
            }
            if($events==null) {
                $events = $this->newEventList();
            }
        }
        return [$events,$waitEvents];
    }

    public function applyAutoEvent($output,$events,$waitEvents)
    {
        if($this->autoEvents) {
            $output->setEvents($events);
        }
    }

    public function scalarNumeric(bool $switch=null) : bool
    {
        if($switch===null) {
            return $this->scalarNumeric;
        }
        $this->scalarNumeric = $switch;
        return $this->scalarNumeric;
    }

    public function fp64() : bool
    {
        return $this->openclmath->fp64();
    }

    public function accelerated() : bool
    {
        return true;
    }

    public function finish()
    {
        $this->queue->finish();
    }

    public function array($array, $flags=null)
    {
        if($array instanceof NDArray) {
            $buffer = $array->buffer();
            if($buffer instanceof LinearBuffer) {
                ;
            } elseif($buffer instanceof OpenCLBuffer) {
                return $array;
            } else {
                throw new InvalidArgumentException('Unsuppored buffer type.');
            }
        } elseif(is_array($array) || is_numeric($array)) {
            $dtype = $flags;
            $flags = null;
            $array = new NDArrayPhp($array,$dtype);
        } else {
            throw new InvalidArgumentException('input value must be NDArray or array');
        }
        if($flags==null) {
            $flags = OpenCL::CL_MEM_READ_WRITE;
        }
        $flags = $flags | OpenCL::CL_MEM_COPY_HOST_PTR;
        return new NDArrayCL(
            $this->context, $this->queue,
            $array->buffer(), $array->dtype(), $array->shape(),
            $array->offset(), $flags);
    }

    public function alloc(array $shape,$dtype=null,$flags=null)
    {
        if($dtype===null)
            $dtype = $this->defaultFloatType;
        return new NDArrayCL(
            $this->context, $this->queue, $buffer=null, $dtype, $shape,
            $offset=null, $flags);
    }

    public function zeros(
        NDArray $X,
        object $events=null, object $waitEvents=null) : NDArray
    {
        $pattern = $this->newHostBuffer(1,$X->dtype());
        $valueSize = $pattern->value_size();
        $N = $X->size();
        $XX = $X->buffer();
        $offX = $X->offset();

        $XX->fill($this->queue,$pattern,$N*$valueSize,$offX*$valueSize,
                    $pattern_size=1,$pattern_offset=0,$events,$waitEvents);
        if($this->blocking) {
            $this->finish();
        }
        return $X;
    }

    public function zerosLike(NDArray $array,$flags=null) : NDArray
    {
        $newArray = $this->alloc($array->shape(),$array->dtype(),$flags);
        $events = $this->newEventList();
        $this->zeros($newArray,$events);
        $events->wait();
        return $newArray;
    }

    public function fill(
        $value,
        NDArray $X,
        object $events=null, object $waitEvents=null) : NDArray
    {
        if(is_scalar($value)) {
            if(is_string($value)) {
                $value = ord($value);
            }
            $pattern = $this->allocHost([1],$X->dtype());
            $pattern[0] = $value;
        } elseif($value instanceof NDArray) {
            ;
        } else {
            throw new InvalidArgumentException('Invalid data type');
        }
        $buffer = $X->buffer();
        $buffer->fill(
            $this->getQueue(),
            $pattern->buffer(),
            $X->size()*$buffer->value_size(),
            0, // buffer offset
            1, // pattern size
            0, // pattern offset
            $events,$waitEvents
        );
        return $X;
    }

    public function allocHost(array $shape,$dtype=null)
    {
        if($dtype===null)
            $dtype = $this->defaultFloatType;
        return new NDArrayPhp(null,$dtype,$shape);
    }

    public function zerosHost(
        NDArray $X) : NDArray
    {
        $N = $X->size();
        $XX = $X->buffer();
        $offX = $X->offset();
        $this->openblasmath->zeros($N,$XX,$offX,1);
        return $X;
    }

    protected function newHostBuffer($size,$dtype)
    {
        return new OpenBlasBuffer($size,$dtype);
    }

    protected function newBuffer(
        int $size, int $flags=null,
        LinearBuffer $hostBuffer=null, int $hostOffset=null,
        int $dtype=null)
    {
        return new OpenCLBuffer($this->context,
            $size,$flags,$hostBuffer,$hostOffset,$dtype);
    }

    public function newEventList()
    {
        return new EventList();
    }

    public function astype(NDArray $X, $dtype, NDArray $Y=null,
        $events=null,$waitEvents=null) : NDArray
    {
        if($Y==null) {
            $Y = $this->alloc($X->shape(),$dtype);
        }
        $n = $X->size();
        $XX = $X->buffer();
        $offX = $X->offset();
        $YY = $Y->buffer();
        $offY = $Y->offset();

        $this->openclmath->astype(
            $n,
            $dtype,
            $XX,$offX,1,
            $YY,$offY,1,
            $events,$waitEvents
        );
        if($this->blocking) {
            $this->finish();
        }
        return $Y;
    }

    /**
    *    Y := X
    */
    public function copy(
        NDArray $X,
        NDArray $Y=null,
        object $events=null) : NDArray
    {
        if($Y===null) {
            $Y = $this->alloc($X->shape(),$X->dtype(),OpenCL::CL_MEM_READ_WRITE);
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
        $this->blas->copy($N,$XX,$offX,1,$YY,$offY,1,$this->queue,$events);
        if($this->blocking) {
            $this->finish();
        }
        return $Y;
    }

    /**
    *    X := alpha * X
    */
    public function scal(
        float $alpha,
        NDArray $X,
        object $events=null) : NDArray
    {
        $N = $X->size();
        $XX = $X->buffer();
        $offX = $X->offset();
        if($alpha===null) {
            $alpha = 1.0;
        }
        $this->blas->scal($N,$alpha,$XX,$offX,1,$this->queue,$events);
        if($this->blocking) {
            $this->finish();
        }
        return $X;
    }

    /**
    *    Y := alpha * X + Y
    */
    public function axpy(
        NDArray $X,
        NDArray $Y,
        float $alpha=null,
        object $events=null) : NDArray
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
        $this->blas->axpy($N,$alpha,$XX,$offX,1,$YY,$offY,1,$this->queue,$events);
        if($this->blocking) {
            $this->finish();
        }
        return $Y;
    }

    /**
    *    ret := X^t Y = x_1 * y_1 + ... + x_n * y_n
    */
    public function dot(
        NDArray $X,
        NDArray $Y,
        NDArray $R=null,
        object $events=null)
    {
        if($X->shape()!=$Y->shape()) {
            $shapeError = '('.implode(',',$X->shape()).'),('.implode(',',$Y->shape()).')';
            throw new InvalidArgumentException("Unmatch shape of dimension: ".$shapeError);
        }
        if($R==null) {
            $R = $this->alloc([],$X->dtype(),OpenCL::CL_MEM_READ_WRITE);
        }
        $N = $X->size();
        $RR = $R->buffer();
        $offR = $R->offset();
        $XX = $X->buffer();
        $offX = $X->offset();
        $YY = $Y->buffer();
        $offY = $Y->offset();
        $this->blas->dot($N,$RR,$offR,$XX,$offX,1,$YY,$offY,1,$this->queue,$events);
        if($this->blocking) {
            $this->finish();
        }
        if($this->scalarNumeric) {
            return $R->toArray();
        }
        return $R;
    }

    /**
    *    ret := |x_1| + ... + |x_n|
    */
    public function asum(
        NDArray $X,
        NDArray $R=null,
        object $events=null)
    {
        if($R==null) {
            $R = $this->alloc([],$X->dtype(),OpenCL::CL_MEM_READ_WRITE);
        }
        $N = $X->size();
        $RR = $R->buffer();
        $offR = $R->offset();
        $XX = $X->buffer();
        $offX = $X->offset();
        $this->blas->asum($N,$RR,$offR,$XX,$offX,1,$this->queue,$events);
        if($this->blocking) {
            $this->finish();
        }
        if($this->scalarNumeric) {
            return $R->toArray();
        }
        return $R;
    }

    /**
    *    ret := arg max |X(i)|
    */
    public function iamax(
        NDArray $X,
        NDArray $R=null,
        object $events=null)
    {
        if($R==null) {
            // *** CAUTION ****
            // Index result is 32bit
            $R = $this->alloc([],NDArray::int32,OpenCL::CL_MEM_READ_WRITE);
        }
        $N = $X->size();
        $RR = $R->buffer();
        $offR = $R->offset();
        $XX = $X->buffer();
        $offX = $X->offset();
        $this->blas->iamax($N,$RR,$offR,$XX,$offX,1,$this->queue,$events);
        if($this->blocking) {
            $this->finish();
        }
        if($this->scalarNumeric) {
            return $R->toArray();
        }
        return $R;
    }

    /**
    *    ret := arg min |X(i)|
    */
    public function iamin(
        NDArray $X,
        NDArray $R=null,
        object $events=null)
    {
        if($R==null) {
            // *** CAUTION ****
            // Index result is 32bit
            $R = $this->alloc([],NDArray::int32,OpenCL::CL_MEM_READ_WRITE);
        }
        $N = $X->size();
        $RR = $R->buffer();
        $offR = $R->offset();
        $XX = $X->buffer();
        $offX = $X->offset();
        $this->blas->iamin($N,$RR,$offR,$XX,$offX,1,$this->queue,$events);
        if($this->blocking) {
            $this->finish();
        }
        if($this->scalarNumeric) {
            return $R->toArray();
        }
        return $R;
    }

    /**
    *    ret := max |X(i)|
    */
    public function amax(
        NDArray $X,
        NDArray $R=null,
        object $events=null)
    {
        if($R==null) {
            $R = $this->alloc([],$X->dtype(),OpenCL::CL_MEM_READ_WRITE);
        }
        // *** CAUTION ****
        // Index result is 32bit
        $IR = $this->alloc([],NDArray::int32,OpenCL::CL_MEM_READ_WRITE);
        $N = $X->size();
        $IRR = $IR->buffer();
        $offIR = $IR->offset();
        $XX = $X->buffer();
        $offX = $X->offset();
        $imaxEvents = $this->newEventList();
        $i = $this->blas->iamax($N,$IRR,$offIR,$XX,$offX,1,$this->queue,$imaxEvents);

        $RR = $R->buffer();
        $offR = $R->offset();
        $this->openclmath->selectAxis1(
            1,$N,$XX,$offX,$N,$IRR,$offIR,1,$RR,$offR,1,$events,$imaxEvents);
        if($this->blocking) {
            $this->finish();
        }
        if($this->scalarNumeric) {
            return $R->toArray();
        }
        return $R;
    }

    /**
    *    ret := min |X(i)|
    */
    public function amin(
        NDArray $X,
        NDArray $R=null,
        object $events=null)
    {
        if($R==null) {
            $R = $this->alloc([],$X->dtype(),OpenCL::CL_MEM_READ_WRITE);
        }
        // *** CAUTION ****
        // Index result is 32bit
        $IR = $this->alloc([],NDArray::int32,OpenCL::CL_MEM_READ_WRITE);
        $N = $X->size();
        $IRR = $IR->buffer();
        $offIR = $IR->offset();
        $XX = $X->buffer();
        $offX = $X->offset();
        $imaxEvents = $this->newEventList();
        $i = $this->blas->iamin($N,$IRR,$offIR,$XX,$offX,1,$this->queue,$imaxEvents);

        $RR = $R->buffer();
        $offR = $R->offset();
        $this->openclmath->selectAxis1(
            1,$N,$XX,$offX,$N,$IRR,$offIR,1,$RR,$offR,1,$events,$imaxEvents);
        if($this->blocking) {
            $this->finish();
        }
        if($this->scalarNumeric) {
            return $R->toArray();
        }
        return $R;
    }

    /**
    *    ret := sqrt(sum(Xn ** 2))
    */
    public function nrm2(
        NDArray $X,
        NDArray $R=null,
        object $events=null)
    {
        if($R==null) {
            $R = $this->alloc([],$X->dtype(),OpenCL::CL_MEM_READ_WRITE);
        }
        $N = $X->size();
        $RR = $R->buffer();
        $offR = $R->offset();
        $XX = $X->buffer();
        $offX = $X->offset();
        $this->blas->nrm2($N,$RR,$offR,$XX,$offX,1,$this->queue,$events);
        if($this->blocking) {
            $this->finish();
        }
        if($this->scalarNumeric) {
            return $R->toArray();
        }
        return $R;
    }

    /**
    *    Y := X
    *    X := Y
    */
    public function swap(
        NDArray $X,
        NDArray $Y,
        object $events=null) : void
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
        $this->blas->swap($N,$XX,$offX,1,$YY,$offY,1,$this->queue,$events);
        if($this->blocking) {
            $this->finish();
        }
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
        bool $trans=null,
        object $events=null) : NDArray
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
            $Y = $this->alloc([$rows],$X->dtype(),OpenCL::CL_MEM_READ_WRITE);
            $beta = 0.0;
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
            $YY,$offY,1,
            $this->queue,$events);

        if($this->blocking) {
            $this->finish();
        }
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
        bool $transB=null,
        object $events=null) : NDArray
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
            $C = $this->alloc([$M,$N]);
            $beta = 0.0;
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
            $CC,$offC,$ldc,
            $this->queue,$events);

        if($this->blocking) {
            $this->finish();
        }
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
        float $beta=null,
        object $events=null
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
            $this->math->gemmStridedBatched(
                BLAS::RowMajor,$transA,$transB,
                $M,$N,$K,
                $alpha,
                $AA,$offA,$lda,$incA,
                $BB,$offB,$ldb,$incB,
                $beta,
                $CC,$offC,$ldc,$incC,
                $broadcastBase,
                $this->queue,$events
            );
            $offA += $incA*$broadcastBase;
            $offB += $incB*$broadcastBase;
            $offC += $incC*$broadcastBase;
        }
        if($this->blocking) {
            $this->finish();
        }
        return $C;
    }

    /**
    *    X := alpha * X + beta
    */
    public function increment(
        NDArray $X,
        float $beta=null,
        float $alpha=null,
        object $events=null,
        object $waitEvents=null
        ) : NDArray
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

        $this->openclmath->increment(
            $n,
            $alpha,
            $XX,$offX,1,
            $beta,
            $events,$waitEvents
        );

        if($this->blocking) {
            $this->finish();
        }
        return $X;
    }

    /**
    *    X := 1 / (a*X + b)
    */
    public function reciprocal(
        NDArray $X,
        float $beta=null,
        float $alpha=null,
        object $events=null,
        object $waitEvents=null
        ) : NDArray
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

        $this->openclmath->reciprocal(
            $n,
            $alpha,
            $XX,$offX,1,
            $beta,
            $events,$waitEvents
        );

        if($this->blocking) {
            $this->finish();
        }
        return $X;
    }

    /**
     *     X := X  (X > a)
     *     X := a  (X <= a)
     */
    public function maximum(
        float $alpha,
        NDArray $X,
        object $events=null,
        object $waitEvents=null
        ) : NDArray
    {
        $n = $X->size();
        $XX = $X->buffer();
        $offX = $X->offset();

        $this->openclmath->maximum(
            $n,
            $alpha,
            $XX,$offX,1,
            $events,$waitEvents
        );

        if($this->blocking) {
            $this->finish();
        }
        return $X;
    }

    /**
     *     X := X  (X < a)
     *     X := a  (X >= a)
     */
    public function minimum(
        float $alpha,
        NDArray $X,
        object $events=null,
        object $waitEvents=null
        ) : NDArray
    {
        $n = $X->size();
        $XX = $X->buffer();
        $offX = $X->offset();

        $this->openclmath->minimum(
            $n,
            $alpha,
            $XX,$offX,1,
            $events,$waitEvents
        );

        if($this->blocking) {
            $this->finish();
        }
        return $X;
    }

    /**
     *     X := 1  (X > a)
     *     X := 0  (X <= a)
     */
    public function greater(
        float $alpha,
        NDArray $X,
        object $events=null,
        object $waitEvents=null
        ) : NDArray
    {
        $n = $X->size();
        $XX = $X->buffer();
        $offX = $X->offset();

        $this->openclmath->greater(
            $n,
            $alpha,
            $XX,$offX,1,
            $events,$waitEvents
        );

        if($this->blocking) {
            $this->finish();
        }
        return $X;
    }

    /**
     *     X := 1  (X < a)
     *     X := 0  (X >= a)
     */
    public function less(
        float $alpha,
        NDArray $X,
        object $events=null,
        object $waitEvents=null
        ) : NDArray
    {
        $n = $X->size();
        $XX = $X->buffer();
        $offX = $X->offset();

        $this->openclmath->less(
            $n,
            $alpha,
            $XX,$offX,1,
            $events,$waitEvents
        );

        if($this->blocking) {
            $this->finish();
        }
        return $X;
    }

    /**
     *    A(m,n) := X(n) * A(m,n)
     */
     public function multiply(
        NDArray $X,
        NDArray $A,
        bool $trans=null,
        object $events=null,
        object $waitEvents=null
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

        $this->openclmath->multiply(
            $trans,
            $m,
            $n,
            $XX,$offX,1,
            $AA,$offA,$n,
            $events,$waitEvents
        );

        if($this->blocking) {
            $this->finish();
        }
        return $A;
    }

    /**
     *    A(m,n) := X(n) * A(m,n)
     */
    public function add(
        NDArray $X,
        NDArray $A,
        float $alpha=null,
        bool $trans=null,
        object $events=null,
        object $waitEvents=null
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

        $this->openclmath->add(
            $trans,
            $m,
            $n,
            $alpha,
            $XX,$offX,1,
            $AA,$offA,$n,
            $events,$waitEvents
        );

        if($this->blocking) {
            $this->finish();
        }
        return $A;
    }

    /**
     *     X := X ^ 2
     */
    public function square(
        NDArray $X,
        object $events=null,
        object $waitEvents=null
        ) : NDArray
    {
        $n = $X->size();
        $XX = $X->buffer();
        $offX = $X->offset();

        $this->openclmath->square(
            $n,
            $XX,$offX,1,
            $events,$waitEvents
        );

        if($this->blocking) {
            $this->finish();
        }
        return $X;
    }

    /**
     *     X := sqrt(X)
     */
    public function sqrt(
        NDArray $X,
        object $events=null,
        object $waitEvents=null
        ) : NDArray
    {
        $n = $X->size();
        $XX = $X->buffer();
        $offX = $X->offset();

        $this->openclmath->sqrt(
            $n,
            $XX,$offX,1,
            $events,$waitEvents
        );

        if($this->blocking) {
            $this->finish();
        }
        return $X;
    }

    /**
     *     X := 1 / (a * sqrt(X) + b)
     */
    public function rsqrt(
        NDArray $X,
        float $beta=null,
        float $alpha=null,
        object $events=null,
        object $waitEvents=null
        ) : NDArray
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

        $this->openclmath->rsqrt(
            $n,
            $alpha,
            $XX,$offX,1,
            $beta,
            $events,$waitEvents
        );

        if($this->blocking) {
            $this->finish();
        }
        return $X;
    }

    /**
     *     X := X ^ a
     */
    public function pow(
        NDArray $X,
        float $alpha,
        object $events=null,
        object $waitEvents=null
        ) : NDArray
    {
        $n = $X->size();
        $XX = $X->buffer();
        $offX = $X->offset();

        $this->openclmath->pow(
            $n,
            $alpha,
            $XX,$offX,1,
            $events,$waitEvents
        );

        if($this->blocking) {
            $this->finish();
        }
        return $X;
    }

    /**
     *     X(i) := e ^ X(i)
     */
    public function exp(
        NDArray $X,
        object $events=null,
        object $waitEvents=null
        ) : NDArray
    {
        $n = $X->size();
        $XX = $X->buffer();
        $offX = $X->offset();

        $this->openclmath->exp(
            $n,
            $XX,$offX,1,
            $events,$waitEvents
        );

        if($this->blocking) {
            $this->finish();
        }
        return $X;
    }

    /**
     *     X := log(X)
     */
    public function log(
        NDArray $X,
        object $events=null,
        object $waitEvents=null
        ) : NDArray
    {
        $n = $X->size();
        $XX = $X->buffer();
        $offX = $X->offset();

        $this->openclmath->log(
            $n,
            $XX,$offX,1,
            $events,$waitEvents
        );

        if($this->blocking) {
            $this->finish();
        }
        return $X;
    }

    /**
     *     X := tanh(X)
     */
    public function tanh(
        NDArray $X,
        object $events=null,
        object $waitEvents=null
        ) : NDArray
    {
        $n = $X->size();
        $XX = $X->buffer();
        $offX = $X->offset();

        $this->openclmath->tanh(
            $n,
            $XX,$offX,1,
            $events,$waitEvents
        );

        if($this->blocking) {
            $this->finish();
        }
        return $X;
    }

    /**
     *     Y(i) := 1 (X(i) = Y(i))
     *     Y(i) := 0 (X(i) = Y(i))
     */
    public function equal(
        NDArray $X,
        NDArray $Y,
        object $events=null,
        object $waitEvents=null
        ) : NDArray
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
        $this->openclmath->equal(
            $N,
            $XX,$offX,$incX,
            $YY,$offY,$incY,
            $events,$waitEvents
        );

        if($this->blocking) {
            $this->finish();
        }
        return $Y;
    }

    /**
     * A(m,n) := X(n)
     */
    public function duplicate(
        NDArray $X,
        int $repeats=null,
        bool $trans=null,
        NDArray $A=null,
        object $events=null,
        object $waitEvents=null
        ) : NDArray
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

        $this->openclmath->duplicate(
            $trans,
            $m,
            $n,
            $XX,$offX,1,
            $AA,$offA,$n,
            $events,$waitEvents
        );

        if($this->blocking) {
            $this->finish();
        }
        return $A;
    }

    /**
    *    ret := x_1 + ... + x_n
    */
    public function sum(
        NDArray $X,
        NDArray $R=null,
        object $events=null
        //object $waitEvents=null
        )
    {
        if($R==null) {
            $R = $this->alloc([],$X->dtype(),OpenCL::CL_MEM_READ_WRITE);
        }
        $N = $X->size();
        $RR = $R->buffer();
        $offR = $R->offset();
        $XX = $X->buffer();
        $offX = $X->offset();
        $dtype = $X->dtype();
        if($dtype==NDArray::float32 || $dtype==NDArray::float64) {
            $this->math->sum($N,$RR,$offR,$XX,$offX,1,$this->queue,$events);
        } else {
            $this->openclmath->sum($N,$RR,$offR,$XX,$offX,1,$events,null);
        }
        if($this->blocking) {
            $this->finish();
        }
        if($this->scalarNumeric) {
            return $R->toArray();
        }
        return $R;
    }

    /**
    *    ret := arg max X(i)
    */
    public function imax(
        NDArray $X,
        NDArray $R=null,
        object $events=null)
    {
        if($R==null) {
            // *** CAUTION ****
            // Index result is 32bit
            $R = $this->alloc([],NDArray::int32,OpenCL::CL_MEM_READ_WRITE);
        }
        $N = $X->size();
        $RR = $R->buffer();
        $offR = $R->offset();
        $XX = $X->buffer();
        $offX = $X->offset();
        $this->math->imax($N,$RR,$offR,$XX,$offX,1,$this->queue,$events);
        if($this->blocking) {
            $this->finish();
        }
        if($this->scalarNumeric) {
            return $R->toArray();
        }
        return $R;
    }

    /**
    *    ret := arg min X(i)
    */
    public function imin(
        NDArray $X,
        NDArray $R=null,
        object $events=null)
    {
        if($R==null) {
            // *** CAUTION ****
            // Index result is 32bit
            $R = $this->alloc([],NDArray::int32,OpenCL::CL_MEM_READ_WRITE);
        }
        $N = $X->size();
        $RR = $R->buffer();
        $offR = $R->offset();
        $XX = $X->buffer();
        $offX = $X->offset();
        $this->math->imin($N,$RR,$offR,$XX,$offX,1,$this->queue,$events);
        if($this->blocking) {
            $this->finish();
        }
        if($this->scalarNumeric) {
            return $R->toArray();
        }
        return $R;
    }

    /**
    *    ret := max X(i)
    */
    public function max(
        NDArray $X,
        NDArray $R=null,
        object $events=null)
    {
        if($R==null) {
            $R = $this->alloc([],$X->dtype(),OpenCL::CL_MEM_READ_WRITE);
        }
        // *** CAUTION ****
        // Index result is 32bit
        $IR = $this->alloc([],NDArray::int32,OpenCL::CL_MEM_READ_WRITE);
        $N = $X->size();
        $IRR = $IR->buffer();
        $offIR = $IR->offset();
        $XX = $X->buffer();
        $offX = $X->offset();
        $imaxEvents = $this->newEventList();
        $this->math->imax($N,$IRR,$offIR,$XX,$offX,1,$this->queue,$imaxEvents);

        $RR = $R->buffer();
        $offR = $R->offset();
        $this->openclmath->selectAxis1(
            1,$N,$XX,$offX,$N,$IRR,$offIR,1,$RR,$offR,1,$events,$imaxEvents);
        if($this->blocking) {
            $this->finish();
        }
        if($this->scalarNumeric) {
            return $R->toArray();
        }
        return $R;
    }


    /**
    *    ret := min X(i)
    */
    public function min(
        NDArray $X,
        NDArray $R=null,
        object $events=null)
    {
        if($R==null) {
            $R = $this->alloc([],$X->dtype(),OpenCL::CL_MEM_READ_WRITE);
        }
        // *** CAUTION ****
        // Index result is 32bit
        $IR = $this->alloc([],NDArray::int32,OpenCL::CL_MEM_READ_WRITE);
        $N = $X->size();
        $IRR = $IR->buffer();
        $offIR = $IR->offset();
        $XX = $X->buffer();
        $offX = $X->offset();
        $imaxEvents = $this->newEventList();
        $i = $this->math->imin($N,$IRR,$offIR,$XX,$offX,1,$this->queue,$imaxEvents);

        $RR = $R->buffer();
        $offR = $R->offset();
        $this->openclmath->selectAxis1(
            1,$N,$XX,$offX,$N,$IRR,$offIR,1,$RR,$offR,1,$events,$imaxEvents);
        if($this->blocking) {
            $this->finish();
        }
        if($this->scalarNumeric) {
            return $R->toArray();
        }
        return $R;
    }

    /**
     * Y := A[X]
     */
    public function select(
        NDArray $A,
        NDArray $X,
        int $axis=null,
        NDArray $Y=null,
        $events=null,$waitEvents=null
        ) : NDArray
    {
        if($axis===null) {
            $axis=0;
        }
        if($X->dtype()!=NDArray::int32 && $X->dtype()!=NDArray::uint32) {
            $waitPrev = $waitEvents;
            $waitEvents = $this->newEventList();
            $X = $this->astype($X,NDArray::int32,null,$waitEvents,$waitPrev);
        }
        if($axis==0) {
            return $this->selectAxis0($A,$X,$Y,$events,$waitEvents);
        } elseif($axis==1) {
            return $this->selectAxis1($A,$X,$Y,$events,$waitEvents);
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
        NDArray $Y=null,
        EventList $events=null,
        EventList $waitEvents=null
        ) : NDArray
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
            $Y = $this->alloc($shape,$A->dtype(),OpenCL::CL_MEM_READ_WRITE);
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

        $this->openclmath->selectAxis0(
            $m,
            $n,
            $countX,
            $AA,$offA,$ldA,
            $XX,$offX,1,
            $YY,$offY,$ldY,
            $events,$waitEvents
        );

        if($this->blocking) {
            $this->finish();
        }
        return $Y;
    }

    /**
     *  Y(i) := A(i,X[i])
     */
    protected function selectAxis1(
        NDArray $A,
        NDArray $X,
        NDArray $Y=null,
        EventList $events=null,
        EventList $waitEvents=null
        ) : NDArray
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
            $Y = $this->alloc([$m],$A->dtype(),OpenCL::CL_MEM_READ_WRITE);
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

        $this->openclmath->selectAxis1(
            $m,
            $n,
            $AA,$offA,$ldA,
            $XX,$offX,1,
            $YY,$offY,1,
            $events,$waitEvents
        );

        if($this->blocking) {
            $this->finish();
        }
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
        NDArray $A=null,
        $events=null,$waitEvents=null
        ) : NDArray
    {
        if($axis===null) {
            $axis=0;
        }
        if($X->dtype()!=NDArray::int32 && $X->dtype()!=NDArray::uint32) {
            $waitPrev = $waitEvents;
            $waitEvents = $this->newEventList();
            $X = $this->astype($X,NDArray::int32,null,$waitEvents,$waitPrev);
        }
        if($axis==0) {
            return $this->scatterAxis0(false,$X,$Y,$numClass,$A,$events,$waitEvents);
        } elseif($axis==1) {
            return $this->scatterAxis1(false,$X,$Y,$numClass,$A,$events,$waitEvents);
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
        int $axis=null,
        $events=null,$waitEvents=null
        ) : NDArray
    {
        if($axis===null) {
            $axis=0;
        }
        if($X->dtype()!=NDArray::int32 && $X->dtype()!=NDArray::uint32) {
            $waitPrev = $waitEvents;
            $waitEvents = $this->newEventList();
            $X = $this->astype($X,NDArray::int32,null,$waitEvents,$waitPrev);
        }
        if($axis==0) {
            return $this->scatterAxis0(true,$X,$Y,null,$A,$events,$waitEvents);
        } elseif($axis==1) {
            return $this->scatterAxis1(true,$X,$Y,null,$A,$events,$waitEvents);
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
        NDArray $A=null,
        $events=null,$waitEvents=null
        ) : NDArray
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
            $waitPrev = $waitEvents;
            $waitEvents = $this->newEventList();
            $this->zeros($A,$waitEvents,$waitPrev);
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

        $this->openclmath->scatterAxis0(
            $m,
            $n,
            $countX,
            $AA,$offA,$ldA,
            $XX,$offX,1,
            $YY,$offY,$ldY,
            $addMode,
            $events,$waitEvents
            );

        if($this->blocking) {
            $this->finish();
        }
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
        NDArray $A=null,
        $events=null,$waitEvents=null
        ) : NDArray
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
            $waitPrev = $waitEvents;
            $waitEvents = $this->newEventList();
            $this->zeros($A,$waitEvents,$waitPrev);
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

        $this->openclmath->scatterAxis1(
            $m,
            $n,
            $AA,$offA,$ldA,
            $XX,$offX,1,
            $YY,$offY,1,
            $addMode,
            $events,$waitEvents
            );

        if($this->blocking) {
            $this->finish();
        }
        return $A;
    }

    public function onehot(
        NDArray $X,
        int $numClass,
        float $a=null,
        NDArray $Y=null,
        $events=null,$waitEvents=null
        ) : NDArray
    {
        if($X->ndim()!=1) {
            throw new InvalidArgumentException('"X" must be 1D-NDArray.');
        }
        if($X->dtype()!=NDArray::int32 && $X->dtype()!=NDArray::uint32) {
            $waitPrev = $waitEvents;
            $waitEvents = $this->newEventList();
            $X = $this->astype($X,NDArray::int32,null,$waitEvents,$waitPrev);
        }
        $sizeX = $X->size();
        $addMode = true;
        if($Y===null) {
            $addMode = false;
            $Y = $this->alloc([$sizeX,$numClass]);
            $waitPrev = $waitEvents;
            $waitEvents = $this->newEventList();
            $this->zeros($Y,$waitEvents,$waitPrev);
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

        $this->openclmath->onehot(
            $m,
            $n,
            $a,
            $XX,$offX,1,
            $YY,$offY,$ldY,
            $addMode,
            $events,$waitEvents
        );

        if($this->blocking) {
            $this->finish();
        }
        return $Y;
    }

    /**
     *     X := softmax(X)
     */
    public function softmax(
        NDArray $X,
        $events=null,$waitEvents=null
        ) : NDArray
    {
        if($X->ndim()!=2) {
            throw new InvalidArgumentException('"X" must be 2-D dimension');
        }

        [$m,$n] = $X->shape();
        $XX = $X->buffer();
        $offX = $X->offset();
        $ldA = $n;
        $this->openclmath->softmax(
            $m,
            $n,
            $XX,$offX,$ldA,
            $events,$waitEvents
        );

        if($this->blocking) {
            $this->finish();
        }
        return $X;
    }

    /**
     *    X(m) := sum( A(m,n) )
     */
    public function reduceSumCLBlast(
        NDArray $A,
        int $axis=null,
        NDArray $X=null,
        $dtypeX=null,
        $events=null
        //,$waitEvents=null
        ) : NDArray
    {
        if($axis===null)
            $axis = 0;
        if($axis!==0 && $axis!==1 && $axis!==-1)
            throw new InvalidArgumentException('"axis" must be 0 or 1 or -1.');
        $shapeA = $A->shape();
        if($axis==0) {
            $trans = true;
            $m = array_shift($shapeA);
            $n = (int)array_product($shapeA);
            $rows = $n;
            $cols = $m;
        } else {
            $trans = false;
            $n = array_pop($shapeA);
            $m = (int)array_product($shapeA);
            $rows = $m;
            $cols = $n;
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

        $AA = $A->buffer();
        $offA = $A->offset();
        $XX = $X->buffer();
        $offX = $X->offset();

        //$this->openclmath->reduceSum(
        //    $trans,
        //    $m,
        //    $n,
        //    $AA,$offA,$n,
        //    $XX,$offX,1,
        //    $events,$waitEvents
        //);
        if($trans) {
            $incA = $n;
            $ldA  = 1;
        } else {
            $incA = 1;
            $ldA  = $n;
        }

        //echo "rows=$rows\n";
        //echo "cols=$cols\n";
        // /$segsize = 65536;
        $segsize = 8192;
        $batches = $rows % $segsize;
        if($batches) {
            $sumEvents = $this->newEventList();
            //echo "ph1-batches=$batches\n";
            for($i=0; $i<$batches; $i++,$offX++,$offA+=$ldA) {
                $this->math->sum($cols,$XX,$offX,$AA,$offA,$incA,$this->queue,$sumEvents);
            }
            $sumEvents->wait();
        }
        $batches = (int)floor($rows / $segsize);
        //echo "ph2-batches=$batches\n";
        for($j=0;$j<$batches;$j++) {
            $sumEvents = $this->newEventList();
            for($i=0; $i<$segsize; $i++,$offX++,$offA+=$ldA) {
                $this->math->sum($cols,$XX,$offX,$AA,$offA,$incA,$this->queue,$sumEvents);
            }
            $sumEvents->wait();
            fwrite(STDERR, "+");
        }

        if($this->blocking) {
            $this->finish();
        }
        return $X;
    }

    /**
     *    X(m) := sum( A(m,n) )
     */
    public function reduceSum(
        NDArray $A,
        int $axis=null,
        NDArray $X=null,
        $dtypeX=null,
        $events=null,$waitEvents=null
        ) : NDArray
    {
        if($axis===null)
            $axis = 0;
        if($axis!==0 && $axis!==1 && $axis!==-1)
            throw new InvalidArgumentException('"axis" must be 0 or 1 or -1.');
        $shapeA = $A->shape();
        if($axis==0) {
            $trans = true;
            $m = array_shift($shapeA);
            $n = (int)array_product($shapeA);
            $cols = $m;
            $rows = $n;
        } else {
            $trans = false;
            $n = array_pop($shapeA);
            $m = (int)array_product($shapeA);
            $cols = $n;
            $rows = $m;
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

        $AA = $A->buffer();
        $offA = $A->offset();
        $XX = $X->buffer();
        $offX = $X->offset();

        $this->openclmath->reduceSum(
            $trans,
            $m,
            $n,
            $AA,$offA,$n,
            $XX,$offX,1,
            $events,$waitEvents
        );

        if($this->blocking) {
            $this->finish();
        }
        return $X;
    }

    /**
     *    X(m) := max( A(m,n) )
     */
    public function reduceMax(
        NDArray $A,
        int $axis,
        NDArray $X=null,
        $dtypeX=null,
        $events=null,$waitEvents=null
        ) : NDArray
    {
        if($axis===null)
            $axis = 0;
        if($axis!==0 && $axis!==1 && $axis!==-1)
            throw new InvalidArgumentException('"axis" must be 0 or 1 or -1.');
        $shapeA = $A->shape();
        if($axis==0) {
            $trans = true;
            $m = array_shift($shapeA);
            $n = (int)array_product($shapeA);
            $rows = $n;
        } else {
            $trans = false;
            $n = array_pop($shapeA);
            $m = (int)array_product($shapeA);
            $rows = $m;
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

        $AA = $A->buffer();
        $offA = $A->offset();
        $XX = $X->buffer();
        $offX = $X->offset();

        $this->openclmath->reduceMax(
            $trans,
            $m,
            $n,
            $AA,$offA,$n,
            $XX,$offX,1,
            $events,$waitEvents
        );

        if($this->blocking) {
            $this->finish();
        }
        return $X;
    }

    /**
     *    X(m) := imax( A(m,n) )
     */
    public function reduceArgMax(
        NDArray $A,
        int $axis,
        NDArray $X=null,
        $dtypeX=null,
        $events=null,$waitEvents=null
        ) : NDArray
    {
        if($axis===null)
            $axis = 0;
        if($axis!==0 && $axis!==1 && $axis!==-1)
            throw new InvalidArgumentException('"axis" must be 0 or 1 or -1.');
        $shapeA = $A->shape();
        if($axis==0) {
            $trans = true;
            $m = array_shift($shapeA);
            $n = (int)array_product($shapeA);
            $rows = $n;
        } else {
            $trans = false;
            $n = array_pop($shapeA);
            $m = (int)array_product($shapeA);
            $rows = $m;
        }

        if($dtypeX==null) {
            $dtypeX = NDArray::int32;
        }
        if($X==null) {
            $X = $this->alloc([$rows],$dtypeX);
        } else {
            if($X->shape()!=[$rows]) {
                $shapeError = '('.implode(',',$A->shape()).'),('.implode(',',$X->shape()).')';
                throw new InvalidArgumentException("Unmatch shape of dimension: ".$shapeError);
            }
        }

        $AA = $A->buffer();
        $offA = $A->offset();
        $XX = $X->buffer();
        $offX = $X->offset();

        $this->openclmath->reduceArgMax(
            $trans,
            $m,
            $n,
            $AA,$offA,$n,
            $XX,$offX,1,
            $events,$waitEvents
        );

        if($this->blocking) {
            $this->finish();
        }
        return $X;
    }

    /**
     *    X(m) := sum( A(m,n)/n )
     */
    public function reduceMean(
        NDArray $A,
        int $axis,
        NDArray $X=null,
        $dtypeX=null,
        $events=null,$waitEvents=null
        ) : NDArray
    {
        $waitPrev = $waitEvents;
        $waitEvents = $this->newEventList();
        $X = $this->reduceSum(
            $A,$axis,$X,$dtypeX,
            $waitEvents,$waitPrev
        );
        $shapeA = $A->shape();
        if($axis==0) {
            $rows = $shapeA[0];
        } else {
            $rows = array_pop($shapeA);
        }
        $waitEvents->wait();
        $this->scal(
            1/$rows,$X,
            $events
        );
        if($this->blocking) {
            $this->finish();
        }
        return $X;
    }

    public function im2col2dclblast(
        bool $reverse,
        int $kernel_mode,
        NDArray $images,
        array $filterSize=null,
        array $strides=null,
        $padding=null,
        array $dilation_rate=null,
        //bool $channels_first=null,
        //bool $cols_channels_first=null,
        NDArray $cols=null,
        $events=null
        ) : NDArray
    {
        $ndim = $images->ndim();
        $images_offset = $images->offset();
        $images_size = $images->size();
        $images_buff = $images->buffer();
        if($ndim!=4) {
            throw new InvalidArgumentException('images must be 4D dimension');
        }
        //if($channels_first) {
            [$batches,
             $channels,
             $im_h,$im_w] =
                $images->shape();
        //} else {
        //    [$batches,
        //     $im_h,$im_w,
        //     $channels] =
        //        $images->shape();
        //}
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
        if($padding===null) {
            $padding = false;
        }
        if($dilation_rate==null) {
            $dilation_rate = [1,1];
        }
        [$dilation_h,$dilation_w] =
            $dilation_rate;
        //$channels_first = ($channels_first) ? true:false;
        //$cols_channels_first = ($cols_channels_first) ? true:false;
        $out_h = intval(floor(($im_h-($filter_h-1)*$dilation_h-1)/$stride_h)+1);
        $out_w = intval(floor(($im_w-($filter_w-1)*$dilation_w-1)/$stride_w)+1);
        if($out_h<=0 && $out_w<=0) {
            throw new InvalidArgumentException('Invalid shape or parameters.');
        }
        if($padding==null) {
            $padding_h = 0;
            $padding_w = 0;
        } elseif(is_bool($padding)) {
            if($padding) {
                $out_h = $im_h;
                $out_w = $im_w;
                $padding_h = (int)(($im_h-1)*$stride_h-$im_h+($filter_h-1)*$dilation_h+1);
                if($padding_h%2) {
                    $out_h++;
                }
                $padding_h = $padding_h ? (int)floor($padding_h/2) : 0;
                $padding_w = (int)(($im_w-1)*$stride_w-$im_w+($filter_w-1)*$dilation_w+1);
                if($padding_w%2) {
                    $out_w++;
                }
                $padding_w = $padding_w ? (int)floor($padding_w/2) : 0;
            }
        } elseif(is_array($padding)) {
            $padding_h = $padding[0];
            $padding_w = $padding[1];
            $out_h = $im_h + $padding_h;
            $out_w = $im_w + $padding_w;
        } else {
            throw new InvalidArgumentException('padding must be bool or array');
        }
        if($cols==null) {
            //if($cols_channels_first) {
                //echo "($batches,$channels,$filter_h,$filter_w,$out_h,$out_w)\n";
                $cols = $this->alloc([
                    $batches,$channels,$filter_h,$filter_w, // channels_first
                    $out_h,$out_w,                          // filters first
                ]);
                $this->zeros($cols);
            //} else {
            //    $cols = $this->alloc([
            //        $batches,$out_h,$out_w,
            //        $filter_h,$filter_w,$channels
            //    ]);
            //    $this->zeros($cols);
            //}
        }

        $col_buffer = $cols->buffer();
        $col_offset = $cols->offset();
        //$out_size = $cols->size();
        if(!$reverse) {
            $this->math->im2col(
                $kernel_mode,
                $channels,
                $im_h,
                $im_w,
                $filter_h,
                $filter_w,
                $padding_h,
                $padding_w,
                $stride_h,
                $stride_w,
                $dilation_h,
                $dilation_w,
                $images_buff,
                $images_offset,
                $col_buffer,
                $col_offset,
                $this->queue,
                //$batches,
                //$channels_first,
                //$cols_channels_first,
                $events
            );
            $result = $cols;
        }  else {
            $this->math->col2im(
                $kernel_mode,
                $channels,
                $im_h,
                $im_w,
                $filter_h,
                $filter_w,
                $padding_h,
                $padding_w,
                $stride_h,
                $stride_w,
                $dilation_h,
                $dilation_w,
                $col_buffer,
                $col_offset,
                $images_buff,
                $images_offset,
                $this->queue,
                //$batches,
                //$channels_first,
                //$cols_channels_first,
                $events
            );
            $result = $images;
        }
        if($this->blocking) {
            $this->finish();
        }
        return $result;
    }

    public function im2col(
        NDArray $images,
        array $filterSize=null,
        array $strides=null,
        bool $padding=null,
        bool $channels_first=null,
        array $dilation_rate=null,
        bool $cols_channels_first=null,
        NDArray $cols=null,
        $events=null,$waitEvents=null
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
                $cols,
                $events,$waitEvents
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
                $cols,
                $events,$waitEvents
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
                $cols,
                $events,$waitEvents
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
        bool $cols_channels_first=null,
        $events=null,$waitEvents=null
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
                $cols,
                $events,$waitEvents
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
                $cols,
                $events,$waitEvents
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
                $cols,
                $events,$waitEvents
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
        NDArray $cols=null,
        $events=null,$waitEvents=null
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
             $im_w] =
                $images->shape();
        } else {
            [$batches,
             $im_w,
             $channels] =
                $images->shape();
        }
        if($filterSize==null) {
            $filterSize = [3];
        }
        [$kernel_w] =
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
                $out_w = $im_w;
            } else {
                $out_w = intval(floor(($im_w-($kernel_w-1)*$dilation_w-1)/$stride_w)+1);
            }
            if($out_w<=0) {
                throw new InvalidArgumentException('Invalid shape or paramaters.');
            }
            if($cols_channels_first) {
                $cols = $this->alloc([
                    $batches,$out_w,
                    $channels,$kernel_w
                ]);
                $waitPrev = $waitEvents;
                $waitEvents = $this->newEventList();
                $this->zeros($cols,$waitEvents,$waitPrev);
            } else {
                $cols = $this->alloc([
                    $batches,$out_w,
                    $kernel_w,$channels
                ]);
                $waitPrev = $waitEvents;
                $waitEvents = $this->newEventList();
                $this->zeros($cols,$waitEvents,$waitPrev);
            }
        }
        $cols_buffer = $cols->buffer();
        $cols_offset = $cols->offset();
        $cols_size = $cols->size();
        $this->openclmath->im2col1d(
            $reverse,
            $images_buff,
            $images_offset,
            $images_size,
            $batches,
            $im_w,
            $channels,
            $kernel_w,
            $stride_w,
            $padding,
            $channels_first,
            $dilation_w,
            $cols_channels_first,
            $cols_buffer,
            $cols_offset,
            $cols_size,
            $events,$waitEvents
        );
        if($this->blocking) {
            $this->finish();
        }
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
        NDArray $cols=null,
        $events=null,$waitEvents=null
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
             $im_h,$im_w] =
                $images->shape();
        } else {
            [$batches,
             $im_h,$im_w,
             $channels] =
                $images->shape();
        }
        if($filterSize==null) {
            $filterSize = [3,3];
        }
        [$kernel_h,$kernel_w] =
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
                $out_h = $im_h;
                $out_w = $im_w;
            } else {
                $out_h = intval(floor(($im_h-($kernel_h-1)*$dilation_h-1)/$stride_h)+1);
                $out_w = intval(floor(($im_w-($kernel_w-1)*$dilation_w-1)/$stride_w)+1);
            }
            if($out_h<=0 && $out_w<=0) {
                throw new InvalidArgumentException('Invalid shape or parameters.');
            }
            if($cols_channels_first) {
                $cols = $this->alloc([
                    $batches,$out_h,$out_w,
                    $channels,$kernel_h,$kernel_w
                ]);
                $waitPrev = $waitEvents;
                $waitEvents = $this->newEventList();
                $this->zeros($cols,$waitEvents,$waitPrev);
            } else {
                $cols = $this->alloc([
                    $batches,$out_h,$out_w,
                    $kernel_h,$kernel_w,$channels
                ]);
                $waitPrev = $waitEvents;
                $waitEvents = $this->newEventList();
                $this->zeros($cols,$waitEvents,$waitPrev);
            }
        }
        $cols_buffer = $cols->buffer();
        $cols_offset = $cols->offset();
        $cols_size = $cols->size();
        $this->openclmath->im2col2d(
            $reverse,
            $images_buff,
            $images_offset,
            $images_size,
            $batches,
            $im_h,
            $im_w,
            $channels,
            $kernel_h,
            $kernel_w,
            $stride_h,
            $stride_w,
            $padding,
            $channels_first,
            $dilation_h,
            $dilation_w,
            $cols_channels_first,
            $cols_buffer,
            $cols_offset,
            $cols_size,
            $events,$waitEvents
        );
        if($this->blocking) {
            $this->finish();
        }
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
        NDArray $cols=null,
        $events=null,$waitEvents=null
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
             $im_d,$im_h,$im_w] =
                $images->shape();
        } else {
            [$batches,
             $im_d,$im_h,$im_w,
             $channels] =
                $images->shape();
        }
        if($filterSize==null) {
            $filterSize = [3,3,3];
        }
        [$kernel_d,$kernel_h,$kernel_w] =
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
        $cols_channels_first = ($cols_channels_first) ? true:false;
        if($cols==null) {
            if($padding) {
                $out_d = $im_d;
                $out_h = $im_h;
                $out_w = $im_w;
            } else {
                $out_d = intval(floor(($im_d-($kernel_d-1)*$dilation_d-1)/$stride_d)+1);
                $out_h = intval(floor(($im_h-($kernel_h-1)*$dilation_h-1)/$stride_h)+1);
                $out_w = intval(floor(($im_w-($kernel_w-1)*$dilation_w-1)/$stride_w)+1);
            }
            if($out_d<=0 || $out_h<=0 || $out_w<=0) {
                throw new InvalidArgumentException('Invalid shape or paramaters.');
            }
            if($cols_channels_first) {
                $cols = $this->alloc([
                    $batches,$out_d,$out_h,$out_w,
                    $channels,$kernel_d,$kernel_h,$kernel_w
                ]);
                $waitPrev = $waitEvents;
                $waitEvents = $this->newEventList();
                $this->zeros($cols,$waitEvents,$waitPrev);
            } else {
                $cols = $this->alloc([
                    $batches,$out_d,$out_h,$out_w,
                    $kernel_d,$kernel_h,$kernel_w,$channels
                ]);
                $waitPrev = $waitEvents;
                $waitEvents = $this->newEventList();
                $this->zeros($cols,$waitEvents,$waitPrev);
            }
        }
        $cols_buffer = $cols->buffer();
        $cols_offset = $cols->offset();
        $cols_size = $cols->size();
        $this->openclmath->im2col3d(
            $reverse,
            $images_buff,
            $images_offset,
            $images_size,
            $batches,
            $im_d,
            $im_h,
            $im_w,
            $channels,
            $kernel_d,
            $kernel_h,
            $kernel_w,
            $stride_d,
            $stride_h,
            $stride_w,
            $padding,
            $channels_first,
            $dilation_d,
            $dilation_h,
            $dilation_w,
            $cols_channels_first,
            $cols_buffer,
            $cols_offset,
            $cols_size,
            $events,$waitEvents
        );
        if($this->blocking) {
            $this->finish();
        }
        return $cols;
    }

    /**
    *  random uniform
    */
    public function randomUniform(
        array $shape,
        $low,
        $high,
        $dtype=null,
        int $seed=null,
        NDArray $X=null
        ) : NDArray
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
        $hostX = $this->allocHost($X->shape(),$X->dtype());
        if($seed===null) {
            $seed = random_int(~PHP_INT_MAX,PHP_INT_MAX);
        }

        $n = $hostX->size();
        $XX = $hostX->buffer();
        $offX = $hostX->offset();

        $this->openblasmath->randomUniform(
            $n,
            $XX,$offX,1,
            $low,
            $high,
            $seed
        );
        $valueSize = $X->buffer()->value_size();
        $X->buffer()->write(
            $this->queue,
            $XX,
            $X->size()*$valueSize,
            $X->offset()*$valueSize,
            $offX,
            $blocking_write=true
        );

        if($this->blocking) {
            $this->finish();
        }
        return $X;
    }

    public function randomNormal(
        array $shape,
        $mean,
        $scale,
        $dtype=null,
        int $seed=null,
        NDArray $X=null
        ) : NDArray
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
        $hostX = $this->allocHost($shape,$dtype);

        $n = $hostX->size();
        $XX = $hostX->buffer();
        $offX = $hostX->offset();

        $this->openblasmath->randomNormal(
            $n,
            $XX,$offX,1,
            $mean,
            $scale,
            $seed);

        $valueSize = $X->buffer()->value_size();
        $X->buffer()->write(
            $this->queue,
            $XX,
            $X->size()*$valueSize,
            $X->offset()*$valueSize,
            $offX,
            $blocking_write=true
        );

        if($this->blocking) {
            $this->finish();
        }
        return $X;
    }

    public function randomSequence(
        int $base,
        int $size=null,
        int $seed=null,
        $events=null,$waitEvents=null
        ) : NDArray
    {
        if($size==null) {
            $size = $base;
        }
        $hostX = $this->allocHost([$base],NDArray::int64);
        if($seed===null) {
            $seed = random_int(~PHP_INT_MAX,PHP_INT_MAX);
        }

        $n = $base;
        $XX = $hostX->buffer();
        $offX = $hostX->offset();

        $this->openblasmath->randomSequence(
            $n,
            $size,
            $XX,$offX,1,
            $seed);
        $hostX = $hostX[[0,$size-1]];
        $X = $this->array($hostX);
        $X = $this->astype(
            $X,NDArray::int32,
            $events,$waitEvents
        );
        if($this->blocking) {
            $this->finish();
        }
        return $X;
    }

    /**
    *  Y = X(n)
    */
    public function slice(
        NDArray $input,
        array $begin,
        array $size,
        NDArray $output=null,
        $events=null,$waitEvents=null
        ) : NDArray
    {
        return $this->doSlice(
            false,
            $input,
            $begin,
            $size,
            $output,
            $events,$waitEvents
        );
    }

    public function stick(
        NDArray $input,
        NDArray $output,
        array $begin,
        array $size,
        $events=null,$waitEvents=null
        ) : NDArray
    {
        return $this->doSlice(
            true,
            $output,
            $begin,
            $size,
            $input,
            $events,$waitEvents
        );
    }

    public function stack(
        array $values,
        int $axis=null,
        $events=null,$waitEvents=null
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
                    $value,
                    $events,$waitEvents
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
                    $value,
                    $events,$waitEvents
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
        int $axis=null,
        $events=null,$waitEvents=null
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
                $value,
                $events,$waitEvents
            );
            $i += $nn;
        }
        $output = $output->reshape(array_merge($shapePrefix,[$n],$dims));
        return $output;
    }

    public function split(
        NDArray $input, array $sizeSplits, $axis=null,
        $events=null,$waitEvents=null
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
                [0,$i],[-1,$size],
                $events,$waitEvents
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
        NDArray $output=null,
        $events=null,$waitEvents=null
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
            if($sizeAxis2<1||$startAxis2+$sizeAxis2>$n){
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
                throw new InvalidArgumentException('Unmatch output shape');
            }
        }

        $A = $input->buffer();
        $offsetA = $input->offset();
        $Y = $output->buffer();
        $offsetY = $output->offset();
        $incA = 1;
        $incY = 1;
        $this->openclmath->slice(
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
            $startAxis2,$sizeAxis2,
            $events,$waitEvents
        );
        if($this->blocking) {
            $this->finish();
        }
        return $output;
    }


    /*
    * repeat
    */
    public function repeat(
        NDArray $A, int $repeats,
        $events=null,$waitEvents=null
        )
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
        $k = 1;
        $size = (int)array_product($shapeCell);
        $AA = $A->buffer();
        $offA = $A->offset();
        $BB = $B->buffer();
        $offB = $B->offset();
        $startAxis0 = 0;
        $sizeAxis0 = $m;
        $startAxis2 = 0;
        $sizeAxis2 = 1;
        for($i=0;$i<$repeats;$i++) {
            $startAxis1 = $i;
            $sizeAxis1 = 1;
            $this->openclmath->slice(
                $reverse=true,
                $addMode=false,
                $m,
                $n,
                $k,
                $size,
                $BB,$offB,1,
                $AA,$offA,1,
                $startAxis0,$sizeAxis0,
                $startAxis1,$sizeAxis1,
                $startAxis2,$sizeAxis2,
                $events,$waitEvents
            );
        }
        if($this->blocking) {
            $this->finish();
        }
        return $B;
    }

    /**
    * reduceSumRepeated
    */
    public function reduceSumRepeated(
        NDArray $A,
        $events=null,$waitEvents=null
        )
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
        $k = 1;
        $size = (int)array_product($shapeCell);
        $AA = $A->buffer();
        $offA = $A->offset();
        $BB = $B->buffer();
        $offB = $B->offset();
        $startAxis0 = 0;
        $sizeAxis0 = $m;
        $startAxis2 = 0;
        $sizeAxis2 = 1;
        for($i=0;$i<$repeats;$i++) {
            if($i<$repeats-1) {
                $waitSlice = $this->newEventList();
            } else {
                $waitSlice = $events;
            }
            $startAxis1 = $i;
            $sizeAxis1 = 1;
            $this->openclmath->slice(
                $reverse=false,
                $addMode=true,
                $m,
                $n,
                $k,
                $size,
                $AA,$offA,1,
                $BB,$offB,1,
                $startAxis0,$sizeAxis0,
                $startAxis1,$sizeAxis1,
                $startAxis2,$sizeAxis2,
                $waitSlice,$waitEvents
            );
            $waitEvents = $waitSlice;
        }
        if($this->blocking) {
            $this->finish();
        }
        return $B;
    }

    public function transpose(
        NDArray $A,
        NDArray $B=null,
        float $alpha=null,
        object $events=null
        )
    {
        if($A->ndim()!=2) {
            throw new InvalidArgumentException('input array must be 2D.');
        }
        $shape = $A->shape();
        $shape = [$shape[1],$shape[0]];
        if($B==null) {
            $B = $this->alloc($shape,$A->dtype());
        }/* else {
            if($B->shape()!=$shape) {
                throw new InvalidArgumentException('output shape must be transpose matrix of input.');
            }
            if($B->dtype()!=$A->dtype()) {
                throw new InvalidArgumentException('output data type must be same with matrix of input.');
            }
        }*/
        if($alpha===null) {
            $alpha = 1.0;
        }
        $m = $shape[1];
        $n = $shape[0];
        $AA = $A->buffer();
        $offA = $A->offset();
        $BB = $B->buffer();
        $offB = $B->offset();
        $this->math->omatcopy(
            BLAS::RowMajor,
            BLAS::Trans,
            $m,
            $n,
            $alpha,
            $AA,$offA,$n,
            $BB,$offB,$m,
            $this->queue,$events
        );
        if($this->blocking) {
            $this->finish();
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

        $S = $this->allocHost([min($m,$n)],$matrix->dtype());
        $this->zerosHost($S);
        $U = $this->allocHost([$m,$ldU],$matrix->dtype());
        $this->zerosHost($U);
        $VT = $this->allocHost([$ldVT,$n],$matrix->dtype());
        $this->zerosHost($VT);
        $SuperB = $this->allocHost([min($m,$n)-1],$matrix->dtype());
        $this->zerosHost($SuperB);

        $hostMatrix = $matrix->toNDArray();

        $AA = $hostMatrix->buffer();
        $offsetA = $hostMatrix->offset();
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
        $U = $this->array($U);
        $S = $this->array($S);
        $VT = $this->array($VT);
        if(!$fullMatrices) {
            // bug in the lapacke ???
            $copyEvents = $this->newEventList();
            $VT = $this->copy($VT[[0,min($m,$n)-1]],null,$copyEvents);
            $copyEvents->wait();
        }
        if($this->blocking) {
            $this->finish();
        }
        return [$U,$S,$VT];
    }

}
