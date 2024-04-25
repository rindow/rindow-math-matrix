<?php
namespace Rindow\Math\Matrix;

use Interop\Polite\Math\Matrix\BLAS;
use Interop\Polite\Math\Matrix\NDArray;
use Interop\Polite\Math\Matrix\OpenCL;
use Interop\Polite\Math\Matrix\LinearBuffer;
use Interop\Polite\Math\Matrix\DeviceBuffer;
use Rindow\OpenCL\EventList;
use Rindow\Math\Matrix\Drivers\Service;
#use Rindow\Math\Matrix\OpenBlasBuffer;
#use Rindow\Math\Matrix\NDArrayPhp;
use InvalidArgumentException;
use LogicException;
use function Rindow\Math\Matrix\R;

class LinearAlgebraCL
{
    use ComplexUtils;

    const LAPACK_ROW_MAJOR = 101;
    const LAPACK_COL_MAJOR = 102;

    protected Service $service;
    protected object $context;
    protected object $queue;
    protected object $blas;
    protected object $lapack;
    protected object $math;
    protected object $openclmath;
    protected object $openblasmath;
    protected int $defaultFloatType = NDArray::float32;
    protected bool $blocking = false;
    protected bool $scalarNumeric = false;
    protected bool $autoEvents;
    protected bool $profiling = false;
    /** @var array<string,float> $profilingStartTime */
    protected $profilingStartTime = [];
    /** @var array<string,int> $profilingCount */
    protected $profilingCount = [];
    /** @var array<string,float> $profilingTotalTime */
    protected array $profilingTotalTime = [];
    protected string $clVersion;
    protected bool $isOpenCL110;

    /** @var array<int,bool> $intTypes */
    protected $intTypes= [
        NDArray::int8   => true,
        NDArray::int16  => true,
        NDArray::int32  => true,
        NDArray::int64  => true,
        NDArray::uint8  => true,
        NDArray::uint16 => true,
        NDArray::uint32 => true,
        NDArray::uint64 => true,
    ];

    public function __construct(
        object $queue, Service $service, int $defaultFloatType=null)
    {
        $this->service = $service;
        $this->queue = $queue;
        $this->context = $queue->getContext();
        $this->blas = $service->blasCL($queue);
        $this->lapack = $service->lapack(Service::LV_ADVANCED);
        $this->math = $service->mathCLBlast($queue);
        $this->openclmath = $service->mathCL($queue);
        $this->openblasmath = $service->math(Service::LV_ADVANCED);
        if($defaultFloatType!==null) {
            $this->defaultFloatType = $defaultFloatType;
        }
        $this->clVersion = $this->context->getInfo(OpenCL::CL_CONTEXT_DEVICES)->getInfo(0,OpenCL::CL_DEVICE_VERSION);
        //                                                    1234567890
        $this->isOpenCL110 = substr($this->clVersion,0,10)==='OpenCL 1.1';
    }

    public function service() : Service
    {
        return $this->service;
    }

    public function getBlas() : object
    {
        return $this->blas;
    }

    public function getMath() : object
    {
        return $this->math;
    }

    public function getOpenCLMath() : object
    {
        return $this->openclmath;
    }

    public function getConfig() : string
    {
        return 'CLBlast';
    }

    public function getContext() : object
    {
        return $this->context;
    }

    public function getQueue() : object
    {
        return $this->queue;
    }

    public function getCLVersion() : string
    {
        return $this->clVersion;
    }

    public function isOpenCL110() : bool
    {
        return $this->isOpenCL110;
    }

    public function setOpenCLTestMode(int $testMode) : void
    {
        $this->openclmath->setTestMode($testMode);
    }

    public function blocking(bool $switch=null) : bool
    {
        if($switch===null) {
            return $this->blocking;
        }
        $this->blocking = $switch;
        return $this->blocking;
    }

    public function setProfiling(bool $profiling) : void
    {
        $this->profiling = $profiling;
    }

    protected function profilingStart(string $name) : void
    {
        if(isset($this->profilingCount[$name])) {
            $this->profilingCount[$name]++;
        } else {
            $this->profilingCount[$name] = 1;
            $this->profilingTotalTime[$name] = 0;
        }
        $this->profilingStartTime[$name] = microtime(true);
    }

    protected function profilingEnd(string $name) : void
    {
        $this->profilingTotalTime[$name] =
            microtime(true) - $this->profilingStartTime[$name];
    }

    public function profilingReport() : void
    {
        asort($this->profilingTotalTime);
        foreach($this->profilingTotalTime as $name => $time) {
            $count = $this->profilingCount[$name];
            echo sprintf("%17s:total %6e, count:%6d, average %6e\n",$name,$time,$count,$time/$count);
        }
    }

    /**
     * @param array<NDArray> $inputs
     * @return array<object|null>
     */
    public function prepareAutoEvent(
        array $inputs, object $events=null, object $waitEvents=null, bool $explicit=null) : array
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

    public function applyAutoEvent(object $output,object $events,object $waitEvents) : void
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

    /**
     * @return array<string>
     */
    public function deviceTypes() : array
    {
        return $this->openclmath->deviceTypes();
    }

    public function finish() : void
    {
        $this->queue->finish();
    }

    protected function printableShapes(mixed $values) : string
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

    protected function isComplex(int $dtype) : bool
    {
        return $this->cistype($dtype);
    }

    /**
     * @return array<bool>
     */
    protected function complementTrans(?bool $trans,?bool $conj,int $dtype) : array
    {
        $trans = $trans ?? false;
        if($this->isComplex($dtype)) {
            $conj = $conj ?? $trans;
        } else {
            $conj = $conj ?? false;
        }
        return [$trans,$conj];
    }

    protected function transToCode(bool $trans,bool $conj) : int
    {
        if($trans) {
            return $conj ? BLAS::ConjTrans : BLAS::Trans;
        } else {
            return $conj ? BLAS::ConjNoTrans : BLAS::NoTrans;
        }
    }

    protected function buildValByType(float|int $value, int $dtype) : float|int|object
    {
        if($this->cistype($dtype)) {
            $value = $this->cbuild($value);
        }
        return $value;
    }

    public function isInt(NDArray $value) : bool
    {
        return array_key_exists($value->dtype(),$this->intTypes);
    }

    public function isFloat(NDArray $value) : bool
    {
        $dtype = $value->dtype();
        return $dtype==NDarray::float32||$dtype==NDarray::float64;
    }

    public function array(mixed $array, int $dtype=null, int $flags=null) : NDArray
    {
        if($this->profiling) {
            $this->profilingStart("array");
        }
        if($array instanceof NDArray) {
            $buffer = $array->buffer();
            if($buffer instanceof LinearBuffer) {
                ;
            } elseif($buffer instanceof DeviceBuffer) {
                return $array;
            } else {
                throw new InvalidArgumentException('Unsuppored buffer type.');
            }
        } elseif(is_array($array) || is_numeric($array) || is_bool($array) || $this->cisobject($array)) {
            $array = new NDArrayPhp($array,dtype:$dtype,service:$this->service);
        } else {
            throw new InvalidArgumentException('input value must be NDArray or array');
        }
        if($flags==null) {
            $flags = OpenCL::CL_MEM_READ_WRITE;
        }
        $flags = $flags | OpenCL::CL_MEM_COPY_HOST_PTR;
        $arrayCL = new NDArrayCL(
            $this->queue,
            $array->buffer(), dtype:$array->dtype(), shape:$array->shape(),
            offset:$array->offset(), flags:$flags, service:$this->service);
        if($this->profiling) {
            $this->profilingEnd("array");
        }
        return $arrayCL;
    }

    public function toNDArray(NDArray $ndarray) : NDArray
    {
        if($ndarray instanceof NDArrayCL) {
            return $ndarray->toNDArray();
        }
        return $ndarray;
    }

    public function scalar(mixed $array) : mixed
    {
        if($array instanceof NDArray) {
            return $array->toArray();
        }
        return $array;
    }

    public function expandDims(NDArray $x, int $axis) : NDArray
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

    public function squeeze(NDArray $x, int $axis=null) : NDArray
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

    /**
     * @param array<int> $shape
     */
    public function alloc(array $shape,int $dtype=null,int $flags=null) : NDArray
    {
        if($this->profiling) {
            $this->profilingStart("alloc");
        }
        if($dtype===null)
            $dtype = $this->defaultFloatType;
        $arrayCL = new NDArrayCL(
            $this->queue, dtype:$dtype, shape:$shape,
            flags:$flags, service:$this->service);

        if($this->profiling) {
            $this->profilingEnd("alloc");
        }
        return $arrayCL;
    }

    public function zeros(
        NDArray $X,
        object $events=null, object $waitEvents=null) : NDArray
    {
        $value = $this->buildValByType(0.0,$X->dtype());
        $this->fill($value,$X,$events,$waitEvents);
        return $X;
    }

    public function ones(
        NDArray $X,
        object $events=null, object $waitEvents=null) : NDArray
    {
        $value = $this->buildValByType(1.0,$X->dtype());
        $this->fill($value,$X,$events,$waitEvents);
        return $X;
    }

    public function zerosLike(NDArray $array,int $flags=null) : NDArray
    {
        $newArray = $this->alloc($array->shape(),dtype:$array->dtype(),flags:$flags);
        $events = $this->newEventList();
        $this->zeros($newArray,events:$events);
        $events->wait();
        return $newArray;
    }

    public function fill(
        mixed $value,
        NDArray $X,
        object $events=null, object $waitEvents=null) : NDArray
    {
        if($this->profiling) {
            $this->profilingStart("fill");
        }
        if($this->isOpenCL110) {
            if(is_scalar($value)) {
                if(is_string($value)) {
                    $value = ord($value);
                }
            } elseif($value instanceof NDArray) {
                $pbuf = $value->buffer();
                if($value->size()!=1) {
                    throw new InvalidArgumentException('Value must be scalar');
                }
                $value = $pbuf[$value->offset()];
            } else {
                throw new InvalidArgumentException('Invalid value type');
            }
            $buffer = $X->buffer();
            $n = $X->size();
            $offsetX = $X->offset();
            $this->openclmath->fill(
                $n,$buffer,$offsetX,1,$value,
                $events,$waitEvents
            );
        } else {
            if(is_scalar($value)||$this->cisobject($value)) {
                if(is_string($value)) {
                    $value = ord($value);
                }
                $pattern = $this->allocHost([1],$X->dtype());
                $pattern[0] = $value;
            } elseif($value instanceof NDArray) {
                if($value->size()!=1) {
                    throw new InvalidArgumentException('Value must be scalar');
                }
                if(!($value->buffer() instanceof LinearBuffer)) {
                    throw new InvalidArgumentException('Value must have a host memory buffer. OpenCL Buffer is not allowed.');
                }
                $pattern = $value;
            } else {
                throw new InvalidArgumentException('Invalid data type');
            }
            $buffer = $X->buffer();
            $buffer->fill(
                $this->getQueue(),
                $pattern->buffer(),
                $X->size()*$buffer->value_size(),
                $X->offset()*$X->valueSize(), // buffer offset
                $pattern->size(), // pattern size
                $pattern->offset(), // pattern offset
                $events,$waitEvents
            );
        }
        if($this->blocking) {
            $this->finish();
        }
        if($this->profiling) {
            $this->profilingEnd("fill");
        }
        return $X;
    }

    public function searchsorted(
        NDArray $A,
        NDArray $X,
        bool $right=null,
        int $dtype=null,
        NDArray $Y=null,
        object $events=null, object $waitEvents=null) : NDArray
    {
        if($this->profiling) {
            $this->profilingStart("searchsorted");
        }
        if($A->ndim()==1) {
            $individual = false;
        } elseif($A->ndim()==2) {
            $individual = true;
        } else {
            throw new InvalidArgumentException('A must be 1D or 2D NDArray.');
        }
        if($right===null) {
            $right = false;
        }
        if($dtype===null) {
            $dtype = NDArray::uint32;
        }
        if($Y===null) {
            $Y = $this->alloc($X->shape(),dtype:$dtype);
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
        if($individual) {
            [$m,$n] = $A->shape();
            if($m!=$X->size()) {
                $shapeError = '('.implode(',',$A->shape()).'),('.implode(',',$X->shape()).')';
                throw new InvalidArgumentException("Unmatch shape of dimension A,X: ".$shapeError);
            }
            $ldA = $n;
        } else {
            $m = $X->size();
            $n = $A->size();
            $ldA = 0;
        }
        $AA = $A->buffer();
        $offA = $A->offset();
        $XX = $X->buffer();
        $offX = $X->offset();
        $YY = $Y->buffer();
        $offY = $Y->offset();

        $this->openclmath->searchsorted(
            $m,
            $n,
            $AA,$offA,$ldA,
            $XX,$offX,1,
            $right,
            $YY,$offY,1,
            $events,$waitEvents
        );

        if($this->blocking) {
            $this->finish();
        }
        if($this->profiling) {
            $this->profilingEnd("searchsorted");
        }
        return $Y;
    }

    public function cumsum(
        NDArray $X,
        bool $exclusive=null,
        bool $reverse=null,
        NDArray $Y=null,
        object $events=null, object $waitEvents=null) : NDArray
    {
        if($this->profiling) {
            $this->profilingStart("cumsum");
        }
        if($exclusive===null) {
            $exclusive = false;
        }
        if($reverse===null) {
            $reverse = false;
        }
        if($Y===null) {
            $Y = $this->alloc($X->shape(),dtype:$X->dtype());
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

        $this->openclmath->cumsum(
            $n,
            $XX,$offX,1,
            $exclusive,
            $reverse,
            $YY,$offY,1,
            $events,$waitEvents
        );

        if($this->blocking) {
            $this->finish();
        }
        if($this->profiling) {
            $this->profilingEnd("cumsum");
        }
        return $Y;
    }

    /**
     *     X := nan2num(X)
     */
    public function nan2num(
        NDArray $X,
        float $alpha=null,
        object $events=null,
        object $waitEvents=null
        ) : NDArray
    {
        if($this->profiling) {
            $this->profilingStart("nan2num");
        }
        $n = $X->size();
        $XX = $X->buffer();
        $offX = $X->offset();

        if($alpha===null) {
            $alpha = 0.0;
        }

        $this->openclmath->nan2num(
            $n,
            $XX,$offX,1,
            $alpha,
            $events,$waitEvents
        );

        if($this->blocking) {
            $this->finish();
        }
        if($this->profiling) {
            $this->profilingEnd("nan2num");
        }
        return $X;
    }

    /**
     *     X := isnan(X)
     */
    public function isnan(
        NDArray $X,
        object $events=null,
        object $waitEvents=null
        ) : NDArray
    {
        if($this->profiling) {
            $this->profilingStart("isnan");
        }
        $n = $X->size();
        $XX = $X->buffer();
        $offX = $X->offset();

        $this->openclmath->isnan(
            $n,
            $XX,$offX,1,
            $events,$waitEvents
        );

        if($this->blocking) {
            $this->finish();
        }
        if($this->profiling) {
            $this->profilingEnd("isnan");
        }
        return $X;
    }

    public function linspace(float $start, float $stop, int $num, int $dtype=null) : NDArray
    {
        if($num<=0) {
            throw new InvalidArgumentException('num must be greater than or equal zero.');
        }
        $array = new NDArrayPhp(null,dtype:$dtype,shape:[$num],service:$this->service);
        $step = ($stop-$start)/($num-1);
        $value = $start;
        for($i=0;$i<$num;$i++) {
            $array[$i] = min($start+$step*$i,$stop);
        }
        $array = $this->array($array);
        return $array;
    }

    /**
     * @param array<int> $shape
     */
    public function allocHost(array $shape,int $dtype=null) : NDArray
    {
        if($dtype===null)
            $dtype = $this->defaultFloatType;
        return new NDArrayPhp(null,dtype:$dtype,shape:$shape,service:$this->service);
    }

    public function zerosHost(
        NDArray $X) : NDArray
    {
        if($this->profiling) {
            $this->profilingStart("zerosHost");
        }
        $N = $X->size();
        $XX = $X->buffer();
        $offX = $X->offset();
        $this->openblasmath->zeros($N,$XX,$offX,1);
        if($this->profiling) {
            $this->profilingEnd("zerosHost");
        }
        return $X;
    }

    #protected function newHostBuffer($size,$dtype)
    #{
    #    return new OpenBlasBuffer($size,$dtype);
    #}

    #protected function newBuffer(
    #    int $size, int $flags=null,
    #    LinearBuffer $hostBuffer=null, int $hostOffset=null,
    #    int $dtype=null)
    #{
    #    return $this->service->buffer()->Buffer($this->context,
    #        $size,$flags,$hostBuffer,$hostOffset,$dtype);
    #}

    public function newEventList() : object
    {
        return $this->service->openCL()->EventList();
    }

    public function astype(NDArray $X, int $dtype, NDArray $Y=null,
        object $events=null,object $waitEvents=null) : NDArray
    {
        if($this->profiling) {
            $this->profilingStart("astype");
        }
        if($Y==null) {
            $Y = $this->alloc($X->shape(),dtype:$dtype);
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
        if($this->profiling) {
            $this->profilingEnd("astype");
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
        if($this->profiling) {
            $this->profilingStart("copy");
        }
        if($Y===null) {
            $Y = $this->alloc($X->shape(),dtype:$X->dtype(),flags:OpenCL::CL_MEM_READ_WRITE);
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
        if($this->profiling) {
            $this->profilingEnd("copy");
        }
        return $Y;
    }

    /**
    *    X := alpha * X
    */
    public function scal(
        float|object $alpha,
        NDArray $X,
        object $events=null) : NDArray
    {
        if($this->profiling) {
            $this->profilingStart("scal");
        }
        $N = $X->size();
        $XX = $X->buffer();
        $offX = $X->offset();
        $this->blas->scal($N,$alpha,$XX,$offX,1,$this->queue,$events);
        if($this->blocking) {
            $this->finish();
        }
        if($this->profiling) {
            $this->profilingEnd("scal");
        }
        return $X;
    }

    /**
    *    Y := alpha * X + Y
    */
    public function axpy(
        NDArray $X,
        NDArray $Y,
        float|object $alpha=null,
        object $events=null) : NDArray
    {
        if($this->profiling) {
            $this->profilingStart("axpy");
        }
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
            $alpha = $this->buildValByType(1.0,$X->dtype());
        }
        $this->blas->axpy($N,$alpha,$XX,$offX,1,$YY,$offY,1,$this->queue,$events);
        if($this->blocking) {
            $this->finish();
        }
        if($this->profiling) {
            $this->profilingEnd("axpy");
        }
        return $Y;
    }

    /**
    *    ret := X^t Y = x_1 * y_1 + ... + x_n * y_n
    */
    public function dot(
        NDArray $X,
        NDArray $Y,
        bool $conj=null,
        NDArray $output=null,
        object $events=null) : float|object
    {
        $R = $output;
        if($this->profiling) {
            $this->profilingStart("dot");
        }
        if($X->shape()!=$Y->shape()) {
            $shapeError = '('.implode(',',$X->shape()).'),('.implode(',',$Y->shape()).')';
            throw new InvalidArgumentException("Unmatch shape of dimension: ".$shapeError);
        }
        if($R==null) {
            $R = $this->alloc([],dtype:$X->dtype(),flags:OpenCL::CL_MEM_READ_WRITE);
        }
        $N = $X->size();
        $RR = $R->buffer();
        $offR = $R->offset();
        $XX = $X->buffer();
        $offX = $X->offset();
        $YY = $Y->buffer();
        $offY = $Y->offset();

        if(!$this->isComplex($X->dtype())) {
            $this->blas->dot($N,$RR,$offR,$XX,$offX,1,$YY,$offY,1,$this->queue,$events);
        } elseif($conj===false) { // explicit conjugation false
            $this->blas->dotu($N,$RR,$offR,$XX,$offX,1,$YY,$offY,1,$this->queue,$events);
        } else {            // implicit conjugation true
            $this->blas->dotc($N,$RR,$offR,$XX,$offX,1,$YY,$offY,1,$this->queue,$events);
        }

        if($this->blocking) {
            $this->finish();
        }
        if($this->profiling) {
            $this->profilingEnd("dot");
        }
        if($this->scalarNumeric) {
            $value = $R->toArray();
            if($this->isComplex($X->dtype())) {
                $value = $this->crebuild($value);
            }
            return $value;
        }
        return $R;
    }

    /**
    *    ret := |x_1| + ... + |x_n|
    */
    public function asum(
        NDArray $X,
        NDArray $output=null,
        object $events=null) : float|NDArray
    {
        $R = $output;
        if($this->profiling) {
            $this->profilingStart("asum");
        }
        if($R==null) {
            $dtype = $X->dtype();
            $R = $this->alloc([],dtype:$dtype,flags:OpenCL::CL_MEM_READ_WRITE);
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
        if($this->profiling) {
            $this->profilingEnd("asum");
        }
        if($this->scalarNumeric) {
            $value = $R->toArray();
            if($this->isComplex($R->dtype())) {
                $value= $value->real;
            }
            return $value;
        }
        return $R;
    }

    /**
    *    ret := arg max |X(i)|
    */
    public function iamax(
        NDArray $X,
        NDArray $R=null,
        object $events=null) : int|NDArray
    {
        if($this->profiling) {
            $this->profilingStart("iamax");
        }
        if($R==null) {
            // *** CAUTION ****
            // Index result is 32bit
            $R = $this->alloc([],dtype:NDArray::int32,flags:OpenCL::CL_MEM_READ_WRITE);
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
        if($this->profiling) {
            $this->profilingEnd("iamax");
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
        object $events=null) : int|NDArray
    {
        if($this->profiling) {
            $this->profilingStart("iamin");
        }
        if($R==null) {
            // *** CAUTION ****
            // Index result is 32bit
            $R = $this->alloc([],dtype:NDArray::int32,flags:OpenCL::CL_MEM_READ_WRITE);
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
        if($this->profiling) {
            $this->profilingEnd("iamin");
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
        object $events=null) : float|NDArray
    {
        if($this->profiling) {
            $this->profilingStart("amax");
        }
        if($R==null) {
            $R = $this->alloc([],dtype:$X->dtype(),flags:OpenCL::CL_MEM_READ_WRITE);
        }
        // *** CAUTION ****
        // Index result is 32bit
        $IR = $this->alloc([],dtype:NDArray::int32,flags:OpenCL::CL_MEM_READ_WRITE);
        $N = $X->size();
        $IRR = $IR->buffer();
        $offIR = $IR->offset();
        $XX = $X->buffer();
        $offX = $X->offset();
        $imaxEvents = $this->newEventList();
        $this->blas->iamax($N,$IRR,$offIR,$XX,$offX,1,$this->queue,$imaxEvents);
        $imaxEvents->wait();

        $idx = $IR->toArray();
        $RR = $R->buffer();
        $offR = $R->offset();
        $valueSize = $RR->value_size();
        if($this->scalarNumeric) {
            $this->blas->copy(1,$XX,$offX+$idx,1,$RR,$offR,1,$this->queue,$events);
        } else {
            $copyEvents = $this->newEventList();
            //$this->blas->copy(1,$XX,$offX+$idx,1,$RR,$offR,1,$this->queue,$copyEvents);
            $RR->copy($this->queue,
                $XX, $valueSize, ($offX+$idx)*$valueSize, $offR*$valueSize,
                $copyEvents,$imaxEvents
            );
            $this->openclmath->abs(1,$RR,$offR,1,$events,$copyEvents);
        }
        //$RR = $R->buffer();
        //$offR = $R->offset();
        //$this->openclmath->reduceGather(false,false,
        //    1,1,$N,$IRR,$offIR,$XX,$offX,$RR,$offR,$events,$imaxEvents);
        //if($this->blocking) {
        //    $this->finish();
        //}
        if($this->profiling) {
            $this->profilingEnd("amax");
        }
        if($this->scalarNumeric) {
            $value = $R->toArray();
            if($this->isComplex($X->dtype())) {
                $value = $this->cabs($value);
            } else {
                $value = abs($value);
            }
            return $value;
        }
        return $R;
    }

    /**
    *    ret := min |X(i)|
    */
    public function amin(
        NDArray $X,
        NDArray $R=null,
        object $events=null) : float|NDArray
    {
        if($this->profiling) {
            $this->profilingStart("amin");
        }
        if($R==null) {
            $R = $this->alloc([],dtype:$X->dtype(),flags:OpenCL::CL_MEM_READ_WRITE);
        }
        // *** CAUTION ****
        // Index result is 32bit
        $IR = $this->alloc([],dtype:NDArray::int32,flags:OpenCL::CL_MEM_READ_WRITE);
        $N = $X->size();
        $IRR = $IR->buffer();
        $offIR = $IR->offset();
        $XX = $X->buffer();
        $offX = $X->offset();
        $imaxEvents = $this->newEventList();
        $this->blas->iamin($N,$IRR,$offIR,$XX,$offX,1,$this->queue,$imaxEvents);
        $imaxEvents->wait();

        $idx = $IR->toArray();
        $RR = $R->buffer();
        $offR = $R->offset();
        $valueSize = $RR->value_size();
        if($this->scalarNumeric) {
            $this->blas->copy(1,$XX,$offX+$idx,1,$RR,$offR,1,$this->queue,$events);
        } else {
            $copyEvents = $this->newEventList();
            //$this->blas->copy(1,$XX,$offX+$idx,1,$RR,$offR,1,$this->queue,$copyEvents);
            $RR->copy($this->queue,
                $XX, $valueSize, ($offX+$idx)*$valueSize, $offR*$valueSize,
                $copyEvents,$imaxEvents
            );
            $this->openclmath->abs(1,$RR,$offR,1,$events,$copyEvents);
        }
        //$RR = $R->buffer();
        //$offR = $R->offset();
        //$this->openclmath->reduceGather(false,false,
        //    1,1,$N,$IRR,$offIR,$XX,$offX,$RR,$offR,$events,$imaxEvents);
        if($this->blocking) {
            $this->finish();
        }
        if($this->profiling) {
            $this->profilingStart("amin");
        }
        if($this->scalarNumeric) {
            $value = $R->toArray();
            if($this->isComplex($X->dtype())) {
                $value = $this->cabs($value);
            } else {
                $value = abs($value);
            }
            return $value;
        }
        return $R;
    }

    /**
    *    ret := sqrt(sum(Xn ** 2))
    */
    public function nrm2(
        NDArray $X,
        NDArray $R=null,
        object $events=null) : mixed
    {
        if($this->profiling) {
            $this->profilingStart("nrm2");
        }
        if($R==null) {
            $R = $this->alloc([],dtype:$X->dtype(),flags:OpenCL::CL_MEM_READ_WRITE);
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
        if($this->profiling) {
            $this->profilingEnd("nrm2");
        }
        if($this->scalarNumeric) {
            $value = $R->toArray();
            if($this->isComplex($R->dtype())) {
                $value= $value->real;
            }
            return $value;
        }
        return $R;
    }

    /**
     * a,b,c,s = rotg(x,y)
     * @return array<NDArray>
     */
    public function rotg(
        NDArray $X,
        NDArray $Y,
        NDArray $R=null,
        NDArray $Z=null,
        NDArray $C=null,
        NDArray $S=null,
        object $events=null) : array
    {
        if($this->profiling) {
            $this->profilingStart("rotg");
        }
        if($X->shape()!=$Y->shape()) {
            $shapeError = '('.implode(',',$X->shape()).'),('.implode(',',$Y->shape()).')';
            throw new InvalidArgumentException("Unmatch shape of dimension: ".$shapeError);
        }
        $copyEvents = $this->newEventList();
        $R = $this->copy($X,$R,$copyEvents);
        $Z = $this->copy($Y,$Z,$copyEvents);
        $copyEvents->wait();
        if($C==null) {
            $C = $this->alloc($X->shape(),dtype:$X->dtype());
        }
        if($S==null) {
            $S = $this->alloc($Y->shape(),dtype:$X->dtype());
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
            $SS,$offS,
            $this->queue,$events,
        );
        if($this->blocking) {
            $this->finish();
        }
        if($this->profiling) {
            $this->profilingEnd("rotg");
        }
        return [$R,$Z,$C,$S];
    }

    /**
    *    x,y := rot(x,y,c,s)
    */
    public function rot(
        NDArray $X,
        NDArray $Y,
        NDArray $C,
        NDArray $S,
        object $events=null) : void
    {
        if($this->profiling) {
            $this->profilingStart("rot");
        }
        if($X->shape()!=$Y->shape()) {
            $shapeError = '('.implode(',',$X->shape()).'),('.implode(',',$Y->shape()).')';
            throw new InvalidArgumentException("Unmatch shape of dimension: ".$shapeError);
        }
        $N = $X->size();
        $XX = $X->buffer();
        $offX = $X->offset();
        $YY = $Y->buffer();
        $offY = $Y->offset();
        $scalarC = $C->reshape([])->toArray();
        $scalarS = $S->reshape([])->toArray();
        $this->blas->rot(
            $N,
            $XX,$offX,1,
            $YY,$offY,1,
            $scalarC,$scalarS,
            $this->queue,$events,
        );
        if($this->blocking) {
            $this->finish();
        }
        if($this->profiling) {
            $this->profilingEnd("rot");
        }
    }

    /**
     * g = rotg(x,y)
     */
    public function rotgxy(
        NDArray $vector,
        NDArray $g=null,
        object $events=null) : NDArray
    {
        if($this->profiling) {
            $this->profilingStart("rotgxy");
        }
        if($vector->shape()!=[2]) {
            throw new InvalidArgumentException("Shape of vector must be [2]: [".implode(',',$vector->shape())."]");
        }
        if($g==null) {
            $g = $this->alloc([4],dtype:$vector->dtype());
        } else {
            if($g->shape()!=[4]) {
                throw new InvalidArgumentException("Shape of g must be [4]: [".implode(',',$g->shape())."]");
            }
        }
        $copyEvents = $this->newEventList();
        $this->copy($vector[R(0,1)],$g[R(0,1)],$copyEvents);
        $this->copy($vector[R(1,2)],$g[R(1,2)],$copyEvents);
        $copyEvents->wait();

        if($g==null) {
            $g = $this->alloc([2],dtype:$vector->dtype());
        }
        $AA = $g->buffer();
        $offA = $g->offset();
        $BB = $g->buffer();
        $offB = $g->offset()+1;
        $CC = $g->buffer();
        $offC = $g->offset()+2;
        $SS = $g->buffer();
        $offS = $g->offset()+3;
        $this->blas->rotg(
            $AA,$offA,
            $BB,$offB,
            $CC,$offC,
            $SS,$offS,
            $this->queue,$events,
        );
        if($this->blocking) {
            $this->finish();
        }
        if($this->profiling) {
            $this->profilingEnd("rotgxy");
        }
        return $g;
    }

    /**
    *    xy := rot(xy,g)
    */
    public function rotxy(
        NDArray $vectors,
        NDArray $g,
        object $events=null) : void
    {
        if($this->profiling) {
            $this->profilingStart("rotxy");
        }
        if($vectors->ndim()!=2) {
            $shapeError = '['.implode(',',$vectors->shape()).']';
            throw new InvalidArgumentException("vectors must be 2D-NDArray: ".$shapeError." given.");
        }
        $shape = $vectors->shape()[1];
        if($shape!=2) {
            $shapeError = '['.implode(',',$vectors->shape()).']';
            throw new InvalidArgumentException("Vectors must be Vectors-NDArray: ".$shapeError." given.");
        }
        if($g->shape()!=[4]) {
            $shapeError = '['.implode(',',$g->shape()).']';
            throw new InvalidArgumentException("shape of g must be [4]: ".$shapeError." given.");
        }
        
        $N = count($vectors);
        $XX = $vectors->buffer();
        $offX = $vectors->offset();
        $YY = $vectors->buffer();
        $offY = $vectors->offset()+1;
        $CC = $g->buffer();
        $offC = $g->offset()+2;
        $SS = $g->buffer();
        $offS = $g->offset()+3;
        $this->blas->rot($N,
            $XX,$offX,2,$YY,$offY,2,
            $CC,$offC,$SS,$offS
        );
        if($this->blocking) {
            $this->finish();
        }
        if($this->profiling) {
            $this->profilingEnd("rotxy");
        }
    }

    /**
     * d1,d2,b1,p = rotmg(x,y)   b1: rotated x   p: params  d1,d2:works
     * @return array<NDArray>
     */
    public function rotmg(
        NDArray $X,
        NDArray $Y,
        NDArray $D1=null,
        NDArray $D2=null,
        NDArray $B1=null,
        NDArray $P=null,
        object $events=null) : array
    {
        if($this->profiling) {
            $this->profilingStart("rotmg");
        }
        if($X->size()!=1||$Y->size()!=1) {
            $shapeError = '('.implode(',',$X->shape()).'),('.implode(',',$Y->shape()).')';
            throw new InvalidArgumentException("Unmatch shape of dimension: ".$shapeError);
        }
        if($D1==null) {
            $D1 = $this->ones($this->alloc([],dtype:$X->dtype()));
        }
        if($D2==null) {
            $D2 = $this->ones($this->alloc([],dtype:$X->dtype()));
        }
        if($B1==null) {
            $B1 = $this->alloc([],dtype:$X->dtype());
        }
        $B2 = $this->alloc([],dtype:$Y->dtype());
        if($P==null) {
            $P = $this->zeros($this->alloc([5],dtype:$X->dtype()));
        }
        $copyEvents = $this->newEventList();
        $this->copy($X->reshape([1]),$B1->reshape([1]),$copyEvents);
        $this->copy($X->reshape([1]),$B2->reshape([1]),$copyEvents);
        $copyEvents->wait();

        $DD1 = $D1->buffer();
        $offD1 = $D1->offset();
        $DD2 = $D2->buffer();
        $offD2 = $D2->offset();
        $BB1 = $B1->buffer();
        $offB1 = $B1->offset();
        $BB2 = $B2->buffer();
        $offB2 = $B2->offset();
        $PP = $P->buffer();
        $offP = $P->offset();
        $this->blas->rotmg(
            $DD1,$offD1,
            $DD2,$offD2,
            $BB1,$offB1,
            $BB2,$offB2,
            $PP,$offP,
            $this->queue,$events,
        );
        if($this->blocking) {
            $this->finish();
        }
        if($this->profiling) {
            $this->profilingEnd("rotmg");
        }
        return [$D1,$D2,$B1,$P];
    }

    /**
    *    x,y := rot(x,y,p)
    */
    public function rotm(
        NDArray $X,
        NDArray $Y,
        NDArray $P,
        object $events=null) : void
    {
        if($this->profiling) {
            $this->profilingStart("rotm");
        }
        if($X->shape()!=$Y->shape()) {
            $shapeError = '('.implode(',',$X->shape()).'),('.implode(',',$Y->shape()).')';
            throw new InvalidArgumentException("Unmatch shape of dimension: ".$shapeError);
        }
        $N = $X->size();
        $XX = $X->buffer();
        $offX = $X->offset();
        $YY = $Y->buffer();
        $offY = $Y->offset();
        $PP = $P->buffer();
        $offP = $P->offset();
        $this->blas->rotm(
            $N,
            $XX,$offX,1,
            $YY,$offY,1,
            $PP,$offP,
            $this->queue,$events,
        );
        if($this->blocking) {
            $this->finish();
        }
        if($this->profiling) {
            $this->profilingEnd("rotm");
        }
    }

    /**
     * d1,d2,b1,p = rotmg(x,y)   b1: rotated x   p: params  d1,d2:works
     */
    public function rotmgxy(
        NDArray $vector,
        NDArray $d=null,
        NDArray $g=null,
        object $events=null) : NDArray
    {
        if($this->profiling) {
            $this->profilingStart("rotmgxy");
        }
        if($vector->shape()!=[2]) {
            throw new InvalidArgumentException("Shape of vector must be [2]: [".implode(',',$vector->shape())."]");
        }
        if($d==null) {
            $d = $this->ones($this->alloc([2],dtype:$vector->dtype()));
        } else {
            if($d->shape()!=[2]) {
                throw new InvalidArgumentException("Shape of d must be [2]: [".implode(',',$d->shape())."]");
            }
        }
        if($g==null) {
            $g = $this->zeros($this->alloc([6],dtype:$vector->dtype()));
        } else {
            if($g->shape()!=[6]) {
                throw new InvalidArgumentException("Shape of g must be [6]: [".implode(',',$g->shape())."]");
            }
        }
        $B2 = $this->alloc([1],dtype:$vector->dtype());
        $copyEvents = $this->newEventList();
        $this->copy($vector[R(0,1)],$g[R(0,1)],$copyEvents);
        $this->copy($vector[R(1,2)],$B2,$copyEvents);
        $copyEvents->wait();

        $DD1 = $d->buffer();
        $offD1 = $d->offset();
        $DD2 = $d->buffer();
        $offD2 = $d->offset()+1;
        $BB1 = $g->buffer();
        $offB1 = $g->offset();
        $BB2 = $B2->buffer();
        $offB2 = $B2->offset();
        $PP = $g->buffer();
        $offP = $g->offset()+1;
        $this->blas->rotmg(
            $DD1,$offD1,
            $DD2,$offD2,
            $BB1,$offB1,
            $BB2,$offB2,
            $PP,$offP,
            $this->queue,$events,
        );
        if($this->blocking) {
            $this->finish();
        }
        if($this->profiling) {
            $this->profilingEnd("rotmgxy");
        }
        return $g;
    }

    /**
    *    x,y := rot(x,y,p)
    */
    public function rotmxy(
        NDArray $vectors,
        NDArray $g,
        object $events=null) : void
    {
        if($this->profiling) {
            $this->profilingStart("rotmxy");
        }
        if($vectors->ndim()!=2) {
            $shapeError = '['.implode(',',$vectors->shape()).']';
            throw new InvalidArgumentException("vectors must be 2D-NDArray: ".$shapeError." given.");
        }
        $shape = $vectors->shape()[1];
        if($shape!=2) {
            $shapeError = '['.implode(',',$vectors->shape()).']';
            throw new InvalidArgumentException("Vectors must be Vectors-NDArray: ".$shapeError." given.");
        }
        if($g->shape()!=[6]) {
            $shapeError = '['.implode(',',$g->shape()).']';
            throw new InvalidArgumentException("shape of g must be [6]: ".$shapeError." given.");
        }

        $N = count($vectors);
        $XX = $vectors->buffer();
        $offX = $vectors->offset();
        $YY = $vectors->buffer();
        $offY = $vectors->offset()+1;
        $PP = $g->buffer();
        $offP = $g->offset()+1;
        $this->blas->rotm(
            $N,
            $XX,$offX,2,
            $YY,$offY,2,
            $PP,$offP,
            $this->queue,$events,
        );
        if($this->blocking) {
            $this->finish();
        }
        if($this->profiling) {
            $this->profilingEnd("rotmxy");
        }
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
        if($this->profiling) {
            $this->profilingStart("swap");
        }
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
        if($this->profiling) {
            $this->profilingEnd("swap");
        }
    }

    /**
    *    y := alpha * Ax + beta * y
    */
    public function gemv(
        NDArray $A,
        NDArray $X,
        float|object $alpha=null,
        float|object $beta=null,
        NDArray $Y=null,
        bool $trans=null,
        bool $conj=null,
        object $events=null) : NDArray
    {
        if($this->profiling) {
            $this->profilingStart("gemv");
        }
        [$trans,$conj] = $this->complementTrans($trans,$conj,$A->dtype());
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
            $alpha = $this->buildValByType(1.0,$A->dtype());
        }
        if($beta===null) {
            $beta = $this->buildValByType(0.0,$A->dtype());
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
            $Y = $this->alloc([$rows],dtype:$X->dtype(),flags:OpenCL::CL_MEM_READ_WRITE);
            // ** CAUTION ** it must fill it with zeros because NAN can't reset by beta.
            $zerosEvents = $this->newEventList();
            $this->zeros($Y,events:$zerosEvents);
            $zerosEvents->wait();
            $beta = $this->buildValByType(0.0,$A->dtype());
        }
        $YY = $Y->buffer();
        $offY = $Y->offset();
        $trans = $this->transToCode($trans,$conj);

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
        if($this->profiling) {
            $this->profilingEnd("gemv");
        }
        return $Y;
    }

    /**
    *    C := alpha * AB + beta * C
    */
    public function gemm(
        NDArray $A,
        NDArray $B,
        float|object $alpha=null,
        float|object $beta=null,
        NDArray $C=null,
        bool $transA=null,
        bool $transB=null,
        bool $conjA=null,
        bool $conjB=null,
        object $events=null) : NDArray
    {
        [$transA,$conjA] = $this->complementTrans($transA,$conjA,$A->dtype());
        [$transB,$conjB] = $this->complementTrans($transB,$conjB,$B->dtype());
        if($this->profiling) {
            $this->profilingStart("gemm");
        }
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
            $alpha = $this->buildValByType(1.0,$A->dtype());
        }
        if($beta===null) {
            $beta = $this->buildValByType(0.0,$A->dtype());
        }
        if($C!=null) {
            $shapeC = $C->shape();
            if($M!=$shapeC[0] || $N!=$shapeC[1]) {
                throw new InvalidArgumentException('"A" and "C" must have the same number of rows."B" and "C" must have the same number of columns');
            }
        } else {
            $C = $this->alloc([$M,$N],dtype:$A->dtype());
            // ** CAUTION ** it must fill it with zeros because NAN can't reset by beta.
            $zerosEvents = $this->newEventList();
            $this->zeros($C,events:$zerosEvents);
            $zerosEvents->wait();
            $beta = $this->buildValByType(0.0,$A->dtype());
        }
        $CC = $C->buffer();
        $offC = $C->offset();

        $lda = ($transA) ? $M : $K;
        $ldb = ($transB) ? $K : $N;
        $ldc = $N;
        $transA = $this->transToCode($transA,$conjA);
        $transB = $this->transToCode($transB,$conjB);

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
        if($this->profiling) {
            $this->profilingEnd("gemm");
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
        float|object $alpha=null,
        float|object $beta=null,
        bool $conjA=null,
        bool $conjB=null,
        object $events=null
        ) : NDArray
    {
        [$transA,$conjA] = $this->complementTrans($transA,$conjA,$A->dtype());
        [$transB,$conjB] = $this->complementTrans($transB,$conjB,$B->dtype());
        if($this->profiling) {
            $this->profilingStart("matmul");
        }
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
            $alpha = $this->buildValByType(1.0,$A->dtype());
        }
        if($beta===null) {
            $beta = $this->buildValByType(0.0,$A->dtype());
        }
        $lda = ($transA) ? $M : $K;
        $ldb = ($transB) ? $K : $N;
        $ldc = $N;
        $transA = $this->transToCode($transA,$conjA);
        $transB = $this->transToCode($transB,$conjB);

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
            $C = $this->alloc($orgShapeC,dtype:$A->dtype());
            // ** CAUTION ** it must fill it with zeros because NAN can't reset by beta.
            $zerosEvents = $this->newEventList();
            $this->zeros($C,events:$zerosEvents);
            $zerosEvents->wait();
        }
        $flatC = $C->reshape(array_merge([$broadcastDest],$shapeEC));
        $CC = $C->buffer();
        $repeats = intdiv($broadcastDest,$broadcastBase);
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
        if($this->profiling) {
            $this->profilingEnd("matmul");
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
        float|object $alpha=null,
        float|object $beta=null,
        NDArray $C=null,
        bool $right=null,
        bool $lower=null,
        object $events=null
        ) : NDArray
    {
        if($this->profiling) {
            $this->profilingStart("symm");
        }
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
            $alpha = $this->buildValByType(1.0,$A->dtype());
        }
        if($beta===null) {
            $beta = $this->buildValByType(0.0,$A->dtype());
        }
        if($C!=null) {
            $shapeC = $C->shape();
            if($M!=$shapeC[0] || $N!=$shapeC[1]) {
                throw new InvalidArgumentException('Matrix "B" and "C" must be same shape');
            }
        } else {
            $C = $this->alloc([$M,$N],dtype:$A->dtype());
            // ** CAUTION ** it must fill it with zeros because NAN can't reset by beta.
            $zerosEvents = $this->newEventList();
            $this->zeros($C,events:$zerosEvents);
            $zerosEvents->wait();
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
            $CC,$offC,$ldc,
            $this->queue,$events);

        if($this->blocking) {
            $this->finish();
        }
        if($this->profiling) {
            $this->profilingEnd("symm");
        }
        return $C;
    }

    /**
    *    C := alpha * A A^T + beta * C  (trans=false)
    *    C := alpha * A^T A + beta * C  (trans=true)
    */
    public function syrk(
        NDArray $A,
        float|object $alpha=null,
        float|object $beta=null,
        NDArray $C=null,
        bool $lower=null,
        bool $trans=null,
        bool $conj=null,
        object $events=null,
        ) : NDArray
    {
        $trans = $trans ?? false;
        // $conj = $conj ?? $trans; // Doing so will result in an error.
        $conj = false;  // conj must be false

        if($this->profiling) {
            $this->profilingStart("syrk");
        }
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
            $alpha = $this->buildValByType(1.0,$A->dtype());
        }
        if($beta===null) {
            $beta = $this->buildValByType(0.0,$A->dtype());
        }
        if($C!=null) {
            $shapeC = $C->shape();
            if($N!=$shapeC[0] || $N!=$shapeC[1]) {
                throw new InvalidArgumentException('"C" rows and cols must have the same number of "A" cols');
            }
        } else {
            $C = $this->alloc([$N,$N],dtype:$A->dtype());
            // ** CAUTION ** it must fill it with zeros because NAN can't reset by beta.
            $zerosEvents = $this->newEventList();
            $this->zeros($C,events:$zerosEvents);
            $zerosEvents->wait();
        }
        $CC = $C->buffer();
        $offC = $C->offset();

        $lda = ($trans) ? $N : $K;
        $ldc = $N;
        $uplo  = ($lower) ? BLAS::Lower : BLAS::Upper;
        $trans = $this->transToCode($trans,$conj);

        $this->blas->syrk(
            BLAS::RowMajor,$uplo,$trans,
            $N,$K,
            $alpha,
            $AA,$offA,$lda,
            $beta,
            $CC,$offC,$ldc,
            $this->queue,$events);

        if($this->blocking) {
            $this->finish();
        }
        if($this->profiling) {
            $this->profilingEnd("syrk");
        }

        return $C;
    }

    /**
    *    C := alpha * A B^T + B A^T + beta * C  (trans=false)
    *    C := alpha * B^T A + A B^T + beta * C  (trans=true)
    */
    public function syr2k(
        NDArray $A,
        NDArray $B,
        float|object $alpha=null,
        float|object $beta=null,
        NDArray $C=null,
        bool $lower=null,
        bool $trans=null,
        bool $conj=null,
        object $events=null
        ) : NDArray
    {
        $trans = $trans ?? false;
        // $conj = $conj ?? $trans; // Doing so will result in an error.
        $conj = false;  // conj must be false

        if($this->profiling) {
            $this->profilingStart("syr2k");
        }
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
            $alpha = $this->buildValByType(1.0,$A->dtype());
        }
        if($beta===null) {
            $beta = $this->buildValByType(0.0,$A->dtype());
        }
        if($C!=null) {
            $shapeC = $C->shape();
            if($N!=$shapeC[0] || $N!=$shapeC[1]) {
                throw new InvalidArgumentException('"C" rows and cols must have the same number of "A" cols');
            }
        } else {
            $C = $this->alloc([$N,$N],dtype:$A->dtype());
            // ** CAUTION ** it must fill it with zeros because NAN can't reset by beta.
            $zerosEvents = $this->newEventList();
            $this->zeros($C,events:$zerosEvents);
            $zerosEvents->wait();
        }
        $CC = $C->buffer();
        $offC = $C->offset();

        $lda = ($trans) ? $N : $K;
        $ldb = ($trans) ? $N : $K;
        $ldc = $N;
        $uplo  = ($lower) ? BLAS::Lower : BLAS::Upper;
        $trans = $this->transToCode($trans,$conj);

        $this->blas->syr2k(
            BLAS::RowMajor,$uplo,$trans,
            $N,$K,
            $alpha,
            $AA,$offA,$lda,
            $BB,$offB,$ldb,
            $beta,
            $CC,$offC,$ldc,
            $this->queue,$events);

        if($this->blocking) {
            $this->finish();
        }
        if($this->profiling) {
            $this->profilingEnd("syr2k");
        }

        return $C;
    }

    /**
    *    C := alpha * A B  (right=false)
    *    C := alpha * B A  (right=true)
    */
    public function trmm(
        NDArray $A,
        NDArray $B,
        float|object $alpha=null,
        bool $right=null,
        bool $lower=null,
        bool $trans=null,
        bool $conj=null,
        bool $unit=null,
        object $events=null
        ) : NDArray
    {
        [$trans,$conj] = $this->complementTrans($trans,$conj,$A->dtype());

        if($this->profiling) {
            $this->profilingStart("trmm");
        }
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
            $alpha = $this->buildValByType(1.0,$A->dtype());
        }

        $lda = ($right) ? $N : $M;
        $ldb = $N;
        $side  = ($right) ? BLAS::Right : BLAS::Left;
        $uplo  = ($lower) ? BLAS::Lower : BLAS::Upper;
        $diag  = ($unit)  ? BLAS::Unit  : BLAS::NonUnit;
        $trans = $this->transToCode($trans,$conj);

        $this->blas->trmm(
            BLAS::RowMajor,$side,$uplo,$trans,$diag,
            $M,$N,
            $alpha,
            $AA,$offA,$lda,
            $BB,$offB,$ldb,
            $this->queue,$events);

        if($this->blocking) {
            $this->finish();
        }
        if($this->profiling) {
            $this->profilingEnd("trmm");
        }

        return $B;
    }

    /**
    *    C := alpha A^-1 B  (right=false)
    *    C := alpha B A^-1  (right=true)
    */
    public function trsm(
        NDArray $A,
        NDArray $B,
        float|object $alpha=null,
        bool $right=null,
        bool $lower=null,
        bool $trans=null,
        bool $conj=null,
        bool $unit=null,
        object $events=null
        ) : NDArray
    {
        [$trans,$conj] = $this->complementTrans($trans,$conj,$A->dtype());

        if($this->profiling) {
            $this->profilingStart("trsm");
        }
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
            $alpha = $this->buildValByType(1.0,$A->dtype());
        }

        $lda = ($right) ? $N : $M;
        $ldb = $N;
        $side  = ($right) ? BLAS::Right : BLAS::Left;
        $uplo  = ($lower) ? BLAS::Lower : BLAS::Upper;
        $diag  = ($unit)  ? BLAS::Unit  : BLAS::NonUnit;
        $trans = $this->transToCode($trans,$conj);

        $this->blas->trsm(
            BLAS::RowMajor,$side,$uplo,$trans,$diag,
            $M,$N,
            $alpha,
            $AA,$offA,$lda,
            $BB,$offB,$ldb,
            $this->queue,$events);

        if($this->blocking) {
            $this->finish();
        }
        if($this->profiling) {
            $this->profilingEnd("trsm");
        }

        return $B;
    }

    /**
     *  B := A    (trans=false)
     *  B := A^T  (trans=true)
     */
    public function omatcopy(
        NDArray $A,
        bool $trans=null,
        bool $conj=null,
        float|object $alpha=null,
        NDArray $B=null,
        object $events=null
        ) : NDArray
    {
        [$trans,$conj] = $this->complementTrans($trans,$conj,$A->dtype());

        if($this->profiling) {
            $this->profilingStart("omatcopy");
        }
        if($A->ndim()!=2) {
            throw new InvalidArgumentException('Dimensions must be 2D-NDArray');
        }
        [$rows,$cols] = $A->shape();
        if($trans) {
            [$rows,$cols] = [$cols,$rows];
        }
        if($B===null) {
            $B = $this->alloc([$rows,$cols],dtype:$A->dtype());
            // ** CAUTION ** it must fill it with zeros because NAN can't reset by beta.
            $zerosEvents = $this->newEventList();
            $this->zeros($B,events:$zerosEvents);
            $zerosEvents->wait();
        } else {
            if($B->shape()!=[$rows,$cols]) {
                $shapeError = '('.implode(',',$A->shape()).'),('.implode(',',$B->shape()).')';
                throw new InvalidArgumentException("Unmatch shape of dimension: ".$shapeError);
            }
        }

        [$M,$N] = $A->shape();
        if($alpha===null) {
            $alpha = $this->buildValByType(1.0,$A->dtype());
        }
        $AA = $A->buffer();
        $offA = $A->offset();
        $ldA = $N;
        $BB = $B->buffer();
        $offB = $B->offset();
        $ldB = $cols;

        $trans = $this->transToCode($trans,$conj);
        $order = BLAS::RowMajor;

        $this->blas->omatcopy(
            $order,$trans,
            $M,$N,
            $alpha,
            $AA, $offA, $ldA,
            $BB, $offB, $ldB,
            $this->queue,$events,
        );

        if($this->blocking) {
            $this->finish();
        }
        if($this->profiling) {
            $this->profilingEnd("omatcopy");
        }

        return $B;
    }

    /**
    *    ret := x_1 + ... + x_n
    */
    public function sum(
        NDArray $X,
        NDArray $R=null,
        object $events=null
        //object $waitEvents=null
        ) : mixed
    {
        if($this->profiling) {
            $this->profilingStart("sum");
        }
        if($R==null) {
            $R = $this->alloc([],dtype:$X->dtype(),flags:OpenCL::CL_MEM_READ_WRITE);
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
        if($this->profiling) {
            $this->profilingEnd("sum");
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
        object $events=null) : int|NDArray
    {
        if($this->profiling) {
            $this->profilingStart("imax");
        }
        if($R==null) {
            // *** CAUTION ****
            // Index result is 32bit
            $R = $this->alloc([],dtype:NDArray::int32,flags:OpenCL::CL_MEM_READ_WRITE);
        }
        $N = $X->size();
        $RR = $R->buffer();
        $offR = $R->offset();
        $XX = $X->buffer();
        $offX = $X->offset();
        if($this->isFloat($X)) {
            $this->math->imax($N,$RR,$offR,$XX,$offX,1,$this->queue,$events);
        } else {
            $this->openclmath->reduceArgMax(
                1,          // $m
                $N,         // $n
                1,          // $k
                $XX, $offX,
                $RR, $offR,
                $events
            );
        }
        if($this->blocking) {
            $this->finish();
        }
        if($this->profiling) {
            $this->profilingEnd("imax");
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
        object $events=null) : int|NDArray
    {
        if($this->profiling) {
            $this->profilingStart("imin");
        }
        if($R==null) {
            // *** CAUTION ****
            // Index result is 32bit
            $R = $this->alloc([],dtype:NDArray::int32,flags:OpenCL::CL_MEM_READ_WRITE);
        }
        $N = $X->size();
        $RR = $R->buffer();
        $offR = $R->offset();
        $XX = $X->buffer();
        $offX = $X->offset();
        if($this->isFloat($X)) {
            $this->math->imin($N,$RR,$offR,$XX,$offX,1,$this->queue,$events);
        } else {
            $this->openclmath->imin($N,$RR,$offR,$XX,$offX,1,$events);
            //throw new InvalidArgumentException("Unsuppored data type. Only float types are supported.");
        }
        if($this->blocking) {
            $this->finish();
        }
        if($this->scalarNumeric) {
            return $R->toArray();
        }
        if($this->profiling) {
            $this->profilingEnd("imin");
        }
        return $R;
    }

    /**
    *    ret := max X(i)
    */
    public function max(
        NDArray $X,
        NDArray $R=null,
        object $events=null) : mixed
    {
        if($this->profiling) {
            $this->profilingStart("max");
        }
        if($R==null) {
            $R = $this->alloc([],dtype:$X->dtype(),flags:OpenCL::CL_MEM_READ_WRITE);
        }
        // *** CAUTION ****
        // Index result is 32bit
        $IR = $this->alloc([],dtype:NDArray::int32,flags:OpenCL::CL_MEM_READ_WRITE);
        $N = $X->size();
        $IRR = $IR->buffer();
        $offIR = $IR->offset();
        $XX = $X->buffer();
        $offX = $X->offset();
        $imaxEvents = $this->newEventList();
        if($this->isFloat($X)) {
            $this->math->imax($N,$IRR,$offIR,$XX,$offX,1,$this->queue,$imaxEvents);
        } else {
            $this->openclmath->reduceArgMax(
                1,          // $m
                $N,         // $n
                1,          // $k
                $XX, $offX,
                $IRR, $offIR,
                $imaxEvents
            );
        }

        $RR = $R->buffer();
        $offR = $R->offset();
        $this->openclmath->reduceGather(false,false,
            1,1,$N,$IRR,$offIR,$XX,$offX,$RR,$offR,$events,$imaxEvents);
        if($this->blocking) {
            $this->finish();
        }
        if($this->profiling) {
            $this->profilingEnd("max");
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
        object $events=null) : mixed
    {
        if($this->profiling) {
            $this->profilingStart("min");
        }
        if($R==null) {
            $R = $this->alloc([],dtype:$X->dtype(),flags:OpenCL::CL_MEM_READ_WRITE);
        }
        // *** CAUTION ****
        // Index result is 32bit
        $IR = $this->alloc([],dtype:NDArray::int32,flags:OpenCL::CL_MEM_READ_WRITE);
        $N = $X->size();
        $IRR = $IR->buffer();
        $offIR = $IR->offset();
        $XX = $X->buffer();
        $offX = $X->offset();
        $imaxEvents = $this->newEventList();
        if($this->isFloat($X)) {
            $this->math->imin($N,$IRR,$offIR,$XX,$offX,1,$this->queue,$imaxEvents);
        } else {
            $this->openclmath->imin($N,$IRR,$offIR,$XX,$offX,1,$imaxEvents);
            //throw new InvalidArgumentException("Unsuppored data type. Only float types are supported.");
        }

        $RR = $R->buffer();
        $offR = $R->offset();
        $this->openclmath->reduceGather(false,false,
            1,1,$N,$IRR,$offIR,$XX,$offX,$RR,$offR,$events,$imaxEvents);
        if($this->blocking) {
            $this->finish();
        }
        if($this->profiling) {
            $this->profilingEnd("min");
        }
        if($this->scalarNumeric) {
            return $R->toArray();
        }
        return $R;
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
        if($this->profiling) {
            $this->profilingStart("increment");
        }
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
        if($this->profiling) {
            $this->profilingEnd("increment");
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
        if($this->profiling) {
            $this->profilingStart("reciprocal");
        }
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
        if($this->profiling) {
            $this->profilingEnd("reciprocal");
        }
        return $X;
    }

    /**
     * @return array<mixed>
     */
    protected function calcBroadcastFormat(NDArray $A,int|float|NDArray $X) : array
    {
        if(is_numeric($X)) {
            $X = new NDArrayPhp($X,dtype:$A->dtype(),service:$this->service);
            $X = $this->array($X);
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
        int|float|NDArray $X,
        object $events=null,
        object $waitEvents=null
        ) : NDArray
    {
        if($this->profiling) {
            $this->profilingStart("maximum");
        }
        [$m,$n,$dmy,$X] = $this->calcBroadcastFormat($A,$X);
        $AA   = $A->buffer();
        $offA = $A->offset();
        $XX = $X->buffer();
        $offX = $X->offset();

        $this->openclmath->maximum(
            $m,
            $n,
            $AA,$offA,$n,
            $XX,$offX,1,
            $events,$waitEvents
        );

        if($this->blocking) {
            $this->finish();
        }
        if($this->profiling) {
            $this->profilingEnd("maximum");
        }
        return $A;
    }

    /**
     *     A[m,n] := A[m,n] (A[m,n] <  X[n])
     *     A[m,n] := X[n]   (A[m,n] >= X[n])
     */
    public function minimum(
        NDArray $A,
        int|float|NDArray $X,
        object $events=null,
        object $waitEvents=null
        ) : NDArray
    {
        if($this->profiling) {
            $this->profilingStart("minimum");
        }
        [$m,$n,$dmy,$X] = $this->calcBroadcastFormat($A,$X);
        $AA   = $A->buffer();
        $offA = $A->offset();
        $XX = $X->buffer();
        $offX = $X->offset();

        $this->openclmath->minimum(
            $m,
            $n,
            $AA,$offA,$n,
            $XX,$offX,1,
            $events,$waitEvents
        );

        if($this->blocking) {
            $this->finish();
        }
        if($this->profiling) {
            $this->profilingEnd("minimum");
        }
        return $A;
    }

    /**
     *     A[m,n] := 1 (A[m,n] >  X[n])
     *     A[m,n] := 0 (A[m,n] <= X[n])
     */
    public function greater(
        NDArray $A,
        int|float|NDArray $X,
        object $events=null,
        object $waitEvents=null
        ) : NDArray
    {
        if($this->profiling) {
            $this->profilingStart("greater");
        }
        [$m,$n,$dmy,$X] = $this->calcBroadcastFormat($A,$X);

        $AA   = $A->buffer();
        $offA = $A->offset();
        $XX = $X->buffer();
        $offX = $X->offset();

        $this->openclmath->greater(
            $m,
            $n,
            $AA,$offA,$n,
            $XX,$offX,1,
            $events,$waitEvents
        );

        if($this->blocking) {
            $this->finish();
        }
        if($this->profiling) {
            $this->profilingEnd("greater");
        }
        return $A;
    }

    /**
     *     A[m,n] := 1 (A[m,n] >= X[n])
     *     A[m,n] := 0 (A[m,n] <  X[n])
     */
    public function greaterEqual(
        NDArray $A,
        int|float|NDArray $X,
        object $events=null,
        object $waitEvents=null
        ) : NDArray
    {
        if($this->profiling) {
            $this->profilingStart("greaterEqual");
        }
        [$m,$n,$dmy,$X] = $this->calcBroadcastFormat($A,$X);

        $AA   = $A->buffer();
        $offA = $A->offset();
        $XX = $X->buffer();
        $offX = $X->offset();

        $this->openclmath->greaterEqual(
            $m,
            $n,
            $AA,$offA,$n,
            $XX,$offX,1,
            $events,$waitEvents
        );

        if($this->blocking) {
            $this->finish();
        }
        if($this->profiling) {
            $this->profilingEnd("greaterEqual");
        }
        return $A;
    }

    /**
     *     A[m,n] := 1 (A[m,n] <  X[n])
     *     A[m,n] := 0 (A[m,n] >= X[n])
     */
    public function less(
        NDArray $A,
        int|float|NDArray $X,
        object $events=null,
        object $waitEvents=null
        ) : NDArray
    {
        if($this->profiling) {
            $this->profilingStart("less");
        }
        [$m,$n,$dmy,$X] = $this->calcBroadcastFormat($A,$X);

        $AA   = $A->buffer();
        $offA = $A->offset();
        $XX = $X->buffer();
        $offX = $X->offset();

        $this->openclmath->less(
            $m,
            $n,
            $AA,$offA,$n,
            $XX,$offX,1,
            $events,$waitEvents
        );

        if($this->blocking) {
            $this->finish();
        }
        if($this->profiling) {
            $this->profilingEnd("less");
        }
        return $A;
    }

    /**
     *     A[m,n] := 1 (A[m,n] <= X[n])
     *     A[m,n] := 0 (A[m,n] >  X[n])
     */
    public function lessEqual(
        NDArray $A,
        int|float|NDArray $X,
        object $events=null,
        object $waitEvents=null
        ) : NDArray
    {
        if($this->profiling) {
            $this->profilingStart("lessEqual");
        }
        [$m,$n,$dmy,$X] = $this->calcBroadcastFormat($A,$X);

        $AA   = $A->buffer();
        $offA = $A->offset();
        $XX = $X->buffer();
        $offX = $X->offset();

        $this->openclmath->lessEqual(
            $m,
            $n,
            $AA,$offA,$n,
            $XX,$offX,1,
            $events,$waitEvents
        );

        if($this->blocking) {
            $this->finish();
        }
        if($this->profiling) {
            $this->profilingEnd("lessEqual");
        }
        return $A;
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
        if($this->profiling) {
            $this->profilingStart("multiply");
        }
        if($trans===null) {
            $trans = false;
        }
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
        if($this->profiling) {
            $this->profilingEnd("multiply");
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
        if($this->profiling) {
            $this->profilingStart("add");
        }
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
        if($this->profiling) {
            $this->profilingEnd("add");
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
        if($this->profiling) {
            $this->profilingStart("square");
        }
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
        if($this->profiling) {
            $this->profilingEnd("square");
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
        if($this->profiling) {
            $this->profilingStart("sqrt");
        }
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
        if($this->profiling) {
            $this->profilingEnd("sqrt");
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
        if($this->profiling) {
            $this->profilingStart("rsqrt");
        }
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
        if($this->profiling) {
            $this->profilingEnd("rsqrt");
        }
        return $X;
    }

    /**
     *     A(m,n) := A(m,n) ** alpha(n)
     */
    public function pow(
        NDArray $A,
        float|NDArray $alpha,
        bool $trans=null,
        object $events=null,
        object $waitEvents=null
        ) : NDArray
    {
        if($this->profiling) {
            $this->profilingStart("pow");
        }
        if($trans===null) {
            $trans = false;
        }
        $shapeA = $A->shape();
        if(is_numeric($alpha)) {
            $alpha = new NDArrayPhp($alpha,dtype:$A->dtype(),service:$this->service);
            $alpha = $this->array($alpha);
        }
        $shapeX = $alpha->shape();
        if(count($shapeX)==0) {
            $trans = false;
            $shapeA = [(int)array_product($shapeA),1];
            $shapeX = [1];
        }

        if($trans) {
            $shapeA = array_reverse($shapeA);
        }
        while(true) {
            $xd = array_pop($shapeX);
            if($xd===null)
                break;
            $ad = array_pop($shapeA);
            if($xd!==$ad) {
                $shapeA = $trans ? array_reverse($A->shape()) : $A->shape();
                throw new InvalidArgumentException('Unmatch dimension size for broadcast.: '.
                    '['.implode(',',$shapeX).'] => ['.implode(',',$shapeA).']');
            }
        }
        $n = $alpha->size();
        $XX = $alpha->buffer();
        $offX = $alpha->offset();
        $m = $A->size()/$n;
        $AA = $A->buffer();
        $offA = $A->offset();
        if($trans) {
            [$m,$n] = [$n,$m];
        }

        $this->openclmath->pow(
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
        if($this->profiling) {
            $this->profilingEnd("pow");
        }
        return $A;
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
        if($this->profiling) {
            $this->profilingStart("exp");
        }
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
        if($this->profiling) {
            $this->profilingEnd("exp");
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
        if($this->profiling) {
            $this->profilingStart("log");
        }
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
        if($this->profiling) {
            $this->profilingEnd("log");
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
        if($this->profiling) {
            $this->profilingStart("tanh");
        }
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
        if($this->profiling) {
            $this->profilingEnd("tanh");
        }
        return $X;
    }

    /**
     *     X := sin(X)
     */
    public function sin(
        NDArray $X,
        object $events=null,
        object $waitEvents=null
        ) : NDArray
    {
        if($this->profiling) {
            $this->profilingStart("sin");
        }
        $n = $X->size();
        $XX = $X->buffer();
        $offX = $X->offset();

        $this->openclmath->sin(
            $n,
            $XX,$offX,1,
            $events,$waitEvents
        );

        if($this->blocking) {
            $this->finish();
        }
        if($this->profiling) {
            $this->profilingEnd("sin");
        }
        return $X;
    }

    /**
     *     X := cos(X)
     */
    public function cos(
        NDArray $X,
        object $events=null,
        object $waitEvents=null
        ) : NDArray
    {
        if($this->profiling) {
            $this->profilingStart("cos");
        }
        $n = $X->size();
        $XX = $X->buffer();
        $offX = $X->offset();

        $this->openclmath->cos(
            $n,
            $XX,$offX,1,
            $events,$waitEvents
        );

        if($this->blocking) {
            $this->finish();
        }
        if($this->profiling) {
            $this->profilingEnd("cos");
        }
        return $X;
    }

    /**
     *     X := tan(X)
     */
    public function tan(
        NDArray $X,
        object $events=null,
        object $waitEvents=null
        ) : NDArray
    {
        if($this->profiling) {
            $this->profilingStart("tan");
        }
        $n = $X->size();
        $XX = $X->buffer();
        $offX = $X->offset();

        $this->openclmath->tan(
            $n,
            $XX,$offX,1,
            $events,$waitEvents
        );

        if($this->blocking) {
            $this->finish();
        }
        if($this->profiling) {
            $this->profilingEnd("tan");
        }
        return $X;
    }

    /**
     *     Y(i) := 1 (X(i) = Y(i))
     *     Y(i) := 0 (X(i) != Y(i))
     */
    public function equal(
        NDArray $X,
        NDArray $Y,
        object $events=null,
        object $waitEvents=null
        ) : NDArray
    {
        if($this->profiling) {
            $this->profilingStart("equal");
        }
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
        if($this->profiling) {
            $this->profilingEnd("equal");
        }
        return $Y;
    }

    /**
     *     Y(i) := 1 (X(i) != Y(i))
     *     Y(i) := 0 (X(i) = Y(i))
     */
    public function notEqual(
        NDArray $X,
        NDArray $Y,
        object $events=null,
        object $waitEvents=null
        ) : NDArray
    {
        if($this->profiling) {
            $this->profilingStart("notEqual");
        }
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
        $this->openclmath->notEqual(
            $N,
            $XX,$offX,$incX,
            $YY,$offY,$incY,
            $events,$waitEvents
        );

        if($this->blocking) {
            $this->finish();
        }
        if($this->profiling) {
            $this->profilingEnd("notEqual");
        }
        return $Y;
    }

    /**
     *     X(i) := 1 (X(i)  = 0)
     *     X(i) := 0 (X(i) != 0)
     */
    public function not(
        NDArray $X,
        object $events=null,
        object $waitEvents=null
        ) : NDArray
    {
        if($this->profiling) {
            $this->profilingStart("notEqual");
        }
        $N = $X->size();
        $XX = $X->buffer();
        $offX = $X->offset();
        $incX = 1;
        $this->openclmath->not(
            $N,
            $XX,
            $offX,
            $incX,
            $events,$waitEvents
        );
        if($this->blocking) {
            $this->finish();
        }
        if($this->profiling) {
            $this->profilingEnd("not");
        }
        return $X;
    }

    /**
     *      input:  X
     *      output: A
     *      A(m,n) := X(n)
     */
    public function duplicate(
        NDArray $input,
        int $repeats=null,
        bool $trans=null,
        NDArray $output=null,
        object $events=null,
        object $waitEvents=null
        ) : NDArray
    {
        if($this->profiling) {
            $this->profilingStart("duplicate");
        }
        if($trans===null)
            $trans = false;
        if($output===null) {
            if($repeats===null)
                $repeats = 1;
            if(!$trans) {
                $output = $this->alloc(array_merge([$repeats],$input->shape()),dtype:$input->dtype());
            } else {
                $output = $this->alloc(array_merge($input->shape(),[$repeats]),dtype:$input->dtype());
            }
        } else {
            $shapeX = $input->shape();
            $shapeA = $output->shape();
            if($trans)
                $shapeA = array_reverse($shapeA);
            while(true) {
                $xd = array_pop($shapeX);
                if($xd===null)
                    break;
                $ad = array_pop($shapeA);
                if($xd!==$ad)
                    throw new InvalidArgumentException('Unmatch dimension size for broadcast.: '.
                        '['.implode(',',$input->shape()).'] => ['.implode(',',$output->shape()).']');
            }
        }

        $n = $input->size();
        $XX = $input->buffer();
        $offX = $input->offset();
        $m = $output->size()/$n;
        $AA = $output->buffer();
        $offA = $output->offset();
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
        if($this->profiling) {
            $this->profilingEnd("duplicate");
        }
        return $output;
    }

    /**
     *      input:  A,X
     *      output: B
     *      B(m,n,k) := A(m,X(m,n),k)
     */
    public function doGather(
        bool $scatterAdd,
        NDArray $A,
        NDArray $X,
        int $axis=null,
        NDArray $output=null,
        int $dtype=null,
        object $events=null,object $waitEvents=null
        ) : NDArray
    {
//echo "shapeX=[".implode(',',$X->shape())."],shapeA=[".implode(',',$A->shape())."]\n";
        if($X->dtype()!=NDArray::int32 && $X->dtype()!=NDArray::uint32) {
            $waitPrev = $waitEvents;
            $waitEvents = $this->newEventList();
            $X = $this->astype($X,NDArray::int32,null,$waitEvents,$waitPrev);
        }
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
        if($output==null) {
            $output = $this->alloc($outputShape,dtype:$dtype);
            $waitPrev = $waitEvents;
            $waitEvents = $this->newEventList();
            $this->zeros($output,$waitEvents,$waitPrev);
        } else {
            if($output->shape()!=$outputShape) {
                throw new InvalidArgumentException("Unmatch output shape of dimension: ".
                                            $this->printableShapes([$outputShape,$output]));
            }
        }

        $AA = $A->buffer();
        $offA = $A->offset();
        $XX = $X->buffer();
        $offX = $X->offset();
        $BB = $output->buffer();
        $offB = $output->offset();

        if($scatterAdd) {
            $reverse=true;
            $addMode=true;
        } else {
            $reverse=false;
            $addMode=false;
        }
        if($reductionDims) {
            $this->openclmath->reduceGather(
                $reverse,
                $addMode,
                $m,
                $n,
                $numClass,
                $XX,$offX,
                $AA,$offA,
                $BB,$offB,
                $events, $waitEvents
            );
        } else {
            if($scatterAdd) {
                $this->openclmath->scatterAdd(
                    $n,
                    $k,
                    $numClass,
                    $XX,$offX,
                    $AA,$offA,
                    $BB,$offB,
                    $events, $waitEvents
                );
            } else {
                $this->openclmath->gather(
                    $reverse,
                    $addMode,
                    $n,
                    $k,
                    $numClass,
                    $XX,$offX,
                    $AA,$offA,
                    $BB,$offB,
                    $events, $waitEvents
                );
            }
        }
        if($this->blocking) {
            $this->finish();
        }
        return $output;
    }

    /**
    *      B(m,n,k) := A(m,X(m,n),k)
    */
    public function gather(
        NDArray $A,
        NDArray $X,
        int $axis=null,
        NDArray $output=null,
        int $dtype=null,
        object $events=null,object $waitEvents=null
        ) : NDArray
    {
        if($this->profiling) {
            $this->profilingStart("gather");
        }
        $output = $this->doGather(
            $scatterAdd=false,
            $A,
            $X,
            $axis,
            $output,
            $dtype,
            $events, $waitEvents
        );
        if($this->profiling) {
            $this->profilingEnd("gather");
        }
        return $output;
    }

    /**
     *      input:  A,X
     *      output: B
     *      B(m,X(m,n),k) += A(m,n,k)
     */
    public function scatterAdd(
        NDArray $X,
        NDArray $A,
        NDArray $output,
        int $axis=null,
        int $dtype=null,
        object $events=null,object $waitEvents=null
        ) : NDArray
    {
        if($this->profiling) {
            $this->profilingStart("scatterAdd");
        }
        $this->doGather(
            $scatterAdd=true,
            $output,
            $X,
            $axis,
            $A,
            $dtype,
            $events, $waitEvents
        );
        if($this->profiling) {
            $this->profilingEnd("scatterAdd");
        }
        return $output;
    }

    /**
     *      input:  A,X
     *      output: B
     *      B(m,X(m,n),k) := A(m,n,k)
     */
    public function scatter(
        NDArray $X,
        NDArray $A,
        int $numClass,
        int $axis=null,
        NDArray $output=null,
        int $dtype=null,
        object $events=null,object $waitEvents=null
        ) : NDArray
    {
        if($this->profiling) {
            $this->profilingStart("scatter");
        }
//echo "shapeX=[".implode(',',$X->shape())."],shapeA=[".implode(',',$A->shape())."]\n";
//echo "axis=$axis,numClass=$numClass\n";
        if($X->dtype()!=NDArray::int32 && $X->dtype()!=NDArray::uint32) {
            $waitPrev = $waitEvents;
            $waitEvents = $this->newEventList();
            $X = $this->astype($X,NDArray::int32,null,$waitEvents,$waitPrev);
        }
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
                throw new InvalidArgumentException('Unmatch Shape:'.
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
        if($output==null) {
            $output = $this->alloc($outputShape,dtype:$dtype);
            $waitPrev = $waitEvents;
            $waitEvents = $this->newEventList();
            $this->zeros($output,$waitEvents,$waitPrev);
        } else {
            if($output->shape()!=$outputShape) {
                $shapeError = '('.implode(',',$A->shape()).'),('.implode(',',$output->shape()).')';
                throw new InvalidArgumentException("Unmatch shape of dimension: ".$shapeError);
            }
        }

        $AA = $A->buffer();
        $offA = $A->offset();
        $XX = $X->buffer();
        $offX = $X->offset();
        $BB = $output->buffer();
        $offB = $output->offset();

        if($expandDims) {
            $this->openclmath->reduceGather(
                $reverse=true,
                $addMode=false,
                $m,
                $n,
                $numClass,
                $XX,$offX,
                $BB,$offB,
                $AA,$offA,
                $events, $waitEvents
            );
        } else {
            $this->openclmath->gather(
                $reverse=true,
                $addMode=false,
                $n,
                $k,
                $numClass,
                $XX,$offX,
                $BB,$offB,
                $AA,$offA,
                $events, $waitEvents
            );
        }
        if($this->blocking) {
            $this->finish();
        }
        if($this->profiling) {
            $this->profilingEnd("scatter");
        }
        return $output;
    }

    public function onehot(
        NDArray $X,
        int $numClass,
        float $alpha=null,
        NDArray $output=null,
        object $events=null,object $waitEvents=null
        ) : NDArray
    {
        if($this->profiling) {
            $this->profilingStart("onehot");
        }
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
        if($output===null) {
            $addMode = false;
            $output = $this->alloc([$sizeX,$numClass],dtype:$this->defaultFloatType);
            $waitPrev = $waitEvents;
            $waitEvents = $this->newEventList();
            $this->zeros($output,$waitEvents,$waitPrev);
        }
        if($output->ndim()!=2) {
            throw new InvalidArgumentException('"Y" must be 2D-NDArray.');
        }
        [$m,$n] = $output->shape();
        if($m!=$sizeX || $n!=$numClass) {
            $shapeError = '('.implode(',',$X->shape()).'),('.implode(',',$output->shape()).')';
            throw new InvalidArgumentException('Unmatch shape of dimension "X" and "Y" and "numClass": '.$shapeError);
        }
        if($alpha===null) {
            $alpha = 1.0;
        }
        $XX = $X->buffer();
        $offX = $X->offset();
        $YY = $output->buffer();
        $offY = $output->offset();
        $ldY = $n;

        $this->openclmath->onehot(
            $m,
            $n,
            $alpha,
            $XX,$offX,1,
            $YY,$offY,$ldY,
            $addMode,
            $events,$waitEvents
        );

        if($this->blocking) {
            $this->finish();
        }
        if($this->profiling) {
            $this->profilingEnd("onehot");
        }
        return $output;
    }

    /**
     *     X := softmax(X)
     */
    public function softmax(
        NDArray $X,
        object $events=null,object $waitEvents=null
        ) : NDArray
    {
        if($this->profiling) {
            $this->profilingStart("softmax");
        }
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
        if($this->profiling) {
            $this->profilingEnd("softmax");
        }
        return $X;
    }

    /**
     *    X(m) := sum( A(m,n) )
     */
    public function reduceSum(//reduceSumEx
        NDArray $input,
        int $axis=null,
        bool $keepdims=null,
        NDArray $output=null,
        int $dtype=null,
        object $events=null,object $waitEvents=null
        ) : NDArray
    {
        if($this->profiling) {
            $this->profilingStart("reduceSum");
        }
        $ndim = $input->ndim();
        $origAxis = $axis;
        if($axis===null) {
            $axis = 0;
        }
        if($axis<0) {
            $axis = $ndim+$axis;
        }
        if($axis<0 || $axis>$ndim-1) {
            $origAxis = $origAxis ?? 'null';
            throw new InvalidArgumentException("Invalid axis: ".$origAxis);
        }
        $postfixShape = $input->shape();
        $prefixShape = [];
        for($i=0;$i<$axis;$i++) {
            $prefixShape[] = array_shift($postfixShape);
        }
        $n = array_shift($postfixShape);
        $m = array_product($prefixShape);
        $k = array_product($postfixShape);
        if($keepdims) {
            $outputShape = array_merge($prefixShape,[1],$postfixShape);
        } else {
            $outputShape = array_merge($prefixShape,$postfixShape);
        }
        if($dtype===null) {
            $dtype = $input->dtype();
        }
        if($output==null) {
            $output = $this->alloc($outputShape,dtype:$dtype);
        } else {
            if($output->shape()!=$outputShape) {
                $shapeError = '('.implode(',',$input->shape()).'),('.implode(',',$output->shape()).')';
                throw new InvalidArgumentException("Unmatch shape of dimension: ".$shapeError);
            }
        }

        $AA = $input->buffer();
        $offA = $input->offset();
        $BB = $output->buffer();
        $offB = $output->offset();

        $this->openclmath->reduceSum(
            $m,
            $n,
            $k,
            $AA,$offA,
            $BB,$offB,
            $events,$waitEvents);

        if($this->blocking) {
            $this->finish();
        }
        if($this->profiling) {
            $this->profilingEnd("reduceSum");
        }
        return $output;
    }

    /**
     *    X(m) := max( A(m,n) )
     */
    public function reduceMax(//reduceMaxEx
        NDArray $input,
        int $axis=null,
        bool $keepdims=null,
        NDArray $output=null,
        int $dtype=null,
        object $events=null,object $waitEvents=null
        ) : NDArray
    {
        if($this->profiling) {
            $this->profilingStart("reduceMax");
        }
        $ndim = $input->ndim();
        $origAxis = $axis;
        if($axis===null) {
            $axis = 0;
        }
        if($axis<0) {
            $axis = $ndim+$axis;
        }
        if($axis<0 || $axis>$ndim-1) {
            $origAxis = $origAxis ?? 'null';
            throw new InvalidArgumentException("Invalid axis: ".$origAxis);
        }
        $postfixShape = $input->shape();
        $prefixShape = [];
        for($i=0;$i<$axis;$i++) {
            $prefixShape[] = array_shift($postfixShape);
        }
        $n = array_shift($postfixShape);
        $m = array_product($prefixShape);
        $k = array_product($postfixShape);
        if($keepdims) {
            $outputShape = array_merge($prefixShape,[1],$postfixShape);
        } else {
            $outputShape = array_merge($prefixShape,$postfixShape);
        }
        if($dtype===null) {
            $dtype = $input->dtype();
        }
        if($output==null) {
            $output = $this->alloc($outputShape,dtype:$dtype);
        } else {
            if($output->shape()!=$outputShape) {
                $shapeError = '('.implode(',',$input->shape()).'),('.implode(',',$output->shape()).')';
                throw new InvalidArgumentException("Unmatch shape of dimension: ".$shapeError);
            }
        }

        $AA = $input->buffer();
        $offA = $input->offset();
        $BB = $output->buffer();
        $offB = $output->offset();

        $this->openclmath->reduceMax(
            $m,
            $n,
            $k,
            $AA,$offA,
            $BB,$offB,
            $events,$waitEvents);

        if($this->blocking) {
            $this->finish();
        }
        if($this->profiling) {
            $this->profilingEnd("reduceMax");
        }
        return $output;
    }

    /**
     *    X(m) := imax( A(m,n) )
     */
    public function reduceArgMax(//reduceArgMaxEx
        NDArray $input,
        int $axis=null,
        bool $keepdims=null,
        NDArray $output=null,
        int $dtype=null,
        object $events=null,object $waitEvents=null
        ) : NDArray
    {
        if($this->profiling) {
            $this->profilingStart("reduceArgMax");
        }
        $ndim = $input->ndim();
        $origAxis = $axis;
        if($axis===null) {
            $axis = 0;
        }
        if($axis<0) {
            $axis = $ndim+$axis;
        }
        if($axis<0 || $axis>$ndim-1) {
            $origAxis = $origAxis ?? 'null';
            throw new InvalidArgumentException("Invalid axis: ".$origAxis);
        }
        $postfixShape = $input->shape();
        $prefixShape = [];
        for($i=0;$i<$axis;$i++) {
            $prefixShape[] = array_shift($postfixShape);
        }
        $n = array_shift($postfixShape);
        $m = array_product($prefixShape);
        $k = array_product($postfixShape);
        if($keepdims) {
            $outputShape = array_merge($prefixShape,[1],$postfixShape);
        } else {
            $outputShape = array_merge($prefixShape,$postfixShape);
        }
        if($dtype===null) {
            $dtype = NDArray::uint32;
        }
        if($output==null) {
            $output = $this->alloc($outputShape,dtype:$dtype);
        } else {
            if($output->shape()!=$outputShape) {
                $shapeError = '('.implode(',',$input->shape()).'),('.implode(',',$output->shape()).')';
                throw new InvalidArgumentException("Unmatch shape of dimension: ".$shapeError);
            }
        }

        $AA = $input->buffer();
        $offA = $input->offset();
        $BB = $output->buffer();
        $offB = $output->offset();

        $this->openclmath->reduceArgMax(
            $m,
            $n,
            $k,
            $AA,$offA,
            $BB,$offB,
            $events,$waitEvents);

        if($this->blocking) {
            $this->finish();
        }
        if($this->profiling) {
            $this->profilingEnd("reduceArgMax");
        }
        return $output;
    }

    /**
     *    X(m) := sum( A(m,n)/n )
     */
    public function reduceMean(
        NDArray $input,
        int $axis=null,
        bool $keepdims=null,
        NDArray $output=null,
        int $dtype=null,
        object $events=null,object $waitEvents=null
        ) : NDArray
    {
        $waitPrev = $waitEvents;
        $waitEvents = $this->newEventList();
        if($axis===null) {
            $axis = 0;
        }
        $output = $this->reduceSum(
            $input,axis:$axis,keepdims:$keepdims,output:$output,dtype:$dtype,
            events:$waitEvents,waitEvents:$waitPrev
        );
        $ndim = $input->ndim();
        if($axis<0) {
            $axis = $ndim+$axis;
        }
        if($ndim<=$axis) {
            throw new InvalidArgumentException('axis must be less then num of dimension');
        }
        $shapeA = $input->shape();
        $rows = $shapeA[$axis];
        $waitEvents->wait();
        $this->scal(
            1/$rows,$output,
            $events
        );
        if($this->blocking) {
            $this->finish();
        }
        return $output;
    }

    /**
     * @param array<int> $filterSize
     * @param array<int> $strides
     * @param array<int> $dilation_rate
     * @param bool|array<int> $padding
     */
    public function im2col2dclblast(
        bool $reverse,
        int $kernel_mode,
        NDArray $images,
        array $filterSize=null,
        array $strides=null,
        bool|array $padding=null,
        array $dilation_rate=null,
        //bool $channels_first=null,
        //bool $cols_channels_first=null,
        NDArray $cols=null,
        object $events=null
        ) : NDArray
    {
        if($this->openclmath->hasDiv5Bug()) {
            throw new LogicException('Not support function on this device.');
        }

        if($this->profiling) {
            $this->profilingStart("im2col2dclblast");
        }
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
        $out_h = intdiv(($im_h-($filter_h-1)*$dilation_h-1),$stride_h)+1;
        $out_w = intdiv(($im_w-($filter_w-1)*$dilation_w-1),$stride_w)+1;
        if($out_h<=0 && $out_w<=0) {
            throw new InvalidArgumentException('Invalid shape or parameters.');
        }
        if(is_bool($padding)) {
            if($padding) {
                $out_h = $im_h;
                $out_w = $im_w;
                $padding_h = (int)(($im_h-1)*$stride_h-$im_h+($filter_h-1)*$dilation_h+1);
                if($padding_h%2) {
                    $out_h++;
                }
                $padding_h = $padding_h ? intdiv($padding_h,2) : 0;
                $padding_w = (int)(($im_w-1)*$stride_w-$im_w+($filter_w-1)*$dilation_w+1);
                if($padding_w%2) {
                    $out_w++;
                }
                $padding_w = $padding_w ? intdiv($padding_w,2) : 0;
            } else {
                $padding_h = 0;
                $padding_w = 0;
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
                ],dtype:$images->dtype());
                $zerosEvents = $this->newEventList();
                $this->zeros($cols,events:$zerosEvents);
                $zerosEvents->wait();
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
        $image_step = $channels*$im_h*$im_w;
        $col_step   = $channels*$filter_h*$filter_w*$out_h*$out_w;
        if(!$reverse) {
            for($i=0;$i<$batches;$i++,$images_offset+=$image_step,$col_offset+=$col_step) {
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
            }
            $result = $cols;
        }  else {
            for($i=0;$i<$batches;$i++,$images_offset+=$image_step,$col_offset+=$col_step) {
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
            }
            $result = $images;
        }
        if($this->blocking) {
            $this->finish();
        }
        if($this->profiling) {
            $this->profilingEnd("im2col2dclblast");
        }
        return $result;
    }

    /**
     * @param array<int> $filterSize
     * @param array<int> $strides
     * @param array<int> $dilation_rate
     */
    public function im2col(
        NDArray $images,
        array $filterSize=null,
        array $strides=null,
        bool $padding=null,
        bool $channels_first=null,
        array $dilation_rate=null,
        bool $cols_channels_first=null,
        NDArray $cols=null,
        object $events=null,object $waitEvents=null
        ) : NDArray
    {
        if($this->profiling) {
            $this->profilingStart("im2col");
        }
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
        if($this->profiling) {
            $this->profilingEnd("im2col");
        }
        return $cols;
    }

    /**
     * @param array<int> $filterSize
     * @param array<int> $strides
     * @param array<int> $dilation_rate
     */
    public function col2im(
        NDArray $cols,
        NDArray $images,
        array $filterSize=null,
        array $strides=null,
        bool $padding=null,
        bool $channels_first=null,
        array $dilation_rate=null,
        bool $cols_channels_first=null,
        object $events=null,object $waitEvents=null
        ) : NDArray
    {
        if($this->profiling) {
            $this->profilingStart("col2im");
        }
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
        if($this->profiling) {
            $this->profilingEnd("col2im");
        }
        return $images;
    }

    /**
     * @param array<int> $filterSize
     * @param array<int> $strides
     * @param array<int> $dilation_rate
     */
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
        object $events=null,object $waitEvents=null
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
                $out_w = intdiv(($im_w-($kernel_w-1)*$dilation_w-1),$stride_w)+1;
            }
            if($out_w<=0) {
                throw new InvalidArgumentException('Invalid shape or paramaters.');
            }
            if($cols_channels_first) {
                $cols = $this->alloc([
                    $batches,$out_w,
                    $channels,$kernel_w
                ],dtype:$images->dtype());
                $waitPrev = $waitEvents;
                $waitEvents = $this->newEventList();
                $this->zeros($cols,$waitEvents,$waitPrev);
            } else {
                $cols = $this->alloc([
                    $batches,$out_w,
                    $kernel_w,$channels
                ],dtype:$images->dtype());
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

    /**
     * @param array<int> $filterSize
     * @param array<int> $strides
     * @param array<int> $dilation_rate
     */
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
        object $events=null,object $waitEvents=null
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
                $out_h = intdiv(($im_h-($kernel_h-1)*$dilation_h-1),$stride_h)+1;
                $out_w = intdiv(($im_w-($kernel_w-1)*$dilation_w-1),$stride_w)+1;
            }
            if($out_h<=0 && $out_w<=0) {
                throw new InvalidArgumentException('Invalid shape or parameters.');
            }
            if($cols_channels_first) {
                $cols = $this->alloc([
                    $batches,$out_h,$out_w,
                    $channels,$kernel_h,$kernel_w
                ],dtype:$images->dtype());
                $waitPrev = $waitEvents;
                $waitEvents = $this->newEventList();
                $this->zeros($cols,$waitEvents,$waitPrev);
            } else {
                $cols = $this->alloc([
                    $batches,$out_h,$out_w,
                    $kernel_h,$kernel_w,$channels
                ],dtype:$images->dtype());
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

    /**
     * @param array<int> $filterSize
     * @param array<int> $strides
     * @param array<int> $dilation_rate
     */
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
        object $events=null,object $waitEvents=null
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
                $out_d = intdiv(($im_d-($kernel_d-1)*$dilation_d-1),$stride_d)+1;
                $out_h = intdiv(($im_h-($kernel_h-1)*$dilation_h-1),$stride_h)+1;
                $out_w = intdiv(($im_w-($kernel_w-1)*$dilation_w-1),$stride_w)+1;
            }
            if($out_d<=0 || $out_h<=0 || $out_w<=0) {
                throw new InvalidArgumentException('Invalid shape or paramaters.');
            }
            if($cols_channels_first) {
                $cols = $this->alloc([
                    $batches,$out_d,$out_h,$out_w,
                    $channels,$kernel_d,$kernel_h,$kernel_w
                ],dtype:$images->dtype());
                $waitPrev = $waitEvents;
                $waitEvents = $this->newEventList();
                $this->zeros($cols,$waitEvents,$waitPrev);
            } else {
                $cols = $this->alloc([
                    $batches,$out_d,$out_h,$out_w,
                    $kernel_d,$kernel_h,$kernel_w,$channels
                ],dtype:$images->dtype());
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
     * @param array<int> $shape
     */
    public function randomUniform(
        array $shape,
        int|float $low,
        int|float $high,
        int $dtype=null,
        int $seed=null,
        NDArray $X=null,
        object $events=null, object $waitEvents=null
        ) : NDArray
    {
        if($this->profiling) {
            $this->profilingStart("randomUniform");
        }
        if($dtype!==null&&$X!==null) {
            if ($X->dtype()!=$dtype) {
                throw new InvalidArgumentException('Unmatch dtype and dtype of X');
            }
        }
        if($X===null) {
            $X = $this->alloc($shape,dtype:$dtype);
        } else {
            if ($X->shape()!=$shape) {
                throw new InvalidArgumentException('Unmatch shape and shape of X');
            }
        }
        if($seed===null) {
            $seed = random_int(~PHP_INT_MAX,PHP_INT_MAX);
        }

        $n = $X->size();
        $XX = $X->buffer();
        $offX = $X->offset();

        $this->openclmath->randomUniform(
            $n,
            $XX,$offX,1,
            $low,
            $high,
            $seed,
            $events, $waitEvents
        );

        if($this->blocking) {
            $this->finish();
        }
        if($this->profiling) {
            $this->profilingEnd("randomUniform");
        }
        return $X;
    }

    /**
     * @param array<int> $shape
     */
    public function randomNormal(
        array $shape,
        float $mean,
        float $scale,
        int $dtype=null,
        int $seed=null,
        NDArray $X=null,
        object $events=null, object $waitEvents=null
        ) : NDArray
    {
        if($this->profiling) {
            $this->profilingStart("randomNormal");
        }
        if($dtype!==null&&$X!==null) {
            if ($X->dtype()!=$dtype) {
                throw new InvalidArgumentException('Unmatch dtype and dtype of X');
            }
        }
        if($X===null) {
            $X = $this->alloc($shape,dtype:$dtype);
        } else {
            if ($X->shape()!=$shape) {
                throw new InvalidArgumentException('Unmatch shape and shape of X');
            }
        }
        if($seed===null) {
            $seed = random_int(~PHP_INT_MAX,PHP_INT_MAX);
        }

        $n = $X->size();
        $XX = $X->buffer();
        $offX = $X->offset();

        $this->openclmath->randomNormal(
            $n,
            $XX,$offX,1,
            $mean,
            $scale,
            $seed,
            $events, $waitEvents
        );

        if($this->blocking) {
            $this->finish();
        }
        if($this->profiling) {
            $this->profilingEnd("randomNormal");
        }
        return $X;
    }

    public function randomSequence(
        int $base,
        int $size=null,
        int $seed=null,
        int $dtype=null,
        NDArray $output=null,
        object $events=null,object $waitEvents=null
        ) : NDArray
    {
        if($this->profiling) {
            $this->profilingStart("randomSequence");
        }
        if($size==null) {
            $size = $base;
        }
        if($output==null) {
            $dtype = $dtype ?? NDArray::int32;
            $hostX = $this->allocHost([$base],$dtype);
        } else {
            $dtype = $dtype ?? $output->dtype();
            if($output->dtype()!=$dtype || $output->size()!=$size) {
                throw new InvalidArgumentException("output size must be the same of size and same dtype");
            }
            $hostX = $this->allocHost([$base],$dtype);
        }
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
        $hostX = $hostX[R(0,$size)];
        if($output==null) {
            $output = $this->array($hostX);
        } else {
            throw new InvalidArgumentException("output option is not supported on OpenCL");
        }
        if($this->blocking) {
            $this->finish();
        }
        if($this->profiling) {
            $this->profilingEnd("randomSequence");
        }
        return $output;
    }

    /**
     *  Y = X(n)
     * @param array<int> $begin
     * @param array<int> $size
     */
    public function slice(
        NDArray $input,
        array $begin,
        array $size,
        NDArray $output=null,
        object $events=null,object $waitEvents=null
        ) : NDArray
    {
        if($this->profiling) {
            $this->profilingStart("slice");
        }
        $output = $this->doSlice(
            false,
            $input,
            $begin,
            $size,
            $output,
            $events,$waitEvents
        );
        if($this->blocking) {
            $this->finish();
        }
        if($this->profiling) {
            $this->profilingEnd("slice");
        }
        return $output;
    }

    /**
     * @param array<int> $begin
     * @param array<int> $size
     */
    public function stick(
        NDArray $input,
        NDArray $output,
        array $begin,
        array $size,
        object $events=null,object $waitEvents=null
        ) : NDArray
    {
        if($this->profiling) {
            $this->profilingStart("stick");
        }
        $output = $this->doSlice(
            true,
            $output,
            $begin,
            $size,
            $input,
            $events,$waitEvents
        );
        if($this->blocking) {
            $this->finish();
        }
        if($this->profiling) {
            $this->profilingEnd("stick");
        }
        return $output;
    }

    /**
     * @param array<NDArray> $values
     */
    public function stack(
        array $values,
        int $axis=null,
        object $events=null,object $waitEvents=null
    ) : NDArray
    {
        if($this->profiling) {
            $this->profilingStart("stack");
        }
        if($axis==null){
            $axis=0;
        }
        if($axis==0){
            $m = count($values);
            $shape = $values[0]->shape();
            array_unshift($shape,$m);
            $output = $this->alloc($shape,dtype:$values[0]->dtype());
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
            $output = $this->alloc($shape,dtype:$values[0]->dtype());
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
        } elseif($axis==2){
            $k = count($values);
            $shape = $values[0]->shape();
            $m = array_shift($shape);
            $n = array_shift($shape);
            array_unshift($shape,$k);
            array_unshift($shape,$n);
            array_unshift($shape,$m);
            $output = $this->alloc($shape,dtype:$values[0]->dtype());
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
                    $value,
                    $events,$waitEvents
                );
                $i++;
            }
        } else {
            throw new InvalidArgumentException('unsuppoted axis');
        }
        if($this->blocking) {
            $this->finish();
        }
        if($this->profiling) {
            $this->profilingEnd("stack");
        }
        return $output;
    }

    /**
     * @param array<NDArray> $values
     */
    public function concat(
        array $values,
        int $axis=null,
        object $events=null,object $waitEvents=null
    ) : NDArray
    {
        if($this->profiling) {
            $this->profilingStart("concat");
        }
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
        $shape = [];
        $shapePrefix = [];
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
        $output = $this->alloc($shape,dtype:$values[0]->dtype());
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
        if($this->blocking) {
            $this->finish();
        }
        if($this->profiling) {
            $this->profilingEnd("concat");
        }
        return $output;
    }

    /**
     * @param array<int> $sizeSplits
     * @return array<NDArray>
     */
    public function split(
        NDArray $input, array $sizeSplits, int $axis=null,
        object $events=null,object $waitEvents=null
        ) : array
    {
        if($this->profiling) {
            $this->profilingStart("split");
        }
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
        $outputs = [];
        $dtype = $input->dtype();
        foreach($sizeSplits as $size) {
            $outputs[] = $this->alloc(array_merge($shapePrefix,[$size],$shape),dtype:$dtype);
        }
        $i = 0;
        $outidx = 0;
        foreach($sizeSplits as $size) {
            $this->doSlice(false,
                $input,
                [0,$i],[-1,$size],
                $outputs[$outidx]->reshape(array_merge(
                    [(int)array_product($shapePrefix)],
                    array_merge([$size],$shape)
                )),
                $events,$waitEvents
            );//->reshape(array_merge($shapePrefix,[$size],$shape));
            $i += $size;
            $outidx++;
        }
        if($this->blocking) {
            $this->finish();
        }
        if($this->profiling) {
            $this->profilingEnd("split");
        }
        return $outputs;
    }

    /**
     * @param array<int> $begin
     * @param array<int> $size
     */
    protected function doSlice(
        bool $reverse,
        NDArray $input,
        array $begin,
        array $size,
        NDArray $output=null,
        object $events=null,object $waitEvents=null
        ) : NDArray
    {
        if($this->profiling) {
            $this->profilingStart("doSlice");
        }
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
            $output = $this->alloc($outputShape,dtype:$input->dtype());
        }else{
            if($outputShape!=$output->shape()){
                throw new InvalidArgumentException('Unmatch output shape: '.
                    $this->printableShapes($output->shape()).'<=>'.
                    $this->printableShapes($outputShape));
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
        if($this->profiling) {
            $this->profilingEnd("doSlice");
        }
        return $output;
    }

    /**
     * repeat
     */
    public function repeat(
        NDArray $A,
        int $repeats,
        int $axis=null,
        bool $keepdims=null,
        object $events=null,object $waitEvents=null
        ) : NDArray
    {
        if($this->profiling) {
            $this->profilingStart("repeat");
        }
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
        for($i=0;$i<$axis;$i++) {
            $outerShape[] = array_shift($innerShape);
        }
        $base = 1;
        if($axis===null) {
            $outputShape = [(int)array_product(
                    array_merge($outerShape,[$repeats],$innerShape))];
        } else {
            if($keepdims) {
                $base = array_shift($innerShape);
                if($base===null) {
                    throw new InvalidArgumentException('dimension rank must be two or greater on keepdims.');
                }
                $outputShape = array_merge($outerShape,[$repeats*$base],$innerShape);
            } else {
                $outputShape = array_merge($outerShape,[$repeats],$innerShape);
            }
        }
        $B = $this->alloc($outputShape,dtype:$A->dtype());
        $m = (int)array_product($outerShape);
        $k = (int)array_product($innerShape)*$base;
        $AA = $A->buffer();
        $offA = $A->offset();
        $BB = $B->buffer();
        $offB = $B->offset();
        $this->openclmath->repeat(
            $m,
            $k,
            $repeats,
            $AA,$offA,
            $BB,$offB,
            $events,$waitEvents
        );
        if($this->blocking) {
            $this->finish();
        }
        if($this->profiling) {
            $this->profilingEnd("repeat");
        }
        return $B;
    }

    /**
     * @param array<int>|NDArray $perm
     */
    public function transpose(
        NDArray $A,
        array|NDArray $perm=null,
        NDArray $B=null,
        object $events=null
        ) : NDArray
    {
        if($A->ndim()<1) {
            throw new InvalidArgumentException('input array must be grator than or equal 1D.');
        }
        if($A->ndim()==2 && $this->isFloat($A)) {
            if($perm) {
                if(count($perm)!=2) {
                    throw new InvalidArgumentException('unmatch sourceshape and perm');
                    //if(!is_array($perm)) {
                    //    $perm = $perm->toArray();
                    //}
                    //if($perm==[0,1]) {
                    //    return $this->copy($A,$B);
                    //}
                    //if($perm!=[1,0]) {
                    //    throw new InvalidArgumentException('unmatch sourceshape and perm');
                    //}
                }
            }
            return $this->transpose2D($A,$B,$events);
        }
        return $this->transposeND($A,$perm,$B,$events);
    }

    /**
     * @param array<int>|NDArray $perm
     */
    protected function transposeND(
        NDArray $A,
        array|NDArray $perm=null,
        NDArray $B=null,
        object $events=null,object $waitEvents=null
        ) : NDArray
    {
        if($this->profiling) {
            $this->profilingStart("transposeND");
        }
        if($perm===null) {
            $perm = range($A->ndim()-1,0,-1);
            $perm = new NDArrayPhp($perm,dtype:NDArray::int32,service:$this->service);
        } elseif(is_array($perm)) {
            $perm = new NDArrayPhp($perm,dtype:NDArray::int32,service:$this->service);
        }
        $perm = $this->toNDArray($perm);
        $shapeA = $A->shape();
        if(count($shapeA)!=count($perm)) {
            throw new InvalidArgumentException('unmatch sourceshape and perm');
        }
        $shapeB = [];
        $checkPerm = [];
        foreach($perm->toArray() as $axis) {
            if(isset($checkPerm[$axis])) {
                throw new InvalidArgumentException('duplicate axis in perm option');
            }
            $checkPerm[$axis] = true;
            $shapeB[] = $shapeA[$axis];
        }
        if($B===null) {
            $B = $this->alloc($shapeB,dtype:$A->dtype());
        } else {
            if($B->shape()!=$shapeB) {
                throw new InvalidArgumentException('output shape must be transpose matrix of input.');
            }
            if($B->dtype()!=$A->dtype()) {
                throw new InvalidArgumentException('output data type must be same with matrix of input.');
            }
        }

        //$AA = $this->toNDArray($A)->buffer();
        //$BB = $this->newHostBuffer($B->size(),$B->dtype());
        $AA = $A->buffer();
        $BB = $B->buffer();
        $offsetA = $A->offset();
        //$offsetB = 0;
        $offsetB = $B->offset();
        $sourceShape = (new NDArrayPhp($shapeA,dtype:NDArray::int32,service:$this->service))->buffer();
        $permBuf = $perm->buffer();
        $this->openclmath->transpose(
            $sourceShape,
            $permBuf,
            $AA, $offsetA,
            $BB, $offsetB, 
            $events,$waitEvents
        );
        //$B->buffer()->write(
        //    $this->queue,
        //    $BB,                            // host buffer
        //    $BB->value_size()*$B->size(),   // bytes
        //    $BB->value_size()*$B->offset(), // offset bytes
        //);
        if($this->blocking) {
            $this->finish();
        }
        if($this->profiling) {
            $this->profilingEnd("transposeND");
        }
        return $B;
    }

    public function transpose2D(
        NDArray $A,
        NDArray $B=null,
        object $events=null
        ) : NDArray
    {
        if($this->profiling) {
            $this->profilingStart("transpose2D");
        }
        if($A->ndim()!=2) {
            throw new InvalidArgumentException('input array must be 2D.');
        }
        $shape = $A->shape();
        $shape = [$shape[1],$shape[0]];
        if($B==null) {
            $B = $this->alloc($shape,dtype:$A->dtype());
        }/* else {
            if($B->shape()!=$shape) {
                throw new InvalidArgumentException('output shape must be transpose matrix of input.');
            }
            if($B->dtype()!=$A->dtype()) {
                throw new InvalidArgumentException('output data type must be same with matrix of input.');
            }
        }*/
        $m = $shape[1];
        $n = $shape[0];
        $AA = $A->buffer();
        $offA = $A->offset();
        $BB = $B->buffer();
        $offB = $B->offset();
        $this->blas->omatcopy(
            BLAS::RowMajor,
            BLAS::Trans,
            $m,
            $n,
            $alpha=1.0,
            $AA,$offA,$n,
            $BB,$offB,$m,
            $this->queue,$events
        );
        if($this->blocking) {
            $this->finish();
        }
        if($this->profiling) {
            $this->profilingEnd("transpose2D");
        }
        return $B;
    }

    public function bandpart(
        NDArray $A,
        int $lower,
        int $upper,
        object $events=null,object $waitEvents=null
    ) : void
    {
        if($this->profiling) {
            $this->profilingStart("bandpart");
        }
        if($A->ndim()<2) {
            throw new InvalidArgumentException('input array must be 2D or upper.');
        }
        $shape = $A->shape();
        $k = array_pop($shape);
        $n = array_pop($shape);
        $m = (int)array_product($shape);
        $buffer = $A->buffer();
        $offset = $A->offset();
        $this->openclmath->bandpart(
            $m,$n,$k,
            $buffer,$offset,
            $lower,
            $upper,
            $events,$waitEvents
        );
        if($this->blocking) {
            $this->finish();
        }
        if($this->profiling) {
            $this->profilingEnd("bandpart");
        }
    }

    public function imagecopy(
        NDArray $A,
        NDArray $B=null,
        bool $channels_first=null,
        int $heightShift=null,
        int $widthShift=null,
        bool $verticalFlip=null,
        bool $horizontalFlip=null,
        bool $rgbFlip=null,
        object $events=null,object $waitEvents=null
        ) : NDArray
    {
        if($this->profiling) {
            $this->profilingStart("imagecopy");
        }
        if($A->ndim()!=3) {
            throw new InvalidArgumentException('input array must be 3D.');
        }
        $shape = $A->shape();
        if($B==null) {
            $B = $this->alloc($shape,dtype:$A->dtype());
            $zerosEvents = $this->newEventList();
            $this->zeros($B,events:$zerosEvents);
            $zerosEvents->wait();
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
        $this->openclmath->imagecopy(
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
            $rgbFlip,
            $events,$waitEvents
        );
        if($this->blocking) {
            $this->finish();
        }
        if($this->profiling) {
            $this->profilingEnd("imagecopy");
        }
        return $B;
    }

    /**
     * @return array<NDArray>
     */
    public function svd(NDArray $matrix,bool $fullMatrices=null) : array
    {
        if($this->profiling) {
            $this->profilingStart("svd");
        }
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
            $VT = $this->copy($VT[R(0,min($m,$n))],null,$copyEvents);
            $copyEvents->wait();
        }
        if($this->blocking) {
            $this->finish();
        }
        if($this->profiling) {
            $this->profilingEnd("svd");
        }
        return [$U,$S,$VT];
    }
}
