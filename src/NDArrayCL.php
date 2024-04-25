<?php
namespace Rindow\Math\Matrix;

require_once __DIR__.'/C.php';
require_once __DIR__.'/R.php';

use Countable;
use IteratorAggregate;
use Traversable;
use InvalidArgumentException;
use OutOfRangeException;
use LogicException;
use RuntimeException;
use Interop\Polite\Math\Matrix\BLAS;
use Interop\Polite\Math\Matrix\NDArray;
use Interop\Polite\Math\Matrix\LinearBuffer;
use Interop\Polite\Math\Matrix\DeviceBuffer;
use Interop\Polite\Math\Matrix\Buffer;
use Interop\Polite\Math\Matrix\OpenCL;
use Rindow\Math\Matrix\Drivers\Service;

/**
 * @implements IteratorAggregate<int, mixed>
 */
class NDArrayCL implements NDArray, Countable, IteratorAggregate
{
    const RANGE_STYLE_DEFAULT = 0;
    const RANGE_STYLE_1 = 1;
    const RANGE_STYLE_FORCE2 = 2;
    public static int $rangeStyle = self::RANGE_STYLE_DEFAULT;

    /** @var array<int,int> $valueSizeTable */
    protected static $valueSizeTable = [
        NDArray::bool  => 1,
        NDArray::int8  => 1,
        NDArray::int16 => 2,
        NDArray::int32 => 4,
        NDArray::int64 => 8,
        NDArray::uint8 => 1,
        NDArray::uint16 => 2,
        NDArray::uint32 => 4,
        NDArray::uint64 => 8,
        NDArray::float8 => 1,
        NDArray::float16 => 2,
        NDArray::float32 => 4,
        NDArray::float64 => 8,
        NDArray::complex16 => 2,
        NDArray::complex32 => 4,
        NDArray::complex64 => 8,
        NDArray::complex128 => 16,
    ];
    protected Service $service;
    protected object $clBufferFactory;
    protected object $context;
    protected object $queue;
    /** @var array<int> $shape */
    protected array $shape;
    protected DeviceBuffer $buffer;
    protected int $offset;
    protected int $dtype;
    protected int $flags;
    protected bool $portableSerializeMode = false;
    protected object $events;

    /**
     * @param array<int> $shape
     */
    final public function __construct(
        object $queue,
        mixed $buffer=null,
        int $dtype=null,
        array $shape = null,
        int $offset=null,
        int $flags=null,
        Service $service=null
    ) {
        if($service===null) {
            throw new InvalidArgumentException("No service specified.");
        }
        $this->service = $service;
        $this->clBufferFactory = $service->buffer(Service::LV_ACCELERATED);
        $context = $queue->getContext();
        $this->context = $context;
        $this->queue = $queue;
        if($dtype===null) {
            $dtype = NDArray::float32;
        } else {
            $dtype = $dtype;
        }
        if($offset===null) {
            $offset = 0;
        }
        if($flags===null) {
            $flags = OpenCL::CL_MEM_READ_WRITE;
        }
        if($shape===null) {
            throw new InvalidArgumentException("Invalid dimension size");
        }

        $this->assertShape($shape);
        $this->shape = $shape;
        $this->flags = $flags;
        $size = (int)array_product($shape);
        if($buffer instanceof DeviceBuffer) {
            if($buffer->bytes()
                < ($size + $offset)*static::$valueSizeTable[$dtype]) {
                throw new InvalidArgumentException("Invalid dimension size");
            }
            $this->dtype  = $dtype;
            $this->buffer = $buffer;
            $this->offset = $offset;
        } elseif($buffer===null) {
            $size = (int)array_product($shape);
            $this->buffer = $this->newBuffer($context, $size, $dtype, $flags);
            $this->dtype  = $dtype;
            $this->offset = 0;
        } elseif($buffer instanceof LinearBuffer) {
            $size = (int)array_product($shape);
            if($size > count($buffer)-$offset) {
                throw new InvalidArgumentException("host buffer is too small");
            }
            $this->buffer = $this->newBuffer(
                $context,
                $size,
                $buffer->dtype(),
                $flags,
                $buffer,
                $offset
            );
            $this->dtype = $buffer->dtype();
            $this->offset = 0;
        } else {
            if(is_object($buffer)) {
                $typename = get_class($buffer);
            } else {
                $typename = gettype($buffer);
            }
            throw new InvalidArgumentException("Invalid type of array: ".$typename);
        }
    }

    public function service() : Service
    {
        return $this->service;
    }

    protected function newBuffer(
        object $context,
        int $size,
        int $dtype,
        int $flags=0,
        object $hostBuffer=null,
        int $hostOffset=0
    ) : DeviceBuffer {
        //if(!extension_loaded('rindow_opencl')) {
        //    throw new LogicException("rindow_opencl extension is not loaded.");
        //}
        //return new OpenCLBuffer($context,static::$valueSizeTable[$dtype]*$size,
        //    $flags,$hostBuffer,$hostOffset,$dtype);
        return $this->clBufferFactory->Buffer(
            $context,
            static::$valueSizeTable[$dtype]*$size,
            $flags,
            $hostBuffer,
            $hostOffset,
            $dtype
        );
    }

    /**
     * @param array<mixed> $shape
     */
    protected function assertShape(array $shape) : void
    {
        foreach($shape as $num) {
            if(!is_int($num)) {
                throw new InvalidArgumentException(
                    "Invalid shape numbers. It gives ".gettype($num)
                );
            }
            if($num<=0) {
                throw new InvalidArgumentException(
                    "Invalid shape numbers. It gives ".$num
                );
            }
        }
    }

    /**
     * @return array<int>
     */
    public function shape() : array
    {
        return $this->shape;
    }

    public function ndim() : int
    {
        return count($this->shape);
    }

    public function dtype() : int
    {
        return $this->dtype;
    }

    public function flags() : int
    {
        return $this->flags;
    }

    public function buffer() : Buffer
    {
        return $this->buffer;
    }

    public function offset() : int
    {
        return $this->offset;
    }

    public function valueSize() : int
    {
        return static::$valueSizeTable[$this->dtype];
    }

    public function size() : int
    {
        return (int)array_product($this->shape);
    }

    /**
     * @param array<int> $shape
     */
    public function reshape(array $shape) : NDArray
    {
        $this->assertShape($shape);
        if($this->size()!=array_product($shape)) {
            throw new InvalidArgumentException("Unmatch size to reshape: ".
                "[".implode(',', $this->shape())."]=>[".implode(',', $shape)."]");
        }
        $newArray = new static($this->queue,$this->buffer,
            $this->dtype,$shape,$this->offset,$this->flags, service:$this->service());
        return $newArray;
    }

    public function toArray() : mixed
    {
        return $this->toNDArray()->toArray();
    }

    public function toNDArray(
        bool $blocking_read=null,
        object $events=null,
        object $waitEvents=null
    ) : NDArray {
        $blocking_read = $blocking_read ?? true;
        $array = new NDArrayPhp(null, $this->dtype, $this->shape, service:$this->service());
        $valueSize = static::$valueSizeTable[$this->dtype];
        $size = array_product($this->shape);
        $this->buffer->read(
            $this->queue,
            $array->buffer(),
            $size*$valueSize,
            $this->offset*$valueSize,
            $hostoffset=0,
            $blocking_read,
            $events,
            $waitEvents
        );
        return $array;
    }

    /**
     * @return int|array<int>|Range
     */
    protected function castOffset(mixed $offset) : int|array|Range
    {
        if(!is_int($offset)&&!is_array($offset)&&!($offset instanceof Range)) {
            throw new InvalidArgumentException("Array offsets must be integers or ranges.");
        }
        return $offset;
    }

    public function offsetExists($offset) : bool
    {
        $offset = $this->castOffset($offset);
        if(is_array($offset) && self::$rangeStyle==self::RANGE_STYLE_FORCE2) {
            throw new InvalidArgumentException("offset style is old renge style.");
        }
        if(is_array($offset)) {
            if(count($offset)!=2 ||
                !array_key_exists(0, $offset) || !array_key_exists(1, $offset) ||
                $offset[0]>$offset[1]) {
                $det = '';
                if(is_numeric($offset[0])&&is_numeric($offset[1])) {
                    $det = ':['. implode(',', $offset).']';
                }
                throw new OutOfRangeException("Illegal range specification.".$det);
            }
            $start = $offset[0];
            $limit = $offset[1];
            if(self::$rangeStyle==self::RANGE_STYLE_1) {
                ++$limit;
            }
        } elseif(is_int($offset)) {
            $start = $offset;
            $limit = $offset+1;
        } else {
            $start = $offset->start();
            $limit = $offset->limit();
            $delta = $offset->delta();
            if($start>=$limit||$delta!=1) {
                $det = ":[$start,$limit".(($delta!=1)?",$delta":"").']';
                throw new OutOfRangeException("Illegal range specification.".$det);
            }
        }
        if($start < 0 || $limit > $this->shape[0]) {
            return false;
        }
        return true;
    }

    public function offsetGet($offset) : mixed
    {
        $offset = $this->castOffset($offset);
        if(!$this->offsetExists($offset)) {
            if(count($this->shape)==0) {
                throw new OutOfRangeException("This object is scalar.");
            } else {
                throw new OutOfRangeException("Index is out of range");
            }
        }

        // for single index specification
        if(is_numeric($offset)) {
            $offset = (int)$offset;
            $shape = $this->shape;
            $max = array_shift($shape);
            if(count($shape)==0) {
                $new = new static($this->queue,$this->buffer,$this->dtype,
                    $shape, $this->offset+$offset, $this->flags, service:$this->service());
                return $new;
            }
            $size = (int)array_product($shape);
            $new = new static($this->queue, $this->buffer, $this->dtype,
                $shape, $this->offset+$offset*$size, $this->flags, service:$this->service());
            return $new;
        }
        
        // for range spesification
        $shape = $this->shape;
        array_shift($shape);
        if(is_array($offset)) {
            $start = (int)$offset[0];
            $limit = (int)$offset[1];
            if(self::$rangeStyle==self::RANGE_STYLE_1) {
                ++$limit;
            }
        } else {
            $start = (int)$offset->start();
            $limit = (int)$offset->limit();
            if($offset->delta()!=1) {
                throw new OutOfRangeException("Illegal range specification.:delta=".$offset->delta());
            }
        }
        $rowsCount = $limit-$start;
        if(count($shape)>0) {
            $itemSize = (int)array_product($shape);
        } else {
            $itemSize = 1;
        }
        if($rowsCount<0) {
            throw new OutOfRangeException('Invalid range');
        }
        array_unshift($shape, $rowsCount);
        $size = (int)array_product($shape);
        $new = new static($this->queue,$this->buffer,$this->dtype,
            $shape,$this->offset+$start*$itemSize, $this->flags, service:$this->service());
        return $new;
    }

    public function offsetSet($offset, $value) : void
    {
        $offset = $this->castOffset($offset);
        if(!$this->offsetExists($offset)) {
            if(count($this->shape)==0) {
                throw new OutOfRangeException("This object is scalar.");
            } else {
                throw new OutOfRangeException("Index is out of range");
            }
        }
        // for range spesification
        if(!is_int($offset)) {
            throw new OutOfRangeException("Unsuppored to set for range specification.");
        }
        // for single index specification
        $shape = $this->shape;
        $max = array_shift($shape);
        if(count($shape)==0) {
            if(!($value instanceof NDArray)) {
                throw new LogicException("Must be NDArray on OpenCL.");
            }
            $buffer = $value->buffer();
            if(!($buffer instanceof DeviceBuffer)) {
                throw new LogicException("Must be NDArray on OpenCL.");
            }
            $valueSize = static::$valueSizeTable[$value->dtype()];
            $this->buffer->copy(
                $this->queue,
                $buffer,
                $valueSize,
                $value->offset()*$valueSize,
                ($this->offset+$offset)*$valueSize
            );
            return;
        }

        if(!($value instanceof static)||$value->shape()!=$shape) {
            throw new LogicException("Unmatch shape numbers");
        }
        $src = $value->buffer();
        $size = (int)array_product($shape);
        $src_idx = $value->offset();
        $idx=$this->offset+$offset*$size;

        $valueSize = static::$valueSizeTable[$value->dtype()];
        $this->buffer->copy(
            $this->queue,
            $src,
            $size*$valueSize,
            $src_idx*$valueSize,
            $idx*$valueSize
        );
    }

    public function offsetUnset($offset) : void
    {
        throw new LogicException("Unsuppored Operation");
    }

    public function count() : int
    {
        return $this->shape[0];
    }

    public function getIterator() : Traversable
    {
        $count = $this->shape[0];
        for($i=0;$i<$count;$i++) {
            yield $i => $this->offsetGet($i);
        }
    }

    public function setPortableSerializeMode(bool $mode) : void
    {
        throw new LogicException("Unsuppored Operation");
    }

    public function getPortableSerializeMode() : bool
    {
        return $this->portableSerializeMode;
    }

    public function __serialize() : array
    {
        throw new LogicException("Unsuppored Operation");
    }

    /**
     * @param array<string,mixed> $serialized
     */
    public function __unserialize(array $serialized) : void
    {
        throw new LogicException("Unsuppored Operation");
    }

    public function setEvents(object $events) : void
    {
        $this->events = $events;
    }

    public function getEvents() : object
    {
        return $this->events;
    }

    public function __clone()
    {
        if(!($this->buffer instanceof DeviceBuffer)) {
            throw new RuntimeException('Unknown buffer type is uncloneable:'.get_class($this->buffer));
        }
        $bytes = $this->buffer->bytes();
        $dtype = $this->buffer->dtype();
        $flags = $this->flags & ~OpenCL::CL_MEM_COPY_HOST_PTR;
        $newBuffer = $this->clBufferFactory->Buffer(
            $this->context,
            $bytes,
            $flags,
            null,
            0,
            $dtype
        );
        $events = $this->service()->openCL()->EventList();
        $newBuffer->copy($this->queue, $this->buffer, 0, 0, 0, $events);
        $events->wait();
        $this->flags = $flags;
        $this->buffer = $newBuffer;
    }
}
