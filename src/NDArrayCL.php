<?php
namespace Rindow\Math\Matrix;

require_once __DIR__.'/C.php';
require_once __DIR__.'/R.php';

use ArrayAccess;
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
use Interop\Polite\Math\Matrix\OpenCL;
use Rindow\Math\Matrix\Drivers\Service;

class NDArrayCL implements NDArray,Countable,IteratorAggregate
{
    const RANGE_STYLE_DEFAULT = 0;
    const RANGE_STYLE_1 = 1;
    const RANGE_STYLE_FORCE2 = 2;
    static public int $rangeStyle = self::RANGE_STYLE_DEFAULT;

    static protected $valueSizeTable = [
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
    protected $service;
    protected $clBufferFactory;
    protected $context;
    protected $queue;
    protected $shape;
    protected $buffer;
    protected $offset;
    protected $dtype;
    protected $flags;
    protected $portableSerializeMode = false;
    protected $events;

    public function __construct(
        object $queue, mixed $buffer=null, int $dtype=null, array $shape = null,
        int $offset=null, int $flags=null,
        Service $service=null)
    {
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
            $this->buffer = $this->newBuffer($context,$size,$dtype,$flags);
            $this->dtype  = $dtype;
            $this->offset = 0;
        } elseif($buffer instanceof LinearBuffer) {
            if($offset===null||!is_int($offset))
                throw new InvalidArgumentException("Must specify offset with the buffer");
            $size = (int)array_product($shape);
            if($size > count($buffer)-$offset)
                throw new InvalidArgumentException("host buffer is too small");
            $this->buffer = $this->newBuffer($context,
                    $size, $buffer->dtype(), $flags,
                    $buffer, $offset);
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

    protected function newBuffer(
        object $context, int $size, int $dtype, int $flags=0,
        object $hostBuffer=null, int $hostOffset=0)
    {
        //if(!extension_loaded('rindow_opencl')) {
        //    throw new LogicException("rindow_opencl extension is not loaded.");
        //}
        //return new OpenCLBuffer($context,static::$valueSizeTable[$dtype]*$size,
        //    $flags,$hostBuffer,$hostOffset,$dtype);
        return $this->clBufferFactory->Buffer($context,static::$valueSizeTable[$dtype]*$size,
            $flags,$hostBuffer,$hostOffset,$dtype);
    }

    protected function assertShape(array $shape)
    {
        foreach($shape as $num) {
            if(!is_int($num)) {
                throw new InvalidArgumentException(
                    "Invalid shape numbers. It gives ".gettype($num));
            }
            if($num<=0) {
                throw new InvalidArgumentException(
                    "Invalid shape numbers. It gives ".$num);
            }
        }
    }

    public function shape() : array
    {
        return $this->shape;
    }

    public function ndim() : int
    {
        return count($this->shape);
    }

    public function dtype()
    {
        return $this->dtype;
    }

    public function flags()
    {
        return $this->flags;
    }

    public function buffer() : ArrayAccess
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

    public function reshape(array $shape) : NDArray
    {
        $this->assertShape($shape);
        if($this->size()!=array_product($shape)) {
            throw new InvalidArgumentException("Unmatch size to reshape: ".
                "[".implode(',',$this->shape())."]=>[".implode(',',$shape)."]");
        }
        $newArray = new static($this->queue,$this->buffer,
            $this->dtype,$shape,$this->offset,$this->flags, service:$this->service);
        return $newArray;
    }

    public function toArray()
    {
        return $this->toNDArray()->toArray();
    }

    public function toNDArray(
        bool $blocking_read=null,EventList $waitEvents=null,
        EventList &$events=null) : NDArray
    {
        $blocking_read = $blocking_read ?? true;
        $array = new NDArrayPhp(null,$this->dtype,$this->shape,service:$this->service);
        $valueSize = static::$valueSizeTable[$this->dtype];
        $size = array_product($this->shape);
        $event = $this->buffer->read($this->queue,$array->buffer(),$size*$valueSize,
            $this->offset*$valueSize,$hostoffset=0,$blocking_read,$waitEvents);
        $events = $event;
        return $array;
    }

    public function offsetExists( $offset ) : bool
    {
        if(is_array($offset) && self::$rangeStyle==self::RANGE_STYLE_FORCE2) {
            throw new InvalidArgumentException("offset style is old renge style.");
        }
        if(is_array($offset)) {
            if(count($offset)!=2 ||
                !array_key_exists(0,$offset) || !array_key_exists(1,$offset) ||
                $offset[0]>$offset[1]) {
                    $det = '';
                    if(is_numeric($offset[0])&&is_numeric($offset[1]))
                        $det = ':['. implode (',',$offset).']';
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
        } elseif($offset instanceof Range) {
            $start = $offset->start();
            $limit = $offset->limit();
            $delta = $offset->delta();
            if($start>=$limit||$delta!=1) {
                $det = ":[$start,$limit".(($delta!=1)?",$delta":"").']';
                throw new OutOfRangeException("Illegal range specification.".$det);
            }
        } else {
            throw new OutOfRangeException("Dimension must be integer");
        }
        if($start < 0 || $limit > $this->shape[0])
            return false;
        return true;
    }

    public function offsetGet( $offset ) : mixed
    {
        if(!$this->offsetExists($offset)) {
            if(count($this->shape)==0) {
                throw new OutOfRangeException("This object is scalar.");
            } else {
                throw new OutOfRangeException("Index is out of range");
            }
        }

        // for single index specification
        if(is_numeric($offset)) {
            $shape = $this->shape;
            $max = array_shift($shape);
            if(count($shape)==0) {
                $new = new static($this->queue,$this->buffer,$this->dtype,
                    $shape, $this->offset+$offset, $this->flags, service:$this->service);
                return $new;
            }
            $size = (int)array_product($shape);
            $new = new static($this->queue, $this->buffer, $this->dtype,
                $shape, $this->offset+$offset*$size, $this->flags, service:$this->service);
            return $new;
        }
        
        // for range spesification
        $shape = $this->shape;
        array_shift($shape);
        if(is_array($offset)) {
            $start = $offset[0];
            $limit = $offset[1];
            if(self::$rangeStyle==self::RANGE_STYLE_1) {
                ++$limit;
            }
        } else {
            $start = $offset->start();
            $limit = $offset->limit();
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
        array_unshift($shape,$rowsCount);
        $size = (int)array_product($shape);
        $new = new static($this->queue,$this->buffer,$this->dtype,
            $shape,$this->offset+$start*$itemSize, $this->flags, service:$this->service);
        return $new;
    }

    public function offsetSet( $offset , $value ) : void
    {
        if(!$this->offsetExists($offset)) {
            if(count($this->shape)==0) {
                throw new OutOfRangeException("This object is scalar.");
            } else {
                throw new OutOfRangeException("Index is out of range");
            }
        }
        // for range spesification
        if(is_array($offset)) {
            throw new OutOfRangeException("Unsuppored to set for range specification.");
        }
        // for single index specification
        $shape = $this->shape;
        $max = array_shift($shape);
        if(count($shape)==0) {
            if(!($value instanceof NDArray))
                throw new LogicException("Must be NDArray on OpenCL.");
            $buffer = $value->buffer();
            if(!($buffer instanceof DeviceBuffer))
                throw new LogicException("Must be NDArray on OpenCL.");
            $valueSize = static::$valueSizeTable[$value->dtype()];
            $this->buffer->copy($this->queue, $buffer, $valueSize,
                $value->offset()*$valueSize, ($this->offset+$offset)*$valueSize);
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
        $this->buffer->copy($this->queue, $src, $size*$valueSize,
            $src_idx*$valueSize ,$idx*$valueSize);
    }

    public function offsetUnset( $offset ) : void
    {
        throw new LogicException("Unsuppored Operation");
    }

    public function count() : int
    {
        return $this->shape[0];
    }

    public function  getIterator() : Traversable
    {
        $count = $this->shape[0];
        for($i=0;$i<$count;$i++) {
            yield $i => $this->offsetGet($i);
        }
    }

    public function setPortableSerializeMode(bool $mode)
    {
        throw new LogicException("Unsuppored Operation");
    }

    public function getPortableSerializeMode()
    {
        return $this->portableSerializeMode;
    }

    public function __serialize()
    {
        throw new LogicException("Unsuppored Operation");
    }

    public function __unserialize($serialized)
    {
        throw new LogicException("Unsuppored Operation");
    }

    public function setEvents($events)
    {
        $this->events = $events;
    }

    public function getEvents()
    {
        return $this->events = $events;
    }

    public function __clone()
    {
        if(!($this->buffer instanceof DeviceBuffer)) {
            throw new RuntimeException('Unknown buffer type is uncloneable:'.get_class($this->_buffer));
        }
        $bytes = $this->buffer->bytes();
        $dtype = $this->buffer->dtype();
        $flags = $this->flags & ~OpenCL::CL_MEM_COPY_HOST_PTR;
        $newBuffer = $this->clBufferFactory->Buffer(
            $this->context,$bytes,
            $flags,null,0,$dtype);
        $events = $this->service->openCL()->EventList();
        $newBuffer->copy($this->queue,$this->buffer,0,0,0,$events);
        $events->wait();
        $this->flags = $flags;
        $this->buffer = $newBuffer;
    }
}
