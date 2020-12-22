<?php
namespace Rindow\Math\Matrix;

use ArrayAccess;
use Countable;
use IteratorAggregate;
use InvalidArgumentException;
use OutOfRangeException;
use LogicException;
use RuntimeException;
use Serializable;
use Interop\Polite\Math\Matrix\BLAS;
use Interop\Polite\Math\Matrix\NDArray;
use Interop\Polite\Math\Matrix\LinearBuffer;
use Interop\Polite\Math\Matrix\OpenCL;
#use Rindow\Math\Matrix\NDArrayPhp;

class NDArrayCL implements NDArray,Serializable,Countable,IteratorAggregate
{
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
    ];
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
        object $context, object $queue, $buffer=null, int $dtype=null, array $shape = null,
        int $offset=null, int $flags=null)
    {
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
        if($buffer instanceof OpenCLBuffer) {
            if($buffer->bytes()
                < ($size + $offset)*self::$valueSizeTable[$dtype]) {
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
            throw new InvalidArgumentException("Invalid type of array");
        }
    }

    protected function newBuffer(
        object $context, int $size, int $dtype, int $flags=0,
        object $hostBuffer=null, int $hostOffset=0)
    {
        if(!extension_loaded('rindow_opencl')) {
            throw new LogicException("rindow_opencl extension is not loaded.");
        }
        return new OpenCLBuffer($context,self::$valueSizeTable[$dtype]*$size,
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
        return self::$valueSizeTable[$this->dtype];
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
        $newArray = new self($this->context,$this->queue,$this->buffer,
            $this->dtype,$shape,$this->offset,$this->flags);
        return $newArray;
    }

    public function toArray()
    {
        return $this->toNDArray()->toArray();
    }

    public function toNDArray(
        bool $blocking_read=true,EventList $waitEvents=null,
        EventList &$events=null) : NDArray
    {
        $array = new NDArrayPhp(null,$this->dtype,$this->shape);
        $valueSize = self::$valueSizeTable[$this->dtype];
        $size = array_product($this->shape);
        $event = $this->buffer->read($this->queue,$array->buffer(),$size*$valueSize,
            $this->offset*$valueSize,$hostoffset=0,$blocking_read,$waitEvents);
        $events = $event;
        return $array;
    }

    public function offsetExists( $offset )
    {
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
            $end   = $offset[1];
        } elseif(is_int($offset)) {
            $start = $offset;
            $end   = $offset;
        } else {
            throw new OutOfRangeException("Dimension must be integer");
        }
        if($start < 0 || $end >= $this->shape[0])
            return false;
        return true;
    }

    public function offsetGet( $offset )
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
            $shape = $this->shape;
            array_shift($shape);
            $rowsCount = $offset[1]-$offset[0]+1;
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
            $new = new self($this->context,$this->queue,$this->buffer,$this->dtype,
                $shape,$this->offset+$offset[0]*$itemSize, $this->flags);
            return $new;
        }

        // for single index specification
        $shape = $this->shape;
        $max = array_shift($shape);
        if(count($shape)==0) {
            $new = new self($this->context,$this->queue,$this->buffer,$this->dtype,
                $shape, $this->offset+$offset, $this->flags);
            return $new;
        }
        $size = (int)array_product($shape);
        $new = new self($this->context, $this->queue, $this->buffer, $this->dtype,
            $shape, $this->offset+$offset*$size, $this->flags);
        return $new;
    }

    public function offsetSet( $offset , $value )
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
            if(!($buffer instanceof OpenCLBuffer))
                throw new LogicException("Must be NDArray on OpenCL.");
            $valueSize = self::$valueSizeTable[$value->dtype()];
            $this->buffer->copy($this->queue, $buffer, $valueSize,
                ($this->offset+$offset)*$valueSize ,$value->offset()*$valueSize);
            return;
        }

        if(!($value instanceof self)||$value->shape()!=$shape) {
            throw new LogicException("Unmatch shape numbers");
        }
        $src = $value->buffer();
        $size = (int)array_product($shape);
        $src_idx = $value->offset();
        $idx=$this->offset+$offset*$size;

        $valueSize = self::$valueSizeTable[$value->dtype()];
        $this->buffer->copy($this->queue, $src, $size*$valueSize,
            $idx*$valueSize ,$src_idx*$valueSize);
    }

    public function offsetUnset( $offset )
    {
        throw new LogicException("Unsuppored Operation");
    }

    public function count()
    {
        return $this->shape[0];
    }

    public function  getIterator()
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

    public function serialize()
    {
        throw new LogicException("Unsuppored Operation");
    }

    public function unserialize($serialized)
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
}
