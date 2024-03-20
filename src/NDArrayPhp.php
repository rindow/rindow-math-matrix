<?php
namespace Rindow\Math\Matrix;

require_once __DIR__.'/R.php';

use ArrayObject;
use ArrayAccess;
use Countable;
use Traversable;
use Serializable;
use IteratorAggregate;
use InvalidArgumentException;
use OutOfRangeException;
use LogicException;
use RuntimeException;
use Interop\Polite\Math\Matrix\BLAS;
use Interop\Polite\Math\Matrix\NDArray;
use Interop\Polite\Math\Matrix\Buffer as BufferInterface;
use Rindow\Math\Matrix\Drivers\Service;
use Rindow\Math\Matrix\Drivers\Selector;

class NDArrayPhp implements NDArray,Countable,Serializable,IteratorAggregate
{
    use ComplexUtils;

    const RANGE_STYLE_DEFAULT = 0;
    const RANGE_STYLE_1 = 1;
    const RANGE_STYLE_FORCE2 = 2;
    static public int $rangeStyle = self::RANGE_STYLE_DEFAULT;

    const SERIALIZE_NDARRAY_KEYWORD = 'NDArray:';
    const SERIALIZE_OLDSTYLE_KEYWORD = 'O:29:"Rindow\Math\Matrix\NDArrayPhp"';
    static public $unserializeWarning = 2;

    protected array $_shape;
    protected BufferInterface $_buffer;
    protected int $_offset;
    protected int $_dtype;
    protected bool $_portableSerializeMode = false;
    protected ?Service $service = null;

    public function __construct(
        mixed $array = null,
        int $dtype=null,
        array $shape = null,
        int $offset=null,
        Service $service=null,
        )
    {
        if($service===null) {
            throw new InvalidArgumentException("No service specified.");
        }
        $this->service = $service;
        if($array===null && $dtype===null && $shape===null && $offset===null) {
            // Empty definition for Unserialize
            return;
        }

        $orgDtype = $dtype;
        if($dtype===null) {
            $dtype = NDArray::float32;
        }

        if($array===null && $shape!==null) {
            $this->assertShape($shape);
            $size = (int)array_product($shape);
            $this->_buffer = $this->newBuffer($size,$dtype);
            $this->_offset = 0;
        } elseif($this->isBuffer($array)) {
            if($offset===null||!is_int($offset))
                throw new InvalidArgumentException("Must specify offset with the buffer");
            if($shape===null)
                throw new InvalidArgumentException("Invalid dimension size");
            $this->_buffer = $array;
            $this->_offset = $offset;
        } elseif(is_array($array)||$array instanceof ArrayObject) {
            $dummyBuffer = new ArrayObject();
            $idx = 0;
            $this->array2Flat($array,$dummyBuffer,$idx,$prepare=true);
            $this->_buffer = $this->newBuffer($idx,$dtype);
            $idx = 0;
            $this->array2Flat($array,$this->_buffer,$idx,$prepare=false);
            $this->_offset = 0;
            if($shape===null) {
                $shape = $this->genShape($array);
            }
        } elseif(is_numeric($array)||is_bool($array)||$this->isComplexObject($array)) {
            if(is_numeric($array)) {
                if($orgDtype==null) {
                    $dtype = NDArray::float32;
                }
            } elseif(is_bool($array)) {
                if($orgDtype==null) {
                    $dtype = NDArray::bool;
                } else {
                    if($dtype!=NDArray::bool) {
                        throw new InvalidArgumentException("unmatch dtype with bool value");
                    }
                }
            } elseif($this->isComplexObject($array)) {
                if($orgDtype==null) {
                    $dtype = NDArray::complex64;
                } else {
                    if(!$this->isComplex($dtype)) {
                        throw new InvalidArgumentException("unmatch dtype with complex value");
                    }
                }
            }
            $this->_buffer = $this->newBuffer(1,$dtype);
            $this->_buffer[0] = $array;
            $this->_offset = 0;
            if($shape===null) {
                $shape = [];
            }
            $this->assertShape($shape);
            if(array_product($shape)!=1) {
                throw new InvalidArgumentException("Invalid dimension size");
            }
        } else {
            throw new InvalidArgumentException("Invalid type of array");
        }
        $this->assertShape($shape);
        $this->_shape = $shape;

        $size = (int)array_product($shape);
        if(count($this->_buffer) - $this->_offset < $size)
            throw new InvalidArgumentException("Invalid dimension size");

        $this->_dtype = $dtype;
    }

    protected function newBuffer($size,$dtype)
    {
        return $this->service->buffer()->Buffer($size,$dtype);
    }

    protected function isBuffer($buffer)
    {
        if($buffer instanceof BufferInterface) {
            return true;
        } else {
            return false;
        }
    }

    protected function isComplex(int $dtype=null) : bool
    {
        $dtype = $dtype ?? $this->_dtype;
        return $this->cistype($dtype);
    }

    public function isComplexObject(mixed $value) : bool
    {
        return $this->cisObject($value);
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

    protected function array2Flat($A, BufferInterface|ArrayObject $F, &$idx, $prepare)
    {
        if(is_array($A)) {
            ksort($A);
        } elseif($A instanceof ArrayObject) {
            $A->ksort();
        }

        $num = null;
        foreach ($A as $key => $value) {
            if(!is_int($key))
                throw new InvalidArgumentException("Dimension must be integer");
            if(is_array($value)||$value instanceof ArrayObject) {
                $num2 = $this->array2Flat($value, $F, $idx, $prepare);
                if($num===null) {
                    $num = $num2;
                } else {
                    if($num!=$num2)
                        throw new InvalidArgumentException("The shape of the dimension is broken");
                }
            } else {
                if($num!==null)
                    throw new InvalidArgumentException("The shape of the dimension is broken");
                if(!$prepare)
                    $F[$idx] = $value;
                $idx++;
            }
        }
        return count($A);
    }

    protected function flat2Array(BufferInterface $F, &$idx, array $shape)
    {
        $size = array_shift($shape);
        if(count($shape)) {
            $A = [];
            for($i=0; $i<$size; $i++) {
                $A[$i] = $this->flat2Array($F,$idx,$shape);
            }
        }  else {
            $A = [];
            for($i=0; $i<$size; $i++) {
                $value = $F[$idx];
                if($this->isComplex()) {
                    $value = new Complex($value->real,$value->imag);
                }
                $A[$i] = $value;
                $idx++;
            }
        }
        return $A;
    }

    protected function genShape($A)
    {
        $shape = [];
        while(is_array($A) || $A instanceof ArrayObject) {
            $shape[] = count($A);
            $A = $A[0];
        }
        return $shape;
    }

    public function service() : ?Service
    {
        return $this->service;
    }

    public function shape() : array
    {
        return $this->_shape;
    }

    public function ndim() : int
    {
        return count($this->_shape);
    }

    public function dtype()
    {
        return $this->_dtype;
    }

    public function buffer() : ArrayAccess
    {
        return $this->_buffer;
    }

    public function offset() : int
    {
        return $this->_offset;
    }

    public function size() : int
    {
        return (int)array_product($this->_shape);
    }

    public function reshape(array $shape) : NDArray
    {
        $this->assertShape($shape);
        if($this->size()!=array_product($shape)) {
            throw new InvalidArgumentException("Unmatch size to reshape: ".
                "[".implode(',',$this->shape())."]=>[".implode(',',$shape)."]");
        }
        $newArray = new self($this->buffer(),$this->dtype(),$shape,$this->offset(),service:$this->service);
        return $newArray;
    }

    public function toArray()
    {
        if(count($this->_shape)==0) {
            $value = $this->_buffer[$this->_offset];
            if($this->isComplex()) {
                $value = new Complex($value->real,$value->imag);
            }
            return $value;
        }
        $idx = $this->_offset;
        return $this->flat2Array($this->_buffer, $idx, $this->_shape);
    }

    public function offsetExists( $offset ) : bool
    {
        if(is_array($offset) && self::$rangeStyle==self::RANGE_STYLE_FORCE2) {
            throw new InvalidArgumentException("offset style is old range style.");
        }
        if(count($this->_shape)==0)
            return false;
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
        if($start < 0 || $limit > $this->_shape[0])
            return false;
        return true;
    }

    public function offsetGet( $offset ) : mixed
    {
        if(!$this->offsetExists($offset)) {
            throw new OutOfRangeException("Index is out of range");
        }

        // for single index specification
        if(is_numeric($offset)) {
            $shape = $this->_shape;
            $max = array_shift($shape);
            if(count($shape)==0) {
                $value = $this->_buffer[$this->_offset+$offset];
                if($this->isComplex()) {
                    $value = new Complex($value->real,$value->imag);
                }
                return $value;
            }
            $size = (int)array_product($shape);
            $new = new self($this->_buffer,$this->_dtype,$shape,$this->_offset+$offset*$size,service:$this->service);
            return $new;
        }

        // for range spesification
        $shape = $this->_shape;
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
        $new = new self($this->_buffer,$this->_dtype,
            $shape,$this->_offset+$start*$itemSize,service:$this->service);
        return $new;
    }

    public function offsetSet( $offset , $value ) : void
    {
        if(!$this->offsetExists($offset))
            throw new OutOfRangeException("Index is out of range");
        // for range spesification
        if(is_array($offset)) {
            throw new OutOfRangeException("Unsuppored to set for range specification.");
        }
        if($offset instanceof Range) {
            throw new OutOfRangeException("Unsuppored to set for range specification.");
        }
        // for single index specification
        $shape = $this->_shape;
        $max = array_shift($shape);
        if(!count($shape)) {
            if($this->isComplex()) {
                if(!($value instanceof Complex)) {
                    throw new InvalidArgumentException("Must be complex type");
                }
            } else {
                if(!is_scalar($value)) {
                    throw new InvalidArgumentException("Must be scalar type");
                }
            }
            $this->_buffer[$this->_offset+$offset] = $value;
            return;
        }

        if(!($value instanceof self)||$value->shape()!=$shape) {
            throw new InvalidArgumentException("Unmatch shape numbers");
        }
        $copy = $value->buffer();
        $size = (int)array_product($shape);
        $src_idx = $value->offset();
        $idx=$this->_offset+$offset*$size;
        for($i=0 ; $i<$size ; $i++,$idx++,$src_idx++) {
            $this->_buffer[$idx] = $copy[$src_idx];
        }
    }

    public function offsetUnset( $offset ) : void
    {
        throw new LogicException("Unsuppored Operation");
    }

    public function count() : int
    {
        if(count($this->_shape)==0)
            return 0;
        return $this->_shape[0];
    }

    public function  getIterator() : Traversable
    {
        if(count($this->_shape)==0)
            return [];
        $count = $this->_shape[0];
        for($i=0;$i<$count;$i++) {
            yield $i => $this->offsetGet($i);
        }
    }

    /**
     * It is an old specification and has no effect.
     */
    public function setPortableSerializeMode(bool $mode)
    {
        $this->_portableSerializeMode = $mode ? true : false;
    }

    /**
     * It is an old specification and has no effect.
     */
    public function getPortableSerializeMode()
    {
        return $this->_portableSerializeMode;
    }

    public function serialize() : string
    {
        return static::SERIALIZE_NDARRAY_KEYWORD.serialize($this->__serialize());
    }

    public function unserialize(string $data) : void
    {
        if(strpos($data,static::SERIALIZE_NDARRAY_KEYWORD)===0) {
            $data = substr($data,strlen(static::SERIALIZE_NDARRAY_KEYWORD));
            $data = unserialize($data);
            if(is_array($data)) {
                $this->__unserialize($data);
                return;
            }
        } elseif(strpos($data,static::SERIALIZE_OLDSTYLE_KEYWORD)===0) {
            $data = unserialize($data);
        } else {
            throw new RuntimeException("Invalid saved data.");
        }
        if(!($data instanceof self)) {
            throw new RuntimeException("Invalid saved data.");
        }
        if(self::$unserializeWarning>=1) {
            echo "Warning: NDarayPhp data is saved using old procedure. We recommend that you resave using the new procedure.\n";
        }
        $buffer = $data->buffer();
        if(get_class($data->service())!==get_class($this->service)) {
            $newBuffer = $this->service->buffer()->Buffer($buffer->count(),$buffer->dtype());
            if($data->service()->serviceLevel()>=Service::LV_ADVANCED &&
                $this->service->serviceLevel()>=Service::LV_ADVANCED) {
                $newBuffer->load($buffer->dump());
            } else {
                $count = $buffer->count();
                for($i=0;$i<$count;$i++) {
                    $newBuffer[$i] = $buffer[$i];
                }
            }
            $buffer = $newBuffer;
        }
        $this->__construct(
            $buffer,
            dtype:$data->dtype(),
            shape:$data->shape(),
            offset:$data->offset(),
            service:$this->service,
        );
    }

    public function __serialize() : array
    {
        $mode = 'machine';
        $buffer = $this->_buffer->dump();
        return [
            'm'=>$mode,
            's'=>$this->_shape,
            'o'=>$this->_offset,
            't'=>$this->_dtype,
            'z'=>count($this->_buffer),
            'b'=>$buffer,
        ];
    }

    public function __unserialize(array $data) : void
    {
        if($this->service===null) {
            if(self::$unserializeWarning>=2) {
                throw new RuntimeException("Please unserialize using the unserialize method.");
            }
            if(self::$unserializeWarning>=1) {
                echo "Warning: NDArrayPhp is not initialized. Please unserialize using the unserialize method.\n";
            }
            $selector = new Selector();
            $this->service = $selector->select();
        }
        $mode = $data['m'];
        $this->_shape = $data['s'];
        $this->_offset = $data['o'];
        $this->_dtype = $data['t'];
        if($mode=='machine' || $mode=='rindow_openblas') {
            //if($this->service->serviceLevel()<Service::LV_ADVANCED) {
            //    throw new RuntimeException('Advanced drivers are not loaded.');
            //}
            $this->_buffer = $this->service->buffer()->Buffer($data['z'],$data['t']);
            $this->_buffer->load($data['b']);
        } elseif($mode=='linear-array') {
            // Compatibility with older specifications
            $this->_buffer = $this->service->buffer()->Buffer($data['z'],$data['t']);
            foreach($data['b'] as $key => $value) {
                $this->_buffer[$key] = $value;
            }
        } else {
            throw new RuntimeException('Illegal save mode: '.$mode);
        }
    }

    public function __clone()
    {
        if($this->service->serviceLevel()>=Service::LV_ADVANCED) {
            $newBuffer = $this->service->buffer()->Buffer(
                count($this->_buffer),$this->_buffer->dtype());
            $newBuffer->load($this->_buffer->dump());
            $this->_buffer = $newBuffer;
        } elseif($this->service->serviceLevel()>=Service::LV_BASIC) {
            $this->_buffer = clone $this->_buffer;
        } else {
            throw new RuntimeException('Unknown buffer type is uncloneable:'.get_class($this->_buffer));
        }
    }
}
