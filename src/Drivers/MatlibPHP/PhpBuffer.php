<?php
namespace Rindow\Math\Matrix\Drivers\MatlibPHP;

use Interop\Polite\Math\Matrix\Buffer as BufferInterface;
use Interop\Polite\Math\Matrix\NDArray;
use TypeError;
use InvalidArgumentException;
use OutOfRangeException;
use LogicException;
use RuntimeException;
use FFI;
use SplFixedArray;
use Traversable;
use Rindow\Math\Matrix\ComplexUtils;

/**
 * @extends SplFixedArray<mixed>
 */
class PhpBuffer extends SplFixedArray implements BufferInterface
{
    use ComplexUtils;

    /** @var array<int,string> $typeString */
    protected static $typeString = [
        NDArray::bool    => 'uint8_t',
        NDArray::int8    => 'int8_t',
        NDArray::int16   => 'int16_t',
        NDArray::int32   => 'int32_t',
        NDArray::int64   => 'int64_t',
        NDArray::uint8   => 'uint8_t',
        NDArray::uint16  => 'uint16_t',
        NDArray::uint32  => 'uint32_t',
        NDArray::uint64  => 'uint64_t',
        //NDArray::float8  => 'N/A',
        //NDArray::float16 => 'N/A',
        NDArray::float32 => 'float',
        NDArray::float64 => 'double',
        //NDArray::complex16  => 'N/A',
        //NDArray::complex32 => 'N/A',
        NDArray::complex64  => 'complex_float',
        NDArray::complex128 => 'complex_double',
    ];

    /** @var array<int,int> $valueSize */
    protected static $valueSize = [
        NDArray::bool    => 1,
        NDArray::int8    => 1,
        NDArray::int16   => 2,
        NDArray::int32   => 4,
        NDArray::int64   => 8,
        NDArray::uint8   => 1,
        NDArray::uint16  => 2,
        NDArray::uint32  => 4,
        NDArray::uint64  => 8,
        //NDArray::float8  => 'N/A',
        //NDArray::float16 => 'N/A',
        NDArray::float32 => 4,
        NDArray::float64 => 8,
        //NDArray::complex16 => 'N/A',
        //NDArray::complex32 => 'N/A',
        NDArray::complex64 => 8,
        NDArray::complex128  => 16,
    ];

    /** @var array<int,string> $pack */
    protected static $pack = [
        NDArray::bool    => 'C',
        NDArray::int8    => 'c',
        NDArray::int16   => 's',
        NDArray::int32   => 'l',
        NDArray::int64   => 'q',
        NDArray::uint8   => 'C',
        NDArray::uint16  => 'S',
        NDArray::uint32  => 'L',
        NDArray::uint64  => 'Q',
        //NDArray::float8  => 'N/A',
        //NDArray::float16 => 'N/A',
        NDArray::float32 => 'g',
        NDArray::float64 => 'e',
        NDArray::complex64 => 'g',
        NDArray::complex128 => 'e',
    ];

    protected int $dtype;

    public function __construct(int $size, int $dtype)
    {
        if(!isset(self::$typeString[$dtype])) {
            throw new InvalidArgumentException("Invalid data type");
        }
        $this->dtype = $dtype;
        parent::__construct($size);
    }

    protected function isComplex(int $dtype=null) : bool
    {
        $dtype = $dtype ?? $this->dtype;
        return $this->cistype($dtype);
    }

    /**
     * @param array<mixed> $array
     */
    public static function fromArray(array $array, bool $preserveKeys = true) : SplFixedArray
    {
        throw new LogicException("Unsupported operation");
    }

    /**
     * @param array<mixed> $array
     */
    public static function fromArrayWithDtype(array $array, int $dtype) : BufferInterface
    {
        $a = new self(count($array), $dtype);
        foreach($array as $i => $v) {
            if(!is_int($i)) {
                throw new InvalidArgumentException("array must contain only positive integer keys");
            }
            $a[$i] = $v;
        }
        return $a;
    }

    public function offsetSet($index, mixed $value) : void
    {
        if($this->isComplex()) {
            if(!$this->cisobject($value)) {
                throw new InvalidArgumentException("Cannot convert to complex number.: ".$this->cobjecttype($value));
            }
            $value = $this->cbuild($value->real,$value->imag);
        }
        parent::offsetSet($index,$value);
    }

    public function dtype() : int
    {
        return $this->dtype;
    }

    public function value_size() : int
    {
        return $this::$valueSize[$this->dtype];
    }

    public function dump() : string
    {
        $size = count($this);
        $fmt = self::$pack[$this->dtype];
        $string = '';
        if($this->isComplex()) {
            $fmt .= $fmt;
            for($i=0;$i<$size;++$i) {
                $v = $this->offsetGet($i);
                $string .= pack($fmt,$v->real,$v->imag);
            }
        } else {
            for($i=0;$i<$size;++$i) {
                $string .= pack($fmt,$this->offsetGet($i));
            }
        }
        return $string;
    }

    public function load(string $string) : void
    {
        $fmt = self::$pack[$this->dtype].'*';
        $data = unpack($fmt,$string);
        if($data===false) {
            throw new RuntimeException('Unpack error');
        }
        $i = 0;
        $real = 0;
        if($this->isComplex()) {
            foreach($data as $value) {
                if($i%2 == 0) {
                    $real = $value;
                } else {
                    $value = $this->cbuild($real,$value);
                    $this->offsetSet(intdiv($i,2) ,$value);
                }
                ++$i;
            }
        } else {
            foreach($data as $value) {
                $this->offsetSet($i,$value);
                ++$i;
            }
        }
    }
}
