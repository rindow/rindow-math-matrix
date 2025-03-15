<?php
namespace Rindow\Math\Matrix\Drivers\MatlibPHP;

use InvalidArgumentException;

use Interop\Polite\Math\Matrix\Buffer as BufferInterface;
use Interop\Polite\Math\Matrix\NDArray;
use Interop\Polite\Math\Matrix\Buffer;

trait Utils
{
    /** @var array<int,bool> $integerDtypes */
    protected array $integerDtypes = [
        NDArray::int8 => true,
        NDArray::uint8 => true,
        NDArray::int16 => true,
        NDArray::uint16 => true,
        NDArray::int32 => true,
        NDArray::uint32 => true,
        NDArray::int64 => true,
        NDArray::uint64 => true,
    ];

    /** @var array<int,bool> $floatDtypes */
    protected array $floatDtypes = [
        NDArray::float32 => true,
        NDArray::float64 => true,
    ];

    protected function assertShapeParameter(
        string $name, int $n) : void
    {
        if($n<1) {
            throw new InvalidArgumentException("Argument $name must be greater than 0.");
        }
    }
    
    protected function assertVectorBufferSpec(
        string $name, BufferInterface $buffer, int $n, int $offset, int $inc) : void
    {
        if($offset<0) {
            throw new InvalidArgumentException("Argument offset$name must be greater than equals 0.");
        }
        if($inc<1) {
            throw new InvalidArgumentException("Argument inc$name must be greater than 0.");
        }
        if($offset+($n-1)*$inc >= count($buffer)) {
            throw new InvalidArgumentException("Vector specification too large for buffer$name.");
        }
    }

    protected function assertMatrixBufferSpec(
        string $name, BufferInterface $buffer,
        int $m, int $n, int $offset, int $ld) : void
    {
        if($offset<0) {
            throw new InvalidArgumentException("Argument offset$name must be greater than equals 0.");
        }
        if($ld<1) {
            throw new InvalidArgumentException("Argument ld$name must be greater than 0.");
        }
        if($offset+($m-1)*$ld+($n-1) >= count($buffer)) {
            throw new InvalidArgumentException("Matrix specification too large for buffer$name.");
        }
    }
    
    protected function assertBufferSize(
        BufferInterface $buffer,
        int $offset, int $size,
        string $message) : void
    {
        if($size<1 || $offset<0 || count($buffer) < $offset+$size) {
            throw new InvalidArgumentException($message);
        }
    }
    
    protected function isIntegerDtype(int $dtype) : bool
    {
        return array_key_exists($dtype, $this->integerDtypes);
    }
}