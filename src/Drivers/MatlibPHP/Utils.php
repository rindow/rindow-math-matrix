<?php
namespace Rindow\Math\Matrix\Drivers\MatlibPHP;

use InvalidArgumentException;

use Interop\Polite\Math\Matrix\Buffer as BufferInterface;

trait Utils
{
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
    
}