<?php
namespace Rindow\Math\Matrix;

use Rindow\OpenCL\Buffer as BufferImplement;
use Interop\Polite\Math\Matrix\Buffer;
use LogicException;

class OpenCLBuffer extends BufferImplement implements Buffer
{
    public function count()
    {
        return $this->bytes()/$this->value_size();
    }

    public function offsetExists( $offset )
    {
        throw new LogicException("Unsuppored Operation");
    }

    public function offsetGet( $offset )
    {
        throw new LogicException("Unsuppored Operation");
    }

    public function offsetSet( $offset , $value )
    {
        throw new LogicException("Unsuppored Operation");
    }

    public function offsetUnset( $offset )
    {
        throw new LogicException("Unsuppored Operation");
    }
}
