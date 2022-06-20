<?php
namespace Rindow\Math\Matrix;

use Rindow\OpenCL\Buffer as BufferImplement;
use Interop\Polite\Math\Matrix\Buffer;
use LogicException;

class OpenCLBuffer extends BufferImplement implements Buffer
{
    public function count() : int
    {
        return $this->bytes()/$this->value_size();
    }

    public function offsetExists( $offset ) : bool
    {
        throw new LogicException("Unsuppored Operation");
    }

    public function offsetGet( $offset ) : mixed
    {
        throw new LogicException("Unsuppored Operation");
    }

    public function offsetSet( $offset , $value ) : void
    {
        throw new LogicException("Unsuppored Operation");
    }

    public function offsetUnset( $offset ) : void
    {
        throw new LogicException("Unsuppored Operation");
    }
}
