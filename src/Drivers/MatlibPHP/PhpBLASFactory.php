<?php
namespace Rindow\Math\Matrix\Drivers\MatlibPHP;

use Interop\Polite\Math\Matrix\Buffer as BufferInterface;
use Rindow\Math\Matrix\Drivers\Driver;
use Rindow\Math\Matrix\Drivers\Service;

class PhpBLASFactory implements Driver
{
    public function isAvailable() : bool
    {
        return true;
    }

    public function name() : string
    {
        return 'phpblas';
    }

    public function Blas(object $blas=null,bool $forceBlas=null) : object
    {
        return new PhpBlas($blas=null,$forceBlas=null);
    }

    public function Lapack(object $blas=null,bool $forceBlas=null) : object
    {
        return new PhpLapack($blas=null,$forceBlas=null);
    }

    public function Math(object $blas=null,bool $forceBlas=null) : object
    {
        return new PhpMath($blas=null,$forceBlas=null);
    }

    public function Buffer(int $size, int $dtype) : BufferInterface
    {
        return new PhpBuffer($size, $dtype);
    }
}