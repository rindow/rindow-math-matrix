<?php
namespace Rindow\Math\Matrix\Drivers\MatlibPHP;

require_once __DIR__.'/../../C.php';
use function Rindow\Math\Matrix\C;
use Interop\Polite\Math\Matrix\NDArray;

class PhpCalcFloat
{
    public function build(float $value) : float
    {
        return $value;
    }

    public function iscomplex(?int $dtype=null) : bool
    {
        return $dtype==NDArray::complex64||$dtype==NDArray::complex128;
    }

    public function iszero(float $value) : bool
    {
        return $value==0;
    }
    
    public function isone(float $value) : bool
    {
        return $value==1;
    }
    
    public function abs(float $value) : float
    {
        return abs($value);
    }

    public function conj(float $value) : float
    {
        return $value;
    }

    public function add(float $x, float $y) : float
    {
        return $x+$y;
    }

    public function sub(float $x, float $y) : float
    {
        return $x-$y;
    }

    public function mul(float $x, float $y) : float
    {
        return $x*$y;
    }

    public function div(float $x, float $y) : float
    {
        if($y==0) {
            return NAN;
        }
        return $x/$y;
    }

    public function scale(float $a, float $x) : float
    {
        return $a*$x;
    }

    public function sqrt(float $x) : float
    {
        return sqrt($x);
    }
}
