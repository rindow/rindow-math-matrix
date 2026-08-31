<?php
namespace Rindow\Math\Matrix\Drivers\MatlibPHP;

require_once __DIR__.'/../../C.php';
use function Rindow\Math\Matrix\C;
use Interop\Polite\Math\Matrix\NDArray;
use Interop\Polite\Math\Matrix\Buffer;

class PhpCalcComplex
{
    public function build(?float $r=null, ?float $i=null) : object
    {
        return C($r, i:$i);
    }

    public function iscomplex(?int $dtype=null) : bool
    {
        return $dtype==NDArray::complex64||$dtype==NDArray::complex128;
    }

    public function iszero(object $value) : bool
    {
        return $value->real==0 && $value->imag==0;
    }
    
    public function isone(object $value) : bool
    {
        return $value->real==1 && $value->imag==0;
    }
    
    public function abs(object $value) : float
    {
        return sqrt($value->real*$value->real + $value->imag*$value->imag);
    }

    public function conj(object $value) : object
    {
        return C($value->real, i:-$value->imag);
    }

    public function add(object $x, object $y) : object
    {
        return C(
            ($x->real+$y->real),
            i:($x->imag+$y->imag)
        );
    }

    public function sub(object $x, object $y) : object
    {
        return C(
            ($x->real-$y->real),
            i:($x->imag-$y->imag)
        );
    }

    public function mul(object $x, object $y) : object
    {
        return C(
            (($x->real*$y->real)-($x->imag*$y->imag)),
            i:(($x->real*$y->imag)+($x->imag*$y->real))
        );
    }

    public function div(object $x, object $y) : object
    {
        $denominator = $y->real * $y->real + $y->imag * $y->imag;
        if($denominator==0) {
            return C(NAN, i:NAN);
        }
        $real = (($x->real*$y->real)+($x->imag*$y->imag))/$denominator;
        $imag = (($x->imag*$y->real)-($x->real*$y->imag))/$denominator;
        return C($real, i:$imag);
    }

    public function scale(float $a, object $x) : object
    {
        return C(
            ($a*$x->real),i:($a*$x->imag)
        );
    }

    public function sqrt(object $x) : object
    {
        $r = sqrt($x->real*$x->real + $x->imag*$x->imag);
        $theta = atan2($x->imag, $x->real) / 2.0;
        return C(
            (sqrt($r)*cos($theta)),
            i:(sqrt($r)*sin($theta))
        );
    }
}
