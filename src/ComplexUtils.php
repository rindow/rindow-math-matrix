<?php
namespace Rindow\Math\Matrix;

require_once __DIR__.'/C.php';
//use function Rindow\Math\Matrix\C;
use Interop\Polite\Math\Matrix\NDArray;

trait ComplexUtils
{
    protected function cbuild(float $r=null, float $i=null) : object
    {
        return C($r, i:$i);
    }

    protected function crebuild(object $value) : object
    {
        return C($value->real, i:$value->imag);
    }

    protected function cisobject(mixed $value) : bool
    {
        if(!is_object($value)) {
            return false;
        }
        return (property_exists($value, 'real') &&
                property_exists($value, 'imag'));
    }

    protected function cobjecttype(mixed $value) : string
    {
        if(is_object($value)) {
            return get_class($value);
        }
        return gettype($value);
    }

    protected function cistype(int $dtype=null) : bool
    {
        return $dtype==NDArray::complex64||$dtype==NDArray::complex128;
    }

    protected function ciszero(object $value) : bool
    {
        return $value->real==0 && $value->imag==0;
    }
    
    protected function cisone(object $value) : bool
    {
        return $value->real==1 && $value->imag==0;
    }
    
    protected function cabs(object $value) : float
    {
        return sqrt($value->real*$value->real + $value->imag*$value->imag);
    }

    protected function cconj(object $value) : object
    {
        return C($value->real, i:-$value->imag);
    }

    protected function cadd(object $x, object $y) : object
    {
        return C(
            ($x->real+$y->real),
            i:($x->imag+$y->imag)
        );
    }

    protected function csub(object $x, object $y) : object
    {
        return C(
            ($x->real-$y->real),
            i:($x->imag-$y->imag)
        );
    }

    protected function cmul(object $x, object $y) : object
    {
        return C(
            (($x->real*$y->real)-($x->imag*$y->imag)),
            i:(($x->real*$y->imag)+($x->imag*$y->real))
        );
    }

    protected function cdiv(object $x, object $y) : object
    {
        $denominator = $y->real * $y->real + $y->imag * $y->imag;
        if($denominator==0) {
            return C(NAN, i:NAN);
        }
        $real = (($x->real*$y->real)+($x->imag*$y->imag))/$denominator;
        $imag = (($x->imag*$y->real)-($x->real*$y->imag))/$denominator;
        return C($real, i:$imag);
    }

    protected function csqrt(object $x) : object
    {
        $r = sqrt($x->real*$x->real + $x->imag*$x->imag);
        $theta = atan2($x->imag, $x->real) / 2.0;
        return C(
            (sqrt($r)*cos($theta)),
            i:(sqrt($r)*sin($theta))
        );
    }
}
