<?php
namespace RindowTest\Math\Matrix\Drivers\MatlibPHP;

use Interop\Polite\Math\Matrix\NDArray;
use Interop\Polite\Math\Matrix\BLAS;
use Rindow\Math\Matrix\MatrixOperator;
use Rindow\Math\Matrix\Drivers\Service;
use InvalidArgumentException;
use Rindow\Math\Matrix\Drivers\MatlibPHP\MatlibPhp;
use function Rindow\Math\Matrix\C;

trait Utils
{
    protected $mo;

    public function setUp() : void
    {
        $service = new MatlibPhp();
        $this->mo = new MatrixOperator(service:$service);
    }

    public function getBlas()
    {
        $blas = $this->mo->service()->blas(Service::LV_BASIC);
        return $blas;
    }

    public function alloc(array $shape, ?int $dtype=null) : NDArray
    {
        return $this->mo->zeros($shape,dtype:$dtype);
    }

    public function zeros(array $shape, ?int $dtype=null) : NDArray
    {
        return $this->mo->zeros($shape,dtype:$dtype);
    }

    public function ones(array $shape, ?int $dtype=null) : NDArray
    {
        return $this->mo->ones($shape,dtype:$dtype);
    }

    public function array(mixed $array, ?int $dtype=null) : NDArray
    {
        return $this->mo->array($array,dtype:$dtype);
    }

    protected function isComplex(int $dtype) : bool
    {
        return $this->cistype($dtype);
    }

    protected function toComplex(mixed $array) : mixed
    {
        if(!is_array($array)) {
            if(is_numeric($array)) {
                return C($array,i:0);
            } else {
                return C($array->real,i:$array->imag);
            }
        }
        $cArray = [];
        foreach($array as $value) {
            $cArray[] = $this->toComplex($value);
        }
        return $cArray;
    }

    protected function complementTrans(?bool $trans,?bool $conj,int $dtype) : array
    {
        $trans = $trans ?? false;
        if($this->isComplex($dtype)) {
            $conj = $conj ?? $trans;
        } else {
            $conj = $conj ?? false;
        }
        return [$trans,$conj];
    }

    protected function buildValByType(float|int $value, int $dtype) : float|int|object
    {
        if($this->cistype($dtype)) {
            $value = $this->cbuild($value);
        }
        return $value;
    }

    protected function transToCode(bool $trans,bool $conj) : int
    {
        if($trans) {
            return $conj ? BLAS::ConjTrans : BLAS::Trans;
        } else {
            return $conj ? BLAS::ConjNoTrans : BLAS::NoTrans;
        }
    }

    protected function abs(float|int|object $value) : float
    {
        if(is_numeric($value)) {
            return abs($value);
        }
        $abs = sqrt(($value->real)**2+($value->imag)**2);
        return $abs;
    }

    protected function copy(NDArray $x, ?NDArray $y=null) : NDArray
    {
        $blas = $this->getBlas();

        if($y==null) {
            $y = $this->zeros($x->shape(),dtype:$x->dtype());
        }
        $blas->copy(...$this->translate_copy($x,$y));
        return $y;
    }

    protected function isclose(
        NDArray $a, NDArray $b,
        ?float $rtol=null, ?float $atol=null
        ) : bool
    {
        $blas = $this->getBlas();

        $isCpx = $this->isComplex($a->dtype());
        if($rtol===null) {
            $rtol = $isCpx?C(1e-04):1e-04;
        }
        if($atol===null) {
            $atol = 1e-07;
        }
        if($a->shape()!=$b->shape()) {
            return false;
        }
        // diff = b - a
        $alpha =  $isCpx?C(-1):-1;
        $diffs = $this->copy($b);
        $blas->axpy(...$this->translate_axpy($a,$diffs,$alpha));
        $iDiffMax = $blas->iamax(...$this->translate_amin($diffs));
        $diff = $this->abs($diffs->buffer()[$iDiffMax]);

        // close = atol + rtol * b
        $scalB = $this->copy($b);
        $blas->scal(...$this->translate_scal($rtol,$scalB));
        $iCloseMax = $blas->iamax(...$this->translate_amin($scalB));
        $close = $atol+$this->abs($scalB->buffer()[$iCloseMax]);

        return $diff < $close;
    }

}