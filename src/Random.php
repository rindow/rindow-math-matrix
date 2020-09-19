<?php
namespace Rindow\Math\Matrix;

use ArrayObject;
use InvalidArgumentException;
use Interop\Polite\Math\Matrix\BLAS;
use Interop\Polite\Math\Matrix\NDArray;

class Random
{
    protected $mo;
    protected $defaultFloatType;

    public function __construct($matrixOperator,$defaultFloatType)
    {
        $this->mo = $matrixOperator;
        $this->defaultFloatType = $defaultFloatType;
    }

    protected function genRandNormal(float $av, float $sd) : float
    {
        $x=random_int(1,PHP_INT_MAX-1)/PHP_INT_MAX;
        $y=random_int(1,PHP_INT_MAX-1)/PHP_INT_MAX;
        return sqrt(-2*log($x))*cos(2*pi()*$y)*$sd+$av;
    }

    protected function checkSize($n)
    {
        if(is_array($n)) {
            $shape = $n;
        } elseif(is_int($n)) {
            $shape = [$n];
        } else {
            throw new InvalidArgumentException('Type of argument must be int or array as shape.');
        }
        return $shape;
    }

    public function rand($n,$dtype=null) : NDArray
    {
        $shape = $this->checkSize($n);
        if($dtype===null)
            $dtype = $this->defaultFloatType;
        $array = $this->mo->zeros($shape,$dtype);
        $buffer = $array->buffer();
        $size = $array->size();
        for($i=0;$i<$size;$i++) {
            $buffer[$i] = $this->randomInt(PHP_INT_MAX)/PHP_INT_MAX;
        }
        return $array;
    }

    public function randn($n,$dtype=null) : NDArray
    {
        $shape = $this->checkSize($n);
        if($dtype===null)
            $dtype = $this->defaultFloatType;
        $array = $this->mo->zeros($shape,$dtype);
        $buffer = $array->buffer();
        $size = $array->size();
        $av = 0.0;
        $sd = 1.0;
        for($i=0;$i<$size;$i++) {
            $buffer[$i] = $this->genRandNormal($av,$sd);
        }
        return $array;
    }

    public function randomInt(int $max) : int
    {
        return random_int(0,$max);
    }

    public function choice($a,int $size=null, bool $replace=null)
    {
        $arangeFlg = false;
        if(is_int($a)) {
            $a = $this->mo->arange($a);
            $arangeFlg = true;
        } elseif($a instanceof NDArray) {
            if($a->ndim()!=1) {
                throw new InvalidArgumentException('NDArray must be 1-D array.');
            }
        } else {
            throw new InvalidArgumentException('First argument must be int or NDArray.');
        }
        if($size===null) {
            $size = 1;
        } elseif($size<1) {
            throw new InvalidArgumentException('Size argument must be greater than or equal 1.');
        }
        if($replace===null)
            $replace = true;

        if($size==1) {
            $idx = $this->randomInt($a->size()-1);
            return $a[$idx];
        }

        $r = $this->mo->zeros([$size],$a->dtype());
        $sourceSize = $a->size();
        if($replace) {
            for($n=0;$n<$size;$n++) {
                $idx = $this->randomInt($sourceSize-1);
                $r[$n] = $a[$idx];
            }
        } else {
            if($size>$sourceSize) {
                throw new InvalidArgumentException("The total number is smaller than the number of samples");
            }
            if($arangeFlg) {
                $select = $a;
            } else {
                $select = $this->mo->arange($sourceSize);
            }
            for($n=0;$n<$size;$n++) {
                $idx = $this->randomInt($sourceSize-$n-1);
                $r[$n] = $a[$select[$idx]];
                $select[$idx] = $select[$sourceSize-$n-1];
            }
        }
        return $r;
    }
}
