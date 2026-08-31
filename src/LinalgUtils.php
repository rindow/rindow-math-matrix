<?php
namespace Rindow\Math\Matrix;

use Interop\Polite\Math\Matrix\NDArray;
use Interop\Polite\Math\Matrix\BLAS;

trait LinalgUtils
{
    /** @var array<int> $intTypes */
    protected array $intTypes= [
        NDArray::int8,NDArray::int16,NDArray::int32,NDArray::int64,
        NDArray::uint8,NDArray::uint16,NDArray::uint32,NDArray::uint64,
    ];

    /** @var array<int,string> $dtypeToString */
    protected array $dtypeToString = [
        NDArray::bool=>'bool',
        NDArray::int8=>'int8',   NDArray::uint8=>'uint8',
        NDArray::int16=>'int16', NDArray::uint16=>'uint16',
        NDArray::int32=>'int32', NDArray::uint32=>'uint32',
        NDArray::int64=>'int64', NDArray::uint64=>'uint64',
        NDArray::float16=>'float16',
        NDArray::float32=>'float32', NDArray::float64=>'float64',
        NDArray::complex64=>'complex64', NDArray::complex128=>'complex128',
    ];

    public function isInt(NDArray $value) : bool
    {
        return in_array($value->dtype(),$this->intTypes);
    }

    public function isFloat(NDArray $value) : bool
    {
        $dtype = $value->dtype();
        return $dtype==NDArray::float32||$dtype==NDArray::float64;
    }

    public function setSeed(int $seed) : void
    {
        srand($seed);
    }

    public function randInt(?int $min=null, ?int $max=null) : int
    {
        return rand($min ?? -2147483648, $max ?? 2147483647); // range of int32
    }
    
    public function dtypeToString(int $dtype) : string
    {
        if(!isset($this->dtypeToString[$dtype])) {
            return 'Unknown';
        }
        return $this->dtypeToString[$dtype];
    }

    protected function dataTypeString(mixed $value) : string
    {
        return $this->cobjecttype($value);
    }

    /**
     * @param array<int,string> $shape
     */
    public function shapeToString(array $shape) : string
    {
        return '('.implode(',',$shape).')';
    }

    public function isComplexDtype(int $dtype) : bool
    {
        return $this->cistype($dtype);
    }

    public function isComplexObject(mixed $value) : bool
    {
        return $this->cisobject($value);
    }

    protected function printableShapes(mixed $values) : string
    {
        if(!is_array($values)) {
            if($values instanceof NDArray)
                return '('.implode(',',$values->shape()).')';
            if(is_object($values))
                return '"'.get_class($values).'"';
            if(is_numeric($values) || is_string($values))
                return strval($values);
            return gettype($values);
        }
        $string = '[';
        foreach($values as $value) {
            if($string!='[') {
                $string .= ',';
            }
            $string .= $this->printableShapes($value);
        }
        $string .= ']';
        return $string;
    }

    public function toString(
        NDArray $array,
        ?string $format=null,
        bool|int|null $indent=null) : string
    {
        $shape = $array->shape();
        if(count($shape)==0) {
            $value = $array->toArray();
            if($format) {
                return sprintf($format,$value);
            } else {
                return strval($value);
            }
        }
        $n = array_shift($shape);
        if(!is_numeric($indent) && $indent===true) {
            $indent=1;
        }
        if(count($shape)==0) {
            if($array->dtype()==NDArray::bool) {
                $str = '';
                foreach($array->toArray() as $value) {
                    $str .= ($str==='') ? '[' : ',';
                    $str .= $value ? 'true' : 'false';
                }
                $str .= ']';
                return $str;
            } else {
                if($format) {
                    return '['.implode(',',array_map(function($x) use ($format,$array) {
                            if($array->dtype()==NDArray::complex64||$array->dtype()==NDArray::complex128) {
                                return sprintf($format,$x->real,$x->imag);
                            } else {
                                return sprintf($format,$x);
                            }
                        },$array->toArray())).']';
                } else {
                    return '['.implode(',',$array->toArray()).']';
                }
            }
        }
        $string = '[';
        if($indent) {
            $string .= "\n";
        }
        for($i=0;$i<$n;$i++) {
            if($i!=0) {
                $string .= ',';
                if($indent) {
                    $string .= "\n";
                }
            }
            if($indent) {
                $string .= str_repeat(' ',$indent);
                $string .= $this->toString($array[$i],$format,$indent+1);
            } else {
                $string .= $this->toString($array[$i],$format,$indent);
            }
        }
        if($indent) {
            $string .= "\n";
            $string .= str_repeat(' ',$indent-1);
        }
        $string .= ']';
        return $string;
    }

    protected function isComplex(int $dtype) : bool
    {
        return $this->cistype($dtype);
    }

    protected function transToCode(bool $trans,bool $conj) : int
    {
        if($trans) {
            return $conj ? BLAS::ConjTrans : BLAS::Trans;
        } else {
            return $conj ? BLAS::ConjNoTrans : BLAS::NoTrans;
        }
    }

    protected function buildValByType(float|int $value, int $dtype) : float|int|object
    {
        if($this->cistype($dtype)) {
            $value = $this->cbuild($value);
        }
        return $value;
    }

}
