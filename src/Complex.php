<?php
namespace Rindow\Math\Matrix;

//  typedef struct { float real, imag; } openblas_complex_float;

class Complex
{
    public function __construct(
        public readonly float $real,
        public readonly float $imag,
    ) {
    }

    public function __toString() : string
    {
        $and = ($this->imag>=0)?'+':'';
        return "{$this->real}{$and}{$this->imag}i";
    }
}
