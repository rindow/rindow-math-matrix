<?php
namespace Rindow\Math\Matrix;

use Interop\Polite\Math\Matrix\NDArray;

function C(
    float $r=null,
    float $i=null,
) : Complex {
    $r = $r ?? 0.0;
    $i = $i ?? 0.0;
    return new Complex($r, $i);
}
