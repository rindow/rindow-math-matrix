<?php
namespace Rindow\Math\Matrix;

/**
*  Alternate anonymous functions.
*  If you store an anonymous function in an instance variable of MatrixOperator,
*  a recursive reference will occur. An alternative to anonymous functions was
*  needed to prevent memory leaks from recursive references.
*/
class MatrixOpelatorFunctions
{
    // broadCastOperators
    //     '+' =>  [null,  function($x,$y) { return $x + $y; }],
    //     '-' =>  [null,  function($x,$y) { return $x - $y; }],
    //     '*' =>  [null,  function($x,$y) { return $x * $y; }],
    //     '/' =>  [null,  function($x,$y) { return $x / $y; }],
    //     '%' =>  [null,  function($x,$y) { return $x % $y; }],
    //     '**' => [null,  function($x,$y) { return $x ** $y; }],
    //     '==' => [NDArray::bool,function($x,$y) { return ($x == $y); }],
    //     '!=' => [NDArray::bool,function($x,$y) { return $x != $y; }],
    //     '>' =>  [NDArray::bool,function($x,$y) { return $x > $y; }],
    //     '>=' => [NDArray::bool,function($x,$y) { return $x >= $y; }],
    //     '<' =>  [NDArray::bool,function($x,$y) { return $x < $y; }],
    //     '<=' => [NDArray::bool,function($x,$y) { return $x <= $y; }],

    public function add($x,$y) { return $x + $y; } // '+'
    public function sub($x,$y) { return $x - $y; } // '-'
    public function mul($x,$y) { return $x * $y; } // '*'
    public function div($x,$y) { return $x / $y; } // '/'
    public function mod($x,$y) { return $x % $y; } // '%'
    public function pow($x,$y) { return $x ** $y; } // '**'
    public function is_equal($x,$y) { return ($x == $y); } // '=='
    public function is_not_equal($x,$y) { return $x != $y; } // '!='
    public function greater($x,$y) { return $x > $y; } // '>'
    public function greater_or_equal($x,$y) { return $x >= $y; } // '>='
    public function smaller($x,$y) { return $x < $y; } // '<'
    public function smaller_or_equal($x,$y) { return $x <= $y; } // '<='

    // updateOperators
    //     '='  =>  [null,  function($x,$y) { return $y; }],
    //     '+=' =>  [null,  function($x,$y) { return $x + $y; }],
    //     '-=' =>  [null,  function($x,$y) { return ($x - $y); }],
    //     '*=' =>  [null,  function($x,$y) { return $x * $y; }],
    //     '/=' =>  [null,  function($x,$y) { return $x / $y; }],
    //     '%=' =>  [null,  function($x,$y) { return $x % $y; }],
    //     '**=' => [null,  function($x,$y) { return $x ** $y; }],

    public function assign($x,$y) { return $y; } // '='
    public function assign_add($x,$y) { return $x + $y; } // '+='
    public function assign_sub($x,$y) { return ($x - $y); } // '-='
    public function assign_mul($x,$y) { return $x * $y; } // '*='
    public function assign_div($x,$y) { return $x / $y; } // '/='
    public function assign_mod($x,$y) { return $x % $y; } // '%='
    public function assign_pow($x,$y) { return $x ** $y; } // '**='

}
