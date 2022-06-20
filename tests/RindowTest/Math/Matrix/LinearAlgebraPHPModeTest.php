<?php
namespace RindowTest\Math\Matrix\LinearAlgebraPHPModeTest;

use PHPUnit\Framework\TestCase;
use Interop\Polite\Math\Matrix\NDArray;
use Rindow\Math\Matrix\MatrixOperator;
use Rindow\Math\Matrix\PhpBlas;
use Rindow\Math\Matrix\PhpLapack;
use Rindow\Math\Matrix\PhpMath;
use ArrayObject;
use SplFixedArray;
use InvalidArgumentException;

if(!class_exists('RindowTest\Math\Matrix\LinearAlgebraTest\Test')) {
    require_once __DIR__.'/LinearAlgebraTest.php';
}
use RindowTest\Math\Matrix\LinearAlgebraTest\Test as ORGTest;

class Test extends ORGTest
{
    public function newMatrixOperator()
    {
        $blas = new PhpBlas();
        $lapack = new PhpLapack();
        $math = new PhpMath();
        $mo = new MatrixOperator($blas,$lapack,$math);
        return $mo;
    }

    public function testTrmmNormal()
    {
        $this->markTestSkipped('Unsuppored function on clblast');
    }

    public function testTrmmTranspose()
    {
        $this->markTestSkipped('Unsuppored function on clblast');
    }

    public function testTrmmUnit()
    {
        $this->markTestSkipped('Unsuppored function on clblast');
    }

    public function testTrmmRight()
    {
        $this->markTestSkipped('Unsuppored function on clblast');
    }

    public function testTrsmNormal()
    {
        $this->markTestSkipped('Unsuppored function on clblast');
    }


    public function testSvdFull1()
    {
        $this->markTestSkipped('Unsuppored function without openblas');
    }

    public function testSvdFull2()
    {
        $this->markTestSkipped('Unsuppored function without openblas');
    }

    public function testSvdSmallVT()
    {
        $this->markTestSkipped('Unsuppored function without openblas');
    }
}
