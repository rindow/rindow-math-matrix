<?php
namespace RindowTest\Math\Matrix\PhpBlasPHPModeTest;

if(!class_exists('RindowTest\Math\Matrix\PhpBlasTest\Test')) {
    include __DIR__.'/PhpBlasTest.php';
}
use RindowTest\Math\Matrix\PhpBlasTest\Test as ORGTest;
use Rindow\Math\Matrix\PhpBlas;
use Rindow\Math\Matrix\MatrixOperator;

class Test extends ORGTest
{
    public function getBlas($mo)
    {
        $blas = new PhpBlas();
        //$blas = $mo->blas();
        return $blas;
    }

    public function testGetConfig()
    {
        $mo = new MatrixOperator();
        $blas = $this->getBlas($mo);
        $this->assertStringStartsWith('PhpBlas',$blas->getConfig());
    }

    public function testGetNumThreads()
    {
        $mo = new MatrixOperator();
        $blas = $this->getBlas($mo);
        $this->assertEquals(1,$blas->getNumThreads());
    }

    public function testGetNumProcs()
    {
        $mo = new MatrixOperator();
        $blas = $this->getBlas($mo);
        $this->assertEquals(1,$blas->getNumProcs());
    }

    public function testGetCorename()
    {
        $mo = new MatrixOperator();
        $blas = $this->getBlas($mo);
        $this->assertTrue(is_string($blas->getCorename()));
    }
}
