<?php
namespace RindowTest\Math\Matrix\Drivers\MatlibPHP\PhpBlasPHPModeTest;

if(!class_exists('RindowTest\Math\Matrix\Drivers\MatlibPHP\PhpBlasTest\Test')) {
    include __DIR__.'/PhpBlasTest.php';
}
use RindowTest\Math\Matrix\Drivers\MatlibPHP\PhpBlasTest\PhpBlasTest as ORGTest;
use Rindow\Math\Matrix\Drivers\Service;
use Rindow\Math\Matrix\MatrixOperator;

class PhpBlasPHPModeTest extends ORGTest
{
    public function getBlas()
    {
        $blas = $this->mo->service()->blas(Service::LV_BASIC);
        //$blas = $mo->blas();
        return $blas;
    }

    public function testGetConfig()
    {
        $blas = $this->getBlas();
        $this->assertStringStartsWith('PhpBlas',$blas->getConfig());
    }

    public function testGetNumThreads()
    {
        $blas = $this->getBlas();
        $this->assertEquals(1,$blas->getNumThreads());
    }

    public function testGetNumProcs()
    {
        $blas = $this->getBlas();
        $this->assertEquals(1,$blas->getNumProcs());
    }

    public function testGetCorename()
    {
        $blas = $this->getBlas();
        $this->assertTrue(is_string($blas->getCorename()));
    }
}
