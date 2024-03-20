<?php
namespace RindowTest\Math\Matrix\Drivers\MatlibPHP\PhpBLASFactoryTest;

use PHPUnit\Framework\TestCase;
use Rindow\Math\Matrix\Drivers\MatlibPHP\PhpBLASFactory;
use Rindow\Math\Matrix\Drivers\MatlibPHP\PhpBuffer;
use Rindow\Math\Matrix\Drivers\MatlibPHP\PhpBlas;
use Rindow\Math\Matrix\Drivers\MatlibPHP\PhpLapack;
use Rindow\Math\Matrix\Drivers\MatlibPHP\PhpMath;
use Interop\Polite\Math\Matrix\NDArray;

class PhpBLASFactoryTest extends TestCase
{
    public function newDriverFactory()
    {
        return new PhpBLASFactory();
    }

    public function testName()
    {
        $factory = $this->newDriverFactory();
        $this->assertEquals('phpblas',$factory->name());
    }

    public function testIsAvailable()
    {
        $factory = $this->newDriverFactory();
        $this->assertTrue($factory->isAvailable());
    }

    //public function testExtName()
    //{
    //    $factory = $this->newDriverFactory();
    //    $this->assertEquals('phpblas',$factory->extName());
    //}

    //public function testVersion()
    //{
    //    $factory = $this->newDriverFactory();
    //    $this->assertTrue(is_string($factory->version()));
    //    //var_dump($factory->version());
    //}

    public function testBlas()
    {
        $factory = $this->newDriverFactory();
        $driver = $factory->Blas();
        $this->assertInstanceOf(PhpBlas::class,$driver);
    }

    public function testLapack()
    {
        $factory = $this->newDriverFactory();
        $driver = $factory->Lapack();
        $this->assertInstanceOf(PhpLapack::class,$driver);
    }

    public function testMath()
    {
        $factory = $this->newDriverFactory();
        $driver = $factory->Math();
        $this->assertInstanceOf(PhpMath::class,$driver);
    }

    public function testBuffer()
    {
        $factory = $this->newDriverFactory();
        $size = 2;
        $dtype = NDArray::float32;
        $driver = $factory->Buffer($size,$dtype);
        $this->assertInstanceOf(PhpBuffer::class,$driver);
    }

}
