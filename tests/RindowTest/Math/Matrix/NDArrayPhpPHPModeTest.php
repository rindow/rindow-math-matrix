<?php
namespace RindowTest\Math\Matrix\NDArrayPhpPHPModeTest;

use PHPUnit\Framework\TestCase;
use Interop\Polite\Math\Matrix\NDArray;
use Rindow\Math\Matrix\MatrixOperator;
use Rindow\Math\Matrix\Drivers\MatlibPHP\MatlibPhp;
use Rindow\Math\Matrix\Drivers\Service;
use ArrayObject;
use InvalidArgumentException;

if(!class_exists('RindowTest\Math\Matrix\NDArrayPhpTest\NDArrayPhpTest')) {
    require_once __DIR__.'/NDArrayPhpTest.php';
}
use RindowTest\Math\Matrix\NDArrayPhpTest\NDArrayPhpTest as ORGTest;

class NDArrayPhpPHPModeTest extends ORGTest
{
    public function setUp() : void
    {
        $this->service = new MatlibPhp();
        if($this->service->serviceLevel()!=Service::LV_BASIC) {
            throw new \Exception("the service is invalid.");
        }
    }
}
