<?php
namespace RindowTest\Math\Matrix\MatrixOperatorPhpModeTest;

use PHPUnit\Framework\TestCase;
use Rindow\Math\Matrix\MatrixOperator;
use Rindow\Math\Matrix\Drivers\MatlibPHP\MatlibPhp;
use Rindow\Math\Matrix\Drivers\Service;

if(!class_exists('RindowTest\Math\Matrix\MatrixOperatorTest\MatrixOperatorTest')) {
    require_once __DIR__.'/MatrixOperatorTest.php';
}
use RindowTest\Math\Matrix\MatrixOperatorTest\MatrixOperatorTest as ORGTest;

class MatrixOperatorPhpModeTest extends ORGTest
{
    public function newMatrixOperator()
    {
        $service = new MatlibPhp();
        $mo = new MatrixOperator(service:$service);
        if($service->serviceLevel()!=Service::LV_BASIC) {
            throw new \Exception("the service is invalid.");
        }
        return $mo;
    }
}
