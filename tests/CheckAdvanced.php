<?php
namespace RindowTest\Math\Matrix\CheckAdvanced;

use PHPUnit\Framework\TestCase;

class CheckAdvanced extends TestCase
{
    public function testAdvanced()
    {
        $mo = new \Rindow\Math\Matrix\MatrixOperator();
        $this->assertTrue($mo->isAdvanced());
    }
}