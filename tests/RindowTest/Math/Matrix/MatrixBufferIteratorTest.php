<?php
namespace RindowTest\Math\Matrix\MatrixBufferIteratorTest;

use PHPUnit\Framework\TestCase;
use Rindow\Math\Matrix\MatrixBufferIterator;

class Test extends TestCase
{
    public function testLinear()
    {
        $shape = [6];
        $skipDims = [];
        $i = new MatrixBufferIterator($shape,$skipDims);
        $keys = [];
        foreach ($i as $value) {
            $keys[] = $value;
        }
        $this->assertEquals([0,1,2,3,4,5],$keys);

    }

    public function testMatrixAll()
    {
        $shape = [3,2];
        $skipDims = [];
        $i = new MatrixBufferIterator($shape,$skipDims);
        $keys = [];
        foreach ($i as $value) {
            $keys[] = $value;
        }
        $this->assertEquals([0,1,2,3,4,5],$keys);
    }

    public function testMatrixwithSkip()
    {
        $shape = [4,3,2];
        $skipDims = [2];
        $i = new MatrixBufferIterator($shape,$skipDims);
        $keys = [];
        foreach ($i as $value) {
            $keys[] = $value;
        }
        $this->assertEquals([0,2,4, 6,8,10, 12,14,16, 18,20,22],$keys);

        $shape = [4,3,2];
        $skipDims = [1];
        $i = new MatrixBufferIterator($shape,$skipDims);
        $keys = [];
        foreach ($i as $value) {
            $keys[] = $value;
        }
        $this->assertEquals([0,1, 6,7, 12,13, 18,19],$keys);
    }
}
