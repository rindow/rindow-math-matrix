<?php
namespace RindowTest\Math\Matrix\MatrixOperatorTest;

use PHPUnit\Framework\TestCase;
use Interop\Polite\Math\Matrix\NDArray;
use Rindow\Math\Matrix\MatrixOperator;
use Rindow\Math\Matrix\Drivers\Selector;
use Rindow\Math\Matrix\Drivers\Service;
use function Rindow\Math\Matrix\C;
use ArrayObject;


class MatrixOperatorTest extends TestCase
{
    public function newMatrixOperator()
    {
        $selector = new Selector();
        $service = $selector->select();
        $mo = new MatrixOperator(service:$service);
        //if($service->serviceLevel()<Service::LV_ADVANCED) {
        //    throw new \Exception("the service is not Advanced.");
        //}
        return $mo;
    }

    public function testCreateFromArray()
    {
        $mo = $this->newMatrixOperator();

        // default dtype
        $nd = $mo->array([[1,2,3],[4,5,6]]);
        $this->assertEquals(NDArray::float32,$nd->dtype());
        $this->assertEquals([[1,2,3],[4,5,6]],$nd->toArray());

        // float64
        $nd = $mo->array([[[1,2],[3,4],[5,6]],[[7,8],[9,10],[11,12]]],NDArray::float64);
        $this->assertEquals(NDArray::float64,$nd->dtype());
        $this->assertEquals([[[1,2],[3,4],[5,6]],[[7,8],[9,10],[11,12]]],$nd->toArray());

        // int32
        $nd = $mo->array([1,2,3],dtype:NDArray::int32);
        $this->assertEquals(NDArray::int32,$nd->dtype());
        $this->assertEquals([1,2,3],$nd->toArray());


        // *** DISCONTINUE auto type select
        // auto type
        // $this->assertEquals(NDArray::int32,$mo->array([1,2,3])->dtype());
        // $this->assertEquals(NDArray::int32,$mo->array([0,0,0])->dtype());
        // $this->assertEquals(NDArray::float32,$mo->array([1.0,2.0,3.0])->dtype());
        // $this->assertEquals(NDArray::float32,$mo->array([0.0,0.0,0.0])->dtype());
        // $this->assertEquals(NDArray::bool,$mo->array([true,false,true])->dtype());
    }

    public function testFull()
    {
        $mo = $this->newMatrixOperator();

        // float64
        $nd = $mo->full([2,3],1.0,NDArray::float64);
        $this->assertEquals(NDArray::float64,$nd->dtype());
        $this->assertEquals([[1.0,1.0,1.0],[1.0,1.0,1.0]],$nd->toArray());

        // int64
        $nd = $mo->full([2,3],1,NDArray::int64);
        $this->assertEquals(NDArray::int64,$nd->dtype());
        $this->assertEquals([[1,1,1],[1,1,1]],$nd->toArray());

        // bool
        $nd = $mo->full([2,3],true,NDArray::bool);
        $this->assertEquals(NDArray::bool,$nd->dtype());
        $this->assertEquals([[true,true,true],[true,true,true]],$nd->toArray());

        // auto type capability is discontinued 
        //$this->assertEquals(NDArray::int32,$mo->full([2,2],1)->dtype());
        //$this->assertEquals(NDArray::int32,$mo->full([2,2],0)->dtype());
        //$this->assertEquals(NDArray::float32,$mo->full([2,2],1.0)->dtype());
        //$this->assertEquals(NDArray::float32,$mo->full([2,2],0.0)->dtype());
        //$this->assertEquals(NDArray::bool,$mo->full([2,2],true)->dtype());
        //$this->assertEquals(NDArray::bool,$mo->full([2,2],false)->dtype());
    }

    public function testCreateZeros()
    {
        $mo = $this->newMatrixOperator();

        $nd = $mo->zeros([2,3]);
        $this->assertEquals(NDArray::float32,$nd->dtype());
        $this->assertEquals([[0,0,0],[0,0,0]],$nd->toArray());

        $nd = $mo->zeros([2,3,2],NDArray::float64);
        $this->assertEquals(NDArray::float64,$nd->dtype());
        $this->assertEquals([[[0,0],[0,0],[0,0]],[[0,0],[0,0],[0,0]]],$nd->toArray());

        $nd = $mo->zeros([3],$dtype=NDArray::int32);
        $this->assertEquals(NDArray::int32,$nd->dtype());
        $this->assertEquals([0,0,0],$nd->toArray());
    }

    public function testArange()
    {
        $mo = $this->newMatrixOperator();
        $this->assertEquals([0,1,2,3,4],$mo->arange(5)->toArray());
        $this->assertEquals([1,2,3,4,5],$mo->arange(5,$start=1)->toArray());
        $this->assertEquals([1,3,5,7,9],$mo->arange(5,$start=1,$step=2)->toArray());
        $this->assertEquals([1.0,1.5,2.0,2.5,3.0],$mo->arange(5,$start=1.0,$step=0.5)->toArray());

        $this->assertEquals(NDArray::int32,$mo->arange(5)->dtype());
        $this->assertEquals(NDArray::float32,$mo->arange(5,$start=1.0)->dtype());
        $this->assertEquals(NDArray::float64,$mo->arange(5,$start=1.0,$step=1.0,$dtype=NDArray::float64)->dtype());
    }

    public function testCopy()
    {
        $mo = $this->newMatrixOperator();

        $nd = $mo->array([[1,2,3],[4,5,6]],dtype:NDArray::float32);
        $copy = $mo->copy($nd);
        $this->assertEquals(NDArray::float32,$copy->dtype());
        $this->assertEquals([[1,2,3],[4,5,6]],$copy->toArray());
        $this->assertNotEquals(
            spl_object_hash($nd->buffer()),
            spl_object_hash($copy->buffer()));

        $this->assertEquals([[1,2,3],[4,5,6]],$nd->toArray());


        $copy = $mo->copy($nd[1]);
        $this->assertEquals(NDArray::float32,$copy->dtype());
        $this->assertEquals([4,5,6],$copy->toArray());
        $this->assertNotEquals(
            spl_object_hash($nd->buffer()),
            spl_object_hash($copy->buffer()));
        $this->assertEquals(0,$copy->offset());
        $this->assertEquals(3,count($copy->buffer()));

        $this->assertEquals([[1,2,3],[4,5,6]],$nd->toArray());
    }

    public function testMatrixMultiply()
    {
        // ######### Numpy compatible ###############
        $mo = $this->newMatrixOperator();

        // Use BLAS
        // 3x3 * 3x3
        $A = $mo->array([[1,2,3],[4,5,6],[7,8,9]],dtype:NDArray::float32);
        $B = $mo->array([[2,3,4],[5,6,7],[8,9,1]],dtype:NDArray::float32);
        $this->assertEquals(
            [[ 36,  42,  21],
             [ 81,  96,  57],
             [126, 150,  93]],
            $mo->cross($A,$B)->toArray());

        $this->assertEquals([[1,2,3],[4,5,6],[7,8,9]],$A->toArray());
        $this->assertEquals([[2,3,4],[5,6,7],[8,9,1]],$B->toArray());

        // Use BLAS
        // 3x1 * 1x3
        $A = $mo->array([[1],[2],[3]],dtype:NDArray::float32);
        $B = $mo->array([[2,3,4]],dtype:NDArray::float32);
        $this->assertEquals(
            [[ 2,  3,  4],
             [ 4,  6,  8],
             [ 6,  9, 12]],
            $mo->cross($A,$B)->toArray());
        $this->assertEquals([[1],[2],[3]],$A->toArray());
        $this->assertEquals([[2,3,4]],$B->toArray());

        // Use BLAS
        // 4x4 * 4x4
        $A = $mo->array([[1,2,3,4],[5,6,7,8],[9,10,11,12],[13,14,15,16]],dtype:NDArray::float32);
        $B = $mo->array([[21,22,23,24],[25,26,27,28],[29,30,31,32],[33,34,35,36]],dtype:NDArray::float32);
        $this->assertEquals(
            [[ 290,  300,  310,  320],
             [ 722,  748,  774,  800],
             [1154, 1196, 1238, 1280],
             [1586, 1644, 1702, 1760]],
            $mo->cross($A,$B)->toArray());

        // Use BLAS
        // 2x2x2 * 2x2
        $A = $mo->array([[[1,2],[3,4]],[[5,6],[7,8]]],dtype:NDArray::float32);
        $B = $mo->array([[1,0],[0,1]],dtype:NDArray::float32);
        $this->assertEquals(
            [[[1, 2],[3, 4]],[[5, 6],[7, 8]]],
            $mo->cross($A,$B)->toArray());

        // 2x2 * 2x2x2
        $A = $mo->array([[1,0],[0,1]],dtype:NDArray::float32);
        $B = $mo->array([[[1,2],[3,4]],[[5,6],[7,8]]],dtype:NDArray::float32);
        $this->assertEquals(
            [[[1, 2],[5, 6]],[[3, 4],[7, 8]]],
            $mo->cross($A,$B)->toArray());

        // Use BLAS
        // 2x2x2 * 2x2
        $A = $mo->array([[[1,2],[3,4]],[[5,6],[7,8]]],dtype:NDArray::float32);
        $B = $mo->array([[4,3],[2,1]],dtype:NDArray::float32);
        $this->assertEquals(
            [[[8, 5],[20, 13]],[[32, 21],[44, 29]]],
            $mo->cross($A,$B)->toArray());

        // 2x2 * 2x2x2
        $A = $mo->array([[4,3],[2,1]],dtype:NDArray::float32);
        $B = $mo->array([[[1,2],[3,4]],[[5,6],[7,8]]],dtype:NDArray::float32);
        $this->assertEquals(
            [[[13, 20],[41, 48]],[[ 5,  8],[17, 20]]],
            $mo->cross($A,$B)->toArray());

        // 2x2 * 4x4x2x4
        $A = $mo->array([[1,0],[0,1]],dtype:NDArray::float32);
        $B = $mo->array([
            [[[1,2,3,4],[5,6,7,8]],[[9,10,11,12],[13,14,15,16]],[[17,18,19,20],[21,22,23,24]],[[25,26,27,28],[29,30,31,32]]],
            [[[31,32,33,34],[35,36,37,38]],[[39,40,41,42],[43,44,45,46]],[[47,48,49,50],[51,52,53,54]],[[55,56,57,58],[59,60,61,62]]],
            [[[63,64,65,66],[67,68,69,70]],[[71,72,73,74],[75,76,77,78]],[[79,80,81,82],[83,84,85,86]],[[87,88,89,90],[91,92,93,94]]],
            [[[95,96,97,98],[99,100,101,102]],[[103,104,105,106],[107,108,109,110]],[[111,112,113,114],[115,116,117,118]],[[119,120,121,122],[123,124,125,126]]]
            ],dtype:NDArray::float32);
        $this->assertEquals(
                  [[[[  1,   2,   3,   4],
                     [  9,  10,  11,  12],
                     [ 17,  18,  19,  20],
                     [ 25,  26,  27,  28]],

                    [[ 31,  32,  33,  34],
                     [ 39,  40,  41,  42],
                     [ 47,  48,  49,  50],
                     [ 55,  56,  57,  58]],

                    [[ 63,  64,  65,  66],
                     [ 71,  72,  73,  74],
                     [ 79,  80,  81,  82],
                     [ 87,  88,  89,  90]],

                    [[ 95,  96,  97,  98],
                     [103, 104, 105, 106],
                     [111, 112, 113, 114],
                     [119, 120, 121, 122]]],


                   [[[  5,   6,   7,   8],
                     [ 13,  14,  15,  16],
                     [ 21,  22,  23,  24],
                     [ 29,  30,  31,  32]],

                    [[ 35,  36,  37,  38],
                     [ 43,  44,  45,  46],
                     [ 51,  52,  53,  54],
                     [ 59,  60,  61,  62]],

                    [[ 67,  68,  69,  70],
                     [ 75,  76,  77,  78],
                     [ 83,  84,  85,  86],
                     [ 91,  92,  93,  94]],

                    [[ 99, 100, 101, 102],
                     [107, 108, 109, 110],
                     [115, 116, 117, 118],
                     [123, 124, 125, 126]]]],

            $mo->cross($A,$B)->toArray());

        // 4x4 * 2x4x2
        $A = $mo->array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]],dtype:NDArray::float32);
        $B = $mo->array([
            [[[1,2],[3,4],[5,6],[7,8]],[[9,10],[11,12],[13,14],[15,16]]],
            [[[17,18],[19,20],[21,22],[23,24]],[[25,26],[27,28],[29,30],[31,32]]]
        ],dtype:NDArray::float32);
        $this->assertEquals(
            [[[[ 1,  2],
               [ 9, 10]],

              [[17, 18],
               [25, 26]]],

             [[[ 3,  4],
               [11, 12]],

              [[19, 20],
               [27, 28]]],

             [[[ 5,  6],
               [13, 14]],

              [[21, 22],
               [29, 30]]],

             [[[ 7,  8],
               [15, 16]],

              [[23, 24],
               [31, 32]]]],
            $mo->cross($A,$B)->toArray());

        // 1x1 * 2x1x2x2
        $A = $mo->array([[1]],dtype:NDArray::float32);
        $B = $mo->array(
            [[[[1,2]],[[3,4]]],[[[5,6]],[[7,8]]]],dtype:NDArray::float32
        );
        $this->assertEquals(
            [[[[1, 2],
               [3, 4]],
              [[5, 6],
               [7, 8]]]],
            $mo->cross($A,$B)->toArray());


        // Use BLAS with Offset
        // 3x3 * 3x3
        $A = $mo->array([[[0,0,0],[0,0,0],[0,0,0]],[[1,2,3],[4,5,6],[7,8,9]]],dtype:NDArray::float32)[1];
        $this->assertEquals(9,$A->offset());
        $B = $mo->array([[[0,0,0],[0,0,0],[0,0,0]],[[2,3,4],[5,6,7],[8,9,1]]],dtype:NDArray::float32)[1];
        $this->assertEquals(9,$B->offset());
        $this->assertEquals(
            [[ 36,  42,  21],
             [ 81,  96,  57],
             [126, 150,  93]],
            $mo->cross($A,$B)->toArray());


        // 2x2 * 2x2x2  without BLAS with offset
        $A = $mo->array([[[0,0],[0,0]],[[1,0],[0,1]]],dtype:NDArray::float32)[1];
        $this->assertEquals(4,$A->offset());
        $B = $mo->array([[[[0,0],[0,0]],[[0,0],[0,0]]],[[[1,2],[3,4]],[[5,6],[7,8]]]],dtype:NDArray::float32)[1];
        $this->assertEquals(8,$B->offset());
        $this->assertEquals(
            [[[1, 2],[5, 6]],[[3, 4],[7, 8]]],
            $mo->cross($A,$B)->toArray());

    }

    public function testVectorTransform()
    {
        $mo = $this->newMatrixOperator();

        $A = $mo->array([[1,2,3],[4,5,6],[7,8,9]],dtype:NDArray::float32);
        $B = $mo->array([2,3,4],dtype:NDArray::float32);
        $this->assertEquals(
            [ 20,  47,  74],
            $mo->cross($A,$B)->toArray());

        $A = $mo->array([[1,2,3],[4,5,6]],dtype:NDArray::float32);
        $B = $mo->array([2,3,4],dtype:NDArray::float32);
        $this->assertEquals(
            [ 20,  47],
            $mo->cross($A,$B)->toArray());

        $A = $mo->array([[[1,2],[3,4]],[[5,6],[7,8]]],dtype:NDArray::float32);
        $B = $mo->array([2,3],dtype:NDArray::float32);
        $this->assertEquals(
            [[ 8,  18],
             [ 28,  38]],
            $mo->cross($A,$B)->toArray());

        $A = $mo->array([[[1,2,3],[4,5,6]],[[7,8,9],[10,11,12]]],dtype:NDArray::float32);
        $B = $mo->array([10,11,12],dtype:NDArray::float32);
        $this->assertEquals(
            [[ 68, 167],[266, 365]],
            $mo->cross($A,$B)->toArray());

        $A = $mo->array(
            [[[1,2],[4,5],[6,7]],
             [[8,9],[10,11],[12,13]],
             [[14,15],[16,17],[18,19]],
             [[20,21],[22,23],[24,25]]],dtype:NDArray::float32);
        $B = $mo->array([1,0],dtype:NDArray::float32);
        $this->assertEquals(
            [[ 1,  4,  6],
             [ 8, 10, 12],
             [14, 16, 18],
             [20, 22, 24]],
            $mo->cross($A,$B)->toArray());

        $A = $mo->array(
            [[[[1,2],[3,4],[5,6]],[[7,8],[9,10],[11,12]],[[13,14],[15,16],[17,18]],[[19,20],[21,22],[23,24]]],
             [[[101,102],[103,104],[105,106]],[[107,108],[109,110],[111,112]],[[113,114],[115,116],[117,118]],[[119,120],[121,122],[123,124]]],
             [[[201,202],[203,104],[205,206]],[[207,208],[209,210],[211,212]],[[213,214],[215,216],[217,218]],[[219,220],[221,222],[223,224]]],
             [[[301,202],[303,304],[305,306]],[[307,308],[309,310],[311,312]],[[313,314],[315,316],[317,318]],[[319,320],[321,322],[323,324]]],
             [[[401,402],[403,404],[405,406]],[[407,408],[409,410],[411,412]],[[413,414],[415,416],[417,418]],[[419,420],[421,422],[423,424]]]]
            ,dtype:NDArray::float32);
        $B = $mo->array([1,0],dtype:NDArray::float32);
        $this->assertEquals(
            [[[  1,   3,   5],[  7,   9,  11],[ 13,  15,  17],[ 19,  21,  23]],
             [[101, 103, 105],[107, 109, 111],[113, 115, 117],[119, 121, 123]],
             [[201, 203, 205],[207, 209, 211],[213, 215, 217],[219, 221, 223]],
             [[301, 303, 305],[307, 309, 311],[313, 315, 317],[319, 321, 323]],
             [[401, 403, 405],[407, 409, 411],[413, 415, 417],[419, 421, 423]]],
            $mo->cross($A,$B)->toArray());


        // With offset
        $A = $mo->array([[[0,0,0],[0,0,0],[0,0,0]],[[1,2,3],[4,5,6],[7,8,9]]],dtype:NDArray::float32)[1];
        $this->assertEquals(9,$A->offset());
        $B = $mo->array([[0,0,0],[2,3,4]],dtype:NDArray::float32)[1];
        $this->assertEquals(3,$B->offset());
        $this->assertEquals(
            [ 20,  47,  74],
            $mo->cross($A,$B)->toArray());

    }


    public function testTranspose()
    {
        $mo = $this->newMatrixOperator();

        // 1D
        $this->assertEquals(
            [1,2,3,4,5,6],
            $mo->transpose($mo->array([1,2,3,4,5,6],dtype:NDArray::float32))->toArray());

        // 2D
        $this->assertEquals(
            [[1,4],
             [2,5],
             [3,6]],
            $mo->transpose($mo->array(
                [[1,2,3],
                 [4,5,6]]
            ,dtype:NDArray::float32))->toArray());

        // 3D
        $this->assertEquals(
            [[[ 1,  7, 13, 19],
              [ 3,  9, 15, 21],
              [ 5, 11, 17, 23]],
             [[ 2,  8, 14, 20],
              [ 4, 10, 16, 22],
              [ 6, 12, 18, 24]]],
            $mo->transpose($mo->array(
                [[[1,2],[3,4],[5,6]],
                 [[7,8],[9,10],[11,12]],
                 [[13,14],[15,16],[17,18]],
                 [[19,20],[21,22],[23,24]]]
            ))->toArray());

        // With Offset

        $A = $mo->array([[0,0,0,0,0,0],[1,2,3,4,5,6]],dtype:NDArray::float32)[1];
        $this->assertEquals(6,$A->offset());
        $this->assertEquals(
            [1,2,3,4,5,6],
            $mo->transpose($A)->toArray());

        $A = $mo->array([[[0,0,0],[0,0,0]],[[1,2,3],[4,5,6]]],dtype:NDArray::float32)[1];
        $this->assertEquals(6,$A->offset());
        $this->assertEquals(
            [[1,4],[2,5],[3,6]],
            $mo->transpose($A)->toArray());

    }


    public function testDot()
    {
        $mo = $this->newMatrixOperator();

        $this->assertEquals(
            133,
            $mo->dot($mo->array([1,2,3,4,5,6],dtype:NDArray::float32),$mo->array([3,4,5,6,7,8],dtype:NDArray::float32)));

        $this->assertEquals(
            133,
            $mo->dot($mo->array([[1,2,3],[4,5,6]],dtype:NDArray::float32),$mo->array([[3,4,5],[6,7,8]],dtype:NDArray::float32)));


        // with offset
        $A = $mo->array([[0,0,0,0,0,0],[1,2,3,4,5,6]],dtype:NDArray::float32)[1];
        $this->assertEquals(6,$A->offset());
        $B = $mo->array([[0,0,0,0,0,0],[3,4,5,6,7,8]],dtype:NDArray::float32)[1];
        $this->assertEquals(6,$B->offset());
        $this->assertEquals(
            133,
            $mo->dot($A,$B));

    }

    public function testAdd()
    {
        $mo = $this->newMatrixOperator();

        $this->assertEquals(
            [ 4,  6,  8, 10, 12, 14],
            $mo->add($mo->array([1,2,3,4,5,6],dtype:NDArray::float32),$mo->array([3,4,5,6,7,8],dtype:NDArray::float32))->toArray());

        $this->assertEquals(
            [[ 4,  6,  8],
             [10, 12, 14]],
            $mo->add($mo->array([[1,2,3],[4,5,6]],dtype:NDArray::float32),$mo->array([[3,4,5],[6,7,8]],dtype:NDArray::float32))->toArray());

        // with offset
        $A = $mo->array([[0,0,0,0,0,0],[1,2,3,4,5,6]],dtype:NDArray::float32)[1];
        $this->assertEquals(6,$A->offset());
        $B = $mo->array([[0,0,0,0,0,0],[3,4,5,6,7,8]],dtype:NDArray::float32)[1];
        $this->assertEquals(6,$B->offset());
        $this->assertEquals(
            [ 4,  6,  8, 10, 12, 14],
            $mo->add($A,$B)->toArray());

    }

    public function testScale()
    {
        $mo = $this->newMatrixOperator();

        $this->assertEquals(
            [ 7, 14, 21,28, 35, 42],
            $mo->scale(7,$mo->array([1,2,3,4,5,6],dtype:NDArray::float32))->toArray());

        $this->assertEquals(
            [[ 7, 14, 21],
             [28, 35, 42]],
            $mo->scale(7,$mo->array([[1,2,3],[4,5,6]],dtype:NDArray::float32))->toArray());

        // with offset
        $A = $mo->array([[0,0,0,0,0,0],[1,2,3,4,5,6]],dtype:NDArray::float32)[1];
        $this->assertEquals(6,$A->offset());
        $this->assertEquals(
            [ 7, 14, 21,28, 35, 42],
            $mo->scale(7,$A)->toArray());
    }

    public function testSumPure()
    {
        $mo = $this->newMatrixOperator();

        $this->assertEquals(
            -1,
            $mo->sum($mo->array([1,2,3,4,-5,-6],dtype:NDArray::float32)));

        $this->assertEquals(
            -1,
            $mo->sum($mo->array([[1,2,3],[4,-5,-6]],dtype:NDArray::float32)));

        // with offset
        $A = $mo->array([[0,0,0,0,0,0],[1,2,3,4,-5,-6]],dtype:NDArray::float32)[1];
        $this->assertEquals(6,$A->offset());
        $this->assertEquals(
            -1,
            $mo->sum($A));

    }

    public function testSumWithAxis()
    {
        $mo = $this->newMatrixOperator();
        $A = $mo->array([[[1,10],[100,1000]],[[10000,100000],[1000000,10000000]]],dtype:NDArray::float32);

        $this->assertEquals(
            [[   10001,   100010],[ 1000100, 10001000]],
            $mo->sum($A,axis:0)->toArray());

        $this->assertEquals(
            [[     101,     1010],[ 1010000, 10100000]],
            $mo->sum($A,axis:1)->toArray());

        $this->assertEquals(
            [[      11,     1100],[  110000, 11000000]],
            $mo->sum($A,axis:2)->toArray());
    }

    public function testASumPure()
    {
        $mo = $this->newMatrixOperator();

        $this->assertEquals(
            21,
            $mo->asum($mo->array([1,2,3,4,-5,-6],dtype:NDArray::float32)));

        $this->assertEquals(
            21,
            $mo->asum($mo->array([[1,2,3],[4,-5,-6]],dtype:NDArray::float32)));

        // with offset
        $A = $mo->array([[0,0,0,0,0,0],[1,2,3,4,-5,-6]],dtype:NDArray::float32)[1];
        $this->assertEquals(6,$A->offset());
        $this->assertEquals(
            21,
            $mo->asum($A));
    }

    public function testAsumWithAxis()
    {
        $mo = $this->newMatrixOperator();
        $A = $mo->array([[[1,-10],[100,-1000]],[[10000,-100000],[1000000,-10000000]]],dtype:NDArray::float32);

        $this->assertEquals(
            [[   10001,   100010],[ 1000100, 10001000]],
            $mo->asum($A,axis:0)->toArray());

        $this->assertEquals(
            [[     101,     1010],[ 1010000, 10100000]],
            $mo->asum($A,axis:1)->toArray());

        $this->assertEquals(
            [[      11,     1100],[  110000, 11000000]],
            $mo->asum($A,axis:2)->toArray());
    }

    public function testMaxPure()
    {
        $mo = $this->newMatrixOperator();

        $this->assertEquals(
            4,
            $mo->max($mo->array([1,2,3,4,-5,-6],dtype:NDArray::float32)));

        $this->assertEquals(
            4,
            $mo->max($mo->array([[1,2,3],[4,-5,-6]],dtype:NDArray::float32)));

        // with offset
        $A = $mo->array([[0,0,0,0,0,0],[1,2,3,4,-5,-6]],dtype:NDArray::float32)[1];
        $this->assertEquals(6,$A->offset());
        $this->assertEquals(
            4,
            $mo->max($A));
    }

    public function testMaxWithAxis()
    {
        $mo = $this->newMatrixOperator();

        $X = $mo->array([[1,2,3],[4,-5,-6]],dtype:NDArray::float32);
        $this->assertEquals([2,3],$X->shape());
        $this->assertEquals(
            [4,2,3],
            $mo->max($X,axis:0)->toArray());

        $X = $mo->array([[1,2,3],[4,-5,-6]],dtype:NDArray::float32);
        $this->assertEquals(
            [3,4],
            $mo->max($X,axis:1)->toArray());

        // with offset
        $X = $mo->array([[[9,9,9],[9,9,9]],[[1,2,3],[4,-5,-6]]],dtype:NDArray::float32);
        $X = $X[1];
        $this->assertEquals([2,3],$X->shape());
        $this->assertEquals(
            [4,2,3],
            $mo->max($X,axis:0)->toArray());

        // with offset
        $X = $mo->array([[[9,9,9],[9,9,9]],[[1,2,3],[4,-5,-6]]],dtype:NDArray::float32);
        $X = $X[1];
        $this->assertEquals(
            [3,4],
            $mo->max($X,axis:1)->toArray());


        $X = $mo->array([[[1,-2],[-3,4]],[[-5,6],[7,-8]]],dtype:NDArray::float32);
        $this->assertEquals(
            [[1, 6],[7, 4]],
            $mo->max($X,axis:0)->toArray());

        $X = $mo->array([[[1,-2],[-3,4]],[[-5,6],[7,-8]]],dtype:NDArray::float32);
        $this->assertEquals(
            [[1, 4],[7, 6]],
            $mo->max($X,axis:1)->toArray());

        $X = $mo->array([[[1,-2],[-3,4]],[[-5,6],[7,-8]]],dtype:NDArray::float32);
        $this->assertEquals(
            [[1, 4],[6, 7]],
            $mo->max($X,axis:2)->toArray());
    }

    public function testArgMaxPure()
    {
        $mo = $this->newMatrixOperator();

        $this->assertEquals(
            3,
            $mo->argMax($mo->array([1,2,3,4,-5,-6],dtype:NDArray::float32)));

        $this->assertEquals(
            3,
            $mo->argMax($mo->array([[1,2,3],[4,-5,-6]],dtype:NDArray::float32)));

        // with offset
        $A = $mo->array([[0,0,0,0,0,0],[1,2,3,4,-5,-6]],dtype:NDArray::float32)[1];
        $this->assertEquals(6,$A->offset());
        $this->assertEquals(
            3,
            $mo->argMax($A));
    }

    public function testArgMaxWithAxis()
    {
        $mo = $this->newMatrixOperator();

        // with axis
        $X = $mo->array([[1,2,3],[4,-5,-6]],dtype:NDArray::float32);
        $this->assertEquals(
            [1,0,0],
            $mo->argMax($X,axis:0)->toArray());

        $X = $mo->array([[1,2,3],[4,-5,-6]],dtype:NDArray::float32);
        $this->assertEquals(
            [2,0],
            $mo->argMax($X,axis:1)->toArray());

        // with offset
        $X = $mo->array([[[0,0,0],[0,0,0]],[[1,2,3],[4,-5,-6]]],dtype:NDArray::float32)[1];
        $this->assertEquals(6,$X->offset());
        $this->assertEquals(
            [1,0,0],
            $mo->argMax($X,axis:0)->toArray());

        $X = $mo->array([[[0,0,0],[0,0,0]],[[1,2,3],[4,-5,-6]]],dtype:NDArray::float32)[1];
        $this->assertEquals(6,$X->offset());
        $this->assertEquals(
            [2,0],
            $mo->argMax($X,axis:1)->toArray());


        $X = $mo->array([[[1,-2],[-3,4]],[[5,6],[-7,-8]]],dtype:NDArray::float32);
        $this->assertEquals(
            [[1, 1],[0, 0]],
            $mo->argMax($X,axis:0)->toArray());

        $X = $mo->array([[[1,-2],[-3,4]],[[5,6],[-7,-8]]],dtype:NDArray::float32);
        $this->assertEquals(
            [[0, 1],[0, 0]],
            $mo->argMax($X,axis:1)->toArray());

        $X = $mo->array([[[1,-2],[-3,4]],[[5,6],[-7,-8]]],dtype:NDArray::float32);
        $this->assertEquals(
            [[0, 1],[1, 0]],
            $mo->argMax($X,axis:2)->toArray());
    }

    public function testAMaxPure()
    {
        $mo = $this->newMatrixOperator();

        $this->assertEquals(
            -5,
            $mo->amax($mo->array([1,2,3,4,-5,-4],dtype:NDArray::float32)));

        $this->assertEquals(
            -5,
            $mo->amax($mo->array([[1,2,3],[4,-5,-4]],dtype:NDArray::float32)));

        // with offset
        $A = $mo->array([[0,0,0,0,0,0],[1,2,3,4,-5,-4]],dtype:NDArray::float32)[1];
        $this->assertEquals(6,$A->offset());
        $this->assertEquals(
            -5,
            $mo->amax($A));
    }

    public function testAmaxWithAxis()
    {
        $mo = $this->newMatrixOperator();
        // with axis

        $X = $mo->array([[1,2,3],[4,-5,-6]],dtype:NDArray::float32);
        $this->assertEquals(
            [4,-5,-6],
            $mo->amax($X,axis:0)->toArray());

        $X = $mo->array([[1,2,3],[4,-5,-6]],dtype:NDArray::float32);
        $this->assertEquals(
            [3,-6],
            $mo->amax($X,axis:1)->toArray());

        $X = $mo->array([[[1,-2],[-3,4]],[[-5,6],[7,-8]]],dtype:NDArray::float32);
        $this->assertEquals(
            [[-5, 6],[7, -8]],
            $mo->amax($X,axis:0)->toArray());

        $X = $mo->array([[[1,-2],[-3,4]],[[-5,6],[7,-8]]],dtype:NDArray::float32);
        $this->assertEquals(
            [[-3, 4],[7, -8]],
            $mo->amax($X,axis:1)->toArray());

        $X = $mo->array([[[1,-2],[-3,4]],[[-5,6],[7,-8]]],dtype:NDArray::float32);
        $this->assertEquals(
            [[-2, 4],[6, -8]],
            $mo->amax($X,axis:2)->toArray());
    }

    public function testArgAMaxPure()
    {
        $mo = $this->newMatrixOperator();

        $this->assertEquals(
            4,
            $mo->argAmax($mo->array([1,2,3,4,-5,-4],dtype:NDArray::float32)));

        $this->assertEquals(
            4,
            $mo->argAmax($mo->array([[1,2,3],[4,-5,-4]],dtype:NDArray::float32)));

        // with offset
        $A = $mo->array([[0,0,0,0,0,0],[1,2,3,4,-5,-4]],dtype:NDArray::float32)[1];
        $this->assertEquals(6,$A->offset());
        $this->assertEquals(
            4,
            $mo->argAmax($A));
    }

    public function testArgAmaxWithAxis()
    {
        $mo = $this->newMatrixOperator();
        // with axis

        $X = $mo->array([[1,2,3],[4,-5,-6]],dtype:NDArray::float32);
        $this->assertEquals(
            [1,1,1],
            $mo->argAmax($X,axis:0)->toArray());

        $X = $mo->array([[1,2,3],[4,-5,-6]],dtype:NDArray::float32);
        $this->assertEquals(
            [2,2],
            $mo->argAmax($X,axis:1)->toArray());

        $X = $mo->array([[[-2,1],[8,-7]],[[3,4],[-5,-6]]],dtype:NDArray::float32);
        $this->assertEquals(
            [[1, 1],[0, 0]],
            $mo->argAmax($X,axis:0)->toArray());

        $X = $mo->array([[[-2,1],[8,-7]],[[3,4],[-5,-6]]],dtype:NDArray::float32);
        $this->assertEquals(
            [[1, 1],[1, 1]],
            $mo->argAmax($X,axis:1)->toArray());

        $X = $mo->array([[[-2,1],[8,-7]],[[3,4],[-5,-6]]],dtype:NDArray::float32);
        $this->assertEquals(
            [[0, 0],[1, 1]],
            $mo->argAmax($X,axis:2)->toArray());
    }

    public function testMinPure()
    {
        $mo = $this->newMatrixOperator();

        $this->assertEquals(
            -6,
            $mo->min($mo->array([1,2,3,4,-5,-6],dtype:NDArray::float32)));

        $this->assertEquals(
            -6,
            $mo->min($mo->array([[1,2,3],[4,-5,-6]],dtype:NDArray::float32)));

        // with offset
        $A = $mo->array([[0,0,0,0,0,0],[1,2,3,4,-5,-6]],dtype:NDArray::float32)[1];
        $this->assertEquals(6,$A->offset());
        $this->assertEquals(
            -6,
            $mo->min($A));
    }

    public function testMinWithAxis()
    {
        $mo = $this->newMatrixOperator();
        // with axis

        $X = $mo->array([[1,2,3],[4,-5,-6]],dtype:NDArray::float32);
        $this->assertEquals(
            [1,-5,-6],
            $mo->min($X,axis:0)->toArray());

        $X = $mo->array([[1,2,3],[4,-5,-6]],dtype:NDArray::float32);
        $this->assertEquals(
            [1,-6],
            $mo->min($X,axis:1)->toArray());

        $X = $mo->array([[[1,-2],[-3,4]],[[-5,6],[7,-8]]],dtype:NDArray::float32);
        $this->assertEquals(
            [[-5, -2],[-3, -8]],
            $mo->min($X,axis:0)->toArray());

        $X = $mo->array([[[1,-2],[-3,4]],[[-5,6],[7,-8]]],dtype:NDArray::float32);
        $this->assertEquals(
            [[-3, -2],[-5, -8]],
            $mo->min($X,axis:1)->toArray());

        $X = $mo->array([[[1,-2],[-3,4]],[[-5,6],[7,-8]]],dtype:NDArray::float32);
        $this->assertEquals(
            [[-2, -3],[-5, -8]],
            $mo->min($X,axis:2)->toArray());
    }

    public function testArgMinPure()
    {
        $mo = $this->newMatrixOperator();

        $this->assertEquals(
            5,
            $mo->argMin($mo->array([1,2,3,4,-5,-6],dtype:NDArray::float32)));

        $this->assertEquals(
            5,
            $mo->argMin($mo->array([[1,2,3],[4,-5,-6]],dtype:NDArray::float32)));

        // with offset
        $A = $mo->array([[0,0,0,0,0,0],[1,2,3,4,-5,-6]],dtype:NDArray::float32)[1];
        $this->assertEquals(6,$A->offset());
        $this->assertEquals(
            5,
            $mo->argMin($A));
    }

    public function testArgMinWithAxis()
    {
        $mo = $this->newMatrixOperator();
        // with axis

        $X = $mo->array([[1,2,3],[4,-5,-6]],dtype:NDArray::float32);
        $this->assertEquals(
            [0, 1, 1],
            $mo->argMin($X,axis:0)->toArray());

        $X = $mo->array([[1,2,3],[4,-5,-6]],dtype:NDArray::float32);
        $this->assertEquals(
            [0, 2],
            $mo->argMin($X,axis:1)->toArray());

        // with offset
        $X = $mo->array([[[0,0,0],[0,0,0]],[[1,2,3],[4,-5,-6]]],dtype:NDArray::float32)[1];
        $this->assertEquals(6,$X->offset());
        $this->assertEquals(
            [0, 1, 1],
            $mo->argMin($X,axis:0)->toArray());

        $X = $mo->array([[[0,0,0],[0,0,0]],[[1,2,3],[4,-5,-6]]],dtype:NDArray::float32)[1];
        $this->assertEquals(6,$X->offset());
        $this->assertEquals(
            [0, 2],
            $mo->argMin($X,axis:1)->toArray());


        $X = $mo->array([[[1,-2],[-3,4]],[[5,6],[-7,-8]]],dtype:NDArray::float32);
        $this->assertEquals(
            [[0, 0],[1, 1]],
            $mo->argMin($X,axis:0)->toArray());

        $X = $mo->array([[[1,-2],[-3,4]],[[5,6],[-7,-8]]],dtype:NDArray::float32);
        $this->assertEquals(
            [[1, 0],[1, 1]],
            $mo->argMin($X,axis:1)->toArray());

        $X = $mo->array([[[1,-2],[-3,4]],[[5,6],[-7,-8]]],dtype:NDArray::float32);
        $this->assertEquals(
            [[1, 0],[0, 1]],
            $mo->argMin($X,axis:2)->toArray());
    }

    public function testAminPure()
    {
        $mo = $this->newMatrixOperator();

        $this->assertEquals(
            1,
            $mo->amin($mo->array([1,2,3,4,-5,-6],dtype:NDArray::float32)));

        $this->assertEquals(
            1,
            $mo->amin($mo->array([[1,2,3],[4,-5,-6]],dtype:NDArray::float32)));

        // with offset
        $A = $mo->array([[0,0,0,0,0,0],[1,2,3,4,-5,-6]],dtype:NDArray::float32)[1];
        $this->assertEquals(6,$A->offset());
        $this->assertEquals(
            1,
            $mo->amin($A));
    }

    public function testAminWithAxis()
    {
        $mo = $this->newMatrixOperator();
        // with axis

        $X = $mo->array([[1,2,3],[4,-5,-6]],dtype:NDArray::float32);
        $this->assertEquals(
            [1,2,3],
            $mo->amin($X,axis:0)->toArray());

        $X = $mo->array([[1,2,3],[4,-5,-6]],dtype:NDArray::float32);
        $this->assertEquals(
            [1,4],
            $mo->amin($X,axis:1)->toArray());

        $X = $mo->array([[[1,-2],[-3,4]],[[-5,6],[7,-8]]],dtype:NDArray::float32);
        $this->assertEquals(
            [[1, -2],[-3, 4]],
            $mo->amin($X,axis:0)->toArray());

        $X = $mo->array([[[1,-2],[-3,4]],[[-5,6],[7,-8]]],dtype:NDArray::float32);
        $this->assertEquals(
            [[1, -2],[-5, 6]],
            $mo->amin($X,axis:1)->toArray());

        $X = $mo->array([[[1,-2],[-3,4]],[[-5,6],[7,-8]]],dtype:NDArray::float32);
        $this->assertEquals(
            [[1, -3],[-5, 7]],
            $mo->amin($X,axis:2)->toArray());
    }

    public function testArgAminPure()
    {
        $mo = $this->newMatrixOperator();

        $this->assertEquals(
            0,
            $mo->argAmin($mo->array([1,2,3,4,-5,-6],dtype:NDArray::float32)));

        $this->assertEquals(
            0,
            $mo->argAmin($mo->array([[1,2,3],[4,-5,-6]],dtype:NDArray::float32)));

        // with offset
        $A = $mo->array([[0,0,0,0,0,0],[1,2,3,4,-5,-6]],dtype:NDArray::float32)[1];
        $this->assertEquals(6,$A->offset());
        $this->assertEquals(
            0,
            $mo->argAmin($A));
    }

    public function testArgAminWithAxis()
    {
        $mo = $this->newMatrixOperator();
        // with axis

        $X = $mo->array([[1,2,3],[4,-5,-6]],dtype:NDArray::float32);
        $this->assertEquals(
            [0,0,0],
            $mo->argAmin($X,axis:0)->toArray());

        $X = $mo->array([[1,2,3],[4,-5,-6]],dtype:NDArray::float32);
        $this->assertEquals(
            [0,0],
            $mo->argAmin($X,axis:1)->toArray());


        $X = $mo->array([[[-2,1],[8,-7]],[[3,4],[-5,-6]]],dtype:NDArray::float32);
        $this->assertEquals(
            [[0, 0],[1, 1]],
            $mo->argAmin($X,axis:0)->toArray());

        $X = $mo->array([[[-2,1],[8,-7]],[[3,4],[-5,-6]]],dtype:NDArray::float32);
        $this->assertEquals(
            [[0, 0],[0, 0]],
            $mo->argAmin($X,axis:1)->toArray());

        $X = $mo->array([[[-2,1],[8,-7]],[[3,4],[-5,-6]]],dtype:NDArray::float32);
        $this->assertEquals(
            [[1, 1],[0, 0]],
            $mo->argAmin($X,axis:2)->toArray());
    }

    public function testMeanPure()
    {
        $mo = $this->newMatrixOperator();

        $this->assertEquals(
            3.5,
            $mo->mean($mo->array([1,2,3,4,5,6],dtype:NDArray::float64)));

        $this->assertEquals(
            3.5,
            $mo->mean($mo->array([[1,2,3],[4,5,6]],dtype:NDArray::float64)));

        // with offset
        $A = $mo->array([[0,0,0,0,0,0],[1,2,3,4,5,6]],dtype:NDArray::float64)[1];
        $this->assertEquals(6,$A->offset());
        $this->assertEquals(
            3.5,
            $mo->mean($A));
    }

    public function testMeanWithAxis()
    {
        $mo = $this->newMatrixOperator();
        $A = $mo->array([[[1,10],
                          [100,1000]],
                         [[10000,100000],
                          [1000000,10000000]]],dtype:NDArray::float32);
        $this->assertEquals([2,2,2],$A->shape());
        $this->assertEquals(
            [[   5000.5,   50005],[ 500050, 5000500]],
            $mo->mean($A,axis:0)->toArray());

        $this->assertEquals(
            [[     50.5,     505],[ 505000, 5050000]],
            $mo->mean($A,axis:1)->toArray());

        $this->assertEquals(
            [[      5.5,     550],[  55000, 5500000]],
            $mo->mean($A,axis:2)->toArray());
    }

    public function testApplyFunction()
    {
        $mo = $this->newMatrixOperator();

        $A = $mo->array([1,4,9,16,25,36],dtype:NDArray::float32);
        $this->assertEquals(
            [1,2,3,4,5,6],
            $mo->f('sqrt',$A)->toArray());

        $this->assertEquals(
            [1,4,9,16,25,36],
            $A->toArray()
        );

        $this->assertEquals(
            [2,5,10,17,26,37],
            $mo->f(function($x,$b){return $x+$b;},$A,1)->toArray());

    }

    public function testUpdateByFunction()
    {
        $mo = $this->newMatrixOperator();

        $A = $mo->array([1,4,9,16,25,36],dtype:NDArray::float32);
        $this->assertEquals(
            [1,2,3,4,5,6],
            $mo->u($A,'sqrt')->toArray());
        $this->assertEquals(
            [1,2,3,4,5,6],
            $A->toArray()
        );

        $A = $mo->array([1,4,9,16,25,36],dtype:NDArray::float32);
        $this->assertEquals(
            [2,5,10,17,26,37],
            $mo->u($A,function($x,$b){return $x+$b;},1)->toArray()
        );
        $this->assertEquals(
            [2,5,10,17,26,37],
            $A->toArray()
        );
    }

    public function testPos2Index()
    {
        $mo = $this->newMatrixOperator();

        $r0 = 3;
        $r1 = 4;
        $r2 = 5;
        $shape = [$r0,$r1,$r2];
        $p0 = 2;
        $p1 = 3;
        $p2 = 4;
        $pos = $p0*($r1*$r2) + $p1*($r2) + $p2;
        $this->assertEquals([$p0,$p1,$p2],$mo->pos2index($pos,$shape));
    }

    public function testProjection()
    {
        $mo = $this->newMatrixOperator();

        $X = $mo->array([[1,2,3,4],[5,6,7,8],[9,10,11,12],[13,14,15,16]],dtype:NDArray::float32);
        $this->assertEquals([5,6,7,8],$mo->projection($X,[1,-1])->toArray());
        $this->assertEquals([2,6,10,14],$mo->projection($X,[-1,1])->toArray());
        $this->assertEquals([9,10,11,12],$mo->projection($X,[2,-1])->toArray());
        $this->assertEquals([3,7,11,15],$mo->projection($X,[-1,2])->toArray());

        $X = $mo->array([[[1,2,3],[4,5,6],[7,8,9]],
                         [[10,11,12],[13,14,15],[16,17,18]],
                         [[19,20,21],[22,23,24],[25,26,27]]],dtype:NDArray::float32);
        $this->assertEquals([13,14,15],$mo->projection($X,[1,1,-1])->toArray());
        $this->assertEquals([11,14,17],$mo->projection($X,[1,-1,1])->toArray());
        $this->assertEquals([5,14,23],$mo->projection($X,[-1,1,1])->toArray());
    }

    public function testBroadCastOpratorsFloat()
    {
        $mo = $this->newMatrixOperator();
        // Matrix + Numeric
        $X = $mo->array([[1,2],[3,4]],dtype:NDArray::float64);
        $this->assertEquals([[2,3],[4,5]],$mo->op($X,'+',1)->toArray());
        $this->assertEquals([[0,1],[2,3]],$mo->op($X,'-',1)->toArray());
        $this->assertEquals([[2,4],[6,8]],$mo->op($X,'*',2)->toArray());
        $this->assertEquals([[0.5,1],[1.5,2]],$mo->op($X,'/',2)->toArray());

        $this->assertEquals([[1,0],[1,0]],$mo->op($X,'%',2)->toArray());
        $this->assertEquals([[1,4],[9,16]],$mo->op($X,'**',2)->toArray());
        $this->assertEquals([[false,true],[false,false]],$mo->op($X,'==',2)->toArray());
        $this->assertEquals([[true,false],[true,true]],$mo->op($X,'!=',2)->toArray());
        $this->assertEquals([[false,false],[true,true]],$mo->op($X,'>',2)->toArray());
        $this->assertEquals([[false,true],[true,true]],$mo->op($X,'>=',2)->toArray());
        $this->assertEquals([[true,false],[false,false]],$mo->op($X,'<',2)->toArray());
        $this->assertEquals([[true,true],[false,false]],$mo->op($X,'<=',2)->toArray());

        // Numeric + Matrix
        $X = $mo->array([[1,2],[3,4]],dtype:NDArray::float64);
        $this->assertEquals([[2,3],[4,5]],$mo->op(1,'+',$X)->toArray());
        $this->assertEquals([[0,-1],[-2,-3]],$mo->op(1,'-',$X)->toArray());
        $this->assertEquals([[2,4],[6,8]],$mo->op(2,'*',$X)->toArray());
        $this->assertEquals([[12,6],[4,3]],$mo->op(12,'/',$X)->toArray());
        $this->assertEquals([[0,1],[1,3]],$mo->op(7,'%',$X)->toArray());
        $this->assertEquals([[2,4],[8,16]],$mo->op(2,'**',$X)->toArray());
        $this->assertEquals([[false,true],[false,false]],$mo->op(2,'==',$X)->toArray());
        $this->assertEquals([[true,false],[true,true]],$mo->op(2,'!=',$X)->toArray());
        $this->assertEquals([[true,false],[false,false]],$mo->op(2,'>',$X)->toArray());
        $this->assertEquals([[true,true],[false,false]],$mo->op(2,'>=',$X)->toArray());
        $this->assertEquals([[false,false],[true,true]],$mo->op(2,'<',$X)->toArray());
        $this->assertEquals([[false,true],[true,true]],$mo->op(2,'<=',$X)->toArray());

        // Matrix + Matrix
        $X = $mo->array([[2.0,4.0],[6.0,8.0]],dtype:NDArray::float64);
        $Y = $mo->array([[1.0,2.0],[3.0,4.0]],dtype:NDArray::float64);

        $this->assertEquals([[3,6],[9,12]],$mo->op($X,'+',$Y)->toArray());
        $this->assertEquals([[1,2],[3,4]],$mo->op($X,'-',$Y)->toArray());
        $this->assertEquals([[2,8],[18,32]],$mo->op($X,'*',$Y)->toArray());
        $this->assertEquals([[2,2],[2,2]],$mo->op($X,'/',$Y)->toArray());
        $this->assertEquals([[0,0],[0,0]],$mo->op($X,'%',$Y)->toArray());
        $this->assertEquals([[2,16],[216,4096]],$mo->op($X,'**',$Y)->toArray());
        $this->assertEquals([[false,false],[false,false]],$mo->op($X,'==',$Y)->toArray());
        $this->assertEquals([[true,true],[true,true]],$mo->op($X,'!=',$Y)->toArray());
        $this->assertEquals([[true,true],[true,true]],$mo->op($X,'>',$Y)->toArray());
        $this->assertEquals([[true,true],[true,true]],$mo->op($X,'>=',$Y)->toArray());
        $this->assertEquals([[false,false],[false,false]],$mo->op($X,'<',$Y)->toArray());
        $this->assertEquals([[false,false],[false,false]],$mo->op($X,'<=',$Y)->toArray());
    }

    public function testBroadCastOpratorsInteger()
    {
        $mo = $this->newMatrixOperator();
        // Matrix + Numeric
        $X = $mo->array([[1,2],[3,4]],dtype:NDArray::int64);
        $this->assertEquals([[2,3],[4,5]],$mo->op($X,'+',1)->toArray());
        $this->assertEquals([[0,1],[2,3]],$mo->op($X,'-',1)->toArray());
        $this->assertEquals([[2,4],[6,8]],$mo->op($X,'*',2)->toArray());
        $Xtmp = $mo->array([[2,4],[6,8]],dtype:NDArray::int64);
        $this->assertEquals([[1,2],[3,4]],$mo->op($Xtmp,'/',2)->toArray());

        $this->assertEquals([[1,0],[1,0]],$mo->op($X,'%',2)->toArray());
        $this->assertEquals([[1,4],[9,16]],$mo->op($X,'**',2)->toArray());
        $this->assertEquals([[false,true],[false,false]],$mo->op($X,'==',2)->toArray());
        $this->assertEquals([[true,false],[true,true]],$mo->op($X,'!=',2)->toArray());
        $this->assertEquals([[false,false],[true,true]],$mo->op($X,'>',2)->toArray());
        $this->assertEquals([[false,true],[true,true]],$mo->op($X,'>=',2)->toArray());
        $this->assertEquals([[true,false],[false,false]],$mo->op($X,'<',2)->toArray());
        $this->assertEquals([[true,true],[false,false]],$mo->op($X,'<=',2)->toArray());

        // Numeric + Matrix
        $X = $mo->array([[1,2],[3,4]],dtype:NDArray::int64);
        $this->assertEquals([[2,3],[4,5]],$mo->op(1,'+',$X)->toArray());
        $this->assertEquals([[0,-1],[-2,-3]],$mo->op(1,'-',$X)->toArray());
        $this->assertEquals([[2,4],[6,8]],$mo->op(2,'*',$X)->toArray());
        $this->assertEquals([[12,6],[4,3]],$mo->op(12,'/',$X)->toArray());
        $this->assertEquals([[0,1],[1,3]],$mo->op(7,'%',$X)->toArray());
        $this->assertEquals([[2,4],[8,16]],$mo->op(2,'**',$X)->toArray());
        $this->assertEquals([[false,true],[false,false]],$mo->op(2,'==',$X)->toArray());
        $this->assertEquals([[true,false],[true,true]],$mo->op(2,'!=',$X)->toArray());
        $this->assertEquals([[true,false],[false,false]],$mo->op(2,'>',$X)->toArray());
        $this->assertEquals([[true,true],[false,false]],$mo->op(2,'>=',$X)->toArray());
        $this->assertEquals([[false,false],[true,true]],$mo->op(2,'<',$X)->toArray());
        $this->assertEquals([[false,true],[true,true]],$mo->op(2,'<=',$X)->toArray());

        // Matrix + Matrix
        $X = $mo->array([[2.0,4.0],[6.0,8.0]],dtype:NDArray::int64);
        $Y = $mo->array([[1.0,2.0],[3.0,4.0]],dtype:NDArray::int64);

        $this->assertEquals([[3,6],[9,12]],$mo->op($X,'+',$Y)->toArray());
        $this->assertEquals([[1,2],[3,4]],$mo->op($X,'-',$Y)->toArray());
        $this->assertEquals([[2,8],[18,32]],$mo->op($X,'*',$Y)->toArray());
        $this->assertEquals([[2,2],[2,2]],$mo->op($X,'/',$Y)->toArray());
        $this->assertEquals([[0,0],[0,0]],$mo->op($X,'%',$Y)->toArray());
        $this->assertEquals([[2,16],[216,4096]],$mo->op($X,'**',$Y)->toArray());
        $this->assertEquals([[false,false],[false,false]],$mo->op($X,'==',$Y)->toArray());
        $this->assertEquals([[true,true],[true,true]],$mo->op($X,'!=',$Y)->toArray());
        $this->assertEquals([[true,true],[true,true]],$mo->op($X,'>',$Y)->toArray());
        $this->assertEquals([[true,true],[true,true]],$mo->op($X,'>=',$Y)->toArray());
        $this->assertEquals([[false,false],[false,false]],$mo->op($X,'<',$Y)->toArray());
        $this->assertEquals([[false,false],[false,false]],$mo->op($X,'<=',$Y)->toArray());
    }

    public function testNTimesShapeBroadCastOprators()
    {
        $mo = $this->newMatrixOperator();
        // Matrix + Matrix (N times shape)
        $X = $mo->array([[[1,2],[3,4]],[[5,6],[7,8]]],dtype:NDArray::float32);
        $Y = $mo->array([10,100],dtype:NDArray::float32);
        $this->assertEquals([[[11,102],[13,104]],[[15,106],[17,108]]],$mo->op($X,'+',$Y)->toArray());
        $this->assertEquals([[[11,102],[13,104]],[[15,106],[17,108]]],$mo->op($Y,'+',$X)->toArray());

        $X = $mo->array([[[[1,2],[3,4]],[[5,6],[7,8]]],[[[11,12],[13,14]],[[15,16],[17,18]]]],dtype:NDArray::float32);
        $Y = $mo->array([100,1000],dtype:NDArray::float32);
        $this->assertEquals(
            [[[[101,1002],[103,1004]],[[105,1006],[107,1008]]],
             [[[111,1012],[113,1014]],[[115,1016],[117,1018]]]]
            ,$mo->op($X,'+',$Y)->toArray());
        $this->assertEquals(
            [[[[101,1002],[103,1004]],[[105,1006],[107,1008]]],
             [[[111,1012],[113,1014]],[[115,1016],[117,1018]]]]
            ,$mo->op($Y,'+',$X)->toArray());
    }

    public function testSelectByMask()
    {
        $mo = $this->newMatrixOperator();

        // Select by Mask
        $X = $mo->array([[-1,2],[-3,4]],
            dtype:NDArray::float32);

        $MASK = $mo->op($X,'>',0);
        $this->assertEquals([2,4],$mo->select($X,$MASK)->toArray());

        $MASK = $mo->op($X,'>=',-1);
        $this->assertEquals([-1,2,4],$mo->select($X,$MASK)->toArray());
    }

    public function testSelectByIndex()
    {
        $mo = $this->newMatrixOperator();

        // Select 1D Matrix by the 1D indexing Matrix
        $X = $mo->array(
            [100,101,102,103,104,105,106,107,108,109,110,111],
            dtype:NDArray::float32
        );
        $MASK = $mo->array(
             [2, 1, 0, 3, 4, 5],dtype:NDArray::int32);
        $this->assertEquals(
            [102,101,100,103,104,105],
            $mo->select($X,$MASK)->toArray());

        // Select 1D Matrix by the 2D indexing Matrix
        $MASK = $mo->array(
             [[2, 1, 0], [3, 4, 5]],dtype:NDArray::int32);
        $this->assertEquals(
            [[102,101,100],[103,104,105]],
            $mo->select($X,$MASK)->toArray());

        // Select 2D Matrix by the 2D indexing Matrix
        $X = $mo->array(
            [[ 0,  1,  2],
             [ 3,  4,  5],
             [ 6,  7,  8],
             [ 9, 10, 11]],
             dtype:NDArray::float32
        );
        $MASK = $mo->array(
            [[0, 1, 2],
             [2, 1, 0]],dtype:NDArray::int32);
        $this->assertEquals([4,3],$X->shape());
        $this->assertEquals([2,3],$MASK->shape());

        $R = $mo->select($X,$MASK);
        $this->assertEquals([2,3,3],$R->shape());
        $this->assertEquals(
            [[[0, 1, 2],
              [3, 4, 5],
              [6, 7, 8]],

             [[6, 7, 8],
              [3, 4, 5],
              [0, 1, 2]]],
            $R->toArray());

        // Select By Multi indexing Matrix
        $X = $mo->array(
            [[100,101,102],[103,104,105],[106,107,108],[109,110,111]],
            dtype:NDArray::float32
        );
        $MASK0 = $mo->array(
            [2, 0, 3],dtype:NDArray::int32);
        $MASK1 = $mo->array(
            [1, 2, 0],dtype:NDArray::int32);
        $this->assertEquals(
            [107, 102, 109],
            $mo->select($X,$MASK0,$MASK1)->toArray());

        // Select high rank Matrix By Multi 1D indexing Matrix
        $X = $mo->array(
            [[[100,101],[102,103]],[[104,105],[106,107]],
             [[108,109],[110,111]],[[112,113],[114,115]]],
             dtype:NDArray::float32
        );
        $MASK0 = $mo->array(
            [1, 0, 0],dtype:NDArray::int32);
        $MASK1 = $mo->array(
            [0, 1, 0],dtype:NDArray::int32);
        $this->assertEquals(
            [[104, 105],
             [102, 103],
             [100, 101]],
            $mo->select($X,$MASK0,$MASK1)->toArray());

        // Select 2D Matrix By Multi 2D indexing Matrix
        $X = $mo->array(
            [[100,101,102],[103,104,105],[106,107,108],[109,110,111]],
            dtype:NDArray::float32
        );
        $MASK0 = $mo->array(
            [[0, 1],[2, 3]],dtype:NDArray::int32);
        $MASK1 = $mo->array(
            [[0, 1],[2, 0]],dtype:NDArray::int32);
        $this->assertEquals(
            [[100, 104],
             [108, 109]],
            $mo->select($X,$MASK0,$MASK1)->toArray());

        // Select high rank Matrix By Multi 2D indexing Matrix
        $X = $mo->array(
            [[[100,101],[102,103]],[[104,105],[106,107]],
             [[108,109],[110,111]],[[112,113],[114,115]]],
             dtype:NDArray::float32
        );
        $MASK0 = $mo->array(
            [[1, 0],[0, 1]],dtype:NDArray::int32);
        $MASK1 = $mo->array(
            [[0, 1],[1, 0]],dtype:NDArray::int32);
        $this->assertEquals(
            [[[104, 105],
              [102, 103]],
             [[102, 103],
              [104, 105]]],
            $mo->select($X,$MASK0,$MASK1)->toArray());
    }

    public function testUdateByMask()
    {
        $mo = $this->newMatrixOperator();

        $X = $mo->array([[-1,2],[-3,4]]);
        $mo->update($X,'=',0,$mo->op($X,'<',0));
        $this->assertEquals([[0,2],[0,4]],$X->toArray());

        $X = $mo->array([[-1,2],[-3,4]]);
        $mo->update($X,'=',0,$mo->op($X,'<',-1));
        $this->assertEquals([[-1,2],[0,4]],$X->toArray());

        $X = $mo->array([[-1,2],[-3,4]]);
        $mo->update($X,'+=',2,$mo->op($X,'<',0));
        $this->assertEquals([[1,2],[-1,4]],$X->toArray());

        $X = $mo->array([[-1,2],[-3,4]]);
        $mo->update($X,'-=',2,$mo->op($X,'<',0));
        $this->assertEquals([[-3,2],[-5,4]],$X->toArray());

        $X = $mo->array([[-1,2],[-3,4]]);
        $mo->update($X,'*=',2,$mo->op($X,'<',0));
        $this->assertEquals([[-2,2],[-6,4]],$X->toArray());

        $X = $mo->array([[-1,2],[-3,4]],dtype:NDArray::float64);
        $mo->update($X,'/=',2,$mo->op($X,'<',0));
        $this->assertEquals([[-0.5,2],[-1.5,4]],$X->toArray());

        $X = $mo->array([[-1,2],[-3,4]]);
        $mo->update($X,'%=',2,$mo->op($X,'<',0));
        $this->assertEquals([[-1,2],[-1,4]],$X->toArray());

        $X = $mo->array([[-1,2],[-3,4]]);
        $mo->update($X,'**=',2,$mo->op($X,'<',0));
        $this->assertEquals([[1,2],[9,4]],$X->toArray());

    }

    public function testUdateByMatrix()
    {
        $mo = $this->newMatrixOperator();

        // Update By Multi indexing Matrix
        $X = $mo->array(
            [[100,101,102],[103,104,105],[106,107,108],[109,110,111]]
        );
        $MASK0 = $mo->array(
            [2, 0, 3],dtype:NDArray::int32);
        $MASK1 = $mo->array(
            [1, 2, 0],dtype:NDArray::int32);
        $mo->update($X,'+=',1000,$MASK0,$MASK1);
        $this->assertEquals(
            [[100,101,1102],[103,104,105],[106,1107,108],[1109,110,111]],
            $X->toArray());

        // Update high rank Matrix By Multi 1D indexing Matrix
        $X = $mo->array(
            [[[100,101],[102,103]],[[104,105],[106,107]],
             [[108,109],[110,111]],[[112,113],[114,115]]]
        );
        $MASK0 = $mo->array(
            [1, 0, 0],dtype:NDArray::int32);
        $MASK1 = $mo->array(
            [0, 1, 0],dtype:NDArray::int32);
        $mo->update($X,'+=',1000,$MASK0,$MASK1);
        $this->assertEquals(
            [[[1100,1101],[1102,1103]],[[1104,1105],[106,107]],
             [[108,109],[110,111]],[[112,113],[114,115]]],
            $X->toArray());

        // Update 2D Matrix By Multi 2D indexing Matrix
        $X = $mo->array(
            [[100,101,102],[103,104,105],[106,107,108],[109,110,111]]
        );
        $MASK0 = $mo->array(
            [[0, 1],[2, 3]],dtype:NDArray::int32);
        $MASK1 = $mo->array(
            [[0, 1],[2, 0]],dtype:NDArray::int32);
        $mo->update($X,'+=',1000,$MASK0,$MASK1);
        $this->assertEquals(
            [[1100,101,102],[103,1104,105],[106,107,1108],[1109,110,111]],
            $X->toArray());

        // Update high rank Matrix By Multi 2D indexing Matrix
        $X = $mo->array(
            [[[100,101],[102,103]],[[104,105],[106,107]],[[108,109],[111,112]]]
        );
        $MASK0 = $mo->array(
            [[1, 0],[0, 1]],dtype:NDArray::int32);
        $MASK1 = $mo->array(
            [[0, 1],[0, 1]],dtype:NDArray::int32);
        $mo->update($X,'+=',1000,$MASK0,$MASK1);
        $this->assertEquals(
            [[[1100,1101],[1102,1103]],[[1104,1105],[1106,1107]],[[108,109],[111,112]]],
            $X->toArray());


        // Update may times
        // ******** CAUTION *******
        // Not compatible to numpy
        // ************************
        $X = $mo->array(
            [[100,101,102],[103,104,105]]
        );
        $MASK0 = $mo->array(
            [0, 0, 0],dtype:NDArray::int32);
        $MASK1 = $mo->array(
            [0, 0, 0],dtype:NDArray::int32);
        $mo->update($X,'+=',1000,$MASK0,$MASK1);
        $this->assertEquals(
            [[3100,101,102],[103,104,105]],
            $X->toArray());

    }

    public function testAstype()
    {
        $mo = $this->newMatrixOperator();
        $X = $mo->array(
            [[100,101,102],[103,104,105]],
            dtype:NDArray::int32
        );
        $Y = $mo->astype($X,NDArray::float64);
        $this->assertEquals(NDArray::int32,$X->dtype());
        $this->assertEquals(NDArray::float64,$Y->dtype());

        $this->assertEquals(NDArray::int32,$X->buffer()->dtype());
        $this->assertEquals(NDArray::float64,$Y->buffer()->dtype());
    }

    public function testDtypeComplex()
    {
        $mo = $this->newMatrixOperator();
        $int8 = $mo->ones([1],NDArray::int8);
        $int32 = $mo->ones([1],NDArray::int32);
        $float32 = $mo->ones([1],NDArray::float32);
        $float64 = $mo->ones([1],NDArray::float64);
        $int8_2D = $mo->ones([2,1],NDArray::int8);
        $int32_2D = $mo->ones([2,1],NDArray::int32);
        $float32_2D = $mo->ones([2,1],NDArray::float32);
        $float64_2D = $mo->ones([2,1],NDArray::float64);

        $this->assertEquals(NDArray::int8,$mo->op($int8,'+',$int8)->dtype());
        $this->assertEquals(NDArray::int32,$mo->op($int8,'+',$int32)->dtype());
        $this->assertEquals(NDArray::float32,$mo->op($int32,'+',$float32)->dtype());
        $this->assertEquals(NDArray::float64,$mo->op($float64,'+',$float32)->dtype());

        $this->assertEquals(NDArray::int8,$mo->op($int8,'+',$int8_2D)->dtype());
        $this->assertEquals(NDArray::int32,$mo->op($int8,'+',$int32_2D)->dtype());
        $this->assertEquals(NDArray::float32,$mo->op($int32,'+',$float32_2D)->dtype());
        $this->assertEquals(NDArray::float64,$mo->op($float64,'+',$float32_2D)->dtype());

        $this->assertEquals(NDArray::int8,$mo->op($int8_2D,'+',$int8)->dtype());
        $this->assertEquals(NDArray::int32,$mo->op($int8_2D,'+',$int32)->dtype());
        $this->assertEquals(NDArray::float32,$mo->op($int32_2D,'+',$float32)->dtype());
        $this->assertEquals(NDArray::float64,$mo->op($float64_2D,'+',$float32)->dtype());

        $this->assertEquals(NDArray::int8,$mo->op($int8,'+',10)->dtype());
        $this->assertEquals(NDArray::int32,$mo->op($int32,'+',10)->dtype());
        $this->assertEquals(NDArray::float32,$mo->op($float32,'+',10)->dtype());
        $this->assertEquals(NDArray::float64,$mo->op($float64,'+',10)->dtype());

        $this->assertEquals(NDArray::float32,$mo->op($int8,'+',1.5)->dtype());
        $this->assertEquals(NDArray::float32,$mo->op($int8,'+',1.5)->dtype());
        $this->assertEquals(NDArray::float32,$mo->op($int32,'+',1.5)->dtype());
        $this->assertEquals(NDArray::float64,$mo->op($float64,'+',1.5)->dtype());

        $this->assertEquals(NDArray::int8,$mo->op(10,'+',$int8)->dtype());
        $this->assertEquals(NDArray::int32,$mo->op(10,'+',$int32)->dtype());
        $this->assertEquals(NDArray::float32,$mo->op(10,'+',$float32)->dtype());
        $this->assertEquals(NDArray::float64,$mo->op(10,'+',$float64)->dtype());

        $this->assertEquals(NDArray::float32,$mo->op(1.5,'+',$int8)->dtype());
        $this->assertEquals(NDArray::float32,$mo->op(1.5,'+',$int32)->dtype());
        $this->assertEquals(NDArray::float32,$mo->op(1.5,'+',$float32)->dtype());
        $this->assertEquals(NDArray::float64,$mo->op(1.5,'+',$float64)->dtype());
    }

    public function testSerializeArray()
    {
        $mo = $this->newMatrixOperator();

        $array = $mo->array([1,2,3],dtype:NDArray::int32);

        $data = $mo->serializeArray($array);

        $newMo = $this->newMatrixOperator();
        $newArray = $newMo->unserializeArray($data);

        $this->assertEquals($array->toArray(),$newArray->toArray());
        $this->assertEquals($array->dtype(),$newArray->dtype());
        $this->assertEquals(
            $array->service()->serviceLevel(),
            $newArray->service()->serviceLevel()
        );

        // array in array
        $arraySet = [
            'one' =>  $mo->array([1,2,3],dtype:NDArray::int32),
            'two' =>  $mo->array([4,5,6],dtype:NDArray::float32),
            'three' => [
                $mo->array([2,3,4],dtype:NDArray::int32),
                $mo->array([5,6,7],dtype:NDArray::int8),
            ]
        ];

        $data = $mo->serializeArray($arraySet);

        $newArraySet = $newMo->unserializeArray($data);

        $this->assertEquals($arraySet['one']->toArray(),$newArraySet['one']->toArray());
        $this->assertEquals($arraySet['two']->toArray(),$newArraySet['two']->toArray());
        $this->assertEquals($arraySet['one']->dtype(),$newArraySet['one']->dtype());
        $this->assertEquals($arraySet['two']->dtype(),$newArraySet['two']->dtype());
        $this->assertEquals(
            $arraySet['one']->service()->serviceLevel(),
            $newArraySet['one']->service()->serviceLevel()
        );
        $this->assertEquals(
            $arraySet['two']->service()->serviceLevel(),
            $newArraySet['two']->service()->serviceLevel()
        );

        $this->assertEquals(
            $arraySet['three'][0]->toArray(),
            $newArraySet['three'][0]->toArray()
        );
        $this->assertEquals(
            $arraySet['three'][1]->toArray(),
            $newArraySet['three'][1]->toArray()
        );
    }

    public function testComplexToString()
    {
        $mo = $this->newMatrixOperator();
        $a = [
            [C(1,i:2),C(3,i:4)],
            [C(5,i:-6),C(-7,i:8)]
        ];
        $array = $mo->array($a,dtype:NDArray::complex64);

        $string = $mo->toString($array);
        $this->assertEquals('[[1+2i,3+4i],[5-6i,-7+8i]]',$string);

        $string = $mo->toString($array,'%3.1f%+3.1fi');
        $this->assertEquals('[[1.0+2.0i,3.0+4.0i],[5.0-6.0i,-7.0+8.0i]]',$string);
        $string = $mo->toString($array,'%5.2f%+5.2fi');
        $this->assertEquals('[[ 1.00+2.00i, 3.00+4.00i],[ 5.00-6.00i,-7.00+8.00i]]',$string);

        // Invalid format
        $string = $mo->toString($array,'%+3.1f');
        $this->assertNotEquals('[[1.0+2.0i,3.0+4.0i],[5.0-6.0i,-7.0+8.0i]]',$string);
    }

    public function testToComplex()
    {
        $mo = $this->newMatrixOperator();

        $array = [[1,2],[3,C(i:4)]];
        $newArray = $mo->toComplex($array);
        $this->assertEquals('1+0i',$newArray[0][0]);
        $this->assertEquals('2+0i',$newArray[0][1]);
        $this->assertEquals('3+0i',$newArray[1][0]);
        $this->assertEquals('0+4i',$newArray[1][1]);

        // implicit converting
        $a = $mo->array($array,dtype:NDArray::complex64);
        $this->assertEquals('[[1+0i,2+0i],[3+0i,0+4i]]',$mo->toString($a));
    }
}
