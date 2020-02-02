<?php
namespace RindowTest\Math\Matrix\NDArrayPhpTest;

use PHPUnit\Framework\TestCase;
use Rindow\Math\Matrix\NDArrayPhp;
use Interop\Polite\Math\Matrix\NDArray;
use ArrayObject;
use SplFixedArray;

class Test extends TestCase
{
    public function bufferToArray($buffer)
    {
        $size = count($buffer);
        $array = [];
        for($i=0;$i<$size;$i++) {
            $array[] = $buffer[$i];
        }
        return $array;
    }
    public function testCreateFromArray()
    {
        $array = [[1,2,3],[4,5,6]];
        $nd = new NDArrayPhp($array);
        $this->assertEquals([2,3],$nd->shape());
        $this->assertEquals(6,$nd->size());
        $this->assertEquals(NDArray::float32,$nd->dtype());
        $this->assertEquals(0,$nd->offset());
        $buffer = $nd->buffer();
        $this->assertEquals(6,count($buffer));
        $this->assertEquals([1,2,3,4,5,6],$this->bufferToArray($buffer));
        $this->assertEquals([[1,2,3],[4,5,6]],$nd->toArray());

        $array = [[[1,2],[3,4],[5,6]],[[7,8],[9,10],[11,12]]];
        $nd = new NDArrayPhp($array,NDArray::float64);
        $this->assertEquals([2,3,2],$nd->shape());
        $this->assertEquals(12,$nd->size());
        $this->assertEquals(NDArray::float64,$nd->dtype());
        $this->assertEquals(0,$nd->offset());
        $buffer = $nd->buffer();
        $this->assertEquals(12,count($buffer));
        $this->assertEquals([1,2,3,4,5,6,7,8,9,10,11,12],$this->bufferToArray($buffer));
        $this->assertEquals([[[1,2],[3,4],[5,6]],[[7,8],[9,10],[11,12]]],$nd->toArray());

        $array = [1,2,3];
        $nd = new NDArrayPhp($array,NDArray::int32);
        $this->assertEquals([3],$nd->shape());
        $this->assertEquals(3,$nd->size());
        $this->assertEquals(NDArray::int32,$nd->dtype());
        $this->assertEquals(0,$nd->offset());
        $buffer = $nd->buffer();
        $this->assertEquals(3,count($buffer));
        $this->assertEquals([1,2,3],$this->bufferToArray($buffer));
        $this->assertEquals([1,2,3],$nd->toArray());
    }

    public function testCreateFromArrayObject()
    {
        $array = new ArrayObject([new ArrayObject([1,2,3]),new ArrayObject([4,5,6])]);
        $nd = new NDArrayPhp($array);
        $this->assertEquals([2,3],$nd->shape());
        $this->assertEquals(6,$nd->size());
        $this->assertEquals(NDArray::float32,$nd->dtype());
        $this->assertEquals(0,$nd->offset());
        $buffer = $nd->buffer();
        $this->assertEquals(6,count($buffer));
        $this->assertEquals([1,2,3,4,5,6],$this->bufferToArray($buffer));
        $this->assertEquals([[1,2,3],[4,5,6]],$nd->toArray());
    }

    public function testCreateFromNumber()
    {
        $array = 123;
        $nd = new NDArrayPhp($array);
        $this->assertEquals([1],$nd->shape());
        $this->assertEquals(1,$nd->size());
        $this->assertEquals(NDArray::float32,$nd->dtype());
        $this->assertEquals(0,$nd->offset());
        $buffer = $nd->buffer();
        $this->assertEquals(1,count($buffer));
        $this->assertEquals([123],$this->bufferToArray($buffer));
        $this->assertEquals([123],$nd->toArray());
    }

    public function testCreateNoInitialize()
    {
        $nd = new NDArrayPhp(null,NDArray::float32,[2,3]);
        $this->assertEquals([2,3],$nd->shape());
        $this->assertEquals(6,$nd->size());
        $this->assertEquals(NDArray::float32,$nd->dtype());
        $this->assertEquals(0,$nd->offset());
        $buffer = $nd->buffer();
        $this->assertEquals(6,count($buffer));
        $this->assertEquals([null,null,null,null,null,null],$this->bufferToArray($buffer));
        $this->assertEquals([[null,null,null],[null,null,null]],$nd->toArray());
    }

    public function testCreateFromBuffer()
    {
        $array = SplFixedArray::fromArray([1,2,3,4,5,6]);
        $nd = new NDArrayPhp($array,NDArray::float32,[1,3],3);
        $this->assertEquals([1,3],$nd->shape());
        $this->assertEquals(3,$nd->size());
        $this->assertEquals(NDArray::float32,$nd->dtype());
        $this->assertEquals(3,$nd->offset());
        $buffer = $nd->buffer();
        $this->assertEquals(6,count($buffer));
        $this->assertEquals([1,2,3,4,5,6],$this->bufferToArray($buffer));
        $this->assertEquals([[4,5,6]],$nd->toArray());
    }

    /**
     * @expectedException        InvalidArgumentException
     * @expectedExceptionMessage The shape of the dimension is broken
     */
    public function testCreateFromIllegalArray()
    {
        $array = [[1,2,3],[4,5]];
        $nd = new NDArrayPhp($array);
    }

    /**
     * @expectedException        InvalidArgumentException
     * @expectedExceptionMessage Invalid dimension size
     */
    public function testCreateFromNumberAndIllegalShape()
    {
        $array = 123;
        $nd = new NDArrayPhp($array,NDArray::float32,[2,3]);
    }

    /**
     * @expectedException       InvalidArgumentException
     * expectedExceptionMessage Invalid type of array
     */
    public function testCreateNullAndNoShape()
    {
        $nd = new NDArrayPhp(null,NDArray::float32);
    }

    /**
     * @expectedException        InvalidArgumentException
     * @expectedExceptionMessage Invalid dimension size
     */
    public function testCreateFromBufferAndIllegalOffset()
    {
        $array = SplFixedArray::fromArray([1,2,3,4,5,6]);
        $nd = new NDArrayPhp($array,NDArray::float32,[1,3],4);
    }

    public function testOffsetExists()
    {
        $array = [[1,2,3],[4,5,6]];
        $nd = new NDArrayPhp($array);
        $this->assertTrue($nd->offsetExists(0));
        $this->assertTrue($nd->offsetExists(1));
        $this->assertFalse($nd->offsetExists(2));
        $this->assertFalse($nd->offsetExists(-1));
        $this->assertTrue(isset($nd[1][2]));
        $this->assertFalse(isset($nd[1][3]));
        $this->assertFalse(isset($nd[2][1]));
    }

    /**
     * @expectedException        OutOfRangeException
     * @expectedExceptionMessage Dimension must be integer
     */
    public function testOffsetExistsWithInvalidNumber()
    {
        $array = [[1,2,3],[4,5,6]];
        $nd = new NDArrayPhp($array);
        $this->assertTrue($nd->offsetExists(0.5));
    }

    public function testOffsetGet()
    {
        $array = [[1,2,3],[4,5,6]];
        $nd = new NDArrayPhp($array);
        $this->assertEquals(1,$nd[0][0]);
        $this->assertEquals(6,$nd[1][2]);
        $this->assertEquals([1,2,3],$nd[0]->toArray());
        $this->assertEquals([4,5,6],$nd[1]->toArray());
        $this->assertEquals(0,$nd[0]->offset());
        $this->assertEquals(3,$nd[1]->offset());
        $this->assertEquals(
            spl_object_hash($nd[0]->buffer()),
            spl_object_hash($nd[1]->buffer()));

        $array = SplFixedArray::fromArray([1,2,3,4,5,6]);
        $nd = new NDArrayPhp($array,NDArray::float32,[1,3],3);
        $this->assertEquals(6,$nd[0][2]);
        $this->assertEquals([4,5,6],$nd[0]->toArray());

        $array = [[[1,2],[3,4]],[[5,6],[7,8]]];
        $nd = new NDArrayPhp($array);
        $this->assertEquals(8,$nd[1][1][1]);
        $this->assertEquals([7,8],$nd[1][1]->toArray());
        $this->assertEquals([[5,6],[7,8]],$nd[1]->toArray());

    }

    public function testOffsetSet()
    {
        $array = [[1,2,3],[4,5,6]];
        $nd = new NDArrayPhp($array);
        $nd[0][0] = 10;
        $this->assertEquals([[10,2,3],[4,5,6]],$nd->toArray());
        $nd[1][2] = 60;
        $this->assertEquals([[10,2,3],[4,5,60]],$nd->toArray());
        $nd[1] = new NDArrayPhp([400,500,600]);
        $this->assertEquals([[10,2,3],[400,500,600]],$nd->toArray());

        $array = SplFixedArray::fromArray([1,2,3,4,5,6]);
        $nd = new NDArrayPhp($array,NDArray::float32,[1,3],3);
        $nd[0][0] = 40;
        $this->assertEquals([[40,5,6]],$nd->toArray());
        $nd[0] = new NDArrayPhp([400,500,600]);
        $this->assertEquals([[400,500,600]],$nd->toArray());
    }

    /**
     * @expectedException        OutOfRangeException
     * @expectedExceptionMessage Index is out of range
     */
    public function testOutOfRangeWithOffgetSet()
    {
        $array = [[1,2,3],[4,5,6]];
        $nd = new NDArrayPhp($array);
        $a = $nd[2];
    }

    /**
     * @expectedException        OutOfRangeException
     * @expectedExceptionMessage Index is out of range
     */
    public function testOutOfRangeWithOffsetSet()
    {
        $array = [[1,2,3],[4,5,6]];
        $nd = new NDArrayPhp($array);
        $nd[3] = new NDArrayPhp([400,500,600]);
    }

    /**
     * @expectedException        LogicException
     * @expectedExceptionMessage Unsuppored Operation
     */
    public function testOffunSet()
    {
        $array = [[1,2,3],[4,5,6]];
        $nd = new NDArrayPhp($array);
        unset($nd[1]);
    }

    public function testReshape()
    {
        $array = [[[0,0,0],[0,0,0]],[[1,2,3],[4,5,6]],[[0,0,0],[0,0,0]]];
        $nd = new NDArrayPhp($array);
        $nd = $nd[1];
        $r = $nd->reshape([6]);
        $this->assertEquals([1,2,3,4,5,6],$r->toArray());
        $this->assertNotEquals(spl_object_hash($r),spl_object_hash($nd));
        $this->assertEquals([6],$r->shape());
        $this->assertEquals(spl_object_hash($r->buffer()),spl_object_hash($nd->buffer()));
    }

    public function testSerializeDefault()
    {
        $array = [[[1,2,3],[4,5,6]],[[7,8,9],[10,11,12]]];
        $a = new NDArrayPhp($array,NDArray::int32);
        $this->assertEquals([2,2,3],$a->shape());
        $this->assertEquals(NDArray::int32,$a->dtype());
        $str = serialize($a);
        if(extension_loaded('rindow_openblas')) {
            $this->assertTrue(strpos($str,'inear-array')===false);
        }
        $b = unserialize($str);
        $this->assertEquals([2,2,3],$b->shape());
        $this->assertEquals(NDArray::int32,$b->dtype());
        $this->assertEquals($array,$b->toArray());
    }

    public function testSerializePortable()
    {
        $array = [[[1,2,3],[4,5,6]],[[7,8,9],[10,11,12]]];
        $a = new NDArrayPhp($array,NDArray::int32);
        $this->assertEquals([2,2,3],$a->shape());
        $this->assertEquals(NDArray::int32,$a->dtype());
        if(extension_loaded('rindow_openblas')) {
            $this->assertInstanceof('Rindow\\OpenBLAS\\Buffer',$a->buffer());
        }
        $this->assertFalse($a->getPortableSerializeMode());
        $a->setPortableSerializeMode(true);
        $this->assertTrue($a->getPortableSerializeMode());
        $str = serialize($a);
        $this->assertTrue(strpos($str,'inear-array')!==null);

        $b = unserialize($str);
        $this->assertEquals([2,2,3],$b->shape());
        $this->assertEquals(NDArray::int32,$b->dtype());
        $this->assertEquals($array,$b->toArray());
        if(extension_loaded('rindow_openblas')) {
            $this->assertInstanceof('Rindow\\OpenBLAS\\Buffer',$b->buffer());
        }
        $this->assertFalse($b->getPortableSerializeMode());
    }

    public function testOffsetGetForRange()
    {
        $array = [1,2,3,4,5,6,7,8,9,10];
        $a = new NDArrayPhp($array,NDArray::int32);
        $this->assertEquals([3,4,5],$a[[2,4]]->toArray());
        $this->assertEquals(
            spl_object_hash($a->buffer()),
            spl_object_hash($a[[2,4]]->buffer()));

        $this->assertEquals([3],$a[[2,2]]->toArray());
        $this->assertEquals(
            spl_object_hash($a->buffer()),
            spl_object_hash($a[[2,2]]->buffer()));

        $this->assertEquals([1,2,3,4,5,6,7,8,9,10],$a[[0,9]]->toArray());
        $this->assertEquals(
            spl_object_hash($a->buffer()),
            spl_object_hash($a[[0,9]]->buffer()));

        $array = [[[1,2],[3,4]],[[5,6],[7,8]],[[9,10],[11,12]]];
        $a = new NDArrayPhp($array,NDArray::int32);
        $this->assertEquals([[[1,2],[3,4]],[[5,6],[7,8]]],$a[[0,1]]->toArray());
        $this->assertEquals([[[5,6],[7,8]],[[9,10],[11,12]]],$a[[1,2]]->toArray());
    }

    public function testOffsetExistsForRange()
    {
        $array = [1,2,3,4,5,6,7,8,9,10];
        $a = new NDArrayPhp($array,NDArray::int32);
        $this->assertTrue($a->offsetExists([2,4]));
        $this->assertTrue($a->offsetExists([0,4]));
        $this->assertTrue($a->offsetExists([4,9]));
        $this->assertTrue($a->offsetExists([0,9]));
        $this->assertFalse($a->offsetExists([-1,9]));
        $this->assertFalse($a->offsetExists([0,10]));
    }

    /**
     * @expectedException        OutOfRangeException
     * @expectedExceptionMessage Illegal range specification
     */
    public function testOffsetExistsForIllegalRange()
    {
        $array = [1,2,3,4,5,6,7,8,9,10];
        $a = new NDArrayPhp($array,NDArray::int32);
        $this->assertFalse($a->offsetExists([1,0]));
    }
}
