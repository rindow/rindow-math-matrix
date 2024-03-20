<?php
namespace RindowTest\Math\Matrix\Drivers\MatlibPHP\PhpBufferTest;

use PHPUnit\Framework\TestCase;
use Interop\Polite\Math\Matrix\NDArray;
use Rindow\Math\Matrix\Drivers\MatlibPHP\PhpBuffer;
use Rindow\Math\Matrix\Complex;
use ArgumentCountError;
use LogicException;
use RuntimeException;
use OutOfRangeException;
use TypeError;

class PhpBufferTest extends TestCase
{
    //public function testExtensionVersion()
    //{
    //    $this->assertEquals('0.1.7',phpversion('rindow_openblas'));
    //}

    public function testNormal()
    {
        $buf = new PhpBuffer(3,NDArray::float32);
        $buf[0] = 0.5;
        $buf[1] = 1.5;
        $buf[2] = 2.5;
        $this->assertEquals(3,count($buf));
        $this->assertEquals(NDArray::float32,$buf->dtype());
        $this->assertTrue(is_float($buf[0]));
        $this->assertEquals(0.5,$buf[0]);
        $this->assertEquals(1.5,$buf[1]);
        $this->assertEquals(2.5,$buf[2]);
    }

    public function testDtypesAndOffsetOfDtypes()
    {
        $buf = new PhpBuffer(3,NDArray::bool);
        $buf[2] = true;
        $this->assertEquals(NDArray::bool,$buf->dtype());
        $this->assertTrue(is_bool($buf[2]));
        $this->assertEquals(true,$buf[2]);

        $buf = new PhpBuffer(3,NDArray::int8);
        $buf[2] = -1;
        $this->assertEquals(NDArray::int8,$buf->dtype());
        $this->assertTrue(is_int($buf[2]));
        $this->assertEquals(-1,$buf[2]);

        $buf = new PhpBuffer(3,NDArray::uint8);
        $buf[2] = -1;
        $this->assertEquals(NDArray::uint8,$buf->dtype());
        $this->assertTrue(is_int($buf[2]));
        $this->assertEquals(-1,$buf[2]); // *** CAUTION ***

        $buf = new PhpBuffer(3,NDArray::int16);
        $buf[2] = -1;
        $this->assertEquals(NDArray::int16,$buf->dtype());
        $this->assertTrue(is_int($buf[2]));
        $this->assertEquals(-1,$buf[2]);

        $buf = new PhpBuffer(3,NDArray::uint16);
        $buf[2] = -1;
        $this->assertEquals(NDArray::uint16,$buf->dtype());
        $this->assertTrue(is_int($buf[2]));
        $this->assertEquals(-1,$buf[2]);  // *** CAUTION ***

        $buf = new PhpBuffer(3,NDArray::int32);
        $buf[2] = -1;
        $this->assertEquals(NDArray::int32,$buf->dtype());
        $this->assertTrue(is_int($buf[2]));
        $this->assertEquals(-1,$buf[2]);

        $buf = new PhpBuffer(3,NDArray::uint32);
        $buf[2] = -1;
        $this->assertEquals(NDArray::uint32,$buf->dtype());
        $this->assertTrue(is_int($buf[2]));
        $this->assertEquals(-1,$buf[2]); // *** CAUTION ***

        $buf = new PhpBuffer(3,NDArray::int64);
        $buf[2] = -1;
        $this->assertEquals(NDArray::int64,$buf->dtype());
        $this->assertTrue(is_int($buf[2]));
        $this->assertEquals(-1,$buf[2]);

        $buf = new PhpBuffer(3,NDArray::uint64);
        $buf[2] = -1;
        $this->assertEquals(NDArray::uint64,$buf->dtype());
        $this->assertTrue(is_int($buf[2]));
        $this->assertEquals(-1,$buf[2]); // *** CAUTION ****

        $buf = new PhpBuffer(3,NDArray::float32);
        $buf[2] = 0.5;
        $this->assertEquals(NDArray::float32,$buf->dtype());
        $this->assertTrue(is_float($buf[2]));
        $this->assertEquals(0.5,$buf[2]);

        $buf = new PhpBuffer(3,NDArray::float64);
        $buf[2] = 0.5;
        $this->assertEquals(NDArray::float64,$buf->dtype());
        $this->assertTrue(is_float($buf[2]));
        $this->assertEquals(0.5,$buf[2]);

        $buf = new PhpBuffer(3,NDArray::complex64);
        $this->assertEquals(NDArray::complex64,$buf->dtype());
        $this->assertEquals(3,count($buf));
        $this->assertEquals(8,$buf->value_size());
        $this->assertEquals(NDArray::complex64,$buf->dtype());
        $buf[1] = new Complex(1.5,2.5);
        $buf[2] = (object)['real'=>3.5,'imag'=>4.5];
        $vv = $buf[1];
        $this->assertEquals(1.5,$vv->real);
        $this->assertEquals(2.5,$vv->imag);
        $vv = $buf[2];
        $this->assertEquals(3.5,$vv->real);
        $this->assertEquals(4.5,$vv->imag);

        $buf = new PhpBuffer(3,NDArray::complex128);
        $this->assertEquals(NDArray::complex128,$buf->dtype());
        $this->assertEquals(3,count($buf));
        $this->assertEquals(16,$buf->value_size());
        $this->assertEquals(NDArray::complex128,$buf->dtype());
        $buf[1] = new Complex(1.5,2.5);
        $buf[2] = (object)['real'=>3.5,'imag'=>4.5];
        $vv = $buf[1];
        $this->assertEquals(1.5,$vv->real);
        $this->assertEquals(2.5,$vv->imag);
        $vv = $buf[2];
        $this->assertEquals(3.5,$vv->real);
        $this->assertEquals(4.5,$vv->imag);
    }

    public function testOffsetExists()
    {
        $buf = new PhpBuffer(3,NDArray::float32);
        $this->assertFalse(isset($buf[0])); // ** CAUTION **
        $this->assertFalse(isset($buf[2])); // ** CAUTION **
        $this->assertFalse(isset($buf[-1]));
        $this->assertFalse(isset($buf[3]));

        $buf[0] = 0;
        $this->assertTrue(isset($buf[0])); // ** CAUTION **
        $this->assertFalse(isset($buf[2])); // ** CAUTION **
    }

    public function testUnset()
    {
        $buf = new PhpBuffer(3,NDArray::float32);
        $buf[0] = 1;
        $this->assertEquals(1,$buf[0]);
        unset($buf[0]); // unset means set zero
        $this->assertEquals(null,$buf[0]);
    }

    public function testDumpAndLoadNormal()
    {
        $buf = new PhpBuffer(3,NDArray::float32);
        $buf[0] = 1;
        $buf[1] = 2;
        $buf[2] = 3;

        $buf2 = new PhpBuffer(3,NDArray::float32);
        $buf2[0] = 0;
        $buf2[1] = 0;
        $buf2[2] = 0;

        $dump = $buf->dump();
        $buf2->load($dump);
        $this->assertEquals(1,$buf2[0]);
        $this->assertEquals(2,$buf2[1]);
        $this->assertEquals(3,$buf2[2]);
    }

    public function testDumpAndLoadComplex()
    {
        $buf = new PhpBuffer(3,NDArray::complex64);
        $buf[0] = (object)['real'=>1,'imag'=>2];
        $buf[1] = (object)['real'=>3,'imag'=>4];
        $buf[2] = (object)['real'=>5,'imag'=>6];

        $buf2 = new PhpBuffer(3,NDArray::complex64);

        $dump = $buf->dump();
        $buf2->load($dump);
        $this->assertEquals(1,$buf2[0]->real);
        $this->assertEquals(2,$buf2[0]->imag);
        $this->assertEquals(3,$buf2[1]->real);
        $this->assertEquals(4,$buf2[1]->imag);
        $this->assertEquals(5,$buf2[2]->real);
        $this->assertEquals(6,$buf2[2]->imag);
    }

    public function testSetOutOfBoundsWithHighOffset()
    {
        //$buf = new \SplFixedArray(3);
        $buf = new PhpBuffer(3,NDArray::float32);
        $this->expectException(RuntimeException::class);
        $this->expectExceptionMessage('Index invalid or out of range');
        $buf[3] = 1;
    }

    public function testSetOutOfBoundsWithLowOffset()
    {
        //$buf = new \SplFixedArray(3);
        $buf = new PhpBuffer(3,NDArray::float32);
        $this->expectException(RuntimeException::class);
        $this->expectExceptionMessage('Index invalid or out of range');
        $buf[-1] = 1;
    }

    public function testGetOutOfBoundsWithHighOffset()
    {
        //$buf = new \SplFixedArray(3);
        $buf = new PhpBuffer(3,NDArray::float32);
        $this->expectException(RuntimeException::class);
        $this->expectExceptionMessage('Index invalid or out of range');
        $x = $buf[3];
    }

    public function testGetOutOfBoundsWithLowOffset()
    {
        //$buf = new \SplFixedArray(3);
        $buf = new PhpBuffer(3,NDArray::float32);
        $this->expectException(RuntimeException::class);
        $this->expectExceptionMessage('Index invalid or out of range');
        $x = $buf[-1];
    }

    public function testUnsetOutOfBoundsWithHighOffset()
    {
        //$buf = new \SplFixedArray(3);
        $buf = new PhpBuffer(3,NDArray::float32);
        $this->expectException(RuntimeException::class);
        $this->expectExceptionMessage('Index invalid or out of range');
        unset($buf[3]);
        //$this->assertTrue(true);
    }

    public function testUnsetOutOfBoundsWithLowOffset()
    {
        //$buf = new \SplFixedArray(3);
        $buf = new PhpBuffer(3,NDArray::float32);
        $this->expectException(RuntimeException::class);
        $this->expectExceptionMessage('Index invalid or out of range');
        unset($buf[-1]);
        $this->assertTrue(true);
    }

    public function testIsExistsOutOfBoundsWithHighOffset()
    {
        //$buf = new \SplFixedArray(3);
        $buf = new PhpBuffer(3,NDArray::float32);
        $this->assertFalse(isset($buf[3]));
    }

    public function testIsExistsOutOfBoundsWithLowOffset()
    {
        //$buf = new \SplFixedArray(3);
        $buf = new PhpBuffer(3,NDArray::float32);
        $this->assertFalse(isset($buf[-1]));
    }

    public function testOffsetSetWithNoOffset()
    {
        $buf = new PhpBuffer(3,NDArray::float32);
        $this->expectException(ArgumentCountError::class);
        //if(version_compare(PHP_VERSION, '8.0.0')<0) {
        //    $this->expectExceptionMessage('offsetSet() expects exactly 2 parameters, 0 given');
        //} else {
        //    $this->expectExceptionMessage('offsetSet() expects exactly 2 arguments, 0 given');
        //}
        $this->expectExceptionMessage('Too few arguments to function');
        $a = $buf->offsetSet();
    }

    public function testOffsetSetIllegalTypeOffset()
    {
        $buf = new PhpBuffer(3,NDArray::float32);
        $this->expectException(TypeError::class);
        if(version_compare(PHP_VERSION, '8.0.0')<0) {
            $this->expectExceptionMessage('offsetSet() expects parameter 1 to be int');
        } elseif(version_compare(PHP_VERSION, '8.3.0')<0) {
            $this->expectExceptionMessage('Illegal offset type');
        } else {
            $this->expectExceptionMessage('Cannot access offset of type stdClass on SplFixedArray');
        }
        $buf->offsetSet(new \stdClass(),1);
    }

    public function testOffsetGetWithNoOffset()
    {
        $buf = new PhpBuffer(3,NDArray::float32);
        $this->expectException(ArgumentCountError::class);
        //if(version_compare(PHP_VERSION, '8.0.0')<0) {
        //    $this->expectExceptionMessage('offsetGet() expects exactly 1 parameter, 0 given');
        //} else {
        //    $this->expectExceptionMessage('offsetGet() expects exactly 1 argument, 0 given');
        //}
        $this->expectExceptionMessage('SplFixedArray::offsetGet() expects exactly 1 argument, 0 given');
        $a = $buf->offsetGet();
    }

    public function testOffsetGetIllegalType()
    {
        $buf = new PhpBuffer(3,NDArray::float32);
        $this->expectException(TypeError::class);
        if(version_compare(PHP_VERSION, '8.0.0')<0) {
            $this->expectExceptionMessage('offsetGet() expects parameter 1 to be int');
        } elseif(version_compare(PHP_VERSION, '8.3.0')<0) {
            $this->expectExceptionMessage('Illegal offset type');
        } else {
            $this->expectExceptionMessage('Cannot access offset of type stdClass on SplFixedArray');
        }
        $a = $buf->offsetGet(new \stdClass());
    }

    public function testOffsetUnsetWithNoOffset()
    {
        $buf = new PhpBuffer(3,NDArray::float32);
        $this->expectException(ArgumentCountError::class);
        //if(version_compare(PHP_VERSION, '8.0.0')<0) {
        //    $this->expectExceptionMessage('offsetUnset() expects exactly 1 parameter, 0 given');
        //} else {
        //    $this->expectExceptionMessage('offsetUnset() expects exactly 1 argument, 0 given');
        //}
        $this->expectExceptionMessage('SplFixedArray::offsetUnset() expects exactly 1 argument, 0 given');
        $buf->offsetUnset();
    }

    public function testOffsetUnsetIllegalType()
    {
        $buf = new PhpBuffer(3,NDArray::float32);
        $this->expectException(TypeError::class);
        if(version_compare(PHP_VERSION, '8.0.0')<0) {
            $this->expectExceptionMessage('offsetUnset() expects parameter 1 to be int');
        } elseif(version_compare(PHP_VERSION, '8.3.0')<0) {
            $this->expectExceptionMessage('Illegal offset type');
        } else {
            $this->expectExceptionMessage('Cannot access offset of type stdClass on SplFixedArray');
        }
        //$this->expectException(LogicException::class);
        //$this->expectExceptionMessage('Illigal Operation');
        $buf->offsetUnset(new \stdClass());
    }

    public function testLoadWithNoOffset()
    {
        $buf = new PhpBuffer(3,NDArray::float32);
        $this->expectException(ArgumentCountError::class);
        //if(version_compare(PHP_VERSION, '8.0.0')<0) {
        //    $this->expectExceptionMessage('load() expects exactly 1 parameter, 0 given');
        //} else {
        //    $this->expectExceptionMessage('load() expects exactly 1 argument, 0 given');
        //}
        $this->expectExceptionMessage('Too few arguments to function');
        $buf->load();
    }

    public function testLoadIllegalType()
    {
        $buf = new PhpBuffer(3,NDArray::float32);
        $this->expectException(TypeError::class);
        if(version_compare(PHP_VERSION, '8.0.0')<0) {
            $this->expectExceptionMessage('load() expects parameter 1 to be string');
        } else {
            //$this->expectExceptionMessage('load(): Argument #1 must be of type string');
            $this->expectExceptionMessage('load(): Argument #1 ($string) must be of type');
        }
        $buf->load(new \stdClass());
    }

    public function testConstractWithNoArgument()
    {
        $this->expectException(ArgumentCountError::class);
        //if(version_compare(PHP_VERSION, '8.0.0')<0) {
        //    $this->expectExceptionMessage('__construct() expects exactly 2 parameters, 0 given');
        //} else {
        //    $this->expectExceptionMessage('__construct() expects exactly 2 arguments, 0 given');
        //}
        $this->expectExceptionMessage('Too few arguments to function');
        $buf = new PhpBuffer();
    }

    public function testConstractIllegalType()
    {
        $this->expectException(TypeError::class);
        if(version_compare(PHP_VERSION, '8.0.0')<0) {
            $this->expectExceptionMessage('__construct() expects parameter 1 to be int');
        } else {
            $this->expectExceptionMessage('__construct(): Argument #1 ($size) must be of type int');
        }
        $buf = new PhpBuffer(new \stdClass(),NDArray::float32);
    }
}
