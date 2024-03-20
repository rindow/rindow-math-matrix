<?php
namespace RindowTest\Math\Matrix\ComplexUtilsTest;

use PHPUnit\Framework\TestCase;
use Interop\Polite\Math\Matrix\NDArray;
use Rindow\Math\Matrix\ComplexUtils;
use function Rindow\Math\Matrix\C;

class TestUtils
{
    use ComplexUtils;

    public function build($r=null,$i=null)
    {
        return $this->cbuild($r,i:$i);
    }

    public function isobject(mixed $value) : bool
    {
        return $this->cisobject($value);
    }

    public function objecttype(mixed $value) : string
    {
        return $this->cobjecttype($value);
    }

    public function add($x,$y)
    {
        return $this->cadd($x,$y);
    }

    public function sub($x,$y)
    {
        return $this->csub($x,$y);
    }

    public function mul($x,$y)
    {
        return $this->cmul($x,$y);
    }

    public function div($x,$y)
    {
        return $this->cdiv($x,$y);
    }

}

class ComplexUtilsTest extends TestCase
{
    public function testcbuild()
    {
        $utils = new TestUtils();
        $this->assertEquals('1+2i',strval($utils->build(1,i:2)));
        $this->assertEquals('0+2i',strval($utils->build(i:2)));
        $this->assertEquals('1+0i',strval($utils->build(1)));
    }

    public function testcisobject()
    {
        $utils = new TestUtils();
        $c = C(0);
        $this->assertTrue($utils->isobject($c));
        $this->assertFalse($utils->isobject(new \stdClass()));
        $this->assertFalse($utils->isobject((object)[]));
        $this->assertTrue($utils->isobject((object)['real'=>0,'imag'=>0]));
    }

    public function testcobjecttype()
    {
        $utils = new TestUtils();
        $this->assertEquals('double',$utils->objecttype(1.1));
        $this->assertEquals('stdClass',$utils->objecttype(new \stdClass()));
    }

    public function testcadd()
    {
        $utils = new TestUtils();
        $c = $utils->add(C(1,i:2),C(2,i:1));
        $this->assertEquals('3+3i',strval($c));
    }

    public function testcsub()
    {
        $utils = new TestUtils();
        $c = $utils->sub(C(1,i:2),C(2,i:1));
        $this->assertEquals('-1+1i',strval($c));
    }

    public function testcmul()
    {
        $utils = new TestUtils();
        $c = $utils->mul(C(1,i:2),C(2,i:1));
        $this->assertEquals('0+5i',strval($c));
    }

    public function testcdiv()
    {
        $utils = new TestUtils();
        $c = $utils->div(C(1,i:2),C(2,i:1));
        $this->assertEquals('0.8+0.6i',strval($c));
    }
}