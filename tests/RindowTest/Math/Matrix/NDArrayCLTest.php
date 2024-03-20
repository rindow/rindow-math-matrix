<?php
namespace RindowTest\Math\Matrix\NDArrayCLTest;

use PHPUnit\Framework\TestCase;
use Rindow\Math\Matrix\NDArrayPhp;
use Rindow\Math\Matrix\NDArrayCL;
use Rindow\Math\Matrix\Drivers\Selector;
use Rindow\Math\Matrix\Drivers\Service;
use Interop\Polite\Math\Matrix\NDArray;
use Interop\Polite\Math\Matrix\OpenCL;
use Interop\Polite\Math\Matrix\DeviceBuffer;
use ArrayObject;
use OutOfRangeException;
use InvalidArgumentException;
use LogicException;
use RuntimeException;
use function Rindow\Math\Matrix\R;

class NDArrayCLTest extends TestCase
{
    protected bool $skipDisplayInfo = true;
    //protected int $default_device_type = OpenCL::CL_DEVICE_TYPE_DEFAULT;
    //protected int $default_device_type = OpenCL::CL_DEVICE_TYPE_GPU;
    static protected int $default_device_type = OpenCL::CL_DEVICE_TYPE_GPU;
    protected $service;

    public function setUp() : void
    {
        $selector = new Selector();
        $this->service = $selector->select();
        if($this->service->serviceLevel()<Service::LV_ACCELERATED) {
            $this->markTestSkipped("The service is not Accelerated.");
        }
    }

    public function getContext()
    {
        //return $this->service->openCL()->Context($this->default_device_type);
        try {
            $context = $this->service->openCL()->Context(self::$default_device_type);
        } catch(RuntimeException $e) {
            if(strpos('clCreateContextFromType',$e->getMessage())===null) {
                throw $e;
            }
            self::$default_device_type = OpenCL::CL_DEVICE_TYPE_DEFAULT;
            $context = $this->service->openCL()->Context(self::$default_device_type);
        }
        return $context;
    }

    public function getQueue($context)
    {
        return $this->service->openCL()->CommandQueue($context);
    }

    public function testSimpleArrayoffsetGet()
    {
        $hostArray = new NDArrayPhp([[1,2],[3,4],[5,6]],service:$this->service);
        $context = $this->getContext();
        $queue = $this->getQueue($context);
        $array = new NDArrayCL($queue,$hostArray->buffer(),$hostArray->dtype(),
            $hostArray->shape(),$hostArray->offset(),
            OpenCL::CL_MEM_READ_ONLY|OpenCL::CL_MEM_COPY_HOST_PTR,service:$this->service);

        $this->assertEquals([3,2],$array->shape());
        $this->assertEquals(NDArray::float32,$array->dtype());
        $this->assertEquals(0,$array->offset());
        $this->assertEquals(2,$array->ndim());
        $this->assertEquals(
            OpenCL::CL_MEM_READ_ONLY|OpenCL::CL_MEM_COPY_HOST_PTR,
            $array->flags());
        $this->assertEquals(2,$array->ndim());
        $this->assertEquals(6,$array->size());
        $this->assertInstanceof(DeviceBuffer::class,$array->buffer());
        $this->assertEquals(6*32/8,$array->buffer()->bytes());
        $this->assertEquals(NDArray::float32,$array->buffer()->dtype());
        $this->assertEquals(32/8,$array->buffer()->value_size());

        // offsetGet rank 1 ndarray object
        $newArray = $array[1];
        $this->assertEquals(1,$newArray->ndim());
        $this->assertEquals([3,4],$newArray->toNDArray()->toArray());

        // offsetGet range axis
        $newArray = $array[R(1,3)];
        $this->assertEquals(2,$newArray->ndim());
        $this->assertEquals([2,2],$newArray->shape());
        $this->assertEquals([[3,4],[5,6]],$newArray->toNDArray()->toArray());

        // offsetGet rank 0 object
        $newArray = $array[1][1];
        $this->assertEquals(0,$newArray->ndim());
        $this->assertEquals([],$newArray->shape());
        $this->assertEquals(4,$newArray->toNDArray()->toArray());
    }

    public function testSimpleArrayoffsetSet()
    {
        $hostArray = new NDArrayPhp([[1,2],[3,4],[5,6]],service:$this->service);
        $context = $this->getContext();
        $queue = $this->getQueue($context);
        $array = new NDArrayCL($queue,$hostArray->buffer(),$hostArray->dtype(),
            $hostArray->shape(),$hostArray->offset(),
            OpenCL::CL_MEM_READ_WRITE|OpenCL::CL_MEM_COPY_HOST_PTR,service:$this->service);

        $hostArray = new NDArrayPhp([11,12],service:$this->service);
        $srcarray = new NDArrayCL($queue,$hostArray->buffer(),$hostArray->dtype(),
            $hostArray->shape(),$hostArray->offset(),
            OpenCL::CL_MEM_READ_ONLY|OpenCL::CL_MEM_COPY_HOST_PTR,service:$this->service);
        $array[1] = $srcarray;
        $queue->finish();
        $this->assertEquals([[1,2],[11,12],[5,6]],$array->toNDArray()->toArray());

        $hostArray = new NDArrayPhp(4,service:$this->service);
        $srcarray = new NDArrayCL($queue,$hostArray->buffer(),$hostArray->dtype(),
            $hostArray->shape(),$hostArray->offset(),
            OpenCL::CL_MEM_READ_ONLY|OpenCL::CL_MEM_COPY_HOST_PTR,service:$this->service);
        $array[1][1] = $srcarray;
        $queue->finish();
        $this->assertEquals([[1,2],[11,4],[5,6]],$array->toNDArray()->toArray());

    }

    public function testClone()
    {
        $hostArray = new NDArrayPhp([[1,2],[3,4],[5,6]],dtype:NDArray::int32,service:$this->service);
        $context = $this->getContext();
        $queue = $this->getQueue($context);
        $array = new NDArrayCL($queue,$hostArray->buffer(),$hostArray->dtype(),
            $hostArray->shape(),$hostArray->offset(),
            OpenCL::CL_MEM_READ_ONLY|OpenCL::CL_MEM_COPY_HOST_PTR,service:$this->service);
        $this->assertEquals([3,2],$array->shape());
        $this->assertEquals(NDArray::int32,$array->dtype());

        $pattern=new NDArrayPhp([[0,0],[0,0],[0,0]],dtype:NDArray::int32,service:$this->service);
        $array2 = clone $array;
        $array->buffer()->write($queue,$pattern->buffer());

        $this->assertEquals([[0,0],[0,0],[0,0]],$array->toArray());
        $this->assertEquals(NDArray::int32,$array2->dtype());
        $this->assertEquals(NDArray::int32,$array2->buffer()->dtype());
        $this->assertEquals([[1,2],[3,4],[5,6]],$array2->toArray());
    }

    public function testToNDArrayStressTest()
    {
        $hostArray = new NDArrayPhp([[1,2],[3,4],[5,6]],service:$this->service);
        $context = $this->getContext();
        $queue = $this->getQueue($context);
        $array = new NDArrayCL($queue,$hostArray->buffer(),$hostArray->dtype(),
            $hostArray->shape(),$hostArray->offset(),
            OpenCL::CL_MEM_READ_ONLY|OpenCL::CL_MEM_COPY_HOST_PTR,service:$this->service);

        for($i=0;$i<1000;++$i) {
            $res = $array->toNDArray();
            if([[1,2],[3,4],[5,6]]!= $res->toArray()) {
                break;
            }
        }
        $this->assertEquals([[1,2],[3,4],[5,6]],$res->toArray());
    }

    public function testRangeStyle()
    {
        $this->assertEquals(NDArrayPhp::RANGE_STYLE_DEFAULT,NDArrayPhp::$rangeStyle);
        $a = [0,1,2,3,4];
        $array = new NDArrayPhp($a,dtype:NDArray::int32,service:$this->service);
        $this->assertEquals([1,2,3],$array[[1,4]]->toArray());
        $this->assertEquals([1,2,3],$array[R(1,4)]->toArray());
        NDArrayPhp::$rangeStyle = NDArrayPhp::RANGE_STYLE_1;
        $this->assertEquals([1,2,3],$array[[1,3]]->toArray());
        $this->assertEquals([1,2,3],$array[R(1,4)]->toArray());
        NDArrayPhp::$rangeStyle = NDArrayPhp::RANGE_STYLE_FORCE2;
        $error = false;
        try {
            $this->assertEquals([1,2,3],$array[[1,4]]->toArray());
        } catch(InvalidArgumentException $e) {
            $error = true;
        }
        $this->assertTrue($error);
        $this->assertEquals([1,2,3],$array[R(1,4)]->toArray());
        NDArrayPhp::$rangeStyle = NDArrayPhp::RANGE_STYLE_DEFAULT;
    }
}
