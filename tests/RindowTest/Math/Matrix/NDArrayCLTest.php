<?php
namespace RindowTest\Math\Matrix\NDArrayCLTest;

use PHPUnit\Framework\TestCase;
use Rindow\Math\Matrix\NDArrayPhp;
use Rindow\Math\Matrix\NDArrayCL;
use Rindow\Math\Matrix\OpenCLBuffer;
use Rindow\OpenCL\Context;
use Rindow\OpenCL\CommandQueue;
use Interop\Polite\Math\Matrix\NDArray;
use Interop\Polite\Math\Matrix\OpenCL;
use ArrayObject;
use SplFixedArray;
use OutOfRangeException;
use InvalidArgumentException;
use LogicException;

/**
  * @requires extension rindow_opencl
  */
class Test extends TestCase
{
    public function getContext()
    {
        return new Context(OpenCL::CL_DEVICE_TYPE_DEFAULT);
    }

    public function getQueue($context)
    {
        return new CommandQueue($context);
    }

    public function testSimpleArrayoffsetGet()
    {
        $hostArray = new NDArrayPHP([[1,2],[3,4],[5,6]]);
        $context = $this->getContext();
        $queue = $this->getQueue($context);
        $array = new NDArrayCL($context,$queue,$hostArray->buffer(),$hostArray->dtype(),
            $hostArray->shape(),$hostArray->offset(),
            OpenCL::CL_MEM_READ_ONLY|OpenCL::CL_MEM_COPY_HOST_PTR);

        $this->assertEquals([3,2],$array->shape());
        $this->assertEquals(NDArray::float32,$array->dtype());
        $this->assertEquals(0,$array->offset());
        $this->assertEquals(2,$array->ndim());
        $this->assertEquals(
            OpenCL::CL_MEM_READ_ONLY|OpenCL::CL_MEM_COPY_HOST_PTR,
            $array->flags());
        $this->assertEquals(2,$array->ndim());
        $this->assertEquals(6,$array->size());
        $this->assertInstanceof(OpenCLBuffer::class,$array->buffer());
        $this->assertEquals(6*32/8,$array->buffer()->bytes());
        $this->assertEquals(NDArray::float32,$array->buffer()->dtype());
        $this->assertEquals(32/8,$array->buffer()->value_size());

        // offsetGet rank 1 ndarray object
        $newArray = $array[1];
        $this->assertEquals(1,$newArray->ndim());
        $this->assertEquals([3,4],$newArray->toNDArray()->toArray());

        // offsetGet range axis
        $newArray = $array[[1,2]];
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
        $hostArray = new NDArrayPHP([[1,2],[3,4],[5,6]]);
        $context = $this->getContext();
        $queue = $this->getQueue($context);
        $array = new NDArrayCL($context,$queue,$hostArray->buffer(),$hostArray->dtype(),
            $hostArray->shape(),$hostArray->offset(),
            OpenCL::CL_MEM_READ_WRITE|OpenCL::CL_MEM_COPY_HOST_PTR);

        $hostArray = new NDArrayPHP([11,12]);
        $srcarray = new NDArrayCL($context,$queue,$hostArray->buffer(),$hostArray->dtype(),
            $hostArray->shape(),$hostArray->offset(),
            OpenCL::CL_MEM_READ_ONLY|OpenCL::CL_MEM_COPY_HOST_PTR);
        $array[1] = $srcarray;
        $queue->finish();
        $this->assertEquals([[1,2],[11,12],[5,6]],$array->toNDArray()->toArray());

        $hostArray = new NDArrayPHP(4);
        $srcarray = new NDArrayCL($context,$queue,$hostArray->buffer(),$hostArray->dtype(),
            $hostArray->shape(),$hostArray->offset(),
            OpenCL::CL_MEM_READ_ONLY|OpenCL::CL_MEM_COPY_HOST_PTR);
        $array[1][1] = $srcarray;
        $queue->finish();
        $this->assertEquals([[1,2],[11,4],[5,6]],$array->toNDArray()->toArray());

    }

    public function testClone()
    {
        $hostArray = new NDArrayPHP([[1,2],[3,4],[5,6]],NDArray::int32);
        $context = $this->getContext();
        $queue = $this->getQueue($context);
        $array = new NDArrayCL($context,$queue,$hostArray->buffer(),$hostArray->dtype(),
            $hostArray->shape(),$hostArray->offset(),
            OpenCL::CL_MEM_READ_ONLY|OpenCL::CL_MEM_COPY_HOST_PTR);
        $this->assertEquals([3,2],$array->shape());
        $this->assertEquals(NDArray::int32,$array->dtype());

        $pattern=new NDArrayPhp([[0,0],[0,0],[0,0]],NDArray::int32);
        $array2 = clone $array;
        $array->buffer()->write($queue,$pattern->buffer());

        $this->assertEquals([[0,0],[0,0],[0,0]],$array->toArray());
        $this->assertEquals(NDArray::int32,$array2->dtype());
        $this->assertEquals(NDArray::int32,$array2->buffer()->dtype());
        $this->assertEquals([[1,2],[3,4],[5,6]],$array2->toArray());
    }
}
