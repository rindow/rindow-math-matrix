<?php
namespace RindowTest\Math\Matrix\LinearAlgebraCLTest;

//if(!class_exists('RindowTest\Math\Matrix\LinearAlgebraTest\Test')) {
//    require_once __DIR__.'/../../../../../../rindow-math-matrix/tests/RindowTest/Math/Matrix/LinearAlgebraTest.php';
//}
if(!class_exists('RindowTest\Math\Matrix\LinearAlgebraTest\Test')) {
    require_once __DIR__.'/LinearAlgebraTest.php';
}

use Rindow\Math\Plot\Plot;
use RindowTest\Math\Matrix\LinearAlgebraTest\Test as ORGTest;
use InvalidArgumentException;
use Rindow\Math\Matrix\MatrixOperator;

use Interop\Polite\Math\Matrix\NDArray;
use Interop\Polite\Math\Matrix\OpenCL;
use Rindow\Math\Matrix\LinearAlgebraCL;
use Rindow\Math\Matrix\NDArrayCL;
use Rindow\Math\Matrix\OpenCLMath;
use Rindow\Math\Matrix\OpenCLMathTunner;
use Rindow\OpenCL\Context;
use Rindow\OpenCL\CommandQueue;
use Rindow\CLBlast\Blas as CLBlastBlas;
use Rindow\CLBlast\Math as CLBlastMath;
use Rindow\OpenBLAS\Math as OpenBLASMath;
use Rindow\OpenBLAS\Lapack as OpenBLASLapack;

class TestMatrixOperator extends MatrixOperator
{
    public function laAccelerated($mode,array $options=null)
    {
        if($mode=='clblast') {
            if($this->clblastLA) {
                return $this->clblastLA;
            }
            if(!extension_loaded('rindow_clblast')) {
                throw new InvalidArgumentException('extension is not loaded');
            }
            if(isset($options['deviceType'])) {
                $deviceType = $options['deviceType'];
            } else {
                $deviceType = OpenCL::CL_DEVICE_TYPE_DEFAULT;
            }
            $context = new Context($deviceType);
            $queue = new CommandQueue($context);
            $clblastblas = new CLBlastBlas();
            $openclmath = new OpenCLMath($context,$queue);
            $clblastmath = new CLBlastMath();
            $la = new TestLinearAlgebraCL($context,$queue,
                $clblastblas,$openclmath,$clblastmath,
                $this->openblasmath,$this->openblaslapack);
            $this->clblastLA = $la;
            return $la;
        }
    }
}

class TestLinearAlgebraCL extends LinearAlgebraCL
{
    public function sumTest(
        NDArray $X,
        NDArray $R=null,
        object $events=null,
        object $waitEvents=null,
        int $mode = null
        )
    {
        if($R==null) {
            $R = $this->alloc([],$X->dtype(),OpenCL::CL_MEM_READ_WRITE);
        }
        $N = $X->size();
        $RR = $R->buffer();
        $offR = $R->offset();
        $XX = $X->buffer();
        $offX = $X->offset();
        $dtype = $X->dtype();

        switch($mode) {
            case 1: {
                $this->openclmath->sum1($N,$RR,$offR,$XX,$offX,1,$events,$waitEvents);
                break;
            }
            case 2: {
                $this->openclmath->sum2($N,$RR,$offR,$XX,$offX,1,$events,$waitEvents);
                break;
            }
            case 3: {
                $this->openclmath->sum3($N,$RR,$offR,$XX,$offX,1,$events,$waitEvents);
                break;
            }
            default:
                throw new InvalidArgumentException("invalid mode: ".$mode);
        }

        if($this->blocking) {
            $this->finish();
        }
        if($this->scalarNumeric) {
            return $R->toArray();
        }
        return $R;
    }
/*
    public function reduceSumTest(
        NDArray $A,
        int $axis=null,
        NDArray $X=null,
        $dtypeX=null,
        $events=null,$waitEvents=null,
        $mode = null
        ) : NDArray
    {
        if($axis===null)
            $axis = 0;
        if($axis!==0 && $axis!==1 && $axis!==-1)
            throw new InvalidArgumentException('"axis" must be 0 or 1 or -1.');
        $shapeA = $A->shape();
        if($axis==0) {
            $trans = true;
            $m = array_shift($shapeA);
            $n = (int)array_product($shapeA);
            $cols = $m;
            $rows = $n;
        } else {
            $trans = false;
            $n = array_pop($shapeA);
            $m = (int)array_product($shapeA);
            $cols = $n;
            $rows = $m;
        }

        if($dtypeX===null) {
            $dtypeX = $A->dtype();
        }
        if($X==null) {
            $X = $this->alloc([$rows],$dtypeX);
        } else {
            if($X->shape()!=[$rows]) {
                $shapeError = '('.implode(',',$A->shape()).'),('.implode(',',$X->shape()).')';
                throw new InvalidArgumentException("Unmatch shape of dimension: ".$shapeError);
            }
        }

        $AA = $A->buffer();
        $offA = $A->offset();
        $XX = $X->buffer();
        $offX = $X->offset();

        switch($mode) {
            case 0: {
                $this->openclmath->reduceSum0(
                    $trans,
                    $m,
                    $n,
                    $AA,$offA,$n,
                    $XX,$offX,1,
                    $events,$waitEvents
                );
                break;
            }
            case 1: {
                $this->openclmath->reduceSum1(
                    $trans,
                    $m,
                    $n,
                    $AA,$offA,$n,
                    $XX,$offX,1,
                    $events,$waitEvents
                );
                break;
            }
            case 2: {
                $this->openclmath->reduceSum2(
                    $trans,
                    $m,
                    $n,
                    $AA,$offA,$n,
                    $XX,$offX,1,
                    $events,$waitEvents
                );
                break;
            }
            case 3: {
                $this->openclmath->reduceSum3(
                    $trans,
                    $m,
                    $n,
                    $AA,$offA,$n,
                    $XX,$offX,1,
                    $events,$waitEvents
                );
                break;
            }
            default:
                throw new InvalidArgumentException("invalid mode: ".$mode);
        }

        if($this->blocking) {
            $this->finish();
        }
        return $X;
    }
*/
    public function reduceSumTest( //reducesumex
        NDArray $A,
        int $axis=null,
        NDArray $B=null,
        $dtype=null,
        $events=null,$waitEvents=null,
        $mode = null
        ) : NDArray
    {
        $ndim = $A->ndim();
        if($axis<0) {
            $axis = $ndim+$axis;
        }
        if($axis<0 || $axis>$ndim-1) {
            throw new InvalidArgumentException("Invalid axis");
        }
        $postfixShape = $A->shape();
        $prefixShape = [];
        for($i=0;$i<$axis;$i++) {
            $prefixShape[] = array_shift($postfixShape);
        }
        $n = array_shift($postfixShape);
        $m = array_product($prefixShape);
        $k = array_product($postfixShape);
        $outputShape = array_merge($prefixShape,$postfixShape);
        if($dtype===null) {
            $dtype = $A->dtype();
        }
        if($B==null) {
            $B = $this->alloc($outputShape,$dtype);
        } else {
            if($B->shape()!=$outputShape) {
                $shapeError = '('.implode(',',$A->shape()).'),('.implode(',',$B->shape()).')';
                throw new InvalidArgumentException("Unmatch shape of dimension: ".$shapeError);
            }
        }

        $AA = $A->buffer();
        $offA = $A->offset();
        $BB = $B->buffer();
        $offB = $B->offset();

        switch($mode) {
            case 0: {
                $this->openclmath->reduceSum0(
                    $m,
                    $n,
                    $k,
                    $AA,$offA,
                    $BB,$offB,
                    $events,$waitEvents
                );
                break;
            }
            case 1: {
                $this->openclmath->reduceSum1(
                    $m,
                    $n,
                    $k,
                    $AA,$offA,
                    $BB,$offB,
                    $events,$waitEvents
                );
                break;
            }
            case 2: {
                $this->openclmath->reduceSum2(
                    $m,
                    $n,
                    $k,
                    $AA,$offA,
                    $BB,$offB,
                    $events,$waitEvents
                );
                break;
            }
            case 3: {
                $this->openclmath->reduceSum3(
                    $m,
                    $n,
                    $k,
                    $AA,$offA,
                    $BB,$offB,
                    $events,$waitEvents
                );
                break;
            }
            default:
                throw new InvalidArgumentException("invalid mode: ".$mode);
        }

        if($this->blocking) {
            $this->finish();
        }
        return $B;
    }
/*
    public function reduceMaxTest(
        NDArray $A,
        int $axis=null,
        NDArray $X=null,
        $dtypeX=null,
        $events=null,$waitEvents=null,
        $mode = null
        ) : NDArray
    {
        if($axis===null)
            $axis = 0;
        if($axis!==0 && $axis!==1 && $axis!==-1)
            throw new InvalidArgumentException('"axis" must be 0 or 1 or -1.');
        $shapeA = $A->shape();
        if($axis==0) {
            $trans = true;
            $m = array_shift($shapeA);
            $n = (int)array_product($shapeA);
            $cols = $m;
            $rows = $n;
        } else {
            $trans = false;
            $n = array_pop($shapeA);
            $m = (int)array_product($shapeA);
            $cols = $n;
            $rows = $m;
        }

        if($dtypeX===null) {
            $dtypeX = $A->dtype();
        }
        if($X==null) {
            $X = $this->alloc([$rows],$dtypeX);
        } else {
            if($X->shape()!=[$rows]) {
                $shapeError = '('.implode(',',$A->shape()).'),('.implode(',',$X->shape()).')';
                throw new InvalidArgumentException("Unmatch shape of dimension: ".$shapeError);
            }
        }

        $AA = $A->buffer();
        $offA = $A->offset();
        $XX = $X->buffer();
        $offX = $X->offset();

        switch($mode) {
            case 0: {
                $this->openclmath->reduceMax0(
                    $trans,
                    $m,
                    $n,
                    $AA,$offA,$n,
                    $XX,$offX,1,
                    $events,$waitEvents
                );
                break;
            }
            case 1: {
                $this->openclmath->reduceMax1(
                    $trans,
                    $m,
                    $n,
                    $AA,$offA,$n,
                    $XX,$offX,1,
                    $events,$waitEvents
                );
                break;
            }
            case 2: {
                $this->openclmath->reduceMax2(
                    $trans,
                    $m,
                    $n,
                    $AA,$offA,$n,
                    $XX,$offX,1,
                    $events,$waitEvents
                );
                break;
            }
            case 3: {
                $this->openclmath->reduceMax3(
                    $trans,
                    $m,
                    $n,
                    $AA,$offA,$n,
                    $XX,$offX,1,
                    $events,$waitEvents
                );
                break;
            }
            default:
                throw new InvalidArgumentException("invalid mode: ".$mode);
        }

        if($this->blocking) {
            $this->finish();
        }
        return $X;
    }
*/
    public function reduceMaxTest( //reducemaxex
        NDArray $A,
        int $axis=null,
        NDArray $B=null,
        $dtype=null,
        $events=null,$waitEvents=null,
        $mode = null
        ) : NDArray
    {
        $ndim = $A->ndim();
        $orgaxis = $axis;
        if($axis<0) {
            $axis = $ndim+$axis;
        }
        if($axis<0 || $axis>$ndim-1) {
            throw new InvalidArgumentException("Invalid axis: ".$orgaxis);
        }
        $postfixShape = $A->shape();
        $prefixShape = [];
        for($i=0;$i<$axis;$i++) {
            $prefixShape[] = array_shift($postfixShape);
        }
        $n = array_shift($postfixShape);
        $m = array_product($prefixShape);
        $k = array_product($postfixShape);
        $outputShape = array_merge($prefixShape,$postfixShape);
        if($dtype===null) {
            $dtype = $A->dtype();
        }
        if($B==null) {
            $B = $this->alloc($outputShape,$dtype);
        } else {
            if($B->shape()!=$outputShape) {
                $shapeError = '('.implode(',',$A->shape()).'),('.implode(',',$B->shape()).')';
                throw new InvalidArgumentException("Unmatch shape of dimension: ".$shapeError);
            }
        }

        $AA = $A->buffer();
        $offA = $A->offset();
        $BB = $B->buffer();
        $offB = $B->offset();

        switch($mode) {
            case 0: {
                $this->openclmath->reduceMax0(
                    $m,
                    $n,
                    $k,
                    $AA,$offA,
                    $BB,$offB,
                    $events,$waitEvents
                );
                break;
            }
            case 1: {
                $this->openclmath->reduceMax1(
                    $m,
                    $n,
                    $k,
                    $AA,$offA,
                    $BB,$offB,
                    $events,$waitEvents
                );
                break;
            }
            case 2: {
                $this->openclmath->reduceMax2(
                    $m,
                    $n,
                    $k,
                    $AA,$offA,
                    $BB,$offB,
                    $events,$waitEvents
                );
                break;
            }
            case 3: {
                $this->openclmath->reduceMax3(
                    $m,
                    $n,
                    $k,
                    $AA,$offA,
                    $BB,$offB,
                    $events,$waitEvents
                );
                break;
            }
            default:
                throw new InvalidArgumentException("invalid mode: ".$mode);
        }

        if($this->blocking) {
            $this->finish();
        }
        return $B;
    }
/*
    public function reduceArgMaxTest(
        NDArray $A,
        int $axis,
        NDArray $X=null,
        $dtypeX=null,
        $events=null,$waitEvents=null,
        $mode=null
        ) : NDArray
    {
        if($axis===null)
            $axis = 0;
        if($axis!==0 && $axis!==1 && $axis!==-1)
            throw new InvalidArgumentException('"axis" must be 0 or 1 or -1.');
        $shapeA = $A->shape();
        if($axis==0) {
            $trans = true;
            $m = array_shift($shapeA);
            $n = (int)array_product($shapeA);
            $rows = $n;
        } else {
            $trans = false;
            $n = array_pop($shapeA);
            $m = (int)array_product($shapeA);
            $rows = $m;
        }

        if($dtypeX==null) {
            $dtypeX = NDArray::int32;
        }
        if($X==null) {
            $X = $this->alloc([$rows],$dtypeX);
        } else {
            if($X->shape()!=[$rows]) {
                $shapeError = '('.implode(',',$A->shape()).'),('.implode(',',$X->shape()).')';
                throw new InvalidArgumentException("Unmatch shape of dimension: ".$shapeError);
            }
        }

        $AA = $A->buffer();
        $offA = $A->offset();
        $XX = $X->buffer();
        $offX = $X->offset();

        switch($mode) {
            case 0: {
                $this->openclmath->reduceArgMax0(
                    $trans,
                    $m,
                    $n,
                    $AA,$offA,$n,
                    $XX,$offX,1,
                    $events,$waitEvents
                );
                break;
            }
            case 1: {
                $this->openclmath->reduceArgMax1(
                    $trans,
                    $m,
                    $n,
                    $AA,$offA,$n,
                    $XX,$offX,1,
                    $events,$waitEvents
                );
                break;
            }
            case 2: {
                $this->openclmath->reduceArgMax2(
                    $trans,
                    $m,
                    $n,
                    $AA,$offA,$n,
                    $XX,$offX,1,
                    $events,$waitEvents
                );
                break;
            }
            case 3: {
                $this->openclmath->reduceArgMax3(
                    $trans,
                    $m,
                    $n,
                    $AA,$offA,$n,
                    $XX,$offX,1,
                    $events,$waitEvents
                );
                break;
            }
            default:
                throw new InvalidArgumentException("invalid mode: ".$mode);
        }

        if($this->blocking) {
            $this->finish();
        }
        return $X;
    }
*/
    public function reduceArgMaxTest( //reduceargmaxex
        NDArray $A,
        int $axis=null,
        NDArray $B=null,
        $dtype=null,
        $events=null,$waitEvents=null,
        $mode = null
        ) : NDArray
    {
        $ndim = $A->ndim();
        if($axis<0) {
            $axis = $ndim+$axis;
        }
        if($axis<0 || $axis>$ndim-1) {
            throw new InvalidArgumentException("Invalid axis");
        }
        $postfixShape = $A->shape();
        $prefixShape = [];
        for($i=0;$i<$axis;$i++) {
            $prefixShape[] = array_shift($postfixShape);
        }
        $n = array_shift($postfixShape);
        $m = array_product($prefixShape);
        $k = array_product($postfixShape);
        $outputShape = array_merge($prefixShape,$postfixShape);
        if($dtype===null) {
            $dtype = NDArray::uint32;
        }
        if($B==null) {
            $B = $this->alloc($outputShape,$dtype);
        } else {
            if($B->shape()!=$outputShape) {
                $shapeError = '('.implode(',',$A->shape()).'),('.implode(',',$B->shape()).')';
                throw new InvalidArgumentException("Unmatch shape of dimension: ".$shapeError);
            }
        }

        $AA = $A->buffer();
        $offA = $A->offset();
        $BB = $B->buffer();
        $offB = $B->offset();

        switch($mode) {
            case 0: {
                $this->openclmath->reduceArgMax0(
                    $m,
                    $n,
                    $k,
                    $AA,$offA,
                    $BB,$offB,
                    $events,$waitEvents
                );
                break;
            }
            case 1: {
                $this->openclmath->reduceArgMax1(
                    $m,
                    $n,
                    $k,
                    $AA,$offA,
                    $BB,$offB,
                    $events,$waitEvents
                );
                break;
            }
            case 2: {
                $this->openclmath->reduceArgMax2(
                    $m,
                    $n,
                    $k,
                    $AA,$offA,
                    $BB,$offB,
                    $events,$waitEvents
                );
                break;
            }
            case 3: {
                $this->openclmath->reduceArgMax3(
                    $m,
                    $n,
                    $k,
                    $AA,$offA,
                    $BB,$offB,
                    $events,$waitEvents
                );
                break;
            }
            default:
                throw new InvalidArgumentException("invalid mode: ".$mode);
        }

        if($this->blocking) {
            $this->finish();
        }
        return $B;
    }

    public function softmaxTest(
        NDArray $X,
        object $events=null, object $waitEvents=null,
        $mode = null
        ) : NDArray
    {
        if($X->ndim()!=2) {
            throw new InvalidArgumentException('"X" must be 2-D dimension');
        }

        [$m,$n] = $X->shape();
        $XX = $X->buffer();
        $offX = $X->offset();
        $ldA = $n;
        switch($mode) {
            case 0: {
                $this->openclmath->softmax0(
                    $m,
                    $n,
                    $XX,$offX,$ldA,
                    $events,$waitEvents);
                break;
            }
            case 1: {
                $this->openclmath->softmax1(
                    $m,
                    $n,
                    $XX,$offX,$ldA,
                    $events,$waitEvents);
                break;
            }
            case 2: {
                $this->openclmath->softmax2(
                    $m,
                    $n,
                    $XX,$offX,$ldA,
                    $events,$waitEvents);
                break;
            }
        }
        if($this->blocking) {
            $this->finish();
        }
        return $X;
    }
}
/**
*   @requires extension rindow_clblast
*/
class Test extends ORGTest
{
    public function newMatrixOperator()
    {
        $mo = new TestMatrixOperator();
        if(extension_loaded('rindow_openblas')) {
            $mo->blas()->forceBlas(true);
            $mo->lapack()->forceLapack(true);
            $mo->math()->forceMath(true);
        }
        return $mo;
    }

    public function newLA($mo)
    {
        $la = $mo->laAccelerated('clblast');
        $la->blocking(true);
        $la->scalarNumeric(true);
        return $la;
    }

    public function ndarray($x)
    {
        if($x instanceof NDArrayCL) {
            $x = $x->toNDArray();
        }
        return $x;
    }

    public function modeProviderNoZero()
    {
        return [
            'mode 1' => [1],
            'mode 2' => [2],
            'mode 3' => [3],
        ];
    }

    public function modeProvider()
    {
        return [
            'mode 0' => [0],
            'mode 1' => [1],
            'mode 2' => [2],
            'mode 3' => [3],
        ];
    }

    public function modeProvider4mode()
    {
        return [
            'mode 0' => [0],
            'mode 1' => [1],
            'mode 2' => [2],
            'mode 3' => [3],
            'mode 4' => [4],
        ];
    }

    public function testRotg()
    {
        $this->markTestSkipped('Unsuppored function on clblast');
    }

    public function testRot()
    {
        $this->markTestSkipped('Unsuppored function on clblast');
    }

    public function testIm2col2dclblastnormal()
    {
        $mo = $this->newMatrixOperator();
        $la = $this->newLA($mo);

        //$kernel_mode = \Rindow\CLBlast\Math::CONVOLUTION;
        $kernel_mode = \Rindow\CLBlast\Math::CROSS_CORRELATION;
        $batches = 1;
        $im_h = 5;
        $im_w = 5;
        $channels = 1;
        $kernel_h = 3;
        $kernel_w = 3;
        $stride_h = 1;
        $stride_w = 1;
        $padding = true;
        $dilation_h = 2;
        $dilation_w = 2;
        //$channels_first = null;
        //$cols_channels_first=null;
        $cols = null;

        $images = $la->array($mo->arange(
            $batches*
            $channels* // channels_first
            $im_h*$im_w,
            null,null,
            NDArray::float32
        ))->reshape([
            $batches,
            $channels, // channels_first
            $im_h,
            $im_w,
        ]);
        $cols = $la->im2col2dclblast(
            $reverse=false,
            $kernel_mode,
            $images,
            $filterSize=[
                $kernel_h,$kernel_w],
            $strides=[
                $stride_h,$stride_w],
            $padding,
            $dilation_rate=[
                $dilation_h,$dilation_w]
            //$channels_first,
            //$cols_channels_first
        );
        //$out_h = 2;
        //$out_w = 2;
        //echo "image=($im_h,$im_w)\n";
        $out_h = intval(floor(($im_h-($kernel_h-1)*$dilation_h-1)/$stride_h)+1);
        $out_w = intval(floor(($im_w-($kernel_w-1)*$dilation_w-1)/$stride_w)+1);
        if($padding) {
            $out_h = $im_h;
            $out_w = $im_w;
            $padding_h = (int)(($im_h-1)*$stride_h-$im_h+($kernel_h-1)*$dilation_h+1);
            if($padding_h%2) {
                $out_h++;
            }
            $padding_h = $padding_h ? (int)floor($padding_h/2+0.5) : 0;
            $padding_w = (int)(($im_w-1)*$stride_w-$im_w+($kernel_w-1)*$dilation_w+1);
            if($padding_w%2) {
                $out_w++;
            }
            $padding_w = $padding_w ? (int)floor($padding_w/2+0.5) : 0;
        }
        //echo "image=($out_h,$out_w)\n";

        $this->assertEquals(
            [
                $batches,
                $channels, // channels_first
                $kernel_h,$kernel_w, // filters first
                $out_h,$out_w,
            ],
            $cols->shape()
        );
        $trues = $this->newArray($cols->shape());
        $truesBuffer = $trues->buffer();
        for($batch_id=0;$batch_id<$batches;$batch_id++) {
            for($channel_id=0;$channel_id<$channels;$channel_id++) {
                for($kernel_y=0;$kernel_y<$kernel_h;$kernel_y++) {
                    for($kernel_x=0;$kernel_x<$kernel_w;$kernel_x++) {
                        for($im_y=0;$im_y<$out_h;$im_y++) {
                            for($im_x=0;$im_x<$out_w;$im_x++) {
                                $input_y = $im_y*$stride_h+$kernel_y*$dilation_h-$padding_h;
                                $input_x = $im_x*$stride_w+$kernel_x*$dilation_w-$padding_w;
                                # channel first
                                $input_id = ((($batch_id*$channels+$channel_id)*$im_h+$input_y)*$im_w+$input_x);
                                # channel kernel stride
                                $cols_id = ((((($batch_id*$channels+$channel_id)
                                            *$kernel_h+$kernel_y)*$kernel_w+$kernel_x)*$out_h+$im_y)*$out_w+$im_x);
                                if($input_y>=0 && $input_y<$im_h && $input_x>=0 && $input_x<$im_w) {
                                    $truesBuffer[$cols_id] = $input_id;
                                }
                            }
                        }
                    }
                }
            }
        }
        $this->assertEquals($trues->toArray(),$cols->toArray());

        // channels_last
        //for($batch_id=0;$batch_id<$batches;$batch_id++) {
        //    for($out_y=0;$out_y<$out_h;$out_y++) {
        //        echo "col_h=$out_y\n";
        //        for($out_x=0;$out_x<$out_w;$out_x++) {
        //            echo "col_w=$out_x\n";
        //            for($kernel_y=$kernel_h-1;$kernel_y>=0;$kernel_y--) {
        //                for($kernel_x=$kernel_w-1;$kernel_x>=0;$kernel_x--) {
        //                    echo "[";
        //                    for($channel_id=0;$channel_id<$channels;$channel_id++) {
        //                        echo $cols[$batch_id][$channel_id][$kernel_y][$kernel_x][$out_y][$out_x]->toArray().",";
        //                    }
        //                    echo "],";
        //                }
        //                echo "\n";
        //            }
        //        }
        //    }
        //}

        // channels_first kernel last
        //for($batch_id=0;$batch_id<$batches;$batch_id++) {
        //    for($channel_id=0;$channel_id<$channels;$channel_id++) {
        //        for($out_y=0;$out_y<$out_h;$out_y++) {
        //            echo "col_h=$out_y\n";
        //            for($out_x=0;$out_x<$out_w;$out_x++) {
        //                echo "col_w=$out_x\n";
        //                for($kernel_y=0;$kernel_y<$kernel_h;$kernel_y++) {
        //                    echo "[";
        //                    for($kernel_x=0;$kernel_x<$kernel_w;$kernel_x++) {
        //                        echo sprintf('%2d',$cols[$batch_id][$channel_id][$kernel_y][$kernel_x][$out_y][$out_x]->toArray()).",";
        //                    }
        //                    echo "],";
        //                    echo "\n";
        //                }
        //            }
        //        }
        //    }
        //}

        //echo "---------------------------\n";
        //foreach ($cols->toArray() as $batch) {
        //    foreach ($batch as $key => $channel) {
        //        echo "channel=$key\n";
        //        foreach ($channel as $key => $kernel_h_value) {
        //            echo "kernel_h=$key\n";
        //            foreach ($kernel_h_value as $key => $kernel_w_value) {
        //                echo "kernel_w=$key\n";
        //                foreach ($kernel_w_value as $k_h => $col_h_value) {
        //                    echo "[";
        //                    foreach ($col_h_value as $k_w => $value) {
        //                        echo sprintf('%2d',(int)$value).",";
        //                    }
        //                    echo "]\n";
        //                }
        //            }
        //        }
        //    }
        //}

        $newImages = $la->zerosLike($images);
        $cols = $la->im2col2dclblast(
            $reverse=true,
            $kernel_mode,
            $newImages,
            $filterSize=[
                $kernel_h,$kernel_w],
            $strides=[
                $stride_h,$stride_w],
            $padding,
            $dilation_rate=[
                $dilation_h,$dilation_w],
            //$channels_first,
            //$cols_channels_first
            $cols
        );
        $imagesTrues = $this->newArray($images->shape());
        $imageBuffer = $imagesTrues->buffer();
        for($batch_id=0;$batch_id<$batches;$batch_id++) {
            for($channel_id=0;$channel_id<$channels;$channel_id++) {
                for($kernel_y=0;$kernel_y<$kernel_h;$kernel_y++) {
                    for($kernel_x=0;$kernel_x<$kernel_w;$kernel_x++) {
                        for($im_y=0;$im_y<$out_h;$im_y++) {
                            for($im_x=0;$im_x<$out_w;$im_x++) {
                                $input_y = $im_y*$stride_h+$kernel_y*$dilation_h-$padding_h;
                                $input_x = $im_x*$stride_w+$kernel_x*$dilation_w-$padding_w;
                                # channel first
                                $input_id = ((($batch_id*$channels+$channel_id)*$im_h+$input_y)*$im_w+$input_x);
                                # channel kernel stride
                                $cols_id = ((((($batch_id*$channels+$channel_id)
                                            *$kernel_h+$kernel_y)*$kernel_w+$kernel_x)*$out_h+$im_y)*$out_w+$im_x);
                                if($input_y>=0 && $input_y<$im_h && $input_x>=0 && $input_x<$im_w) {
                                    $value = $imageBuffer[$input_id];
                                    $imageBuffer[$input_id] = $value + $truesBuffer[$cols_id];
                                }
                            }
                        }
                    }
                }
            }
        }
        $this->assertEquals($imagesTrues->toArray(),$newImages->toArray());
        // result is Not equal to original
        // because to sum for back propagation
        //$this->assertEquals(
        //    $images->toArray(),
        //    $newImages->toArray()
        //);
        #foreach ($newImages->toArray() as $batch) {
        #    foreach ($batch as $key => $channel) {
        #        foreach ($channel as $key => $im_y) {
        #            #echo "kernel_h=$key\n";
        #            echo "[";
        #            foreach ($im_y as $key => $value) {
        #                #echo "kernel_w=$key\n";
        #                echo sprintf('%3d',$value).",";
        #            }
        #            echo "],";
        #            echo "\n";
        #        }
        #        echo "\n";
        #    }
        #}
    }

    public function testIm2col2dclblastlarge()
    {
        $mo = $this->newMatrixOperator();
        $la = $this->newLA($mo);

        //$kernel_mode = \Rindow\CLBlast\Math::CONVOLUTION;
        $kernel_mode = \Rindow\CLBlast\Math::CROSS_CORRELATION;
        $batches = 1;
        $im_h = 128;
        $im_w = 128;
        $channels = 1;
        $kernel_h = 3;
        $kernel_w = 3;
        $stride_h = 1;
        $stride_w = 1;
        $padding = null;
        $dilation_h = 1;
        $dilation_w = 1;
        //$channels_first = null;
        //$cols_channels_first=null;
        $cols = null;

        $images = $la->array($mo->arange(
            $batches*
            $channels* // channels_first
            $im_h*$im_w,
            null,null,
            NDArray::float32
        ))->reshape([
            $batches,
            $channels, // channels_first
            $im_h,
            $im_w,
        ]);
        $cols = $la->im2col2dclblast(
            $reverse=false,
            $kernel_mode,
            $images,
            $filterSize=[
                $kernel_h,$kernel_w],
            $strides=[
                $stride_h,$stride_w],
            $padding,
            $dilation_rate=[
                $dilation_h,$dilation_w]
            //$channels_first,
            //$cols_channels_first
        );
        $newImages = $la->zerosLike($images);
        $cols = $la->im2col2dclblast(
            $reverse=true,
            $kernel_mode,
            $newImages,
            $filterSize=[
                $kernel_h,$kernel_w],
            $strides=[
                $stride_h,$stride_w],
            $padding,
            $dilation_rate=[
                $dilation_h,$dilation_w],
            //$channels_first,
            //$cols_channels_first
            $cols
        );
        $this->assertTrue(true);
    }

    public function testIm2col2dSpeedCL()
    {
        $mo = $this->newMatrixOperator();
        $la = $this->newLA($mo);
        if(!self::$speedtest) {
            $this->markTestSkipped('Speed measurement');
            return;
        }
        $mo = $this->newMatrixOperator();
        $la = $this->newLA($mo);
        if($la->getConfig()=='PhpBlas') {
            $this->assertTrue(true);
            return;
        }
        echo "\n";
        $kernel_mode = \Rindow\CLBlast\Math::CROSS_CORRELATION;
        $batches = 8;
        $im_h = 512;
        $im_w = 512;
        $channels = 3;
        $images = $la->alloc([$batches,$channels,$im_h,$im_w]);
        $la->ones($images);
        $kernel_h = 3;
        $kernel_w = 3;
        $stride_h = 1;
        $stride_w = 1;
        $padding = null;
        //$channels_first = null;
        $dilation_h = 1;
        $dilation_w = 1;
        //$cols_channels_first=null;
        $cols = null;
        echo "im=($im_h,$im_w),knl=($kernel_h,$kernel_w),batches=$batches\n";

        $cols = $la->im2col2dclblast(
            $reverse=false,
            $kernel_mode,
            $images,
            $filterSize=[
                $kernel_h,$kernel_w],
            $strides=[
                $stride_h,$stride_w],
            $padding,
            //$channels_first,
            $dilation_rate=[
                $dilation_h,$dilation_w]
            //$cols_channels_first
        );
        $start = hrtime(true);
        $cols = $la->im2col2dclblast(
            $reverse=false,
            $kernel_mode,
            $images,
            $filterSize=[
                $kernel_h,$kernel_w],
            $strides=[
                $stride_h,$stride_w],
            $padding,
            //$channels_first,
            $dilation_rate=[
                $dilation_h,$dilation_w]
            //$cols_channels_first
        );
        $end = hrtime(true);
        echo (explode(' ',$la->getConfig()))[0].'CL='.number_format($end-$start)."\n";

        $newImages = $la->alloc([$batches,$channels,$im_h,$im_w]);
        $la->im2col2dclblast(
            $reverse=true,
            $kernel_mode,
            $newImages,
            $filterSize=[
                $kernel_h,$kernel_w],
            $strides=[
                $stride_h,$stride_w],
            $padding,
            //$channels_first,
            $dilation_rate=[
                $dilation_h,$dilation_w],
            //$cols_channels_first
            $cols
        );
        $start = hrtime(true);
        $la->im2col2dclblast(
            $reverse=true,
            $kernel_mode,
            $newImages,
            $filterSize=[
                $kernel_h,$kernel_w],
            $strides=[
                $stride_h,$stride_w],
            $padding,
            //$channels_first,
            $dilation_rate=[
                $dilation_h,$dilation_w],
            //$cols_channels_first
            $cols
        );
        $end = hrtime(true);
        echo (explode(' ',$la->getConfig()))[0].'CL='.number_format($end-$start)."\n";

        $kernel_mode = \Rindow\CLBlast\Math::CROSS_CORRELATION;
        $batches = 256;
        $im_h = 28;
        $im_w = 28;
        $channels = 3;
        $images = $la->alloc([$batches,$channels,$im_h,$im_w]);
        $la->ones($images);
        $kernel_h = 3;
        $kernel_w = 3;
        $stride_h = 1;
        $stride_w = 1;
        $padding = null;
        //$channels_first = null;
        $dilation_h = 1;
        $dilation_w = 1;
        //$cols_channels_first=null;
        $cols = null;
        echo "im=($im_h,$im_w),knl=($kernel_h,$kernel_w),batches=$batches\n";

        $cols = $la->im2col2dclblast(
            $reverse=false,
            $kernel_mode,
            $images,
            $filterSize=[
                $kernel_h,$kernel_w],
            $strides=[
                $stride_h,$stride_w],
            $padding,
            //$channels_first,
            $dilation_rate=[
                $dilation_h,$dilation_w]
            //$cols_channels_first
        );
        $start = hrtime(true);
        $cols = $la->im2col2dclblast(
            $reverse=false,
            $kernel_mode,
            $images,
            $filterSize=[
                $kernel_h,$kernel_w],
            $strides=[
                $stride_h,$stride_w],
            $padding,
            //$channels_first,
            $dilation_rate=[
                $dilation_h,$dilation_w]
            //$cols_channels_first
        );
        $end = hrtime(true);
        echo (explode(' ',$la->getConfig()))[0].'CL='.number_format($end-$start)."\n";

        $newImages = $la->alloc([$batches,$channels,$im_h,$im_w]);
        $la->im2col2dclblast(
            $reverse=true,
            $kernel_mode,
            $newImages,
            $filterSize=[
                $kernel_h,$kernel_w],
            $strides=[
                $stride_h,$stride_w],
            $padding,
            //$channels_first,
            $dilation_rate=[
                $dilation_h,$dilation_w],
            //$cols_channels_first
            $cols
        );
        $start = hrtime(true);
        $la->im2col2dclblast(
            $reverse=true,
            $kernel_mode,
            $newImages,
            $filterSize=[
                $kernel_h,$kernel_w],
            $strides=[
                $stride_h,$stride_w],
            $padding,
            //$channels_first,
            $dilation_rate=[
                $dilation_h,$dilation_w],
            //$cols_channels_first
            $cols
        );
        $end = hrtime(true);
        echo (explode(' ',$la->getConfig()))[0].'CL='.number_format($end-$start)."\n";

        $this->assertTrue(true);
    }

    /**
    * @dataProvider modeProviderNoZero
    */
    public function testSumOpenCL($mode)
    {
        $dtype = NDArray::float32;
        $mo = $this->newMatrixOperator();
        $la = $this->newLA($mo);
        $x = $la->array([[1,2,-3],[-4,5,-6]],$dtype);
        $ret = $la->sumTest($x,null,null,null,$mode);
        $this->assertEquals(1+2-3-4+5-6,$ret);

        // 1
        $x = $la->alloc([1],$dtype);
        $la->fill(1,$x);
        $ret = $la->sumTest($x,null,null,null,$mode);
        $this->assertEquals(1,$ret);

        // 2
        $x = $la->alloc([2],$dtype);
        $la->fill(1,$x);
        $ret = $la->sumTest($x,null,null,null,$mode);
        $this->assertEquals(2,$ret);

        // 3
        $x = $la->alloc([3],$dtype);
        $la->fill(1,$x);
        $ret = $la->sumTest($x,null,null,null,$mode);
        $this->assertEquals(3,$ret);

        // 4
        $x = $la->alloc([4],$dtype);
        $la->fill(1,$x);
        $ret = $la->sumTest($x,null,null,null,$mode);
        $this->assertEquals(4,$ret);

        // 255
        $x = $la->alloc([256],$dtype);
        $la->fill(1,$x);
        $ret = $la->sumTest($x,null,null,null,$mode);
        $this->assertEquals(256,$ret);

        // 256
        $x = $la->alloc([256],$dtype);
        $la->fill(1,$x);
        $ret = $la->sumTest($x,null,null,null,$mode);
        $this->assertEquals(256,$ret);

        if($mode>1) {
            // 257
            $x = $la->alloc([257],$dtype);
            $la->fill(1,$x);
            $ret = $la->sumTest($x,null,null,null,$mode);
            $this->assertEquals(257,$ret);

            // 65535
            $x = $la->alloc([65536],$dtype);
            $la->fill(1,$x);
            $ret = $la->sumTest($x,null,null,null,$mode);
            $this->assertEquals(65536,$ret);

            // 65536
            $x = $la->alloc([65536],$dtype);
            $la->fill(1,$x);
            $ret = $la->sumTest($x,null,null,null,$mode);
            $this->assertEquals(65536,$ret);

            // 65537
            $x = $la->alloc([65537],$dtype);
            $la->fill(1,$x);
            $ret = $la->sumTest($x,null,null,null,$mode);
            $this->assertEquals(65537,$ret);
        }
    }

    /**
    * dataProvider modeProvider
    */
/*
    public function testReduceSumOpenCL($mode)
    {
        $mo = $this->newMatrixOperator();
        $la = $this->newLA($mo);
        $x = $la->array([[1,2,3],[4,5,6]]);
        $y = $la->reduceSumTest($x,$axis=0,null,null,null,null,$mode);
        $this->assertEquals([5,7,9],$y->toArray());
        $y = $la->reduceSumTest($x,$axis=1,null,null,null,null,$mode);
        $this->assertEquals([6,15],$y->toArray());

        // ***** CAUTION ******
        // 3d array as 2d array
        $x = $la->array([[[1,2],[3,4]],[[5,6],[7,8]]]);
        $y = $la->reduceSumTest($x,$axis=0,null,null,null,null,$mode);
        $this->assertEquals([6,8,10,12],$y->toArray());
        $x = $la->array([[[1,2],[3,4]],[[5,6],[7,8]]]);
        $y = $la->reduceSumTest($x,$axis=1,null,null,null,null,$mode);
        $this->assertEquals([3,7,11,15],$y->toArray());

        // with offset
        $x = $la->array([[[9,9,9],[9,9,9]],[[1,2,3],[4,5,6]]]);
        $x = $x[1];
        $this->assertEquals([5,7,9],$la->reduceSumTest($x,$axis=0,null,null,null,null,$mode)->toArray());
        $this->assertEquals([6,15],$la->reduceSumTest($x,$axis=1,null,null,null,null,$mode)->toArray());

        // ceil thread
        $x = $la->alloc([33,33]);
        $la->ones($x);
        $trues = $mo->full([33],33)->toArray();
        $this->assertEquals($trues,$la->reduceSumTest($x,$axis=0,null,null,null,null,$mode)->toArray());
        $this->assertEquals($trues,$la->reduceSumTest($x,$axis=1,null,null,null,null,$mode)->toArray());


        if($mode!=1) {
            // 257
            $x = $la->alloc([257],NDArray::float32);
            $la->fill(1,$x);
            $ret = $la->reduceSumTest($x,$axis=1,null,null,null,null,$mode);
            $this->assertEquals(257,$la->asum($ret));
            // 65535
            $x = $la->alloc([65535],NDArray::float32);
            $la->fill(1,$x);
            $ret = $la->reduceSumTest($x,$axis=1,null,null,null,null,$mode);
            $this->assertEquals(65535,$la->asum($ret));
            // 65536
            $x = $la->alloc([65536],NDArray::float32);
            $la->fill(1,$x);
            $ret = $la->reduceSumTest($x,$axis=1,null,null,null,null,$mode);
            $this->assertEquals(65536,$la->asum($ret));
            // 65537
            $x = $la->alloc([65537],NDArray::float32);
            $la->fill(1,$x);
            $ret = $la->reduceSumTest($x,$axis=1,null,null,null,null,$mode);
            $this->assertEquals(65537,$la->asum($ret));
        }
    }
*/
    /**
    * @dataProvider modeProvider
    */
    public function testReduceSumOpenCL($mode)  // reducesumex
    {
        $mo = $this->newMatrixOperator();
        $la = $this->newLA($mo);
        $x = $la->array([[1,2,3],[4,5,6]]);
        $y = $la->reduceSumTest($x,$axis=0,null,null,null,null,$mode);
        $this->assertEquals([5,7,9],$y->toArray());
        $y = $la->reduceSumTest($x,$axis=1,null,null,null,null,$mode);
        $this->assertEquals([6,15],$y->toArray());

        // ***** CAUTION ******
        // 3d array
        $x = $la->array([[[1,2],[3,4]],[[5,6],[7,8]]]);
        $y = $la->reduceSumTest($x,$axis=0);
        $this->assertEquals([[6,8],[10,12]],$y->toArray());
        $x = $la->array([[[1,2],[3,4]],[[5,6],[7,8]]]);
        $y = $la->reduceSumTest($x,$axis=1);
        $this->assertEquals([[4,6],[12,14]],$y->toArray());

        // with offset
        $x = $la->array([[[9,9,9],[9,9,9]],[[1,2,3],[4,5,6]]]);
        $x = $x[1];
        $this->assertEquals([5,7,9],$la->reduceSumTest($x,$axis=0,null,null,null,null,$mode)->toArray());
        $this->assertEquals([6,15],$la->reduceSumTest($x,$axis=1,null,null,null,null,$mode)->toArray());

        // ceil thread
        $x = $la->alloc([33,33]);
        $la->ones($x);
        $trues = $mo->full([33],33)->toArray();
        $this->assertEquals($trues,$la->reduceSumTest($x,$axis=0,null,null,null,null,$mode)->toArray());
        $this->assertEquals($trues,$la->reduceSumTest($x,$axis=1,null,null,null,null,$mode)->toArray());


        if($mode!=1) {
            // 257
            $x = $la->alloc([1,257],NDArray::float32);
            $la->fill(1,$x);
            $ret = $la->reduceSumTest($x,$axis=1,null,null,null,null,$mode);
            $this->assertEquals(257,$la->asum($ret));
            // 65535
            $x = $la->alloc([1,65535],NDArray::float32);
            $la->fill(1,$x);
            $ret = $la->reduceSumTest($x,$axis=1,null,null,null,null,$mode);
            $this->assertEquals(65535,$la->asum($ret));
            // 65536
            $x = $la->alloc([1,65536],NDArray::float32);
            $la->fill(1,$x);
            $ret = $la->reduceSumTest($x,$axis=1,null,null,null,null,$mode);
            $this->assertEquals(65536,$la->asum($ret));
            // 65537
            $x = $la->alloc([1,65537],NDArray::float32);
            $la->fill(1,$x);
            $ret = $la->reduceSumTest($x,$axis=1,null,null,null,null,$mode);
            $this->assertEquals(65537,$la->asum($ret));
        }
    }

    /**
    * @dataProvider modeProvider
    */
    public function testReduceMaxOpenCL($mode)
    {
        $mo = $this->newMatrixOperator();
        $la = $this->newLA($mo);
        $x = $la->array([[1,2,3],[4,5,6]]);
        $this->assertEquals([4,5,6],$la->reduceMaxTest($x,$axis=0,null,null,null,null,$mode)->toArray());
        $this->assertEquals([3,6],$la->reduceMaxTest($x,$axis=1,null,null,null,null,$mode)->toArray());

        $mo = $this->newMatrixOperator();
        $la = $this->newLA($mo);
        $x = $la->array([[-1,-2,-3],[-4,-5,-6]]);
        $this->assertEquals([-1,-2,-3],$la->reduceMax($x,$axis=0)->toArray());
        $this->assertEquals([-1,-4],$la->reduceMax($x,$axis=1)->toArray());

        // ***** CAUTION ******
        // 3d array
        $x = $la->array([[[1,2],[3,4]],[[5,6],[7,8]]]);
        $y = $la->reduceMaxTest($x,$axis=0,null,null,null,null,$mode);
        $this->assertEquals([[5,6],[7,8]],$y->toArray());
        $x = $la->array([[[1,2],[3,4]],[[5,6],[7,8]]]);
        $y = $la->reduceMaxTest($x,$axis=1,null,null,null,null,$mode);
        $this->assertEquals([[3,4],[7,8]],$y->toArray());

        // with offset
        $x = $la->array([[[9,9,9],[9,9,9]],[[1,2,3],[4,5,6]]]);
        $x = $x[1];
        $this->assertEquals([4,5,6],$la->reduceMaxTest($x,$axis=0,null,null,null,null,$mode)->toArray());
        $this->assertEquals([3,6],$la->reduceMaxTest($x,$axis=1,null,null,null,null,$mode)->toArray());

        if($mode>1) {
            // 257
            $x = $la->alloc([1,257],NDArray::float32);
            $la->fill(1,$x);
            $ret = $la->reduceMaxTest($x,$axis=1,null,null,null,null,$mode);
            $this->assertEquals(1,$la->amax($ret));
            // 65535
            $x = $la->alloc([1,65535],NDArray::float32);
            $la->fill(1,$x);
            $ret = $la->reduceMaxTest($x,$axis=1,null,null,null,null,$mode);
            $this->assertEquals(1,$la->amax($ret));
            // 65536
            $x = $la->alloc([1,65536],NDArray::float32);
            $la->fill(1,$x);
            $ret = $la->reduceMaxTest($x,$axis=1,null,null,null,null,$mode);
            $this->assertEquals(1,$la->amax($ret));
            // 65537
            $x = $la->alloc([1,65537],NDArray::float32);
            $la->fill(1,$x);
            $ret = $la->reduceMaxTest($x,$axis=1,null,null,null,null,$mode);
            $this->assertEquals(1,$la->amax($ret));
        }
    }

    /**
    * @dataProvider modeProvider
    */
    public function testReduceArgMaxOpenCL($mode)
    {
        $mo = $this->newMatrixOperator();
        $la = $this->newLA($mo);
        $x = $la->array([[1,2,3],[4,5,6]]);
        $this->assertEquals([1,1,1],$la->reduceArgMaxTest($x,$axis=0,null,null,null,null,$mode)->toArray());
        $this->assertEquals([2,2],$la->reduceArgMaxTest($x,$axis=1,null,null,null,null,$mode)->toArray());

        // ***** CAUTION ******
        // 3d array as 2d array
        $x = $la->array([[[1,2],[3,4]],[[5,6],[7,8]]]);
        $y = $la->reduceArgMaxTest($x,$axis=0,null,null,null,null,$mode);
        $this->assertEquals([[1,1],[1,1]],$y->toArray());
        $x = $la->array([[[1,2],[3,4]],[[5,6],[7,8]]]);
        $y = $la->reduceArgMaxTest($x,$axis=1,null,null,null,null,$mode);
        $this->assertEquals([[1,1],[1,1]],$y->toArray());

        // with offset
        $x = $la->array([[[9,9,9],[9,9,9]],[[1,2,3],[4,5,6]]]);
        $x = $x[1];
        $this->assertEquals([1,1,1],$la->reduceArgMaxTest($x,$axis=0,null,null,null,null,$mode)->toArray());
        $this->assertEquals([2,2],$la->reduceArgMaxTest($x,$axis=1,null,null,null,null,$mode)->toArray());

    }

    /**
    * @dataProvider modeProvider4mode
    */
    public function testScatterAddOpenCL($mode)
    {
        $mo = $this->newMatrixOperator();
        $la = $this->newLA($mo);
        $la->setOpenCLTestMode($mode);
        // float32
        $x = $la->array([0,2],NDArray::int64);
        $y = $la->array([[1,2,3],[7,8,9]],NDArray::float32);
        $a = $la->array($mo->ones([4,3],NDArray::float32));
        $la->scatterAdd($x,$y,$a);
        #foreach($a->toArray() as $pr) {
        #    echo "\n";
        #    foreach($pr as $value) {
        #        echo $value.",";
        #    }
        #}
        #echo "\n";
        #$this->assertTrue(false);
        $this->assertEquals(
           [[2,3,4],
            [1,1,1],
            [8,9,10],
            [1,1,1]],
            $a->toArray()
        );
        // float64
        if($la->fp64()) {
            $x = $la->array([0,2],NDArray::int64);
            $y = $la->array([[1,2,3],[7,8,9]],NDArray::float64);
            $a = $la->array($mo->ones([4,3],NDArray::float64));
            $la->scatterAdd($x,$y,$a);
            $this->assertEquals(
                [[2,3,4],
                [1,1,1],
                [8,9,10],
                [1,1,1]],
                $a->toArray()
            );
        }
        // int64
        $x = $la->array([0,2],NDArray::int64);
        $y = $la->array([[1,2,3],[7,8,9]],NDArray::int64);
        $a = $la->array($mo->ones([4,3],NDArray::int64));
        $la->scatterAdd($x,$y,$a);
        $this->assertEquals(
           [[2,3,4],
            [1,1,1],
            [8,9,10],
            [1,1,1]],
            $a->toArray()
        );
        // uint8
        $x = $la->array([0,2],NDArray::int64);
        $y = $la->array([[1,2,3],[7,8,9]],NDArray::uint8);
        $a = $la->array($mo->ones([4,3],NDArray::uint8));
        $la->scatterAdd($x,$y,$a);
        $this->assertEquals(
           [[2,3,4],
            [1,1,1],
            [8,9,10],
            [1,1,1]],
            $a->toArray()
        );
    }

    public function testSumCompareSpeed()
    {
        // Comment out when compare speed
        if(!self::$speedtest) {
            $this->markTestSkipped('Speed measurement');
            return;
        }
        $mo = $this->newMatrixOperator();
        $la = $this->newLA($mo);
        if($la->getConfig()=='PhpBlas') {
            $this->assertTrue(true);
            return;
        }

        echo "\n";
        $size = 256;
        echo "\n";
        echo "small size($size)\n";
        echo "mode=1\n";
        $x = $la->alloc([$size],NDArray::float32);
        $la->fill(1.0,$x);
        $sum = $la->sumTest($x,null,null,null,1);
        $start = hrtime(true);
        $sum = $la->sumTest($x,null,null,null,1);
        $end = hrtime(true);
        echo (explode(' ',$la->getConfig()))[0].'='.number_format($end-$start)."\n";
        if(!is_scalar($sum)) {
            $sum = $sum->toArray();
        }
        $this->assertLessThan(1e-3,$size-$sum);
        echo "mode=2\n";
        $x = $la->alloc([$size],NDArray::float32);
        $la->fill(1.0,$x);
        $sum = $la->sumTest($x,null,null,null,2);
        $start = hrtime(true);
        $sum = $la->sumTest($x,null,null,null,2);
        $end = hrtime(true);
        echo (explode(' ',$la->getConfig()))[0].'='.number_format($end-$start)."\n";
        if(!is_scalar($sum)) {
            $sum = $sum->toArray();
        }
        $this->assertLessThan(1e-3,$size-$sum);

        $size = 32768;
        echo "midle size($size)\n";
        echo "mode=2\n";
        $x = $la->alloc([$size],NDArray::float32);
        $la->fill(1.0,$x);
        $sum = $la->sumTest($x,null,null,null,2);
        $start = hrtime(true);
        $sum = $la->sumTest($x,null,null,null,2);
        $end = hrtime(true);
        echo (explode(' ',$la->getConfig()))[0].'='.number_format($end-$start)."\n";
        if(!is_scalar($sum)) {
            $sum = $sum->toArray();
        }
        $this->assertLessThan(1e-3,$size-$sum);
        echo "mode=3\n";
        $x = $la->alloc([$size],NDArray::float32);
        $la->fill(1.0,$x);
        $sum = $la->sumTest($x,null,null,null,3);
        $start = hrtime(true);
        $sum = $la->sumTest($x,null,null,null,3);
        $end = hrtime(true);
        echo (explode(' ',$la->getConfig()))[0].'='.number_format($end-$start)."\n";
        if(!is_scalar($sum)) {
            $sum = $sum->toArray();
        }
        $this->assertLessThan(1e-3,$size-$sum);

        $size = 65536;
        echo "midle size($size)\n";
        echo "mode=2\n";
        $x = $la->alloc([$size],NDArray::float32);
        $la->fill(1.0,$x);
        $sum = $la->sumTest($x,null,null,null,2);
        $start = hrtime(true);
        $sum = $la->sumTest($x,null,null,null,2);
        $end = hrtime(true);
        echo (explode(' ',$la->getConfig()))[0].'='.number_format($end-$start)."\n";
        if(!is_scalar($sum)) {
            $sum = $sum->toArray();
        }
        $this->assertLessThan(1e-3,$size-$sum);
        echo "mode=3\n";
        $x = $la->alloc([$size],NDArray::float32);
        $la->fill(1.0,$x);
        $sum = $la->sumTest($x,null,null,null,3);
        $start = hrtime(true);
        $sum = $la->sumTest($x,null,null,null,3);
        $end = hrtime(true);
        echo (explode(' ',$la->getConfig()))[0].'='.number_format($end-$start)."\n";
        if(!is_scalar($sum)) {
            $sum = $sum->toArray();
        }
        $this->assertLessThan(1e-3,$size-$sum);

        $size = 131072;
        echo "midle size($size)\n";
        echo "mode=2\n";
        $x = $la->alloc([$size],NDArray::float32);
        $la->fill(1.0,$x);
        $sum = $la->sumTest($x,null,null,null,2);
        $start = hrtime(true);
        $sum = $la->sumTest($x,null,null,null,2);
        $end = hrtime(true);
        echo (explode(' ',$la->getConfig()))[0].'='.number_format($end-$start)."\n";
        if(!is_scalar($sum)) {
            $sum = $sum->toArray();
        }
        $this->assertLessThan(1e-3,$size-$sum);
        echo "mode=3\n";
        $x = $la->alloc([$size],NDArray::float32);
        $la->fill(1.0,$x);
        $sum = $la->sumTest($x,null,null,null,3);
        $start = hrtime(true);
        $sum = $la->sumTest($x,null,null,null,3);
        $end = hrtime(true);
        echo (explode(' ',$la->getConfig()))[0].'='.number_format($end-$start)."\n";
        if(!is_scalar($sum)) {
            $sum = $sum->toArray();
        }
        $this->assertLessThan(1e-3,$size-$sum);

        $size = 2000000;
        echo "large size($size)\n";
        echo "mode=2\n";
        $x = $la->alloc([$size],NDArray::float32);
        $la->fill(1.0,$x);
        $sum = $la->sumTest($x,null,null,null,2);
        $start = hrtime(true);
        $sum = $la->sumTest($x,null,null,null,2);
        $end = hrtime(true);
        echo (explode(' ',$la->getConfig()))[0].'='.number_format($end-$start)."\n";
        if(!is_scalar($sum)) {
            $sum = $sum->toArray();
        }
        $this->assertLessThan(1e-3,$size-$sum);
        echo "mode=3\n";
        $x = $la->alloc([$size],NDArray::float32);
        $la->fill(1.0,$x);
        $sum = $la->sumTest($x,null,null,null,3);
        $start = hrtime(true);
        $sum = $la->sumTest($x,null,null,null,3);
        $end = hrtime(true);
        echo (explode(' ',$la->getConfig()))[0].'='.number_format($end-$start)."\n";
        if(!is_scalar($sum)) {
            $sum = $sum->toArray();
        }
        $this->assertLessThan(1e-3,$size-$sum);
    }
/*
    public function testReduceSumCompareSpeed()
    {
        // Comment out when compare speed
        if(!self::$speedtest) {
            $this->markTestSkipped('Speed measurement');
            return;
        }
        $mo = $this->newMatrixOperator();
        $la = $this->newLA($mo);
        if($la->getConfig()=='PhpBlas') {
            $this->assertTrue(true);
            return;
        }
        echo "\n";
        $colsize = 131072;#0;
        $rowsize = 64;
        echo "==large cols($rowsize,$colsize)==\n";
        fwrite(STDERR,"*");
        echo "mode=2\n";
        $x = $la->alloc([$rowsize,$colsize],NDArray::float32);
        $la->fill(1.0,$x);
        $sum = $la->reduceSumTest($x,$axis=1,null,null,null,null,2);
        $start = hrtime(true);
        $sum = $la->reduceSumTest($x,$axis=1,null,null,null,null,2);
        $end = hrtime(true);
        $this->assertEquals($x->size(),$la->asum($sum));
        echo (explode(' ',$la->getConfig()))[0].'='.number_format($end-$start)."\n";

        fwrite(STDERR,"*");
        echo "mode=3\n";
        $x = $la->alloc([$rowsize,$colsize],NDArray::float32);
        $la->fill(1.0,$x);
        $sum = $la->reduceSumTest($x,$axis=1,null,null,null,null,3);
        $start = hrtime(true);
        $sum = $la->reduceSumTest($x,$axis=1,null,null,null,null,3);
        $end = hrtime(true);
        $this->assertEquals($x->size(),$la->asum($sum));
        echo (explode(' ',$la->getConfig()))[0].'='.number_format($end-$start)."\n";

        $colsize = 64;
        $rowsize = 1000000;
        echo "==large rows($rowsize,$colsize)==\n";
        fwrite(STDERR,"*");
        echo "mode=0\n";
        $x = $la->alloc([$rowsize,$colsize],NDArray::float32);
        $la->fill(1.0,$x);
        $sum = $la->reduceSumTest($x,$axis=1,null,null,null,null,0);
        $start = hrtime(true);
        $sum = $la->reduceSumTest($x,$axis=1,null,null,null,null,0);
        $end = hrtime(true);
        $this->assertEquals($x->size(),$la->asum($sum));
        echo (explode(' ',$la->getConfig()))[0].'='.number_format($end-$start)."\n";

        fwrite(STDERR,"*");
        echo "mode=1\n";
        $x = $la->alloc([$rowsize,$colsize],NDArray::float32);
        $la->fill(1.0,$x);
        $sum = $la->reduceSumTest($x,$axis=1,null,null,null,null,1);
        $start = hrtime(true);
        $sum = $la->reduceSumTest($x,$axis=1,null,null,null,null,1);
        $end = hrtime(true);
        $this->assertEquals($x->size(),$la->asum($sum));
        echo (explode(' ',$la->getConfig()))[0].'='.number_format($end-$start)."\n";

        fwrite(STDERR,"*");
        echo "mode=2\n";
        $x = $la->alloc([$rowsize,$colsize],NDArray::float32);
        $la->fill(1.0,$x);
        $sum = $la->reduceSumTest($x,$axis=1,null,null,null,null,2);
        $start = hrtime(true);
        $sum = $la->reduceSumTest($x,$axis=1,null,null,null,null,2);
        $end = hrtime(true);
        $this->assertEquals($x->size(),$la->asum($sum));
        echo (explode(' ',$la->getConfig()))[0].'='.number_format($end-$start)."\n";

        $colsize = 256;
        $rowsize = 12500;#0;
        echo "==midle rows,cols($rowsize,$colsize)==\n";
        fwrite(STDERR,"*");
        echo "mode=0\n";
        $x = $la->alloc([$rowsize,$colsize],NDArray::float32);
        $la->fill(1.0,$x);
        $sum = $la->reduceSumTest($x,$axis=1,null,null,null,null,0);
        $start = hrtime(true);
        $sum = $la->reduceSumTest($x,$axis=1,null,null,null,null,0);
        $end = hrtime(true);
        $this->assertEquals($x->size(),$la->asum($sum));
        echo (explode(' ',$la->getConfig()))[0].'='.number_format($end-$start)."\n";

        fwrite(STDERR,"*");
        echo "mode=1\n";
        $x = $la->alloc([$rowsize,$colsize],NDArray::float32);
        $la->fill(1.0,$x);
        $sum = $la->reduceSumTest($x,$axis=1,null,null,null,null,2);
        $start = hrtime(true);
        $sum = $la->reduceSumTest($x,$axis=1,null,null,null,null,2);
        $end = hrtime(true);
        $this->assertEquals($x->size(),$la->asum($sum));
        echo (explode(' ',$la->getConfig()))[0].'='.number_format($end-$start)."\n";

        fwrite(STDERR,"*");
        echo "mode=2\n";
        $x = $la->alloc([$rowsize,$colsize],NDArray::float32);
        $la->fill(1.0,$x);
        $sum = $la->reduceSumTest($x,$axis=1,null,null,null,null,2);
        $start = hrtime(true);
        $sum = $la->reduceSumTest($x,$axis=1,null,null,null,null,2);
        $end = hrtime(true);
        $this->assertEquals($x->size(),$la->asum($sum));
        echo (explode(' ',$la->getConfig()))[0].'='.number_format($end-$start)."\n";

        $colsize = 257;
        $rowsize = 12500;#0;
        echo "==midle rows,cols($rowsize,$colsize)==\n";
        fwrite(STDERR,"*");
        echo "mode=0\n";
        $x = $la->alloc([$rowsize,$colsize],NDArray::float32);
        $la->fill(1.0,$x);
        $sum = $la->reduceSumTest($x,$axis=1,null,null,null,null,0);
        $start = hrtime(true);
        $sum = $la->reduceSumTest($x,$axis=1,null,null,null,null,0);
        $end = hrtime(true);
        $this->assertEquals($x->size(),$la->asum($sum));
        echo (explode(' ',$la->getConfig()))[0].'='.number_format($end-$start)."\n";

        fwrite(STDERR,"*");
        echo "mode=2\n";
        $x = $la->alloc([$rowsize,$colsize],NDArray::float32);
        $la->fill(1.0,$x);
        $sum = $la->reduceSumTest($x,$axis=1,null,null,null,null,2);
        $start = hrtime(true);
        $sum = $la->reduceSumTest($x,$axis=1,null,null,null,null,2);
        $end = hrtime(true);
        $this->assertEquals($x->size(),$la->asum($sum));
        echo (explode(' ',$la->getConfig()))[0].'='.number_format($end-$start)."\n";

        $colsize = 65535;//256*256-1
        $rowsize = 512;#0;
        echo "==midle rows,cols($rowsize,$colsize)==\n";
        $trues = $mo->full([$rowsize],$colsize);
        fwrite(STDERR,"*");
        echo "mode=0\n";
        $x = $la->alloc([$rowsize,$colsize],NDArray::float32);
        $la->fill(1.0,$x);
        $sum = $la->reduceSumTest($x,$axis=1,null,null,null,null,0);
        $start = hrtime(true);
        $sum = $la->reduceSumTest($x,$axis=1,null,null,null,null,0);
        $end = hrtime(true);
        $this->assertEquals($trues->toArray(),$sum->toArray());
        echo (explode(' ',$la->getConfig()))[0].'='.number_format($end-$start)."\n";

        fwrite(STDERR,"*");
        echo "mode=2\n";
        $x = $la->alloc([$rowsize,$colsize],NDArray::float32);
        $la->fill(1.0,$x);
        $sum = $la->reduceSumTest($x,$axis=1,null,null,null,null,2);
        $start = hrtime(true);
        $sum = $la->reduceSumTest($x,$axis=1,null,null,null,null,2);
        $end = hrtime(true);
        $this->assertEquals($trues->toArray(),$sum->toArray());
        echo (explode(' ',$la->getConfig()))[0].'='.number_format($end-$start)."\n";

        $colsize = 65536/2;//256*256-1
        $rowsize = 512;#0;
        echo "==midle rows,cols($rowsize,$colsize)==\n";
        $trues = $mo->full([$rowsize],$colsize);
        fwrite(STDERR,"*");
        echo "mode=0\n";
        $x = $la->alloc([$rowsize,$colsize],NDArray::float32);
        $la->fill(1.0,$x);
        $sum = $la->reduceSumTest($x,$axis=1,null,null,null,null,0);
        $start = hrtime(true);
        $sum = $la->reduceSumTest($x,$axis=1,null,null,null,null,0);
        $end = hrtime(true);
        $this->assertEquals($trues->toArray(),$sum->toArray());
        echo (explode(' ',$la->getConfig()))[0].'='.number_format($end-$start)."\n";

        fwrite(STDERR,"*");
        echo "mode=2\n";
        $x = $la->alloc([$rowsize,$colsize],NDArray::float32);
        $la->fill(1.0,$x);
        $sum = $la->reduceSumTest($x,$axis=1,null,null,null,null,2);
        $start = hrtime(true);
        $sum = $la->reduceSumTest($x,$axis=1,null,null,null,null,2);
        $end = hrtime(true);
        $this->assertEquals($trues->toArray(),$sum->toArray());
        echo (explode(' ',$la->getConfig()))[0].'='.number_format($end-$start)."\n";

        $colsize = 65536/2-1;//256*256-1
        $rowsize = 800;#0;
        echo "==midle rows,cols($rowsize,$colsize)==\n";
        $trues = $mo->full([$rowsize],$colsize);
        fwrite(STDERR,"*");
        echo "mode=0\n";
        $x = $la->alloc([$rowsize,$colsize],NDArray::float32);
        $la->fill(1.0,$x);
        $sum = $la->reduceSumTest($x,$axis=1,null,null,null,null,0);
        $start = hrtime(true);
        $sum = $la->reduceSumTest($x,$axis=1,null,null,null,null,0);
        $end = hrtime(true);
        $this->assertEquals($trues->toArray(),$sum->toArray());
        echo (explode(' ',$la->getConfig()))[0].'='.number_format($end-$start)."\n";

        fwrite(STDERR,"*");
        echo "mode=2\n";
        $x = $la->alloc([$rowsize,$colsize],NDArray::float32);
        $la->fill(1.0,$x);
        $sum = $la->reduceSumTest($x,$axis=1,null,null,null,null,2);
        $start = hrtime(true);
        $sum = $la->reduceSumTest($x,$axis=1,null,null,null,null,2);
        $end = hrtime(true);
        $this->assertEquals($trues->toArray(),$sum->toArray());
        echo (explode(' ',$la->getConfig()))[0].'='.number_format($end-$start)."\n";

        $colsize = 65536/2;//256*256-1
        $rowsize = 800;#0;
        echo "==midle rows,cols($rowsize,$colsize)==\n";
        $trues = $mo->full([$rowsize],$colsize);
        fwrite(STDERR,"*");
        echo "mode=0\n";
        $x = $la->alloc([$rowsize,$colsize],NDArray::float32);
        $la->fill(1.0,$x);
        $sum = $la->reduceSumTest($x,$axis=1,null,null,null,null,0);
        $start = hrtime(true);
        $sum = $la->reduceSumTest($x,$axis=1,null,null,null,null,0);
        $end = hrtime(true);
        $this->assertEquals($trues->toArray(),$sum->toArray());
        echo (explode(' ',$la->getConfig()))[0].'='.number_format($end-$start)."\n";

        fwrite(STDERR,"*");
        echo "mode=2\n";
        $x = $la->alloc([$rowsize,$colsize],NDArray::float32);
        $la->fill(1.0,$x);
        $sum = $la->reduceSumTest($x,$axis=1,null,null,null,null,2);
        $start = hrtime(true);
        $sum = $la->reduceSumTest($x,$axis=1,null,null,null,null,2);
        $end = hrtime(true);
        $this->assertEquals($trues->toArray(),$sum->toArray());
        echo (explode(' ',$la->getConfig()))[0].'='.number_format($end-$start)."\n";

        $colsize = 65536/2-1;//256*256-1
        $rowsize = 1023;#0;
        echo "==midle rows,cols($rowsize,$colsize)==\n";
        $trues = $mo->full([$rowsize],$colsize);
        fwrite(STDERR,"*");
        echo "mode=0\n";
        $x = $la->alloc([$rowsize,$colsize],NDArray::float32);
        $la->fill(1.0,$x);
        $sum = $la->reduceSumTest($x,$axis=1,null,null,null,null,0);
        $start = hrtime(true);
        $sum = $la->reduceSumTest($x,$axis=1,null,null,null,null,0);
        $end = hrtime(true);
        $this->assertEquals($trues->toArray(),$sum->toArray());
        echo (explode(' ',$la->getConfig()))[0].'='.number_format($end-$start)."\n";

        fwrite(STDERR,"*");
        echo "mode=2\n";
        $x = $la->alloc([$rowsize,$colsize],NDArray::float32);
        $la->fill(1.0,$x);
        $sum = $la->reduceSumTest($x,$axis=1,null,null,null,null,2);
        $start = hrtime(true);
        $sum = $la->reduceSumTest($x,$axis=1,null,null,null,null,2);
        $end = hrtime(true);
        $this->assertEquals($trues->toArray(),$sum->toArray());
        echo (explode(' ',$la->getConfig()))[0].'='.number_format($end-$start)."\n";

        $colsize = 65536/2;//256*256
        $rowsize = 1023;#0;
        echo "==midle rows,cols($rowsize,$colsize)==\n";
        $trues = $mo->full([$rowsize],$colsize);
        fwrite(STDERR,"*");
        echo "mode=0\n";
        $x = $la->alloc([$rowsize,$colsize],NDArray::float32);
        $la->fill(1.0,$x);
        $sum = $la->reduceSumTest($x,$axis=1,null,null,null,null,0);
        $start = hrtime(true);
        $sum = $la->reduceSumTest($x,$axis=1,null,null,null,null,0);
        $end = hrtime(true);
        $this->assertEquals($trues->toArray(),$sum->toArray());
        echo (explode(' ',$la->getConfig()))[0].'='.number_format($end-$start)."\n";

        fwrite(STDERR,"*");
        echo "mode=2\n";
        $x = $la->alloc([$rowsize,$colsize],NDArray::float32);
        $la->fill(1.0,$x);
        $sum = $la->reduceSumTest($x,$axis=1,null,null,null,null,2);
        $start = hrtime(true);
        $sum = $la->reduceSumTest($x,$axis=1,null,null,null,null,2);
        $end = hrtime(true);
        $this->assertEquals($trues->toArray(),$sum->toArray());
        echo (explode(' ',$la->getConfig()))[0].'='.number_format($end-$start)."\n";

        $colsize = (int)(32768/4-1);//256*256
        $rowsize = 2048*2;#0;
        echo "==midle rows,cols($rowsize,$colsize)==\n";
        $trues = $mo->full([$rowsize],$colsize);
        fwrite(STDERR,"*");
        echo "mode=0\n";
        $x = $la->alloc([$rowsize,$colsize],NDArray::float32);
        $la->fill(1.0,$x);
        $sum = $la->reduceSumTest($x,$axis=1,null,null,null,null,0);
        $start = hrtime(true);
        $sum = $la->reduceSumTest($x,$axis=1,null,null,null,null,0);
        $end = hrtime(true);
        $this->assertEquals($trues->toArray(),$sum->toArray());
        echo (explode(' ',$la->getConfig()))[0].'='.number_format($end-$start)."\n";

        fwrite(STDERR,"*");
        echo "mode=2\n";
        $x = $la->alloc([$rowsize,$colsize],NDArray::float32);
        $la->fill(1.0,$x);
        $sum = $la->reduceSumTest($x,$axis=1,null,null,null,null,2);
        $start = hrtime(true);
        $sum = $la->reduceSumTest($x,$axis=1,null,null,null,null,2);
        $end = hrtime(true);
        $this->assertEquals($trues->toArray(),$sum->toArray());
        echo (explode(' ',$la->getConfig()))[0].'='.number_format($end-$start)."\n";

        $colsize = (int)(32768/4);//256*256
        $rowsize = 2048*2;#0;
        echo "==midle rows,cols($rowsize,$colsize)==\n";
        $trues = $mo->full([$rowsize],$colsize);
        fwrite(STDERR,"*");
        echo "mode=0\n";
        $x = $la->alloc([$rowsize,$colsize],NDArray::float32);
        $la->fill(1.0,$x);
        $sum = $la->reduceSumTest($x,$axis=1,null,null,null,null,0);
        $start = hrtime(true);
        $sum = $la->reduceSumTest($x,$axis=1,null,null,null,null,0);
        $end = hrtime(true);
        $this->assertEquals($trues->toArray(),$sum->toArray());
        echo (explode(' ',$la->getConfig()))[0].'='.number_format($end-$start)."\n";

        fwrite(STDERR,"*");
        echo "mode=2\n";
        $x = $la->alloc([$rowsize,$colsize],NDArray::float32);
        $la->fill(1.0,$x);
        $sum = $la->reduceSumTest($x,$axis=1,null,null,null,null,2);
        $start = hrtime(true);
        $sum = $la->reduceSumTest($x,$axis=1,null,null,null,null,2);
        $end = hrtime(true);
        $this->assertEquals($trues->toArray(),$sum->toArray());
        echo (explode(' ',$la->getConfig()))[0].'='.number_format($end-$start)."\n";

        $colsize = 65536;
        $rowsize = 100;#0;
        echo "==midle rows,cols($rowsize,$colsize)==\n";
        $trues = $mo->full([$rowsize],$colsize);
        fwrite(STDERR,"*");
        echo "mode=2\n";
        $x = $la->alloc([$rowsize,$colsize],NDArray::float32);
        $la->fill(1.0,$x);
        $sum = $la->reduceSumTest($x,$axis=1,null,null,null,null,2);
        $start = hrtime(true);
        $sum = $la->reduceSumTest($x,$axis=1,null,null,null,null,2);
        $end = hrtime(true);
        $this->assertEquals($trues->toArray(),$sum->toArray());
        echo (explode(' ',$la->getConfig()))[0].'='.number_format($end-$start)."\n";

        fwrite(STDERR,"*");
        echo "mode=3\n";
        $x = $la->alloc([$rowsize,$colsize],NDArray::float32);
        $la->fill(1.0,$x);
        $sum = $la->reduceSumTest($x,$axis=1,null,null,null,null,3);
        $start = hrtime(true);
        $sum = $la->reduceSumTest($x,$axis=1,null,null,null,null,3);
        $end = hrtime(true);
        $this->assertEquals($trues->toArray(),$sum->toArray());
        echo (explode(' ',$la->getConfig()))[0].'='.number_format($end-$start)."\n";

    }
*/
    public function testReduceSumCompareSpeed() // reducesumex
    {
        // Comment out when compare speed
        if(!self::$speedtest) {
            $this->markTestSkipped('Speed measurement');
            return;
        }
        $mo = $this->newMatrixOperator();
        $la = $this->newLA($mo);
        if($la->getConfig()=='PhpBlas') {
            $this->assertTrue(true);
            return;
        }
        echo "\n";
        $depth = 8;
        $rowsize = 800000;
        $colsize = 8;
        echo "==large rows($depth,$rowsize,$colsize)==\n";
        fwrite(STDERR,"*");
        $mode = 0;
        echo "mode=$mode\n";
        $x = $la->alloc([$depth,$rowsize,$colsize],NDArray::float32);
        $la->fill(1.0,$x);
        $sum = $la->reduceSumTest($x,$axis=1,null,null,null,null,$mode);
        $start = hrtime(true);
        $sum = $la->reduceSumTest($x,$axis=1,null,null,null,null,$mode);
        $end = hrtime(true);
        $this->assertEquals($x->size(),$la->asum($sum));
        echo (explode(' ',$la->getConfig()))[0].'='.number_format($end-$start)."\n";

        fwrite(STDERR,"*");
        $mode = 2;
        echo "mode=$mode\n";
        $x = $la->alloc([$depth,$rowsize,$colsize],NDArray::float32);
        $la->fill(1.0,$x);
        $sum = $la->reduceSumTest($x,$axis=1,null,null,null,null,$mode);
        $start = hrtime(true);
        $sum = $la->reduceSumTest($x,$axis=1,null,null,null,null,$mode);
        $end = hrtime(true);
        $this->assertEquals($x->size(),$la->asum($sum));
        echo (explode(' ',$la->getConfig()))[0].'='.number_format($end-$start)."\n";

        fwrite(STDERR,"*");
        $mode = 3;
        echo "mode=$mode\n";
        $x = $la->alloc([$depth,$rowsize,$colsize],NDArray::float32);
        $la->fill(1.0,$x);
        $sum = $la->reduceSumTest($x,$axis=1,null,null,null,null,$mode);
        $start = hrtime(true);
        $sum = $la->reduceSumTest($x,$axis=1,null,null,null,null,$mode);
        $end = hrtime(true);
        $this->assertEquals($x->size(),$la->asum($sum));
        echo (explode(' ',$la->getConfig()))[0].'='.number_format($end-$start)."\n";

        $depth = 800000;
        $rowsize = 8;
        $colsize = 8;
        echo "==large depth($depth,$rowsize,$colsize)==\n";
        fwrite(STDERR,"*");
        $mode = 0;
        echo "mode=$mode\n";
        $x = $la->alloc([$depth,$rowsize,$colsize],NDArray::float32);
        $la->fill(1.0,$x);
        $sum = $la->reduceSumTest($x,$axis=1,null,null,null,null,$mode);
        $start = hrtime(true);
        $sum = $la->reduceSumTest($x,$axis=1,null,null,null,null,$mode);
        $end = hrtime(true);
        $this->assertEquals($x->size(),$la->asum($sum));
        echo (explode(' ',$la->getConfig()))[0].'='.number_format($end-$start)."\n";

        fwrite(STDERR,"*");
        $mode = 1;
        echo "mode=$mode\n";
        $x = $la->alloc([$depth,$rowsize,$colsize],NDArray::float32);
        $la->fill(1.0,$x);
        $sum = $la->reduceSumTest($x,$axis=1,null,null,null,null,$mode);
        $start = hrtime(true);
        $sum = $la->reduceSumTest($x,$axis=1,null,null,null,null,$mode);
        $end = hrtime(true);
        $this->assertEquals($x->size(),$la->asum($sum));
        echo (explode(' ',$la->getConfig()))[0].'='.number_format($end-$start)."\n";

        fwrite(STDERR,"*");
        $mode = 2;
        echo "mode=$mode\n";
        $x = $la->alloc([$depth,$rowsize,$colsize],NDArray::float32);
        $la->fill(1.0,$x);
        $sum = $la->reduceSumTest($x,$axis=1,null,null,null,null,$mode);
        $start = hrtime(true);
        $sum = $la->reduceSumTest($x,$axis=1,null,null,null,null,$mode);
        $end = hrtime(true);
        $this->assertEquals($x->size(),$la->asum($sum));
        echo (explode(' ',$la->getConfig()))[0].'='.number_format($end-$start)."\n";

        $depth = 8;
        $rowsize = 8;#0;
        $colsize = 800000;
        echo "==large cols($depth,$rowsize,$colsize)==\n";
        $trues = $mo->full([$rowsize,$depth],$colsize);
        fwrite(STDERR,"*");
        $mode = 0;
        echo "mode=$mode\n";
        $x = $la->alloc([$depth,$rowsize,$colsize],NDArray::float32);
        $la->fill(1.0,$x);
        $sum = $la->reduceSumTest($x,$axis=1,null,null,null,null,$mode);
        $start = hrtime(true);
        $sum = $la->reduceSumTest($x,$axis=1,null,null,null,null,$mode);
        $end = hrtime(true);
        $this->assertEquals($x->size(),$la->asum($sum));
        echo (explode(' ',$la->getConfig()))[0].'='.number_format($end-$start)."\n";

        fwrite(STDERR,"*");
        $mode = 1;
        echo "mode=$mode\n";
        $x = $la->alloc([$depth,$rowsize,$colsize],NDArray::float32);
        $la->fill(1.0,$x);
        $sum = $la->reduceSumTest($x,$axis=1,null,null,null,null,$mode);
        $start = hrtime(true);
        $sum = $la->reduceSumTest($x,$axis=1,null,null,null,null,$mode);
        $end = hrtime(true);
        $this->assertEquals($x->size(),$la->asum($sum));
        echo (explode(' ',$la->getConfig()))[0].'='.number_format($end-$start)."\n";

        fwrite(STDERR,"*");
        $mode = 2;
        echo "mode=$mode\n";
        $x = $la->alloc([$depth,$rowsize,$colsize],NDArray::float32);
        $la->fill(1.0,$x);
        $sum = $la->reduceSumTest($x,$axis=1,null,null,null,null,$mode);
        $start = hrtime(true);
        $sum = $la->reduceSumTest($x,$axis=1,null,null,null,null,$mode);
        $end = hrtime(true);
        $this->assertEquals($x->size(),$la->asum($sum));
        echo (explode(' ',$la->getConfig()))[0].'='.number_format($end-$start)."\n";

    }

    public function testReduceMaxCompareSpeed()
    {
        // Comment out when compare speed
        if(!self::$speedtest) {
            $this->markTestSkipped('Speed measurement');
            return;
        }
        $mo = $this->newMatrixOperator();
        $la = $this->newLA($mo);
        if($la->getConfig()=='PhpBlas') {
            $this->assertTrue(true);
            return;
        }
        echo "==large cols==\n";
        echo "mode=2\n";
        $colsize = 131072;#0;
        $rowsize = 64;
        $x = $la->alloc([$rowsize,$colsize],NDArray::float32);
        $la->fill(1.0,$x);
        $max = $la->reduceMaxTest($x,$axis=1,null,null,null,null,2);
        $start = hrtime(true);
        $max = $la->reduceMaxTest($x,$axis=1,null,null,null,null,2);
        $end = hrtime(true);
        $this->assertEquals($max->size(),$la->asum($max));
        echo (explode(' ',$la->getConfig()))[0].'='.number_format($end-$start)."\n";

        echo "mode=3\n";
        $colsize = 131072;#0;
        $rowsize = 64;
        $x = $la->alloc([$rowsize,$colsize],NDArray::float32);
        $la->fill(1.0,$x);
        $max = $la->reduceMaxTest($x,$axis=1,null,null,null,null,3);
        $start = hrtime(true);
        $max = $la->reduceMaxTest($x,$axis=1,null,null,null,null,3);
        $end = hrtime(true);
        $this->assertEquals($max->size(),$la->asum($max));
        echo (explode(' ',$la->getConfig()))[0].'='.number_format($end-$start)."\n";

        echo "==large rows==\n";
        echo "mode=0\n";
        $colsize = 64;
        $rowsize = 1000000;
        $x = $la->alloc([$rowsize,$colsize],NDArray::float32);
        $la->fill(1.0,$x);
        $max = $la->reduceMaxTest($x,$axis=1,null,null,null,null,0);
        $start = hrtime(true);
        $max = $la->reduceMaxTest($x,$axis=1,null,null,null,null,0);
        $end = hrtime(true);
        $this->assertEquals($max->size(),$la->asum($max));
        echo (explode(' ',$la->getConfig()))[0].'='.number_format($end-$start)."\n";

        echo "mode=1\n";
        $colsize = 64;
        $rowsize = 1000000;
        $x = $la->alloc([$rowsize,$colsize],NDArray::float32);
        $la->fill(1.0,$x);
        $max = $la->reduceMaxTest($x,$axis=1,null,null,null,null,1);
        $start = hrtime(true);
        $max = $la->reduceMaxTest($x,$axis=1,null,null,null,null,1);
        $end = hrtime(true);
        $this->assertEquals($max->size(),$la->asum($max));
        echo (explode(' ',$la->getConfig()))[0].'='.number_format($end-$start)."\n";

        echo "mode=2\n";
        $colsize = 64;
        $rowsize = 1000000;
        $x = $la->alloc([$rowsize,$colsize],NDArray::float32);
        $la->fill(1.0,$x);
        $max = $la->reduceMaxTest($x,$axis=1,null,null,null,null,2);
        $start = hrtime(true);
        $max = $la->reduceMaxTest($x,$axis=1,null,null,null,null,2);
        $end = hrtime(true);
        $this->assertEquals($max->size(),$la->asum($max));
        echo (explode(' ',$la->getConfig()))[0].'='.number_format($end-$start)."\n";

        echo "==midle rows,cols==\n";
        echo "mode=2\n";
        $colsize = 4096;
        $rowsize = 12500;#0;
        $x = $la->alloc([$rowsize,$colsize],NDArray::float32);
        $la->fill(1.0,$x);
        $max = $la->reduceMaxTest($x,$axis=1,null,null,null,null,2);
        $start = hrtime(true);
        $max = $la->reduceMaxTest($x,$axis=1,null,null,null,null,2);
        $end = hrtime(true);
        $this->assertEquals($max->size(),$la->asum($max));
        echo (explode(' ',$la->getConfig()))[0].'='.number_format($end-$start)."\n";

        echo "mode=3\n";
        $colsize = 4096;
        $rowsize = 12500;#0;
        $x = $la->alloc([$rowsize,$colsize],NDArray::float32);
        $la->fill(1.0,$x);
        $max = $la->reduceMaxTest($x,$axis=1,null,null,null,null,3);
        $start = hrtime(true);
        $max = $la->reduceMaxTest($x,$axis=1,null,null,null,null,3);
        $end = hrtime(true);
        $this->assertEquals($max->size(),$la->asum($max));
        echo (explode(' ',$la->getConfig()))[0].'='.number_format($end-$start)."\n";
    }

    public function testSoftmaxCompareSpeed()
    {
        if(!self::$speedtest) {
            $this->markTestSkipped('Speed measurement');
            return;
        }
        $mo = $this->newMatrixOperator();
        $la = $this->newLA($mo);
        if($la->getConfig()=='PhpBlas') {
            $this->assertTrue(true);
            return;
        }

        // large size
        $colsize = 100000;
        $rowsize = 64;
        echo "($rowsize,$colsize)\n";
        $x = $la->alloc([$rowsize,$colsize],NDArray::float32);
        fwrite(STDERR,"fill-start0\n");
        $la->fill(1.0,$x);
        fwrite(STDERR,"fill-end0\n");
        fwrite(STDERR,"pre-start0\n");
        $r = $la->softmaxTest($x,null,null,0);
        fwrite(STDERR,"pre-end0\n");
        $start = hrtime(true);
        $r = $la->softmaxTest($x,null,null,0);
        $end = hrtime(true);
        echo (explode(' ',$la->getConfig()))[0].'0='.number_format($end-$start)."\n";

        $x = $la->alloc([$rowsize,$colsize],NDArray::float32);
        fwrite(STDERR,"fill-start2\n");
        $la->fill(1.0,$x);
        fwrite(STDERR,"fill-end2\n");
        fwrite(STDERR,"pre-start2\n");
        $r = $la->softmaxTest($x,null,null,2);
        fwrite(STDERR,"pre-end2\n");
        $start = hrtime(true);
        $r = $la->softmaxTest($x,null,null,2);
        $end = hrtime(true);
        echo (explode(' ',$la->getConfig()))[0].'2='.number_format($end-$start)."\n";

        // col256
        $colsize = 256;
        $rowsize = 100000;
        echo "($rowsize,$colsize)\n";
        $x = $la->alloc([$rowsize,$colsize],NDArray::float32);
        fwrite(STDERR,"fill-start0\n");
        $la->fill(1.0,$x);
        fwrite(STDERR,"fill-end0\n");
        fwrite(STDERR,"pre-start0\n");
        $r = $la->softmaxTest($x,null,null,0);
        fwrite(STDERR,"pre-end0\n");
        $start = hrtime(true);
        $r = $la->softmaxTest($x,null,null,0);
        $end = hrtime(true);
        echo (explode(' ',$la->getConfig()))[0].'0='.number_format($end-$start)."\n";

        $x = $la->alloc([$rowsize,$colsize],NDArray::float32);
        fwrite(STDERR,"fill-start1\n");
        $la->fill(1.0,$x);
        fwrite(STDERR,"fill-end1\n");
        fwrite(STDERR,"pre-start1\n");
        $r = $la->softmaxTest($x,null,null,1);
        fwrite(STDERR,"pre-end1\n");
        $start = hrtime(true);
        $r = $la->softmaxTest($x,null,null,1);
        $end = hrtime(true);
        echo (explode(' ',$la->getConfig()))[0].'1='.number_format($end-$start)."\n";

        // large low
        $colsize = 64;
        $rowsize = 300000;
        echo "($rowsize,$colsize)\n";
        $x = $la->alloc([$rowsize,$colsize],NDArray::float32);
        fwrite(STDERR,"fill-start0\n");
        $la->fill(1.0,$x);
        fwrite(STDERR,"fill-end0\n");
        fwrite(STDERR,"pre-start0\n");
        $r = $la->softmaxTest($x,null,null,0);
        fwrite(STDERR,"pre-end0\n");
        $start = hrtime(true);
        $r = $la->softmaxTest($x,null,null,0);
        $end = hrtime(true);
        echo (explode(' ',$la->getConfig()))[0].'0='.number_format($end-$start)."\n";

        $x = $la->alloc([$rowsize,$colsize],NDArray::float32);
        fwrite(STDERR,"fill-start1\n");
        $la->fill(1.0,$x);
        fwrite(STDERR,"fill-end1\n");
        fwrite(STDERR,"pre-start1\n");
        $r = $la->softmaxTest($x,null,null,1);
        fwrite(STDERR,"pre-end1\n");
        $start = hrtime(true);
        $r = $la->softmaxTest($x,null,null,1);
        $end = hrtime(true);
        echo (explode(' ',$la->getConfig()))[0].'1='.number_format($end-$start)."\n";

        // col256
        $colsize = 257;
        $rowsize = 100000;
        echo "($rowsize,$colsize)\n";
        $x = $la->alloc([$rowsize,$colsize],NDArray::float32);
        fwrite(STDERR,"fill-start0\n");
        $la->fill(1.0,$x);
        fwrite(STDERR,"fill-end0\n");
        fwrite(STDERR,"pre-start0\n");
        $r = $la->softmaxTest($x,null,null,0);
        fwrite(STDERR,"pre-end0\n");
        $start = hrtime(true);
        $r = $la->softmaxTest($x,null,null,0);
        $end = hrtime(true);
        echo (explode(' ',$la->getConfig()))[0].'0='.number_format($end-$start)."\n";

        $x = $la->alloc([$rowsize,$colsize],NDArray::float32);
        fwrite(STDERR,"fill-start1\n");
        $la->fill(1.0,$x);
        fwrite(STDERR,"fill-end1\n");
        fwrite(STDERR,"pre-start1\n");
        $r = $la->softmaxTest($x,null,null,2);
        fwrite(STDERR,"pre-end1\n");
        $start = hrtime(true);
        $r = $la->softmaxTest($x,null,null,2);
        $end = hrtime(true);
        echo (explode(' ',$la->getConfig()))[0].'2='.number_format($end-$start)."\n";

        // col256
        $colsize = 512;
        $rowsize = 100000;
        echo "($rowsize,$colsize)\n";
        $x = $la->alloc([$rowsize,$colsize],NDArray::float32);
        fwrite(STDERR,"fill-start0\n");
        $la->fill(1.0,$x);
        fwrite(STDERR,"fill-end0\n");
        fwrite(STDERR,"pre-start0\n");
        $r = $la->softmaxTest($x,null,null,0);
        fwrite(STDERR,"pre-end0\n");
        $start = hrtime(true);
        $r = $la->softmaxTest($x,null,null,0);
        $end = hrtime(true);
        echo (explode(' ',$la->getConfig()))[0].'0='.number_format($end-$start)."\n";

        $x = $la->alloc([$rowsize,$colsize],NDArray::float32);
        fwrite(STDERR,"fill-start1\n");
        $la->fill(1.0,$x);
        fwrite(STDERR,"fill-end1\n");
        fwrite(STDERR,"pre-start1\n");
        $r = $la->softmaxTest($x,null,null,2);
        fwrite(STDERR,"pre-end1\n");
        $start = hrtime(true);
        $r = $la->softmaxTest($x,null,null,2);
        $end = hrtime(true);
        echo (explode(' ',$la->getConfig()))[0].'2='.number_format($end-$start)."\n";
    }

    public function testScatterAddCompareSpeed()
    {
        // Comment out when compare speed
        if(!self::$speedtest) {
            $this->markTestSkipped('Speed measurement');
            return;
        }
        $mo = $this->newMatrixOperator();
        $la = $this->newLA($mo);
        if($la->getConfig()=='PhpBlas') {
            $this->assertTrue(true);
            return;
        }
        echo "\n";
        $rows = 8;
        $cols = 8;
        $numClass = 800000;#65536;
        echo "==large cols($rows,$cols,$numClass)==\n";
        fwrite(STDERR,"*");
        $mode = 0;
        echo "mode=$mode\n";
        $x = $la->alloc([$rows],NDArray::int32);
        $la->fill(1,$x);
        $y = $la->alloc([$rows,$cols],NDArray::float32);
        $la->fill(1.0,$y);
        $a = $la->alloc([$numClass,$cols],NDArray::float32);
        $la->fill(0.0,$a);
        $la->scatterAdd($x,$y,$a);
        $start = hrtime(true);
        $la->scatterAdd($x,$y,$a);
        $end = hrtime(true);
        echo (explode(' ',$la->getConfig()))[0].'='.number_format($end-$start)."\n";

        fwrite(STDERR,"*");
        $mode = 2;
        echo "mode=$mode\n";
        $x = $la->alloc([$rows],NDArray::int32);
        $la->fill(1,$x);
        $y = $la->alloc([$rows,$cols],NDArray::float32);
        $la->fill(1.0,$y);
        $a = $la->alloc([$numClass,$cols],NDArray::float32);
        $la->fill(0.0,$a);
        $la->scatterAdd($x,$y,$a);
        $start = hrtime(true);
        $la->scatterAdd($x,$y,$a);
        $end = hrtime(true);
        echo (explode(' ',$la->getConfig()))[0].'='.number_format($end-$start)."\n";

        fwrite(STDERR,"*");
        $mode = 4;
        echo "mode=$mode\n";
        $x = $la->alloc([$rows],NDArray::int32);
        $la->fill(1,$x);
        $y = $la->alloc([$rows,$cols],NDArray::float32);
        $la->fill(1.0,$y);
        $a = $la->alloc([$numClass,$cols],NDArray::float32);
        $la->fill(0.0,$a);
        $la->scatterAdd($x,$y,$a);
        $start = hrtime(true);
        $la->scatterAdd($x,$y,$a);
        $end = hrtime(true);
        echo (explode(' ',$la->getConfig()))[0].'='.number_format($end-$start)."\n";

        $rows = 800000;
        $cols = 8;
        $numClass = 8;#65536;
        echo "==large cols($rows,$cols,$numClass)==\n";
        fwrite(STDERR,"*");
        $mode = 0;
        echo "mode=$mode\n";
        $x = $la->alloc([$rows],NDArray::int32);
        $la->fill(1,$x);
        $y = $la->alloc([$rows,$cols],NDArray::float32);
        $la->fill(1.0,$y);
        $a = $la->alloc([$numClass,$cols],NDArray::float32);
        $la->fill(0.0,$a);
        $la->scatterAdd($x,$y,$a);
        $start = hrtime(true);
        $la->scatterAdd($x,$y,$a);
        $end = hrtime(true);
        echo (explode(' ',$la->getConfig()))[0].'='.number_format($end-$start)."\n";

        fwrite(STDERR,"*");
        $mode = 2;
        echo "mode=$mode\n";
        $x = $la->alloc([$rows],NDArray::int32);
        $la->fill(1,$x);
        $y = $la->alloc([$rows,$cols],NDArray::float32);
        $la->fill(1.0,$y);
        $a = $la->alloc([$numClass,$cols],NDArray::float32);
        $la->fill(0.0,$a);
        $la->scatterAdd($x,$y,$a);
        $start = hrtime(true);
        $la->scatterAdd($x,$y,$a);
        $end = hrtime(true);
        echo (explode(' ',$la->getConfig()))[0].'='.number_format($end-$start)."\n";

        fwrite(STDERR,"*");
        $mode = 3;
        echo "mode=$mode\n";
        $x = $la->alloc([$rows],NDArray::int32);
        $la->fill(1,$x);
        $y = $la->alloc([$rows,$cols],NDArray::float32);
        $la->fill(1.0,$y);
        $a = $la->alloc([$numClass,$cols],NDArray::float32);
        $la->fill(0.0,$a);
        $la->scatterAdd($x,$y,$a);
        $start = hrtime(true);
        $la->scatterAdd($x,$y,$a);
        $end = hrtime(true);
        echo (explode(' ',$la->getConfig()))[0].'='.number_format($end-$start)."\n";

        fwrite(STDERR,"*");
        $mode = 4;
        echo "mode=$mode\n";
        $x = $la->alloc([$rows],NDArray::int32);
        $la->fill(1,$x);
        $y = $la->alloc([$rows,$cols],NDArray::float32);
        $la->fill(1.0,$y);
        $a = $la->alloc([$numClass,$cols],NDArray::float32);
        $la->fill(0.0,$a);
        $la->scatterAdd($x,$y,$a);
        $start = hrtime(true);
        $la->scatterAdd($x,$y,$a);
        $end = hrtime(true);
        echo (explode(' ',$la->getConfig()))[0].'='.number_format($end-$start)."\n";

        $rows = 8;
        $cols = 800000;
        $numClass = 8;#65536;
        echo "==large cols($rows,$cols,$numClass)==\n";
        fwrite(STDERR,"*");
        $mode = 0;
        echo "mode=$mode\n";
        $x = $la->alloc([$rows],NDArray::int32);
        $la->fill(1,$x);
        $y = $la->alloc([$rows,$cols],NDArray::float32);
        $la->fill(1.0,$y);
        $a = $la->alloc([$numClass,$cols],NDArray::float32);
        $la->fill(0.0,$a);
        $la->scatterAdd($x,$y,$a);
        $start = hrtime(true);
        $la->scatterAdd($x,$y,$a);
        $end = hrtime(true);
        echo (explode(' ',$la->getConfig()))[0].'='.number_format($end-$start)."\n";

        fwrite(STDERR,"*");
        $mode = 2;
        echo "mode=$mode\n";
        $x = $la->alloc([$rows],NDArray::int32);
        $la->fill(1,$x);
        $y = $la->alloc([$rows,$cols],NDArray::float32);
        $la->fill(1.0,$y);
        $a = $la->alloc([$numClass,$cols],NDArray::float32);
        $la->fill(0.0,$a);
        $la->scatterAdd($x,$y,$a);
        $start = hrtime(true);
        $la->scatterAdd($x,$y,$a);
        $end = hrtime(true);
        echo (explode(' ',$la->getConfig()))[0].'='.number_format($end-$start)."\n";

        fwrite(STDERR,"*");
        $mode = 4;
        echo "mode=$mode\n";
        $x = $la->alloc([$rows],NDArray::int32);
        $la->fill(1,$x);
        $y = $la->alloc([$rows,$cols],NDArray::float32);
        $la->fill(1.0,$y);
        $a = $la->alloc([$numClass,$cols],NDArray::float32);
        $la->fill(0.0,$a);
        $la->scatterAdd($x,$y,$a);
        $start = hrtime(true);
        $la->scatterAdd($x,$y,$a);
        $end = hrtime(true);
        echo (explode(' ',$la->getConfig()))[0].'='.number_format($end-$start)."\n";

        $this->assertTrue(true);
    }

    public function testSolve()
    {
        $this->markTestSkipped('Unsuppored function on opencl');
    }

    public function testTunnerScatterAdd()
    {
        $this->markTestSkipped('Tunning only');
        return;
        $mo = $this->newMatrixOperator();
        $tunner = new OpenCLMathTunner($mo);
        //$tunner->tunningScatterAdd($mode=0,$maxTime=10**9.5,$limitTime=10**9.7);
        //$tunner->tunningScatterAdd($mode=1,$maxTime=10**9.5,$limitTime=10**9.7);
        //$tunner->tunningScatterAdd($mode=2,$maxTime=10**9.5,$limitTime=10**9.7);
        //$tunner->tunningScatterAdd($mode=3,$maxTime=10**9.5,$limitTime=10**9.7);
        $tunner->tunningScatterAdd($mode=4,$maxTime=10**9.5,$limitTime=10**9.65);
    }

    public function testShowGraphScatterAdd()
    {
        $this->markTestSkipped('Tunning only');
        return;
        $mo = $this->newMatrixOperator();
        $tunner = new OpenCLMathTunner($mo);
        $tunner->showGraphScatterAdd($mode=0,$details=true);
        $this->assertTrue(true);
    }

    public function testShowDetailGraphScatterAdd()
    {
        $this->markTestSkipped('Tunning only');
        return;
        $mo = $this->newMatrixOperator();
        $plt = new Plot(null,$mo);
        $tunner = new OpenCLMathTunner($mo);
        $mode = 4;
        for($n=8;$n<=1048576;$n<<=1) {
            $plt->figure();
            $axes = $plt->getAxes();
            //$tunner->drawGraphRowsCols3($n,$mo,$mode,$axes,$details=true,$marker=null);
            $tunner->drawGraphColsRows3($n,$mo,$mode,$axes,$details=true,$marker=null);
            //$tunner->drawGraphNumClassRows3($n,$mo,$mode,$axes,$details=true,$marker=null);
            //$tunner->drawGraphRowsNumClass3($n,$mo,$mode,$axes,$details=true,$marker=null);
            //$tunner->drawGraphNumClassCols3($n,$mo,$mode,$axes,$details=true,$marker=null);
            //$tunner->drawGraphColsNumClass3($n,$mo,$mode,$axes,$details=true,$marker=null);
        }
        //$plt->figure();
        //$axes = $plt->getAxes();
        //$tunner->drawGraphRowsCols3($nc=16,$mo,$mode,$axes,$details=true,$marker=null);
        //$tunner->drawGraphColsRows3($nc=16,$mo,$mode,$axes,$details=true,$marker=null);
        //$tunner->drawGraphNumClassRows3($co=16,$mo,$mode,$axes,$details=true,$marker=null);
        //$tunner->drawGraphRowsNumClass3($co=8,$mo,$mode,$axes,$details=true,$marker=null);
        //$tunner->drawGraphNumClassCols3($ro=8,$mo,$mode,$axes,$details=true,$marker=null);
        //$tunner->drawGraphColsNumClass3($ro=8,$mo,$mode,$axes,$details=true,$marker=null);
        $plt->show();
        $this->assertTrue(true);
    }

    public function testEditGraphScatterAdd()
    {
        $this->markTestSkipped('Tunning only');
        return;
        $mo = $this->newMatrixOperator();
        $tunner = new OpenCLMathTunner($mo);
        //$times[8][8][8] = 0.3;
        $tunner->editGraphScatterAdd($mode=4,$times);
        $this->assertTrue(true);
    }

    //public function testEditParams()
    //{
    //    $mo = $this->newMatrixOperator();
    //    $tunner = new OpenCLMathTunner($mo);
    //    $tunner->getScatterAddParameter($mode=4,$details=true);
    //}

    public function testTunnerReduceSum()
    {
        $this->markTestSkipped('Tunning only');
        return;
        $mo = $this->newMatrixOperator();
        $tunner = new OpenCLMathTunner($mo);
        //$tunner->tunningReduceSum($mode=0,$maxTime=10**9.5,$limitTime=10**9.7);
        //$tunner->tunningReduceSum($mode=1,$maxTime=10**9.5,$limitTime=10**9.7);
        //$tunner->tunningReduceSum($mode=2,$maxTime=10**9.5,$limitTime=10**9.7);
        $tunner->tunningReduceSum($mode=3,$maxTime=10**9.5,$limitTime=10**9.7);
    }

    public function testShowGraphReduceSum()
    {
        $this->markTestSkipped('Tunning only');
        return;
        $mo = $this->newMatrixOperator();
        $tunner = new OpenCLMathTunner($mo);
        $tunner->showGraphReduceSum($mode=3,$details=true);
        $this->assertTrue(true);
    }

    public function testEditGraphReduceSum()
    {
        $this->markTestSkipped('Tunning only');
        return;
        $mo = $this->newMatrixOperator();
        $tunner = new OpenCLMathTunner($mo);
        $times[8][8] = 1.0;
        $tunner->EditGraphReduceSum($mode=3,$times);
        $tunner->showGraphReduceSum($mode=3,$details=true);
        $this->assertTrue(true);
    }
}
