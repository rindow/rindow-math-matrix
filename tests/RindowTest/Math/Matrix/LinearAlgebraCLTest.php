<?php
namespace RindowTest\Math\Matrix\LinearAlgebraCLTest;

//if(!class_exists('RindowTest\Math\Matrix\LinearAlgebraTest\Test')) {
//    require_once __DIR__.'/../../../../../../rindow-math-matrix/tests/RindowTest/Math/Matrix/LinearAlgebraTest.php';
//}
if(!class_exists('RindowTest\Math\Matrix\LinearAlgebraTest\Test')) {
    require_once __DIR__.'/LinearAlgebraTest.php';
}

use RindowTest\Math\Matrix\LinearAlgebraTest\Test as ORGTest;
use Rindow\Math\Matrix\MatrixOperator;

use Interop\Polite\Math\Matrix\NDArray;
use Interop\Polite\Math\Matrix\OpenCL;
use Rindow\Math\Matrix\LinearAlgebraCL;
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

    /**
     * A(X) := Y
     */
    public function scatterAddTest(
        NDArray $X,
        NDArray $Y,
        NDArray $A,
        int $axis=null,
        $events=null,$waitEvents=null,
        int $mode = null
        ) : NDArray
    {
        if($axis===null) {
            $axis=0;
        }
        if($X->dtype()!=NDArray::int32 && $X->dtype()!=NDArray::uint32) {
            $waitPrev = $waitEvents;
            $waitEvents = $this->newEventList();
            $X = $this->astype($X,NDArray::int32,null,$waitEvents,$waitPrev);
        }
        if($axis==0) {
            return $this->scatterAddAxis0Test(true,$X,$Y,null,$A,$events,$waitEvents,$mode);
        } elseif($axis==1) {
            return $this->scatterAddAxis1Test(true,$X,$Y,null,$A,$events,$waitEvents,$mode);
        } else {
            throw new InvalidArgumentException('axis must be 0 or 1');
        }
    }

    protected function scatterAddAxis0Test(
        bool $addMode,
        NDArray $X,
        NDArray $Y,
        int $numClass=null,
        NDArray $A=null,
        $events=null,$waitEvents=null,
        int $mode=null
        ) : NDArray
    {
        if($X->ndim()!=1) {
            throw new InvalidArgumentException('"X" must be 1D-NDArray.');
        }
        $countX = $X->shape()[0];
        $shape = $Y->shape();
        $countY = array_shift($shape);
        if($countX!=$countY) {
            throw new InvalidArgumentException('Unmatch size "Y" with "X".');
        }
        $n = (int)array_product($shape);
        if($A==null) {
            $m = $numClass;
            array_unshift($shape,$numClass);
            $A = $this->alloc($shape,$Y->dtype());
            $waitPrev = $waitEvents;
            $waitEvents = $this->newEventList();
            $this->zeros($A,$waitEvents,$waitPrev);
        } else {
            $m = $A->shape()[0];
            array_unshift($shape,$m);
            if($A->shape()!=$shape){
                throw new InvalidArgumentException('Unmatch size "Y" with "A" .');
            }
        }

        $AA = $A->buffer();
        $offA = $A->offset();
        $ldA = $n;
        $XX = $X->buffer();
        $offX = $X->offset();
        $YY = $Y->buffer();
        $offY = $Y->offset();
        $ldY = $n;

        switch($mode) {
            case 1: {
                $this->openclmath->scatterAddAxis0_1(
                    $m,
                    $n,
                    $countX,
                    $AA,$offA,$ldA,
                    $XX,$offX,1,
                    $YY,$offY,$ldY,
                    $addMode,
                    $events,$waitEvents
                    );
                break;
            }
            case 2: {
                $this->openclmath->scatterAddAxis0_2(
                    $m,
                    $n,
                    $countX,
                    $AA,$offA,$ldA,
                    $XX,$offX,1,
                    $YY,$offY,$ldY,
                    $addMode,
                    $events,$waitEvents
                    );
                break;
            }
            case 3: {
                $this->openclmath->scatterAddAxis0_3(
                    $m,
                    $n,
                    $countX,
                    $AA,$offA,$ldA,
                    $XX,$offX,1,
                    $YY,$offY,$ldY,
                    $addMode,
                    $events,$waitEvents
                    );
                break;
            }
            case 4: {
                $this->openclmath->scatterAddAxis0_4(
                    $m,
                    $n,
                    $countX,
                    $AA,$offA,$ldA,
                    $XX,$offX,1,
                    $YY,$offY,$ldY,
                    $addMode,
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
        return $A;
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

    public function modeProvider()
    {
        return [
            'mode 1' => [1],
            'mode 2' => [2],
            'mode 3' => [3],
        ];
    }

    public function modeProvider4mode()
    {
        return [
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

    /**
    * @dataProvider modeProvider
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
    * @dataProvider modeProvider
    */
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

        if($mode>1) {
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
        // 3d array as 2d array
        $x = $la->array([[[1,2],[3,4]],[[5,6],[7,8]]]);
        $y = $la->reduceMaxTest($x,$axis=0,null,null,null,null,$mode);
        $this->assertEquals([5,6,7,8],$y->toArray());
        $x = $la->array([[[1,2],[3,4]],[[5,6],[7,8]]]);
        $y = $la->reduceMaxTest($x,$axis=1,null,null,null,null,$mode);
        $this->assertEquals([2,4,6,8],$y->toArray());

        // with offset
        $x = $la->array([[[9,9,9],[9,9,9]],[[1,2,3],[4,5,6]]]);
        $x = $x[1];
        $this->assertEquals([4,5,6],$la->reduceMaxTest($x,$axis=0,null,null,null,null,$mode)->toArray());
        $this->assertEquals([3,6],$la->reduceMaxTest($x,$axis=1,null,null,null,null,$mode)->toArray());

        if($mode>1) {
            // 257
            $x = $la->alloc([257],NDArray::float32);
            $la->fill(1,$x);
            $ret = $la->reduceMaxTest($x,$axis=1,null,null,null,null,$mode);
            $this->assertEquals(1,$la->amax($ret));
            // 65535
            $x = $la->alloc([65535],NDArray::float32);
            $la->fill(1,$x);
            $ret = $la->reduceMaxTest($x,$axis=1,null,null,null,null,$mode);
            $this->assertEquals(1,$la->amax($ret));
            // 65536
            $x = $la->alloc([65536],NDArray::float32);
            $la->fill(1,$x);
            $ret = $la->reduceMaxTest($x,$axis=1,null,null,null,null,$mode);
            $this->assertEquals(1,$la->amax($ret));
            // 65537
            $x = $la->alloc([65537],NDArray::float32);
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
        $this->assertEquals([1,1,1,1],$y->toArray());
        $x = $la->array([[[1,2],[3,4]],[[5,6],[7,8]]]);
        $y = $la->reduceArgMaxTest($x,$axis=1,null,null,null,null,$mode);
        $this->assertEquals([1,1,1,1],$y->toArray());

        // with offset
        $x = $la->array([[[9,9,9],[9,9,9]],[[1,2,3],[4,5,6]]]);
        $x = $x[1];
        $this->assertEquals([1,1,1],$la->reduceArgMaxTest($x,$axis=0,null,null,null,null,$mode)->toArray());
        $this->assertEquals([2,2],$la->reduceArgMaxTest($x,$axis=1,null,null,null,null,$mode)->toArray());

    }

    /**
    * @dataProvider modeProvider4mode
    */
    public function testScatterAddAxis0OpenCL($mode)
    {
        $mo = $this->newMatrixOperator();
        $la = $this->newLA($mo);
        // float32
        $x = $la->array([0,2],NDArray::int64);
        $y = $la->array([[1,2,3],[7,8,9]],NDArray::float32);
        $a = $la->array($mo->ones([4,3],NDArray::float32));
        $la->scatterAddTest($x,$y,$a,$axis=0,null,null,$mode);
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
            $la->scatterAddTest($x,$y,$a,$axis=0,null,null,$mode);
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
        $la->scatterAddTest($x,$y,$a,$axis=0,null,null,$mode);
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
        $la->scatterAddTest($x,$y,$a,$axis=0,null,null,$mode);
        $this->assertEquals(
           [[2,3,4],
            [1,1,1],
            [8,9,10],
            [1,1,1]],
            $a->toArray()
        );
    }

    public function testSolve()
    {
        $this->markTestSkipped('Unsuppored function on opencl');
    }

    public function testTunner()
    {
        $this->assertTrue(true);
        return;
        $mo = $this->newMatrixOperator();
        $tunner = new OpenCLMathTunner($mo);
        $tunner->tunningScatterAdd($mode=1,$maxTime=10**9.5,$limitTime=10**9.7);
        //$tunner->tunningScatterAdd($mode=2,$maxTime=10**9.5,$limitTime=10**9.7);
        //$tunner->tunningScatterAdd($mode=3,$maxTime=10**9.5,$limitTime=10**9.7);
        //$tunner->tunningScatterAdd($mode=4,$maxTime=10**9.5,$limitTime=10**9.65);
    }

    public function testShowGraph()
    {
        $this->assertTrue(true);
        return;
        $mo = $this->newMatrixOperator();
        $tunner = new OpenCLMathTunner($mo);
        $tunner->showGraphScatterAdd($mode=1,$details=true);
    }

    //public function testEditParams()
    //{
    //    $mo = $this->newMatrixOperator();
    //    $tunner = new OpenCLMathTunner($mo);
    //    $tunner->getScatterAddParameter($mode=4,$details=true);
    //}
}
