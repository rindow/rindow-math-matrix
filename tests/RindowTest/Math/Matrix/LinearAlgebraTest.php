<?php
namespace RindowTest\Math\Matrix\LinearAlgebraTest;

use PHPUnit\Framework\TestCase;
use Interop\Polite\Math\Matrix\NDArray;
use Interop\Polite\Math\Matrix\OpenCL;
use Rindow\Math\Matrix\MatrixOperator;
use Rindow\Math\Matrix\NDArrayPhp;
use Rindow\Math\Plot\Plot;
use ArrayObject;
use SplFixedArray;
use InvalidArgumentException;
use Rindow\Math\Matrix\LinearAlgebraCL;

class Test extends TestCase
{
    static protected $speedtest = false;
    protected $equalEpsilon = 1e-04;

    public function newMatrixOperator()
    {
        $mo = new MatrixOperator();
        if(extension_loaded('rindow_openblas')) {
            $mo->blas()->forceBlas(true);
            $mo->lapack()->forceLapack(true);
            $mo->math()->forceMath(true);
        }
        return $mo;
    }

    public function newLA($mo)
    {
        return $mo->la();
    }

    public function newArray(array $shape,$dtype=null)
    {
        if($dtype===null)
            $dtype = NDArray::float32;
        $array = new NDArrayPhp(null,$dtype,$shape);
        $size = $array->size();
        $buffer = $array->buffer();
        for($i=0;$i<$size;$i++) {
            $buffer[$i] = 0.0;
        }
        return $array;
    }

    public function ndarray($x)
    {
        return $x;
    }

    public function equalTest($a,$b)
    {
        $mo = $this->newMatrixOperator();
        $la = $this->newLA($mo);
        if($a instanceof NDArray) {
            if(!($b instanceof NDArray))
                throw new InvalidArgumentException('NDArrays must be of the same type.');
            if($a->shape()!=$b->shape())
                return false;
            $delta = $la->zerosLike($b);
            $la->copy($b,$delta);
            $la->axpy($a,$delta,-1.0);
            $delta = $la->asum($delta);
        } elseif(is_numeric($a)) {
            if(!is_numeric($b))
                throw new InvalidArgumentException('Values must be of the same type.');
            $delta = abs($a - $b);
        } elseif(is_bool($a)) {
            if(!is_bool($b))
                throw new InvalidArgumentException('Values must be of the same type.');
            $delta = ($a==$b)? 0 : 1;
        } else {
            throw new InvalidArgumentException('Values must be DNArray or float or int.');
        }

        if($delta < $this->equalEpsilon) {
            return true;
        } else {
            return false;
        }
    }

    public function testSpeedTest()
    {
        // ==============================================
        // The speed test should normally be False.
        // Temporarily change to True only when performing
        // the corresponding test individually.
        // ==============================================
        $this->assertFalse(self::$speedtest);
    }

    public function testExpandDims()
    {
        $mo = $this->newMatrixOperator();
        $la = $this->newLA($mo);
        $x = $la->alloc([2,3]);
        $this->assertEquals([2,3],$x->shape());
        $this->assertEquals([1,2,3],$la->expandDims($x,$axis=0)->shape());
        $this->assertEquals([2,1,3],$la->expandDims($x,$axis=1)->shape());
        $this->assertEquals([2,3,1],$la->expandDims($x,$axis=2)->shape());
        $this->assertEquals([2,3,1],$la->expandDims($x,$axis=-1)->shape());
        $this->assertEquals([2,1,3],$la->expandDims($x,$axis=-2)->shape());
        $this->assertEquals([1,2,3],$la->expandDims($x,$axis=-3)->shape());

        $x = $la->alloc([]);
        $this->assertEquals([],$x->shape());
        $this->assertEquals(1,$x->size());
        $this->assertEquals(1,count($x->buffer()));
        $this->assertEquals([1],$la->expandDims($x,$axis=0)->shape());
        $this->assertEquals([1],$la->expandDims($x,$axis=-1)->shape());
    }

    public function testExpandDimsOutofDims()
    {
        $mo = $this->newMatrixOperator();
        $la = $this->newLA($mo);
        $x = $la->alloc([2,3]);
        $this->expectException(InvalidArgumentException::class);
        $this->expectExceptionMessage('axis is out of range: -4');
        $la->expandDims($x,$axis=-4);
    }

    public function testSqueeze()
    {
        $mo = $this->newMatrixOperator();
        $la = $this->newLA($mo);
        $x = $la->alloc([1,2,1,3,1]);
        $this->assertEquals([1,2,1,3,1],$x->shape());
        $this->assertEquals([2,3],$la->squeeze($x)->shape());
        $this->assertEquals([2,1,3,1],$la->squeeze($x,$axis=0)->shape());
        $this->assertEquals([1,2,3,1],$la->squeeze($x,$axis=2)->shape());
        $this->assertEquals([1,2,1,3],$la->squeeze($x,$axis=4)->shape());
        $this->assertEquals([1,2,1,3],$la->squeeze($x,$axis=-1)->shape());
        $this->assertEquals([1,2,3,1],$la->squeeze($x,$axis=-3)->shape());
        $this->assertEquals([2,1,3,1],$la->squeeze($x,$axis=-5)->shape());
    }

    public function testSqueezeOutofDims()
    {
        $mo = $this->newMatrixOperator();
        $la = $this->newLA($mo);
        $x = $la->alloc([1,2,1,3,1]);
        $this->expectException(InvalidArgumentException::class);
        $this->expectExceptionMessage('axis is out of range: -6');
        $la->squeeze($x,$axis=-6);
    }

    public function testSqueezeCanNot()
    {
        $mo = $this->newMatrixOperator();
        $la = $this->newLA($mo);
        $x = $la->alloc([1,2,1,3,1]);
        $this->expectException(InvalidArgumentException::class);
        $this->expectExceptionMessage('Can not squeeze dim[3]');
        $la->squeeze($x,$axis=3);
    }

    public function testAlloc()
    {
        $mo = $this->newMatrixOperator();
        $la = $this->newLA($mo);
        $x = $la->alloc([2,3]);
        $this->assertEquals([2,3],$x->shape());
        $this->assertEquals(NDArray::float32,$x->dtype());
        $this->assertEquals(6,$x->size());
        $this->assertEquals(6,count($x->buffer()));
    }

    public function testScal()
    {
        $mo = $this->newMatrixOperator();
        $la = $this->newLA($mo);
        $x = $la->array([[1,2,3],[4,5,6],[7,8,9]],NDArray::float32);
        $la->scal(2,$x);
        $this->assertEquals([[2,4,6],[8,10,12],[14,16,18]],$x->toArray());
    }

    /**
    *    Y := alpha * X + Y
    */
    public function testaxpy()
    {
        $mo = $this->newMatrixOperator();
        $la = $this->newLA($mo);
        $x = $la->array([[1,2,3],[4,5,6]],NDArray::float32);
        $y = $la->array([[10,20,30],[40,50,60]],NDArray::float32);
        $la->axpy($x,$y,2);
        $this->assertEquals([[12,24,36],[48,60,72]],$y->toArray());
    }

    /**
    *    ret := X^t Y = x_1 * y_1 + ... + x_n * y_n
    */
    public function testdot()
    {
        $mo = $this->newMatrixOperator();
        $la = $this->newLA($mo);
        $x = $la->array([[1,2,3],[4,5,6]],NDArray::float32);
        $y = $la->array([[10,20,30],[40,50,60]],NDArray::float32);
        $ret = $la->dot($x,$y);
        $this->assertEquals(1*10+2*20+3*30+4*40+5*50+6*60,$ret);
    }

    /**
    *    ret := |x_1| + ... + |x_n|
    */
    public function testasum()
    {
        $mo = $this->newMatrixOperator();
        $la = $this->newLA($mo);
        $x = $la->array([[-1,2,-3],[-4,5,-6]],NDArray::float32);
        $ret = $la->asum($x);
        $this->assertEquals(1+2+3+4+5+6,$ret);
    }

    /**
    *    ret := arg max X(i)
    */
    public function testimax()
    {
        $mo = $this->newMatrixOperator();
        $la = $this->newLA($mo);
        $x = $la->array([[-1,2,-3],[-4,5,-6]],NDArray::float32);
        $ret = $la->imax($x);
        $this->assertEquals(4,$ret);
    }

    /**
    *    ret := arg max |X(i)|
    */
    public function testiamax()
    {
        $mo = $this->newMatrixOperator();
        $la = $this->newLA($mo);
        $x = $la->array([[-1,2,-3],[-4,5,-6]],NDArray::float32);
        $ret = $la->iamax($x);
        $this->assertEquals(5,$ret);
    }

    /**
    *    ret := arg min X(i)
    */
    public function testimin()
    {
        $mo = $this->newMatrixOperator();
        $la = $this->newLA($mo);
        $x = $la->array([[-1,2,-3],[-4,5,-6]],NDArray::float32);
        $ret = $la->imin($x);
        $this->assertEquals(5,$ret);
    }

    /**
    *    ret := arg min |X(i)|
    */
    public function testiamin()
    {
        $mo = $this->newMatrixOperator();
        $la = $this->newLA($mo);
        $x = $la->array([[-1,2,-3],[-4,5,-6]],NDArray::float32);
        $ret = $la->iamin($x);
        $this->assertEquals(0,$ret);
    }

    /**
    *    ret := max X(i)
    */
    public function testMaxNormal()
    {
        $mo = $this->newMatrixOperator();
        $la = $this->newLA($mo);
        $x = $la->array([[-1,2,-3],[-4,5,-6]],NDArray::float32);
        $ret = $la->max($x);
        $this->assertEquals(5,$ret);


        // INFINITY & NaN
        // *** CAUTION ****
        // This function is not compatible with numpy
        // and is compatible with argmax in tensorflow 2.6.
        $x = $la->array([0,INF,-INF],NDArray::float32);
        $ret = $la->max($x);
        $this->assertTrue(INF==$ret);

        $x = $la->array([0,INF,-INF,NAN],NDArray::float32);
        $ret = $la->max($x);
        $this->assertTrue($ret==INF);

        $x = $la->array([0,1,-1,NAN],NDArray::float32);
        $ret = $la->max($x);
        $this->assertEquals(1.0,$ret);

        $x = $la->array([NAN,1,-1,0],NDArray::float32);
        $ret = $la->max($x);
        $this->assertEquals(1.0,$ret);
    }

    /**
    *    ret := max |X(i)|
    */
    public function testamax()
    {
        $mo = $this->newMatrixOperator();
        $la = $this->newLA($mo);
        $x = $la->array([[-1,2,-3],[-4,5,-6]],NDArray::float32);
        $ret = $la->amax($x);
        $this->assertEquals(-6,$ret);

        // INFINITY & NaN
        // *** CAUTION ****
        // This function is not compatible with numpy
        // and is compatible with argmax in tensorflow 2.6.
        $x = $la->array([0,INF,-INF],NDArray::float32);
        $ret = $la->amax($x);
        //$this->assertTrue(INF==$ret);
        // -INF or INF
        $this->assertTrue($ret!=0);

        $x = $la->array([0,INF,-INF,NAN],NDArray::float32);
        $ret = $la->amax($x);
        //$this->assertTrue($ret==INF);
        // -INF or INF
        $this->assertTrue($ret!=0);
        $this->assertTrue(!is_nan($ret));

        $x = $la->array([0,1,-1,NAN],NDArray::float32);
        $ret = $la->amax($x);
        // -1 or 1
        $this->assertTrue($ret!=0);
        $this->assertTrue(!is_nan($ret));
    }

    /**
    *    ret := min X(i)
    */
    public function testminNormal()
    {
        $mo = $this->newMatrixOperator();
        $la = $this->newLA($mo);
        $x = $la->array([[-1,2,-3],[-4,5,-6]],NDArray::float32);
        $ret = $la->min($x);
        $this->assertEquals(-6,$ret);

        // INFINITY & NaN
        // *** CAUTION ****
        // This function is not compatible with numpy
        // and is compatible with argmax in tensorflow 2.6.
        $x = $la->array([0,INF,-INF],NDArray::float32);
        $ret = $la->min($x);
        $this->assertTrue(-INF==$ret);

        $x = $la->array([0,INF,-INF,NAN],NDArray::float32);
        $ret = $la->min($x);
        $this->assertTrue($ret==-INF);

        $x = $la->array([0,1,-1,NAN],NDArray::float32);
        $ret = $la->min($x);
        $this->assertEquals(-1.0,$ret);
    }

    /**
    *    ret := min |X(i)|
    */
    public function testamin()
    {
        $mo = $this->newMatrixOperator();
        $la = $this->newLA($mo);
        $x = $la->array([[-1,2,-3],[-4,5,-6]],NDArray::float32);
        $ret = $la->amin($x);
        $this->assertEquals(-1,$ret);

        // INFINITY & NaN
        // *** CAUTION ****
        // This function is not compatible with numpy
        // and is compatible with argmax in tensorflow 2.6.
        $x = $la->array([0,INF,-INF],NDArray::float32);
        $ret = $la->amin($x);
        $this->assertTrue(0==$ret);

        $x = $la->array([0,INF,-INF,NAN],NDArray::float32);
        $ret = $la->amin($x);
        // *** CAUTION ***
        // Platform dependent
        // $this->assertTrue($ret==0);

        $x = $la->array([0,1,-1,NAN],NDArray::float32);
        $ret = $la->amin($x);
        // *** CAUTION ***
        // Platform dependent
        // $this->assertEquals(0,$ret);
    }

    /**
    *    Y := X
    */
    public function testcopy()
    {
        $mo = $this->newMatrixOperator();
        $la = $this->newLA($mo);
        $x = $la->array([[-1,2,-3],[-4,5,-6]],NDArray::float32);
        $y = $la->zerosLike($x);
        $ret = $la->copy($x,$y);
        $this->assertEquals([[-1,2,-3],[-4,5,-6]],$y->toArray());
    }

    /**
    *    Y := sqrt(sum(Xn ** 2))
    */
    public function testNrm2()
    {
        $mo = $this->newMatrixOperator();
        $la = $this->newLA($mo);
        $x = $la->array([[1,2],[3,4]],NDArray::float32);
        $nrm2 = sqrt(1+2**2+3**2+4**2);
        $this->assertLessThan(0.00001,abs($nrm2-
            $la->nrm2($x)
        ));
    }

    /**
    *    a,b,cos,sin := rotg(x,y)
    *
    *   @requires extension rindow_openblas
    */
    public function testRotg()
    {
        $mo = $this->newMatrixOperator();
        $la = $this->newLA($mo);
        $x = $la->array([1,2,3,4,5],NDArray::float32);
        $y = $la->array([1,2,3,4,5],NDArray::float32);
        $x = $x->reshape([$x->size(),1]);
        $y = $y->reshape([$y->size(),1]);
        for($i=0;$i<5;$i++) {
            $xx = $x[$i][0];
            $yy = $y[$i][0];
            [$r,$z,$cos,$sin] = $la->rotg($x[$i],$y[$i]);
            $rr = $r[0];
            $zz = $z[0];
            $cc = $cos[0];
            $ss = $sin[0];
            #echo "(x,y)=(".$x[$i][0].", ".$y[$i][0].")\n";
            #echo "(r,z)=(".$rr.", ".$zz.")\n";
            #echo "(c,s)=(".$cc.", ".$ss.")\n";
            $this->assertLessThan(1e-7,abs($xx-$x[$i][0]));
            $this->assertLessThan(1e-7,abs($yy-$y[$i][0]));
            $rx =  $cc * $xx + $ss * $yy;
            $ry = -$ss * $xx + $cc * $yy;
            #echo "(rx,ry)=(".$rx.",".$ry.")\n";
            $this->assertLessThan(1e-6,abs($rr-$rx));
            $this->assertLessThan(1e-6,abs(0-$ry));
        }
    }

    public function testRot()
    {
        $mo = $this->newMatrixOperator();
        $la = $this->newLA($mo);
        $x = $la->array([1,2,3,4,5],NDArray::float32);
        $y = $la->array([1,2,3,4,5],NDArray::float32);
        $c = $la->array([cos(pi()/4)],NDArray::float32);
        $s = $la->array([sin(pi()/4)],NDArray::float32);
        $la->rot($x,$y,$c,$s);
        for($i=0;$i<5;$i++) {
            $this->assertLessThan(1e-6,abs(sqrt(2)*($i+1)-$x[$i]));
            $this->assertLessThan(1e-6,abs($y[$i]));
        }
        $la->rot($x,$y,$c,$s);
        for($i=0;$i<5;$i++) {
            $this->assertLessThan(1e-6,abs(($i+1)-$x[$i]));
            $this->assertLessThan(1e-6,abs((-$i-1)-$y[$i]));
        }
    }

    /**
    *    Y := X
    *    X := Y
    */
    public function testSwap()
    {
        $mo = $this->newMatrixOperator();
        $la = $this->newLA($mo);
        $x = $la->array($mo->arange(16,null,null,NDArray::float32));
        $y = $la->array($mo->arange(16,15,-1,NDArray::float32));
        $la->swap($x,$y);
        for($i=0;$i<16;$i++) {
            if(is_scalar($x[$i])) {
                $this->assertEquals(15-$i,$x[$i]);
                $this->assertEquals($i,$y[$i]);
            } else {
                $this->assertEquals(15-$i,$x[$i]->toArray());
                $this->assertEquals($i,$y[$i]->toArray());
            }
        }
    }

    public function testGemvNormal()
    {
        $mo = $this->newMatrixOperator();
        $la = $this->newLA($mo);
        $A = $la->array([[1,2,3],[4,5,6]]);
        $X = $la->array([100,10,1]);

        $Y = $la->gemv($A,$X);
        $this->assertEquals(
            [123,456]
        ,$Y->toArray());
    }

    public function testGemvTranspose()
    {
        $mo = $this->newMatrixOperator();
        $la = $this->newLA($mo);
        $A = $la->array([[1,2,3],[4,5,6]]);
        $X = $la->array([10,1]);

        $Y = $la->gemv($A,$X,null,null,null,$trans=true);
        $this->assertEquals(
            [14,25,36]
        ,$Y->toArray());
    }

    public function testGemmNormal()
    {
        $mo = $this->newMatrixOperator();
        $la = $this->newLA($mo);
        $A = $la->array([[1,2,3],[4,5,6],[7,8,9]]);
        $B = $la->array([[1,0,0],[0,1,0],[0,0,1]]);

        $C = $la->gemm($A,$B);
        $this->assertEquals([
            [1,2,3],
            [4,5,6],
            [7,8,9]
        ],$C->toArray());
    }

    public function testGemmScaleAlpha()
    {
        $mo = $this->newMatrixOperator();
        $la = $this->newLA($mo);
        $A = $la->array([[1,2,3],[4,5,6],[7,8,9]]);
        $B = $la->array([[1,0,0],[0,1,0],[0,0,1]]);

        $C = $la->gemm($A,$B,10);
        $this->assertEquals([
            [10,20,30],
            [40,50,60],
            [70,80,90]
        ],$C->toArray());
    }

    public function testGemmScaleBeta()
    {
        $mo = $this->newMatrixOperator();
        $la = $this->newLA($mo);
        $A = $la->array([[1,2,3],[4,5,6],[7,8,9]]);
        $B = $la->array([[1,0,0],[0,1,0],[0,0,1]]);

        $C = $la->array($mo->ones([$A->shape()[0],$B->shape()[1]]));
        $la->gemm($A,$B,null,10,$C);
        $this->assertEquals([
            [11,12,13],
            [14,15,16],
            [17,18,19]
        ],$C->toArray());
    }

    public function testGemmTransposeSquareA()
    {
        $mo = $this->newMatrixOperator();
        $la = $this->newLA($mo);
        $A = $la->array([[1,2,3],[4,5,6],[7,8,9]]);
        $B = $la->array([[1,0,0],[0,1,0],[0,0,1]]);

        $C = $la->gemm($A,$B,null,null,null,$transA=true);
        $this->assertEquals([
            [1,4,7],
            [2,5,8],
            [3,6,9]
        ],$C->toArray());
    }

    public function testGemmTransposeSquareB()
    {
        $mo = $this->newMatrixOperator();
        $la = $this->newLA($mo);
        $A = $la->array([[1,0,0],[0,1,0],[0,0,1]]);
        $B = $la->array([[1,2,3],[4,5,6],[7,8,9]]);

        $C = $la->gemm($A,$B,null,null,null,null,$transB=true);
        $this->assertEquals([
            [1,4,7],
            [2,5,8],
            [3,6,9]
        ],$C->toArray());
    }

    public function testGemmNoTransRectangleA23()
    {
        $mo = $this->newMatrixOperator();
        $la = $this->newLA($mo);
        $A = $la->array([[1,2,3],[4,5,6]]);
        $B = $la->array([[1,0,0],[0,1,0],[0,0,1]]);

        $C = $la->gemm($A,$B);
        $this->assertEquals([
            [1,2,3],
            [4,5,6],
        ],$C->toArray());
    }

    public function testGemmTransposeRectangleA32()
    {
        $mo = $this->newMatrixOperator();
        $la = $this->newLA($mo);
        $A = $la->array([[1,2],[3,4],[5,6]]);
        $B = $la->array([[1,0,0],[0,1,0],[0,0,1]]);
        $C = $la->gemm($A,$B,null,null,null,$transA=true);
        $this->assertEquals([
            [1,3,5],
            [2,4,6],
        ],$C->toArray());
    }

    public function testGemmNoTransRectangleB32()
    {
        $mo = $this->newMatrixOperator();
        $la = $this->newLA($mo);
        $A = $la->array([[1,0,0],[0,1,0],[0,0,1]]);
        $B = $la->array([[1,2],[3,4],[5,6]]);
        $C = $la->gemm($A,$B);
        $this->assertEquals([
            [1,2],
            [3,4],
            [5,6],
        ],$C->toArray());
    }

    public function testGemmTransposeRectangleB23()
    {
        $mo = $this->newMatrixOperator();
        $la = $this->newLA($mo);
        $A = $la->array([[1,0,0],[0,1,0],[0,0,1]]);
        $B = $la->array([[1,2,3],[4,5,6]]);
        $C = $la->gemm($A,$B,null,null,null,null,$transB=true);
        $this->assertEquals([
            [1,4],
            [2,5],
            [3,6],
        ],$C->toArray());
    }

    public function testGemmUnmatchShapeNoTransRectangleA32()
    {
        $mo = $this->newMatrixOperator();
        $la = $this->newLA($mo);
        $A = $la->array([[1,2],[3,4],[5,6]]);
        $B = $la->array([[1,0,0],[0,1,0],[0,0,1]]);
        $this->expectException(InvalidArgumentException::class);
        $this->expectExceptionMessage('The number of columns in "A" and the number of rows in "B" must be the same');
        $C = $la->gemm($A,$B);
    }

    public function testGemmUnmatchShapeTransposeRectangleA23()
    {
        $mo = $this->newMatrixOperator();
        $la = $this->newLA($mo);
        $A = $la->array([[1,2,3],[4,5,6]]);
        $B = $la->array([[1,0,0],[0,1,0],[0,0,1]]);
        $this->expectException(InvalidArgumentException::class);
        $this->expectExceptionMessage('The number of columns in "A" and the number of rows in "B" must be the same');
        $C = $la->gemm($A,$B,null,null,null,$transA=true);
    }

    public function testGemmUnmatchShapeNoTransRectangleB23()
    {
        $mo = $this->newMatrixOperator();
        $la = $this->newLA($mo);
        $A = $la->array([[1,0,0],[0,1,0],[0,0,1]]);
        $B = $la->array([[1,2,3],[4,5,6]]);
        $this->expectException(InvalidArgumentException::class);
        $this->expectExceptionMessage('The number of columns in "A" and the number of rows in "B" must be the same');
        $C = $la->gemm($A,$B);
    }

    public function testGemmUnmatchShapeTransposeRectangleB32()
    {
        $mo = $this->newMatrixOperator();
        $la = $this->newLA($mo);
        $A = $la->array([[1,0,0],[0,1,0],[0,0,1]]);
        $B = $la->array([[1,2],[3,4],[5,6]]);
        $this->expectException(InvalidArgumentException::class);
        $this->expectExceptionMessage('The number of columns in "A" and the number of rows in "B" must be the same');
        $C = $la->gemm($A,$B,null,null,null,null,$transB=true);
    }

    public function testGemmUnmatchOutputShapeNoTransA()
    {
        $mo = $this->newMatrixOperator();
        $la = $this->newLA($mo);
        $A = $la->array([[1,2,3],[4,5,6]]);
        $B = $la->array([[1,0,0],[0,1,0],[0,0,1]]);

        $C = $la->array($mo->zeros([3,3]));
        $this->expectException(InvalidArgumentException::class);
        $this->expectExceptionMessage('"A" and "C" must have the same number of rows."B" and "C" must have the same number of columns');
        $la->gemm($A,$B,null,null,$C);
    }

    public function testGemmUnmatchOutputShapeNoTransB()
    {
        $mo = $this->newMatrixOperator();
        $la = $this->newLA($mo);
        $A = $la->array([[1,2,3],[4,5,6]]);
        $B = $la->array([[1,0,0],[0,1,0],[0,0,1]]);

        $C = $la->array($mo->zeros([2,2]));
        $this->expectException(InvalidArgumentException::class);
        $this->expectExceptionMessage('"A" and "C" must have the same number of rows."B" and "C" must have the same number of columns');
        $la->gemm($A,$B,null,null,$C);
    }

    public function testGemmUnmatchOutputShapeTransposeA()
    {
        $mo = $this->newMatrixOperator();
        $la = $this->newLA($mo);
        $A = $la->array([[1,2],[3,4],[5,6]]);
        $B = $la->array([[1,0,0],[0,1,0],[0,0,1]]);

        $C = $la->array($mo->zeros([3,3]));
        $this->expectException(InvalidArgumentException::class);
        $this->expectExceptionMessage('"A" and "C" must have the same number of rows."B" and "C" must have the same number of columns');
        $la->gemm($A,$B,null,null,$C,$transA=true);
    }

    public function testGemmUnmatchOutputShapeTransposeB()
    {
        $mo = $this->newMatrixOperator();
        $la = $this->newLA($mo);
        $A = $la->array([[1,0,0],[0,1,0],[0,0,1]]);
        $B = $la->array([[1,2,3],[4,5,6]]);

        $C = $la->array($mo->zeros([3,3]));
        $this->expectException(InvalidArgumentException::class);
        $this->expectExceptionMessage('"A" and "C" must have the same number of rows."B" and "C" must have the same number of columns');
        $la->gemm($A,$B,null,null,$C,null,$transB=true);
    }

    public function testGemmSpeed()
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

        $rows = 2000;
        $cols = 2000;
        $a = $la->alloc([$rows,$cols],NDArray::float32);
        $b = $la->alloc([$cols,$rows],NDArray::float32);
        $la->fill(1.0,$a);
        $la->fill(1.0,$b);
        $c = $la->gemm($a,$b);
        $start = hrtime(true);
        $c = $la->gemm($a,$b);
        $end = hrtime(true);
        echo "\n".(explode(' ',$la->getConfig()))[0].'='.number_format($end-$start)."\n";
        $this->assertTrue(true);
    }

    public function testSymmNormal()
    {
        $mo = $this->newMatrixOperator();
        $la = $this->newLA($mo);
        $A = $la->array([
            [1,2,3],
            [2,4,5],
            [3,5,6],
        ]);
        $B = $la->array([
            [1,2,3,4],
            [5,6,7,8],
            [9,10,11,12],
        ]);
        $trues = $la->gemm($A,$B);
        $A = $la->array([
            [1,2,3],
            [0,4,5],
            [0,0,6],
        ]);
        $C = $la->symm($A,$B);
        $this->assertEquals($trues->toArray(),$C->toArray());
        $this->assertEquals([
            [38, 44, 50, 56],
            [67, 78, 89,100],
            [82, 96,110,124]
        ],$C->toArray());

        // lower
        $A = $la->array([
            [1,0,0],
            [2,4,0],
            [3,5,6],
        ]);
        $C = $la->symm($A,$B,null,null,null,null,true);
        $this->assertEquals($trues->toArray(),$C->toArray());
        $this->assertEquals([
            [38, 44, 50, 56],
            [67, 78, 89,100],
            [82, 96,110,124]
        ],$C->toArray());

    }

    public function testSymmRight()
    {
        $mo = $this->newMatrixOperator();
        $la = $this->newLA($mo);
        $A = $la->array([
            [1,2,3],
            [2,4,5],
            [3,5,6],
        ]);
        $B = $la->array([
            [1,2,3],
            [4,5,6],
            [7,8,9],
            [10,11,12],
        ]);
        $trues = $la->gemm($B,$A);
        $A = $la->array([
            [1,2,3],
            [0,4,5],
            [0,0,6],
        ]);
        $C = $la->symm($A,$B,null,null,null,true);
        $this->assertEquals($trues->toArray(),$C->toArray());
        $this->assertEquals([
            [14, 25, 31],
            [32, 58, 73],
            [50, 91,115],
            [68,124,157]
        ],$C->toArray());
        // lower
        $A = $la->array([
            [1,0,0],
            [2,4,0],
            [3,5,6],
        ]);
        $C = $la->symm($A,$B,null,null,null,true,true);
        $this->assertEquals($trues->toArray(),$C->toArray());
        $this->assertEquals([
            [14, 25, 31],
            [32, 58, 73],
            [50, 91,115],
            [68,124,157]
        ],$C->toArray());
    }

    public function testSyrkNormal()
    {
        $mo = $this->newMatrixOperator();
        $la = $this->newLA($mo);
        $A = $la->array([
            [1,2,3],
            [4,5,6],
            [7,8,9],
            [10,11,12],
        ]);
        [$n,$k] = $A->shape();
        $AT = $la->transpose($A);
        $trues = $la->gemm($A,$AT)->toArray();
        for($i=0;$i<$n;$i++) {
            for($j=0;$j<$i;$j++) {
                $trues[$i][$j] = 0;
            }
        }
        $C = $la->syrk($A);
        $this->assertEquals($trues,$C->toArray());
        $this->assertEquals([
            [14, 32, 50, 68],
            [ 0, 77,122,167],
            [ 0,  0,194,266],
            [ 0,  0,  0,365],
        ],$C->toArray());

        // lower
        $trues = $la->gemm($A,$AT)->toArray();
        for($i=0;$i<$n;$i++) {
            for($j=$i+1;$j<$n;$j++) {
                $trues[$i][$j] = 0;
            }
        }
        $C = $la->syrk($A,null,null,null,true);
        $this->assertEquals($trues,$C->toArray());
        $this->assertEquals([
            [14,  0,  0,  0],
            [32, 77,  0,  0],
            [50,122,194,  0],
            [68,167,266,365],
        ],$C->toArray());
    }

    public function testSyrkTranspose()
    {
        $mo = $this->newMatrixOperator();
        $la = $this->newLA($mo);
        $A = $la->array([
            [1,2,3],
            [4,5,6],
            [7,8,9],
            [10,11,12],
        ]);
        [$n,$k] = $A->shape();
        $AT = $la->transpose($A);
        $trues = $la->gemm($AT,$A)->toArray();
        for($i=0;$i<$k;$i++) {
            for($j=0;$j<$i;$j++) {
                $trues[$i][$j] = 0;
            }
        }
        $C = $la->syrk($A,null,null,null,null,true);
        $this->assertEquals($trues,$C->toArray());
        $this->assertEquals([
            [ 166, 188, 210],
            [   0, 214, 240],
            [   0,   0, 270]
        ],$C->toArray());

        // lower
        $trues = $la->gemm($AT,$A)->toArray();
        for($i=0;$i<$k;$i++) {
            for($j=$i+1;$j<$k;$j++) {
                $trues[$i][$j] = 0;
            }
        }
        $C = $la->syrk($A,null,null,null,true,true);
        $this->assertEquals($trues,$C->toArray());
        $this->assertEquals([
            [ 166,   0,   0],
            [ 188, 214,   0],
            [ 210, 240, 270]
        ],$C->toArray());
    }

    public function testSyr2kNormal()
    {
        $mo = $this->newMatrixOperator();
        $la = $this->newLA($mo);
        $A = $la->array([
            [1,2,3],
            [4,5,6],
            [7,8,9],
            [10,11,12],
        ]);
        $B = $la->array([
            [1,3,5],
            [2,4,6],
            [7,9,11],
            [8,10,12],
        ]);
        $AT = $la->transpose($A);
        $BT = $la->transpose($B);
        [$n,$k] = $A->shape();
        $trues = $la->gemm($A,$BT);
        $trues = $la->gemm($B,$AT,null,1.0,$trues)->toArray();
        for($i=0;$i<$n;$i++) {
            for($j=0;$j<$i;$j++) {
                $trues[$i][$j] = 0;
            }
        }
        $C = $la->syr2k($A,$B);
        $this->assertEquals($trues,$C->toArray());
        $this->assertEquals([
            [44, 77,134,167],
            [ 0,128,239,290],
            [ 0,  0,440,545],
            [ 0,  0,  0,668]
        ],$C->toArray());

        // lower
        $trues = $la->gemm($A,$BT);
        $trues = $la->gemm($B,$AT,null,1.0,$trues)->toArray();
        for($i=0;$i<$n;$i++) {
            for($j=$i+1;$j<$n;$j++) {
                $trues[$i][$j] = 0;
            }
        }
        $C = $la->syr2k($A,$B,null,null,null,true);
        $this->assertEquals($trues,$C->toArray());
        $this->assertEquals([
            [ 44,  0,  0,  0],
            [ 77,128,  0,  0],
            [134,239,440,  0],
            [167,290,545,668]
        ],$C->toArray());
    }

    public function testSyr2kTranspose()
    {
        $mo = $this->newMatrixOperator();
        $la = $this->newLA($mo);
        $A = $la->array([
            [1,2,3],
            [4,5,6],
            [7,8,9],
            [10,11,12],
        ]);
        $B = $la->array([
            [1,3,5],
            [2,4,6],
            [7,9,11],
            [8,10,12],
        ]);
        $AT = $la->transpose($A);
        $BT = $la->transpose($B);
        [$n,$k] = $A->shape();
        $trues = $la->gemm($BT,$A);
        $trues = $la->gemm($AT,$B,null,1.0,$trues)->toArray();
        for($i=0;$i<$k;$i++) {
            for($j=0;$j<$i;$j++) {
                $trues[$i][$j] = 0;
            }
        }
        $C = $la->syr2k($A,$B,null,null,null,null,true);
        $this->assertEquals($trues,$C->toArray());
        $this->assertEquals([
            [276,338,400],
            [  0,416,494],
            [  0,  0,588]
        ],$C->toArray());

        // lower
        $trues = $la->gemm($BT,$A);
        $trues = $la->gemm($AT,$B,null,1.0,$trues)->toArray();
        for($i=0;$i<$k;$i++) {
            for($j=$i+1;$j<$k;$j++) {
                $trues[$i][$j] = 0;
            }
        }
        $C = $la->syr2k($A,$B,null,null,null,true,true);
        $this->assertEquals($trues,$C->toArray());
        $this->assertEquals([
            [276,  0,  0],
            [338,416,  0],
            [400,494,588]
        ],$C->toArray());
    }

    /**
    *   @requires extension rindow_clblast
    */
    public function testTrmmNormal()
    {
        $mo = $this->newMatrixOperator();
        $la = $this->newLA($mo);
        $A = $la->array([
            [1,2,3],
            [0,4,5],
            [0,0,6],
        ]);
        $B = $la->array([
            [1,2,3,4],
            [5,6,7,8],
            [9,10,11,12],
        ]);
        $trues = $la->gemm($A,$B);
        $A = $la->array([
            [1,2,3],
            [9,4,5],
            [9,9,6],
        ]);
        $la->trmm($A,$B);
        $this->assertEquals($trues->toArray(),$B->toArray());
        $this->assertEquals([
            [ 38, 44, 50, 56],
            [ 65, 74, 83, 92],
            [ 54, 60, 66, 72]
        ],$trues->toArray());

        // lower
        $A = $la->array([
            [1,0,0],
            [2,4,0],
            [3,5,6],
        ]);
        $B = $la->array([
            [1,2,3,4],
            [5,6,7,8],
            [9,10,11,12],
        ]);
        $trues = $la->gemm($A,$B);
        $A = $la->array([
            [1,9,9],
            [2,4,9],
            [3,5,6],
        ]);
        $C = $la->trmm($A,$B,null,null,true);
        $this->assertEquals($trues->toArray(),$C->toArray());
        $this->assertEquals([
            [  1,  2,  3,  4],
            [ 22, 28, 34, 40],
            [ 82, 96,110,124]
        ],$trues->toArray());
    }

    /**
    *   @requires extension rindow_clblast
    */
    public function testTrmmTranspose()
    {
        $mo = $this->newMatrixOperator();
        $la = $this->newLA($mo);
        $A = $la->array([
            [1,2,3],
            [0,4,5],
            [0,0,6],
        ]);
        $B = $la->array([
            [1,2,3,4],
            [5,6,7,8],
            [9,10,11,12],
        ]);
        $trues = $la->gemm($la->transpose($A),$B);
        $A = $la->array([
            [1,2,3],
            [9,4,5],
            [9,9,6],
        ]);
        $la->trmm($A,$B,null,null,null,true);
        $this->assertEquals($trues->toArray(),$B->toArray());
        $this->assertEquals([
            [  1,  2,  3,  4],
            [ 22, 28, 34, 40],
            [ 82, 96,110,124]
        ],$trues->toArray());

        // lower
        $A = $la->array([
            [1,0,0],
            [2,4,0],
            [3,5,6],
        ]);
        $B = $la->array([
            [1,2,3,4],
            [5,6,7,8],
            [9,10,11,12],
        ]);
        $trues = $la->gemm($la->transpose($A),$B);
        $A = $la->array([
            [1,9,9],
            [2,4,9],
            [3,5,6],
        ]);
        $C = $la->trmm($A,$B,null,null,true,true);
        $this->assertEquals($trues->toArray(),$C->toArray());
        $this->assertEquals([
            [ 38, 44, 50, 56],
            [ 65, 74, 83, 92],
            [ 54, 60, 66, 72]
        ],$trues->toArray());
    }

    /**
    *   @requires extension rindow_clblast
    */
    public function testTrmmUnit()
    {
        $mo = $this->newMatrixOperator();
        $la = $this->newLA($mo);
        $A = $la->array([
            [1,2,3],
            [0,1,5],
            [0,0,1],
        ]);
        $B = $la->array([
            [1,2,3,4],
            [5,6,7,8],
            [9,10,11,12],
        ]);
        $trues = $la->gemm($A,$B);
        $A = $la->array([
            [9,2,3],
            [9,9,5],
            [9,9,9],
        ]);
        $la->trmm($A,$B,null,null,null,null,true);
        $this->assertEquals($trues->toArray(),$B->toArray());
        $this->assertEquals([
            [ 38, 44, 50, 56],
            [ 50, 56, 62, 68],
            [  9, 10, 11, 12]
        ],$trues->toArray());

        // lower
        $A = $la->array([
            [1,0,0],
            [2,1,0],
            [3,5,1],
        ]);
        $B = $la->array([
            [1,2,3,4],
            [5,6,7,8],
            [9,10,11,12],
        ]);
        $trues = $la->gemm($A,$B);
        $A = $la->array([
            [9,9,9],
            [2,9,9],
            [3,5,9],
        ]);
        $C = $la->trmm($A,$B,null,null,true,null,true);
        $this->assertEquals($trues->toArray(),$C->toArray());
        $this->assertEquals([
            [  1,  2,  3,  4],
            [  7, 10, 13, 16],
            [ 37, 46, 55, 64]
        ],$trues->toArray());
    }

    /**
    *   @requires extension rindow_clblast
    */
    public function testTrmmRight()
    {
        $mo = $this->newMatrixOperator();
        $la = $this->newLA($mo);
        $A = $la->array([
            [1,2,3],
            [0,4,5],
            [0,0,6],
        ]);
        $B = $la->array([
            [1,2,3],
            [4,5,6],
            [7,8,9],
            [10,11,12],
        ]);
        $trues = $la->gemm($B,$A);
        $A = $la->array([
            [1,2,3],
            [9,4,5],
            [9,9,6],
        ]);
        $la->trmm($A,$B,null,true);
        $this->assertEquals($trues->toArray(),$B->toArray());
        $this->assertEquals([
            [  1, 10, 31],
            [  4, 28, 73],
            [  7, 46,115],
            [ 10, 64,157]
        ],$trues->toArray());

        // lower
        $A = $la->array([
            [1,0,0],
            [2,4,0],
            [3,5,6],
        ]);
        $B = $la->array([
            [1,2,3],
            [4,5,6],
            [7,8,9],
            [10,11,12],
        ]);
        $trues = $la->gemm($B,$A);
        $A = $la->array([
            [1,9,9],
            [2,4,9],
            [3,5,6],
        ]);
        $la->trmm($A,$B,null,true,true);
        $this->assertEquals($trues->toArray(),$B->toArray());
        $this->assertEquals([
            [ 14, 23, 18],
            [ 32, 50, 36],
            [ 50, 77, 54],
            [ 68,104, 72]
        ],$trues->toArray());
    }

    /**
    *   @requires extension rindow_clblast
    */
    public function testTrsmNormal()
    {
        $mo = $this->newMatrixOperator();
        $la = $this->newLA($mo);
        $A = $la->array([
            [1,2,3],
            [0,4,5],
            [0,0,6],
        ]);
        $B = $la->array([
            [1,2,3,4],
            [5,6,7,8],
            [9,10,11,12],
        ]);
        $trues = $la->gemm($A,$B);
        $A = $la->array([
            [1,2,3],
            [9,4,5],
            [9,9,6],
        ]);
        $la->trsm($A,$B);
        $this->markTestSkipped('trsm');
    }

    public function testMatmulNormal()
    {
        $mo = $this->newMatrixOperator();
        $la = $this->newLA($mo);
        $A = $la->array([[[1,2,3],[4,5,6]],  [[60,50,40],[30,20,10]]]);
        $B = $la->array([[[1,0],[0,1],[0,0]],[[2,0],[0,2],[0,0]]]);

        $C = $la->matmul($A,$B);
        $this->assertEquals([
            [[1,2],
             [4,5]],
            [[120,100],
             [60,40]],
        ],$C->toArray());

        $C = $la->matmul($B,$A);
        $this->assertEquals([
            [[1,2,3],
             [4,5,6],
             [0,0,0]],
            [[120,100,80],
             [60,40,20],
             [0,0,0]],
        ],$C->toArray());


        $A = $la->array([[[1,2,3],[4,5,6]],  [[60,50,40],[30,20,10]]]);
        $B = $la->array([[1,0],[0,1],[0,0]]);

        $C = $la->matmul($A,$B);
        $this->assertEquals([
            [[1,2],
             [4,5]],
            [[60,50],
             [30,20]],
        ],$C->toArray());

        $C = $la->matmul($B,$A);
        $this->assertEquals([
            [[1,2,3],
             [4,5,6],
             [0,0,0]],
            [[60,50,40],
             [30,20,10],
             [0,0,0]],
        ],$C->toArray());
    }

    public function testMatmulTransposeA()
    {
        $mo = $this->newMatrixOperator();
        $la = $this->newLA($mo);
        $A = $la->array([[[1,2,3],[4,5,6]],  [[60,50,40],[30,20,10]]]);
        $B = $la->array([[[1,0,0],[0,1,0]],[[2,0,0],[0,2,0]]]);

        $C = $la->matmul($A,$B,$transA=true);
        $this->assertEquals([
            [[1,4,0],
             [2,5,0],
             [3,6,0]],
            [[120,60,0],
             [100,40,0],
             [80,20,0]],
        ],$C->toArray());

        $C = $la->matmul($B,$A,$transA=true);
        $this->assertEquals([
            [[1,2,3],
             [4,5,6],
             [0,0,0]],
            [[120,100,80],
             [60,40,20],
             [0,0,0]],
        ],$C->toArray());

        $A = $la->array([[[1,2,3],[4,5,6]],  [[60,50,40],[30,20,10]]]);
        $B = $la->array([[1,0,0],[0,1,0]]);

        $C = $la->matmul($A,$B,$transA=true);
        $this->assertEquals([
            [[1,4,0],
             [2,5,0],
             [3,6,0]],
            [[60,30,0],
             [50,20,0],
             [40,10,0]],
        ],$C->toArray());

        $C = $la->matmul($B,$A,$transA=true);
        $this->assertEquals([
            [[1,2,3],
             [4,5,6],
             [0,0,0]],
            [[60,50,40],
             [30,20,10],
             [0,0,0]],
        ],$C->toArray());
    }

    public function testMatmulTransposeB()
    {
        $mo = $this->newMatrixOperator();
        $la = $this->newLA($mo);
        $A = $la->array([[[1,2,3],[4,5,6]],  [[60,50,40],[30,20,10]]]);
        $B = $la->array([[[1,0,0],[0,1,0]],[[2,0,0],[0,2,0]]]);

        $C = $la->matmul($A,$B,null,$transB=true);
        $this->assertEquals([
            [[1,2],
             [4,5]],
            [[120,100],
             [60,40]],
        ],$C->toArray());

        $C = $la->matmul($B,$A,null,$transB=true);
        $this->assertEquals([
            [[1,4],
             [2,5]],
            [[120,60],
             [100,40]],
        ],$C->toArray());

        $A = $la->array([[[1,2,3],[4,5,6]],  [[60,50,40],[30,20,10]]]);
        $B = $la->array([[1,0,0],[0,1,0]]);

        $C = $la->matmul($A,$B,null,$transB=true);
        $this->assertEquals([
            [[1,2],
             [4,5]],
            [[60,50],
             [30,20]],
        ],$C->toArray());

        $C = $la->matmul($B,$A,null,$transB=true);
        $this->assertEquals([
            [[1,4],
             [2,5]],
            [[60,30],
             [50,20]],
        ],$C->toArray());
    }

    public function testMatmul4d()
    {
        $mo = $this->newMatrixOperator();
        $la = $this->newLA($mo);
        $A = $la->array([[[[1,2,3],[4,5,6]],  [[60,50,40],[30,20,10]]],
                         [[[1,2,3],[4,5,6]],  [[60,50,40],[30,20,10]]]]);
        $B = $la->array([[[[1,0],[0,1],[0,0]],[[2,0],[0,2],[0,0]]],
                         [[[1,0],[0,1],[0,0]],[[2,0],[0,2],[0,0]]]]);

        $C = $la->matmul($A,$B);
        $this->assertEquals([
            [[[1,2],
              [4,5]],
             [[120,100],
              [60,40]]],
            [[[1,2],
              [4,5]],
             [[120,100],
              [60,40]]],
        ],$C->toArray());
    }

    public function testMatmulUnmatchBroadcast()
    {
        $mo = $this->newMatrixOperator();
        $la = $this->newLA($mo);
        $A = $la->array([[[1,2],[3,4],[5,6]],[[1,2],[3,4],[5,6]]]);
        $B = $la->array([[[1,0],[0,1]],[[1,0],[0,1]],[[1,0],[0,1]]]);

        $this->expectException(InvalidArgumentException::class);
        $this->expectExceptionMessage('Matrix size-incompatible for broadcast:[2,3,2]<=>[3,2,2]');
        $C = $la->matmul($A,$B);
    }

    public function testMatmulUnmatchBaseMatrix()
    {
        $mo = $this->newMatrixOperator();
        $la = $this->newLA($mo);
        $A = $la->array([[[1,2],[3,4],[5,6]],[[1,2],[3,4],[5,6]]]);
        $B = $la->array([[[1,0],[0,1],[1,0]],[[1,0],[0,1],[1,0]]]);

        $this->expectException(InvalidArgumentException::class);
        $this->expectExceptionMessage('The number of columns in "A" and the number of rows in "B" must be the same:[2,3,2]<=>[2,3,2]');
        $C = $la->matmul($A,$B);
    }

    public function testMatmulUnmatchOutputShape()
    {
        $mo = $this->newMatrixOperator();
        $la = $this->newLA($mo);
        $A = $la->array([[[1,2],[3,4],[5,6]],[[1,2],[3,4],[5,6]]]);
        $B = $la->array([[[1,0],[0,1]],[[1,0],[0,1]]]);
        $C = $la->array($mo->zeros([2,2,2]));

        $this->expectException(InvalidArgumentException::class);
        $this->expectExceptionMessage('"A" and "C" must have the same number of rows."B" and "C" must have the same number of columns:[2,3,2] , [2,2,2] => [2,2,2]');
        $C = $la->matmul($A,$B,null,null,$C);
    }

    public function providerSumNormal()
    {
        return [
            'float32' => [NDArray::float32],
            'int32'   => [NDArray::int32],
        ];
    }
    /**
    *    ret := |x_1| + ... + |x_n|
    *
    * @dataProvider providerSumNormal
    */
    public function testSumNormal($dtype)
    {
        $mo = $this->newMatrixOperator();
        $la = $this->newLA($mo);
        $x = $la->array([[1,2,-3],[-4,5,-6]],$dtype);
        $ret = $la->sum($x);
        $this->assertEquals(1+2-3-4+5-6,$ret);

        // 1
        $x = $la->alloc([1],$dtype);
        $la->fill(1,$x);
        $ret = $la->sum($x);
        $this->assertEquals(1,$ret);

        // 2
        $x = $la->alloc([2],$dtype);
        $la->fill(1,$x);
        $ret = $la->sum($x);
        $this->assertEquals(2,$ret);

        // 3
        $x = $la->alloc([3],$dtype);
        $la->fill(1,$x);
        $ret = $la->sum($x);
        $this->assertEquals(3,$ret);

        // 4
        $x = $la->alloc([4],$dtype);
        $la->fill(1,$x);
        $ret = $la->sum($x);
        $this->assertEquals(4,$ret);

        // 256
        $x = $la->alloc([256],$dtype);
        $la->fill(1,$x);
        $ret = $la->sum($x);
        $this->assertEquals(256,$ret);

        // over 256
        $x = $la->alloc([1000],$dtype);
        $la->fill(1,$x);
        $ret = $la->sum($x);
        $this->assertEquals(1000,$ret);

        // over 65536
        $x = $la->alloc([70000],$dtype);
        $la->fill(1,$x);
        $ret = $la->sum($x);
        $this->assertEquals(70000,$ret);
    }

    public function testSumIntegerAndBool()
    {
        $mo = $this->newMatrixOperator();
        $la = $this->newLA($mo);
        $x = $la->array([[1,2,-3],[-4,5,-6]],NDArray::int32);
        $ret = $la->sum($x);
        $this->assertEquals(1+2-3-4+5-6,$ret);

        $x = $la->array([[true,false,true],[false,true,false]],NDArray::bool);
        $ret = $la->sum($x);
        $this->assertEquals(3,$ret);
    }

    public function testSumLarge()
    {
        $mo = $this->newMatrixOperator();
        $la = $this->newLA($mo);
        if(!$la->accelerated()) {
            $this->markTestSkipped('Skip due to high load');
            return;
        }

        // large size
        $size = 2000000;
        $x = $la->alloc([$size],NDArray::float32);
        $la->fill(1.0,$x);
        $sum = $la->sum($x);
        $this->assertLessThan(1e-3,$size-$sum);
    }

    public function testSumSpeed()
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
        $size = 2000000;
        $x = $la->alloc([$size],NDArray::float32);
        $la->fill(1.0,$x);
        $sum = $la->sum($x);
        $start = hrtime(true);
        $sum = $la->sum($x);
        $end = hrtime(true);
        echo "\n".(explode(' ',$la->getConfig()))[0].'='.number_format($end-$start)."\n";
        if(!is_scalar($sum)) {
            $sum = $sum->toArray();
        }
        $this->assertLessThan(1e-3,$size-$sum);
    }

    public function testIncrement()
    {
        $mo = $this->newMatrixOperator();
        $la = $this->newLA($mo);
        $X = $la->array([[1,2,3],[4,5,6]]);

        // X := X + 1
        $la->increment($X,1.0);
        $this->assertEquals(
            [[2,3,4],[5,6,7]]
        ,$X->toArray());

        // X := 8 - X
        $la->increment($X,8.0,-1.0);
        $this->assertEquals(
            [[6,5,4],[3,2,1]]
        ,$X->toArray());

        // X := 2 * X
        $la->increment($X,null,2.0);
        $this->assertEquals(
            [[12,10,8],[6,4,2]]
        ,$X->toArray());
    }

    public function testReciprocal()
    {
        $mo = $this->newMatrixOperator();
        $la = $this->newLA($mo);

        // X := 1 / X
        $X = $la->array([[1,2,4],[8,16,32]]);
        $la->reciprocal($X);
        $this->assertEquals(
            [[1,0.5,0.25],[0.125,0.0625, 0.03125]]
        ,$X->toArray());

        // X := 1 / (X + 1)
        $X = $la->array([[0,1,3],[7,15,31]]);
        $la->reciprocal($X,1.0);
        $this->assertEquals(
            [[1,0.5,0.25],[0.125,0.0625, 0.03125]]
        ,$X->toArray());

        // X := 1 / (32 - X)
        $X = $la->array([[31,30,28],[24,16,0]]);
        $la->reciprocal($X,32,-1.0);
        $this->assertEquals(
            [[1,0.5,0.25],[0.125,0.0625, 0.03125]]
        ,$X->toArray());

        // INFINITY & NaN
        $X = $la->array([4,2,0,INF,-INF,NAN]);
        $la->reciprocal($X);
        $X = $la->toNDArray($X);
        $this->assertEquals(0.25,$X[0]);
        $this->assertEquals(0.5, $X[1]);
        $this->assertTrue(INF==$X[2]);
        $this->assertEquals(0, $X[3]);
        $this->assertEquals(0, $X[4]);
        $this->assertTrue(is_nan($X[5]));
    }

    public function testMaximumFloat()
    {
        $mo = $this->newMatrixOperator();
        $la = $this->newLA($mo);

        // X := maximum(1,X)
        $X = $la->array([[-1,0,1],[2,3,4]]);
        $la->maximum($X,1);
        $this->assertEquals(
            [[1,1,1],[2,3,4]]
        ,$X->toArray());

        // INFINITY & NaN
        // Compatible with tenserflow 2.6
        $X = $la->array([1,0,-1,INF,-INF,NAN]);
        $la->maximum($X,0);
        $X = $la->toNDArray($X);
        $this->assertEquals(1,   $X[0]);
        $this->assertEquals(0,   $X[1]);
        $this->assertEquals(0,   $X[2]);
        $this->assertTrue(INF==  $X[3]);
        $this->assertEquals(0,   $X[4]);
        $this->assertTrue(is_nan($X[5]));

        $X = $la->array([1,0,-1,INF,-INF,NAN]);
        $la->maximum($X,NAN);
        $X = $la->toNDArray($X);
        $this->assertTrue(is_nan($X[0]));
        $this->assertTrue(is_nan($X[1]));
        $this->assertTrue(is_nan($X[2]));
        $this->assertTrue(is_nan($X[3]));
        $this->assertTrue(is_nan($X[4]));
        $this->assertTrue(is_nan($X[5]));
    }

    public function testMaximumCompDim()
    {
        $mo = $this->newMatrixOperator();
        $la = $this->newLA($mo);

        // Same dim
        $X = $la->array([[-1,0,1], [2,3,4]]);
        $Y = $la->array([[-2,-1,2],[4,3,2]]);
        $la->maximum($X,$Y);
        $this->assertEquals(
            [[-1,0,2],[4,3,4]]
        ,$X->toArray());

        // Broadcast
        $X = $la->array([[-1,0,4], [2,4,1]]);
        $Y = $la->array([-2,1,2]);
        $la->maximum($X,$Y);
        $this->assertEquals(
            [[-1,1,4],[2,4,2]]
        ,$X->toArray());
    }

    public function testMinimumFloat()
    {
        $mo = $this->newMatrixOperator();
        $la = $this->newLA($mo);

        // X := minimum(1,X)
        $X = $la->array([[-1,0,1],[2,3,4]]);
        $la->minimum($X,1);
        $this->assertEquals(
            [[-1,0,1],[1,1,1]]
        ,$X->toArray());

        // INFINITY & NaN
        // Compatible with tenserflow 2.6
        $X = $la->array([1,0,-1,INF,-INF,NAN]);
        $la->minimum($X,0);
        $X = $la->toNDArray($X);
        $this->assertEquals(0,   $X[0]);
        $this->assertEquals(0,   $X[1]);
        $this->assertEquals(-1,  $X[2]);
        $this->assertEquals(0,   $X[3]);
        $this->assertTrue(-INF== $X[4]);
        $this->assertTrue(is_nan($X[5]));

        $X = $la->array([1,0,-1,INF,-INF,NAN]);
        $la->minimum($X,NAN);
        $X = $la->toNDArray($X);
        $this->assertTrue(is_nan($X[0]));
        $this->assertTrue(is_nan($X[1]));
        $this->assertTrue(is_nan($X[2]));
        $this->assertTrue(is_nan($X[3]));
        $this->assertTrue(is_nan($X[4]));
        $this->assertTrue(is_nan($X[5]));
    }

    public function testMinimumCompDim()
    {
        $mo = $this->newMatrixOperator();
        $la = $this->newLA($mo);

        // Same dim
        $X = $la->array([[-1,0,1], [2,3,4]]);
        $Y = $la->array([[-2,-1,2],[4,3,2]]);
        $la->minimum($X,$Y);
        $this->assertEquals(
            [[-2,-1,1],[2,3,2]]
        ,$X->toArray());

        // Broadcast
        $X = $la->array([[-1,0,4], [2,4,1]]);
        $Y = $la->array([-2,1,2]);
        $la->minimum($X,$Y);
        $this->assertEquals(
            [[-2,0,2],[-2,1,1]]
        ,$X->toArray());
    }

    public function testGreaterFloat()
    {
        $mo = $this->newMatrixOperator();
        $la = $this->newLA($mo);

        // X := greater(X,1)
        $X = $la->array([[-1,0,1],[1.5,2,3]]);
        $la->greater($X,1);
        $this->assertEquals(
            [[0,0,0],[1,1,1]]
        ,$X->toArray());

        // INFINITY & NaN
        $X = $la->array([INF,0,-INF,NAN]);
        $la->greater($X,0);
        $X = $la->toNDArray($X);
        $this->assertEquals(1.0, $X[0]); // INF
        $this->assertEquals(0.0, $X[1]); // 0
        $this->assertEquals(0.0, $X[2]); // -INF
        $this->assertEquals(0.0, $X[3]); // NAN

        $X = $la->array([INF,0,-INF,NAN]);
        $la->greater($X,NAN);
        $X = $la->toNDArray($X);
        $this->assertEquals(0.0, $X[0]);
        $this->assertEquals(0.0, $X[1]);
        $this->assertEquals(0.0, $X[2]);
        $this->assertEquals(0.0, $X[3]);
    }

    public function testGreaterCompDim()
    {
        $mo = $this->newMatrixOperator();
        $la = $this->newLA($mo);

        // Same dim
        $X = $la->array([[-1,0,1], [2,3,4]]);
        $Y = $la->array([[-2,-1,2],[4,3,2]]);
        $la->greater($X,$Y);
        $this->assertEquals(
            [[1,1,0],[0,0,1]]
        ,$X->toArray());

        // Broadcast
        $X = $la->array([[-1,0,4], [2,4,1]]);
        $Y = $la->array([-2,1,2]);
        $la->greater($X,$Y);
        $this->assertEquals(
            [[1,0,1],[1,1,0]]
        ,$X->toArray());
    }

    public function testGreaterSpeed()
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

        $n = 1000000;
        $X = $la->alloc([$n]);
        $la->ones($X);
        $la->greater($X,1);
        $start = hrtime(true);
        $la->greater($X,1);
        $end = hrtime(true);
        echo (explode(' ',$la->getConfig()))[0].'='.number_format($end-$start)."\n";
        $this->assertTrue(true);
    }

    public function testGreaterEqualFloat()
    {
        $mo = $this->newMatrixOperator();
        $la = $this->newLA($mo);

        // X := greater_equal(X,1)
        $X = $la->array([[-1,0,0.5],[1,2,3]]);
        $la->greaterEqual($X,1);
        $this->assertEquals(
            [[0,0,0],[1,1,1]]
        ,$X->toArray());

        // INFINITY & NaN
        $X = $la->array([INF,0,-INF,NAN]);
        $la->greaterEqual($X,0);
        $X = $la->toNDArray($X);
        $this->assertEquals(1.0, $X[0]); // INF
        $this->assertEquals(1.0, $X[1]); // 0
        $this->assertEquals(0.0, $X[2]); // -INF
        $this->assertEquals(0.0, $X[3]); // NAN

        $X = $la->array([INF,0,-INF,NAN]);
        $la->greaterEqual($X,NAN);
        $X = $la->toNDArray($X);
        $this->assertEquals(0.0, $X[0]);
        $this->assertEquals(0.0, $X[1]);
        $this->assertEquals(0.0, $X[2]);
        $this->assertEquals(0.0, $X[3]);
    }

    public function testGreaterEqualCompDim()
    {
        $mo = $this->newMatrixOperator();
        $la = $this->newLA($mo);

        // Same dim
        $X = $la->array([[-1,0,1], [2,3,4]]);
        $Y = $la->array([[-2,-1,2],[4,3,2]]);
        $la->greaterEqual($X,$Y);
        $this->assertEquals(
            [[1,1,0],[0,1,1]]
        ,$X->toArray());

        // Broadcast
        $X = $la->array([[-1,0,4], [2,0,2]]);
        $Y = $la->array([-2,1,2]);
        $la->greaterEqual($X,$Y);
        $this->assertEquals(
            [[1,0,1],[1,0,1]]
        ,$X->toArray());
    }

    public function testLessFloat()
    {
        $mo = $this->newMatrixOperator();
        $la = $this->newLA($mo);

        // X := less(1,X)
        $X = $la->array([[-1,0,0.5],[1,2,3]]);
        $la->less($X,1);
        $this->assertEquals(
            [[1,1,1],[0,0,0]]
        ,$X->toArray());

        // INFINITY & NaN
        $X = $la->array([INF,0,-INF,NAN]);
        $la->less($X,0);
        $X = $la->toNDArray($X);
        $this->assertEquals(0.0, $X[0]); // INF
        $this->assertEquals(0.0, $X[1]); // 0
        $this->assertEquals(1.0, $X[2]); // -INF
        $this->assertEquals(0.0, $X[3]); // NAN

        $X = $la->array([INF,0,-INF,NAN]);
        $la->less($X,NAN);
        $X = $la->toNDArray($X);
        $this->assertEquals(0.0, $X[0]);
        $this->assertEquals(0.0, $X[1]);
        $this->assertEquals(0.0, $X[2]);
        $this->assertEquals(0.0, $X[3]);
    }

    public function testLessCompDim()
    {
        $mo = $this->newMatrixOperator();
        $la = $this->newLA($mo);

        // Same dim
        $X = $la->array([[-1,0,1], [2,3,4]]);
        $Y = $la->array([[-2,-1,2],[4,3,2]]);
        $la->less($X,$Y);
        $this->assertEquals(
            [[0,0,1],[1,0,0]]
        ,$X->toArray());

        // Broadcast
        $X = $la->array([[-1,0,4], [2,0,2]]);
        $Y = $la->array([-2,1,2]);
        $la->less($X,$Y);
        $this->assertEquals(
            [[0,1,0],[0,1,0]]
        ,$X->toArray());
    }

    public function testLessEqualFloat()
    {
        $mo = $this->newMatrixOperator();
        $la = $this->newLA($mo);

        // X := less(1,X)
        $X = $la->array([[-1,0,1],[1.5,2,3]]);
        $la->lessEqual($X,1);
        $this->assertEquals(
            [[1,1,1],[0,0,0]]
        ,$X->toArray());

        // INFINITY & NaN
        $X = $la->array([INF,0,-INF,NAN]);
        $la->lessEqual($X,0);
        $X = $la->toNDArray($X);
        $this->assertEquals(0.0, $X[0]);
        $this->assertEquals(1.0, $X[1]);
        $this->assertEquals(1.0, $X[2]);
        $this->assertEquals(0.0, $X[3]);

        $X = $la->array([INF,0,-INF,NAN]);
        $la->lessEqual($X,NAN);
        $X = $la->toNDArray($X);
        $this->assertEquals(0.0, $X[0]);
        $this->assertEquals(0.0, $X[1]);
        $this->assertEquals(0.0, $X[2]);
        $this->assertEquals(0.0, $X[3]);
    }

    public function testLessEqualCompDim()
    {
        $mo = $this->newMatrixOperator();
        $la = $this->newLA($mo);

        // Same dim
        $X = $la->array([[-1,0,1], [2,3,4]]);
        $Y = $la->array([[-2,-1,2],[4,3,2]]);
        $la->lessEqual($X,$Y);
        $this->assertEquals(
            [[0,0,1],[1,1,0]]
        ,$X->toArray());

        // Broadcast
        $X = $la->array([[-1,0,4], [2,0,2]]);
        $Y = $la->array([-2,1,2]);
        $la->lessEqual($X,$Y);
        $this->assertEquals(
            [[0,1,0],[0,1,1]]
        ,$X->toArray());
    }

    public function testMultiplyNormal()
    {
        $mo = $this->newMatrixOperator();
        $la = $this->newLA($mo);

        // Y := X(i) * Y(i)
        $X = $la->array([[1,2,3],[4,5,6]]);
        $Y = $la->array([[1,10,100],[-1,-10,-100]]);
        $la->multiply($X,$Y);
        $this->assertEquals(
            [[1,2,3],[4,5,6]]
        ,$X->toArray());
        $this->assertEquals(
            [[1,20,300],[-4,-50,-600]]
        ,$Y->toArray());

        // broadcast
        $X = $la->array([1,2,3]);
        $Y = $la->array([[1,10,100],[-1,-10,-100]]);
        $this->assertEquals([[1,10,100],[-1,-10,-100]],$Y->toArray());
        $la->multiply($X,$Y);
        $this->assertEquals(
            [1,2,3]
        ,$X->toArray());
        $this->assertEquals(
            [[1,20,300],[-1,-20,-300]]
        ,$Y->toArray());

        // transpose and broadcast
        $X = $la->array([1,2,3]);
        $Y = $la->array([[1,10],[100,-1],[-10,-100]]);
        $la->multiply($X,$Y,$trans=true);
        $this->assertEquals(
            [1,2,3]
        ,$X->toArray());
        $this->assertEquals(
            [[1,10],[200,-2],[-30,-300]]
        ,$Y->toArray());

    }

    public function testMultiplyLarge()
    {
        $mo = $this->newMatrixOperator();
        $la = $this->newLA($mo);
        if(!$la->accelerated()) {
            $this->markTestSkipped('Skip due to high load');
            return;
        }
        // large size
        $rows = 800000;
        $cols = 16;
        $x = $la->alloc([$rows,$cols],NDArray::float32);
        $la->fill(2.0,$x);
        $y = $la->alloc([$rows,$cols],NDArray::float32);
        $la->fill(3.0,$y);
        $r = $la->multiply($x,$y);
        $trues = $la->alloc([$rows,$cols],NDArray::float32);
        $la->fill(6,$trues);
        $this->assertLessThan(1e-3,$la->amax($la->axpy(
            $trues,$r,-1)));
    }


    public function testMultiplySpeed()
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
        $rows = 1000000;
        $cols = 16;
        $x = $la->alloc([$rows,$cols],NDArray::float32);
        $y = $la->alloc([$rows,$cols],NDArray::float32);
        fwrite(STDERR,"fill-x\n");
        $la->fill(2.0,$x);
        fwrite(STDERR,"fill-y\n");
        $la->fill(3.0,$y);
        fwrite(STDERR,"pre-execute\n");
        $r = $la->multiply($x,$y);
        fwrite(STDERR,"execute\n");
        $start = hrtime(true);
        $r = $la->multiply($x,$y);
        $end = hrtime(true);
        fwrite(STDERR,"done\n");
        echo "\n".(explode(' ',$la->getConfig()))[0].'='.number_format($end-$start)."\n";
        $this->assertTrue(true);
    }

    public function testAdd()
    {
        $mo = $this->newMatrixOperator();
        $la = $this->newLA($mo);

        // Y := X(i) * Y(i)
        $X = $la->array([[1,2,3],[4,5,6]]);
        $Y = $la->array([[1,10,100],[-1,-10,-100]]);
        $la->add($X,$Y);
        $this->assertEquals(
            [[1,2,3],[4,5,6]]
        ,$X->toArray());
        $this->assertEquals(
            [[2,12,103],[3,-5,-94]]
        ,$Y->toArray());

        // broadcast and alpha = -1
        $X = $la->array([1,2,3]);
        $Y = $la->array([[1,10,100],[-1,-10,-100]]);
        $la->add($X,$Y,-1);
        $this->assertEquals(
            [1,2,3]
        ,$X->toArray());
        $this->assertEquals(
            [[0,8,97],[-2,-12,-103]]
        ,$Y->toArray());

        // transpose and broadcast
        $X = $la->array([1,2,3]);
        $Y = $la->array([[1,10],[100,-1],[-10,-100]]);
        $la->add($X,$Y,null,$trans=true);
        $this->assertEquals(
            [1,2,3]
        ,$X->toArray());
        $this->assertEquals(
            [[2,11],[102,1],[-7,-97]]
        ,$Y->toArray());
    }

    public function testAddLarge()
    {
        $mo = $this->newMatrixOperator();
        $la = $this->newLA($mo);
        if(!$la->accelerated()) {
            $this->markTestSkipped('Skip due to high load');
            return;
        }
        // large size
        $rows = 800000;
        $cols = 16;
        $x = $la->alloc([$rows,$cols],NDArray::float32);
        $la->fill(2.0,$x);
        $y = $la->alloc([$rows,$cols],NDArray::float32);
        $la->fill(3.0,$y);
        $r = $la->add($x,$y);
        $trues = $la->alloc([$rows,$cols],NDArray::float32);
        $la->fill(5,$trues);
        $this->assertLessThan(1e-3,$la->amax($la->axpy(
            $trues,$r,-1)));
    }

    public function testSquareNormal()
    {
        $mo = $this->newMatrixOperator();
        $la = $this->newLA($mo);

        // X := X ^ 2
        $X = $la->array([[1,2,3],[4,5,6]]);
        $la->square($X);
        $this->assertEquals(
            [[1,4,9],[16,25,36]]
        ,$X->toArray());
    }

    public function testSquareSpeed()
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

        // X := X ^ 2
        $rows = 1000000;#;
        $cols = 8;
        $X = $la->alloc([$rows,$cols],NDArray::float32);
        $la->fill(2.0,$X);
        fwrite(STDERR,"pre-execute\n");
        $la->square($X);
        fwrite(STDERR,"execute\n");
        $start = hrtime(true);
        $la->square($X);
        $end = hrtime(true);
        fwrite(STDERR,"done\n");
        echo "\n".(explode(' ',$la->getConfig()))[0].'='.number_format($end-$start)."\n";
        $trues = $la->alloc([$rows,$cols],NDArray::float32);
        $la->fill(16.0,$trues);
        $la->axpy($X,$trues,-1);
        $this->assertLessThan(1e-4,$la->asum($trues));
    }

    public function testSqrt()
    {
        $mo = $this->newMatrixOperator();
        $la = $this->newLA($mo);

        // X := sqrt(X)
        $X = $la->array([[1,4,9],[16,25,36]]);
        $la->sqrt($X);
        $this->assertEquals(
            [[1,2,3],[4,5,6]]
        ,$X->toArray());

        // INFINITY & NaN
        $X = $la->array([4,1,0,-1,INF,-INF,NAN]);
        $la->sqrt($X);
        $X = $la->toNDArray($X);
        $this->assertEquals(2.0, $X[0]);
        $this->assertEquals(1.0, $X[1]);
        $this->assertEquals(0.0, $X[2]);
        $this->assertTrue(is_nan($X[3])); // -NAN
        $this->assertTrue(INF==  $X[4]);
        $this->assertTrue(is_nan($X[5]));
        $this->assertTrue(is_nan($X[6]));
    }

    public function testRsqrt()
    {
        $mo = $this->newMatrixOperator();
        $la = $this->newLA($mo);

        // X := 1 / sqrt(X)
        $X = $la->array([[1,4],[16,64]]);
        $la->rsqrt($X);
        $this->assertEquals(
            [[1/1,1/2],[1/4,1/8]]
        ,$X->toArray());

        // X := 1 / ( 1 - sqrt(X))
        $X = $la->array([[10,40],[80,160]]);
        $la->rsqrt($X,1,-1);

        $la->reciprocal($X);
        $la->increment($X,1,-1);
        $la->square($X);

        $this->assertLessThan(1e-4,$la->amax($la->axpy(
            $X,
            $la->array([[10,40],[80,160]]),-1
        )));


        // INFINITY & NaN
        $X = $la->array([4,1,0,-1,INF,-INF,NAN]);
        $la->rsqrt($X);
        $X = $la->toNDArray($X);
        $this->assertEquals(0.5, $X[0]);
        $this->assertEquals(1.0, $X[1]);
        $this->assertTrue(INF==  $X[2]);
        $this->assertTrue(is_nan($X[3])); // -NAN
        $this->assertEquals(0, $X[4]);
        $this->assertTrue(is_nan($X[5]));
        $this->assertTrue(is_nan($X[6]));
    }

    public function testPow()
    {
        $mo = $this->newMatrixOperator();
        $la = $this->newLA($mo);

        // X := sqrt(X)
        $X = $la->array([[1,2,3],[4,5,6]]);
        $la->pow($X,3);
        $X = $la->toNDArray($X);
        $this->assertTrue($mo->la()->isclose($mo->array([[1,8,27],[64,125,216]]),$X));
    }

    public function testExp()
    {
        $mo = $this->newMatrixOperator();
        $la = $this->newLA($mo);

        // X := exp(X)
        $X = $mo->array([[1,2,3],[4,5,6]]);
        $trues = $mo->f(function ($x) { return exp($x);},$X);

        $X = $la->array($X);
        $trues = $la->array($trues);
        $la->exp($X);
        $this->assertLessThan(1e-5,$la->asum($la->axpy($trues,$X,-1)));
    }

    public function testLog()
    {
        $mo = $this->newMatrixOperator();
        $la = $this->newLA($mo);

        // X := log(X)
        $X = $mo->array([[1,2,3],[4,5,6]]);
        $trues = $mo->f(function ($x) { return log($x);},$X);

        $X = $la->array($X);
        $trues = $la->array($trues);
        $la->log($X);
        $this->assertLessThan(1e-5,$la->asum($la->axpy($trues,$X,-1)));

        // INFINITY & NaN
        $X = $la->array([1,0,-1,INF,-INF,NAN]);
        $la->log($X);
        $X = $la->toNDArray($X);
        $this->assertEquals(0.0,  $X[0]);
        $this->assertTrue(-INF==  $X[1]);
        $this->assertTrue(is_nan( $X[2]));
        $this->assertTrue(INF==   $X[3]);
        $this->assertTrue(is_nan( $X[4]));
        $this->assertTrue(is_nan( $X[5]));
    }

    public function testTanh()
    {
        $mo = $this->newMatrixOperator();
        $la = $this->newLA($mo);

        // X := tanh(X)
        $X = $mo->array([[1,2,3],[4,5,6]]);
        $trues = $mo->f(function ($x) { return tanh($x);},$X);

        $X = $la->array($X);
        $trues = $la->array($trues);
        $la->tanh($X);
        $this->assertLessThan(1e-5,$la->asum($la->axpy($trues,$X,-1)));
    }

    public function testSin()
    {
        $mo = $this->newMatrixOperator();
        $la = $this->newLA($mo);

        // X := sin(X)
        $X = $mo->array([[1,2,3],[4,5,6]]);
        $trues = $mo->f(function ($x) { return sin($x);},$X);

        $X = $la->array($X);
        $trues = $la->array($trues);
        $la->sin($X);
        $this->assertLessThan(1e-5,$la->asum($la->axpy($trues,$X,-1)));
    }

    public function testCos()
    {
        $mo = $this->newMatrixOperator();
        $la = $this->newLA($mo);

        // X := cos(X)
        $X = $mo->array([[1,2,3],[4,5,6]]);
        $trues = $mo->f(function ($x) { return cos($x);},$X);

        $X = $la->array($X);
        $trues = $la->array($trues);
        $la->cos($X);
        $this->assertLessThan(1e-5,$la->asum($la->axpy($trues,$X,-1)));
    }

    public function testTan()
    {
        $mo = $this->newMatrixOperator();
        $la = $this->newLA($mo);

        // X := tan(X)
        $X = $mo->array([[1,2,3],[4,5,6]]);
        $trues = $mo->f(function ($x) { return tan($x);},$X);

        $X = $la->array($X);
        $trues = $la->array($trues);
        $la->tan($X);
        $this->assertLessThan(1e-5,$la->asum($la->axpy($trues,$X,-1)));
    }

    public function testDuplicate()
    {
        $mo = $this->newMatrixOperator();
        $la = $this->newLA($mo);

        // Y := X (duplicate 2 times)
        $X = $la->array([[1,2,3],[4,5,6]]);
        $Y = $la->duplicate($X,2);
        $this->assertEquals(
            [[1,2,3],[4,5,6]]
        ,$X->toArray());
        $this->assertEquals([
            [[1,2,3],[4,5,6]],
            [[1,2,3],[4,5,6]],
        ],$Y->toArray());

        // 1 time
        $X = $la->array([[1,2,3],[4,5,6]]);
        $Y = $la->duplicate($X,1);
        $this->assertEquals(
            [[1,2,3],[4,5,6]]
        ,$X->toArray());
        $this->assertEquals(
            [[[1,2,3],[4,5,6]]]
        ,$Y->toArray());

        // transpose
        $X = $la->array([[1,2,3],[4,5,6]]);
        $Y = $la->duplicate($X,2,true);
        $this->assertEquals(
            [[1,2,3],[4,5,6]]
        ,$X->toArray());
        $this->assertEquals([
            [[1,1],[2,2],[3,3]],
            [[4,4],[5,5],[6,6]],
        ],$Y->toArray());
    }

    public function testZeros()
    {
        $mo = $this->newMatrixOperator();
        $la = $this->newLA($mo);
        $x = $la->array([1,2,3]);
        $la->zeros($x);
        $this->assertEquals([0,0,0],$x->toArray());
    }

    public function testSelect()
    {
        $mo = $this->newMatrixOperator();
        $la = $this->newLA($mo);

        $a = $la->array([
            [1,2,3],
            [4,5,6],
            [7,8,9],
            [10,11,12],
        ],NDArray::float32);
        $x = $la->array([0,2],NDArray::int32);
        $y = $la->gather($a,$x);
        $this->assertEquals([[1,2,3],[7,8,9]],$y->toArray());
        $y = $la->gather($a,$x);
        $this->assertEquals([[1,2,3],[7,8,9]],$y->toArray());

        $a = $la->array([
            [1,2,3],
            [4,5,6],
            [7,8,9],
            [10,11,12],
        ]);
        $x = $la->array([0,1,2,0],NDArray::int32);
        $y = $la->gather($a,$x,$axis=1);
        $this->assertEquals([1,5,9,10],$y->toArray());
    }

    public function testSelectAxis0()
    {
        $mo = $this->newMatrixOperator();
        $la = $this->newLA($mo);
        $a = $la->array([
            [1,2,3],
            [4,5,6],
            [7,8,9],
            [10,11,12],
        ],NDArray::float32);
        $x = $la->array([0,2],NDArray::int32);
        $y = $la->gather($a,$x);
        $this->assertEquals([[1,2,3],[7,8,9]],$y->toArray());

        if($la->fp64()) {
            $a = $la->array([
                [1,2,3],
                [4,5,6],
                [7,8,9],
                [10,11,12],
            ],NDArray::float64);
            $x = $la->array([0,2],NDArray::int64);
            $y = $la->gather($a,$x);
            $this->assertEquals([[1,2,3],[7,8,9]],$y->toArray());
        }

        $a = $la->array([
            [1,2,3],
            [4,5,6],
            [7,8,9],
            [10,11,12],
        ],NDArray::int64);
        $x = $la->array([0,2],NDArray::int64);
        $y = $la->gather($a,$x);
        $this->assertEquals([[1,2,3],[7,8,9]],$y->toArray());

        $a = $la->array([
            [1,2,3],
            [4,5,6],
            [7,8,9],
            [10,11,12],
        ],NDArray::uint8);
        $x = $la->array([0,2],NDArray::uint8);
        $y = $la->gather($a,$x);
        $this->assertEquals([[1,2,3],[7,8,9]],$y->toArray());

        $a = $la->array([1,2,3,4],NDArray::float32);
        $x = $la->array([0,2],NDArray::int32);
        $y = $la->gather($a,$x);
        $this->assertEquals([1,3],$y->toArray());

        if($la->fp64()) {
            $a = $la->array([1,2,3,4],NDArray::float64);
            $x = $la->array([0,2],NDArray::int64);
            $y = $la->gather($a,$x);
            $this->assertEquals([1,3],$y->toArray());
        }

        $a = $la->array([1,2,3,4],NDArray::int64);
        $x = $la->array([0,2],NDArray::int64);
        $y = $la->gather($a,$x);
        $this->assertEquals([1,3],$y->toArray());

        $a = $la->array([252,253,254,255],NDArray::uint8);
        $x = $la->array([0,2],NDArray::uint8);
        $y = $la->gather($a,$x);
        $this->assertEquals([252,254],$y->toArray());

        $a = $la->array($mo->full([256],255,NDArray::uint8));
        $x = $la->array([0,2],NDArray::uint8);
        $y = $la->gather($a,$x);
        $a2 = $la->array($mo->full([256],255,NDArray::uint8));
        $x2 = $la->array([0,2],NDArray::uint8);
        $y2 = $la->gather($a2,$x2);

    }

    public function testSelectAxis1()
    {
        $mo = $this->newMatrixOperator();
        $la = $this->newLA($mo);
        $a = $la->array([
            [1,2,3],
            [4,5,6],
            [7,8,9],
            [10,11,12],
        ]);
        $x = $la->array([0,1,2,0],NDArray::int32);
        $y = $la->gather($a,$x,$axis=1);
        $this->assertEquals([1,5,9,10],$y->toArray());

        $x = $la->array([0,1,2,0],NDArray::int64);
        $y = $la->gather($a,$x,$axis=1);
        $this->assertEquals([1,5,9,10],$y->toArray());

        $x = $la->array([0,1,2,0],NDArray::float32);
        $y = $la->gather($a,$x,$axis=1);
        $this->assertEquals([1,5,9,10],$y->toArray());

        if($la->fp64()) {
            $x = $la->array([0,1,2,0],NDArray::float64);
            $y = $la->gather($a,$x,$axis=1);
            $this->assertEquals([1,5,9,10],$y->toArray());
        }
    }

    public function testScatterAxisNullNormal()
    {
        $mo = $this->newMatrixOperator();
        $la = $this->newLA($mo);

        // float32
        // 1D 2D
        $x = $la->array([0,2],NDArray::int64);
        $y = $la->array([[1,2,3],[7,8,9]],NDArray::float32);
        $a = $la->scatter($x,$y,$numClass=4);

        $this->assertEquals([2],$x->shape());
        $this->assertEquals([2,3],$y->shape());
        $this->assertEquals([4,3],$a->shape());

        $this->assertEquals(
           [[1,2,3],
            [0,0,0],
            [7,8,9],
            [0,0,0]],
            $a->toArray()
        );

        // 1D 1D (must be same shape)
        $x = $la->array([0,2],NDArray::int64);
        $y = $la->array([1,2],NDArray::float32);
        $a = $la->scatter($x,$y,$numClass=4);

        $this->assertEquals([2],$x->shape());
        $this->assertEquals([2],$y->shape());
        $this->assertEquals([4],$a->shape());

        $this->assertEquals(
           [1,0,2,0],
            $a->toArray()
        );

        // 2D 3D
        $x = $la->array([[0,1,2],[5,4,3]],NDArray::int64);
        $y = $la->array([[[1,2],[3,4],[5,6]],[[7,8],[9,10],[11,12]]],NDArray::float32);
        $a = $la->scatter($x,$y,$numClass=8);

        $this->assertEquals([2,3],$x->shape());
        $this->assertEquals([2,3,2],$y->shape());
        $this->assertEquals([8,2],$a->shape());
        $this->assertEquals(
           [[1,2],
            [3,4],
            [5,6],
            [11,12],
            [9,10],
            [7,8],
            [0,0],
            [0,0]],
            $a->toArray()
        );

        /// index bit variation

        // float64
        if($la->fp64()) {
            $x = $la->array([0,2],NDArray::int64);
            $y = $la->array([[1,2,3],[7,8,9]],NDArray::float64);
            $a = $la->scatter($x,$y,$numClass=4);
            $this->assertEquals(
                [[1,2,3],
                [0,0,0],
                [7,8,9],
                [0,0,0]],
                $a->toArray()
            );
        }
        // int64
        $x = $la->array([0,2],NDArray::int64);
        $y = $la->array([[1,2,3],[7,8,9]],NDArray::int64);
        $a = $la->scatter($x,$y,$numClass=4);
        $this->assertEquals(
           [[1,2,3],
            [0,0,0],
            [7,8,9],
            [0,0,0]],
            $a->toArray()
        );
        // uint8
        $x = $la->array([0,2],NDArray::int64);
        $y = $la->array([[1,2,3],[7,8,9]],NDArray::uint8);
        $a = $la->scatter($x,$y,$numClass=4);
        $this->assertEquals(
           [[1,2,3],
            [0,0,0],
            [7,8,9],
            [0,0,0]],
            $a->toArray()
        );

        // float32
        $x = $la->array([0,2],NDArray::int64);
        $y = $la->array([1,3],NDArray::float32);
        $a = $la->scatter($x,$y,$numClass=4);
        $this->assertEquals(
           [1,0,3,0],
            $a->toArray()
        );

        // int32
        $x = $la->array([0,2],NDArray::int64);
        $y = $la->array([1,3],NDArray::int32);
        $a = $la->scatter($x,$y,$numClass=4);
        $this->assertEquals(
           [1,0,3,0],
            $a->toArray()
        );

        // float64
        if($la->fp64()) {
            $x = $la->array([0,2],NDArray::int64);
            $y = $la->array([1,3],NDArray::float64);
            $a = $la->scatter($x,$y,$numClass=4);
            $this->assertEquals(
                [1,0,3,0],
                $a->toArray()
            );
        }
        // int64
        $x = $la->array([0,2],NDArray::int64);
        $y = $la->array([1,3],NDArray::int64);
        $a = $la->scatter($x,$y,$numClass=4);
        $this->assertEquals(
           [1,0,3,0],
            $a->toArray()
        );
        // uint8
        $x = $la->array([0,2],NDArray::int64);
        $y = $la->array([252,254],NDArray::uint8);
        $a = $la->scatter($x,$y,$numClass=4);
        $this->assertEquals(
           [252,0,254,0],
            $a->toArray()
        );
        // x=uint8
        $x = $la->array([0,255],NDArray::uint8);
        $y = $la->array([252,254],NDArray::uint8);
        $a = $la->scatter($x,$y,$numClass=256);
        if(is_scalar($a[0])) {
            $this->assertEquals(252,$a[0]);
            $this->assertEquals(254,$a[255]);
        } else {
            $this->assertEquals(252,$a[0]->toArray());
            $this->assertEquals(254,$a[255]->toArray());
        }
    }

    public function testScatterAxisNullSpeed()
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

        echo "small rows\n";
        // small
        $rows = 256;
        $cols = 8;
        $numClass = 8;
        $x = $la->alloc([$rows],NDArray::int32);
        $la->fill(1,$x);
        $y = $la->alloc([$rows,$cols],NDArray::float32);
        $la->fill(1.0,$y);
        $a = $la->alloc([$numClass,$cols],NDArray::float32);
        $la->fill(0.0,$a);
        $la->scatter($x,$y,$numClass,$axis=null,$a);
        $start = hrtime(true);
        $la->scatter($x,$y,$numClass,$axis=null,$a);
        $end = hrtime(true);
        echo (explode(' ',$la->getConfig()))[0].'='.number_format($end-$start)."\n";
        echo "\n";

        echo "medium rows\n";
        // medium
        $rows = 65536;#131072;
        $cols = 8;
        $numClass = 8;
        $x = $la->alloc([$rows],NDArray::int32);
        $la->fill(1,$x);
        $y = $la->alloc([$rows,$cols],NDArray::float32);
        $la->fill(1.0,$y);
        $a = $la->alloc([$numClass,$cols],NDArray::float32);
        $la->fill(0.0,$a);
        $la->scatter($x,$y,$numClass,$axis=null,$a);
        $start = hrtime(true);
        $la->scatter($x,$y,$numClass,$axis=null,$a);
        $end = hrtime(true);
        echo (explode(' ',$la->getConfig()))[0].'='.number_format($end-$start)."\n";
        echo "\n";

        echo "large rows\n";
        // large
        $rows = 1000000;
        $cols = 8;
        $numClass = 8;
        $x = $la->alloc([$rows],NDArray::int32);
        $la->fill(1,$x);
        $y = $la->alloc([$rows,$cols],NDArray::float32);
        $la->fill(1.0,$y);
        $a = $la->alloc([$numClass,$cols],NDArray::float32);
        $la->fill(0.0,$a);
        $la->scatter($x,$y,$numClass,$axis=null,$a);
        $start = hrtime(true);
        $la->scatter($x,$y,$numClass,$axis=null,$a);
        $end = hrtime(true);
        echo (explode(' ',$la->getConfig()))[0].'='.number_format($end-$start)."\n";
        echo "\n";

        echo "medium cols\n";
        // medium cols
        $rows = 8;
        $cols = 65536;
        $numClass = 8;
        $x = $la->alloc([$rows],NDArray::int32);
        $la->fill(1,$x);
        $y = $la->alloc([$rows,$cols],NDArray::float32);
        $la->fill(1.0,$y);
        $a = $la->alloc([$numClass,$cols],NDArray::float32);
        $la->fill(0.0,$a);
        $la->scatter($x,$y,$numClass,$axis=null,$a);
        $start = hrtime(true);
        $la->scatter($x,$y,$numClass,$axis=null,$a);
        $end = hrtime(true);
        echo (explode(' ',$la->getConfig()))[0].'='.number_format($end-$start)."\n";
        echo "\n";

        echo "large cols\n";
        // large cols
        $rows = 8;
        $cols = 1000000;
        $numClass = 8;
        $x = $la->alloc([$rows],NDArray::int32);
        $la->fill(1,$x);
        $y = $la->alloc([$rows,$cols],NDArray::float32);
        $la->fill(1.0,$y);
        $a = $la->alloc([$numClass,$cols],NDArray::float32);
        $la->fill(0.0,$a);
        $la->scatter($x,$y,$numClass,$axis=null,$a);
        $start = hrtime(true);
        $la->scatter($x,$y,$numClass,$axis=null,$a);
        $end = hrtime(true);
        echo (explode(' ',$la->getConfig()))[0].'='.number_format($end-$start)."\n";
        echo "\n";

        echo "medium class\n";
        // medium class
        $rows = 8;
        $cols = 8;
        $numClass = 131072;
        $x = $la->alloc([$rows],NDArray::int32);
        $la->fill(1,$x);
        $y = $la->alloc([$rows,$cols],NDArray::float32);
        $la->fill(1.0,$y);
        $a = $la->alloc([$numClass,$cols],NDArray::float32);
        $la->fill(0.0,$a);
        $la->scatter($x,$y,$numClass,$axis=null,$a);
        $start = hrtime(true);
        $la->scatter($x,$y,$numClass,$axis=null,$a);
        $end = hrtime(true);
        echo (explode(' ',$la->getConfig()))[0].'='.number_format($end-$start)."\n";
        echo "\n";

        echo "large class\n";
        // large class
        $rows = 8;
        $cols = 8;
        $numClass = 1000000;
        $x = $la->alloc([$rows],NDArray::int32);
        $la->fill(1,$x);
        $y = $la->alloc([$rows,$cols],NDArray::float32);
        $la->fill(1.0,$y);
        $a = $la->alloc([$numClass,$cols],NDArray::float32);
        $la->fill(0.0,$a);
        $la->scatter($x,$y,$numClass,$axis=null,$a);
        $start = hrtime(true);
        $la->scatter($x,$y,$numClass,$axis=null,$a);
        $end = hrtime(true);
        echo (explode(' ',$la->getConfig()))[0].'='.number_format($end-$start)."\n";
        echo "\n";

        //echo "mode4\n";
        // small
        //
        //$rows = 131072;
        //$cols = 8;
        //$numClass = 8;
        //$x = $la->alloc([$rows],NDArray::int32);
        //$la->fill(1,$x);
        //$y = $la->alloc([$rows,$cols],NDArray::float32);
        //$la->fill(1.0,$y);
        //$a = $la->alloc([$numClass,$cols],NDArray::float32);
        //$la->fill(0.0,$a);
        //$la->scatterTest($x,$y,$numClass,$axis=0,$a,null,null,$mode=4);
        //$start = hrtime(true);
        //$la->scatterTest($x,$y,$numClass,$axis=0,$a,null,null,$mode=4);
        //$end = hrtime(true);
        //echo (explode(' ',$la->getConfig()))[0].'='.number_format($end-$start)."\n";
        //echo "\n";

        $this->assertTrue(true);
    }

    public function testScatterAxis1()
    {
        $mo = $this->newMatrixOperator();
        $la = $this->newLA($mo);

        // 1D 1D
        $x = $la->array([0,1,2,0],NDArray::int32);
        $y = $la->array([1,5,9,10],NDArray::float32);
        $a = $la->scatter($x,$y,$numClass=3,$axis=1);
        //foreach($a->toArray() as $pr) {
        //    echo "\n";
        //    foreach($pr as $value) {
        //        echo $value.",";
        //    }
        //}
        //echo "\n";
        //$this->assertTrue(false);
        $this->assertEquals([4],$x->shape());
        $this->assertEquals([4],$y->shape());
        $this->assertEquals([4,3],$a->shape());

        $this->assertEquals(
           [[1,0,0],
            [0,5,0],
            [0,0,9],
            [10,0,0]],
            $a->toArray());

        // 1D 2D is unmatched shapes
        // 2D 1D is unmatched shapes

        // 2D 2D
        $x = $la->array([[0,1,2],[2,1,0]],NDArray::int32);
        $y = $la->array([[1,2,3],[4,5,6]],NDArray::float32);
        $a = $la->scatter($x,$y,$numClass=4,$axis=1);

        $this->assertEquals([2,3],$x->shape());
        $this->assertEquals([2,3],$y->shape());
        $this->assertEquals([2,4,3],$a->shape());
        $this->assertEquals(
          [[[1,0,0],
            [0,2,0],
            [0,0,3],
            [0,0,0]],
           [[0,0,6],
            [0,5,0],
            [4,0,0],
            [0,0,0]]],
            $a->toArray());

        /// index bit variation
        $x = $la->array([0,1,2,0],NDArray::int64);
        $y = $la->array([1,5,9,10],NDArray::float32);
        $a = $la->scatter($x,$y,$numClass=3,$axis=1);
        $this->assertEquals(
           [[1,0,0],
            [0,5,0],
            [0,0,9],
            [10,0,0]],
            $a->toArray());

        $x = $la->array([0,1,2,0],NDArray:: float32);
        $a = $la->scatter($x,$y,$numClass=3,$axis=1);
        $this->assertEquals(
           [[1,0,0],
            [0,5,0],
            [0,0,9],
            [10,0,0]],
            $a->toArray());

        if($la->fp64()) {
            $x = $la->array([0,1,2,0],NDArray:: float64);
            $a = $la->scatter($x,$y,$numClass=3,$axis=1);
            $this->assertEquals(
                [[1,0,0],
                [0,5,0],
                [0,0,9],
                [10,0,0]],
                $a->toArray());
        }

    }

    public function testScatterAxis0()
    {
        $mo = $this->newMatrixOperator();
        $la = $this->newLA($mo);

        // 1D 1D
        $x = $la->array([0,1,2,0],NDArray::int32);
        $y = $la->array([1,5,9,10],NDArray::float32);
        $a = $la->scatter($x,$y,$numClass=3,$axis=0);

        $this->assertEquals([4],$x->shape());
        $this->assertEquals([4],$y->shape());
        $this->assertEquals([3,4],$a->shape());

        $this->assertEquals(
           [[1, 0, 0,10],
            [0, 5, 0, 0],
            [0, 0, 9, 0]],
            $a->toArray());

        // 1D 2D is unmatched shapes
        // 2D 1D is unmatched shapes

        // 2D 2D
        $x = $la->array([[0,1,2],[2,1,0]],NDArray::int32);
        $y = $la->array([[1,2,3],[4,5,6]],NDArray::float32);
        $a = $la->scatter($x,$y,$numClass=4,$axis=0);

        $this->assertEquals([2,3],$x->shape());
        $this->assertEquals([2,3],$y->shape());
        $this->assertEquals([4,2,3],$a->shape());
        $this->assertEquals(
          [[[1,0,0],
            [0,0,6]],
           [[0,2,0],
            [0,5,0]],
           [[0,0,3],
            [4,0,0]],
           [[0,0,0],
            [0,0,0]]],
            $a->toArray());
    }

    public function testScatterAxis2()
    {
        $mo = $this->newMatrixOperator();
        $la = $this->newLA($mo);

        // 1D 1D is unmatched shapes
        // 1D 2D is unmatched shapes
        // 2D 1D is unmatched shapes

        // 2D 2D
        $x = $la->array([[0,1,2],[2,1,0]],NDArray::int32);
        $y = $la->array([[1,2,3],[4,5,6]],NDArray::float32);
        $a = $la->scatter($x,$y,$numClass=4,$axis=2);

        $this->assertEquals([2,3],$x->shape());
        $this->assertEquals([2,3],$y->shape());
        $this->assertEquals([2,3,4],$a->shape());

        $this->assertEquals(
          [[[1,0,0,0],
            [0,2,0,0],
            [0,0,3,0]],
           [[0,0,4,0],
            [0,5,0,0],
            [6,0,0,0]]],
            $a->toArray());
    }

    public function testGatherNormal()
    {
        $mo = $this->newMatrixOperator();
        $la = $this->newLA($mo);

        // 1D by 1D
        $x = $la->array([3,2,1,1],NDArray::int32);
        $a = $la->array([10,11,12,13,14,15,16,17,18,19]);
        $b = $la->gather($a,$x);
        $this->assertEquals([4],$x->shape());
        $this->assertEquals([10],$a->shape());
        $this->assertEquals([4],$b->shape()); // replace axis0
        $trues = $la->array([13,12,11,11]);
        //echo $mo->toString($b,null,true)."\n";
        $mo_gather = $mo->select($this->ndarray($a),$this->ndarray($x));
        //echo $mo->toString($mo_gather,null,true)."\n";
        $this->assertEquals($mo_gather->toArray(),$b->toArray());
        $this->assertEquals($trues->toArray(),$b->toArray());

        // 1D by 2D
        $x = $la->array([[3,2,1],[1,2,3]],NDArray::int32);
        $a = $la->array([10,11,12,13,14,15,16,17,18,19]);
        $b = $la->gather($a,$x);
        $this->assertEquals([2,3],$x->shape());
        $this->assertEquals([10],$a->shape());
        $this->assertEquals([2,3],$b->shape()); // replace axis0
        $trues = $la->array([[13,12,11],[11,12,13]]);
        //echo $mo->toString($b,null,true)."\n";
        $mo_gather = $mo->select($this->ndarray($a),$this->ndarray($x));
        //echo $mo->toString($mo_gather,null,true)."\n";
        $this->assertEquals($mo_gather->toArray(),$b->toArray());
        $this->assertEquals($trues->toArray(),$b->toArray());

        // 2D by 1D
        $x = $la->array([3,1],NDArray::int32);
        $a = $la->array([
            [ 0, 0, 3],
            [ 0, 0, 4],
            [ 0, 2, 0],
            [ 1, 0, 0]]);
        $b = $la->gather($a,$x);
        $this->assertEquals([2],$x->shape());
        $this->assertEquals([4,3],$a->shape());
        $this->assertEquals([2,3],$b->shape()); // replace axis0
        $trues = $la->array([
            [1,0,0],
            [0,0,4]
        ]);
        //echo $mo->toString($b,null,true)."\n";
        $mo_gather = $mo->select($this->ndarray($a),$this->ndarray($x));
        //echo $mo->toString($mo_gather,null,true)."\n";
        $this->assertEquals($mo_gather->toArray(),$b->toArray());
        $this->assertEquals($trues->toArray(),$b->toArray());

        // 2D by 2D
        $x = $la->array([
            [2, 1, 0],
            [1, 2, 3],
        ],NDArray::int32);
        $a = $la->array([
            [ 1, 0, 0],
            [ 0, 2, 0],
            [ 0, 0, 3],
            [ 4, 0, 0]]);
        $b = $la->gather($a,$x);
        $this->assertEquals([2,3],$x->shape());
        $this->assertEquals([4,3],$a->shape());
        $this->assertEquals([2,3,3],$b->shape()); // replace axis0
        $trues = $la->array([
            [[0,0,3],
             [0,2,0],
             [1,0,0]],
            [[0,2,0],
             [0,0,3],
             [4,0,0]],
        ]);
        //echo $mo->toString($b,null,true)."\n";
        $mo_gather = $mo->select($this->ndarray($a),$this->ndarray($x));
        //echo $mo->toString($mo_gather,null,true)."\n";
        $this->assertEquals($mo_gather->toArray(),$b->toArray());
        $this->assertEquals($trues->toArray(),$b->toArray());
    }

    public function testGatherReduction()
    {
        $mo = $this->newMatrixOperator();
        $la = $this->newLA($mo);
        // axis = 0
        // 1D indices
        $x = $la->array([3,2,1],NDArray::int32);
        $a = $la->array([
            [ 0, 0, 3],
            [ 0, 0, 4],
            [ 0, 2, 0],
            [ 1, 0, 0]]);
        $b = $la->gather($a,$x,$axis=0);
        $this->assertEquals([3],$x->shape());
        $this->assertEquals([4,3],$a->shape());
        $this->assertEquals([3],$b->shape()); // reduction axis0
        $trues = $la->array([1,2,4]);
        //echo $mo->toString($b,null,true)."\n";
        $max = $la->reduceMax($a,$axis=0);
        //echo $mo->toString($max,null,true)."\n";
        $max = $la->reduceArgMax($a,$axis=0);
        //echo $mo->toString($max,null,true)."\n";
        $this->assertEquals($max->toArray(),$x->toArray());
        $max = $la->reduceMax($a,$axis=0);
        $this->assertEquals($max->toArray(),$trues->toArray());
        $this->assertEquals($trues->toArray(),$b->toArray());

        // axis = 0
        // 2D indices
        $x = $la->array([
            [0,2,2,1],
            [2,0,1,2],
        ],NDArray::int32);
        $a = $la->array([
            [[ 1, 0, 0, 0],
             [ 0, 6, 0, 0]],
            [[ 0, 0, 0, 4],
             [ 0, 0, 7, 0]],
            [[ 0, 2, 3, 0],
             [ 5, 0, 0, 8]],
        ]);
        $b = $la->gather($a,$x,$axis=0);
        $this->assertEquals([2,4],$x->shape());
        $this->assertEquals([3,2,4],$a->shape());
        $this->assertEquals([2,4],$b->shape()); // reduction axis0
        $trues = $la->array([
            [1,2,3,4],
            [5,6,7,8],
        ]);
        //echo $mo->toString($b,null,true)."\n";
        $max = $la->reduceMax($a,$axis=0);
        //echo $mo->toString($max,null,true)."\n";
        $max = $la->reduceArgMax($a,$axis=0);
        //echo $mo->toString($max,null,true)."\n";
        $this->assertEquals($max->toArray(),$x->toArray());
        $max = $la->reduceMax($a,$axis=0);
        $this->assertEquals($max->toArray(),$trues->toArray());
        $this->assertEquals($trues->toArray(),$b->toArray());

        // axis = 1
        // 1D indices
        $x = $la->array([2,2,1,0],NDArray::int32);
        $a = $la->array([
            [ 0, 0, 3],
            [ 0, 0, 4],
            [ 0, 2, 0],
            [ 1, 0, 0]]);
        $b = $la->gather($a,$x,$axis=1);
        $this->assertEquals([4],$x->shape());
        $this->assertEquals([4,3],$a->shape());
        $this->assertEquals([4],$b->shape()); // reduction axis1
        $trues = $la->array([3,4,2,1]);
        //echo $mo->toString($b,null,true)."\n";
        $max = $la->reduceMax($a,$axis=1);
        //echo $mo->toString($max,null,true)."\n";
        $max = $la->reduceArgMax($a,$axis=1);
        //echo $mo->toString($max,null,true)."\n";
        $this->assertEquals($max->toArray(),$x->toArray());
        $max = $la->reduceMax($a,$axis=1);
        $this->assertEquals($max->toArray(),$trues->toArray());
        $this->assertEquals($trues->toArray(),$b->toArray());

        // axis = 1
        // 2D indices
        $x = $la->array([
            [1,0,1,0],
            [1,0,1,0],
            [1,0,1,0],
        ],NDArray::int32);
        $a = $la->array([
            [[ 0, 2, 0, 4],
             [ 1, 0, 3, 0]],
            [[ 0, 6, 0, 8],
             [ 5, 0, 7, 0]],
            [[ 0,10, 0,12],
             [ 9, 0,11, 0]],
        ]);
        $b = $la->gather($a,$x,$axis=1);
        $this->assertEquals([3,4],$x->shape());
        $this->assertEquals([3,2,4],$a->shape());
        $this->assertEquals([3,4],$b->shape()); // reduction axis1
        $trues = $la->array([
            [1,2,3,4],
            [5,6,7,8],
            [9,10,11,12]
        ]);
        //echo $mo->toString($b,null,true)."\n";
        $max = $la->reduceMax($a,$axis=1);
        //echo $mo->toString($max,null,true)."\n";
        $max = $la->reduceArgMax($a,$axis=1);
        //echo $mo->toString($max,null,true)."\n";
        $this->assertEquals($max->toArray(),$x->toArray());
        $max = $la->reduceMax($a,$axis=1);
        $this->assertEquals($max->toArray(),$trues->toArray());
        $this->assertEquals($trues->toArray(),$b->toArray());
    }

    public function testScatterExNormal()
    {
        $mo = $this->newMatrixOperator();
        $la = $this->newLA($mo);

        // 1D by 1D
        $x = $la->array([3,2,1,1],NDArray::int32);
        $a = $la->array([13,12,11,11]);
        $b = $la->scatter($x,$a,$numClass=10);
        $this->assertEquals([4],$x->shape());
        $this->assertEquals([4],$a->shape());
        $this->assertEquals([10],$b->shape()); // replace axis0
        $trues = $la->array([0,11,12,13,0,0,0,0,0,0]);
        //echo $mo->toString($b,null,true)."\n";
        $mo_gather = $mo->select($this->ndarray($b),$this->ndarray($x));
        //echo $mo->toString($mo_gather,null,true)."\n";
        $this->assertEquals($mo_gather->toArray(),$a->toArray());
        $this->assertEquals($trues->toArray(),$b->toArray());

        // 1D by 2D
        $x = $la->array([[3,2,1],[5,6,7]],NDArray::int32);
        $a = $la->array([[13,12,11],[15,16,17]]);
        $b = $la->scatter($x,$a,$numClass=10);
        $this->assertEquals([2,3],$x->shape());
        $this->assertEquals([2,3],$a->shape());
        $this->assertEquals([10],$b->shape()); // replace axis0
        $trues = $la->array([0,11,12,13,0,15,16,17,0,0]);
        //echo $mo->toString($b,null,true)."\n";
        $mo_gather = $mo->select($this->ndarray($b),$this->ndarray($x));
        //echo $mo->toString($mo_gather,null,true)."\n";
        $this->assertEquals($mo_gather->toArray(),$a->toArray());
        $this->assertEquals($trues->toArray(),$b->toArray());

        // 2D by 1D
        $x = $la->array([3,1],NDArray::int32);
        $a = $la->array([
            [1,2,3],
            [2,3,4]
        ]);
        $b = $la->scatter($x,$a,$numClass=4);
        $this->assertEquals([2],$x->shape());
        $this->assertEquals([2,3],$a->shape());
        $this->assertEquals([4,3],$b->shape()); // replace axis0
        $trues = $la->array([
            [ 0, 0, 0],
            [ 2, 3, 4],
            [ 0, 0, 0],
            [ 1, 2, 3]
        ]);
        //echo $mo->toString($b,null,true)."\n";
        $mo_gather = $mo->select($this->ndarray($b),$this->ndarray($x));
        //echo $mo->toString($mo_gather,null,true)."\n";
        $this->assertEquals($mo_gather->toArray(),$a->toArray());
        $this->assertEquals($trues->toArray(),$b->toArray());

        // 2D by 2D
        $x = $la->array([
            [2, 1, 0],
            [1, 2, 3],
        ],NDArray::int32);
        $a = $la->array([
            [[0,0,3],
             [0,2,0],
             [1,0,0]],
            [[0,2,0],
             [0,0,3],
             [4,0,0]],
        ]);
        $b = $la->scatter($x,$a,$numClass=4);
        $this->assertEquals([2,3],$x->shape());
        $this->assertEquals([2,3,3],$a->shape());
        $this->assertEquals([4,3],$b->shape()); // replace axis0
        $trues = $la->array([
            [ 1, 0, 0],
            [ 0, 2, 0],
            [ 0, 0, 3],
            [ 4, 0, 0]
        ]);
        //echo $mo->toString($b,null,true)."\n";
        $mo_gather = $mo->select($this->ndarray($b),$this->ndarray($x));
        //echo $mo->toString($mo_gather,null,true)."\n";
        $this->assertEquals($mo_gather->toArray(),$a->toArray());
        $this->assertEquals($trues->toArray(),$b->toArray());
    }

    public function testScatterExExpandDims()
    {
        $mo = $this->newMatrixOperator();
        $la = $this->newLA($mo);

        //
        // axis = 0
        //
        //  1D inputs
        $x = $la->array([3,2,0],NDArray::int32);
        $a = $la->array([1,2,3],NDArray::float32);
        $b = $la->scatter($x,$a,$numClass=4,$axis=0);
        $this->assertEquals([3],$x->shape());
        $this->assertEquals([3],$a->shape());
        $this->assertEquals([4,3],$b->shape()); // insert axis0
        $trues = $la->array([
            [ 0, 0, 3],
            [ 0, 0, 0],
            [ 0, 2, 0],
            [ 1, 0, 0]]);
        $max = $la->reduceArgMax($trues,$axis=0);
        $this->assertEquals($max->toArray(),$x->toArray());
        $max = $la->reduceMax($trues,$axis=0);
        $this->assertEquals($max->toArray(),$a->toArray());
        $this->assertEquals($trues->toArray(),$b->toArray());

        //  2D inputs
        $x = $la->array([
            [0,2,2,1],
            [2,0,1,2],
        ]);
        $a = $la->array([
            [1,2,3,4],
            [5,6,7,8],
        ]);
        $b = $la->scatter($x,$a,$numClass=3,$axis=0);
        $this->assertEquals([2,4],$x->shape());
        $this->assertEquals([2,4],$a->shape());
        $this->assertEquals([3,2,4],$b->shape()); // insert axis0
        $trues = $la->array([
            [[ 1, 0, 0, 0],
             [ 0, 6, 0, 0]],
            [[ 0, 0, 0, 4],
             [ 0, 0, 7, 0]],
            [[ 0, 2, 3, 0],
             [ 5, 0, 0, 8]],
        ]);
        $max = $la->reduceArgMax($trues,$axis=0);
        //echo $mo->toString($max,null,true);
        $this->assertEquals($max->toArray(),$x->toArray());
        $max = $la->reduceMax($trues,$axis=0);
        //echo $mo->toString($max,null,true);
        $this->assertEquals($max->toArray(),$a->toArray());
        $this->assertEquals($trues->toArray(),$b->toArray());

        //
        // axis = 1
        //
        //  1D inputs
        $x = $la->array([0,1,2,0],NDArray::int32);
        $a = $la->array([1,5,9,10],NDArray::float32);
        $b = $la->scatter($x,$a,$numClass=3,$axis=1);
        $this->assertEquals([4],$x->shape());
        $this->assertEquals([4],$a->shape());
        $this->assertEquals([4,3],$b->shape()); // insert axis1
        $trues = $la->array([
            [1,0,0],
            [0,5,0],
            [0,0,9],
            [10,0,0]]);
        $max = $la->reduceArgMax($trues,$axis=1);
        $this->assertEquals($max->toArray(),$x->toArray());
        $max = $la->reduceMax($trues,$axis=1);
        $this->assertEquals($max->toArray(),$a->toArray());
        $this->assertEquals($trues->toArray(),$b->toArray());

        //  2D inputs
        $x = $la->array([
            [1,0,1,0],
            [1,0,1,0],
            [1,0,1,0],
        ]);
        $a = $la->array([
            [1,2,3,4],
            [5,6,7,8],
            [9,10,11,12]
        ]);
        $b = $la->scatter($x,$a,$numClass=2,$axis=1);
        $this->assertEquals([3,4],$x->shape());
        $this->assertEquals([3,4],$a->shape());
        $this->assertEquals([3,2,4],$b->shape()); // insert axis1
        $trues = $la->array([
            [[0,2,0,4],
             [1,0,3,0]],
            [[0,6,0,8],
             [5,0,7,0]],
            [[ 0,10, 0,12],
             [ 9, 0,11, 0]],
        ]);
        $max = $la->reduceArgMax($trues,$axis=1);
        $this->assertEquals($max->toArray(),$x->toArray());
        $max = $la->reduceMax($trues,$axis=1);
        $this->assertEquals($max->toArray(),$a->toArray());
        $this->assertEquals($trues->toArray(),$b->toArray());

        //  3D inputs
        $x = $la->array([
            [[1,0],[1,0]],
            [[1,0],[1,0]],
            [[1,0],[1,0]],
        ]);
        $a = $la->array([
            [[1,2],[3,4]],
            [[5,6],[7,8]],
            [[9,10],[11,12]],
        ]);
        $b = $la->scatter($x,$a,$numClass=2,$axis=1);
        $this->assertEquals([3,2,2],$x->shape());
        $this->assertEquals([3,2,2],$a->shape());
        $this->assertEquals([3,2,2,2],$b->shape()); // insert axis1
        $trues = $la->array([
            [[[0,2],[0,4]],
             [[1,0],[3,0]]],
            [[[0,6],[0,8]],
             [[5,0],[7,0]]],
            [[[0,10],[0,12]],
             [[9, 0],[11,0]]],
        ]);
        $max = $la->reduceArgMax($trues,$axis=1);
        $this->assertEquals($max->toArray(),$x->toArray());
        $max = $la->reduceMax($trues,$axis=1);
        $this->assertEquals($max->toArray(),$a->toArray());
        $this->assertEquals($trues->toArray(),$b->toArray());

/*
        $x = $la->array([0,1,2,0],NDArray::int32);
        $y = $la->array([
            [1,2],
            [5,6],
            [9,10],
            [11,12]
        ],NDArray::float32);
        $a = $la->scatter($x,$y,$numClass=3,$axis=1);
        $trues = $la->array([
            [1,0,0],
            [0,5,0],
            [0,0,9],
            [10,0,0]]);
        $this->assertEquals($trues->toArray(),$a->toArray());
*/

        $y = $la->array([1,5,9,10],NDArray::float32);
        $x = $la->array([0,1,2,0],NDArray::int64);
        $a = $la->scatter($x,$y,$numClass=3,$axis=1);
        $this->assertEquals(
           [[1,0,0],
            [0,5,0],
            [0,0,9],
            [10,0,0]],
            $a->toArray());

        $x = $la->array([0,1,2,0],NDArray:: float32);
        $a = $la->scatter($x,$y,$numClass=3,$axis=1);
        $this->assertEquals(
           [[1,0,0],
            [0,5,0],
            [0,0,9],
            [10,0,0]],
            $a->toArray());

        if($la->fp64()) {
            $x = $la->array([0,1,2,0],NDArray:: float64);
            $a = $la->scatter($x,$y,$numClass=3,$axis=1);
            $this->assertEquals(
                [[1,0,0],
                [0,5,0],
                [0,0,9],
                [10,0,0]],
                $a->toArray());
        }

    }

    public function testScatterAddExNormal()
    {
        $mo = $this->newMatrixOperator();
        $la = $this->newLA($mo);

        // 1D by 1D
        $x = $la->array([3,2,1,1],NDArray::int32);
        $a = $la->array([13,12,11,11]);
        $b = $la->alloc([10]);
        $la->ones($b);
        $la->scatterAdd($x,$a,$b);
        $this->assertEquals([4],$x->shape());
        $this->assertEquals([4],$a->shape());
        $this->assertEquals([10],$b->shape()); // replace axis0
        $trues = $la->array([1,23,13,14,1,1,1,1,1,1]);
        //echo $mo->toString($b,null,true)."\n";
        $this->assertEquals($trues->toArray(),$b->toArray());
    }

    public function testScatterAddExExpandDims()
    {
        $mo = $this->newMatrixOperator();
        $la = $this->newLA($mo);

        //
        // axis = 0
        //
        //  1D inputs
        $x = $la->array([3,2,0],NDArray::int32);
        $a = $la->array([1,2,3],NDArray::float32);
        $b = $la->alloc([4,3]);
        $la->ones($b);
        $la->scatterAdd($x,$a,$b,$axis=0);
        $this->assertEquals([3],$x->shape());
        $this->assertEquals([3],$a->shape());
        $this->assertEquals([4,3],$b->shape()); // insert axis0
        $trues = $la->array([
            [ 1, 1, 4],
            [ 1, 1, 1],
            [ 1, 3, 1],
            [ 2, 1, 1]]);
        $this->assertEquals($trues->toArray(),$b->toArray());
    }



    public function testScatterAddAxis0Normal()
    {
        $mo = $this->newMatrixOperator();
        $la = $this->newLA($mo);
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

    public function testScatterAddLarge()
    {
        $mo = $this->newMatrixOperator();
        $la = $this->newLA($mo);
        if(!$la->accelerated()) {
            $this->markTestSkipped('Skip due to high load');
            return;
        }
        // medium size
        $rows = 8;
        $cols = 8;
        $numClass = 65536;
        $x = $la->alloc([$rows],NDArray::int32);
        $la->fill(1,$x);
        $y = $la->alloc([$rows,$cols],NDArray::float32);
        $la->fill(1.0,$y);
        $a = $la->alloc([$numClass,$cols],NDArray::float32);
        $la->fill(0.0,$a);
        $la->scatterAdd($x,$y,$a);
        $trues = $la->alloc([$numClass,$cols],NDArray::float32);
        $la->fill($rows,$trues);
        $this->assertLessThan(1e-3,$la->amax($la->axpy(
            $trues,$a,-1)));
        // large size
        $rows = 8;
        $cols = 8;
        $numClass = 1000000;
        $x = $la->alloc([$rows],NDArray::int32);
        $la->fill(1,$x);
        $y = $la->alloc([$rows,$cols],NDArray::float32);
        $la->fill(1.0,$y);
        $a = $la->alloc([$numClass,$cols],NDArray::float32);
        $la->fill(0.0,$a);
        $la->scatterAdd($x,$y,$a);
        $trues = $la->alloc([$numClass,$cols],NDArray::float32);
        $la->fill($rows,$trues);
        $this->assertLessThan(1e-3,$la->amax($la->axpy(
            $trues,$a,-1)));
    }

    public function testScatterAddAxis0Speed()
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

        echo "small rows\n";
        // small
        $rows = 256;
        $cols = 8;
        $numClass = 8;
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
        echo "\n";

        echo "medium rows\n";
        // medium
        $rows = 65536;#131072;
        $cols = 8;
        $numClass = 8;
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
        echo "\n";

        echo "large rows\n";
        // large
        $rows = 1000000;
        $cols = 8;
        $numClass = 8;
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
        echo "\n";

        echo "medium cols\n";
        // medium cols
        $rows = 8;
        $cols = 65536;
        $numClass = 8;
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
        echo "\n";

        echo "large cols\n";
        // large cols
        $rows = 8;
        $cols = 1000000;
        $numClass = 8;
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
        echo "\n";

        echo "medium class\n";
        // medium class
        $rows = 8;
        $cols = 8;
        $numClass = 131072;
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
        echo "\n";

        echo "large class\n";
        // large class
        $rows = 8;
        $cols = 8;
        $numClass = 1000000;
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
        echo "\n";

        //echo "mode4\n";
        // small
        //
        //$rows = 131072;
        //$cols = 8;
        //$numClass = 8;
        //$x = $la->alloc([$rows],NDArray::int32);
        //$la->fill(1,$x);
        //$y = $la->alloc([$rows,$cols],NDArray::float32);
        //$la->fill(1.0,$y);
        //$a = $la->alloc([$numClass,$cols],NDArray::float32);
        //$la->fill(0.0,$a);
        //$la->scatterAddTest($x,$y,$a,$axis=0,null,null,$mode=4);
        //$start = hrtime(true);
        //$la->scatterAddTest($x,$y,$a,$axis=0,null,null,$mode=4);
        //$end = hrtime(true);
        //echo (explode(' ',$la->getConfig()))[0].'='.number_format($end-$start)."\n";
        //echo "\n";


        $this->assertTrue(true);
    }

    public function testScatterAddAxis1()
    {
        $mo = $this->newMatrixOperator();
        $la = $this->newLA($mo);
        $x = $la->array([0,1,2,0],NDArray::int32);
        $y = $la->array([1,5,9,10],NDArray::float32);
        $a = $la->array($mo->ones([4,3],NDArray::float32));
        $la->scatterAdd($x,$y,$a,$axis=1);
        $this->assertEquals(
           [[2,1,1],
            [1,6,1],
            [1,1,10],
            [11,1,1]],
            $a->toArray());

        if($la->fp64()) {
            $x = $la->array([0,1,2,0],NDArray::int32);
            $y = $la->array([1,5,9,10],NDArray::float64);
            $a = $la->array($mo->ones([4,3],NDArray::float64));
            $la->scatterAdd($x,$y,$a,$axis=1);
            $this->assertEquals(
                [[2,1,1],
                [1,6,1],
                [1,1,10],
                [11,1,1]],
                $a->toArray());
        }
    }

    public function testOnehot()
    {
        $mo = $this->newMatrixOperator();
        $la = $this->newLA($mo);
        $x = $la->array([0,1,2,0]);

        $this->assertEquals([
            [1,0,0],
            [0,1,0],
            [0,0,1],
            [1,0,0],
        ],$la->onehot($x,3)->toArray());

        $y = $la->array($mo->ones([4,3]));
        $this->assertEquals([
            [-1, 1, 1],
            [ 1,-1, 1],
            [ 1, 1,-1],
            [-1, 1, 1],
        ],$la->onehot($x,3,-2,$y)->toArray());
    }
/*
    public function testReduceSumOLDNormal()
    {
        $mo = $this->newMatrixOperator();
        $la = $this->newLA($mo);
        $x = $la->array([[1,2,3],[4,5,6]]);
        $y = $la->reduceSum($x,$axis=0);
        $this->assertEquals([5,7,9],$y->toArray());
        $y = $la->reduceSum($x,$axis=1);
        $this->assertEquals([6,15],$y->toArray());

        // ***** CAUTION ******
        // 3d array as 2d array
        $x = $la->array([[[1,2],[3,4]],[[5,6],[7,8]]]);
        $y = $la->reduceSum($x,$axis=0);
        $this->assertEquals([6,8,10,12],$y->toArray());
        $x = $la->array([[[1,2],[3,4]],[[5,6],[7,8]]]);
        $y = $la->reduceSum($x,$axis=1);
        $this->assertEquals([3,7,11,15],$y->toArray());

        // with offset
        $x = $la->array([[[9,9,9],[9,9,9]],[[1,2,3],[4,5,6]]]);
        $x = $x[1];
        $this->assertEquals([5,7,9],$la->reduceSum($x,$axis=0)->toArray());
        $this->assertEquals([6,15],$la->reduceSum($x,$axis=1)->toArray());
    }
*/
    public function testReduceSumNormal()
    {
        $mo = $this->newMatrixOperator();
        $la = $this->newLA($mo);
        $x = $la->array([[1,2,3],[4,5,6]]);
        $y = $la->reduceSum($x,$axis=0);
        $this->assertEquals([5,7,9],$y->toArray());
        $y = $la->reduceSum($x,$axis=1);
        $this->assertEquals([6,15],$y->toArray());
        $y = $la->reduceSum($x,$axis=-1);
        $this->assertEquals([6,15],$y->toArray());
        $y = $la->reduceSum($x,$axis=null); // ** CAUTION null is 0
        $this->assertEquals([5,7,9],$y->toArray());

        // 3d array
        $x = $la->array([[[1,2],[3,4]],[[5,6],[7,8]]]);
        $y = $la->reduceSum($x,$axis=0);
        $this->assertEquals([[6,8],[10,12]],$y->toArray());
        $x = $la->array([[[1,2],[3,4]],[[5,6],[7,8]]]);
        $y = $la->reduceSum($x,$axis=1);
        $this->assertEquals([[4,6],[12,14]],$y->toArray());

        // with offset
        $x = $la->array([[[9,9,9],[9,9,9]],[[1,2,3],[4,5,6]]]);
        $x = $x[1];
        $this->assertEquals([5,7,9],$la->reduceSum($x,$axis=0)->toArray());
        $this->assertEquals([6,15],$la->reduceSum($x,$axis=1)->toArray());
    }
/*
    public function testReduceSumOLDLarge()
    {
        $mo = $this->newMatrixOperator();
        $la = $this->newLA($mo);
        if(!$la->accelerated()) {
            $this->markTestSkipped('Skip due to high load');
            return;
        }
        // large size
        $colsize = 1000000;
        $rowsize = 64;
        $x = $la->alloc([$rowsize,$colsize],NDArray::float32);
        $la->fill(1.0,$x);
        $sum = $la->reduceSum($x,$axis=1);
        $trues = $la->alloc([$rowsize],NDArray::float32);
        $la->fill($colsize,$trues);
        $this->assertLessThan(1e-3,$la->amax($la->axpy(
            $trues,$sum,-1)));

        // large size
        $colsize = 64;
        $rowsize = 10000;#00;
        $x = $la->alloc([$rowsize,$colsize],NDArray::float32);
        $la->fill(1.0,$x);
        $sum = $la->reduceSum($x,$axis=1);
        $trues = $la->alloc([$rowsize],NDArray::float32);
        $la->fill($colsize,$trues);
        $this->assertLessThan(1e-3,$la->amax($la->axpy(
            $trues,$sum,-1)));
    }
*/
    public function testReduceSumLarge()
    {
        $mo = $this->newMatrixOperator();
        $la = $this->newLA($mo);
        if(!$la->accelerated()) {
            $this->markTestSkipped('Skip due to high load');
            return;
        }
        // large size
        $colsize = 800000;
        $rowsize = 64;
        $x = $la->alloc([$rowsize,$colsize],NDArray::float32);
        $la->fill(1.0,$x);
        $sum = $la->reduceSum($x,$axis=1);
        $trues = $la->alloc([$rowsize],NDArray::float32);
        $la->fill($colsize,$trues);
        $this->assertLessThan(1e-3,$la->amax($la->axpy(
            $trues,$sum,-1)));

        // large size
        $colsize = 64;
        $rowsize = 10000;#00;
        $x = $la->alloc([$rowsize,$colsize],NDArray::float32);
        $la->fill(1.0,$x);
        $sum = $la->reduceSum($x,$axis=1);
        $trues = $la->alloc([$rowsize],NDArray::float32);
        $la->fill($colsize,$trues);
        $this->assertLessThan(1e-3,$la->amax($la->axpy(
            $trues,$sum,-1)));
    }
/*
    public function testReduceSumOLDSpeed()
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

        echo "\n";

        $colsize = 1000000;
        $rowsize = 64;
        $x = $la->alloc([$rowsize,$colsize],NDArray::float32);
        fwrite(STDERR,"Start fill\n");
        $la->fill(1.0,$x);
        fwrite(STDERR,"End fill\n");
        fwrite(STDERR,"Start prepare\n");
        $sum = $la->reduceSum($x,$axis=1);
        fwrite(STDERR,"End prepare\n");
        $start = hrtime(true);
        $sum = $la->reduceSum($x,$axis=1);
        $end = hrtime(true);
        $this->assertEquals($x->size(),$la->asum($sum));
        echo (explode(' ',$la->getConfig()))[0].'='.number_format($end-$start)."\n";

        $colsize = 64;
        $rowsize = 1000000;
        $x = $la->alloc([$rowsize,$colsize],NDArray::float32);
        fwrite(STDERR,"Start fill\n");
        $la->fill(1.0,$x);
        fwrite(STDERR,"End fill\n");
        fwrite(STDERR,"Start prepare\n");
        $sum = $la->reduceSum($x,$axis=1);
        fwrite(STDERR,"End prepare\n");
        $start = hrtime(true);
        $sum = $la->reduceSum($x,$axis=1);
        $end = hrtime(true);
        $this->assertEquals($x->size(),$la->asum($sum));
        echo (explode(' ',$la->getConfig()))[0].'='.number_format($end-$start)."\n";

        $colsize = 4096;
        $rowsize = 12500;#0;
        $x = $la->alloc([$rowsize,$colsize],NDArray::float32);
        fwrite(STDERR,"Start fill\n");
        $la->fill(1.0,$x);
        fwrite(STDERR,"End fill\n");
        fwrite(STDERR,"Start prepare\n");
        $sum = $la->reduceSum($x,$axis=1);
        fwrite(STDERR,"End prepare\n");
        $start = hrtime(true);
        $sum = $la->reduceSum($x,$axis=1);
        $end = hrtime(true);
        $this->assertEquals($x->size(),$la->asum($sum));
        echo (explode(' ',$la->getConfig()))[0].'='.number_format($end-$start)."\n";
    }
*/
    public function testReduceSumSpeed()
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

        echo "\n";

        $colsize = 1000000;
        $rowsize = 64;
        $x = $la->alloc([$rowsize,$colsize],NDArray::float32);
        fwrite(STDERR,"Start fill1\n");
        $la->fill(1.0,$x);
        fwrite(STDERR,"End fill1\n");
        fwrite(STDERR,"Start prepare1\n");
        $sum = $la->reduceSum($x,$axis=1);
        fwrite(STDERR,"End prepare1\n");
        $start = hrtime(true);
        $sum = $la->reduceSum($x,$axis=1);
        $end = hrtime(true);
        $this->assertEquals($x->size(),$la->asum($sum));
        echo (explode(' ',$la->getConfig()))[0].'='.number_format($end-$start)."\n";

        $colsize = 64;
        $rowsize = 1000000;
        $x = $la->alloc([$rowsize,$colsize],NDArray::float32);
        fwrite(STDERR,"Start fill2\n");
        $la->fill(1.0,$x);
        fwrite(STDERR,"End fill2\n");
        fwrite(STDERR,"Start prepare2\n");
        $sum = $la->reduceSum($x,$axis=1);
        fwrite(STDERR,"End prepare2\n");
        $start = hrtime(true);
        $sum = $la->reduceSum($x,$axis=1);
        $end = hrtime(true);
        $this->assertEquals($x->size(),$la->asum($sum));
        echo (explode(' ',$la->getConfig()))[0].'='.number_format($end-$start)."\n";

        $colsize = 4096;
        $rowsize = 8000;#0;
        $x = $la->alloc([$rowsize,$colsize],NDArray::float32);
        fwrite(STDERR,"Start fill3\n");
        $la->fill(1.0,$x);
        fwrite(STDERR,"End fill3\n");
        fwrite(STDERR,"Start prepare3\n");
        $sum = $la->reduceSum($x,$axis=1);
        fwrite(STDERR,"End prepare3\n");
        $start = hrtime(true);
        $sum = $la->reduceSum($x,$axis=1);
        $end = hrtime(true);
        $this->assertEquals($x->size(),$la->asum($sum));
        echo (explode(' ',$la->getConfig()))[0].'='.number_format($end-$start)."\n";
    }

    public function testReduceMaxNormal()
    {
        $mo = $this->newMatrixOperator();
        $la = $this->newLA($mo);
        $x = $la->array([[1,2,3],[4,5,6]]);
        $this->assertEquals([4,5,6],$la->reduceMax($x,$axis=0)->toArray());
        $this->assertEquals([3,6],$la->reduceMax($x,$axis=1)->toArray());

        $mo = $this->newMatrixOperator();
        $la = $this->newLA($mo);
        $x = $la->array([[-1,-2,-3],[-4,-5,-6]]);
        $this->assertEquals([-1,-2,-3],$la->reduceMax($x,$axis=0)->toArray());
        $this->assertEquals([-1,-4],$la->reduceMax($x,$axis=1)->toArray());


        // ***** CAUTION ******
        // 3d array
        $x = $la->array([[[1,2],[3,4]],[[5,6],[7,8]]]);
        $y = $la->reduceMax($x,$axis=0);
        $this->assertEquals([[5,6],[7,8]],$y->toArray());
        $x = $la->array([[[1,2],[3,4]],[[5,6],[7,8]]]);
        $y = $la->reduceMax($x,$axis=1);
        $this->assertEquals([[3,4],[7,8]],$y->toArray());

        $x = $la->array([
            [[1,4,5,8],
             [2,3,6,7]],
            [[1,4,5,8],
             [2,3,6,7]],
            [[1,4,5,8],
             [2,3,6,7]],
         ]);
         $y = $la->reduceMax($x,$axis=1);
         $this->assertEquals([
             [2,4,6,8],
             [2,4,6,8],
             [2,4,6,8]
         ],$y->toArray());

        // with offset
        $x = $la->array([[[9,9,9],[9,9,9]],[[1,2,3],[4,5,6]]]);
        $x = $x[1];
        $this->assertEquals([4,5,6],$la->reduceMax($x,$axis=0)->toArray());
        $this->assertEquals([3,6],$la->reduceMax($x,$axis=1)->toArray());

        // *** CAUTION ***
        // if NaN set NaN
        // Compatible with reduce_max of tensorflow 2.6
        $x = $la->array([
            [1.0, 2.0],
            [INF, 1.0],
            [INF,INF],
            [-INF,INF],
            [0.0, NAN],
            [INF, NAN]]);
        $x = $la->reduceMax($x,$axis=1);
        $x = $la->toNDArray($x);
        $this->assertEquals(2,   $x[0]);
        $this->assertTrue(INF==  $x[1]);
        $this->assertTrue(INF==  $x[2]);
        $this->assertTrue(INF==  $x[3]);
        $this->assertTrue(is_nan($x[4]));
        $this->assertTrue(is_nan($x[5]));
    }

    public function testReduceMaxLarge()
    {
        $mo = $this->newMatrixOperator();
        $la = $this->newLA($mo);
        if(!$la->accelerated()) {
            $this->markTestSkipped('Skip due to high load');
            return;
        }
        // large size
        $colsize = 1000000;
        $rowsize = 64;
        $x = $la->alloc([$rowsize,$colsize],NDArray::float32);
        $la->fill(1.0,$x);
        $r = $la->reduceMax($x,$axis=1);
        $trues = $la->alloc([$rowsize],NDArray::float32);
        $la->fill(1.0,$trues);
        $this->assertLessThan(1e-3,$la->amax($la->axpy(
            $trues,$r,-1)));

        // *** CAUTION ***
        // if NaN set NaN
        // Compatible with reduce_max of tensorflow 2.6

        //    mode:  0   1    2     3
        $colslist = [10, 200, 2000, 200000];
        foreach($colslist as $cols) {
            $x = $la->alloc([6,$cols],NDArray::float32);
            $la->fill(0.0,$x);
            $la->copy($la->array([1.0, 2.0]),   $x[0][[0,1]]);
            $la->copy($la->array([INF, 1.0]),   $x[1][[0,1]]);
            $la->copy($la->array([INF, INF]),   $x[2][[0,1]]);
            $la->copy($la->array([-INF,INF]),   $x[3][[0,1]]);
            $la->copy($la->array([1.0, NAN]),   $x[4][[0,1]]);
            $la->copy($la->array([INF, NAN]),   $x[5][[0,1]]);

            $x = $la->reduceMax($x,$axis=1);
            $x = $la->toNDArray($x);

            $this->assertEquals(2,   $x[0]);
            $this->assertTrue(INF==  $x[1]);
            $this->assertTrue(INF==  $x[2]);
            $this->assertTrue(INF==  $x[3]);
            $this->assertTrue(is_nan($x[4]));
            $this->assertTrue(is_nan($x[5]));
        }
    }

    public function testReduceMaxSpeed()
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

        $colsize = 1000000;
        $rowsize = 64;
        $x = $la->alloc([$rowsize,$colsize],NDArray::float32);
        $la->fill(1.0,$x);
        $max = $la->reduceMax($x,$axis=1);
        $start = hrtime(true);
        $max = $la->reduceMax($x,$axis=1);
        $end = hrtime(true);
        $this->assertEquals(1,$la->amax($max));
        echo "\n".(explode(' ',$la->getConfig()))[0].'='.number_format($end-$start)."\n";

        $colsize = 64;
        $rowsize = 1000000;
        $x = $la->alloc([$rowsize,$colsize],NDArray::float32);
        $la->fill(1.0,$x);
        $max = $la->reduceMax($x,$axis=1);
        $start = hrtime(true);
        $max = $la->reduceMax($x,$axis=1);
        $end = hrtime(true);
        $this->assertEquals(1,$la->amax($max));
        echo "\n".(explode(' ',$la->getConfig()))[0].'='.number_format($end-$start)."\n";

        $colsize = 4096;
        $rowsize = 12500;#0;
        $x = $la->alloc([$rowsize,$colsize],NDArray::float32);
        $la->fill(1.0,$x);
        $max = $la->reduceMax($x,$axis=1);
        $start = hrtime(true);
        $max = $la->reduceMax($x,$axis=1);
        $end = hrtime(true);
        $this->assertEquals(1,$la->amax($max));
        echo "\n".(explode(' ',$la->getConfig()))[0].'='.number_format($end-$start)."\n";
    }

    public function testReduceArgMaxNormal()
    {
        $mo = $this->newMatrixOperator();
        $la = $this->newLA($mo);
        $x = $la->array([[1,2,3],[4,5,6]]);
        $this->assertEquals([1,1,1],$la->reduceArgMax($x,$axis=0)->toArray());
        $this->assertEquals([2,2],$la->reduceArgMax($x,$axis=1)->toArray());

        // ***** CAUTION ******
        // 3d array
        $x = $la->array([[[1,2],[3,4]],[[5,6],[7,8]]]);
        $y = $la->reduceArgMax($x,$axis=0);
        $this->assertEquals([[1,1],[1,1]],$y->toArray());
        $x = $la->array([
            [[1,4,5,8],
             [2,3,6,7]],
            [[1,4,5,8],
             [2,3,6,7]],
            [[1,4,5,8],
             [2,3,6,7]],
         ]);
        $y = $la->reduceArgMax($x,$axis=1);
        $this->assertEquals([
            [1,0,1,0],
            [1,0,1,0],
            [1,0,1,0],
        ],$y->toArray());

        // with offset
        $x = $la->array([[[9,9,9],[9,9,9]],[[1,2,3],[4,5,6]]]);
        $x = $x[1];
        $this->assertEquals([1,1,1],$la->reduceArgMax($x,$axis=0)->toArray());
        $this->assertEquals([2,2],$la->reduceArgMax($x,$axis=1)->toArray());

    }

    public function testReduceArgMaxLarge()
    {
        $mo = $this->newMatrixOperator();
        $la = $this->newLA($mo);
        if(!$la->accelerated()) {
            $this->markTestSkipped('Skip due to high load');
            return;
        }
        // large size
        $colsize = 500000;
        $rowsize = 64;
        $x = $la->alloc([$rowsize,$colsize],NDArray::float32);
        $la->fill(1.0,$x);
        $max = $la->reduceArgMax($x,$axis=1);
        $trues = $la->alloc([$rowsize],NDArray::int32);
        $this->assertTrue($la->sum($max)==0);
    }

    public function testReduceArgMaxSpeed()
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

        $colsize = 1000000;
        $rowsize = 64;
        $x = $la->alloc([$rowsize,$colsize],NDArray::float32);
        $la->fill(1.0,$x);
        $max = $la->reduceArgMax($x,$axis=1);
        $start = hrtime(true);
        $max = $la->reduceArgMax($x,$axis=1);
        $end = hrtime(true);
        $this->assertTrue($la->sum($max)==0);
        echo "\n".(explode(' ',$la->getConfig()))[0].'='.number_format($end-$start)."\n";

        $colsize = 64;
        $rowsize = 1000000;
        $x = $la->alloc([$rowsize,$colsize],NDArray::float32);
        $la->fill(1.0,$x);
        $max = $la->reduceArgMax($x,$axis=1);
        $start = hrtime(true);
        $max = $la->reduceArgMax($x,$axis=1);
        $end = hrtime(true);
        $this->assertTrue($la->sum($max)==0);
        echo "\n".(explode(' ',$la->getConfig()))[0].'='.number_format($end-$start)."\n";

        $colsize = 4096;
        $rowsize = 12500;#0;
        $x = $la->alloc([$rowsize,$colsize],NDArray::float32);
        $la->fill(1.0,$x);
        $max = $la->reduceArgMax($x,$axis=1);
        $start = hrtime(true);
        $max = $la->reduceArgMax($x,$axis=1);
        $end = hrtime(true);
        $this->assertTrue($la->sum($max)==0);
        echo "\n".(explode(' ',$la->getConfig()))[0].'='.number_format($end-$start)."\n";
    }

    public function testReduceMean()
    {
        $mo = $this->newMatrixOperator();
        $la = $this->newLA($mo);
        $x = $la->array([[1,2,3],[4,5,6]]);
        $this->assertEquals([2.5,3.5,4.5],$la->reduceMean($x,$axis=0)->toArray());
        $this->assertEquals([2,5],$la->reduceMean($x,$axis=1)->toArray());

        // n-dimensions
        $x = $la->array([[[1,2,3],[4,5,6]],[[1,2,3],[4,5,6]],[[1,2,3],[4,5,6]]]);
        $y = $la->reduceMean($x,$axis=0);
        $this->assertEquals([3,2,3],$x->shape());
        $this->assertEquals([2,3],$y->shape());
        $this->assertEquals([[1,2,3],[4,5,6]],$y->toArray());
        $y = $la->reduceMean($x,$axis=1);
        $this->assertEquals([3,3],$y->shape());
        $this->assertEquals([[2.5,3.5,4.5],[2.5,3.5,4.5],[2.5,3.5,4.5]],$y->toArray());

        // with offset
        $x = $la->array([[[9,9,9],[9,9,9]],[[1,2,3],[4,5,6]]]);
        $x = $x[1];
        $this->assertEquals([2.5,3.5,4.5],$la->reduceMean($x,$axis=0)->toArray());
        $this->assertEquals([2,5],$la->reduceMean($x,$axis=1)->toArray());
    }

    public function testEqualNormal()
    {
        $mo = $this->newMatrixOperator();
        $la = $this->newLA($mo);

        $X = $la->array([100,10,-1000]);
        $Y = $la->array([100,-10,-1000]);
        $this->assertEquals([1,0,1],$la->equal($X,$Y)->toArray());
    }

    public function testSoftmax()
    {
        $mo = $this->newMatrixOperator();
        $la = $this->newLA($mo);

        $x = $la->array([
            [-1.0,-0.5,0.0,0.5,1.0],
            [-1.0,-0.5,0.0,0.5,1.0],
            [-1.0,-0.5,0.0,0.5,1.0],
            [-1.0,-0.5,0.0,0.5,1.0],
            [-1.0,-0.5,0.0,0.5,1.0],
        ]);
        $y = $la->softmax($x);
        if(is_scalar($y[0][0])) {
            $this->assertTrue($this->equalTest(0.05801,$y[0][0]));
            $this->assertTrue($this->equalTest(0.09564,$y[0][1]));
            $this->assertTrue($this->equalTest(0.15769,$y[0][2]));
            $this->assertTrue($this->equalTest(0.25999,$y[0][3]));
            $this->assertTrue($this->equalTest(0.42865,$y[0][4]));
        } else {
            $this->assertTrue($this->equalTest(0.05801,$y[0][0]->toArray()));
            $this->assertTrue($this->equalTest(0.09564,$y[0][1]->toArray()));
            $this->assertTrue($this->equalTest(0.15769,$y[0][2]->toArray()));
            $this->assertTrue($this->equalTest(0.25999,$y[0][3]->toArray()));
            $this->assertTrue($this->equalTest(0.42865,$y[0][4]->toArray()));
        }
    }

    public function testSoftmaxLarge()
    {
        $mo = $this->newMatrixOperator();
        $la = $this->newLA($mo);
        if(!$la->accelerated()) {
            $this->markTestSkipped('Skip due to high load');
            return;
        }
            // large size
        $colsize = 600000;
        $rowsize = 64;
        $x = $la->alloc([$rowsize,$colsize],NDArray::float32);
        $la->fill(1.0,$x);
        $r = $la->softmax($x);
        $trues = $la->alloc([$rowsize,$colsize],NDArray::float32);
        $la->fill(1.0,$trues);
        $this->assertLessThan(1e-3,$la->amax($la->axpy(
            $trues,$r,-1)));

        // large size
        $colsize = 64;
        $rowsize = 800000;
        $x = $la->alloc([$rowsize,$colsize],NDArray::float32);
        $la->fill(1.0,$x);
        $r = $la->softmax($x);
        $trues = $la->alloc([$rowsize,$colsize],NDArray::float32);
        $la->fill(1.0,$trues);
        $this->assertLessThan(1e-3,$la->amax($la->axpy(
            $trues,$r,-1)));
    }

    public function testSoftmaxSpeed()
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
        $colsize = 600000;
        $rowsize = 64;
        $x = $la->alloc([$rowsize,$colsize],NDArray::float32);
        fwrite(STDERR,"fill-start\n");
        $la->fill(1.0,$x);
        fwrite(STDERR,"fill-end\n");
        fwrite(STDERR,"pre-start\n");
        $r = $la->softmax($x);
        fwrite(STDERR,"pre-end\n");
        $start = hrtime(true);
        $r = $la->softmax($x);
        $end = hrtime(true);
        echo "\n".(explode(' ',$la->getConfig()))[0].'='.number_format($end-$start)."\n";

        // large size
        $colsize = 64;
        $rowsize = 1000000;
        $x = $la->alloc([$rowsize,$colsize],NDArray::float32);
        fwrite(STDERR,"fill-start\n");
        $la->fill(1.0,$x);
        fwrite(STDERR,"fill-end\n");
        fwrite(STDERR,"pre-start\n");
        $r = $la->softmax($x);
        fwrite(STDERR,"pre-end\n");
        $start = hrtime(true);
        $r = $la->softmax($x);
        $end = hrtime(true);
        echo "\n".(explode(' ',$la->getConfig()))[0].'='.number_format($end-$start)."\n";
    }

    public function testastype()
    {
        $mo = $this->newMatrixOperator();
        $la = $this->newLA($mo);
        $math = $la;
        if($la->accelerated()) {
            $devType = $math->getContext()->getInfo(OpenCL::CL_CONTEXT_DEVICES)->getInfo(0,OpenCL::CL_DEVICE_TYPE);
            $devName = $math->getContext()->getInfo(OpenCL::CL_CONTEXT_DEVICES)->getInfo(0,OpenCL::CL_DEVICE_NAME);
        } else {
            $devType = OpenCL::CL_DEVICE_TYPE_CPU;
            $devName = "CPU";
        }

        #### int to any
        $X = $la->array([-1,0,1,2,3],NDArray::int32);
        $dtype = NDArray::float32;
        $Y = $math->astype($X, $dtype);
        $this->assertEquals(NDArray::float32,$Y->dtype());
        $this->assertEquals([-1,0,1,2,3],$Y->toArray());

        if($la->fp64()) {
            $dtype = NDArray::float64;
            $Y = $math->astype($X, $dtype);
            $this->assertEquals([-1,0,1,2,3],$Y->toArray());
        }

        $dtype = NDArray::int8;
        $Y = $math->astype($X, $dtype);
        $this->assertEquals([-1,0,1,2,3],$Y->toArray());

        $dtype = NDArray::int16;
        $Y = $math->astype($X, $dtype);
        $this->assertEquals([-1,0,1,2,3],$Y->toArray());

        $dtype = NDArray::int32;
        $Y = $math->astype($X, $dtype);
        $this->assertEquals([-1,0,1,2,3],$Y->toArray());

        $dtype = NDArray::int64;
        $Y = $math->astype($X, $dtype);
        $this->assertEquals([-1,0,1,2,3],$Y->toArray());

        $dtype = NDArray::bool;
        $Y = $math->astype($X, $dtype);
        $this->assertEquals([true,false,true,true,true],$Y->toArray());

        #### float to any ######
        $X = $la->array([-1,0,1,2,3],NDArray::float32);
        $dtype = NDArray::float32;
        $Y = $math->astype($X, $dtype);
        $this->assertEquals([-1,0,1,2,3],$Y->toArray());

        if($la->fp64()) {
            $dtype = NDArray::float64;
            $Y = $math->astype($X, $dtype);
            $this->assertEquals([-1,0,1,2,3],$Y->toArray());
        }

        $dtype = NDArray::int8;
        $Y = $math->astype($X, $dtype);
        $this->assertEquals([-1,0,1,2,3],$Y->toArray());

        $dtype = NDArray::int16;
        $Y = $math->astype($X, $dtype);
        $this->assertEquals([-1,0,1,2,3],$Y->toArray());

        $dtype = NDArray::int32;
        $Y = $math->astype($X, $dtype);
        $this->assertEquals([-1,0,1,2,3],$Y->toArray());

        $dtype = NDArray::int64;
        $Y = $math->astype($X, $dtype);
        $this->assertEquals([-1,0,1,2,3],$Y->toArray());

        $dtype = NDArray::bool;
        $Y = $math->astype($X, $dtype);
        if($devType==OpenCL::CL_DEVICE_TYPE_GPU&&
            strpos($devName,'Intel')!==false ) {
            $this->assertEquals([false,false,true,true,true],$Y->toArray());
        } else {
            $this->assertEquals([true,false,true,true,true],$Y->toArray());
        }

        #### bool to any ######
        $X = $la->array([true,false,true,true,true],NDArray::bool);
        $dtype = NDArray::float32;
        $Y = $math->astype($X, $dtype);
        $this->assertEquals([1,0,1,1,1],$Y->toArray());

        if($la->fp64()) {
            $dtype = NDArray::float64;
            $Y = $math->astype($X, $dtype);
            $this->assertEquals([1,0,1,1,1],$Y->toArray());
        }

        $dtype = NDArray::int8;
        $Y = $math->astype($X, $dtype);
        $this->assertEquals([1,0,1,1,1],$Y->toArray());

        $dtype = NDArray::int16;
        $Y = $math->astype($X, $dtype);
        $this->assertEquals([1,0,1,1,1],$Y->toArray());

        $dtype = NDArray::int32;
        $Y = $math->astype($X, $dtype);
        $this->assertEquals([1,0,1,1,1],$Y->toArray());

        $dtype = NDArray::int64;
        $Y = $math->astype($X, $dtype);
        $this->assertEquals([1,0,1,1,1],$Y->toArray());

        $dtype = NDArray::bool;
        $Y = $math->astype($X, $dtype);
        $this->assertEquals([true,false,true,true,true],$Y->toArray());

        #### float to unsigned ######
        $X = $la->array([-1,0,1,2,3],NDArray::float32);
        $dtype = NDArray::uint8;
        $Y = $math->astype($X, $dtype);
        if($devType==OpenCL::CL_DEVICE_TYPE_GPU&&
            strpos($devName,'Intel')!==false ) {
            $this->assertEquals([0,0,1,2,3], $Y->toArray());
        } else {
            $this->assertEquals([255,0,1,2,3], $Y->toArray());
        } 

        $dtype = NDArray::uint16;
        $Y = $math->astype($X, $dtype);
        if($devType==OpenCL::CL_DEVICE_TYPE_GPU&&
            strpos($devName,'Intel')!==false ) {
            $this->assertEquals([0,0,1,2,3], $Y->toArray());
        } else {
            $this->assertEquals([65535,0,1,2,3], $Y->toArray());
        }

        // ***** CAUTION ******
        $X = $la->array([-1000,0,1,2,4294967295],NDArray::float32);
        if($la->accelerated()) {
            // GPU
            $dtype = NDArray::uint32;
            $Y = $math->astype($X, $dtype);
            if($devType===OpenCL::CL_DEVICE_TYPE_GPU) {
                $this->assertEquals([0,0,1,2,4294967295],$Y->toArray());
            } else {
                $this->assertEquals([4294966296,0,1,2,0],$Y->toArray());
            }
        } else {
            // CPU
            $dtype = NDArray::uint32;
            $Y = $math->astype($X, $dtype);
            if(extension_loaded('rindow_openblas')) {
                $this->assertEquals([4294966296,0,1,2,0],$Y->toArray());
            } else {
                $this->assertEquals([4294966296,0,1,2,4294967295],$Y->toArray());
            }
        }

        // ***** CAUTION ******
        $X = $la->array([-1000,0,1,2,3],NDArray::float32);
        if($la->accelerated()) {
            // GPU
            $dtype = NDArray::uint64;
            $Y = $math->astype($X, $dtype);
            if($devType===OpenCL::CL_DEVICE_TYPE_GPU) {
                if($devName=='Loveland') {
                    $this->assertEquals([0,0,1,2,3],$Y->toArray());
                } elseif(strpos($devName,'Intel')!==false) {
                    $this->assertEquals([1000,0,1,2,3],$Y->toArray());
                } else {
                    $this->assertEquals([-1000,0,1,2,3],$Y->toArray());
                }
            } else {
                $this->assertEquals([-1000,0,1,2,3],$Y->toArray());
            }
        } elseif($la->getConfig()=='PhpBlas') {
            // CPU
            $dtype = NDArray::uint64;
            $Y = $math->astype($X, $dtype);
            $this->assertEquals([-1000,0,1,2,3],$Y->toArray());
        }
    }


    public function providerIm2col2dNormal()
    {
        return [
            'normal' => [[
                'batches' => 2,
                'im_h' => 8,
                'im_w' => 8,
                'channels' => 3,
                'kernel_h' => 3,
                'kernel_w' => 3,
                'stride_h' => 1,
                'stride_w' => 1,
                'padding' => null,
                'channels_first' => null,
                'dilation_h' => 1,
                'dilation_w' => 1,
                'cols_channels_first' => null,
            ]],
            'kernel_h' => [[
                'batches' => 2,
                'im_h' => 8,
                'im_w' => 8,
                'channels' => 3,
                'kernel_h' => 4,
                'kernel_w' => 3,
                'stride_h' => 1,
                'stride_w' => 1,
                'padding' => null,
                'channels_first' => null,
                'dilation_h' => 1,
                'dilation_w' => 1,
                'cols_channels_first' => null,
            ]],
            'kernel_w' => [[
                'batches' => 2,
                'im_h' => 8,
                'im_w' => 8,
                'channels' => 3,
                'kernel_h' => 3,
                'kernel_w' => 4,
                'stride_h' => 1,
                'stride_w' => 1,
                'padding' => null,
                'channels_first' => null,
                'dilation_h' => 1,
                'dilation_w' => 1,
                'cols_channels_first' => null,
            ]],
            'stride_h' => [[
                'batches' => 2,
                'im_h' => 8,
                'im_w' => 8,
                'channels' => 3,
                'kernel_h' => 3,
                'kernel_w' => 3,
                'stride_h' => 2,
                'stride_w' => 1,
                'padding' => null,
                'channels_first' => null,
                'dilation_h' => 1,
                'dilation_w' => 1,
                'cols_channels_first' => null,
            ]],
            'stride_w' => [[
                'batches' => 2,
                'im_h' => 8,
                'im_w' => 8,
                'channels' => 3,
                'kernel_h' => 3,
                'kernel_w' => 3,
                'stride_h' => 1,
                'stride_w' => 2,
                'padding' => null,
                'channels_first' => null,
                'dilation_h' => 1,
                'dilation_w' => 1,
                'cols_channels_first' => null,
            ]],
            'dilation_h' => [[
                'batches' => 2,
                'im_h' => 8,
                'im_w' => 8,
                'channels' => 3,
                'kernel_h' => 3,
                'kernel_w' => 3,
                'stride_h' => 1,
                'stride_w' => 1,
                'padding' => null,
                'channels_first' => null,
                'dilation_h' => 2,
                'dilation_w' => 1,
                'cols_channels_first' => null,
            ]],
            'dilation_w' => [[
                'batches' => 2,
                'im_h' => 8,
                'im_w' => 8,
                'channels' => 3,
                'kernel_h' => 3,
                'kernel_w' => 3,
                'stride_h' => 1,
                'stride_w' => 1,
                'padding' => null,
                'channels_first' => null,
                'dilation_h' => 1,
                'dilation_w' => 2,
                'cols_channels_first' => null,
            ]],
            'normal channels_first' => [[
                'batches' => 2,
                'im_h' => 8,
                'im_w' => 8,
                'channels' => 3,
                'kernel_h' => 3,
                'kernel_w' => 3,
                'stride_h' => 1,
                'stride_w' => 1,
                'padding' => null,
                'channels_first' => true,
                'dilation_h' => 1,
                'dilation_w' => 1,
                'cols_channels_first' => null,
            ]],
            'kernel_h channels_first' => [[
                'batches' => 2,
                'im_h' => 8,
                'im_w' => 8,
                'channels' => 3,
                'kernel_h' => 4,
                'kernel_w' => 3,
                'stride_h' => 1,
                'stride_w' => 1,
                'padding' => null,
                'channels_first' => true,
                'dilation_h' => 1,
                'dilation_w' => 1,
                'cols_channels_first' => null,
            ]],
            'kernel_w channels_first' => [[
                'batches' => 2,
                'im_h' => 8,
                'im_w' => 8,
                'channels' => 3,
                'kernel_h' => 3,
                'kernel_w' => 4,
                'stride_h' => 1,
                'stride_w' => 1,
                'padding' => null,
                'channels_first' => true,
                'dilation_h' => 1,
                'dilation_w' => 1,
                'cols_channels_first' => null,
            ]],
            'stride_h channels_first' => [[
                'batches' => 2,
                'im_h' => 8,
                'im_w' => 8,
                'channels' => 3,
                'kernel_h' => 3,
                'kernel_w' => 3,
                'stride_h' => 2,
                'stride_w' => 1,
                'padding' => null,
                'channels_first' => true,
                'dilation_h' => 1,
                'dilation_w' => 1,
                'cols_channels_first' => null,
            ]],
            'stride_w channels_first' => [[
                'batches' => 2,
                'im_h' => 8,
                'im_w' => 8,
                'channels' => 3,
                'kernel_h' => 3,
                'kernel_w' => 3,
                'stride_h' => 1,
                'stride_w' => 2,
                'padding' => null,
                'channels_first' => true,
                'dilation_h' => 1,
                'dilation_w' => 1,
                'cols_channels_first' => null,
            ]],
            'dilation_h channels_first' => [[
                'batches' => 2,
                'im_h' => 8,
                'im_w' => 8,
                'channels' => 3,
                'kernel_h' => 3,
                'kernel_w' => 3,
                'stride_h' => 1,
                'stride_w' => 1,
                'padding' => null,
                'channels_first' => true,
                'dilation_h' => 2,
                'dilation_w' => 1,
                'cols_channels_first' => null,
            ]],
            'dilation_w channels_first' => [[
                'batches' => 2,
                'im_h' => 8,
                'im_w' => 8,
                'channels' => 3,
                'kernel_h' => 3,
                'kernel_w' => 3,
                'stride_h' => 1,
                'stride_w' => 1,
                'padding' => null,
                'channels_first' => true,
                'dilation_h' => 1,
                'dilation_w' => 2,
                'cols_channels_first' => null,
            ]],
            'normal padding' => [[
                'batches' => 2,
                'im_h' => 8,
                'im_w' => 8,
                'channels' => 3,
                'kernel_h' => 3,
                'kernel_w' => 3,
                'stride_h' => 1,
                'stride_w' => 1,
                'padding' => true,
                'channels_first' => null,
                'dilation_h' => 1,
                'dilation_w' => 1,
                'cols_channels_first' => null,
            ]],
            'kernel_h padding' => [[
                'batches' => 2,
                'im_h' => 8,
                'im_w' => 8,
                'channels' => 3,
                'kernel_h' => 4,
                'kernel_w' => 3,
                'stride_h' => 1,
                'stride_w' => 1,
                'padding' => true,
                'channels_first' => null,
                'dilation_h' => 1,
                'dilation_w' => 1,
                'cols_channels_first' => null,
            ]],
            'kernel_w padding' => [[
                'batches' => 2,
                'im_h' => 8,
                'im_w' => 8,
                'channels' => 3,
                'kernel_h' => 3,
                'kernel_w' => 4,
                'stride_h' => 1,
                'stride_w' => 1,
                'padding' => true,
                'channels_first' => null,
                'dilation_h' => 1,
                'dilation_w' => 1,
                'cols_channels_first' => null,
            ]],
            'stride_h padding' => [[
                'batches' => 2,
                'im_h' => 8,
                'im_w' => 8,
                'channels' => 3,
                'kernel_h' => 3,
                'kernel_w' => 3,
                'stride_h' => 2,
                'stride_w' => 1,
                'padding' => true,
                'channels_first' => null,
                'dilation_h' => 1,
                'dilation_w' => 1,
                'cols_channels_first' => null,
            ]],
            'stride_w padding' => [[
                'batches' => 2,
                'im_h' => 8,
                'im_w' => 8,
                'channels' => 3,
                'kernel_h' => 3,
                'kernel_w' => 3,
                'stride_h' => 1,
                'stride_w' => 2,
                'padding' => true,
                'channels_first' => null,
                'dilation_h' => 1,
                'dilation_w' => 1,
                'cols_channels_first' => null,
            ]],
            'dilation_h padding' => [[
                'batches' => 2,
                'im_h' => 8,
                'im_w' => 8,
                'channels' => 3,
                'kernel_h' => 3,
                'kernel_w' => 3,
                'stride_h' => 1,
                'stride_w' => 1,
                'padding' => true,
                'channels_first' => null,
                'dilation_h' => 2,
                'dilation_w' => 1,
                'cols_channels_first' => null,
            ]],
            'dilation_w padding' => [[
                'batches' => 2,
                'im_h' => 8,
                'im_w' => 8,
                'channels' => 3,
                'kernel_h' => 3,
                'kernel_w' => 3,
                'stride_h' => 1,
                'stride_w' => 1,
                'padding' => true,
                'channels_first' => null,
                'dilation_h' => 1,
                'dilation_w' => 2,
                'cols_channels_first' => null,
            ]],
            'normal cols_channels_first' => [[
                'batches' => 2,
                'im_h' => 8,
                'im_w' => 8,
                'channels' => 3,
                'kernel_h' => 3,
                'kernel_w' => 3,
                'stride_h' => 1,
                'stride_w' => 1,
                'padding' => null,
                'channels_first' => null,
                'dilation_h' => 1,
                'dilation_w' => 1,
                'cols_channels_first' => true,
            ]],
            'kernel_h cols_channels_first' => [[
                'batches' => 2,
                'im_h' => 8,
                'im_w' => 8,
                'channels' => 3,
                'kernel_h' => 4,
                'kernel_w' => 3,
                'stride_h' => 1,
                'stride_w' => 1,
                'padding' => null,
                'channels_first' => null,
                'dilation_h' => 1,
                'dilation_w' => 1,
                'cols_channels_first' => true,
            ]],
            'kernel_w cols_channels_first' => [[
                'batches' => 2,
                'im_h' => 8,
                'im_w' => 8,
                'channels' => 3,
                'kernel_h' => 3,
                'kernel_w' => 4,
                'stride_h' => 1,
                'stride_w' => 1,
                'padding' => null,
                'channels_first' => null,
                'dilation_h' => 1,
                'dilation_w' => 1,
                'cols_channels_first' => true,
            ]],
            'stride_h cols_channels_first' => [[
                'batches' => 2,
                'im_h' => 8,
                'im_w' => 8,
                'channels' => 3,
                'kernel_h' => 3,
                'kernel_w' => 3,
                'stride_h' => 2,
                'stride_w' => 1,
                'padding' => null,
                'channels_first' => null,
                'dilation_h' => 1,
                'dilation_w' => 1,
                'cols_channels_first' => true,
            ]],
            'stride_w cols_channels_first' => [[
                'batches' => 2,
                'im_h' => 8,
                'im_w' => 8,
                'channels' => 3,
                'kernel_h' => 3,
                'kernel_w' => 3,
                'stride_h' => 1,
                'stride_w' => 2,
                'padding' => null,
                'channels_first' => null,
                'dilation_h' => 1,
                'dilation_w' => 1,
                'cols_channels_first' => true,
            ]],
            'dilation_h cols_channels_first' => [[
                'batches' => 2,
                'im_h' => 8,
                'im_w' => 8,
                'channels' => 3,
                'kernel_h' => 3,
                'kernel_w' => 3,
                'stride_h' => 1,
                'stride_w' => 1,
                'padding' => null,
                'channels_first' => null,
                'dilation_h' => 2,
                'dilation_w' => 1,
                'cols_channels_first' => true,
            ]],
            'dilation_w cols_channels_first' => [[
                'batches' => 2,
                'im_h' => 8,
                'im_w' => 8,
                'channels' => 3,
                'kernel_h' => 3,
                'kernel_w' => 3,
                'stride_h' => 1,
                'stride_w' => 1,
                'padding' => null,
                'channels_first' => null,
                'dilation_h' => 1,
                'dilation_w' => 2,
                'cols_channels_first' => true,
            ]],
        ];
    }
    /**
    * @dataProvider providerIm2col2dNormal
    */
    public function testIm2col2dNormal($params)
    {
        $mo = $this->newMatrixOperator();
        $la = $this->newLA($mo);

        extract($params);

        //$batches = 1;
        //$im_h = 4;
        //$im_w = 4;
        //$channels = 3;
        //$kernel_h = 3;
        //$kernel_w = 3;
        //$stride_h = 1;
        //$stride_w = 1;
        //$padding = null;
        //$channels_first = null;
        //$dilation_h = 1;
        //$dilation_w = 1;
        //$cols_channels_first=null;
        $cols = null;

        $images = $la->array($mo->arange(
            $batches*
            $im_h*$im_w*
            $channels,
            null,null,
            NDArray::float32
        ));
        if($channels_first) {
            $images = $images->reshape([
                $batches,
                $channels,
                $im_h,
                $im_w
            ]);
        } else {
            $images = $images->reshape([
                $batches,
                $im_h,
                $im_w,
                $channels
            ]);
        }
        $cols = $la->im2col(
            $images,
            $filterSize=[
                $kernel_h,$kernel_w],
            $strides=[
                $stride_h,$stride_w],
            $padding,
            $channels_first,
            $dilation_rate=[
                $dilation_h,$dilation_w],
            $cols_channels_first
        );
        $out_h = intval(floor(($im_h-($kernel_h-1)*$dilation_h-1)/$stride_h)+1);
        $out_w = intval(floor(($im_w-($kernel_w-1)*$dilation_w-1)/$stride_w)+1);
        if($padding) {
            $padding_h = (int)floor((($im_h-1)*$stride_h-$im_h+($kernel_h-1)*$dilation_h+1)/2);
            $padding_w = (int)floor((($im_w-1)*$stride_w-$im_w+($kernel_w-1)*$dilation_w+1)/2);
            $out_h = $im_h;
            $out_w = $im_w;
        } else {
            $padding_h = 0;
            $padding_w = 0;
        }

        if($cols_channels_first) {
            $this->assertEquals(
                [
                    $batches,
                    $out_h,$out_w,
                    $channels,
                    $kernel_h,$kernel_w,
                ],
                $cols->shape()
            );
        } else {
            $this->assertEquals(
                [
                    $batches,
                    $out_h,$out_w,
                    $kernel_h,$kernel_w,
                    $channels,
                ],
                $cols->shape()
            );
        }
        $trues = $this->newArray($cols->shape());
        $truesBuffer = $trues->buffer();
        for($batch_id=0;$batch_id<$batches;$batch_id++) {
            for($channel_id=0;$channel_id<$channels;$channel_id++) {
                for($im_y=0;$im_y<$out_h;$im_y++) {
                    for($im_x=0;$im_x<$out_w;$im_x++) {
                        for($kernel_y=0;$kernel_y<$kernel_h;$kernel_y++) {
                            for($kernel_x=0;$kernel_x<$kernel_w;$kernel_x++) {
                                $input_y = $im_y*$stride_h+$kernel_y*$dilation_h-$padding_h;
                                $input_x = $im_x*$stride_w+$kernel_x*$dilation_w-$padding_w;
                                if($channels_first) {
                                    $input_id = ((($batch_id*$channels+$channel_id)*$im_h+$input_y)*$im_w+$input_x);
                                } else {
                                    $input_id = ((($batch_id*$im_h+$input_y)*$im_w+$input_x)*$channels+$channel_id);
                                }
                                if($cols_channels_first) {
                                    $cols_id = ((((($batch_id*$out_h+$im_y)*$out_w+$im_x)
                                                *$channels+$channel_id)*$kernel_h+$kernel_y)*$kernel_w+$kernel_x);
                                } else {
                                    $cols_id = ((((($batch_id*$out_h+$im_y)*$out_w+$im_x)
                                                *$kernel_h+$kernel_y)*$kernel_w+$kernel_x)*$channels+$channel_id);
                                }
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
        // channels_first kernel stride
        //for($batch_id=0;$batch_id<$batches;$batch_id++) {
        //    for($kernel_y=0;$kernel_y<$kernel_h;$kernel_y++) {
        //        echo "kernel_h=$kernel_y\n";
        //        for($kernel_x=0;$kernel_x<$kernel_w;$kernel_x++) {
        //            echo "kernel_w=$kernel_x\n";
        //            for($channel_id=0;$channel_id<$channels;$channel_id++) {
        //                echo "channel=$channel_id\n";
        //                for($out_y=0;$out_y<$out_h;$out_y++) {
        //                    echo "[";
        //                    for($out_x=0;$out_x<$out_w;$out_x++) {
        //                        $value = $cols[$batch_id][$out_y][$out_x][$kernel_y][$kernel_x][$channel_id];
        //                        if(!is_scalar($value)) { $value = $value->toArray(); }
        //                        echo sprintf('%2d',intval($value)).",";
        //                    }
        //                    echo "],\n";
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
        //                        $value = $cols[$batch_id][$out_y][$out_x][$kernel_y][$kernel_x][$channel_id];
        //                        if(!is_scalar($value))
        //                            $value = $value->toArray();
        //                        echo sprintf('%3d',$value).",";
        //                    }
        //                    echo "],";
        //                    echo "\n";
        //                }
        //            }
        //        }
        //    }
        //}

        //foreach ($cols->toArray() as $batch) {
        //    foreach ($batch as $col_h_id => $col_h_value) {
        //        echo "stride_h=$col_h_id\n";
        //        foreach ($col_h_value as $col_w_id => $col_w_value) {
        //            echo "stride($col_h_id,$col_w_id)\n";
        //            foreach ($col_w_value as $key => $kernel_h_value) {
        //                #echo "kernel_h=$key\n";
        //                foreach ($kernel_h_value as $key => $kernel_w_value) {
        //                    #echo "kernel_w=$key\n";
        //                    echo "[";
        //                    foreach ($kernel_w_value as $key => $channel_value) {
        //                            echo sprintf('%2d',$channel_value).",";
        //                    }
        //                    echo "],";
        //                }
        //                echo "\n";
        //            }
        //        }
        //    }
        //}
        //echo "==== cols trues ======\n";
        //foreach ($trues->toArray() as $batch) {
        //    foreach ($batch as $col_h_id => $col_h_value) {
        //        echo "stride_h=$col_h_id\n";
        //        foreach ($col_h_value as $col_w_id => $col_w_value) {
        //            echo "stride($col_h_id,$col_w_id)\n";
        //            foreach ($col_w_value as $key => $kernel_h_value) {
        //                #echo "kernel_h=$key\n";
        //                foreach ($kernel_h_value as $key => $kernel_w_value) {
        //                    #echo "kernel_w=$key\n";
        //                    echo "[";
        //                    foreach ($kernel_w_value as $key => $channel_value) {
        //                            echo sprintf('%2d',$channel_value).",";
        //                    }
        //                    echo "],";
        //                }
        //                echo "\n";
        //            }
        //        }
        //    }
        //}
        //$this->assertTrue(false);
        //$this->assertEquals(
        //[[
        //  [
        //   [[[ 0, 1, 2],[ 3, 4, 5],[ 6, 7, 8]],
        //    [[12,13,14],[15,16,17],[18,19,20]],
        //    [[24,25,26],[27,28,29],[30,31,32]],],
        //   [[[ 3, 4, 5],[ 6, 7, 8],[ 9,10,11]],
        //    [[15,16,17],[18,19,20],[21,22,23]],
        //    [[27,28,29],[30,31,32],[33,34,35]],],
        //  ],
        //  [
        //   [[[12,13,14],[15,16,17],[18,19,20]],
        //    [[24,25,26],[27,28,29],[30,31,32]],
        //    [[36,37,38],[39,40,41],[42,43,44]],],
        //   [[[15,16,17],[18,19,20],[21,22,23]],
        //    [[27,28,29],[30,31,32],[33,34,35]],
        //    [[39,40,41],[42,43,44],[45,46,47]],],
        //  ],
        //]],
        //$cols->toArray()
        //);

        $newImages = $la->zerosLike($images);
        $la->col2im(
            $cols,
            $newImages,
            $filterSize=[
                $kernel_h,$kernel_w],
            $strides=[
                $stride_h,$stride_w],
            $padding,
            $channels_first,
            $dilation_rate=[
                $dilation_h,$dilation_w],
            $cols_channels_first
        );
        // result is Not equal to original
        // because to sum for back propagation
        //$this->assertEquals(
        //    $images->toArray(),
        //    $newImages->toArray()
        //);
        //foreach ($newImages->toArray() as $batch) {
        //    foreach ($batch as $key => $im_y) {
        //        #echo "kernel_h=$key\n";
        //        foreach ($im_y as $key => $im_x) {
        //            #echo "kernel_w=$key\n";
        //            echo "[";
        //            foreach ($im_x as $key => $channel) {
        //                    echo sprintf('%3d',$channel).",";
        //            }
        //            echo "],";
        //        }
        //        echo "\n";
        //    }
        //}
        // channels_first
        //for($batch_id=0;$batch_id<$batches;$batch_id++) {
        //    for($channel_id=0;$channel_id<$channels;$channel_id++) {
        //        echo "channel=$channel_id\n";
        //        for($im_y=0;$im_y<$im_h;$im_y++) {
        //            echo "[";
        //            for($im_x=0;$im_x<$im_w;$im_x++) {
        //                echo sprintf('%3d',intval($newImages[$batch_id][$im_y][$im_x][$channel_id]->toArray())).",";
        //            }
        //            echo "],\n";
        //        }
        //        echo "\n";
        //    }
        //}

        $imagesTrues = $this->newArray($images->shape());
        $imageBuffer = $imagesTrues->buffer();
        for($batch_id=0;$batch_id<$batches;$batch_id++) {
            for($channel_id=0;$channel_id<$channels;$channel_id++) {
                for($im_y=0;$im_y<$out_h;$im_y++) {
                    for($im_x=0;$im_x<$out_w;$im_x++) {
                        for($kernel_y=0;$kernel_y<$kernel_h;$kernel_y++) {
                            for($kernel_x=0;$kernel_x<$kernel_w;$kernel_x++) {
                                $input_y = $im_y*$stride_h+$kernel_y*$dilation_h-$padding_h;
                                $input_x = $im_x*$stride_w+$kernel_x*$dilation_w-$padding_w;
                                if($channels_first) {
                                    $input_id = ((($batch_id*$channels+$channel_id)*$im_h+$input_y)*$im_w+$input_x);
                                } else {
                                    $input_id = ((($batch_id*$im_h+$input_y)*$im_w+$input_x)*$channels+$channel_id);
                                }
                                if($cols_channels_first) {
                                    $cols_id = ((((($batch_id*$out_h+$im_y)*$out_w+$im_x)
                                                *$channels+$channel_id)*$kernel_h+$kernel_y)*$kernel_w+$kernel_x);
                                } else {
                                    $cols_id = ((((($batch_id*$out_h+$im_y)*$out_w+$im_x)
                                                *$kernel_h+$kernel_y)*$kernel_w+$kernel_x)*$channels+$channel_id);
                                }
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
        //echo "==== reverse trues ======\n";
        //foreach ($imagesTrues->toArray() as $batch) {
        //    foreach ($batch as $key => $im_y) {
        //        #echo "kernel_h=$key\n";
        //        foreach ($im_y as $key => $im_x) {
        //            #echo "kernel_w=$key\n";
        //            echo "[";
        //            foreach ($im_x as $key => $channel) {
        //                    echo sprintf('%3d',$channel).",";
        //            }
        //            echo "],";
        //        }
        //        echo "\n";
        //    }
        //}
    }

    public function testIm2col2dLarge()
    {
        $mo = $this->newMatrixOperator();
        $la = $this->newLA($mo);
        if(!$la->accelerated()) {
            $this->markTestSkipped('Skip due to high load');
            return;
        }

        $batches = 1;
        $im_h = 1024;
        $im_w = 1024;
        $channels = 3;
        $kernel_h = 3;
        $kernel_w = 3;
        $stride_h = 1;
        $stride_w = 1;
        $padding = null;
        $channels_first = null;
        $dilation_h = 1;
        $dilation_w = 1;
        $cols_channels_first=null;
        $cols = null;

        $images = $la->array($mo->arange(
            $batches*
            $im_h*$im_w*
            $channels,
            null,null,
            NDArray::float32
        ));
        //$images = $la->array($mo->ones(
        //    [$batches*
        //    $im_h*$im_w*
        //    $channels],
        //    NDArray::float32
        //));
        if($channels_first) {
            $images = $images->reshape([
                $batches,
                $channels,
                $im_h,
                $im_w
            ]);
        } else {
            $images = $images->reshape([
                $batches,
                $im_h,
                $im_w,
                $channels
            ]);
        }
        $cols = $la->im2col(
            $images,
            $filterSize=[
                $kernel_h,$kernel_w],
            $strides=[
                $stride_h,$stride_w],
            $padding,
            $channels_first,
            $dilation_rate=[
                $dilation_h,$dilation_w],
            $cols_channels_first
        );
        $out_h = intval(floor(($im_h-($kernel_h-1)*$dilation_h-1)/$stride_h)+1);
        $out_w = intval(floor(($im_w-($kernel_w-1)*$dilation_w-1)/$stride_w)+1);
        //var_dump($cols->shape());
        #echo "===== cols =====\n";
        #echo $mo->toString($cols->toNDArray(),'%2d',true);
        $newImages = $la->zerosLike($images);
        $la->col2im(
            $cols,
            $newImages,
            $filterSize=[
                $kernel_h,$kernel_w],
            $strides=[
                $stride_h,$stride_w],
            $padding,
            $channels_first,
            $dilation_rate=[
                $dilation_h,$dilation_w],
            $cols_channels_first
        );
        $this->assertTrue(true);
    }

    public function testIm2col2dForPool()
    {
        $mo = $this->newMatrixOperator();
        $la = $this->newLA($mo);

        $batches = 1;
        $im_h = 4;
        $im_w = 4;
        $channels = 3;
        $kernel_h = 2;
        $kernel_w = 2;
        $stride_h = 2;
        $stride_w = 2;
        $padding = null;
        $channels_first = null;
        $dilation_h = 1;
        $dilation_w = 1;
        $cols_channels_first=true;
        $cols = null;

        $images = $la->array($mo->arange(
            $batches*
            $im_h*$im_w*
            $channels,
            null,null,
            NDArray::float32
        ))->reshape([
            $batches,
            $im_h,
            $im_w,
            $channels
        ]);
        $cols = $la->im2col(
            $images,
            $filterSize=[
                $kernel_h,$kernel_w],
            $strides=[
                $stride_h,$stride_w],
            $padding,
            $channels_first,
            $dilation_rate=[
                $dilation_h,$dilation_w],
            $cols_channels_first
        );
        $out_h = intval(floor(($im_h-($kernel_h-1)*$dilation_h-1)/$stride_h)+1);
        $out_w = intval(floor(($im_w-($kernel_w-1)*$dilation_w-1)/$stride_w)+1);
        if($padding) {
            $out_h = $im_h;
            $out_w = $im_w;
        }

        $this->assertEquals(
            [
                $batches,
                $out_h,$out_w,
                $channels,
                $kernel_h,$kernel_w,
            ],
            $cols->shape()
        );
        $this->assertEquals(
        [[
          [
           [[[0,3],[12,15]],
            [[1,4],[13,16]],
            [[2,5],[14,17]],],
           [[[6,9],[18,21]],
            [[7,10],[19,22]],
            [[8,11],[20,23]],],
          ],
          [
           [[[24,27],[36,39]],
            [[25,28],[37,40]],
            [[26,29],[38,41]],],
           [[[30,33],[42,45]],
            [[31,34],[43,46]],
            [[32,35],[44,47]],],
          ],
        ]],
        $cols->toArray()
        );

        $newImages = $la->zerosLike($images);
        $la->col2im(
            $cols,
            $newImages,
            $filterSize=[
                $kernel_h,$kernel_w],
            $strides=[
                $stride_h,$stride_w],
            $padding,
            $channels_first,
            $dilation_rate=[
                $dilation_h,$dilation_w],
            $cols_channels_first
        );

        // result is Not equal to original
        // because to sum for back propagation
        //$this->assertEquals(
        //    $images->toArray(),
        //    $newImages->toArray()
        //);
    }

    public function providerIm2col1dNormal()
    {
        return [
            'normal' => [[
                'batches' => 2,
                'im_w' => 8,
                'channels' => 3,
                'kernel_w' => 1,
                'stride_w' => 1,
                'padding' => null,
                'dilation_w' => 1,
                'channels_first' => null,
                'cols_channels_first' => null,
            ]],
            'kernel_w' => [[
                'batches' => 2,
                'im_w' => 8,
                'channels' => 3,
                'kernel_w' => 2,
                'stride_w' => 1,
                'padding' => null,
                'dilation_w' => 1,
                'channels_first' => null,
                'cols_channels_first' => null,
            ]],
            'stride_w' => [[
                'batches' => 2,
                'im_w' => 8,
                'channels' => 3,
                'kernel_w' => 1,
                'stride_w' => 2,
                'padding' => null,
                'dilation_w' => 1,
                'channels_first' => null,
                'cols_channels_first' => null,
            ]],
            'dilation_w' => [[
                'batches' => 2,
                'im_w' => 8,
                'channels' => 3,
                'kernel_w' => 1,
                'stride_w' => 1,
                'padding' => null,
                'dilation_w' => 2,
                'channels_first' => null,
                'cols_channels_first' => null,
            ]],
            'normal padding' => [[
                'batches' => 2,
                'im_w' => 8,
                'channels' => 3,
                'kernel_w' => 1,
                'stride_w' => 1,
                'padding' => true,
                'dilation_w' => 1,
                'channels_first' => null,
                'cols_channels_first' => null,
            ]],
            'kernel_w padding' => [[
                'batches' => 2,
                'im_w' => 8,
                'channels' => 3,
                'kernel_w' => 2,
                'stride_w' => 1,
                'padding' => true,
                'dilation_w' => 1,
                'channels_first' => null,
                'cols_channels_first' => null,
            ]],
            'stride_w padding' => [[
                'batches' => 2,
                'im_w' => 8,
                'channels' => 3,
                'kernel_w' => 1,
                'stride_w' => 2,
                'padding' => true,
                'dilation_w' => 1,
                'channels_first' => null,
                'cols_channels_first' => null,
            ]],
            'dilation_w padding' => [[
                'batches' => 2,
                'im_w' => 8,
                'channels' => 3,
                'kernel_w' => 1,
                'stride_w' => 1,
                'padding' => true,
                'dilation_w' => 2,
                'channels_first' => null,
                'cols_channels_first' => null,
            ]],
            'normal channels_first' => [[
                'batches' => 2,
                'im_w' => 8,
                'channels' => 3,
                'kernel_w' => 1,
                'stride_w' => 1,
                'padding' => null,
                'dilation_w' => 1,
                'channels_first' => true,
                'cols_channels_first' => null,
            ]],
            'kernel_w channels_first' => [[
                'batches' => 2,
                'im_w' => 8,
                'channels' => 3,
                'kernel_w' => 2,
                'stride_w' => 1,
                'padding' => null,
                'dilation_w' => 1,
                'channels_first' => true,
                'cols_channels_first' => null,
            ]],
            'stride_w channels_first' => [[
                'batches' => 2,
                'im_w' => 8,
                'channels' => 3,
                'kernel_w' => 1,
                'stride_w' => 2,
                'padding' => null,
                'dilation_w' => 1,
                'channels_first' => true,
                'cols_channels_first' => null,
            ]],
            'dilation_w channels_first' => [[
                'batches' => 2,
                'im_w' => 8,
                'channels' => 3,
                'kernel_w' => 1,
                'stride_w' => 1,
                'padding' => null,
                'dilation_w' => 2,
                'channels_first' => true,
                'cols_channels_first' => null,
            ]],
            'normal cols_channels_first' => [[
                'batches' => 2,
                'im_w' => 8,
                'channels' => 3,
                'kernel_w' => 1,
                'stride_w' => 1,
                'padding' => null,
                'dilation_w' => 1,
                'channels_first' => null,
                'cols_channels_first' => true,
            ]],
            'kernel_w cols_channels_first' => [[
                'batches' => 2,
                'im_w' => 8,
                'channels' => 3,
                'kernel_w' => 2,
                'stride_w' => 1,
                'padding' => null,
                'dilation_w' => 1,
                'channels_first' => null,
                'cols_channels_first' => true,
            ]],
            'stride_w cols_channels_first' => [[
                'batches' => 2,
                'im_w' => 8,
                'channels' => 3,
                'kernel_w' => 1,
                'stride_w' => 2,
                'padding' => null,
                'dilation_w' => 1,
                'channels_first' => null,
                'cols_channels_first' => true,
            ]],
            'dilation_w cols_channels_first' => [[
                'batches' => 2,
                'im_w' => 8,
                'channels' => 3,
                'kernel_w' => 1,
                'stride_w' => 1,
                'padding' => null,
                'dilation_w' => 2,
                'channels_first' => null,
                'cols_channels_first' => true,
            ]],
        ];
    }
    /**
    * @dataProvider providerIm2col1dNormal
    */
    public function testIm2col1dNormal($params)
    {
        $mo = $this->newMatrixOperator();
        $la = $this->newLA($mo);

        extract($params);

        //$batches = 1;
        //$im_w = 4;
        //$channels = 3;
        //$kernel_w = 3;
        //$stride_w = 1;
        //$padding = null;
        //$dilation_w = 1;
        //$channels_first = null;
        //$cols_channels_first = null;
        $cols = null;

        $images = $la->array($mo->arange(
            $batches*
            $im_w*
            $channels,
            null,null,
            NDArray::float32
        ));
        if($channels_first) {
            $images = $images->reshape([
                $batches,
                $channels,
                $im_w,
            ]);
        } else {
            $images = $images->reshape([
                $batches,
                $im_w,
                $channels,
            ]);
        }
        $cols = $la->im2col(
            $images,
            $filterSize=[
                $kernel_w],
            $strides=[
                $stride_w],
            $padding,
            $channels_first,
            $dilation_rate=[
                $dilation_w],
            $cols_channels_first
        );
        $out_w = intval(floor(($im_w-($kernel_w-1)*$dilation_w-1)/$stride_w)+1);
        if($padding) {
            $padding_w = (int)floor((($im_w-1)*$stride_w-$im_w+($kernel_w-1)*$dilation_w+1)/2);
            $out_w = $im_w;
        } else {
            $padding_w = 0;
        }

        if($cols_channels_first) {
            $this->assertEquals(
                [
                    $batches,
                    $out_w,
                    $channels,
                    $kernel_w,
                ],
                $cols->shape()
            );
        } else {
            $this->assertEquals(
                [
                    $batches,
                    $out_w,
                    $kernel_w,
                    $channels,
                ],
                $cols->shape()
            );
        }
        //$this->assertEquals(
        //[[
        //   [[0,1,2],[3,4,5],[6,7,8]],
        //   [[3,4,5],[6,7,8],[9,10,11]],
        //]],
        //$cols->toArray()
        //);
        $trues = $this->newArray($cols->shape());
        $truesBuffer = $trues->buffer();
        for($batch_id=0;$batch_id<$batches;$batch_id++) {
            for($channel_id=0;$channel_id<$channels;$channel_id++) {
                for($kernel_x=0;$kernel_x<$kernel_w;$kernel_x++) {
                    for($im_x=0;$im_x<$out_w;$im_x++) {
                        $input_x = $im_x*$stride_w+$kernel_x*$dilation_w-$padding_w;
                        if($channels_first) {
                            $input_id = (($batch_id*$channels+$channel_id)*$im_w+$input_x);
                        } else {
                            $input_id = (($batch_id*$im_w+$input_x)*$channels+$channel_id);
                        }
                        if($cols_channels_first) {
                            $cols_id = ((($batch_id*$out_w+$im_x)
                                *$channels+$channel_id)*$kernel_w+$kernel_x);
                        } else {
                            $cols_id = ((($batch_id*$out_w+$im_x)
                                *$kernel_w+$kernel_x)*$channels+$channel_id);
                        }
                        if($input_x>=0 && $input_x<$im_w) {
                            $truesBuffer[$cols_id] = $input_id;
                        }
                    }
                }
            }
        }
        $this->assertEquals($trues->toArray(),$cols->toArray());

        $newImages = $la->zerosLike($images);
        $la->col2im(
            $cols,
            $newImages,
            $filterSize=[
                $kernel_w],
            $strides=[
                $stride_w],
            $padding,
            $channels_first,
            $dilation_rate=[
                $dilation_w],
            $cols_channels_first
        );

        // result is Not equal to original
        // because to sum for back propagation
        //$this->assertEquals(
        //    $images->toArray(),
        //    $newImages->toArray()
        //);

        $imagesTrues = $this->newArray($images->shape());
        $imageBuffer = $imagesTrues->buffer();
        for($batch_id=0;$batch_id<$batches;$batch_id++) {
            for($channel_id=0;$channel_id<$channels;$channel_id++) {
                for($kernel_x=0;$kernel_x<$kernel_w;$kernel_x++) {
                    for($im_x=0;$im_x<$out_w;$im_x++) {
                        $input_x = $im_x*$stride_w+$kernel_x*$dilation_w-$padding_w;
                        if($channels_first) {
                            $input_id = (($batch_id*$channels+$channel_id)*$im_w+$input_x);
                        } else {
                            $input_id = (($batch_id*$im_w+$input_x)*$channels+$channel_id);
                        }
                        if($cols_channels_first) {
                            $cols_id = ((($batch_id*$out_w+$im_x)
                                *$channels+$channel_id)*$kernel_w+$kernel_x);
                        } else {
                            $cols_id = ((($batch_id*$out_w+$im_x)
                                *$kernel_w+$kernel_x)*$channels+$channel_id);
                        }
                        if($input_x>=0 && $input_x<$im_w) {
                            $value = $imageBuffer[$input_id];
                            $imageBuffer[$input_id] = $value + $truesBuffer[$cols_id];
                        }
                    }
                }
            }
        }
        $this->assertEquals($imagesTrues->toArray(),$newImages->toArray());
    }

    public function testIm2col1dForPool()
    {
        $mo = $this->newMatrixOperator();
        $la = $this->newLA($mo);

        $batches = 1;
        $im_w = 4;
        $channels = 3;
        $kernel_w = 2;
        $stride_w = 2;
        $padding = null;
        $channels_first = null;
        $dilation_w = 1;
        $cols_channels_first=true;
        $cols = null;

        $images = $la->array($mo->arange(
            $batches*
            $im_w*
            $channels,
            null,null,
            NDArray::float32
        ))->reshape([
            $batches,
            $im_w,
            $channels
        ]);
        $cols = $la->im2col(
            $images,
            $filterSize=[
                $kernel_w],
            $strides=[
                $stride_w],
            $padding,
            $channels_first,
            $dilation_rate=[
                $dilation_w],
            $cols_channels_first
        );
        $out_w = 2;

        $this->assertEquals(
            [
                $batches,
                $out_w,
                $channels,
                $kernel_w,
            ],
            $cols->shape()
        );
        $this->assertEquals(
          [
           [[[0,3],[1,4],[2,5]],
            [[6,9],[7,10],[8,11]]],
          ],
        $cols->toArray()
        );

        $newImages = $la->zerosLike($images);
        $la->col2im(
            $cols,
            $newImages,
            $filterSize=[
                $kernel_w],
            $strides=[
                $stride_w],
            $padding,
            $channels_first,
            $dilation_rate=[
                $dilation_w],
            $cols_channels_first
        );

        // result is Not equal to original
        // because to sum for back propagation
        //$this->assertEquals(
        //    $images->toArray(),
        //    $newImages->toArray()
        //);
    }

    public function providerIm2col3dNormal()
    {
        return [
            'normal' => [[
                'batches' => 2,
                'im_d' => 8,
                'im_h' => 8,
                'im_w' => 8,
                'channels' => 3,
                'kernel_d' => 3,
                'kernel_h' => 3,
                'kernel_w' => 3,
                'stride_d' => 1,
                'stride_h' => 1,
                'stride_w' => 1,
                'padding' => null,
                'channels_first' => null,
                'dilation_d' => 1,
                'dilation_h' => 1,
                'dilation_w' => 1,
                'cols_channels_first' => null,
            ]],
            'kernel_d' => [[
                'batches' => 2,
                'im_d' => 8,
                'im_h' => 8,
                'im_w' => 8,
                'channels' => 3,
                'kernel_d' => 4,
                'kernel_h' => 3,
                'kernel_w' => 3,
                'stride_d' => 1,
                'stride_h' => 1,
                'stride_w' => 1,
                'padding' => null,
                'channels_first' => null,
                'dilation_d' => 1,
                'dilation_h' => 1,
                'dilation_w' => 1,
                'cols_channels_first' => null,
            ]],
            'kernel_h' => [[
                'batches' => 2,
                'im_d' => 8,
                'im_h' => 8,
                'im_w' => 8,
                'channels' => 3,
                'kernel_d' => 3,
                'kernel_h' => 4,
                'kernel_w' => 3,
                'stride_d' => 1,
                'stride_h' => 1,
                'stride_w' => 1,
                'padding' => null,
                'channels_first' => null,
                'dilation_d' => 1,
                'dilation_h' => 1,
                'dilation_w' => 1,
                'cols_channels_first' => null,
            ]],
            'kernel_w' => [[
                'batches' => 2,
                'im_d' => 8,
                'im_h' => 8,
                'im_w' => 8,
                'channels' => 3,
                'kernel_d' => 3,
                'kernel_h' => 3,
                'kernel_w' => 4,
                'stride_d' => 1,
                'stride_h' => 1,
                'stride_w' => 1,
                'padding' => null,
                'channels_first' => null,
                'dilation_d' => 1,
                'dilation_h' => 1,
                'dilation_w' => 1,
                'cols_channels_first' => null,
            ]],
            'stride_d' => [[
                'batches' => 2,
                'im_d' => 8,
                'im_h' => 8,
                'im_w' => 8,
                'channels' => 3,
                'kernel_d' => 3,
                'kernel_h' => 3,
                'kernel_w' => 3,
                'stride_d' => 2,
                'stride_h' => 1,
                'stride_w' => 1,
                'padding' => null,
                'channels_first' => null,
                'dilation_d' => 1,
                'dilation_h' => 1,
                'dilation_w' => 1,
                'cols_channels_first' => null,
            ]],
            'stride_h' => [[
                'batches' => 2,
                'im_d' => 8,
                'im_h' => 8,
                'im_w' => 8,
                'channels' => 3,
                'kernel_d' => 3,
                'kernel_h' => 3,
                'kernel_w' => 3,
                'stride_d' => 1,
                'stride_h' => 2,
                'stride_w' => 1,
                'padding' => null,
                'channels_first' => null,
                'dilation_d' => 1,
                'dilation_h' => 1,
                'dilation_w' => 1,
                'cols_channels_first' => null,
            ]],
            'stride_w' => [[
                'batches' => 2,
                'im_d' => 8,
                'im_h' => 8,
                'im_w' => 8,
                'channels' => 3,
                'kernel_d' => 3,
                'kernel_h' => 3,
                'kernel_w' => 3,
                'stride_d' => 1,
                'stride_h' => 1,
                'stride_w' => 2,
                'padding' => null,
                'channels_first' => null,
                'dilation_d' => 1,
                'dilation_h' => 1,
                'dilation_w' => 1,
                'cols_channels_first' => null,
            ]],
            'dilation_d' => [[
                'batches' => 2,
                'im_d' => 8,
                'im_h' => 8,
                'im_w' => 8,
                'channels' => 3,
                'kernel_d' => 3,
                'kernel_h' => 3,
                'kernel_w' => 3,
                'stride_d' => 1,
                'stride_h' => 1,
                'stride_w' => 1,
                'padding' => null,
                'channels_first' => null,
                'dilation_d' => 2,
                'dilation_h' => 1,
                'dilation_w' => 1,
                'cols_channels_first' => null,
            ]],
            'dilation_h' => [[
                'batches' => 2,
                'im_d' => 8,
                'im_h' => 8,
                'im_w' => 8,
                'channels' => 3,
                'kernel_d' => 3,
                'kernel_h' => 3,
                'kernel_w' => 3,
                'stride_d' => 1,
                'stride_h' => 1,
                'stride_w' => 1,
                'padding' => null,
                'channels_first' => null,
                'dilation_d' => 1,
                'dilation_h' => 2,
                'dilation_w' => 1,
                'cols_channels_first' => null,
            ]],
            'dilation_w' => [[
                'batches' => 2,
                'im_d' => 8,
                'im_h' => 8,
                'im_w' => 8,
                'channels' => 3,
                'kernel_d' => 3,
                'kernel_h' => 3,
                'kernel_w' => 3,
                'stride_d' => 1,
                'stride_h' => 1,
                'stride_w' => 1,
                'padding' => null,
                'channels_first' => null,
                'dilation_d' => 1,
                'dilation_h' => 1,
                'dilation_w' => 2,
                'cols_channels_first' => null,
            ]],
            'normal channels_first' => [[
                'batches' => 2,
                'im_d' => 8,
                'im_h' => 8,
                'im_w' => 8,
                'channels' => 3,
                'kernel_d' => 3,
                'kernel_h' => 3,
                'kernel_w' => 3,
                'stride_d' => 1,
                'stride_h' => 1,
                'stride_w' => 1,
                'padding' => null,
                'channels_first' => true,
                'dilation_d' => 1,
                'dilation_h' => 1,
                'dilation_w' => 1,
                'cols_channels_first' => null,
            ]],
            'kernel_d channels_first' => [[
                'batches' => 2,
                'im_d' => 8,
                'im_h' => 8,
                'im_w' => 8,
                'channels' => 3,
                'kernel_d' => 4,
                'kernel_h' => 3,
                'kernel_w' => 3,
                'stride_d' => 1,
                'stride_h' => 1,
                'stride_w' => 1,
                'padding' => null,
                'channels_first' => true,
                'dilation_d' => 1,
                'dilation_h' => 1,
                'dilation_w' => 1,
                'cols_channels_first' => null,
            ]],
            'kernel_h channels_first' => [[
                'batches' => 2,
                'im_d' => 8,
                'im_h' => 8,
                'im_w' => 8,
                'channels' => 3,
                'kernel_d' => 3,
                'kernel_h' => 4,
                'kernel_w' => 3,
                'stride_d' => 1,
                'stride_h' => 1,
                'stride_w' => 1,
                'padding' => null,
                'channels_first' => true,
                'dilation_d' => 1,
                'dilation_h' => 1,
                'dilation_w' => 1,
                'cols_channels_first' => null,
            ]],
            'kernel_w channels_first' => [[
                'batches' => 2,
                'im_d' => 8,
                'im_h' => 8,
                'im_w' => 8,
                'channels' => 3,
                'kernel_d' => 3,
                'kernel_h' => 3,
                'kernel_w' => 4,
                'stride_d' => 1,
                'stride_h' => 1,
                'stride_w' => 1,
                'padding' => null,
                'channels_first' => true,
                'dilation_d' => 1,
                'dilation_h' => 1,
                'dilation_w' => 1,
                'cols_channels_first' => null,
            ]],
            'stride_d channels_first' => [[
                'batches' => 2,
                'im_d' => 8,
                'im_h' => 8,
                'im_w' => 8,
                'channels' => 3,
                'kernel_d' => 3,
                'kernel_h' => 3,
                'kernel_w' => 3,
                'stride_d' => 2,
                'stride_h' => 1,
                'stride_w' => 1,
                'padding' => null,
                'channels_first' => true,
                'dilation_d' => 1,
                'dilation_h' => 1,
                'dilation_w' => 1,
                'cols_channels_first' => null,
            ]],
            'stride_h channels_first' => [[
                'batches' => 2,
                'im_d' => 8,
                'im_h' => 8,
                'im_w' => 8,
                'channels' => 3,
                'kernel_d' => 3,
                'kernel_h' => 3,
                'kernel_w' => 3,
                'stride_d' => 1,
                'stride_h' => 2,
                'stride_w' => 1,
                'padding' => null,
                'channels_first' => true,
                'dilation_d' => 1,
                'dilation_h' => 1,
                'dilation_w' => 1,
                'cols_channels_first' => null,
            ]],
            'stride_w channels_first' => [[
                'batches' => 2,
                'im_d' => 8,
                'im_h' => 8,
                'im_w' => 8,
                'channels' => 3,
                'kernel_d' => 3,
                'kernel_h' => 3,
                'kernel_w' => 3,
                'stride_d' => 1,
                'stride_h' => 1,
                'stride_w' => 2,
                'padding' => null,
                'channels_first' => true,
                'dilation_d' => 1,
                'dilation_h' => 1,
                'dilation_w' => 1,
                'cols_channels_first' => null,
            ]],
            'dilation_d channels_first' => [[
                'batches' => 2,
                'im_d' => 8,
                'im_h' => 8,
                'im_w' => 8,
                'channels' => 3,
                'kernel_d' => 3,
                'kernel_h' => 3,
                'kernel_w' => 3,
                'stride_d' => 1,
                'stride_h' => 1,
                'stride_w' => 1,
                'padding' => null,
                'channels_first' => true,
                'dilation_d' => 2,
                'dilation_h' => 1,
                'dilation_w' => 1,
                'cols_channels_first' => null,
            ]],
            'dilation_h channels_first' => [[
                'batches' => 2,
                'im_d' => 8,
                'im_h' => 8,
                'im_w' => 8,
                'channels' => 3,
                'kernel_d' => 3,
                'kernel_h' => 3,
                'kernel_w' => 3,
                'stride_d' => 1,
                'stride_h' => 1,
                'stride_w' => 1,
                'padding' => null,
                'channels_first' => true,
                'dilation_d' => 1,
                'dilation_h' => 2,
                'dilation_w' => 1,
                'cols_channels_first' => null,
            ]],
            'dilation_w channels_first' => [[
                'batches' => 2,
                'im_d' => 8,
                'im_h' => 8,
                'im_w' => 8,
                'channels' => 3,
                'kernel_d' => 3,
                'kernel_h' => 3,
                'kernel_w' => 3,
                'stride_d' => 1,
                'stride_h' => 1,
                'stride_w' => 1,
                'padding' => null,
                'channels_first' => true,
                'dilation_d' => 1,
                'dilation_h' => 1,
                'dilation_w' => 2,
                'cols_channels_first' => null,
            ]],
            'normal padding' => [[
                'batches' => 2,
                'im_d' => 8,
                'im_h' => 8,
                'im_w' => 8,
                'channels' => 3,
                'kernel_d' => 3,
                'kernel_h' => 3,
                'kernel_w' => 3,
                'stride_d' => 1,
                'stride_h' => 1,
                'stride_w' => 1,
                'padding' => true,
                'channels_first' => null,
                'dilation_d' => 1,
                'dilation_h' => 1,
                'dilation_w' => 1,
                'cols_channels_first' => null,
            ]],
            'kernel_d padding' => [[
                'batches' => 2,
                'im_d' => 8,
                'im_h' => 8,
                'im_w' => 8,
                'channels' => 3,
                'kernel_d' => 4,
                'kernel_h' => 3,
                'kernel_w' => 3,
                'stride_d' => 1,
                'stride_h' => 1,
                'stride_w' => 1,
                'padding' => true,
                'channels_first' => null,
                'dilation_d' => 1,
                'dilation_h' => 1,
                'dilation_w' => 1,
                'cols_channels_first' => null,
            ]],
            'kernel_h padding' => [[
                'batches' => 2,
                'im_d' => 8,
                'im_h' => 8,
                'im_w' => 8,
                'channels' => 3,
                'kernel_d' => 3,
                'kernel_h' => 4,
                'kernel_w' => 3,
                'stride_d' => 1,
                'stride_h' => 1,
                'stride_w' => 1,
                'padding' => true,
                'channels_first' => null,
                'dilation_d' => 1,
                'dilation_h' => 1,
                'dilation_w' => 1,
                'cols_channels_first' => null,
            ]],
            'kernel_w padding' => [[
                'batches' => 2,
                'im_d' => 8,
                'im_h' => 8,
                'im_w' => 8,
                'channels' => 3,
                'kernel_d' => 3,
                'kernel_h' => 3,
                'kernel_w' => 4,
                'stride_d' => 1,
                'stride_h' => 1,
                'stride_w' => 1,
                'padding' => true,
                'channels_first' => null,
                'dilation_d' => 1,
                'dilation_h' => 1,
                'dilation_w' => 1,
                'cols_channels_first' => null,
            ]],
            'stride_d padding' => [[
                'batches' => 2,
                'im_d' => 8,
                'im_h' => 8,
                'im_w' => 8,
                'channels' => 3,
                'kernel_d' => 3,
                'kernel_h' => 3,
                'kernel_w' => 3,
                'stride_d' => 2,
                'stride_h' => 1,
                'stride_w' => 1,
                'padding' => true,
                'channels_first' => null,
                'dilation_d' => 1,
                'dilation_h' => 1,
                'dilation_w' => 1,
                'cols_channels_first' => null,
            ]],
            'stride_h padding' => [[
                'batches' => 2,
                'im_d' => 8,
                'im_h' => 8,
                'im_w' => 8,
                'channels' => 3,
                'kernel_d' => 3,
                'kernel_h' => 3,
                'kernel_w' => 3,
                'stride_d' => 1,
                'stride_h' => 2,
                'stride_w' => 1,
                'padding' => true,
                'channels_first' => null,
                'dilation_d' => 1,
                'dilation_h' => 1,
                'dilation_w' => 1,
                'cols_channels_first' => null,
            ]],
            'stride_w padding' => [[
                'batches' => 2,
                'im_d' => 8,
                'im_h' => 8,
                'im_w' => 8,
                'channels' => 3,
                'kernel_d' => 3,
                'kernel_h' => 3,
                'kernel_w' => 3,
                'stride_d' => 1,
                'stride_h' => 1,
                'stride_w' => 2,
                'padding' => true,
                'channels_first' => null,
                'dilation_d' => 1,
                'dilation_h' => 1,
                'dilation_w' => 1,
                'cols_channels_first' => null,
            ]],
            'dilation_d padding' => [[
                'batches' => 2,
                'im_d' => 8,
                'im_h' => 8,
                'im_w' => 8,
                'channels' => 3,
                'kernel_d' => 3,
                'kernel_h' => 3,
                'kernel_w' => 3,
                'stride_d' => 1,
                'stride_h' => 1,
                'stride_w' => 1,
                'padding' => true,
                'channels_first' => null,
                'dilation_d' => 2,
                'dilation_h' => 1,
                'dilation_w' => 1,
                'cols_channels_first' => null,
            ]],
            'dilation_h padding' => [[
                'batches' => 2,
                'im_d' => 8,
                'im_h' => 8,
                'im_w' => 8,
                'channels' => 3,
                'kernel_d' => 3,
                'kernel_h' => 3,
                'kernel_w' => 3,
                'stride_d' => 1,
                'stride_h' => 1,
                'stride_w' => 1,
                'padding' => true,
                'channels_first' => null,
                'dilation_d' => 1,
                'dilation_h' => 2,
                'dilation_w' => 1,
                'cols_channels_first' => null,
            ]],
            'dilation_w padding' => [[
                'batches' => 2,
                'im_d' => 8,
                'im_h' => 8,
                'im_w' => 8,
                'channels' => 3,
                'kernel_d' => 3,
                'kernel_h' => 3,
                'kernel_w' => 3,
                'stride_d' => 1,
                'stride_h' => 1,
                'stride_w' => 1,
                'padding' => true,
                'channels_first' => null,
                'dilation_d' => 1,
                'dilation_h' => 1,
                'dilation_w' => 2,
                'cols_channels_first' => null,
            ]],
            'normal cols_channels_first' => [[
                'batches' => 2,
                'im_d' => 8,
                'im_h' => 8,
                'im_w' => 8,
                'channels' => 3,
                'kernel_d' => 3,
                'kernel_h' => 3,
                'kernel_w' => 3,
                'stride_d' => 1,
                'stride_h' => 1,
                'stride_w' => 1,
                'padding' => null,
                'channels_first' => null,
                'dilation_d' => 1,
                'dilation_h' => 1,
                'dilation_w' => 1,
                'cols_channels_first' => true,
            ]],
            'kernel_d cols_channels_first' => [[
                'batches' => 2,
                'im_d' => 8,
                'im_h' => 8,
                'im_w' => 8,
                'channels' => 3,
                'kernel_d' => 4,
                'kernel_h' => 3,
                'kernel_w' => 3,
                'stride_d' => 1,
                'stride_h' => 1,
                'stride_w' => 1,
                'padding' => null,
                'channels_first' => null,
                'dilation_d' => 1,
                'dilation_h' => 1,
                'dilation_w' => 1,
                'cols_channels_first' => true,
            ]],
            'kernel_h cols_channels_first' => [[
                'batches' => 2,
                'im_d' => 8,
                'im_h' => 8,
                'im_w' => 8,
                'channels' => 3,
                'kernel_d' => 3,
                'kernel_h' => 4,
                'kernel_w' => 3,
                'stride_d' => 1,
                'stride_h' => 1,
                'stride_w' => 1,
                'padding' => null,
                'channels_first' => null,
                'dilation_d' => 1,
                'dilation_h' => 1,
                'dilation_w' => 1,
                'cols_channels_first' => true,
            ]],
            'kernel_w cols_channels_first' => [[
                'batches' => 2,
                'im_d' => 8,
                'im_h' => 8,
                'im_w' => 8,
                'channels' => 3,
                'kernel_d' => 3,
                'kernel_h' => 3,
                'kernel_w' => 4,
                'stride_d' => 1,
                'stride_h' => 1,
                'stride_w' => 1,
                'padding' => null,
                'channels_first' => null,
                'dilation_d' => 1,
                'dilation_h' => 1,
                'dilation_w' => 1,
                'cols_channels_first' => true,
            ]],
            'stride_d cols_channels_first' => [[
                'batches' => 2,
                'im_d' => 8,
                'im_h' => 8,
                'im_w' => 8,
                'channels' => 3,
                'kernel_d' => 3,
                'kernel_h' => 3,
                'kernel_w' => 3,
                'stride_d' => 2,
                'stride_h' => 1,
                'stride_w' => 1,
                'padding' => null,
                'channels_first' => null,
                'dilation_d' => 1,
                'dilation_h' => 1,
                'dilation_w' => 1,
                'cols_channels_first' => true,
            ]],
            'stride_h cols_channels_first' => [[
                'batches' => 2,
                'im_d' => 8,
                'im_h' => 8,
                'im_w' => 8,
                'channels' => 3,
                'kernel_d' => 3,
                'kernel_h' => 3,
                'kernel_w' => 3,
                'stride_d' => 1,
                'stride_h' => 2,
                'stride_w' => 1,
                'padding' => null,
                'channels_first' => null,
                'dilation_d' => 1,
                'dilation_h' => 1,
                'dilation_w' => 1,
                'cols_channels_first' => true,
            ]],
            'stride_w cols_channels_first' => [[
                'batches' => 2,
                'im_d' => 8,
                'im_h' => 8,
                'im_w' => 8,
                'channels' => 3,
                'kernel_d' => 3,
                'kernel_h' => 3,
                'kernel_w' => 3,
                'stride_d' => 1,
                'stride_h' => 1,
                'stride_w' => 2,
                'padding' => null,
                'channels_first' => null,
                'dilation_d' => 1,
                'dilation_h' => 1,
                'dilation_w' => 1,
                'cols_channels_first' => true,
            ]],
            'dilation_d cols_channels_first' => [[
                'batches' => 2,
                'im_d' => 8,
                'im_h' => 8,
                'im_w' => 8,
                'channels' => 3,
                'kernel_d' => 3,
                'kernel_h' => 3,
                'kernel_w' => 3,
                'stride_d' => 1,
                'stride_h' => 1,
                'stride_w' => 1,
                'padding' => null,
                'channels_first' => null,
                'dilation_d' => 2,
                'dilation_h' => 1,
                'dilation_w' => 1,
                'cols_channels_first' => true,
            ]],
            'dilation_h cols_channels_first' => [[
                'batches' => 2,
                'im_d' => 8,
                'im_h' => 8,
                'im_w' => 8,
                'channels' => 3,
                'kernel_d' => 3,
                'kernel_h' => 3,
                'kernel_w' => 3,
                'stride_d' => 1,
                'stride_h' => 1,
                'stride_w' => 1,
                'padding' => null,
                'channels_first' => null,
                'dilation_d' => 1,
                'dilation_h' => 2,
                'dilation_w' => 1,
                'cols_channels_first' => true,
            ]],
            'dilation_w cols_channels_first' => [[
                'batches' => 2,
                'im_d' => 8,
                'im_h' => 8,
                'im_w' => 8,
                'channels' => 3,
                'kernel_d' => 3,
                'kernel_h' => 3,
                'kernel_w' => 3,
                'stride_d' => 1,
                'stride_h' => 1,
                'stride_w' => 1,
                'padding' => null,
                'channels_first' => null,
                'dilation_d' => 1,
                'dilation_h' => 1,
                'dilation_w' => 2,
                'cols_channels_first' => true,
            ]],
        ];
    }

    /**
    * @dataProvider providerIm2col3dNormal
    */
    public function testIm2col3dNormal($params)
    {
        $mo = $this->newMatrixOperator();
        $la = $this->newLA($mo);

        extract($params);

        //$batches = 1;
        //$im_d = 5;
        //$im_h = 5;
        //$im_w = 5;
        //$channels = 3;
        //$kernel_d = 3;
        //$kernel_h = 3;
        //$kernel_w = 3;
        //$stride_d = 1;
        //$stride_h = 1;
        //$stride_w = 1;
        //$padding = null;
        //$channels_first = null;
        //$dilation_d = 2;
        //$dilation_h = 1;
        //$dilation_w = 1;
        //$cols_channels_first=null;
        $cols = null;

        $images = $la->array($mo->arange(
            $batches*
            $im_d*$im_h*$im_w*
            $channels,
            null,null,
            NDArray::float32
        ));
        if($channels_first) {
            $images = $images->reshape([
                $batches,
                $channels,
                $im_d,
                $im_h,
                $im_w,
            ]);
        } else {
            $images = $images->reshape([
                $batches,
                $im_d,
                $im_h,
                $im_w,
                $channels,
            ]);
        }
        $cols = $la->im2col(
            $images,
            $filterSize=[
                $kernel_d,$kernel_h,$kernel_w],
            $strides=[
                $stride_d,$stride_h,$stride_w],
            $padding,
            $channels_first,
            $dilation_rate=[
                $dilation_d,$dilation_h,$dilation_w],
            $cols_channels_first
        );
        $out_d = intval(floor(($im_d-($kernel_d-1)*$dilation_d-1)/$stride_d)+1);
        $out_h = intval(floor(($im_h-($kernel_h-1)*$dilation_h-1)/$stride_h)+1);
        $out_w = intval(floor(($im_w-($kernel_w-1)*$dilation_w-1)/$stride_w)+1);
        if($padding) {
            $padding_d = (int)floor((($im_d-1)*$stride_d-$im_d+($kernel_d-1)*$dilation_d+1)/2);
            $padding_h = (int)floor((($im_h-1)*$stride_h-$im_h+($kernel_h-1)*$dilation_h+1)/2);
            $padding_w = (int)floor((($im_w-1)*$stride_w-$im_w+($kernel_w-1)*$dilation_w+1)/2);
            $out_d = $im_d;
            $out_h = $im_h;
            $out_w = $im_w;
        } else {
            $padding_d = 0;
            $padding_h = 0;
            $padding_w = 0;
        }

        if($cols_channels_first) {
            $this->assertEquals(
                [
                    $batches,
                    $out_d,$out_h,$out_w,
                    $channels,
                    $kernel_d,$kernel_h,$kernel_w,
                ],
                $cols->shape()
            );
        } else {
            $this->assertEquals(
                [
                    $batches,
                    $out_d,$out_h,$out_w,
                    $kernel_d,$kernel_h,$kernel_w,
                    $channels,
                ],
                $cols->shape()
            );
        }
        $trues = $this->newArray($cols->shape());
        $truesBuffer = $trues->buffer();
        for($batch_id=0;$batch_id<$batches;$batch_id++) {
            for($channel_id=0;$channel_id<$channels;$channel_id++) {
                for($im_z=0;$im_z<$out_d;$im_z++) {
                    for($im_y=0;$im_y<$out_h;$im_y++) {
                        for($im_x=0;$im_x<$out_w;$im_x++) {
                            for($kernel_z=0;$kernel_z<$kernel_d;$kernel_z++) {
                                for($kernel_y=0;$kernel_y<$kernel_h;$kernel_y++) {
                                    for($kernel_x=0;$kernel_x<$kernel_w;$kernel_x++) {
                                        $input_z = $im_z*$stride_d+$kernel_z*$dilation_d-$padding_d;
                                        $input_y = $im_y*$stride_h+$kernel_y*$dilation_h-$padding_h;
                                        $input_x = $im_x*$stride_w+$kernel_x*$dilation_w-$padding_w;
                                        if($channels_first) {
                                            $input_id = (((($batch_id*$channels+$channel_id)*$im_d+$input_z)*$im_h+$input_y)*$im_w+$input_x);
                                        } else {
                                            $input_id = (((($batch_id*$im_d+$input_z)*$im_h+$input_y)*$im_w+$input_x)*$channels+$channel_id);
                                        }
                                        if($cols_channels_first) {
                                            $cols_id = ((((((($batch_id*$out_d+$im_z)*$out_h+$im_y)*$out_w+$im_x)
                                                        *$channels+$channel_id)*$kernel_d+$kernel_z)*$kernel_h+$kernel_y)*$kernel_w+$kernel_x);
                                        } else {
                                            $cols_id = ((((((($batch_id*$out_d+$im_z)*$out_h+$im_y)*$out_w+$im_x)
                                                        *$kernel_d+$kernel_z)*$kernel_h+$kernel_y)*$kernel_w+$kernel_x)*$channels+$channel_id);
                                        }
                                        if($input_z>=0 && $input_z<$im_d && $input_y>=0 && $input_y<$im_h && $input_x>=0 && $input_x<$im_w) {
                                            $truesBuffer[$cols_id] = $input_id;
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        $this->assertEquals($trues->toArray(),$cols->toArray());
        //$this->assertEquals(
        //    [],$cols->toArray()
        //);
        //$this->assertNotEquals(
        //    $la->zerosLike($cols)->toArray(),
        //    $cols->toArray()
        //);
        //foreach ($cols->toArray() as $batch) {
        //    foreach ($batch as $im_z => $im_z_value) {
        //        echo "im_z=$im_z\n";
        //        foreach ($im_z_value as $im_y => $im_y_value) {
        //            echo "im_y=$im_y\n";
        //            foreach ($im_y_value as $im_x => $im_x_value) {
        //                echo "im($im_z,$im_y,$im_x)\n";
        //                foreach ($im_x_value as $kernel_z => $kernel_z_value) {
        //                    foreach ($kernel_z_value as $kernel_y => $kernel_y_value) {
        //                        foreach ($kernel_y_value as $kernel_x => $kernel_x_value) {
        //                            echo "[";
        //                            foreach ($kernel_x_value as $channel_id => $channel_value) {
        //                                    echo sprintf('%3d',$channel_value).",";
        //                            }
        //                            echo "],";
        //                        }
        //                        echo "\n";
        //                    }
        //                    echo "\n";
        //                }
        //            }
        //        }
        //    }
        //}

        $newImages = $la->zerosLike($images);
        $la->col2im(
            $cols,
            $newImages,
            $filterSize=[
                $kernel_d,$kernel_h,$kernel_w],
            $strides=[
                $stride_d,$stride_h,$stride_w],
            $padding,
            $channels_first,
            $dilation_rate=[
                $dilation_d,$dilation_h,$dilation_w],
            $cols_channels_first
        );

        // result is Not equal to original
        // because to sum for back propagation
        //$this->assertEquals(
        //    $images->toArray(),
        //    $newImages->toArray()
        //);
        $imagesTrues = $this->newArray($images->shape());
        $imageBuffer = $imagesTrues->buffer();
        for($batch_id=0;$batch_id<$batches;$batch_id++) {
            for($channel_id=0;$channel_id<$channels;$channel_id++) {
                for($im_z=0;$im_z<$out_d;$im_z++) {
                    for($im_y=0;$im_y<$out_h;$im_y++) {
                        for($im_x=0;$im_x<$out_w;$im_x++) {
                            for($kernel_z=0;$kernel_z<$kernel_d;$kernel_z++) {
                                for($kernel_y=0;$kernel_y<$kernel_h;$kernel_y++) {
                                    for($kernel_x=0;$kernel_x<$kernel_w;$kernel_x++) {
                                        $input_z = $im_z*$stride_d+$kernel_z*$dilation_d-$padding_d;
                                        $input_y = $im_y*$stride_h+$kernel_y*$dilation_h-$padding_h;
                                        $input_x = $im_x*$stride_w+$kernel_x*$dilation_w-$padding_w;
                                        if($channels_first) {
                                            $input_id = (((($batch_id*$channels+$channel_id)*$im_d+$input_z)*$im_h+$input_y)*$im_w+$input_x);
                                        } else {
                                            $input_id = (((($batch_id*$im_d+$input_z)*$im_h+$input_y)*$im_w+$input_x)*$channels+$channel_id);
                                        }
                                        if($cols_channels_first) {
                                            $cols_id = ((((((($batch_id*$out_d+$im_z)*$out_h+$im_y)*$out_w+$im_x)
                                                        *$channels+$channel_id)*$kernel_d+$kernel_z)*$kernel_h+$kernel_y)*$kernel_w+$kernel_x);
                                        } else {
                                            $cols_id = ((((((($batch_id*$out_d+$im_z)*$out_h+$im_y)*$out_w+$im_x)
                                                        *$kernel_d+$kernel_z)*$kernel_h+$kernel_y)*$kernel_w+$kernel_x)*$channels+$channel_id);
                                        }
                                        if($input_z>=0 && $input_z<$im_d && $input_y>=0 && $input_y<$im_h && $input_x>=0 && $input_x<$im_w) {
                                            $value = $imageBuffer[$input_id];
                                            $imageBuffer[$input_id] = $value + $truesBuffer[$cols_id];
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        $this->assertEquals($imagesTrues->toArray(),$newImages->toArray());
    }

    public function testIm2col3dForPool()
    {
        $mo = $this->newMatrixOperator();
        $la = $this->newLA($mo);

        $batches = 1;
        $im_d = 4;
        $im_h = 4;
        $im_w = 4;
        $channels = 3;
        $kernel_d = 2;
        $kernel_h = 2;
        $kernel_w = 2;
        $stride_d = 2;
        $stride_h = 2;
        $stride_w = 2;
        $padding = null;
        $channels_first = null;
        $dilation_d = 1;
        $dilation_h = 1;
        $dilation_w = 1;
        $cols_channels_first=true;
        $cols = null;

        $images = $la->array($mo->arange(
            $batches*
            $im_d*$im_h*$im_w*
            $channels,
            null,null,
            NDArray::float32
        ))->reshape([
            $batches,
            $im_d,
            $im_h,
            $im_w,
            $channels
        ]);
        $cols = $la->im2col(
            $images,
            $filterSize=[
                $kernel_d,$kernel_h,$kernel_w],
            $strides=[
                $stride_d,$stride_h,$stride_w],
            $padding,
            $channels_first,
            $dilation_rate=[
                $dilation_d,$dilation_h,$dilation_w],
            $cols_channels_first
        );
        $out_d = intval(floor(($im_d-($kernel_d-1)*$dilation_d-1)/$stride_d)+1);
        $out_h = intval(floor(($im_h-($kernel_h-1)*$dilation_h-1)/$stride_h)+1);
        $out_w = intval(floor(($im_w-($kernel_w-1)*$dilation_w-1)/$stride_w)+1);

        $this->assertEquals(
            [
                $batches,
                $out_d,$out_h,$out_w,
                $channels,
                $kernel_d,$kernel_h,$kernel_w,
            ],
            $cols->shape()
        );
        //$this->assertEquals(
        //[],
        //$cols->toArray()
        //);
        $this->assertNotEquals(
            $la->zerosLike($cols)->toArray(),
            $cols->toArray()
        );

        $newImages = $la->zerosLike($images);
        $la->col2im(
            $cols,
            $newImages,
            $filterSize=[
                $kernel_d,$kernel_h,$kernel_w],
            $strides=[
                $stride_d,$stride_h,$stride_w],
            $padding,
            $channels_first,
            $dilation_rate=[
                $dilation_d,$dilation_h,$dilation_w],
            $cols_channels_first
        );

        // result is Not equal to original
        // because to sum for back propagation
        //$this->assertEquals(
        //    $images->toArray(),
        //    $newImages->toArray()
        //);
    }

    public function testIm2col2dSpeed()
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
        $batches = 8;
        $im_h = 512;
        $im_w = 512;
        $channels = 3;
        $images = $la->alloc([$batches,$im_h,$im_w,$channels]);
        $la->ones($images);
        $kernel_h = 3;
        $kernel_w = 3;
        $stride_h = 1;
        $stride_w = 1;
        $padding = null;
        $channels_first = null;
        $dilation_h = 1;
        $dilation_w = 1;
        $cols_channels_first=null;
        $cols = null;
        echo "im=($im_h,$im_w),knl=($kernel_h,$kernel_w),batches=$batches\n";

        $cols = $la->im2col(
            $images,
            $filterSize=[
                $kernel_h,$kernel_w],
            $strides=[
                $stride_h,$stride_w],
            $padding,
            $channels_first,
            $dilation_rate=[
                $dilation_h,$dilation_w],
            $cols_channels_first
        );
        $start = hrtime(true);
        $cols = $la->im2col(
            $images,
            $filterSize=[
                $kernel_h,$kernel_w],
            $strides=[
                $stride_h,$stride_w],
            $padding,
            $channels_first,
            $dilation_rate=[
                $dilation_h,$dilation_w],
            $cols_channels_first
        );
        $end = hrtime(true);
        echo (explode(' ',$la->getConfig()))[0].'='.number_format($end-$start)."\n";

        $newImages = $la->alloc([$batches,$im_h,$im_w,$channels]);
        $la->col2im(
            $cols,
            $newImages,
            $filterSize=[
                $kernel_h,$kernel_w],
            $strides=[
                $stride_h,$stride_w],
            $padding,
            $channels_first,
            $dilation_rate=[
                $dilation_h,$dilation_w],
            $cols_channels_first
        );
        $start = hrtime(true);
        $la->col2im(
            $cols,
            $newImages,
            $filterSize=[
                $kernel_h,$kernel_w],
            $strides=[
                $stride_h,$stride_w],
            $padding,
            $channels_first,
            $dilation_rate=[
                $dilation_h,$dilation_w],
            $cols_channels_first
        );
        $end = hrtime(true);
        echo (explode(' ',$la->getConfig()))[0].'='.number_format($end-$start)."\n";

        $batches = 256;
        $im_h = 28;
        $im_w = 28;
        $channels = 1;
        $images = $la->alloc([$batches,$im_h,$im_w,$channels]);
        $la->ones($images);
        $kernel_h = 3;
        $kernel_w = 3;
        $stride_h = 1;
        $stride_w = 1;
        $padding = null;
        $channels_first = null;
        $dilation_h = 1;
        $dilation_w = 1;
        $cols_channels_first=null;
        $cols = null;
        echo "im=($im_h,$im_w),knl=($kernel_h,$kernel_w),batches=$batches\n";

        $cols = $la->im2col(
            $images,
            $filterSize=[
                $kernel_h,$kernel_w],
            $strides=[
                $stride_h,$stride_w],
            $padding,
            $channels_first,
            $dilation_rate=[
                $dilation_h,$dilation_w],
            $cols_channels_first
        );
        $start = hrtime(true);
        $cols = $la->im2col(
            $images,
            $filterSize=[
                $kernel_h,$kernel_w],
            $strides=[
                $stride_h,$stride_w],
            $padding,
            $channels_first,
            $dilation_rate=[
                $dilation_h,$dilation_w],
            $cols_channels_first
        );
        $end = hrtime(true);
        echo (explode(' ',$la->getConfig()))[0].'='.number_format($end-$start)."\n";

        $newImages = $la->alloc([$batches,$im_h,$im_w,$channels]);
        $la->col2im(
            $cols,
            $newImages,
            $filterSize=[
                $kernel_h,$kernel_w],
            $strides=[
                $stride_h,$stride_w],
            $padding,
            $channels_first,
            $dilation_rate=[
                $dilation_h,$dilation_w],
            $cols_channels_first
        );
        $start = hrtime(true);
        $la->col2im(
            $cols,
            $newImages,
            $filterSize=[
                $kernel_h,$kernel_w],
            $strides=[
                $stride_h,$stride_w],
            $padding,
            $channels_first,
            $dilation_rate=[
                $dilation_h,$dilation_w],
            $cols_channels_first
        );
        $end = hrtime(true);
        echo (explode(' ',$la->getConfig()))[0].'='.number_format($end-$start)."\n";
        $this->assertTrue(true);
    }

    public function testRandomUniform()
    {
        $mo = $this->newMatrixOperator();
        $la = $this->newLA($mo);

        $x = $la->randomUniform(
            $shape=[20,30],
            $low=-1.0,
            $high=1.0);
        $y = $la->randomUniform(
            $shape=[20,30],
            $low=-1,
            $high=1);
        $this->assertEquals(
            NDArray::float32,$x->dtype());
        $this->assertNotEquals(
            $x->toArray(),
            $y->toArray());
        $this->assertLessThanOrEqual(1,$la->max($x));
        $this->assertGreaterThanOrEqual(-1,$la->min($x));

        $x = $la->randomUniform(
            $shape=[20,30],
            $low=-1,
            $high=1,
            $dtype=NDArray::int32
            );
        $y = $la->randomUniform(
            $shape=[20,30],
            $low=-1,
            $high=1,
            $dtype=NDArray::int32);
        $this->assertEquals(
            NDArray::int32,$x->dtype());
        $this->assertNotEquals(
            $x->toArray(),
            $y->toArray());
        $x = $la->astype($x,NDArray::float32);
        if(extension_loaded('rindow_openblas')) {
            $this->assertEquals(1,$la->max($x));
            $this->assertEquals(-1,$la->min($x));
        } else {
            $this->assertEquals(1,round($la->max($x)));
            $this->assertEquals(-1,round($la->min($x)));
        }
    }

    public function testRandomNormal()
    {
        $mo = $this->newMatrixOperator();
        $la = $this->newLA($mo);

        $x = $la->randomNormal(
            $shape=[20,30],
            $mean=0.0,
            $scale=1.0);
        $y = $la->randomNormal(
            $shape=[20,30],
            $mean=0.0,
            $scale=1.0);
        $this->assertEquals(
            NDArray::float32,$x->dtype());
        $this->assertNotEquals(
            $x->toArray(),
            $y->toArray());
        $this->assertLessThanOrEqual(5,$la->max($x));
        $this->assertGreaterThanOrEqual(-5,$la->min($x));
    }

    public function testRandomSequence()
    {
        $mo = $this->newMatrixOperator();
        $la = $this->newLA($mo);

        $x = $la->randomSequence(
            $base=500,
            $size=100
            );
        $y = $la->randomSequence(
            $base=500,
            $size=100
            );
        if($la->accelerated()) {
            $this->assertEquals(
                NDArray::int32,$x->dtype());
        } else {
            $this->assertEquals(
                NDArray::int64,$x->dtype());
        }
        $this->assertEquals(
            [100],$x->shape());
        $this->assertNotEquals(
            $x->toArray(),
            $y->toArray());
    }

    public function testSlice()
    {
        $mo = $this->newMatrixOperator();
        $la = $this->newLA($mo);

        // 3D
        $x = $la->array([
            [[0,1,2],
             [3,4,5],
             [6,7,8],
             [9,10,11]],
            [[12,13,14],
             [15,16,17],
             [18,19,20],
             [21,22,23]],
        ]);
        $this->assertEquals(3,$x->ndim());
        $y = $la->slice(
            $x,
            $start=[0,1],
            $size=[-1,2]
            );
        $this->assertEquals([
            [[3,4,5],
             [6,7,8],],
            [[15,16,17],
             [18,19,20],],
        ],$y->toArray());

        $y = $la->slice(
            $x,
            $start=[0,1],
            $size=[-1,1]
            );
        $this->assertEquals([
            [[3,4,5],],
            [[15,16,17],]
        ],$y->toArray());

        $y = $la->slice(
            $x,
            $start=[0,-1],
            $size=[-1,1]
            );
        $this->assertEquals([
            [[9,10,11],],
            [[21,22,23],]
        ],$y->toArray());

        $y = $la->slice(
            $x,
            $start=[1],
            $size=[1]
            );
        $this->assertEquals([
            [[12,13,14],
             [15,16,17],
             [18,19,20],
             [21,22,23],],
        ],$y->toArray());

        // 2D
        $x = $la->array($mo->arange(8,null,null,NDArray::float32)->reshape([2,4]));
        $this->assertEquals(2,$x->ndim());
        $x = $la->array([
            [0,1,2,3],
            [4,5,6,7],
        ]);
        $y = $la->slice(
            $x,
            $start=[0,1],
            $size=[-1,2]
            );
        $this->assertEquals([
            [1,2],
            [5,6]
        ],$y->toArray());

        $y = $la->slice(
            $x,
            $start=[0,0],
            $size=[2,4]
            );
        $this->assertEquals([
            [0,1,2,3],
            [4,5,6,7],
        ],$y->toArray());

        // 4D
        $x = $la->array([
            [[[0,1,2],
              [3,4,5]],
             [[6,7,8],
              [9,10,11]]],
            [[[12,13,14],
              [15,16,17]],
             [[18,19,20],
              [21,22,23]]],
        ]);
        $this->assertEquals(4,$x->ndim());
        $y = $la->slice(
            $x,
            $start=[1,1,1],
            $size=[-1,-1,-1]);
        $this->assertEquals([
            [[[21,22,23]]],
        ],$y->toArray());

        $y = $la->slice(
            $x,
            $start=[1,1],
            $size=[-1,-1]);
        $this->assertEquals([
            [[[18,19,20],
              [21,22,23]]],
        ],$y->toArray());

        $y = $la->slice(
            $x,
            $start=[1],
            $size=[-1]);
        $this->assertEquals([
            [[[12,13,14],
              [15,16,17]],
             [[18,19,20],
              [21,22,23]]],
        ],$y->toArray());
    }

    public function testStick()
    {
        $mo = $this->newMatrixOperator();
        $la = $this->newLA($mo);

        $x = $la->array($mo->arange(12,null,null,NDArray::float32)->reshape([2,2,3]));
        $y = $la->array($mo->zeros([2,4,3]));
        $la->stick(
            $x,
            $y,
            $start=[0,1],
            $size=[-1,2]
            );
        $this->assertEquals([
            [[0,0,0],
             [0,1,2],
             [3,4,5],
             [0,0,0]],
            [[0,0,0],
             [6,7,8],
             [9,10,11],
             [0,0,0]],
        ],$y->toArray());

        $x = $la->array($mo->arange(6,null,null,NDArray::float32)->reshape([2,1,3]));
        $y = $la->array($mo->zeros([2,4,3]));
        $la->stick(
            $x,
            $y,
            $start=[0,1],
            $size=[-1,1]
            );
        $this->assertEquals([
            [[0,0,0],
             [0,1,2],
             [0,0,0],
             [0,0,0]],
            [[0,0,0],
             [3,4,5],
             [0,0,0],
             [0,0,0]],
        ],$y->toArray());

        $x = $la->array($mo->arange(6,null,null,NDArray::float32)->reshape([2,1,3]));
        $y = $la->array($mo->zeros([2,4,3]));
        $la->stick(
            $x,
            $y,
            $start=[0,-1],
            $size=[-1,1]
            );
        $this->assertEquals([
            [[0,0,0],
             [0,0,0],
             [0,0,0],
             [0,1,2]],
            [[0,0,0],
             [0,0,0],
             [0,0,0],
             [3,4,5]],
        ],$y->toArray());

        $x = $la->array($mo->arange(12,null,null,NDArray::float32)->reshape([1,4,3]));
        $y = $la->array($mo->zeros([2,4,3]));
        $la->stick(
            $x,
            $y,
            $start=[1],
            $size=[1]
            );
        $this->assertEquals([
            [[0,0,0],
             [0,0,0],
             [0,0,0],
             [0,0,0]],
            [[0,1,2],
             [3,4,5],
             [6,7,8],
             [9,10,11]],
        ],$y->toArray());

        $x = $la->array($mo->arange(4,null,null,NDArray::float32)->reshape([2,2]));
        $y = $la->array($mo->zeros([2,4]));
        $la->stick(
            $x,
            $y,
            $start=[0,1],
            $size=[-1,2]
            );
        $this->assertEquals([
            [0,0,1,0],
            [0,2,3,0],
        ],$y->toArray());

        // 4D
        $x = $la->array([
            [[[0,1,2],
              [3,4,5]],
             [[6,7,8],
              [9,10,11]]],
            [[[12,13,14],
              [15,16,17]],
             [[18,19,20],
              [21,22,23]]],
        ]);
        $this->assertEquals(4,$x->ndim());
        $this->assertEquals([2,2,2,3],$x->shape());
        $y = $la->array($mo->zeros([2,2,4,3]));
        $la->stick(
            $x,
            $y,
            $start=[ 0, 0, 1],
            $size= [-1,-1, 2]
            );
        $this->assertEquals([
            [[[0,0,0],
              [0,1,2],
              [3,4,5],
              [0,0,0]],
             [[0,0,0],
              [6,7,8],
              [9,10,11],
              [0,0,0]]],
            [[[0,0,0],
              [12,13,14],
              [15,16,17],
              [0,0,0]],
             [[0,0,0],
              [18,19,20],
              [21,22,23],
              [0,0,0]]],
        ],$y->toArray());
    }

    public function testStack()
    {
        $mo = $this->newMatrixOperator();
        $la = $this->newLA($mo);

        $a = $la->array($mo->arange(6,0,null,NDArray::float32)->reshape([2,3]));
        $b = $la->array($mo->arange(6,6,null,NDArray::float32)->reshape([2,3]));
        $y = $la->stack(
            [$a,$b],
            $axis=0
            );
        $this->assertEquals([
            [[0,1,2],
             [3,4,5]],
            [[6,7,8],
             [9,10,11]],
        ],$y->toArray());

        $a = $la->array($mo->arange(6,0,null,NDArray::float32)->reshape([2,3]));
        $b = $la->array($mo->arange(6,6,null,NDArray::float32)->reshape([2,3]));
        $y = $la->stack(
            [$a,$b],
            $axis=1
            );
        $this->assertEquals([
            [[0,1,2],
             [6,7,8]],
            [[3,4,5],
             [9,10,11]],
        ],$y->toArray());

        $a = $la->array($mo->arange(12,0,null,NDArray::float32)->reshape([2, 2,3]));
        $b = $la->array($mo->arange(12,12,null,NDArray::float32)->reshape([2,2,3]));
        $y = $la->stack(
            [$a,$b],
            $axis=0
            );
        $this->assertEquals([
           [[[0,1,2],
             [3,4,5]],
            [[6,7,8],
             [9,10,11]]],
           [[[12,13,14],
             [15,16,17]],
            [[18,19,20],
             [21,22,23]]],
        ],$y->toArray());

        $a = $la->array($mo->arange(12,0,null,NDArray::float32)->reshape([2, 2,3]));
        $b = $la->array($mo->arange(12,12,null,NDArray::float32)->reshape([2,2,3]));
        $y = $la->stack(
            [$a,$b],
            $axis=1
            );
        $this->assertEquals([
           [[[0,1,2],
             [3,4,5]],
            [[12,13,14],
             [15,16,17]]],
           [[[6,7,8],
             [9,10,11]],
            [[18,19,20],
             [21,22,23]]],
        ],$y->toArray());

        // 4D
        $a = $la->array($mo->arange(24, 0,null,NDArray::float32)->reshape([2,2,2,3]));
        $b = $la->array($mo->arange(24,24,null,NDArray::float32)->reshape([2,2,2,3]));
        $y = $la->stack(
            [$a,$b],
            $axis=2
            );
        $this->assertEquals([
             [[[[ 0,  1,  2],
                [ 3,  4,  5]],
               [[24, 25, 26],
                [27, 28, 29]]],
              [[[ 6,  7,  8],
                [ 9, 10, 11]],
               [[30, 31, 32],
                [33, 34, 35]]]],
             [[[[12, 13, 14],
                [15, 16, 17]],
               [[36, 37, 38],
                [39, 40, 41]]],
              [[[18, 19, 20],
                [21, 22, 23]],
               [[42, 43, 44],
                [45, 46, 47]]]]
        ],$y->toArray());
    }

    public function testAnytypeSlice()
    {
        $mo = $this->newMatrixOperator();
        $la = $this->newLA($mo);

        if($la->fp64()) {
            $dtypes = [NDArray::float32,NDArray::float64,NDArray::uint8,NDArray::int32,NDArray::int64];
        } else {
            $dtypes = [NDArray::float32,NDArray::uint8,NDArray::int32,NDArray::int64];
        }
        foreach($dtypes as $dtype) {
            // forward slice
            $x = $la->array($mo->arange(24,null,null,$dtype)->reshape([2,4,3]));
            $y = $la->slice(
                $x,
                $start=[0,1],
                $size=[-1,2]
                );
            $this->assertEquals([
                [[3,4,5],
                 [6,7,8],],
                [[15,16,17],
                 [18,19,20],],
            ],$y->toArray());

            // reverse slice
            $x = $la->array($mo->arange(12,null,null,$dtype)->reshape([2,2,3]));
            $y = $la->array($mo->zeros([2,4,3],$dtype));
            $la->stick(
                $x,
                $y,
                $start=[0,1],
                $size=[-1,2]
                );
            $this->assertEquals([
                [[0,0,0],
                 [0,1,2],
                 [3,4,5],
                 [0,0,0]],
                [[0,0,0],
                 [6,7,8],
                 [9,10,11],
                 [0,0,0]],
            ],$y->toArray());

            // reverse and add
            // $Y = $la->array([
            //     [[1,2,3],[1,2,3]],
            //     [[4,5,6],[4,5,6]],
            // ],$dtype);
            // $X = $la->reduceSumRepeated($Y);
            // $this->assertEquals([2,2,3],$Y->shape());
            // $this->assertEquals([2,3],$X->shape());
            // $this->assertEquals([
            //     [[1,2,3],[1,2,3]],
            //     [[4,5,6],[4,5,6]],
            // ],$Y->toArray());
            // $this->assertEquals([
            //     [2,4,6],
            //     [8,10,12]
            // ],$X->toArray());

        }
    }

    public function testConcat()
    {
        $mo = $this->newMatrixOperator();
        $la = $this->newLA($mo);

        $a = $la->array($mo->arange(6,$start=0,null,NDArray::float32)->reshape([3,2]));
        $b = $la->array($mo->arange(4,$start=6,null,NDArray::float32)->reshape([2,2]));
        $y = $la->concat(
            [$a,$b],
            $axis=0
            );
        $this->assertEquals([
            [0,1],
            [2,3],
            [4,5],
            [6,7],
            [8,9],
        ],$y->toArray());

        $a = $la->array($mo->arange(6,$start=0,null,NDArray::float32)->reshape([2,3]));
        $b = $la->array($mo->arange(4,$start=6,null,NDArray::float32)->reshape([2,2]));
        $y = $la->concat(
            [$a,$b],
            $axis=1
            );
        $this->assertEquals([
            [0,1,2,6,7],
            [3,4,5,8,9],
        ],$y->toArray());

        $a = $la->array($mo->arange(12,$start=0,null,NDArray::float32)->reshape([3,2,2]));
        $b = $la->array($mo->arange(8,$start=12,null,NDArray::float32)->reshape([2,2,2]));
        $y = $la->concat(
            [$a,$b],
            $axis=0
            );
        $this->assertEquals([
            [[0,1],[2,3]],
            [[4,5],[6,7]],
            [[8,9],[10,11]],
            [[12,13],[14,15]],
            [[16,17],[18,19]],
        ],$y->toArray());

        $a = $la->array($mo->arange(12,$start=0,null,NDArray::float32)->reshape([2,3,2]));
        $b = $la->array($mo->arange(8,$start=12,null,NDArray::float32)->reshape([2,2,2]));
        $y = $la->concat(
            [$a,$b],
            $axis=1
            );
        $this->assertEquals([
            [[0,1],
             [2,3],
             [4,5],
             [12,13],
             [14,15]],
            [[6,7],
             [8,9],
             [10,11],
             [16,17],[18,19]],
        ],$y->toArray());

        $a = $la->array($mo->arange(12,$start=0,null,NDArray::float32)->reshape([2,2,3]));
        $b = $la->array($mo->arange(8,$start=12,null,NDArray::float32)->reshape([2,2,2]));
        $y = $la->concat(
            [$a,$b],
            $axis=2
            );
        $this->assertEquals([
            [[0,1,2,12,13],
             [3,4,5,14,15]],
            [[6,7,8,16,17],
             [9,10,11,18,19]],
        ],$y->toArray());

        $y = $la->concat(
            [$a,$b],
            $axis=-1
            );
        $this->assertEquals([
            [[0,1,2,12,13],
             [3,4,5,14,15]],
            [[6,7,8,16,17],
             [9,10,11,18,19]],
        ],$y->toArray());
    }

    public function testRepeat()
    {
        $mo = $this->newMatrixOperator();
        $la = $this->newLA($mo);

        // Y := X (duplicate 2 times)
        $X = $la->array([
            [1,2,3],
            [4,5,6]
        ]);
        $Y = $la->repeat($X,$repeats=2,$axis=1);
        $this->assertEquals([2,3],$X->shape());
        $this->assertEquals([2,2,3],$Y->shape());
        $this->assertEquals([
            [1,2,3],
            [4,5,6]
        ],$X->toArray());
        $this->assertEquals([
            [[1,2,3],[1,2,3]],
            [[4,5,6],[4,5,6]],
        ],$Y->toArray());

        // 1 time
        $X = $la->array([[1,2,3],[4,5,6]]);
        $Y = $la->repeat($X,$repeats=1,$axis=1);
        $this->assertEquals(
            [[1,2,3],[4,5,6]]
        ,$X->toArray());
        $this->assertEquals([2,3],$X->shape());
        $this->assertEquals([2,1,3],$Y->shape());
        $this->assertEquals(
            [[[1,2,3]],[[4,5,6]]]
        ,$Y->toArray());

        //
        $X = $la->array([
            [[1,2,3],[4,5,6]],
            [[7,8,9],[10,11,12]]
        ]);
        $Y = $la->repeat($X,$repeats=4,$axis=1);
        $this->assertEquals([
            [[1,2,3],[4,5,6]],
            [[7,8,9],[10,11,12]]
        ],$X->toArray());
        $this->assertEquals([2,2,3],$X->shape());
        $this->assertEquals([2,4,2,3],$Y->shape());
        $this->assertEquals([
            [[[1,2,3],[4,5,6]],
             [[1,2,3],[4,5,6]],
             [[1,2,3],[4,5,6]],
             [[1,2,3],[4,5,6]]],
            [[[7,8,9],[10,11,12]],
             [[7,8,9],[10,11,12]],
             [[7,8,9],[10,11,12]],
             [[7,8,9],[10,11,12]]],
        ],$Y->toArray());

        // axis = 0
        // Y := X (duplicate 2 times)
        $X = $la->array([
            [1,2,3],
            [4,5,6]
        ]);
        $Y = $la->repeat($X,$repeats=2,$axis=0);
        $this->assertEquals([2,3],$X->shape());
        $this->assertEquals([2,2,3],$Y->shape());
        $this->assertEquals([
            [1,2,3],
            [4,5,6]
        ],$X->toArray());
        $this->assertEquals([
            [[1,2,3],[4,5,6]],
            [[1,2,3],[4,5,6]],
        ],$Y->toArray());

        // axis = 0
        // Y := X (duplicate 1D)
        $X = $la->array([1,2,3]);
        $Y = $la->repeat($X,$repeats=2,$axis=0);
        $this->assertEquals([3],$X->shape());
        $this->assertEquals([2,3],$Y->shape());
        $this->assertEquals([1,2,3],$X->toArray());
        $this->assertEquals([
            [1,2,3],
            [1,2,3],
        ],$Y->toArray());

        // axis = 1
        // Y := X (duplicate 1D)
        $X = $la->array([1,2,3]);
        $Y = $la->repeat($X,$repeats=2,$axis=1);
        $this->assertEquals([3],$X->shape());
        $this->assertEquals([3,2],$Y->shape());
        $this->assertEquals([1,2,3],$X->toArray());
        $this->assertEquals([
            [1,1],
            [2,2],
            [3,3],
        ],$Y->toArray());

        // axis = NULL
        // Y := X (duplicate 2 times)
        $X = $la->array([
            [1,2,3],
            [4,5,6]
        ]);
        $Y = $la->repeat($X,$repeats=2,$axis=null);
        $this->assertEquals([2,3],$X->shape());
        $this->assertEquals([12],$Y->shape());
        $this->assertEquals([
            [1,2,3],
            [4,5,6]
        ],$X->toArray());
        $this->assertEquals([
            1,2,3,4,5,6,
            1,2,3,4,5,6,
        ],$Y->toArray());

        // axis = -1
        // Y := X (duplicate 2 times)
        $X = $la->array([
            [1,2,3],
            [4,5,6]
        ]);
        $Y = $la->repeat($X,$repeats=2,$axis=-1);
        $this->assertEquals([2,3],$X->shape());
        $this->assertEquals([2,2,3],$Y->shape());
        $this->assertEquals([
            [1,2,3],
            [4,5,6]
        ],$X->toArray());
        $this->assertEquals([
            [[1,2,3],[1,2,3]],
            [[4,5,6],[4,5,6]],
        ],$Y->toArray());
    }

    public function testReduceSum3d()
    {
        $mo = $this->newMatrixOperator();
        $la = $this->newLA($mo);

        // Y := X (sum 2 times)
        $Y = $la->array([
            [[1,2,3],[1,2,3]],
            [[4,5,6],[4,5,6]],
        ]);
        $X = $la->reduceSum($Y,$axis=1);
        $this->assertEquals([2,2,3],$Y->shape());
        $this->assertEquals([2,3],$X->shape());
        $this->assertEquals([
            [[1,2,3],[1,2,3]],
            [[4,5,6],[4,5,6]],
        ],$Y->toArray());
        $this->assertEquals([
            [2,4,6],
            [8,10,12]
        ],$X->toArray());

        // 1 time
        $Y = $la->array([
            [[1,2,3]],
            [[4,5,6]]
        ]);
        $X = $la->reduceSum($Y,$axis=1);
        $this->assertEquals([2,1,3],$Y->shape());
        $this->assertEquals([2,3],$X->shape());
        $this->assertEquals([
            [1,2,3],
            [4,5,6]
        ],$X->toArray());
        $this->assertEquals([
            [[1,2,3]],
            [[4,5,6]]
        ],$Y->toArray());

        $Y = $la->array([
            [[[1,2,3],[4,5,6]],
             [[1,2,3],[4,5,6]],
             [[1,2,3],[4,5,6]],
             [[1,2,3],[4,5,6]]],
            [[[7,8,9],[10,11,12]],
             [[7,8,9],[10,11,12]],
             [[7,8,9],[10,11,12]],
             [[7,8,9],[10,11,12]]],
        ]);
        $X = $la->reduceSum($Y,$axis=1);
        $this->assertEquals([2,4,2,3],$Y->shape());
        $this->assertEquals([2,2,3],$X->shape());
        $this->assertEquals([
            [[4,8,12],[16,20,24]],
            [[28,32,36],[40,44,48]]
        ],$X->toArray());
        $this->assertEquals([
            [[[1,2,3],[4,5,6]],
             [[1,2,3],[4,5,6]],
             [[1,2,3],[4,5,6]],
             [[1,2,3],[4,5,6]]],
            [[[7,8,9],[10,11,12]],
             [[7,8,9],[10,11,12]],
             [[7,8,9],[10,11,12]],
             [[7,8,9],[10,11,12]]],
        ],$Y->toArray());
    }

    public function testSplit()
    {
        $mo = $this->newMatrixOperator();
        $la = $this->newLA($mo);

        $x = $la->array([
            [0,1],
            [2,3],
            [4,5],
            [6,7],
            [8,9],
        ]);
        $y = $la->split(
            $x,
            [3,2],
            $axis=0
        );
        $a = $la->array($mo->arange(6,$start=0,null,NDArray::float32)->reshape([3,2]));
        $b = $la->array($mo->arange(4,$start=6,null,NDArray::float32)->reshape([2,2]));
        $this->assertCount(2,$y);
        $this->assertEquals($a->toArray(),$y[0]->toArray());
        $this->assertEquals($b->toArray(),$y[1]->toArray());

        $x = $la->array([
            [0,1,2,6,7],
            [3,4,5,8,9],
        ]);
        $y = $la->split(
            $x,
            [3,2],
            $axis=1
            );
        $a = $la->array($mo->arange(6,$start=0,null,NDArray::float32)->reshape([2,3]));
        $b = $la->array($mo->arange(4,$start=6,null,NDArray::float32)->reshape([2,2]));
        $this->assertCount(2,$y);
        $this->assertEquals($a->toArray(),$y[0]->toArray());
        $this->assertEquals($b->toArray(),$y[1]->toArray());

        $x = $la->array([
            [[0,1],[2,3]],
            [[4,5],[6,7]],
            [[8,9],[10,11]],
            [[12,13],[14,15]],
            [[16,17],[18,19]],
        ]);
        $y = $la->split(
            $x,
            [3,2],
            $axis=0
            );
        $a = $la->array($mo->arange(12,$start=0,null,NDArray::float32)->reshape([3,2,2]));
        $b = $la->array($mo->arange(8,$start=12,null,NDArray::float32)->reshape([2,2,2]));
        $this->assertCount(2,$y);
        $this->assertEquals($a->toArray(),$y[0]->toArray());
        $this->assertEquals($b->toArray(),$y[1]->toArray());

        $x = $la->array([
            [[0,1],
             [2,3],
             [4,5],
             [12,13],
             [14,15]],
            [[6,7],
             [8,9],
             [10,11],
             [16,17],[18,19]],
        ]);
        $y = $la->split(
            $x,
            [3,2],
            $axis=1
            );
        $a = $la->array($mo->arange(12,$start=0,null,NDArray::float32)->reshape([2,3,2]));
        $b = $la->array($mo->arange(8,$start=12,null,NDArray::float32)->reshape([2,2,2]));
        $this->assertCount(2,$y);
        $this->assertEquals($a->toArray(),$y[0]->toArray());
        $this->assertEquals($b->toArray(),$y[1]->toArray());

        $x = $la->array([
            [[0,1,2,12,13],
             [3,4,5,14,15]],
            [[6,7,8,16,17],
             [9,10,11,18,19]],
        ]);
        $y = $la->split(
            $x,
            [3,2],
            $axis=2
            );
        $a = $la->array($mo->arange(12,$start=0,null,NDArray::float32)->reshape([2,2,3]));
        $b = $la->array($mo->arange(8,$start=12,null,NDArray::float32)->reshape([2,2,2]));
        $this->assertCount(2,$y);
        $this->assertEquals($a->toArray(),$y[0]->toArray());
        $this->assertEquals($b->toArray(),$y[1]->toArray());

        $y = $la->split(
            $x,
            [3,2],
            $axis=-1
            );
        $this->assertCount(2,$y);
        $this->assertEquals($a->toArray(),$y[0]->toArray());
        $this->assertEquals($b->toArray(),$y[1]->toArray());
    }

    public function testTranspose()
    {
        $mo = $this->newMatrixOperator();
        $la = $this->newLA($mo);
        $a = $la->array([
            [0,1,2],
            [3,4,5],
        ]);
        $b = $la->transpose($a);
        $this->assertEquals([
            [0,3],
            [1,4],
            [2,5]
        ],$b->toArray());
    }

    public function testImagecopy()
    {
        $mo = $this->newMatrixOperator();
        $la = $this->newLA($mo);
        $a = $la->array([
            [[0],[1],[2]],
            [[3],[4],[5]],
            [[6],[7],[8]],
        ]);
        $b = $la->imagecopy($a,null,null,
            $heightShift=1
        );
        $this->assertEquals([
            [[0],[1],[2]],
            [[0],[1],[2]],
            [[3],[4],[5]],
        ],$b->toArray());
        $b = $la->imagecopy($a,null,null,
            $heightShift=-1
        );
        $this->assertEquals([
            [[3],[4],[5]],
            [[6],[7],[8]],
            [[6],[7],[8]],
        ],$b->toArray());
        $b = $la->imagecopy($a,null,null,
            $heightShift=null,
            $widthShift=1
        );
        $this->assertEquals([
            [[0],[0],[1]],
            [[3],[3],[4]],
            [[6],[6],[7]],
        ],$b->toArray());
        $b = $la->imagecopy($a,null,null,
            $heightShift=null,
            $widthShift=-1
        );
        $this->assertEquals([
            [[1],[2],[2]],
            [[4],[5],[5]],
            [[7],[8],[8]],
        ],$b->toArray());
        $b = $la->imagecopy($a,null,null,
            $heightShift=1,
            $widthShift=1
        );
        $this->assertEquals([
            [[0],[0],[1]],
            [[0],[0],[1]],
            [[3],[3],[4]],
        ],$b->toArray());
        $b = $la->imagecopy($a,null,null,
            $heightShift=-1,
            $widthShift=-1
        );
        $this->assertEquals([
            [[4],[5],[5]],
            [[7],[8],[8]],
            [[7],[8],[8]],
        ],$b->toArray());
        $b = $la->imagecopy($a,null,null,
            $heightShift=null,
            $widthShift=null,
            $verticalFlip=true,
            $horizontalFlip=null
        );
        $this->assertEquals([
            [[6],[7],[8]],
            [[3],[4],[5]],
            [[0],[1],[2]],
        ],$b->toArray());
        $b = $la->imagecopy($a,null,null,
            $heightShift=null,
            $widthShift=null,
            $verticalFlip=null,
            $horizontalFlip=true
        );
        $this->assertEquals([
            [[2],[1],[0]],
            [[5],[4],[3]],
            [[8],[7],[6]],
        ],$b->toArray());

        $a = $la->array([
            [[0], [1], [2], [3] ],
            [[4], [5], [6], [7] ],
            [[8], [9], [10],[11]],
            [[12],[13],[14],[15]],
        ]);
        $b = $la->imagecopy($a,null,null,
            $heightShift=1,
            $widthShift=null,
            $verticalFlip=true,
            $horizontalFlip=null
        );
        $this->assertEquals([
            [[12],[13],[14],[15]],
            [[12],[13],[14],[15]],
            [[8],[9],[10],[11]],
            [[4],[5],[6],[7]],
        ],$b->toArray());
        $b = $la->imagecopy($a,null,null,
            $heightShift=-1,
            $widthShift=null,
            $verticalFlip=true,
            $horizontalFlip=null
        );
        $this->assertEquals([
            [[8], [9], [10],[11]],
            [[4], [5], [6], [7] ],
            [[0], [1], [2], [3] ],
            [[0], [1], [2], [3] ],
        ],$b->toArray());
        $b = $la->imagecopy($a,null,null,
            $heightShift=null,
            $widthShift=1,
            $verticalFlip=null,
            $horizontalFlip=true
        );
        $this->assertEquals([
            [[3] ,[3] ,[2], [1] ],
            [[7] ,[7] ,[6], [5] ],
            [[11],[11],[10],[9] ],
            [[15],[15],[14],[13]],
        ],$b->toArray());
        $b = $la->imagecopy($a,null,null,
            $heightShift=null,
            $widthShift=-1,
            $verticalFlip=null,
            $horizontalFlip=true
        );
        $this->assertEquals([
            [[2 ],[1 ],[0] ,[0] ],
            [[6 ],[5 ],[4] ,[4] ],
            [[10],[9 ],[8] ,[8] ],
            [[14],[13],[12],[12]],
        ],$b->toArray());
        $b = $la->imagecopy($a,null,null,
            $heightShift=null,
            $widthShift=null,
            $verticalFlip=true,
            $horizontalFlip=true
        );
        $this->assertEquals([
            [[15],[14],[13],[12]],
            [[11],[10],[9] ,[8] ],
            [[7] ,[6] ,[5] ,[4] ],
            [[3] ,[2] ,[1] ,[0] ],
        ],$b->toArray());
        $b = $la->imagecopy($a,null,null,
            $heightShift=1,
            $widthShift=null,
            $verticalFlip=true,
            $horizontalFlip=true
        );
        $this->assertEquals([
            [[15],[14],[13],[12]],
            [[15],[14],[13],[12]],
            [[11],[10],[9] ,[8]],
            [[7] ,[6], [5] ,[4]],
        ],$b->toArray());
        $b = $la->imagecopy($a,null,null,
            $heightShift=-1,
            $widthShift=null,
            $verticalFlip=true,
            $horizontalFlip=true
        );
        $this->assertEquals([
            [[11],[10],[9] ,[8]],
            [[7] ,[6], [5] ,[4]],
            [[3] ,[2], [1] ,[0]],
            [[3] ,[2], [1] ,[0]],
        ],$b->toArray());
        $b = $la->imagecopy($a,null,null,
            $heightShift=null,
            $widthShift=1,
            $verticalFlip=true,
            $horizontalFlip=true
        );
        $this->assertEquals([
            [[15],[15],[14],[13]],
            [[11],[11],[10],[9 ]],
            [[7 ],[7 ],[6 ],[5 ]],
            [[3 ],[3 ],[2 ],[1 ]],
        ],$b->toArray());
        $b = $la->imagecopy($a,null,null,
            $heightShift=null,
            $widthShift=-1,
            $verticalFlip=true,
            $horizontalFlip=true
        );
        $this->assertEquals([
            [[14],[13],[12],[12]],
            [[10],[9 ],[8 ],[8 ]],
            [[6 ],[5 ],[4 ],[4 ]],
            [[2 ],[1 ],[0 ],[0 ]],
        ],$b->toArray());
        $b = $la->imagecopy($a,null,null,
            $heightShift=1,
            $widthShift=1,
            $verticalFlip=true,
            $horizontalFlip=true
        );
        $this->assertEquals([
            [[15],[15],[14],[13]],
            [[15],[15],[14],[13]],
            [[11],[11],[10],[9] ],
            [[7] ,[7] ,[6] ,[5] ],
        ],$b->toArray());
        $b = $la->imagecopy($a,null,null,
            $heightShift=-1,
            $widthShift=-1,
            $verticalFlip=true,
            $horizontalFlip=true
        );
        $this->assertEquals([
            [[10],[9] ,[8] ,[8] ],
            [[6] ,[5] ,[4] ,[4] ],
            [[2] ,[1] ,[0] ,[0] ],
            [[2] ,[1] ,[0] ,[0] ],
        ],$b->toArray());

        $a = $la->array([
            [[1,2,3],
             [1,2,3]],
            [[4,5,6],
             [4,5,6]],
            [[7,8,9],
             [7,8,9]],
        ]);
        $this->assertEquals([3,2,3],$a->shape());
        $b = $la->imagecopy($a,null,null,
            $heightShift=1,
            $widthShift=0,
            $verticalFlip=false,
            $horizontalFlip=false
        );
        //echo $mo->toString($b,null,true);
        $this->assertEquals([
            [[1,2,3],
             [1,2,3]],
            [[1,2,3],
             [1,2,3]],
            [[4,5,6],
             [4,5,6]],
        ],$b->toArray());

        // flip rgb
        $a = $la->array([
            [[1,2,3],
             [1,2,3]],
            [[4,5,6],
             [4,5,6]],
            [[7,8,9],
             [7,8,9]],
        ]);
        $this->assertEquals([3,2,3],$a->shape());
        $b = $la->imagecopy($a,null,null,
            $heightShift=0,
            $widthShift=0,
            $verticalFlip=false,
            $horizontalFlip=false,
            $rgbFlip=true
        );
        //echo $mo->toString($b,null,true);
        $this->assertEquals([
            [[3,2,1],
             [3,2,1]],
            [[6,5,4],
             [6,5,4]],
            [[9,8,7],
             [9,8,7]],
        ],$b->toArray());

        // flip rgb with alpha
        $a = $la->array([
            [[1,2,3,4],
             [1,2,3,4]],
            [[4,5,6,7],
             [4,5,6,7]],
            [[7,8,9,10],
             [7,8,9,10]],
        ]);
        $this->assertEquals([3,2,4],$a->shape());
        $b = $la->imagecopy($a,null,null,
            $heightShift=0,
            $widthShift=0,
            $verticalFlip=false,
            $horizontalFlip=false,
            $rgbFlip=true
        );
        //echo $mo->toString($b,null,true);
        $this->assertEquals([
            [[3,2,1,4],
             [3,2,1,4]],
            [[6,5,4,7],
             [6,5,4,7]],
            [[9,8,7,10],
             [9,8,7,10]],
        ],$b->toArray());
    }

        public function testImagecopychannelsfirst()
    {
        $mo = $this->newMatrixOperator();
        $la = $this->newLA($mo);
        $a = $la->array([
            [[1,2,3],
             [4,5,6]],
            [[11,12,13],
             [14,15,16]],
            [[21,22,23],
             [24,25,26]],
        ]);
        $this->assertEquals([3,2,3],$a->shape());
        $b = $la->imagecopy($a,null,true,
            $heightShift=1,
            $widthShift=0,
            $verticalFlip=false,
            $horizontalFlip=false
        );
        $this->assertEquals([
            [[1,2,3],
             [1,2,3]],
            [[11,12,13],
             [11,12,13]],
            [[21,22,23],
             [21,22,23]],
        ],$b->toArray());
        $b = $la->imagecopy($a,null,true,
            $heightShift=0,
            $widthShift=1,
            $verticalFlip=false,
            $horizontalFlip=false
        );
        $this->assertEquals([
            [[1,1,2],
             [4,4,5]],
            [[11,11,12],
             [14,14,15]],
            [[21,21,22],
             [24,24,25]],
        ],$b->toArray());
        $b = $la->imagecopy($a,null,true,
            $heightShift=0,
            $widthShift=0,
            $verticalFlip=true,
            $horizontalFlip=false
        );
        $this->assertEquals([
            [[4,5,6],
             [1,2,3]],
            [[14,15,16],
             [11,12,13]],
            [[24,25,26],
             [21,22,23]],
        ],$b->toArray());
        $b = $la->imagecopy($a,null,true,
            $heightShift=0,
            $widthShift=0,
            $verticalFlip=false,
            $horizontalFlip=true
        );
        //echo $mo->toString($b,null,true);
        $this->assertEquals([
            [[3,2,1],
             [6,5,4]],
            [[13,12,11],
             [16,15,14]],
            [[23,22,21],
             [26,25,24]],
        ],$b->toArray());
    }

    public function testfill()
    {
        $mo = $this->newMatrixOperator();
        $la = $this->newLA($mo);
        $x = $la->alloc([2,3],NDArray::float32);
        $b = $la->fill(123,$x);
        $this->assertEquals([
            [123,123,123],
            [123,123,123],
        ],$b->toArray());

        $x = $la->alloc([2,3],NDArray::int64);
        $b = $la->fill(123,$x);
        $this->assertEquals([
            [123,123,123],
            [123,123,123],
        ],$b->toArray());

        $x = $la->alloc([2,3],NDArray::int8);
        $b = $la->fill(123,$x);
        $this->assertEquals([
            [123,123,123],
            [123,123,123],
        ],$b->toArray());
    }

    public function testSearchsorted()
    {
        $mo = $this->newMatrixOperator();
        $la = $this->newLA($mo);

        $A = $mo->array([0.1,0.3,0.5,0.7,0.9]);
        $A = $la->array($A);
        $X = $mo->array([0.0,0.5,1.0]);
        $X = $la->array($X);
        $Y = $la->searchsorted($A,$X);
        $this->assertEquals(
            [0,2,5],
            $Y->toArray()
        );

        // right=true
        $A = $mo->array([0.1,0.3,0.5,0.7,0.9]);
        $A = $la->array($A);
        $X = $mo->array([0.0,0.5,1.0]);
        $X = $la->array($X);
        $Y = $la->searchsorted($A,$X,true);
        $this->assertEquals(
            [0,3,5],
            $Y->toArray()
        );

        // after nan2num and cumsum
        //  nan nan 5 nan 4
        $A = $mo->array([NAN,NAN,0.5,NAN,0.5]);
        $A = $la->array($A);
        $A = $la->nan2num($A,0);
        $total = $la->sum($A);
        $this->assertEquals(1.0,$total);
        $size = $A->size();
        $this->assertEquals(5,$size);
        $A = $la->cumsum($A[[0,$size-2]]);
        $this->assertEquals([0.0,0.0,0.5,0.5],$A->toArray());
        $X = $mo->array([0.0,0.4,0.6,$total]);
        $X = $la->array($X);
        $Y = $la->searchsorted($A,$X,true);
        $this->assertEquals(
            [2,2,4,4],
            $Y->toArray()
        );

        // right=true
        $A = $mo->array([0.5,1.0,1.0]);
        $A = $la->array($A);
        $X = $mo->array([0.9]);
        $X = $la->array($X);
        $Y = $la->searchsorted($A,$X,true);
        $this->assertEquals(
            [1],
            $Y->toArray()
        );


        // nan data
        $A = $mo->array([1,3,5,7,9]);
        $A = $la->array($A);
        $X = $mo->array([NAN,5,10]);
        $X = $la->array($X);
        $Y = $la->searchsorted($A,$X);
        $this->assertEquals(
            [0,2,5],
            $Y->toArray()
        );

        // nan seq
        $A = $mo->array([0.1,0.3,NAN,0.7,0.9]);
        $A = $la->array($A);
        $X = $mo->array([0.0,0.5,1.0]);
        $X = $la->array($X);
        $Y = $la->searchsorted($A,$X);
        $this->assertEquals(
            [0,2,2],
            $Y->toArray()
        );
    }

    public function testcumsum()
    {
        $mo = $this->newMatrixOperator();
        $la = $this->newLA($mo);

        $X = $mo->array([1,2,1,2]);
        $X = $la->array($X);
        $Y = $la->cumsum($X);
        $this->assertEquals(
            [1,3,4,6],
            $Y->toArray()
        );

        // exclusive=true
        $X = $mo->array([1,2,1,2]);
        $X = $la->array($X);
        $Y = $la->cumsum($X,$exclusive=true);
        $this->assertEquals(
            [0,1,3,4],
            $Y->toArray()
        );

        // reverse=true
        $X = $mo->array([1,2,1,2]);
        $X = $la->array($X);
        $Y = $la->cumsum($X,null,$reverse=true);
        $this->assertEquals(
            [6,4,3,1],
            $Y->toArray()
        );

        // nan data
        $X = $mo->array([1,2,NAN,2]);
        $X = $la->array($X);
        $Y = $la->cumsum($X);
        $Y = $la->toNDArray($Y);
        $this->assertEquals(1.0,$Y[0]);
        $this->assertEquals(3.0,$Y[1]);
        $this->assertTrue(is_nan($Y[2]));
        $this->assertTrue(is_nan($Y[3]));

        $X = $mo->array([1,2,NAN,2]);
        $X = $la->array($X);
        $Y = $la->cumsum($X,null,$reverse=true);
        $Y = $la->toNDArray($Y);
        $this->assertTrue(is_nan($Y[0]));
        $this->assertTrue(is_nan($Y[1]));
        $this->assertEquals(3.0,$Y[2]);
        $this->assertEquals(1.0,$Y[3]);
    }

    public function testNan2num()
    {
        $mo = $this->newMatrixOperator();
        $la = $this->newLA($mo);

        // X := nan2num(X)
        $X = $mo->array([[NAN,2,NAN],[4,NAN,6]]);
        $X = $la->array($X);
        $la->nan2num($X);
        $this->assertEquals(
            [[0,2,0],[4,0,6]],
            $X->toArray()
        );

        $X = $mo->array([[NAN,2,NAN],[4,NAN,6]]);
        $X = $la->array($X);
        $la->nan2num($X,1.0);
        $this->assertEquals(
            [[1,2,1],[4,1,6]],
            $X->toArray()
        );
    }

    public function testIsnan()
    {
        $mo = $this->newMatrixOperator();
        $la = $this->newLA($mo);

        // X := nan2num(X)
        $X = $mo->array([[NAN,2,NAN],[4,NAN,6]]);
        $X = $la->array($X);
        $la->isnan($X);
        $this->assertEquals(
            [[1,0,1],[0,1,0]],
            $X->toArray()
        );
    }

    public function testLinspace()
    {
        $mo = $this->newMatrixOperator();
        $la = $this->newLA($mo);

        $X = $la->linspace($start=10,$stop=100,$num=10);
        $this->assertEquals(
            [10,20,30,40,50,60,70,80,90,100],
            $X->toArray()
        );
    }


    public function testSvdFull1()
    {
        $mo = $this->newMatrixOperator();
        $la = $this->newLA($mo);
        $a = $la->array([
            [ 8.79,  9.93,  9.83,  5.45,  3.16,],
            [ 6.11,  6.91,  5.04, -0.27,  7.98,],
            [-9.15, -7.93,  4.86,  4.85,  3.01,],
            [ 9.57,  1.64,  8.83,  0.74,  5.80,],
            [-3.49,  4.02,  9.80, 10.00,  4.27,],
            [ 9.84,  0.15, -8.99, -6.02, -5.31,],
        ]);
        $this->assertEquals([6,5],$a->shape());
        [$u,$s,$vt] = $la->svd($a);
        $this->assertEquals([6,6],$u->shape());
        $this->assertEquals([5],$s->shape());
        $this->assertEquals([5,5],$vt->shape());

        # echo "---- u ----\n";
        # foreach($u->toArray() as $array)
        #     echo '['.implode(',',array_map(function($a){return sprintf('%5.2f',$a);},$array))."],\n";
        # echo "---- s ----\n";
        # echo '['.implode(',',array_map(function($a){return sprintf('%5.2f',$a);},$s->toArray()))."],\n";
        # echo "---- vt ----\n";
        # foreach($vt->toArray() as $array)
        #     echo '['.implode(',',array_map(function($a){return sprintf('%5.2f',$a);},$array))."],\n";

        # ---- u ----
        $correctU = $la->array([
            [-0.59, 0.26, 0.36, 0.31, 0.23, 0.55],
            [-0.40, 0.24,-0.22,-0.75,-0.36, 0.18],
            [-0.03,-0.60,-0.45, 0.23,-0.31, 0.54],
            [-0.43, 0.24,-0.69, 0.33, 0.16,-0.39],
            [-0.47,-0.35, 0.39, 0.16,-0.52,-0.46],
            [ 0.29, 0.58,-0.02, 0.38,-0.65, 0.11],
        ]);
        //$this->assertTrue(false);
        $this->assertLessThan(0.01,abs($la->amax($la->axpy($u,$correctU,-1))));
        # ---- s ----
        $correctS = $la->array(
            [27.47,22.64, 8.56, 5.99, 2.01]
        );
        $this->assertLessThan(0.01,abs($la->amax($la->axpy($s,$correctS,-1))));
        # ---- vt ----
        $correctVT = $la->array([
            [-0.25,-0.40,-0.69,-0.37,-0.41],
            [ 0.81, 0.36,-0.25,-0.37,-0.10],
            [-0.26, 0.70,-0.22, 0.39,-0.49],
            [ 0.40,-0.45, 0.25, 0.43,-0.62],
            [-0.22, 0.14, 0.59,-0.63,-0.44],
        ]);
        $this->assertLessThan(0.01,abs($la->amax($la->axpy($vt,$correctVT,-1))));
        $this->assertTrue(true);
    }

    /**
    *   @requires extension rindow_openblas
    */
    public function testSvdFull2()
    {
        $mo = $this->newMatrixOperator();
        $la = $this->newLA($mo);
        $a = $la->array([
            [ 8.79,  9.93,  9.83,  5.45,  3.16,],
            [ 6.11,  6.91,  5.04, -0.27,  7.98,],
            [-9.15, -7.93,  4.86,  4.85,  3.01,],
            [ 9.57,  1.64,  8.83,  0.74,  5.80,],
            [-3.49,  4.02,  9.80, 10.00,  4.27,],
            [ 9.84,  0.15, -8.99, -6.02, -5.31,],
        ]);
        $a = $la->transpose($a);
        $this->assertEquals([5,6],$a->shape());
        [$u,$s,$vt] = $la->svd($a);
        $this->assertEquals([5,5],$u->shape());
        $this->assertEquals([5],$s->shape());
        $this->assertEquals([6,6],$vt->shape());

        # echo "---- u ----\n";
        # foreach($u->toArray() as $array)
        #     echo '['.implode(',',array_map(function($a){return sprintf('%5.2f',$a);},$array))."],\n";
        # echo "---- s ----\n";
        # echo '['.implode(',',array_map(function($a){return sprintf('%5.2f',$a);},$s->toArray()))."],\n";
        # echo "---- vt ----\n";
        # foreach($vt->toArray() as $array)
        #     echo '['.implode(',',array_map(function($a){return sprintf('%5.2f',$a);},$array))."],\n";

        # ---- u ----
        $correctU = $la->array([
            [ 0.25, 0.40, 0.69, 0.37, 0.41],
            [ 0.81, 0.36,-0.25,-0.37,-0.10],
            [-0.26, 0.70,-0.22, 0.39,-0.49],
            [ 0.40,-0.45, 0.25, 0.43,-0.62],
            [-0.22, 0.14, 0.59,-0.63,-0.44],
        ]);
        $correctU = $la->transpose($correctU);

        $correctU = $la->square($correctU);
        $u = $la->square($u);
        $this->assertLessThan(0.01,abs($la->amax($la->axpy($u,$correctU,-1))));
        # ---- s ----
        $correctS = $la->array(
            [27.47,22.64, 8.56, 5.99, 2.01]
        );
        $this->assertLessThan(0.01,abs($la->amax($la->axpy($s,$correctS,-1))));
        # ---- vt ----
        $correctVT = $la->array([
            [ 0.59, 0.26, 0.36, 0.31, 0.23, 0.55],
            [ 0.40, 0.24,-0.22,-0.75,-0.36, 0.18],
            [ 0.03,-0.60,-0.45, 0.23,-0.31, 0.54],
            [ 0.43, 0.24,-0.69, 0.33, 0.16,-0.39],
            [ 0.47,-0.35, 0.39, 0.16,-0.52,-0.46],
            [-0.29, 0.58,-0.02, 0.38,-0.65, 0.11],
        ]);
        $correctVT = $la->transpose($correctVT);
        $correctVT = $la->square($correctVT);
        $vt = $la->square($vt);
        $this->assertLessThan(0.01,abs($la->amax($la->axpy($vt,$correctVT,-1))));
        $this->assertTrue(true);
    }

    public function testSvdSmallU()
    {
        $mo = $this->newMatrixOperator();
        $la = $this->newLA($mo);
        $a = $la->array([
            [ 8.79,  9.93,  9.83,  5.45,  3.16,],
            [ 6.11,  6.91,  5.04, -0.27,  7.98,],
            [-9.15, -7.93,  4.86,  4.85,  3.01,],
            [ 9.57,  1.64,  8.83,  0.74,  5.80,],
            [-3.49,  4.02,  9.80, 10.00,  4.27,],
            [ 9.84,  0.15, -8.99, -6.02, -5.31,],
        ]);
        [$u,$s,$vt] = $la->svd($a,$full_matrices=false);

        # echo "---- u ----\n";
        # foreach($u->toArray() as $array)
        #     echo '['.implode(',',array_map(function($a){return sprintf('%5.2f',$a);},$array))."],\n";
        # echo "---- s ----\n";
        # echo '['.implode(',',array_map(function($a){return sprintf('%5.2f',$a);},$s->toArray()))."],\n";
        # echo "---- vt ----\n";
        # foreach($vt->toArray() as $array)
        #     echo '['.implode(',',array_map(function($a){return sprintf('%5.2f',$a);},$array))."],\n";

        # ---- u ----
        $correctU = $la->array([
            [-0.59, 0.26, 0.36, 0.31, 0.23],
            [-0.40, 0.24,-0.22,-0.75,-0.36],
            [-0.03,-0.60,-0.45, 0.23,-0.31],
            [-0.43, 0.24,-0.69, 0.33, 0.16],
            [-0.47,-0.35, 0.39, 0.16,-0.52],
            [ 0.29, 0.58,-0.02, 0.38,-0.65],
        ]);
        $this->assertLessThan(0.01,abs($la->amax($la->axpy($u,$correctU,-1))));
        # ---- s ----
        $correctS = $la->array(
            [27.47,22.64, 8.56, 5.99, 2.01]
        );
        $this->assertLessThan(0.01,abs($la->amax($la->axpy($s,$correctS,-1))));
        # ---- vt ----
        $correctVT = $la->array([
            [-0.25,-0.40,-0.69,-0.37,-0.41],
            [ 0.81, 0.36,-0.25,-0.37,-0.10],
            [-0.26, 0.70,-0.22, 0.39,-0.49],
            [ 0.40,-0.45, 0.25, 0.43,-0.62],
            [-0.22, 0.14, 0.59,-0.63,-0.44],
        ]);
        $this->assertLessThan(0.01,abs($la->amax($la->axpy($vt,$correctVT,-1))));
        $this->assertTrue(true);
    }

    /**
    *   @requires extension rindow_openblas
    */
    public function testSvdSmallVT()
    {
        $mo = $this->newMatrixOperator();
        $la = $this->newLA($mo);
        $a = $la->array([
            [ 8.79,  9.93,  9.83,  5.45,  3.16,],
            [ 6.11,  6.91,  5.04, -0.27,  7.98,],
            [-9.15, -7.93,  4.86,  4.85,  3.01,],
            [ 9.57,  1.64,  8.83,  0.74,  5.80,],
            [-3.49,  4.02,  9.80, 10.00,  4.27,],
            [ 9.84,  0.15, -8.99, -6.02, -5.31,],
        ]);
        $a = $la->transpose($a);
        [$u,$s,$vt] = $la->svd($a,$full_matrices=false);

        # echo "---- u ----\n";
        # foreach($u->toArray() as $array)
        #  echo '['.implode(',',array_map(function($a){return sprintf('%5.2f',$a);},$array))."],\n";
        # echo "---- s ----\n";
        # echo '['.implode(',',array_map(function($a){return sprintf('%5.2f',$a);},$s->toArray()))."],\n";
        # echo "---- vt ----\n";
        # foreach($vt->toArray() as $array)
        #  echo '['.implode(',',array_map(function($a){return sprintf('%5.2f',$a);},$array))."],\n";

        # ---- u ----
        $correctU = $la->array([
            [ 0.25, 0.40, 0.69, 0.37, 0.41],
            [ 0.81, 0.36,-0.25,-0.37,-0.10],
            [-0.26, 0.70,-0.22, 0.39,-0.49],
            [ 0.40,-0.45, 0.25, 0.43,-0.62],
            [-0.22, 0.14, 0.59,-0.63,-0.44],
        ]);
        $correctU = $la->transpose($correctU);
        $correctU = $la->square($correctU);
        $u = $la->square($u);
        $this->assertLessThan(0.01,abs($la->amax($la->axpy($u,$correctU,-1))));
        # ---- s ----
        $correctS = $la->array(
            [27.47,22.64, 8.56, 5.99, 2.01]
        );
        $this->assertLessThan(0.01,abs($la->amax($la->axpy($s,$correctS,-1))));
        # ---- vt ----
        $correctVT = $la->array([
            [ 0.59, 0.26, 0.36, 0.31, 0.23,],
            [ 0.40, 0.24,-0.22,-0.75,-0.36,],
            [ 0.03,-0.60,-0.45, 0.23,-0.31,],
            [ 0.43, 0.24,-0.69, 0.33, 0.16,],
            [ 0.47,-0.35, 0.39, 0.16,-0.52,],
            [-0.29, 0.58,-0.02, 0.38,-0.65,],
        ]);
        $correctVT = $la->transpose($correctVT);
        $correctVT = $la->square($correctVT);
        $vt = $la->square($vt);
        $this->assertLessThan(0.01,abs($la->amax($la->axpy($vt,$correctVT,-1))));
        $this->assertTrue(true);
    }

    public function testSolve()
    {
        $mo = $this->newMatrixOperator();
        $la = $this->newLA($mo);
        $a = $la->array([
            [1, 1, 1],
            [2, 4, 6],
            [2, 0, 4],
        ]);
        $b = $la->array(
             [10, 38, 14]
        );
        $solve = $la->solve($a,$b);
        //echo $mo->toString($solve,'%f',true);
        $this->assertEquals([3,5,2],$solve->toArray());
    }

}
