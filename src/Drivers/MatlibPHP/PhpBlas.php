<?php
namespace Rindow\Math\Matrix\Drivers\MatlibPHP;

use LogicException;
use InvalidArgumentException;
use RuntimeException;
use Interop\Polite\Math\Matrix\BLAS;
use Interop\Polite\Math\Matrix\NDArray;
use Interop\Polite\Math\Matrix\Buffer;
use Rindow\Math\Matrix\ComplexUtils;

class PhpBlas
{
    use Utils;
    use ComplexUtils;

    const GAM = 4096;
    const GAMSQ = 16777216;
    const RGAMSQ = 5.9604645e-8;

    protected ?object $blas;
    protected ?bool $forceBlas;
    /** @var array<int> $floatTypes */
    protected $floatTypes= [
        NDArray::float16,NDArray::float32,NDArray::float64,
    ];

    public function __construct(object $blas=null,bool $forceBlas=null)
    {
        //$this->blas = $blas;
        //$this->forceBlas = $forceBlas;
        $this->blas = null;
        $this->forceBlas = null;
    }

    //public function forceBlas($forceBlas)
    //{
    //    $this->forceBlas = $forceBlas;
    //}

    //protected function useBlas(Buffer $X)
    //{
    //    //if($this->blas===null)
    //    //    return false;
    //    //return $this->forceBlas || in_array($X->dtype(),$this->floatTypes);
    //    return false;
    //}

    public function getNumThreads() : int
    {
        if($this->blas===null)
            return 1;
        return $this->blas->getNumThreads();
    }

    public function getNumProcs() : int
    {
        if($this->blas===null)
            return 1;
        return $this->blas->getNumProcs();
    }

    public function getConfig() : string
    {
        if($this->blas===null)
            return 'PhpBlas';
        return $this->blas->getConfig();
    }

    public function getCorename() : string
    {
        if($this->blas===null)
            return 'PHP';
        return $this->blas->getCorename();
    }

    public function getParallel() : int
    {
        return 0; // parallel mode = 0 : serial 
    }

    protected function sign(float $x,float $y) : float
    {
        if($y<0) {
            $x = -$x;
        }
        return $x;
    }

    /**
     * @param  int $trans BLAS::NoTrans, BLAS::Trans, BLAS::ConjTrans, BLAS::ConjNoTrans
     * @return array<bool> [bool $trans, bool $conj]
     */
    protected function codeToTrans(int $trans) : array
    {
        switch($trans) {
            case BLAS::NoTrans: {
                return [false,false];
            }
            case BLAS::Trans: {
                return [true,false];
            }
            case BLAS::ConjTrans: {
                return [true,true];
            }
            case BLAS::ConjNoTrans: {
                return [false,true];
            }
            default: {
                throw new InvalidArgumentException('Unknown Tranpose Code: '.$trans);
            }
        }
    }

    protected function cleanComplexNumber(mixed $value,string $name) : object
    {
        if(!$this->cisobject($value)) {
            throw new RuntimeException("$name is not complex");
        }
        return $value;
    }

    protected function cleanFloatNumber(mixed $value,string $name) : float
    {
        if(!is_numeric($value)) {
            throw new RuntimeException("$name is not float");
        }
        return (float)$value;
    }

    public function scal(
        int $n,
        float|object $alpha,
        Buffer $X, int $offsetX, int $incX) : void
    {
        $this->assertShapeParameter('n',$n);
        $this->assertVectorBufferSpec('X', $X, $n, $offsetX, $incX);

        $idx = $offsetX;
        if($this->cistype($X->dtype())) {
            $alpha = $this->cleanComplexNumber($alpha,'alpha');
            for ($i=0; $i<$n; $i++,$idx+=$incX) {
                $X[$idx] = $this->cmul($X[$idx],$alpha);
            }
        } else {
            $alpha = $this->cleanFloatNumber($alpha,'alpha');
            for ($i=0; $i<$n; $i++,$idx+=$incX) {
                $X[$idx] = $X[$idx] * $alpha;
            }
        }
    }
    /**
     *  Y := alpha * X + Y
     */
    public function axpy(
        int $n,
        float|object $alpha,
        Buffer $X, int $offsetX, int $incX,
        Buffer $Y, int $offsetY, int $incY ) : void
    {
        $this->assertShapeParameter('n',$n);
        $this->assertVectorBufferSpec('X', $X, $n, $offsetX, $incX);
        $this->assertVectorBufferSpec('Y', $Y, $n, $offsetY, $incY);

        $idxX = $offsetX;
        $idxY = $offsetY;
        if($this->cistype($X->dtype())) {
            $alpha = $this->cleanComplexNumber($alpha,'alpha');
            for ($i=0; $i<$n; $i++,$idxX+=$incX,$idxY+=$incY) {
                $Y[$idxY] = $this->cadd($this->cmul($alpha,$X[$idxX]),$Y[$idxY]);
            }
        } else {
            $alpha = $this->cleanFloatNumber($alpha,'alpha');
            if($alpha==1.0) {   // Y := X + Y
                for ($i=0; $i<$n; $i++,$idxX+=$incX,$idxY+=$incY) {
                    $Y[$idxY] = $X[$idxX] + $Y[$idxY];
                }
            } else {            // Y := a*X + Y
                for ($i=0; $i<$n; $i++,$idxX+=$incX,$idxY+=$incY) {
                    $Y[$idxY] = $alpha * $X[$idxX] + $Y[$idxY];
                }
            }
        }
    }

    public function dot(
        int $n,
        Buffer $X, int $offsetX, int $incX,
        Buffer $Y, int $offsetY, int $incY ) : float|object
    {
        if($this->cistype($X->dtype())) {
            throw new InvalidArgumentException('Unsuppored data type.');
        }
        $this->assertShapeParameter('n',$n);
        $this->assertVectorBufferSpec('X', $X, $n, $offsetX, $incX);
        $this->assertVectorBufferSpec('Y', $Y, $n, $offsetY, $incY);

        $idxX = $offsetX;
        $idxY = $offsetY;
        if($this->cistype($X->dtype())) {
            $acc = $this->cbuild(0.0);
            for ($i=0; $i<$n; $i++,$idxX+=$incX,$idxY+=$incY) {
                $acc = $this->cadd($acc,$this->cmul($X[$idxX],$Y[$idxY]));
            }
        } else {
            $acc = 0.0;
            for ($i=0; $i<$n; $i++,$idxX+=$incX,$idxY+=$incY) {
                $acc += $X[$idxX] * $Y[$idxY];
            }
        }
        return $acc;
    }

    public function dotu(
        int $n,
        Buffer $X, int $offsetX, int $incX,
        Buffer $Y, int $offsetY, int $incY ) : float|object
    {
        if(!$this->cistype($X->dtype())) {
            throw new InvalidArgumentException('Unsuppored data type.');
        }
        $this->assertShapeParameter('n',$n);
        $this->assertVectorBufferSpec('X', $X, $n, $offsetX, $incX);
        $this->assertVectorBufferSpec('Y', $Y, $n, $offsetY, $incY);

        $idxX = $offsetX;
        $idxY = $offsetY;
        $acc = $this->cbuild(0.0);
        for ($i=0; $i<$n; $i++,$idxX+=$incX,$idxY+=$incY) {
            $acc = $this->cadd($acc,$this->cmul($X[$idxX],$Y[$idxY]));
        }
        return $acc;
    }

    public function dotc(
        int $n,
        Buffer $X, int $offsetX, int $incX,
        Buffer $Y, int $offsetY, int $incY ) : float|object
    {
        if(!$this->cistype($X->dtype())) {
            throw new InvalidArgumentException('Unsuppored data type.');
        }
        $this->assertShapeParameter('n',$n);
        $this->assertVectorBufferSpec('X', $X, $n, $offsetX, $incX);
        $this->assertVectorBufferSpec('Y', $Y, $n, $offsetY, $incY);

        $idxX = $offsetX;
        $idxY = $offsetY;
        $acc = $this->cbuild(0.0);
        for ($i=0; $i<$n; $i++,$idxX+=$incX,$idxY+=$incY) {
            $acc = $this->cadd($acc,$this->cmul($this->cconj($X[$idxX]),$Y[$idxY]));
        }
        return $acc;
    }

    public function asum(
        int $n,
        Buffer $X, int $offsetX, int $incX ) : float
    {
        $this->assertShapeParameter('n',$n);
        $this->assertVectorBufferSpec('X', $X, $n, $offsetX, $incX);

        $idxX = $offsetX;
        if($this->cistype($X->dtype())) {
            $acc = 0.0;
            for ($i=0; $i<$n; $i++,$idxX+=$incX) {
                $acc += $this->cabs($X[$idxX]);
            }
        } else {
            $acc = 0.0;
            for ($i=0; $i<$n; $i++,$idxX+=$incX) {
                $acc += abs($X[$idxX]);
            }
        }
        return $acc;
    }

    public function iamax(
        int $n,
        Buffer $X, int $offsetX, int $incX ) : int
    {
        $this->assertShapeParameter('n',$n);
        $this->assertVectorBufferSpec('X', $X, $n, $offsetX, $incX);

        $idxX = $offsetX+$incX;
        $idx = 0;
        if($this->cistype($X->dtype())) {
            $acc = $this->cabs($X[$offsetX]);
            for($i=1; $i<$n; $i++,$idxX+=$incX) {
                $abs = $this->cabs($X[$idxX]);
                if($acc < $abs) {
                    $acc = $abs;
                    $idx = $i;
                }
            }
        } else {
            $acc = abs($X[$offsetX]);
            for($i=1; $i<$n; $i++,$idxX+=$incX) {
                $abs = abs($X[$idxX]);
                if($acc < $abs) {
                    $acc = $abs;
                    $idx = $i;
                }
            }
        }
        return $idx;
    }

    public function iamin(
        int $n,
        Buffer $X, int $offsetX, int $incX ) : int
    {
        $this->assertShapeParameter('n',$n);
        $this->assertVectorBufferSpec('X', $X, $n, $offsetX, $incX);

        $idxX = $offsetX+$incX;
        $idx = 0;
        if($this->cistype($X->dtype())) {
            $acc = $this->cabs($X[$offsetX]);
            for($i=1; $i<$n; $i++,$idxX+=$incX) {
                $abs = $this->cabs($X[$idxX]);
                if($acc > $abs) {
                    $acc = $abs;
                    $idx = $i;
                }
            }
        } else {
            $acc = abs($X[$offsetX]);
            for($i=1; $i<$n; $i++,$idxX+=$incX) {
                $abs = abs($X[$idxX]);
                if($acc > $abs) {
                    $acc = $abs;
                    $idx = $i;
                }
            }
        }
        return $idx;
    }

    public function copy(
        int $n,
        Buffer $X, int $offsetX, int $incX,
        Buffer $Y, int $offsetY, int $incY ) : void
    {
        $this->assertShapeParameter('n',$n);
        $this->assertVectorBufferSpec('X', $X, $n, $offsetX, $incX);
        $this->assertVectorBufferSpec('Y', $Y, $n, $offsetY, $incY);

        $idxX = $offsetX;
        $idxY = $offsetY;
        for($i=0; $i<$n; $i++,$idxX+=$incX,$idxY+=$incY) {
            $Y[$idxY] = $X[$idxX];
        }
    }

    public function nrm2(
        int $n,
        Buffer $X, int $offsetX, int $incX
        ) : float
    {
        $this->assertShapeParameter('n',$n);
        $this->assertVectorBufferSpec('X', $X, $n, $offsetX, $incX);

        $idxX = $offsetX;
        // Y := sqrt(sum(Xn ** 2))
        if($this->cistype($X->dtype())) {
            $sum = 0.0;
            for ($i=0; $i<$n; $i++,$idxX+=$incX) {
                $real = $X[$idxX]->real;
                $imag = $X[$idxX]->imag;
                $sum += $real*$real +  $imag*$imag;
            }
            $Y = sqrt($sum);
        } else {
            $sum = 0.0;
            for ($i=0; $i<$n; $i++,$idxX+=$incX) {
                $v = $X[$idxX];
                $sum += $v*$v;
            }
            $Y = sqrt($sum);
        }
        return $Y;
    }
/*
    public function rotg(
        Buffer $A, int $offsetA,
        Buffer $B, int $offsetB,
        Buffer $C, int $offsetC,
        Buffer $S, int $offsetS
        ) : void
    {
        if($this->cistype($A->dtype())) {
            throw new InvalidArgumentException('Unsuppored data type.');
        }
        $a = $A[$offsetA];
        $b = $B[$offsetB];
        // r
        if(abs($a)>abs($b)) {
            $r = $this->sign(sqrt($a**2 + $b**2),$a);
        } else {
            $r = $this->sign(sqrt($a**2 + $b**2),$b);
        }
        // c
        if($r!=0) {
            $c = $a/$r;
        } else {
            $c = 1;
        }
        // s
        if($r!=0) {
            $s = $a/$r;
        } else {
            $s = 0;
        }
        // z
        if(abs($a)>abs($b)) {
            $z = $s;
        } else {
            if($r!=0) {
                if($c!=0) {
                    $z = 1/$c;
                } else {
                    $z = 1;
                }
            } else {
                $z = 0;
            }
        }
        $A[$offsetA] = $r;
        $B[$offsetB] = $z;
        $C[$offsetC] = $c;
        $S[$offsetS] = $s;
    }
*/
    public function rotg(
        Buffer $A, int $offsetA,
        Buffer $B, int $offsetB,
        Buffer $C, int $offsetC,
        Buffer $S, int $offsetS
        ) : void
    {
        if($this->cistype($A->dtype())) {
            throw new InvalidArgumentException('Unsuppored data type.');
        }
        $a = $A[$offsetA];
        $b = $B[$offsetB];
        $absa = abs($a);
        $absb = abs($b);

        if($absb == 0.0) {
            $c = 1.0;
            $s = 0.0;
            $r = $a;
            $z =  0.0;
        } elseif($absa == 0.0) {
            $c = 0.0;
            $s = 1.0;
            $r = $b;
            $z = 1.0;
        } else {
            $safmin = 1.0e-37;
            $safmax = 1/$safmin;
            $scale = min( max($safmin,max($absa,$absb)), $safmax);
            if ($absa > $absb) {
                $sigma = ($a>=0.0)? 1.0:-1.0;
            } else {
                $sigma = ($b>=0.0)? 1.0:-1.0;
            }
            $dascal = $a / $scale;
            $dbscal = $b / $scale;
            $r = $sigma * ($scale * sqrt($dascal * $dascal + $dbscal * $dbscal));
            $c = $a / $r;
            $s = $b / $r;
            $z = 1.0;
            if($absa > $absb) {
                $z = $s;
            }
            if(($absa <= $absb) && ($c != 0.0)) {
                $z = 1.0 / $c;
            }
        }

        $A[$offsetA] = $r;
        $B[$offsetB] = $z;
        $C[$offsetC] = $c;
        $S[$offsetS] = $s;
    }

    public function rotmg(
        Buffer $D1, int $offsetD1,
        Buffer $D2, int $offsetD2,
        Buffer $B1, int $offsetB1,
        Buffer $B2, int $offsetB2,
        Buffer $P,  int $offsetP,
    ) : void
    {
        $dd1 = $D1[$offsetD1];
        $dd2 = $D2[$offsetD2];
        $dx1 = $B1[$offsetB1];
        $dy1 = $B2[$offsetB2];
        $dh11 = 0.0;
        $dh21 = 0.0;
        $dh12 = 0.0;
        $dh22 = 0.0;
        $dflag = -1;
        
        if($dd2 == 0.0 || $dy1 == 0.0)
        {
            $dflag = -2.0;
            $P[$offsetP+0] = $dflag;
            return;
        }
    
        if($dd1 < 0.0)
        {
            $dflag = -1.0;
            $dh11  = 0.0;
            $dh12  = 0.0;
            $dh21  = 0.0;
            $dh22  = 0.0;
    
            $dd1  = 0.0;
            $dd2  = 0.0;
            $dx1  = 0.0;
        }
        else if (($dd1 == 0.0 || $dx1 == 0.0) && $dd2 > 0.0)
        {
            $dflag = 1.0;
            $dh12 = 1;
            $dh21 = -1;
            $dx1 = $dy1;
            $dtemp = $dd1;
            $dd1 = $dd2;
            $dd2 = $dtemp;
        } 
        else
        {
            $dp2 = $dd2 * $dy1;
            $dp1 = $dd1 * $dx1;
            $dq2 =  $dp2 * $dy1;
            $dq1 =  $dp1 * $dx1;
            if(abs($dq1) > abs($dq2))
            {
                $dh11  =  1.0;
                $dh22  =  1.0;
                $dh21 = -  $dy1 / $dx1;
                $dh12 =    $dp2 /  $dp1;
    
                $du   = 1.0 - $dh12 * $dh21;
                $dflag = 0.0;
                $dd1  = $dd1 / $du;
                $dd2  = $dd2 / $du;
                $dx1  = $dx1 * $du;
                
            }
            else
            {
                if($dq2 < 0.0)
                {
                    $dflag = -1.0;
    
                    $dh11  = 0.0;
                    $dh12  = 0.0;
                    $dh21  = 0.0;
                    $dh22  = 0.0;
    
                    $dd1  = 0.0;
                    $dd2  = 0.0;
                    $dx1  = 0.0;
                }
                else
                {
                    $dflag =  1.0;
                    $dh21  = -1.0;
                    $dh12  =  1.0;
    
                    $dh11  =  $dp1 /  $dp2;
                    $dh22  = $dx1 /  $dy1;
                    $du    =  1.0 + $dh11 * $dh22;
                    $dtemp = $dd2 / $du;
    
                    $dd2  = $dd1 / $du;
                    $dd1  = $dtemp;
                    $dx1  = $dy1 * $du;
                }
            }
    
    
            while ( $dd1 <= self::RGAMSQ && $dd1 != 0.0)
            {
                $dflag = -1.0;
                $dd1  = $dd1 * (self::GAM * self::GAM);
                $dx1  = $dx1 / self::GAM;
                $dh11  = $dh11 / self::GAM;
                $dh12  = $dh12 / self::GAM;
            }
            while (abs($dd1) > self::GAMSQ) {
                $dflag = -1.0;
                $dd1  = $dd1 / (self::GAM * self::GAM);
                $dx1  = $dx1 * self::GAM;
                $dh11  = $dh11 * self::GAM;
                $dh12  = $dh12 * self::GAM;
            }
    
            while (abs($dd2) <= self::RGAMSQ && $dd2 != 0.0) {
                $dflag = -1.0;
                $dd2  = $dd2 * (self::GAM * self::GAM);
                $dh21  = $dh21 / self::GAM;
                $dh22  = $dh22 / self::GAM;
            }
            while (abs($dd2) > self::GAMSQ) {
                $dflag = -1.0;
                $dd2  = $dd2 / (self::GAM * self::GAM);
                $dh21  = $dh21 * self::GAM;
                $dh22  = $dh22 * self::GAM;
            }
    
        }
    
        if($dflag < 0.0)
        {
            $P[$offsetP+1] = $dh11;
            $P[$offsetP+2] = $dh21;
            $P[$offsetP+3] = $dh12;
            $P[$offsetP+4] = $dh22;
        }
        else
        {
            if($dflag == 0.0)
            {
                $P[$offsetP+2] = $dh21;
                $P[$offsetP+3] = $dh12;
            }
            else
            {
                $P[$offsetP+1] = $dh11;
                $P[$offsetP+4] = $dh22;
            }
        }
    
        $P[$offsetP+0] = $dflag;

        $D1[$offsetD1] = $dd1;
        $D2[$offsetD2] = $dd2;
        $B1[$offsetB1] = $dx1;
    }

    public function rot(
        int $n,
        Buffer $X, int $offsetX, int $incX,
        Buffer $Y, int $offsetY, int $incY,
        Buffer $C, int $offsetC,
        Buffer $S, int $offsetS
        ) : void
    {
        if($this->cistype($X->dtype())) {
            throw new InvalidArgumentException('Unsuppored data type.');
        }
        $cc = $C[$offsetC];
        $ss = $S[$offsetS];
        $idX = $offsetX;
        $idY = $offsetY;
        for($i=0;$i<$n;$i++,$idX+=$incX,$idY+=$incY) {
            $xx = $X[$idX];
            $yy = $Y[$idY];
            $X[$idX] =  $cc * $xx + $ss * $yy;
            $Y[$idY] = -$ss * $xx + $cc * $yy;
        }
    }

    public function rotm(
        int $n,
        Buffer $X, int $offsetX, int $incX,
        Buffer $Y, int $offsetY, int $incY,
        Buffer $P,  int $offsetP,
    ) : void
    {
        $dflag = $P[$offsetP+0];
        if ($n <= 0 || $dflag == - 2.0) {
            // flag -2
            return;
        }
    
        $kx = $offsetX;
        $ky = $offsetY;
        if ($incX < 0) {
            $kx = (1 - $n) * $incX + $offsetX;
        }
        if ($incY < 0) {
            $ky = (1 - $n) * $incY + $offsetY;
        }
    
        if ($dflag < 0) {
            // flag = -1
            $dh11 = $P[$offsetP+1];
            $dh12 = $P[$offsetP+3];
            $dh21 = $P[$offsetP+2];
            $dh22 = $P[$offsetP+4];
            for ($i=0; $i<$n; ++$i) {
                $w = $X[$kx];
                $z = $Y[$ky];
                $X[$kx] = $w * $dh11 + $z * $dh12;
                $Y[$ky] = $w * $dh21 + $z * $dh22;
                $kx += $incX;
                $ky += $incY;
            }
        } elseif($dflag == 0) {
            // flag = 0
            $dh12 = $P[$offsetP+3];
            $dh21 = $P[$offsetP+2];
            for ($i=0; $i<$n; ++$i) {
                $w = $X[$kx];
                $z = $Y[$ky];
                $X[$kx] = $w + $z * $dh12;
                $Y[$ky] = $w * $dh21 + $z;
                $kx += $incX;
                $ky += $incY;
            }
        } else {
            // flag = 1
            $dh11 = $P[$offsetP+1];
            $dh22 = $P[$offsetP+4];
            for ($i=0; $i<$n; ++$i) {
                $w = $X[$kx];
                $z = $Y[$ky];
                $X[$kx] = $w * $dh11 + $z;
                $Y[$ky] = -$w + $dh22 * $z;
                $kx += $incX;
                $ky += $incY;
            }
        }
    }

    public function swap(
        int $n,
        Buffer $X, int $offsetX, int $incX,
        Buffer $Y, int $offsetY, int $incY ) : void
    {
        $this->assertShapeParameter('n',$n);
        $this->assertVectorBufferSpec('X', $X, $n, $offsetX, $incX);
        $this->assertVectorBufferSpec('Y', $Y, $n, $offsetY, $incY);

        $idxX = $offsetX;
        $idxY = $offsetY;
        for($i=0; $i<$n; $i++,$idxX+=$incX,$idxY+=$incY) {
            $tmp = $Y[$idxY];
            $Y[$idxY] = $X[$idxX];
            $X[$idxX] = $tmp;
        }
    }

    public function gemv(
        int $order,
        int $trans,
        int $m,
        int $n,
        float|object $alpha,
        Buffer $A, int $offsetA, int $ldA,
        Buffer $X, int $offsetX, int $incX,
        float|object $beta,
        Buffer $Y, int $offsetY, int $incY ) : void
    {
        if($order==BLAS::ColMajor) {
            [$m,$n] = [$n,$m];
        } elseif($order!=BLAS::RowMajor) {
            throw new InvalidArgumentException('Invalid Order type');
        }
        [$trans,$conj] = $this->codeToTrans($trans);
        $rows = (!$trans) ? $m : $n;
        $cols = (!$trans) ? $n : $m;

        $this->assertShapeParameter('m',$m);
        $this->assertShapeParameter('n',$n);
        $this->assertMatrixBufferSpec("A", $A, $m, $n, $offsetA, $ldA);

        $this->assertVectorBufferSpec('X', $X, $cols, $offsetX, $incX);
        $this->assertVectorBufferSpec('Y', $Y, $rows, $offsetY, $incY);

        $ldA_i = (!$trans) ? $ldA : 1;
        $ldA_j = (!$trans) ? 1 : $ldA;

        $idA_i = $offsetA;
        $idY = $offsetY;
        if($this->cistype($X->dtype())) {
            $alpha = $this->cleanComplexNumber($alpha,'alpha');
            $beta = $this->cleanComplexNumber($beta,'beta');
            $hasAlpha = !$this->cisone($alpha);
            $hasBeta = !$this->ciszero($beta);
            $betaIsNotOne = !$this->cisone($beta);
            for($i=0; $i<$rows; $i++,$idA_i+=$ldA_i,$idY+=$incY) {
                $idA = $idA_i;
                $idX = $offsetX;
                $acc = $this->cbuild(0.0);
                for($j=0; $j<$cols; $j++,$idA+=$ldA_j,$idX+=$incX) {
                    // acc += alpha*A*X
                    $v = $A[$idA];
                    if($conj) {
                        $v = $this->cconj($v);
                    }
                    $v = $this->cmul($v,$X[$idX]);
                    if($hasAlpha) {
                        $v = $this->cmul($alpha,$v);
                    }
                    $acc = $this->cadd($acc,$v);
                }
                // Y = acc+beta*Y
                if($hasBeta) {
                    $v = $Y[$idY];
                    if($betaIsNotOne) {
                        $v = $this->cmul($beta,$v);
                    }
                    $acc = $this->cadd($acc,$v);
                }
                $Y[$idY] = $acc;
            }
        } else {
            $alpha = $this->cleanFloatNumber($alpha,'alpha');
            $beta = $this->cleanFloatNumber($beta,'beta');
            $hasBeta  = $beta!=0.0;
            for ($i=0; $i<$rows; $i++,$idA_i+=$ldA_i,$idY+=$incY) {
                $idA = $idA_i;
                $idX = $offsetX;
                $acc = 0.0;
                for ($j=0; $j<$cols; $j++,$idA+=$ldA_j,$idX+=$incX) {
                    $acc += $alpha * $A[$idA] * $X[$idX];
                }
                if($hasBeta) {
                    $Y[$idY] = $acc + $beta * $Y[$idY];
                } else {
                    $Y[$idY] = $acc;
                }
            }
        }
    }

    public function gemm(
        int $order,
        int $transA,
        int $transB,
        int $m,
        int $n,
        int $k,
        float|object $alpha,
        Buffer $A, int $offsetA, int $ldA,
        Buffer $B, int $offsetB, int $ldB,
        float|object $beta,
        Buffer $C, int $offsetC, int $ldC ) : void
    {
        if($order==BLAS::ColMajor) {
            [$m,$n] = [$n,$m];
        } elseif($order!=BLAS::RowMajor) {
            throw new InvalidArgumentException('Invalid Order type');
        }
        [$transA,$conjA] = $this->codeToTrans($transA);
        [$transB,$conjB] = $this->codeToTrans($transB);

        $this->assertShapeParameter('m',$m);
        $this->assertShapeParameter('n',$n);
        $this->assertShapeParameter('k',$k);

        $rowsA = (!$transA) ? $m : $k;
        $colsA = (!$transA) ? $k : $m;
        $rowsB = (!$transB) ? $k : $n;
        $colsB = (!$transB) ? $n : $k;

        $this->assertMatrixBufferSpec("A", $A, $rowsA, $colsA, $offsetA, $ldA);
        $this->assertMatrixBufferSpec("B", $B, $rowsB, $colsB, $offsetB, $ldB);
        $this->assertMatrixBufferSpec("C", $C, $m, $n, $offsetC, $ldC);

        $ldA_m = (!$transA) ? $ldA : 1;
        $ldA_k = (!$transA) ? 1 : $ldA;
        $ldB_k = (!$transB) ? $ldB : 1;
        $ldB_n = (!$transB) ? 1 : $ldB;

        $idA_m = $offsetA;
        $idC_m = $offsetC;
        if($this->cistype($A->dtype())) {
            $alpha = $this->cleanComplexNumber($alpha,'alpha');
            $beta = $this->cleanComplexNumber($beta,'beta');
            $hasAlpha = !$this->cisone($alpha);
            $hasBeta = !$this->ciszero($beta);
            $betaIsNotOne = !$this->cisone($beta);
            for ($im=0; $im<$m; $im++,$idA_m+=$ldA_m,$idC_m+=$ldC) {
                $idB_n = $offsetB;
                $idC = $idC_m;
                for ($in=0; $in<$n; $in++,$idB_n+=$ldB_n,$idC++) {
                    $idA = $idA_m;
                    $idB = $idB_n;
                    $acc = $this->cbuild(0.0);
                    for ($ik=0; $ik<$k; $ik++,$idA+=$ldA_k,$idB+=$ldB_k) {
                        $valueA = $A[$idA];
                        $valueB = $B[$idB];
                        if($conjA) {
                            $valueA = $this->cconj($valueA);
                        }
                        if($conjB) {
                            $valueB = $this->cconj($valueB);
                        }
                        $acc = $this->cadd($acc,$this->cmul($valueA,$valueB));
                    }
                    if($hasAlpha) {
                        $acc = $this->cmul($alpha,$acc);
                    }
                    if($hasBeta) {
                        $v = $C[$idC];
                        if($betaIsNotOne) {
                            $v = $this->cmul($beta,$v);
                        }
                        $acc = $this->cadd($acc,$v);
                    }
                    $C[$idC] = $acc;
                }
            }
        } else {
            $alpha = $this->cleanFloatNumber($alpha,'alpha');
            $beta = $this->cleanFloatNumber($beta,'beta');
            for ($im=0; $im<$m; $im++,$idA_m+=$ldA_m,$idC_m+=$ldC) {
                $idB_n = $offsetB;
                $idC = $idC_m;
                for ($in=0; $in<$n; $in++,$idB_n+=$ldB_n,$idC++) {
                    $idA = $idA_m;
                    $idB = $idB_n;
                    $acc = 0.0;
                    for ($ik=0; $ik<$k; $ik++,$idA+=$ldA_k,$idB+=$ldB_k) {
                        $acc += $A[$idA] * $B[$idB];
                    }
                    if($beta==0.0) {
                        $C[$idC] = $alpha * $acc;
                    } else {
                        $C[$idC] = $alpha * $acc + $beta * $C[$idC];
                    }
                }
            }
        }
    }

    public function symm(
        int $order,
        int $side,
        int $uplo,
        int $m,
        int $n,
        float|object $alpha,
        Buffer $A, int $offsetA, int $ldA,
        Buffer $B, int $offsetB, int $ldB,
        float|object $beta,
        Buffer $C, int $offsetC, int $ldC ) : void
    {
        if($order==BLAS::ColMajor) {
            [$m,$n] = [$n,$m];
        } elseif($order!=BLAS::RowMajor) {
            throw new InvalidArgumentException('Invalid Order type');
        }
        $this->assertShapeParameter('m',$m);
        $this->assertShapeParameter('n',$n);

        $sizeA = ($side==BLAS::Left) ? $m : $n;
        $this->assertMatrixBufferSpec("A", $A, $sizeA, $sizeA, $offsetA, $ldA);
        $this->assertMatrixBufferSpec("B", $B, $m, $n, $offsetB, $ldB);
        $this->assertMatrixBufferSpec("C", $C, $m, $n, $offsetC, $ldC);

        $ldA_m = ($uplo==BLAS::Upper) ? $ldA : 1;
        $ldA_k = ($uplo==BLAS::Upper) ? 1 : $ldA;
        $ldB_k = ($side==BLAS::Left) ? $ldB : 1;
        $ldB_n = ($side==BLAS::Left) ? 1 : $ldB;
        $ldC_m = ($side==BLAS::Left) ? $ldC : 1;
        $ldC_n = ($side==BLAS::Left) ? 1 : $ldC;
        if($side==BLAS::Right) {
            [$n,$m] = [$m,$n];
        }

        $idA_m = $offsetA;
        $idC_m = $offsetC;
        if($this->cistype($A->dtype())) {
            $alpha = $this->cleanComplexNumber($alpha,'alpha');
            $beta = $this->cleanComplexNumber($beta,'beta');
            $hasAlpha = !$this->cisone($alpha);
            $hasBeta = !$this->ciszero($beta);
            $betaIsNotOne = !$this->cisone($beta);
            for($im=0; $im<$m; $im++,$idC_m+=$ldC_m) {
                $idB_n = $offsetB;
                $idC = $idC_m;
                for ($in=0; $in<$n; $in++,$idB_n+=$ldB_n,$idC+=$ldC_n) {
                    $idB = $idB_n;
                    $acc = $this->cbuild(0.0);
                    for ($ik=0; $ik<$sizeA; $ik++,$idB+=$ldB_k) {
                        if($ik<$im) {
                            $idA = $offsetA+$ik*$ldA_m+$im*$ldA_k;
                        } else {
                            $idA = $offsetA+$im*$ldA_m+$ik*$ldA_k;
                        }
                        $acc = $this->cadd($acc,$this->cmul($A[$idA],$B[$idB]));
                    }
                    if($hasAlpha) {
                        $acc = $this->cmul($alpha,$acc);
                    }
                    if($hasBeta) {
                        $v = $C[$idC];
                        if($betaIsNotOne) {
                            $v = $this->cmul($beta,$v);
                        }
                        $acc = $this->cadd($acc,$v);
                    }
                    $C[$idC] = $acc;
                }
            }
        } else {
            $alpha = $this->cleanFloatNumber($alpha,'alpha');
            $beta = $this->cleanFloatNumber($beta,'beta');
            for ($im=0; $im<$m; $im++,$idC_m+=$ldC_m) {
                $idB_n = $offsetB;
                $idC = $idC_m;
                for ($in=0; $in<$n; $in++,$idB_n+=$ldB_n,$idC+=$ldC_n) {
                    $idB = $idB_n;
                    $acc = 0.0;
                    for ($ik=0; $ik<$sizeA; $ik++,$idB+=$ldB_k) {
                        if($ik<$im) {
                            $idA = $offsetA+$ik*$ldA_m+$im*$ldA_k;
                        } else {
                            $idA = $offsetA+$im*$ldA_m+$ik*$ldA_k;
                        }
                        $acc += $A[$idA] * $B[$idB];
                    }
                    if($beta==0.0) {
                        $C[$idC] = $alpha * $acc;
                    } else {
                        $C[$idC] = $alpha * $acc + $beta * $C[$idC];
                    }
                }
            }
        }
    }

    public function syrk(
        int $order,
        int $uplo,
        int $trans,
        int $n,
        int $k,
        float|object $alpha,
        Buffer $A, int $offsetA, int $ldA,
        float|object $beta,
        Buffer $C, int $offsetC, int $ldC ) : void
    {
        if($order==BLAS::ColMajor) {
            [$n,$k] = [$k,$n];
        } elseif($order!=BLAS::RowMajor) {
            throw new InvalidArgumentException('Invalid Order type');
        }
        [$trans,$conj] = $this->codeToTrans($trans);
        $this->assertShapeParameter('n',$n);
        $this->assertShapeParameter('k',$k);

        $rows = (!$trans) ? $n : $k;
        $cols = (!$trans) ? $k : $n;
        $this->assertMatrixBufferSpec("A", $A, $rows, $cols, $offsetA, $ldA);
        $this->assertMatrixBufferSpec("C", $C, $n, $n, $offsetC, $ldC);

        $ldA_m  = (!$trans) ? $ldA : 1;
        $ldA_k  = (!$trans) ? 1 : $ldA;
        $ldAT_k = ($trans)  ? $ldA : 1;
        $ldAT_n = ($trans)  ? 1 : $ldA;

        $idA_m = $offsetA;
        $idC_m = $offsetC;
        if($this->cistype($A->dtype())) {
            $alpha = $this->cleanComplexNumber($alpha,'alpha');
            $beta = $this->cleanComplexNumber($beta,'beta');
            $hasAlpha = !$this->cisone($alpha);
            $hasBeta = !$this->ciszero($beta);
            $betaIsNotOne = !$this->cisone($beta);
            for ($im=0; $im<$n; $im++,$idA_m+=$ldA_m,$idC_m+=$ldC) {
                $idAT_n = $offsetA;
                $idC = $idC_m;
                if($uplo==Blas::Upper) {
                    $start_n = $im;
                    $end_n = $n;
                } else {
                    $start_n = 0;
                    $end_n = $im+1;
                }
                for ($in=$start_n; $in<$end_n; $in++,$idAT_n+=$ldAT_n,$idC++) {
                    $acc = $this->cbuild(0.0);
                    for ($ik=0; $ik<$k; $ik++) {
                        $idA  = $offsetA+$im*$ldA_m+$ik*$ldA_k;
                        $idAT = $offsetA+$ik*$ldAT_k+$in*$ldAT_n;
                        $valueA  = $A[$idA];
                        $valueAT = $A[$idAT];
                        if($conj) {
                            $valueA  = $this->cconj($valueA);
                            $valueAT = $this->cconj($valueAT);
                        }
                        $acc = $this->cadd($acc,$this->cmul($valueA,$valueAT));
                    }
                    $idC = $im*$ldC+$in;
                    if($hasAlpha) {
                        $acc = $this->cmul($alpha,$acc);
                    }
                    if($hasBeta) {
                        $v = $C[$idC];
                        if($betaIsNotOne) {
                            $v = $this->cmul($beta,$v);
                        }
                        $acc = $this->cadd($acc,$v);
                    }
                    $C[$idC] = $acc;
                }
            }
        } else {
            $alpha = $this->cleanFloatNumber($alpha,'alpha');
            $beta = $this->cleanFloatNumber($beta,'beta');
            for ($im=0; $im<$n; $im++,$idA_m+=$ldA_m,$idC_m+=$ldC) {
                $idAT_n = $offsetA;
                $idC = $idC_m;
                if($uplo==Blas::Upper) {
                    $start_n = $im;
                    $end_n = $n;
                } else {
                    $start_n = 0;
                    $end_n = $im+1;
                }
                for ($in=$start_n; $in<$end_n; $in++,$idAT_n+=$ldAT_n,$idC++) {
                    $acc = 0.0;
                    for ($ik=0; $ik<$k; $ik++) {
                        $idA  = $offsetA+$im*$ldA_m+$ik*$ldA_k;
                        $idAT = $offsetA+$ik*$ldAT_k+$in*$ldAT_n;
                        $acc += $A[$idA] * $A[$idAT];
                    }
                    $idC = $im*$ldC+$in;
                    if($beta==0.0) {
                        $C[$idC] = $alpha * $acc;
                    } else {
                        $C[$idC] = $alpha * $acc + $beta * $C[$idC];
                    }
                }
            }
        }
    }

    public function syr2k(
        int $order,
        int $uplo,
        int $trans,
        int $n,
        int $k,
        float|object $alpha,
        Buffer $A, int $offsetA, int $ldA,
        Buffer $B, int $offsetB, int $ldB,
        float|object $beta,
        Buffer $C, int $offsetC, int $ldC ) : void
    {
        if($order==BLAS::ColMajor) {
            [$n,$k] = [$k,$n];
        } elseif($order!=BLAS::RowMajor) {
            throw new InvalidArgumentException('Invalid Order type');
        }
        [$trans,$conj] = $this->codeToTrans($trans);

        $this->assertShapeParameter('n',$n);
        $this->assertShapeParameter('k',$k);

        $rows = (!$trans) ? $n : $k;
        $cols = (!$trans) ? $k : $n;

        $this->assertMatrixBufferSpec("A", $A, $rows, $cols, $offsetA, $ldA);
        $this->assertMatrixBufferSpec("B", $B, $rows, $cols, $offsetB, $ldB);
        $this->assertMatrixBufferSpec("C", $C, $n, $n, $offsetC, $ldC);

        $ldA_m  = (!$trans) ? $ldA : 1;
        $ldA_k  = (!$trans) ? 1 : $ldA;
        $ldAT_k = ($trans) ? $ldA : 1;
        $ldAT_n = ($trans) ? 1 : $ldA;
        $ldB_m  = (!$trans) ? $ldB : 1;
        $ldB_k  = (!$trans) ? 1 : $ldB;
        $ldBT_k = ($trans) ? $ldB : 1;
        $ldBT_n = ($trans) ? 1 : $ldB;

        $idA_m = $offsetA;
        $idC_m = $offsetC;
        if($this->cistype($A->dtype())) {
            $alpha = $this->cleanComplexNumber($alpha,'alpha');
            $beta = $this->cleanComplexNumber($beta,'beta');
            $hasAlpha = !$this->cisone($alpha);
            $hasBeta = !$this->ciszero($beta);
            $betaIsNotOne = !$this->cisone($beta);
            for ($im=0; $im<$n; $im++,$idA_m+=$ldA_m,$idC_m+=$ldC) {
                $idAT_n = $offsetA;
                $idC = $idC_m;
                if($uplo==Blas::Upper) {
                    $start_n = $im;
                    $end_n = $n;
                } else {
                    $start_n = 0;
                    $end_n = $im+1;
                }
                for ($in=$start_n; $in<$end_n; $in++,$idAT_n+=$ldAT_n,$idC++) {
                    $acc = $this->cbuild(0.0);
                    for ($ik=0; $ik<$k; $ik++) {
                        $idA  = $offsetA+$im*$ldA_m+ $ik*$ldA_k;
                        $idB  = $offsetB+$im*$ldB_m+ $ik*$ldB_k;
                        $idAT = $offsetA+$ik*$ldAT_k+$in*$ldAT_n;
                        $idBT = $offsetB+$ik*$ldBT_k+$in*$ldBT_n;
                        $valueA  = $A[$idA];
                        $valueAT = $A[$idAT];
                        if($conj) {
                            $valueA  = $this->cconj($valueA);
                            $valueAT = $this->cconj($valueAT);
                        }
                        $acc = $this->cadd($acc,$this->cmul($valueA,$B[$idBT]));
                        $acc = $this->cadd($acc,$this->cmul($valueAT,$B[$idB]));
                    }
                    $idC = $im*$ldC+$in;
                    if($hasAlpha) {
                        $acc = $this->cmul($alpha,$acc);
                    }
                    if($hasBeta) {
                        $v = $C[$idC];
                        if($betaIsNotOne) {
                            $v = $this->cmul($beta,$v);
                        }
                        $acc = $this->cadd($acc,$v);
                    }
                    $C[$idC] = $acc;
                }
            }
        } else {
            $alpha = $this->cleanFloatNumber($alpha,'alpha');
            $beta = $this->cleanFloatNumber($beta,'beta');
            for ($im=0; $im<$n; $im++,$idA_m+=$ldA_m,$idC_m+=$ldC) {
                $idAT_n = $offsetA;
                $idC = $idC_m;
                if($uplo==Blas::Upper) {
                    $start_n = $im;
                    $end_n = $n;
                } else {
                    $start_n = 0;
                    $end_n = $im+1;
                }
                for ($in=$start_n; $in<$end_n; $in++,$idAT_n+=$ldAT_n,$idC++) {
                    $acc = 0.0;
                    for ($ik=0; $ik<$k; $ik++) {
                        $idA  = $offsetA+$im*$ldA_m+ $ik*$ldA_k;
                        $idB  = $offsetB+$im*$ldB_m+ $ik*$ldB_k;
                        $idAT = $offsetA+$ik*$ldAT_k+$in*$ldAT_n;
                        $idBT = $offsetB+$ik*$ldBT_k+$in*$ldBT_n;
                        $acc += $A[$idA]  * $B[$idBT];
                        $acc += $A[$idAT] * $B[$idB];
                    }
                    $idC = $im*$ldC+$in;
                    if($beta==0.0) {
                        $C[$idC] = $alpha * $acc;
                    } else {
                        $C[$idC] = $alpha * $acc + $beta * $C[$idC];
                    }
                }
            }
        }
    }


    /**
     *   B(m,n) = alpha * A(m,m)B(m,n)  : side=Left
     *   B(m,n) = alpha * B(m,n)A(n,n)  : side=Right
     */
    public function trmm(
        int $order,
        int $side,  // left or right
        int $uplo,  // upper or lower
        int $trans, // trans A
        int $diag,  // no unit or unit
        int $m,
        int $n,
        float|object $alpha,
        Buffer $A, int $offsetA, int $ldA,
        Buffer $B, int $offsetB, int $ldB) : void
    {
        if($order==BLAS::ColMajor) {
            [$m,$n] = [$n,$m];
        } elseif($order!=BLAS::RowMajor) {
            throw new InvalidArgumentException('Invalid Order type');
        }
        [$trans,$conj] = $this->codeToTrans($trans);

        $this->assertShapeParameter('m',$m);
        $this->assertShapeParameter('n',$n);

        if($side==BLAS::Left) {
            $sizeA = $m;
            $sizeB = $n;
            $right = false;
        } elseif($side==BLAS::Right) {
            $sizeA = $n;
            $sizeB = $m;
            $right = true;
        } else {
            throw new InvalidArgumentException('Invalid side value: '.$side);
        }
        if($uplo==BLAS::Upper) {
            $lower = false;
        } elseif($uplo==BLAS::Lower) {
            $lower = true;
        } else {
            throw new InvalidArgumentException('Invalid uplo value: '.$uplo);
        }
        if($diag==BLAS::NonUnit) {
            $unit = false;
        } elseif($diag==BLAS::Unit) {
            $unit = true;
        } else {
            throw new InvalidArgumentException('Invalid diag value: '.$diag);
        }
        $rowsB = $m;
        $colsB = $n;

        $this->assertMatrixBufferSpec("A", $A, $sizeA, $sizeA, $offsetA, $ldA);
        $this->assertMatrixBufferSpec("B", $B, $rowsB, $colsB, $offsetB, $ldB);

        $trans = $right ? !$trans : $trans;
        $lower = $trans ? !$lower : $lower;
        $ldA_m = $trans ? 1 : $ldA;
        $ldA_k = $trans ? $ldA : 1;
        $ldB_k = $right ? 1 : $ldB;
        $ldB_n = $right ? $ldB : 1;

        $startm = $lower?($sizeA-1):0;
        $stepm =  $lower?(-1):1;
        if($this->cistype($A->dtype())) {
            $alpha = $this->cleanComplexNumber($alpha,'alpha');
            $hasAlpha = !$this->cisone($alpha);
            for($cm=0,$im=$startm;$cm<$sizeA;$cm++,$im+=$stepm) {
                for($in=0;$in<$sizeB;$in++) {
                    if($unit) {
                        $startk = $lower?0:$im+1;
                        $countk = $sizeA-$cm-1;
                        $acc = $B[$offsetB+$im*$ldB_k+$in*$ldB_n];
                    } else {
                        $startk = $lower?0:$im;
                        $countk = $sizeA-$cm;
                        $acc = $this->cbuild(0.0);
                    }
                    for($ck=0,$ik=$startk; $ck<$countk; $ck++,$ik++) {
                        $v = $A[$offsetA+$im*$ldA_m+$ik*$ldA_k];
                        if($conj) {
                            $v = $this->cconj($v);
                        }
                        $acc = $this->cadd($acc,$this->cmul($v,$B[$offsetB+$ik*$ldB_k+$in*$ldB_n]));
                    }
                    if($hasAlpha) {
                        $acc = $this->cmul($alpha,$acc);
                    }
                    $B[$offsetB+$im*$ldB_k+$in*$ldB_n] = $acc;
                }
            }
        } else {
            $alpha = $this->cleanFloatNumber($alpha,'alpha');
            for($cm=0,$im=$startm;$cm<$sizeA;$cm++,$im+=$stepm) {
                for($in=0;$in<$sizeB;$in++) {
                    if($unit) {
                        $startk = $lower?0:$im+1;
                        $countk = $sizeA-$cm-1;
                        $acc = $B[$offsetB+$im*$ldB_k+$in*$ldB_n];
                    } else {
                        $startk = $lower?0:$im;
                        $countk = $sizeA-$cm;
                        $acc = 0.0;
                    }
                    for($ck=0,$ik=$startk; $ck<$countk; $ck++,$ik++) {
                        $acc += $A[$offsetA+$im*$ldA_m+$ik*$ldA_k]*$B[$offsetB+$ik*$ldB_k+$in*$ldB_n];
                    }
                    $B[$offsetB+$im*$ldB_k+$in*$ldB_n] = $alpha * $acc;
                }
            }
        }
    }

    public function trsm(
        int $order,
        int $side,
        int $uplo,
        int $trans,
        int $diag,
        int $m,
        int $n,
        float|object $alpha,
        Buffer $A, int $offsetA, int $ldA,
        Buffer $B, int $offsetB, int $ldB) : void
    {
        if($order==BLAS::ColMajor) {
            [$m,$n] = [$n,$m];
        } elseif($order!=BLAS::RowMajor) {
            throw new InvalidArgumentException('Invalid Order type');
        }
        [$trans,$conj] = $this->codeToTrans($trans);

        $this->assertShapeParameter('m',$m);
        $this->assertShapeParameter('n',$n);

        if($side==BLAS::Left) {
            $sizeA = $m;
            $sizeB = $n;
            $right = false;
        } elseif($side==BLAS::Right) {
            $sizeA = $n;
            $sizeB = $m;
            $right = true;
        } else {
            throw new InvalidArgumentException('Invalid side value: '.$side);
        }
        if($uplo==BLAS::Upper) {
            $lower = false;
        } elseif($uplo==BLAS::Lower) {
            $lower = true;
        } else {
            throw new InvalidArgumentException('Invalid uplo value: '.$uplo);
        }
        if($diag==BLAS::NonUnit) {
            $unit = false;
        } elseif($diag==BLAS::Unit) {
            $unit = true;
        } else {
            throw new InvalidArgumentException('Invalid diag value: '.$diag);
        }
        $rowsB = $m;
        $colsB = $n;
        $this->assertMatrixBufferSpec("A", $A, $sizeA, $sizeA, $offsetA, $ldA);
        $this->assertMatrixBufferSpec("B", $B, $rowsB, $colsB, $offsetB, $ldB);

        $trans = $right ? !$trans : $trans;
        $lower = $trans ? !$lower : $lower;
        $ldA_m = $trans ? 1 : $ldA;
        $ldA_k = $trans ? $ldA : 1;
        $ldB_k = $right ? 1 : $ldB;
        $ldB_n = $right ? $ldB : 1;

        $startm = $lower?0:($sizeA-1);
        $stepm =  $lower?1:(-1);
        if($this->cistype($A->dtype())) {
            $alpha = $this->cleanComplexNumber($alpha,'alpha');
            $hasAlpha = !$this->cisone($alpha);
            // loop(i)
            for($cm=0,$im=$startm;$cm<$sizeA;$cm++,$im+=$stepm) {
                // A[i,i]
                if($unit) {
                    $denomi = 1.0;
                    $denomiFlag = 1; // is_one
                } else {
                    $denomi = $A[$offsetA+$im*$ldA_m+$im*$ldA_k];
                    if($this->ciszero($denomi)) {
                        $denomiFlag = 0; // is_zero
                        $denomi = $this->cbuild(NAN,i:NAN);
                    } else {
                        $denomiFlag = 2; // is_normal
                        if($conj) {
                            //echo "C";
                            $denomi = $this->cconj($denomi);
                        }
                    }
                }
                //echo "denomi:$denomi\n";
                // loop(j)
                //echo "for(j)[$in,$sizeB]\n";
                for($in=0;$in<$sizeB;$in++) {
                    // acc = 0;
                    $startk = $lower?0:($im+1);
                    $countk = $cm;
                    $acc = $this->cbuild(0.0);
                    // loop(k)
                    //echo "for(k)[$startk,$countk]\n";
                    for($ck=0,$ik=$startk; $ck<$countk; $ck++,$ik++) {
                        // acc += A[i,k] * B[k,j];
                        //echo "a[$im,$ik]:";
                        //echo $A[$offsetA+$im*$ldA_m+$ik*$ldA_k].",";
                        //echo "b[$ik,$in],";
                        //echo "b[".($offsetB+$ik*$ldB_k+$in*$ldB_n)."]:";
                        //echo $B[$offsetB+$ik*$ldB_k+$in*$ldB_n].",";
                        $v = $A[$offsetA+$im*$ldA_m+$ik*$ldA_k];
                        if($conj) {
                            //echo "C";
                            $v = $this->cconj($v);
                        }
                        $acc = $this->cadd($acc,$this->cmul($v,$B[$offsetB+$ik*$ldB_k+$in*$ldB_n]));
                        //echo "acc:".$acc.",";
                    }
                    //echo "endfor(k)\n";
                    //echo "acc:".$acc.",";
                    //echo "\n";
                    // B[i,j] = (B[i,j] - acc) / A[i,i];
                    //echo "B[$im,$in]";
                    //echo "[$ldB_k,$ldB_n]";
                    if($hasAlpha) {
                        $acc = $this->cmul($alpha,$acc);
                    }
                    if($denomiFlag==0) { // NAN
                        $B[$offsetB+$im*$ldB_k+$in*$ldB_n] = $denomi;
                    } elseif($denomiFlag==1) { // denomi == 1.0
                        $B[$offsetB+$im*$ldB_k+$in*$ldB_n] = $this->csub($B[$offsetB+$im*$ldB_k+$in*$ldB_n],$acc);
                    } else {
                        $B[$offsetB+$im*$ldB_k+$in*$ldB_n] = $this->cdiv($this->csub($B[$offsetB+$im*$ldB_k+$in*$ldB_n],$acc),$denomi);
                    }
                    //echo "B[".($offsetB+$im*$ldB_k+$in*$ldB_n)."]:";
                    //echo $B[$offsetB+$im*$ldB_k+$in*$ldB_n];
                    //echo "\n";
                }
                //echo "endfor(j)\n";
            }
        } else {
            $alpha = $this->cleanFloatNumber($alpha,'alpha');
            // loop(i)
            for($cm=0,$im=$startm;$cm<$sizeA;$cm++,$im+=$stepm) {
                // A[i,i]
                if($unit) {
                    $denomi = 1.0;
                } else {
                    $denomi = $A[$offsetA+$im*$ldA_m+$im*$ldA_k];
                    if($denomi==0) {
                        $denomi = NAN;
                    }
                }
                // loop(j)
                for($in=0;$in<$sizeB;$in++) {
                    // acc = 0;
                    $startk = $lower?0:($im+1);
                    $countk = $cm;
                    $acc = 0.0;
                    // loop(k)
                    for($ck=0,$ik=$startk; $ck<$countk; $ck++,$ik++) {
                        // acc += A[i,k] * B[k,j];
                        $acc += $A[$offsetA+$im*$ldA_m+$ik*$ldA_k]*$B[$offsetB+$ik*$ldB_k+$in*$ldB_n];
                    }
                    // B[i,j] = (B[i,j] - acc) / A[i,i];
                    $B[$offsetB+$im*$ldB_k+$in*$ldB_n] = ($B[$offsetB+$im*$ldB_k+$in*$ldB_n] - $alpha*$acc) / $denomi;
                }
            }
        }
    }

    public function omatcopy(
        int $order,
        int $trans,
        int $m,
        int $n,
        float|object $alpha,
        Buffer $A, int $offsetA, int $ldA,
        Buffer $B, int $offsetB, int $ldB,
    ) : void
    {
        if($order==BLAS::ColMajor) {
            [$m,$n] = [$n,$m];
        } elseif($order!=BLAS::RowMajor) {
            throw new InvalidArgumentException('Invalid Order type');
        }
        [$trans,$conj] = $this->codeToTrans($trans);
        $this->assertShapeParameter('m',$m);
        $this->assertShapeParameter('n',$n);
        $rows = (!$trans) ? $m : $n;
        $cols = (!$trans) ? $n : $m;

        $this->assertMatrixBufferSpec("A", $A, $m, $n, $offsetA, $ldA);
        $this->assertMatrixBufferSpec("B", $B, $rows, $cols, $offsetB, $ldB);

        // Check Buffer A and B
        if($A->dtype()!=$B->dtype()) {
            throw new InvalidArgumentException("Unmatch data type for A and B");
        }

        $ldA_i = (!$trans) ? $ldA : 1;
        $ldA_j = (!$trans) ? 1 : $ldA;

        $idA_i = $offsetA;
        $idB_i = $offsetB;

        if($this->cistype($A->dtype())) {
            $alpha = $this->cleanComplexNumber($alpha,'alpha');
            $hasAlpha = !$this->cisone($alpha);
            for($i=0; $i<$rows; $i++,$idA_i+=$ldA_i,$idB_i+=$ldB) {
                $idA = $idA_i;
                $idB = $idB_i;
                for($j=0; $j<$cols; $j++,$idA+=$ldA_j,$idB++) {
                    $v = $A[$idA];
                    if($conj) {
                        $v = $this->cconj($v);
                    }
                    if($hasAlpha) {
                        $v = $this->cmul($alpha,$v);
                    }
                    $B[$idB] = $v;
                }
            }
        } else {
            $alpha = $this->cleanFloatNumber($alpha,'alpha');
            for($i=0; $i<$rows; $i++,$idA_i+=$ldA_i,$idB_i+=$ldB) {
                $idA = $idA_i;
                $idB = $idB_i;
                for($j=0; $j<$cols; $j++,$idA+=$ldA_j,$idB++) {
                    $B[$idB] = $alpha * $A[$idA];
                }
            }
        }
    }
    
}
