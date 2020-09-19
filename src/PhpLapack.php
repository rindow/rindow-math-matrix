<?php
namespace Rindow\Math\Matrix;

use ArrayAccess as Buffer;
use RuntimeException;
use InvalidArgumentException;
use Interop\Polite\Math\Matrix\NDArray;

class PhpLapack
{
    protected $defaultFloatType = NDArray::float32;
    protected $lapack;
    protected $forceLapack;
    protected $intTypes= [
        NDArray::int8,NDArray::int16,NDArray::int32,NDArray::int64,
        NDArray::uint8,NDArray::uint16,NDArray::uint32,NDArray::uint64,
    ];
    protected $floatTypes= [
        NDArray::float16,NDArray::float32,NDArray::float64,
    ];

    public function __construct($lapack=null,$forceLapack=null)
    {
        $this->lapack = $lapack;
        $this->forceLapack = $forceLapack;
    }

    public function forceLapack($forceLapack)
    {
        $this->forceLapack = $forceLapack;
    }

    protected function useLapack(Buffer $X)
    {
        if($this->lapack===null)
            return false;
        return $this->forceLapack || in_array($X->dtype(),$this->floatTypes);
    }

    /**
     * Below is the author of the original code
     * @author Yehia Abed
     * @copyright 2010
     * @see https://github.com/d3veloper/SVD
     *
     * **************** CAUTION *********************
     * OpenBLAS has the DGESVD. But it is difficult to use DGESVD in LAPACKE
     * on PHP script. it is the implement temporarily.
     *
     */
    public function gesvd(
        int $matrix_layout,
        int $jobu,
        int $jobvt,
        int $m,
        int $n,
        Buffer $A,  int $offsetA,  int $ldA,
        Buffer $S,  int $offsetS,
        Buffer $U,  int $offsetU,  int $ldU,
        Buffer $VT, int $offsetVT, int $ldVT,
        Buffer $SuperB,  int $offsetSuperB
    ) : void
    {
        if($this->useLapack($A)) {
            $this->lapack->gesvd(
                $matrix_layout,
                $jobu,
                $jobvt,
                $m,
                $n,
                $A,  $offsetA,  $ldA,
                $S,  $offsetS,
                $U,  $offsetU,  $ldU,
                $VT, $offsetVT, $ldVT,
                $SuperB,  $offsetSuperB
            );
            return;
        }
        if(method_exists($A,'dtype')) {
            $dtype = $A->dtype();
        } else {
            $dtype = $this->defaultFloatType;
        }

        $transposed = false;
        $A = new NDArrayPhp($A,$dtype,[$m,$n],$offsetA);
        $offsetV = $V = null;
        //if($m<$n) {
        //    throw new InvalidArgumentException('unsuppoted shape by PhpLapack. M must be greater than N or equal.');
        //}
        if($m<$n) {
         $transposed = true;
         $A = $this->transpose($A);
         [$m,$n] = [$n,$m];
         [$ldU,$ldVT] = [$ldVT,$ldU];
         [$U,$V] = [$V,$U];
         [$offsetU,$offsetV] = [$offsetV,$offsetU];
        }

        $U = new NDArrayPhp($U,$dtype,[$m,$ldU],$offsetU);
        $V = new NDArrayPhp($V,$dtype,[$n,$ldVT],$offsetV);

        $min_num = min($m,$n);
        for($i=0;$i<$m;$i++) {
            $this->copy($A[$i][[0,(int)($min_num-1)]],
                $U[$i][[0,(int)($min_num-1)]]);
        }
        for($i=0;$i<$min_num;$i++) {
            $this->copy($A[$i][[0,(int)($n-1)]],
                $V[$i][[0,(int)($n-1)]]);
        }

        #echo "--- a ---\n";
        #$this->print($A);
        #echo "--- u ---\n";
        #$this->print($U);
        #echo "--- v ---\n";
        #$this->print($V);

        $eps = 2.22045e-016;

        // Decompose Phase

        // Householder reduction to bidiagonal form.
        $g = $scale = $anorm = 0.0;
        for($i = 0; $i < $ldU; $i++){
            $l = $i + 2;
            $rv1[$i] = $scale * $g;
            $g = $s = $scale = 0.0;
            if($i < $m){
                for($k = $i; $k < $m; $k++)
                    $scale += abs($U[$k][$i]);
                if($scale != 0.0) {
                    for($k = $i; $k < $m; $k++) {
                        $U[$k][$i] = $U[$k][$i] / $scale;
                        $s += $U[$k][$i] * $U[$k][$i];
                    }
                    $f = $U[$i][$i];
                    $g = - $this->sameSign(sqrt($s), $f);
                    $h = $f * $g - $s;
                    $U[$i][$i] = $f - $g;
                    for($j = $l - 1; $j < $ldU; $j++){
                        for($s = 0.0, $k = $i; $k < $m; $k++)
                            $s += $U[$k][$i] * $U[$k][$j];
                        $f = $s / $h;
                        for($k = $i; $k < $m; $k++)
                            $U[$k][$j] = $U[$k][$j] + $f * $U[$k][$i];
                    }
                    for($k = $i; $k < $m; $k++)
                        $U[$k][$i] = $U[$k][$i] * $scale;
                }
            }
            $W[$i] = $scale * $g;
            $g = $s = $scale = 0.0;
            if($i + 1 <= $m && $i + 1 != $ldU){
                for ($k= $l - 1; $k < $ldU; $k++)
                    $scale += abs($U[$i][$k]);
                if($scale != 0.0){
                    for ($k= $l - 1; $k < $ldU; $k++){
                        $U[$i][$k] = $U[$i][$k] / $scale;
                        $s += $U[$i][$k] * $U[$i][$k];
                    }
                    $f = $U[$i][$l - 1];
                    $g = - $this->sameSign(sqrt($s), $f);
                    $h = $f * $g - $s;
                    $U[$i][$l - 1] = $f - $g;
                    for($k = $l - 1; $k < $ldU; $k++)
                        $rv1[$k] = $U[$i][$k] / $h;
                    for($j = $l - 1; $j < $m; $j++){
                        for($s = 0.0, $k = $l - 1; $k < $ldU; $k++)
                            $s += $U[$j][$k] * $U[$i][$k];
                        for($k = $l - 1; $k < $ldU; $k++)
                            $U[$j][$k] = $U[$j][$k] + $s * $rv1[$k];
                    }
                    for($k= $l - 1; $k < $ldU; $k++) $U[$i][$k] *= $scale;
                }
            }
            $anorm = max($anorm, (abs($W[$i]) + abs($rv1[$i])));
        }

        // Accumulation of right-hand transformations.
        for($i = $n - 1; $i >= 0; $i--){
            if($i < $n - 1){
                if($g != 0.0){
                    for($j = $l; $j < $n; $j++) // Double division to avoid possible underflow.
                    $V[$j][$i] = ($U[$i][$j] / $U[$i][$l]) / $g;
                    for($j = $l; $j < $n; $j++){
                        for($s = 0.0, $k = $l; $k < $n; $k++)
                            $s += ($U[$i][$k] * $V[$k][$j]);
                        for($k = $l; $k < $n; $k++)
                            $V[$k][$j] = $V[$k][$j] + $s * $V[$k][$i];
                    }
                }
                for($j = $l; $j < $n; $j++)
                    $V[$i][$j] = $V[$j][$i] = 0.0;
            }
            $V[$i][$i] = 1.0;
            $g = $rv1[$i];
            $l = $i;
        }

        // Accumulation of left-hand transformations.
        for($i = min($m, $ldU) - 1; $i >= 0; $i--){
            $l = $i + 1;
            $g = $W[$i];
            for($j = $l; $j < $ldU; $j++)
                $U[$i][$j] = 0.0;
            if($g != 0.0){
                $g = 1.0 / $g;
                for($j = $l; $j < $ldU; $j++){
                    for($s = 0.0, $k = $l; $k < $m; $k++)
                        $s += $U[$k][$i] * $U[$k][$j];
                    $f = ($s / $U[$i][$i]) * $g;
                    for($k = $i; $k < $m; $k++)
                        $U[$k][$j] = $U[$k][$j] + $f * $U[$k][$i];
                }
                for($j = $i; $j < $m; $j++)
                    $U[$j][$i] = $U[$j][$i] * $g;
            }else {
                for($j = $i; $j < $m; $j++)
                    $U[$j][$i] = 0.0;
            }
            $U[$i][$i] = $U[$i][$i]+1;
        }

        // Diagonalization of the bidiagonal form
        // Loop over singular values, and over allowed iterations.
        for($k = $n - 1; $k >= 0; $k--){
            for($its = 0; $its < 30; $its++){
                $flag = true;
                for($l = $k; $l >= 0; $l--){
                    $nm = $l - 1;
                    if( $l == 0 || abs($rv1[$l]) <= $eps*$anorm){
                        $flag = false;
                        break;
                    }
                    if(abs($W[$nm]) <= $eps*$anorm) break;
                }
                if($flag){
                    $c = 0.0;  // Cancellation of rv1[l], if l > 0.
                    $s = 1.0;
                    for($i = $l; $i < $k + 1; $i++){
                        $f = $s * $rv1[$i];
                        $rv1[$i] = $c * $rv1[$i];
                        if(abs($f) <= $eps*$anorm) break;
                        $g = $W[$i];
                        $h = $this->pythag($f,$g);
                        $W[$i] = $h;
                        $h = 1.0 / $h;
                        $c = $g * $h;
                        $s = -$f * $h;
                        for($j = 0; $j < $m; $j++){
                            $y = $U[$j][$nm];
                            $z = $U[$j][$i];
                            $U[$j][$nm] = $y * $c + $z * $s;
                            $U[$j][$i] = $z * $c - $y * $s;
                        }
                    }
                }
                $z = $W[$k];
                if($l == $k){
                    if($z < 0.0){
                        $W[$k] = -$z; // Singular value is made nonnegative.
                        for($j = 0; $j < $n; $j++)
                            $V[$j][$k] = -$V[$j][$k];
                    }
                    break;
                }
                if($its == 29) print("no convergence in 30 svd iterations");
                $x = $W[$l]; // Shift from bottom 2-by-2 minor.
                $nm = $k - 1;
                $y = $W[$nm];
                $g = $rv1[$nm];
                $h = $rv1[$k];
                $f = (($y - $z) * ($y + $z) + ($g - $h) * ($g + $h)) / (2.0 * $h * $y);
                $g = $this->pythag($f,1.0);
                $f = (($x - $z) * ($x + $z) + $h * (($y / ($f + $this->sameSign($g,$f))) - $h)) / $x;
                $c = $s = 1.0;
                for($j = $l; $j <= $nm; $j++){
                    $i = $j + 1;
                    $g = $rv1[$i];
                    $y = $W[$i];
                    $h = $s * $g;
                    $g = $c * $g;
                    $z = $this->pythag($f,$h);
                    $rv1[$j] = $z;
                    $c = $f / $z;
                    $s = $h / $z;
                    $f = $x * $c + $g * $s;
                    $g = $g * $c - $x * $s;
                    $h = $y * $s;
                    $y *= $c;
                    for($jj = 0; $jj < $n; $jj++){
                        $x = $V[$jj][$j];
                        $z = $V[$jj][$i];
                        $V[$jj][$j] = $x * $c + $z * $s;
                        $V[$jj][$i] = $z * $c - $x * $s;
                    }
                    $z = $this->pythag($f,$h);
                    $W[$j] = $z;  // Rotation can be arbitrary if z = 0.
                    if($z){
                        $z = 1.0 / $z;
                        $c = $f * $z;
                        $s = $h * $z;
                    }
                    $f = $c * $g + $s * $y;
                    $x = $c * $y - $s * $g;
                    for($jj = 0; $jj < $m; $jj++){
                        $y = $U[$jj][$j];
                        $z = $U[$jj][$i];
                        $U[$jj][$j] = $y * $c + $z * $s;
                        $U[$jj][$i] = $z * $c - $y * $s;
                    }
                }
                $rv1[$l] = 0.0;
                $rv1[$k] = $f;
                $W[$k] = $x;
            }
        }

        // Reorder Phase
        // Sort. The method is Shell's sort.
        // (The work is negligible as compared to that already done in decompose phase.)
        $inc = 1;
        do {
            $inc = (int)($inc * 3);
            $inc++;
        }   while($inc <= $n);

        do {
            $inc = (int)($inc / 3);
            for($i = $inc; $i < $n; $i++){
                $sw = $W[$i];
                for($k = 0; $k < $m; $k++) $su[$k] = $U[$k][$i];
                for($k = 0; $k < $n; $k++) $sv[$k] = $V[$k][$i];
                $j = $i;
                while($W[$j - $inc] < $sw){
                    $W[$j] = $W[$j - $inc];
                    for($k = 0; $k < $m; $k++) $U[$k][$j] = $U[$k][$j - $inc];
                    for($k = 0; $k < $n; $k++) $V[$k][$j] = $V[$k][$j - $inc];
                    $j -= $inc;
                    if($j < $inc) break;
                }
                $W[$j] = $sw;
                for($k = 0; $k < $m; $k++) $U[$k][$j] = $su[$k];
                for($k = 0; $k < $n; $k++) $V[$k][$j] = $sv[$k];
            }
        }  while($inc > 1);

        #for($k = 0; $k < $n; $k++){
        #    $s = 0;
        #    for($i = 0; $i < $m; $i++) if ($U[$i][$k] < 0.0) $s++;
        #    for($j = 0; $j < $n; $j++) if ($V[$j][$k] < 0.0) $s++;
        #    if($s > ($m + $n)/2) {
        #        for($i = 0; $i < $m; $i++) $U[$i][$k] = - $U[$i][$k];
        #        for($j = 0; $j < $n; $j++) $V[$j][$k] = - $V[$j][$k];
        #    }
        #}

        // calculate the rank
        $rank = 0;
        for($i = 0; $i < count($W); $i++){
            if(round($W[$i], 4) > 0){
                $rank += 1;
            }
        }

        // Low-Rank Approximation
        $q = 0.9;
        $k = 0;
        $frobA = $frobAk = 0;
        for($i = 0; $i < $rank; $i++) $frobA += $W[$i];
        do{
            for($i = 0; $i <= $k; $i++) $frobAk += $W[$i];
            $clt = $frobAk / $frobA;
            $k++;
        }   while($clt < $q);

        // prepare S matrix as n*n daigonal matrix of singular values
        for($i = 0; $i < $n; $i++){
            $S[$i] = $W[$i];
        }

        if($transposed) {
            $UT = $this->transpose($V);
            $this->copy($UT,$V);
            $VT = new NDArrayPhp($VT,$dtype,[$ldU,$m],$offsetVT);
            $this->copy($U,$VT);
        } else {
            $VT = new NDArrayPhp($VT,$dtype,[$ldVT,$n],$offsetVT);
            for($i=0;$i<$ldVT;$i++) {
                for($j=0;$j<$n;$j++) {
                    $VT[$i][$j] = $V[$j][$i];
                }
            }
        }
        //$matrices['U'] = $U;
        //$matrices['S'] = $S;
        //$matrices['W'] = $W;
        //$matrices['V'] = $this->matrixTranspose($V);
        //$matrices['Rank'] = $rank;
        //$matrices['K'] = $k;

        #return [$U,$S,$this->transpose($V)];
    }

    private function print($a)
    {
        foreach($a->toArray() as $array)
            echo '['.implode(',',array_map(function($a){return sprintf('%5.2f',$a);},$array))."],\n";
    }

    private function alloc(array $shape, $dtype)
    {
        return new NDArrayPhp(null,$dtype,$shape);
    }

    private function zeros(NDArray $X)
    {
        $N = $X->size();
        $XX = $X->buffer();
        $offX = $X->offset();
        $posX = $offX;
        for($i=0;$i<$N;$i++) {
            $XX[$posX] = 0;
            $posX++;
        }
    }

    private function copy(NDArray $X, NDArray $Y)
    {
        if($X->shape()!=$Y->shape()) {
            throw new InvalidArgumentException('unmatch shape:');
        }
        $N = $X->size();
        $XX = $X->buffer();
        $offX = $X->offset();
        $YY = $Y->buffer();
        $offY = $Y->offset();
        $posX = $offX;
        $posY = $offY;
        for($i=0;$i<$N;$i++) {
            $YY[$posY] = $XX[$posX];
            $posX++;
            $posY++;
        }
    }

    private function sameSign($a, $b)
    {
        if($b >= 0){
            $result = abs($a);
        } else {
            $result = - abs($a);
        }
        return $result;
    }

    private function pythag($a, $b)
    {
        $absa = abs($a);
        $absb = abs($b);

        if( $absa > $absb ){
            return $absa * sqrt( 1.0 + pow( $absb / $absa , 2) );
        } else {
            if( $absb > 0.0 ){
                return $absb * sqrt( 1.0 + pow( $absa / $absb, 2 ) );
            } else {
                return 0.0;
            }
        }
    }

    /**
    *   copied from MatrixOperator
    */
    public function transpose(NDArray $X) : NDArray
    {
        $shape = $X->shape();
        $newShape = array_reverse($shape);
        $Y = $this->alloc($newShape, $X->dtype());
        $w = 1;
        $posY = 0;
        $posX = 0;
        $this->_transpose($newShape, $w, $X->buffer(), $X->offset(), $posX, $Y->buffer(), $posY);
        return $Y;
    }

    protected function _transpose($shape, $w, $bufX, $offX, $posX, $bufY, &$posY)
    {
        $n=array_shift($shape);
        $W = $w*$n;
        $deps = count($shape);
        for($i=0;$i<$n;$i++) {
            if($deps) {
                $this->_transpose($shape, $W, $bufX, $offX, $posX+$w*$i, $bufY, $posY);
            } else {
                $bufY[$posY] = $bufX[$offX + $posX+$w*$i];
                $posY++;
            }
        }
    }
}
