<?php
namespace Rindow\Math\Matrix;

use Interop\Polite\Math\Matrix\NDArray;
use Interop\Polite\Math\Matrix\OpenCL;
use Rindow\Math\Plot\Plot;
use InvalidArgumentException;

class OpenCLMathTunner
{
    protected $mo;
    protected $la;
    protected $maxWorkItem;

    public function __construct($mo)
    {
        $la = $mo->laAccelerated('clblast');
        $la->blocking(true);
        $la->scalarNumeric(true);
        $this->mo = $mo;
        $this->la = $la;
        $this->maxWorkItem = $la->getOpenCLMath()->maxWorkItem()[0];
    }

    public function tunningScatterAdd($mode,$maxTime,$limitTime)
    {
        $la = $this;
        $rowMax = 1048576;
        $colMax = 1048576;
        $classMax = 1048576;
        $times = $this->loadParameter('TempTimesMode'.$mode.'.php');
        //$times['pointer'] = [32768,16,8];
        //$times['prevTime'] = 0;//$times[32768][16][256];
        //$this->saveParameter('TempTimesMode'.$mode.'.php',$times);
        //return;
        if($times) {
            [$startI,$startJ,$startK] = $times['pointer'];
            $prevTime = $times['prevTime'];
        } else {
            [$startI,$startJ,$startK] = [8,8,8];
            $times = [];
            $prevTime = 0;
        }
        for($i=$startI;$i<=$rowMax;$i<<=1) {
            $startI=8;
            $times[$i] = [];
            fwrite(STDERR, "rows=$i\n");
            $timeover=false;
            $prevEndK = 0;
            for($j=$startJ;$j<=$colMax;$j<<=1) {
                $startJ=8;
                $times[$i][$j] = [];
                fwrite(STDERR, "rows=$i,cols=$j\n");
                $timeover=false;
                $predictTimeover=false;
                for($k=$startK;$k<=$classMax;$k<<=1) {
                    $startK=8;
                    fwrite(STDERR, $k);
                    if($mode==4&&$k!=8) {
                        $times[$i][$j][$k] = $time;
                        fwrite(STDERR, ",");
                        continue;
                    }
                    //if($k==8||$k==16) {
                    //    fwrite(STDERR,"<i=$i,j=$j>");
                    //    fwrite(STDERR,"<".(isset($times[$i][$j>>1][$k])?'true':'false').">");
                    //    fwrite(STDERR,"<".(isset($times[$i][$j>>2][$k])?'true':'false').">");
                    //    var_dump($times[$i>>1]);
                    //    var_dump($times[$i]);
                    //    var_dump($times[$i][$j>>2]);
                    //}
                    if(($mode==4 && $i==524288 && $j>=32 && $k==8)||
                        ($mode==4 && $i==1048576 && $j>=8 && $k==8)) {
                        fwrite(STDERR,"X");
                        $times['prevTime'] = $prevTime;
                        $times['pointer'] = [$i,$j,$k];
                        $this->saveParameter('TempTimesMode'.$mode.'.php',$times);
                        break;
                    }
                    if(($k==8||$k==16)&&isset($times[$i][$j>>1][$k])&&isset($times[$i][$j>>2][$k])&&
                        $times[$i][$j>>1][$k]>0&&$times[$i][$j>>2][$k]>0) {
                        $point = $j>>1;
                        $prevPoint = $j>>2;
                        $nextPoint = $j;
                        $timep = $times[$i][$j>>1][$k];
                        $prevTimep = $times[$i][$j>>2][$k];
                        $predictTime = ($timep-$prevTimep)/($point-$prevPoint)*($nextPoint-$point)+$timep;
                        //fwrite(STDERR,"<".sprintf("%3.2f",log10($predictTime)).">");
                        if($predictTime > $limitTime) {
                            $timeover=true;
                            $predictTimeover=true;
                            fwrite(STDERR,"p");
                            fwrite(STDERR,"(".sprintf("%3.2f",log10($predictTime)).")");
                            $times['prevTime'] = $prevTime;
                            $times['pointer'] = [$i,$j,$k];
                            $this->saveParameter('TempTimesMode'.$mode.'.php',$times);
                            if($k!=8) {
                                $prevTime = 0;
                            }
                            break;
                        }
                    }
                    try {
                        $times[$i][$j][$k] = $this->timeScatterAdd(
                            $la,$try=10,$mode,$i,$j,$k);
                    } catch(\Exception $e) {
                        fwrite(STDERR, $e->getMessage());
                        $times[$i][$j][$k] = 0;
                    }
                    $time = $times[$i][$j][$k];
                    //fwrite(STDERR, '('.sprintf("%3.2f",log10($time)).')');
                    fwrite(STDERR, ",");
                    //if($i==1048576&&$j==8&&$k==8) {
                    //    break;
                    //}
                    if($time>$maxTime) {
                        $timeover=true;
                        fwrite(STDERR,"T");
                        fwrite(STDERR,"(".sprintf("%3.2f",log10($time)).")");
                    }
                    if($k!=8&&$prevTime!=0) {
                        $point = $k;
                        $prevPoint = $k>>1;
                        $nextPoint = $k<<1;
                        $predictTime = ($time-$prevTime)/($point-$prevPoint)*($nextPoint-$point)+$time;
                        if($predictTime > $limitTime) {
                            $timeover=true;
                            $predictTimeover=true;
                            fwrite(STDERR,"P");
                            fwrite(STDERR,"(".sprintf("%3.2f",log10($predictTime)).")");
                        }
                    } if($k==8&&$prevEndK==16&&$predictTimeover) {
                        $timeover=true;
                        fwrite(STDERR,"(p)");
                    }
                    $prevTime = $time;
                    $prevEndK = $k;
                    if($timeover) {
                        $times['prevTime'] = $prevTime;
                        $times['pointer'] = [$i,$j,$k];
                        $this->saveParameter('TempTimesMode'.$mode.'.php',$times);
                        if($k!=8) {
                            $prevTime = 0;
                        }
                        break;
                    }
                }
                //if($i==1048576&&$j==8&&$k==8) {
                //    break;
                //}
                //if($k==8&&$j!=8&&$prevTime!=0) {
                //    $point = $j;
                //    $prevPoint = $j>>1;
                //    $nextPoint = $j<<1;
                //    if(($time-$prevTime)/($point-$prevPoint)*($nextPoint-$point)+$time > 10**10) {
                //        $timeover=true;
                //    }
                //}
                if($timeover&&$k==8) {
                    if($j!=8) {
                        $prevTime = 0;
                    }
                    break;
                }
                $timeover=false;
                fwrite(STDERR, "\n");
            }
            //if($i==1048576&&$j==8&&$k==8) {
            //    break;
            //}
            //if($j==8&&$i!=8&&$prevTime!=0) {
            //    $point = $i;
            //    $prevPoint = $i>>1;
            //    $nextPoint = $i<<1;
            //    if(($time-$prevTime)/($point-$prevPoint)*($nextPoint-$point)+$time > 10**10) {
            //        $timeover=true;
            //    }
            //}
            if($timeover&&$j==8) {
                if($i!=8) {
                    $prevTime = 0;
                }
                break;
            }
        }
        unset($times['pointer']);
        unset($times['prevTime']);
        $this->saveParameter('ScatterAddTimesMode'.$mode.'.php',$times);
        $this->deleteParameter('TempTimesMode'.$mode.'.php');
    }

    protected function timeScatterAdd($la,$try,$mode,$rows,$cols,$numClass)
    {
        switch($mode) {
            case 0:{
                return $this->timeSimulation($rows,$cols,$numClass);
                break;
            }
            case 1:{
                if($rows>$this->maxWorkItem) {// || $rows*$cols*$numClass>256*256*256*4) {
                    fwrite(STDERR,"x");
                    return 0;
                }
                break;
            }
            case 2:{
                //if($rows*$cols*$numClass>256*256*256*4) {
                //    fwrite(STDERR,"x");
                //    return 0;
                //}
                break;
            }
            case 3:{
                //if($rows*$cols*$numClass>256*256*64) { #*128
                //    fwrite(STDERR,"x");
                //    return 0;
                //}
                break;
            }
            case 4:{
                //if($cols*$rows>256*256*128) { #*128
                //    fwrite(STDERR,"x");
                //    return 0;
                //}
                break;
            }
            default:
                return 0;
        }

        $x = $this->la->alloc([$rows],NDArray::int32);
        $y = $this->la->alloc([$rows,$cols],NDArray::float32);
        $a = $this->la->alloc([$numClass,$cols],NDArray::float32);

        $this->la->fill(1,$x);
        $this->la->fill(1.0,$y);
        $this->la->fill(0.0,$a);
        $this->scatterAddTest($x,$y,$a,$axis=0,null,null,$mode);
        $time = 0;
        for($i=0;$i<$try;$i++) {
            $start = hrtime(true);
            $this->scatterAddTest($x,$y,$a,$axis=0,null,null,$mode);
            $end = hrtime(true);
            $time += $end-$start;
        }
        return $time/$try;
    }

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
            $waitEvents = $this->la->newEventList();
            $X = $this->la->astype($X,NDArray::int32,null,$waitEvents,$waitPrev);
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
            $A = $this->la->alloc($shape,$Y->dtype());
            $waitPrev = $waitEvents;
            $waitEvents = $this->la->newEventList();
            $this->la->zeros($A,$waitEvents,$waitPrev);
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

        //fwrite(STDERR,"(m=$m,n=$n,k=$countX)");
        switch($mode) {
            case 1: {
                $this->la->getOpenCLMath()->scatterAddAxis0_1(
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
                $this->la->getOpenCLMath()->scatterAddAxis0_2(
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
                $this->la->getOpenCLMath()->scatterAddAxis0_3(
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
                $this->la->getOpenCLMath()->scatterAddAxis0_4(
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

        $this->la->finish();
        return $A;
    }

    protected function timeSimulation($rows,$cols,$numClass)
    {
        $a[0] = 10; $b[0] = 10;
        $a[1] = 10; $b[1] = 10;
        $a[2] = 10; $b[2] = 10;
        $c = 10000;
        return ($a[0]*$rows+$b[0])*($a[1]*$cols+$b[1])*($a[2]*$numClass+$b[2])+$c;
    }

    public function showGraphScatterAdd($mode=null,$details=null)
    {
        $marker = null;
        $colors = [1=>'b',2=>'g',3=>'r',4=>'m'];
        $mo = $this->mo;
        $plt = new Plot(null,$mo);
        if($mode===null) {
            $modes = range(1,4);
        } else {
            $modes = [$mode];
        }
        if($details) {
            $marker = null;
            foreach($modes as $mode) {
                $plt->figure();
                $axes = $plt->getAxes();
                $this->drawGraphRowsCols($mo,$mode,$axes,$details,$marker);
                $plt->figure();
                $axes = $plt->getAxes();
                $this->drawGraphColsRows($mo,$mode,$axes,$details,$marker);
                $plt->figure();
                $axes = $plt->getAxes();
                $this->drawGraphNumClassRows($mo,$mode,$axes,$details,$marker);
                $plt->figure();
                $axes = $plt->getAxes();
                $this->drawGraphRowsNumClass($mo,$mode,$axes,$details,$marker);
                $plt->figure();
                $axes = $plt->getAxes();
                $this->drawGraphNumClassCols($mo,$mode,$axes,$details,$marker);
                $plt->figure();
                $axes = $plt->getAxes();
                $this->drawGraphColsNumClass($mo,$mode,$axes,$details,$marker);
            }
        } else {
            [$fig,$axes] = $plt->subplots(3,2);
            foreach($modes as $mode) {
                if(count($modes)==1) {
                    $marker = null;
                } else {
                    $marker = $colors[$mode];
                }
                $this->drawGraphRowsCols($mo,$mode,$axes[0],$details,$marker);
                $this->drawGraphColsRows($mo,$mode,$axes[1],$details,$marker);
                $this->drawGraphNumClassRows($mo,$mode,$axes[2],$details,$marker);
                $this->drawGraphRowsNumClass($mo,$mode,$axes[3],$details,$marker);
                $this->drawGraphNumClassCols($mo,$mode,$axes[4],$details,$marker);
                $this->drawGraphColsNumClass($mo,$mode,$axes[5],$details,$marker);
            }
        }
        $plt->show();
    }

    protected function drawGraphRowsCols($mo,$mode,$ax,$details,$marker)
    {
        $times = $this->loadParameter('ScatterAddTimesMode'.$mode.'.php');
        // rows <-> cols
        $graph = [];
        foreach ($times as $rows => $colsData) {
            foreach ($colsData as $cols => $numClassData) {
                foreach ($numClassData as $numClass => $value) {
                    if(!isset($graph[$numClass])) {
                        $graph[$numClass] = [];
                    }
                    if(!isset($graph[$numClass][$cols])) {
                        $graph[$numClass][$cols] = [];
                    }
                    if($value>0) {
                        $graph[$numClass][$cols][] = [$rows,$value];
                    }
                }
            }
        }
        $numClass = 8;
        foreach ($graph[$numClass] as $cols => $gr) {
            if(count($gr)) {
                $gr = $mo->transpose($mo->array($gr));
                $ax->plot($gr[0],$gr[1],$marker,'cols='.$cols);
            }
        }
        $ax->setYScale('log');
        $ax->setXScale('log');
        if($details) {
            $ax->legend();
            $ax->setXLabel('rows');
        }
    }

    protected function drawGraphColsRows($mo,$mode,$ax,$details,$marker)
    {
        $times = $this->loadParameter('ScatterAddTimesMode'.$mode.'.php');
        // cols <-> rows
        foreach ($times as $rows => $colsData) {
            $graph = [];
            foreach ($colsData as $cols => $numClassData) {
                foreach ($numClassData as $numClass => $value) {
                    if(!isset($graph[$numClass])) {
                        $graph[$numClass] = [];
                    }
                    if($value>0) {
                        $graph[$numClass][] = [$cols,$value];
                    }
                }
            }
            $numClass = 8;
            if(isset($graph[$numClass]) && count($graph[$numClass])) {
                $gr = $mo->transpose($mo->array($graph[$numClass]));
                $ax->plot($gr[0],$gr[1],$marker,'rows='.$rows);
            }
        }
        $ax->setYScale('log');
        $ax->setXScale('log');
        if($details) {
            $ax->legend();
            $ax->setXLabel('cols');
        }
    }

    protected function drawGraphNumClassRows($mo,$mode,$ax,$details,$marker)
    {
        $times = $this->loadParameter('ScatterAddTimesMode'.$mode.'.php');
        // numClass <-> rows
        $graph = [];
        foreach ($times as $rows => $colsData) {
            foreach ($colsData as $cols => $numClassData) {
                foreach ($numClassData as $numClass => $value) {
                    if(!isset($graph[$cols])) {
                        $graph[$cols] = [];
                    }
                    if(!isset($graph[$cols][$rows])) {
                        $graph[$cols][$rows] = [];
                    }
                    if($value>0) {
                        $graph[$cols][$rows][] = [$numClass,$value];
                    }
                }
            }
        }
        $cols = 8;
        foreach ($graph[$cols] as $rows => $gr) {
            if(count($gr)) {
                $gr = $mo->transpose($mo->array($gr));
                $ax->plot($gr[0],$gr[1],$marker,'rows='.$rows);
            }
        }
        $ax->setYScale('log');
        $ax->setXScale('log');
        if($details) {
            $ax->legend();
            $ax->setXLabel('numClass');
        }
    }

    protected function drawGraphRowsNumClass($mo,$mode,$ax,$details,$marker)
    {
        $times = $this->loadParameter('ScatterAddTimesMode'.$mode.'.php');
        // rows <-> numClass
        $graph = [];
        foreach ($times as $rows => $colsData) {
            foreach ($colsData as $cols => $numClassData) {
                foreach ($numClassData as $numClass => $value) {
                    if(!isset($graph[$cols])) {
                        $graph[$cols] = [];
                    }
                    if(!isset($graph[$cols][$numClass])) {
                        $graph[$cols][$numClass] = [];
                    }
                    if($value>0) {
                        $graph[$cols][$numClass][] = [$rows,$value];
                    }
                }
            }
        }
        $cols = 8;
        foreach ($graph[$cols] as $numClass => $gr) {
            if(count($gr)) {
                $gr = $mo->transpose($mo->array($gr));
                $ax->plot($gr[0],$gr[1],$marker,'numClass='.$numClass);
            }
        }
        $ax->setYScale('log');
        $ax->setXScale('log');
        if($details) {
            $ax->legend();
            $ax->setXLabel('rows');
        }
    }

    protected function drawGraphNumClassCols($mo,$mode,$ax,$details,$marker)
    {
        $times = $this->loadParameter('ScatterAddTimesMode'.$mode.'.php');
        // numClass <-> cols
        $graph = [];
        foreach ($times as $rows => $colsData) {
            foreach ($colsData as $cols => $numClassData) {
                foreach ($numClassData as $numClass => $value) {
                    if(!isset($graph[$rows])) {
                        $graph[$rows] = [];
                    }
                    if(!isset($graph[$rows][$cols])) {
                        $graph[$rows][$cols] = [];
                    }
                    if($value>0) {
                        $graph[$rows][$cols][] = [$numClass,$value];
                    }
                }
            }
        }
        $rows = 8;
        foreach ($graph[$rows] as $cols => $gr) {
            if(count($gr)) {
                $gr = $mo->transpose($mo->array($gr));
                $ax->plot($gr[0],$gr[1],$marker,'cols='.$cols);
            }
        }
        $ax->setYScale('log');
        $ax->setXScale('log');
        if($details) {
            $ax->legend();
            $ax->setXLabel('numClass');
        }
    }

    protected function drawGraphColsNumClass($mo,$mode,$ax,$details,$marker)
    {
        $times = $this->loadParameter('ScatterAddTimesMode'.$mode.'.php');
        // cols <-> numClass
        $graph = [];
        foreach ($times as $rows => $colsData) {
            foreach ($colsData as $cols => $numClassData) {
                foreach ($numClassData as $numClass => $value) {
                    if(!isset($graph[$rows])) {
                        $graph[$rows] = [];
                    }
                    if(!isset($graph[$rows][$numClass])) {
                        $graph[$rows][$numClass] = [];
                    }
                    if($value>0) {
                        $graph[$rows][$numClass][] = [$cols,$value];
                    }
                }
            }
        }
        $rows = 8;
        foreach ($graph[$rows] as $numClass => $gr) {
            if(count($gr)) {
                $gr = $mo->transpose($mo->array($gr));
                $ax->plot($gr[0],$gr[1],$marker,'numClass='.$numClass);
            }
        }
        $ax->setYScale('log');
        $ax->setXScale('log');
        if($details) {
            $ax->legend();
            $ax->setXLabel('cols');
        }
    }

    protected function getHomeDirectory()
    {
        if(PHP_OS=='WINNT') {
            return getenv('USERPROFILE');
        } elseif(PHP_OS=='Linux') {
            return getenv('HOME');
        }
    }

    protected function saveParameter($filename,$value)
    {
        $dir = $this->getHomeDirectory().'/.rindow';
        if(!file_exists($dir)) {
            mkdir($dir,0755,true);
        }
        $code = "<?php\nreturn unserialize('".str_replace(array('\\','\''), array('\\\\','\\\''), serialize($value))."');";
        file_put_contents($dir.'/'.$filename,$code);
    }

    protected function loadParameter($filename)
    {
        $filepath = $this->getHomeDirectory().'/.rindow/'.$filename;
        if(!file_exists($filepath)) {
            $filepath = __DIR__.'/params/'.$filename;
        }
        if(!file_exists($filepath)) {
            return null;
        }
        $times = include $filepath;
        return $times;
    }

    protected function deleteParameter($filename)
    {
        $filepath = $this->getHomeDirectory().'/.rindow/'.$filename;
        if(file_exists($filepath)) {
            unlink($filepath);
        }
    }

    public function getScatterAddParameter($mode,$rows,$cols,$numClass)
    {
        $times = $this->loadParameter('ScatterAddTimesMode'.$mode.'.php');
        if(isset($times[$rows][$cols][$numClass])) {
            return $times[$rows][$cols][$numClass];
        } else {
            return 0;
        }
    }

    public function setScatterAddParameter($mode,$rows,$cols,$numClass,$value,$force=null)
    {
        $times = $this->loadParameter('ScatterAddTimesMode'.$mode.'.php');
        if(isset($times[$rows][$cols][$numClass])) {
            $times[$rows][$cols][$numClass] = $value;
        } else {
            if($force) {
                $times[$rows][$cols][$numClass] = $value;
            } else {
                echo 'OutOfRange!';
                return;
            }
        }
        $this->saveParameter('ScatterAddTimesMode'.$mode.'.php',$times);
    }
}
