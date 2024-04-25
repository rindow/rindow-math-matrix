<?php
namespace Rindow\Math\Matrix\Drivers\MatlibCL;

use Interop\Polite\Math\Matrix\NDArray;
use Interop\Polite\Math\Matrix\OpenCL;
use Rindow\Math\Plot\Plot;
use InvalidArgumentException;

class OpenCLMathTunner
{
    protected object $mo;
    protected object $la;
    protected int $maxWorkItem;

    public function __construct(object $mo)
    {
        $la = $mo->laAccelerated('clblast');
        $la->blocking(true);
        $la->scalarNumeric(true);
        $this->mo = $mo;
        $this->la = $la;
        $this->maxWorkItem = $la->getOpenCLMath()->maxWorkItem()[0];
    }

    public function tunningScatterAdd(int $mode,int $maxTime,int $limitTime) : void
    {
        $la = $this;
        $rowMax = 1048576;
        $colMax = 1048576;
        $classMax = 1048576;
        $times = $this->loadParameter('TempTimesMode'.$mode.'.php');
        //$times['pointer'] = [131072,8,8];
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
            if(!array_key_exists($i,$times)) {
                $times[$i] = [];
            }
            fwrite(STDERR, "rows=$i\n");
            $timeover=false;
            $prevEndK = 0;
            for($j=$startJ;$j<=$colMax;$j<<=1) {
                $startJ=8;
                if(!array_key_exists($j,$times[$i])) {
                    $times[$i][$j] = [];
                }
                fwrite(STDERR, "rows=$i,cols=$j\n");
                $timeover=false;
                $predictTimeover=false;
                for($k=$startK;$k<=$classMax;$k<<=1) {
                    $startK=8;
                    fwrite(STDERR, $k);
                    if($mode==4&&$k!=8) {
                        $times[$i][$j][$k] = 0; // $time
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
                        ($mode==4 && $i==1048576 && $j>=8 && $k==8)||
                        ($mode==0 && $i==131072 && $j==256 && $k==128)||
                        ($mode==2 && $i==1048576 && $j==8 && $k==16)
                    ) {
                        fwrite(STDERR,"X");
                        $times['prevTime'] = $prevTime;
                        $times['pointer'] = [$i,$j*2,8];
                        $this->saveParameter('TempTimesMode'.$mode.'.php',$times);
                        break;
                    }
                    $limitTimeN = $limitTime;
                    if(($mode==0 && $i>=262144)) {
                        fwrite(STDERR,"L");
                        $limitTimeN = 10**9.6;
                    }
                    if(($mode==0 && $i>=524288 && $j>=32)) {
                        fwrite(STDERR,"L");
                        $limitTimeN = 10**9.4;
                    }
                    if(($mode==0 && $i>=524288 && $j>=256)) {
                        fwrite(STDERR,"L");
                        $limitTimeN = 10**9.3;
                    }
                    if(($mode==0 && $i>=1048576)) {
                        fwrite(STDERR,"L");
                        $limitTimeN = 10**9.3;
                    }
                    if(($k==8||$k==16)&&
                       isset($times[$i][$j>>1][$k])&&
                       isset($times[$i][$j>>2][$k])&&
                       $times[$i][$j>>1][$k]>0&&
                       $times[$i][$j>>2][$k]>0) {
                        fwrite(STDERR,"K");
                        $point = $j>>1;
                        $prevPoint = $j>>2;
                        $nextPoint = $j;
                        $timep = $times[$i][$j>>1][$k];
                        $prevTimep = $times[$i][$j>>2][$k];
                        $predictTime = ($timep-$prevTimep)/($point-$prevPoint)*($nextPoint-$point)+$timep;
                        //fwrite(STDERR,"<".sprintf("%3.2f",log10($predictTime)).">");
                        if($predictTime > $limitTimeN) {
                            $timeover=true;
                            $predictTimeover=true;
                            fwrite(STDERR,"p");
                            fwrite(STDERR,"(".sprintf("%3.2f",log10($predictTime)).")");
                            $times['prevTime'] = $prevTime;
                            $times['pointer'] = [$i,$j*2,8];
                            $this->saveParameter('TempTimesMode'.$mode.'.php',$times);
                            if($k!=8) {
                                $prevTime = 0;
                            }
                            break;
                        }
                    }
                    if(($j>16 && $k==8 && !isset($times[$i][$j>>1][$k]))||
                       ($j>16 && $k==16 && !isset($times[$i][$j>>1][$k]))
                       ) {
                           fwrite(STDERR,"x");
                           $times['prevTime'] = $prevTime;
                           $times['pointer'] = [$i,$j*2,8];
                           $this->saveParameter('TempTimesMode'.$mode.'.php',$times);
                           if($k!=8) {
                               $prevTime = 0;
                           }
                           $timeover=true;
                           break;
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
                        if($predictTime > $limitTimeN) {
                            $timeover=true;
                            $predictTimeover=true;
                            fwrite(STDERR,"P");
                            fwrite(STDERR,"(".sprintf("%3.2f",log10($predictTime)).")");
                        }
                    } elseif($k==8&&$prevEndK==16&&$predictTimeover) {
                        $timeover=true;
                        fwrite(STDERR,"(p)");
                    }
                    $prevTime = $time;
                    $prevEndK = $k;
                    if($timeover) {
                        $times['prevTime'] = $prevTime;
                        $times['pointer'] = [$i,$j*2,8];
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

    protected function timeScatterAdd(
        object $la,int $try,int $mode,int $rows,int $cols,int $numClass) : float
    {
        switch($mode) {
            case 0:{
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
        $a = $this->la->alloc([$rows,$cols],NDArray::float32);
        $b = $this->la->alloc([$numClass,$cols],NDArray::float32);

        $this->la->fill(1,$x);
        $this->la->fill(1.0,$a);
        $this->la->fill(0.0,$b);
        $this->scatterAddTest($x,$a,$b,$axis=0,null,null,$mode);
        $time = 0;
        for($i=0;$i<$try;$i++) {
            $start = hrtime(true);
            $this->scatterAddTest($x,$a,$b,$axis=0,null,null,$mode);
            $end = hrtime(true);
            $time += $end-$start;
        }
        return $time/$try;
    }

    public function scatterAddTest(
        NDArray $X,
        NDArray $A,
        NDArray $output,
        int $axis=null,
        object $events=null,object $waitEvents=null,
        int $mode = null
        ) : NDArray
    {
        if($X->dtype()!=NDArray::int32 && $X->dtype()!=NDArray::uint32) {
            $waitPrev = $waitEvents;
            $waitEvents = $this->la->newEventList();
            $X = $this->la->astype($X,NDArray::int32,null,$waitEvents,$waitPrev);
        }
        $dtype = $X->dtype();
        if($axis===null) {
            return $this->scatterAddAxisNullTest( // doGather()
                $scatterAdd=true,
                $output,
                $X,
                $axis,
                $A,
                $dtype,
                $events, $waitEvents,
                $mode,
            );
            //} elseif($axis==1) {
        //    return $this->scatterAddAxis1Test(true,$X,$A,null,$B,$events,$waitEvents,$mode);
        } else {
            throw new InvalidArgumentException('axis must be 0 or 1');
        }
    }

    protected function scatterAddAxisNullTest( // doGather()
        bool $scatterAdd,
        NDArray $A,
        NDArray $X,
        int $axis=null,
        NDArray $output=null,
        int $dtype=null,
        object $events=null,object $waitEvents=null,
        int $mode=null,
        ) : NDArray
    {
        // if($axis===null) {
            $postfixShape = $A->shape();
            $prefixShape = $X->shape();
            $numClass = array_shift($postfixShape);
            $m = 1;
            $n = array_product($prefixShape);
            $k = array_product($postfixShape);
            $reductionDims = false;
            $outputShape = array_merge($prefixShape,$postfixShape);
        // }
//echo "outputShape=[".implode(',',$outputShape)."]\n";
        if($dtype===null) {
            $dtype = $A->dtype();
        }
        if($output->shape()!=$outputShape) {
            throw new InvalidArgumentException("Unmatch output shape of dimension: ".
                                        '['.implode(',',$outputShape).']'.'['.implode(',',$output).']');
        }

        $AA = $A->buffer();
        $offA = $A->offset();
        $XX = $X->buffer();
        $offX = $X->offset();
        $BB = $output->buffer();
        $offB = $output->offset();

        // if($scatterAdd) {
            $reverse=true;
            $addMode=true;
        // }

        //fwrite(STDERR,"(m=$m,n=$n,k=$countX)");
        switch($mode) {
            case 0: {
                $this->la->getOpenCLMath()->scatterAdd_0(
                    $n,
                    $k,
                    $numClass,
                    $XX,$offX,
                    $AA,$offA,
                    $BB,$offB,
                    $events, $waitEvents
                );
                break;
            }
            case 1: {
                $this->la->getOpenCLMath()->scatterAdd_1(
                    $n,
                    $k,
                    $numClass,
                    $XX,$offX,
                    $AA,$offA,
                    $BB,$offB,
                    $events, $waitEvents
                );
                break;
            }
            case 2: {
                $this->la->getOpenCLMath()->scatterAdd_2(
                    $n,
                    $k,
                    $numClass,
                    $XX,$offX,
                    $AA,$offA,
                    $BB,$offB,
                    $events, $waitEvents
                );
                break;
            }
            case 3: {
                $this->la->getOpenCLMath()->scatterAdd_3(
                    $n,
                    $k,
                    $numClass,
                    $XX,$offX,
                    $AA,$offA,
                    $BB,$offB,
                    $events, $waitEvents
                );
                break;
            }
            case 4: {
                $this->la->getOpenCLMath()->scatterAdd_4(
                    $n,
                    $k,
                    $numClass,
                    $XX,$offX,
                    $AA,$offA,
                    $BB,$offB,
                    $events, $waitEvents
                );
                break;
            }
            default:
                throw new InvalidArgumentException("invalid mode: ".$mode);
        }

        $this->la->finish();
        return $A;
    }

    protected function timeSimulation(int $rows,int $cols,int $numClass) : int
    {
        $a[0] = 10; $b[0] = 10;
        $a[1] = 10; $b[1] = 10;
        $a[2] = 10; $b[2] = 10;
        $c = 10000;
        return ($a[0]*$rows+$b[0])*($a[1]*$cols+$b[1])*($a[2]*$numClass+$b[2])+$c;
    }

    public function showGraphScatterAdd(int $mode=null,bool $details=null) : void
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
                $this->drawGraphRowsCols3(8,$mo,$mode,$axes,$details,$marker);
                $plt->figure();
                $axes = $plt->getAxes();
                $this->drawGraphColsRows3(8,$mo,$mode,$axes,$details,$marker);
                $plt->figure();
                $axes = $plt->getAxes();
                $this->drawGraphNumClassRows3(8,$mo,$mode,$axes,$details,$marker);
                $plt->figure();
                $axes = $plt->getAxes();
                $this->drawGraphRowsNumClass3(8,$mo,$mode,$axes,$details,$marker);
                $plt->figure();
                $axes = $plt->getAxes();
                $this->drawGraphNumClassCols3(8,$mo,$mode,$axes,$details,$marker);
                $plt->figure();
                $axes = $plt->getAxes();
                $this->drawGraphColsNumClass3(8,$mo,$mode,$axes,$details,$marker);
            }
        } else {
            [$fig,$axes] = $plt->subplots(3,2);
            foreach($modes as $mode) {
                if(count($modes)==1) {
                    $marker = null;
                } else {
                    $marker = $colors[$mode];
                }
                $this->drawGraphRowsCols3(8,$mo,$mode,$axes[0],$details,$marker);
                $this->drawGraphColsRows3(8,$mo,$mode,$axes[1],$details,$marker);
                $this->drawGraphNumClassRows3(8,$mo,$mode,$axes[2],$details,$marker);
                $this->drawGraphRowsNumClass3(8,$mo,$mode,$axes[3],$details,$marker);
                $this->drawGraphNumClassCols3(8,$mo,$mode,$axes[4],$details,$marker);
                $this->drawGraphColsNumClass3(8,$mo,$mode,$axes[5],$details,$marker);
            }
        }
        $plt->show();
    }

    public function drawGraphRowsCols3(
        int $nc,object $mo,int $mode,object $ax,bool $details,string $marker) : void
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
        $numClass = $nc;
        if(isset($graph[$numClass])) {
            foreach ($graph[$numClass] as $cols => $gr) {
                if(count($gr)) {
                    $gr = $mo->transpose($mo->array($gr));
                    $ax->plot($gr[0],$gr[1],$marker,'cols='.$cols);
                }
            }
        }
        $ax->setYScale('log');
        $ax->setXScale('log');
        if($details) {
            $ax->legend();
            $ax->setXLabel('rows');
            $ax->setTitle("mode$mode(numClass=$numClass)");
        }
    }

    public function drawGraphColsRows3(
        int $nc,object $mo,int $mode,object $ax,bool $details,stirng $marker)
    {
        $times = $this->loadParameter('ScatterAddTimesMode'.$mode.'.php');
        // cols <-> rows
        $numClass = 0;
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
            $numClass = $nc;
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
            $ax->setTitle("mode$mode(numClass=$numClass)");
        }
    }

    public function drawGraphNumClassRows3(
        int $co,object $mo,int $mode,object $ax,bool $details,string $marker) : void
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
        $cols = $co;
        if(isset($graph[$cols])) {
            foreach ($graph[$cols] as $rows => $gr) {
                if(count($gr)) {
                    $gr = $mo->transpose($mo->array($gr));
                    $ax->plot($gr[0],$gr[1],$marker,'rows='.$rows);
                }
            }
        }
        $ax->setYScale('log');
        $ax->setXScale('log');
        if($details) {
            $ax->legend();
            $ax->setXLabel('numClass');
            $ax->setTitle("mode$mode(cols=$cols)");
        }
    }

    public function drawGraphRowsNumClass3(
        int $co,object $mo,int $mode,object $ax,bool $details,string $marker) : void
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
        $cols = $co;
        if(isset($graph[$cols])) {
            foreach ($graph[$cols] as $numClass => $gr) {
                if(count($gr)) {
                    $gr = $mo->transpose($mo->array($gr));
                    $ax->plot($gr[0],$gr[1],$marker,'numClass='.$numClass);
                }
            }
        }
        $ax->setYScale('log');
        $ax->setXScale('log');
        if($details) {
            $ax->legend();
            $ax->setXLabel('rows');
            $ax->setTitle("mode$mode(cols=$cols)");
        }
    }

    public function drawGraphNumClassCols3(
        int $ro,object $mo,int $mode,object $ax,bool $details,string $marker) : void
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
        $rows = $ro;
        if(isset($graph[$rows])) {
            foreach ($graph[$rows] as $cols => $gr) {
                if(count($gr)) {
                    $gr = $mo->transpose($mo->array($gr));
                    $ax->plot($gr[0],$gr[1],$marker,'cols='.$cols);
                }
            }
        }
        $ax->setYScale('log');
        $ax->setXScale('log');
        if($details) {
            $ax->legend();
            $ax->setXLabel('numClass');
            $ax->setTitle("mode$mode(rows=$rows)");
        }
    }

    public function drawGraphColsNumClass3(
        int $ro,object $mo,int $mode,object $ax,bool $details,string $marker) : void
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
        $rows = $ro;
        if(isset($graph[$rows])) {
            foreach ($graph[$rows] as $numClass => $gr) {
                if(count($gr)) {
                    $gr = $mo->transpose($mo->array($gr));
                    $ax->plot($gr[0],$gr[1],$marker,'numClass='.$numClass);
                }
            }
        }
        $ax->setYScale('log');
        $ax->setXScale('log');
        if($details) {
            $ax->legend();
            $ax->setXLabel('cols');
            $ax->setTitle("mode$mode(rows=$rows)");
        }
    }

    protected function getHomeDirectory() : string
    {
        if(PHP_OS=='WINNT') {
            return getenv('USERPROFILE');
        } elseif(PHP_OS=='Linux') {
            return getenv('HOME');
        }
    }

    protected function saveParameter(string $filename,mixed $value) : void
    {
        $dir = $this->getHomeDirectory().'/.rindow';
        if(!file_exists($dir)) {
            mkdir($dir,0755,true);
        }
        $code = "<?php\nreturn unserialize('".str_replace(array('\\','\''), array('\\\\','\\\''), serialize($value))."');";
        file_put_contents($dir.'/'.$filename,$code);
    }

    protected function loadParameter(string $filename,bool $default=null) : ?int
    {
        $filepath = __DIR__.'/params/'.$filename;
        if(!$default) {
            $tmpfilepath = $this->getHomeDirectory().'/.rindow/'.$filename;
            if(file_exists($tmpfilepath)) {
                $filepath = $tmpfilepath;
            }
        }
        if(!file_exists($filepath)) {
            return null;
        }
        $times = include $filepath;
        return $times;
    }

    protected function deleteParameter(string $filename) : void
    {
        $filepath = $this->getHomeDirectory().'/.rindow/'.$filename;
        if(file_exists($filepath)) {
            unlink($filepath);
        }
    }

    public function getScatterAddParameter(
        int $mode,int $rows,int $cols,int $numClass) : int
    {
        $times = $this->loadParameter('ScatterAddTimesMode'.$mode.'.php');
        if(isset($times[$rows][$cols][$numClass])) {
            return $times[$rows][$cols][$numClass];
        } else {
            return 0;
        }
    }

    public function setScatterAddParameter(
        int $mode,int $rows,int $cols,int $numClass,int $value,bool $force=null) : void
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

    /**
     * @param array<int,array<int,int>> $data
     */
    public function editGraphScatterAdd(int $mode,array $data) : void
    {
        echo "mode$mode\n";
        $times = $this->loadParameter('ScatterAddTimesMode'.$mode.'.php',$default=true);
        foreach ($data as $rows => $colsData) {
            if(!array_key_exists($rows,$times)) {
                throw new \Exception("Invalid rows:[$rows] in mode".$mode);
            }
            foreach ($colsData as $cols => $numClassData) {
                if(!array_key_exists($rows,$times)) {
                    throw new \Exception("Invalid cols:[$rows][$cols] in mode".$mode);
                }
                foreach ($numClassData as $numClass => $value) {
                    echo "[$rows][$cols][$numClass]: ".$times[$rows][$cols][$numClass]." * $value\n";
                    $value = (int)($times[$rows][$cols][$numClass]*$value);
                    $times[$rows][$cols][$numClass] = $value;
                }
            }
        }
        $this->saveParameter('ScatterAddTimesMode'.$mode.'.php',$times);
    }

    public function tunningReduceSum(int $mode,int $maxTime,int $limitTime) : void
    {
        fwrite(STDERR,"\n");
        $la = $this;
        $rowMax = 1048576*256;
        $colMax = 1048576*256;
        //$classMax = 1048576;
        $times = $this->loadParameter('TempRedSumTimesMode'.$mode.'.php');
        //$times['pointer'] = [131072,8];
        //$times['prevTime'] = 0;//$times[32768][16][256];
        //unset($times[131072]);
        //unset($times[262144]);
        //unset($times[524288]);
        //unset($times[1048576]);
        //$this->saveParameter('TempTimesMode'.$mode.'.php',$times);
        //return;
        if($times) {
            [$startI,$startJ] = $times['pointer'];
            $prevTime = $times['prevTime'];
        } else {
            [$startI,$startJ] = [8,8];
            $times = [];
            $prevTime = 0;
        }
        fwrite(STDERR,"start=($startI,$startJ)\n");
        for($i=$startI;$i<=$rowMax;$i<<=1) {
            $startI=8;
            if(!array_key_exists($i,$times)) {
                $times[$i] = [];
            }
            fwrite(STDERR, "rows=$i\n");
            $timeover=false;
            $prevEndK = 0;
            for($j=$startJ;$j<=$colMax;$j<<=1) {
                $startJ=8;
                fwrite(STDERR, $j);
                $timeover=false;
                $predictTimeover=false;
                $limitTimeN = $limitTime;
                if(($mode==2 && $i>=131072)) {
                    $limitTimeN = 10**9.6;
                    fwrite(STDERR,"L");
                }
                if(($mode==4 && $i==524288 && $j>=32 )||
                    ($mode==4 && $i==1048576 && $j>=8)
                ){
                    fwrite(STDERR,"X");
                    $timeover=true;
                    $predictTimeover=true;
                } elseif(($j==8||$j==16)&&
                   isset($times[$i>>1][$j])&&
                   isset($times[$i>>2][$j])&&
                   $times[$i>>1][$j]>0&&
                   $times[$i>>2][$j]>0
                ) {
                    fwrite(STDERR,"K");
                    $prev1point = $i>>1;
                    $prev2point = $i>>2;
                    $nextPoint = $i;
                    $prev1timep = $times[$i>>1][$j];
                    $prev2timep = $times[$i>>2][$j];
                    $predictTime = ($prev1timep-$prev2timep)/
                                    ($prev1point-$prev2point)*
                                    ($nextPoint-$prev2point)+$prev1timep;
                    //fwrite(STDERR,"<".sprintf("%3.2f",log10($predictTime)).">");
                    if($predictTime > $limitTimeN) {
                        $timeover=true;
                        $predictTimeover=true;
                        fwrite(STDERR,"p");
                        fwrite(STDERR,"(".sprintf("%3.2f",log10($predictTime)).")");
                    }
                } elseif(($i>16 && $j==8  && !isset($times[$i>>1][$j]))||
                         ($i>16 && $j==16 && !isset($times[$i>>1][$j]))
                ) {
                    fwrite(STDERR,"x");
                    $times['prevTime'] = $prevTime;
                    $times['pointer'] = [$i,$j*2,8];
                    $predictTimeover=true;
                    $timeover=true;
                } elseif(($j>16)&&
                   isset($times[$i][$j>>1])&&
                   isset($times[$i][$j>>2])&&
                   $times[$i][$j>>1]>0&&
                   $times[$i][$j>>2]>0
                ) {
                    $prev1point = $j>>1;
                    $prev2point = $j>>2;
                    $nextPoint = $j;
                    $prev1timep = $times[$i][$j>>1];
                    $prev2timep = $times[$i][$j>>2];
                    $predictTime = ($prev1timep-$prev2timep)/
                                    ($prev1point-$prev2point)*
                                    ($nextPoint-$prev2point)+$prev1timep;
                    if($predictTime > $limitTimeN) {
                        $timeover=true;
                        $predictTimeover=true;
                        fwrite(STDERR,"P");
                        fwrite(STDERR,"(".sprintf("%3.2f",log10($predictTime)).")");
                    }
                }
                if(!$predictTimeover) {
                    try {
                        $times[$i][$j] = $this->timeReduceSum(
                            $la,$try=10,$mode,$i,$j);
                    } catch(\Exception $e) {
                        fwrite(STDERR, ",".$e->getMessage());
                        $times[$i][$j] = 0;
                    }
                    $time = $times[$i][$j];
                    if($time>$maxTime) {
                        $timeover=true;
                        fwrite(STDERR,"T");
                        fwrite(STDERR,"(".sprintf("%3.2f",log10($time)).")");
                    }
                    if($time==0) {
                        $timeover=true;
                    }
                }
                fwrite(STDERR, ",");
                if($timeover) {
                    break;
                }
            }
            $times['prevTime'] = $prevTime;
            $times['pointer'] = [$i*2,8];
            $this->saveParameter('TempRedSumTimesMode'.$mode.'.php',$times);
            fwrite(STDERR, "\n");
        }
        unset($times['pointer']);
        unset($times['prevTime']);
        $this->saveParameter('ReduceSumTimesMode'.$mode.'.php',$times);
        $this->deleteParameter('TempRedSumTimesMode'.$mode.'.php');
    }

    protected function timeReduceSum(object $la,int $try,int $mode,int $rows,int $cols) : int
    {
        switch($mode) {
            case 0:{
                break;
            }
            case 1:{
                if($rows > $this->maxWorkItem) {// || $rows*$cols*$numClass>256*256*256*4) {
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

        $a = $this->la->alloc([$rows,$cols],NDArray::float32);
        $x = $this->la->alloc([$cols],NDArray::float32);

        $this->la->fill(0.0,$a);
        $this->la->fill(1,$x);
        $this->reduceSumExTest($a,$axis=0,$x,null,null,null,$mode);
        $time = 0;
        for($i=0;$i<$try;$i++) {
            $start = hrtime(true);
            $this->reduceSumExTest($a,$axis=0,$x,null,null,null,$mode);
            $end = hrtime(true);
            $time += $end-$start;
        }
        return $time/$try;
    }

    public function reduceSumTest(
        NDArray $A,
        int $axis=null,
        NDArray $X=null,
        int $dtypeX=null,
        object $events=null,object $waitEvents=null,
        int $mode = null
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
            $X = $this->la->alloc([$rows],$dtypeX);
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

        $math = $this->la->getOpenCLMath();
        switch($mode) {
            case 0: {
                $math->reduceSum0(
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
                $math->reduceSum1(
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
                $math->reduceSum2(
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
                $math->reduceSum3(
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

        $this->la->finish();
        return $X;
    }

    public function reduceSumExTest(
        NDArray $A,
        int $axis=null,
        NDArray $B=null,
        int $dtype=null,
        object $events=null,object $waitEvents=null,
        int $mode = null
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
            $B = $this->la->alloc($outputShape,$dtype);
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

        $math = $this->la->getOpenCLMath();
        switch($mode) {
            case 0: {
                $math->reduceSumEx0(
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
                $math->reduceSumEx1(
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
                $math->reduceSumEx2(
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
                $math->reduceSumEx3(
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

        $this->la->finish();
        return $B;
    }

    public function showGraphReduceSum(int $mode=null,bool $details=null) : void
    {
        $marker = null;
        $colors = [0=>'m',1=>'b',2=>'g',3=>'r'];
        $mo = $this->mo;
        $plt = new Plot(null,$mo);
        if($mode===null) {
            $modes = range(0,3);
        } elseif(is_int($mode)) {
            $modes = [$mode];
        } elseif(is_array($mode)) {
            $modes = $mode;
        } else {
            throw new InvalidArgumentException("Invalid mode");
        }
        if($details) {
            $marker = null;
            $plt->figure();
            $axes[0] = $plt->getAxes();
            $plt->figure();
            $axes[1] = $plt->getAxes();
            $arts0 = [];
            $arts1 = [];
            $labels0 = [];
            $labels1 = [];
            foreach($modes as $mode) {
                if(count($modes)==1) {
                    $marker = null;
                    $legend = true;
                } else {
                    $marker = $colors[$mode];
                    $legend = false;
                }
                $artists = $this->drawGraphRowsCols2($mo,$mode,$axes[0],$details,$marker,$legend);
                if(!$legend) {
                    $arts0[] = $artists[0];
                    $labels0[] = 'mode='.$mode;
                }
                $this->drawGraphColsRows2($mo,$mode,$axes[1],$details,$marker,$legend);
                if(!$legend) {
                    $arts1[] = $artists[0];
                    $labels1[] = 'mode='.$mode;
                } else {
                    $axes[0]->setTitle('mode='.$mode);
                    $axes[1]->setTitle('mode='.$mode);
                }
            }
            $axes[0]->legend($arts0,$labels0);
            $axes[1]->legend($arts1,$labels1);
        } else {
            [$fig,$axes] = $plt->subplots(1,2);
            $legend = false;
            foreach($modes as $mode) {
                if(count($modes)==1) {
                    $marker = null;
                } else {
                    $marker = $colors[$mode];
                }
                $this->drawGraphRowsCols2($mo,$mode,$axes[0],$details,$marker,$legend);
                $this->drawGraphColsRows2($mo,$mode,$axes[1],$details,$marker,$legend);
            }
        }
        $plt->show();
    }

    /**
     * @return array<object>
     */
    protected function drawGraphRowsCols2(
        object $mo,int $mode,object $ax,bool $details,string $marker,bool $legend) : array
    {
        $times = $this->loadParameter('ReduceSumExTimesMode'.$mode.'.php');
        // rows <-> cols
        $graph = [];
        foreach ($times as $rows => $colsData) {
            foreach ($colsData as $cols => $value) {
                if(!isset($graph[$cols])) {
                    $graph[$cols] = [];
                }
                if($value>0) {
                    $graph[$cols][] = [$rows,$value];
                }
            }
        }
        $artists = [];
        foreach ($graph as $cols => $gr) {
            if(count($gr)) {
                $gr = $mo->transpose($mo->array($gr));
                $artists = array_merge($artists,$ax->plot($gr[0],$gr[1],$marker,'cols='.$cols));
            }
        }
        $ax->setYScale('log');
        $ax->setXScale('log');
        if($legend) {
            $ax->legend();
        }
        if($details) {
            $ax->setXLabel('rows');
        }
        return $artists;
    }

    /**
     * @return array<object>
     */
    protected function drawGraphColsRows2(
        object $mo,int $mode,object $ax,bool $details,string $marker,bool $legend) : array
    {
        $times = $this->loadParameter('ReduceSumExTimesMode'.$mode.'.php');
        // cols <-> rows
        $artists = [];
        foreach ($times as $rows => $colsData) {
            $graph = [];
            foreach ($colsData as $cols => $value) {
                if(!isset($graph[$rows])) {
                    $graph[$rows] = [];
                }
                if($value>0) {
                    $graph[$rows][] = [$cols,$value];
                }
            }
            if(isset($graph[$rows]) && count($graph[$rows])) {
                $gr = $mo->transpose($mo->array($graph[$rows]));
                $artists = array_merge($artists,$ax->plot($gr[0],$gr[1],$marker,'rows='.$rows));
            }
        }
        $ax->setYScale('log');
        $ax->setXScale('log');
        if($legend) {
            $ax->legend();
        }
        if($details) {
            $ax->setXLabel('cols');
        }
        return $artists;
    }

    /**
     * @param array<int,array<int,int>> $data
     */
    public function editGraphReduceSum(int $mode,array $data) : void
    {
        echo "mode$mode\n";
        $times = $this->loadParameter('ReduceSumExTimesMode'.$mode.'.php',$default=true);
        foreach ($data as $rows => $colsData) {
            if(!array_key_exists($rows,$times)) {
                throw new \Exception("Invalid rows:[$rows] in mode".$mode);
            }
            foreach ($colsData as $cols => $value) {
                if(!array_key_exists($rows,$times)) {
                    throw new \Exception("Invalid cols:[$rows][$cols] in mode".$mode);
                }
                echo "[$rows][$cols]: ".$times[$rows][$cols]." * $value\n";
                $value = (int)($times[$rows][$cols]*$value);
                $times[$rows][$cols] = $value;
            }
        }
        $this->saveParameter('ReduceSumExTimesMode'.$mode.'.php',$times);
    }

}
