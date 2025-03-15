<?php
namespace Rindow\Math\Matrix\Drivers\MatlibCL;

//use Rindow\OpenCL\Buffer as Buffer;
use RuntimeException;
use InvalidArgumentException;
use LogicException;
use Interop\Polite\Math\Matrix\NDArray;
use Interop\Polite\Math\Matrix\OpenCL;
use Interop\Polite\Math\Matrix\LinearBuffer as HostBufferInterface;
use Interop\Polite\Math\Matrix\DeviceBuffer as BufferInterface;
use Interop\Polite\Math\Matrix\Buffer as AnyBuffer;
use Interop\Polite\Math\Matrix\BLAS;
use Rindow\Math\Matrix\NDArrayPhp;
use Rindow\Math\Matrix\NDArrayCL;
use Rindow\Math\Matrix\Drivers\Service;
use Rindow\Math\Matrix\ComplexUtils;

class OpenCLBlas
{
    use ComplexUtils;

    /** @var array<int,string> $dtypeToString */
    protected $dtypeToString = [
        NDArray::bool=>'bool',
        NDArray::int8=>'int8',   NDArray::uint8=>'uint8',
        NDArray::int16=>'int16', NDArray::uint16=>'uint16',
        NDArray::int32=>'int32', NDArray::uint32=>'uint32',
        NDArray::int64=>'int64', NDArray::uint64=>'uint64',
        NDArray::float16=>'float16',
        NDArray::float32=>'float32', NDArray::float64=>'float64',
        NDArray::complex64=>'complex64', NDArray::complex128=>'complex128',
    ];

    /** @var array<int,string> $dtypeToOpenCLType */
    protected $dtypeToOpenCLType = [
        NDArray::bool=>'uchar',
        NDArray::int8=>'char',   NDArray::uint8=>'uchar',
        NDArray::int16=>'short', NDArray::uint16=>'ushort',
        NDArray::int32=>'int', NDArray::uint32=>'uint',
        NDArray::int64=>'long', NDArray::uint64=>'ulong',
        NDArray::float16=>'half',
        NDArray::float32=>'float', NDArray::float64=>'double',
        NDArray::complex64=>'float', NDArray::complex128=>'double',
    ];

    /** @var array<int,string> $alternativeUnsignedCLType */
    protected $alternativeUnsignedCLType = [
        NDArray::bool=>'char',
        NDArray::uint8=>'char',
        NDArray::uint16=>'short',
        NDArray::uint32=>'int',
        NDArray::uint64=>'long',
    ];

    /** @var array<int,int|float> $smallests */
    protected $smallests = [
        NDArray::bool  => 0,
        NDArray::int8  => -128,         NDArray::uint8  => 0,
        NDArray::int16 => -32768,       NDArray::uint16 => 0,
        NDArray::int32 => -2147483648,  NDArray::uint32 => 0,
        NDArray::int64 => -9223372036854775808, NDArray::uint64 => 0,
        NDArray::float16 => -1.0e+14,
        NDArray::float32 => -1.0e+37, NDArray::float64 => -1.0e+37,
        NDArray::complex64 => -1.0e+37, NDArray::complex128 => -1.0e+37,
    ];
    /** @var array<int,int|float> $largests */
    protected $largests = [
        NDArray::bool  => 1,
        NDArray::int8  => 127,          NDArray::uint8  => 255,
        NDArray::int16 => 32767,        NDArray::uint16 => 65535,
        NDArray::int32 => 2147483647,  NDArray::uint32 => 4294967295,
        NDArray::int64 => 9223372036854775807, NDArray::uint64 => 9223372036854775807, //
        NDArray::float16 => 1.0e+14,
        NDArray::float32 => 1.0e+37, NDArray::float64 => 1.0e+37,
        NDArray::complex64 => 1.0e+37, NDArray::complex128 => 1.0e+37,
    ];

    /** @var array<int> $intTypes */
    protected $intTypes= [
        NDArray::int8,NDArray::int16,NDArray::int32,NDArray::int64,
        NDArray::uint8,NDArray::uint16,NDArray::uint32,NDArray::uint64,
    ];

    protected object $context;
    protected object $queue;
    protected Service $service;
    /** @var array<string> $deviceTypes */
    protected array $deviceTypes = [];
    /** @var array<string,string> $sources */
    protected array $sources = [];
    /** @var array<string,object> $program */
    protected array $program = [];
    protected bool $fp64;
    /** @var array<int> $maxWorkItem */
    protected array $maxWorkItem;
    protected int $localMemSize;
    protected int $maxComputeUnits;
    protected ?int $kernelMultiple=null;
    protected ?int $testMode=null;
    /** @var array<mixed> $timesPredictionScatterAdd */
    protected $timesPredictionScatterAdd = [];
    /** @var array<mixed> $timesPredictionReduceSum */
    protected $timesPredictionReduceSum = [];

    public function __construct(object $queue, Service $service)
    {
        $this->queue = $queue;
        $this->context = $queue->getContext();
        $this->service = $service;
        $devices = $this->context->getInfo(OpenCL::CL_CONTEXT_DEVICES);
        $extensions = $devices->getInfo(0,OpenCL::CL_DEVICE_EXTENSIONS);
        if(strpos($extensions,'cl_khr_fp64')===false) {
            $this->fp64 = false;
        } else {
            $this->fp64 = true;
        }
        $this->maxWorkItem = $devices->getInfo(0,OpenCL::CL_DEVICE_MAX_WORK_ITEM_SIZES);
        $this->localMemSize = $devices->getInfo(0,OpenCL::CL_DEVICE_LOCAL_MEM_SIZE);
        $this->maxComputeUnits = $devices->getInfo(0,OpenCL::CL_DEVICE_MAX_COMPUTE_UNITS);
        
        $deviceType = $devices->getInfo(0,OpenCL::CL_DEVICE_TYPE);
        $nDev = count($devices);
        $types = [];
        for($i=0;$i<$nDev;$i++) {
            if($deviceType&OpenCL::CL_DEVICE_TYPE_CPU) { $types[] = "CPU"; }
            if($deviceType&OpenCL::CL_DEVICE_TYPE_GPU) { $types[] = "GPU"; }
            if($deviceType&OpenCL::CL_DEVICE_TYPE_ACCELERATOR) { $types[] = "ACCEL"; }
            if($deviceType&OpenCL::CL_DEVICE_TYPE_CUSTOM) { $types[] = "CUSTOM"; }
        }
        $this->deviceTypes = $types;
    }

    public function setTestMode(?int $testMode) : void
    {
        $this->testMode = $testMode;
    }

    public function fp64() : bool
    {
        return $this->fp64;
    }

    public function dtypeToString(int $dtype) : string
    {
        return $this->dtypeToString[$dtype];
    }

    protected function assertFP64() : void
    {
        if(!$this->fp64) {
            throw new RuntimeException('This device does not support 64-bit floating point.');
        }
    }

    /**
     * @return array<int>
     */
    public function maxWorkItem() : array
    {
        return $this->maxWorkItem;
    }

    /**
     * @return array<string>
     */
    public function deviceTypes() : array
    {
        return $this->deviceTypes;
    }

    protected function kernelMultiple(object $kernel) : int
    {
        if($this->kernelMultiple) {
            return $this->kernelMultiple;
        }
        $this->kernelMultiple = $kernel->getWorkGroupInfo(OpenCL::CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE);
        return $this->kernelMultiple;
    }

    protected function createKernel(string $name) : object
    {
        if(!isset($this->program[$name])) {
            $source = $this->sources[$name];
            $program = $this->service->opencl()->Program($this->context,$source);
            try {
                $program->build();
            } catch (\RuntimeException $e) {
                echo get_class($e)."\n";
                echo $e->getMessage();
                if($e->getCode()==OpenCL::CL_BUILD_PROGRAM_FAILURE) {
                    echo "CL_PROGRAM_BUILD_STATUS=".$program->getBuildInfo(OpenCL::CL_PROGRAM_BUILD_STATUS)."\n";
                    echo "CL_PROGRAM_BUILD_OPTIONS=".$program->getBuildInfo(OpenCL::CL_PROGRAM_BUILD_OPTIONS)."\n";
                    echo "CL_PROGRAM_BUILD_LOG=".$program->getBuildInfo(OpenCL::CL_PROGRAM_BUILD_LOG)."\n";
                    echo "CL_PROGRAM_BINARY_TYPE=".$program->getBuildInfo(OpenCL::CL_PROGRAM_BINARY_TYPE)."\n";
                }
                throw $e;
            }
            $this->program[$name] = $program;
        } else {
            $program = $this->program[$name];
        }
        $kernel = $this->service->opencl()->Kernel($program,$name);
        return $kernel;
    }

    protected function assertShapeParameter(
        string $name, int $n) : void
    {
        if($n<1) {
            throw new InvalidArgumentException("Argument $name must be greater than 0.");
        }
    }
    
    protected function assertVectorBufferSpec(
        string $name, BufferInterface $buffer, int $n, int $offset, int $inc) : void
    {
        if($offset<0) {
            throw new InvalidArgumentException("Argument offset$name must be greater than equals 0.");
        }
        if($inc<1) {
            throw new InvalidArgumentException("Argument inc$name must be greater than 0.");
        }
        if($offset+($n-1)*$inc >= count($buffer)) {
            throw new InvalidArgumentException("Vector specification too large for buffer$name.");
        }
    }

    protected function assertMatrixBufferSpec(
        string $name, BufferInterface $buffer,
        int $m, int $n, int $offset, int $ld) : void
    {
        if($offset<0) {
            throw new InvalidArgumentException("Argument offset$name must be greater than equals 0.");
        }
        if($ld<1) {
            throw new InvalidArgumentException("Argument ld$name must be greater than 0.");
        }
        if($offset+($m-1)*$ld+($n-1) >= count($buffer)) {
            throw new InvalidArgumentException("Matrix specification too large for buffer$name.");
        }
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

    protected function isComplex(int $dtypeX) : bool
    {
        return ($dtypeX==NDArray::complex64||$dtypeX==NDArray::complex128);
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
        BufferInterface $X, int $offsetX, int $incX,
        ?object $events=null, ?object $waitEvents=null
        ) : void
    {
        $this->assertShapeParameter('n',$n);
        $this->assertVectorBufferSpec('X', $X, $n, $offsetX, $incX);

        //$idx = $offsetX;
        //$alpha = $this->cleanFloatNumber($alpha,'alpha');
        //for ($i=0; $i<$n; $i++,$idx+=$incX) {
        //    $X[$idx] = $X[$idx] * $alpha;
        //}
        $dtype = $X->dtype();
        if($dtype==NDArray::float64) {
            $this->assertFP64();
        }
        $type = $this->dtypeToOpenCLType[$dtype];
        if(!$this->isComplex($dtype)) {
            $alpha = $this->cleanFloatNumber($alpha,'alpha');
            $kernel_name = "scal_{$type}";
            if(!isset($this->sources[$kernel_name])) {
                $this->sources[$kernel_name] =
                    "__kernel void {$kernel_name}(\n".
                    "    const        {$type} alpha,\n".
                    "        __global {$type} * x,\n".
                    "    const        int offsetX,\n".
                    "    const        int incX)\n".
                    "{\n".
                    "    int i = get_global_id(0);\n".
                    "    x[offsetX+i*incX] = x[offsetX+i*incX] * alpha;\n".
                    "}\n";
            }
            $kernel = $this->createKernel($kernel_name);
            $kernel->setArg(0,$alpha,$dtype);
            $kernel->setArg(1,$X);
            $kernel->setArg(2,$offsetX,NDArray::int32);
            $kernel->setArg(3,$incX,NDArray::int32);
        } else {
            $alpha = $this->cleanComplexNumber($alpha,'alpha');
            $kernel_name = "cscal_{$type}";
            if(!isset($this->sources[$kernel_name])) {
                $this->sources[$kernel_name] =
                    "__kernel void {$kernel_name}(\n".
                    "    const        {$type} alpha_real,\n".
                    "    const        {$type} alpha_imag,\n".
                    "        __global {$type} * x,\n".
                    "    const        int offsetX,\n".
                    "    const        int incX)\n".
                    "{\n".
                    "    int i = get_global_id(0);\n".
                    "    int pos = (offsetX+i*incX)*2;\n".
                    "    {$type} x_real = x[pos]*alpha_real - x[pos+1]*alpha_imag;\n".
                    "    {$type} x_imag = x[pos]*alpha_imag + x[pos+1]*alpha_real;\n".
                    "    x[pos] = x_real;\n".
                    "    x[pos+1] = x_imag;\n".
                    "}\n";
            }
            $dtype = ($dtype==NDArray::complex64) ? NDArray::float32 : NDArray::float64;
            $kernel = $this->createKernel($kernel_name);
            $kernel->setArg(0,$alpha->real,$dtype);
            $kernel->setArg(1,$alpha->imag,$dtype);
            $kernel->setArg(2,$X);
            $kernel->setArg(3,$offsetX,NDArray::int32);
            $kernel->setArg(4,$incX,NDArray::int32);
        }
        $global_work_size = [$n];
        $kernel->enqueueNDRange($this->queue,$global_work_size,null,null,
            $events,$waitEvents);
    }

    /**
     *  Y := alpha * X + Y
     */
    public function axpy(
        int $n,
        float|object $alpha,
        BufferInterface $X, int $offsetX, int $incX,
        BufferInterface $Y, int $offsetY, int $incY,
        ?object $events=null, ?object $waitEvents=null
        ) : void
    {
        $this->assertShapeParameter('n',$n);
        $this->assertVectorBufferSpec('X', $X, $n, $offsetX, $incX);
        $this->assertVectorBufferSpec('Y', $Y, $n, $offsetY, $incY);

        //$idxX = $offsetX;
        //$idxY = $offsetY;
        //if($this->cistype($X->dtype())) {
        //    $alpha = $this->cleanComplexNumber($alpha,'alpha');
        //    for ($i=0; $i<$n; $i++,$idxX+=$incX,$idxY+=$incY) {
        //        $Y[$idxY] = $this->cadd($this->cmul($alpha,$X[$idxX]),$Y[$idxY]);
        //    }
        //} else {
        //    $alpha = $this->cleanFloatNumber($alpha,'alpha');
        //    if($alpha==1.0) {   // Y := X + Y
        //        for ($i=0; $i<$n; $i++,$idxX+=$incX,$idxY+=$incY) {
        //            $Y[$idxY] = $X[$idxX] + $Y[$idxY];
        //        }
        //    } else {            // Y := a*X + Y
        //        for ($i=0; $i<$n; $i++,$idxX+=$incX,$idxY+=$incY) {
        //            $Y[$idxY] = $alpha * $X[$idxX] + $Y[$idxY];
        //        }
        //    }
        //}
        $dtype = $X->dtype();
        if($dtype==NDArray::float64) {
            $this->assertFP64();
        }
        $type = $this->dtypeToOpenCLType[$dtype];
        if(!$this->isComplex($dtype)) {
            $alpha = $this->cleanFloatNumber($alpha,'alpha');
            $kernel_name = "axpy_{$type}";
            if(!isset($this->sources[$kernel_name])) {
                $this->sources[$kernel_name] =
                    "__kernel void {$kernel_name}(\n".
                    "    const        {$type} alpha,\n".
                    "        __global {$type} * x,\n".
                    "    const        int offsetX,\n".
                    "    const        int incX,\n".
                    "        __global {$type} * y,\n".
                    "    const        int offsetY,\n".
                    "    const        int incY)\n".
                    "{\n".
                    "    int i = get_global_id(0);\n".
                    "    y[offsetY+i*incY] = y[offsetY+i*incY] + alpha * x[offsetX+i*incX];\n".
                    "}\n";
            }
            $kernel = $this->createKernel($kernel_name);
            $kernel->setArg(0,$alpha,$dtype);
            $kernel->setArg(1,$X);
            $kernel->setArg(2,$offsetX,NDArray::int32);
            $kernel->setArg(3,$incX,NDArray::int32);
            $kernel->setArg(4,$Y);
            $kernel->setArg(5,$offsetY,NDArray::int32);
            $kernel->setArg(6,$incY,NDArray::int32);
        } else {
            $alpha = $this->cleanComplexNumber($alpha,'alpha');
            $kernel_name = "caxpy_{$type}";
            if(!isset($this->sources[$kernel_name])) {
                $this->sources[$kernel_name] =
                    "__kernel void {$kernel_name}(\n".
                    "    const        {$type} alpha_real,\n".
                    "    const        {$type} alpha_imag,\n".
                    "        __global {$type} * x,\n".
                    "    const        int offsetX,\n".
                    "    const        int incX,\n".
                    "        __global {$type} * y,\n".
                    "    const        int offsetY,\n".
                    "    const        int incY)\n".
                    "{\n".
                    "    int i = get_global_id(0);\n".
                    "    int posX = (offsetX+i*incX)*2;\n".
                    "    int posY = (offsetY+i*incY)*2;\n".
                    "    {$type} y_real = y[posY]   + x[posX]*alpha_real - x[posX+1]*alpha_imag;\n".
                    "    {$type} y_imag = y[posY+1] + x[posX]*alpha_imag + x[posX+1]*alpha_real;\n".
                    "    y[posY] = y_real;\n".
                    "    y[posY+1] = y_imag;\n".
                    "}\n";
            }
            $dtype = ($dtype==NDArray::complex64) ? NDArray::float32 : NDArray::float64;
            $kernel = $this->createKernel($kernel_name);
            $kernel->setArg(0,$alpha->real,$dtype);
            $kernel->setArg(1,$alpha->imag,$dtype);
            $kernel->setArg(2,$X);
            $kernel->setArg(3,$offsetX,NDArray::int32);
            $kernel->setArg(4,$incX,NDArray::int32);
            $kernel->setArg(5,$Y);
            $kernel->setArg(6,$offsetY,NDArray::int32);
            $kernel->setArg(7,$incY,NDArray::int32);
        }
        $global_work_size = [$n];
        $kernel->enqueueNDRange($this->queue,$global_work_size,null,null,
            $events,$waitEvents);
    }

    public function copy(
        int $n,
        BufferInterface $X, int $offsetX, int $incX,
        BufferInterface $Y, int $offsetY, int $incY,
        ?object $events=null, ?object $waitEvents=null
        ) : void
    {
        $this->assertShapeParameter('n',$n);
        $this->assertVectorBufferSpec('X', $X, $n, $offsetX, $incX);
        $this->assertVectorBufferSpec('Y', $Y, $n, $offsetY, $incY);

        //$idxX = $offsetX;
        //$idxY = $offsetY;
        //for($i=0; $i<$n; $i++,$idxX+=$incX,$idxY+=$incY) {
        //    $Y[$idxY] = $X[$idxX];
        //}
        $dtype = $X->dtype();
        if($dtype==NDArray::float64) {
            $this->assertFP64();
        }
        $type = $this->dtypeToOpenCLType[$dtype];
        if(!$this->isComplex($dtype)) {
            $kernel_name = "copy_{$type}";
            if(!isset($this->sources[$kernel_name])) {
                $this->sources[$kernel_name] =
                    "__kernel void {$kernel_name}(\n".
                    "        __global {$type} * x,\n".
                    "    const        int offsetX,\n".
                    "    const        int incX,\n".
                    "        __global {$type} * y,\n".
                    "    const        int offsetY,\n".
                    "    const        int incY)\n".
                    "{\n".
                    "    int i = get_global_id(0);\n".
                    "    y[offsetY+i*incY] = x[offsetX+i*incX];\n".
                    "}\n";
            }
        } else {
            $kernel_name = "ccopy_{$type}";
            if(!isset($this->sources[$kernel_name])) {
                $this->sources[$kernel_name] =
                    "__kernel void {$kernel_name}(\n".
                    "        __global {$type} * x,\n".
                    "    const        int offsetX,\n".
                    "    const        int incX,\n".
                    "        __global {$type} * y,\n".
                    "    const        int offsetY,\n".
                    "    const        int incY)\n".
                    "{\n".
                    "    int i = get_global_id(0);\n".
                    "    int posX = (offsetX+i*incX)*2;\n".
                    "    int posY = (offsetY+i*incY)*2;\n".
                    "    y[posY] = x[posX];\n".
                    "    y[posY+1] = x[posX+1];\n".
                    "}\n";
            }
        }
        $kernel = $this->createKernel($kernel_name);
        $kernel->setArg(0,$X);
        $kernel->setArg(1,$offsetX,NDArray::int32);
        $kernel->setArg(2,$incX,NDArray::int32);
        $kernel->setArg(3,$Y);
        $kernel->setArg(4,$offsetY,NDArray::int32);
        $kernel->setArg(5,$incY,NDArray::int32);
        $global_work_size = [$n];
        $kernel->enqueueNDRange($this->queue,$global_work_size,null,null,
            $events,$waitEvents);
    }

    public function gemm(
        int $order,
        int $transA,
        int $transB,
        int $m,
        int $n,
        int $k,
        float|object $alpha,
        BufferInterface $A, int $offsetA, int $ldA,
        BufferInterface $B, int $offsetB, int $ldB,
        float|object $beta,
        BufferInterface $C, int $offsetC, int $ldC,
        ?object $events=null, ?object $waitEvents=null
        ) : void
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

        //$idA_m = $offsetA;
        //$idC_m = $offsetC;
        //$alpha = $this->cleanFloatNumber($alpha,'alpha');
        //$beta = $this->cleanFloatNumber($beta,'beta');
        //for ($im=0; $im<$m; $im++,$idA_m+=$ldA_m,$idC_m+=$ldC) {
        //    $idB_n = $offsetB;
        //    $idC = $idC_m;
        //    for ($in=0; $in<$n; $in++,$idB_n+=$ldB_n,$idC++) {
        //        $idA = $idA_m;
        //        $idB = $idB_n;
        //        $acc = 0.0;
        //        for ($ik=0; $ik<$k; $ik++,$idA+=$ldA_k,$idB+=$ldB_k) {
        //            $acc += $A[$idA] * $B[$idB];
        //        }
        //        if($beta==0.0) {
        //            $C[$idC] = $alpha * $acc;
        //        } else {
        //            $C[$idC] = $alpha * $acc + $beta * $C[$idC];
        //        }
        //    }
        //}
        $dtype = $A->dtype();
        if($dtype==NDArray::float64) {
            $this->assertFP64();
        }
        $type = $this->dtypeToOpenCLType[$dtype];
        if(!$this->isComplex($dtype)) {
            $alpha = $this->cleanFloatNumber($alpha,'alpha');
            $beta = $this->cleanFloatNumber($beta,'beta');
            $kernel_name = "gemm_{$type}";
            if(!isset($this->sources[$kernel_name])) {
                $this->sources[$kernel_name] =
                    "__kernel void {$kernel_name}(\n".
                    "    const        int k,\n".
                    "    const        {$type} alpha,\n".
                    "        __global {$type} * a,\n".
                    "    const        int offsetA,\n".
                    "    const        int ldA_m,\n".
                    "    const        int ldA_k,\n".
                    "        __global {$type} * b,\n".
                    "    const        int offsetB,\n".
                    "    const        int ldB_n,\n".
                    "    const        int ldB_k,\n".
                    "    const        {$type} beta,\n".
                    "        __global {$type} * c,\n".
                    "    const        int offsetC,\n".
                    "    const        int ldC)\n".
                    "{\n".
                    "    int i = get_global_id(0);\n".
                    "    int j = get_global_id(1);\n".
                    "    int posC = offsetC+i*ldC+j;\n".
                    "    {$type} acc = 0.0;\n".
                    "    for(int p=0;p<k;p++) {\n".
                    "        acc += a[offsetA+i*ldA_m+p*ldA_k]*b[offsetB+j*ldB_n+p*ldB_k];\n".
                    "    }\n".
                    "    if(beta==0.0) {\n".
                    "        c[posC] = alpha*acc;\n".
                    "    } else {\n".
                    "        c[posC] = alpha*acc + beta*c[posC];\n".
                    "    }\n".
                    "}\n";
            }
            $kernel = $this->createKernel($kernel_name);
            $kernel->setArg(0,$k,NDArray::int32);
            $kernel->setArg(1,$alpha,$dtype);
            $kernel->setArg(2,$A);
            $kernel->setArg(3,$offsetA,NDArray::int32);
            $kernel->setArg(4,$ldA_m,NDArray::int32);
            $kernel->setArg(5,$ldA_k,NDArray::int32);
            $kernel->setArg(6,$B);
            $kernel->setArg(7,$offsetB,NDArray::int32);
            $kernel->setArg(8,$ldB_n,NDArray::int32);
            $kernel->setArg(9,$ldB_k,NDArray::int32);
            $kernel->setArg(10,$beta,$dtype);
            $kernel->setArg(11,$C);
            $kernel->setArg(12,$offsetC,NDArray::int32);
            $kernel->setArg(13,$ldC,NDArray::int32);
        } else {
            $conjA = $conjA ? 1:0;
            $conjB = $conjB ? 1:0;
            $alpha = $this->cleanComplexNumber($alpha,'alpha');
            $beta = $this->cleanComplexNumber($beta,'beta');
            //$hasAlpha = !$this->cisone($alpha);
            $hasBeta = (!$this->ciszero($beta)) ? 1:0;
            $kernel_name = "cgemm_{$type}";
            if(!isset($this->sources[$kernel_name])) {
                $this->sources[$kernel_name] =
                    "__kernel void {$kernel_name}(\n".
                    "    const        int k,\n".
                    "    const        {$type} alpha_real,\n".
                    "    const        {$type} alpha_imag,\n".
                    "        __global {$type} * a,\n".
                    "    const        int offsetA,\n".
                    "    const        int ldA_m,\n".
                    "    const        int ldA_k,\n".
                    "        __global {$type} * b,\n".
                    "    const        int offsetB,\n".
                    "    const        int ldB_n,\n".
                    "    const        int ldB_k,\n".
                    "    const        {$type} beta_real,\n".
                    "    const        {$type} beta_imag,\n".
                    "        __global {$type} * c,\n".
                    "    const        int offsetC,\n".
                    "    const        int ldC,\n".
                    "    const        int conjA,\n".
                    "    const        int conjB,\n".
                    "    const        int hasBeta)\n".
                    "{\n".
                    "    int i = get_global_id(0);\n".
                    "    int j = get_global_id(1);\n".
                    "    int posC = (offsetC+i*ldC+j)*2;\n".
                    "    {$type} acc_real = 0.0;\n".
                    "    {$type} acc_imag = 0.0;\n".
                    "    for(int p=0;p<k;p++) {\n".
                    "        int posA = (offsetA+i*ldA_m+p*ldA_k)*2;\n".
                    "        int posB = (offsetB+j*ldB_n+p*ldB_k)*2;\n".
                    "        {$type} valueA_real = a[posA];\n".
                    "        {$type} valueA_imag = a[posA+1];\n".
                    "        {$type} valueB_real = b[posB];\n".
                    "        {$type} valueB_imag = b[posB+1];\n".
                    "        if(conjA) {\n".
                    "            valueA_imag = -valueA_imag;\n".
                    "        }\n".
                    "        if(conjB) {\n".
                    "            valueB_imag = -valueB_imag;\n".
                    "        }\n".
                    "        acc_real += valueA_real*valueB_real - valueA_imag*valueB_imag;\n".
                    "        acc_imag += valueA_real*valueB_imag + valueA_imag*valueB_real;\n".
                    "    }\n".
                    "    {$type} tmp_real = alpha_real*acc_real - alpha_imag*acc_imag;\n".
                    "    {$type} tmp_imag = alpha_real*acc_imag + alpha_imag*acc_real;\n".
                    "    acc_real = tmp_real;\n".
                    "    acc_imag = tmp_imag;\n".
                    "    if(hasBeta) {\n".
                    "        {$type} valueC_real = beta_real*c[posC]   - beta_imag*c[posC+1];\n".
                    "        {$type} valueC_imag = beta_real*c[posC+1] + beta_imag*c[posC];\n".
                    "        acc_real += valueC_real;\n".
                    "        acc_imag += valueC_imag;\n".
                    "    }\n".
                    "    c[posC]   = acc_real;\n".
                    "    c[posC+1] = acc_imag;\n".
                    "}\n";
            }
            $dtype = ($dtype==NDArray::complex64) ? NDArray::float32 : NDArray::float64;
            $kernel = $this->createKernel($kernel_name);
            $kernel->setArg(0,$k,NDArray::int32);
            $kernel->setArg(1,$alpha->real,$dtype);
            $kernel->setArg(2,$alpha->imag,$dtype);
            $kernel->setArg(3,$A);
            $kernel->setArg(4,$offsetA,NDArray::int32);
            $kernel->setArg(5,$ldA_m,NDArray::int32);
            $kernel->setArg(6,$ldA_k,NDArray::int32);
            $kernel->setArg(7,$B);
            $kernel->setArg(8,$offsetB,NDArray::int32);
            $kernel->setArg(9,$ldB_n,NDArray::int32);
            $kernel->setArg(10,$ldB_k,NDArray::int32);
            $kernel->setArg(11,$beta->real,$dtype);
            $kernel->setArg(12,$beta->imag,$dtype);
            $kernel->setArg(13,$C);
            $kernel->setArg(14,$offsetC,NDArray::int32);
            $kernel->setArg(15,$ldC,NDArray::int32);
            $kernel->setArg(16,$conjA,NDArray::int32);
            $kernel->setArg(17,$conjB,NDArray::int32);
            $kernel->setArg(18,$hasBeta,NDArray::int32);
        }
        $global_work_size = [$m,$n];
        $kernel->enqueueNDRange($this->queue,$global_work_size,null,null,
            $events,$waitEvents);
    }
}
