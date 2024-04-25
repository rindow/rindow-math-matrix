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
use Rindow\Math\Matrix\NDArrayPhp;
use Rindow\Math\Matrix\NDArrayCL;
use Rindow\Math\Matrix\Drivers\Service;

class OpenCLMath
{
    /** @var array<int,string> $dtypeToString */
    protected $dtypeToString = [
        NDArray::bool=>'bool',
        NDArray::int8=>'int8',   NDArray::uint8=>'uint8',
        NDArray::int16=>'int16', NDArray::uint16=>'uint16',
        NDArray::int32=>'int32', NDArray::uint32=>'uint32',
        NDArray::int64=>'int64', NDArray::uint64=>'uint64',
        NDArray::float16=>'float16',
        NDArray::float32=>'float32', NDArray::float64=>'float64',
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
    ];

    /** @var array<int> $intTypes */
    protected $intTypes= [
        NDArray::int8,NDArray::int16,NDArray::int32,NDArray::int64,
        NDArray::uint8,NDArray::uint16,NDArray::uint32,NDArray::uint64,
    ];

    /** @var array<string,string> $kernelCoreOperation */
    protected $kernelCoreOperation = [
        'qsum' =>
            "i >>= 1;\n".
            "if(lid < i) {\n".
            "    local_work[lid] += local_work[lid + i];\n".
            "}\n".
            "barrier(CLK_LOCAL_MEM_FENCE);\n",
        'qmax' =>
            "i >>= 1;\n".
            "if(lid < i) {\n".
            //   *** CAUTION ***
            //   if NaN set NaN
            //   Compatible with reduce_max of tensorflow 2.6
            "    if((local_work[lid] < local_work[lid + i])||isnan(local_work[lid + i])) {\n".
            "        local_work[lid] = local_work[lid + i];\n".
            "    }\n".
            "}\n".
            "barrier(CLK_LOCAL_MEM_FENCE);\n",
        'qimax' =>
            "i >>= 1;\n".
            "if(lid < i) {\n".
            //   *** CAUTION ***
            //   if NaN set NaN
            //   Compatible with reduce_max of tensorflow 2.6
            "    if(!(local_work[lid] >= local_work[lid + i])) {\n".
            "        local_work[lid]  = local_work[lid + i];\n".
            "        local_iwork[lid] = local_iwork[lid + i];\n".
            "    }\n".
            "}\n".
            "barrier(CLK_LOCAL_MEM_FENCE);\n",
        'qimin' =>
            "i >>= 1;\n".
            "if(lid < i) {\n".
            //   *** CAUTION ***
            //   if NaN set NaN
            //   Compatible with reduce_max of tensorflow 2.6
            "    if(!(local_work[lid] <= local_work[lid + i])) {\n".
            "        local_work[lid]  = local_work[lid + i];\n".
            "        local_iwork[lid] = local_iwork[lid + i];\n".
            "    }\n".
            "}\n".
            "barrier(CLK_LOCAL_MEM_FENCE);\n",

        'lsum-1' =>
            "value += input;\n",
        'lsum-2' =>
            "local_work[lid] = local_work[lid] + local_work[lid + i];\n",
        'lsum-3' =>
            "temp_buffer[grid] = local_work[0];\n",
        'lsum-4' =>
            "local_work[lid] = temp_buffer[lid] + temp_buffer[lws+lid];\n",

        'lrsum-1' =>
            "value += input;\n",
        'lrsum-2' =>
            "local_work[lid] = local_work[lid] + local_work[lid + i];\n",
        'lrsum-3' =>
            "temp_buffer[parallel_item_id*grs+grid] = local_work[0];\n",
        'lrsum-4' =>
            "local_work[lid] = temp_buffer[parallel_item_id*lws*2 + lid] + temp_buffer[parallel_item_id*lws*2 + lws+lid];\n",

        'lrmax-1' =>
            "if(value < input||isnan(input)) {\n".
            "    value = input;\n".
            "}\n",
        'lrmax-2' =>
            "if(local_work[lid] < local_work[lid + i] || isnan(local_work[lid + i])) {\n".
            "    local_work[lid] = local_work[lid + i];\n".
            "}\n",
        'lrmax-3' =>
            "temp_buffer[parallel_item_id*grs+grid] = local_work[0];\n",
        'lrmax-4' =>
            "if((temp_buffer[parallel_item_id*lws*2 + lid] < temp_buffer[parallel_item_id*lws*2 + lws+lid]) ||isnan(temp_buffer[parallel_item_id*lws*2 + lws+lid])) {\n".
            "    local_work[lid] = temp_buffer[parallel_item_id*lws*2 + lws+lid];\n".
            "} else {\n".
            "    local_work[lid] = temp_buffer[parallel_item_id*lws*2 + lid];\n".
            "}\n",

        'lrimax-1' =>
            "if(value < input) {\n".
            "    value = input;\n".
            "    ivalue = input_index;\n".
            "}\n",
        'lrimax-2' =>
            "if(local_work[lid] < local_work[lid + i]) {\n".
            "    local_work[lid] = local_work[lid + i];\n".
            "    local_iwork[lid] = local_iwork[lid + i];\n".
            "}\n",
        'lrimax-3' =>
            "temp_buffer[parallel_item_id*grs+grid] = local_work[0];\n".
            "temp_ibuffer[parallel_item_id*grs+grid] = local_iwork[0];\n",
        'lrimax-4' =>
            "if(temp_buffer[parallel_item_id*lws*2 + lid] < temp_buffer[parallel_item_id*lws*2 + lws+lid]) {\n".
            "    local_work[lid] = temp_buffer[parallel_item_id*lws*2 + lws+lid];\n".
            "    local_iwork[lid] = temp_ibuffer[parallel_item_id*lws*2 + lws+lid];\n".
            "} else {\n".
            "    local_work[lid] = temp_buffer[parallel_item_id*lws*2 + lid];\n".
            "    local_iwork[lid] = temp_ibuffer[parallel_item_id*lws*2 + lid];\n".
            "}\n",

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
    protected ?int $kernelMultiple=null;
    protected bool $hasDiv5Bug;
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
        $this->checkDiv5Bug();
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

    public function checkDiv5Bug() : void
    {
        $kernel_name = "checkDiv5Bug";
        if(!isset($this->sources[$kernel_name])) {
            $this->sources[$kernel_name] =
                "__kernel void {$kernel_name}(\n".
                "    const        int alpha,\n".
                "    const        int beta,\n".
                "        __global int * results\n".
                ")\n".
                "{\n".
                "    results[0] = alpha%beta;\n".
                "    results[1] = alpha/beta;\n".
                "}\n";
        }

        $alpha = 3;
        $beta = 5;
        $results = new NDArrayPhp([0,0],NDArray::int32,service:$this->service);
        $flags = OpenCL::CL_MEM_READ_WRITE | OpenCL::CL_MEM_COPY_HOST_PTR;
        $resultsCL = new NDArrayCL(
            $this->queue,
            $results->buffer(), $results->dtype(), $results->shape(),
            $results->offset(), $flags,
            service:$this->service
        );
        
        $kernel = $this->createKernel($kernel_name);
        $kernel->setArg(0,$alpha,NDArray::int32);
        $kernel->setArg(1,$beta,NDArray::int32);
        $kernel->setArg(2,$resultsCL->buffer());
        $global_work_size = [1];
        $kernel->enqueueNDRange($this->queue,$global_work_size,null,null,
            null,null);
        $this->queue->finish();
        $results = $resultsCL->toNDArray();
        if($results[0]!=3 || $results[1]!=0) {
            $this->hasDiv5Bug = true;
        } else {
            $this->hasDiv5Bug = false;
        }
    }

    public function hasDiv5Bug() : bool
    {
        return $this->hasDiv5Bug;
    }

    protected function splitPointer(
        string $low,
        string $high,
        string $pointer,
        string $base
    ) : string
    {
        return
            "    const uint {$low} = {$pointer}%{$base};\n".
            "    const uint {$high} = {$pointer}/{$base};\n";
    }

    public function kernelTemplateQSum(string $inputs,string $outputs) : string
    {
        $operation = $this->kernelCoreOperation['qsum'];
        $initial = 0;
        return $this->kernelQTemplate($operation,$inputs,$outputs,$initial);
    }

    public function kernelTemplateSSum(string $inputs,string $outputs) : string
    {
        $operation = $this->kernelCoreOperation['qsum'];
        $initial = 0;
        return $this->kernelSTemplate($operation,$inputs,$outputs,$initial);
    }

    public function kernelTemplateLSingleSum1(string $inputs,int $dtype) : string
    {
        $operation1 = $this->kernelCoreOperation['lsum-1'];
        $operation2 = $this->kernelCoreOperation['lsum-2'];
        $operation3 = $this->kernelCoreOperation['lsum-3'];
        $type = $this->dtypeToOpenCLType[$dtype];
        $initial = 0;
        return $this->kernelLTemplate1(
            $operation1,$operation2,$operation3,$inputs,$type,$initial);
    }

    public function kernelTemplateLSingleSum2(string $output) : string
    {
        $operation2 = $this->kernelCoreOperation['lsum-2'];
        $operation4 = $this->kernelCoreOperation['lsum-4'];
        return $this->kernelLTemplate2(
            $operation2,$operation4,$output);
    }

    public function kernelTemplateLSum1(string $inputs,int $dtype) : string
    {
        $operation1 = $this->kernelCoreOperation['lrsum-1'];
        $operation2 = $this->kernelCoreOperation['lrsum-2'];
        $operation3 = $this->kernelCoreOperation['lrsum-3'];
        $type = $this->dtypeToOpenCLType[$dtype];
        $initial = 0;
        return $this->kernelLTemplate1(
            $operation1,$operation2,$operation3,$inputs,$type,$initial);
    }

    public function kernelTemplateLSum2(string $output) : string
    {
        $operation2 = $this->kernelCoreOperation['lrsum-2'];
        $operation4 = $this->kernelCoreOperation['lrsum-4'];
        return $this->kernelLTemplate2(
            $operation2,$operation4,$output);
    }

    public function kernelTemplateQMax(string $inputs,string $outputs,int $dtype) : string
    {
        $operation = $this->kernelCoreOperation['qmax'];
        $initial = $this->smallests[$dtype];
        return $this->kernelQTemplate($operation,$inputs,$outputs,$initial);
    }

    public function kernelTemplateSMax(string $inputs,string $outputs,int $dtype) : string
    {
        $operation = $this->kernelCoreOperation['qmax'];
        $initial = $this->smallests[$dtype];
        return $this->kernelSTemplate($operation,$inputs,$outputs,$initial);
    }

    public function kernelTemplateQiMax(string $inputs,string $outputs,int $dtype) : string
    {
        $operation = $this->kernelCoreOperation['qimax'];
        $initial = $this->smallests[$dtype];
        return $this->kernelQiTemplate($operation,$inputs,$outputs,$initial);
    }

    public function kernelTemplateSiMax(string $inputs,string $outputs,int $dtype) : string
    {
        $operation = $this->kernelCoreOperation['qimax'];
        $initial = $this->smallests[$dtype];
        return $this->kernelSiTemplate($operation,$inputs,$outputs,$initial);
    }

    public function kernelTemplateLMax1(string $inputs,int $dtype) : string
    {
        $operation1 = $this->kernelCoreOperation['lrmax-1'];
        $operation2 = $this->kernelCoreOperation['lrmax-2'];
        $operation3 = $this->kernelCoreOperation['lrmax-3'];
        $type = $this->dtypeToOpenCLType[$dtype];
        $initial = $this->smallests[$dtype];
        return $this->kernelLTemplate1(
            $operation1,$operation2,$operation3,$inputs,$type,$initial);
    }

    public function kernelTemplateLMax2(string $output) : string
    {
        $operation2 = $this->kernelCoreOperation['lrmax-2'];
        $operation4 = $this->kernelCoreOperation['lrmax-4'];
        return $this->kernelLTemplate2(
            $operation2,$operation4,$output);
    }

    public function kernelTemplateLiMax1(string $inputs,int $dtype) : string
    {
        $operation1 = $this->kernelCoreOperation['lrimax-1'];
        $operation2 = $this->kernelCoreOperation['lrimax-2'];
        $operation3 = $this->kernelCoreOperation['lrimax-3'];
        $type = $this->dtypeToOpenCLType[$dtype];
        $initial = $this->smallests[$dtype];
        return $this->kernelLiTemplate1(
            $operation1,$operation2,$operation3,$inputs,$type,$initial);
    }

    public function kernelTemplateLiMax2(string $output) : string
    {
        $operation2 = $this->kernelCoreOperation['lrimax-2'];
        $operation4 = $this->kernelCoreOperation['lrimax-4'];
        return $this->kernelLTemplate2(
            $operation2,$operation4,$output);
    }

    public function kernelTemplateQiMin(
        string $inputs,
        string $outputs,
        int $dtype
        ) : string
    {
        $operation = $this->kernelCoreOperation['qimin'];
        $initial = $this->largests[$dtype];
        return $this->kernelQiTemplate($operation,$inputs,$outputs,$initial);
    }

    public function kernelTemplateSiMin(
        string $inputs,
        string $outputs,
        int $dtype
        ) : string 
    {
        $operation = $this->kernelCoreOperation['qimin'];
        $initial = $this->largests[$dtype];
        return $this->kernelSiTemplate($operation,$inputs,$outputs,$initial);
    }

    public function kernelQTemplate(
        string $operation,
        string $inputs,
        string $outputs,
        int|float $initial
        ) : string
    {
        return
        "{\n".
        "    const uint lid = get_local_id(0);\n".
        "    const uint lws = get_local_size(0);\n".
        "    uint seg_count = segments;\n".
        "    uint local_items = work_items;\n".
        "    uint left_local_items = total_local_items;\n".
        "    int is_first=1;\n".
        "    while(1) {\n".
        "        for(int seg=0;seg<seg_count;seg++) {\n".
        "            if(lid<local_items) {\n".
        "                if(is_first) {\n".
        "                    {$inputs}\n".
        "                } else {\n".
        "                    local_work[lid] = seg_work[seg*lws+lid];\n".
        "                }\n".
        "            } else {\n".
        "                local_work[lid] = {$initial};\n".
        "            }\n".
        "            barrier(CLK_LOCAL_MEM_FENCE);\n".
        "            int i = lws;\n".
        "            while( i>1 ) {\n".
        "                {$operation}\n".
        "            }\n".
        "            if(lid == 0) {\n".
        "                seg_work[seg] = local_work[0];\n".
        "            }\n".
        "            barrier(CLK_LOCAL_MEM_FENCE);\n".
        "            left_local_items -= local_items;\n".
        "            if(left_local_items<local_items) {\n".
        "                local_items = left_local_items;\n".
        "            }\n".
        "        }\n".
        "        if(seg_count<=1) {\n".
        "            break;\n".
        "        }\n".
        "        is_first = 0;\n".
        "        left_local_items = seg_count;\n".
        "        if(left_local_items<lws) {\n".
        "            local_items = left_local_items;\n".
        "        } else {\n".
        "            local_items = lws;\n".
        "        }\n".
        "        seg_count = (seg_count+lws-1)/lws;\n". // Not covered by div5bug. because lws is always a power of 2
        "    }\n".
        "    if(lid == 0) {\n".
        "        {$outputs}\n".
        "    }\n".
        "}\n";
    }

    public function kernelQiTemplate(
        string $operation,
        string $inputs,
        string $outputs,
        int|float $initial
        ) : string
    {
        return
        "{\n".
        "    const uint lid = get_local_id(0);\n".
        "    const uint lws = get_local_size(0);\n".
        "    uint seg_count = segments;\n".
        "    uint local_items = work_items;\n".
        "    uint left_local_items = total_local_items;\n".
        "    int is_first=1;\n".
        "    while(1) {\n".
        "        for(int seg=0;seg<seg_count;seg++) {\n".
        "            if(lid<local_items) {\n".
        "                if(is_first) {\n".
        "                    {$inputs}\n".
        "                } else {\n".
        "                    local_work[lid] = seg_work[seg*lws+lid];\n".
        "                    local_iwork[lid] = seg_iwork[seg*lws+lid];\n".
        "                }\n".
        "            } else {\n".
        "                local_work[lid] = {$initial};\n".
        "                local_iwork[lid] = 0;\n".
        "            }\n".
        "            barrier(CLK_LOCAL_MEM_FENCE);\n".
        "            int i = lws;\n".
        "            while( i>1 ) {\n".
        "                {$operation}\n".
        "            }\n".
        "            if(lid == 0) {\n".
        "                seg_work[seg] = local_work[0];\n".
        "                seg_iwork[seg] = local_iwork[0];\n".
        "            }\n".
        "            barrier(CLK_LOCAL_MEM_FENCE);\n".
        "            left_local_items -= local_items;\n".
        "            if(left_local_items<local_items) {\n".
        "                local_items = left_local_items;\n".
        "            }\n".
        "        }\n".
        "        if(seg_count<=1) {\n".
        "            break;\n".
        "        }\n".
        "        is_first = 0;\n".
        "        left_local_items = seg_count;\n".
        "        if(left_local_items<lws) {\n".
        "            local_items = left_local_items;\n".
        "        } else {\n".
        "            local_items = lws;\n".
        "        }\n".
        "        seg_count = (seg_count+lws-1)/lws;\n". // Not covered by div5bug. because lws is always a power of 2
        "    }\n".
        "    if(lid == 0) {\n".
        "        {$outputs}\n".
        "    }\n".
        "}\n";
    }

    public function kernelSTemplate(
        string $operation,
        string $inputs,
        string $outputs,
        int|float $initial
        ) : string
    {
        return
        "{\n".
        "    const uint lid = get_local_id(0);\n".
        "    const uint lws = get_local_size(0);\n".
        #"    local_work[lid] = 0;\n".
        #"    {$type} input = 0;\n".
        "    if(lid<total_local_items) {\n".
        "        {$inputs}\n".
        "    } else {\n".
        "        local_work[lid]  = {$initial};\n".
        "    }\n".
        #"    local_work[lid] = input;\n".
        "    barrier(CLK_LOCAL_MEM_FENCE);\n".
        "    int i = lws;\n".
        "    while( i>1 ) {\n".
        "        {$operation}\n".
        "    }\n".
        "    if(lid == 0) {\n".
        "        {$outputs}\n".
        "    }\n".
        "}\n";
    }

    public function kernelSiTemplate(
        string $operation,
        string $inputs,
        string $outputs,
        int|float $initial
        ) : string
    {
        return
        "{\n".
        "    const uint lid = get_local_id(0);\n".
        "    const uint lws = get_local_size(0);\n".
        #"    local_work[lid] = 0;\n".
        #"    {$type} input = 0;\n".
        "    if(lid<total_local_items) {\n".
        "        {$inputs}\n".
        "    } else {\n".
        "        local_work[lid] = {$initial};\n".
        "        local_iwork[lid] = 0;\n".
        "    }\n".
        #"    local_work[lid] = input;\n".
        "    barrier(CLK_LOCAL_MEM_FENCE);\n".
        "    int i = lws;\n".
        "    while( i>1 ) {\n".
        "        {$operation}\n".
        "    }\n".
        "    if(lid == 0) {\n".
        "        {$outputs}\n".
        "    }\n".
        "}\n";
    }

    public function kernelLTemplate1(
        string $operation1,
        string $operation2,
        string $operation3,
        string $inputs,
        string $type,
        int|float $initial
        ) : string
    {
        return
        "{\n".
        "    const uint lid = get_local_id(0);\n".
        "    const uint grid = get_group_id(0);\n".
        "    const uint lws = get_local_size(0);\n".
        "    const uint grs = get_num_groups(0);\n".
        "    {$type} value = {$initial};\n".
        "    uint local_item_id = grid*lws + lid;\n".
        "    while(local_item_id < total_local_items) {\n".
        "        {$inputs}\n".
        "        {$operation1}\n".
        "        local_item_id += lws*grs;\n".
        "    }\n".
        "    local_work[lid] = value;\n".
        "    barrier(CLK_LOCAL_MEM_FENCE);\n".
        "    for(int i=lws/2; i>0; i>>=1) {\n".
        "        if(lid < i) {\n".
        "            {$operation2}\n".
        "        }\n".
        "        barrier(CLK_LOCAL_MEM_FENCE);\n".
        "    }\n".
        "    if(lid == 0) {\n".
        "        {$operation3}\n".
        "    }\n".
        "}\n";
    }

    public function kernelLTemplate2(
        string $operation2,
        string $operation4,
        string $output
        ) : string 
    {
        return
        "{\n".
        "    const uint lid = get_local_id(0);\n".
        "    const uint lws = get_local_size(0);\n".
        "    {$operation4}\n".
        "    barrier(CLK_LOCAL_MEM_FENCE);\n".
        "    for(uint i=lws/2; i>0; i>>=1) {\n".
        "        if (lid < i) {\n".
        "            {$operation2}\n".
        "        }\n".
        "        barrier(CLK_LOCAL_MEM_FENCE);\n".
        "    }\n".
        "    if (lid == 0) {\n".
        "        {$output}\n".
        "    }\n".
        "}\n";
    }

    public function kernelLiTemplate1(
        string $operation1,
        string $operation2,
        string $operation3,
        string $inputs,
        string $type,
        int|float $initial
        ) : string
    {
        return
        "{\n".
        "    const uint lid = get_local_id(0);\n".
        "    const uint grid = get_group_id(0);\n".
        "    const uint lws = get_local_size(0);\n".
        "    const uint grs = get_num_groups(0);\n".
        "    {$type} value = {$initial};\n".
        "    uint    ivalue = 0;\n".
        "    uint local_item_id = grid*lws + lid;\n".
        "    while(local_item_id < total_local_items) {\n".
        "        {$inputs}\n".
        "        {$operation1}\n".
        "        local_item_id += lws*grs;\n".
        "    }\n".
        "    local_work[lid] = value;\n".
        "    local_iwork[lid] = ivalue;\n".
        "    barrier(CLK_LOCAL_MEM_FENCE);\n".
        "    for(int i=lws/2; i>0; i>>=1) {\n".
        "        if(lid < i) {\n".
        "            {$operation2}\n".
        "        }\n".
        "        barrier(CLK_LOCAL_MEM_FENCE);\n".
        "    }\n".
        "    if(lid == 0) {\n".
        "        {$operation3}\n".
        "    }\n".
        "}\n";
    }

    protected function newEventList() : object
    {
        return $this->service->opencl()->EventList();
    }

    protected function newBuffer(
        int $size,int $flags=null,
        HostBufferInterface $hostBuffer=null, int $hostOffset=null,
        int $dtype=null) : BufferInterface
    {
        $hostOffset = $hostOffset ?? 0;
        return $this->service->buffer(Service::LV_ACCELERATED)->Buffer($this->context,
            $size,$flags,$hostBuffer,$hostOffset,$dtype);
    }

    protected function newHostBuffer(int $size, int $dtype) : HostBufferInterface
    {
        return $this->service->buffer(Service::LV_ADVANCED)->Buffer($size,$dtype);
    }

    /**
     * Rounding up boundaries
     *  ex. 1 base 4 => 4 
     *      5 base 4 => 8 
     *      6 base 4 => 12
     */
    protected function ceil(int $value,int $base) : int
    {
        return intdiv(($value+$base-1),$base)*$base;
    }

    protected function adjBoundary(int $bytes) : int
    {
        $bytes += ($bytes%4) ? 4-($bytes%4) : 0; // Adjust word boundary
        return (int)$bytes;
    }

    /**
     * Y := sum( X )
     */
    public function sum(
        int $n,
        BufferInterface $R, int $offsetR,
        BufferInterface $X, int $offsetX, int $incX,
        object $events=null, object $waitEvents=null
        ) : void
    {
        $dtype = $X->dtype();
        if($R->dtype()!=$dtype) {
            throw new InvalidArgumentException("Unmatch data type R and X:".
            $this->dtypeToString($R->dtype()).",".$this->dtypeToString($dtype));
        }
        if($dtype==NDArray::float64) {
            $this->assertFP64();
        }
        $max_work_items = $this->maxWorkItem[0];
        if($n <= $max_work_items) {
            $mode = 1;
        } elseif($n <= 30000) { // php74=131072
            $mode = 2;
        } else {
            $mode = 3;
        }
        if($this->testMode!==null) {
            $mode = $this->testMode;
        }
        //echo "mode=$mode($m,$n,$k)\n";
        switch($mode) {
            case 1:{
                $this->sum1(
                    $n,
                    $R, $offsetR,
                    $X, $offsetX, $incX,
                    $events, $waitEvents
                );
                break;
            }
            case 2:{
                $this->sum2(
                    $n,
                    $R, $offsetR,
                    $X, $offsetX, $incX,
                    $events, $waitEvents
                );
                break;
            }
            case 3:{
                $this->sum3(
                    $n,
                    $R, $offsetR,
                    $X, $offsetX, $incX,
                    $events, $waitEvents
                );
                break;
            }
            default: {
                throw new LogicException('Invalid Mode in sum(): mode='.$mode);
            }
        }
    }

    /**
     * Y := sum( X )
     */
    public function sum1(
        int $n,
        BufferInterface $R, int $offsetR,
        BufferInterface $X, int $offsetX, int $incX,
        object $events=null, object $waitEvents=null
        ) : void
    {
        $dtype = $X->dtype();

        $index_x = 'lid+offset_x';
        $total_local_items = $n;
        $max_work_items = $this->maxWorkItem[0];
        if($total_local_items>$max_work_items) {
            throw new InvalidArgumentException('too large array');
        } else {
            for($max_work_items=1; $max_work_items<$total_local_items;$max_work_items<<=1) {
                ;
            }
        }
        $value_size = $X->value_size();
        $type = $this->dtypeToOpenCLType[$dtype];
        $kernel_name = "sum_S_{$type}";
        if(!isset($this->sources[$kernel_name])) {
            $this->sources[$kernel_name] =
                "__kernel void {$kernel_name}(\n".
                "    const        uint total_local_items,\n".
                "        __global {$type} * r,\n".
                "    const        uint offset_r,\n".
                "        __global {$type} * x,\n".
                "    const        uint offset_x,\n".
                "    const        uint incx,\n".
                "         __local {$type} * local_work)\n".
                "{\n".
                    $this->kernelTemplateSSum(
                        "local_work[lid] = x[{$index_x}];",
                        "r[offset_r] = local_work[0];"
                    ).
                "}\n";
        }
        $kernel = $this->createKernel($kernel_name);

        $kernel->setArg(0,$total_local_items,NDArray::uint32);
        $kernel->setArg(1,$R);
        $kernel->setArg(2,$offsetR,NDArray::uint32);
        $kernel->setArg(3,$X);
        $kernel->setArg(4,$offsetX,NDArray::uint32);
        $kernel->setArg(5,$incX,NDArray::uint32);
        $kernel->setArg(6,null,$this->adjBoundary($max_work_items*$value_size));
        $global_work_size = [$max_work_items];
        $local_work_size = [$max_work_items];
        $kernel->enqueueNDRange($this->queue,$global_work_size,$local_work_size,null,
                $events,$waitEvents);
    }

    /**
     * Y := sum( X )
     */
    public function sum2(
        int $n,
        BufferInterface $R, int $offsetR,
        BufferInterface $X, int $offsetX, int $incX,
        object $events=null, object $waitEvents=null
        ) : void
    {
        $dtype = $X->dtype();

        $index_x = '(seg*lws+lid)+offset_x';
        $total_local_items = $n;
        $max_work_items = $this->maxWorkItem[0];
        if($total_local_items>$max_work_items) {
            $segments = (int)ceil($total_local_items/$max_work_items); // round up float
            $work_items = $max_work_items;
        } else {
            for($max_work_items=1; $max_work_items<$total_local_items;$max_work_items<<=1) {
                ;
            }
            $segments = 1; // round up float
            $work_items = $total_local_items;
        }
        $value_size = $X->value_size();
        $type = $this->dtypeToOpenCLType[$dtype];
        $kernel_name = "sum_M_{$type}";
        if(!isset($this->sources[$kernel_name])) {
            $this->sources[$kernel_name] =
                "__kernel void {$kernel_name}(\n".
                "    const        uint total_local_items,\n".
                "    const        uint segments,\n".
                "        __global {$type} * r,\n".
                "    const        uint offset_r,\n".
                "        __global {$type} * x,\n".
                "    const        uint offset_x,\n".
                "    const        uint incx,\n".
                "         __local {$type} * local_work,\n".
                "         __local {$type} * seg_work,\n".
                "    const        uint work_items)\n".
                "{\n".
                    $this->kernelTemplateQSum(
                        "local_work[lid] = x[{$index_x}];",
                        "r[offset_r] = seg_work[0];"
                    ).
                "}\n";
        }
        $kernel = $this->createKernel($kernel_name);

        //$seg_work_bytes = $segments*$value_size;
        //$seg_work_bytes += ($seg_work_bytes%4) ? 4-($seg_work_bytes%4) : 0; // Adjust Addressing Boundary
        $kernel->setArg(0,$total_local_items,NDArray::uint32);
        $kernel->setArg(1,$segments,NDArray::uint32);
        $kernel->setArg(2,$R);
        $kernel->setArg(3,$offsetR,NDArray::uint32);
        $kernel->setArg(4,$X);
        $kernel->setArg(5,$offsetX,NDArray::uint32);
        $kernel->setArg(6,$incX,NDArray::uint32);
        $kernel->setArg(7,null,$this->adjBoundary($max_work_items*$value_size));
        $kernel->setArg(8,null,$this->adjBoundary($segments*$value_size));
        $kernel->setArg(9,$work_items,NDArray::uint32);
        $global_work_size = [$max_work_items];
        $local_work_size = [$max_work_items];
        $kernel->enqueueNDRange($this->queue,$global_work_size,$local_work_size,null,
                $events,$waitEvents);
    }

    /**
     * Y := sum( X )
     */
    public function sum3(
        int $n,
        BufferInterface $R, int $offsetR,
        BufferInterface $X, int $offsetX, int $incX,
        object $events=null, object $waitEvents=null
        ) : void
    {
        $dtype = $X->dtype();

        $total_local_items = $n;
        $work_items1 = $this->maxWorkItem[0];
        $work_items2 = $this->maxWorkItem[0];
        if($total_local_items<$work_items1) {
            for($work_items1=1;$work_items1<$total_local_items;$work_items1<<=1) {
                ;
            }
        }
        if($total_local_items<$work_items2) {
            for($work_items2=1;$work_items2<$total_local_items;$work_items2<<=1) {
                ;
            }
        }

        $value_size = $X->value_size();
        $temp_size = 2*$work_items2;
        $temp_buffer = $this->newBuffer(
            $value_size*$temp_size,
            OpenCL::CL_MEM_READ_WRITE,null,null,$dtype);

        $type = $this->dtypeToOpenCLType[$dtype];
        $kernel_name1 = "sum_L1_{$type}";
        $kernel_name2 = "sum_L2_{$type}";
        if(!isset($this->sources[$kernel_name1])) {
            $this->sources[$kernel_name1] =
                "__kernel void {$kernel_name1}(\n".
                "    const        uint total_local_items,\n".
                "    const __global {$type} * x,\n".
                "    const        uint offset_x,\n".
                "    const        uint incx,\n".
                "        __global {$type} * temp_buffer,\n".
                "         __local {$type} * local_work)\n".
                "{\n".
                    $this->kernelTemplateLSingleSum1(
                        "{$type} input = x[local_item_id*incx + offset_x];",
                        $dtype
                    ).
                "}\n";
        }
        $kernel = $this->createKernel($kernel_name1);

        if(!isset($this->sources[$kernel_name2])) {
            $this->sources[$kernel_name2] =
                "__kernel void {$kernel_name2}(\n".
                "    const __global {$type} * temp_buffer,\n".
                "        __global {$type} * r,\n".
                "    const        uint offset_r,\n".
                "         __local {$type} * local_work)\n".
                "{\n".
                    $this->kernelTemplateLSingleSum2(
                        "r[offset_r] = local_work[0];"
                    ).
                "}\n";
        }
        $kernel2 = $this->createKernel($kernel_name2);

        $kernel->setArg(0,$total_local_items,NDArray::uint32);
        $kernel->setArg(1,$X);
        $kernel->setArg(2,$offsetX,NDArray::uint32);
        $kernel->setArg(3,$incX,NDArray::uint32);
        $kernel->setArg(4,$temp_buffer);
        $kernel->setArg(5,null,$this->adjBoundary($work_items1*$value_size));
        $global_work_size = [$work_items1*$temp_size];
        $local_work_size = [$work_items1];
        $sum1Events = $this->newEventList();
        $kernel->enqueueNDRange($this->queue,$global_work_size,$local_work_size,null,
                $sum1Events,$waitEvents);

        $kernel2->setArg(0,$temp_buffer);
        $kernel2->setArg(1,$R);
        $kernel2->setArg(2,$offsetR,NDArray::uint32);
        $kernel2->setArg(3,null,$this->adjBoundary($work_items2*$value_size));
        $global_work_size = [$work_items2];
        $local_work_size = [$work_items2];
        $kernel2->enqueueNDRange($this->queue,$global_work_size,$local_work_size,null,
                $events,$sum1Events);
    }

    /**
     * Y := imin( X )
     */
    public function imin(
        int $n,
        BufferInterface $R, int $offsetR,
        BufferInterface $X, int $offsetX, int $incX,
        object $events=null, object $waitEvents=null
        ) : void
    {
        $dtype = $X->dtype();
        if($R->dtype()!=NDArray::int32 && $R->dtype()!=NDArray::uint32) {
            throw new InvalidArgumentException("R must be 32bit integer:".
                                            $this->dtypeToString($R->dtype()));
        }
        //if($dtype!=NDArray::float64 && $dtype!=NDArray::float32)
        if($dtype==NDArray::bool) {
            throw new InvalidArgumentException("Unsuppored data type:".
                                            $this->dtypeToString($dtype));
        }

        if($dtype==NDArray::float64) {
            $this->assertFP64();
        }
        $max_work_items = $this->maxWorkItem[0];
        if($n <= $max_work_items) {
            $mode = 1;
        } else {
            $mode = 2;
        }
        if($this->testMode!==null) {
            $mode = $this->testMode;
        }
        //echo "mode=$mode($m,$n,$k)\n";
        switch($mode) {
            case 1:{
                $this->imin1(
                    $n,
                    $R, $offsetR,
                    $X, $offsetX, $incX,
                    $events, $waitEvents
                );
                break;
            }
            case 2:{
                $this->imin2(
                    $n,
                    $R, $offsetR,
                    $X, $offsetX, $incX,
                    $events, $waitEvents
                );
                break;
            }
            default: {
                throw new LogicException('Invalid Mode in imin(): mode='.$mode);
            }
        }
    }

    /**
     * Y := imin( X )
     */
    public function imin1(
        int $n,
        BufferInterface $R, int $offsetR,
        BufferInterface $X, int $offsetX, int $incX,
        object $events=null, object $waitEvents=null
        ) : void
    {
        $dtype = $X->dtype();

        $index_x = 'lid+offset_x';
        $total_local_items = $n;
        $max_work_items = $this->maxWorkItem[0];
        if($total_local_items>$max_work_items) {
            throw new InvalidArgumentException('too large array');
        } else {
            for($max_work_items=1; $max_work_items<$total_local_items;$max_work_items<<=1) {
                ;
            }
        }
        $value_size = $X->value_size();
        $index_value_size = (int)(32/8); // uint32 size
        $type = $this->dtypeToOpenCLType[$dtype];
        $kernel_name = "imin_S_{$type}";
        if(!isset($this->sources[$kernel_name])) {
            $this->sources[$kernel_name] =
                "__kernel void {$kernel_name}(\n".
                "    const        uint total_local_items,\n".
                "        __global {$type} * r,\n".
                "    const        uint offset_r,\n".
                "        __global {$type} * x,\n".
                "    const        uint offset_x,\n".
                "    const        uint incx,\n".
                "         __local {$type} * local_work,\n".
                "         __local uint * local_iwork)\n".
                "{\n".
                    $this->kernelTemplateSiMin(
                        "local_work[lid] = x[{$index_x}];\n".
                        "local_iwork[lid] = lid;",
                        "r[offset_r] = local_iwork[0];\n",
                        $dtype
                    ).
               "}\n";
        }
        $kernel = $this->createKernel($kernel_name);

        $kernel->setArg(0,$total_local_items,NDArray::uint32);
        $kernel->setArg(1,$R);
        $kernel->setArg(2,$offsetR,NDArray::uint32);
        $kernel->setArg(3,$X);
        $kernel->setArg(4,$offsetX,NDArray::uint32);
        $kernel->setArg(5,$incX,NDArray::uint32);
        $kernel->setArg(6,null,$this->adjBoundary($max_work_items*$value_size));
        $kernel->setArg(7,null,$this->adjBoundary($max_work_items*$index_value_size));
        $global_work_size = [$max_work_items];
        $local_work_size = [$max_work_items];
        $kernel->enqueueNDRange($this->queue,$global_work_size,$local_work_size,null,
                $events,$waitEvents);
    }

    /**
     * Y := imin( X )
     */
    public function imin2(
        int $n,
        BufferInterface $R, int $offsetR,
        BufferInterface $X, int $offsetX, int $incX,
        object $events=null, object $waitEvents=null
        ) : void
    {
        $dtype = $X->dtype();

        $index_x = '(seg*lws+lid)+offset_x';
        $total_local_items = $n;
        $max_work_items = $this->maxWorkItem[0];
        if($total_local_items>$max_work_items) {
            $segments = (int)ceil($total_local_items/$max_work_items); // round up float
            $work_items = $max_work_items;
        } else {
            for($max_work_items=1; $max_work_items<$total_local_items;$max_work_items<<=1) {
                ;
            }
            $segments = 1; // round up float
            $work_items = $total_local_items;
        }
        $value_size = $X->value_size();
        $index_value_size = (int)(32/8); // uint32 size
        $type = $this->dtypeToOpenCLType[$dtype];
        $kernel_name = "imin_M_{$type}";
        if(!isset($this->sources[$kernel_name])) {
            $this->sources[$kernel_name] =
                "__kernel void {$kernel_name}(\n".
                "    const        uint total_local_items,\n".
                "    const        uint segments,\n".
                "        __global {$type} * r,\n".
                "    const        uint offset_r,\n".
                "        __global {$type} * x,\n".
                "    const        uint offset_x,\n".
                "    const        uint incx,\n".
                "         __local {$type} * local_work,\n".
                "         __local {$type} * seg_work,\n".
                "         __local uint * local_iwork,\n".
                "         __local uint * seg_iwork,\n".
                "    const        uint work_items)\n".
                "{\n".
                    $this->kernelTemplateQiMin(
                        "local_work[lid] = x[{$index_x}];\n".
                        "local_iwork[lid] = {$index_x};\n",
                        "r[offset_r] = seg_iwork[0];\n",
                        $dtype
                    ).
                    "}\n";
        }
        $kernel = $this->createKernel($kernel_name);

        //$seg_work_bytes = $segments*$value_size;
        //$seg_work_bytes += ($seg_work_bytes%4) ? 4-($seg_work_bytes%4) : 0; // Adjust Addressing Boundary
        $kernel->setArg(0,$total_local_items,NDArray::uint32);
        $kernel->setArg(1,$segments,NDArray::uint32);
        $kernel->setArg(2,$R);
        $kernel->setArg(3,$offsetR,NDArray::uint32);
        $kernel->setArg(4,$X);
        $kernel->setArg(5,$offsetX,NDArray::uint32);
        $kernel->setArg(6,$incX,NDArray::uint32);
        $kernel->setArg(7,null,$this->adjBoundary($max_work_items*$value_size));
        $kernel->setArg(8,null,$this->adjBoundary($segments*$value_size));
        $kernel->setArg(9,null,$this->adjBoundary($max_work_items*$index_value_size));
        $kernel->setArg(10,null,$this->adjBoundary($segments*$index_value_size));
        $kernel->setArg(11,$work_items,NDArray::uint32);
        $global_work_size = [$max_work_items];
        $local_work_size = [$max_work_items];
        $kernel->enqueueNDRange($this->queue,$global_work_size,$local_work_size,null,
                $events,$waitEvents);
    }

    /**
     *     X := a*X + b
     */
    public function increment(
        int $n,
        float $alpha,
        BufferInterface $X, int $offsetX, int $incX,
        float $beta,
        object $events=null, object $waitEvents=null
        ) : void
    {
        $dtypeX = $X->dtype();
        if($dtypeX==NDArray::float64) {
            $this->assertFP64();
        }
        $type = $this->dtypeToOpenCLType[$dtypeX];
        $kernel_name = "increment_{$type}";
        if(!isset($this->sources[$kernel_name])) {
            $this->sources[$kernel_name] =
                "__kernel void {$kernel_name}(\n".
                "    const        {$type} alpha,\n".
                "        __global {$type} * x,\n".
                "    const        uint offset_x,\n".
                "    const        uint incx,\n".
                "    const        {$type} beta)\n".
                "{\n".
                "    uint idx = get_global_id(0)*incx+offset_x;\n".
                "    x[idx] = alpha*x[idx]+beta;\n".
                "}\n";
        }
        $kernel = $this->createKernel($kernel_name);
        $kernel->setArg(0,$alpha,NDArray::float32);
        $kernel->setArg(1,$X);
        $kernel->setArg(2,$offsetX,NDArray::uint32);
        $kernel->setArg(3,$incX,NDArray::uint32);
        $kernel->setArg(4,$beta,NDArray::float32);
        $global_work_size = [$n];
        $kernel->enqueueNDRange($this->queue,$global_work_size,null,null,
            $events,$waitEvents);
    }

    /**
     *     X := 1 / (a*X + b)
     */
    public function reciprocal(
        int $n,
        float $alpha,
        BufferInterface $X, int $offsetX, int $incX,
        float $beta,
        object $events=null, object $waitEvents=null
        ) : void
    {
        $dtypeX = $X->dtype();
        if($dtypeX==NDArray::float64) {
            $this->assertFP64();
        }
        $type = $this->dtypeToOpenCLType[$dtypeX];
        $kernel_name = "reciprocal_{$type}";
        if(!isset($this->sources[$kernel_name])) {
            $this->sources[$kernel_name] =
                "__kernel void {$kernel_name}(\n".
                "    const        {$type} alpha,\n".
                "        __global {$type} * x,\n".
                "    const        uint offset_x,\n".
                "    const        uint incx,\n".
                "    const        {$type} beta)\n".
                "{\n".
                "    uint idx = get_global_id(0)*incx+offset_x;\n".
                "    x[idx] = 1.0/(alpha*x[idx]+beta);\n".
                "}\n";
        }
        $kernel = $this->createKernel($kernel_name);
        $kernel->setArg(0,$alpha,NDArray::float32);
        $kernel->setArg(1,$X);
        $kernel->setArg(2,$offsetX,NDArray::uint32);
        $kernel->setArg(3,$incX,NDArray::uint32);
        $kernel->setArg(4,$beta,NDArray::float32);
        $global_work_size = [$n];
        $kernel->enqueueNDRange($this->queue,$global_work_size,null,null,
            $events,$waitEvents);
    }

    /**
     *     A[m,n] := A[m,n] (A[m,n] >  X[n])
     *     A[m,n] := X[n]   (A[m,n] <= X[n])
     */
    public function maximum(
        int $m,
        int $n,
        BufferInterface $A, int $offsetA, int $ldA,
        BufferInterface $X, int $offsetX, int $incX,
        object $events=null, object $waitEvents=null
        ) : void
    {
        $dtypeX = $X->dtype();
        if($dtypeX==NDArray::float64) {
            $this->assertFP64();
        }
        if($m<=0 || $n<=0) {
            throw new InvalidArgumentException("m and n must be greater than 0");
        }
        if(($m-1)*$ldA+($n-1)+$offsetA>=count($A)) {
            throw new InvalidArgumentException("Matrix A is too small");
        }
        if(($n-1)*$incX+$offsetX>=count($X)) {
            throw new InvalidArgumentException("Buffer X is too small");
        }
        $type = $this->dtypeToOpenCLType[$dtypeX];
        $kernel_name = "maximum_{$type}";
        if(!isset($this->sources[$kernel_name])) {
            $this->sources[$kernel_name] =
                "__kernel void {$kernel_name}(\n".
                "        __global {$type} * a,\n".
                "    const        uint offset_a,\n".
                "    const        uint lda,\n".
                "        __global {$type} * x,\n".
                "    const        uint offset_x,\n".
                "    const        uint incx)\n".
                "{\n".
                "    uint row_id = get_global_id(0);\n".
                "    uint col_id = get_global_id(1);\n".
                "    uint ida = row_id*lda+col_id+offset_a;\n".
                "    uint idx = col_id*incx+offset_x;\n".
                "    {$type} tmp_a = a[ida];\n".
                "    {$type} tmp_x = x[idx];\n".
                "    if(isnan(tmp_x)) {\n".
                "        a[ida] = tmp_x;\n".
                "    } else {\n".
                "        if(tmp_a < tmp_x) {\n".
                "            a[ida] = tmp_x;\n".
                "        }\n".
                "    }\n".
                "}\n";
        }
        $kernel = $this->createKernel($kernel_name);
        $kernel->setArg(0,$A);
        $kernel->setArg(1,$offsetA,NDArray::uint32);
        $kernel->setArg(2,$ldA,NDArray::uint32);
        $kernel->setArg(3,$X);
        $kernel->setArg(4,$offsetX,NDArray::uint32);
        $kernel->setArg(5,$incX,NDArray::uint32);
        $global_work_size = [$m,$n];
        $kernel->enqueueNDRange($this->queue,$global_work_size,null,null,
            $events,$waitEvents);
    }

    /**
     *     A[m,n] := A[m,n] (A[m,n] <  X[n])
     *     A[m,n] := X[n]   (A[m,n] >= X[n])
     */
    public function minimum(
        int $m,
        int $n,
        BufferInterface $A, int $offsetA, int $ldA,
        BufferInterface $X, int $offsetX, int $incX,
        object $events=null, object $waitEvents=null
        ) : void
    {
        $dtypeX = $X->dtype();
        if($dtypeX==NDArray::float64) {
            $this->assertFP64();
        }
        if($m<=0 || $n<=0) {
            throw new InvalidArgumentException("m and n must be greater than 0");
        }
        if(($m-1)*$ldA+($n-1)+$offsetA>=count($A)) {
            throw new InvalidArgumentException("Matrix A is too small");
        }
        if(($n-1)*$incX+$offsetX>=count($X)) {
            throw new InvalidArgumentException("Buffer X is too small");
        }
        $type = $this->dtypeToOpenCLType[$dtypeX];
        $kernel_name = "minimum_{$type}";
        if(!isset($this->sources[$kernel_name])) {
            $this->sources[$kernel_name] =
                "__kernel void {$kernel_name}(\n".
                "        __global {$type} * a,\n".
                "    const        uint offset_a,\n".
                "    const        uint lda,\n".
                "        __global {$type} * x,\n".
                "    const        uint offset_x,\n".
                "    const        uint incx)\n".
                "{\n".
                "    uint row_id = get_global_id(0);\n".
                "    uint col_id = get_global_id(1);\n".
                "    uint ida = row_id*lda+col_id+offset_a;\n".
                "    uint idx = col_id*incx+offset_x;\n".
                "    {$type} tmp_a = a[ida];\n".
                "    {$type} tmp_x = x[idx];\n".
                "    if(isnan(tmp_x)) {\n".
                "        a[ida] = tmp_x;\n".
                "    } else {\n".
                "        if(tmp_a > tmp_x) {\n".
                "            a[ida] = tmp_x;\n".
                "        }\n".
                "    }\n".
                "}\n";
        }
        $kernel = $this->createKernel($kernel_name);
        $kernel->setArg(0,$A);
        $kernel->setArg(1,$offsetA,NDArray::uint32);
        $kernel->setArg(2,$ldA,NDArray::uint32);
        $kernel->setArg(3,$X);
        $kernel->setArg(4,$offsetX,NDArray::uint32);
        $kernel->setArg(5,$incX,NDArray::uint32);
        $global_work_size = [$m,$n];
        $kernel->enqueueNDRange($this->queue,$global_work_size,null,null,
            $events,$waitEvents);
    }

    /**
     *     A[m,n] := 1 (A[m,n] >  X[n])
     *     A[m,n] := 0 (A[m,n] <= X[n])
     */
    public function greater(
        int $m,
        int $n,
        BufferInterface $A, int $offsetA, int $ldA,
        BufferInterface $X, int $offsetX, int $incX,
        object $events=null, object $waitEvents=null
        ) : void
    {
        $dtypeX = $X->dtype();
        if($dtypeX==NDArray::float64) {
            $this->assertFP64();
        }
        if($m<=0 || $n<=0) {
            throw new InvalidArgumentException("m and n must be greater than 0");
        }
        if(($m-1)*$ldA+($n-1)+$offsetA>=count($A)) {
            throw new InvalidArgumentException("Matrix A is too small");
        }
        if(($n-1)*$incX+$offsetX>=count($X)) {
            throw new InvalidArgumentException("Buffer X is too small");
        }
        //$segment_size = 2**9;
        //$segments = (int)ceil($n/$segment_size);
        //$fraction = (int)($n%$segment_size);
        $type = $this->dtypeToOpenCLType[$dtypeX];
        $kernel_name = "greater_{$type}";
        if(!isset($this->sources[$kernel_name])) {
            $this->sources[$kernel_name] =
                "__kernel void {$kernel_name}(\n".
                "        __global {$type} * a,\n".
                "    const        uint offset_a,\n".
                "    const        uint lda,\n".
                "        __global {$type} * x,\n".
                "    const        uint offset_x,\n".
                "    const        uint incx)\n".
                "{\n".
                "    uint row_id = get_global_id(0);\n".
                "    uint col_id = get_global_id(1);\n".
                "    uint ida = row_id*lda+col_id+offset_a;\n".
                "    uint idx = col_id*incx+offset_x;\n".
                "    {$type} value;\n".
                //   if NaN set 0.0
                //   if equal set 0.0
                "    if(a[ida] > x[idx]) {\n".
                "        value = 1.0;\n".
                "    } else {\n".
                "        value = 0.0;\n".
                "    }\n".
                "    a[ida] = value;\n".
                "}\n";
        }
        $kernel = $this->createKernel($kernel_name);
        $kernel->setArg(0,$A);
        $kernel->setArg(1,$offsetA,NDArray::uint32);
        $kernel->setArg(2,$ldA,NDArray::uint32);
        $kernel->setArg(3,$X);
        $kernel->setArg(4,$offsetX,NDArray::uint32);
        $kernel->setArg(5,$incX,NDArray::uint32);
        $global_work_size = [$m,$n];
        $local_work_size = null;
        //$multiple = $this->kernelMultiple($kernel);
        //$global_work_size = [$this->ceil($n,$multiple)];
        //$local_work_size = [$multiple];
        $kernel->enqueueNDRange($this->queue,$global_work_size,$local_work_size,null,
            $events,$waitEvents);
    }

    /**
     *     A[m,n] := 1 (A[m,n] >= X[n])
     *     A[m,n] := 0 (A[m,n] <  X[n])
     */
    public function greaterEqual(
        int $m,
        int $n,
        BufferInterface $A, int $offsetA, int $ldA,
        BufferInterface $X, int $offsetX, int $incX,
        object $events=null, object $waitEvents=null
        ) : void
    {
        $dtypeX = $X->dtype();
        if($dtypeX==NDArray::float64) {
            $this->assertFP64();
        }
        if($m<=0 || $n<=0) {
            throw new InvalidArgumentException("m and n must be greater than 0");
        }
        if(($m-1)*$ldA+($n-1)+$offsetA>=count($A)) {
            throw new InvalidArgumentException("Matrix A is too small");
        }
        if(($n-1)*$incX+$offsetX>=count($X)) {
            throw new InvalidArgumentException("Buffer X is too small");
        }
        //$segment_size = 2**9;
        //$segments = (int)ceil($n/$segment_size);
        //$fraction = (int)($n%$segment_size);
        $type = $this->dtypeToOpenCLType[$dtypeX];
        $kernel_name = "greater_equal_{$type}";
        if(!isset($this->sources[$kernel_name])) {
            $this->sources[$kernel_name] =
                "__kernel void {$kernel_name}(\n".
                "        __global {$type} * a,\n".
                "    const        uint offset_a,\n".
                "    const        uint lda,\n".
                "        __global {$type} * x,\n".
                "    const        uint offset_x,\n".
                "    const        uint incx)\n".
                "{\n".
                "    uint row_id = get_global_id(0);\n".
                "    uint col_id = get_global_id(1);\n".
                "    uint ida = row_id*lda+col_id+offset_a;\n".
                "    uint idx = col_id*incx+offset_x;\n".
                "    {$type} value;\n".
                //   if NaN set 0.0
                //   if equal set 1.0
                "    if(a[ida] >= x[idx]) {\n".
                "        value = 1.0;\n".
                "    } else {\n".
                "        value = 0.0;\n".
                "    }\n".
                "    a[ida] = value;\n".
                "}\n";
        }
        $kernel = $this->createKernel($kernel_name);
        $kernel->setArg(0,$A);
        $kernel->setArg(1,$offsetA,NDArray::uint32);
        $kernel->setArg(2,$ldA,NDArray::uint32);
        $kernel->setArg(3,$X);
        $kernel->setArg(4,$offsetX,NDArray::uint32);
        $kernel->setArg(5,$incX,NDArray::uint32);
        $global_work_size = [$m,$n];
        $local_work_size = null;
        //$multiple = $this->kernelMultiple($kernel);
        //$global_work_size = [$this->ceil($n,$multiple)];
        //$local_work_size = [$multiple];
        $kernel->enqueueNDRange($this->queue,$global_work_size,$local_work_size,null,
            $events,$waitEvents);
    }
    /**
     *     A[m,n] := 1 (A[m,n] <  X[n])
     *     A[m,n] := 0 (A[m,n] >= X[n])
     */
    public function less(
        int $m,
        int $n,
        BufferInterface $A, int $offsetA, int $ldA,
        BufferInterface $X, int $offsetX, int $incX,
        object $events=null, object $waitEvents=null
        ) : void
    {
        $dtypeX = $X->dtype();
        if($dtypeX==NDArray::float64) {
            $this->assertFP64();
        }
        if($m<=0 || $n<=0) {
            throw new InvalidArgumentException("m and n must be greater than 0");
        }
        if(($m-1)*$ldA+($n-1)+$offsetA>=count($A)) {
            throw new InvalidArgumentException("Matrix A is too small");
        }
        if(($n-1)*$incX+$offsetX>=count($X)) {
            throw new InvalidArgumentException("Buffer X is too small");
        }
        $type = $this->dtypeToOpenCLType[$dtypeX];
        $kernel_name = "less_{$type}";
        if(!isset($this->sources[$kernel_name])) {
            $this->sources[$kernel_name] =
                "__kernel void {$kernel_name}(\n".
                "        __global {$type} * a,\n".
                "    const        uint offset_a,\n".
                "    const        uint lda,\n".
                "        __global {$type} * x,\n".
                "    const        uint offset_x,\n".
                "    const        uint incx)\n".
                "{\n".
                "    uint row_id = get_global_id(0);\n".
                "    uint col_id = get_global_id(1);\n".
                "    uint ida = row_id*lda+col_id+offset_a;\n".
                "    uint idx = col_id*incx+offset_x;\n".
                "    {$type} value;\n".
                //   if NaN set 0.0
                //   if equal set 0.0
                "    if(a[ida] < x[idx]) {\n".
                "        value = 1.0;\n".
                "    } else {\n".
                "        value = 0.0;\n".
                "    }\n".
                "    a[ida] = value;\n".
                "}\n";
        }
        $kernel = $this->createKernel($kernel_name);
        $kernel->setArg(0,$A);
        $kernel->setArg(1,$offsetA,NDArray::uint32);
        $kernel->setArg(2,$ldA,NDArray::uint32);
        $kernel->setArg(3,$X);
        $kernel->setArg(4,$offsetX,NDArray::uint32);
        $kernel->setArg(5,$incX,NDArray::uint32);
        $global_work_size = [$m,$n];
        $kernel->enqueueNDRange($this->queue,$global_work_size,null,null,
            $events,$waitEvents);
    }

    /**
     *     A[m,n] := 1 (A[m,n] <= X[n])
     *     A[m,n] := 0 (A[m,n] >  X[n])
     */
    public function lessEqual(
        int $m,
        int $n,
        BufferInterface $A, int $offsetA, int $ldA,
        BufferInterface $X, int $offsetX, int $incX,
        object $events=null, object $waitEvents=null
        ) : void
    {
        $dtypeX = $X->dtype();
        if($dtypeX==NDArray::float64) {
            $this->assertFP64();
        }
        if($m<=0 || $n<=0) {
            throw new InvalidArgumentException("m and n must be greater than 0");
        }
        if(($m-1)*$ldA+($n-1)+$offsetA>=count($A)) {
            throw new InvalidArgumentException("Matrix A is too small");
        }
        if(($n-1)*$incX+$offsetX>=count($X)) {
            throw new InvalidArgumentException("Buffer X is too small");
        }
        $type = $this->dtypeToOpenCLType[$dtypeX];
        $kernel_name = "less_{$type}";
        if(!isset($this->sources[$kernel_name])) {
            $this->sources[$kernel_name] =
                "__kernel void {$kernel_name}(\n".
                "        __global {$type} * a,\n".
                "    const        uint offset_a,\n".
                "    const        uint lda,\n".
                "        __global {$type} * x,\n".
                "    const        uint offset_x,\n".
                "    const        uint incx)\n".
                "{\n".
                "    uint row_id = get_global_id(0);\n".
                "    uint col_id = get_global_id(1);\n".
                "    uint ida = row_id*lda+col_id+offset_a;\n".
                "    uint idx = col_id*incx+offset_x;\n".
                "    {$type} value;\n".
                //   if NaN set 0.0
                //   if equal set 1.0
                "    if(a[ida] <= x[idx]) {\n".
                "        value = 1.0;\n".
                "    } else {\n".
                "        value = 0.0;\n".
                "    }\n".
                "    a[ida] = value;\n".
                "}\n";
        }
        $kernel = $this->createKernel($kernel_name);
        $kernel->setArg(0,$A);
        $kernel->setArg(1,$offsetA,NDArray::uint32);
        $kernel->setArg(2,$ldA,NDArray::uint32);
        $kernel->setArg(3,$X);
        $kernel->setArg(4,$offsetX,NDArray::uint32);
        $kernel->setArg(5,$incX,NDArray::uint32);
        $global_work_size = [$m,$n];
        $kernel->enqueueNDRange($this->queue,$global_work_size,null,null,
            $events,$waitEvents);
    }

    /**
     *    A(m,n) := X(n) * A(m,n)
     */
    public function multiply(
        bool $trans,
        int $m,
        int $n,
        BufferInterface $X, int $offsetX, int $incX,
        BufferInterface $A, int $offsetA, int $ldA,
        object $events=null, object $waitEvents=null
        ) : void
    {
        if($trans) {
            $trans = 'trans';
            $rows = $n;
            $cols = $m;
        } else {
            $trans = 'norm';
            $rows = $m;
            $cols = $n;
        }
        $dtypeX = $X->dtype();
        if($dtypeX==NDArray::float64) {
            $this->assertFP64();
        }
        $type = $this->dtypeToOpenCLType[$dtypeX];
        $kernel_name = "multiply_{$type}_{$trans}";
        if(!isset($this->sources[$kernel_name])) {
            if($trans=='trans') {
                $index_a = 'col_id*lda+row_id+offset_a';
            } else {
                $index_a = 'row_id*lda+col_id+offset_a';
            }
            $this->sources[$kernel_name] =
                "__kernel void {$kernel_name}(\n".
                "    const global {$type} * x,\n".
                "    const        uint offset_x,\n".
                "    const        uint incx,\n".
                "        __global {$type} * a,\n".
                "    const        uint offset_a,\n".
                "    const        uint lda)\n".
                "{\n".
                "    const uint row_id = get_global_id(0);\n".
                "    const uint col_id = get_global_id(1);\n".
                "    const uint index_a = {$index_a};\n".
                "    const uint index_x = col_id*incx+offset_x;\n".
                "    const {$type} work_x = x[index_x];\n".
                "    a[index_a] = a[index_a] * work_x;\n".
                "}\n";
        }
        $kernel = $this->createKernel($kernel_name);
        $kernel->setArg(0,$X);
        $kernel->setArg(1,$offsetX,NDArray::uint32);
        $kernel->setArg(2,$incX,NDArray::uint32);
        $kernel->setArg(3,$A);
        $kernel->setArg(4,$offsetA,NDArray::uint32);
        $kernel->setArg(5,$ldA,NDArray::uint32);
        $global_work_size = [$rows,$cols];
        $kernel->enqueueNDRange($this->queue,$global_work_size,null,null,
            $events,$waitEvents);
    }

    /**
     *    A(m,n) := X(n) + A(m,n)
     */
    public function add(
        bool $trans,
        int $m,
        int $n,
        float $alpha,
        BufferInterface $X, int $offsetX, int $incX,
        BufferInterface $A, int $offsetA, int $ldA,
        object $events=null, object $waitEvents=null
        ) : void
    {
        if($trans) {
            $trans = 'trans';
            $rows = $n;
            $cols = $m;
        } else {
            $trans = 'norm';
            $rows = $m;
            $cols = $n;
        }
        $dtypeX = $X->dtype();
        if($dtypeX==NDArray::float64) {
            $this->assertFP64();
        }
        $type = $this->dtypeToOpenCLType[$dtypeX];
        $kernel_name = "add_{$type}_{$trans}";
        if(!isset($this->sources[$kernel_name])) {
            if($trans=='trans') {
                $index_a = 'col_id*lda+row_id+offset_a';
            } else {
                $index_a = 'row_id*lda+col_id+offset_a';
            }
            $this->sources[$kernel_name] =
                "__kernel void {$kernel_name}(\n".
                "    const        {$type} alpha,\n".
                "    const global {$type} * x,\n".
                "    const        uint offset_x,\n".
                "    const        uint incx,\n".
                "        __global {$type} * a,\n".
                "    const        uint offset_a,\n".
                "    const        uint lda)\n".
                "{\n".
                "    const uint row_id = get_global_id(0);\n".
                "    const uint col_id = get_global_id(1);\n".
                "    const uint index_a = {$index_a};\n".
                "    const uint index_x = col_id*incx+offset_x;\n".
                "    const {$type} work_x = x[index_x];\n".
                "    a[index_a] = a[index_a] + alpha * work_x;\n".
                "}\n";
        }
        $kernel = $this->createKernel($kernel_name);
        $kernel->setArg(0,$alpha,NDArray::float32);
        $kernel->setArg(1,$X);
        $kernel->setArg(2,$offsetX,NDArray::uint32);
        $kernel->setArg(3,$incX,NDArray::uint32);
        $kernel->setArg(4,$A);
        $kernel->setArg(5,$offsetA,NDArray::uint32);
        $kernel->setArg(6,$ldA,NDArray::uint32);
        $global_work_size = [$rows,$cols];
        $kernel->enqueueNDRange($this->queue,$global_work_size,null,null,
            $events,$waitEvents);
    }

    /**
     *     X := X ^ 2
     */

    public function square(
        int $n,
        BufferInterface $X, int $offsetX, int $incX,
        object $events=null, object $waitEvents=null
        ) : void
    {
        $dtypeX = $X->dtype();
        if($dtypeX==NDArray::float64) {
            $this->assertFP64();
        }
        $type = $this->dtypeToOpenCLType[$dtypeX];
        $kernel_name = "square_{$type}";
        if(!isset($this->sources[$kernel_name])) {
            $this->sources[$kernel_name] =
                "__kernel void {$kernel_name}(\n".
                "    const        uint n,\n".
                "        __global {$type} * x,\n".
                "    const        uint offset_x,\n".
                "    const        uint incx)\n".
                "{\n".
                "    uint gid = get_global_id(0);\n".
                //"    if(gid<n) {\n".
                "        uint idx = gid*incx+offset_x;\n".
                "        x[idx] = x[idx] * x[idx];\n".
                //"    }\n".
                "}\n";
        }
        $kernel = $this->createKernel($kernel_name);
        $kernel->setArg(0,$n,NDArray::uint32);
        $kernel->setArg(1,$X);
        $kernel->setArg(2,$offsetX,NDArray::uint32);
        $kernel->setArg(3,$incX,NDArray::uint32);
        $global_work_size = [$n];//[$this->ceil($n,32)];
        $local_work_size = null;//[32];
        $kernel->enqueueNDRange($this->queue,$global_work_size,$local_work_size,null,
            $events,$waitEvents);
    }

    /**
     *     X := sqrt(X)
     */
    public function sqrt(
        int $n,
        BufferInterface $X, int $offsetX, int $incX,
        object $events=null, object $waitEvents=null
        ) : void
    {
        $dtypeX = $X->dtype();
        if($dtypeX==NDArray::float64) {
            $this->assertFP64();
        }
        $type = $this->dtypeToOpenCLType[$dtypeX];
        $kernel_name = "sqrt_{$type}";
        if(!isset($this->sources[$kernel_name])) {
            $this->sources[$kernel_name] =
                "__kernel void {$kernel_name}(\n".
                "        __global {$type} * x,\n".
                "    const        uint offset_x,\n".
                "    const        uint incx)\n".
                "{\n".
                "    uint idx = get_global_id(0)*incx+offset_x;\n".
                "    x[idx] = sqrt(x[idx]);\n".
                "}\n";
        }
        $kernel = $this->createKernel($kernel_name);
        $kernel->setArg(0,$X);
        $kernel->setArg(1,$offsetX,NDArray::uint32);
        $kernel->setArg(2,$incX,NDArray::uint32);
        $global_work_size = [$n];
        $kernel->enqueueNDRange($this->queue,$global_work_size,null,null,
            $events,$waitEvents);
    }

    /**
     *     X := 1 / (a * sqrt(X) + b)
     */
    public function rsqrt(
        int $n,
        float $alpha,
        BufferInterface $X, int $offsetX, int $incX,
        float $beta,
        object $events=null, object $waitEvents=null
        ) : void
    {
        $dtypeX = $X->dtype();
        if($dtypeX==NDArray::float64) {
            $this->assertFP64();
        }
        $type = $this->dtypeToOpenCLType[$dtypeX];
        $kernel_name = "rsqrt_{$type}";
        if(!isset($this->sources[$kernel_name])) {
            $this->sources[$kernel_name] =
                "__kernel void {$kernel_name}(\n".
                "    const        {$type} alpha,\n".
                "        __global {$type} * x,\n".
                "    const        uint offset_x,\n".
                "    const        uint incx,\n".
                "    const        {$type} beta)\n".
                "{\n".
                "    uint idx = get_global_id(0)*incx+offset_x;\n".
                "    x[idx] = 1.0/(alpha * sqrt(x[idx]) + beta);\n".
                "}\n";
        }
        $kernel = $this->createKernel($kernel_name);
        $kernel->setArg(0,$alpha,NDArray::float32);
        $kernel->setArg(1,$X);
        $kernel->setArg(2,$offsetX,NDArray::uint32);
        $kernel->setArg(3,$incX,NDArray::uint32);
        $kernel->setArg(4,$beta,NDArray::float32);
        $global_work_size = [$n];
        $kernel->enqueueNDRange($this->queue,$global_work_size,null,null,
            $events,$waitEvents);
    }

    /**
     *    A(m,n) := A(m,n) ** X(n)
     */
    public function pow(
        bool $trans,
        int $m,
        int $n,
        BufferInterface $A, int $offsetA, int $ldA,
        BufferInterface $X, int $offsetX, int $incX,
        object $events=null, object $waitEvents=null
        ) : void
    {
        if($trans) {
            $trans = 'trans';
            $rows = $n;
            $cols = $m;
        } else {
            $trans = 'norm';
            $rows = $m;
            $cols = $n;
        }
        $dtypeX = $X->dtype();
        if($dtypeX==NDArray::float64) {
            $this->assertFP64();
        }
        $type = $this->dtypeToOpenCLType[$dtypeX];
        $kernel_name = "pow_{$type}_{$trans}";
        if(!isset($this->sources[$kernel_name])) {
            if($trans=='trans') {
                $index_a = 'col_id*lda+row_id+offset_a';
            } else {
                $index_a = 'row_id*lda+col_id+offset_a';
            }
            $this->sources[$kernel_name] =
                "__kernel void {$kernel_name}(\n".
                "    const global {$type} * x,\n".
                "    const        uint offset_x,\n".
                "    const        uint incx,\n".
                "        __global {$type} * a,\n".
                "    const        uint offset_a,\n".
                "    const        uint lda)\n".
                "{\n".
                "    const uint row_id = get_global_id(0);\n".
                "    const uint col_id = get_global_id(1);\n".
                "    const uint index_a = {$index_a};\n".
                "    const uint index_x = col_id*incx+offset_x;\n".
                "    const {$type} work_x = x[index_x];\n".
                "    a[index_a] = pow(a[index_a],work_x);\n".
                "}\n";
        }
        $kernel = $this->createKernel($kernel_name);
        $kernel->setArg(0,$X);
        $kernel->setArg(1,$offsetX,NDArray::uint32);
        $kernel->setArg(2,$incX,NDArray::uint32);
        $kernel->setArg(3,$A);
        $kernel->setArg(4,$offsetA,NDArray::uint32);
        $kernel->setArg(5,$ldA,NDArray::uint32);
        $global_work_size = [$rows,$cols];
        $kernel->enqueueNDRange($this->queue,$global_work_size,null,null,
            $events,$waitEvents);
    }


    /**
     *     X(i) := e ^ X(i)
     */
    public function exp(
        int $n,
        BufferInterface $X, int $offsetX, int $incX,
        object $events=null, object $waitEvents=null
        ) : void
    {
        $dtypeX = $X->dtype();
        if($dtypeX==NDArray::float64) {
            $this->assertFP64();
        }
        $type = $this->dtypeToOpenCLType[$dtypeX];
        $kernel_name = "exp_{$type}";
        if(!isset($this->sources[$kernel_name])) {
            $this->sources[$kernel_name] =
                "__kernel void {$kernel_name}(\n".
                "        __global {$type} * x,\n".
                "    const        uint offset_x,\n".
                "    const        uint incx)\n".
                "{\n".
                "    uint idx = get_global_id(0)*incx+offset_x;\n".
                "    x[idx] = exp(x[idx]);\n".
                "}\n";
        }
        $kernel = $this->createKernel($kernel_name);
        $kernel->setArg(0,$X);
        $kernel->setArg(1,$offsetX,NDArray::uint32);
        $kernel->setArg(2,$incX,NDArray::uint32);
        $global_work_size = [$n];
        $kernel->enqueueNDRange($this->queue,$global_work_size,null,null,
            $events,$waitEvents);
    }

    /**
     *     X := log(X)
     */
    public function log(
        int $n,
        BufferInterface $X, int $offsetX, int $incX,
        object $events=null, object $waitEvents=null
        ) : void
    {
        $dtypeX = $X->dtype();
        if($dtypeX==NDArray::float64) {
            $this->assertFP64();
        }
        $type = $this->dtypeToOpenCLType[$dtypeX];
        $kernel_name = "log_{$type}";
        if(!isset($this->sources[$kernel_name])) {
            $this->sources[$kernel_name] =
                "__kernel void {$kernel_name}(\n".
                "        __global {$type} * x,\n".
                "    const        uint offset_x,\n".
                "    const        uint incx)\n".
                "{\n".
                "    uint idx = get_global_id(0)*incx+offset_x;\n".
                "    x[idx] = log(x[idx]);\n".
                "}\n";
        }
        $kernel = $this->createKernel($kernel_name);
        $kernel->setArg(0,$X);
        $kernel->setArg(1,$offsetX,NDArray::uint32);
        $kernel->setArg(2,$incX,NDArray::uint32);
        $global_work_size = [$n];
        $kernel->enqueueNDRange($this->queue,$global_work_size,null,null,
            $events,$waitEvents);
    }

    /**
     *     X := tanh(X)
     */
    public function tanh(
        int $n,
        BufferInterface $X, int $offsetX, int $incX,
        object $events=null, object $waitEvents=null
        ) : void
    {
        $dtypeX = $X->dtype();
        if($dtypeX==NDArray::float64) {
            $this->assertFP64();
        }
        $type = $this->dtypeToOpenCLType[$dtypeX];
        $kernel_name = "tanh_{$type}";
        if(!isset($this->sources[$kernel_name])) {
            $this->sources[$kernel_name] =
                "__kernel void {$kernel_name}(\n".
                "        __global {$type} * x,\n".
                "    const        uint offset_x,\n".
                "    const        uint incx)\n".
                "{\n".
                "    uint idx = get_global_id(0)*incx+offset_x;\n".
                "    x[idx] = tanh(x[idx]);\n".
                "}\n";
        }
        $kernel = $this->createKernel($kernel_name);
        $kernel->setArg(0,$X);
        $kernel->setArg(1,$offsetX,NDArray::uint32);
        $kernel->setArg(2,$incX,NDArray::uint32);
        $global_work_size = [$n];
        $kernel->enqueueNDRange($this->queue,$global_work_size,null,null,
            $events,$waitEvents);
    }

    /**
     *     X := sin(X)
     */
    public function sin(
        int $n,
        BufferInterface $X, int $offsetX, int $incX,
        object $events=null, object $waitEvents=null
        ) : void
    {
        $dtypeX = $X->dtype();
        if($dtypeX==NDArray::float64) {
            $this->assertFP64();
        }
        $type = $this->dtypeToOpenCLType[$dtypeX];
        $kernel_name = "sin_{$type}";
        if(!isset($this->sources[$kernel_name])) {
            $this->sources[$kernel_name] =
                "__kernel void {$kernel_name}(\n".
                "        __global {$type} * x,\n".
                "    const        uint offset_x,\n".
                "    const        uint incx)\n".
                "{\n".
                "    uint idx = get_global_id(0)*incx+offset_x;\n".
                "    x[idx] = sin(x[idx]);\n".
                "}\n";
        }
        $kernel = $this->createKernel($kernel_name);
        $kernel->setArg(0,$X);
        $kernel->setArg(1,$offsetX,NDArray::uint32);
        $kernel->setArg(2,$incX,NDArray::uint32);
        $global_work_size = [$n];
        $kernel->enqueueNDRange($this->queue,$global_work_size,null,null,
            $events,$waitEvents);
    }

    /**
     *     X := cos(X)
     */
    public function cos(
        int $n,
        BufferInterface $X, int $offsetX, int $incX,
        object $events=null, object $waitEvents=null
        ) : void
    {
        $dtypeX = $X->dtype();
        if($dtypeX==NDArray::float64) {
            $this->assertFP64();
        }
        $type = $this->dtypeToOpenCLType[$dtypeX];
        $kernel_name = "cos_{$type}";
        if(!isset($this->sources[$kernel_name])) {
            $this->sources[$kernel_name] =
                "__kernel void {$kernel_name}(\n".
                "        __global {$type} * x,\n".
                "    const        uint offset_x,\n".
                "    const        uint incx)\n".
                "{\n".
                "    uint idx = get_global_id(0)*incx+offset_x;\n".
                "    x[idx] = cos(x[idx]);\n".
                "}\n";
        }
        $kernel = $this->createKernel($kernel_name);
        $kernel->setArg(0,$X);
        $kernel->setArg(1,$offsetX,NDArray::uint32);
        $kernel->setArg(2,$incX,NDArray::uint32);
        $global_work_size = [$n];
        $kernel->enqueueNDRange($this->queue,$global_work_size,null,null,
            $events,$waitEvents);
    }

    /**
     *     X := tan(X)
     */
    public function tan(
        int $n,
        BufferInterface $X, int $offsetX, int $incX,
        object $events=null, object $waitEvents=null
        ) : void
    {
        $dtypeX = $X->dtype();
        if($dtypeX==NDArray::float64) {
            $this->assertFP64();
        }
        $type = $this->dtypeToOpenCLType[$dtypeX];
        $kernel_name = "tan_{$type}";
        if(!isset($this->sources[$kernel_name])) {
            $this->sources[$kernel_name] =
                "__kernel void {$kernel_name}(\n".
                "        __global {$type} * x,\n".
                "    const        uint offset_x,\n".
                "    const        uint incx)\n".
                "{\n".
                "    uint idx = get_global_id(0)*incx+offset_x;\n".
                "    x[idx] = tan(x[idx]);\n".
                "}\n";
        }
        $kernel = $this->createKernel($kernel_name);
        $kernel->setArg(0,$X);
        $kernel->setArg(1,$offsetX,NDArray::uint32);
        $kernel->setArg(2,$incX,NDArray::uint32);
        $global_work_size = [$n];
        $kernel->enqueueNDRange($this->queue,$global_work_size,null,null,
            $events,$waitEvents);
    }

    /**
     * Y(i) := 1  ( X(i) == Y(i) )
     * Y(i) := 0  ( X(i) != Y(i) )
     */
    public function equal(
        int $n,
        BufferInterface $X, int $offsetX, int $incX,
        BufferInterface $Y, int $offsetY, int $incY,
        object $events=null, object $waitEvents=null
        ) : void
    {
        $dtypeX = $X->dtype();
        $dtypeY = $Y->dtype();
        if($dtypeX != $dtypeY) {
            throw new InvalidArgumentException('X and Y must be same data type.');
        }
        if($dtypeX==NDArray::float64 || $dtypeY==NDArray::float64) {
            $this->assertFP64();
        }
        $dtype = $this->dtypeToOpenCLType[$dtypeX];
        $kernel_name = "equal_{$dtype}";
        if(!isset($this->sources[$kernel_name])) {
            $this->sources[$kernel_name] =
                "__kernel void {$kernel_name}(\n".
                "    const global {$dtype} * x,\n".
                "    const        uint offset_x,\n".
                "    const        uint incx,\n".
                "        __global {$dtype} * y,\n".
                "    const        uint offset_y,\n".
                "    const        uint incy)\n".
                "{\n".
                "    uint gid = get_global_id(0);\n".
                "    uint index_x = gid*incx+offset_x;\n".
                "    uint index_y = gid*incy+offset_y;\n".
                "    if(x[index_x]==y[index_y]) {\n".
                "        y[index_y] = 1;\n".
                "    } else {\n".
                "        y[index_y] = 0;\n".
                "    }\n".
                "}\n";
        }
        $kernel = $this->createKernel($kernel_name);

        $kernel->setArg(0,$X);
        $kernel->setArg(1,$offsetX,NDArray::uint32);
        $kernel->setArg(2,$incX,NDArray::uint32);
        $kernel->setArg(3,$Y);
        $kernel->setArg(4,$offsetY,NDArray::uint32);
        $kernel->setArg(5,$incY,NDArray::uint32);
        $global_work_size = [$n];
        $kernel->enqueueNDRange($this->queue,$global_work_size,null,null,
            $events,$waitEvents);
    }

    /**
     * Y(i) := 1  ( X(i) != Y(i) )
     * Y(i) := 0  ( X(i) == Y(i) )
     */
    public function notEqual(
        int $n,
        BufferInterface $X, int $offsetX, int $incX,
        BufferInterface $Y, int $offsetY, int $incY,
        object $events=null, object $waitEvents=null
        ) : void
    {
        $dtypeX = $X->dtype();
        $dtypeY = $Y->dtype();
        if($dtypeX != $dtypeY) {
            throw new InvalidArgumentException('X and Y must be same data type.');
        }
        if($dtypeX==NDArray::float64 || $dtypeY==NDArray::float64) {
            $this->assertFP64();
        }
        $dtype = $this->dtypeToOpenCLType[$dtypeX];
        $kernel_name = "notEqual_{$dtype}";
        if(!isset($this->sources[$kernel_name])) {
            $this->sources[$kernel_name] =
                "__kernel void {$kernel_name}(\n".
                "    const global {$dtype} * x,\n".
                "    const        uint offset_x,\n".
                "    const        uint incx,\n".
                "        __global {$dtype} * y,\n".
                "    const        uint offset_y,\n".
                "    const        uint incy)\n".
                "{\n".
                "    uint gid = get_global_id(0);\n".
                "    uint index_x = gid*incx+offset_x;\n".
                "    uint index_y = gid*incy+offset_y;\n".
                "    if(x[index_x]!=y[index_y]) {\n".
                "        y[index_y] = 1;\n".
                "    } else {\n".
                "        y[index_y] = 0;\n".
                "    }\n".
                "}\n";
        }
        $kernel = $this->createKernel($kernel_name);

        $kernel->setArg(0,$X);
        $kernel->setArg(1,$offsetX,NDArray::uint32);
        $kernel->setArg(2,$incX,NDArray::uint32);
        $kernel->setArg(3,$Y);
        $kernel->setArg(4,$offsetY,NDArray::uint32);
        $kernel->setArg(5,$incY,NDArray::uint32);
        $global_work_size = [$n];
        $kernel->enqueueNDRange($this->queue,$global_work_size,null,null,
            $events,$waitEvents);
    }

    /**
     * X(i) := 1  ( X(i) != 0 )
     * X(i) := 0  ( X(i) == 0 )
     */
    public function not(
        int $n,
        BufferInterface $X, int $offsetX, int $incX,
        object $events=null, object $waitEvents=null
        ) : void
    {
        $dtypeX = $X->dtype();
        if($dtypeX==NDArray::float64) {
            $this->assertFP64();
        }
        $dtype = $this->dtypeToOpenCLType[$dtypeX];
        $kernel_name = "not_{$dtype}";
        if(!isset($this->sources[$kernel_name])) {
            $this->sources[$kernel_name] =
                "__kernel void {$kernel_name}(\n".
                "        __global {$dtype} * x,\n".
                "    const        uint offset_x,\n".
                "    const        uint incx)\n".
                "{\n".
                "    uint gid = get_global_id(0);\n".
                "    uint index_x = gid*incx+offset_x;\n".
                "    if(x[index_x]==0) {\n".
                "        x[index_x] = 1;\n".
                "    } else {\n".
                "        x[index_x] = 0;\n".
                "    }\n".
                "}\n";
        }
        $kernel = $this->createKernel($kernel_name);

        $kernel->setArg(0,$X);
        $kernel->setArg(1,$offsetX,NDArray::uint32);
        $kernel->setArg(2,$incX,NDArray::uint32);
        $global_work_size = [$n];
        $kernel->enqueueNDRange($this->queue,$global_work_size,null,null,
            $events,$waitEvents);
    }

    protected function isComplex(int $dtypeX) : bool
    {
        return ($dtypeX==NDArray::complex64||$dtypeX==NDArray::complex128);
    }

    /**
     * X(i) := abs(X(i))
     */
    public function abs(
        int $n,
        BufferInterface $X, int $offsetX, int $incX,
        object $events=null, object $waitEvents=null
        ) : void
    {
        $dtypeX = $X->dtype();
        if($dtypeX==NDArray::float64||$dtypeX==NDArray::complex128) {
            $this->assertFP64();
        }
        $isInt = array_key_exists($dtypeX,$this->intTypes);
        $abs = $isInt ? 'abs' : 'fabs';
        $isComplex = $this->isComplex($dtypeX);
        if($isComplex) {
            $dtype = ($dtypeX==NDArray::complex64)?'float':'double';
            $statement  = "    x[index_x] = sqrt(x[index_x]*x[index_x] + x[index_x+1]*x[index_x+1]);\n";
            $statement .= "    x[index_x+1] = 0;\n";
            $incX *= 2;
            $offsetX *= 2;
        } else {
            $dtype = $this->dtypeToOpenCLType[$dtypeX];
            $statement = "    x[index_x] = {$abs}(x[index_x]);\n";
        }


        $kernel_name = "abs_{$dtype}";
        if(!isset($this->sources[$kernel_name])) {
            $this->sources[$kernel_name] =
                "__kernel void {$kernel_name}(\n".
                "        __global {$dtype} * x,\n".
                "    const        uint offset_x,\n".
                "    const        uint incx)\n".
                "{\n".
                "    uint gid = get_global_id(0);\n".
                "    uint index_x = gid*incx+offset_x;\n".
                $statement.
                "}\n";
        }
        $kernel = $this->createKernel($kernel_name);

        $kernel->setArg(0,$X);
        $kernel->setArg(1,$offsetX,NDArray::uint32);
        $kernel->setArg(2,$incX,NDArray::uint32);
        $global_work_size = [$n];
        $kernel->enqueueNDRange($this->queue,$global_work_size,null,null,
            $events,$waitEvents);
    }

    /**
     *     A(m,n) := X(n)
     */
    public function duplicate(
        bool $trans,
        int $m,
        int $n,
        BufferInterface $X, int $offsetX, int $incX,
        BufferInterface $A, int $offsetA, int $ldA,
        object $events=null, object $waitEvents=null
        ) : void
    {
        if($trans) {
            $trans = 'trans';
        } else {
            $trans = 'norm';
        }
        $dtype = $X->dtype();
        if($dtype==NDArray::float64) {
            $this->assertFP64();
        }
        $type = $this->dtypeToOpenCLType[$dtype];
        $kernel_name = "duplicate_{$type}_{$trans}";
        if(!isset($this->sources[$kernel_name])) {
            if($trans=='trans') {
                $idxI = '1';
                $idxJ = '0';
                $index_a = 'j*lda+i+offset_a';
            } else {
                $idxI = '0';
                $idxJ = '1';
                $index_a = 'i*lda+j+offset_a';
            }
            $this->sources[$kernel_name] =
                "__kernel void {$kernel_name}(\n".
                "    const global {$type} * x,\n".
                "    const        uint offset_x,\n".
                "    const        uint incx,\n".
                "        __global {$type} * a,\n".
                "    const        uint offset_a,\n".
                "    const        uint lda)\n".
                "{\n".
                "    uint i = get_global_id({$idxI});\n".
                "    uint j = get_global_id({$idxJ});\n".
                "    uint index_a = {$index_a};\n".
                "    uint index_x = j*incx+offset_x;\n".
                "    a[index_a] = x[index_x];\n".
                "}\n";
        }
        $kernel = $this->createKernel($kernel_name);

        $kernel->setArg(0,$X);
        $kernel->setArg(1,$offsetX,NDArray::uint32);
        $kernel->setArg(2,$incX,NDArray::uint32);
        $kernel->setArg(3,$A);
        $kernel->setArg(4,$offsetA,NDArray::uint32);
        $kernel->setArg(5,$ldA,NDArray::uint32);
        $global_work_size = [$m,$n];
        $kernel->enqueueNDRange($this->queue,$global_work_size,null,null,
            $events,$waitEvents);
    }

    public function astype(
        int $n,
        int $dtype,
        BufferInterface $X, int $offsetX, int $incX,
        BufferInterface $Y, int $offsetY, int $incY,
        object $events=null, object $waitEvents=null
        ) : void
    {
        $dtypeX = $X->dtype();
        $dtypeY = $Y->dtype();
        if($dtypeX==NDArray::float64 || $dtypeY==NDArray::float64) {
            $this->assertFP64();
        }
        if($dtype!=$dtypeY) {
            throw new InvalidArgumentException('unmatch data type between dtype and output buffer type');
        }
        $from = $this->dtypeToOpenCLType[$dtypeX];
        $to = $this->dtypeToOpenCLType[$dtypeY];
        $toOrg = $to;
        if($dtypeY==NDArray::bool) {
            $toOrg = 'bool';
        }
        $kernel_name = "astype_{$from}_{$toOrg}";
        if(!isset($this->sources[$kernel_name])) {
            $this->sources[$kernel_name] =
                "__kernel void {$kernel_name}(\n".
                "    const global {$from} * x,\n".
                "    const        uint offset_x,\n".
                "    const        uint incx,\n".
                "        __global {$to} * y,\n".
                "    const        uint offset_y,\n".
                "    const        uint incy)\n".
                "{\n".
                "    uint gid = get_global_id(0);\n";
            if($dtypeY==NDArray::bool) {
                if($dtypeX==NDArray::float16||$dtypeX==NDArray::float32||$dtypeX==NDArray::float64) {
                $this->sources[$kernel_name] .=
                "    int tmp = x[gid*incx+offset_x];\n".
                "    if(tmp==0) {\n".
                "        y[gid*incy+offset_y] = 0;\n".
                "    } else {\n".
                "        y[gid*incy+offset_y] = 1;\n".
                "    }\n";
                } else {
                $this->sources[$kernel_name] .=
                "    if(x[gid*incx+offset_x]==0) {\n".
                "        y[gid*incy+offset_y] = 0;\n".
                "    } else {\n".
                "        y[gid*incy+offset_y] = 1;\n".
                "    }\n";
                }
            } else {
                $this->sources[$kernel_name] .=
                "    y[gid*incy+offset_y] = x[gid*incx+offset_x];\n";
            }
            $this->sources[$kernel_name] .=
                "}\n";
        }
        $kernel = $this->createKernel($kernel_name);

        $kernel->setArg(0,$X);
        $kernel->setArg(1,$offsetX,NDArray::uint32);
        $kernel->setArg(2,$incX,NDArray::uint32);
        $kernel->setArg(3,$Y);
        $kernel->setArg(4,$offsetY,NDArray::uint32);
        $kernel->setArg(5,$incY,NDArray::uint32);
        $global_work_size = [$n];
        $kernel->enqueueNDRange($this->queue,$global_work_size,null,null,
            $events,$waitEvents);
    }

    /**
    *      B(n,k) := A(X(n),k)
    */
    public function gather(
        bool $reverse,
        bool $addMode,
        int $n,
        int $k,
        int $numClass,
        BufferInterface $X, int $offsetX,
        BufferInterface $A, int $offsetA,
        BufferInterface $B, int $offsetB,
        object $events=null, object $waitEvents=null
        ) : void
    {
        if($reverse==true && $addMode==true) {
            $this->scatterAdd(
                $n,
                $k,
                $numClass,
                $X, $offsetX,
                $A, $offsetA,
                $B, $offsetB,
                $events, $waitEvents
            );
            return;
        }
        $dtype = $A->dtype();
        if($X->dtype()!=NDArray::int32 && $X->dtype()!=NDArray::uint32) {
            throw new InvalidArgumentException("X must be 32bit integer:".
                                            $this->dtypeToString($X->dtype()));
        }
        if($dtype!=$B->dtype()) {
            throw new InvalidArgumentException("Unmatch data type A and B:".
            $this->dtypeToString($dtype).",".$this->dtypeToString($B->dtype()));
        }
        if($dtype==NDArray::float64) {
            $this->assertFP64();
        }

        if($addMode) {
            $op = 'add';
        } else {
            $op = 'set';
        }
        if($reverse) {
            $direction = 'r';
        } else {
            $direction = 'f';
        }
        $type = $this->dtypeToOpenCLType[$dtype];
        $kernel_name = "gather_{$op}_{$type}_{$direction}";
        if(!isset($this->sources[$kernel_name])) {
            $a_variable = "a[offset_a+label*k+p]";
            $b_variable = "b[offset_b+j*k+p]";
            if($reverse) {
                $from = $b_variable;
                $to = $a_variable;
                $b_arg_type = 'const global';
                $a_arg_type = '__global';
            } else {
                $from = $a_variable;
                $to = $b_variable;
                $a_arg_type = 'const global';
                $b_arg_type = '__global';
            }
            if($addMode) {
                $operator = '+=';
            } else {
                $operator = '=';
            }
            $this->sources[$kernel_name] =
                "__kernel void {$kernel_name}(\n".
                "    const        uint n,\n".
                "    const        uint k,\n".
                "    const        uint numClass,\n".
                "    const global uint * x,\n".
                "    const        uint offset_x,\n".
                "    {$a_arg_type} {$type} * a,\n".
                "    const        uint offset_a,\n".
                "    {$b_arg_type} {$type} * b,\n".
                "    const        uint offset_b)\n".
                "{\n".
                "    uint p = get_global_id(0);\n".
                "    uint j = get_global_id(1);\n".
                "    uint label = x[j+offset_x];\n".
                "    if(label<numClass) {\n".
                "        {$to} {$operator} {$from};\n".
                "    }\n".
                "}\n";
        }
        $kernel = $this->createKernel($kernel_name);
        $kernel->setArg(0,$n,NDArray::uint32);
        $kernel->setArg(1,$k,NDArray::uint32);
        $kernel->setArg(2,$numClass,NDArray::uint32);
        $kernel->setArg(3,$X);
        $kernel->setArg(4,$offsetX,NDArray::uint32);
        $kernel->setArg(5,$A);
        $kernel->setArg(6,$offsetA,NDArray::uint32);
        $kernel->setArg(7,$B);
        $kernel->setArg(8,$offsetB,NDArray::uint32);
        $global_work_size = [$k,$n];
        $kernel->enqueueNDRange($this->queue,$global_work_size,null,null,
            $events,$waitEvents);
    }

    /**
    *      B(n,k) := A(X(n),k)
    */
    public function reduceGather(
        bool $reverse,
        bool $addMode,
        int $m,
        int $n,
        int $numClass,
        BufferInterface $X, int $offsetX,
        BufferInterface $A, int $offsetA,
        BufferInterface $B, int $offsetB,
        object $events=null, object $waitEvents=null
        ) : void
    {
        $dtype = $A->dtype();
        if($X->dtype()!=NDArray::int32 && $X->dtype()!=NDArray::uint32) {
            throw new InvalidArgumentException("X must be 32bit integer:".
                                            $this->dtypeToString($X->dtype()));
        }
        if($dtype!=$B->dtype()) {
            throw new InvalidArgumentException("Unmatch data type A and B:".
            $this->dtypeToString($dtype).",".$this->dtypeToString($B->dtype()));
        }
        if($dtype==NDArray::float64) {
            $this->assertFP64();
        }

        if($addMode) {
            $op = 'add';
        } else {
            $op = 'set';
        }
        if($reverse) {
            $direction = 'r';
        } else {
            $direction = 'f';
        }
        $type = $this->dtypeToOpenCLType[$dtype];
        $kernel_name = "reduceGather_{$op}_{$type}_{$direction}";
        if(!isset($this->sources[$kernel_name])) {
            $a_variable = "a[offset_a+i*n*numClass+j+label*n]";
            $b_variable = "b[offset_b+i*n+j]";
            if($reverse) {
                $from = $b_variable;
                $to = $a_variable;
                $b_arg_type = 'const global';
                $a_arg_type = '__global';
            } else {
                $from = $a_variable;
                $to = $b_variable;
                $a_arg_type = 'const global';
                $b_arg_type = '__global';
            }
            if($addMode) {
                $operator = '+=';
            } else {
                $operator = '=';
            }
            $this->sources[$kernel_name] =
                "__kernel void {$kernel_name}(\n".
                "    const        uint m,\n".
                "    const        uint n,\n".
                "    const        uint numClass,\n".
                "    const global uint * x,\n".
                "    const        uint offset_x,\n".
                "    {$a_arg_type} {$type} * a,\n".
                "    const        uint offset_a,\n".
                "    {$b_arg_type} {$type} * b,\n".
                "    const        uint offset_b)\n".
                "{\n".
                "    uint j = get_global_id(0);\n".
                "    uint i = get_global_id(1);\n".
                "    uint label = x[j+offset_x+n*i];\n".
                "    if(label<numClass) {\n".
                "        {$to} {$operator} {$from};\n".
                "    }\n".
                "}\n";
        }
        $kernel = $this->createKernel($kernel_name);
        $kernel->setArg(0,$m,NDArray::uint32);
        $kernel->setArg(1,$n,NDArray::uint32);
        $kernel->setArg(2,$numClass,NDArray::uint32);
        $kernel->setArg(3,$X);
        $kernel->setArg(4,$offsetX,NDArray::uint32);
        $kernel->setArg(5,$A);
        $kernel->setArg(6,$offsetA,NDArray::uint32);
        $kernel->setArg(7,$B);
        $kernel->setArg(8,$offsetB,NDArray::uint32);
        $global_work_size = [$n,$m];
        $kernel->enqueueNDRange($this->queue,$global_work_size,null,null,
            $events,$waitEvents);
    }

    protected function getHomeDirectory() : string
    {
        $path = '';
        if(PHP_OS=='WINNT') {
            $path = getenv('USERPROFILE');
        } elseif(PHP_OS=='Linux') {
            $path = getenv('HOME');
        }
        if(!is_string($path)) {
            $path = '';
        }
        return $path;
    }

    /**
     * @return array<mixed>
     */
    protected function loadParameter(string $filename) : array
    {
        $filepath = $this->getHomeDirectory().'/.rindow/'.$filename;
        if(!file_exists($filepath)) {
            $filepath = __DIR__.'/params/'.$filename;
        }
        $times = include $filepath;
        return $times;
    }

    public function predictTimeScatterAdd(int $mode,int $numClass,int $cols,int $rows) : int
    {
        if(isset($this->timesPredictionScatterAdd[$mode])) {
            $times = $this->timesPredictionScatterAdd[$mode];
        } else {
            $times = $this->loadParameter('ScatterAddTimesMode'.$mode.'.php');
            $this->timesPredictionScatterAdd[$mode] = $times;
        }
        if($mode==4) {
            $numClass = 8;
        } elseif($mode==3 && $rows>256) {
            $rows = 256;
        }
        if(isset($times[$rows][$cols][$numClass])) {
            $time = $times[$rows][$cols][$numClass];
            if($time==0)
                return PHP_INT_MAX;
            return $time;
        } else {
            return PHP_INT_MAX;
        }
    }
    /**
     * A(X[k],n) := B[k,n] ,  m > max(X[k])
     */
    public function scatterAdd(
        int $n,
        int $k,
        int $numClass,
        BufferInterface $X, int $offsetX,
        BufferInterface $A, int $offsetA,
        BufferInterface $B, int $offsetB,
        object $events=null, object $waitEvents=null
        ) : void
    {
        if($X->dtype()!=NDArray::int32 && $X->dtype()!=NDArray::uint32) {
            throw new InvalidArgumentException("X must be 32bit integer:".
                                            $this->dtypeToString($X->dtype()));
        }
        if($A->dtype()!=$B->dtype()) {
            throw new InvalidArgumentException("Unmatch data type A and B:".
            $this->dtypeToString($A->dtype()).",".$this->dtypeToString($B->dtype()));
        }
        $dtype = $A->dtype();
        if($dtype==NDArray::float64) {
            $this->assertFP64();
        }
        //if($dtype==NDArray::bool||$dtype==NDArray::int8||$dtype==NDArray::uint8||$dtype==NDArray::int16||$dtype==NDArray::uint16) {
        //    throw new LogicException('Not supported data tape dtype='.$this->dtypeToString($dtype));
        //}

        $small = $max_work_items = $this->maxWorkItem[0];
        $mediam = $max_work_items*$max_work_items*2;
        // m(rows) => numClass
        // n(cols) => k
        // k       => n
        for($bn=8; $bn<$n;$bn<<=1) { ; }
        for($bk=8; $bk<$n;$bk<<=1) { ; }
        for($bclass=8; $bclass<$numClass;$bclass<<=1) { ; }
        //echo "($m,$n,$k)\n";
        $mode0 = $this->predictTimeScatterAdd(0,$bclass,$bk,$bn);
        //$mode1 = $this->predictTimeScatterAdd(1,$bm,$bn,$bk);
        $mode2 = $this->predictTimeScatterAdd(2,$bclass,$bk,$bn);
        $mode3 = $this->predictTimeScatterAdd(3,$bclass,$bk,$bn);
        $mode4 = $this->predictTimeScatterAdd(4,$bclass,$bk,$bn);
        //echo 'mode0='.number_format($mode0)."\n";
        ////echo 'mode1='.number_format($mode1)."\n";
        //echo 'mode2='.number_format($mode2)."\n";
        //echo 'mode3='.number_format($mode3)."\n";
        //echo 'mode4='.number_format($mode4)."\n";
        $imin1 = ($mode0 < $mode2) ? 0 : 2;
        $min1 = ($mode0 < $mode2) ? $mode0 : $mode2;
        $imin2 = ($mode3 < $mode4) ? 3 : 4;
        $min2 = ($mode3 < $mode4) ? $mode3 : $mode4;
        $mode = ($min1 < $min2) ? $imin1 : $imin2;
        //$min = ($min1 < $min2) ? $min1 : $min2;
        if($mode==2 && $bk<=$small) {
            $mode=1;
        }
        if($this->testMode!==null) {
            $mode = $this->testMode;
        }
        //echo "mode=$mode($m,$n,$k)\n";
        switch($mode) {
            case 0:{
                $this->scatterAdd_0(
                    $n,$k,$numClass,
                    $X, $offsetX,
                    $A, $offsetA,
                    $B, $offsetB,
                    $events, $waitEvents
                );
                break;
            }
            case 1:{
                $this->scatterAdd_1(
                    $n,$k,$numClass,
                    $X, $offsetX,
                    $A, $offsetA,
                    $B, $offsetB,
                    $events, $waitEvents
                );
                break;
            }
            case 2:{
                $this->scatterAdd_2(
                    $n,$k,$numClass,
                    $X, $offsetX,
                    $A, $offsetA,
                    $B, $offsetB,
                    $events, $waitEvents
                );
                break;
            }
            case 3:{
                $this->scatterAdd_3(
                    $n,$k,$numClass,
                    $X, $offsetX,
                    $A, $offsetA,
                    $B, $offsetB,
                    $events, $waitEvents
                );
                break;
            }
            case 4:{
                $this->scatterAdd_4(
                    $n,$k,$numClass,
                    $X, $offsetX,
                    $A, $offsetA,
                    $B, $offsetB,
                    $events, $waitEvents
                );
                break;
            }
        }
    }

    /**
    *      B(m,X(m,n),k) += A(m,n,k)
    */
    public function scatterAdd_0(
        int $n,
        int $k,
        int $numClass,
        BufferInterface $X, int $offsetX,
        BufferInterface $A, int $offsetA,
        BufferInterface $B, int $offsetB,
        object $events=null, object $waitEvents=null
        ) : void
    {
//echo "mode=0\n";
        $dtype = $A->dtype();
        $total_local_items = $n;
        $type = $this->dtypeToOpenCLType[$dtype];
        $kernel_name = "scatterAdd_0_{$type}";
        if(!isset($this->sources[$kernel_name])) {
            $this->sources[$kernel_name] =
                "__kernel void {$kernel_name}(\n".
                "    const        uint total_local_items,\n".
                "    const        uint k,\n".
                "    const        uint numclass,\n".
                "    const global uint * x,\n".
                "    const        uint offset_x,\n".
                "        __global {$type} * a,\n".
                "    const        uint offset_a,\n".
                "    const global {$type} * b,\n".
                "    const        uint offset_b)\n".
                "{\n".
                "    const uint p = get_global_id(0);\n".  // A col id(p)
                "    const uint i = get_global_id(1);\n".  // A row id(i)
                "    uint pos_x = offset_x;\n".  // A col id
                "    uint pos_b = offset_b+p;\n".  // A col id
                "    {$type} sum = 0;\n".
                //"    if(p<k && i<numclass) {\n".
                "        for(int j=0;j<total_local_items;j++,pos_x++,pos_b+=k) {\n".
                "            uint label = x[pos_x];\n".
                "            if(label==i) {\n".
                "                sum += b[pos_b];\n".
                "            }\n".
                "        }\n".
                "        a[offset_a+i*k+p] += sum;\n".
                //"    }\n".
                "}\n";
        }
        $kernel = $this->createKernel($kernel_name);

        $kernel->setArg(0,$total_local_items,NDArray::uint32);
        $kernel->setArg(1,$k,NDArray::uint32);
        $kernel->setArg(2,$numClass,NDArray::uint32);
        $kernel->setArg(3,$X);
        $kernel->setArg(4,$offsetX,NDArray::uint32);
        $kernel->setArg(5,$A);
        $kernel->setArg(6,$offsetA,NDArray::uint32);
        $kernel->setArg(7,$B);
        $kernel->setArg(8,$offsetB,NDArray::uint32);
        $global_work_size = [$k,$numClass];
        $local_work_size = null;
        $kernel->enqueueNDRange($this->queue,$global_work_size,$local_work_size,null,
            $events,$waitEvents);
    }

    /**
    * A(X[k],n) := B[k,n] ,  m > max(X[k])
    */
    public function scatterAdd_1(
        int $n,
        int $k,
        int $numClass,
        BufferInterface $X, int $offsetX,
        BufferInterface $A, int $offsetA,
        BufferInterface $B, int $offsetB,
        object $events=null, object $waitEvents=null
        ) : void
    {
//echo "mode=1\n";
        $dtype = $A->dtype();
        $total_local_items = $n;
        $max_work_items = $this->maxWorkItem[0];
        if($total_local_items>$max_work_items) {
            throw new InvalidArgumentException('too many cols');
        } else {
            for($max_work_items=1; $max_work_items<$total_local_items;$max_work_items<<=1) {
                ;
            }
            $segments = 1;
            $work_items = $total_local_items;
        }
        $value_size = $A->value_size();
        $type = $this->dtypeToOpenCLType[$dtype];
        $kernel_name = "scatterAdd_S_{$type}";
        if(!isset($this->sources[$kernel_name])) {
            $this->sources[$kernel_name] =
                "__kernel void {$kernel_name}(\n".
                "    const        uint total_local_items,\n".
                "    const        uint k,\n".
                "    const        uint numclass,\n".
                "    const global uint * x,\n".
                "    const        uint offset_x,\n".
                "        __global {$type} * a,\n".
                "    const        uint offset_a,\n".
                "    const global {$type} * b,\n".
                "    const        uint offset_b,\n".
                "         __local {$type} * local_work,\n".
                "    const        uint work_items)\n".
                "{\n".
                "    const uint grid = get_group_id(0);\n".  // A col id (p)
                "    const uint gid1 = get_global_id(1);\n". // A row id(i)
                "    const uint pos_b = grid+offset_b;\n".
                     $this->kernelTemplateSSum(
                         "uint label = x[lid+offset_x];\n".  // pos_x (j)
                         "if(label==gid1) {\n".
                         "    local_work[lid] = b[lid*k+pos_b];\n".
                         "} else {\n".
                         "    local_work[lid] = 0;\n".
                         "}\n",
                         "a[gid1*k+grid+offset_a] += local_work[0];\n"
                     ).
                "}\n";
        }
        $kernel = $this->createKernel($kernel_name);

        $kernel->setArg(0,$total_local_items,NDArray::uint32);
        $kernel->setArg(1,$k,NDArray::uint32);
        $kernel->setArg(2,$numClass,NDArray::uint32);
        $kernel->setArg(3,$X);
        $kernel->setArg(4,$offsetX,NDArray::uint32);
        $kernel->setArg(5,$A);
        $kernel->setArg(6,$offsetA,NDArray::uint32);
        $kernel->setArg(7,$B);
        $kernel->setArg(8,$offsetB,NDArray::uint32);
        $kernel->setArg(9,null,$this->adjBoundary($max_work_items*$value_size));
        $kernel->setArg(10,$work_items,NDArray::uint32);
        $global_work_size = [$max_work_items*$k,$numClass];
        $local_work_size = [$max_work_items,1];
        $kernel->enqueueNDRange($this->queue,$global_work_size,$local_work_size,null,
            $events,$waitEvents);
    }

    /**
    * A(X[k],n) := B[k,n] ,  m > max(X[k])
    */
    public function scatterAdd_2(
        int $n,
        int $k,
        int $numClass,
        BufferInterface $X, int $offsetX,
        BufferInterface $A, int $offsetA,
        BufferInterface $B, int $offsetB,
        object $events=null, object $waitEvents=null
        ) : void
    {
//echo "mode=2\n";
        $dtype = $A->dtype();
        $total_local_items = $n;
        $max_work_items = $this->maxWorkItem[0];
        if($total_local_items>$max_work_items) {
            $segments = (int)ceil($total_local_items/$max_work_items); // round up float
            $work_items = $max_work_items;
        } else {
            for($max_work_items=1; $max_work_items<$total_local_items;$max_work_items<<=1) {
                ;
            }
            $segments = 1;
            $work_items = $total_local_items;
        }
        $value_size = $A->value_size();
        $type = $this->dtypeToOpenCLType[$dtype];
        $kernel_name = "scatterAdd_M_{$type}";
        if(!isset($this->sources[$kernel_name])) {
            $this->sources[$kernel_name] =
                "__kernel void {$kernel_name}(\n".
                "    const        uint total_local_items,\n". // k => n
                "    const        uint k,\n".
                "    const        uint numclass,\n".
                "    const global uint * x,\n".
                "    const        uint offset_x,\n".
                "        __global {$type} * a,\n".
                "    const        uint offset_a,\n".
                "    const global {$type} * b,\n".
                "    const        uint offset_b,\n".
                "    const        uint segments,\n".
                "         __local {$type} * local_work,\n".
                "         __local {$type} * seg_work,\n".
                "    const        uint work_items)\n".
                "{\n".
                "    const uint grid = get_group_id(0);\n".  // A col id(p)(n => k)
                "    const uint gid1 = get_global_id(1);\n".  // A row id(i)(m => class)
                "    const uint pos_b = grid+offset_b;\n".
                     $this->kernelTemplateQSum(
                         "uint label = x[seg*lws+lid+offset_x];\n". // pos_x (i)
                         "if(label==gid1) {\n".
                         "    local_work[lid] = b[(seg*lws+lid)*k+pos_b];\n".
                         "} else {\n".
                         "    local_work[lid] = 0;\n".
                         "}\n",
                         "a[gid1*k+grid+offset_a] += seg_work[0];\n"
                     ).
                "}\n";
        }
        $kernel = $this->createKernel($kernel_name);

        $kernel->setArg(0,$total_local_items,NDArray::uint32);
        $kernel->setArg(1,$k,NDArray::uint32);
        $kernel->setArg(2,$numClass,NDArray::uint32);
        $kernel->setArg(3,$X);
        $kernel->setArg(4,$offsetX,NDArray::uint32);
        $kernel->setArg(5,$A);
        $kernel->setArg(6,$offsetA,NDArray::uint32);
        $kernel->setArg(7,$B);
        $kernel->setArg(8,$offsetB,NDArray::uint32);
        $kernel->setArg(9,$segments,NDArray::uint32);
        $kernel->setArg(10,null,$this->adjBoundary($max_work_items*$value_size));
        $kernel->setArg(11,null,$this->adjBoundary($segments*$value_size));
        $kernel->setArg(12,$work_items,NDArray::uint32);
        $global_work_size = [$max_work_items*$k,$numClass];
        $local_work_size = [$max_work_items,1];
        $kernel->enqueueNDRange($this->queue,$global_work_size,$local_work_size,null,
            $events,$waitEvents);
    }

    /**
    * A(X[k],n) := B[k,n] ,  m > max(X[k])
    */
    public function scatterAdd_3(
        int $n,
        int $k,
        int $numClass,
        BufferInterface $X, int $offsetX,
        BufferInterface $A, int $offsetA,
        BufferInterface $B, int $offsetB,
        object $events=null, object $waitEvents=null
        ) : void
    {
//echo "mode=3\n";
        $dtype = $A->dtype();
        $total_local_items = $n;
        $work_items1 = $this->maxWorkItem[0];
        $work_items2 = $this->maxWorkItem[0];
        if($total_local_items<$work_items1) {
            for($work_items1=1;$work_items1<$total_local_items;$work_items1<<=1) {
                ;
            }
        }
        if($total_local_items<$work_items2) {
            for($work_items2=1;$work_items2<$total_local_items;$work_items2<<=1) {
                ;
            }
        }
        $value_size = $A->value_size();
        $temp_size = 2*$work_items2;
        $temp_buffer = $this->newBuffer(
            $value_size*$temp_size*$k*$numClass,
            OpenCL::CL_MEM_READ_WRITE,null,null,$dtype);

        $type = $this->dtypeToOpenCLType[$dtype];
        $kernel_name1 = "scatterAdd_L1_{$type}";
        $kernel_name2 = "scatterAdd_L2_{$type}";
        if(!isset($this->sources[$kernel_name1])) {
            $this->sources[$kernel_name1] =
                "__kernel void {$kernel_name1}(\n".
                "    const        uint total_local_items,\n".
                "    const        uint k,\n".
                "    const        uint numclass,\n".
                "    const global uint * x,\n".
                "    const        uint offset_x,\n".
                "    const global {$type} * b,\n".
                "    const        uint offset_b,\n".
                "        __global {$type} * temp_buffer,\n".
                "         __local {$type} * local_work)\n".
                "{\n".
                "    const uint parallel_item_id = get_global_id(1);\n".
                $this->splitPointer('inner_id','outer_id','parallel_item_id','k').
                $this->kernelTemplateLSum1(
                        "{$type} input;\n".
                        // inner_id = (p)(cols k)
                        // outer_id = (i)(rows class)
                        "const uint label = x[local_item_id+offset_x];\n".
                        "if(label==outer_id) {\n".
                        "    input = b[local_item_id*k+inner_id+offset_b];\n".
                        "} else {\n".
                        "    input = 0;\n".
                        "}\n",
                        $dtype
                    ).
                "}\n";
        }
        $kernel = $this->createKernel($kernel_name1);

        if(!isset($this->sources[$kernel_name2])) {
            $this->sources[$kernel_name2] =
                "__kernel void {$kernel_name2}(\n".
                "    const        uint k,\n".
                "    const __global {$type} * temp_buffer,\n".
                "          __global {$type} * a,\n".
                "    const        uint offset_a,\n".
                "         __local {$type} * local_work)\n".
                "{\n".
                "    const uint parallel_item_id = get_global_id(1);\n".
                    $this->kernelTemplateLSum2(
                        #"const uint inner_id = parallel_item_id%k;\n". // (p)(cols k)
                        #"const uint outer_id = parallel_item_id/k;\n". // (i)(rows class)
                        "a[parallel_item_id+offset_a] += local_work[0];\n"
                    ).
                "}\n";
        }
        $kernel2 = $this->createKernel($kernel_name2);
        //
        $kernel->setArg(0,$total_local_items,NDArray::uint32);
        $kernel->setArg(1,$k,NDArray::uint32);
        $kernel->setArg(2,$numClass,NDArray::uint32);
        $kernel->setArg(3,$X);
        $kernel->setArg(4,$offsetX,NDArray::uint32);
        $kernel->setArg(5,$B);
        $kernel->setArg(6,$offsetB,NDArray::uint32);
        $kernel->setArg(7,$temp_buffer);
        $kernel->setArg(8,null,$this->adjBoundary($work_items1*$value_size));
        $global_work_size = [$work_items1*$temp_size,$k*$numClass];
        $local_work_size = [$work_items1,1];
        $phase1Events = $this->newEventList();
        $kernel->enqueueNDRange($this->queue,$global_work_size,$local_work_size,null,
                $phase1Events,$waitEvents);
        //
        $kernel2->setArg(0,$k,NDArray::uint32);
        $kernel2->setArg(1,$temp_buffer);
        $kernel2->setArg(2,$A);
        $kernel2->setArg(3,$offsetA,NDArray::uint32);
        $kernel2->setArg(4,null,$this->adjBoundary($work_items2*$value_size));
        $global_work_size = [$work_items2,$k*$numClass];
        $local_work_size = [$work_items2,1];
        $kernel2->enqueueNDRange($this->queue,$global_work_size,$local_work_size,null,
                $events,$phase1Events);
    }

    /**
     * A(X[k],n) := B[k,n] ,  m > max(X[k])
     */
     public function scatterAdd_4(
         int $n,
         int $k,
         int $numClass,
         BufferInterface $X, int $offsetX,
         BufferInterface $A, int $offsetA,
         BufferInterface $B, int $offsetB,
         object $events=null, object $waitEvents=null
         ) : void
    {
//echo "mode=4\n";
        $type = $this->dtypeToOpenCLType[$B->dtype()];
        $kernel_name = "scatterAdd_4_{$type}";
        if(!isset($this->sources[$kernel_name])) {
            $this->sources[$kernel_name] =
                "__kernel void {$kernel_name}(\n".
                "    const        uint n,\n".
                "    const        uint k,\n".
                "    const        uint numclass,\n".
                "    const global uint * x,\n".
                "    const        uint offset_x,\n".
                "        __global {$type} * a,\n".
                "    const        uint offset_a,\n".
                "    const global {$type} * b,\n".
                "    const        uint offset_b)\n".
                "{\n".
                "    const uint gid = get_global_id(0);\n".
                //"    if(gid<n) {\n".
                "        uint pos_x = offset_x;\n".
                "        uint pos_b = offset_b;\n".
                "        uint pos_a = gid+offset_a;\n".
                "        for(uint j=0;j<n;j++,pos_x++,pos_b+=k) {\n".
                "            uint label = x[pos_x];\n".
                "            if(label<numclass) {\n".
                "                a[label*k+pos_a] += b[pos_b+gid];\n".
                "            }\n".
                "        }\n".
                //"    }\n".
                "}\n";
        }
        $kernel = $this->createKernel($kernel_name);

        $kernel->setArg(0,$n,NDArray::uint32);
        $kernel->setArg(1,$k,NDArray::uint32);
        $kernel->setArg(2,$numClass,NDArray::uint32);
        $kernel->setArg(3,$X);
        $kernel->setArg(4,$offsetX,NDArray::uint32);
        $kernel->setArg(5,$A);
        $kernel->setArg(6,$offsetA,NDArray::uint32);
        $kernel->setArg(7,$B);
        $kernel->setArg(8,$offsetB,NDArray::uint32);
        //$multiple = $this->kernelMultiple($kernel);
        //$global_work_size = [$this->ceil($k,$multiple)];
        //$local_work_size = [$multiple];
        $global_work_size = [$k];
        $local_work_size = null;
        $kernel->enqueueNDRange($this->queue,$global_work_size,$local_work_size,null,
            $events,$waitEvents);
    }

    /**
     *
     */
     public function repeat(
         int $m,
         int $k,
         int $repeats,
         BufferInterface $A, int $offsetA,
         BufferInterface $B, int $offsetB,
         object $events=null, object $waitEvents=null
         ) : void
    {
//echo "mode=4\n";
        $type = $this->dtypeToOpenCLType[$B->dtype()];
        $kernel_name = "repeat_{$type}";
        if(!isset($this->sources[$kernel_name])) {
            $this->sources[$kernel_name] =
                "__kernel void {$kernel_name}(\n".
                "    const        uint m,\n".
                "    const        uint k,\n".
                "    const        uint repeats,\n".
                "    const global {$type} * a,\n".
                "    const        uint offset_a,\n".
                "        __global {$type} * b,\n".
                "    const        uint offset_b)\n".
                "{\n".
                "    const uint gid = get_global_id(0);\n".
                     $this->splitPointer('p','j','gid','k').
                "    const uint i = get_global_id(1);\n".
                //"    if(gid<n) {\n".
                "        uint pos_a = i*k+offset_a;\n".
                "        uint pos_b = i*repeats*k+offset_b;\n".
                "        b[pos_b+j*k+p] = a[pos_a+p];\n".
                //"    }\n".
                "}\n";
        }
        $kernel = $this->createKernel($kernel_name);

        $kernel->setArg(0,$m,NDArray::uint32);
        $kernel->setArg(1,$k,NDArray::uint32);
        $kernel->setArg(2,$repeats,NDArray::uint32);
        $kernel->setArg(3,$A);
        $kernel->setArg(4,$offsetA,NDArray::uint32);
        $kernel->setArg(5,$B);
        $kernel->setArg(6,$offsetB,NDArray::uint32);
        //$multiple = $this->kernelMultiple($kernel);
        //$global_work_size = [$this->ceil($k,$multiple)];
        //$local_work_size = [$multiple];
        $global_work_size = [$k*$repeats,$m];
        $local_work_size = null;
        $kernel->enqueueNDRange($this->queue,$global_work_size,$local_work_size,null,
            $events,$waitEvents);
    }

    /**
     *     Y := onehot(X,a)
     */
    public function onehot(
        int $m,
        int $n,
        float $alpha,
        BufferInterface $X, int $offsetX, int $incX,
        BufferInterface $Y, int $offsetY, int $ldY,
        bool $addMode,
        object $events=null, object $waitEvents=null
        ) : void
    {
        if($X->dtype()!=NDArray::int32 && $X->dtype()!=NDArray::uint32) {
            throw new InvalidArgumentException("X must be 32bit integer:".
                                            $this->dtypeToString($X->dtype()));
        }
        if($Y->dtype()!=NDArray::float64 && $Y->dtype()!=NDArray::float32) {
            throw new InvalidArgumentException("Unsuppored data type:".
                                            $this->dtypeToString($Y->dtype()));
        }
        if($Y->dtype()==NDArray::float64) {
            $this->assertFP64();
        }

        $type = $this->dtypeToOpenCLType[$Y->dtype()];
        $mode = $addMode ? 'add' : 'set';
        $operator = $addMode ? '+=' : '=';
        if(!isset($this->sources["onehot_{$type}_{$mode}"])) {
            $this->sources["onehot_{$type}_{$mode}"] =
                "__kernel void onehot_{$type}_{$mode}(\n".
                "    const        uint n,\n".
                "    const        {$type} alpha,\n".
                "    const global uint * x,\n".
                "    const        uint offset_x,\n".
                "    const        uint incx,\n".
                "        __global {$type} * y,\n".
                "    const        uint offset_y,\n".
                "    const        uint ldy)\n".
                "{\n".
                "    uint gid = get_global_id(0);\n".
                "    uint label = x[gid*incx+offset_x];\n".
                "    if(label<n) {\n".
                "        y[gid*ldy+label+offset_y] {$operator} alpha;\n".
                "    }\n".
                "}\n";
        }
        $kernel = $this->createKernel("onehot_{$type}_{$mode}");

        $kernel->setArg(0,$n,NDArray::uint32);
        $kernel->setArg(1,$alpha,$Y->dtype());
        $kernel->setArg(2,$X);
        $kernel->setArg(3,$offsetX,NDArray::uint32);
        $kernel->setArg(4,$incX,NDArray::uint32);
        $kernel->setArg(5,$Y);
        $kernel->setArg(6,$offsetY,NDArray::uint32);
        $kernel->setArg(7,$ldY,NDArray::uint32);
        $global_work_size = [$m];
        $kernel->enqueueNDRange($this->queue,$global_work_size,null,null,
            $events,$waitEvents);
    }

    public function predictTimeReduceSum(int $mode,int $cols,int $rows) : int
    {
        if(isset($this->timesPredictionReduceSum[$mode])) {
            $times = $this->timesPredictionReduceSum[$mode];
        } else {
            $times = $this->loadParameter('ReduceSumTimesMode'.$mode.'.php');
            $this->timesPredictionReduceSum[$mode] = $times;
        }
        if(isset($times[$rows][$cols])) {
            $time = $times[$rows][$cols];
            if($time==0)
                return PHP_INT_MAX;
            return $time;
        } else {
            return PHP_INT_MAX;
        }
    }

    /**
     * X(m) := sum( A(m,n) )
     */
    public function reduceSum(
        int $m,
        int $n,
        int $k,
        BufferInterface $A, int $offsetA,
        BufferInterface $B, int $offsetB,
        object $events=null, object $waitEvents=null
        ) : void
    {
        $dtype = $A->dtype();
        if($dtype!=$B->dtype()) {
            throw new InvalidArgumentException("Unmatch data type A and B:".
            $this->dtypeToString($dtype).",".$this->dtypeToString($B->dtype()));
        }
        if($dtype!=NDArray::float64 && $dtype!=NDArray::float32) {
            throw new InvalidArgumentException("Unsuppored data type:".
                                            $this->dtypeToString($dtype));
        }
        if($dtype==NDArray::float64) {
            $this->assertFP64();
        }
        $mk = $m*$k;
        for($bm=8; $bm<$mk;$bm<<=1) { ; }
        for($bn=8; $bn<$n;$bn<<=1) { ; }
        $mode0 = $this->predictTimeReduceSum(0,$bm,$bn);
        $mode2 = $this->predictTimeReduceSum(2,$bm,$bn);
        $mode3 = $this->predictTimeReduceSum(3,$bm,$bn);

        //echo number_format($mode0)."   ".number_format($mode2)."   ".number_format($mode3)."\n";
        $min1 =  ($mode2 < $mode0) ? $mode2 : $mode0;
        $imin1 = ($mode2 < $mode0) ? 2 : 0;
        $min2 =  ($mode3 < $min1)  ? $mode3 : $min1;
        $mode =  ($mode3 < $min1)  ? 3 : $imin1;
        if($mode==2 && $n<=256) {
            $mode = 1;
        }
        if($this->testMode!==null) {
            $mode = $this->testMode;
        }
        //echo "mode=$mode\n";
        switch($mode) {
            case 0: {
                $this->reduceSum0(
                    $m,
                    $n,
                    $k,
                    $A,$offsetA,
                    $B,$offsetB,
                    $events,$waitEvents
                );
                break;
            }
            case 1: {
                $this->reduceSum1(
                    $m,
                    $n,
                    $k,
                    $A,$offsetA,
                    $B,$offsetB,
                    $events,$waitEvents
                );
                break;
            }
            case 2: {
                $this->reduceSum2(
                    $m,
                    $n,
                    $k,
                    $A,$offsetA,
                    $B,$offsetB,
                    $events,$waitEvents
                );
                break;
            }
            case 3: {
                $this->reduceSum3(
                    $m,
                    $n,
                    $k,
                    $A,$offsetA,
                    $B,$offsetB,
                    $events,$waitEvents
                );
            }
        }
    }

    /**
     * X(m) := sum( A(m,n) )
     */
    public function reduceSum0(
        int $m,
        int $n,
        int $k,
        BufferInterface $A, int $offsetA,
        BufferInterface $B, int $offsetB,
        object $events=null, object $waitEvents=null
        ) : void
    {
        $dtype = $A->dtype();
        $total_local_items = $n;
        $rows = $m;
        $ldA = $n*$k;
        $ldB = $k;
        $type = $this->dtypeToOpenCLType[$dtype];
        $kernel_name = "reduceSum_0_{$type}";
        if(!isset($this->sources[$kernel_name])) {
            $index_a = 'gid0*lda+gid1+offset_a';
            $index_b = 'gid0*ldb+gid1+offset_b';
            $this->sources[$kernel_name] =
                "__kernel void {$kernel_name}(\n".
                "    const        uint rows,\n".
                "    const        uint total_local_items,\n".
                "    const        uint k,\n".
                "    const global {$type} * a,\n".
                "    const        uint offset_a,\n".
                "    const        uint lda,\n".
                "          global {$type} * b,\n".
                "    const        uint offset_b,\n".
                "    const        uint ldb)\n".
                "{\n".
                "    const uint gid = get_global_id(0);\n".
                     $this->splitPointer('gid1','gid0','gid','k').
                "    {$type} sum = 0;\n".
                //"    if(gid0<rows) {\n".
                "        uint pos = {$index_a};\n".
                "        for(uint i=total_local_items; i>0; i--,pos+=k) {\n".
                "            sum += a[pos];\n".
                "        }\n".
                "        b[{$index_b}] = sum;\n".
                //"    }\n".
                "}\n";
        }
        $kernel = $this->createKernel($kernel_name);

        $kernel->setArg(0,$rows,NDArray::uint32);
        $kernel->setArg(1,$total_local_items,NDArray::uint32);
        $kernel->setArg(2,$k,NDArray::uint32);
        $kernel->setArg(3,$A);
        $kernel->setArg(4,$offsetA,NDArray::uint32);
        $kernel->setArg(5,$ldA,NDArray::uint32);
        $kernel->setArg(6,$B);
        $kernel->setArg(7,$offsetB,NDArray::uint32);
        $kernel->setArg(8,$ldB,NDArray::uint32);
        //$multiple = $this->kernelMultiple($kernel);
        //$global_work_size = [$this->ceil($rows*$k,$multiple)];
        //$local_work_size = [$multiple];
        $global_work_size = [$rows*$k];
        $local_work_size = null;
        $kernel->enqueueNDRange($this->queue,$global_work_size,$local_work_size,null,
                $events,$waitEvents);
    }

    /**
     * X(m) := sum( A(m,n) )
     */
    public function reduceSum1(
        int $m,
        int $n,
        int $k,
        BufferInterface $A, int $offsetA,
        BufferInterface $B, int $offsetB,
        object $events=null, object $waitEvents=null
        ) : void
    {
        $dtype = $A->dtype();
        $total_local_items = $n;
        $rows = $m;
        $ldA = $n*$k;
        $ldB = $k;
        $max_work_items = $this->maxWorkItem[0];
        if($total_local_items>$max_work_items) {
            throw new InvalidArgumentException('too many cols');
        } else {
            for($max_work_items=1; $max_work_items<$total_local_items;$max_work_items<<=1) {
                ;
            }
        }
        $value_size = $A->value_size();
        $type = $this->dtypeToOpenCLType[$dtype];
        $kernel_name = "reduceSum_S_{$type}";
        if(!isset($this->sources[$kernel_name])) {
            $this->sources[$kernel_name] =
                "__kernel void {$kernel_name}(\n".
                "    const        uint total_local_items,\n".
                "    const        uint k,\n".
                "    const global {$type} * a,\n".
                "    const        uint offset_a,\n".
                "    const        uint lda,\n".
                "        __global {$type} * b,\n".
                "    const        uint offset_b,\n".
                "    const        uint ldb,\n".
                "         __local {$type} * local_work)\n".
                "{\n".
                "    const uint grid = get_group_id(0);\n".
                     $this->splitPointer('gid_r','gid_l','grid','k').
                "    const uint pos_a = gid_l*lda+gid_r+offset_a;\n".
                "    const uint pos_b = gid_l*ldb+gid_r+offset_b;\n".
                     $this->kernelTemplateSSum(
                         "local_work[lid] = a[pos_a+lid*k];\n",
                         "b[pos_b] = local_work[0];\n"
                     ).
                "}\n";
        }
        $kernel = $this->createKernel($kernel_name);

        $kernel->setArg(0,$total_local_items,NDArray::uint32);
        $kernel->setArg(1,$k,NDArray::uint32);
        $kernel->setArg(2,$A);
        $kernel->setArg(3,$offsetA,NDArray::uint32);
        $kernel->setArg(4,$ldA,NDArray::uint32);
        $kernel->setArg(5,$B);
        $kernel->setArg(6,$offsetB,NDArray::uint32);
        $kernel->setArg(7,$ldB,NDArray::uint32);
        $kernel->setArg(8,null,$this->adjBoundary($max_work_items*$value_size));
        $global_work_size = [$max_work_items*$k*$rows];
        $local_work_size = [$max_work_items];
        $kernel->enqueueNDRange($this->queue,$global_work_size,$local_work_size,null,
                $events,$waitEvents);
    }

    /**
     * X(m) := sum( A(m,n) )
     */
    public function reduceSum2(
        int $m,
        int $n,
        int $k,
        BufferInterface $A, int $offsetA,
        BufferInterface $B, int $offsetB,
        object $events=null, object $waitEvents=null
        ) : void
    {
        $dtype = $A->dtype();
        $total_local_items = $n;
        $rows = $m;
        $ldA = $n*$k;
        $ldB = $k;
        $max_work_items = $this->maxWorkItem[0];
        if($total_local_items>$max_work_items) {
            $segments = (int)ceil($total_local_items/$max_work_items); // round up float
            $work_items = $max_work_items;
        } else {
            for($max_work_items=1; $max_work_items<$total_local_items;$max_work_items<<=1) {
                ;
            }
            $segments = 1; // round up float
            $work_items = $total_local_items;
        }
        $value_size = $A->value_size();
        $type = $this->dtypeToOpenCLType[$dtype];
        $kernel_name = "reduceSum_M_{$type}";
        if(!isset($this->sources[$kernel_name])) {
            $this->sources[$kernel_name] =
                "__kernel void {$kernel_name}(\n".
                "    const        uint total_local_items,\n".
                "    const        uint segments,\n".
                "    const        uint k,\n".
                "    const global {$type} * a,\n".
                "    const        uint offset_a,\n".
                "    const        uint lda,\n".
                "        __global {$type} * b,\n".
                "    const        uint offset_b,\n".
                "    const        uint ldb,\n".
                "         __local {$type} * local_work,\n".
                "         __local {$type} * seg_work,\n".
                "    const        uint work_items)\n".
                "{\n".
                "    const uint grid = get_group_id(0);\n".
                     $this->splitPointer('gid_r','gid_l','grid','k').
                "    const uint pos_a = gid_l*lda+gid_r+offset_a;\n".
                "    const uint pos_b = gid_l*ldb+gid_r+offset_b;\n".
                     $this->kernelTemplateQSum(
                         "local_work[lid] = a[pos_a+(seg*lws+lid)*k];\n",
                         "b[pos_b] = seg_work[0];\n"
                     ).
                "}\n";
        }
        $kernel = $this->createKernel($kernel_name);

        $kernel->setArg(0,$total_local_items,NDArray::uint32);
        $kernel->setArg(1,$segments,NDArray::uint32);
        $kernel->setArg(2,$k,NDArray::uint32);
        $kernel->setArg(3,$A);
        $kernel->setArg(4,$offsetA,NDArray::uint32);
        $kernel->setArg(5,$ldA,NDArray::uint32);
        $kernel->setArg(6,$B);
        $kernel->setArg(7,$offsetB,NDArray::uint32);
        $kernel->setArg(8,$ldB,NDArray::uint32);

        $kernel->setArg(9,null,$this->adjBoundary($max_work_items*$value_size));
        $kernel->setArg(10,null,$this->adjBoundary($segments*$value_size));
        $kernel->setArg(11,$work_items,NDArray::uint32);
        $global_work_size = [$max_work_items*$k*$rows];
        $local_work_size = [$max_work_items];
        $kernel->enqueueNDRange($this->queue,$global_work_size,$local_work_size,null,
                $events,$waitEvents);
    }

    /**
     * X(m) := sum( A(m,n) )
     */
    public function reduceSum3(
        int $m,
        int $n,
        int $k,
        BufferInterface $A, int $offsetA,
        BufferInterface $B, int $offsetB,
        object $events=null, object $waitEvents=null
        ) : void
    {
        $dtype = $A->dtype();
        $total_local_items = $n;
        $rows = $m;
        $ldA = $n*$k;
        $ldB = $k;
        $work_items1 = $this->maxWorkItem[0];
        $work_items2 = $this->maxWorkItem[0];
        if($total_local_items<$work_items1) {
            for($work_items1=1;$work_items1<$total_local_items;$work_items1<<=1) {
                ;
            }
        }
        if($total_local_items<$work_items2) {
            for($work_items2=1;$work_items2<$total_local_items;$work_items2<<=1) {
                ;
            }
        }
        $value_size = $A->value_size();
        $temp_size = 2*$work_items2;
        $temp_buffer = $this->newBuffer(
            $value_size*$temp_size*$rows*$k,
            OpenCL::CL_MEM_READ_WRITE,null,null,$dtype);

        $type = $this->dtypeToOpenCLType[$dtype];
        $kernel_name1 = "reduceSum_L1_{$type}";
        $kernel_name2 = "reduceSum_L2_{$type}";
        if(!isset($this->sources[$kernel_name1])) {
            $this->sources[$kernel_name1] =
                "__kernel void {$kernel_name1}(\n".
                "    const        uint total_local_items,\n".
                "    const        uint k,\n".
                "    const __global {$type} * a,\n".
                "    const        uint offset_a,\n".
                "    const        uint lda,\n".
                "        __global {$type} * temp_buffer,\n".
                "         __local {$type} * local_work)\n".
                "{\n".
                "    const uint parallel_item_id = get_global_id(1);\n".
                     $this->splitPointer('gid_r','gid_l','parallel_item_id','k').
                "    const uint pos_a = gid_l*lda+gid_r+offset_a;\n".
                    $this->kernelTemplateLSum1(
                        "{$type} input = a[pos_a+local_item_id*k];",
                        $dtype
                    ).
                "}\n";
        }
        $kernel = $this->createKernel($kernel_name1);

        if(!isset($this->sources[$kernel_name2])) {
            $this->sources[$kernel_name2] =
                "__kernel void {$kernel_name2}(\n".
                "    const        uint k,\n".
                "    const __global {$type} * temp_buffer,\n".
                "        __global {$type} * b,\n".
                "    const        uint offset_b,\n".
                "    const        uint ldb,\n".
                "         __local {$type} * local_work)\n".
                "{\n".
                "    const uint parallel_item_id = get_global_id(1);\n".
                     $this->splitPointer('gid_r','gid_l','parallel_item_id','k').
                "    const uint pos_b = gid_l*ldb+gid_r+offset_b;\n".
                    $this->kernelTemplateLSum2(
                        "b[pos_b] = local_work[0];"
                    ).
                "}\n";
        }
        $kernel2 = $this->createKernel($kernel_name2);

        $kernel->setArg(0,$total_local_items,NDArray::uint32);
        $kernel->setArg(1,$k,NDArray::uint32);
        $kernel->setArg(2,$A);
        $kernel->setArg(3,$offsetA,NDArray::uint32);
        $kernel->setArg(4,$ldA,NDArray::uint32);
        $kernel->setArg(5,$temp_buffer);
        $kernel->setArg(6,null,$this->adjBoundary($work_items1*$value_size));
        $global_work_size = [$work_items1*$temp_size,$rows*$k];
        $local_work_size  = [$work_items1,1];
        $phase1Events = $this->newEventList();
        $kernel->enqueueNDRange($this->queue,$global_work_size,$local_work_size,null,
                $phase1Events,$waitEvents);

        $kernel2->setArg(0,$k,NDArray::uint32);
        $kernel2->setArg(1,$temp_buffer);
        $kernel2->setArg(2,$B);
        $kernel2->setArg(3,$offsetB,NDArray::uint32);
        $kernel2->setArg(4,$ldB,NDArray::uint32);
        $kernel2->setArg(5,null,$this->adjBoundary($work_items2*$value_size));
        $global_work_size = [$work_items2,$rows*$k];
        $local_work_size = [$work_items2,1];
        $kernel2->enqueueNDRange($this->queue,$global_work_size,$local_work_size,null,
                $events,$phase1Events);
    }

    /**
     * X(m) := sum( A(m,n) )
     */
    public function reduceMax(
        int $m,
        int $n,
        int $k,
        BufferInterface $A, int $offsetA,
        BufferInterface $B, int $offsetB,
        object $events=null, object $waitEvents=null
        ) : void
    {
        $dtype = $A->dtype();
        if($dtype!=$B->dtype()) {
            throw new InvalidArgumentException("Unmatch data type A and B:".
            $this->dtypeToString($dtype).",".$this->dtypeToString($B->dtype()));
        }
        if($dtype!=NDArray::float64 && $dtype!=NDArray::float32) {
            throw new InvalidArgumentException("Unsuppored data type:".
                                            $this->dtypeToString($dtype));
        }
        if($dtype==NDArray::float64) {
            $this->assertFP64();
        }
        $mk = $m*$k;
        for($bm=8; $bm<$mk;$bm<<=1) { ; }
        for($bn=8; $bn<$n;$bn<<=1) { ; }
        $mode0 = $this->predictTimeReduceSum(0,$bm,$bn);
        $mode2 = $this->predictTimeReduceSum(2,$bm,$bn);
        $mode3 = $this->predictTimeReduceSum(3,$bm,$bn);

        //echo number_format($mode0)."   ".number_format($mode2)."   ".number_format($mode3)."\n";
        $min1 =  ($mode2 < $mode0) ? $mode2 : $mode0;
        $imin1 = ($mode2 < $mode0) ? 2 : 0;
        $min2 =  ($mode3 < $min1)  ? $mode3 : $min1;
        $mode =  ($mode3 < $min1)  ? 3 : $imin1;
        if($mode==2 && $n<=256) {
            $mode = 1;
        }
        if($this->testMode!==null) {
            $mode = $this->testMode;
        }
        //echo "mode=$mode\n";
        switch($mode) {
            case 0: {
                $this->reduceMax0(
                    $m,
                    $n,
                    $k,
                    $A,$offsetA,
                    $B,$offsetB,
                    $events,$waitEvents
                );
                break;
            }
            case 1: {
                $this->reduceMax1(
                    $m,
                    $n,
                    $k,
                    $A,$offsetA,
                    $B,$offsetB,
                    $events,$waitEvents
                );
                break;
            }
            case 2: {
                $this->reduceMax2(
                    $m,
                    $n,
                    $k,
                    $A,$offsetA,
                    $B,$offsetB,
                    $events,$waitEvents
                );
                break;
            }
            case 3: {
                $this->reduceMax3(
                    $m,
                    $n,
                    $k,
                    $A,$offsetA,
                    $B,$offsetB,
                    $events,$waitEvents
                );
            }
        }
    }

    /**
     * X(m) := sum( A(m,n) )
     */
    public function reduceMax0(
        int $m,
        int $n,
        int $k,
        BufferInterface $A, int $offsetA,
        BufferInterface $B, int $offsetB,
        object $events=null, object $waitEvents=null
        ) : void
    {
        $dtype = $A->dtype();
        $total_local_items = $n;
        $rows = $m;
        $ldA = $n*$k;
        $ldB = $k;
        $type = $this->dtypeToOpenCLType[$dtype];
        $kernel_name = "reduceMax_0_{$type}";
        if(!isset($this->sources[$kernel_name])) {
            $index_a = 'gid0*lda+gid1+offset_a';
            $index_b = 'gid0*ldb+gid1+offset_b';
            $this->sources[$kernel_name] =
                "__kernel void {$kernel_name}(\n".
                "    const        uint rows,\n".
                "    const        uint total_local_items,\n".
                "    const        uint k,\n".
                "    const global {$type} * a,\n".
                "    const        uint offset_a,\n".
                "    const        uint lda,\n".
                "          global {$type} * b,\n".
                "    const        uint offset_b,\n".
                "    const        uint ldb)\n".
                "{\n".
                "    const uint gid = get_global_id(0);\n".
                     $this->splitPointer('gid1','gid0','gid','k').
                //"    if(gid0<rows) {\n".
                "        uint i=0;\n".
                "        uint pos = {$index_a};\n".
                "        {$type} max = a[pos];\n".
                "        pos += k;\n".
                "        for(uint i=1; i<total_local_items; i++,pos+=k) {\n".
                "            {$type} value = a[pos];\n".
                //           if NaN set NaN
                //           Compatible with reduce_max of tensorflow 2.6
                "            if(max<value||isnan(value)) {\n".
                "                max = value;\n".
                "            }\n".
                "        }\n".
                "        b[{$index_b}] = max;\n".
                //"    }\n".
                "}\n";
        }
        $kernel = $this->createKernel($kernel_name);

        $kernel->setArg(0,$rows,NDArray::uint32);
        $kernel->setArg(1,$total_local_items,NDArray::uint32);
        $kernel->setArg(2,$k,NDArray::uint32);
        $kernel->setArg(3,$A);
        $kernel->setArg(4,$offsetA,NDArray::uint32);
        $kernel->setArg(5,$ldA,NDArray::uint32);
        $kernel->setArg(6,$B);
        $kernel->setArg(7,$offsetB,NDArray::uint32);
        $kernel->setArg(8,$ldB,NDArray::uint32);
        //$multiple = $this->kernelMultiple($kernel);
        //$global_work_size = [$this->ceil($rows*$k,$multiple)];
        //$local_work_size = [$multiple];
        $global_work_size = [$rows*$k];
        $local_work_size = null;
        $kernel->enqueueNDRange($this->queue,$global_work_size,$local_work_size,null,
                $events,$waitEvents);
    }

    /**
     * X(m) := sum( A(m,n) )
     */
    public function reduceMax1(
        int $m,
        int $n,
        int $k,
        BufferInterface $A, int $offsetA,
        BufferInterface $B, int $offsetB,
        object $events=null, object $waitEvents=null
        ) : void
    {
        $dtype = $A->dtype();
        $total_local_items = $n;
        $rows = $m;
        $ldA = $n*$k;
        $ldB = $k;
        $max_work_items = $this->maxWorkItem[0];
        if($total_local_items>$max_work_items) {
            throw new InvalidArgumentException('too many cols');
        } else {
            for($max_work_items=1; $max_work_items<$total_local_items;$max_work_items<<=1) {
                ;
            }
        }
        $value_size = $A->value_size();
        $type = $this->dtypeToOpenCLType[$dtype];
        $kernel_name = "reduceMax_S_{$type}";
        if(!isset($this->sources[$kernel_name])) {
            $this->sources[$kernel_name] =
                "__kernel void {$kernel_name}(\n".
                "    const        uint total_local_items,\n".
                "    const        uint k,\n".
                "    const global {$type} * a,\n".
                "    const        uint offset_a,\n".
                "    const        uint lda,\n".
                "        __global {$type} * b,\n".
                "    const        uint offset_b,\n".
                "    const        uint ldb,\n".
                "         __local {$type} * local_work)\n".
                "{\n".
                "    const uint grid = get_group_id(0);\n".
                     $this->splitPointer('gid_r','gid_l','grid','k').
                "    const uint pos_a = gid_l*lda+gid_r+offset_a;\n".
                "    const uint pos_b = gid_l*ldb+gid_r+offset_b;\n".
                     $this->kernelTemplateSMax(
                         "local_work[lid] = a[pos_a+lid*k];\n",
                         "b[pos_b] = local_work[0];\n",
                         $dtype
                     ).
                "}\n";
        }
        $kernel = $this->createKernel($kernel_name);

        $kernel->setArg(0,$total_local_items,NDArray::uint32);
        $kernel->setArg(1,$k,NDArray::uint32);
        $kernel->setArg(2,$A);
        $kernel->setArg(3,$offsetA,NDArray::uint32);
        $kernel->setArg(4,$ldA,NDArray::uint32);
        $kernel->setArg(5,$B);
        $kernel->setArg(6,$offsetB,NDArray::uint32);
        $kernel->setArg(7,$ldB,NDArray::uint32);
        $kernel->setArg(8,null,$this->adjBoundary($max_work_items*$value_size));
        $global_work_size = [$max_work_items*$k*$rows];
        $local_work_size = [$max_work_items];
        $kernel->enqueueNDRange($this->queue,$global_work_size,$local_work_size,null,
                $events,$waitEvents);
    }

    /**
     * X(m) := sum( A(m,n) )
     */
    public function reduceMax2(
        int $m,
        int $n,
        int $k,
        BufferInterface $A, int $offsetA,
        BufferInterface $B, int $offsetB,
        object $events=null, object $waitEvents=null
        ) : void
    {
        $dtype = $A->dtype();
        $total_local_items = $n;
        $rows = $m;
        $ldA = $n*$k;
        $ldB = $k;
        $max_work_items = $this->maxWorkItem[0];
        if($total_local_items>$max_work_items) {
            $segments = (int)ceil($total_local_items/$max_work_items); // round up float
            $work_items = $max_work_items;
        } else {
            for($max_work_items=1; $max_work_items<$total_local_items;$max_work_items<<=1) {
                ;
            }
            $segments = 1; // round up float
            $work_items = $total_local_items;
        }
        $value_size = $A->value_size();
        $type = $this->dtypeToOpenCLType[$dtype];
        $kernel_name = "reduceMax_M_{$type}";
        if(!isset($this->sources[$kernel_name])) {
            $this->sources[$kernel_name] =
                "__kernel void {$kernel_name}(\n".
                "    const        uint total_local_items,\n".
                "    const        uint segments,\n".
                "    const        uint k,\n".
                "    const global {$type} * a,\n".
                "    const        uint offset_a,\n".
                "    const        uint lda,\n".
                "        __global {$type} * b,\n".
                "    const        uint offset_b,\n".
                "    const        uint ldb,\n".
                "         __local {$type} * local_work,\n".
                "         __local {$type} * seg_work,\n".
                "    const        uint work_items)\n".
                "{\n".
                "    const uint grid = get_group_id(0);\n".
                     $this->splitPointer('gid_r','gid_l','grid','k').
                "    const uint pos_a = gid_l*lda+gid_r+offset_a;\n".
                "    const uint pos_b = gid_l*ldb+gid_r+offset_b;\n".
                     $this->kernelTemplateQMax(
                         "local_work[lid] = a[pos_a+(seg*lws+lid)*k];\n",
                         "b[pos_b] = seg_work[0];\n",
                         $dtype
                     ).
                "}\n";
        }
        $kernel = $this->createKernel($kernel_name);

        $kernel->setArg(0,$total_local_items,NDArray::uint32);
        $kernel->setArg(1,$segments,NDArray::uint32);
        $kernel->setArg(2,$k,NDArray::uint32);
        $kernel->setArg(3,$A);
        $kernel->setArg(4,$offsetA,NDArray::uint32);
        $kernel->setArg(5,$ldA,NDArray::uint32);
        $kernel->setArg(6,$B);
        $kernel->setArg(7,$offsetB,NDArray::uint32);
        $kernel->setArg(8,$ldB,NDArray::uint32);

        $kernel->setArg(9,null,$this->adjBoundary($max_work_items*$value_size));
        $kernel->setArg(10,null,$this->adjBoundary($segments*$value_size));
        $kernel->setArg(11,$work_items,NDArray::uint32);
        $global_work_size = [$max_work_items*$k*$rows];
        $local_work_size = [$max_work_items];
        $kernel->enqueueNDRange($this->queue,$global_work_size,$local_work_size,null,
                $events,$waitEvents);
    }

    /**
     * X(m) := sum( A(m,n) )
     */
    public function reduceMax3(
        int $m,
        int $n,
        int $k,
        BufferInterface $A, int $offsetA,
        BufferInterface $B, int $offsetB,
        object $events=null, object $waitEvents=null
        ) : void
    {
        $dtype = $A->dtype();
        $total_local_items = $n;
        $rows = $m;
        $ldA = $n*$k;
        $ldB = $k;
        $work_items1 = $this->maxWorkItem[0];
        $work_items2 = $this->maxWorkItem[0];
        if($total_local_items<$work_items1) {
            for($work_items1=1;$work_items1<$total_local_items;$work_items1<<=1) {
                ;
            }
        }
        if($total_local_items<$work_items2) {
            for($work_items2=1;$work_items2<$total_local_items;$work_items2<<=1) {
                ;
            }
        }
        $value_size = $A->value_size();
        $temp_size = 2*$work_items2;
        $temp_buffer = $this->newBuffer(
            $value_size*$temp_size*$rows*$k,
            OpenCL::CL_MEM_READ_WRITE,null,null,$dtype);

        $type = $this->dtypeToOpenCLType[$dtype];
        $kernel_name1 = "reduceMax_L1_{$type}";
        $kernel_name2 = "reduceMax_L2_{$type}";
        if(!isset($this->sources[$kernel_name1])) {
            $this->sources[$kernel_name1] =
                "__kernel void {$kernel_name1}(\n".
                "    const        uint total_local_items,\n".
                "    const        uint k,\n".
                "    const __global {$type} * a,\n".
                "    const        uint offset_a,\n".
                "    const        uint lda,\n".
                "        __global {$type} * temp_buffer,\n".
                "         __local {$type} * local_work)\n".
                "{\n".
                "    const uint parallel_item_id = get_global_id(1);\n".
                     $this->splitPointer('gid_r','gid_l','parallel_item_id','k').
                "    const uint pos_a = gid_l*lda+gid_r+offset_a;\n".
                    $this->kernelTemplateLMax1(
                        "{$type} input = a[pos_a+local_item_id*k];",
                        $dtype
                    ).
                "}\n";
        }
        $kernel = $this->createKernel($kernel_name1);

        if(!isset($this->sources[$kernel_name2])) {
            $this->sources[$kernel_name2] =
                "__kernel void {$kernel_name2}(\n".
                "    const        uint k,\n".
                "    const __global {$type} * temp_buffer,\n".
                "        __global {$type} * b,\n".
                "    const        uint offset_b,\n".
                "    const        uint ldb,\n".
                "         __local {$type} * local_work)\n".
                "{\n".
                "    const uint parallel_item_id = get_global_id(1);\n".
                "    const uint gid_l = parallel_item_id/k;\n".
                "    const uint gid_r = parallel_item_id%k;\n".
                "    const uint pos_b = gid_l*ldb+gid_r+offset_b;\n".
                    $this->kernelTemplateLMax2(
                        "b[pos_b] = local_work[0];"
                    ).
                "}\n";
        }
        $kernel2 = $this->createKernel($kernel_name2);

        $kernel->setArg(0,$total_local_items,NDArray::uint32);
        $kernel->setArg(1,$k,NDArray::uint32);
        $kernel->setArg(2,$A);
        $kernel->setArg(3,$offsetA,NDArray::uint32);
        $kernel->setArg(4,$ldA,NDArray::uint32);
        $kernel->setArg(5,$temp_buffer);
        $kernel->setArg(6,null,$this->adjBoundary($work_items1*$value_size));
        $global_work_size = [$work_items1*$temp_size,$rows*$k];
        $local_work_size  = [$work_items1,1];
        $phase1Events = $this->newEventList();
        $kernel->enqueueNDRange($this->queue,$global_work_size,$local_work_size,null,
                $phase1Events,$waitEvents);

        $kernel2->setArg(0,$k,NDArray::uint32);
        $kernel2->setArg(1,$temp_buffer);
        $kernel2->setArg(2,$B);
        $kernel2->setArg(3,$offsetB,NDArray::uint32);
        $kernel2->setArg(4,$ldB,NDArray::uint32);
        $kernel2->setArg(5,null,$this->adjBoundary($work_items2*$value_size));
        $global_work_size = [$work_items2,$rows*$k];
        $local_work_size = [$work_items2,1];
        $kernel2->enqueueNDRange($this->queue,$global_work_size,$local_work_size,null,
                $events,$phase1Events);
    }

    /**
     * X(m) := argMax( A(m,n) )
     */
    public function reduceArgMax(
        int $m,
        int $n,
        int $k,
        BufferInterface $A, int $offsetA,
        BufferInterface $B, int $offsetB,
        object $events=null, object $waitEvents=null
        ) : void
    {
        $dtype = $A->dtype();
        if($B->dtype()!=NDArray::int32 && $B->dtype()!=NDArray::uint32) {
            throw new InvalidArgumentException("B must be 32bit integer:".
                                            $this->dtypeToString($B->dtype()));
        }
        //if($dtype!=NDArray::float64 && $dtype!=NDArray::float32)
        if($dtype==NDArray::bool) {
            throw new InvalidArgumentException("Unsuppored data type:".
                                            $this->dtypeToString($dtype));
        }
        if($dtype==NDArray::float64) {
            $this->assertFP64();
        }
        $mk = $m*$k;
        for($bm=8; $bm<$mk;$bm<<=1) { ; }
        for($bn=8; $bn<$n;$bn<<=1) { ; }
        $mode0 = $this->predictTimeReduceSum(0,$bm,$bn);
        $mode2 = $this->predictTimeReduceSum(2,$bm,$bn);
        $mode3 = $this->predictTimeReduceSum(3,$bm,$bn);

        //echo number_format($mode0)."   ".number_format($mode2)."   ".number_format($mode3)."\n";
        $min1 =  ($mode2 < $mode0) ? $mode2 : $mode0;
        $imin1 = ($mode2 < $mode0) ? 2 : 0;
        $min2 =  ($mode3 < $min1)  ? $mode3 : $min1;
        $mode =  ($mode3 < $min1)  ? 3 : $imin1;
        if($mode==2 && $n<=256) {
            $mode = 1;
        }
        if($this->testMode!==null) {
            $mode = $this->testMode;
        }
        //echo "mode=$mode\n";
        switch($mode) {
            case 0: {
                $this->reduceArgMax0(
                    $m,
                    $n,
                    $k,
                    $A,$offsetA,
                    $B,$offsetB,
                    $events,$waitEvents
                );
                break;
            }
            case 1: {
                $this->reduceArgMax1(
                    $m,
                    $n,
                    $k,
                    $A,$offsetA,
                    $B,$offsetB,
                    $events,$waitEvents
                );
                break;
            }
            case 2: {
                $this->reduceArgMax2(
                    $m,
                    $n,
                    $k,
                    $A,$offsetA,
                    $B,$offsetB,
                    $events,$waitEvents
                );
                break;
            }
            case 3: {
                $this->reduceArgMax3(
                    $m,
                    $n,
                    $k,
                    $A,$offsetA,
                    $B,$offsetB,
                    $events,$waitEvents
                );
            }
        }
    }

    /**
     * X(m) := sum( A(m,n) )
     */
    public function reduceArgMax0(
        int $m,
        int $n,
        int $k,
        BufferInterface $A, int $offsetA,
        BufferInterface $B, int $offsetB,
        object $events=null, object $waitEvents=null
        ) : void
    {
        $dtype = $A->dtype();
        $total_local_items = $n;
        $rows = $m;
        $ldA = $n*$k;
        $ldB = $k;
        $type = $this->dtypeToOpenCLType[$dtype];
        $kernel_name = "reduceArgMax_0_{$type}";
        if(!isset($this->sources[$kernel_name])) {
            $index_a = 'gid0*lda+gid1+offset_a';
            $index_b = 'gid0*ldb+gid1+offset_b';
            $this->sources[$kernel_name] =
                "__kernel void {$kernel_name}(\n".
                "    const        uint rows,\n".
                "    const        uint total_local_items,\n".
                "    const        uint k,\n".
                "    const global {$type} * a,\n".
                "    const        uint offset_a,\n".
                "    const        uint lda,\n".
                "          global uint * b,\n".
                "    const        uint offset_b,\n".
                "    const        uint ldb)\n".
                "{\n".
                "    const uint gid = get_global_id(0);\n".
                     $this->splitPointer('gid1','gid0','gid','k').
                //"    if(gid0<rows) {\n".
                "        uint i=0;\n".
                "        uint pos = {$index_a};\n".
                "        {$type} max = a[pos];\n".
                "        uint imax = i;\n".
                "        pos += k;\n".
                "        for(uint i=1; i<total_local_items; i++,pos+=k) {\n".
                "            {$type} value = a[pos];\n".
                "            if(value>max) {\n".
                "                max = value;\n".
                "                imax = i;\n".
                "            }\n".
                "        }\n".
                "        b[{$index_b}] = imax;\n".
                //"    }\n".
                "}\n";
        }
        $kernel = $this->createKernel($kernel_name);

        $kernel->setArg(0,$rows,NDArray::uint32);
        $kernel->setArg(1,$total_local_items,NDArray::uint32);
        $kernel->setArg(2,$k,NDArray::uint32);
        $kernel->setArg(3,$A);
        $kernel->setArg(4,$offsetA,NDArray::uint32);
        $kernel->setArg(5,$ldA,NDArray::uint32);
        $kernel->setArg(6,$B);
        $kernel->setArg(7,$offsetB,NDArray::uint32);
        $kernel->setArg(8,$ldB,NDArray::uint32);
        //$multiple = $this->kernelMultiple($kernel);
        //$global_work_size = [$this->ceil($rows*$k,$multiple)];
        //$local_work_size = [$multiple];
        $global_work_size = [$rows*$k];
        $local_work_size = null;
        $kernel->enqueueNDRange($this->queue,$global_work_size,$local_work_size,null,
                $events,$waitEvents);
    }

    /**
     * X(m) := sum( A(m,n) )
     */
    public function reduceArgMax1(
        int $m,
        int $n,
        int $k,
        BufferInterface $A, int $offsetA,
        BufferInterface $B, int $offsetB,
        object $events=null, object $waitEvents=null
        ) : void
    {
        $dtype = $A->dtype();
        $total_local_items = $n;
        $rows = $m;
        $ldA = $n*$k;
        $ldB = $k;
        $max_work_items = $this->maxWorkItem[0];
        if($total_local_items>$max_work_items) {
            throw new InvalidArgumentException('too many cols');
        } else {
            for($max_work_items=1; $max_work_items<$total_local_items;$max_work_items<<=1) {
                ;
            }
        }
        $value_size = $A->value_size();
        //$index_value_size = $B->value_size();
        $index_value_size = (int)(32/8); // uint32 size
        $type = $this->dtypeToOpenCLType[$dtype];
        $kernel_name = "reduceArgMax_S_{$type}";
        if(!isset($this->sources[$kernel_name])) {
            $this->sources[$kernel_name] =
                "__kernel void {$kernel_name}(\n".
                "    const        uint total_local_items,\n".
                "    const        uint k,\n".
                "    const global {$type} * a,\n".
                "    const        uint offset_a,\n".
                "    const        uint lda,\n".
                "        __global uint * b,\n".
                "    const        uint offset_b,\n".
                "    const        uint ldb,\n".
                "         __local {$type} * local_work,\n".
                "         __local uint * local_iwork)\n".
                "{\n".
                "    const uint grid = get_group_id(0);\n".
                     $this->splitPointer('gid_r','gid_l','grid','k').
                "    const uint pos_a = gid_l*lda+gid_r+offset_a;\n".
                "    const uint pos_b = gid_l*ldb+gid_r+offset_b;\n".
                     $this->kernelTemplateSiMax(
                         "local_work[lid] = a[pos_a+lid*k];\n".
                         "local_iwork[lid] = lid;",
                         "b[pos_b] = local_iwork[0];\n",
                         $dtype
                     ).
                "}\n";
        }
        $kernel = $this->createKernel($kernel_name);

        $kernel->setArg(0,$total_local_items,NDArray::uint32);
        $kernel->setArg(1,$k,NDArray::uint32);
        $kernel->setArg(2,$A);
        $kernel->setArg(3,$offsetA,NDArray::uint32);
        $kernel->setArg(4,$ldA,NDArray::uint32);
        $kernel->setArg(5,$B);
        $kernel->setArg(6,$offsetB,NDArray::uint32);
        $kernel->setArg(7,$ldB,NDArray::uint32);
        $kernel->setArg(8,null,$this->adjBoundary($max_work_items*$value_size));
        $kernel->setArg(9,null,$this->adjBoundary($max_work_items*$index_value_size));
        $global_work_size = [$max_work_items*$k*$rows];
        $local_work_size = [$max_work_items];
        $kernel->enqueueNDRange($this->queue,$global_work_size,$local_work_size,null,
                $events,$waitEvents);
    }

    /**
     * X(m) := sum( A(m,n) )
     */
    public function reduceArgMax2(
        int $m,
        int $n,
        int $k,
        BufferInterface $A, int $offsetA,
        BufferInterface $B, int $offsetB,
        object $events=null, object $waitEvents=null
        ) : void
    {
        $dtype = $A->dtype();
        $total_local_items = $n;
        $rows = $m;
        $ldA = $n*$k;
        $ldB = $k;
        $max_work_items = $this->maxWorkItem[0];
        if($total_local_items>$max_work_items) {
            $segments = (int)ceil($total_local_items/$max_work_items); // round up float
            $work_items = $max_work_items;
        } else {
            for($max_work_items=1; $max_work_items<$total_local_items;$max_work_items<<=1) {
                ;
            }
            $segments = 1; // round up float
            $work_items = $total_local_items;
        }
        $value_size = $A->value_size();
        //$index_value_size = $B->value_size();
        $index_value_size = (int)(32/8); // uint32 size
        $type = $this->dtypeToOpenCLType[$dtype];
        $kernel_name = "reduceArgMax_M_{$type}";
        if(!isset($this->sources[$kernel_name])) {
            $this->sources[$kernel_name] =
                "__kernel void {$kernel_name}(\n".
                "    const        uint total_local_items,\n".
                "    const        uint segments,\n".
                "    const        uint k,\n".
                "    const global {$type} * a,\n".
                "    const        uint offset_a,\n".
                "    const        uint lda,\n".
                "        __global uint * b,\n".
                "    const        uint offset_b,\n".
                "    const        uint ldb,\n".
                "         __local {$type} * local_work,\n".
                "         __local {$type} * seg_work,\n".
                "         __local uint * local_iwork,\n".
                "         __local uint * seg_iwork,\n".
                "    const        uint work_items)\n".
                "{\n".
                "    const uint grid = get_group_id(0);\n".
                     $this->splitPointer('gid_r','gid_l','grid','k').
                "    const uint pos_a = gid_l*lda+gid_r+offset_a;\n".
                "    const uint pos_b = gid_l*ldb+gid_r+offset_b;\n".
                     $this->kernelTemplateQiMax(
                         "local_work[lid] = a[pos_a+(seg*lws+lid)*k];\n".
                         "local_iwork[lid] = seg*lws+lid;\n",
                         "b[pos_b] = seg_iwork[0];\n",
                         $dtype
                     ).
                "}\n";
        }
        $kernel = $this->createKernel($kernel_name);

        $kernel->setArg(0,$total_local_items,NDArray::uint32);
        $kernel->setArg(1,$segments,NDArray::uint32);
        $kernel->setArg(2,$k,NDArray::uint32);
        $kernel->setArg(3,$A);
        $kernel->setArg(4,$offsetA,NDArray::uint32);
        $kernel->setArg(5,$ldA,NDArray::uint32);
        $kernel->setArg(6,$B);
        $kernel->setArg(7,$offsetB,NDArray::uint32);
        $kernel->setArg(8,$ldB,NDArray::uint32);

        $kernel->setArg(9,null,$this->adjBoundary($max_work_items*$value_size));
        $kernel->setArg(10,null,$this->adjBoundary($segments*$value_size));
        $kernel->setArg(11,null,$this->adjBoundary($max_work_items*$index_value_size));
        $kernel->setArg(12,null,$this->adjBoundary($segments*$index_value_size));
        $kernel->setArg(13,$work_items,NDArray::uint32);
        $global_work_size = [$max_work_items*$k*$rows];
        $local_work_size = [$max_work_items];
        $kernel->enqueueNDRange($this->queue,$global_work_size,$local_work_size,null,
                $events,$waitEvents);
    }

    /**
     * X(m) := sum( A(m,n) )
     */
    public function reduceArgMax3(
        int $m,
        int $n,
        int $k,
        BufferInterface $A, int $offsetA,
        BufferInterface $B, int $offsetB,
        object $events=null, object $waitEvents=null
        ) : void
    {
        $dtype = $A->dtype();
        $total_local_items = $n;
        $rows = $m;
        $ldA = $n*$k;
        $ldB = $k;
        $work_items1 = $this->maxWorkItem[0];
        $work_items2 = $this->maxWorkItem[0];
        if($total_local_items<$work_items1) {
            for($work_items1=1;$work_items1<$total_local_items;$work_items1<<=1) {
                ;
            }
        }
        if($total_local_items<$work_items2) {
            for($work_items2=1;$work_items2<$total_local_items;$work_items2<<=1) {
                ;
            }
        }
        $value_size = $A->value_size();
        $index_value_size = $B->value_size();
        $temp_size = 2*$work_items2;
        $temp_buffer = $this->newBuffer(
            $value_size*$temp_size*$rows*$k,
            OpenCL::CL_MEM_READ_WRITE,null,null,$dtype);
        $temp_ibuffer = $this->newBuffer(
            $index_value_size*$temp_size*$rows*$k,
            OpenCL::CL_MEM_READ_WRITE,null,null,$B->dtype());
        $type = $this->dtypeToOpenCLType[$dtype];
        $kernel_name1 = "reduceArgMax_L1_{$type}";
        $kernel_name2 = "reduceArgMax_L2_{$type}";
        if(!isset($this->sources[$kernel_name1])) {
            $this->sources[$kernel_name1] =
                "__kernel void {$kernel_name1}(\n".
                "    const        uint total_local_items,\n".
                "    const        uint k,\n".
                "    const __global {$type} * a,\n".
                "    const        uint offset_a,\n".
                "    const        uint lda,\n".
                "        __global {$type} * temp_buffer,\n".
                "         __local {$type} * local_work,\n".
                "        __global uint * temp_ibuffer,\n".
                "         __local uint * local_iwork)\n".
                "{\n".
                "    const uint parallel_item_id = get_global_id(1);\n".
                     $this->splitPointer('gid_r','gid_l','parallel_item_id','k').
                "    const uint pos_a = gid_l*lda+gid_r+offset_a;\n".
                    $this->kernelTemplateLiMax1(
                        "{$type} input = a[pos_a+local_item_id*k];".
                        "uint input_index = local_item_id;\n",
                        $dtype
                    ).
                "}\n";
        }
        $kernel = $this->createKernel($kernel_name1);

        if(!isset($this->sources[$kernel_name2])) {
            $this->sources[$kernel_name2] =
                "__kernel void {$kernel_name2}(\n".
                "    const        uint k,\n".
                "    const __global {$type} * temp_buffer,\n".
                "    const __global uint * temp_ibuffer,\n".
                "        __global uint * b,\n".
                "    const        uint offset_b,\n".
                "    const        uint ldb,\n".
                "         __local {$type} * local_work,\n".
                "         __local uint * local_iwork)\n".
                "{\n".
                "    const uint parallel_item_id = get_global_id(1);\n".
                    $this->splitPointer('gid_r','gid_l','parallel_item_id','k').
                "    const uint pos_b = gid_l*ldb+gid_r+offset_b;\n".
                    $this->kernelTemplateLiMax2(
                        "b[pos_b] = local_iwork[0];"
                    ).
                "}\n";
        }
        $kernel2 = $this->createKernel($kernel_name2);

        $kernel->setArg(0,$total_local_items,NDArray::uint32);
        $kernel->setArg(1,$k,NDArray::uint32);
        $kernel->setArg(2,$A);
        $kernel->setArg(3,$offsetA,NDArray::uint32);
        $kernel->setArg(4,$ldA,NDArray::uint32);
        $kernel->setArg(5,$temp_buffer);
        $kernel->setArg(6,null,$this->adjBoundary($work_items1*$value_size));
        $kernel->setArg(7,$temp_ibuffer);
        $kernel->setArg(8,null,$this->adjBoundary($work_items1*$index_value_size));
        $global_work_size = [$work_items1*$temp_size,$rows*$k];
        $local_work_size  = [$work_items1,1];
        $phase1Events = $this->newEventList();
        $kernel->enqueueNDRange($this->queue,$global_work_size,$local_work_size,null,
                $phase1Events,$waitEvents);

        $kernel2->setArg(0,$k,NDArray::uint32);
        $kernel2->setArg(1,$temp_buffer);
        $kernel2->setArg(2,$temp_ibuffer);
        $kernel2->setArg(3,$B);
        $kernel2->setArg(4,$offsetB,NDArray::uint32);
        $kernel2->setArg(5,$ldB,NDArray::uint32);
        $kernel2->setArg(6,null,$this->adjBoundary($work_items2*$value_size));
        $kernel2->setArg(7,null,$this->adjBoundary($work_items2*$index_value_size));
        $global_work_size = [$work_items2,$rows*$k];
        $local_work_size = [$work_items2,1];
        $kernel2->enqueueNDRange($this->queue,$global_work_size,$local_work_size,null,
                $events,$phase1Events);
    }

    /**
    *     X := softmax(X)
    */
    public function softmax(
        int $m,
        int $n,
        BufferInterface $A, int $offsetA, int $ldA,
        object $events=null, object $waitEvents=null
        ) : void
    {
        $dtype = $A->dtype();
        if($dtype!=NDArray::float64 && $dtype!=NDArray::float32) {
            throw new InvalidArgumentException("Unsuppored data type:".
                                            $this->dtypeToString($dtype));
        }
        if($dtype==NDArray::float64) {
            $this->assertFP64();
        }
        $cols = $n;
        $max_work_items = $this->maxWorkItem[0];
        if($cols <= $max_work_items*2) {
            $this->softmax0(
                $m,
                $n,
                $A,$offsetA,$ldA,
                $events,$waitEvents
            );
        } else {
            $this->softmax2(
                $m,
                $n,
                $A,$offsetA,$ldA,
                $events,$waitEvents
            );
        }
    }

    public function softmax0(
        int $m,
        int $n,
        BufferInterface $A, int $offsetA, int $ldA,
        object $events=null, object $waitEvents=null
        ) : void
    {
        //$trans = false;  // disable transpose function yet
        $dtype = $A->dtype();
        //if($trans) {
        //    $trans = 'trans';
        //    $rows = $n;
        //    $cols = $m;
        //} else {
            $trans = 'norm';
            $rows = $m;
            $cols = $n;
        //}
        $total_local_items = $cols;
        $type = $this->dtypeToOpenCLType[$dtype];
        $kernel_name = "softmax_4_{$type}_{$trans}";
        if(!isset($this->sources[$kernel_name])) {
            //if($trans=='trans') {
            //    $index_a = 'i*lda+gid+offset_a';
            //} else {
                $index_a = 'gid*lda+i+offset_a';
            //}
            $this->sources[$kernel_name] =
                "__kernel void {$kernel_name}(\n".
                "    const        uint rows,\n".
                "    const        uint total_local_items,\n".
                "        __global {$type} * a,\n".
                "    const        uint offset_a,\n".
                "    const        uint lda)\n".
                "{\n".
                "    const uint gid = get_global_id(0);\n".
                //"    if(gid<rows) {\n".
                "        uint i=0;\n".
                "        {$type} max = a[{$index_a}];\n".
                "        for(i=1;i<total_local_items;i++) {\n".
                "            {$type} value = a[{$index_a}];\n".
                "            if(value > max) {\n".
                "                max = value;\n".
                "            }\n".
                "        }\n".
                "        {$type} sum=0;\n".
                "        for(i=0;i<total_local_items;i++) {\n".
                "            sum += exp(a[{$index_a}]-max);".
                "        }\n".
                "        for(i=0;i<total_local_items;i++) {\n".
                "            a[{$index_a}] = exp(a[{$index_a}]-max)/sum;\n".
                "        }\n".
                //"    }\n".
                "}\n";
        }
        $kernel = $this->createKernel($kernel_name);

        $kernel->setArg(0,$rows,NDArray::uint32);
        $kernel->setArg(1,$total_local_items,NDArray::uint32);
        $kernel->setArg(2,$A);
        $kernel->setArg(3,$offsetA,NDArray::uint32);
        $kernel->setArg(4,$ldA,NDArray::uint32);

        //$multiple = $this->kernelMultiple($kernel);
        //$global_work_size = [$this->ceil($rows,$multiple)];
        //$local_work_size = [$multiple];
        $global_work_size = [$rows];
        $local_work_size = null;
        $kernel->enqueueNDRange($this->queue,$global_work_size,$local_work_size,null,
            $events,$waitEvents);
    }

    public function softmax1(
        int $m,
        int $n,
        BufferInterface $A, int $offsetA, int $ldA,
        object $events=null, object $waitEvents=null
        ) : void
    {
        //$trans = false;  // disable transpose function yet
        $dtype = $A->dtype();
        //if($trans) {
        //    $trans = 'trans';
        //    $rows = $n;
        //    $cols = $m;
        //} else {
            $trans = 'norm';
            $rows = $m;
            $cols = $n;
        //}
        $total_local_items = $cols;
        $max_work_items = $this->maxWorkItem[0];
        if($total_local_items>$max_work_items) {
            throw new InvalidArgumentException('too many cols');
        } else {
            for($max_work_items=1; $max_work_items<$total_local_items;$max_work_items<<=1) {
                ;
            }
        }
        $value_size = $A->value_size();
        $type = $this->dtypeToOpenCLType[$dtype];
        $kernel_name = "softmax_S_{$type}_{$trans}";
        if(!isset($this->sources[$kernel_name])) {
            //if($trans=='trans') {
            //    $index_a = 'lid*lda+grid+offset_a';
            //} else {
                $index_a = 'grid*lda+lid+offset_a';
            //}
            $this->sources[$kernel_name] =
                "__kernel void {$kernel_name}(\n".
                "    const        uint total_local_items,\n".
                "        __global {$type} * a,\n".
                "    const        uint offset_a,\n".
                "    const        uint lda,\n".
                "         __local {$type} * local_work)\n".
                "{\n".
                "    const uint grid = get_group_id(0);\n".
                "    __local {$type} max;\n".
                "    __local {$type} sum;\n".
                     $this->kernelTemplateSMax(
                         "local_work[lid] = a[{$index_a}];\n",
                         "max = local_work[0];\n",
                         $dtype
                     ).
                     "barrier(CLK_LOCAL_MEM_FENCE);\n".
                     $this->kernelTemplateSSum(
                         "local_work[lid] = exp(a[{$index_a}]-max);",
                         "sum = local_work[0];\n"
                     ).
                     "barrier(CLK_LOCAL_MEM_FENCE);\n".
                "    {\n".
                "        const uint lid = get_local_id(0);\n".
                "        const uint lws = get_local_size(0);\n".
                "        if(lid<total_local_items) {\n".
                "            a[{$index_a}] = exp(a[{$index_a}]-max)/sum;\n".
                "        }\n".
                "    }\n".
                "}\n";
        }
        $kernel = $this->createKernel($kernel_name);

        $kernel->setArg(0,$total_local_items,NDArray::uint32);
        $kernel->setArg(1,$A);
        $kernel->setArg(2,$offsetA,NDArray::uint32);
        $kernel->setArg(3,$ldA,NDArray::uint32);
        $kernel->setArg(4,null,$this->adjBoundary($max_work_items*$value_size));
        $global_work_size = [$max_work_items*$rows];
        $local_work_size = [$max_work_items];
        $kernel->enqueueNDRange($this->queue,$global_work_size,$local_work_size,null,
            $events,$waitEvents);
    }

    public function softmax2(
        int $m,
        int $n,
        BufferInterface $A, int $offsetA, int $ldA,
        object $events=null, object $waitEvents=null
        ) : void
    {
        //$trans = false;  // disable transpose function yet
        $dtype = $A->dtype();
        //if($trans) {
        //    $trans = 'trans';
        //    $rows = $n;
        //    $cols = $m;
        //} else {
            $trans = 'norm';
            $rows = $m;
            $cols = $n;
        //}
        $total_local_items = $cols;
        $max_work_items = $this->maxWorkItem[0];
        if($total_local_items>$max_work_items) {
            $segments = (int)ceil($total_local_items/$max_work_items); // round up float
            $work_items = $max_work_items;
        } else {
            for($max_work_items=1; $max_work_items<$total_local_items;$max_work_items<<=1) {
                ;
            }
            $segments = 1; // round up float
            $work_items = $total_local_items;
        }
        $value_size = $A->value_size();
        $type = $this->dtypeToOpenCLType[$dtype];
        $kernel_name = "softmax_M_{$type}_{$trans}";
        if(!isset($this->sources[$kernel_name])) {
            //if($trans=='trans') {
            //    $index_a = '(seg*lws+lid)*lda+grid+offset_a';
            //} else {
                $index_a = 'grid*lda+(seg*lws+lid)+offset_a';
            //}
            $this->sources[$kernel_name] =
                "__kernel void {$kernel_name}(\n".
                "    const        uint total_local_items,\n".
                "    const        uint segments,\n".
                "        __global {$type} * a,\n".
                "    const        uint offset_a,\n".
                "    const        uint lda,\n".
                "         __local {$type} * local_work,\n".
                "         __local {$type} * seg_work,\n".
                "    const        uint work_items)\n".
                "{\n".
                "    const uint grid = get_group_id(0);\n".
                "    __local {$type} max;\n".
                "    __local {$type} sum;\n".
                     $this->kernelTemplateQMax(
                         "local_work[lid] = a[{$index_a}];\n",
                         "max = seg_work[0];\n",
                         $dtype
                     ).
                     "barrier(CLK_LOCAL_MEM_FENCE);\n".
                     $this->kernelTemplateQSum(
                         "local_work[lid] = exp(a[{$index_a}]-max);",
                         "sum = seg_work[0];\n"
                     ).
                     "barrier(CLK_LOCAL_MEM_FENCE);\n".
                "    {\n".
                "        const uint lid = get_local_id(0);\n".
                "        const uint lws = get_local_size(0);\n".
                "        uint seg_count = segments;\n".
                "        uint local_items = work_items;\n".
                "        uint left_local_items = total_local_items;\n".
                "        for(int seg=0;seg<seg_count;seg++) {\n".
                "            if(lid<local_items) {\n".
                "                a[{$index_a}] = exp(a[{$index_a}]-max)/sum;\n".
                "            }\n".
                "            barrier(CLK_LOCAL_MEM_FENCE);\n".
                "            left_local_items -= local_items;\n".
                "            if(left_local_items<local_items) {\n".
                "                local_items = left_local_items;\n".
                "            }\n".
                "        }\n".
                "    }\n".
                "}\n";
        }
        $kernel = $this->createKernel($kernel_name);

        $kernel->setArg(0,$total_local_items,NDArray::uint32);
        $kernel->setArg(1,$segments,NDArray::uint32);
        $kernel->setArg(2,$A);
        $kernel->setArg(3,$offsetA,NDArray::uint32);
        $kernel->setArg(4,$ldA,NDArray::uint32);

        $kernel->setArg(5,null,$this->adjBoundary($max_work_items*$value_size));
        $kernel->setArg(6,null,$this->adjBoundary($segments*$value_size));
        $kernel->setArg(7,$work_items,NDArray::uint32);
        $global_work_size = [$max_work_items*$rows];
        $local_work_size = [$max_work_items];
        $kernel->enqueueNDRange($this->queue,$global_work_size,$local_work_size,null,
            $events,$waitEvents);
    }

    /**
    * slice
    */
    public function slice(
        bool $reverse,
        bool $addMode,
        int $m,
        int $n,
        int $k,
        int $size,
        BufferInterface $A, int $offsetA, int $incA,
        BufferInterface $Y, int $offsetY, int $incY,
        int $startAxis0,
        int $sizeAxis0,
        int $startAxis1,
        int $sizeAxis1,
        int $startAxis2,
        int $sizeAxis2,
        object $events=null, object $waitEvents=null
        ) : void
    {
        if($A->dtype()!=$Y->dtype()) {
            throw new InvalidArgumentException("Unmatch data type A and Y:".
            $this->dtypeToString($A->dtype()).",".$this->dtypeToString($Y->dtype()));
        }
        if($A->dtype()==NDArray::float64) {
            $this->assertFP64();
        }
        if($addMode) {
            $op = 'add';
        } else {
            $op = 'set';
        }
        if($reverse) {
            $direction = 'r';
        } else {
            $direction = 'f';
        }
        $type = $this->dtypeToOpenCLType[$A->dtype()];
        $kernel_name = "slice_{$type}_{$direction}_{$op}";
        if(!isset($this->sources[$kernel_name])) {
            $y_variable = 'y[i0*i1size*i2size*size+'.
                            'i1*i2size*size+'.
                            'i2*size+lid*incy+offset_y]';
            $a_variable = 'a[(startAxis0+i0)*n*k*size+'.
                            '(startAxis1+i1)*k*size+'.
                            '(startAxis2+i2)*size+lid*inca+offset_a]';
            if($reverse) {
                $from = $y_variable;
                $to = $a_variable;
                $y_arg_type = 'const global';
                $a_arg_type = '__global';
            } else {
                $from = $a_variable;
                $to = $y_variable;
                $a_arg_type = 'const global';
                $y_arg_type = '__global';
            }
            if($addMode) {
                $operator = '+=';
            } else {
                $operator = '=';
            }
            $this->sources[$kernel_name] =
                "__kernel void {$kernel_name}(\n".
                "    const        uint n,\n".
                "    const        uint k,\n".
                "    $a_arg_type {$type} * a,\n".
                "    const        uint offset_a,\n".
                "    const        uint inca,\n".
                "    $y_arg_type {$type} * y,\n".
                "    const        uint offset_y,\n".
                "    const        uint incy,\n".
                "    const        uint startAxis0,\n".
                "    const        uint startAxis1,\n".
                "    const        uint startAxis2,\n".
                "    const        uint i0size,\n".
                "    const        uint i1size,\n".
                "    const        uint i2size,\n".
                "    const        uint size)\n".
                "{\n".
                "    uint gid0 = get_global_id(0);\n".
                "    uint gid1 = get_global_id(1);\n".
                    $this->splitPointer('lid','i2','gid0','size').
                    $this->splitPointer('i1','i0','gid1','i1size').
                "    {$to} {$operator} {$from};\n".
                "}\n";
        }

        $kernel = $this->createKernel($kernel_name);
        $kernel->setArg(0,$n,NDArray::uint32);
        $kernel->setArg(1,$k,NDArray::uint32);
        $kernel->setArg(2,$A);
        $kernel->setArg(3,$offsetA,NDArray::uint32);
        $kernel->setArg(4,$incA,NDArray::uint32);
        $kernel->setArg(5,$Y);
        $kernel->setArg(6,$offsetY,NDArray::uint32);
        $kernel->setArg(7,$incY,NDArray::uint32);
        $kernel->setArg(8,$startAxis0,NDArray::uint32);
        $kernel->setArg(9,$startAxis1,NDArray::uint32);
        $kernel->setArg(10,$startAxis2,NDArray::uint32);
        $kernel->setArg(11,$sizeAxis0,NDArray::uint32);
        $kernel->setArg(12,$sizeAxis1,NDArray::uint32);
        $kernel->setArg(13,$sizeAxis2,NDArray::uint32);
        $kernel->setArg(14,$size,NDArray::uint32);
        $global_work_size = [$size*$sizeAxis2,$sizeAxis1*$sizeAxis0];
        $local_work_size=null;
        $kernel->enqueueNDRange($this->queue,$global_work_size,$local_work_size,null,
            $events,$waitEvents);
    }

    /**
     *     X := searchsorted(A,X)
     */
    public function searchsorted(
        int $m,
        int $n,
        BufferInterface $A, int $offsetA, int $ldA, // float
        BufferInterface $X, int $offsetX, int $incX, // float
        bool $right,
        BufferInterface $Y, int $offsetY, int $incY, // int
        object $events=null, object $waitEvents=null
        ) : void
    {
        $dtypeX = $X->dtype();
        $dtypeY = $Y->dtype();
        if($dtypeX==NDArray::float64) {
            $this->assertFP64();
        }
        if($dtypeY!=NDArray::int32&&$dtypeY!=NDArray::uint32) {
            throw new InvalidArgumentException('dtype of Y must be int32.');
        }
        if($A->dtype()!=$X->dtype()) {
            throw new InvalidArgumentException('dtype of A and X must be same.');
        }
        if($right) {
            $subname = 'r';
            $cmp = ">=";
        } else {
            $subname = 'l';
            $cmp = ">";
        }
        $type = $this->dtypeToOpenCLType[$dtypeX];
        $kernel_name = "searchsorted_{$type}_{$subname}";
        if(!isset($this->sources[$kernel_name])) {
            $this->sources[$kernel_name] =
                "__kernel void {$kernel_name}(\n".
                "    const        uint n,\n".
                "        __global {$type} * a,\n".
                "    const        uint offset_a,\n".
                "    const        uint lda,\n".
                "        __global {$type} * x,\n".
                "    const        uint offset_x,\n".
                "    const        uint incx,\n".
                "        __global uint * y,\n".
                "    const        uint offset_y,\n".
                "    const        uint incy)\n".
                "{\n".
                "    uint ida = get_global_id(0)*lda+offset_a;\n".
                "    uint idx = get_global_id(0)*incx+offset_x;\n".
                "    uint idy = get_global_id(0)*incy+offset_y;\n".
                "    uint i;\n".
                "    {$type} v = x[idx];\n".
                "    for(i=0;i<n;i++) {\n".
                "        if(!(v {$cmp} a[ida])) {\n".
                "            break;\n".
                "        }\n".
                "        ++ida;\n".
                "    }\n".
                "    y[idy] = i;\n".
                "}\n";
        }

        $kernel = $this->createKernel($kernel_name);
        $kernel->setArg(0,$n,NDArray::uint32);
        $kernel->setArg(1,$A);
        $kernel->setArg(2,$offsetA,NDArray::uint32);
        $kernel->setArg(3,$ldA,NDArray::uint32);
        $kernel->setArg(4,$X);
        $kernel->setArg(5,$offsetX,NDArray::uint32);
        $kernel->setArg(6,$incX,NDArray::uint32);
        $kernel->setArg(7,$Y);
        $kernel->setArg(8,$offsetY,NDArray::uint32);
        $kernel->setArg(9,$incY,NDArray::uint32);
        $global_work_size = [$m];
        $kernel->enqueueNDRange($this->queue,$global_work_size,null,null,
            $events,$waitEvents);
    }

    /**
     *     Y(n) := X(n) + Y(n-1)
     */
    public function cumsum(
        int $n,
        BufferInterface $X, int $offsetX, int $incX, // float
        bool $exclusive,
        bool $reverse,
        BufferInterface $Y, int $offsetY, int $incY, // float
        object $events=null, object $waitEvents=null
        ) : void
    {
        $dtypeX = $X->dtype();
        $dtypeY = $Y->dtype();
        if($dtypeX==NDArray::float64) {
            $this->assertFP64();
        }
        if($X->dtype()!=$Y->dtype()) {
            throw new InvalidArgumentException('dtype of X and Y must be same.');
        }
        if($reverse) {
            $offsetY = $offsetY+$incY*($n-1);
            $incY = -$incY;
        }
        if($exclusive) {
            $execute =
            "        y[idy] = value;\n".
            "        value += x[idx];\n";
            $subname = 'e';
        } else {
            $execute =
            "        value += x[idx];\n".
            "        y[idy] = value;\n";
            $subname = 'n';
        }
        $type = $this->dtypeToOpenCLType[$dtypeX];
        $kernel_name = "cumsum_{$type}_{$subname}";
        if(!isset($this->sources[$kernel_name])) {
            $this->sources[$kernel_name] =
                "__kernel void {$kernel_name}(\n".
                "    const        int n,\n".
                "        __global {$type} * x,\n".
                "    const        int offset_x,\n".
                "    const        int incx,\n".
                "        __global {$type} * y,\n".
                "    const        int offset_y,\n".
                "    const        int incy)\n".
                "{\n".
                "    int idx = offset_x;\n".
                "    int idy = offset_y;\n".
                "    {$type} value = 0;\n".
                "    for(int i=0;i<n;i++,idx+=incx,idy+=incy) {\n".
                        $execute.
                "    }\n".
                "}\n";
        }

        $kernel = $this->createKernel($kernel_name);
        $kernel->setArg(0,$n,NDArray::int32);
        $kernel->setArg(1,$X);
        $kernel->setArg(2,$offsetX,NDArray::int32);
        $kernel->setArg(3,$incX,NDArray::int32);
        $kernel->setArg(4,$Y);
        $kernel->setArg(5,$offsetY,NDArray::int32);
        $kernel->setArg(6,$incY,NDArray::int32);
        $global_work_size = [1];
        $kernel->enqueueNDRange($this->queue,$global_work_size,null,null,
            $events,$waitEvents);
    }

    /**
     *     X := nan2num(X)
     */
    public function nan2num(
        int $n,
        BufferInterface $X, int $offsetX, int $incX,
        float $alpha,
        object $events=null, object $waitEvents=null
        ) : void
    {
        $dtypeX = $X->dtype();
        if($dtypeX==NDArray::float64) {
            $this->assertFP64();
        }
        $type = $this->dtypeToOpenCLType[$dtypeX];
        $kernel_name = "nan2num_{$type}";
        if(!isset($this->sources[$kernel_name])) {
            $this->sources[$kernel_name] =
                "__kernel void {$kernel_name}(\n".
                "    const        {$type} alpha,\n".
                "        __global {$type} * x,\n".
                "    const        uint offset_x,\n".
                "    const        uint incx)\n".
                "{\n".
                "    uint idx = get_global_id(0)*incx+offset_x;\n".
                "    if(isnan(x[idx])) {\n".
                "        x[idx] = alpha;\n".
                "    }\n".
                "}\n";
        }
        $kernel = $this->createKernel($kernel_name);
        $kernel->setArg(0,$alpha,NDArray::float32);
        $kernel->setArg(1,$X);
        $kernel->setArg(2,$offsetX,NDArray::uint32);
        $kernel->setArg(3,$incX,NDArray::uint32);
        $global_work_size = [$n];
        $kernel->enqueueNDRange($this->queue,$global_work_size,null,null,
            $events,$waitEvents);
    }

    /**
     *     X := isnan(X)
     */
    public function isnan(
        int $n,
        BufferInterface $X, int $offsetX, int $incX,
        object $events=null, object $waitEvents=null
        ) : void
    {
        $dtypeX = $X->dtype();
        if($dtypeX==NDArray::float64) {
            $this->assertFP64();
        }
        $type = $this->dtypeToOpenCLType[$dtypeX];
        $kernel_name = "isnan_{$type}";
        if(!isset($this->sources[$kernel_name])) {
            $this->sources[$kernel_name] =
                "__kernel void {$kernel_name}(\n".
                "        __global {$type} * x,\n".
                "    const        uint offset_x,\n".
                "    const        uint incx)\n".
                "{\n".
                "    uint idx = get_global_id(0)*incx+offset_x;\n".
                "    if(isnan(x[idx])) {\n".
                "        x[idx] = 1.0;\n".
                "    } else {\n".
                "        x[idx] = 0.0;\n".
                "    }\n".
                "}\n";
        }
        $kernel = $this->createKernel($kernel_name);
        $kernel->setArg(0,$X);
        $kernel->setArg(1,$offsetX,NDArray::uint32);
        $kernel->setArg(2,$incX,NDArray::uint32);
        $global_work_size = [$n];
        $kernel->enqueueNDRange($this->queue,$global_work_size,null,null,
            $events,$waitEvents);
    }

    /**
    * imagecopy
    */
    public function imagecopy(
        int $height,
        int $width,
        int $channels,
        BufferInterface $A, int $offsetA,
        BufferInterface $B, int $offsetB,
        bool $channelsFirst,
        int $heightShift,
        int $widthShift,
        bool $verticalFlip,
        bool $horizontalFlip,
        bool $rgbFlip,
        object $events=null, object $waitEvents=null
        ) : void
    {
        if($A->dtype()!=$B->dtype()) {
            throw new InvalidArgumentException("Unmatch data type A and B:".
            $this->dtypeToString($A->dtype()).",".$this->dtypeToString($B->dtype()));
        }
        if($A->dtype()==NDArray::float64) {
            $this->assertFP64();
        }

        if($channelsFirst) {
            $ldC = $width*$height;
            $ldY = $width;
            $ldX = 1;
        } else {
            $ldY = $width*$channels;
            $ldX = $channels;
            $ldC = 1;
        }
        $directionY = $directionX = 1;
        $biasY = $biasX = 0;
        if($verticalFlip) {
            $directionY = -$directionY;
            $biasY = $height-1;
        }
        if($horizontalFlip) {
            $directionX = -$directionX;
            $biasX = $width-1;
        }
        if($rgbFlip) {
            $rgbFlip = 1;
        } else {
            $rgbFlip = 0;
        }
        $biasY -= $heightShift*$directionY;
        $biasX -= $widthShift*$directionX;

        $type = $this->dtypeToOpenCLType[$A->dtype()];
        $kernel_name = "imagecopy_{$type}";
        if(!isset($this->sources[$kernel_name])) {
            $this->sources[$kernel_name] =
                "__kernel void {$kernel_name}(\n".
                "    const        int height,\n".
                "    const        int width,\n".
                "    const        int directionY,\n".
                "    const        int biasY,\n".
                "    const        int directionX,\n".
                "    const        int biasX,\n".
                "    const        int ldY,\n".
                "    const        int ldX,\n".
                "    const        int ldC,\n".
                "    const        int rgbFlip,\n".
                "    const global {$type} * a,\n".
                "    const        uint offset_a,\n".
                "        __global {$type} * b,\n".
                "    const        uint offset_b)\n".
                "{\n".
                "    uint gid0 = get_global_id(0);\n".
                "    uint gid1 = get_global_id(1);\n".
                "    int c  = gid0;\n".
                    $this->splitPointer('x','y','gid1','width').
                "    int sy = y*directionY+biasY;\n".
                "    int sx = x*directionX+biasX;\n".
                "    if(sy<0) {\n".
                "        sy = 0;\n".
                "    } else if(sy>=height) {\n".
                "        sy = height-1;\n".
                "    }\n".
                "    if(sx<0) {\n".
                "        sx = 0;\n".
                "    } else if(sx>=width) {\n".
                "        sx = width-1;\n".
                "    }\n".
                "    int srcC = (rgbFlip&&c<3)?(2-c):c;\n".
                "    b[y*ldY+x*ldX+c*ldC+offset_b] =\n".
                "        a[sy*ldY+sx*ldX+srcC*ldC+offset_a];\n".
                "}\n";
        }

        $kernel = $this->createKernel($kernel_name);
        $kernel->setArg(0,$height,NDArray::int32);
        $kernel->setArg(1,$width,NDArray::int32);
        $kernel->setArg(2,$directionY,NDArray::int32);
        $kernel->setArg(3,$biasY,NDArray::int32);
        $kernel->setArg(4,$directionX,NDArray::int32);
        $kernel->setArg(5,$biasX,NDArray::int32);
        $kernel->setArg(6,$ldY,NDArray::int32);
        $kernel->setArg(7,$ldX,NDArray::int32);
        $kernel->setArg(8,$ldC,NDArray::int32);
        $kernel->setArg(9,$rgbFlip,NDArray::int32);
        $kernel->setArg(10,$A);
        $kernel->setArg(11,$offsetA,NDArray::uint32);
        $kernel->setArg(12,$B);
        $kernel->setArg(13,$offsetB,NDArray::uint32);
        $global_work_size = [$channels,$width*$height];
        $local_work_size=null;
        $kernel->enqueueNDRange($this->queue,$global_work_size,$local_work_size,null,
            $events,$waitEvents);
    }

    /**
    * images: (n,w,c) : channels_last
    *         (n,c,w) : channels_first
    * kernels:
    * strides:
    * padding:
    * dilation:
    * data_format:
    * output:(n,ow,kw,c)
    */
    public function im2col1d(
        bool $reverse,
        BufferInterface $images,
        int $images_offset,
        int $images_size,
        int $batches,
        int $im_w,
        int $channels,
        int $kernel_w,
        int $stride_w,
        bool $padding,
        bool $channels_first,
        int $dilation_w,
        bool $cols_channels_first,
        BufferInterface $cols,
        int $cols_offset,
        int $cols_size,
        object $events=null, object $waitEvents=null
        ) : void
    {
        $dtype = $images->dtype();
        if($dtype!=$cols->dtype()) {
            throw new InvalidArgumentException("Unmatch data type images and cols:".
            $this->dtypeToString($dtype).",".$this->dtypeToString($cols->dtype()));
        }
        if($dtype==NDArray::float64) {
            $this->assertFP64();
        }

        $output_w = intdiv(($im_w-($kernel_w-1)*$dilation_w-1),$stride_w)+1;
        if($padding) {
            $pad_w = intdiv((($im_w-1)*$stride_w-$im_w+($kernel_w-1)*$dilation_w+1),2);
            $output_w = $im_w;
        } else {
            $pad_w = 0;
        }
        if($channels_first) {
            $channel_mode = 'cf';
        } else {
            $channel_mode = 'cl';
        }
        if($cols_channels_first) {
            $cols_mode = 'cf';
        } else {
            $cols_mode = 'cl';
        }
        if($reverse) {
            $tmode = 'B';
        } else {
            $tmode = 'F';
        }
        $type = $this->dtypeToOpenCLType[$dtype];
        $kernel_name = "im2col1d_{$tmode}_{$type}_{$channel_mode}_{$cols_mode}";
        if(!isset($this->sources[$kernel_name])) {
            if($reverse) {
                $col_arg_type = 'const global';
                $im_arg_type = '__global';
            } else {
                $im_arg_type = 'const global';
                $col_arg_type = '__global';
            }
            if($channels_first) {
                $input_id = '((batch_id*channels+channel_id)*im_w+input_x)';
            } else {
                $input_id = '((batch_id*im_w+input_x)*channels+channel_id)';
            }
            if($cols_channels_first) {
                $cols_id = '(((batch_id*output_w+im_x)'.
                            '*channels+channel_id)*kernel_w+kernel_x)';
            } else {
                $cols_id = '(((batch_id*output_w+im_x)'.
                            '*kernel_w+kernel_x)*channels+channel_id)';
            }
            $input_x = '(im_x*stride_w+kernel_x*dilation_w-pad_w)';
            if($reverse) {
                $this->sources[$kernel_name] =
                    "__kernel void {$kernel_name}(\n".
                    "    $im_arg_type {$type} * images,\n".
                    "    const        uint offset_images,\n".
                    "    $col_arg_type {$type} * cols,\n".
                    "    const        uint offset_cols,\n".
                    "    const        uint batches,\n".
                    "    const        uint im_w,\n".
                    "    const        uint channels,\n".
                    "    const        uint kernel_w,\n".
                    "    const        uint stride_w,\n".
                    "    const        uint pad_w,\n".
                    "    const        uint dilation_w,\n".
                    "    const        uint output_w)\n".
                    "{\n".
                    "    const uint gid0 = get_global_id(0);\n".
                    "    const uint gid1 = get_global_id(1);\n".
                        $this->splitPointer('channel_id','input_x','gid0','channels').
                    "    const uint batch_id   = gid1;\n".
                    "    if(input_x<im_w && batch_id<batches){\n".
                    "        {$type} value=0;\n".
                    "        for(int kernel_x=0;kernel_x<kernel_w;kernel_x++) {\n".
                    "            const int tmp_x = input_x-kernel_x*dilation_w+pad_w;\n".
                    "            const int im_x = tmp_x/stride_w;\n". // div5bug
                    "            if(tmp_x%stride_w==0 &&\n". // div5bug
                    "               im_x>=0 && im_x<output_w) {\n".
                    "                value += cols[{$cols_id}];\n".
                    "            }\n".
                    "        }\n".
                    "        images[{$input_id}] += value;\n".
                    "    }\n".
                    "}\n";
            } else {
                $this->sources[$kernel_name] =
                    "__kernel void {$kernel_name}(\n".
                    "    $im_arg_type {$type} * images,\n".
                    "    const        uint offset_images,\n".
                    "    $col_arg_type {$type} * cols,\n".
                    "    const        uint offset_cols,\n".
                    "    const        uint batches,\n".
                    "    const        uint im_w,\n".
                    "    const        uint channels,\n".
                    "    const        uint kernel_w,\n".
                    "    const        uint stride_w,\n".
                    "    const        uint pad_w,\n".
                    "    const        uint dilation_w,\n".
                    "    const        uint output_w)\n".
                    "{\n".
                    "    const uint gid = get_global_id(0);\n".
                        $this->splitPointer('channel_id','im_x','gid','channels').
                    "    uint batch_id = get_global_id(1);\n".
                    //"    uint kernel_x = get_global_id(2);\n".
                    "    if(im_x<output_w && batch_id<batches){\n".
                    "        for(uint kernel_x=0;kernel_x<kernel_w;kernel_x++) {\n".
                    "            int input_x = {$input_x};\n".
                    "            {$type} value;\n".
                    "            if(input_x>=0 && input_x<im_w) {\n".
                    "                uint input_id = {$input_id};\n".
                    "                value = images[input_id];\n".
                    "            } else {\n".
                    "                value = 0;\n".
                    "            }\n".
                    "            uint cols_id = {$cols_id};\n".
                    "            cols[cols_id] = value;\n".
                    "        }\n".
                    "    }\n".
                    "}\n";
            }
        }

        $kernel = $this->createKernel($kernel_name);
        $kernel->setArg(0,$images);
        $kernel->setArg(1,$images_offset,NDArray::uint32);
        $kernel->setArg(2,$cols);
        $kernel->setArg(3,$cols_offset,NDArray::uint32);
        $kernel->setArg(4,$batches,NDArray::uint32);
        $kernel->setArg(5,$im_w,NDArray::uint32);
        $kernel->setArg(6,$channels,NDArray::uint32);
        $kernel->setArg(7,$kernel_w,NDArray::uint32);
        $kernel->setArg(8,$stride_w,NDArray::uint32);
        $kernel->setArg(9,$pad_w,NDArray::uint32);
        $kernel->setArg(10,$dilation_w,NDArray::uint32);
        $kernel->setArg(11,$output_w,NDArray::uint32);
        if($reverse) {
            $global_work_size = [$this->ceil($im_w*$channels,4),$this->ceil($batches,8)];
            $local_work_size = [4,8];
        } else {
            $global_work_size = [$this->ceil($output_w*$channels,4),$this->ceil($batches,8)];
            $local_work_size=[4,8];
        }
        $kernel->enqueueNDRange($this->queue,$global_work_size,$local_work_size,null,
            $events,$waitEvents);
    }

    /**
    * images: (n,h,w,c) : channels_last
    *        (n,c,h,w) : channels_first
    * kernels:
    * strides:
    * padding:
    * dilation:
    * data_format:
    * output:(n,oh,ow,kh,kw,c)
    */
    public function im2col2d(
        bool $reverse,
        BufferInterface $images,
        int $images_offset,
        int $images_size,
        int $batches,
        int $im_h,
        int $im_w,
        int $channels,
        int $kernel_h,
        int $kernel_w,
        int $stride_h,
        int $stride_w,
        bool $padding,
        bool $channels_first,
        int $dilation_h,
        int $dilation_w,
        bool $cols_channels_first,
        BufferInterface $cols,
        int $cols_offset,
        int $cols_size,
        object $events=null, object $waitEvents=null
        ) : void
    {
        $dtype = $images->dtype();
        if($dtype!=$cols->dtype()) {
            throw new InvalidArgumentException("Unmatch data type images and cols:".
            $this->dtypeToString($dtype).",".$this->dtypeToString($cols->dtype()));
        }
        if($dtype==NDArray::float64) {
            $this->assertFP64();
        }

        $output_h = intdiv(($im_h-($kernel_h-1)*$dilation_h-1),$stride_h)+1;
        $output_w = intdiv(($im_w-($kernel_w-1)*$dilation_w-1),$stride_w)+1;
        if($padding) {
            $pad_h = intdiv((($im_h-1)*$stride_h-$im_h+($kernel_h-1)*$dilation_h+1),2);
            $pad_w = intdiv((($im_w-1)*$stride_w-$im_w+($kernel_w-1)*$dilation_w+1),2);
            $output_h = $im_h;
            $output_w = $im_w;
        } else {
            $pad_h = $pad_w = 0;
        }
        if($channels_first) {
            $channel_mode = 'cf';
        } else {
            $channel_mode = 'cl';
        }
        if($cols_channels_first) {
            $cols_mode = 'cf';
        } else {
            $cols_mode = 'cl';
        }
        if($reverse) {
            $tmode = 'R';
        } else {
            $tmode = 'F';
        }
        $type = $this->dtypeToOpenCLType[$dtype];
        $kernel_name = "im2col2d_{$tmode}_{$type}_{$channel_mode}_{$cols_mode}";
        if(!isset($this->sources[$kernel_name])) {
            if($channels_first) {
                $input_id = '(((batch_id*channels+channel_id)*im_h+input_y)*im_w+input_x)';
            } else {
                $input_id = '(((batch_id*im_h+input_y)*im_w+input_x)*channels+channel_id)';
            }
            if($cols_channels_first) {
                $cols_id = '(((((batch_id*output_h+im_y)*output_w+im_x)'.
                            '*channels+channel_id)*kernel_h+kernel_y)*kernel_w+kernel_x)';
            } else {
                $cols_id = '(((((batch_id*output_h+im_y)*output_w+im_x)'.
                            '*kernel_h+kernel_y)*kernel_w+kernel_x)*channels+channel_id)';
            }
            $input_y = '(im_y*stride_h+kernel_y*dilation_h-pad_h)';
            $input_x = '(im_x*stride_w+kernel_x*dilation_w-pad_w)';
            if($reverse) {
                $col_arg_type = 'const global';
                $im_arg_type = '__global';
                $this->sources[$kernel_name] =
                    "__kernel void {$kernel_name}(\n".
                    "    $im_arg_type {$type} * images,\n".
                    "    const        uint offset_images,\n".
                    "    $col_arg_type {$type} * cols,\n".
                    "    const        uint offset_cols,\n".
                    "    const        uint batches,\n".
                    "    const        uint im_h,\n".
                    "    const        uint im_w,\n".
                    "    const        uint channels,\n".
                    "    const        uint kernel_h,\n".
                    "    const        uint kernel_w,\n".
                    "    const        uint stride_h,\n".
                    "    const        uint stride_w,\n".
                    "    const        uint pad_h,\n".
                    "    const        uint pad_w,\n".
                    "    const        uint dilation_h,\n".
                    "    const        uint dilation_w,\n".
                    "    const        uint output_h,\n".
                    "    const        uint output_w\n".
                    "    )\n".
                    "{\n".
                    "    const uint gid0 = get_global_id(0);\n".
                    "    const uint gid1 = get_global_id(1);\n".
                         $this->splitPointer('channel_id','input_x','gid0','channels').
                         $this->splitPointer('input_y','batch_id','gid1','im_h').
                    //"    const uint kernel_idx = {$kernel_idx};\n".
                    //"    const uint kernel_y = kernel_idx/kernel_w;\n".
                    //"    const uint kernel_x = kernel_idx%kernel_w;\n".
                    "    if(input_x<im_w && batch_id<batches){\n".
                    "        {$type} value=0;\n".
                    "        for(int kernel_y=0;kernel_y<kernel_h;kernel_y++) {".
                    "            for(int kernel_x=0;kernel_x<kernel_w;kernel_x++) {".
                    "                const int tmp_y = input_y-kernel_y*dilation_h+pad_h;\n".
                    "                const int tmp_x = input_x-kernel_x*dilation_w+pad_w;\n".
                    "                const int im_y = tmp_y/stride_h;\n". // div5bug
                    "                const int im_x = tmp_x/stride_w;\n". // div5bug
                    "                if(tmp_y%stride_h==0 && tmp_x%stride_w==0 &&\n". // div5bug
                    "                   im_y>=0 && im_y<output_h && im_x>=0 && im_x<output_w) {\n".
                    "                    value += cols[{$cols_id}];\n".
                    "                }\n".
                    "            }\n".
                    "        }\n".
                    "        images[{$input_id}] += value;\n".
                    "    }\n".
                    "}\n";
            } else {
                $im_arg_type = 'const global';
                $col_arg_type = '__global';
                $this->sources[$kernel_name] =
                    "__kernel void {$kernel_name}(\n".
                    "    $im_arg_type {$type} * images,\n".
                    "    const        uint offset_images,\n".
                    "    $col_arg_type {$type} * cols,\n".
                    "    const        uint offset_cols,\n".
                    "    const        uint batches,\n".
                    "    const        uint im_h,\n".
                    "    const        uint im_w,\n".
                    "    const        uint channels,\n".
                    "    const        uint kernel_h,\n".
                    "    const        uint kernel_w,\n".
                    "    const        uint stride_h,\n".
                    "    const        uint stride_w,\n".
                    "    const        uint pad_h,\n".
                    "    const        uint pad_w,\n".
                    "    const        uint dilation_h,\n".
                    "    const        uint dilation_w,\n".
                    "    const        uint output_h,\n".
                    "    const        uint output_w\n".
                    "    )\n".
                    "{\n".
                    "    const uint gid0 = get_global_id(0);\n".
                    "    const uint gid1 = get_global_id(1);\n".
                         $this->splitPointer('channel_id','im_x','gid0','channels').
                         $this->splitPointer('im_y','batch_id','gid1','output_h').
                    "    if(im_x<output_w && batch_id<batches){\n".
                    "        for(uint kernel_y=0;kernel_y<kernel_h;kernel_y++) {\n".
                    "            for(uint kernel_x=0;kernel_x<kernel_w;kernel_x++) {\n".
                    "                int input_y = {$input_y};\n".
                    "                int input_x = {$input_x};\n".
                    "                {$type} value;\n".
                    "                if(input_y>=0 && input_y<im_h && input_x>=0 && input_x<im_w) {\n".
                    "                    uint input_id = {$input_id};\n".
                    "                    value = images[input_id];\n".
                    "                } else {\n".
                    "                    value = 0;\n".
                    "                }\n".
                    "                uint cols_id = {$cols_id};\n".
                    "                cols[cols_id] = value;\n".
                    "            }\n".
                    "        }\n".
                    "    }\n".
                    "}\n";
            }
        }
        $kernel = $this->createKernel($kernel_name);
        $kernel->setArg(0,$images);
        $kernel->setArg(1,$images_offset,NDArray::uint32);
        $kernel->setArg(2,$cols);
        $kernel->setArg(3,$cols_offset,NDArray::uint32);
        $kernel->setArg(4,$batches,NDArray::uint32);
        $kernel->setArg(5,$im_h,NDArray::uint32);
        $kernel->setArg(6,$im_w,NDArray::uint32);
        $kernel->setArg(7,$channels,NDArray::uint32);
        $kernel->setArg(8,$kernel_h,NDArray::uint32);
        $kernel->setArg(9,$kernel_w,NDArray::uint32);
        $kernel->setArg(10,$stride_h,NDArray::uint32);
        $kernel->setArg(11,$stride_w,NDArray::uint32);
        $kernel->setArg(12,$pad_h,NDArray::uint32);
        $kernel->setArg(13,$pad_w,NDArray::uint32);
        $kernel->setArg(14,$dilation_h,NDArray::uint32);
        $kernel->setArg(15,$dilation_w,NDArray::uint32);
        $kernel->setArg(16,$output_h,NDArray::uint32);
        $kernel->setArg(17,$output_w,NDArray::uint32);
        if($reverse) {
            $global_work_size = [$this->ceil($im_w*$channels,4),$this->ceil($batches*$im_h,8)];
            $local_work_size = [4,8];
        } else {
            $global_work_size = [$this->ceil($output_w*$channels,4),$this->ceil($batches*$output_h,8)];
            $local_work_size=[4,8];
            //$global_work_size = [$output_w*$channels,$batches*$output_h];
            //$local_work_size=null;
        }
        $kernel->enqueueNDRange($this->queue,$global_work_size,$local_work_size,null,
            $events,$waitEvents);
    }

    /**
    * images: (n,d,h,w,c) : channels_last
    *         (n,c,d,h,w) : channels_first
    * kernels:
    * strides:
    * padding:
    * dilation:
    * data_format:
    * output: (n,od,oh,ow,kh,kw,c) : cols_channels_last
    *         (n,od,oh,ow,c,kh,kw) : cols_channels_first
    */
    public function im2col3d(
        bool $reverse,
        BufferInterface $images,
        int $images_offset,
        int $images_size,
        int $batches,
        int $im_d,
        int $im_h,
        int $im_w,
        int $channels,
        int $kernel_d,
        int $kernel_h,
        int $kernel_w,
        int $stride_d,
        int $stride_h,
        int $stride_w,
        bool $padding,
        bool $channels_first,
        int $dilation_d,
        int $dilation_h,
        int $dilation_w,
        bool $cols_channels_first,
        BufferInterface $cols,
        int $cols_offset,
        int $cols_size,
        object $events=null, object $waitEvents=null
        ) : void
    {
        $dtype = $images->dtype();
        if($dtype!=$cols->dtype()) {
            throw new InvalidArgumentException("Unmatch data type images and cols:".
            $this->dtypeToString($dtype).",".$this->dtypeToString($cols->dtype()));
        }
        if($dtype==NDArray::float64) {
            $this->assertFP64();
        }

        $output_d = intdiv(($im_d-($kernel_d-1)*$dilation_d-1),$stride_d)+1;
        $output_h = intdiv(($im_h-($kernel_h-1)*$dilation_h-1),$stride_h)+1;
        $output_w = intdiv(($im_w-($kernel_w-1)*$dilation_w-1),$stride_w)+1;
        if($padding) {
            $pad_d = intdiv((($im_d-1)*$stride_d-$im_d+($kernel_d-1)*$dilation_d+1),2);
            $pad_h = intdiv((($im_h-1)*$stride_h-$im_h+($kernel_h-1)*$dilation_h+1),2);
            $pad_w = intdiv((($im_w-1)*$stride_w-$im_w+($kernel_w-1)*$dilation_w+1),2);
            $output_d = $im_d;
            $output_h = $im_h;
            $output_w = $im_w;
        } else {
            $pad_d = $pad_h = $pad_w = 0;
        }

        if($channels_first) {
            $channel_mode = 'cf';
        } else {
            $channel_mode = 'cl';
        }
        if($cols_channels_first) {
            $cols_mode = 'cf';
        } else {
            $cols_mode = 'cl';
        }
        if($reverse) {
            $tmode = 'B';
        } else {
            $tmode = 'F';
        }
        $type = $this->dtypeToOpenCLType[$dtype];
        $kernel_name = "im2col3d_{$tmode}_{$type}_{$channel_mode}_{$cols_mode}";
        if(!isset($this->sources[$kernel_name])) {
            if($reverse) {
                $col_arg_type = 'const global';
                $im_arg_type = '__global';
            } else {
                $im_arg_type = 'const global';
                $col_arg_type = '__global';
            }
            if($channels_first) {
                $input_id = '((((batch_id*channels+channel_id)*im_d+input_z)*im_h+input_y)*im_w+input_x)';
            } else {
                $input_id = '((((batch_id*im_d+input_z)*im_h+input_y)*im_w+input_x)*channels+channel_id)';
            }
            if($cols_channels_first) {
                $cols_id = '(((((((batch_id*output_d+im_z)*output_h+im_y)*output_w+im_x)'.
                            '*channels+channel_id)*kernel_d+kernel_z)*kernel_h+kernel_y)*kernel_w+kernel_x)';
            } else {
                $cols_id = '(((((((batch_id*output_d+im_z)*output_h+im_y)*output_w+im_x)'.
                            '*kernel_d+kernel_z)*kernel_h+kernel_y)*kernel_w+kernel_x)*channels+channel_id)';
            }
            $input_z = '(im_z*stride_d+kernel_z*dilation_d-pad_d)';
            $input_y = '(im_y*stride_h+kernel_y*dilation_h-pad_h)';
            $input_x = '(im_x*stride_w+kernel_x*dilation_w-pad_w)';
            if($reverse) {
                $this->sources[$kernel_name] =
                    "__kernel void {$kernel_name}(\n".
                    "    $im_arg_type {$type} * images,\n".
                    "    const        uint offset_images,\n".
                    "    $col_arg_type {$type} * cols,\n".
                    "    const        uint offset_cols,\n".
                    "    const        uint batches,\n".
                    "    const        uint im_d,\n".
                    "    const        uint im_h,\n".
                    "    const        uint im_w,\n".
                    "    const        uint channels,\n".
                    "    const        uint kernel_d,\n".
                    "    const        uint kernel_h,\n".
                    "    const        uint kernel_w,\n".
                    "    const        uint stride_d,\n".
                    "    const        uint stride_h,\n".
                    "    const        uint stride_w,\n".
                    "    const        uint pad_d,\n".
                    "    const        uint pad_h,\n".
                    "    const        uint pad_w,\n".
                    "    const        uint dilation_d,\n".
                    "    const        uint dilation_h,\n".
                    "    const        uint dilation_w,\n".
                    "    const        uint output_d,\n".
                    "    const        uint output_h,\n".
                    "    const        uint output_w)\n".
                    "{\n".
                    "    const uint gid0 = get_global_id(0);\n".
                    "    const uint gid1 = get_global_id(1);\n".
                    "    const uint gid2 = get_global_id(2);\n".
                         $this->splitPointer('channel_id','input_x','gid0','channels').
                         $this->splitPointer('input_y','input_z','gid1','im_h').
                    "    const int batch_id   = gid2;\n".
                    "    if(input_x<im_w && input_z<im_d && batch_id<batches){\n".
                    "        {$type} value=0;\n".
                    "        for(int kernel_z=0;kernel_z<kernel_d;kernel_z++) {".
                    "            for(int kernel_y=0;kernel_y<kernel_h;kernel_y++) {".
                    "                for(int kernel_x=0;kernel_x<kernel_w;kernel_x++) {".
                    "                    const int tmp_z = input_z-kernel_z*dilation_d+pad_d;\n".
                    "                    const int tmp_y = input_y-kernel_y*dilation_h+pad_h;\n".
                    "                    const int tmp_x = input_x-kernel_x*dilation_w+pad_w;\n".
                    "                    const int im_z = tmp_z/stride_d;\n". // div5bug
                    "                    const int im_y = tmp_y/stride_h;\n". // div5bug
                    "                    const int im_x = tmp_x/stride_w;\n". // div5bug
                    "                    if(tmp_z%stride_d==0 && tmp_y%stride_h==0 && tmp_x%stride_w==0 &&\n". // div5bug
                    "                        im_z>=0 && im_z<output_d &&\n".
                    "                        im_y>=0 && im_y<output_h &&\n".
                    "                        im_x>=0 && im_x<output_w) {\n".
                    "                        value += cols[{$cols_id}];\n".
                    "                    }\n".
                    "                }\n".
                    "            }\n".
                    "        }\n".
                    "        images[{$input_id}] += value;\n".
                    "    }\n".
                    "}\n";
            } else {
                $this->sources[$kernel_name] =
                    "__kernel void {$kernel_name}(\n".
                    "    $im_arg_type {$type} * images,\n".
                    "    const        uint offset_images,\n".
                    "    $col_arg_type {$type} * cols,\n".
                    "    const        uint offset_cols,\n".
                    "    const        uint batches,\n".
                    "    const        uint im_d,\n".
                    "    const        uint im_h,\n".
                    "    const        uint im_w,\n".
                    "    const        uint channels,\n".
                    "    const        uint kernel_d,\n".
                    "    const        uint kernel_h,\n".
                    "    const        uint kernel_w,\n".
                    "    const        uint stride_d,\n".
                    "    const        uint stride_h,\n".
                    "    const        uint stride_w,\n".
                    "    const        uint pad_d,\n".
                    "    const        uint pad_h,\n".
                    "    const        uint pad_w,\n".
                    "    const        uint dilation_d,\n".
                    "    const        uint dilation_h,\n".
                    "    const        uint dilation_w,\n".
                    "    const        uint output_d,\n".
                    "    const        uint output_h,\n".
                    "    const        uint output_w)\n".
                    "{\n".
                    "    const uint gid0 = get_global_id(0);\n".
                    "    const uint gid1 = get_global_id(1);\n".
                    "    const uint gid2 = get_global_id(2);\n".
                         $this->splitPointer('channel_id','im_x','gid0','channels').
                         $this->splitPointer('im_y','im_z','gid1','output_h').
                    "    const int batch_id =   gid2;\n".
                    "    if(im_x<output_w && im_z<output_d && batch_id<batches){\n".
                    "        for(uint kernel_z=0;kernel_z<kernel_d;kernel_z++) {\n".
                    "            for(uint kernel_y=0;kernel_y<kernel_h;kernel_y++) {\n".
                    "                for(uint kernel_x=0;kernel_x<kernel_w;kernel_x++) {\n".
                    "                    int input_z = {$input_z};\n".
                    "                    int input_y = {$input_y};\n".
                    "                    int input_x = {$input_x};\n".
                    "                    {$type} value;\n".
                    "                    if(input_z>=0 && input_z<im_d && input_y>=0 && input_y<im_h && input_x>=0 && input_x<im_w) {\n".
                    "                        uint input_id = {$input_id};\n".
                    "                        value = images[input_id];\n".
                    "                    } else {\n".
                    "                        value = 0;\n".
                    "                    }\n".
                    "                    uint cols_id = {$cols_id};\n".
                    "                    cols[cols_id] = value;\n".
                    "                }\n".
                    "            }\n".
                    "        }\n".
                    "    }\n".
                    "}\n";
            }
        }

        $kernel = $this->createKernel($kernel_name);
        $kernel->setArg(0,$images);
        $kernel->setArg(1,$images_offset,NDArray::uint32);
        $kernel->setArg(2,$cols);
        $kernel->setArg(3,$cols_offset,NDArray::uint32);
        $kernel->setArg(4,$batches,NDArray::uint32);
        $kernel->setArg(5,$im_d,NDArray::uint32);
        $kernel->setArg(6,$im_h,NDArray::uint32);
        $kernel->setArg(7,$im_w,NDArray::uint32);
        $kernel->setArg(8,$channels,NDArray::uint32);
        $kernel->setArg(9,$kernel_d,NDArray::uint32);
        $kernel->setArg(10,$kernel_h,NDArray::uint32);
        $kernel->setArg(11,$kernel_w,NDArray::uint32);
        $kernel->setArg(12,$stride_d,NDArray::uint32);
        $kernel->setArg(13,$stride_h,NDArray::uint32);
        $kernel->setArg(14,$stride_w,NDArray::uint32);
        $kernel->setArg(15,$pad_d,NDArray::uint32);
        $kernel->setArg(16,$pad_h,NDArray::uint32);
        $kernel->setArg(17,$pad_w,NDArray::uint32);
        $kernel->setArg(18,$dilation_d,NDArray::uint32);
        $kernel->setArg(19,$dilation_h,NDArray::uint32);
        $kernel->setArg(20,$dilation_w,NDArray::uint32);
        $kernel->setArg(21,$output_d,NDArray::uint32);
        $kernel->setArg(22,$output_h,NDArray::uint32);
        $kernel->setArg(23,$output_w,NDArray::uint32);
        if($reverse) {
            $global_work_size = [$this->ceil($channels*$im_w,2),$this->ceil($im_h*$im_d,2),$this->ceil($batches,8)];
            $local_work_size=[2,2,8];
        } else {
            $global_work_size = [$this->ceil($output_w*$channels,2),$this->ceil($output_h*$output_d,2),$this->ceil($batches,8)];
            $local_work_size=[2,2,8];
        }
        $kernel->enqueueNDRange($this->queue,$global_work_size,$local_work_size,null,
            $events,$waitEvents);
    }

    /**
    * randomUniform
    */
    public function randomUniform(
        int $n,
        BufferInterface $X, int $offsetX, int $incX,
        int|float $low,
        int|float $high,
        int $seed,
        object $events=null, object $waitEvents=null
        ) : void
    {
        $dtype = $X->dtype();
        if($dtype==NDArray::float64) {
            $this->assertFP64();
        }
        $isInt = array_key_exists($dtype,$this->intTypes);
        $itype = ($isInt) ? 'i' : 'f';
        $type = $this->dtypeToOpenCLType[$dtype];
        $kernel_name = "randomUniform_{$type}_{$itype}";
        if(!isset($this->sources[$kernel_name])) {
            if($isInt) {
                $this->sources[$kernel_name] =
                "uint wang_hash(uint seed)\n".
                "{\n".
                "    seed = (seed ^ 61) ^ (seed >> 16);\n".
                "    seed *= 9;\n".
                "    seed = seed ^ (seed >> 4);\n".
                "    seed *= 0x27d4eb2d;\n".
                "    seed = seed ^ (seed >> 15);\n".
                "    return seed;\n".
                "}\n".
                "__kernel void {$kernel_name}(const uint seed,\n".
                "                      const {$type} low,\n".
                "                      const {$type} high,\n".
                "                    __global {$type} * x,\n".
                "                      const uint offsetX,\n".
                "                      const uint incX)\n".
                "{\n".
                "   uint gid = get_global_id(0);\n".
                "   uint randmax = 0;\n".
                "   randmax--;\n".
                "   {$type} randx = ({$type})floor((float)wang_hash(gid+seed)*\n".
                "                         ((float)(high+1-low)/(float)randmax)+(float)low);\n".
                "   if(randx>=high) {\n".
                "       randx = high;\n".
                "   }\n".
                "   x[offsetX+gid*incX] = randx;\n".
                "}\n";
            } else {
                $this->sources[$kernel_name] =
                "uint wang_hash(uint seed)\n".
                "{\n".
                "    seed = (seed ^ 61) ^ (seed >> 16);\n".
                "    seed *= 9;\n".
                "    seed = seed ^ (seed >> 4);\n".
                "    seed *= 0x27d4eb2d;\n".
                "    seed = seed ^ (seed >> 15);\n".
                "    return seed;\n".
                "}\n".
                "__kernel void {$kernel_name}(const uint seed,\n".
                "                      const {$type} low,\n".
                "                      const {$type} high,\n".
                "                    __global {$type} * x,\n".
                "                      const uint offsetX,\n".
                "                      const uint incX)\n".
                "{\n".
                "   uint gid = get_global_id(0);\n".
                "   uint randmax = 0;\n".
                "   randmax--;\n".
                "   {$type} randx = ({$type})(({$type})wang_hash(gid+seed)*\n".
                "                         ((high-low)/({$type})randmax)+low);\n".
                "   x[offsetX+gid*incX] = randx;\n".
                "}\n";
            }
        }

        $kernel = $this->createKernel($kernel_name);
        $kernel->setArg(0,$seed,NDArray::uint32);
        $kernel->setArg(1,$low,$dtype);
        $kernel->setArg(2,$high,$dtype);
        $kernel->setArg(3,$X);
        $kernel->setArg(4,$offsetX,NDArray::uint32);
        $kernel->setArg(5,$incX,NDArray::uint32);
        $global_work_size = [$n];
        $local_work_size=null;
        #$local_work_size = [1,1,$size];
        $kernel->enqueueNDRange($this->queue,$global_work_size,$local_work_size,null,
            $events,$waitEvents);
    }

    /**
    * randomNormal
    */
    public function randomNormal(
        int $n,
        BufferInterface $X, int $offsetX, int $incX,
        float $mean,
        float $scale,
        int $seed,
        object $events=null, object $waitEvents=null
        ) : void
    {
        $dtype = $X->dtype();
        if($dtype!=NDArray::float32 && $dtype!=NDArray::float64) {
            throw new InvalidArgumentException('Data type of X must be float32 or float64');
        }
        if($dtype==NDArray::float64) {
            $this->assertFP64();
        }
        $isInt = array_key_exists($dtype,$this->intTypes);
        $type = $this->dtypeToOpenCLType[$dtype];
        $kernel_name = "randomNormal_{$type}";
        if(!isset($this->sources[$kernel_name])) {
            $PI = ($dtype==NDArray::float32) ? 'M_PI_F' : 'M_PI';
            $this->sources[$kernel_name] =
            "uint wang_hash(uint seed)\n".
            "{\n".
            "    seed = (seed ^ 61) ^ (seed >> 16);\n".
            "    seed *= 9;\n".
            "    seed = seed ^ (seed >> 4);\n".
            "    seed *= 0x27d4eb2d;\n".
            "    seed = seed ^ (seed >> 15);\n".
            "    return seed;\n".
            "}\n".
            "__kernel void {$kernel_name}(const uint seed,\n".
            "                      const {$type} mean,\n".
            "                      const {$type} scale,\n".
            "                    __global {$type} * x,\n".
            "                      const uint offsetX,\n".
            "                      const uint incX)\n".
            "{\n".
            "   uint gid = get_global_id(0);\n".
            "   uint randmax = 0;\n".
            "   randmax--;\n".
            "   uint seed1 = wang_hash(gid+seed);\n".
            "   uint seed2 = wang_hash(gid+seed1);\n".
            "   {$type} randx = ({$type})seed1/({$type})randmax;\n".
            "   {$type} randy = ({$type})seed2/({$type})randmax;\n".
            "   x[offsetX+gid*incX] = sqrt(-2*log(randx))*cos(2*{$PI}*randy)*scale+mean;\n".
            "}\n";
        }

        $kernel = $this->createKernel($kernel_name);
        $kernel->setArg(0,$seed,NDArray::uint32);
        $kernel->setArg(1,$mean,$dtype);
        $kernel->setArg(2,$scale,$dtype);
        $kernel->setArg(3,$X);
        $kernel->setArg(4,$offsetX,NDArray::uint32);
        $kernel->setArg(5,$incX,NDArray::uint32);
        $global_work_size = [$n];
        $local_work_size=null;
        #$local_work_size = [1,1,$size];
        $kernel->enqueueNDRange($this->queue,$global_work_size,$local_work_size,null,
            $events,$waitEvents);
    }

    public function fill(
        int $n,
        BufferInterface $X, int $offsetX, int $incX,
        mixed $pattern,
        object $events=null, object $waitEvents=null
        ) : void
    {
        if(!is_scalar($pattern)) {
            throw new InvalidArgumentException('Pattern must be scalar type.');
        }
        $dtype = $X->dtype();
        if($dtype==NDArray::float64) {
            $this->assertFP64();
        }
        $isInt = array_key_exists($dtype,$this->intTypes);
        $type = $this->dtypeToOpenCLType[$dtype];
        if(is_bool($pattern)) {
            $pattern = ($pattern)?1:0;
        }
        if($dtype==NDArray::bool) {
            $dtype = NDArray::uint8;
        } elseif($isInt) {
            $pattern = (int)$pattern;
        } else {
            $pattern = (float)$pattern;
        }
        $kernel_name = "fill_{$type}";
        if(!isset($this->sources[$kernel_name])) {
            $this->sources[$kernel_name] =
            "__kernel void {$kernel_name}(\n".
            "                    __global {$type} * x,\n".
            "                      const uint offsetX,\n".
            "                      const uint incX,\n".
            "                      const {$type} pattern)\n".
            "{\n".
            "   uint gid = get_global_id(0);\n".
            "   x[offsetX+gid*incX] = pattern;\n".
            "}\n";
        }

        $kernel = $this->createKernel($kernel_name);
        $kernel->setArg(0,$X);
        $kernel->setArg(1,$offsetX,NDArray::uint32);
        $kernel->setArg(2,$incX,NDArray::uint32);
        $kernel->setArg(3,$pattern,$dtype);
        $global_work_size = [$n];
        $local_work_size=null;
        #$local_work_size = [1,1,$size];
        $kernel->enqueueNDRange($this->queue,$global_work_size,$local_work_size,null,
            $events,$waitEvents);
    }

    protected function toHostBuffer(
        BufferInterface $clBuffer,
        int $offset,
        bool $blocking_read=true,
        object $events=null,
        object $waitEvents=null) : HostBufferInterface
    {
        $dtype = $clBuffer->dtype();
        $bytes = $clBuffer->bytes();
        $valueSize = $clBuffer->value_size();
        $size = $clBuffer->count();
        $hostBuffer = $this->newHostBuffer($size,$dtype);
        $event = $clBuffer->read($this->queue,$hostBuffer,$bytes,
            $offset*$valueSize,$hostoffset=0,$blocking_read,$events,$waitEvents);
        return $hostBuffer;
    }

    public function transpose(
        HostBufferInterface|BufferInterface $shape,   // DeviceBuffer or HostBuffer
        HostBufferInterface|BufferInterface $perm,    // DeviceBuffer or HostBuffer
        BufferInterface $A, int $offsetA,
        BufferInterface $B, int $offsetB, 
        object $events=null, object $waitEvents=null
        ) : void
    {
        $dtype = $A->dtype();
        if($dtype==NDArray::float64) {
            $this->assertFP64();
        }
        $isInt = array_key_exists($dtype,$this->intTypes);
        $type = $this->dtypeToOpenCLType[$dtype];
        $ndim = count($shape);

        // check shape
        if($shape->dtype()!=NDArray::int32) {
            throw new InvalidArgumentException('data type of shape buffer must be int32.');
        }
        $orgShape = $shape;
        if($shape instanceof BufferInterface) {
            $shape = $this->toHostBuffer($shape,0);
        }
        $size = 1;
        for($i=0;$i<$ndim;$i++) {
            $size *= $shape[$i];
        }
        
        // check shape
        if($ndim!=count($perm)) {
            throw new InvalidArgumentException('matrix shape and perm must be same size.');
        }
        if($perm->dtype()!=NDArray::int32) {
            throw new InvalidArgumentException('data type of perm buffer must be int32.');
        }
        $orgPerm = $perm;
        if($perm instanceof BufferInterface) {
            $perm = $this->toHostBuffer($perm,0);
        }

        if($dtype!=$B->dtype()) {
            throw new InvalidArgumentException("Unmatch data type for A and B.".
            $this->dtypeToString($dtype).",".$this->dtypeToString($B->dtype()));
        }

        $strides = $this->newHostBuffer($ndim,NDArray::int32);
        $targetStrides = $this->newHostBuffer($ndim,NDArray::int32);
        $stride = 1;
        $targetStride = 1;
        for($dimDepth=$ndim-1;$dimDepth>=0;$dimDepth--) {
            $strides[$dimDepth] = $stride;
            $stride *= $shape[$dimDepth];
            $targDepth = $perm[$dimDepth];
            if($targDepth>=$ndim) {
                throw new InvalidArgumentException('perm contained an out-of-bounds axis.');
            }
            $targetStrides[$targDepth] = $targetStride;
            $targetStride *= $shape[$targDepth];
        }
        if($stride!=$targetStride) {
            throw new InvalidArgumentException('Perm contained duplicate axis.');
        }
    
        if($dtype==NDArray::bool) {
            $dtype = NDArray::uint8;
        }

        $repeat = $shape[0];
        $stride = $strides[0];
        $targetStride = $targetStrides[0];

        $shape = $orgShape;
        if(!($shape instanceof BufferInterface)) {
            $valueSize = $shape->value_size();
            $shape = $this->newBuffer(
                count($shape)*$valueSize,
                OpenCL::CL_MEM_READ_ONLY|OpenCL::CL_MEM_COPY_HOST_PTR,
                $shape, 0,
                $shape->dtype());
        }

        $valueSize = $strides->value_size();
        $strides = $this->newBuffer(
            $strides->count()*$valueSize,
            OpenCL::CL_MEM_READ_ONLY|OpenCL::CL_MEM_COPY_HOST_PTR,
            $strides, 0,
            $strides->dtype()
        );
        $valueSize = $strides->value_size();
        $targetStrides = $this->newBuffer(
            $targetStrides->count()*$valueSize,
            OpenCL::CL_MEM_READ_ONLY|OpenCL::CL_MEM_COPY_HOST_PTR,
            $targetStrides, 0,
            $targetStrides->dtype()
        );

        $kernel_name = "transpose_{$type}";
        if(!isset($this->sources[$kernel_name])) {
            $this->sources[$kernel_name] =
            "void {$kernel_name}_copy(\n".
            "                      const  int n,\n".
            "                    __global {$type} * a,\n".
            "                             int offsetA,\n".
            "                      const  int incA,\n".
            "                    __global {$type} * b,\n".
            "                             int offsetB,\n".
            "                      const  int incB)\n".
            "{\n".
            "    for(int i=0;i<n;i++,offsetA+=incA,offsetB+=incB) {\n".
            "        b[offsetB] = a[offsetA];\n".
            "    }\n".
            "}\n".
            "__kernel void {$kernel_name}(\n".
            "                      const  int ndim,\n".
            "                    __global int * shape,\n".
            "                    __global int * strides,\n".
            "                    __global int * targetStrides,\n".
            "                    __global {$type} * a,\n".
            "                             int offsetA,\n".
            "                    __global {$type} * b,\n".
            "                             int offsetB,\n".
            "                    __local {$type} * stackPos,\n".
            "                    __local {$type} * stackOfsA,\n".
            "                    __local {$type} * stackOfsB)\n".
            "{\n".
            //"   const int dbg = -1;"."\n".
            "   const int gid = get_global_id(0);\n".
            "   const int surface = 1;\n".
            "   const int bed = 2;\n".
            "   int depth = surface;\n".
            "   int pos = 0;\n".
            //'   if(gid==dbg) printf("start kernel(%d)\n",gid);'."\n".
            "   offsetA = offsetA+strides[0]*gid;\n".
            "   offsetB = offsetB+targetStrides[0]*gid;\n".
            "   if(depth>=ndim) {\n".
            "       b[offsetB] = a[offsetA];\n".
            "       return;\n".
            "   }\n".
            "   while(depth>=surface) {\n".
            //'       if(gid==dbg) printf("top(%d):  dep=%d,pos=%d,rep=%d,ofsA=%d,ofsB=%d,st=%d,ta=%d\n",'."\n".
            //'           gid,depth,pos,shape[depth],offsetA,offsetB,strides[depth],targetStrides[depth]);'."\n".
            "       while(depth<ndim-bed) {\n".
            //'           if(gid==dbg) printf("push(%d): dep=%d,pos=%d,rep=%d,ofsA=%d,ofsB=%d,st=%d,ta=%d\n",'."\n".
            //'               gid,depth,pos,shape[depth],offsetA,offsetB,strides[depth],targetStrides[depth]);'."\n".
            "           stackPos[depth] = pos;\n".
            "           stackOfsA[depth] = offsetA;\n".
            "           stackOfsB[depth] = offsetB;\n".
            "           pos = 0;\n".
            "           depth++;\n".
            //'           if(gid==dbg) printf("psh2(%d): dep=%d,pos=%d,rep=%d,ofsA=%d,ofsB=%d,st=%d,ta=%d\n",'."\n".
            //'               gid,depth,pos,shape[depth],offsetA,offsetB,strides[depth],targetStrides[depth]);'."\n".
            "       }\n".
            "       if(depth>=ndim-bed) {\n".
            "           int dp=0;\n".
            "           if(ndim>2) {\n".
            "               depth++;\n".
            "               dp=1;\n".
            "           }\n".
            //'           if(gid==dbg) printf("copy(%d): dep=%d,      n=  %d,ofsA=%d,ofsB=%d,st=%d,ta=%d\n",'."\n".
            //'               gid,depth,shape[depth],offsetA,offsetB,strides[depth],targetStrides[depth]);'."\n".
            "           {$kernel_name}_copy(\n".
            "               shape[depth],\n".
            "               a,offsetA,strides[depth],\n".
            "               b,offsetB,targetStrides[depth]\n".
            "           );\n".
            "           if(dp!=0) {\n".
            "               depth--;\n".
            "           }\n".
            "       }\n".
            "       while(true) {\n".
            "           if(depth<=ndim-bed) {\n".
            "               offsetA += strides[depth];\n".
            "               offsetB += targetStrides[depth];\n".
            "               pos++;\n".
            //'               if(gid==dbg) printf("incr(%d): dep=%d,pos=%d,rep=%d,ofsA=%d,ofsB=%d,st=%d,ta=%d\n",'."\n".
            //'                   gid,depth,pos,shape[depth],offsetA,offsetB,strides[depth],targetStrides[depth]);'."\n".
            "               if(pos<shape[depth]) {\n".
            "                   break;\n".
            "               }\n".
            "           }\n".
            "           depth--;\n".
            "           if(depth<surface) {\n".
            //'               if(gid==dbg) printf("dep(%d)=%d\n",gid,depth);'."\n".
            "               break;\n".
            "           }\n".
            "           pos = stackPos[depth];\n".
            "           offsetA = stackOfsA[depth];\n".
            "           offsetB = stackOfsB[depth];\n".
            //'           if(gid==dbg) printf("pop(%d):  dep=%d,pos=%d,rep=%d,ofsA=%d,ofsB=%d,st=%d,ta=%d\n",'."\n".
            //'               gid,depth,pos,shape[depth],offsetA,offsetB,strides[depth],targetStrides[depth]);'."\n".
            "       }\n".
            "   }\n".
            "}\n";
        }

        $intSize = $strides->value_size();
        $kernel = $this->createKernel($kernel_name);
        $kernel->setArg(0,$ndim,NDArray::int32);
        $kernel->setArg(1,$shape);
        $kernel->setArg(2,$strides);
        $kernel->setArg(3,$targetStrides);
        $kernel->setArg(4,$A);
        $kernel->setArg(5,$offsetA,NDArray::int32);
        $kernel->setArg(6,$B);
        $kernel->setArg(7,$offsetB,NDArray::int32);
        $kernel->setArg(8,null,$ndim*$intSize);
        $kernel->setArg(9,null,$ndim*$intSize);
        $kernel->setArg(10,null,$ndim*$intSize);
        $global_work_size = [$repeat];
        $local_work_size = [1];
        $kernel->enqueueNDRange($this->queue,$global_work_size,$local_work_size,null,
            $events,$waitEvents);
    }

    /**
     *     X := bandpart(X,lower,upper)
     */
    public function bandpart(
        int $m,
        int $n,
        int $k,
        BufferInterface $A, int $offset,
        int $lower,
        int $upper,
        object $events=null, object $waitEvents=null
        ) : void
    {
        $dtype = $A->dtype();
        if($dtype==NDArray::float64) {
            $this->assertFP64();
        }
        $type = $this->dtypeToOpenCLType[$dtype];
        $kernel_name = "bandpart_{$type}";
        if(!isset($this->sources[$kernel_name])) {
            $this->sources[$kernel_name] =
                "__kernel void {$kernel_name}(\n".
                "    const        int n,\n".
                "    const        int k,\n".
                "        __global {$type} * a,\n".
                "    const        int offset,\n".
                "    const        int lower,\n".
                "    const        int upper)\n".
                "{\n".
                "    int idx = get_global_id(0)*n*k+offset;\n".
                "    int i = get_global_id(1);\n".
                "    for(int j=0;j<k;j++) {\n".
                "        if((lower >= 0 && (i-j) > lower) || (upper >= 0 && (j-i) > upper)) {\n".
                "            a[idx+i*k+j] = 0;\n".
                "        }\n".
                "    }\n".
                "}\n";
        }
        $kernel = $this->createKernel($kernel_name);
        $kernel->setArg(0,$n,NDArray::uint32);
        $kernel->setArg(1,$k,NDArray::uint32);
        $kernel->setArg(2,$A);
        $kernel->setArg(3,$offset,NDArray::uint32);
        $kernel->setArg(4,$lower,NDArray::uint32);
        $kernel->setArg(5,$upper,NDArray::uint32);
        $global_work_size = [$m,$n];
        $kernel->enqueueNDRange($this->queue,$global_work_size,null,null,
            $events,$waitEvents);
    }
}
