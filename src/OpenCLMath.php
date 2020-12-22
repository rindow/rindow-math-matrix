<?php
namespace Rindow\Math\Matrix;

use Rindow\OpenCL\Buffer as Buffer;
use RuntimeException;
use InvalidArgumentException;
use Interop\Polite\Math\Matrix\NDArray;
use Interop\Polite\Math\Matrix\OpenCL;
use Interop\Polite\Math\Matrix\LinearBuffer;
use Rindow\OpenCL\Program;
use Rindow\OpenCL\Kernel;
use Rindow\OpenCL\EventList;

class OpenCLMath
{
    protected $dtypeToString = [
        NDArray::bool=>'bool',
        NDArray::int8=>'int8',   NDArray::uint8=>'uint8',
        NDArray::int16=>'int16', NDArray::uint16=>'uint16',
        NDArray::int32=>'int32', NDArray::uint32=>'uint32',
        NDArray::int64=>'int64', NDArray::uint64=>'uint64',
        NDArray::float16=>'float16',
        NDArray::float32=>'float32', NDArray::float64=>'float64',
    ];

    protected $dtypeToOpenCLType = [
        NDArray::bool=>'uchar',
        NDArray::int8=>'char',   NDArray::uint8=>'uchar',
        NDArray::int16=>'short', NDArray::uint16=>'ushort',
        NDArray::int32=>'int', NDArray::uint32=>'uint',
        NDArray::int64=>'long', NDArray::uint64=>'ulong',
        NDArray::float16=>'half',
        NDArray::float32=>'float', NDArray::float64=>'double',
    ];

    protected $smallests = [
        NDArray::bool  => 0,
        NDArray::int8  => -128,         NDArray::uint8  => 0,
        NDArray::int16 => -32768,       NDArray::uint16 => 0,
        NDArray::int32 => -2147483648,  NDArray::uint32 => 0,
        NDArray::int64 => -9223372036854775808, NDArray::uint64 => 0,
        NDArray::float16 => -1.0e+14,
        NDArray::float32 => -1.0e+37, NDArray::float64 => -1.0e+37,
    ];

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
            "    if(local_work[lid] < local_work[lid + i]) {\n".
            "        local_work[lid] = local_work[lid + i];\n".
            "    }\n".
            "}\n".
            "barrier(CLK_LOCAL_MEM_FENCE);\n",
        'qimax' =>
            "i >>= 1;\n".
            "if(lid < i) {\n".
            "    if(local_work[lid] < local_work[lid + i]) {\n".
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
            "if(value < input) {\n".
            "    value = input;\n".
            "}\n",
        'lrmax-2' =>
            "if(local_work[lid] < local_work[lid + i]) {\n".
            "    local_work[lid] = local_work[lid + i];\n".
            "}\n",
        'lrmax-3' =>
            "temp_buffer[parallel_item_id*grs+grid] = local_work[0];\n",
        'lrmax-4' =>
            "if(temp_buffer[parallel_item_id*lws*2 + lid] < temp_buffer[parallel_item_id*lws*2 + lws+lid]) {\n".
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

    protected $context;
    protected $queue;
    protected $fp64;
    protected $maxWorkItem;

    public function __construct(object $context,object $queue)
    {
        $this->context = $context;
        $this->queue = $queue;
        $devices = $context->getInfo(OpenCL::CL_CONTEXT_DEVICES);
        $extensions = $devices->getInfo(0,OpenCL::CL_DEVICE_EXTENSIONS);
        if(strpos($extensions,'cl_khr_fp64')===false) {
            $this->fp64 = false;
        } else {
            $this->fp64 = true;
        }
        $this->maxWorkItem = $devices->getInfo(0,OpenCL::CL_DEVICE_MAX_WORK_ITEM_SIZES);
    }

    public function fp64() : bool
    {
        return $this->fp64;
    }

    public function dtypeToString($dtype)
    {
        return $this->dtypeToString[$dtype];
    }

    protected function assertFP64() : void
    {
        if(!$this->fp64) {
            throw new RuntimeException('This device does not support 64-bit floating point.');
        }
    }

    public function maxWorkItem()
    {
        return $this->maxWorkItem;
    }

    protected function createKernel($name)
    {
        if(!isset($this->program[$name])) {
            $source = $this->sources[$name];
            $program = new Program($this->context,$source);
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
        $kernel = new Kernel($program,$name);
        return $kernel;
    }

    public function kernelTemplateQSum($inputs,$outputs)
    {
        $operation = $this->kernelCoreOperation['qsum'];
        $initial = 0;
        return $this->kernelQTemplate($operation,$inputs,$outputs,$initial);
    }

    public function kernelTemplateSSum($inputs,$outputs)
    {
        $operation = $this->kernelCoreOperation['qsum'];
        $initial = 0;
        return $this->kernelSTemplate($operation,$inputs,$outputs,$initial);
    }

    public function kernelTemplateLSingleSum1($inputs,$dtype)
    {
        $operation1 = $this->kernelCoreOperation['lsum-1'];
        $operation2 = $this->kernelCoreOperation['lsum-2'];
        $operation3 = $this->kernelCoreOperation['lsum-3'];
        $type = $this->dtypeToOpenCLType[$dtype];
        $initial = 0;
        return $this->kernelLTemplate1(
            $operation1,$operation2,$operation3,$inputs,$type,$initial);
    }

    public function kernelTemplateLSingleSum2($output)
    {
        $operation2 = $this->kernelCoreOperation['lsum-2'];
        $operation4 = $this->kernelCoreOperation['lsum-4'];
        return $this->kernelLTemplate2(
            $operation2,$operation4,$output);
    }

    public function kernelTemplateLSum1($inputs,$dtype)
    {
        $operation1 = $this->kernelCoreOperation['lrsum-1'];
        $operation2 = $this->kernelCoreOperation['lrsum-2'];
        $operation3 = $this->kernelCoreOperation['lrsum-3'];
        $type = $this->dtypeToOpenCLType[$dtype];
        $initial = 0;
        return $this->kernelLTemplate1(
            $operation1,$operation2,$operation3,$inputs,$type,$initial);
    }

    public function kernelTemplateLSum2($output)
    {
        $operation2 = $this->kernelCoreOperation['lrsum-2'];
        $operation4 = $this->kernelCoreOperation['lrsum-4'];
        return $this->kernelLTemplate2(
            $operation2,$operation4,$output);
    }

    public function kernelTemplateQMax($inputs,$outputs,$dtype)
    {
        $operation = $this->kernelCoreOperation['qmax'];
        $initial = $this->smallests[$dtype];
        return $this->kernelQTemplate($operation,$inputs,$outputs,$initial);
    }

    public function kernelTemplateSMax($inputs,$outputs,$dtype)
    {
        $operation = $this->kernelCoreOperation['qmax'];
        $initial = $this->smallests[$dtype];
        return $this->kernelSTemplate($operation,$inputs,$outputs,$initial);
    }

    public function kernelTemplateQiMax($inputs,$outputs,$dtype)
    {
        $operation = $this->kernelCoreOperation['qimax'];
        $initial = $this->smallests[$dtype];
        return $this->kernelQiTemplate($operation,$inputs,$outputs,$initial);
    }

    public function kernelTemplateSiMax($inputs,$outputs,$dtype)
    {
        $operation = $this->kernelCoreOperation['qimax'];
        $initial = $this->smallests[$dtype];
        return $this->kernelSiTemplate($operation,$inputs,$outputs,$initial);
    }

    public function kernelTemplateLMax1($inputs,$dtype)
    {
        $operation1 = $this->kernelCoreOperation['lrmax-1'];
        $operation2 = $this->kernelCoreOperation['lrmax-2'];
        $operation3 = $this->kernelCoreOperation['lrmax-3'];
        $type = $this->dtypeToOpenCLType[$dtype];
        $initial = $this->smallests[$dtype];
        return $this->kernelLTemplate1(
            $operation1,$operation2,$operation3,$inputs,$type,$initial);
    }

    public function kernelTemplateLMax2($output)
    {
        $operation2 = $this->kernelCoreOperation['lrmax-2'];
        $operation4 = $this->kernelCoreOperation['lrmax-4'];
        return $this->kernelLTemplate2(
            $operation2,$operation4,$output);
    }

    public function kernelTemplateLiMax1($inputs,$dtype)
    {
        $operation1 = $this->kernelCoreOperation['lrimax-1'];
        $operation2 = $this->kernelCoreOperation['lrimax-2'];
        $operation3 = $this->kernelCoreOperation['lrimax-3'];
        $type = $this->dtypeToOpenCLType[$dtype];
        $initial = $this->smallests[$dtype];
        return $this->kernelLiTemplate1(
            $operation1,$operation2,$operation3,$inputs,$type,$initial);
    }

    public function kernelTemplateLiMax2($output)
    {
        $operation2 = $this->kernelCoreOperation['lrimax-2'];
        $operation4 = $this->kernelCoreOperation['lrimax-4'];
        return $this->kernelLTemplate2(
            $operation2,$operation4,$output);
    }

    public function kernelQTemplate($operation,$inputs,$outputs,$initial)
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
        "                    ${inputs}\n".
        "                } else {\n".
        "                    local_work[lid] = seg_work[seg*lws+lid];\n".
        "                }\n".
        "            } else {\n".
        "                local_work[lid] = ${initial};\n".
        "            }\n".
        "            barrier(CLK_LOCAL_MEM_FENCE);\n".
        "            int i = lws;\n".
        "            while( i>1 ) {\n".
        "                ${operation}\n".
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
        "        seg_count = (seg_count+lws-1)/lws;\n".
        "    }\n".
        "    if(lid == 0) {\n".
        "        ${outputs}\n".
        "    }\n".
        "}\n";
    }

    public function kernelQiTemplate($operation,$inputs,$outputs,$initial)
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
        "                    ${inputs}\n".
        "                } else {\n".
        "                    local_work[lid] = seg_work[seg*lws+lid];\n".
        "                    local_iwork[lid] = seg_iwork[seg*lws+lid];\n".
        "                }\n".
        "            } else {\n".
        "                local_work[lid] = ${initial};\n".
        "                local_iwork[lid] = 0;\n".
        "            }\n".
        "            barrier(CLK_LOCAL_MEM_FENCE);\n".
        "            int i = lws;\n".
        "            while( i>1 ) {\n".
        "                ${operation}\n".
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
        "        seg_count = (seg_count+lws-1)/lws;\n".
        "    }\n".
        "    if(lid == 0) {\n".
        "        ${outputs}\n".
        "    }\n".
        "}\n";
    }

    public function kernelSTemplate($operation,$inputs,$outputs,$initial)
    {
        return
        "{\n".
        "    const uint lid = get_local_id(0);\n".
        "    const uint lws = get_local_size(0);\n".
        #"    local_work[lid] = 0;\n".
        #"    ${type} input = 0;\n".
        "    if(lid<total_local_items) {\n".
        "        ${inputs}\n".
        "    } else {\n".
        "        local_work[lid]  = ${initial};\n".
        "    }\n".
        #"    local_work[lid] = input;\n".
        "    barrier(CLK_LOCAL_MEM_FENCE);\n".
        "    int i = lws;\n".
        "    while( i>1 ) {\n".
        "        ${operation}\n".
        "    }\n".
        "    if(lid == 0) {\n".
        "        ${outputs}\n".
        "    }\n".
        "}\n";
    }

    public function kernelSiTemplate($operation,$inputs,$outputs,$initial)
    {
        return
        "{\n".
        "    const uint lid = get_local_id(0);\n".
        "    const uint lws = get_local_size(0);\n".
        #"    local_work[lid] = 0;\n".
        #"    ${type} input = 0;\n".
        "    if(lid<total_local_items) {\n".
        "        ${inputs}\n".
        "    } else {\n".
        "        local_work[lid] = ${initial};\n".
        "        local_iwork[lid] = 0;\n".
        "    }\n".
        #"    local_work[lid] = input;\n".
        "    barrier(CLK_LOCAL_MEM_FENCE);\n".
        "    int i = lws;\n".
        "    while( i>1 ) {\n".
        "        ${operation}\n".
        "    }\n".
        "    if(lid == 0) {\n".
        "        ${outputs}\n".
        "    }\n".
        "}\n";
    }

    public function kernelLTemplate1(
        $operation1,$operation2,$operation3,$inputs,$type,$initial)
    {
        return
        "{\n".
        "    const uint lid = get_local_id(0);\n".
        "    const uint grid = get_group_id(0);\n".
        "    const uint lws = get_local_size(0);\n".
        "    const uint grs = get_num_groups(0);\n".
        "    ${type} value = ${initial};\n".
        "    uint local_item_id = grid*lws + lid;\n".
        "    while(local_item_id < total_local_items) {\n".
        "        ${inputs}\n".
        "        ${operation1}\n".
        "        local_item_id += lws*grs;\n".
        "    }\n".
        "    local_work[lid] = value;\n".
        "    barrier(CLK_LOCAL_MEM_FENCE);\n".
        "    for(int i=lws/2; i>0; i>>=1) {\n".
        "        if(lid < i) {\n".
        "            ${operation2}\n".
        "        }\n".
        "        barrier(CLK_LOCAL_MEM_FENCE);\n".
        "    }\n".
        "    if(lid == 0) {\n".
        "        ${operation3}\n".
        "    }\n".
        "}\n";
    }

    public function kernelLTemplate2(
        $operation2,$operation4,$output)
    {
        return
        "{\n".
        "    const uint lid = get_local_id(0);\n".
        "    const uint lws = get_local_size(0);\n".
        "    ${operation4}\n".
        "    barrier(CLK_LOCAL_MEM_FENCE);\n".
        "    for(uint i=lws/2; i>0; i>>=1) {\n".
        "        if (lid < i) {\n".
        "            ${operation2}\n".
        "        }\n".
        "        barrier(CLK_LOCAL_MEM_FENCE);\n".
        "    }\n".
        "    if (lid == 0) {\n".
        "        ${output}\n".
        "    }\n".
        "}\n";
    }

    public function kernelLiTemplate1(
        $operation1,$operation2,$operation3,$inputs,$type,$initial)
    {
        return
        "{\n".
        "    const uint lid = get_local_id(0);\n".
        "    const uint grid = get_group_id(0);\n".
        "    const uint lws = get_local_size(0);\n".
        "    const uint grs = get_num_groups(0);\n".
        "    ${type} value = ${initial};\n".
        "    uint    ivalue = 0;\n".
        "    uint local_item_id = grid*lws + lid;\n".
        "    while(local_item_id < total_local_items) {\n".
        "        ${inputs}\n".
        "        ${operation1}\n".
        "        local_item_id += lws*grs;\n".
        "    }\n".
        "    local_work[lid] = value;\n".
        "    local_iwork[lid] = ivalue;\n".
        "    barrier(CLK_LOCAL_MEM_FENCE);\n".
        "    for(int i=lws/2; i>0; i>>=1) {\n".
        "        if(lid < i) {\n".
        "            ${operation2}\n".
        "        }\n".
        "        barrier(CLK_LOCAL_MEM_FENCE);\n".
        "    }\n".
        "    if(lid == 0) {\n".
        "        ${operation3}\n".
        "    }\n".
        "}\n";
    }

    protected function newEventList()
    {
        return new EventList();
    }

    protected function newBuffer(
        int $size,int $flags=null,
        LinearBuffer $hostBuffer=null, int $hostOffset=null,
        int $dtype=null)
    {
        return new OpenCLBuffer($this->context,
            $size,$flags,$hostBuffer,$hostOffset,$dtype);
    }

    protected function newHostBuffer($size,$dtype)
    {
        return new OpenBlasBuffer($size,$dtype);
    }

    /**
     * Y := sum( X )
     */
    public function sum(
        int $n,
        Buffer $R, int $offsetR,
        Buffer $X, int $offsetX, int $incX,
        EventList $events=null, EventList $waitEvents=null
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
            $this->sum1(
                $n,
                $R, $offsetR,
                $X, $offsetX, $incX,
                $events, $waitEvents
            );
        } elseif($n <= $max_work_items*$max_work_items*2) {
            $this->sum2(
                $n,
                $R, $offsetR,
                $X, $offsetX, $incX,
                $events, $waitEvents
            );
        } else {
            $this->sum3(
                $n,
                $R, $offsetR,
                $X, $offsetX, $incX,
                $events, $waitEvents
            );
        }
    }

    /**
     * Y := sum( X )
     */
    public function sum1(
        int $n,
        Buffer $R, int $offsetR,
        Buffer $X, int $offsetX, int $incX,
        EventList $events=null, EventList $waitEvents=null
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
        $kernel_name = "sum_S_${type}";
        if(!isset($this->sources[$kernel_name])) {
            $this->sources[$kernel_name] =
                "__kernel void ${kernel_name}(\n".
                "    const        uint total_local_items,\n".
                "        __global ${type} * r,\n".
                "    const        uint offset_r,\n".
                "        __global ${type} * x,\n".
                "    const        uint offset_x,\n".
                "    const        uint incx,\n".
                "         __local ${type} * local_work)\n".
                "{\n".
                    $this->kernelTemplateSSum(
                        "local_work[lid] = x[${index_x}];",
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
        $kernel->setArg(6,null,$max_work_items*$value_size);
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
        Buffer $R, int $offsetR,
        Buffer $X, int $offsetX, int $incX,
        EventList $events=null, EventList $waitEvents=null
        ) : void
    {
        $dtype = $X->dtype();

        $index_x = '(seg*lws+lid)+offset_x';
        $total_local_items = $n;
        $max_work_items = $this->maxWorkItem[0];
        if($total_local_items>$max_work_items) {
            $segments = (int)floor(($total_local_items+$max_work_items-1)/$max_work_items); // round up float
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
        $kernel_name = "sum_M_${type}";
        if(!isset($this->sources[$kernel_name])) {
            $this->sources[$kernel_name] =
                "__kernel void ${kernel_name}(\n".
                "    const        uint total_local_items,\n".
                "    const        uint segments,\n".
                "        __global ${type} * r,\n".
                "    const        uint offset_r,\n".
                "        __global ${type} * x,\n".
                "    const        uint offset_x,\n".
                "    const        uint incx,\n".
                "         __local ${type} * local_work,\n".
                "         __local ${type} * seg_work,\n".
                "    const        uint work_items)\n".
                "{\n".
                    $this->kernelTemplateQSum(
                        "local_work[lid] = x[${index_x}];",
                        "r[offset_r] = seg_work[0];"
                    ).
                "}\n";
        }
        $kernel = $this->createKernel($kernel_name);

        $kernel->setArg(0,$total_local_items,NDArray::uint32);
        $kernel->setArg(1,$segments,NDArray::uint32);
        $kernel->setArg(2,$R);
        $kernel->setArg(3,$offsetR,NDArray::uint32);
        $kernel->setArg(4,$X);
        $kernel->setArg(5,$offsetX,NDArray::uint32);
        $kernel->setArg(6,$incX,NDArray::uint32);
        $kernel->setArg(7,null,$max_work_items*$value_size);
        $kernel->setArg(8,null,$segments*$value_size);
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
        Buffer $R, int $offsetR,
        Buffer $X, int $offsetX, int $incX,
        EventList $events=null, EventList $waitEvents=null
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
        $kernel_name1 = "sum_L1_${type}";
        $kernel_name2 = "sum_L2_${type}";
        if(!isset($this->sources[$kernel_name1])) {
            $this->sources[$kernel_name1] =
                "__kernel void ${kernel_name1}(\n".
                "    const        uint total_local_items,\n".
                "    const __global ${type} * x,\n".
                "    const        uint offset_x,\n".
                "    const        uint incx,\n".
                "        __global ${type} * temp_buffer,\n".
                "         __local ${type} * local_work)\n".
                "{\n".
                    $this->kernelTemplateLSingleSum1(
                        "${type} input = x[local_item_id*incx + offset_x];",
                        $dtype
                    ).
                "}\n";
        }
        $kernel = $this->createKernel($kernel_name1);

        if(!isset($this->sources[$kernel_name2])) {
            $this->sources[$kernel_name2] =
                "__kernel void ${kernel_name2}(\n".
                "    const __global ${type} * temp_buffer,\n".
                "        __global ${type} * r,\n".
                "    const        uint offset_r,\n".
                "         __local ${type} * local_work)\n".
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
        $kernel->setArg(5,null,$work_items1*$value_size);
        $global_work_size = [$work_items1*$temp_size];
        $local_work_size = [$work_items1];
        $sum1Events = $this->newEventList();
        $kernel->enqueueNDRange($this->queue,$global_work_size,$local_work_size,null,
                $sum1Events,$waitEvents);

        $kernel2->setArg(0,$temp_buffer);
        $kernel2->setArg(1,$R);
        $kernel2->setArg(2,$offsetR,NDArray::uint32);
        $kernel2->setArg(3,null,$work_items2*$value_size);
        $global_work_size = [$work_items2];
        $local_work_size = [$work_items2];
        $kernel2->enqueueNDRange($this->queue,$global_work_size,$local_work_size,null,
                $events,$sum1Events);
    }

    /**
     *     X := a*X + b
     */
    public function increment(
        int $n,
        float $alpha,
        Buffer $X, int $offsetX, int $incX,
        float $beta,
        EventList $events=null, EventList $waitEvents=null
        ) : void
    {
        $dtypeX = $X->dtype();
        if($dtypeX==NDArray::float64) {
            $this->assertFP64();
        }
        $type = $this->dtypeToOpenCLType[$dtypeX];
        $kernel_name = "increment_${type}";
        if(!isset($this->sources[$kernel_name])) {
            $this->sources[$kernel_name] =
                "__kernel void ${kernel_name}(\n".
                "    const        ${type} alpha,\n".
                "        __global ${type} * x,\n".
                "    const        uint offset_x,\n".
                "    const        uint incx,\n".
                "    const        ${type} beta)\n".
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
        Buffer $X, int $offsetX, int $incX,
        float $beta,
        EventList $events=null, EventList $waitEvents=null
        ) : void
    {
        $dtypeX = $X->dtype();
        if($dtypeX==NDArray::float64) {
            $this->assertFP64();
        }
        $type = $this->dtypeToOpenCLType[$dtypeX];
        $kernel_name = "reciprocal_${type}";
        if(!isset($this->sources[$kernel_name])) {
            $this->sources[$kernel_name] =
                "__kernel void ${kernel_name}(\n".
                "    const        ${type} alpha,\n".
                "        __global ${type} * x,\n".
                "    const        uint offset_x,\n".
                "    const        uint incx,\n".
                "    const        ${type} beta)\n".
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
     *     X := X  (X > a)
     *     X := a  (X <= a)
     */
    public function maximum(
        int $n,
        float $alpha,
        Buffer $X, int $offsetX, int $incX,
        EventList $events=null, EventList $waitEvents=null
        ) : void
    {
        $dtypeX = $X->dtype();
        if($dtypeX==NDArray::float64) {
            $this->assertFP64();
        }
        $type = $this->dtypeToOpenCLType[$dtypeX];
        $kernel_name = "maximum_${type}";
        if(!isset($this->sources[$kernel_name])) {
            $this->sources[$kernel_name] =
                "__kernel void ${kernel_name}(\n".
                "    const        ${type} alpha,\n".
                "        __global ${type} * x,\n".
                "    const        uint offset_x,\n".
                "    const        uint incx)\n".
                "{\n".
                "    uint idx = get_global_id(0)*incx+offset_x;\n".
                "    ${type} tmp = x[idx];\n".
                "    if(tmp < alpha) {\n".
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
     *     X := X  (X < a)
     *     X := a  (X >= a)
     */
    public function minimum(
        int $n,
        float $alpha,
        Buffer $X, int $offsetX, int $incX,
        EventList $events=null, EventList $waitEvents=null
        ) : void
    {
        $dtypeX = $X->dtype();
        if($dtypeX==NDArray::float64) {
            $this->assertFP64();
        }
        $type = $this->dtypeToOpenCLType[$dtypeX];
        $kernel_name = "minimum_${type}";
        if(!isset($this->sources[$kernel_name])) {
            $this->sources[$kernel_name] =
                "__kernel void ${kernel_name}(\n".
                "    const        ${type} alpha,\n".
                "        __global ${type} * x,\n".
                "    const        uint offset_x,\n".
                "    const        uint incx)\n".
                "{\n".
                "    uint idx = get_global_id(0)*incx+offset_x;\n".
                "    ${type} tmp = x[idx];\n".
                "    if(tmp > alpha) {\n".
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
     *     X := 1  (X > a)
     *     X := 0  (X <= a)
     */
    public function greater(
        int $n,
        float $alpha,
        Buffer $X, int $offsetX, int $incX,
        EventList $events=null, EventList $waitEvents=null
        ) : void
    {
        $dtypeX = $X->dtype();
        if($dtypeX==NDArray::float64) {
            $this->assertFP64();
        }
        $type = $this->dtypeToOpenCLType[$dtypeX];
        $kernel_name = "greater_${type}";
        if(!isset($this->sources[$kernel_name])) {
            $this->sources[$kernel_name] =
                "__kernel void ${kernel_name}(\n".
                "    const        ${type} alpha,\n".
                "        __global ${type} * x,\n".
                "    const        uint offset_x,\n".
                "    const        uint incx)\n".
                "{\n".
                "    uint idx = get_global_id(0)*incx+offset_x;\n".
                "    if(x[idx] > alpha) {\n".
                "        x[idx] = 1.0;\n".
                "    } else {\n".
                "        x[idx] = 0.0;\n".
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
     *     X := 1  (X < a)
     *     X := 0  (X >= a)
     */
    public function less(
        int $n,
        float $alpha,
        Buffer $X, int $offsetX, int $incX,
        EventList $events=null, EventList $waitEvents=null
        ) : void
    {
        $dtypeX = $X->dtype();
        if($dtypeX==NDArray::float64) {
            $this->assertFP64();
        }
        $type = $this->dtypeToOpenCLType[$dtypeX];
        $kernel_name = "less_${type}";
        if(!isset($this->sources[$kernel_name])) {
            $this->sources[$kernel_name] =
                "__kernel void ${kernel_name}(\n".
                "    const        ${type} alpha,\n".
                "        __global ${type} * x,\n".
                "    const        uint offset_x,\n".
                "    const        uint incx)\n".
                "{\n".
                "    uint idx = get_global_id(0)*incx+offset_x;\n".
                "    if(x[idx] < alpha) {\n".
                "        x[idx] = 1.0;\n".
                "    } else {\n".
                "        x[idx] = 0.0;\n".
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
     *    A(m,n) := X(n) * A(m,n)
     */
    public function multiply(
        bool $trans,
        int $m,
        int $n,
        Buffer $X, int $offsetX, int $incX,
        Buffer $A, int $offsetA, int $ldA,
        EventList $events=null, EventList $waitEvents=null
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
        $kernel_name = "multiply_${type}_${trans}";
        if(!isset($this->sources[$kernel_name])) {
            if($trans=='trans') {
                $index_a = 'col_id*lda+row_id+offset_a';
            } else {
                $index_a = 'row_id*lda+col_id+offset_a';
            }
            $this->sources[$kernel_name] =
                "__kernel void ${kernel_name}(\n".
                "    const global ${type} * x,\n".
                "    const        uint offset_x,\n".
                "    const        uint incx,\n".
                "        __global ${type} * a,\n".
                "    const        uint offset_a,\n".
                "    const        uint lda)\n".
                "{\n".
                "    const uint row_id = get_global_id(0);\n".
                "    const uint col_id = get_global_id(1);\n".
                "    const uint index_a = ${index_a};\n".
                "    const uint index_x = col_id*incx+offset_x;\n".
                "    const ${type} work_x = x[index_x];\n".
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
        Buffer $X, int $offsetX, int $incX,
        Buffer $A, int $offsetA, int $ldA,
        EventList $events=null, EventList $waitEvents=null
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
        $kernel_name = "add_${type}_${trans}";
        if(!isset($this->sources[$kernel_name])) {
            if($trans=='trans') {
                $index_a = 'col_id*lda+row_id+offset_a';
            } else {
                $index_a = 'row_id*lda+col_id+offset_a';
            }
            $this->sources[$kernel_name] =
                "__kernel void ${kernel_name}(\n".
                "    const        ${type} alpha,\n".
                "    const global ${type} * x,\n".
                "    const        uint offset_x,\n".
                "    const        uint incx,\n".
                "        __global ${type} * a,\n".
                "    const        uint offset_a,\n".
                "    const        uint lda)\n".
                "{\n".
                "    const uint row_id = get_global_id(0);\n".
                "    const uint col_id = get_global_id(1);\n".
                "    const uint index_a = ${index_a};\n".
                "    const uint index_x = col_id*incx+offset_x;\n".
                "    const ${type} work_x = x[index_x];\n".
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
        Buffer $X, int $offsetX, int $incX,
        EventList $events=null, EventList $waitEvents=null
        ) : void
    {
        $dtypeX = $X->dtype();
        if($dtypeX==NDArray::float64) {
            $this->assertFP64();
        }
        $type = $this->dtypeToOpenCLType[$dtypeX];
        $kernel_name = "square_${type}";
        if(!isset($this->sources[$kernel_name])) {
            $this->sources[$kernel_name] =
                "__kernel void ${kernel_name}(\n".
                "        __global ${type} * x,\n".
                "    const        uint offset_x,\n".
                "    const        uint incx)\n".
                "{\n".
                "    uint idx = get_global_id(0)*incx+offset_x;\n".
                "    x[idx] = x[idx] * x[idx];\n".
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
     *     X := sqrt(X)
     */
    public function sqrt(
        int $n,
        Buffer $X, int $offsetX, int $incX,
        EventList $events=null, EventList $waitEvents=null
        ) : void
    {
        $dtypeX = $X->dtype();
        if($dtypeX==NDArray::float64) {
            $this->assertFP64();
        }
        $type = $this->dtypeToOpenCLType[$dtypeX];
        $kernel_name = "sqrt_${type}";
        if(!isset($this->sources[$kernel_name])) {
            $this->sources[$kernel_name] =
                "__kernel void ${kernel_name}(\n".
                "        __global ${type} * x,\n".
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
        Buffer $X, int $offsetX, int $incX,
        float $beta,
        EventList $events=null, EventList $waitEvents=null
        ) : void
    {
        $dtypeX = $X->dtype();
        if($dtypeX==NDArray::float64) {
            $this->assertFP64();
        }
        $type = $this->dtypeToOpenCLType[$dtypeX];
        $kernel_name = "rsqrt_${type}";
        if(!isset($this->sources[$kernel_name])) {
            $this->sources[$kernel_name] =
                "__kernel void ${kernel_name}(\n".
                "    const        ${type} alpha,\n".
                "        __global ${type} * x,\n".
                "    const        uint offset_x,\n".
                "    const        uint incx,\n".
                "    const        ${type} beta)\n".
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
     *     X := X ^ a
     */
    public function pow(
        int $n,
        float $alpha,
        Buffer $X, int $offsetX, int $incX,
        EventList $events=null, EventList $waitEvents=null
        ) : void
    {
        $dtypeX = $X->dtype();
        if($dtypeX==NDArray::float64) {
            $this->assertFP64();
        }
        $type = $this->dtypeToOpenCLType[$dtypeX];
        $kernel_name = "pow_${type}";
        if(!isset($this->sources[$kernel_name])) {
            $this->sources[$kernel_name] =
                "__kernel void ${kernel_name}(\n".
                "    const        ${type} alpha,\n".
                "        __global ${type} * x,\n".
                "    const        uint offset_x,\n".
                "    const        uint incx)\n".
                "{\n".
                "    uint idx = get_global_id(0)*incx+offset_x;\n".
                "    x[idx] = pow(x[idx],alpha);\n".
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
     *     X(i) := e ^ X(i)
     */
    public function exp(
        int $n,
        Buffer $X, int $offsetX, int $incX,
        EventList $events=null, EventList $waitEvents=null
        ) : void
    {
        $dtypeX = $X->dtype();
        if($dtypeX==NDArray::float64) {
            $this->assertFP64();
        }
        $type = $this->dtypeToOpenCLType[$dtypeX];
        $kernel_name = "exp_${type}";
        if(!isset($this->sources[$kernel_name])) {
            $this->sources[$kernel_name] =
                "__kernel void ${kernel_name}(\n".
                "        __global ${type} * x,\n".
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
        Buffer $X, int $offsetX, int $incX,
        EventList $events=null, EventList $waitEvents=null
        ) : void
    {
        $dtypeX = $X->dtype();
        if($dtypeX==NDArray::float64) {
            $this->assertFP64();
        }
        $type = $this->dtypeToOpenCLType[$dtypeX];
        $kernel_name = "log_${type}";
        if(!isset($this->sources[$kernel_name])) {
            $this->sources[$kernel_name] =
                "__kernel void ${kernel_name}(\n".
                "        __global ${type} * x,\n".
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
        Buffer $X, int $offsetX, int $incX,
        EventList $events=null, EventList $waitEvents=null
        ) : void
    {
        $dtypeX = $X->dtype();
        if($dtypeX==NDArray::float64) {
            $this->assertFP64();
        }
        $type = $this->dtypeToOpenCLType[$dtypeX];
        $kernel_name = "tanh_${type}";
        if(!isset($this->sources[$kernel_name])) {
            $this->sources[$kernel_name] =
                "__kernel void ${kernel_name}(\n".
                "        __global ${type} * x,\n".
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
     * Y(i) := 1  ( X(i) == Y(i) )
     * Y(i) := 0  ( X(i) != Y(i) )
     */
    public function equal(
        int $n,
        Buffer $X, int $offsetX, int $incX,
        Buffer $Y, int $offsetY, int $incY,
        EventList $events=null, EventList $waitEvents=null
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
        $kernel_name = "equal_${dtype}";
        if(!isset($this->sources[$kernel_name])) {
            $this->sources[$kernel_name] =
                "__kernel void ${kernel_name}(\n".
                "    const global ${dtype} * x,\n".
                "    const        uint offset_x,\n".
                "    const        uint incx,\n".
                "        __global ${dtype} * y,\n".
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
     *     A(m,n) := X(n)
     */
    public function duplicate(
        bool $trans,
        int $m,
        int $n,
        Buffer $X, int $offsetX, int $incX,
        Buffer $A, int $offsetA, int $ldA,
        EventList $events=null, EventList $waitEvents=null
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
        $kernel_name = "duplicate_${type}_${trans}";
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
                "__kernel void ${kernel_name}(\n".
                "    const global ${type} * x,\n".
                "    const        uint offset_x,\n".
                "    const        uint incx,\n".
                "        __global ${type} * a,\n".
                "    const        uint offset_a,\n".
                "    const        uint lda)\n".
                "{\n".
                "    uint i = get_global_id(${idxI});\n".
                "    uint j = get_global_id(${idxJ});\n".
                "    uint index_a = ${index_a};\n".
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
        Buffer $X, int $offsetX, int $incX,
        Buffer $Y, int $offsetY, int $incY,
        EventList $events=null, EventList $waitEvents=null
        ) : void
    {
        $dtypeX = $X->dtype();
        $dtypeY = $Y->dtype();
        if($dtypeX==NDArray::float64 || $dtypeY==NDArray::float64) {
            $this->assertFP64();
        }
        $from = $this->dtypeToOpenCLType[$dtypeX];
        $to = $this->dtypeToOpenCLType[$dtypeY];
        $kernel_name = "astype_${from}_${to}";
        if(!isset($this->sources[$kernel_name])) {
            $this->sources[$kernel_name] =
                "__kernel void ${kernel_name}(\n".
                "    const global ${from} * x,\n".
                "    const        uint offset_x,\n".
                "    const        uint incx,\n".
                "        __global ${to} * y,\n".
                "    const        uint offset_y,\n".
                "    const        uint incy)\n".
                "{\n".
                "    uint gid = get_global_id(0);\n".
                "    y[gid*incy+offset_y] = x[gid*incx+offset_x];\n".
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
     *     B(k,n) := A(X(k),n)
     */
    public function selectAxis0(
        int $m,
        int $n,
        int $k,
        Buffer $A, int $offsetA, int $ldA,
        Buffer $X, int $offsetX, int $incX,
        Buffer $B, int $offsetB, int $ldB,
        EventList $events=null, EventList $waitEvents=null
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

        $type = $this->dtypeToOpenCLType[$dtype];
        $kernel_name = "selectAxis0_${type}";
        if(!isset($this->sources[$kernel_name])) {
            $this->sources[$kernel_name] =
                "__kernel void ${kernel_name}(\n".
                "    const        uint m,\n".
                "    const global ${type} * a,\n".
                "    const        uint offset_a,\n".
                "    const        uint lda,\n".
                "    const global uint * x,\n".
                "    const        uint offset_x,\n".
                "    const        uint incx,\n".
                "        __global ${type} * b,\n".
                "    const        uint offset_b,\n".
                "    const        uint ldb)\n".
                "{\n".
                "    uint i = get_global_id(0);\n".
                "    uint j = get_global_id(1);\n".
                "    ulong label = x[i*incx+offset_x];\n".
                "    if(label<m) {\n".
                "        b[i*ldb+j+offset_b] = a[label*lda+j+offset_a];\n".
                "    }\n".
                "}\n";
        }
        $kernel = $this->createKernel($kernel_name);
        $kernel->setArg(0,$m,NDArray::uint32);
        $kernel->setArg(1,$A);
        $kernel->setArg(2,$offsetA,NDArray::uint32);
        $kernel->setArg(3,$ldA,NDArray::uint32);
        $kernel->setArg(4,$X);
        $kernel->setArg(5,$offsetX,NDArray::uint32);
        $kernel->setArg(6,$incX,NDArray::uint32);
        $kernel->setArg(7,$B);
        $kernel->setArg(8,$offsetB,NDArray::uint32);
        $kernel->setArg(9,$ldB,NDArray::uint32);
        $global_work_size = [$k,$n];
        $kernel->enqueueNDRange($this->queue,$global_work_size,null,null,
            $events,$waitEvents);
    }

    /**
     *     Y(k) := A(k,X(m))
     */
    public function selectAxis1(
        int $m,
        int $n,
        Buffer $A, int $offsetA, int $ldA,
        Buffer $X, int $offsetX, int $incX,
        Buffer $Y, int $offsetY, int $incY,
        EventList $events=null, EventList $waitEvents=null
        ) : void
    {
        $dtype = $A->dtype();
        if($X->dtype()!=NDArray::int32 && $X->dtype()!=NDArray::uint32) {
            throw new InvalidArgumentException("X must be 32bit integer:".
                                            $this->dtypeToString($X->dtype()));
        }
        if($dtype!=$Y->dtype()) {
            throw new InvalidArgumentException("Unmatch data type A and Y:".
                $this->dtypeToString($A->dtype()).",".$this->dtypeToString($Y->dtype()));
        }
        if($dtype==NDArray::float64) {
            $this->assertFP64();
        }
        $type = $this->dtypeToOpenCLType[$dtype];
        $kernel_name = "selectAxis1_${type}";
        if(!isset($this->sources[$kernel_name])) {
            $this->sources[$kernel_name] =
                "__kernel void ${kernel_name}(\n".
                "    const        uint n,\n".
                "    const global ${type} * a,\n".
                "    const        uint offset_a,\n".
                "    const        uint lda,\n".
                "    const global uint * x,\n".
                "    const        uint offset_x,\n".
                "    const        uint incx,\n".
                "        __global ${type} * y,\n".
                "    const        uint offset_y,\n".
                "    const        uint incy)\n".
                "{\n".
                "    uint gid = get_global_id(0);\n".
                "    ulong label = x[gid*incx+offset_x];\n".
                "    if(label<n) {\n".
                "        y[gid*incy+offset_y] = a[lda*gid+label+offset_a];\n".
                "    }\n".
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

    protected function getHomeDirectory()
    {
        if(PHP_OS=='WINNT') {
            return getenv('USERPROFILE');
        } elseif(PHP_OS=='Linux') {
            return getenv('HOME');
        }
    }

    protected function loadParameter($filename)
    {
        $filepath = $this->getHomeDirectory().'/.rindow/'.$filename;
        if(!file_exists($filepath)) {
            $filepath = __DIR__.'/params/'.$filename;
        }
        $times = include $filepath;
        return $times;
    }

    public function predictTimeScatterAddAxis0($mode,$numClass,$cols,$rows)
    {
        if(isset($this->timesPredictionScatterAddAxis0[$mode])) {
            $times = $this->timesPredictionScatterAddAxis0[$mode];
        } else {
            $times = $this->loadParameter('ScatterAddTimesMode'.$mode.'.php');
            $this->timesPredictionScatterAddAxis0[$mode] = $times;
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
    public function scatterAxis0(
        int $m,
        int $n,
        int $k,
        Buffer $A, int $offsetA, int $ldA,
        Buffer $X, int $offsetX, int $incX,
        Buffer $B, int $offsetB, int $ldB,
        bool $addMode,
        EventList $events=null, EventList $waitEvents=null
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
        if($A->dtype()==NDArray::float64) {
            $this->assertFP64();
        }

        if($addMode) {
            $small = $max_work_items = $this->maxWorkItem[0];
            $mediam = $max_work_items*$max_work_items*2;
            for($bm=8; $bm<$m;$bm<<=1) { ; }
            for($bn=8; $bn<$n;$bn<<=1) { ; }
            for($bk=8; $bk<$k;$bk<<=1) { ; }
            //echo "($m,$n,$k)\n";
            $mode1 = $this->predictTimeScatterAddAxis0(1,$bm,$bn,$bk);
            $mode2 = $this->predictTimeScatterAddAxis0(2,$bm,$bn,$bk);
            $mode3 = $this->predictTimeScatterAddAxis0(3,$bm,$bn,$bk);
            $mode4 = $this->predictTimeScatterAddAxis0(4,$bm,$bn,$bk);
            //echo 'mode1='.number_format($mode1)."\n";
            //echo 'mode2='.number_format($mode2)."\n";
            //echo 'mode3='.number_format($mode3)."\n";
            //echo 'mode4='.number_format($mode4)."\n";
            $imin1 = ($mode1 < $mode2) ? 1 : 2;
            $min1 = ($mode1 < $mode2) ? $mode1 : $mode2;
            $imin2 = ($mode3 < $mode4) ? 3 : 4;
            $min2 = ($mode3 < $mode4) ? $mode3 : $mode4;
            $mode = ($min1 < $min2) ? $imin1 : $imin2;
            //$min = ($min1 < $min2) ? $min1 : $min2;
            //if($min==PHP_INT_MAX) { echo "scatterAxis0,"; }
            if($mode==2 && $bk<=$small) {
                $mode=1;
            }
            switch($mode) {
                case 1:{
                    $this->scatterAddAxis0_1(
                        $m,$n,$k,
                        $A, $offsetA, $ldA,
                        $X, $offsetX, $incX,
                        $B, $offsetB, $ldB,
                        $addMode,
                        $events, $waitEvents
                    );
                    break;
                }
                case 2:{
                    $this->scatterAddAxis0_2(
                        $m,$n,$k,
                        $A, $offsetA, $ldA,
                        $X, $offsetX, $incX,
                        $B, $offsetB, $ldB,
                        $addMode,
                        $events, $waitEvents
                    );
                    break;
                }
                case 3:{
                    $this->scatterAddAxis0_3(
                        $m,$n,$k,
                        $A, $offsetA, $ldA,
                        $X, $offsetX, $incX,
                        $B, $offsetB, $ldB,
                        $addMode,
                        $events, $waitEvents
                    );
                    break;
                }
                case 4:{
                    $this->scatterAddAxis0_4(
                        $m,$n,$k,
                        $A, $offsetA, $ldA,
                        $X, $offsetX, $incX,
                        $B, $offsetB, $ldB,
                        $addMode,
                        $events, $waitEvents
                    );
                    break;
                }
            }
            return;
        }

        $type = $this->dtypeToOpenCLType[$B->dtype()];
        $kernel_name = "scatterAxis0_${type}_set";
        if(!isset($this->sources[$kernel_name])) {
            $this->sources[$kernel_name] =
                "__kernel void ${kernel_name}(\n".
                "    const        uint m,\n".
                "        __global ${type} * a,\n".
                "    const        uint offset_a,\n".
                "    const        uint lda,\n".
                "    const global uint * x,\n".
                "    const        uint offset_x,\n".
                "    const        uint incx,\n".
                "    const global ${type} * b,\n".
                "    const        uint offset_b,\n".
                "    const        uint ldb)\n".
                "{\n".
                "    uint i = get_global_id(0);\n".
                "    uint j = get_global_id(1);\n".
                "    uint label = x[i*incx+offset_x];\n".
                "    if(label<m) {\n".
                "        a[label*lda+j+offset_a] = b[i*ldb+j+offset_b];\n".
                "    }\n".
                "}\n";
        }
        $kernel = $this->createKernel($kernel_name);
        $kernel->setArg(0,$m,NDArray::uint32);
        $kernel->setArg(1,$A);
        $kernel->setArg(2,$offsetA,NDArray::uint32);
        $kernel->setArg(3,$ldA,NDArray::uint32);
        $kernel->setArg(4,$X);
        $kernel->setArg(5,$offsetX,NDArray::uint32);
        $kernel->setArg(6,$incX,NDArray::uint32);
        $kernel->setArg(7,$B);
        $kernel->setArg(8,$offsetB,NDArray::uint32);
        $kernel->setArg(9,$ldB,NDArray::uint32);
        $global_work_size = [$k,$n];
        $local_work_size = null;
        $kernel->enqueueNDRange($this->queue,$global_work_size,$local_work_size,null,
            $events,$waitEvents);
    }

    /**
     * A(X[k],n) := B[k,n] ,  m > max(X[k])
     */
    public function scatterAddAxis0_1(
        int $m,
        int $n,
        int $k,
        Buffer $A, int $offsetA, int $ldA,
        Buffer $X, int $offsetX, int $incX,
        Buffer $B, int $offsetB, int $ldB,
        bool $addMode,
        EventList $events=null, EventList $waitEvents=null
        ) : void
    {
//echo "mode=1\n";
        $dtype = $A->dtype();
        $total_local_items = $k;
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
        $kernel_name = "scatterAxis0_S_${type}_add";
        if(!isset($this->sources[$kernel_name])) {
            $this->sources[$kernel_name] =
                "__kernel void ${kernel_name}(\n".
                "    const        uint total_local_items,\n".
                "        __global ${type} * a,\n".
                "    const        uint offset_a,\n".
                "    const        uint lda,\n".
                "    const global uint * x,\n".
                "    const        uint offset_x,\n".
                "    const        uint incx,\n".
                "    const global ${type} * b,\n".
                "    const        uint offset_b,\n".
                "    const        uint ldb,\n".
                "         __local ${type} * local_work,\n".
                "    const        uint work_items)\n".
                "{\n".
                "    const uint grid = get_group_id(0);\n".  // A row id
                "    const uint gid1 = get_global_id(1);\n".  // A col id
                     $this->kernelTemplateSSum(
                         "uint label = x[lid*incx+offset_x];\n".
                         "if(label==grid) {\n".
                         "    local_work[lid] = b[lid*ldb+gid1+offset_b];\n".
                         "} else {\n".
                         "    local_work[lid] = 0;\n".
                         "}\n",
                         "a[grid*lda+gid1+offset_a] += local_work[0];\n"
                     ).
                "}\n";
        }
        $kernel = $this->createKernel($kernel_name);

        $kernel->setArg(0,$total_local_items,NDArray::uint32);
        $kernel->setArg(1,$A);
        $kernel->setArg(2,$offsetA,NDArray::uint32);
        $kernel->setArg(3,$ldA,NDArray::uint32);
        $kernel->setArg(4,$X);
        $kernel->setArg(5,$offsetX,NDArray::uint32);
        $kernel->setArg(6,$incX,NDArray::uint32);
        $kernel->setArg(7,$B);
        $kernel->setArg(8,$offsetB,NDArray::uint32);
        $kernel->setArg(9,$ldB,NDArray::uint32);
        $kernel->setArg(10,null,$max_work_items*$value_size);
        $kernel->setArg(11,$work_items,NDArray::uint32);
        $global_work_size = [$max_work_items*$m,$n];
        $local_work_size = [$max_work_items,1];
        $kernel->enqueueNDRange($this->queue,$global_work_size,$local_work_size,null,
            $events,$waitEvents);

    }

    /**
     * A(X[k],n) := B[k,n] ,  m > max(X[k])
     */
    public function scatterAddAxis0_2(
        int $m,
        int $n,
        int $k,
        Buffer $A, int $offsetA, int $ldA,
        Buffer $X, int $offsetX, int $incX,
        Buffer $B, int $offsetB, int $ldB,
        bool $addMode,
        EventList $events=null, EventList $waitEvents=null
        ) : void
    {
//echo "mode=2\n";
        $dtype = $A->dtype();
        $total_local_items = $k;
        $max_work_items = $this->maxWorkItem[0];
        if($total_local_items>$max_work_items) {
            $segments = (int)floor(($total_local_items+$max_work_items-1)/$max_work_items); // round up float
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
        $kernel_name = "scatterAxis0_M_${type}_add";
        if(!isset($this->sources[$kernel_name])) {
            $this->sources[$kernel_name] =
                "__kernel void ${kernel_name}(\n".
                "    const        uint total_local_items,\n".
                "        __global ${type} * a,\n".
                "    const        uint offset_a,\n".
                "    const        uint lda,\n".
                "    const global uint * x,\n".
                "    const        uint offset_x,\n".
                "    const        uint incx,\n".
                "    const global ${type} * b,\n".
                "    const        uint offset_b,\n".
                "    const        uint ldb,\n".
                "    const        uint segments,\n".
                "         __local ${type} * local_work,\n".
                "         __local ${type} * seg_work,\n".
                "    const        uint work_items)\n".
                "{\n".
                "    const uint grid = get_group_id(0);\n".  // A row id
                "    const uint gid1 = get_global_id(1);\n".  // A col id
                     $this->kernelTemplateQSum(
                         "uint label = x[(seg*lws+lid)*incx+offset_x];\n".
                         "if(label==grid) {\n".
                         "    local_work[lid] = b[(seg*lws+lid)*ldb+gid1+offset_b];\n".
                         "} else {\n".
                         "    local_work[lid] = 0;\n".
                         "}\n",
                         "a[grid*lda+gid1+offset_a] += seg_work[0];\n"
                     ).
                "}\n";
        }
        $kernel = $this->createKernel($kernel_name);

        $kernel->setArg(0,$total_local_items,NDArray::uint32);
        $kernel->setArg(1,$A);
        $kernel->setArg(2,$offsetA,NDArray::uint32);
        $kernel->setArg(3,$ldA,NDArray::uint32);
        $kernel->setArg(4,$X);
        $kernel->setArg(5,$offsetX,NDArray::uint32);
        $kernel->setArg(6,$incX,NDArray::uint32);
        $kernel->setArg(7,$B);
        $kernel->setArg(8,$offsetB,NDArray::uint32);
        $kernel->setArg(9,$ldB,NDArray::uint32);
        $kernel->setArg(10,$segments,NDArray::uint32);
        $kernel->setArg(11,null,$max_work_items*$value_size);
        $kernel->setArg(12,null,$segments*$value_size);
        $kernel->setArg(13,$work_items,NDArray::uint32);
        $global_work_size = [$max_work_items*$m,$n];
        $local_work_size = [$max_work_items,1];
        $kernel->enqueueNDRange($this->queue,$global_work_size,$local_work_size,null,
            $events,$waitEvents);
    }

    /**
     * A(X[k],n) := B[k,n] ,  m > max(X[k])
     */
    public function scatterAddAxis0_3(
        int $m,
        int $n,
        int $k,
        Buffer $A, int $offsetA, int $ldA,
        Buffer $X, int $offsetX, int $incX,
        Buffer $B, int $offsetB, int $ldB,
        bool $addMode,
        EventList $events=null, EventList $waitEvents=null
        ) : void
    {
//echo "mode=3\n";
        $dtype = $A->dtype();
        $total_local_items = $k;
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
            $value_size*$temp_size*$m*$n,
            OpenCL::CL_MEM_READ_WRITE,null,null,$dtype);

        $type = $this->dtypeToOpenCLType[$dtype];
        $kernel_name1 = "scatterAxis0_L1_${type}_add";
        $kernel_name2 = "scatterAxis0_L2_${type}_add";
        if(!isset($this->sources[$kernel_name1])) {
            $this->sources[$kernel_name1] =
                "__kernel void ${kernel_name1}(\n".
                "    const        uint total_local_items,\n".
                "    const        uint cols,\n".
                "    const global uint * x,\n".
                "    const        uint offset_x,\n".
                "    const        uint incx,\n".
                "    const global ${type} * b,\n".
                "    const        uint offset_b,\n".
                "    const        uint ldb,\n".
                "        __global ${type} * temp_buffer,\n".
                "         __local ${type} * local_work)\n".
                "{\n".
                "    const uint parallel_item_id = get_global_id(1);\n".
                    $this->kernelTemplateLSum1(
                        "${type} input;\n".
                        "const uint row_id = parallel_item_id/cols;\n".
                        "const uint col_id = parallel_item_id%cols;\n".
                        "const uint label = x[local_item_id*incx+offset_x];\n".
                        "if(label==row_id) {\n".
                        "    input = b[local_item_id*ldb+col_id+offset_b];\n".
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
                "__kernel void ${kernel_name2}(\n".
                "    const        uint cols,\n".
                "    const __global ${type} * temp_buffer,\n".
                "        __global ${type} * a,\n".
                "    const        uint offset_a,\n".
                "    const        uint lda,\n".
                "         __local ${type} * local_work)\n".
                "{\n".
                "    const uint parallel_item_id = get_global_id(1);\n".
                    $this->kernelTemplateLSum2(
                        "const uint row_id = parallel_item_id/cols;\n".
                        "const uint col_id = parallel_item_id%cols;\n".
                        "a[row_id*lda+col_id+offset_a] += local_work[0];\n"
                    ).
                "}\n";
        }
        $kernel2 = $this->createKernel($kernel_name2);

        $kernel->setArg(0,$total_local_items,NDArray::uint32);
        $kernel->setArg(1,$n,NDArray::uint32);
        $kernel->setArg(2,$X);
        $kernel->setArg(3,$offsetX,NDArray::uint32);
        $kernel->setArg(4,$incX,NDArray::uint32);
        $kernel->setArg(5,$B);
        $kernel->setArg(6,$offsetB,NDArray::uint32);
        $kernel->setArg(7,$ldB,NDArray::uint32);
        $kernel->setArg(8,$temp_buffer);
        $kernel->setArg(9,null,$work_items1*$value_size);
        $global_work_size = [$work_items1*$temp_size,$m*$n];
        $local_work_size = [$work_items1,1];
        $phase1Events = $this->newEventList();
        $kernel->enqueueNDRange($this->queue,$global_work_size,$local_work_size,null,
                $phase1Events,$waitEvents);

        $kernel2->setArg(0,$n,NDArray::uint32);
        $kernel2->setArg(1,$temp_buffer);
        $kernel2->setArg(2,$A);
        $kernel2->setArg(3,$offsetA,NDArray::uint32);
        $kernel2->setArg(4,$ldA,NDArray::uint32);
        $kernel2->setArg(5,null,$work_items2*$value_size);
        $global_work_size = [$work_items2,$m*$n];
        $local_work_size = [$work_items2,1];
        $kernel2->enqueueNDRange($this->queue,$global_work_size,$local_work_size,null,
                $events,$phase1Events);
    }

    /**
     * A(X[k],n) := B[k,n] ,  m > max(X[k])
     */
    public function scatterAddAxis0_4(
        int $m,
        int $n,
        int $k,
        Buffer $A, int $offsetA, int $ldA,
        Buffer $X, int $offsetX, int $incX,
        Buffer $B, int $offsetB, int $ldB,
        bool $addMode,
        EventList $events=null, EventList $waitEvents=null
        ) : void
    {
//echo "mode=4\n";
        $type = $this->dtypeToOpenCLType[$B->dtype()];
        $kernel_name = "scatterAxis0_4_${type}_add";
        if(!isset($this->sources[$kernel_name])) {
            $this->sources[$kernel_name] =
                "__kernel void ${kernel_name}(\n".
                "    const        uint k,\n".
                "    const        uint m,\n".
                "        __global ${type} * a,\n".
                "    const        uint offset_a,\n".
                "    const        uint lda,\n".
                "    const global uint * x,\n".
                "    const        uint offset_x,\n".
                "    const        uint incx,\n".
                "    const global ${type} * b,\n".
                "    const        uint offset_b,\n".
                "    const        uint ldb)\n".
                "{\n".
                "    const uint grid = get_group_id(0);\n".
                "    uint xpos = offset_x;\n".
                "    uint bpos = offset_b;\n".
                "    for(uint i=0;i<k;i++,xpos+=incx,bpos+=ldb) {\n".
                "        uint label = x[xpos];\n".
                "        if(label<m) {\n".
                "            a[label*lda+grid+offset_a] += b[bpos+grid];\n".
                "        }\n".
                "    }\n".
                "}\n";
        }
        $kernel = $this->createKernel($kernel_name);

        $kernel->setArg(0,$k,NDArray::uint32);
        $kernel->setArg(1,$m,NDArray::uint32);
        $kernel->setArg(2,$A);
        $kernel->setArg(3,$offsetA,NDArray::uint32);
        $kernel->setArg(4,$ldA,NDArray::uint32);
        $kernel->setArg(5,$X);
        $kernel->setArg(6,$offsetX,NDArray::uint32);
        $kernel->setArg(7,$incX,NDArray::uint32);
        $kernel->setArg(8,$B);
        $kernel->setArg(9,$offsetB,NDArray::uint32);
        $kernel->setArg(10,$ldB,NDArray::uint32);
        $global_work_size = [$n];
        $local_work_size = [1];
        $kernel->enqueueNDRange($this->queue,$global_work_size,$local_work_size,null,
            $events,$waitEvents);
    }

    /**
     *      A(m,n) := B(m,1)  ( n = X(m) )
     */
    public function scatterAxis1(
        int $m,
        int $n,
        Buffer $A, int $offsetA, int $ldA,
        Buffer $X, int $offsetX, int $incX,
        Buffer $Y, int $offsetY, int $incY,
        bool $addMode,
        EventList $events=null, EventList $waitEvents=null
        ) : void
    {
        if($X->dtype()!=NDArray::int32 && $X->dtype()!=NDArray::uint32) {
            throw new InvalidArgumentException("X must be 32bit integer:".
                                            $this->dtypeToString($X->dtype()));
        }
        if($A->dtype()!=$Y->dtype()) {
            throw new InvalidArgumentException("Unmatch data type A and Y:".
            $this->dtypeToString($A->dtype()).",".$this->dtypeToString($Y->dtype()));
        }
        if($A->dtype()==NDArray::float64) {
            $this->assertFP64();
        }

        $type = $this->dtypeToOpenCLType[$Y->dtype()];
        $mode = $addMode ? 'add' : 'set';
        $operator = $addMode ? '+=' : '=';
        if(!isset($this->sources["scatterAxis1_${type}_${mode}"])) {
            $this->sources["scatterAxis1_${type}_${mode}"] =
                "__kernel void scatterAxis1_${type}_${mode}(\n".
                "    const        uint n,\n".
                "        __global ${type} * a,\n".
                "    const        uint offset_a,\n".
                "    const        uint lda,\n".
                "    const global uint * x,\n".
                "    const        uint offset_x,\n".
                "    const        uint incx,\n".
                "    const global ${type} * y,\n".
                "    const        uint offset_y,\n".
                "    const        uint incy)\n".
                "{\n".
                "    uint gid = get_global_id(0);\n".
                "    uint label = x[gid*incx+offset_x];\n".
                "    if(label<n) {\n".
                "        a[gid*lda+label+offset_a] ${operator} y[gid*incy+offset_y];\n".
                "    }\n".
                "}\n";
        }
        $kernel = $this->createKernel("scatterAxis1_${type}_${mode}");

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
     *     Y := onehot(X,a)
     */
    public function onehot(
        int $m,
        int $n,
        float $alpha,
        Buffer $X, int $offsetX, int $incX,
        Buffer $Y, int $offsetY, int $ldY,
        bool $addMode,
        EventList $events=null, EventList $waitEvents=null
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
        if(!isset($this->sources["onehot_${type}_${mode}"])) {
            $this->sources["onehot_${type}_${mode}"] =
                "__kernel void onehot_${type}_${mode}(\n".
                "    const        uint n,\n".
                "    const        ${type} alpha,\n".
                "    const global uint * x,\n".
                "    const        uint offset_x,\n".
                "    const        uint incx,\n".
                "        __global ${type} * y,\n".
                "    const        uint offset_y,\n".
                "    const        uint ldy)\n".
                "{\n".
                "    uint gid = get_global_id(0);\n".
                "    uint label = x[gid*incx+offset_x];\n".
                "    if(label<n) {\n".
                "        y[gid*ldy+label+offset_y] ${operator} alpha;\n".
                "    }\n".
                "}\n";
        }
        $kernel = $this->createKernel("onehot_${type}_${mode}");

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

    /**
     * X(m) := sum( A(m,n) )
     */
    public function reduceSum(
        bool $trans,
        int $m,
        int $n,
        Buffer $A, int $offsetA, int $ldA,
        Buffer $X, int $offsetX, int $incX,
        EventList $events=null, EventList $waitEvents=null
        ) : void
    {
        $dtype = $A->dtype();
        if($dtype!=$X->dtype()) {
            throw new InvalidArgumentException("Unmatch data type A and X:".
            $this->dtypeToString($dtype).",".$this->dtypeToString($X->dtype()));
        }
        if($dtype!=NDArray::float64 && $dtype!=NDArray::float32) {
            throw new InvalidArgumentException("Unsuppored data type:".
                                            $this->dtypeToString($dtype));
        }
        if($dtype==NDArray::float64) {
            $this->assertFP64();
        }
        if($trans) {
            $cols = $m;
        } else {
            $cols = $n;
        }
        $max_work_items = $this->maxWorkItem[0];
        if($cols <= $max_work_items) {
            $this->reduceSum1(
                $trans,
                $m,
                $n,
                $A,$offsetA,$ldA,
                $X,$offsetX,$incX,
                $events,$waitEvents
            );
        } elseif($cols <= $max_work_items*$max_work_items*2) {
            $this->reduceSum2(
                $trans,
                $m,
                $n,
                $A,$offsetA,$ldA,
                $X,$offsetX,$incX,
                $events,$waitEvents
            );
        } else {
            $this->reduceSum3(
                $trans,
                $m,
                $n,
                $A,$offsetA,$ldA,
                $X,$offsetX,$incX,
                $events,$waitEvents
            );
        }
    }

    /**
     * X(m) := sum( A(m,n) )
     */
    public function reduceSum1(
        bool $trans,
        int $m,
        int $n,
        Buffer $A, int $offsetA, int $ldA,
        Buffer $X, int $offsetX, int $incX,
        EventList $events=null, EventList $waitEvents=null
        ) : void
    {
        $dtype = $A->dtype();
        if($trans) {
            $trans = 'trans';
            $rows = $n;
            $cols = $m;
        } else {
            $trans = 'norm';
            $rows = $m;
            $cols = $n;
        }
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
        $kernel_name = "reduceSum_S_${type}_${trans}";
        if(!isset($this->sources[$kernel_name])) {
            if($trans=='trans') {
                $index_a = 'lid*lda+grid+offset_a';
            } else {
                $index_a = 'grid*lda+lid+offset_a';
            }
            $this->sources[$kernel_name] =
                "__kernel void ${kernel_name}(\n".
                "    const        uint total_local_items,\n".
                "        __global ${type} * x,\n".
                "    const        uint offset_x,\n".
                "    const        uint incx,\n".
                "    const global ${type} * a,\n".
                "    const        uint offset_a,\n".
                "    const        uint lda,\n".
                "         __local ${type} * local_work)\n".
                "{\n".
                "    const uint grid = get_group_id(0);\n".
                     $this->kernelTemplateSSum(
                         "local_work[lid] = a[${index_a}];\n",
                         "x[grid*incx+offset_x] = local_work[0];\n"
                     ).
                "}\n";
        }
        $kernel = $this->createKernel($kernel_name);

        $kernel->setArg(0,$total_local_items,NDArray::uint32);
        $kernel->setArg(1,$X);
        $kernel->setArg(2,$offsetX,NDArray::uint32);
        $kernel->setArg(3,$incX,NDArray::uint32);
        $kernel->setArg(4,$A);
        $kernel->setArg(5,$offsetA,NDArray::uint32);
        $kernel->setArg(6,$ldA,NDArray::uint32);
        $kernel->setArg(7,null,$max_work_items*$value_size);
        $global_work_size = [$max_work_items*$rows];
        $local_work_size = [$max_work_items];
        $kernel->enqueueNDRange($this->queue,$global_work_size,$local_work_size,null,
                $events,$waitEvents);
    }

    /**
     * X(m) := sum( A(m,n) )
     */
    public function reduceSum2(
        bool $trans,
        int $m,
        int $n,
        Buffer $A, int $offsetA, int $ldA,
        Buffer $X, int $offsetX, int $incX,
        EventList $events=null, EventList $waitEvents=null
        ) : void
    {
        $dtype = $A->dtype();
        if($trans) {
            $trans = 'trans';
            $rows = $n;
            $cols = $m;
        } else {
            $trans = 'norm';
            $rows = $m;
            $cols = $n;
        }
        $total_local_items = $cols;
        $max_work_items = $this->maxWorkItem[0];
        if($total_local_items>$max_work_items) {
            $segments = (int)floor(($total_local_items+$max_work_items-1)/$max_work_items); // round up float
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
        $kernel_name = "reduceSum_M_${type}_${trans}";
        if(!isset($this->sources[$kernel_name])) {
            if($trans=='trans') {
                $index_a = '(seg*lws+lid)*lda+grid+offset_a';
            } else {
                $index_a = 'grid*lda+(seg*lws+lid)+offset_a';
            }
            $this->sources[$kernel_name] =
                "__kernel void ${kernel_name}(\n".
                "    const        uint total_local_items,\n".
                "    const        uint segments,\n".
                "        __global ${type} * x,\n".
                "    const        uint offset_x,\n".
                "    const        uint incx,\n".
                "    const global ${type} * a,\n".
                "    const        uint offset_a,\n".
                "    const        uint lda,\n".
                "         __local ${type} * local_work,\n".
                "         __local ${type} * seg_work,\n".
                "    const        uint work_items)\n".
                "{\n".
                "    const uint grid = get_group_id(0);\n".
                     $this->kernelTemplateQSum(
                         "local_work[lid] = a[${index_a}];\n",
                         "x[grid*incx+offset_x] = seg_work[0];\n"
                     ).
                "}\n";
        }
        $kernel = $this->createKernel($kernel_name);

        $kernel->setArg(0,$total_local_items,NDArray::uint32);
        $kernel->setArg(1,$segments,NDArray::uint32);
        $kernel->setArg(2,$X);
        $kernel->setArg(3,$offsetX,NDArray::uint32);
        $kernel->setArg(4,$incX,NDArray::uint32);
        $kernel->setArg(5,$A);
        $kernel->setArg(6,$offsetA,NDArray::uint32);
        $kernel->setArg(7,$ldA,NDArray::uint32);

        $kernel->setArg(8,null,$max_work_items*$value_size);
        $kernel->setArg(9,null,$segments*$value_size);
        $kernel->setArg(10,$work_items,NDArray::uint32);
        $global_work_size = [$max_work_items*$rows];
        $local_work_size = [$max_work_items];
        $kernel->enqueueNDRange($this->queue,$global_work_size,$local_work_size,null,
                $events,$waitEvents);
    }

    /**
     * X(m) := sum( A(m,n) )
     */
    public function reduceSum3(
        bool $trans,
        int $m,
        int $n,
        Buffer $A, int $offsetA, int $ldA,
        Buffer $X, int $offsetX, int $incX,
        EventList $events=null, EventList $waitEvents=null
        ) : void
    {
        $dtype = $A->dtype();
        if($trans) {
            $trans = 'trans';
            $rows = $n;
            $cols = $m;
        } else {
            $trans = 'norm';
            $rows = $m;
            $cols = $n;
        }
        $total_local_items = $cols;
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
            $value_size*$temp_size*$rows,
            OpenCL::CL_MEM_READ_WRITE,null,null,$dtype);

        $type = $this->dtypeToOpenCLType[$dtype];
        $kernel_name1 = "reduceSum_L1_${type}_${trans}";
        $kernel_name2 = "reduceSum_L2_${type}_${trans}";
        if(!isset($this->sources[$kernel_name1])) {
            if($trans=='trans') {
                $index_a = 'local_item_id*lda + parallel_item_id + offset_a';
            } else {
                $index_a = 'parallel_item_id*lda + local_item_id + offset_a';
            }
            $this->sources[$kernel_name1] =
                "__kernel void ${kernel_name1}(\n".
                "    const        uint total_local_items,\n".
                "    const __global ${type} * a,\n".
                "    const        uint offset_a,\n".
                "    const        uint lda,\n".
                "        __global ${type} * temp_buffer,\n".
                "         __local ${type} * local_work)\n".
                "{\n".
                "    const uint parallel_item_id = get_global_id(1);\n".
                    $this->kernelTemplateLSum1(
                        "${type} input = a[${index_a}];",
                        $dtype
                    ).
                "}\n";
        }
        $kernel = $this->createKernel($kernel_name1);

        if(!isset($this->sources[$kernel_name2])) {
            $this->sources[$kernel_name2] =
                "__kernel void ${kernel_name2}(\n".
                "    const __global ${type} * temp_buffer,\n".
                "        __global ${type} * x,\n".
                "    const        uint offset_x,\n".
                "    const        uint incx,\n".
                "         __local ${type} * local_work)\n".
                "{\n".
                "    const uint parallel_item_id = get_global_id(1);\n".
                    $this->kernelTemplateLSum2(
                        "x[parallel_item_id*incx+offset_x] = local_work[0];"
                    ).
                "}\n";
        }
        $kernel2 = $this->createKernel($kernel_name2);

        $kernel->setArg(0,$cols,NDArray::uint32);
        $kernel->setArg(1,$A);
        $kernel->setArg(2,$offsetA,NDArray::uint32);
        $kernel->setArg(3,$ldA,NDArray::uint32);
        $kernel->setArg(4,$temp_buffer);
        $kernel->setArg(5,null,$work_items1*$value_size);
        $global_work_size = [$work_items1*$temp_size,$rows];
        $local_work_size = [$work_items1,1];
        $phase1Events = $this->newEventList();
        $kernel->enqueueNDRange($this->queue,$global_work_size,$local_work_size,null,
                $phase1Events,$waitEvents);

        $kernel2->setArg(0,$temp_buffer);
        $kernel2->setArg(1,$X);
        $kernel2->setArg(2,$offsetX,NDArray::uint32);
        $kernel2->setArg(3,$incX,NDArray::uint32);
        $kernel2->setArg(4,null,$work_items2*$value_size);
        $global_work_size = [$work_items2,$rows];
        $local_work_size = [$work_items2,1];
        $kernel2->enqueueNDRange($this->queue,$global_work_size,$local_work_size,null,
                $events,$phase1Events);
    }

    /**
     * X(m) := max( A(m,n) )
     */
    public function reduceMax(
        bool $trans,
        int $m,
        int $n,
        Buffer $A, int $offsetA, int $ldA,
        Buffer $X, int $offsetX, int $incX,
        EventList $events=null, EventList $waitEvents=null
        ) : void
    {
        $dtype = $A->dtype();
        if($dtype!=$X->dtype()) {
            throw new InvalidArgumentException("Unmatch data type A and X:".
            $this->dtypeToString($dtype).",".$this->dtypeToString($X->dtype()));
        }
        if($dtype!=NDArray::float64 && $dtype!=NDArray::float32) {
            throw new InvalidArgumentException("Unsuppored data type:".
                                            $this->dtypeToString($dtype));
        }
        if($dtype==NDArray::float64) {
            $this->assertFP64();
        }
        if($trans) {
            $cols = $m;
        } else {
            $cols = $n;
        }
        $max_work_items = $this->maxWorkItem[0];
        if($cols <= $max_work_items) {
            $this->reduceMax1(
                $trans,
                $m,
                $n,
                $A,$offsetA,$ldA,
                $X,$offsetX,$incX,
                $events,$waitEvents
            );
        } elseif($cols <= $max_work_items*$max_work_items*2) {
            $this->reduceMax2(
                $trans,
                $m,
                $n,
                $A,$offsetA,$ldA,
                $X,$offsetX,$incX,
                $events,$waitEvents
            );
        } else {
            $this->reduceMax3(
                $trans,
                $m,
                $n,
                $A,$offsetA,$ldA,
                $X,$offsetX,$incX,
                $events,$waitEvents
            );
        }
    }

    /**
    * X(m) := max( A(m,n) )
    */
    public function reduceMax1(
        bool $trans,
        int $m,
        int $n,
        Buffer $A, int $offsetA, int $ldA,
        Buffer $X, int $offsetX, int $incX,
        EventList $events=null, EventList $waitEvents=null
        ) : void
    {
        $dtype = $A->dtype();
        if($trans) {
            $trans = 'trans';
            $rows = $n;
            $cols = $m;
        } else {
            $trans = 'norm';
            $rows = $m;
            $cols = $n;
        }
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
        $kernel_name = "reduceMax_S_${type}_${trans}";
        if(!isset($this->sources[$kernel_name])) {
            if($trans=='trans') {
                $index_a = 'lid*lda+grid+offset_a';
            } else {
                $index_a = 'grid*lda+lid+offset_a';
            }
            $this->sources[$kernel_name] =
                "__kernel void ${kernel_name}(\n".
                "    const        uint total_local_items,\n".
                "        __global ${type} * x,\n".
                "    const        uint offset_x,\n".
                "    const        uint incx,\n".
                "    const global ${type} * a,\n".
                "    const        uint offset_a,\n".
                "    const        uint lda,\n".
                "         __local ${type} * local_work)\n".
                "{\n".
                "    const uint grid = get_group_id(0);\n".
                     $this->kernelTemplateSMax(
                         "local_work[lid] = a[${index_a}];",
                         "x[grid*incx+offset_x] = local_work[0];",
                         $dtype
                     ).
                "}\n";
        }
        $kernel = $this->createKernel($kernel_name);

        $kernel->setArg(0,$total_local_items,NDArray::uint32);
        $kernel->setArg(1,$X);
        $kernel->setArg(2,$offsetX,NDArray::uint32);
        $kernel->setArg(3,$incX,NDArray::uint32);
        $kernel->setArg(4,$A);
        $kernel->setArg(5,$offsetA,NDArray::uint32);
        $kernel->setArg(6,$ldA,NDArray::uint32);
        $kernel->setArg(7,null,$max_work_items*$value_size);
        $global_work_size = [$max_work_items*$rows];
        $local_work_size = [$max_work_items];
        $kernel->enqueueNDRange($this->queue,$global_work_size,$local_work_size,null,
                $events,$waitEvents);
    }

    /**
     * X(m) := max( A(m,n) )
     */
    public function reduceMax2(
        bool $trans,
        int $m,
        int $n,
        Buffer $A, int $offsetA, int $ldA,
        Buffer $X, int $offsetX, int $incX,
        EventList $events=null, EventList $waitEvents=null
        ) : void
    {
        $dtype = $A->dtype();
        if($trans) {
            $trans = 'trans';
            $rows = $n;
            $cols = $m;
        } else {
            $trans = 'norm';
            $rows = $m;
            $cols = $n;
        }
        $total_local_items = $cols;
        $max_work_items = $this->maxWorkItem[0];
        if($total_local_items>$max_work_items) {
            $segments = (int)floor(($total_local_items+$max_work_items-1)/$max_work_items); // round up float
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
        $kernel_name = "reduceMax_M_${type}_${trans}";
        if(!isset($this->sources[$kernel_name])) {
            if($trans=='trans') {
                $index_a = '(seg*lws+lid)*lda+grid+offset_a';
            } else {
                $index_a = 'grid*lda+(seg*lws+lid)+offset_a';
            }
            $this->sources[$kernel_name] =
                "__kernel void ${kernel_name}(\n".
                "    const        uint total_local_items,\n".
                "    const        uint segments,\n".
                "        __global ${type} * x,\n".
                "    const        uint offset_x,\n".
                "    const        uint incx,\n".
                "    const global ${type} * a,\n".
                "    const        uint offset_a,\n".
                "    const        uint lda,\n".
                "         __local ${type} * local_work,\n".
                "         __local ${type} * seg_work,\n".
                "    const        uint work_items)\n".
                "{\n".
                "    const uint grid = get_group_id(0);\n".
                     $this->kernelTemplateQMax(
                         "local_work[lid] = a[${index_a}];\n",
                         "x[grid*incx+offset_x] = seg_work[0];\n",
                         $dtype
                     ).
                "}\n";
        }
        $kernel = $this->createKernel($kernel_name);

        $kernel->setArg(0,$total_local_items,NDArray::uint32);
        $kernel->setArg(1,$segments,NDArray::uint32);
        $kernel->setArg(2,$X);
        $kernel->setArg(3,$offsetX,NDArray::uint32);
        $kernel->setArg(4,$incX,NDArray::uint32);
        $kernel->setArg(5,$A);
        $kernel->setArg(6,$offsetA,NDArray::uint32);
        $kernel->setArg(7,$ldA,NDArray::uint32);

        $kernel->setArg(8,null,$max_work_items*$value_size);
        $kernel->setArg(9,null,$segments*$value_size);
        $kernel->setArg(10,$work_items,NDArray::uint32);
        $global_work_size = [$max_work_items*$rows];
        $local_work_size = [$max_work_items];
        $kernel->enqueueNDRange($this->queue,$global_work_size,$local_work_size,null,
                $events,$waitEvents);
    }

    /**
     * X(m) := max( A(m,n) )
     */
    public function reduceMax3(
        bool $trans,
        int $m,
        int $n,
        Buffer $A, int $offsetA, int $ldA,
        Buffer $X, int $offsetX, int $incX,
        EventList $events=null, EventList $waitEvents=null
        ) : void
    {
        $dtype = $A->dtype();
        if($trans) {
            $trans = 'trans';
            $rows = $n;
            $cols = $m;
        } else {
            $trans = 'norm';
            $rows = $m;
            $cols = $n;
        }
        $total_local_items = $cols;
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
            $value_size*$temp_size*$rows,
            OpenCL::CL_MEM_READ_WRITE,null,null,$dtype);

        $type = $this->dtypeToOpenCLType[$dtype];
        $kernel_name1 = "reduceMax_L1_${type}_${trans}";
        $kernel_name2 = "reduceMax_L2_${type}_${trans}";
        if(!isset($this->sources[$kernel_name1])) {
            if($trans=='trans') {
                $index_a = 'local_item_id*lda + parallel_item_id + offset_a';
            } else {
                $index_a = 'parallel_item_id*lda + local_item_id + offset_a';
            }
            $this->sources[$kernel_name1] =
                "__kernel void ${kernel_name1}(\n".
                "    const        uint total_local_items,\n".
                "    const __global ${type} * a,\n".
                "    const        uint offset_a,\n".
                "    const        uint lda,\n".
                "        __global ${type} * temp_buffer,\n".
                "         __local ${type} * local_work)\n".
                "{\n".
                "    const uint parallel_item_id = get_global_id(1);\n".
                    $this->kernelTemplateLMax1(
                        "${type} input = a[${index_a}];",
                        $dtype
                    ).
                "}\n";
        }
        $kernel = $this->createKernel($kernel_name1);

        if(!isset($this->sources[$kernel_name2])) {
            $this->sources[$kernel_name2] =
                "__kernel void ${kernel_name2}(\n".
                "    const __global ${type} * temp_buffer,\n".
                "        __global ${type} * x,\n".
                "    const        uint offset_x,\n".
                "    const        uint incx,\n".
                "         __local ${type} * local_work)\n".
                "{\n".
                "    const uint parallel_item_id = get_global_id(1);\n".
                    $this->kernelTemplateLMax2(
                        "x[parallel_item_id*incx+offset_x] = local_work[0];"
                    ).
                "}\n";
        }
        $kernel2 = $this->createKernel($kernel_name2);

        $kernel->setArg(0,$total_local_items,NDArray::uint32);
        $kernel->setArg(1,$A);
        $kernel->setArg(2,$offsetA,NDArray::uint32);
        $kernel->setArg(3,$ldA,NDArray::uint32);
        $kernel->setArg(4,$temp_buffer);
        $kernel->setArg(5,null,$work_items1*$value_size);
        $global_work_size = [$work_items1*$temp_size,$rows];
        $local_work_size = [$work_items1,1];
        $phase1Events = $this->newEventList();
        $kernel->enqueueNDRange($this->queue,$global_work_size,$local_work_size,null,
                $phase1Events,$waitEvents);

        $kernel2->setArg(0,$temp_buffer);
        $kernel2->setArg(1,$X);
        $kernel2->setArg(2,$offsetX,NDArray::uint32);
        $kernel2->setArg(3,$incX,NDArray::uint32);
        $kernel2->setArg(4,null,$work_items2*$value_size);
        $global_work_size = [$work_items2,$rows];
        $local_work_size = [$work_items2,1];
        $kernel2->enqueueNDRange($this->queue,$global_work_size,$local_work_size,null,
                $events,$phase1Events);
    }

    /**
    * X(m) := argMax( A(m,n) )
    */
    public function reduceArgMax(
        bool $trans,
        int $m,
        int $n,
        Buffer $A, int $offsetA, int $ldA,
        Buffer $X, int $offsetX, int $incX,
        EventList $events=null, EventList $waitEvents=null
        ) : void
    {
        if($X->dtype()!=NDArray::int32 && $X->dtype()!=NDArray::uint32) {
            throw new InvalidArgumentException("X must be 32bit integer:".
                                            $this->dtypeToString($X->dtype()));
        }
        if($A->dtype()!=NDArray::float64 && $A->dtype()!=NDArray::float32) {
            throw new InvalidArgumentException("Unsuppored data type:".
                                            $this->dtypeToString($A->dtype()));
        }
        if($A->dtype()==NDArray::float64) {
            $this->assertFP64();
        }
        if($trans) {
            $cols = $m;
        } else {
            $cols = $n;
        }
        $max_work_items = $this->maxWorkItem[0];
        if($cols <= $max_work_items) {
            $this->reduceArgMax1(
                $trans,
                $m,
                $n,
                $A,$offsetA,$ldA,
                $X,$offsetX,$incX,
                $events,$waitEvents
            );
        } elseif($cols <= $max_work_items*$max_work_items*2) {
            $this->reduceArgMax2(
                $trans,
                $m,
                $n,
                $A,$offsetA,$ldA,
                $X,$offsetX,$incX,
                $events,$waitEvents
            );
        } else {
            $this->reduceArgMax3(
                $trans,
                $m,
                $n,
                $A,$offsetA,$ldA,
                $X,$offsetX,$incX,
                $events,$waitEvents
            );
        }
    }

    /**
    * X(m) := argmax( A(m,n) )
    */
    public function reduceArgMax1(
        bool $trans,
        int $m,
        int $n,
        Buffer $A, int $offsetA, int $ldA,
        Buffer $X, int $offsetX, int $incX,
        EventList $events=null, EventList $waitEvents=null
        ) : void
    {
        $dtype = $A->dtype();
        if($trans) {
            $trans = 'trans';
            $rows = $n;
            $cols = $m;
        } else {
            $trans = 'norm';
            $rows = $m;
            $cols = $n;
        }
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
        $index_value_size = $X->value_size();
        $type = $this->dtypeToOpenCLType[$dtype];
        $kernel_name = "reduceArgMax_S_${type}_${trans}";
        if(!isset($this->sources[$kernel_name])) {
            if($trans=='trans') {
                $index_a = 'lid*lda+grid+offset_a';
            } else {
                $index_a = 'grid*lda+lid+offset_a';
            }
            $this->sources[$kernel_name] =
                "__kernel void ${kernel_name}(\n".
                "    const        uint total_local_items,\n".
                "        __global uint * x,\n".
                "    const        uint offset_x,\n".
                "    const        uint incx,\n".
                "    const global ${type} * a,\n".
                "    const        uint offset_a,\n".
                "    const        uint lda,\n".
                "         __local ${type} * local_work,\n".
                "         __local uint * local_iwork)\n".
                "{\n".
                "    const uint grid = get_group_id(0);\n".
                     $this->kernelTemplateSiMax(
                         "local_work[lid] = a[${index_a}];\n".
                         "local_iwork[lid] = lid;",
                         "x[grid*incx+offset_x] = local_iwork[0];",
                         $dtype
                         ).
                "}\n";
        }
        $kernel = $this->createKernel($kernel_name);

        $kernel->setArg(0,$total_local_items,NDArray::uint32);
        $kernel->setArg(1,$X);
        $kernel->setArg(2,$offsetX,NDArray::uint32);
        $kernel->setArg(3,$incX,NDArray::uint32);
        $kernel->setArg(4,$A);
        $kernel->setArg(5,$offsetA,NDArray::uint32);
        $kernel->setArg(6,$ldA,NDArray::uint32);
        $kernel->setArg(7,null,$max_work_items*$value_size);
        $kernel->setArg(8,null,$max_work_items*$index_value_size);
        $global_work_size = [$max_work_items*$rows];
        $local_work_size = [$max_work_items];
        $kernel->enqueueNDRange($this->queue,$global_work_size,$local_work_size,null,
                $events,$waitEvents);
    }

    /**
     * X(m) := max( A(m,n) )
     */
    public function reduceArgMax2(
        bool $trans,
        int $m,
        int $n,
        Buffer $A, int $offsetA, int $ldA,
        Buffer $X, int $offsetX, int $incX,
        EventList $events=null, EventList $waitEvents=null
        ) : void
    {
        $dtype = $A->dtype();
        if($trans) {
            $trans = 'trans';
            $rows = $n;
            $cols = $m;
        } else {
            $trans = 'norm';
            $rows = $m;
            $cols = $n;
        }
        $total_local_items = $cols;
        $max_work_items = $this->maxWorkItem[0];
        if($total_local_items>$max_work_items) {
            $segments = (int)floor(($total_local_items+$max_work_items-1)/$max_work_items); // round up float
            $work_items = $max_work_items;
        } else {
            for($max_work_items=1; $max_work_items<$total_local_items;$max_work_items<<=1) {
                ;
            }
            $segments = 1; // round up float
            $work_items = $total_local_items;
        }
        $value_size = $A->value_size();
        $index_value_size = $X->value_size();
        $type = $this->dtypeToOpenCLType[$dtype];
        $kernel_name = "reduceArgMax_M_${type}_${trans}";
        if(!isset($this->sources[$kernel_name])) {
            if($trans=='trans') {
                $index_a = '(seg*lws+lid)*lda+grid+offset_a';
            } else {
                $index_a = 'grid*lda+(seg*lws+lid)+offset_a';
            }
            $this->sources[$kernel_name] =
                "__kernel void ${kernel_name}(\n".
                "    const        uint total_local_items,\n".
                "    const        uint segments,\n".
                "        __global uint * x,\n".
                "    const        uint offset_x,\n".
                "    const        uint incx,\n".
                "    const global ${type} * a,\n".
                "    const        uint offset_a,\n".
                "    const        uint lda,\n".
                "         __local ${type} * local_work,\n".
                "         __local ${type} * seg_work,\n".
                "         __local uint * local_iwork,\n".
                "         __local uint * seg_iwork,\n".
                "    const        uint work_items)\n".
                "{\n".
                "    const uint grid = get_group_id(0);\n".
                     $this->kernelTemplateQiMax(
                         "local_work[lid] = a[${index_a}];\n".
                         "local_iwork[lid] = seg*lws+lid;\n",
                         "x[grid*incx+offset_x] = seg_iwork[0];\n",
                         $dtype
                     ).
                "}\n";
        }
        $kernel = $this->createKernel($kernel_name);

        $kernel->setArg(0,$total_local_items,NDArray::uint32);
        $kernel->setArg(1,$segments,NDArray::uint32);
        $kernel->setArg(2,$X);
        $kernel->setArg(3,$offsetX,NDArray::uint32);
        $kernel->setArg(4,$incX,NDArray::uint32);
        $kernel->setArg(5,$A);
        $kernel->setArg(6,$offsetA,NDArray::uint32);
        $kernel->setArg(7,$ldA,NDArray::uint32);

        $kernel->setArg(8,null,$max_work_items*$value_size);
        $kernel->setArg(9,null,$segments*$value_size);
        $kernel->setArg(10,null,$max_work_items*$index_value_size);
        $kernel->setArg(11,null,$segments*$index_value_size);
        $kernel->setArg(12,$work_items,NDArray::uint32);
        $global_work_size = [$max_work_items*$rows];
        $local_work_size = [$max_work_items];
        $kernel->enqueueNDRange($this->queue,$global_work_size,$local_work_size,null,
                $events,$waitEvents);
    }

    /**
     * X(m) := max( A(m,n) )
     */
    public function reduceArgMax3(
        bool $trans,
        int $m,
        int $n,
        Buffer $A, int $offsetA, int $ldA,
        Buffer $X, int $offsetX, int $incX,
        EventList $events=null, EventList $waitEvents=null
        ) : void
    {
        $dtype = $A->dtype();
        if($trans) {
            $trans = 'trans';
            $rows = $n;
            $cols = $m;
        } else {
            $trans = 'norm';
            $rows = $m;
            $cols = $n;
        }
        $total_local_items = $cols;
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
        $index_value_size = $X->value_size();
        $temp_size = 2*$work_items2;
        $temp_buffer = $this->newBuffer(
            $value_size*$temp_size*$rows,
            OpenCL::CL_MEM_READ_WRITE,null,null,$dtype);
        $temp_ibuffer = $this->newBuffer(
            $index_value_size*$temp_size*$rows,
            OpenCL::CL_MEM_READ_WRITE,null,null,$X->dtype());
        $type = $this->dtypeToOpenCLType[$dtype];
        $kernel_name1 = "reduceArgMax_L1_${type}_${trans}";
        $kernel_name2 = "reduceArgMax_L2_${type}_${trans}";
        if(!isset($this->sources[$kernel_name1])) {
            if($trans=='trans') {
                $index_a = 'local_item_id*lda + parallel_item_id + offset_a';
            } else {
                $index_a = 'parallel_item_id*lda + local_item_id + offset_a';
            }
            $this->sources[$kernel_name1] =
                "__kernel void ${kernel_name1}(\n".
                "    const        uint total_local_items,\n".
                "    const __global ${type} * a,\n".
                "    const        uint offset_a,\n".
                "    const        uint lda,\n".
                "        __global ${type} * temp_buffer,\n".
                "         __local ${type} * local_work,\n".
                "        __global uint * temp_ibuffer,\n".
                "         __local uint * local_iwork)\n".
                "{\n".
                "    const uint parallel_item_id = get_global_id(1);\n".
                    $this->kernelTemplateLiMax1(
                        "${type} input = a[${index_a}];\n".
                        "uint input_index = local_item_id;\n",
                        $dtype
                    ).
                "}\n";
        }
        $kernel = $this->createKernel($kernel_name1);

        if(!isset($this->sources[$kernel_name2])) {
            $this->sources[$kernel_name2] =
                "__kernel void ${kernel_name2}(\n".
                "    const __global ${type} * temp_buffer,\n".
                "    const __global uint * temp_ibuffer,\n".
                "        __global uint * x,\n".
                "    const        uint offset_x,\n".
                "    const        uint incx,\n".
                "         __local ${type} * local_work,\n".
                "         __local uint * local_iwork)\n".
                "{\n".
                "    const uint parallel_item_id = get_global_id(1);\n".
                    $this->kernelTemplateLiMax2(
                        "x[parallel_item_id*incx+offset_x] = local_iwork[0];"
                    ).
                "}\n";
        }
        $kernel2 = $this->createKernel($kernel_name2);

        $kernel->setArg(0,$total_local_items,NDArray::uint32);
        $kernel->setArg(1,$A);
        $kernel->setArg(2,$offsetA,NDArray::uint32);
        $kernel->setArg(3,$ldA,NDArray::uint32);
        $kernel->setArg(4,$temp_buffer);
        $kernel->setArg(5,null,$work_items1*$value_size);
        $kernel->setArg(6,$temp_ibuffer);
        $kernel->setArg(7,null,$work_items1*$index_value_size);
        $global_work_size = [$work_items1*$temp_size,$rows];
        $local_work_size = [$work_items1,1];
        $phase1Events = $this->newEventList();
        $kernel->enqueueNDRange($this->queue,$global_work_size,$local_work_size,null,
                $phase1Events,$waitEvents);

        $kernel2->setArg(0,$temp_buffer);
        $kernel2->setArg(1,$temp_ibuffer);
        $kernel2->setArg(2,$X);
        $kernel2->setArg(3,$offsetX,NDArray::uint32);
        $kernel2->setArg(4,$incX,NDArray::uint32);
        $kernel2->setArg(5,null,$work_items2*$value_size);
        $kernel2->setArg(6,null,$work_items2*$index_value_size);
        $global_work_size = [$work_items2,$rows];
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
        Buffer $A, int $offsetA, int $ldA,
        EventList $events=null, EventList $waitEvents=null
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
        if($cols <= $max_work_items) {
            $this->softmax1(
                $m,
                $n,
                $A,$offsetA,$ldA,
                $events,$waitEvents
            );
        } elseif($cols <= $max_work_items*$max_work_items*2) {
            $this->softmax2(
                $m,
                $n,
                $A,$offsetA,$ldA,
                $events,$waitEvents
            );
        }
    }

    public function softmax1(
        int $m,
        int $n,
        Buffer $A, int $offsetA, int $ldA,
        EventList $events=null, EventList $waitEvents=null
        ) : void
    {
        $trans = false;
        $dtype = $A->dtype();
        if($trans) {
            $trans = 'trans';
            $rows = $n;
            $cols = $m;
        } else {
            $trans = 'norm';
            $rows = $m;
            $cols = $n;
        }
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
        $kernel_name = "softmax_S_${type}_${trans}";
        if(!isset($this->sources[$kernel_name])) {
            if($trans=='trans') {
                $index_a = 'lid*lda+grid+offset_a';
            } else {
                $index_a = 'grid*lda+lid+offset_a';
            }
            $this->sources[$kernel_name] =
                "__kernel void ${kernel_name}(\n".
                "    const        uint total_local_items,\n".
                "        __global ${type} * a,\n".
                "    const        uint offset_a,\n".
                "    const        uint lda,\n".
                "         __local ${type} * local_work)\n".
                "{\n".
                "    const uint grid = get_group_id(0);\n".
                "    __local ${type} max;\n".
                "    __local ${type} sum;\n".
                     $this->kernelTemplateSMax(
                         "local_work[lid] = a[${index_a}];\n",
                         "max = local_work[0];\n",
                         $dtype
                     ).
                     "barrier(CLK_LOCAL_MEM_FENCE);\n".
                     $this->kernelTemplateSSum(
                         "local_work[lid] = exp(a[${index_a}]-max);",
                         "sum = local_work[0];\n"
                     ).
                     "barrier(CLK_LOCAL_MEM_FENCE);\n".
                "    {\n".
                "        const uint lid = get_local_id(0);\n".
                "        const uint lws = get_local_size(0);\n".
                "        if(lid<total_local_items) {\n".
                "            a[${index_a}] = exp(a[${index_a}]-max)/sum;\n".
                "        }\n".
                "    }\n".
                "}\n";
        }
        $kernel = $this->createKernel($kernel_name);

        $kernel->setArg(0,$total_local_items,NDArray::uint32);
        $kernel->setArg(1,$A);
        $kernel->setArg(2,$offsetA,NDArray::uint32);
        $kernel->setArg(3,$ldA,NDArray::uint32);
        $kernel->setArg(4,null,$max_work_items*$value_size);
        $global_work_size = [$max_work_items*$rows];
        $local_work_size = [$max_work_items];
        $kernel->enqueueNDRange($this->queue,$global_work_size,$local_work_size,null,
            $events,$waitEvents);
    }

    public function softmax2(
        int $m,
        int $n,
        Buffer $A, int $offsetA, int $ldA,
        EventList $events=null, EventList $waitEvents=null
        ) : void
    {
        $trans = false;
        $dtype = $A->dtype();
        if($trans) {
            $trans = 'trans';
            $rows = $n;
            $cols = $m;
        } else {
            $trans = 'norm';
            $rows = $m;
            $cols = $n;
        }
        $total_local_items = $cols;
        $max_work_items = $this->maxWorkItem[0];
        if($total_local_items>$max_work_items) {
            $segments = (int)floor(($total_local_items+$max_work_items-1)/$max_work_items); // round up float
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
        $kernel_name = "softmax_M_${type}_${trans}";
        if(!isset($this->sources[$kernel_name])) {
            if($trans=='trans') {
                $index_a = '(seg*lws+lid)*lda+grid+offset_a';
            } else {
                $index_a = 'grid*lda+(seg*lws+lid)+offset_a';
            }
            $this->sources[$kernel_name] =
                "__kernel void ${kernel_name}(\n".
                "    const        uint total_local_items,\n".
                "    const        uint segments,\n".
                "        __global ${type} * a,\n".
                "    const        uint offset_a,\n".
                "    const        uint lda,\n".
                "         __local ${type} * local_work,\n".
                "         __local ${type} * seg_work,\n".
                "    const        uint work_items)\n".
                "{\n".
                "    const uint grid = get_group_id(0);\n".
                "    __local ${type} max;\n".
                "    __local ${type} sum;\n".
                     $this->kernelTemplateQMax(
                         "local_work[lid] = a[${index_a}];\n",
                         "max = seg_work[0];\n",
                         $dtype
                     ).
                     "barrier(CLK_LOCAL_MEM_FENCE);\n".
                     $this->kernelTemplateQSum(
                         "local_work[lid] = exp(a[${index_a}]-max);",
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
                "                a[${index_a}] = exp(a[${index_a}]-max)/sum;\n".
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

        $kernel->setArg(5,null,$max_work_items*$value_size);
        $kernel->setArg(6,null,$segments*$value_size);
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
        Buffer $A, int $offsetA, int $incA,
        Buffer $Y, int $offsetY, int $incY,
        int $startAxis0,
        int $sizeAxis0,
        int $startAxis1,
        int $sizeAxis1,
        int $startAxis2,
        int $sizeAxis2,
        EventList $events=null, EventList $waitEvents=null
        )
    {
        if($A->dtype()!=$Y->dtype()) {
            throw new InvalidArgumentException("Unmatch data type A and Y:".
            $this->dtypeToString($A->dtype()).",".$this->dtypeToString($Y->dtype()));
        }
        if($A->dtype()==NDArray::float64) {
            $this->assertFP64();
        }

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
            $direction = 'r';
        } else {
            $from = $a_variable;
            $to = $y_variable;
            $a_arg_type = 'const global';
            $y_arg_type = '__global';
            $direction = 'f';
        }
        if($addMode) {
            $op = 'add';
            $operator = '+=';
        } else {
            $op = 'set';
            $operator = '=';
        }
        $type = $this->dtypeToOpenCLType[$A->dtype()];
        if(!isset($this->sources["slice_${type}_${direction}_${op}"])) {
            $this->sources["slice_${type}_${direction}_${op}"] =
                "__kernel void slice_${type}_${direction}_${op}(\n".
                "    const        uint n,\n".
                "    const        uint k,\n".
                "    $a_arg_type ${type} * a,\n".
                "    const        uint offset_a,\n".
                "    const        uint inca,\n".
                "    $y_arg_type ${type} * y,\n".
                "    const        uint offset_y,\n".
                "    const        uint incy,\n".
                "    const        uint startAxis0,\n".
                "    const        uint startAxis1,\n".
                "    const        uint startAxis2)\n".
                "{\n".
                "    uint i0 = get_global_id(0);\n".
                "    uint i1 = get_global_id(1);\n".
                "    uint i2 = get_group_id(2);\n".
                "    uint lid = get_local_id(2);\n".
                "    uint i1size = get_local_size(1);\n".
                "    uint i2size = get_num_groups(2);\n".
                "    uint size = get_local_size(2);\n".
                "    ${to} ${operator} ${from};\n".
                "}\n";
        }
//echo "dtype=".$this->dtypeToString($A->dtype())."\n";
//echo "m,n,k,size=$m,$n,$k,$size\n";
//echo "startAxis0,startAxis1,startAxis2=$startAxis0,$startAxis1,$startAxis2\n";
//echo "sizeAxis0,sizeAxis1,sizeAxis2=$sizeAxis0,$sizeAxis1,$sizeAxis2\n";

        $kernel = $this->createKernel("slice_${type}_${direction}_${op}");
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
        $global_work_size = [$sizeAxis0,$sizeAxis1,$sizeAxis2*$size];
        $local_work_size=null;
        #$local_work_size = [1,1,$size];
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
        Buffer $images,
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
        Buffer $cols,
        int $cols_offset,
        int $cols_size,
        EventList $events=null, EventList $waitEvents=null
        )
    {
        $dtype = $images->dtype();
        if($dtype!=$cols->dtype()) {
            throw new InvalidArgumentException("Unmatch data type images and cols:".
            $this->dtypeToString($dtype).",".$this->dtypeToString($cols->dtype()));
        }
        if($dtype==NDArray::float64) {
            $this->assertFP64();
        }

        $output_w = intval(floor(($im_w-($kernel_w-1)*$dilation_w-1)/$stride_w)+1);
        if($padding) {
            $pad_w = (int)floor((($im_w-1)*$stride_w-$im_w+($kernel_w-1)*$dilation_w+1)/2);
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
            $total_local_items = $kernel_w;
            $max_work_items = $this->maxWorkItem[0];
            if($total_local_items <= $max_work_items) {
                $tmode = 'S';
                for($max_work_items=1; $max_work_items<$total_local_items;$max_work_items<<=1) {
                    ;
                }
            } else {
                $tmode = 'M';
                $segments = (int)floor(($total_local_items+$max_work_items-1)/$max_work_items); // round up float
                $work_items = $max_work_items;
            }
            $value_size = $images->value_size();
        } else {
            $tmode = 'F';
        }
        $type = $this->dtypeToOpenCLType[$dtype];
        $kernel_name = "im2col1d_${tmode}_${type}_${channel_mode}_${cols_mode}";
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
                if($tmode=='S'){
                    $kernelTemplateSum = 'kernelTemplateSSum';
                    $kernel_idx = 'lid';
                    $sum_output = 'local_work';
                } else {
                    $kernelTemplateSum = 'kernelTemplateQSum';
                    $kernel_idx = 'seg*lws+lid';
                    $sum_output = 'seg_work';
                }
                $this->sources[$kernel_name] =
                    "__kernel void ${kernel_name}(\n".
                    "    $im_arg_type ${type} * images,\n".
                    "    const        uint offset_images,\n".
                    "    $col_arg_type ${type} * cols,\n".
                    "    const        uint offset_cols,\n".
                    "    const        uint batches,\n".
                    "    const        uint im_w,\n".
                    "    const        uint channels,\n".
                    "    const        uint kernel_w,\n".
                    "    const        uint stride_w,\n".
                    "    const        uint pad_w,\n".
                    "    const        uint dilation_w,\n".
                    "    const        uint output_w,\n".
                    "    const        uint total_local_items,\n".
                    (($tmode=='S') ? (
                    "         __local ${type} * local_work)\n"
                    ) : (
                    "    const        uint segments,\n".
                    "         __local ${type} * local_work,\n".
                    "         __local ${type} * seg_work,\n".
                    "    const        uint work_items)\n"
                    )).
                    "{\n".
                    "    const uint grid = get_group_id(0);\n".  // A row id
                    "    const int input_x = grid;\n".
                    "    const uint batch_id   = get_global_id(1)/channels;\n".
                    "    const uint channel_id = get_global_id(1)%channels;\n".
                         $this->$kernelTemplateSum(
                             "const uint kernel_x = ${kernel_idx};\n".
                             "const int tmp_x = input_x-kernel_x*dilation_w+pad_w;\n".
                             "const int im_x = tmp_x/stride_w;\n".
                             "if(tmp_x%stride_w==0 &&
                                im_x>=0 && im_x<output_w) {\n".
                             "    local_work[lid] = cols[${cols_id}];\n".
                             "} else {\n".
                             "    local_work[lid] = 0;\n".
                             "}\n",
                             "images[${input_id}] += ${sum_output}[0];\n"
                         ).
                    "}\n";
            } else {
                $this->sources[$kernel_name] =
                    "__kernel void ${kernel_name}(\n".
                    "    $im_arg_type ${type} * images,\n".
                    "    const        uint offset_images,\n".
                    "    $col_arg_type ${type} * cols,\n".
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
                    "    uint batch_id = get_global_id(0)/channels;\n".
                    "    uint im_x = get_global_id(1);\n".
                    "    uint channel_id = get_global_id(0)%channels;\n".
                    "    uint kernel_x = get_global_id(2);\n".
                    "    int input_x = ${input_x};\n".
                    "    ${type} value;\n".
                    "    if(input_x>=0 && input_x<im_w) {\n".
                    "        uint input_id = ${input_id};\n".
                    "        value = images[input_id];\n".
                    "    } else {\n".
                    "        value = 0;\n".
                    "    }\n".
                    "    uint cols_id = ${cols_id};\n".
                    "    cols[cols_id] = value;\n".
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
            if($tmode=='S') {
                $kernel->setArg(12,$total_local_items,NDArray::uint32);
                $kernel->setArg(13,null,$max_work_items*$value_size);
                $global_work_size = [$max_work_items*$im_w,$batches*$channels];
                $local_work_size = [$max_work_items,1];
            } else {
                $kernel->setArg(12,$total_local_items,NDArray::uint32);
                $kernel->setArg(13,$segments,NDArray::uint32);
                $kernel->setArg(14,null,$max_work_items*$value_size);
                $kernel->setArg(15,null,$segments*$value_size);
                $kernel->setArg(16,$work_items,NDArray::uint32);
                $global_work_size = [$max_work_items*$im_w,$batches*$channels];
                $local_work_size = [$max_work_items,1];
            }
        } else {
            $global_work_size = [$batches*$channels,$output_w,$kernel_w];
            $local_work_size=null;
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
        Buffer $images,
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
        Buffer $cols,
        int $cols_offset,
        int $cols_size,
        EventList $events=null, EventList $waitEvents=null
        )
    {
        $dtype = $images->dtype();
        if($dtype!=$cols->dtype()) {
            throw new InvalidArgumentException("Unmatch data type images and cols:".
            $this->dtypeToString($dtype).",".$this->dtypeToString($cols->dtype()));
        }
        if($dtype==NDArray::float64) {
            $this->assertFP64();
        }

        $output_h = intval(floor(($im_h-($kernel_h-1)*$dilation_h-1)/$stride_h)+1);
        $output_w = intval(floor(($im_w-($kernel_w-1)*$dilation_w-1)/$stride_w)+1);
        if($padding) {
            $pad_h = (int)floor((($im_h-1)*$stride_h-$im_h+($kernel_h-1)*$dilation_h+1)/2);
            $pad_w = (int)floor((($im_w-1)*$stride_w-$im_w+($kernel_w-1)*$dilation_w+1)/2);
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
            $value_size = $images->value_size();
            $total_local_items = $kernel_h*$kernel_w;
            $max_work_items = $this->maxWorkItem[0];
            if($total_local_items <= $max_work_items) {
                $tmode = 'S';
                for($max_work_items=1; $max_work_items<$total_local_items;$max_work_items<<=1) {
                    ;
                }
            } else {
                $tmode = 'M';
                $segments = (int)floor(($total_local_items+$max_work_items-1)/$max_work_items); // round up float
                $work_items = $max_work_items;
            }
        } else {
            $tmode = 'F';
        }
        $type = $this->dtypeToOpenCLType[$dtype];
        $kernel_name = "im2col2d_${tmode}_${type}_${channel_mode}_${cols_mode}";
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
                if($tmode=='S') {
                    $kernelTemplateSum = 'kernelTemplateSSum';
                    $kernel_idx = 'lid';
                    $sum_output = 'local_work';
                } else {
                    $kernelTemplateSum = 'kernelTemplateQSum';
                    $kernel_idx = 'seg*lws+lid';
                    $sum_output = 'seg_work';
                }
                $this->sources[$kernel_name] =
                    "__kernel void ${kernel_name}(\n".
                    "    $im_arg_type ${type} * images,\n".
                    "    const        uint offset_images,\n".
                    "    $col_arg_type ${type} * cols,\n".
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
                    "    const        uint output_w,\n".
                    "    const        uint total_local_items,\n".
                    (($tmode=='S') ? (
                    "         __local ${type} * local_work)\n"
                    ) : (
                    "    const        uint segments,\n".
                    "         __local ${type} * local_work,\n".
                    "         __local ${type} * seg_work,\n".
                    "    const        uint work_items)\n"
                    )).
                    "{\n".
                    "    const uint grid = get_group_id(0);\n".
                    "    const int input_y = grid/im_w;\n".
                    "    const int input_x = grid%im_w;\n".
                    "    const uint batch_id   = get_global_id(1)/channels;\n".
                    "    const uint channel_id = get_global_id(1)%channels;\n".
                         $this->$kernelTemplateSum(
                             "const uint kernel_idx = ${kernel_idx};\n".
                             "const uint kernel_y = kernel_idx/kernel_w;\n".
                             "const uint kernel_x = kernel_idx%kernel_w;\n".
                             "const int tmp_y = input_y-kernel_y*dilation_h+pad_h;\n".
                             "const int tmp_x = input_x-kernel_x*dilation_w+pad_w;\n".
                             "const int im_y = tmp_y/stride_h;\n".
                             "const int im_x = tmp_x/stride_w;\n".
                             "if(tmp_y%stride_h==0 && tmp_x%stride_w==0 &&
                                im_y>=0 && im_y<output_h && im_x>=0 && im_x<output_w) {\n".
                             "    local_work[lid] = cols[${cols_id}];\n".
                             "} else {\n".
                             "    local_work[lid] = 0;\n".
                             "}\n",
                             "images[${input_id}] += ${sum_output}[0];\n"
                         ).
                    "}\n";
            } else {
                $im_arg_type = 'const global';
                $col_arg_type = '__global';
                $this->sources[$kernel_name] =
                    "__kernel void ${kernel_name}(\n".
                    "    $im_arg_type ${type} * images,\n".
                    "    const        uint offset_images,\n".
                    "    $col_arg_type ${type} * cols,\n".
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
                    "    const        uint output_w)\n".
                    "{\n".
                    "    uint batch_id = get_global_id(0)/channels;\n".
                    "    uint im_y = get_global_id(1)/output_w;\n".
                    "    uint im_x = get_global_id(1)%output_w;\n".
                    "    uint channel_id = get_global_id(0)%channels;\n".
                    "    uint kernel_y = get_global_id(2)/kernel_w;\n".
                    "    uint kernel_x = get_global_id(2)%kernel_w;\n".
                    "    int input_y = ${input_y};\n".
                    "    int input_x = ${input_x};\n".
                    "    ${type} value;\n".
                    "    if(input_y>=0 && input_y<im_h && input_x>=0 && input_x<im_w) {\n".
                    "        uint input_id = ${input_id};\n".
                    "        value = images[input_id];\n".
                    "    } else {\n".
                    "        value = 0;\n".
                    "    }\n".
                    "    uint cols_id = ${cols_id};\n".
                    "    cols[cols_id] = value;\n".
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
            if($tmode=='S') {
                $kernel->setArg(18,$total_local_items,NDArray::uint32);
                $kernel->setArg(19,null,$max_work_items*$value_size);
                $global_work_size = [$max_work_items*$im_h*$im_w,$batches*$channels];
                $local_work_size = [$max_work_items,1];
            } else {
                $kernel->setArg(18,$total_local_items,NDArray::uint32);
                $kernel->setArg(19,$segments,NDArray::uint32);
                $kernel->setArg(20,null,$max_work_items*$value_size);
                $kernel->setArg(21,null,$segments*$value_size);
                $kernel->setArg(22,$work_items,NDArray::uint32);
                $global_work_size = [$max_work_items*$im_h*$im_w,$batches*$channels];
                $local_work_size = [$max_work_items,1];
            }
        } else {
            $global_work_size = [$batches*$channels,$output_h*$output_w,$kernel_h*$kernel_w];
            $local_work_size=null;
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
        Buffer $images,
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
        Buffer $cols,
        int $cols_offset,
        int $cols_size,
        EventList $events=null, EventList $waitEvents=null
        )
    {
        $dtype = $images->dtype();
        if($dtype!=$cols->dtype()) {
            throw new InvalidArgumentException("Unmatch data type images and cols:".
            $this->dtypeToString($dtype).",".$this->dtypeToString($cols->dtype()));
        }
        if($dtype==NDArray::float64) {
            $this->assertFP64();
        }

        $output_d = intval(floor(($im_d-($kernel_d-1)*$dilation_d-1)/$stride_d)+1);
        $output_h = intval(floor(($im_h-($kernel_h-1)*$dilation_h-1)/$stride_h)+1);
        $output_w = intval(floor(($im_w-($kernel_w-1)*$dilation_w-1)/$stride_w)+1);
        if($padding) {
            $pad_d = (int)floor((($im_d-1)*$stride_d-$im_d+($kernel_d-1)*$dilation_d+1)/2);
            $pad_h = (int)floor((($im_h-1)*$stride_h-$im_h+($kernel_h-1)*$dilation_h+1)/2);
            $pad_w = (int)floor((($im_w-1)*$stride_w-$im_w+($kernel_w-1)*$dilation_w+1)/2);
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
            $total_local_items = $kernel_d*$kernel_h*$kernel_w;
            $max_work_items = $this->maxWorkItem[0];
            if($total_local_items <= $max_work_items) {
                $tmode = 'S';
                for($max_work_items=1; $max_work_items<$total_local_items;$max_work_items<<=1) {
                    ;
                }
            } else {
                $tmode = 'M';
                $segments = (int)floor(($total_local_items+$max_work_items-1)/$max_work_items); // round up float
                $work_items = $max_work_items;
            }
            $value_size = $images->value_size();
        } else {
            $tmode = 'G';
        }
        $type = $this->dtypeToOpenCLType[$dtype];
        $kernel_name = "im2col3d_${tmode}_${type}_${channel_mode}_${cols_mode}";
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
                if($tmode=='S') {
                    $kernelTemplateSum = 'kernelTemplateSSum';
                    $kernel_idx = 'lid';
                    $sum_output = 'local_work';
                } else {
                    $kernelTemplateSum = 'kernelTemplateQSum';
                    $kernel_idx = 'seg*lws+lid';
                    $sum_output = 'seg_work';
                }
                $this->sources[$kernel_name] =
                    "__kernel void ${kernel_name}(\n".
                    "    $im_arg_type ${type} * images,\n".
                    "    const        uint offset_images,\n".
                    "    $col_arg_type ${type} * cols,\n".
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
                    "    const        uint output_w,\n".
                    "    const        uint total_local_items,\n".
                    (($tmode=='S') ? (
                    "         __local ${type} * local_work)\n"
                    ) : (
                    "    const        uint segments,\n".
                    "         __local ${type} * local_work,\n".
                    "         __local ${type} * seg_work,\n".
                    "    const        uint work_items)\n"
                    )).
                    "{\n".
                    "    const uint grid = get_group_id(0);\n".
                    "    const uint input_z = grid/im_h/im_w;\n".
                    "    const uint input_y = grid/im_w-input_z*im_h;\n".
                    "    const uint input_x = grid%im_w;\n".
                    "    const uint batch_id   = get_global_id(1)/channels;\n".
                    "    const uint channel_id = get_global_id(1)%channels;\n".
                         $this->$kernelTemplateSum(
                             "const uint kernel_idx = ${kernel_idx};\n".
                             "const uint kernel_z = kernel_idx/kernel_h/kernel_w;\n".
                             "const uint kernel_y = kernel_idx/kernel_w-kernel_z*kernel_h;\n".
                             "const uint kernel_x = kernel_idx%kernel_w;\n".
                             "const int tmp_z = input_z-kernel_z*dilation_d+pad_d;\n".
                             "const int tmp_y = input_y-kernel_y*dilation_h+pad_h;\n".
                             "const int tmp_x = input_x-kernel_x*dilation_w+pad_w;\n".
                             "const int im_z = tmp_z/stride_d;\n".
                             "const int im_y = tmp_y/stride_h;\n".
                             "const int im_x = tmp_x/stride_w;\n".
                             "if(tmp_z%stride_d==0 && tmp_y%stride_h==0 && tmp_x%stride_w==0 &&\n".
                             "    im_z>=0 && im_z<output_d &&\n".
                             "    im_y>=0 && im_y<output_h &&\n".
                             "    im_x>=0 && im_x<output_w) {\n".
                             "    local_work[lid] = cols[${cols_id}];\n".
                             "} else {\n".
                             "    local_work[lid] = 0;\n".
                             "}\n",
                             "images[${input_id}] += ${sum_output}[0];\n"
                         ).
                    "}\n";
            } else {
                $this->sources[$kernel_name] =
                    "__kernel void ${kernel_name}(\n".
                    "    $im_arg_type ${type} * images,\n".
                    "    const        uint offset_images,\n".
                    "    $col_arg_type ${type} * cols,\n".
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
                    "    uint batch_id = get_global_id(0)/channels;\n".
                    "    uint im_z = get_global_id(1)/output_h/output_w;\n".
                    "    uint im_y = get_global_id(1)/output_w-im_z*output_h;\n".
                    "    uint im_x = get_global_id(1)%output_w;\n".
                    "    uint channel_id = get_global_id(0)%channels;\n".
                    "    uint kernel_z = get_global_id(2)/kernel_h/kernel_w;\n".
                    "    uint kernel_y = get_global_id(2)/kernel_w-kernel_z*kernel_h;\n".
                    "    uint kernel_x = get_global_id(2)%kernel_w;\n".
                    "    int input_z = ${input_z};\n".
                    "    int input_y = ${input_y};\n".
                    "    int input_x = ${input_x};\n".
                    "    ${type} value;\n".
                    "    if(input_z>=0 && input_z<im_d && input_y>=0 && input_y<im_h && input_x>=0 && input_x<im_w) {\n".
                    "        uint input_id = ${input_id};\n".
                    "        value = images[input_id];\n".
                    "    } else {\n".
                    "        value = 0;\n".
                    "    }\n".
                    "    uint cols_id = ${cols_id};\n".
                    "    cols[cols_id] = value;\n".
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
            if($tmode=='S') {
                $kernel->setArg(24,$total_local_items,NDArray::uint32);
                $kernel->setArg(25,null,$max_work_items*$value_size);
                $global_work_size = [$max_work_items*$im_d*$im_h*$im_w,$batches*$channels];
                $local_work_size = [$max_work_items,1];
            } else {
                $kernel->setArg(24,$total_local_items,NDArray::uint32);
                $kernel->setArg(25,$segments,NDArray::uint32);
                $kernel->setArg(26,null,$max_work_items*$value_size);
                $kernel->setArg(27,null,$segments*$value_size);
                $kernel->setArg(28,$work_items,NDArray::uint32);
                $global_work_size = [$max_work_items*$im_d*$im_h*$im_w,$batches*$channels];
                $local_work_size = [$max_work_items,1];
            }
        } else {
            $global_work_size = [$batches*$channels,$output_d*$output_h*$output_w,$kernel_d*$kernel_h*$kernel_w];
            $local_work_size=null;
        }
        $kernel->enqueueNDRange($this->queue,$global_work_size,$local_work_size,null,
            $events,$waitEvents);
    }
/*
    public function im2col3d(
        bool $reverse,
        Buffer $images,
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
        Buffer $cols,
        int $cols_offset,
        int $cols_size,
        EventList $events=null, EventList $waitEvents=null
        )
    {
        if($images->dtype()!=$cols->dtype()) {
            throw new InvalidArgumentException("Unmatch data type images and cols:".
            $this->dtypeToString($images->dtype()).",".$this->dtypeToString($cols->dtype()));
        }
        if($images->dtype()==NDArray::float64) {
            $this->assertFP64();
        }

        $output_d = intval(floor(($im_d-($kernel_d-1)*$dilation_d-1)/$stride_d)+1);
        $output_h = intval(floor(($im_h-($kernel_h-1)*$dilation_h-1)/$stride_h)+1);
        $output_w = intval(floor(($im_w-($kernel_w-1)*$dilation_w-1)/$stride_w)+1);
        if($padding) {
            $pad_d = (int)floor((($im_d-1)*$stride_d-$im_d+($kernel_d-1)*$dilation_d+1)/2);
            $pad_h = (int)floor((($im_h-1)*$stride_h-$im_h+($kernel_h-1)*$dilation_h+1)/2);
            $pad_w = (int)floor((($im_w-1)*$stride_w-$im_w+($kernel_w-1)*$dilation_w+1)/2);
            $output_d = $im_d;
            $output_h = $im_h;
            $output_w = $im_w;
        } else {
            $pad_d = $pad_h = $pad_w = 0;
        }

        if($reverse) {
            $col_arg_type = 'const global';
            $im_arg_type = '__global';
            $direction = 'r';
        } else {
            $im_arg_type = 'const global';
            $col_arg_type = '__global';
            $direction = 'f';
        }
        if($channels_first) {
            $input_id = '((((batch_id*channels+channel_id)*im_d+input_z)*im_h+input_y)*im_w+input_x)';
            $channel_mode = 'cf';
        } else {
            $input_id = '((((batch_id*im_d+input_z)*im_h+input_y)*im_w+input_x)*channels+channel_id)';
            $channel_mode = 'cl';
        }
        if($cols_channels_first) {
            $cols_id = '(((((((batch_id*output_d+im_z)*output_h+im_y)*output_w+im_x)'.
                        '*channels+channel_id)*kernel_d+kernel_z)*kernel_h+kernel_y)*kernel_w+kernel_x)';
            $cols_mode = 'cf';
        } else {
            $cols_id = '(((((((batch_id*output_d+im_z)*output_h+im_y)*output_w+im_x)'.
                        '*kernel_d+kernel_z)*kernel_h+kernel_y)*kernel_w+kernel_x)*channels+channel_id)';
            $cols_mode = 'cl';
        }
        $input_z = '(im_z*stride_d+kernel_z*dilation_d-pad_d)';
        $input_y = '(im_y*stride_h+kernel_y*dilation_h-pad_h)';
        $input_x = '(im_x*stride_w+kernel_x*dilation_w-pad_w)';
        $total_local_items = $kernel_d*$kernel_h*$kernel_w;
        $max_work_items = $this->maxWorkItem[0];
        if($total_local_items>$max_work_items) {
            $segments = (int)floor(($total_local_items+$max_work_items-1)/$max_work_items); // round up float
            $work_items = $max_work_items;
        } else {
            $segments = 1; // round up float
            $work_items = $total_local_items;
        }
        $type = $this->dtypeToOpenCLType[$images->dtype()];
        if($reverse) {
            if(!isset($this->sources["im2col3d_${type}_${direction}_${channel_mode}_${cols_mode}"])) {
                $this->sources["im2col3d_${type}_${direction}_${channel_mode}_${cols_mode}"] =
                    "__kernel void im2col3d_${type}_${direction}_${channel_mode}_${cols_mode}(\n".
                    "    $im_arg_type ${type} * images,\n".
                    "    const        uint offset_images,\n".
                    "    $col_arg_type ${type} * cols,\n".
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
                    "    const        uint output_w,\n".
                    "    const        uint total_local_items,\n".
                    "    const        uint segments,\n".
                    "         __local ${type} * local_work,\n".
                    "         __local ${type} * seg_work)\n".
                    "{\n".
                    "    const uint grid = get_group_id(0);\n".
                    "    const uint input_z = grid/im_h/im_w;\n".
                    "    const uint input_y = grid/im_w-input_z*im_h;\n".
                    "    const uint input_x = grid%im_w;\n".
                    "    const uint batch_id   = get_global_id(1)/channels;\n".
                    "    const uint channel_id = get_global_id(1)%channels;\n".
                         $this->kernelTemplateSum(
                             "const uint kernel_idx = seg*lws+lid;\n".
                             "const uint kernel_z = kernel_idx/kernel_h/kernel_w;\n".
                             "const uint kernel_y = kernel_idx/kernel_w-kernel_z*kernel_h;\n".
                             "const uint kernel_x = kernel_idx%kernel_w;\n".
                             "const int tmp_z = input_z-kernel_z*dilation_d+pad_d;\n".
                             "const int tmp_y = input_y-kernel_y*dilation_h+pad_h;\n".
                             "const int tmp_x = input_x-kernel_x*dilation_w+pad_w;\n".
                             "const int im_z = tmp_z/stride_d;\n".
                             "const int im_y = tmp_y/stride_h;\n".
                             "const int im_x = tmp_x/stride_w;\n".
                             "if(tmp_z%stride_d==0 && tmp_y%stride_h==0 && tmp_x%stride_w==0 &&\n".
                             "    im_z>=0 && im_z<output_d &&\n".
                             "    im_y>=0 && im_y<output_h &&\n".
                             "    im_x>=0 && im_x<output_w) {\n".
                             "    local_work[lid] = cols[${cols_id}];\n".
                             "} else {\n".
                             "    local_work[lid] = 0;\n".
                             "}\n",
                             "images[${input_id}] += seg_work[0];\n"
                             #"images[${input_id}] += 1;\n"
                         ).
                    "}\n";
            }
        } else {
            if(!isset($this->sources["im2col3d_${type}_${direction}_${channel_mode}_${cols_mode}"])) {
                $this->sources["im2col3d_${type}_${direction}_${channel_mode}_${cols_mode}"] =
                    "__kernel void im2col3d_${type}_${direction}_${channel_mode}_${cols_mode}(\n".
                    "    $im_arg_type ${type} * images,\n".
                    "    const        uint offset_images,\n".
                    "    $col_arg_type ${type} * cols,\n".
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
                    "    uint batch_id = get_global_id(0)/channels;\n".
                    "    uint im_z = get_global_id(1)/output_h/output_w;\n".
                    "    uint im_y = get_global_id(1)/output_w-im_z*output_h;\n".
                    "    uint im_x = get_global_id(1)%output_w;\n".
                    "    uint channel_id = get_global_id(0)%channels;\n".
                    "    uint kernel_z = get_global_id(2)/kernel_h/kernel_w;\n".
                    "    uint kernel_y = get_global_id(2)/kernel_w-kernel_z*kernel_h;\n".
                    "    uint kernel_x = get_global_id(2)%kernel_w;\n".
                    "    int input_z = ${input_z};\n".
                    "    int input_y = ${input_y};\n".
                    "    int input_x = ${input_x};\n".
                    "    ${type} value;\n".
                    "    if(input_z>=0 && input_z<im_d && input_y>=0 && input_y<im_h && input_x>=0 && input_x<im_w) {\n".
                    "        uint input_id = ${input_id};\n".
                    "        value = images[input_id];\n".
                    "    } else {\n".
                    "        value = 0;\n".
                    "    }\n".
                    "    uint cols_id = ${cols_id};\n".
                    "    cols[cols_id] = value;\n".
                    "}\n";
            }
        }

#echo "gids=".($batches*$output_h*$output_w*$channels*$kernel_h*$kernel_w)."\n";
#echo "colssize=".($cols->bytes()/32*8)."\n";

        $kernel = $this->createKernel("im2col3d_${type}_${direction}_${channel_mode}_${cols_mode}");
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
            $kernel->setArg(24,$total_local_items,NDArray::uint32);
            $kernel->setArg(25,$segments,NDArray::uint32);
            $kernel->setArg(26,null,intval($work_items*$cols->value_size()));
            $kernel->setArg(27,null,intval($segments*$cols->value_size()));
            $global_work_size = [$work_items*$im_d*$im_h*$im_w,$batches*$channels];
            $local_work_size = [$work_items,1];
        } else {
            $global_work_size = [$batches*$channels,$output_d*$output_h*$output_w,$kernel_d*$kernel_h*$kernel_w];
            $local_work_size=null;
        }
        $kernel->enqueueNDRange($this->queue,$global_work_size,$local_work_size,null,
            $events,$waitEvents);
    }
*/
}
