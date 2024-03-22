<?php
namespace Rindow\Math\Matrix\Drivers;

use Interop\Polite\Math\Matrix\OpenCL;
use Rindow\Math\Matrix\Drivers\MatlibPHP\PhpBLASFactory;
use Rindow\Math\Matrix\Drivers\MatlibCL\OpenCLMath;
use Rindow\Math\Matrix\Drivers\MatlibCL\OpenCLMathFixDiv5Bug;

use InvalidArgumentException;

abstract class AbstractMatlibService implements Service
{
    protected array $levelString = [
        Service::LV_BASIC => 'Basic',
        Service::LV_ADVANCED => 'Advanced',
        Service::LV_ACCELERATED => 'Accelerated',
    ];

    protected object $phpBLASFactory;
    protected object $phpblas;
    protected object $phplapack;
    protected object $phpmath;
    protected object $phpbuffer;

    protected object $blas;
    protected object $lapack;
    protected object $math;
    protected object $buffer;
    protected ?object $openclMath=null;
    protected ?object $clblastBlas=null;
    protected ?object $clblastMath=null;

    protected ?int $serviceLevel=null;

    public function __construct(
        protected ?object $bufferFactory=null,
        protected ?object $openblasFactory=null,
        protected ?object $mathFactory=null,
        protected ?object $openclFactory=null,
        protected ?object $clblastFactory=null,
        protected ?object $blasCLFactory=null,
        protected ?object $mathCLFactory=null,
        protected ?object $bufferCLFactory=null,
    )
    {
        $this->phpBLASFactory = new PhpBLASFactory($this);
        $this->phpblas = $this->phpBLASFactory->Blas();
        $this->phplapack = $this->phpBLASFactory->Lapack();
        $this->phpmath = $this->phpBLASFactory->Math();
        $this->phpbuffer = $this->phpBLASFactory;

        $level = $this->serviceLevel();
        if($level>=Service::LV_ADVANCED) {
            $this->blas = $this->openblasFactory->Blas();
            $this->lapack = $this->openblasFactory->Lapack();
            $this->math = $this->mathFactory->Math();
            $this->buffer = $this->bufferFactory;
        } else {
            $this->blas = $this->phpblas;
            $this->lapack = $this->phplapack;
            $this->math = $this->phpmath;
            $this->buffer = $this->phpbuffer;
        }
    }

    public function serviceLevel() : int
    {
        if($this->serviceLevel!==null) {
            $this->serviceLevel;
        }

        $level = Service::LV_BASIC;
        while(true) {
            if($this->bufferFactory===null ||
                $this->openblasFactory===null ||
                $this->mathFactory===null) {
                break;
            }
            if(!$this->bufferFactory->isAvailable()||
                !$this->openblasFactory->isAvailable()||
                !$this->mathFactory->isAvailable()) {
                break;
            }
            $level = Service::LV_ADVANCED;

            if($this->openclFactory==null||
                $this->clblastFactory==null) {
                break;
            }
            if(!$this->openclFactory->isAvailable()||
                !$this->clblastFactory->isAvailable()) {
                break;
            }
            $level = Service::LV_ACCELERATED;
            break;
        }

        $this->serviceLevel = $level;
        return $level;
    }

    public function info() : string
    {
        $info =  "Service Level   : ".$this->levelString[$this->serviceLevel]."\n";
        $info .= "Buffer Factory  : ".get_class($this->buffer)."\n";
        $info .= "BLAS Driver     : ".get_class($this->blas)."\n";
        $info .= "LAPACK Driver   : ".get_class($this->lapack)."\n";
        $info .= "Math Driver     : ".get_class($this->math)."\n";
        if($this->serviceLevel()>=Service::LV_ACCELERATED) {
            $info .= "OpenCL Factory  : ".get_class($this->openclFactory)."\n";
            $info .= "CLBlast Factory : ".get_class($this->clblastFactory)."\n";
        }
        return $info;
    }

    public function name() : string
    {
        return $this->name;
    }

    public function blas(int $level=null) : object
    {
        $level = $level ?? Service::LV_ADVANCED;
        switch($level) {
            case Service::LV_BASIC: {
                return $this->phpblas;
            }
            case Service::LV_ADVANCED: {
                return $this->blas;
            }
            default: {
                throw new InvalidArgumentException('Unknown service level.');
            }
        }
        if($level===Service::LV_BASIC) {
        }
    }

    public function lapack(int $level=null) : object
    {
        $level = $level ?? Service::LV_ADVANCED;
        switch($level) {
            case Service::LV_BASIC: {
                return $this->phplapack;
            }
            case Service::LV_ADVANCED: {
                return $this->lapack;
            }
            default: {
                throw new InvalidArgumentException('Unknown service level.');
            }
        }
    }

    public function math(int $level=null) : object
    {
        $level = $level ?? Service::LV_ADVANCED;
        switch($level) {
            case Service::LV_BASIC: {
                return $this->phpmath;
            }
            case Service::LV_ADVANCED: {
                return $this->math;
            }
            default: {
                throw new InvalidArgumentException('Unknown service level.');
            }
        }
    }

    public function buffer(int $level=null) : object
    {
        $level = $level ?? Service::LV_ADVANCED;
        switch($level) {
            case Service::LV_BASIC: {
                return $this->phpbuffer;
            }
            case Service::LV_ADVANCED: {
                return $this->buffer;
            }
            case Service::LV_ACCELERATED: {
                return $this->bufferCLFactory;
            }
            default: {
                throw new InvalidArgumentException('Unknown service level.');
            }
        }
    }

    public function opencl() : object
    {
        return $this->openclFactory;
    }

    public function blasCL(object $queue) : object
    {
        return $this->blasCLFactory->Blas($queue,service:$this);
    }

    public function mathCL(object $queue) : object
    {
        return $this->mathCLFactory->Math($queue,service:$this);
    }

    public function mathCLBlast(object $queue) : object
    {
        return $this->clblastFactory->Math($queue,service:$this);
    }

    public function createQueue(array $options=null) : object
    {
        if($this->serviceLevel()<Service::LV_ACCELERATED) {
            throw new InvalidArgumentException('Service level requirements not met.');
        }
        if(isset($options['device'])) {
            $device = $this->getDevice($options['device']);
        } elseif(isset($options['deviceType'])) {
            $device = $this->searchDevice($options['deviceType']);
        } else {
            $device = OpenCL::CL_DEVICE_TYPE_DEFAULT;
        }
        $context = $this->openclFactory->Context($device);
        $queue = $this->openclFactory->CommandQueue($context);
        return $queue;
    }

    protected function getDevice($devOption) : object
    {
        $devOption = explode(',',$devOption);
        if(count($devOption)!=2) {
            throw new InvalidArgumentException('Device option must be two numeric with comma, etc."0,1"');
        }
        [$platformId,$deviceId] = $devOption;
        if(!is_numeric($platformId)||!is_numeric($deviceId)) {
            throw new InvalidArgumentException('platformId and deviceId must be integer, etc."0,1"');
        }
        $platformId = intval($platformId);
        $deviceId = intval($deviceId);
        $platform = $this->openclFactory->PlatformList();
        $platform = $platform->getOne($platformId);
        $device = $this->openclFactory->DeviceList($platform);
        $device = $device->getOne($deviceId);
        return $device;
    }

    protected function searchDevice($deviceType) : object
    {
        $platformList = $this->openclFactory->PlatformList();
        $platformCount = $platformList->count();
        for($p=0;$p<$platformCount;$p++) {
            $deviceList = $this->openclFactory->DeviceList($platformList->getOne($p));
            $deviceCount = $deviceList->count();
            for($d=0;$d<$deviceCount;$d++) {
                if($deviceType==OpenCL::CL_DEVICE_TYPE_DEFAULT) {
                    return $deviceList->getOne($d);
                }
                if($deviceList->getInfo($d,OpenCL::CL_DEVICE_TYPE)===$deviceType) {
                    return $deviceList->getOne($d);
                }
            }
        }
        throw new InvalidArgumentException('The specified device type cannot be found');
    }
}