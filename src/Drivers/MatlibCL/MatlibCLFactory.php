<?php
namespace Rindow\Math\Matrix\Drivers\MatlibCL;

use Rindow\Math\Matrix\Drivers\Driver;
use Rindow\Math\Matrix\Drivers\Service;

use RuntimeException;

class MatlibCLFactory implements Driver
{
    public function isAvailable() : bool
    {
        return true;
    }

    public function Math(object $queue, Service $service) : object
    {
        $openclmath = new OpenCLMath($queue, $service);
        if($openclmath->hasDiv5Bug()) {
            throw new RuntimeException("OpenCL Device has The Div5bug.");
        }
        return $openclmath;
    }
}
