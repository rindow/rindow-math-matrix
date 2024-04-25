<?php
namespace Rindow\Math\Matrix\Drivers;

use Rindow\Math\Matrix\Drivers\MatlibPHP\MatlibPhp;
use LogicException;
use RuntimeException;

class Selector
{
    /** @var array<string> $catalog */
    protected array $catalog;
    protected ?Service $recommended=null;

    /**
     * @param array<string> $catalog
     */
    public function __construct(array $catalog = null)
    {
        $catalog = $catalog ?? [
            'Rindow\Math\Matrix\Drivers\MatlibFFI\MatlibFFI',
            'Rindow\Math\Matrix\Drivers\MatlibExt\MatlibExt',
        ];
        $this->catalog = $catalog;
    }

    public function select() : Service
    {
        if($this->recommended) {
            return $this->recommended;
        }
        $recommended = null;
        $highestLevel = 0;
        foreach ($this->catalog as $name) {
            if(class_exists($name)) {
                $service = new $name;
                if(!($service instanceof Service)) {
                    throw new LogicException('Not service class: '.$name);
                }
                $level = $service->serviceLevel();
                if($level>$highestLevel) {
                    $highestLevel = $level;
                    $recommended = $service;
                }
            }
        }
        if($highestLevel<=Service::LV_BASIC) {
            $recommended = new MatlibPhp();
        }
        if($recommended==null) {
            throw new RuntimeException('Service not found');
        }
        $this->recommended = $recommended;
        return $recommended;
    }
}
