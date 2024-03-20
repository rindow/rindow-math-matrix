<?php
namespace Rindow\Math\Matrix\Drivers;

use LogicException;

abstract class AbstractDriver implements Driver
{
    protected string $LOWEST_VERSION = '1000.1000.1000';
    protected string $OVER_VERSION   = '0.0.0';

    protected function assertExtensionVersion($name,$lowestVersion,$overVersion)
    {
        $currentVersion = phpversion($name);
        if(version_compare($currentVersion,$lowestVersion)<0||
            version_compare($currentVersion,$overVersion)>=0 ) {
                throw new LogicException($name.' '.$currentVersion.' is an unsupported version. '.
                'Supported versions are greater than or equal to '.$lowestVersion.
                ' and less than '.$overVersion.'.');
        }
    }

    protected function assertVersion()
    {
        $this->assertExtensionVersion($this->extName,
            $this->LOWEST_VERSION,
            $this->OVER_VERSION);
    }

    public function name() : string
    {
        return $this->extName();
    }

    public function isAvailable() : bool
    {
        return extension_loaded($this->extName);
    }

    public function extName() : string
    {
        return $this->extName;
    }

    public function version() : string
    {
        return phpversion($this->extName);
    }

}
