<?php
namespace Rindow\Math\Matrix\Drivers;

use LogicException;

abstract class AbstractDriver implements Driver
{
    // abstract properties
    protected string $LOWEST_VERSION = '1000.1000.1000';
    protected string $OVER_VERSION   = '0.0.0';
    protected string $extName        = 'unknown';

    protected function strVersion(string $name=null) : string
    {
        if($name==null) {
            $version = phpversion();
        } else {
            $version = phpversion($name);
        }
        if($version===false) {
            $version = '0.0.0';
        }
        return $version;
    }

    protected function assertExtensionVersion(string $name, string $lowestVersion, string $overVersion) : void
    {
        $currentVersion = $this->strVersion($name);
        if(version_compare($currentVersion, $lowestVersion)<0||
            version_compare($currentVersion, $overVersion)>=0) {
            throw new LogicException($name.' '.$currentVersion.' is an unsupported version. '.
            'Supported versions are greater than or equal to '.$lowestVersion.
            ' and less than '.$overVersion.'.');
        }
    }

    protected function assertVersion() : void
    {
        $this->assertExtensionVersion(
            $this->extName,
            $this->LOWEST_VERSION,
            $this->OVER_VERSION
        );
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
        $version = $this->strVersion($this->extName);
        return $version;
    }
}
