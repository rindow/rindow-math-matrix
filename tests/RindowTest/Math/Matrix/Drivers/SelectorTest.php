<?php
namespace RindowTest\Math\Matrix\Drivers\SelectorTest;

use PHPUnit\Framework\TestCase;
use Rindow\Math\Matrix\Drivers\MatlibExt\MatlibExt;
use Rindow\Math\Matrix\Drivers\MatlibFFI\MatlibFFI;
use Rindow\Math\Matrix\Drivers\MatlibPHP\MatlibPhp;
use Rindow\Math\Matrix\Drivers\Selector;
use Rindow\Math\Matrix\Drivers\Service;

use Interop\Polite\Math\Matrix\NDArray;

class SelectorTest extends TestCase
{
    public function newSelector($catalog=null)
    {
        return new Selector($catalog);
    }

    public function testDefault()
    {
        $ext = null;
        $extLevel = 0;
        if(class_exists('Rindow\Math\Matrix\Drivers\MatlibExt\MatlibExt')) {
            $ext = new MatlibExt();
        }
        $ffi = null;
        $ffiLevel = 0;
        if(class_exists('Rindow\Math\Matrix\Drivers\MatlibFFI\MatlibFFI')) {
            $ffi = new MatlibFFI();
        }
        $php = new MatlibPhp();
        $selector = $this->newSelector();
        $service = $selector->select();
        if($ext != null) {
            $extLevel = $ext->serviceLevel();
        }
        if($ffi != null) {
            $ffiLevel = $ffi->serviceLevel();
        }

        if($ffiLevel==0 && $extLevel==0) {
            $this->assertInstanceOf(MatlibPhp::class,$service);
        } elseif($ffiLevel>$extLevel) {
            $this->assertInstanceOf(MatlibFFI::class,$service);
        } else {
            $this->assertInstanceOf(MatlibExt::class,$service);
        }
    }

    public function testCatalog()
    {
        $classFFI = 'Rindow\Math\Matrix\Drivers\MatlibFFI\MatlibFFI';
        $classExt = 'Rindow\Math\Matrix\Drivers\MatlibExt\MatlibExt';
        $ffi = null;
        if(class_exists($classFFI)) {
            $ffi = new MatlibFFI();
        }
        $ext = null;
        if(class_exists($classExt)) {
            $ext = new MatlibExt();
        }
        $php = new MatlibPhp();

        $truesrv = $php;
        $level = Service::LV_BASIC;
        if($ffi!==null &&
            $ffi->serviceLevel()>$level) {
            $truesrv = $ffi;
            $level = $ffi->serviceLevel();
        }
        if($ext!==null &&
            $ext->serviceLevel()>$level) {
            $truesrv = $ext;
            $level = $ext->serviceLevel();
        }
        $catalog = [$classFFI,$classExt,MatlibPhp::class];
        $selector = $this->newSelector($catalog);
        $service = $selector->select();
        $this->assertInstanceOf(get_class($truesrv),$service);
    }
}
