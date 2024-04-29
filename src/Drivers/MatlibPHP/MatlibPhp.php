<?php
namespace Rindow\Math\Matrix\Drivers\MatlibPHP;

use Rindow\Math\Matrix\Drivers\AbstractMatlibService;

class MatlibPhp extends AbstractMatlibService
{
    protected string $name = 'matlib_php';

    protected function injectDefaultFactories() : void
    {
    }
}
