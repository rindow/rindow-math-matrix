<?php
namespace Rindow\Math\Matrix\Drivers;

interface Service
{
    const LV_UNAVAILABLE = 0;
    const LV_BASIC       = 1;   // Pure PHP
    const LV_ADVANCED    = 2;   // Use external library
    const LV_ACCELERATED = 3;   // Use some accelerater like a GPU

    public function serviceLevel() : int;
    public function info() : string;
    public function name() : string;
    public function blas(int $level=null) : object;
    public function lapack(int $level=null) : object;
    public function math(int $level=null) : object;
    public function buffer(int $level=null) : object;
    public function openCL() : object;
    public function blasCL(object $queue) : object;
    public function mathCL(object $queue) : object;
    public function mathCLBlast(object $queue) : object;
    /**
     * @param array<string,mixed> $options
     */
    public function createQueue(array $options=null) : object;
}
