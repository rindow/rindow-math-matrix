<?php
namespace Rindow\Math\Matrix\Drivers;

interface Service
{
    const LV_UNAVAILABLE = 0;
    const LV_BASIC       = 1;   // Pure PHP
    const LV_ADVANCED    = 2;   // Use external library
    const LV_ACCELERATED = 3;   // Use some accelerater like a GPU

    public function serviceLevel() : int;
}
