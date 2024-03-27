<?php
ini_set('short_open_tag', '1');

date_default_timezone_set('UTC');
#ini_set('short_open_tag',true);
if(file_exists(__DIR__.'/../vendor/autoload.php')) {
    $loader = require_once __DIR__.'/../vendor/autoload.php';
} else {
    $loader = require_once __DIR__.'/init_autoloader.php';
}
if(file_exists(__DIR__.'/../addpack/vendor/autoload.php')) {
    $addpack = __DIR__.'/../addpack/vendor/rindow/';
    $loader->addPsr4('Rindow\\Math\\Matrix\\Drivers\\MatlibFFI\\',$addpack.'rindow-math-matrix-matlibffi/src');
    $loader->addPsr4('Rindow\\Math\\Buffer\\FFI\\',$addpack.'rindow-math-buffer-ffi/src');
    $loader->addPsr4('Rindow\\OpenBLAS\\FFI\\',$addpack.'rindow-openblas-ffi/src');
    $loader->addPsr4('Rindow\\Matlib\\FFI\\',$addpack.'rindow-matlib-ffi/src');
    $loader->addPsr4('Rindow\\OpenCL\\FFI\\',$addpack.'rindow-opencl-ffi/src');
    $loader->addPsr4('Rindow\\CLBlast\\FFI\\',$addpack.'rindow-clblast-ffi/src');
}
#if(!class_exists('PHPUnit\Framework\TestCase')) {
#    include __DIR__.'/travis/patch55.php';
#}
