<?php
$paths = [
    __DIR__.'/../../../autoload.php',
    __DIR__.'/../vendor/autoload.php',
];
foreach($paths as $path) {
    if(file_exists($path)) {
        include_once $path;
        break;
    }
}
use Rindow\Math\Matrix\MatrixOperator;
$mo = new MatrixOperator();
echo $mo->service()->info();
