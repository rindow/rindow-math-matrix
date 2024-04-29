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

$verbose = null;
if($argc>1) {
    if($argv[1]=='-v') {
        $verbose = 10;
    }
}
$mo = new MatrixOperator(verbose:$verbose);
if($verbose!==null) {
    echo "\n";
}
echo $mo->service()->info();
