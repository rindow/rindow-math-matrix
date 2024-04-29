<?php
if(!defined('COMPOSER_LIBRARY_PATH')) {
    define('COMPOSER_LIBRARY_PATH', getenv('COMPOSER_LIBRARY_PATH'));
}
if(COMPOSER_LIBRARY_PATH && file_exists(COMPOSER_LIBRARY_PATH.'/vendor/autoload.php')) {
    $loader = include COMPOSER_LIBRARY_PATH.'/vendor/autoload.php';
} else {
    throw new \Exception("Loader is not found.");
}
$loader->addPsr4('Rindow\\Math\\Matrix\\', __DIR__.'/../src');
$loader->addPsr4('Rindow\\Math\\Plot\\', __DIR__.'/../../rindow-math-plot/src');
$loader->addPsr4('Interop\\Polite\\Math\\', __DIR__.'/../../../interop-phpobjects/polite-math/src');

return $loader;
