<?php
namespace Rindow\Math\Matrix;

use InvalidArgumentException;

function R(
    int $start,
    int $limit,
) : Range {
    if(func_num_args()!=2) {
        throw new InvalidArgumentException('R must have only two arguments: "start" and "limit".');
    }
    return new Range(start:$start, limit:$limit);
}
