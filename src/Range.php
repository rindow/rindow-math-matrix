<?php
namespace Rindow\Math\Matrix;

use IteratorAggregate;
use Traversable;

class Range implements IteratorAggregate
{
    protected mixed $start;
    protected mixed $limit;
    protected mixed $delta;

    public function __construct(
        int|float $limit,
        int|float $start=null,
        int|float $delta=null)
    {
        $this->limit = $limit;
        $this->start = $start ?? 0;
        $this->delta = $delta ?? (($limit>=$start)? 1 : -1);
    }

    public function start() : mixed
    {
        return $this->start;
    }

    public function limit() : mixed
    {
        return $this->limit;
    }

    public function delta() : mixed
    {
        return $this->delta;
    }

    public function  getIterator() : Traversable
    {
        $index = 0;
        $value = $this->start;
        if($this->delta > 0) {
            while($value < $this->limit) {
                yield $index => $value;
                $index++;
                $value += $this->delta;
            }
        } else {
            while($value > $this->limit) {
                yield $index => $value;
                $index++;
                $value += $this->delta;
            }
        }
    }
}
