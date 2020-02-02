<?php
namespace Rindow\Math\Matrix;

use Iterator;
use RangeException;
use Interop\Polite\Math\Matrix\BLAS;
use Interop\Polite\Math\Matrix\NDArray;

class MatrixBufferIterator implements Iterator
{
    protected $shape;
    protected $skipDims;
    protected $current;
    protected $endOfItem = false;

    public function __construct(array $shape, array $skipDims)
    {
        $this->shape = $shape;
        $this->skipDims = $skipDims;
        $this->current = array_fill(0,count($shape),0);
    }

    public function getCurrentIndex()
    {
        return $this->current;
    }

    public function current()
    {
        if($this->endOfItem) {
            throw new RangeException('End of buffer');
        }
        $w = 1;
        $pos = 0;
        for($i=count($this->shape)-1; $i>=0; $i--) {
            $pos += $this->current[$i]*$w;
            $w *= $this->shape[$i];
        }
        return $pos;
    }

    public function key()
    {
        return null;
    }

    public function next()
    {
        if($this->endOfItem)
            return;

        for($dimNum=count($this->shape)-1; $dimNum>=0; $dimNum--) {
            if(in_array($dimNum,$this->skipDims))
                continue;
            $this->current[$dimNum]++;
            if($this->current[$dimNum]<$this->shape[$dimNum]) {
                return;
            }
            $this->current[$dimNum] = 0;
        }
        $this->endOfItem = true;
    }

    public function rewind()
    {
        $this->current = array_fill(0,count($this->shape),0);
        $this->endOfItem = false;
    }

    public function valid()
    {
        if($this->endOfItem)
            return false;

        return true;
    }
}
