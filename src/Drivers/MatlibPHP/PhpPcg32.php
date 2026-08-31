<?php
/**
 * PCG32 - Extension-independent pure-PHP version
 *
 * This implementation does not rely on any PHP extensions (such as GMP).
 * Instead, it represents 64-bit integers as an array of four 16-bit 
 * "limbs" (in little-endian order, where `$limbs[0]` holds the least
 * significant 16 bits and `$limbs[3]` the most significant) and 
 * implements addition, multiplication, and bit-shifting using custom
 * carry-handling logic. 
 *
 * It generates an identical sequence of random numbers to `Pcg32.php`
 * (the GMP version) and `pcg32.c` (the native C version) when using
 * the same algorithm and seed. As its execution speed is significantly
 * slower than the native and GMP versions, it is intended for use as
 * a fallback in environments where those extensions are unavailable.
 * 
 */
namespace Rindow\Math\Matrix\Drivers\MatlibPHP;

class PhpPcg32
{
    /** Decompose 6364136223846793005 = 0x5851F42D4C957F2D into four 16-bit limbs. */
    private const MULT = [0x7F2D, 0x4C95, 0xF42D, 0x5851];

    /** @var int[] 4 elements, each 0-65535 (lower to higher) */
    private array $state;

    /** @var int[] 4 elements, each 0-65535 */
    private array $inc;

    public function __construct(int $seed = 0, int $sequence = 1)
    {
        // inc = (sequence << 1) | 1 (always odd)
        $this->inc = self::fromInt(($sequence << 1) | 1);

        $this->setSeed($seed);
    }

    /** 
     * 64bit arithmetic helpers (all return limb arrays)
     * 
     * @param int $v 
     * @return array<int,int>
     * */
    private static function fromInt(int $v): array
    {
        return [
            $v & 0xFFFF,
            ($v >> 16) & 0xFFFF,
            ($v >> 32) & 0xFFFF,
            ($v >> 48) & 0xFFFF,
        ];
    }

    /** 
     * mod 2^64 addition
     * 
     * @param array<int,int> $a 
     * @param array<int,int> $b 
     * @return array<int,int>
     * */
    private static function add64(array $a, array $b): array
    {
        $carry = 0;
        $r = [0, 0, 0, 0];
        for ($i = 0; $i < 4; $i++) {
            $sum = $a[$i] + $b[$i] + $carry;
            $r[$i] = $sum & 0xFFFF;
            $carry = $sum >> 16;
        }
        return $r; // Final carry is discarded by mod 2^64
    }

    /** 
     * mod 2^64 multiplication (manual long multiplication with 16bit limbs)
     * 
     * @param array<int,int> $a 
     * @param array<int,int> $b 
     * @return array<int,int>
     * */
    private static function mul64(array $a, array $b): array
    {
        $tmp = [0, 0, 0, 0, 0, 0, 0, 0];
        for ($i = 0; $i < 4; $i++) {
            $carry = 0;
            for ($j = 0; $j < 4; $j++) {
                $pos = $i + $j;
                $prod = $a[$i] * $b[$j] + $tmp[$pos] + $carry;
                $tmp[$pos] = $prod & 0xFFFF;
                $carry = intdiv($prod, 0x10000);
            }
            $k = $i + 4;
            while ($carry > 0 && $k < 8) {
                $prod = $tmp[$k] + $carry;
                $tmp[$k] = $prod & 0xFFFF;
                $carry = intdiv($prod, 0x10000);
                $k++;
            }
        }
        return [$tmp[0], $tmp[1], $tmp[2], $tmp[3]]; // Higher 4 limbs are truncated by mod 2^64
    }

    /** 
     * XOR of each limb
     * 
     * @param array<int,int> $a 
     * @param array<int,int> $b 
     * @return array<int,int>
     * */
    private static function xor64(array $a, array $b): array
    {
        return [$a[0] ^ $b[0], $a[1] ^ $b[1], $a[2] ^ $b[2], $a[3] ^ $b[3]];
    }

    /** 
     * Logical right shift (unsigned, $n is 0-63)
     * 
     * @param array<int,int> $a 
     * @return array<int,int>
     * */
    private static function shr64(array $a, int $n): array
    {
        if ($n <= 0) {
            return $a;
        }
        if ($n >= 64) {
            return [0, 0, 0, 0];
        }
        $limbShift = intdiv($n, 16);
        $bitShift = $n % 16;
        $r = [0, 0, 0, 0];
        for ($i = 0; $i < 4; $i++) {
            $lowIdx = $i + $limbShift;
            $highIdx = $lowIdx + 1;
            $low = ($lowIdx <= 3) ? $a[$lowIdx] : 0;
            $high = ($highIdx <= 3) ? $a[$highIdx] : 0;
            $val = ($bitShift === 0)
                ? $low
                : (($low >> $bitShift) | (($high << (16 - $bitShift)) & 0xFFFF));
            $r[$i] = $val & 0xFFFF;
        }
        return $r;
    }

    /** 
     * Extract lower 32 bits of a limb array as a PHP integer 
     * 
     * @param array<int,int> $a 
     * @return int
     * */
    private static function toUint32(array $a): int
    {
        return $a[0] | ($a[1] << 16);
    }

    private function step(): void
    {
        $this->state = self::add64(self::mul64($this->state, self::MULT), $this->inc);
    }

    // ---- Public API (same interface as Pcg32.php / pcg32.c) ----

    public function setSeed(int $seed): void
    {
        $this->state = [0, 0, 0, 0];
        $this->step();
        $this->state = self::add64($this->state, self::fromInt($seed));
        $this->step();
    }

    public function randUint32(): int
    {
        $oldstate = $this->state;
        $this->step();

        $xorshifted64 = self::shr64(
            self::xor64(self::shr64($oldstate, 18), $oldstate),
            27
        );
        $xorshifted = self::toUint32($xorshifted64);

        $rot = self::toUint32(self::shr64($oldstate, 59)) & 31;

        return (($xorshifted >> $rot) | ($xorshifted << ((-$rot) & 31))) & 0xFFFFFFFF;
    }

    /**
     * Convert Uint32(0 to 4294967295) to Int32(-2147483648 to 2147483647).
     * The bit pattern is preserved, with bit 31 reinterpreted as the sign bit (two's complement).
     */
    public static function uint32ToInt32(int $u): int
    {
        $u &= 0xFFFFFFFF;
        return ($u >= 0x80000000) ? $u - 0x100000000 : $u;
    }
 
    /**
     * Return the next random number as a signed 32-bit integer (-2147483648 to 2147483647).
     */
    public function nextInt32(): int
    {
        return self::uint32ToInt32($this->randUint32());
    }

    public function nextFloat(): float
    {
        return $this->randUint32() / 4294967296.0;
    }

    public function nextInt(int $min, int $max): int
    {
        $range = $max - $min + 1;
        if ($range <= 0) {
            throw new \InvalidArgumentException('max must be >= min');
        }
        $threshold = (0x100000000 - $range) % $range;
        do {
            $r = $this->randUint32();
        } while ($r < $threshold);

        return $min + ($r % $range);
    }

    /** @return float[] */
    public function fillUniform(int $size, float $low = 0.0, float $high = 1.0): array
    {
        $out = [];
        $scale = $high - $low;
        for ($i = 0; $i < $size; $i++) {
            $out[] = $low + $this->nextFloat() * $scale;
        }
        return $out;
    }

    /** @return array{state:int[], inc:int[]} */
    public function getState(): array
    {
        return ['state' => $this->state, 'inc' => $this->inc];
    }

    /** @param array{state:int[], inc:int[]} $stateData */
    public function setState(array $stateData): void
    {
        $this->state = $stateData['state'];
        $this->inc = $stateData['inc'];
    }
}