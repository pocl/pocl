
#define assert(x)

// OpenCL decls
#include "CL/cl.h"

/* plain frexp(double, int*) doesn't work because address spaces
 * mangling in OpenCL. expfrexp is a pocl-specific kernel library function. */
cl_long _CL_OVERLOADABLE _cl_expfrexp(double x);
double _CL_OVERLOADABLE _cl_floor(double);

#define expfrexp _cl_expfrexp
#define floor _cl_floor

#define NULL ((void*)0)

#include "errol.h"
#include "itoa_c.h"

/*
 * floating point format definitions
 */

typedef double fpnum_t;

static inline double fpnext(double val)
{
        errol_bits_t bits = { val };
        bits.i++;
        return bits.d;
}

static inline double fpprev(double val)
{
        errol_bits_t bits = { val };
        bits.i--;
        return bits.d;
}

static inline __uint128_t fpeint(double from)
{
        errol_bits_t bits = { from };
        assert((bits.i & ((1ULL << 52) - 1)) == 0 && (bits.i >> 52) >= 1023);

        return (__uint128_t)1 << ((bits.i >> 52) - 1023);
}

#define ERROL0_EPSILON  0.0000001
#define ERROL1_EPSILON  8.77e-15


/**
 * High-precision data structure.
 *   @val, off: The value and offset.
 */

struct hp_t {
        fpnum_t val, off;
};


/*
 * lookup table data
 */

#include "lookup.h"
#include "enum3.h"
#include "enum4.h"

/*
 * high-precision constants
 */

/*
 * local function declarations
 */

static void inline hp_normalize(struct hp_t *hp);
static void inline hp_mul10(struct hp_t *hp);
static void inline hp_div10(struct hp_t *hp);
static struct hp_t hp_prod(struct hp_t in, double val);
static int inline mismatch10(cl_ulong a, cl_ulong b);
static int inline table_lower_bound(cl_ulong *table, int n, cl_ulong k);

/*
 * inline function instantiations
 */

extern inline char *u32toa(cl_int value, char *buffer);
extern inline char *u64toa(cl_long value, char *buffer);

/*
 * intrinsics
 */

extern __uint128_t __udivmodti4(__uint128_t a, __uint128_t b,
                                __uint128_t* rem);

static char *
strcpy(char *dest, const char *src)
{
  char *temp = dest;
  while ((*dest++ = *src++));
  return temp;
}

/**
 * Errol0 double to ASCII conversion, guaranteed correct but possibly not
 * optimal. Useful for embedded systems.
 *   @val: The value.
 *   @buf: The output buffer.
 *   &returns: The exponent.
 */

int errol0_dtoa(double val, char *buf)
{
        double ten;
        int exp;
        struct hp_t mid, inhi, inlo;

        if(val == CL_DBL_MAX) {
                strcpy(buf, "17976931348623157");
                return 309;
        }

        ten = 1.0;
        exp = 1;

        /* normalize the midpoint */

        mid.val = val;
        mid.off = 0.0;

        while(((mid.val > 10.0) || ((mid.val == 10.0) && (mid.off >= 0.0))) && (exp < 308))
                exp++, hp_div10(&mid), ten /= 10.0;

        while(((mid.val < 1.0) || ((mid.val == 1.0) && (mid.off < 0.0))) && (exp > -307))
                exp--, hp_mul10(&mid), ten *= 10.0;

        inhi.val = mid.val;
        inhi.off = mid.off + (fpnext(val) - val) * ten / (2.0 + ERROL0_EPSILON);
        inlo.val = mid.val;
        inlo.off = mid.off + (fpprev(val) - val) * ten / (2.0 + ERROL0_EPSILON);

        hp_normalize(&inhi);
        hp_normalize(&inlo);

        /* normalized boundaries */

        while(inhi.val > 10.0 || (inhi.val == 10.0 && (inhi.off >= 0.0)))
                exp++, hp_div10(&inhi), hp_div10(&inlo);

        while(inhi.val < 1.0 || (inhi.val == 1.0 && (inhi.off < 0.0)))
                exp--, hp_mul10(&inhi), hp_mul10(&inlo);

        /* digit generation */

        while(inhi.val != 0.0 || inhi.off != 0.0) {
                uint8_t ldig, hdig = 0;

                hdig = (uint8_t)(inhi.val);
                if((inhi.val == hdig) && (inhi.off < 0))
                        hdig -= 1;

                ldig = (uint8_t)(inlo.val);
                if((inlo.val == ldig) && (inlo.off < 0))
                        ldig -= 1;

                if(ldig != hdig)
                        break;

                *buf++ = hdig + '0';

                inhi.val -= hdig;
                hp_mul10(&inhi);

                inlo.val -= ldig;
                hp_mul10(&inlo);
        }

        double mdig = (inhi.val + inlo.val) / 2.0 + 0.5;
        *buf++ = (uint8_t)(mdig) + '0';
        *buf = '\0';

        return exp;
}

/**
 * Errol double to ASCII conversion.
 *   @val: The value.
 *   @buf: The output buffer.
 *   @opt: The optimality flag.
 *   &returns: The exponent.
 */

int errol1_dtoa(double val, char *buf, bool *opt)
{
        double ten, lten;
        int e;
        int exp;
        struct hp_t mid, inhi, inlo, outhi, outlo;

        if(val == CL_DBL_MAX) {
                strcpy(buf, "17976931348623157");
                return 309;
        }

        /* normalize the midpoint */

        e = expfrexp(val);
        exp = 307 + (double)e*0.30103;
        if(exp < 20)
                exp = 20;
        else if(exp >= LOOKUP_TABLE_LEN)
                exp = LOOKUP_TABLE_LEN - 1;

        mid = lookup_table[exp];
        mid = hp_prod(mid, val);
        lten = lookup_table[exp].val;
        ten = 1.0;

        exp -= 307;

        while((mid.val > 10.0) || ((mid.val == 10.0 && mid.off >= 0.0)))
                exp++, hp_div10(&mid), ten /= 10.0;

        while((mid.val < 1.0) || ((mid.val == 1.0) && (mid.off < 0.0)))
                exp--, hp_mul10(&mid), ten *= 10.0;

        inhi.val = mid.val;
        inhi.off = mid.off + (fpnext(val) - val) * lten * ten / (2.0 + ERROL1_EPSILON);
        inlo.val = mid.val;
        inlo.off = mid.off + (fpprev(val) - val) * lten * ten / (2.0 + ERROL1_EPSILON);
        outhi.val = mid.val;
        outhi.off = mid.off + (fpnext(val) - val) * lten * ten / (2.0 - ERROL1_EPSILON);
        outlo.val = mid.val;
        outlo.off = mid.off + (fpprev(val) - val) * lten * ten / (2.0 - ERROL1_EPSILON);

        hp_normalize(&inhi);
        hp_normalize(&inlo);
        hp_normalize(&outhi);
        hp_normalize(&outlo);

        /* normalized boundaries */

        while(inhi.val > 10.0 || (inhi.val == 10.0 && inhi.off >= 0.0))
                exp++, hp_div10(&inhi), hp_div10(&inlo), hp_div10(&outhi), hp_div10(&outlo);

        while(inhi.val < 1.0 || (inhi.val == 1.0 && inhi.off < 0.0))
                exp--, hp_mul10(&inhi), hp_mul10(&inlo), hp_mul10(&outhi), hp_mul10(&outlo);

        /* digit generation */

        *opt = true;

        while(inhi.val != 0.0 || inhi.off != 0.0) {
                uint8_t ldig, hdig;

                hdig = (uint8_t)inhi.val;
                if((inhi.val == hdig) && (inhi.off < 0))
                        hdig -= 1;

                ldig = (uint8_t)inlo.val;
                if((inlo.val == ldig) && (inlo.off < 0))
                        ldig -= 1;

                if(ldig != hdig)
                        break;

                *buf++ = hdig + '0';
                inhi.val -= hdig;
                inlo.val -= ldig;
                hp_mul10(&inhi);
                hp_mul10(&inlo);

                hdig = (uint8_t)outhi.val;
                if((outhi.val == hdig) && (outhi.off < 0))
                        hdig -= 1;

                ldig = (uint8_t)outlo.val;
                if((outlo.val == ldig) && (outlo.off < 0))
                        ldig -= 1;

                if(ldig != hdig)
                        *opt = false;

                outhi.val -= hdig;
                outlo.val -= ldig;
                hp_mul10(&outhi);
                hp_mul10(&outlo);
        }

        double mdig = (inhi.val + inlo.val) / 2.0 + 0.5;
        *buf++ = (uint8_t)mdig + '0';
        *buf = '\0';

        return exp;
}

/**
 * Errol2 double to ASCII conversion.
 *   @val: The value.
 *   @buf: The output buffer.
 *   @opt: The optimality flag.
 *   &returns: The exponent.
 */

int errol2_dtoa(double val, char *buf, bool *opt)
{
        if((val <= 9.007199254740992e15) || (val >= 3.40282366920938e38))
                return errol1_dtoa(val, buf, opt);

        int8_t i, j;
        int exp;
        errol_bits_t bits;
        char lstr[42], hstr[42], mstr[41];
        cl_ulong l64, m64, h64;
        __uint128_t low, mid, high, pow19 = (__uint128_t)1e19;

        mid = (__uint128_t)val;
        low = mid - fpeint((fpnext(val) - val) / 2.0);
        high = mid + fpeint((val - fpprev(val)) / 2.0);

        bits.d = val;
        if(bits.i & 0x1)
                high--;
        else
                low--;

        i = 39;
        lstr[41] = hstr[41] = mstr[40] = '\0';
        lstr[40] = hstr[40] = '5';
        while(high != 0) {
                __uint128_t tmp1;
                low = __udivmodti4(low, pow19, &tmp1);
                l64 = tmp1;
                mid = __udivmodti4(mid, pow19, &tmp1);
                m64 = tmp1;
                high = __udivmodti4(high, pow19, &tmp1);
                h64 = tmp1;

                for(j = 0; ((high != 0) && (j < 19)) || ((high == 0) && (h64 != 0)); j++, i--) {
                        lstr[i] = '0' + l64 % (cl_ulong)10;
                        mstr[i] = '0' + m64 % (cl_ulong)10;
                        hstr[i] = '0' + h64 % (cl_ulong)10;

                        l64 /= 10;
                        m64 /= 10;
                        h64 /= 10;
                }
        }

        exp = 39 - i++;

        do
                *buf++ = hstr[i++];
        while(hstr[i] != '\0' && hstr[i] == lstr[i]);

        *buf++ = mstr[i] + ((mstr[i+1] >= '5') ? 1 : 0);

        *buf = '\0';
        *opt = true;

        return exp;
}


/**
 * Corrected Errol3 double to ASCII conversion.
 *   @val: The value.
 *   @buf: The output buffer.
 *   &returns: The exponent.
 */

int errol3_dtoa(double val, char *buf)
{
        errol_bits_t k = { val };

        int n = sizeof(errol_enum3) / sizeof(cl_ulong);
        int i = table_lower_bound(errol_enum3, n, k.i);
        if (i < n && errol_enum3[i] == k.i)
        {
                strcpy(buf, errol_enum3_data[i].str);
                return errol_enum3_data[i].exp;
        }

        return errol3u_dtoa(val, buf);
}

/**
 * Uncorrected Errol3 double to ASCII conversion.
 *   @val: The value.
 *   @buf: The output buffer.
 *   &returns: The exponent.
 */

int errol3u_dtoa(double val, char *buf)
{
        int e;
        double ten, lten;
        int exp;
        struct hp_t mid;
        struct hp_t high = { val, 0.0 };
        struct hp_t low = { val, 0.0 };

        /* check if in integer or fixed range */

        if((val > 9.007199254740992e15) && (val < 3.40282366920938e+38))
                return errol_int(val, buf);
        else if((val >= 16.0) && (val <= 9.007199254740992e15))
                return errol_fixed(val, buf);

        /* normalize the midpoint */

        e = expfrexp(val);
        exp = 307 + (double)e*0.30103;
        if(exp < 20)
                exp = 20;
        else if(exp >= LOOKUP_TABLE_LEN)
                exp = LOOKUP_TABLE_LEN - 1;

        mid = lookup_table[exp];
        mid = hp_prod(mid, val);
        lten = lookup_table[exp].val;
        ten = 1.0;

        exp -= 307;

        while(mid.val > 10.0 || (mid.val == 10.0 && mid.off >= 0.0))
                exp++, hp_div10(&mid), ten /= 10.0;

        while(mid.val < 1.0 || (mid.val == 1.0 && mid.off < 0.0))
                exp--, hp_mul10(&mid), ten *= 10.0;

        /* compute boundaries */

        high.val = mid.val;
        high.off = mid.off + (fpnext(val) - val) * lten * ten / 2.0;
        low.val = mid.val;
        low.off = mid.off + (fpprev(val) - val) * lten * ten / 2.0;

        hp_normalize(&high);
        hp_normalize(&low);

        /* normalized boundaries */

        while(high.val > 10.0 || (high.val == 10.0 && high.off >= 0.0))
                exp++, hp_div10(&high), hp_div10(&low);

        while(high.val < 1.0 || (high.val == 1.0 && high.off < 0.0))
                exp--, hp_mul10(&high), hp_mul10(&low);

        /* digit generation */

        while(true) {
                int8_t ldig, hdig;

                hdig = (uint8_t)(high.val);
                if((high.val == hdig) && (high.off < 0))
                        hdig -= 1;

                ldig = (uint8_t)(low.val);
                if((low.val == ldig) && (low.off < 0))
                        ldig -= 1;

                if(ldig != hdig)
                        break;

                *buf++ = hdig + '0';
                high.val -= hdig;
                low.val -= ldig;
                hp_mul10(&high);
                hp_mul10(&low);
        }

        double tmp = (high.val + low.val) / 2.0;
        uint8_t mdig = tmp + 0.5;
        if((mdig - tmp) == 0.5 && (mdig & 0x1))
                mdig--;

        *buf++ = mdig + '0';
        *buf = '\0';

        return exp;
}

/**
 * Corrected Errol4 double to ASCII conversion.
 *   @val: The value.
 *   @buf: The output buffer.
 *   &returns: The exponent.
 */

int errol4_dtoa(double val, char *buf)
{
        errol_bits_t k = { val };

        int n = sizeof(errol_enum4) / sizeof(cl_ulong);
        int i = table_lower_bound(errol_enum4, n, k.i);
        if (i < n && errol_enum4[i] == k.i)
        {
                strcpy(buf, errol_enum4_data[i].str);
                return errol_enum4_data[i].exp;
        }

        return errol4u_dtoa(val, buf);
}

/**
 * Uncorrected Errol4 double to ASCII conversion.
 *   @val: The value.
 *   @buf: The output buffer.
 *   &returns: The exponent.
 */

int errol4u_dtoa(double val, char *buf)
{
        int e;
        int exp;
        struct hp_t mid;
        double ten, lten;

        /* check if in integer or fixed range */

        if((val >= 1.80143985094820e+16) && (val < 3.40282366920938e+38))
                return errol_int(val, buf);
        else if((val >= 16.0) && (val <= 9.007199254740992e15))
                return errol_fixed(val, buf);
        
        /* normalize the midpoint */

        e = expfrexp(val);
        exp = 290 + (double)e*0.30103;
        if(exp < 20)
                exp = 20;
        else if(exp >= LOOKUP_TABLE_LEN)
                exp = LOOKUP_TABLE_LEN - 1;

        mid = lookup_table[exp];
        lten = mid.val;
        mid = hp_prod(mid, val);

        ten = 1.0;
        exp -= 290;

        if(mid.val < 1.00000000000000016e+17) {
                e = expfrexp(val);
                exp = 290 + (double)e*0.30103;

                mid = hp_prod(mid, val);
                lten *= mid.val;
                mid = hp_prod(mid, val);
        }

        while(mid.val < 1.00000000000000016e+17)
                exp--, hp_mul10(&mid), ten *= 10.0;

        double diff = (fpnext(val) - val) * lten * ten / 2.0;
        cl_ulong val64 = (cl_ulong)mid.val;
        cl_ulong lo64 = val64 + (cl_ulong)floor(mid.off - diff);
        cl_ulong mid64;
        cl_ulong hi64 = val64 + (cl_ulong)floor(mid.off + diff);

        if(hi64 >= 1e18)
                exp++;

        cl_ulong iten;
        for(iten = 1; ; iten *= 10) {
                lo64 /= 10;
                hi64 /= 10;

                if(lo64 == hi64)
                        break;
        }

        mid64 = (val64 + (cl_ulong)floor(mid.off + iten * 0.5)) / iten;

        if (hi64 > 0)
                buf = u64toa(hi64, buf);
        *buf++ = mid64 % 10 + '0';
        *buf = '\0';

        return exp;
}


/**
 * Integer conversion algorithm, guaranteed correct, optimal, and best.
 *   @val: The val.
 *   @buf: The output buffer.
 *   &return: The exponent.
 */

int errol_int(double val, char *buf)
{
        int exp;
        errol_bits_t bits;
        __uint128_t low, mid, high;
        static __uint128_t pow19 = (__uint128_t)1e19;

        assert((val > 9.007199254740992e15) && val < (3.40282366920938e38));

        mid = (__uint128_t)val;
        low = mid - fpeint((fpnext(val) - val) / 2.0);
        high = mid + fpeint((val - fpprev(val)) / 2.0);

        bits.d = val;
        if(bits.i & 0x1)
                high--;
        else
                low--;

        __uint128_t tmp1, tmp2;
        __udivmodti4(__udivmodti4(low, pow19, &tmp1), pow19, &tmp2);
        cl_ulong l64 = tmp1;
        cl_ulong lf = tmp2;
        __udivmodti4(__udivmodti4(high, pow19, &tmp1), pow19, &tmp2);
        cl_ulong h64 = tmp1;
        cl_ulong hf = tmp2;

        if (lf != hf)
        {
                l64 = lf;
                h64 = hf;
                mid = __udivmodti4(mid, pow19 / 10, NULL);
        }

        int mi = mismatch10(l64, h64);
        cl_ulong x = 1;
        for (int i = (lf == hf); i < mi; i++)
                x *= 10;
        cl_ulong m64 = __udivmodti4(mid, x, NULL);

        if (lf != hf)
                mi += 19;

        char *p = u64toa(m64, buf) - 1;

        if (mi != 0)
                p[-1] += (*p >= '5');
        else
                ++p;

        exp = p - buf + mi;
        *p = '\0';

        return exp;
}

/**
 * Fixed point conversion algorithm, guaranteed correct, optimal, and best.
 *   @val: The val.
 *   @buf: The output buffer.
 *   &return: The exponent.
 */

int errol_fixed(double val, char *buf)
{
        char *p;
        int j, exp;
        double n, mid, lo, hi;
        cl_ulong u;

        assert((val >= 16.0) && (val <= 9.007199254740992e15));

        u = (cl_ulong)val;
        n = (double)u;

        mid = val - n;
        lo = ((fpprev(val) - n) + mid) / 2.0;
        hi = ((fpnext(val) - n) + mid) / 2.0;

        p = u64toa(u, buf);
        j = exp = p - buf;
        buf[j] = '\0';

        if(mid != 0.0) {
                while(mid != 0.0) {
                        int ldig, mdig, hdig;

                        lo *= 10.0;
                        ldig = (int)lo;
                        lo -= ldig;

                        mid *= 10.0;
                        mdig = (int)mid;
                        mid -= mdig;

                        hi *= 10.0;
                        hdig = (int)hi;
                        hi -= hdig;

                        buf[j++] = mdig + '0';

                        if(hdig != ldig || j > 50)
                                break;
                }

                if(mid > 0.5)
                        buf[j-1]++;
                else if((mid == 0.5) && (buf[j-1] & 0x1))
                        buf[j-1]++;
        }
        else {
                while(buf[j-1] == '0') {
                        buf[j-1] = '\0';
                        j--;
                }
        }

        buf[j] = '\0';

        return exp;
}


/**
 * Normalize the number by factoring in the error.
 *   @hp: The float pair.
 */

static inline void hp_normalize(struct hp_t *hp)
{
        fpnum_t val = hp->val;

        hp->val += hp->off;
        hp->off += val - hp->val;
}

/**
 * Multiply the high-precision number by ten.
 *   @hp: The high-precision number
 */

static inline void hp_mul10(struct hp_t *hp)
{
        fpnum_t off, val = hp->val;

        hp->val *= 10.0;
        hp->off *= 10.0;
        
        off = hp->val;
        off -= val * 8.0;
        off -= val * 2.0;

        hp->off -= off;

        hp_normalize(hp);
}

/**
 * Divide the high-precision number by ten.
 *   @hp: The high-precision number
 */

static inline void hp_div10(struct hp_t *hp)
{
        double val = hp->val;

        hp->val /= 10.0;
        hp->off /= 10.0;

        val -= hp->val * 8.0;
        val -= hp->val * 2.0;

        hp->off += val / 10.0;

        hp_normalize(hp);
}

static inline double gethi(double in)
{
        errol_bits_t v = { .d = in };

        //v.i += 0x0000000004000000;
        v.i &= 0xFFFFFFFFF8000000;

        return v.d;
}

/**
 * Split a double into two halves.
 *   @val: The double.
 *   @hi: The high bits.
 *   @lo: The low bits.
 */

static inline void split(double val, double *hi, double *lo)
{
        //double t = (134217728.0 + 1.0) * val;

        //*hi = t - (t - val);
        //if(*hi != gethi(val)) 
        //*lo = val - *hi;

        *hi = gethi(val);
        *lo = val - *hi;
}

/**
 * Compute the product of an HP number and a double.
 *   @in: The HP number.
 *   @val: The double.
 *   &returns: The HP number.
 */

static struct hp_t hp_prod(struct hp_t in, double val)
{
        double p, hi, lo, e;

        double hi2, lo2;
        split(in.val, &hi, &lo);
        split(val, &hi2, &lo2);

        p = in.val * val;
        e = ((hi * hi2 - p) + lo * hi2 + hi * lo2) + lo * lo2;

        return (struct hp_t){ p, in.off * val + e };
}

/**
 * Given two different integers with the same length in terms of the number
 * of decimal digits, index the digits from the right-most position starting
 * from zero, find the first index where the digits in the two integers
 * divergent starting from the highest index.
 *   @a: Integer a.
 *   @b: Integer b.
 *   &returns: An index within [0, 19).
 */

static inline int mismatch10(cl_ulong a, cl_ulong b)
{
        cl_ulong pow10 = 10000000000U;
        cl_ulong af = a / pow10;
        cl_ulong bf = b / pow10;
        int i = 0;

        if (af != bf)
        {
                i = 10;
                a = af;
                b = bf;
        }

        for (;; ++i)
        {
                a /= 10;
                b /= 10;

                if (a == b)
                        return i;
        }
}

/**
 * Find the insertion point for a key in a level-order array.
 *   @table: The target array.
 *   @n: Array size.
 *   @k: The key to find.
 *   &returns: The insertion point.
 */

static inline int table_lower_bound(cl_ulong *table, int n, cl_ulong k)
{
        int i = n, j = 0;

        while (j < n)
        {
                if (table[j] < k)
                        j = 2 * j + 2;
                else
                {
                        i = j;
                        j = 2 * j + 1;
                }
        }

        return i;
}
