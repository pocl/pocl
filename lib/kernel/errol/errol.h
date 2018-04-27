#ifndef ERROL_H
#define ERROL_H

#ifdef __cplusplus
extern "C" {
#endif

/*
 * errol declarations
 */

// TODO include stdbool ?
typedef int bool;
#define true 1
#define false 0

#define ERR_LEN   512
#define ERR_DEPTH 4

int errol0_dtoa(double val, char *buf);
int errol1_dtoa(double val, char *buf, bool *opt);
int errol2_dtoa(double val, char *buf, bool *opt);
int errol3_dtoa(double val, char *buf);
int errol3u_dtoa(double val, char *buf);
int errol4_dtoa(double val, char *buf);
int errol4u_dtoa(double val, char *buf);

int errol_int(double val, char *buf);
int errol_fixed(double val, char *buf);

struct errol_err_t {
        double val;
        cl_char str[18];
        cl_int exp;
};

struct errol_slab_t {
        char str[18];
        cl_int exp;
};

typedef union {
        double d;
        cl_long i;
} errol_bits_t;

#ifdef __cplusplus
}
#endif

#endif
