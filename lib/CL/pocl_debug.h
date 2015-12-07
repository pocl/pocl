#ifndef POCL_DEBUG_H
#define POCL_DEBUG_H

#ifdef _WIN32
#  include <stdint.h>
#  include <stddef.h> // size_t
#  define PRIu64 "I64u"
#  define PRIX64 "I64x"
#  define PRIXPTR "p"
#  define PRIuS "Iu"
#else
# ifndef __STDC_FORMAT_MACROS
# define __STDC_FORMAT_MACROS
# endif
# include <inttypes.h>
#endif

// size_t print spec
#ifndef PRIuS
# define PRIuS "zu"
#endif

#ifdef __cplusplus
extern "C" {
#endif

// should use some terminfo library, but..
#define RESET   "\033[0m"
#define BLACK   "\033[30m"      /* Black */
#define RED     "\033[31m"      /* Red */
#define GREEN   "\033[32m"      /* Green */
#define YELLOW  "\033[33m"      /* Yellow */
#define BLUE    "\033[34m"      /* Blue */
#define MAGENTA "\033[35m"      /* Magenta */
#define CYAN    "\033[36m"      /* Cyan */
#define WHITE   "\033[37m"      /* White */
#define BOLDBLACK   "\033[1m\033[30m"      /* Bold Black */
#define BOLDRED     "\033[1m\033[31m"      /* Bold Red */
#define BOLDGREEN   "\033[1m\033[32m"      /* Bold Green */
#define BOLDYELLOW  "\033[1m\033[33m"      /* Bold Yellow */
#define BOLDBLUE    "\033[1m\033[34m"      /* Bold Blue */
#define BOLDMAGENTA "\033[1m\033[35m"      /* Bold Magenta */
#define BOLDCYAN    "\033[1m\033[36m"      /* Bold Cyan */
#define BOLDWHITE   "\033[1m\033[37m"      /* Bold White */

#ifdef __GNUC__
#pragma GCC visibility push(hidden)
#endif

/* Debugging macros. Also macros for marking unimplemented parts of specs or
   untested parts of the implementation. */

#define POCL_ABORT_UNIMPLEMENTED(MSG)                                   \
    do {                                                                \
        fprintf(stderr,"%s is unimplemented (%s:%d)\n",                 \
                        MSG, __FILE__, __LINE__);                       \
        exit(2);                                                        \
    } while (0)

#define POCL_WARN_UNTESTED()                                            \
    do {                                                                \
        fprintf( stderr,                                                \
            "pocl warning: encountered untested part of the "           \
            "implementation in %s:%d\n", __FILE__, __LINE__);           \
    } while (0)

#define POCL_WARN_INCOMPLETE()                                          \
    do {                                                                \
        fprintf( stderr,                                                \
            "pocl warning: encountered incomplete implementation"       \
            " in %s:%d\n", __FILE__, __LINE__);                         \
    } while (0)

#define POCL_ABORT(...)                                                 \
    do {                                                                \
        fprintf(stderr, __VA_ARGS__);                                  \
        exit(2);                                                        \
    } while (0)

#define POCL_ERROR(x) do { if (errcode_ret != NULL) {                   \
                              *errcode_ret = (x);                       \
                           } return NULL; } while (0)

#define POCL_SUCCESS() do { if (errcode_ret != NULL) {                  \
                              *errcode_ret = CL_SUCCESS;                \
                            } } while (0)



#include "config.h"

#ifdef POCL_DEBUG_MESSAGES

    extern int pocl_debug_messages;
    extern int stderr_is_a_tty;

    #if __GNUC__ >= 2
    #define __func__ __PRETTY_FUNCTION__
    #else
    #define __func__ __FUNCTION__
    #endif

    #ifdef HAVE_CLOCK_GETTIME
        #define POCL_DEBUG_HEADER pocl_debug_print_header(__func__, __LINE__);
        extern void pocl_debug_print_header(const char * func, unsigned line);
        extern void pocl_debug_measure_start(void* start);
        extern void pocl_debug_measure_finish(void* start, void* finish,
                                              const char* msg,
                                              const char *func,
                                              unsigned line);
        extern void pocl_debug_print_duration(const char* func, unsigned line,
                                              const char* msg, uint64_t nanosecs);
        #define POCL_MEASURE_START(SUFFIX) \
          struct timespec pocl_time_start_ ## SUFFIX, pocl_time_finish_ ## SUFFIX; \
          pocl_debug_measure_start(&pocl_time_start_ ## SUFFIX);

        #define POCL_MEASURE_FINISH(SUFFIX) \
          pocl_debug_measure_finish(&pocl_time_start_ ## SUFFIX, \
                         &pocl_time_finish_ ## SUFFIX, "API: " #SUFFIX, \
                         __func__, __LINE__);
    #else
        #define POCL_DEBUG_HEADER                                           \
            fprintf(stderr, "** POCL ** : in function %s"                   \
            " at line %u:\n", __func__, __LINE__);
        #define POCL_MEASURE_START(SUFFIX)
        #define POCL_MEASURE_FINISH(SUFFIX)
        #define pocl_debug_print_duration(X)
    #endif

    #define POCL_MSG_PRINT(TYPE, ERRCODE, ...)                              \
        do {                                                                \
            if (pocl_debug_messages) {                                      \
                POCL_DEBUG_HEADER                                           \
                if (stderr_is_a_tty)                                        \
                  fprintf(stderr, TYPE CYAN ERRCODE " "  RESET);            \
                else                                                        \
                  fprintf(stderr, TYPE ERRCODE " ");                        \
                fprintf(stderr, __VA_ARGS__);                               \
            }                                                               \
        } while (0)

    #define POCL_MSG_PRINT2(func, line, ...)                                \
        do {                                                                \
            if (pocl_debug_messages) {                                      \
                pocl_debug_print_header(func, line);                        \
                fprintf(stderr, __VA_ARGS__);                               \
            }                                                               \
        } while (0)

    #define POCL_MSG_WARN2(errcode, ...)   if (stderr_is_a_tty) \
          POCL_MSG_PRINT(YELLOW " *** WARNING *** ", errcode, __VA_ARGS__); \
          else POCL_MSG_PRINT(" *** WARNING *** ", errcode, __VA_ARGS__)
    #define POCL_MSG_WARN(...)  POCL_MSG_WARN2("", __VA_ARGS__)

    #define POCL_MSG_ERR2(errcode, ...)    if (stderr_is_a_tty) \
          POCL_MSG_PRINT(RED " *** ERROR *** ", errcode, __VA_ARGS__); \
          else POCL_MSG_PRINT(" *** ERROR *** ", errcode, __VA_ARGS__)
    #define POCL_MSG_ERR(...)  POCL_MSG_ERR2("", __VA_ARGS__)

    #define POCL_MSG_PRINT_INFO2(errcode, ...) if (stderr_is_a_tty) \
          POCL_MSG_PRINT(GREEN " *** INFO *** ", errcode, __VA_ARGS__); \
          else POCL_MSG_PRINT(" *** INFO *** ", errcode, __VA_ARGS__)
    #define POCL_MSG_PRINT_INFO(...) POCL_MSG_PRINT_INFO2("", __VA_ARGS__)

#else

    #define POCL_MSG_WARN(...)
    #define POCL_MSG_ERR(...)
    #define POCL_MSG_PRINT(...)
    #define POCL_MSG_PRINT2(...)
    #define POCL_MSG_PRINT_INFO(...)

#endif

#define POCL_DEBUG_EVENT_TIME(eventp, msg) \
        pocl_debug_print_duration(__func__, __LINE__, "Event " msg, (uint64_t)((*eventp)->time_end - (*eventp)->time_start))

#define POCL_GOTO_ERROR_ON(cond, err_code, ...)                             \
  do                                                                        \
    {                                                                       \
      if (cond)                                                             \
        {                                                                   \
            POCL_MSG_ERR2(#err_code, __VA_ARGS__);                          \
            errcode = err_code;                                             \
            goto ERROR;                                                     \
        }                                                                   \
    }                                                                       \
  while (0)

#define POCL_RETURN_ERROR_ON(cond, err_code, ...)                           \
  do                                                                        \
    {                                                                       \
      if (cond)                                                             \
        {                                                                   \
            POCL_MSG_ERR2(#err_code, __VA_ARGS__);                          \
            return err_code;                                                \
        }                                                                   \
    }                                                                       \
  while (0)

#define POCL_RETURN_ERROR_COND(cond, err_code)                              \
  do                                                                        \
    {                                                                       \
      if (cond)                                                             \
        {                                                                   \
          POCL_MSG_ERR2(#err_code, "%s\n", #cond);                          \
          return err_code;                                                  \
        }                                                                   \
    }                                                                       \
  while (0)

#define POCL_GOTO_ERROR_COND(cond, err_code)                                \
  do                                                                        \
    {                                                                       \
      if (cond)                                                             \
        {                                                                   \
          POCL_MSG_ERR2(#err_code, "%s\n", #cond);                          \
          errcode = err_code;                                               \
          goto ERROR;                                                       \
        }                                                                   \
    }                                                                       \
  while (0)



#ifdef __GNUC__
#pragma GCC visibility pop
#endif

#ifdef __cplusplus
}
#endif

#endif
