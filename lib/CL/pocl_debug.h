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

#define POCL_ABORT(__MSG__)                                             \
    do {                                                                \
        fprintf(stderr, __MSG__);                                       \
        exit(2);                                                        \
    } while (0)

#define POCL_ERROR(x) do { if (errcode_ret != NULL) {                   \
                              *errcode_ret = (x);                       \
                           } return NULL; } while (0)

#define POCL_SUCCESS() do { if (errcode_ret != NULL) {                  \
                              *errcode_ret = CL_SUCCESS;                \
                            } } while (0)




#ifdef POCL_DEBUG_MESSAGES

extern int pocl_debug_messages;

  #ifdef HAVE_CLOCK_GETTIME

extern struct timespec pocl_debug_timespec;
  #define POCL_MSG_PRINT_INFO(...)                                            \
    do {                                                                      \
        if (pocl_debug_messages) {                                            \
            clock_gettime(CLOCK_REALTIME, &pocl_debug_timespec);              \
            fprintf(stderr,                                                   \
                    "[%li.%09li] POCL: in function %s at line %u:",           \
                    (long)pocl_debug_timespec.tv_sec,                         \
                    (long)pocl_debug_timespec.tv_nsec,                        \
                    __func__, __LINE__);                                      \
            fprintf(stderr, __VA_ARGS__);                                     \
        }                                                                     \
    } while (0)

  #define POCL_MSG_PRINT(TYPE, ERRCODE, ...)                                  \
    do {                                                                      \
        if (pocl_debug_messages) {                                            \
            clock_gettime(CLOCK_REALTIME, &pocl_debug_timespec);              \
            fprintf(stderr, "[%li.%09li] POCL: " TYPE ERRCODE                 \
                    " in function %s at line %u: \n",                         \
                    (long)pocl_debug_timespec.tv_sec,                         \
                    (long)pocl_debug_timespec.tv_nsec,                        \
                    __func__, __LINE__);                                      \
            fprintf(stderr, __VA_ARGS__);                                     \
        }                                                                     \
    } while (0)

  #else

  #define POCL_MSG_PRINT_INFO(...)                                            \
    do {                                                                      \
        if (pocl_debug_messages) {                                            \
            fprintf(stderr, "** POCL ** : in function %s"                     \
                    " at line %u:", __func__, __LINE__);                      \
            fprintf(stderr, __VA_ARGS__);                                     \
        }                                                                     \
    } while (0)

  #define POCL_MSG_PRINT(TYPE, ERRCODE, ...)                                  \
    do {                                                                      \
        if (pocl_debug_messages) {                                            \
            fprintf(stderr, "** POCL ** : " TYPE ERRCODE " in function %s"    \
                    " at line %u: \n",  __func__, __LINE__);                  \
            fprintf(stderr, __VA_ARGS__);                                     \
        }                                                                     \
    } while (0)

  #endif


#define POCL_MSG_WARN(...)    POCL_MSG_PRINT("WARNING", "", __VA_ARGS__)
#define POCL_MSG_ERR(...)     POCL_MSG_PRINT("ERROR", "", __VA_ARGS__)

#else

#define POCL_MSG_WARN(...)
#define POCL_MSG_ERR(...)
#define POCL_MSG_PRINT(...)
#define POCL_MSG_PRINT_INFO(...)

#endif


#define POCL_GOTO_ERROR_ON(cond, err_code, ...)                             \
    if (cond)                                                               \
    {                                                                       \
        POCL_MSG_PRINT("ERROR : ", # err_code, __VA_ARGS__);                \
        errcode = err_code;                                                 \
        goto ERROR;                                                         \
    }                                                                       \

#define POCL_RETURN_ERROR_ON(cond, err_code, ...)                           \
    if (cond)                                                               \
    {                                                                       \
        POCL_MSG_PRINT("ERROR : ", # err_code, __VA_ARGS__);                \
        return err_code;                                                    \
    }                                                                       \

#define POCL_RETURN_ERROR_COND(cond, err_code)                              \
    if (cond)                                                               \
    {                                                                       \
        POCL_MSG_PRINT("ERROR : ", # err_code, "%s\n", # cond);             \
        return err_code;                                                    \
    }                                                                       \

#define POCL_GOTO_ERROR_COND(cond, err_code)                                \
    if (cond)                                                               \
    {                                                                       \
        POCL_MSG_PRINT("ERROR : ", # err_code, "%s\n", # cond);             \
        errcode = err_code;                                                 \
        goto ERROR;                                                         \
    }                                                                       \
